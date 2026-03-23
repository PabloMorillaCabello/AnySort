#!/usr/bin/env python3
"""
TEST VERSION — Persistent SAM3 server variant of demo_orbbec_gemini2.py.

Difference from the original:
  - SAM3 model is loaded ONCE at startup via a persistent subprocess (sam3_server.py).
  - Each keypress snapshot sends a request to the already-running server instead of
    spawning a new process and reloading model.safetensors from scratch.

Original script: scripts/demo_orbbec_gemini2.py
SAM3 server:     scripts/TEST/sam3_server.py
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh.transformations as tra

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.robot import get_gripper_info
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    get_color_from_score,
    make_frame,
    visualize_grasp,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import (
    depth_and_segmentation_to_point_clouds,
    filter_colliding_grasps,
    point_cloud_outlier_removal,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture one frame from Orbbec Gemini 2 and run GraspGen inference"
    )
    parser.add_argument("--gripper_config", type=str, required=True,
                        help="Path to checkpoint config yml")
    parser.add_argument("--segmentation_mask_path", type=str, default="",
                        help="Optional path to segmentation mask (.npy or image)")
    parser.add_argument("--target_object_id", type=int, default=1,
                        help="Object id in segmentation mask used as grasp target")
    parser.add_argument("--num_grasps", type=int, default=200)
    parser.add_argument("--grasp_threshold", type=float, default=-1.0)
    parser.add_argument("--topk_num_grasps", type=int, default=100)
    parser.add_argument("--return_topk", action="store_true")
    parser.add_argument("--collision_filter", action="store_true")
    parser.add_argument("--collision_threshold", type=float, default=0.02)
    parser.add_argument("--max_scene_points", type=int, default=8192)
    parser.add_argument("--max_object_points", type=int, default=12000)
    parser.add_argument("--scene_point_size", type=float, default=0.008)
    parser.add_argument("--object_point_size", type=float, default=0.012)
    parser.add_argument("--auto_mask_depth_delta", type=float, default=0.08)
    parser.add_argument("--auto_mask_min_pixels", type=int, default=800)
    parser.add_argument("--save_capture_prefix", type=str, default="")
    parser.add_argument("--keypress", action="store_true",
                        help="Interactive mode: press Enter to capture, q to quit")
    # SAM3
    parser.add_argument("--sam3_use", action="store_true",
                        help="Use SAM3 (text prompt) to obtain segmentation mask from RGB")
    parser.add_argument("--sam3_prompt", type=str, default="",
                        help="Text prompt for SAM3 (e.g. 'red mug')")
    parser.add_argument("--sam3_device", type=str, default="cuda:0",
                        help="Device for SAM3 server (default: cuda:0)")
    parser.add_argument("--sam3_no_fp16", action="store_true",
                        help="Use FP32 for SAM3 instead of FP16")
    parser.add_argument("--sam3_no_transformers", action="store_true",
                        help="Use native SAM3 API instead of HuggingFace Transformers")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# SAM3 persistent server helpers
# ---------------------------------------------------------------------------

_SAM3_PYTHON_CANDIDATES = [
    "/opt/sam3env/bin/python3.12",
    "/opt/sam3env/bin/python3",
    "/opt/sam3env/bin/python",
    "/usr/bin/python3.12",
]
_SAM3_SERVER_SCRIPT = "/ros2_ws/scripts/TEST/sam3_server.py"


def _find_sam3_python() -> str:
    for p in _SAM3_PYTHON_CANDIDATES:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise RuntimeError(
        "Cannot find SAM3 Python interpreter. Tried:\n"
        + "\n".join(f"  {p}" for p in _SAM3_PYTHON_CANDIDATES)
    )


def start_sam3_server(args) -> subprocess.Popen:
    """
    Launch sam3_server.py as a persistent subprocess.
    Blocks until the server prints {"status": "ready"} on stdout,
    meaning model.safetensors has been loaded into GPU memory.
    Returns the Popen handle to pass around for later requests.
    """
    python_bin = _find_sam3_python()

    cmd = [python_bin, _SAM3_SERVER_SCRIPT, "--device", args.sam3_device]
    if args.sam3_no_fp16:
        cmd.append("--no-fp16")
    if args.sam3_no_transformers:
        cmd.append("--no-transformers")

    # Strip GraspGen venv from env so SAM3 Python 3.12 finds its own stdlib
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)

    print(f"[SAM3] Starting persistent server: {' '.join(cmd)}", flush=True)
    print("[SAM3] Loading model.safetensors — this happens ONCE, please wait...", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,   # SAM3 server stderr passes through to our terminal
        env=env,
        text=True,
        bufsize=1,     # line-buffered
    )

    # Wait for the server to signal it is ready
    start = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(
                f"SAM3 server exited unexpectedly (code {proc.returncode}) before sending 'ready'."
            )
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.05)
            continue
        try:
            msg = json.loads(line.strip())
        except json.JSONDecodeError:
            print(f"[SAM3 server raw] {line.rstrip()}", flush=True)
            continue

        if msg.get("status") == "ready":
            print(f"[SAM3] Server ready in {time.time() - start:.1f}s. Model stays in GPU memory.", flush=True)
            return proc
        elif msg.get("status") == "load_error":
            proc.terminate()
            raise RuntimeError(f"SAM3 server failed to load model: {msg.get('error')}")
        else:
            print(f"[SAM3 server] {msg}", flush=True)


def segment_with_sam3(rgb: np.ndarray, text_prompt: str, sam3_proc: subprocess.Popen) -> np.ndarray:
    """
    Send one inference request to the already-running SAM3 server.
    Returns binary mask (H, W) uint8 {0, 1}.
    No model reload — just the GPU forward pass.
    """
    if not text_prompt:
        raise ValueError("sam3_prompt is empty")

    tmp_dir = Path("/ros2_ws/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    img_path = tmp_dir / "sam3_input.png"
    mask_path = tmp_dir / "sam3_mask.npy"

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_path), rgb_bgr)

    request = json.dumps({
        "image": str(img_path),
        "prompt": text_prompt,
        "mask_out": str(mask_path),
    })

    t0 = time.time()
    sam3_proc.stdin.write(request + "\n")
    sam3_proc.stdin.flush()

    # Read response line (blocking)
    while True:
        if sam3_proc.poll() is not None:
            raise RuntimeError(f"SAM3 server died unexpectedly (code {sam3_proc.returncode})")
        line = sam3_proc.stdout.readline()
        if not line:
            time.sleep(0.02)
            continue
        break

    elapsed = time.time() - t0
    try:
        resp = json.loads(line.strip())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"SAM3 server bad response: {line!r}") from e

    if resp.get("status") != "ok":
        raise RuntimeError(f"SAM3 server error: {resp.get('error', resp)}")

    print(f"[SAM3] Inference done in {elapsed:.2f}s, {resp.get('pixels', '?')} pixels masked", flush=True)

    mask = np.load(mask_path)
    if mask.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"SAM3 mask shape {mask.shape} does not match RGB shape {rgb.shape}")
    return (mask > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Camera helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _extract_intrinsics(profile, frame):
    fx = fy = cx = cy = None
    for obj in [frame, profile]:
        if obj is None:
            continue
        for method_name in ["get_intrinsic", "get_camera_intrinsic", "get_intrinsics"]:
            if not hasattr(obj, method_name):
                continue
            intr = getattr(obj, method_name)()
            for attr_name in ["fx", "focal_x"]:
                if hasattr(intr, attr_name):
                    fx = float(getattr(intr, attr_name))
                    break
            for attr_name in ["fy", "focal_y"]:
                if hasattr(intr, attr_name):
                    fy = float(getattr(intr, attr_name))
                    break
            for attr_name in ["cx", "ppx", "principal_x"]:
                if hasattr(intr, attr_name):
                    cx = float(getattr(intr, attr_name))
                    break
            for attr_name in ["cy", "ppy", "principal_y"]:
                if hasattr(intr, attr_name):
                    cy = float(getattr(intr, attr_name))
                    break
            if None not in [fx, fy, cx, cy]:
                return fx, fy, cx, cy
    raise RuntimeError("Could not read camera intrinsics from Orbbec SDK objects.")


def _to_rgb_array(color_frame, ob_format):
    from io import BytesIO
    from PIL import Image

    h, w = color_frame.get_height(), color_frame.get_width()
    raw = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    fmt = color_frame.get_format()
    fmt_name = str(fmt).upper()

    if hasattr(ob_format, "RGB") and fmt == ob_format.RGB:
        return raw.reshape(h, w, 3)
    if hasattr(ob_format, "BGR") and fmt == ob_format.BGR:
        return raw.reshape(h, w, 3)[:, :, ::-1].copy()
    if (hasattr(ob_format, "MJPG") and fmt == ob_format.MJPG) or "MJPG" in fmt_name:
        with Image.open(BytesIO(raw.tobytes())) as img:
            return np.array(img.convert("RGB"))
    if (hasattr(ob_format, "YUYV") and fmt == ob_format.YUYV) or "YUYV" in fmt_name:
        import cv2 as cv2_local
        return cv2_local.cvtColor(raw.reshape(h, w, 2), cv2_local.COLOR_YUV2RGB_YUY2)
    if (hasattr(ob_format, "UYVY") and fmt == ob_format.UYVY) or "UYVY" in fmt_name:
        import cv2 as cv2_local
        return cv2_local.cvtColor(raw.reshape(h, w, 2), cv2_local.COLOR_YUV2RGB_UYVY)

    expected_rgb_size = h * w * 3
    if raw.size == expected_rgb_size:
        return raw.reshape(h, w, 3)
    if raw.size == h * w * 2:
        import cv2 as cv2_local
        return cv2_local.cvtColor(raw.reshape(h, w, 2), cv2_local.COLOR_YUV2RGB_YUY2)

    raise RuntimeError(f"Unsupported color frame format {fmt} with buffer size {raw.size} for {w}x{h}")


def capture_orbbec_frame():
    try:
        from pyorbbecsdk import Config, OBSensorType, OBFormat, Pipeline
    except Exception as error:
        raise RuntimeError("pyorbbecsdk is not installed.") from error

    pipeline = Pipeline()
    config = Config()

    profile_list_depth = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list_depth.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    profile_list_color = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list_color.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    pipeline.start(config)
    try:
        frames = None
        for _ in range(40):
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            if frames.get_depth_frame() is not None and frames.get_color_frame() is not None:
                break

        if frames is None:
            raise RuntimeError("Failed to receive frames from Orbbec pipeline")

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if depth_frame is None or color_frame is None:
            raise RuntimeError("Could not get both depth and color frames")

        depth_h, depth_w = depth_frame.get_height(), depth_frame.get_width()
        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(depth_h, depth_w)
        depth_scale = float(depth_frame.get_depth_scale())
        depth_m = depth_raw.astype(np.float32) * depth_scale

        valid_depth = depth_m[(depth_m > 0.05) & np.isfinite(depth_m)]
        if valid_depth.size > 0:
            median_depth = float(np.median(valid_depth))
            if median_depth > 20.0:
                depth_m = depth_m / 1000.0
                print(f"Depth in mm (median={median_depth:.3f}); converting to meters")

        from pyorbbecsdk import OBFormat as OBFormat_local
        rgb = _to_rgb_array(color_frame, OBFormat_local)
        if rgb.shape[0] != depth_h or rgb.shape[1] != depth_w:
            from PIL import Image as PILImage
            rgb = np.array(PILImage.fromarray(rgb).resize((depth_w, depth_h), resample=PILImage.BILINEAR))

        fx, fy, cx, cy = _extract_intrinsics(depth_profile, depth_frame)
        return depth_m, rgb, (fx, fy, cx, cy)
    finally:
        pipeline.stop()


def load_mask(mask_path):
    if mask_path.endswith(".npy"):
        return np.load(mask_path)
    from PIL import Image as PILImage
    return np.array(PILImage.open(mask_path))


def auto_mask_from_depth(depth_m, depth_delta, min_pixels):
    h, w = depth_m.shape
    cy, cx = h // 2, w // 2
    valid = (depth_m > 0.1) & np.isfinite(depth_m)
    if not valid.any():
        raise RuntimeError("No valid depth values found for auto mask")

    center_depth = depth_m[cy, cx]
    if not np.isfinite(center_depth) or center_depth <= 0.1:
        center_depth = np.percentile(depth_m[valid], 35)

    lower = max(0.1, center_depth - depth_delta)
    upper = center_depth + depth_delta
    foreground = valid & (depth_m >= lower) & (depth_m <= upper)

    if foreground.sum() < min_pixels:
        lo = np.percentile(depth_m[valid], 10)
        hi = np.percentile(depth_m[valid], 45)
        foreground = valid & (depth_m >= lo) & (depth_m <= hi)

    if foreground.sum() < min_pixels:
        raise RuntimeError(f"Auto mask too small ({foreground.sum()} pixels).")

    mask = np.zeros_like(depth_m, dtype=np.uint8)
    mask[foreground] = 1
    return mask


def process_point_cloud(pc, grasps, grasp_conf):
    scores = get_color_from_score(grasp_conf, use_255_scale=True)
    grasps[:, 3, 3] = 1
    t_center = tra.translation_matrix(-pc.mean(axis=0))
    pc_centered = tra.transform_points(pc, t_center)
    grasps_centered = np.array([t_center @ np.array(g) for g in grasps.tolist()])
    return pc_centered, grasps_centered, scores, t_center


# ---------------------------------------------------------------------------
# Main snapshot logic (sam3_proc added as parameter)
# ---------------------------------------------------------------------------

def run_snapshot(args, vis, sam3_proc):
    vis.delete()

    print("Capturing one frame from Orbbec Gemini 2...")
    cap_start = time.time()
    depth_m, rgb, (fx, fy, cx, cy) = capture_orbbec_frame()
    print(
        f"Capture complete in {time.time() - cap_start:.2f}s | "
        f"depth={depth_m.shape} rgb={rgb.shape} intrinsics={(fx, fy, cx, cy)}"
    )

    # --- Segmentation mask: SAM3 > external > auto depth ---
    if args.sam3_use and args.sam3_prompt:
        print(f"[SAM3] Sending request with prompt: '{args.sam3_prompt}'")
        sam3_mask = segment_with_sam3(rgb, args.sam3_prompt, sam3_proc)
        if sam3_mask.shape != depth_m.shape:
            raise ValueError(f"SAM3 mask shape {sam3_mask.shape} != depth shape {depth_m.shape}")
        segmentation_mask = sam3_mask.astype(np.uint8)
        print("Segmentation mask obtained from SAM3.")
    elif args.segmentation_mask_path:
        segmentation_mask = load_mask(args.segmentation_mask_path)
        print(f"Loaded segmentation mask from {args.segmentation_mask_path}")
    else:
        segmentation_mask = auto_mask_from_depth(
            depth_m, depth_delta=args.auto_mask_depth_delta, min_pixels=args.auto_mask_min_pixels
        )
        print("Generated segmentation mask automatically from depth")

    if segmentation_mask.shape != depth_m.shape:
        raise ValueError(f"Mask shape {segmentation_mask.shape} != depth shape {depth_m.shape}")

    if args.save_capture_prefix:
        np.save(f"{args.save_capture_prefix}_depth.npy", depth_m)
        np.save(f"{args.save_capture_prefix}_rgb.npy", rgb)
        np.save(f"{args.save_capture_prefix}_mask.npy", segmentation_mask)
        print(f"Saved capture to prefix: {args.save_capture_prefix}")

    scene_pc, object_pc, scene_colors, object_colors = depth_and_segmentation_to_point_clouds(
        depth_image=depth_m,
        segmentation_mask=segmentation_mask,
        fx=fx, fy=fy, cx=cx, cy=cy,
        rgb_image=rgb,
        target_object_id=args.target_object_id,
        remove_object_from_scene=True,
    )

    if len(object_pc) > args.max_object_points:
        keep_idx = np.random.choice(len(object_pc), args.max_object_points, replace=False)
        object_pc = object_pc[keep_idx]
        if object_colors is not None:
            object_colors = object_colors[keep_idx]
        print(f"Downsampled object point cloud to {len(object_pc)} points")

    object_pc_torch = torch.from_numpy(object_pc)
    pc_filtered, _ = point_cloud_outlier_removal(object_pc_torch)
    pc_filtered = pc_filtered.numpy()

    if len(pc_filtered) > 0:
        t_center = tra.translation_matrix(-pc_filtered.mean(axis=0))
        pc_centered = tra.transform_points(pc_filtered, t_center)
    else:
        t_center = np.eye(4)
        pc_centered = pc_filtered

    if scene_colors is None:
        scene_colors = np.tile(np.array([[120, 120, 120]], dtype=np.uint8), (len(scene_pc), 1))

    object_vis_colors = np.tile(np.array([[255, 255, 255]], dtype=np.uint8), (len(pc_centered), 1))
    scene_centered = tra.transform_points(scene_pc, t_center)

    visualize_pointcloud(vis, "pc_scene", scene_centered, scene_colors, size=args.scene_point_size)
    visualize_pointcloud(vis, "pc_obj", pc_centered, object_vis_colors, size=args.object_point_size)

    if len(scene_centered) > 0:
        print(f"Scene bounds (centered): min={scene_centered.min(axis=0)}, max={scene_centered.max(axis=0)}")

    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    sampler = GraspGenSampler(grasp_cfg)

    grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
        pc_filtered, sampler,
        grasp_threshold=args.grasp_threshold,
        num_grasps=args.num_grasps,
        topk_num_grasps=args.topk_num_grasps,
    )

    if len(grasps_inferred) == 0:
        print("No grasps found.")
        return

    grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
    grasps_inferred = grasps_inferred.cpu().numpy()
    pc_centered, grasps_centered, scores, t_center = process_point_cloud(
        pc_filtered, grasps_inferred, grasp_conf_inferred
    )

    if args.collision_filter:
        gripper_info = get_gripper_info(gripper_name)
        collision_mesh = gripper_info.collision_mesh
        if len(scene_centered) > args.max_scene_points:
            idx = np.random.choice(len(scene_centered), args.max_scene_points, replace=False)
            scene_for_collision = scene_centered[idx]
        else:
            scene_for_collision = scene_centered

        collision_free_mask = filter_colliding_grasps(
            scene_pc=scene_for_collision,
            grasp_poses=grasps_centered,
            gripper_collision_mesh=collision_mesh,
            collision_threshold=args.collision_threshold,
        )

        free_grasps = grasps_centered[collision_free_mask]
        colliding_grasps = grasps_centered[~collision_free_mask]
        free_scores = scores[collision_free_mask]

        for j, grasp in enumerate(free_grasps):
            visualize_grasp(vis, f"grasps/free/{j:03d}/grasp", grasp,
                            color=free_scores[j], gripper_name=gripper_name, linewidth=1.5)
        for j, grasp in enumerate(colliding_grasps[:40]):
            visualize_grasp(vis, f"grasps/colliding/{j:03d}/grasp", grasp,
                            color=[255, 0, 0], gripper_name=gripper_name, linewidth=0.4)

        print(f"Collision filter: {collision_free_mask.sum()}/{len(collision_free_mask)} grasps collision-free")
    else:
        for j, grasp in enumerate(grasps_centered):
            visualize_grasp(vis, f"grasps/{j:03d}/grasp", grasp,
                            color=scores[j], gripper_name=gripper_name, linewidth=1.2)

    print(
        f"Done. Inferred {len(grasps_inferred)} grasps, confidence "
        f"[{grasp_conf_inferred.min():.3f}, {grasp_conf_inferred.max():.3f}]"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.exists(args.gripper_config):
        raise FileNotFoundError(args.gripper_config)

    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    # Start SAM3 server ONCE before the capture loop
    sam3_proc = None
    if args.sam3_use:
        if not args.sam3_prompt:
            raise ValueError("--sam3_use requires --sam3_prompt")
        sam3_proc = start_sam3_server(args)

    vis = create_visualizer()
    make_frame(vis, "world", h=0.12, radius=0.004)
    print("Meshcat visualization active. Open http://127.0.0.1:7000 in your browser.")

    try:
        if args.keypress:
            print("Keypress mode: Enter=Capture, q=Quit")
            while True:
                user_in = input("[Enter=Capture, q=Quit] > ").strip().lower()
                if user_in in ["q", "quit", "exit"]:
                    print("Exiting.")
                    break
                try:
                    run_snapshot(args, vis, sam3_proc)
                except Exception as error:
                    print(f"Snapshot failed: {error}")
        else:
            run_snapshot(args, vis, sam3_proc)
            print("Visualizer running. Keep this process alive.")
            while True:
                time.sleep(1.0)
    finally:
        if sam3_proc is not None:
            print("[SAM3] Shutting down server...", flush=True)
            try:
                sam3_proc.stdin.close()
                sam3_proc.wait(timeout=5)
            except Exception:
                sam3_proc.terminate()


if __name__ == "__main__":
    main()
