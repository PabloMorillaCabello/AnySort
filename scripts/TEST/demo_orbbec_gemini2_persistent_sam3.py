#!/usr/bin/env python3
"""
GraspGen Pipeline — Tkinter UI with persistent SAM3 socket server.

Workflow:
  1. SAM3 runs as a persistent Unix socket server (scripts/sam3_server.py),
     loading model.safetensors ONCE into GPU memory.
  2. Orbbec Gemini 2 streams live RGB to the Tkinter preview window.
  3. User selects a gripper config from the dropdown, types an object prompt,
     and clicks "Capture & Run".
  4. Latest frame → SAM3 (via socket) → GraspGen → Meshcat visualisation.
  5. Segmentation mask is overlaid on the live preview after each inference.
  6. Clicking "Capture & Run" again re-runs with the same (cached) model —
     no restart needed.

Usage — two-terminal workflow:
  # Terminal 1: start SAM3 server (sam3 Python env, loads model once)
  /opt/sam3env/bin/python /ros2_ws/scripts/sam3_server.py [--device cuda:0]

  # Terminal 2: run this UI (GraspGen Python env)
  python /ros2_ws/scripts/TEST/demo_orbbec_gemini2_persistent_sam3.py

  # Or let this script auto-start the SAM3 server:
  python /ros2_ws/scripts/TEST/demo_orbbec_gemini2_persistent_sam3.py --sam3_autostart

Meshcat visualisation: http://127.0.0.1:7000
"""

import gc
import json
import os
import queue
import socket as _socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Redirect OrbbecSDK C-level stderr to file — prevents timestamp-anomaly spam
# from flooding the terminal. Full log available at /tmp/orbbec_sdk.log
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orbbec_quiet  # noqa: E402

import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, scrolledtext
import trimesh.transformations as tra

try:
    from PIL import Image as PILImage, ImageTk
    _PIL_OK = True
except ImportError:
    PILImage = None
    ImageTk = None
    _PIL_OK = False

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
# Constants
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = "/opt/GraspGen/GraspGenModels/checkpoints"
SAM3_SERVER_SCRIPT = "/ros2_ws/scripts/sam3_server.py"

_SAM3_PYTHON_CANDIDATES = [
    "/opt/sam3env/bin/python3.12",
    "/opt/sam3env/bin/python3",
    "/opt/sam3env/bin/python",
    "/usr/bin/python3.12",
]

_PREVIEW_W = 640
_PREVIEW_H = 480
_MASK_GREEN_RGB = np.array([0, 210, 90], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def scan_checkpoints(directory: str) -> dict:
    """Return {basename: full_path} for every .yml found in directory."""
    d = Path(directory)
    if not d.exists():
        return {}
    return {p.name: str(p) for p in sorted(d.glob("*.yml"))}


# ---------------------------------------------------------------------------
# SAM3 server management
# ---------------------------------------------------------------------------
def _find_sam3_python() -> str:
    for p in _SAM3_PYTHON_CANDIDATES:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise RuntimeError(
        "Cannot find SAM3 Python interpreter. Tried:\n"
        + "\n".join(f"  {p}" for p in _SAM3_PYTHON_CANDIDATES)
        + "\nRun `ls -la /opt/sam3env/bin/` inside the container."
    )


def _wait_for_sam3_socket(sock_path: str, timeout: float = 120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(sock_path)
            s.close()
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(
        f"SAM3 server did not become ready within {timeout:.0f}s at '{sock_path}'"
    )


def start_sam3_server(sock_path: str, device: str = "cuda:0",
                      fp16: bool = True) -> subprocess.Popen:
    python_bin = _find_sam3_python()
    cmd = [python_bin, SAM3_SERVER_SCRIPT, "--socket", sock_path, "--device", device]
    if not fp16:
        cmd.append("--no-fp16")
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    print(f"[SAM3] Starting server: {' '.join(cmd)}", flush=True)
    print("[SAM3] Loading model — this happens ONCE, please wait...", flush=True)
    proc = subprocess.Popen(cmd, env=env)
    _wait_for_sam3_socket(sock_path, timeout=120.0)
    print("[SAM3] Server ready.", flush=True)
    return proc


# ---------------------------------------------------------------------------
# Mask post-processing
# ---------------------------------------------------------------------------
def _select_largest_component(mask: np.ndarray) -> tuple:
    """Keep only the largest connected component in a binary mask (H,W) uint8."""
    if mask.sum() == 0:
        return mask, 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    n_components = num_labels - 1
    if n_components <= 1:
        return mask, n_components

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(areas.argmax()) + 1
    result = (labels == best_label).astype(np.uint8)
    return result, n_components


# ---------------------------------------------------------------------------
# SAM3 socket client
# ---------------------------------------------------------------------------
def _recv_exactly(s: _socket.socket, n: int) -> bytes:
    chunks, received = [], 0
    while received < n:
        chunk = s.recv(min(n - received, 65536))
        if not chunk:
            raise ConnectionError("SAM3 server disconnected mid-transfer")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def _recv_json_line(s: _socket.socket) -> dict:
    buf = b""
    while True:
        byte = s.recv(1)
        if not byte:
            raise ConnectionError("SAM3 server disconnected")
        if byte == b"\n":
            break
        buf += byte
    return json.loads(buf.decode("utf-8"))


def segment_with_sam3_server(rgb: np.ndarray, prompt: str, sock_path: str) -> np.ndarray:
    """Send one inference request to the SAM3 Unix socket server."""
    if not prompt:
        raise ValueError("SAM3 prompt is empty")

    h, w = rgb.shape[:2]
    rgb_bytes = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    header = json.dumps({
        "width": w,
        "height": h,
        "prompt": prompt,
        "size": len(rgb_bytes),
    }) + "\n"

    t0 = time.time()
    try:
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.settimeout(90.0)
        s.connect(sock_path)
        s.sendall(header.encode("utf-8"))
        s.sendall(rgb_bytes)
        resp = _recv_json_line(s)
        if not resp.get("ok"):
            raise RuntimeError(f"SAM3 server error: {resp.get('error', resp)}")
        num_masks = resp.get("num_masks", 1)
        mask_bytes = _recv_exactly(s, resp["size"]) if resp["size"] > 0 else b""
        s.close()
    except OSError as e:
        raise RuntimeError(
            f"Cannot connect to SAM3 server at '{sock_path}': {e}\n"
            "Start it with:  /opt/sam3env/bin/python /ros2_ws/scripts/sam3_server.py"
        ) from e

    if num_masks == 0 or not mask_bytes:
        print(f"[SAM3] {time.time()-t0:.2f}s — no detections", flush=True)
        return np.zeros((h, w), dtype=np.uint8)

    all_masks = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(num_masks, h, w)
    best_idx = 0
    best_mask, n_objects = _select_largest_component(
        (all_masks[best_idx] > 0).astype(np.uint8)
    )

    print(
        f"[SAM3] {time.time()-t0:.2f}s — "
        f"{n_objects} '{prompt}' found — "
        f"selected largest, {int(best_mask.sum())} px",
        flush=True,
    )
    return best_mask


# ---------------------------------------------------------------------------
# Orbbec camera helpers
# ---------------------------------------------------------------------------
def _to_rgb_array(color_frame, ob_format) -> np.ndarray:
    from io import BytesIO

    h, w = color_frame.get_height(), color_frame.get_width()
    raw = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    fmt = color_frame.get_format()
    fmt_name = str(fmt).upper()

    if hasattr(ob_format, "RGB") and fmt == ob_format.RGB:
        return raw.reshape(h, w, 3).copy()
    if hasattr(ob_format, "BGR") and fmt == ob_format.BGR:
        return raw.reshape(h, w, 3)[:, :, ::-1].copy()
    if (hasattr(ob_format, "MJPG") and fmt == ob_format.MJPG) or "MJPG" in fmt_name:
        img = PILImage.open(BytesIO(raw.tobytes()))
        return np.array(img.convert("RGB"))
    if (hasattr(ob_format, "YUYV") and fmt == ob_format.YUYV) or "YUYV" in fmt_name:
        return cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUY2)
    if (hasattr(ob_format, "UYVY") and fmt == ob_format.UYVY) or "UYVY" in fmt_name:
        return cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2RGB_UYVY)
    if raw.size == h * w * 3:
        return raw.reshape(h, w, 3).copy()
    if raw.size == h * w * 2:
        return cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUY2)
    raise RuntimeError(f"Unsupported color format {fmt} size={raw.size} for {w}x{h}")


def _extract_intrinsics(depth_profile, depth_frame):
    fx = fy = cx = cy = None
    for obj in [depth_frame, depth_profile]:
        if obj is None:
            continue
        for method in ["get_intrinsic", "get_camera_intrinsic", "get_intrinsics"]:
            if not hasattr(obj, method):
                continue
            intr = getattr(obj, method)()
            for attr in ["fx", "focal_x"]:
                if hasattr(intr, attr):
                    fx = float(getattr(intr, attr))
                    break
            for attr in ["fy", "focal_y"]:
                if hasattr(intr, attr):
                    fy = float(getattr(intr, attr))
                    break
            for attr in ["cx", "ppx", "principal_x"]:
                if hasattr(intr, attr):
                    cx = float(getattr(intr, attr))
                    break
            for attr in ["cy", "ppy", "principal_y"]:
                if hasattr(intr, attr):
                    cy = float(getattr(intr, attr))
                    break
            if None not in [fx, fy, cx, cy]:
                return fx, fy, cx, cy
    raise RuntimeError("Could not read camera intrinsics from Orbbec SDK.")


# ---------------------------------------------------------------------------
# Persistent Orbbec camera
# ---------------------------------------------------------------------------
class OrbbecCamera:
    """Keeps the Orbbec Pipeline alive and continuously exposes the latest frame."""

    def __init__(self):
        try:
            from pyorbbecsdk import Config, OBSensorType, Pipeline, OBFormat
        except ImportError as e:
            raise RuntimeError("pyorbbecsdk is not installed.") from e

        self._Pipeline = Pipeline
        self._Config = Config
        self._OBSensorType = OBSensorType
        self._OBFormat = OBFormat

        self._lock = threading.Lock()
        self._latest_rgb = None
        self._latest_depth_m = None
        self._latest_intrinsics = None
        self._running = False
        self._thread = None
        self._depth_profile = None

    def start(self):
        pipeline = self._Pipeline()
        config = self._Config()

        depth_list = pipeline.get_stream_profile_list(self._OBSensorType.DEPTH_SENSOR)
        # Prefer Y16 — RLE is not supported by post-processing filters or pixel-size API
        depth_profile = None
        for _j in range(len(depth_list)):
            if depth_list[_j].get_format() == self._OBFormat.Y16:
                depth_profile = depth_list[_j]
                break
        if depth_profile is None:
            depth_profile = depth_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        self._depth_profile = depth_profile

        color_list = pipeline.get_stream_profile_list(self._OBSensorType.COLOR_SENSOR)
        color_profile = color_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        pipeline.start(config)
        self._pipeline = pipeline
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self):
        OBFormat = self._OBFormat
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(100)
            except Exception:
                continue
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue
            try:
                rgb = _to_rgb_array(color_frame, OBFormat)
            except Exception:
                continue
            try:
                dh = depth_frame.get_height()
                dw = depth_frame.get_width()
                depth_raw = np.frombuffer(
                    depth_frame.get_data(), dtype=np.uint16
                ).reshape(dh, dw)
                depth_scale = float(depth_frame.get_depth_scale())
                depth_m = depth_raw.astype(np.float32) * depth_scale

                valid = depth_m[(depth_m > 0.05) & np.isfinite(depth_m)]
                if valid.size > 0 and float(np.median(valid)) > 20.0:
                    depth_m = depth_m / 1000.0

                if rgb.shape[0] != dh or rgb.shape[1] != dw:
                    rgb = np.array(
                        PILImage.fromarray(rgb).resize((dw, dh), PILImage.BILINEAR)
                    )
                intr = _extract_intrinsics(self._depth_profile, depth_frame)
            except Exception:
                continue

            with self._lock:
                self._latest_rgb = rgb
                self._latest_depth_m = depth_m
                self._latest_intrinsics = intr

    def get_latest(self):
        with self._lock:
            if self._latest_rgb is None:
                return None, None, None
            return (
                self._latest_rgb.copy(),
                self._latest_depth_m.copy(),
                self._latest_intrinsics,
            )

    def stop(self):
        self._running = False
        try:
            self._pipeline.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# GraspGen helpers
# ---------------------------------------------------------------------------
def _process_point_cloud(pc, grasps, grasp_conf):
    scores = get_color_from_score(grasp_conf, use_255_scale=True)
    grasps[:, 3, 3] = 1
    t_center = tra.translation_matrix(-pc.mean(axis=0))
    pc_centered = tra.transform_points(pc, t_center)
    grasps_centered = np.array([t_center @ np.array(g) for g in grasps.tolist()])
    return pc_centered, grasps_centered, scores, t_center


def _save_best_grasp(grasps_cam, scores, t_center) -> dict:
    if len(grasps_cam) == 0:
        return {}
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    pose = grasps_cam[best_idx].copy()
    translation = pose[:3, 3]
    rot_mat = pose[:3, :3]
    try:
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(rot_mat)
        euler_deg = r.as_euler("xyz", degrees=True)
        quat_xyzw = r.as_quat()
    except Exception:
        euler_deg = np.zeros(3)
        quat_xyzw = np.array([0., 0., 0., 1.])

    sep = "=" * 56
    print(f"\n{sep}\n  BEST GRASP\n{sep}")
    print(f"  Rank:       #{best_idx + 1} of {len(grasps_cam)}")
    print(f"  Confidence: {best_score:.4f}  [{scores.min():.4f}–{scores.max():.4f}]")
    print(f"  Position:   X={translation[0]:+.4f}  Y={translation[1]:+.4f}  Z={translation[2]:+.4f} m")
    print(f"  Euler XYZ:  Roll={euler_deg[0]:+.2f}°  Pitch={euler_deg[1]:+.2f}°  Yaw={euler_deg[2]:+.2f}°")
    print(f"  Quaternion: [{quat_xyzw[0]:+.4f}, {quat_xyzw[1]:+.4f}, {quat_xyzw[2]:+.4f}, {quat_xyzw[3]:+.4f}]")
    print(sep)

    os.makedirs("/ros2_ws/results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": ts,
        "rank": best_idx + 1,
        "total_candidates": len(grasps_cam),
        "confidence": best_score,
        "score_range": [float(scores.min()), float(scores.max())],
        "position_xyz_m": translation.tolist(),
        "euler_xyz_deg": euler_deg.tolist(),
        "quaternion_xyzw": quat_xyzw.tolist(),
        "pose_matrix_4x4": pose.tolist(),
    }
    out_path = f"/ros2_ws/results/best_grasp_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {out_path}\n")
    return out


# ---------------------------------------------------------------------------
# Tkinter Application
# ---------------------------------------------------------------------------
class GraspGenApp:
    def __init__(self, root: tk.Tk, args, camera: OrbbecCamera,
                 config_map: dict, sam3_proc=None, vis=None):
        self.root = root
        self.args = args
        self.camera = camera
        self.config_map = config_map
        self.sam3_proc = sam3_proc

        self._loaded_config_name = None
        self._grasp_cfg = None
        self._sampler = None

        self._last_mask = None

        self._inference_running = False
        self._config_loading = False
        self._log_queue = queue.Queue()
        self._cb_queue: queue.Queue = queue.Queue()
        self._vis = vis
        self._show_mask_var = tk.BooleanVar(value=True)

        self._build_ui()

        if config_map:
            first = next(iter(config_map))
            if args.gripper_config:
                preferred = Path(args.gripper_config).name
                if preferred in config_map:
                    first = preferred
            self._config_combo.set(first)
            self._start_config_load(first)

        self._schedule_preview_update()
        self._schedule_log_flush()

    def _build_ui(self):
        self.root.title("GraspGen Pipeline")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(False, False)

        left = tk.Frame(self.root, bg="#1e1e1e")
        left.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=6)

        self._canvas = tk.Canvas(
            left, width=_PREVIEW_W, height=_PREVIEW_H,
            bg="#111111", highlightthickness=1, highlightbackground="#444",
        )
        self._canvas.pack()
        self._canvas_img_id = self._canvas.create_image(0, 0, anchor="nw")
        self._no_cam_txt = self._canvas.create_text(
            _PREVIEW_W // 2, _PREVIEW_H // 2,
            text="Waiting for camera…",
            fill="#555555", font=("Helvetica", 14),
        )

        self._mask_chk = tk.Checkbutton(
            left,
            text="Show mask overlay",
            variable=self._show_mask_var,
            bg="#1e1e1e", fg="#aaa",
            activebackground="#1e1e1e", activeforeground="white",
            selectcolor="#2d2d2d",
        )
        self._mask_chk.pack(anchor="w", padx=4, pady=(4, 0))

        right = tk.Frame(self.root, bg="#2d2d2d", width=340)
        right.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=6)
        right.grid_propagate(False)

        tk.Label(right, text="GraspGen", bg="#2d2d2d", fg="#61afef",
                 font=("Helvetica", 16, "bold")).pack(pady=(14, 0))
        tk.Label(right, text="SAM3  ·  Orbbec Gemini 2  ·  GraspGen",
                 bg="#2d2d2d", fg="#666", font=("Helvetica", 9)).pack(pady=(0, 10))

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=2)

        tk.Label(right, text="Gripper / Tool:", bg="#2d2d2d", fg="#ccc",
                 font=("Helvetica", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 2))

        names = list(self.config_map.keys())
        self._config_combo = ttk.Combobox(
            right, values=names, state="readonly", font=("Helvetica", 9), width=36,
        )
        self._config_combo.pack(padx=12, pady=(0, 4), fill="x")
        if not names:
            self._config_combo.set("(no configs found)")
            self._config_combo.configure(state="disabled")
        self._config_combo.bind("<<ComboboxSelected>>", self._on_config_change)

        self._config_hint = tk.Label(
            right, text="", bg="#2d2d2d", fg="#555",
            font=("Courier", 7), wraplength=300, justify="left",
        )
        self._config_hint.pack(anchor="w", padx=12, pady=(0, 6))

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=2)

        tk.Label(right, text="Object prompt:", bg="#2d2d2d", fg="#ccc",
                 font=("Helvetica", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 2))

        self._prompt_var = tk.StringVar(value=self.args.sam3_prompt or "")
        self._prompt_entry = tk.Entry(
            right, textvariable=self._prompt_var,
            bg="#3a3a3a", fg="white", insertbackground="white",
            relief="flat", font=("Helvetica", 11), bd=4,
        )
        self._prompt_entry.pack(padx=12, pady=(0, 10), fill="x")
        self._prompt_entry.bind("<Return>", lambda _e: self._on_run())

        self._run_btn = tk.Button(
            right, text="▶  Capture & Run GraspGen",
            bg="#61afef", fg="#1e1e1e", activebackground="#4d9bd6",
            font=("Helvetica", 11, "bold"),
            relief="flat", cursor="hand2", bd=0,
            command=self._on_run,
        )
        self._run_btn.pack(padx=12, pady=4, fill="x", ipady=10)

        tk.Button(
            right, text="✕  Clear mask overlay",
            bg="#3a3a3a", fg="#aaa", activebackground="#444",
            font=("Helvetica", 9),
            relief="flat", cursor="hand2", bd=0,
            command=self._clear_mask,
        ).pack(padx=12, pady=(2, 8), fill="x", ipady=4)

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=4)

        tk.Label(right, text="Status:", bg="#2d2d2d", fg="#666",
                 font=("Helvetica", 9)).pack(anchor="w", padx=12)
        self._status_var = tk.StringVar(value="Waiting for camera…")
        tk.Label(
            right, textvariable=self._status_var,
            bg="#2d2d2d", fg="#98c379",
            font=("Helvetica", 9), wraplength=300, justify="left",
        ).pack(anchor="w", padx=12, pady=(2, 8))

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=2)

        tk.Label(right, text="Log:", bg="#2d2d2d", fg="#666",
                 font=("Helvetica", 9)).pack(anchor="w", padx=12)
        self._log_text = scrolledtext.ScrolledText(
            right, width=38, height=14,
            bg="#1e1e1e", fg="#abb2bf",
            font=("Courier", 8), state="disabled", relief="flat",
        )
        self._log_text.pack(padx=12, pady=(2, 8), fill="both", expand=True)

        tk.Label(
            right,
            text=f"SAM3: {self.args.sam3_socket}   Meshcat: http://127.0.0.1:7000",
            bg="#2d2d2d", fg="#444", font=("Courier", 7),
        ).pack(anchor="w", padx=12, pady=(0, 8))

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

    def _start_config_load(self, name: str):
        if self._config_loading or name == self._loaded_config_name:
            return
        self._config_loading = True
        self.root.after(0, lambda: self._run_btn.configure(
            state="disabled", text="Loading model…"
        ))
        self._set_status(f"Loading {name}…")
        threading.Thread(
            target=self._load_grasp_config, args=(name,), daemon=True
        ).start()

    def _load_grasp_config(self, name: str):
        path = self.config_map.get(name)
        if not path:
            self._log(f"[Config] Unknown config: {name}")
            self._config_loading = False
            self._cb_queue.put(self._restore_run_btn)
            return
        try:
            if self._sampler is not None:
                self._log("[Config] Releasing previous model from GPU…")
                try:
                    del self._sampler
                except Exception:
                    pass
                self._sampler = None
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()

            self._log(f"[Config] Loading {name}…")
            new_cfg = load_grasp_cfg(path)
            new_sampler = GraspGenSampler(new_cfg)

            self._grasp_cfg = new_cfg
            self._sampler = new_sampler
            self._loaded_config_name = name

            self._cb_queue.put(lambda p=path: self._config_hint.configure(text=p))
            self._log(f"[Config] Ready: {new_cfg.data.gripper_name}")
            self._set_status(f"Ready — {name}")
        except Exception as e:
            self._log(f"[Config] ERROR loading {name}: {e}")
            self._set_status(f"Config error: {e}")
            try:
                del new_sampler
            except Exception:
                pass
            self._grasp_cfg = None
            self._sampler = None
            self._loaded_config_name = None
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
        finally:
            self._config_loading = False
            self._cb_queue.put(self._restore_run_btn)

    def _restore_run_btn(self):
        if not self._inference_running and not self._config_loading:
            self._run_btn.configure(state="normal", text="▶  Capture & Run GraspGen")

    def _on_config_change(self, _event=None):
        name = self._config_combo.get()
        if not name or name == self._loaded_config_name:
            return
        if self._config_loading or self._inference_running:
            self._config_combo.set(self._loaded_config_name or "")
            self._set_status("Busy — wait for current operation to finish.")
            return
        self._start_config_load(name)

    def _schedule_preview_update(self):
        self._update_preview()
        self.root.after(80, self._schedule_preview_update)

    def _update_preview(self):
        if not _PIL_OK:
            return
        rgb, _depth, _intr = self.camera.get_latest()
        if rgb is None:
            return

        self._canvas.itemconfig(self._no_cam_txt, state="hidden")
        display = rgb.copy()

        mask = self._last_mask
        if self._show_mask_var.get() and mask is not None:
            mh, mw = mask.shape[:2]
            dh, dw = display.shape[:2]
            if (mh, mw) != (dh, dw):
                mask = cv2.resize(
                    mask.astype(np.float32), (dw, dh),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.uint8)
            overlay = display.copy()
            overlay[mask > 0] = _MASK_GREEN_RGB
            cv2.addWeighted(display, 0.55, overlay, 0.45, 0, dst=display)

        h, w = display.shape[:2]
        scale = min(_PREVIEW_W / w, _PREVIEW_H / h)
        nw, nh = int(w * scale), int(h * scale)
        display = cv2.resize(display, (nw, nh), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((_PREVIEW_H, _PREVIEW_W, 3), dtype=np.uint8)
        yo = (_PREVIEW_H - nh) // 2
        xo = (_PREVIEW_W - nw) // 2
        padded[yo:yo + nh, xo:xo + nw] = display

        pil_img = PILImage.fromarray(padded)
        tk_img = ImageTk.PhotoImage(pil_img)
        self._canvas.itemconfig(self._canvas_img_id, image=tk_img)
        self._canvas._tk_img_ref = tk_img

        if self._status_var.get().startswith("Waiting for camera"):
            self._status_var.set("Camera ready.")

    def _clear_mask(self):
        self._last_mask = None

    def _log(self, msg: str):
        self._log_queue.put(msg)

    def _schedule_log_flush(self):
        self._flush_log()
        self._flush_cb_queue()
        self.root.after(200, self._schedule_log_flush)

    def _flush_cb_queue(self):
        try:
            while True:
                cb = self._cb_queue.get_nowait()
                cb()
        except queue.Empty:
            pass

    def _flush_log(self):
        msgs = []
        try:
            while True:
                msgs.append(self._log_queue.get_nowait())
        except queue.Empty:
            pass
        if msgs:
            self._log_text.configure(state="normal")
            for m in msgs:
                self._log_text.insert("end", m + "\n")
            self._log_text.see("end")
            self._log_text.configure(state="disabled")

    def _set_status(self, msg: str):
        self._cb_queue.put(lambda m=msg: self._status_var.set(m))

    def _on_run(self):
        if self._inference_running or self._config_loading:
            return

        prompt = self._prompt_var.get().strip()
        if not prompt:
            self._set_status("Enter an object prompt first.")
            return

        if self._sampler is None or self._grasp_cfg is None:
            self._set_status("No gripper config loaded — select one from the dropdown.")
            return

        rgb, depth_m, intrinsics = self.camera.get_latest()
        if rgb is None or depth_m is None:
            self._set_status("No camera frame available yet.")
            return

        self._inference_running = True
        self._run_btn.configure(state="disabled", text="Running…")
        self._set_status("Running pipeline…")
        self._last_mask = None

        threading.Thread(
            target=self._run_pipeline,
            args=(rgb.copy(), depth_m.copy(), intrinsics, prompt,
                  self._sampler, self._grasp_cfg),
            daemon=True,
        ).start()

    def _run_pipeline(self, rgb, depth_m, intrinsics, prompt, sampler, grasp_cfg):
        args = self.args
        try:
            self._log(f"[SAM3] prompt='{prompt}'")
            t0 = time.time()
            mask = segment_with_sam3_server(rgb, prompt, args.sam3_socket)
            n_px = int(mask.sum())
            self._log(f"[SAM3] {time.time()-t0:.2f}s — {n_px} px masked")

            if mask.shape != depth_m.shape:
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (depth_m.shape[1], depth_m.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.uint8)
                n_px = int(mask.sum())

            self._last_mask = mask

            if n_px < 50:
                raise RuntimeError(
                    f"Only {n_px} pixels masked — try a more specific prompt"
                )

            fx, fy, cx, cy = intrinsics
            scene_pc, object_pc, scene_colors, object_colors = \
                depth_and_segmentation_to_point_clouds(
                    depth_image=depth_m,
                    segmentation_mask=mask,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    rgb_image=rgb,
                    target_object_id=args.target_object_id,
                    remove_object_from_scene=True,
                )

            if len(object_pc) > args.max_object_points:
                idx = np.random.choice(len(object_pc), args.max_object_points, replace=False)
                object_pc = object_pc[idx]
                if object_colors is not None:
                    object_colors = object_colors[idx]

            pc_torch = torch.from_numpy(object_pc)
            pc_filtered, _ = point_cloud_outlier_removal(pc_torch)
            pc_filtered = pc_filtered.numpy()

            if len(pc_filtered) == 0:
                raise RuntimeError("Object point cloud is empty after outlier removal")

            t_center = tra.translation_matrix(-pc_filtered.mean(axis=0))
            pc_centered = tra.transform_points(pc_filtered, t_center)
            scene_centered = tra.transform_points(scene_pc, t_center)

            if self._vis is not None:
                self._vis.delete()
                make_frame(self._vis, "world", h=0.12, radius=0.004)

                _sc = scene_colors if scene_colors is not None else \
                    np.tile([[120, 120, 120]], (len(scene_pc), 1)).astype(np.uint8)
                _oc = np.tile([[255, 255, 255]], (len(pc_centered), 1)).astype(np.uint8)
                visualize_pointcloud(self._vis, "pc_scene", scene_centered, _sc,
                                     size=args.scene_point_size)
                visualize_pointcloud(self._vis, "pc_obj", pc_centered, _oc,
                                     size=args.object_point_size)

            self._log("[GraspGen] Running inference…")
            t1 = time.time()
            grasps, grasp_conf = GraspGenSampler.run_inference(
                pc_filtered, sampler,
                grasp_threshold=args.grasp_threshold,
                num_grasps=args.num_grasps,
                topk_num_grasps=args.topk_num_grasps,
            )
            self._log(f"[GraspGen] {time.time()-t1:.2f}s — {len(grasps)} grasps")

            if len(grasps) == 0:
                raise RuntimeError("GraspGen found no grasps for this point cloud")

            grasp_conf_np = grasp_conf.cpu().numpy()
            grasps_np = grasps.cpu().numpy()
            pc_centered, grasps_centered, scores, t_center = _process_point_cloud(
                pc_filtered, grasps_np, grasp_conf_np
            )

            best = _save_best_grasp(grasps_np, grasp_conf_np, t_center)
            gripper_name = grasp_cfg.data.gripper_name

            if self._vis is not None:
                if args.collision_filter:
                    gripper_info = get_gripper_info(gripper_name)
                    scene_col = scene_centered
                    if len(scene_centered) > args.max_scene_points:
                        idx = np.random.choice(
                            len(scene_centered), args.max_scene_points, replace=False
                        )
                        scene_col = scene_centered[idx]

                    free_mask = filter_colliding_grasps(
                        scene_pc=scene_col,
                        grasp_poses=grasps_centered,
                        gripper_collision_mesh=gripper_info.collision_mesh,
                        collision_threshold=args.collision_threshold,
                    )
                    for j, g in enumerate(grasps_centered[free_mask]):
                        visualize_grasp(self._vis, f"grasps/free/{j:03d}/grasp", g,
                                        color=scores[free_mask][j],
                                        gripper_name=gripper_name, linewidth=1.5)
                    for j, g in enumerate(grasps_centered[~free_mask][:40]):
                        visualize_grasp(self._vis, f"grasps/colliding/{j:03d}/grasp", g,
                                        color=[255, 0, 0],
                                        gripper_name=gripper_name, linewidth=0.4)
                    n_free = int(free_mask.sum())
                    self._log(f"[Collision] {n_free}/{len(free_mask)} grasps free")
                else:
                    for j, g in enumerate(grasps_centered):
                        visualize_grasp(self._vis, f"grasps/{j:03d}/grasp", g,
                                        color=scores[j],
                                        gripper_name=gripper_name, linewidth=1.2)

            conf_min = float(grasp_conf_np.min())
            conf_max = float(grasp_conf_np.max())
            best_conf = best.get("confidence", 0.0)
            best_z = best.get("position_xyz_m", [0, 0, 0])[2]
            summary = (
                f"{len(grasps_np)} grasps [{conf_min:.3f}–{conf_max:.3f}]\n"
                f"Best conf: {best_conf:.4f}  depth: {best_z:.3f}m\n"
                f"Meshcat: {getattr(self._vis, 'viewer_url', 'http://127.0.0.1:7000/static/')}"
            )
            self._log("─" * 36)
            self._log(summary)
            self._set_status(summary)

        except Exception as exc:
            err = str(exc)
            self._log(f"[ERROR] {err}")
            self._set_status(f"Error: {err}")

        finally:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

            self._inference_running = False
            self._cb_queue.put(self._restore_run_btn)


# ---------------------------------------------------------------------------
# Simple configuration (no CLI arguments)
# ---------------------------------------------------------------------------
class _Args:
    gripper_config = None
    checkpoints_dir = CHECKPOINTS_DIR
    num_grasps = 200
    grasp_threshold = -1.0
    topk_num_grasps = 100
    return_topk = False
    collision_filter = False
    collision_threshold = 0.02
    max_scene_points = 8192
    max_object_points = 12000
    scene_point_size = 0.008
    object_point_size = 0.012
    target_object_id = 1
    sam3_prompt = ""
    sam3_socket = "/tmp/sam3_server.sock"
    sam3_device = "cuda:0"
    sam3_no_fp16 = False


def get_args():
    return _Args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = get_args()

    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    config_map = scan_checkpoints(args.checkpoints_dir)
    if config_map:
        print(f"[Config] Found {len(config_map)} config(s) in {args.checkpoints_dir}:", flush=True)
        for name in config_map:
            print(f"  {name}", flush=True)
    else:
        print(
            f"[Config] WARNING: No .yml files found in '{args.checkpoints_dir}'.\n"
            "         Edit _Args.checkpoints_dir or _Args.gripper_config to point\n"
            "         to your GraspGen .yml files.",
            flush=True,
        )
        if args.gripper_config and os.path.isfile(args.gripper_config):
            name = Path(args.gripper_config).name
            config_map = {name: args.gripper_config}

    sam3_proc = None
    print(f"[SAM3] Checking server at '{args.sam3_socket}'…", flush=True)
    try:
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(args.sam3_socket)
        s.close()
        print("[SAM3] Connected to existing server.", flush=True)
    except OSError:
        print("[SAM3] No server found, auto-starting sam3_server.py…", flush=True)
        sam3_proc = start_sam3_server(
            args.sam3_socket,
            device=args.sam3_device,
            fp16=not args.sam3_no_fp16,
        )

    vis = None
    try:
        vis = create_visualizer()
        meshcat_url = getattr(vis, "viewer_url", "http://127.0.0.1:7000/static/")
        print(f"[Meshcat] Visualizer ready → {meshcat_url}", flush=True)
    except Exception as e:
        print(f"[Meshcat] WARNING: could not create visualizer: {e}", flush=True)
        print("[Meshcat] Grasps will not be visualized.", flush=True)

    print("[Camera] Starting Orbbec pipeline…", flush=True)
    camera = OrbbecCamera()
    camera.start()
    print("[Camera] Live stream running.", flush=True)

    if not _PIL_OK:
        print(
            "[UI] WARNING: Pillow not found — camera preview will be blank.\n"
            "     pip install Pillow",
            flush=True,
        )

    root = tk.Tk()
    GraspGenApp(root, args, camera, config_map, sam3_proc=sam3_proc, vis=vis)

    def _on_close():
        print("Shutting down…", flush=True)
        camera.stop()
        if sam3_proc is not None:
            try:
                sam3_proc.terminate()
                sam3_proc.wait(timeout=5)
            except Exception:
                try:
                    sam3_proc.kill()
                except Exception:
                    pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)

    print(
        "UI ready.\n"
        "  • Select a gripper from the dropdown.\n"
        "  • Type a prompt and click 'Capture & Run'.\n"
        f"  • Meshcat: {getattr(vis, 'viewer_url', 'http://127.0.0.1:7000/static/')}",
        flush=True,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
