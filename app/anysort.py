#!/usr/bin/env python3
"""
AnySort — Grasp Execute Pipeline
=================================
Full pipeline: Camera → SAM3 → GraspGen → hand-eye transform → Dobot execution.

Single-command launch (SAM3 + Meshcat auto-start):
  cd /ros2_ws/app && python anysort.py

Or from Windows: double-click AnySort.vbs at repo root.
"""

import gc
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import trimesh.transformations as tra

try:
    from PIL import Image as PILImage, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    from scipy.spatial.transform import Rotation
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    print("[WARN] scipy not found — rotation conversion limited")

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.robot import get_gripper_info
from grasp_gen.utils.meshcat_utils import (
    create_visualizer, get_color_from_score, make_frame,
    visualize_grasp, visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import (
    depth_and_segmentation_to_point_clouds,
    filter_colliding_grasps,
    point_cloud_outlier_removal,
)

# ===========================================================================
# Constants
# ===========================================================================
CHECKPOINTS_DIR  = "/opt/GraspGen/GraspGenModels/checkpoints"
SAM3_SERVER_SCRIPT = "/ros2_ws/app/sam3_server.py"
CALIB_FILE       = "/ros2_ws/data/calibration/hand_eye_calib.npz"
RESULTS_DIR      = Path("/ros2_ws/results")
ROI_SAVE_PATH    = Path(__file__).parent / "pipeline_roi.json"
POSITIONS_SAVE_PATH  = Path(__file__).parent / "pipeline_positions.json"
OBJECT_LISTS_DIR     = Path(__file__).parent.parent.parent / "data" / "object_lists"

ROBOT_IP_DEFAULT = "192.168.5.1"
APPROACH_OFFSET  = 40       # mm above grasp position for pre-grasp approach
HOME_POSE        = [300, 0, 450, 0, 0, 0]  # [X, Y, Z, Rx, Ry, Rz] safe home

_PREVIEW_W  = 640
_PREVIEW_H  = 480
_MASK_GREEN = np.array([0, 210, 90], dtype=np.uint8)

_SAM3_PYTHON_CANDIDATES = [
    "/opt/sam3env/bin/python3.12",
    "/opt/sam3env/bin/python3",
    "/opt/sam3env/bin/python",
]

# ===========================================================================
# SAM3 server helpers  (identical to demo script)
# ===========================================================================
def _find_sam3_python():
    for p in _SAM3_PYTHON_CANDIDATES:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise RuntimeError("Cannot find SAM3 Python interpreter.")


def _wait_for_sam3_socket(sock_path, timeout=600.0, proc=None):
    """Wait for the SAM3 server Unix socket to appear and accept connections.

    The socket is created only AFTER the model finishes loading, which can take
    several minutes on first run (downloading facebook/sam3 weights).
    """
    import socket as _s
    deadline = time.time() + timeout
    last_log = time.time()
    while time.time() < deadline:
        # Fail fast if the subprocess already died
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"SAM3 server process exited early (code {proc.returncode})")
        try:
            s = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM)
            s.settimeout(2.0); s.connect(sock_path); s.close(); return
        except OSError:
            pass
        now = time.time()
        if now - last_log >= 30.0:
            elapsed = now - (deadline - timeout)
            print(f"[SAM3] Still loading model… ({elapsed:.0f}s elapsed, "
                  f"timeout={timeout:.0f}s) — waiting for {sock_path}")
            last_log = now
        time.sleep(0.5)
    raise RuntimeError(f"SAM3 server not ready within {timeout:.0f}s")


def start_sam3_server(sock_path, device="cuda:0", fp16=True):
    python_bin = _find_sam3_python()
    cmd = [python_bin, SAM3_SERVER_SCRIPT, "--socket", sock_path, "--device", device]
    if not fp16:
        cmd.append("--no-fp16")
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None); env.pop("PYTHONHOME", None)
    print(f"[SAM3] Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env)
    _wait_for_sam3_socket(sock_path, timeout=600.0, proc=proc)
    print("[SAM3] Server ready.")
    return proc


def _recv_exactly(s, n):
    chunks, received = [], 0
    while received < n:
        chunk = s.recv(min(n - received, 65536))
        if not chunk:
            raise ConnectionError("SAM3 disconnected")
        chunks.append(chunk); received += len(chunk)
    return b"".join(chunks)


def _recv_json_line(s):
    buf = b""
    while True:
        byte = s.recv(1)
        if not byte:
            raise ConnectionError("SAM3 disconnected")
        if byte == b"\n":
            break
        buf += byte
    return json.loads(buf.decode())


def _select_largest_component(mask):
    if mask.sum() == 0:
        return mask, 0
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num <= 1:
        return mask, 0
    best = int(stats[1:, cv2.CC_STAT_AREA].argmax()) + 1
    return (labels == best).astype(np.uint8), num - 1


def segment_with_sam3(rgb, prompt, sock_path):
    import socket as _s
    h, w = rgb.shape[:2]
    rgb_bytes = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    header = json.dumps({"width": w, "height": h, "prompt": prompt,
                          "size": len(rgb_bytes)}) + "\n"
    s = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM)
    s.settimeout(90.0); s.connect(sock_path)
    s.sendall(header.encode()); s.sendall(rgb_bytes)
    resp = _recv_json_line(s)
    if not resp.get("ok"):
        raise RuntimeError(f"SAM3 error: {resp.get('error', resp)}")
    mask_bytes = _recv_exactly(s, resp["size"]) if resp["size"] > 0 else b""
    s.close()
    n_masks = resp.get("num_masks", 1)
    if n_masks == 0 or not mask_bytes:
        return np.zeros((h, w), dtype=np.uint8)
    all_masks = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(n_masks, h, w)
    best, _ = _select_largest_component((all_masks[0] > 0).astype(np.uint8))
    return best


# ===========================================================================
# Orbbec camera  (identical to demo script)
# ===========================================================================
def _to_rgb_array(color_frame, OBFormat):
    from io import BytesIO
    h, w = color_frame.get_height(), color_frame.get_width()
    raw = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    fmt = color_frame.get_format()
    fmt_s = str(fmt).upper()
    if hasattr(OBFormat, "RGB") and fmt == OBFormat.RGB:
        return raw.reshape(h, w, 3).copy()
    if hasattr(OBFormat, "BGR") and fmt == OBFormat.BGR:
        return raw.reshape(h, w, 3)[:, :, ::-1].copy()
    if (hasattr(OBFormat, "MJPG") and fmt == OBFormat.MJPG) or "MJPG" in fmt_s:
        return np.array(PILImage.open(BytesIO(raw.tobytes())).convert("RGB"))
    if (hasattr(OBFormat, "YUYV") and fmt == OBFormat.YUYV) or "YUYV" in fmt_s:
        return cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUY2)
    if raw.size == h * w * 3:
        return raw.reshape(h, w, 3).copy()
    raise RuntimeError(f"Unsupported format {fmt}")


def _extract_intrinsics(profile_or_frame):
    """Try to read (fx, fy, cx, cy) from a stream profile or frame object."""
    if profile_or_frame is None:
        raise RuntimeError("Cannot read intrinsics from None")
    for method in ["get_intrinsic", "get_camera_intrinsic", "get_intrinsics"]:
        if not hasattr(profile_or_frame, method):
            continue
        intr = getattr(profile_or_frame, method)()
        try:
            fx = float(next(getattr(intr, a) for a in ["fx","focal_x"] if hasattr(intr,a)))
            fy = float(next(getattr(intr, a) for a in ["fy","focal_y"] if hasattr(intr,a)))
            cx = float(next(getattr(intr, a) for a in ["cx","ppx","principal_x"] if hasattr(intr,a)))
            cy = float(next(getattr(intr, a) for a in ["cy","ppy","principal_y"] if hasattr(intr,a)))
            return fx, fy, cx, cy
        except StopIteration:
            continue
    raise RuntimeError("Cannot read camera intrinsics")


class OrbbecCamera:
    """Orbbec Gemini 2 camera with hardware D2C alignment and frame sync.

    After start(), depth is hardware-aligned to the color sensor so every
    depth pixel corresponds 1:1 to the same color pixel.  Intrinsics are
    taken from the **color** profile (the alignment target).
    """

    def __init__(self):
        from pyorbbecsdk import (Config, OBAlignMode, OBFormat,
                                 OBSensorType, Pipeline)
        self._OBFormat = OBFormat
        self._OBAlignMode = OBAlignMode
        self._OBSensorType = OBSensorType
        self._started = False
        self._color_profile = None    # intrinsics come from colour (D2C target)
        self._depth_filters = []      # SDK post-processing filters
        self._pipeline = None

    # ------------------------------------------------------------------ #
    def start(self):
        from pyorbbecsdk import Config, OBAlignMode, OBFormat, OBSensorType, Pipeline
        pipeline = Pipeline()

        # ── Enable hardware frame sync to avoid timestamp anomalies ──
        try:
            pipeline.enable_frame_sync()
            print("[Camera] Frame sync enabled")
        except Exception as e:
            print(f"[Camera] Frame sync failed: {e}")

        config = Config()

        # ── Find an RGB colour profile and a matching HW-D2C depth profile ──
        # IMPORTANT: prefer 1280×720 to match the hand-eye and intrinsics calibration.
        # Using a different resolution with calibrated K (cx/cy) causes deformed point clouds.
        PREFERRED_W, PREFERRED_H = 1280, 720
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        hw_d2c_ok = False

        def _try_color_profile(cp):
            """Try to pair a color profile with a Y16 D2C depth profile.
            Returns (cp, dp) on success, None on failure."""
            if cp.get_format() != OBFormat.RGB:
                return None
            try:
                d2c_list = pipeline.get_d2c_depth_profile_list(cp, OBAlignMode.HW_MODE)
            except Exception:
                return None
            if len(d2c_list) == 0:
                return None
            dp = None
            for j in range(len(d2c_list)):
                if d2c_list[j].get_format() == OBFormat.Y16:
                    dp = d2c_list[j]
                    break
            if dp is None:
                dp = d2c_list[0]
            return (cp, dp)

        # Pass 1: try to find 1280×720 RGB with HW D2C
        chosen = None
        for i in range(len(color_profiles)):
            cp = color_profiles[i]
            if cp.get_width() != PREFERRED_W or cp.get_height() != PREFERRED_H:
                continue
            result = _try_color_profile(cp)
            if result:
                chosen = result
                break

        # Pass 2: fall back to any RGB resolution with HW D2C
        if chosen is None:
            print(f"[Camera] WARNING: no {PREFERRED_W}×{PREFERRED_H} RGB+D2C profile found — "
                  f"trying other resolutions (point cloud may not match calibration!)")
            for i in range(len(color_profiles)):
                result = _try_color_profile(color_profiles[i])
                if result:
                    chosen = result
                    break

        if chosen is not None:
            cp, dp = chosen
            config.enable_stream(dp)
            config.enable_stream(cp)
            config.set_align_mode(OBAlignMode.HW_MODE)
            self._color_profile = cp
            hw_d2c_ok = True
            print(f"[Camera] HW D2C: color {cp.get_width()}x{cp.get_height()} "
                  f"depth {dp.get_width()}x{dp.get_height()}"
                  + (" OK matches calibration" if cp.get_width() == PREFERRED_W else " [!] resolution mismatch"))

        if not hw_d2c_ok:
            # Fallback: default profiles, no alignment
            print("[Camera] WARNING: HW D2C unavailable — falling back to defaults")
            depth_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            dp = None
            for j in range(len(depth_list)):
                if depth_list[j].get_format() == OBFormat.Y16:
                    dp = depth_list[j]
                    break
            if dp is None:
                dp = depth_list.get_default_video_stream_profile()
            config.enable_stream(dp)
            color_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            cp = color_list.get_default_video_stream_profile()
            config.enable_stream(cp)
            self._color_profile = cp

        # Start briefly to discover depth filters, then stop immediately.
        # The pipeline is only running during capture_frame() calls.
        pipeline.start(config)

        # ── Depth post-processing: selective filter enable ──
        #
        # DISABLED — HoleFilling: interpolates across invalid pixels → creates
        #   smooth gradients on flat surfaces ("hill" artefact).
        #
        # DISABLED — Temporal: averages depth across consecutive frames.
        #   Fine for a live stream, but this camera runs start-stop per capture.
        #   On restart the filter retains stale state from the previous scene,
        #   so a newly placed object appears at a blended intermediate depth
        #   until the filter accumulates enough frames to converge.
        #
        # KEPT — Spatial (bilateral-style): reduces per-frame sensor noise
        #   within a single frame without blending across time or bridging
        #   depth discontinuities.
        try:
            device = pipeline.get_device()
            sensor = device.get_sensor(OBSensorType.DEPTH_SENSOR)
            filters = sensor.get_recommended_filters()
            for f in filters:
                fname = f.get_name().lower()
                skip = any(kw in fname for kw in
                           ("hole", "fill", "inpaint", "temporal", "time"))
                if skip:
                    f.enable(False)
                    print(f"[Camera] Depth filter: {f.get_name()} (DISABLED)")
                else:
                    f.enable(True)
                    print(f"[Camera] Depth filter: {f.get_name()} (enabled)")
            self._depth_filters = filters
        except Exception as e:
            print(f"[Camera] Depth filters unavailable: {e}")
            self._depth_filters = []

        # Stabilise sensor (15 warm-up frames)
        print("[Camera] Stabilising sensor (15 frames)…")
        for _ in range(15):
            try:
                pipeline.wait_for_frames(200)
            except Exception:
                pass

        # Stop immediately — pipeline only runs during capture_frame()
        # This prevents the SDK from streaming USB data continuously
        # (which causes usbipd memory leak on Windows/WSL2).
        pipeline.stop()

        self._pipeline = pipeline
        self._config = config          # stored for re-start in capture_frame()
        self._started = True
        print("[Camera] Ready (idle — streams only during capture)")

    # ------------------------------------------------------------------ #
    def capture_frame(self, timeout_ms=3000):
        """Start the pipeline, grab one frame, stop the pipeline.
        Blocking — call from a worker thread.
        Returns (rgb uint8, depth_m float32, intrinsics tuple) or raises RuntimeError."""
        OBFormat = self._OBFormat

        # Re-start the pipeline for this capture
        self._pipeline.start(self._config)
        # Warm-up: discard frames while AE / depth processor settles.
        # 10 frames at ~30 fps ≈ 330 ms — fast enough to be unnoticeable but
        # sufficient for exposure and the spatial filter to converge on the
        # current scene without temporal filter interference.
        for _ in range(10):
            try:
                self._pipeline.wait_for_frames(200)
            except Exception:
                pass

        try:
            deadline = time.monotonic() + timeout_ms / 1000.0
            while time.monotonic() < deadline:
                try:
                    frames = self._pipeline.wait_for_frames(200)
                except Exception:
                    continue
                if not frames:
                    continue
                cf = frames.get_color_frame()
                df = frames.get_depth_frame()
                if cf is None or df is None:
                    continue
                try:
                    rgb = _to_rgb_array(cf, OBFormat)
                except Exception:
                    continue
                try:
                    for filt in self._depth_filters:
                        if filt.is_enabled():
                            try:
                                df = filt.process(df)
                            except Exception:
                                pass
                    if hasattr(df, 'as_depth_frame'):
                        df = df.as_depth_frame()
                    dh, dw = df.get_height(), df.get_width()
                    depth_raw = np.frombuffer(df.get_data(),
                                              dtype=np.uint16).reshape(dh, dw)
                    scale = float(df.get_depth_scale())
                    depth_m = depth_raw.astype(np.float32) * scale
                    valid = depth_m[(depth_m > 0.05) & np.isfinite(depth_m)]
                    if valid.size > 0 and float(np.median(valid)) > 20.0:
                        depth_m /= 1000.0

                    if rgb.shape[0] != dh or rgb.shape[1] != dw:
                        rgb = np.array(PILImage.fromarray(rgb).resize(
                            (dw, dh), PILImage.BILINEAR))

                    intr = _extract_intrinsics(self._color_profile)
                    return rgb, depth_m, intr
                except Exception:
                    continue
            raise RuntimeError("Camera capture timed out — no valid frame received")
        finally:
            # ALWAYS stop the pipeline after capture to release USB bandwidth
            # and prevent usbipd memory leak.
            try:
                self._pipeline.stop()
            except Exception:
                pass

    def stop(self):
        self._started = False
        try:
            self._pipeline.stop()
        except Exception:
            pass


# ===========================================================================
# Modular robot drivers  — see app/robots/ for available backends
# ===========================================================================
sys.path.insert(0, os.path.dirname(__file__))  # ensure 'robots' package is importable
from robots import RobotBase, get_driver_names, get_available_drivers, create_robot
from tools import get_tool_names, create_tool


# ===========================================================================
# Hand-eye calibration helpers
# ===========================================================================
def load_calibration(path=CALIB_FILE):
    """Load T_cam2base (4×4).  Translation stored in mm."""
    data = np.load(path)
    T = data["T_cam2base"].copy()       # 4×4, t in mm
    K = data["camera_matrix"].copy() if "camera_matrix" in data else None
    return T, K


def transform_grasp_to_robot(grasp_4x4_cam: np.ndarray,
                              T_cam2base_mm: np.ndarray
                              ) -> tuple:
    """
    Transform a GraspGen pose (camera frame, meters) → robot base frame.

    Returns (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg) for Dobot MovJ.

    grasp_4x4_cam  : 4×4 float64, translation in METERS  (GraspGen output)
    T_cam2base_mm  : 4×4 float64, translation in MILLIMETRES (calibration output)
    """
    # Bring calibration translation to meters so units match
    T = T_cam2base_mm.copy()
    T[:3, 3] /= 1000.0                              # mm → m

    grasp_base = T @ grasp_4x4_cam                  # 4×4, t in meters

    # Position → mm for Dobot
    pos_mm = grasp_base[:3, 3] * 1000.0

    # Rotation → Euler ZYX degrees (Dobot convention)
    R = grasp_base[:3, :3]
    if _SCIPY_OK:
        rz, ry, rx = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)
    else:
        # Fallback: no rotation (point straight down)
        rx, ry, rz = 0.0, 0.0, 0.0

    return float(pos_mm[0]), float(pos_mm[1]), float(pos_mm[2]), \
           float(rx), float(ry), float(rz)


# ===========================================================================
# GraspGen helpers  (identical to demo)
# ===========================================================================
def scan_checkpoints(directory):
    d = Path(directory)
    if not d.exists():
        return {}
    return {p.name: str(p) for p in sorted(d.glob("*.yml"))}


def _process_point_cloud(pc, grasps, grasp_conf):
    scores = get_color_from_score(grasp_conf, use_255_scale=True)
    grasps[:, 3, 3] = 1
    t_center = tra.translation_matrix(-pc.mean(axis=0))
    pc_centered = tra.transform_points(pc, t_center)
    grasps_centered = t_center @ grasps   # (4,4) @ (N,4,4) → (N,4,4) via numpy broadcast
    return pc_centered, grasps_centered, scores, t_center


def _save_best_grasp(grasps_cam, conf) -> dict:
    """conf: 1-D float array of raw GraspGen confidence scores (grasp_conf_np)."""
    if len(grasps_cam) == 0:
        return {}
    conf = np.asarray(conf).ravel()          # ensure 1-D
    best_idx = int(np.argmax(conf))
    pose = grasps_cam[best_idx].copy()
    t = pose[:3, 3]
    R = pose[:3, :3]
    try:
        euler = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
        quat  = Rotation.from_matrix(R).as_quat()
    except Exception:
        euler = np.zeros(3); quat = np.array([0.,0.,0.,1.])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"timestamp": ts, "rank": best_idx+1, "total": len(grasps_cam),
           "confidence": float(conf[best_idx]),
           "score_range": [float(conf.min()), float(conf.max())],
           "position_xyz_m": t.tolist(), "euler_xyz_deg": euler.tolist(),
           "quat_xyzw": quat.tolist(), "pose_4x4": pose.tolist()}
    path = RESULTS_DIR / f"best_grasp_{ts}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


# ===========================================================================
# Application
# ===========================================================================
class GraspExecuteApp:

    def __init__(self, root: tk.Tk, args, camera: OrbbecCamera,
                 config_map: dict, sam3_proc=None, vis=None,
                 preloaded_config=None):
        self.root       = root
        self.args       = args
        self.camera     = camera
        self.config_map = config_map
        self.sam3_proc  = sam3_proc
        self._vis       = vis

        # Replay mode — (rgb uint8, depth_m float32, intrinsics tuple) or None for live
        self._replay_frame = None
        self._replay_label_var = tk.StringVar(value="")

        # GraspGen state — may be pre-populated from splash loader
        if preloaded_config:
            self._loaded_config_name = preloaded_config["name"]
            self._grasp_cfg  = preloaded_config["cfg"]
            self._sampler    = preloaded_config["sampler"]
        else:
            self._loaded_config_name = None
            self._grasp_cfg  = None
            self._sampler    = None
        self._last_mask  = None
        self._best_grasp_base: np.ndarray = None  # 4×4, robot base frame, meters
        self._best_grasp_info: dict = {}
        self._all_grasps_base: np.ndarray = None  # (N,4,4) sorted conf desc, base frame
        self._all_grasps_work: np.ndarray = None  # (N,4,4) Z-up work frame for vis
        self._all_grasps_centered: np.ndarray = None  # same, centered for meshcat
        self._all_grasp_scores = None             # (N,3) uint8 original colors
        self._current_grasp_idx: int = 0

        # Hand-eye calibration
        self._T_cam2base = None    # 4×4, t in mm
        self._calib_K    = None

        # Robot
        self._robot: RobotBase = None
        self._robot_connected = False

        # Pipeline options
        self._collision_var  = tk.BooleanVar(value=False)  # enable GraspGen collision check
        self._reach_var      = tk.BooleanVar(value=False)  # filter unreachable grasps
        self._debug_var      = tk.BooleanVar(value=False)  # step-by-step debug mode
        self._debug_event    = threading.Event()
        self._debug_event.set()  # start in "go" state

        # Flags
        self._inference_running = False
        self._config_loading    = False
        self._executing         = False
        self._show_mask         = tk.BooleanVar(value=True)
        self._show_mask.trace_add("write", lambda *_: self._refresh_canvas())
        self._current_frame     = None   # (rgb, depth_m, intrinsics) from last capture

        self._log_queue = queue.Queue()
        self._cb_queue  = queue.Queue()

        # ROI selection (4-point polygon)
        self._roi_poly_img: list = None  # [(x,y),...] image coords, 4 pts when complete
        self._roi_selecting: bool = False
        self._roi_canvas_pts: list = []  # canvas (cx,cy) accumulated during selection

        self._canvas_scale: float = 1.0
        self._canvas_xo: int = 0
        self._canvas_yo: int = 0

        # Orientation flip for vacuum gripper — set automatically when config is loaded
        self._flip_orient: bool = False

        # Retry on robot error — iterate through next best grasps
        self._retry_grasps_var = tk.BooleanVar(value=False)
        self._retry_stop_event = threading.Event()

        # Sort / drop position for object placement
        self._sort_joints = None
        self._home_joints = None

        # Batch word-list mode
        self._batch_running    = False
        self._batch_stop_event = threading.Event()
        self._last_list_path   = None

        self._build_ui()
        self._load_calibration()
        self._load_roi()
        self._load_positions()
        self._start_config_if_available()
        self._schedule_flush()

        # Auto-capture first frame if camera is already live
        if self.camera._started:
            self.root.after(800, self._auto_capture_on_startup)

    # ------------------------------------------------------------------
    # Startup helpers
    # ------------------------------------------------------------------
    def _save_roi(self):
        """Persist the current ROI polygon to disk."""
        try:
            if self._roi_poly_img:
                ROI_SAVE_PATH.write_text(json.dumps({"poly": self._roi_poly_img}))
            else:
                ROI_SAVE_PATH.write_text(json.dumps({"poly": None}))
        except Exception as e:
            self._log(f"[ROI] Could not save ROI: {e}")

    def _load_roi(self):
        """Restore a previously saved ROI polygon."""
        try:
            if not ROI_SAVE_PATH.exists():
                return
            data = json.loads(ROI_SAVE_PATH.read_text())
            poly = data.get("poly")
            if poly and len(poly) == 4:
                self._roi_poly_img = [tuple(p) for p in poly]
                xs = [p[0] for p in self._roi_poly_img]
                ys = [p[1] for p in self._roi_poly_img]
                self._roi_info_var.set(
                    f"ROI poly  bbox ({min(xs)},{min(ys)})→({max(xs)},{max(ys)})  [restored]")
                self._log(f"[ROI] Restored saved polygon: {self._roi_poly_img}")
        except Exception as e:
            self._log(f"[ROI] Could not load saved ROI: {e}")

    def _auto_capture_on_startup(self):
        """Capture one frame automatically when the camera is already live at startup."""
        if self.camera._started and self._current_frame is None:
            self._log("[Camera] Auto-capturing startup frame…")
            threading.Thread(target=self._capture_frame_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Position persistence (home + sort joint angles)
    # ------------------------------------------------------------------
    def _save_positions(self):
        """Persist home/sort joints, word list, and last list path to disk."""
        try:
            data = {}
            if self._home_joints:
                data["home_joints"] = self._home_joints
            if self._sort_joints:
                data["sort_joints"] = self._sort_joints
            if self._last_list_path:
                data["last_list_path"] = self._last_list_path
            data["tcp_z_offset"] = float(self._tcp_z_var.get())
            try:
                words = list(self._batch_listbox.get(0, "end"))
                if words:
                    data["word_list"] = words
            except Exception:
                pass
            POSITIONS_SAVE_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            self._log(f"[Positions] Could not save: {e}")

    def _load_positions(self):
        """Restore saved joint positions, word list, and last list path from disk."""
        try:
            if not POSITIONS_SAVE_PATH.exists():
                return
            data = json.loads(POSITIONS_SAVE_PATH.read_text())
            if "home_joints" in data:
                self._home_joints = data["home_joints"]
                self._log(f"[Positions] Restored home: "
                          f"[{', '.join(f'{v:.1f}' for v in self._home_joints)}]")
            if "sort_joints" in data:
                self._sort_joints = data["sort_joints"]
                self._log(f"[Positions] Restored sort: "
                          f"[{', '.join(f'{v:.1f}' for v in self._sort_joints)}]")
                try:
                    self._sort_status.config(
                        text=f"Sort: [{', '.join(f'{v:.0f}' for v in self._sort_joints)}]",
                        fg="#98c379")
                except Exception:
                    pass
            if "tcp_z_offset" in data:
                self._tcp_z_var.set(str(data["tcp_z_offset"]))
            # Restore word list: prefer last saved file, fall back to embedded list
            last_path = data.get("last_list_path")
            if last_path and Path(last_path).exists():
                try:
                    words = [w.strip() for w in
                             Path(last_path).read_text().splitlines() if w.strip()]
                    for w in words:
                        self._batch_listbox.insert("end", w)
                    self._last_list_path = last_path
                    self._list_file_label.config(
                        text=Path(last_path).name, fg="#98c379")
                    self._log(f"[Positions] Restored list from {Path(last_path).name}"
                              f" ({len(words)} word(s))")
                except Exception as _le:
                    self._log(f"[Positions] Could not reload list file: {_le}")
            elif "word_list" in data:
                for w in data["word_list"]:
                    self._batch_listbox.insert("end", w)
                self._log(f"[Positions] Restored {len(data['word_list'])} word(s) from cache")
        except Exception as e:
            self._log(f"[Positions] Could not load: {e}")

    def _on_stop_retry(self):
        self._retry_stop_event.set()
        self._log("[Retry] Stop requested — will halt after current attempt.")
        self._stop_retry_btn.config(state="disabled")

    def _load_calibration(self):
        path = self.args.calib_file
        try:
            T, K = load_calibration(path)
            self._T_cam2base = T
            self._calib_K    = K
            t = T[:3, 3]
            self._log(f"[Calib] Loaded {path}")
            self._log(f"[Calib] t_cam2base = [{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}] mm")
            self._cb_queue.put(lambda: self._calib_status.config(
                text=f"OK  Calibration loaded  t=[{t[0]:.0f},{t[1]:.0f},{t[2]:.0f}]mm",
                fg="#98c379"))
        except Exception as e:
            self._log(f"[Calib] WARNING: {e}")
            self._cb_queue.put(lambda: self._calib_status.config(
                text="[!]  No calibration -- run hand_eye_calibration.py first",
                fg="#e06c75"))
        # Load calibrated camera intrinsics (K + distortion) from camera_calibration.py
        self._calib_dist = np.zeros(5, np.float64)
        intrinsics_path = Path(path).parent / "camera_intrinsics.npz"
        try:
            ci = np.load(str(intrinsics_path))
            self._calib_dist = ci["dist_coeffs"].ravel().astype(np.float64)
            if self._calib_K is None and "camera_matrix" in ci:
                self._calib_K = ci["camera_matrix"].astype(np.float64)
            self._log(f"[Calib] Loaded camera intrinsics: dist={self._calib_dist.round(4).tolist()}")
        except Exception as e:
            self._log(f"[Calib] camera_intrinsics.npz not found — using SDK intrinsics (no distortion correction)")

    @staticmethod
    def _calib_label(stem: str) -> str:
        """Convert a file stem to a short display label."""
        if stem == "hand_eye_calib":
            return "(default)"
        if stem.startswith("hand_eye_calib_"):
            return stem[len("hand_eye_calib_"):]
        return stem

    def _refresh_calib_combo(self):
        """Scan data/calibration/ for .npz files and populate combo with short labels."""
        self._calib_file_map: dict = {}  # display label → filename
        calib_dir = Path(self.args.calib_file).parent
        if calib_dir.is_dir():
            for f in sorted(calib_dir.glob("hand_eye_calib*.npz")):
                label = self._calib_label(f.stem)
                self._calib_file_map[label] = f.name
        try:
            labels = list(self._calib_file_map.keys())
            self._calib_combo["values"] = labels
            current_label = self._calib_label(Path(self.args.calib_file).stem)
            if current_label in labels:
                self._calib_combo.set(current_label)
            elif labels:
                self._calib_combo.set(labels[0])
        except Exception:
            pass

    def _on_calib_selected(self, *_):
        """User picked a different calibration file from the combo."""
        selected = self._calib_combo.get()
        if not selected:
            return
        filename = self._calib_file_map.get(selected, selected)
        calib_dir = Path(self.args.calib_file).parent
        new_path = str(calib_dir / filename)
        if new_path == self.args.calib_file:
            return
        self.args.calib_file = new_path
        self._load_calibration()

    def _start_config_if_available(self):
        if self._loaded_config_name and self._sampler:
            # Already pre-loaded from splash — just update combo + hints
            self._config_combo.set(self._loaded_config_name)
            path = self.config_map.get(self._loaded_config_name, "")
            self._config_hint.configure(text=str(path))
            _gname = self._grasp_cfg.data.gripper_name if self._grasp_cfg else ""
            self._flip_orient = any(kw in _gname.lower()
                                    for kw in ("vacuum", "suction", "cup"))
            self._log(f"[Config] Pre-loaded: {self._loaded_config_name}")
            self._set_status(f"Ready — {self._loaded_config_name}")
            return
        if self.config_map:
            first = next(iter(self.config_map))
            self._config_combo.set(first)
            self._start_config_load(first)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    # ==================================================================
    # UI — top-level builder (delegates to per-panel helpers)
    # ==================================================================
    def _build_ui(self):
        self.root.title("AnySort")
        self.root.configure(bg="#181818")
        self.root.resizable(True, True)
        self.root.minsize(1280, 720)
        self.root.geometry("1700x960")

        # 5 content columns + 1 log row
        self.root.columnconfigure(0, weight=3, minsize=660)  # camera
        self.root.columnconfigure(1, weight=2, minsize=230)  # grasp
        self.root.columnconfigure(2, weight=2, minsize=230)  # robot
        self.root.columnconfigure(3, weight=2, minsize=230)  # word list
        self.root.columnconfigure(4, weight=2, minsize=230)  # execution
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0, minsize=118)

        _BG = ["#1e1e1e", "#252525", "#2a2a2a", "#252525", "#2a2a2a"]
        _PAD = [(6,2), (2,2), (2,2), (2,2), (2,6)]
        _panels = []
        for c, (bg, px) in enumerate(zip(_BG, _PAD)):
            f = tk.Frame(self.root, bg=bg)
            f.grid(row=0, column=c, sticky="nsew", padx=px, pady=(6,2))
            _panels.append(f)

        self._build_camera_panel(_panels[0])
        self._build_grasp_panel(_panels[1])
        self._build_robot_panel(_panels[2])
        self._build_list_panel(_panels[3])
        self._build_exec_panel(_panels[4])

        # ── Log bar (spans all columns, drag-resizable) ───────────────────
        self.root.rowconfigure(1, weight=0, minsize=140)

        _log_bar = tk.Frame(self.root, bg="#161616")
        _log_bar.grid(row=1, column=0, columnspan=5, sticky="nsew",
                      padx=6, pady=(0,6))
        _log_bar.rowconfigure(0, weight=0)
        _log_bar.rowconfigure(1, weight=1)
        _log_bar.columnconfigure(0, weight=1)

        # Drag handle — thin bar at the top of the log
        _drag_handle = tk.Frame(_log_bar, bg="#333", height=5, cursor="sb_v_double_arrow")
        _drag_handle.grid(row=0, column=0, sticky="ew")
        tk.Label(_log_bar, text="LOG", bg="#161616", fg="#444",
                 font=("Helvetica", 8, "bold"), anchor="w"
                 ).grid(row=0, column=0, sticky="w", padx=(6,0), pady=0)

        self._log_text = scrolledtext.ScrolledText(
            _log_bar, bg="#161616", fg="#6b737f",
            font=("Courier", 10), state="disabled",
            relief="flat", height=7, wrap="word")
        self._log_text.grid(row=1, column=0, sticky="nsew", padx=(4,4), pady=(0,2))

        # Drag-to-resize: dragging handle up/down changes log area height
        def _log_drag_start(e):
            self._log_drag_y0 = e.y_root
            self._log_drag_ms = self.root.rowconfigure(1)["minsize"]
            if isinstance(self._log_drag_ms, str):
                self._log_drag_ms = int(self._log_drag_ms) if self._log_drag_ms else 140
        def _log_drag_motion(e):
            dy = self._log_drag_y0 - e.y_root   # up = positive = grow log
            new_h = max(80, min(600, self._log_drag_ms + dy))
            self.root.rowconfigure(1, minsize=new_h)
        _drag_handle.bind("<Button-1>", _log_drag_start)
        _drag_handle.bind("<B1-Motion>", _log_drag_motion)

    # ------------------------------------------------------------------
    # Panel builders
    # ------------------------------------------------------------------
    def _build_camera_panel(self, p):
        bg = p["bg"]

        def _section(parent, title, fg="#abb2bf"):
            """Labelled sub-group frame with a thin top border."""
            outer = tk.Frame(parent, bg=bg)
            outer.pack(fill="x", padx=8, pady=(4, 2))
            hdr = tk.Frame(outer, bg=bg)
            hdr.pack(fill="x")
            tk.Label(hdr, text=title, bg=bg, fg=fg,
                     font=("Helvetica", 8, "bold")).pack(side="left")
            ttk.Separator(hdr, orient="horizontal").pack(side="left", fill="x",
                                                          expand=True, padx=(4, 0))
            inner = tk.Frame(outer, bg=bg)
            inner.pack(fill="x", pady=(3, 0))
            return inner

        # ── Panel title ─────────────────────────────────────────────────
        tk.Label(p, text="Camera & Scene", bg=bg, fg="#61afef",
                 font=("Helvetica", 11, "bold")).pack(anchor="w", padx=10, pady=(8, 2))

        # ── Live preview canvas ─────────────────────────────────────────
        self._canvas = tk.Canvas(p, width=_PREVIEW_W, height=_PREVIEW_H,
                                  bg="#111", highlightthickness=1,
                                  highlightbackground="#333")
        self._canvas.pack(padx=8, pady=(0, 4))
        self._canvas_img_id = self._canvas.create_image(0, 0, anchor="nw")
        self._no_cam_txt = self._canvas.create_text(
            _PREVIEW_W//2, _PREVIEW_H//2,
            text="No camera  ·  connect or load a scene",
            fill="#444", font=("Helvetica", 13))
        self._roi_dot_ids = [
            self._canvas.create_oval(-6, -6, -6, -6, outline="#FFD700",
                                     fill="#FFD700", state="hidden")
            for _ in range(4)]
        self._roi_edge_ids = [
            self._canvas.create_line(0, 0, 0, 0, fill="#FFD700",
                                     width=2, state="hidden")
            for _ in range(4)]
        self._roi_preview_id = self._canvas.create_line(
            0, 0, 0, 0, fill="#FFD700", width=1, dash=(4, 2), state="hidden")
        self._canvas.bind("<ButtonPress-1>", self._on_roi_click)
        self._canvas.bind("<Motion>",        self._on_roi_hover)

        # ── Sub-group: Camera ───────────────────────────────────────────
        cam = _section(p, "Camera", fg="#61afef")

        r_cam = tk.Frame(cam, bg=bg); r_cam.pack(fill="x", pady=(0, 3))
        self._cam_connect_btn = tk.Button(
            r_cam, text="Connect Camera",
            bg="#3a3a3a", fg="#61afef", activebackground="#4a4a4a",
            relief="flat", cursor="hand2", bd=0, font=("Helvetica", 8),
            command=self._on_camera_connect)
        self._cam_connect_btn.pack(side="left", ipady=4, ipadx=6)
        tk.Button(r_cam, text="Capture",
                  bg="#3a3a3a", fg="#98c379", activebackground="#4a4a4a",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 8),
                  command=self._on_capture_frame
                  ).pack(side="left", ipady=4, ipadx=6, padx=(4, 0))
        self._cam_status_var = tk.StringVar(
            value="-- Connected" if self.camera._started else "-- No camera")
        self._cam_status_lbl = tk.Label(
            r_cam, textvariable=self._cam_status_var, bg=bg,
            fg="#98c379" if self.camera._started else "#e06c75",
            font=("Helvetica", 8))
        self._cam_status_lbl.pack(side="left", padx=(8, 0))

        # ── Sub-group: Scene ────────────────────────────────────────────
        scn = _section(p, "Scene", fg="#e5c07b")

        r_scn = tk.Frame(scn, bg=bg); r_scn.pack(fill="x", pady=(0, 2))
        for txt, fg_c, cmd in [
                ("Save Scene", "#98c379", self._on_save_scene),
                ("Load Scene", "#e5c07b", self._on_load_replay),
                ("[x] Clear",  "#aaa",    self._on_clear_replay)]:
            tk.Button(r_scn, text=txt, bg="#3a3a3a", fg=fg_c,
                      activebackground="#4a4a4a", relief="flat",
                      cursor="hand2", bd=0, font=("Helvetica", 8),
                      command=cmd
                      ).pack(side="left", ipady=3, ipadx=6, padx=(0, 4))
        self._replay_info = tk.Label(scn, textvariable=self._replay_label_var,
                                      bg=bg, fg="#c678dd",
                                      font=("Courier", 7), anchor="w")
        self._replay_info.pack(fill="x", pady=(1, 0))

        # ── Sub-group: Mask & ROI ───────────────────────────────────────
        roi = _section(p, "Mask & ROI", fg="#FFD700")

        r_mask = tk.Frame(roi, bg=bg); r_mask.pack(fill="x", pady=(0, 2))
        tk.Checkbutton(r_mask, text="Show mask overlay", variable=self._show_mask,
                       bg=bg, fg="#aaa", activebackground=bg,
                       selectcolor="#2d2d2d", font=("Helvetica", 8)
                       ).pack(side="left")

        r_roi = tk.Frame(roi, bg=bg); r_roi.pack(fill="x", pady=(0, 2))
        self._roi_btn = tk.Button(
            r_roi, text="[ ] Select ROI",
            bg="#3a3a3a", fg="#FFD700", activebackground="#4a4a4a",
            relief="flat", cursor="hand2", bd=0, font=("Helvetica", 8),
            command=self._on_roi_select)
        self._roi_btn.pack(side="left", ipady=3, ipadx=6)
        tk.Button(r_roi, text="[x] Clear ROI",
                  bg="#3a3a3a", fg="#aaa", activebackground="#4a4a4a",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 8),
                  command=self._on_roi_clear
                  ).pack(side="left", ipady=3, ipadx=6, padx=(4, 0))
        self._roi_info_var = tk.StringVar(value="")
        tk.Label(r_roi, textvariable=self._roi_info_var,
                 bg=bg, fg="#888", font=("Courier", 7)
                 ).pack(side="left", padx=(6, 0))

    def _build_grasp_panel(self, p):
        bg = p["bg"]
        def _sep(): ttk.Separator(p, orient="horizontal").pack(fill="x", padx=8, pady=4)
        def _lbl(t, fg="#e5c07b"):
            tk.Label(p, text=t, bg=bg, fg=fg,
                     font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10, pady=(4,1))

        tk.Label(p, text="[G]  GraspGen", bg=bg, fg="#98c379",
                 font=("Helvetica", 11, "bold")).pack(anchor="w", padx=10, pady=(8,0))
        _sep()

        self._calib_status = tk.Label(p, text="Loading calibration...",
                                       bg=bg, fg="#888",
                                       font=("Courier", 8), wraplength=240, anchor="w")
        self._calib_status.pack(anchor="w", padx=10, pady=(0,2))

        # Calibration file selector
        r_cal = tk.Frame(p, bg=bg); r_cal.pack(fill="x", padx=10, pady=(0,4))
        tk.Label(r_cal, text="Calib:", bg=bg, fg="#ccc",
                 font=("Helvetica", 8), anchor="w").pack(side="left")
        self._calib_combo = ttk.Combobox(r_cal, font=("Helvetica", 8), width=28)
        self._calib_combo.pack(side="left", fill="x", expand=True, ipady=1, padx=(4,3))
        self._calib_combo.bind("<<ComboboxSelected>>", self._on_calib_selected)
        self._refresh_calib_combo()
        _sep()

        _lbl("Gripper / Tool:")
        names = list(self.config_map.keys())
        self._config_combo = ttk.Combobox(p, values=names, state="readonly",
                                           font=("Helvetica", 9))
        self._config_combo.pack(padx=10, fill="x", ipady=2)
        if not names:
            self._config_combo.set("(no configs found)")
            self._config_combo.configure(state="disabled")
        self._config_combo.bind("<<ComboboxSelected>>", self._on_config_change)
        self._config_hint = tk.Label(p, text="", bg=bg, fg="#555",
                                      font=("Courier", 8), wraplength=230, justify="left")
        self._config_hint.pack(anchor="w", padx=10)
        _sep()

        _lbl("Object Prompt:")
        self._prompt_var = tk.StringVar(value="")
        self._prompt_entry = tk.Entry(p, textvariable=self._prompt_var,
                                       bg="#3a3a3a", fg="white",
                                       insertbackground="white",
                                       relief="flat", font=("Helvetica", 10), bd=4)
        self._prompt_entry.pack(padx=10, fill="x", pady=(0,4), ipady=2)
        self._prompt_entry.bind("<Return>", lambda _: self._on_run())
        _sep()

        _lbl("Options:")
        for var, txt in [(self._collision_var, "Check collisions"),
                         (self._reach_var,    "Filter unreachable (needs robot)"),
                         (self._debug_var,    "Step-by-step debug")]:
            tk.Checkbutton(p, text=txt, variable=var,
                           bg=bg, fg="#aaa", activebackground=bg,
                           selectcolor="#3a3a3a", font=("Helvetica", 9)
                           ).pack(anchor="w", padx=10, pady=1)
        _sep()

        _lbl("Inference Parameters:")
        r = tk.Frame(p, bg=bg); r.pack(fill="x", padx=10, pady=1)
        tk.Label(r, text="Grasps:", bg=bg, fg="#ccc",
                 font=("Helvetica", 9), anchor="w", width=13).pack(side="left")
        self._topk_grasps_var = tk.StringVar(value=str(self.args.topk_num_grasps))
        self._num_grasps_var  = self._topk_grasps_var   # same var — both use it
        tk.Entry(r, textvariable=self._topk_grasps_var, width=6,
                 bg="#3a3a3a", fg="white", insertbackground="white",
                 relief="flat", font=("Helvetica", 9)).pack(side="right")
        tk.Label(p, text="samples N candidates, returns best N by confidence",
                 bg=bg, fg="#555", font=("Helvetica", 8), wraplength=230,
                 justify="left").pack(anchor="w", padx=10)
        _sep()

        _lbl("Selected Grasp  (Robot Frame):")
        self._grasp_display = tk.Label(p, text="--  run GraspGen first",
                                        bg=bg, fg="#555",
                                        font=("Courier", 9), justify="left",
                                        wraplength=230, anchor="w")
        self._grasp_display.pack(anchor="w", padx=10, pady=(0,4))
        nav = tk.Frame(p, bg=bg); nav.pack(fill="x", padx=10)
        self._prev_grasp_btn = tk.Button(
            nav, text="<  Prev", width=7,
            bg="#3a3a3a", fg="#ccc", activebackground="#444",
            relief="flat", cursor="hand2", bd=0,
            font=("Helvetica", 9), state="disabled",
            command=self._on_prev_grasp)
        self._prev_grasp_btn.pack(side="left")
        self._grasp_idx_label = tk.Label(nav, text="-- / --", bg=bg, fg="#abb2bf",
                                          font=("Courier", 10, "bold"))
        self._grasp_idx_label.pack(side="left", expand=True)
        self._next_grasp_btn = tk.Button(
            nav, text="Next  >", width=7,
            bg="#3a3a3a", fg="#ccc", activebackground="#444",
            relief="flat", cursor="hand2", bd=0,
            font=("Helvetica", 9), state="disabled",
            command=self._on_next_grasp)
        self._next_grasp_btn.pack(side="right")

    def _build_robot_panel(self, p):
        bg = p["bg"]
        def _sep(): ttk.Separator(p, orient="horizontal").pack(fill="x", padx=8, pady=4)
        def _lbl(t): tk.Label(p, text=t, bg=bg, fg="#e5c07b",
                               font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10, pady=(4,1))
        def _btn(text, fg_c, cmd, state="disabled"):
            b = tk.Button(p, text=text, bg="#3a3a3a", fg=fg_c,
                          activebackground="#444", relief="flat",
                          cursor="hand2", bd=0, font=("Helvetica", 9),
                          state=state, command=cmd)
            b.pack(padx=10, pady=2, fill="x", ipady=5)
            return b

        tk.Label(p, text="[R]  Robot", bg=bg, fg="#c678dd",
                 font=("Helvetica", 11, "bold")).pack(anchor="w", padx=10, pady=(8,0))
        _sep()

        # Robot type selector
        r_type = tk.Frame(p, bg=bg); r_type.pack(fill="x", padx=10, pady=(0,3))
        tk.Label(r_type, text="Type:", bg=bg, fg="#ccc",
                 font=("Helvetica", 9), width=5, anchor="w").pack(side="left")
        driver_names = get_driver_names()
        self._robot_type_var = tk.StringVar(value=driver_names[0] if driver_names else "")
        type_menu = ttk.Combobox(r_type, textvariable=self._robot_type_var,
                                  values=driver_names, state="readonly",
                                  font=("Helvetica", 9), width=18)
        type_menu.pack(side="left", fill="x", expand=True, ipady=2)

        # Tool selector
        r_tool = tk.Frame(p, bg=bg); r_tool.pack(fill="x", padx=10, pady=(0,3))
        tk.Label(r_tool, text="Tool:", bg=bg, fg="#ccc",
                 font=("Helvetica", 9), width=5, anchor="w").pack(side="left")
        tool_names = ["(built-in / none)"] + get_tool_names()
        self._tool_var = tk.StringVar(value=tool_names[0])
        tool_menu = ttk.Combobox(r_tool, textvariable=self._tool_var,
                                  values=tool_names, state="readonly",
                                  font=("Helvetica", 9), width=18)
        tool_menu.pack(side="left", fill="x", expand=True, ipady=2)
        self._tool_status = tk.Label(p, text="", bg=bg, fg="#555", font=("Courier", 8))
        self._tool_status.pack(anchor="w", padx=10)

        r = tk.Frame(p, bg=bg); r.pack(fill="x", padx=10, pady=(0,5))
        tk.Label(r, text="IP:", bg=bg, fg="#ccc",
                 font=("Helvetica", 9), width=5, anchor="w").pack(side="left")
        self._ip_var = tk.StringVar(value=self.args.robot_ip)
        tk.Entry(r, textvariable=self._ip_var, bg="#3a3a3a", fg="white",
                 insertbackground="white", relief="flat",
                 font=("Helvetica", 9)).pack(side="left", fill="x", expand=True, ipady=3)
        self._connect_btn = tk.Button(p, text=">>  Connect Robot",
                                       bg="#4a5568", fg="white",
                                       activebackground="#5a6578",
                                       relief="flat", cursor="hand2", bd=0,
                                       font=("Helvetica", 9),
                                       command=self._on_connect)
        self._connect_btn.pack(padx=10, pady=(0,4), fill="x", ipady=5)
        self._robot_status = tk.Label(p, text="-- Disconnected",
                                       bg=bg, fg="#e06c75", font=("Courier", 9))
        self._robot_status.pack(anchor="w", padx=10, pady=(0,5))
        _sep()

        _lbl("Motion")
        for label, attr, default in [
                ("Speed %",     "_speed_var",    "15"),
                ("Approach mm", "_approach_var", str(APPROACH_OFFSET)),
                ("TCP Z mm",    "_tcp_z_var",    "0")]:
            r2 = tk.Frame(p, bg=bg); r2.pack(fill="x", padx=10, pady=2)
            tk.Label(r2, text=label+":", bg=bg, fg="#ccc",
                     font=("Helvetica", 9), anchor="w", width=13).pack(side="left")
            v = tk.StringVar(value=default); setattr(self, attr, v)
            tk.Entry(r2, textvariable=v, width=6, bg="#3a3a3a", fg="white",
                     insertbackground="white", relief="flat",
                     font=("Helvetica", 9)).pack(side="right", ipady=2)
        _sep()

        _lbl("Actions")
        self._recover_btn = tk.Button(
            p, text="[!]  Recover Robot",
            bg="#d19a66", fg="#1e1e1e", activebackground="#b8844a",
            relief="flat", cursor="hand2", bd=0,
            font=("Helvetica", 9, "bold"), state="disabled",
            command=self._on_recover_robot)
        self._recover_btn.pack(padx=10, pady=2, fill="x", ipady=5)
        self._home_btn = _btn("[H]  Move to Home", "#ccc", self._on_home)

        # Gripper manual controls
        r_grip = tk.Frame(p, bg=bg); r_grip.pack(fill="x", padx=10, pady=2)
        self._gripper_close_btn = tk.Button(
            r_grip, text="Gripper Close", bg="#3a3a3a", fg="#98c379",
            activebackground="#444", relief="flat", cursor="hand2", bd=0,
            font=("Helvetica", 9), state="disabled",
            command=self._on_gripper_close)
        self._gripper_close_btn.pack(side="left", fill="x", expand=True, ipady=5, padx=(0,2))
        self._gripper_open_btn = tk.Button(
            r_grip, text="Gripper Open", bg="#3a3a3a", fg="#e06c75",
            activebackground="#444", relief="flat", cursor="hand2", bd=0,
            font=("Helvetica", 9), state="disabled",
            command=self._on_gripper_open)
        self._gripper_open_btn.pack(side="left", fill="x", expand=True, ipady=5, padx=(2,0))
        _sep()

        _lbl("Positions")
        self._save_home_btn = _btn("[*]  Save as Home", "#e5c07b", self._on_save_home)
        self._save_sort_btn = _btn("[*]  Save as Sort", "#56b6c2", self._on_save_sort)
        self._go_sort_btn   = _btn("[v]  Go to Sort",    "#56b6c2", self._on_go_sort)
        self._sort_status = tk.Label(p, text="No sort position saved",
                                      bg=bg, fg="#555", font=("Courier", 8))
        self._sort_status.pack(anchor="w", padx=10, pady=(0,4))

    def _build_list_panel(self, p):
        bg = p["bg"]
        def _sep(): ttk.Separator(p, orient="horizontal").pack(fill="x", padx=8, pady=4)

        tk.Label(p, text="[W]  Word List", bg=bg, fg="#e5c07b",
                 font=("Helvetica", 11, "bold")).pack(anchor="w", padx=10, pady=(8,0))
        tk.Label(p, text="top→bottom  ·  3 tries/word  ·  loops forever",
                 bg=bg, fg="#555", font=("Helvetica", 7)
                 ).pack(anchor="w", padx=10, pady=(0,2))
        _sep()

        lf = tk.Frame(p, bg=bg)
        lf.pack(fill="both", expand=True, padx=10, pady=(0,4))
        self._batch_listbox = tk.Listbox(
            lf, bg="#1a1a1a", fg="#abb2bf",
            selectbackground="#3a3a3a", selectforeground="#61afef",
            font=("Courier", 10), relief="flat", bd=0,
            highlightthickness=1, highlightbackground="#333",
            activestyle="none")
        self._batch_listbox.pack(side="left", fill="both", expand=True)
        _sb = tk.Scrollbar(lf, orient="vertical", command=self._batch_listbox.yview)
        _sb.pack(side="right", fill="y")
        self._batch_listbox.config(yscrollcommand=_sb.set)

        ar = tk.Frame(p, bg=bg); ar.pack(fill="x", padx=10, pady=(0,3))
        self._batch_word_var = tk.StringVar()
        self._batch_entry = tk.Entry(
            ar, textvariable=self._batch_word_var,
            bg="#3a3a3a", fg="white", insertbackground="white",
            relief="flat", font=("Helvetica", 9), bd=3)
        self._batch_entry.pack(side="left", fill="x", expand=True, padx=(0,4))
        self._batch_entry.bind("<Return>", lambda _: self._on_add_batch_word())
        tk.Button(ar, text="Add",
                  bg="#3a3a3a", fg="#98c379", activebackground="#444",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 8),
                  command=self._on_add_batch_word
                  ).pack(side="right", ipady=4, ipadx=6)

        rr = tk.Frame(p, bg=bg); rr.pack(fill="x", padx=10, pady=(0,4))
        tk.Button(rr, text="Remove",
                  bg="#3a3a3a", fg="#e06c75", activebackground="#444",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 8),
                  command=self._on_remove_batch_word
                  ).pack(side="left", ipady=3, ipadx=4)
        tk.Button(rr, text="Clear All",
                  bg="#3a3a3a", fg="#aaa", activebackground="#444",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 8),
                  command=lambda: self._batch_listbox.delete(0, "end")
                  ).pack(side="left", ipady=3, ipadx=4, padx=(4,0))
        _sep()

        tk.Label(p, text="List Files", bg=bg, fg="#e5c07b",
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=10, pady=(0,2))
        fr = tk.Frame(p, bg=bg); fr.pack(fill="x", padx=10, pady=(0,3))
        tk.Button(fr, text="Load List",
                  bg="#3a3a3a", fg="#e5c07b", activebackground="#444",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 8),
                  command=self._on_load_list
                  ).pack(side="left", ipady=4, ipadx=6)
        tk.Button(fr, text="Save List",
                  bg="#3a3a3a", fg="#98c379", activebackground="#444",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 8),
                  command=self._on_save_list
                  ).pack(side="left", ipady=4, ipadx=6, padx=(4,0))
        self._list_file_label = tk.Label(p, text="No file loaded",
                                          bg=bg, fg="#555",
                                          font=("Courier", 7), anchor="w",
                                          wraplength=210)
        self._list_file_label.pack(anchor="w", padx=10, pady=(2,4))

    def _build_exec_panel(self, p):
        bg = p["bg"]
        def _sep(): ttk.Separator(p, orient="horizontal").pack(fill="x", padx=8, pady=4)
        def _lbl(t): tk.Label(p, text=t, bg=bg, fg="#e5c07b",
                               font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10, pady=(4,1))

        tk.Label(p, text="[>]  Execution", bg=bg, fg="#56b6c2",
                 font=("Helvetica", 11, "bold")).pack(anchor="w", padx=10, pady=(8,0))
        _sep()

        _lbl("Status")
        self._status_var = tk.StringVar(value="Waiting for camera...")
        tk.Label(p, textvariable=self._status_var, bg=bg, fg="#98c379",
                 font=("Helvetica", 10), wraplength=230, justify="left", anchor="w"
                 ).pack(anchor="w", padx=10, pady=(0,5))
        _sep()

        _lbl("Single Run")
        self._run_btn = tk.Button(
            p, text="[>]  Capture & Run GraspGen",
            bg="#61afef", fg="#1e1e1e", activebackground="#4d9bd6",
            font=("Helvetica", 9, "bold"), relief="flat",
            cursor="hand2", bd=0, command=self._on_run)
        self._run_btn.pack(padx=10, pady=2, fill="x", ipady=8)

        self._continue_btn = tk.Button(
            p, text="[>]  Continue (debug step)",
            bg="#e5c07b", fg="#1e1e1e", activebackground="#c9a44e",
            font=("Helvetica", 9, "bold"), relief="flat",
            cursor="hand2", bd=0, state="disabled",
            command=self._on_debug_continue)
        self._continue_btn.pack(padx=10, pady=2, fill="x", ipady=5)

        self._execute_btn = tk.Button(
            p, text="[>]  Execute Selected Grasp",
            bg="#c678dd", fg="white", activebackground="#a85dc0",
            relief="flat", cursor="hand2", bd=0,
            font=("Helvetica", 9, "bold"), state="disabled",
            command=self._on_execute)
        self._execute_btn.pack(padx=10, pady=2, fill="x", ipady=8)

        tk.Checkbutton(p, text="Auto-retry next grasp on error",
                       variable=self._retry_grasps_var,
                       bg=bg, fg="#aaa", activebackground=bg,
                       selectcolor="#3a3a3a", font=("Helvetica", 9)
                       ).pack(anchor="w", padx=10, pady=(2,1))
        self._stop_retry_btn = tk.Button(
            p, text="[x]  Stop Retry",
            bg="#e06c75", fg="white", activebackground="#c0545e",
            relief="flat", cursor="hand2", bd=0,
            font=("Helvetica", 9, "bold"), state="disabled",
            command=self._on_stop_retry)
        self._stop_retry_btn.pack(padx=10, pady=(0,2), fill="x", ipady=4)
        tk.Button(p, text="[x]  Clear Mask",
                  bg="#3a3a3a", fg="#aaa", activebackground="#444",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 9),
                  command=lambda: setattr(self, "_last_mask", None)
                  ).pack(padx=10, pady=2, fill="x", ipady=4)
        _sep()

        _lbl("Batch  (continuous loop)")
        self._batch_run_btn = tk.Button(
            p, text="[>]  Run Batch",
            bg="#e5c07b", fg="#1e1e1e", activebackground="#c9a44e",
            relief="flat", cursor="hand2", bd=0,
            font=("Helvetica", 10, "bold"),
            command=self._on_run_batch)
        self._batch_run_btn.pack(padx=10, pady=2, fill="x", ipady=8)

        self._batch_stop_btn = tk.Button(
            p, text="[x]  Stop Batch",
            bg="#e06c75", fg="white", activebackground="#c0545e",
            relief="flat", cursor="hand2", bd=0,
            font=("Helvetica", 9, "bold"), state="disabled",
            command=self._on_stop_batch)
        self._batch_stop_btn.pack(padx=10, pady=(0,4), fill="x", ipady=5)
        _sep()

        _lbl("Visualisation")
        tk.Button(p, text="Open Meshcat Viewer",
                  bg="#3a3a3a", fg="#56b6c2", activebackground="#444",
                  relief="flat", cursor="hand2", bd=0,
                  font=("Helvetica", 8),
                  command=self._on_open_meshcat
                  ).pack(padx=10, pady=2, fill="x", ipady=5)

    # ------------------------------------------------------------------
    # ROI selection
    # ------------------------------------------------------------------
    def _on_roi_select(self):
        """Start 4-point polygon ROI selection."""
        self._roi_poly_img = None
        self._roi_canvas_pts = []
        self._roi_selecting = True
        for _id in self._roi_dot_ids + self._roi_edge_ids:
            self._canvas.itemconfig(_id, state="hidden")
        self._canvas.itemconfig(self._roi_preview_id, state="hidden")
        self._roi_btn.config(text="Click point 1 / 4")
        self._canvas.config(cursor="crosshair")
        self._cb_queue.put(self._refresh_canvas)

    def _on_roi_clear(self):
        self._roi_poly_img = None
        self._roi_selecting = False
        self._roi_canvas_pts = []
        for _id in self._roi_dot_ids + self._roi_edge_ids:
            self._canvas.itemconfig(_id, state="hidden")
        self._canvas.itemconfig(self._roi_preview_id, state="hidden")
        self._roi_info_var.set("")
        self._roi_btn.config(text="[ ] Select ROI")
        self._canvas.config(cursor="")
        self._save_roi()
        self._cb_queue.put(self._refresh_canvas)

    def _on_roi_click(self, event):
        if not self._roi_selecting:
            return
        cx, cy = event.x, event.y
        self._roi_canvas_pts.append((cx, cy))
        n = len(self._roi_canvas_pts)

        # Place dot for this point
        r = 5
        self._canvas.coords(self._roi_dot_ids[n - 1],
                            cx - r, cy - r, cx + r, cy + r)
        self._canvas.itemconfig(self._roi_dot_ids[n - 1], state="normal")

        # Edge from previous point to this one
        if n > 1:
            px, py = self._roi_canvas_pts[n - 2]
            self._canvas.coords(self._roi_edge_ids[n - 2], px, py, cx, cy)
            self._canvas.itemconfig(self._roi_edge_ids[n - 2], state="normal")

        if n < 4:
            self._roi_btn.config(text=f"Click point {n + 1} / 4")
        else:
            # Close the polygon (edge from point 4 back to point 1)
            p0x, p0y = self._roi_canvas_pts[0]
            self._canvas.coords(self._roi_edge_ids[3], cx, cy, p0x, p0y)
            self._canvas.itemconfig(self._roi_edge_ids[3], state="normal")
            self._canvas.itemconfig(self._roi_preview_id, state="hidden")

            # Convert all 4 canvas pts → image coords
            scale = self._canvas_scale
            xo, yo = self._canvas_xo, self._canvas_yo
            frame = self._current_frame or self._replay_frame
            fh, fw = (frame[0].shape[:2] if frame is not None else (99999, 99999))
            img_pts = []
            for ccx, ccy in self._roi_canvas_pts:
                ix = int(max(0, min((ccx - xo) / scale, fw - 1)))
                iy = int(max(0, min((ccy - yo) / scale, fh - 1)))
                img_pts.append((ix, iy))

            self._roi_poly_img = img_pts
            self._roi_selecting = False
            self._roi_canvas_pts = []
            self._roi_btn.config(text="[ ] Select ROI")
            self._canvas.config(cursor="")
            xs = [p[0] for p in img_pts]
            ys = [p[1] for p in img_pts]
            self._roi_info_var.set(
                f"ROI poly  bbox ({min(xs)},{min(ys)})→({max(xs)},{max(ys)})")
            self._log(f"[ROI] 4-pt polygon (image coords): {img_pts}")
            self._save_roi()

    def _on_roi_hover(self, event):
        if not self._roi_selecting or not self._roi_canvas_pts:
            return
        px, py = self._roi_canvas_pts[-1]
        self._canvas.coords(self._roi_preview_id, px, py, event.x, event.y)
        self._canvas.itemconfig(self._roi_preview_id, state="normal")

    # ------------------------------------------------------------------
    # Camera connect / disconnect
    # ------------------------------------------------------------------
    def _on_camera_connect(self):
        if self.camera._started:
            # Disconnect
            self.camera.stop()
            self._cam_status_var.set("-- No camera")
            self._cam_status_lbl.config(fg="#e06c75")
            self._cam_connect_btn.config(text="Connect Camera Camera")
            self._log("[Camera] Disconnected.")
        else:
            # Connect in background (Pipeline() can take a moment)
            self._cam_connect_btn.config(state="disabled", text="Connecting…")
            self._cam_status_var.set("-- Connecting…")
            self._cam_status_lbl.config(fg="#e5c07b")
            threading.Thread(target=self._camera_connect_worker, daemon=True).start()

    def _camera_connect_worker(self):
        try:
            self.camera.start()
            self._log("[Camera] Connected — live stream running.")
            def _ok():
                self._cam_status_var.set("-- Connected")
                self._cam_status_lbl.config(fg="#98c379")
                self._cam_connect_btn.config(state="normal", text="Disconnect Camera")
            self._cb_queue.put(_ok)
        except Exception as e:
            self._log(f"[Camera] Failed to connect: {e}")
            def _err():
                self._cam_status_var.set(f"-- Error: {e}")
                self._cam_status_lbl.config(fg="#e06c75")
                self._cam_connect_btn.config(state="normal", text="Connect Camera Camera")
            self._cb_queue.put(_err)

    # ------------------------------------------------------------------
    # Replay mode
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Save scene (all data needed for offline pipeline replay)
    # ------------------------------------------------------------------
    def _on_save_scene(self):
        """Save current frame + calibration + prompt + ROI to a single .npz."""
        frame = self._current_frame or self._replay_frame
        if frame is None:
            self._log("[Save] No frame captured yet — capture first.")
            return
        rgb, depth_m, intrinsics = frame

        from tkinter import filedialog
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"scene_{ts}.npz"
        save_path = filedialog.asksaveasfilename(
            title="Save scene as…",
            initialdir=str(RESULTS_DIR),
            initialfile=default_name,
            defaultextension=".npz",
            filetypes=[("NumPy scene", "*.npz"), ("All files", "*.*")])
        if not save_path:
            return

        # Calibration
        T_cam2base = self._T_cam2base if self._T_cam2base is not None else np.eye(4)
        K = self._calib_K if self._calib_K is not None \
            else np.array([[intrinsics[0], 0, intrinsics[2]],
                           [0, intrinsics[1], intrinsics[3]],
                           [0, 0, 1]], dtype=np.float64)
        dist = getattr(self, '_calib_dist', None)
        dist_arr = dist if dist is not None else np.zeros(5)

        # ROI polygon
        roi = np.array(self._roi_poly_img, dtype=np.float32) \
            if self._roi_poly_img else np.zeros((0, 2), dtype=np.float32)

        np.savez_compressed(
            save_path,
            rgb=rgb,                        # uint8 (H, W, 3)
            depth_m=depth_m,                # float32 (H, W) metres
            intrinsics=np.array(intrinsics, dtype=np.float64),  # [fx, fy, cx, cy]
            K=K,                            # 3×3 calibrated camera matrix
            dist=dist_arr,                  # distortion coefficients
            T_cam2base=T_cam2base,          # 4×4 mm, camera → robot base
            roi_poly=roi,                   # (4, 2) image-coord polygon or (0,2)
        )
        self._log(f"[Save] Scene saved -> {Path(save_path).name}  "
                  f"rgb={rgb.shape}  depth={depth_m.shape}  roi={len(roi)} pts")

    def _on_load_replay(self):
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Select .npz scene or RGB image",
            initialdir=str(RESULTS_DIR),
            filetypes=[("Scene / PNG", "*.npz *.png"), ("NumPy scene", "*.npz"),
                       ("PNG images", "*.png"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            # ── .npz scene: contains everything ──────────────────────────────
            if file_path.lower().endswith(".npz"):
                data = np.load(file_path, allow_pickle=True)
                rgb     = data["rgb"]                       # uint8 (H,W,3)
                depth_m = data["depth_m"]                   # float32 (H,W) metres
                intr_arr = data["intrinsics"]               # [fx, fy, cx, cy]
                intr = tuple(float(v) for v in intr_arr)

                # Restore calibration from scene
                if "T_cam2base" in data:
                    T = data["T_cam2base"]
                    if T.shape == (4, 4) and not np.allclose(T, np.eye(4)):
                        self._T_cam2base = T
                        self._log(f"[Replay] Restored T_cam2base from scene")
                if "K" in data:
                    K = data["K"]
                    if K.shape == (3, 3):
                        self._calib_K = K
                if "dist" in data:
                    self._calib_dist = data["dist"]

                # Restore ROI
                if "roi_poly" in data:
                    roi = data["roi_poly"]
                    if roi.shape[0] >= 3:
                        self._roi_poly_img = [(int(p[0]), int(p[1])) for p in roi]
                        self._log(f"[Replay] Restored ROI ({len(self._roi_poly_img)} pts)")


            # ── PNG image: find matching depth ───────────────────────────────
            else:
                rgb = np.array(__import__("PIL").Image.open(file_path).convert("RGB"))

                import re as _re
                ts_match = _re.search(r"(\d{8}_\d{6})", file_path)
                depth_m = None
                if ts_match:
                    ts = ts_match.group(1)
                    data_root = Path(file_path).parent.parent
                    for candidate in [
                        data_root / "depth_aligned" / f"depth_aligned_{ts}.npy",
                        data_root / "depth"         / f"depth_{ts}.npy",
                    ]:
                        if candidate.exists():
                            raw = np.load(str(candidate))
                            depth_m = raw.astype(np.float32) / 1000.0
                            self._log(f"[Replay] Depth: {candidate.name}")
                            break

                if depth_m is None:
                    depth_path = filedialog.askopenfilename(
                        title="Select depth .npy (uint16 mm)",
                        initialdir=str(Path(file_path).parent.parent / "depth_aligned"),
                        filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")])
                    if not depth_path:
                        return
                    raw = np.load(depth_path)
                    depth_m = raw.astype(np.float32) / 1000.0

                if self._calib_K is not None:
                    K = self._calib_K
                    intr = (float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2]))
                else:
                    h, w = rgb.shape[:2]
                    intr = (684.7, 685.9, w / 2.0, h / 2.0)

            self._replay_frame = (rgb, depth_m, intr)
            fname = Path(file_path).name
            self._replay_label_var.set(f"REPLAY: {fname}")
            self._log(f"[Replay] Loaded {fname}  rgb={rgb.shape}  "
                      f"depth={depth_m.shape} ({depth_m[depth_m>0].mean()*1000:.0f}mm avg)")
            self._show_replay_in_canvas(rgb)
        except Exception as e:
            self._log(f"[Replay] Load error: {e}")
            import traceback; traceback.print_exc()

    def _on_clear_replay(self):
        self._replay_frame = None
        self._current_frame = None
        self._replay_label_var.set("")
        self._last_mask = None
        # Blank the canvas and show the placeholder text
        self._canvas.itemconfig(self._canvas_img_id, image="")
        self._canvas.itemconfig(self._no_cam_txt, state="normal")
        self._log("[Scene] Cleared.")

    def _show_replay_in_canvas(self, rgb):
        """Show a replay frame — delegates to _refresh_canvas (called on main thread)."""
        self._refresh_canvas(rgb)

    # ------------------------------------------------------------------
    # Snapshot capture & canvas refresh
    # ------------------------------------------------------------------
    def _on_capture_frame(self):
        """Take one frame from the live camera and display it."""
        if not self.camera._started:
            self._log("[Camera] Not connected — cannot capture.")
            return
        self._log("[Camera] Capturing frame…")
        threading.Thread(target=self._capture_frame_worker, daemon=True).start()

    def _capture_frame_worker(self):
        try:
            rgb, depth_m, intrinsics = self.camera.capture_frame()
            self._current_frame = (rgb, depth_m, intrinsics)
            self._last_mask = None
            self._cb_queue.put(lambda r=rgb: self._refresh_canvas(r))
            self._log(f"[Camera] Captured  {rgb.shape[1]}×{rgb.shape[0]}")
        except Exception as e:
            self._log(f"[Camera] Capture failed: {e}")

    def _refresh_canvas(self, rgb=None):
        """Render rgb (with mask overlay) onto the preview canvas. Must run on main thread."""
        if not _PIL_OK:
            return
        if rgb is None:
            if self._current_frame is not None:
                rgb = self._current_frame[0]
            elif self._replay_frame is not None:
                rgb = self._replay_frame[0]
            else:
                return

        display = rgb.copy()
        mask = self._last_mask
        if self._show_mask.get() and mask is not None:
            mh, mw = mask.shape[:2]
            dh, dw = display.shape[:2]
            if (mh, mw) != (dh, dw):
                mask = cv2.resize(mask.astype(np.float32), (dw, dh),
                                  interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            overlay = display.copy()
            overlay[mask > 0] = _MASK_GREEN
            cv2.addWeighted(display, 0.55, overlay, 0.45, 0, dst=display)

        h, w = display.shape[:2]
        scale = min(_PREVIEW_W / w, _PREVIEW_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(display, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((_PREVIEW_H, _PREVIEW_W, 3), dtype=np.uint8)
        yo, xo = (_PREVIEW_H - nh) // 2, (_PREVIEW_W - nw) // 2
        padded[yo:yo+nh, xo:xo+nw] = resized

        # Store for ROI coord mapping
        self._canvas_scale = scale
        self._canvas_xo    = xo
        self._canvas_yo    = yo

        # Draw completed ROI polygon overlay
        if self._roi_poly_img is not None:
            _pts = np.array(
                [[int(p[0] * scale + xo), int(p[1] * scale + yo)]
                 for p in self._roi_poly_img], dtype=np.int32)
            cv2.polylines(padded, [_pts], isClosed=True, color=(255, 215, 0), thickness=2)
            for _pt in _pts:
                cv2.circle(padded, tuple(_pt), 5, (255, 215, 0), -1)

        pil = PILImage.fromarray(padded)
        tk_img = ImageTk.PhotoImage(pil)
        self._canvas.itemconfig(self._canvas_img_id, image=tk_img)
        self._canvas.itemconfig(self._no_cam_txt, state="hidden")
        self._canvas._tk_ref = tk_img

    # ------------------------------------------------------------------
    # Log / cb queue
    # ------------------------------------------------------------------
    def _schedule_flush(self):
        self._flush_log(); self._flush_cb()
        self.root.after(150, self._schedule_flush)

    def _flush_cb(self):
        try:
            while True:
                self._cb_queue.get_nowait()()
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

    def _log(self, msg):
        print(msg, flush=True)
        self._log_queue.put(msg)

    def _set_status(self, msg):
        self._cb_queue.put(lambda m=msg: self._status_var.set(m))

    # ------------------------------------------------------------------
    # GraspGen config loading  (identical to demo)
    # ------------------------------------------------------------------
    def _start_config_load(self, name):
        if self._config_loading or name == self._loaded_config_name:
            return
        self._config_loading = True
        self._cb_queue.put(lambda: self._run_btn.configure(
            state="disabled", text="Loading model…"))
        self._set_status(f"Loading {name}…")
        threading.Thread(target=self._load_config_worker, args=(name,), daemon=True).start()

    def _load_config_worker(self, name):
        path = self.config_map.get(name)
        try:
            if self._sampler:
                del self._sampler
                self._sampler = None
                try:
                    torch.cuda.synchronize(); torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()
            self._log(f"[Config] Loading {name}…")
            cfg = load_grasp_cfg(path)
            sampler = GraspGenSampler(cfg)
            self._grasp_cfg = cfg
            self._sampler = sampler
            self._loaded_config_name = name
            # Auto-detect vacuum/suction gripper → enable orientation flip
            _gname = cfg.data.gripper_name.lower()
            self._flip_orient = any(kw in _gname for kw in ("vacuum", "suction", "cup"))
            self._cb_queue.put(lambda p=path: self._config_hint.configure(text=p))
            self._log(f"[Config] Ready: {cfg.data.gripper_name}  "
                      f"(orientation flip {'ON — vacuum/suction detected' if self._flip_orient else 'OFF'})")
            self._set_status(f"Ready — {name}")
        except Exception as e:
            self._log(f"[Config] ERROR: {e}")
            self._set_status(f"Config error: {e}")
            self._grasp_cfg = None; self._sampler = None
            self._loaded_config_name = None
        finally:
            self._config_loading = False
            self._cb_queue.put(self._restore_run_btn)

    def _restore_run_btn(self):
        if not self._inference_running and not self._config_loading:
            self._run_btn.configure(state="normal", text="[>]  Capture & Run GraspGen")

    def _on_config_change(self, _=None):
        name = self._config_combo.get()
        if not name or name == self._loaded_config_name:
            return
        if self._config_loading or self._inference_running:
            self._config_combo.set(self._loaded_config_name or "")
            return
        self._start_config_load(name)

    # ------------------------------------------------------------------
    # Debug step control
    # ------------------------------------------------------------------
    def _debug_pause(self, step_label):
        """Block the pipeline worker at a debug checkpoint until the user continues."""
        if not self._debug_var.get():
            return
        self._debug_event.clear()
        self._set_status(f"Debug — {step_label}  > click Continue")
        self._cb_queue.put(lambda: self._continue_btn.configure(state="normal"))
        self._debug_event.wait(timeout=600)   # auto-release after 10 min
        self._cb_queue.put(lambda: self._continue_btn.configure(state="disabled"))

    def _on_debug_continue(self):
        self._debug_event.set()

    # ------------------------------------------------------------------
    # Run GraspGen pipeline
    # ------------------------------------------------------------------
    def _on_run(self):
        if self._inference_running or self._config_loading:
            return
        prompt = self._prompt_var.get().strip()
        if not prompt:
            self._set_status("Enter an object prompt first.")
            return
        if self._sampler is None:
            self._set_status("No gripper config loaded.")
            return
        if self._replay_frame is not None:
            source = "replay"
        elif self.camera._started:
            source = "camera"      # will capture inside the worker thread
        elif self._current_frame is not None:
            source = "cached"
        else:
            self._set_status("Connect camera or load a saved frame first.")
            return
        self._inference_running = True
        self._best_grasp_base = None
        self._cb_queue.put(lambda: self._execute_btn.configure(state="disabled"))
        self._run_btn.configure(state="disabled", text="Running…")
        self._set_status("Running pipeline…")
        self._last_mask = None
        threading.Thread(target=self._pipeline_worker,
                          args=(source, prompt, self._sampler, self._grasp_cfg),
                          daemon=True).start()

    def _pipeline_worker(self, source, prompt, sampler, grasp_cfg):
        args = self.args
        try:
            # ── Acquire frame ────────────────────────────────────────────────
            if source == "replay":
                rgb, depth_m, intrinsics = self._replay_frame
                self._log("[Source] Saved frame (replay mode)")
            elif source == "camera":
                self._log("[Camera] Capturing frame…")
                self._set_status("Capturing frame…")
                rgb, depth_m, intrinsics = self.camera.capture_frame()
                self._current_frame = (rgb, depth_m, intrinsics)
                self._log(f"[Camera] Captured  {rgb.shape[1]}×{rgb.shape[0]}")
                self._cb_queue.put(lambda r=rgb: self._refresh_canvas(r))
            else:
                rgb, depth_m, intrinsics = self._current_frame
                self._log("[Source] Last captured frame")
            # ─────────────────────────────────────────────────────────────────
            self._log(f"[SAM3] prompt='{prompt}'")
            t0 = time.time()
            _roi_poly = self._roi_poly_img
            if _roi_poly is not None:
                fH, fW = rgb.shape[:2]
                _xs = [p[0] for p in _roi_poly]
                _ys = [p[1] for p in _roi_poly]
                _rx1, _ry1 = max(0, min(_xs)), max(0, min(_ys))
                _rx2, _ry2 = min(fW, max(_xs)), min(fH, max(_ys))
                if _rx2 > _rx1 + 10 and _ry2 > _ry1 + 10:
                    rgb_crop = rgb[_ry1:_ry2, _rx1:_rx2].copy()
                    # Mask out pixels outside the polygon within the crop
                    _poly_in_crop = np.array(
                        [[(p[0] - _rx1), (p[1] - _ry1)] for p in _roi_poly],
                        dtype=np.int32)
                    _poly_mask = np.zeros(rgb_crop.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(_poly_mask, [_poly_in_crop], 1)
                    rgb_sam = rgb_crop.copy()
                    rgb_sam[_poly_mask == 0] = 0
                    self._log(f"[SAM3] ROI 4-pt polygon, bbox "
                              f"({_rx1},{_ry1})→({_rx2},{_ry2}) "
                              f"{_rx2-_rx1}×{_ry2-_ry1} px")
                else:
                    rgb_sam, _roi_poly = rgb, None
            else:
                rgb_sam = rgb
            mask_crop = segment_with_sam3(rgb_sam, prompt, args.sam3_socket)
            if _roi_poly is not None:
                mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
                crop_h, crop_w = _ry2 - _ry1, _rx2 - _rx1
                if mask_crop.shape != (crop_h, crop_w):
                    mask_crop = cv2.resize(mask_crop.astype(np.float32),
                                           (crop_w, crop_h),
                                           interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                # Only keep mask pixels that are inside the polygon
                mask_crop = mask_crop & _poly_mask
                mask[_ry1:_ry2, _rx1:_rx2] = mask_crop
            else:
                mask = mask_crop
            self._log(f"[SAM3] {time.time()-t0:.2f}s — {int(mask.sum())} px")

            if mask.shape != depth_m.shape:
                mask = cv2.resize(mask.astype(np.float32),
                                   (depth_m.shape[1], depth_m.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(np.uint8)

            self._last_mask = mask
            self._cb_queue.put(self._refresh_canvas)   # show mask overlay on captured frame
            if mask.sum() < 50:
                raise RuntimeError(f"Only {mask.sum()} px — try a different prompt")

            # ── Debug step 1: SAM3 segmentation ─────────────────────────────
            self._debug_pause(f"Step 1/7 — SAM3  ({int(mask.sum())} px masked)")
            # ────────────────────────────────────────────────────────────────

            self._log(f"[PC] Frame size: rgb={rgb.shape[1]}×{rgb.shape[0]}  "
                      f"depth={depth_m.shape[1]}×{depth_m.shape[0]}")

            # Prefer calibrated intrinsics over SDK defaults
            if self._calib_K is not None:
                K = self._calib_K
                fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
                calib_w = int(round(cx * 2))  # approximate calibrated width from cx
                if abs(rgb.shape[1] - calib_w) > 50:
                    self._log(f"[PC] WARNING: frame width {rgb.shape[1]} differs from "
                              f"calibration width ≈{calib_w} — point cloud may be deformed!")
                else:
                    self._log("[PC] Using calibrated camera intrinsics ✓")
            else:
                fx, fy, cx, cy = intrinsics
                self._log("[PC] Using SDK camera intrinsics")

            # Undistort RGB only — depth must NOT be undistorted with bilinear
            # interpolation: at object edges cv2.undistort blends foreground and
            # background depth values, producing interpolated points that project
            # to 3D positions floating between surfaces (the "flying pixel" halo).
            dist = getattr(self, '_calib_dist', None)
            if dist is not None and np.any(dist != 0):
                K_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
                rgb = cv2.undistort(rgb, K_mat, dist)
                self._log("[PC] Applied lens distortion correction (RGB only)")


            # Erode the mask by 4 pixels to strip the depth-edge border.
            # At the boundary between object and background the Orbbec D2C
            # alignment produces "flying pixels" — depth values that are a blend
            # of foreground and background.  These project to 3-D positions that
            # float between the two surfaces and distort the object shape (the
            # classic "hill on a flat face" artefact).  A small erosion removes
            # only the outermost ring of pixels where flying pixels live.
            _ero_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask_pc = cv2.erode(mask, _ero_k, iterations=1)
            n_lost  = int(mask.sum()) - int(mask_pc.sum())
            if n_lost > 0:
                self._log(f"[PC] Edge erosion removed {n_lost} border px (flying-pixel guard)")
            if mask_pc.sum() < 30:
                self._log("[PC] Mask too small after erosion — using original mask")
                mask_pc = mask

            scene_pc, object_pc, scene_colors, object_colors = \
                depth_and_segmentation_to_point_clouds(
                    depth_image=depth_m, segmentation_mask=mask_pc,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    rgb_image=rgb, target_object_id=args.target_object_id,
                    remove_object_from_scene=True)

            if len(object_pc) > args.max_object_points:
                idx = np.random.choice(len(object_pc), args.max_object_points, replace=False)
                object_pc = object_pc[idx]
                object_colors = object_colors[idx]

            # ── Debug step 2: raw object point cloud (camera frame) ──────────
            if self._debug_var.get() and self._vis:
                self._vis.delete()
                visualize_pointcloud(self._vis, "pc_raw_cam", object_pc, object_colors,
                                     size=args.scene_point_size)
            self._debug_pause(
                f"Step 2/7 — Raw object cloud  ({len(object_pc)} pts, camera frame)")
            # ────────────────────────────────────────────────────────────────

            pc_torch = torch.from_numpy(object_pc)
            pc_filtered, _ = point_cloud_outlier_removal(pc_torch)
            pc_filtered = pc_filtered.numpy()
            if len(pc_filtered) == 0:
                raise RuntimeError("Point cloud empty after outlier removal")

            # Match colors to the surviving points after outlier removal.
            # pc_filtered ⊆ object_pc (outlier removal only drops points),
            # so every filtered point has an exact match in object_pc.
            # Use cKDTree instead of broadcasting to avoid O(N*M) memory blowup.
            from scipy.spatial import cKDTree
            _tree = cKDTree(object_pc)
            _, nn_idx = _tree.query(pc_filtered, k=1, workers=1)
            object_colors_filtered = object_colors[nn_idx]

            # ── Debug step 3: before vs after outlier removal (camera frame) ──
            if self._debug_var.get() and self._vis:
                # Find which object_pc points were removed as outliers
                _outlier_mask = np.ones(len(object_pc), dtype=bool)
                _outlier_mask[nn_idx] = False
                _outlier_pc = object_pc[_outlier_mask]
                self._vis.delete()
                # Kept points in their real colors
                visualize_pointcloud(self._vis, "pc_kept", pc_filtered,
                                     object_colors_filtered, size=args.scene_point_size)
                # Removed outliers in red so the difference is obvious
                if len(_outlier_pc) > 0:
                    _red = np.tile([[255, 50, 50]], (len(_outlier_pc), 1)).astype(np.uint8)
                    visualize_pointcloud(self._vis, "pc_outliers", _outlier_pc, _red,
                                         size=args.scene_point_size)
            self._debug_pause(
                f"Step 3/7 — Outlier removal  "
                f"({len(pc_filtered)} kept, "
                f"{len(object_pc)-len(pc_filtered)} removed in red)")
            # ────────────────────────────────────────────────────────────────

            # ── Transform point clouds from camera frame to robot base
            #    frame using the FULL T_cam2base (rotation + translation)
            #    so everything is referenced from the robot base.
            if self._T_cam2base is not None:
                T_cam2base_m = self._T_cam2base.copy()
                T_cam2base_m[:3, 3] /= 1000.0          # mm → m
            else:
                T_cam2base_m = np.eye(4)
            R_cam2base = T_cam2base_m[:3, :3]
            self._log(f"[GraspGen] Transforming PC to robot base frame "
                      f"(det(R)={np.linalg.det(R_cam2base):.4f})")

            # Full rigid transform: p_base = R * p_cam + t
            pc_base = (T_cam2base_m[:3, :3] @ pc_filtered.T).T + T_cam2base_m[:3, 3]
            scene_base = (T_cam2base_m[:3, :3] @ scene_pc.T).T + T_cam2base_m[:3, 3] \
                if scene_pc is not None and len(scene_pc) > 0 else scene_pc

            # ── Ensure Z-up for GraspGen ──
            # Camera forward [0,0,1] in cam frame → "down" in real world
            # (camera looks down at the table).  In base frame this becomes:
            cam_down_in_base = R_cam2base @ np.array([0.0, 0.0, 1.0])
            z_needs_flip = cam_down_in_base[2] > 0   # base Z is DOWN → flip
            if z_needs_flip:
                # Rotate 180° around X to flip Y and Z (preserves right-hand rule)
                R_flip = np.diag([1.0, -1.0, -1.0])
                pc_base = (R_flip @ pc_base.T).T
                if scene_base is not None and len(scene_base) > 0:
                    scene_base = (R_flip @ scene_base.T).T
                self._log("[GraspGen] Z-axis was pointing down — flipped to Z-up")
            else:
                R_flip = np.eye(3)
                self._log("[GraspGen] Z-axis already points up — no flip needed")

            # Store object centroid in base frame (for un-centering later)
            obj_centroid_base = pc_base.mean(axis=0)
            self._log(f"[GraspGen] Object centroid (work frame): "
                      f"[{obj_centroid_base[0]*1000:.1f}, "
                      f"{obj_centroid_base[1]*1000:.1f}, "
                      f"{obj_centroid_base[2]*1000:.1f}] mm")
            t_center     = tra.translation_matrix(-obj_centroid_base)
            t_center_inv = tra.translation_matrix(obj_centroid_base)   # inverse = +centroid
            pc_centered = tra.transform_points(pc_base, t_center)
            scene_centered = tra.transform_points(scene_base, t_center) \
                if scene_base is not None and len(scene_base) > 0 \
                else np.empty((0, 3))

            # Camera position in the (possibly flipped) base frame
            cam_pos_base = R_flip @ T_cam2base_m[:3, 3]

            if self._vis:
                self._vis.delete()
                # Robot base frame at origin
                make_frame(self._vis, "robot_base", h=0.15, radius=0.005)
                # Camera frame
                cam_frame_4x4 = np.eye(4)
                cam_frame_4x4[:3, :3] = R_flip @ T_cam2base_m[:3, :3]
                cam_frame_4x4[:3, 3]  = cam_pos_base
                self._vis["camera_frame"].set_transform(cam_frame_4x4)
                make_frame(self._vis["camera_frame"], "axes", h=0.08, radius=0.003)
                # Object centroid frame
                obj_frame = np.eye(4)
                obj_frame[:3, 3] = obj_centroid_base
                self._vis["object_frame"].set_transform(obj_frame)
                make_frame(self._vis["object_frame"], "axes", h=0.06, radius=0.003)

                _sc = scene_colors if scene_colors is not None else \
                    np.tile([[120,120,120]], (len(scene_pc),1)).astype(np.uint8)
                # Show full point cloud in work frame (Z-up)
                full_pc_base = np.vstack([scene_base, pc_base]) \
                    if scene_base is not None and len(scene_base) > 0 else pc_base
                full_colors = np.vstack([_sc, object_colors_filtered]) \
                    if scene_base is not None and len(scene_base) > 0 \
                    else object_colors_filtered
                visualize_pointcloud(self._vis, "pc_full", full_pc_base, full_colors,
                                     size=args.scene_point_size)

            # ── Debug step 4: point cloud in robot base frame ────────────────
            self._debug_pause(
                f"Step 4/7 — Point cloud in robot frame  ({len(pc_filtered)} obj pts, "
                f"{len(scene_pc) if scene_pc is not None else 0} scene pts)")
            # ────────────────────────────────────────────────────────────────

            _ng  = max(1, int(self._num_grasps_var.get() or args.num_grasps))
            _topk = max(1, int(self._topk_grasps_var.get() or args.topk_num_grasps))
            self._log(f"[GraspGen] Running inference  num_grasps={_ng}  topk={_topk}…")
            t1 = time.time()
            grasps, grasp_conf = GraspGenSampler.run_inference(
                pc_centered, sampler,
                grasp_threshold=args.grasp_threshold,
                num_grasps=_ng,
                topk_num_grasps=_topk,
                min_grasps=_topk)
            self._log(f"[GraspGen] {time.time()-t1:.2f}s — {len(grasps)} grasps")

            if len(grasps) == 0:
                raise RuntimeError("No grasps found")

            grasps_obj_np = grasps.cpu().numpy()   # object/centered frame
            grasp_conf_np = grasp_conf.cpu().numpy()
            n_generated = len(grasps_obj_np)

            # ── Debug step 5: all raw GraspGen poses (work frame) ────────────
            if self._debug_var.get() and self._vis:
                _grasps_all_work = t_center_inv @ grasps_obj_np
                self._vis.delete()
                visualize_pointcloud(self._vis, "pc_obj", pc_base, object_colors_filtered,
                                     size=args.scene_point_size)
                _gn = grasp_cfg.data.gripper_name
                for _i, _g in enumerate(_grasps_all_work):
                    visualize_grasp(self._vis, f"grasps_all/{_i:03d}/grasp", _g,
                                    color=[180, 180, 255], gripper_name=_gn, linewidth=0.8)
            self._debug_pause(
                f"Step 5/7 — All GraspGen poses  ({len(grasps_obj_np)} raw grasps)")
            # ────────────────────────────────────────────────────────────────

            # Filter: keep only grasps approaching from above.
            # When the orientation flip is enabled (GraspGen outputs reversed
            # orientations for this tool), the grasps that are visually correct
            # top-down have approach_z > 0 in GraspGen's frame — after the 180°
            # flip applied later they become approach_z < 0 (downward) for
            # execution.  Without the flip, the standard convention applies.
            approach_z = grasps_obj_np[:, 2, 2]
            if self._flip_orient:
                top_down = approach_z > 0   # inverted: GraspGen orientation is reversed
                filter_label = "top-down (inverted — flip enabled)"
            else:
                top_down = approach_z < 0   # standard: approach Z points down
                filter_label = "top-down"
            if top_down.sum() == 0:
                self._log(f"[GraspGen] WARNING: no {filter_label} grasps — keeping all")
            else:
                n_before = len(grasps_obj_np)
                grasps_obj_np = grasps_obj_np[top_down]
                grasp_conf_np = grasp_conf_np[top_down]
                n_topdown_removed = n_before - len(grasps_obj_np)
                self._log(f"[GraspGen] {filter_label.capitalize()} filter: {n_before} → "
                          f"{len(grasps_obj_np)} grasps")

            # ── Debug step 6: top-down filtered grasps ───────────────────────
            if self._debug_var.get() and self._vis:
                _grasps_td_work = t_center_inv @ grasps_obj_np
                self._vis.delete()
                visualize_pointcloud(self._vis, "pc_obj", pc_base, object_colors_filtered,
                                     size=args.scene_point_size)
                _gn = grasp_cfg.data.gripper_name
                for _i, _g in enumerate(_grasps_td_work):
                    visualize_grasp(self._vis, f"grasps_td/{_i:03d}/grasp", _g,
                                    color=[180, 255, 180], gripper_name=_gn, linewidth=0.8)
            self._debug_pause(
                f"Step 6/7 — Top-down filtered  ({len(grasps_obj_np)} grasps)")
            # ────────────────────────────────────────────────────────────────

            # Un-center: grasps go from centered → work frame (Z-up base)
            grasps_work_np = t_center_inv @ grasps_obj_np   # (4,4) @ (N,4,4) → (N,4,4)

            # grasps_work_np are in the Z-up "work" frame.
            # Flip them back to the real robot base frame for motor commands.
            if z_needs_flip:
                R_unflip_4x4 = np.eye(4)
                R_unflip_4x4[:3, :3] = R_flip  # 180° around X is its own inverse
                grasps_base_np = R_unflip_4x4 @ grasps_work_np   # (4,4) @ (N,4,4) → (N,4,4)
            else:
                grasps_base_np = grasps_work_np

            # Visualization uses the Z-up work frame (same as point cloud above)
            _, grasps_centered, scores, _ = _process_point_cloud(
                pc_base, grasps_work_np, grasp_conf_np)

            n_topdown_removed  = 0
            n_collision_removed = 0
            n_reach_removed    = 0

            # ── Collision filtering ──────────────────────────────────────────
            if self._collision_var.get():
                try:
                    gripper_info = get_gripper_info(grasp_cfg.data.gripper_name)
                    scene_col = scene_centered
                    if scene_col is not None and len(scene_col) > args.max_scene_points:
                        idx = np.random.choice(len(scene_col), args.max_scene_points,
                                               replace=False)
                        scene_col = scene_col[idx]
                    free_mask = filter_colliding_grasps(
                        scene_pc=scene_col,
                        grasp_poses=grasps_centered,
                        gripper_collision_mesh=gripper_info.collision_mesh,
                        collision_threshold=args.collision_threshold,
                    )
                    # ── Debug step 6b: collision filter result ────────────────
                    if self._debug_var.get() and self._vis:
                        self._vis.delete()
                        visualize_pointcloud(self._vis, "pc_obj", pc_base,
                                             object_colors_filtered, size=args.scene_point_size)
                        _gn = grasp_cfg.data.gripper_name
                        for _i, _g in enumerate(grasps_work_np):
                            _col = [50, 255, 50] if free_mask[_i] else [255, 50, 50]
                            visualize_grasp(self._vis, f"grasps_col/{_i:03d}/grasp", _g,
                                            color=_col, gripper_name=_gn, linewidth=0.8)
                        self._debug_pause(
                            f"Step 6b — Collision filter  "
                            f"({int(free_mask.sum())} free / "
                            f"{int((~free_mask).sum())} colliding)")
                    # ─────────────────────────────────────────────────────────
                    n_before = len(grasps_work_np)
                    grasps_work_np  = grasps_work_np[free_mask]
                    grasps_base_np  = grasps_base_np[free_mask]
                    grasps_centered = grasps_centered[free_mask]
                    grasp_conf_np   = grasp_conf_np[free_mask]
                    scores          = scores[free_mask]
                    n_collision_removed = n_before - int(free_mask.sum())
                    self._log(f"[Collision] {n_before} → {free_mask.sum()} "
                              f"collision-free grasps")
                    if len(grasps_work_np) == 0:
                        raise RuntimeError("All grasps filtered by collision check")
                except Exception as ce:
                    self._log(f"[Collision] WARNING: {ce}")
            # ── Reachability filtering ───────────────────────────────────────
            if self._reach_var.get():
                if not self._robot_connected or self._robot is None:
                    self._log("[Reach] WARNING: robot not connected — filter skipped")
                else:
                    _approach_m = float(self._approach_var.get()) / 1000.0
                    self._log(f"[Reach] Checking {len(grasps_work_np)} grasps…")
                    reach_mask = []
                    for _g in grasps_base_np:   # already in real robot base frame (m)
                        try:
                            _x, _y, _z, _rx, _ry, _rz = \
                                self._grasp_base_to_robot_coords(_g)
                            # pre-grasp along tool Z-axis (same logic as _on_execute)
                            _tool_z = _g[:3, 2]
                            _pre_mm = (_g[:3, 3] - _approach_m * _tool_z) * 1000.0
                            ok_g, _, _ = self._check_pose_valid(
                                _x, _y, _z, _rx, _ry, _rz)
                            ok_a, _, _ = self._check_pose_valid(
                                float(_pre_mm[0]), float(_pre_mm[1]), float(_pre_mm[2]),
                                _rx, _ry, _rz)
                            reach_mask.append(ok_g and ok_a)
                        except Exception:
                            reach_mask.append(False)
                    reach_mask = np.array(reach_mask, dtype=bool)
                    n_reach_removed = int((~reach_mask).sum())
                    grasps_work_np  = grasps_work_np[reach_mask]
                    grasps_base_np  = grasps_base_np[reach_mask]
                    grasps_centered = grasps_centered[reach_mask]
                    grasp_conf_np   = grasp_conf_np[reach_mask]
                    scores          = scores[reach_mask]
                    self._log(f"[Reach] {n_reach_removed} unreachable removed → "
                              f"{len(grasps_work_np)} reachable grasps")
                    if len(grasps_work_np) == 0:
                        raise RuntimeError("All grasps filtered by reachability check")
            # ────────────────────────────────────────────────────────────────

            # ── Orientation flip for vacuum gripper (execution only) ─────────
            # Visualization stays unchanged.  For the execution poses we flip the
            # approach direction (tool Z) so it always points downward in the robot
            # base frame (negative Z component), meaning the robot descends from
            # above rather than coming up from below.
            if self._flip_orient:
                _R_flip = np.eye(4)
                _R_flip[:3, :3] = np.array([[1., 0., 0.],
                                             [0., -1., 0.],
                                             [0., 0., -1.]])   # 180° around X
                grasps_base_np = np.array([g @ _R_flip for g in grasps_base_np])
                self._log("[GraspGen] Approach Z flipped for execution (tool approaches from above)")
            # ────────────────────────────────────────────────────────────────

            # Sort all grasps by confidence descending and store for retry loop
            sort_idx = np.argsort(grasp_conf_np.ravel())[::-1]
            self._all_grasps_base     = grasps_base_np[sort_idx]   # real base frame
            self._all_grasps_work     = grasps_work_np[sort_idx]   # Z-up for vis
            self._all_grasps_centered = grasps_centered[sort_idx]
            self._all_grasp_scores    = scores[sort_idx]
            self._current_grasp_idx   = 0

            if self._vis:
                gripper_name = grasp_cfg.data.gripper_name
                if self._debug_var.get():
                    # Rebuild full scene so step 7 looks identical to the normal result
                    self._vis.delete()
                    make_frame(self._vis, "robot_base", h=0.15, radius=0.005)
                    _cf = np.eye(4)
                    _cf[:3, :3] = R_flip @ T_cam2base_m[:3, :3]
                    _cf[:3, 3]  = cam_pos_base
                    self._vis["camera_frame"].set_transform(_cf)
                    make_frame(self._vis["camera_frame"], "axes", h=0.08, radius=0.003)
                    _of = np.eye(4); _of[:3, 3] = obj_centroid_base
                    self._vis["object_frame"].set_transform(_of)
                    make_frame(self._vis["object_frame"], "axes", h=0.06, radius=0.003)
                    _sc = scene_colors if scene_colors is not None else \
                        np.tile([[120,120,120]], (len(scene_pc),1)).astype(np.uint8)
                    _fp = np.vstack([scene_base, pc_base]) \
                        if scene_base is not None and len(scene_base) > 0 else pc_base
                    _fc = np.vstack([_sc, object_colors_filtered]) \
                        if scene_base is not None and len(scene_base) > 0 \
                        else object_colors_filtered
                    visualize_pointcloud(self._vis, "pc_full", _fp, _fc,
                                         size=args.scene_point_size)
                # Visualize grasps in Z-up work frame (matches point cloud)
                for i, g in enumerate(self._all_grasps_work):
                    col = [255, 255, 0] if i == 0 else self._all_grasp_scores[i].tolist()
                    visualize_grasp(self._vis, f"grasps/{i:03d}/grasp", g,
                                    color=col, gripper_name=gripper_name, linewidth=1.2)

            # ── Debug step 7: final sorted grasps ────────────────────────────
            self._debug_pause(
                f"Step 7/7 — Final sorted grasps  ({len(self._all_grasps_work)} poses, "
                f"best conf {float(grasp_conf_np.max()):.3f})")
            # ────────────────────────────────────────────────────────────────

            best_info = _save_best_grasp(grasps_base_np, grasp_conf_np)
            self._best_grasp_base = self._all_grasps_base[0].copy()  # robot base frame, meters

            conf_min, conf_max = float(grasp_conf_np.min()), float(grasp_conf_np.max())
            best_conf = best_info.get("confidence", 0.0)
            n_final = len(grasps_base_np)

            # ── Filter summary ───────────────────────────────────────────────
            self._log("─" * 36)
            self._log(f"[Summary] Generated:      {n_generated}")
            if n_topdown_removed:
                self._log(f"[Summary] − Top-down:     {n_topdown_removed}")
            if n_collision_removed:
                self._log(f"[Summary] − Collision:    {n_collision_removed}")
            if n_reach_removed:
                self._log(f"[Summary] − Unreachable:  {n_reach_removed}")
            self._log(f"[Summary] = Shown:        {n_final}")
            self._log(f"[Summary] Conf range:     [{conf_min:.3f} – {conf_max:.3f}]")
            self._log(f"[Summary] Best conf:      {best_conf:.4f}")
            # ────────────────────────────────────────────────────────────────

            summary = (f"{n_final} grasps [{conf_min:.3f}–{conf_max:.3f}]\n"
                       f"Best conf: {best_conf:.4f}")
            self._set_status(summary)

            # Transform to robot frame and update display
            self._update_grasp_display()

        except Exception as e:
            self._log(f"[ERROR] {e}")
            self._set_status(f"Error: {e}")
        finally:
            try:
                torch.cuda.synchronize(); torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            self._inference_running = False
            self._cb_queue.put(self._restore_run_btn)

    @staticmethod
    def _grasp_base_to_robot_coords(grasp_4x4_base: np.ndarray) -> tuple:
        """Extract (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg) from a
        grasp pose already in robot base frame (meters)."""
        pos_mm = grasp_4x4_base[:3, 3] * 1000.0
        R = grasp_4x4_base[:3, :3]
        if _SCIPY_OK:
            rz, ry, rx = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)
        else:
            rx, ry, rz = 0.0, 0.0, 0.0
        return float(pos_mm[0]), float(pos_mm[1]), float(pos_mm[2]), \
               float(rx), float(ry), float(rz)

    def _check_pose_valid(self, x, y, z, rx, ry, rz):
        """Workspace bounds check — delegates to the active robot driver."""
        if not self._robot_connected or self._robot is None:
            return False, None, "Robot not connected"
        return self._robot.check_reachability(x, y, z, rx, ry, rz)

    def _update_grasp_display(self):
        """Read robot-frame pose directly and update the display label + nav buttons."""
        if self._best_grasp_base is None:
            return
        idx = self._current_grasp_idx
        total = len(self._all_grasps_base) if self._all_grasps_base is not None else 0
        conf = float(self._all_grasp_scores[idx].mean()) if self._all_grasp_scores is not None else 0
        try:
            x, y, z, rx, ry, rz = self._grasp_base_to_robot_coords(
                self._best_grasp_base)
            txt = (f"X  = {x:+8.1f} mm\n"
                   f"Y  = {y:+8.1f} mm\n"
                   f"Z  = {z:+8.1f} mm\n"
                   f"Rx = {rx:+7.2f} °\n"
                   f"Ry = {ry:+7.2f} °\n"
                   f"Rz = {rz:+7.2f} °")
            # Approach direction = Z-axis of the grasp rotation matrix
            approach = self._best_grasp_base[:3, 2]  # 3rd column = local Z
            self._log(f"[Grasp {idx+1}/{total}] Robot frame: "
                      f"X={x:.1f} Y={y:.1f} Z={z:.1f} mm  "
                      f"Rx={rx:.1f} Ry={ry:.1f} Rz={rz:.1f} deg  "
                      f"approach=[{approach[0]:+.2f},{approach[1]:+.2f},{approach[2]:+.2f}]")
            def _ui_update(t=txt, i=idx, n=total):
                self._grasp_display.config(text=t, fg="#98c379")
                self._grasp_idx_label.config(text=f"{i+1} / {n}")
                self._execute_btn.configure(
                    state="normal" if self._robot_connected else "disabled")
                self._prev_grasp_btn.config(
                    state="normal" if i > 0 else "disabled")
                self._next_grasp_btn.config(
                    state="normal" if i < n - 1 else "disabled")
            self._cb_queue.put(_ui_update)
        except Exception as e:
            self._log(f"[Transform] ERROR: {e}")

    def _select_grasp(self, idx: int):
        """Switch to grasp at index idx, update Meshcat highlight and display."""
        if self._all_grasps_base is None or idx < 0:
            return
        total = len(self._all_grasps_base)
        if idx >= total:
            return
        # Un-highlight previous, highlight new
        prev_idx = self._current_grasp_idx
        if prev_idx != idx:
            self._highlight_grasp(prev_idx,
                                  self._all_grasp_scores[prev_idx].tolist()
                                  if self._all_grasp_scores is not None
                                  else [128, 128, 128])
        self._highlight_grasp(idx, [255, 255, 0])
        self._current_grasp_idx = idx
        self._best_grasp_base = self._all_grasps_base[idx].copy()
        self._update_grasp_display()

    def _on_prev_grasp(self):
        self._select_grasp(self._current_grasp_idx - 1)

    def _on_next_grasp(self):
        self._select_grasp(self._current_grasp_idx + 1)

    # ------------------------------------------------------------------
    # Robot connection
    # ------------------------------------------------------------------
    def _on_connect(self):
        if self._robot_connected:
            # Disconnect
            if self._robot:
                self._robot.close(); self._robot = None
            self._robot_connected = False
            self._robot_status.config(text="-- Disconnected", fg="#e06c75")
            self._connect_btn.config(text=">>  Connect Robot")
            for btn in (self._execute_btn, self._home_btn, self._save_home_btn,
                        self._save_sort_btn, self._go_sort_btn, self._recover_btn,
                        self._gripper_close_btn, self._gripper_open_btn):
                btn.config(state="disabled")
            self._tool_status.config(text="", fg="#555")
            return
        ip = self._ip_var.get().strip()
        self._connect_btn.config(state="disabled", text="Connecting…")
        threading.Thread(target=self._connect_worker, args=(ip,), daemon=True).start()

    def _connect_worker(self, ip):
        try:
            driver_name = self._robot_type_var.get()
            self._log(f"[Robot] Connecting to {ip} via {driver_name!r}…")
            robot = create_robot(driver_name, ip)
            self._log(f"[Robot] Connected to {ip}")
            # Clear any stale alarms first
            robot.clear_error()
            time.sleep(0.5)
            # Power on (takes ~10 s for controller boot)
            self._log("[Robot] PowerOn…")
            robot.power_on()
            time.sleep(3.0)
            # Enable
            self._log("[Robot] EnableRobot…")
            robot.enable()
            time.sleep(2.0)
            # Check mode; clear errors and retry once if needed
            mode = robot.get_mode()
            if mode == robot.MODE_ERROR:
                self._log("[Robot] Error state — clearing and re-enabling…")
                robot.clear_error(); time.sleep(1.0)
                robot.enable();      time.sleep(2.0)
            # Set default speed (low for safety)
            robot.set_speed(15)
            # Attach tool driver if selected
            tool_name = self._tool_var.get()
            tool_msg = ""
            if tool_name and tool_name != "(built-in / none)":
                try:
                    self._log(f"[Tool] Attaching '{tool_name}'…")
                    tool = create_tool(tool_name, robot=robot)
                    robot.attach_tool(tool)
                    tool_msg = f"Tool: {tool_name}"
                    self._log(f"[Tool] '{tool_name}' attached OK")
                except Exception as te:
                    tool_msg = f"Tool FAILED: {te}"
                    self._log(f"[Tool] Attach failed: {te}")
            self._robot = robot
            self._robot_connected = True
            self._log(f"[Robot] Ready  mode={robot.get_mode()}")
            self._cb_queue.put(lambda m=tool_msg: self._on_robot_connected_ui(m))
        except Exception as e:
            self._log(f"[Robot] Connection failed: {e}")
            self._cb_queue.put(lambda err=e: (
                self._robot_status.config(text=f"-- Error: {err}", fg="#e06c75"),
                self._connect_btn.config(state="normal", text=">>  Connect Robot")))

    def _on_robot_connected_ui(self, tool_msg: str = ""):
        self._robot_status.config(
            text=f"-- {self._robot_type_var.get()} @ {self._ip_var.get()}", fg="#98c379")
        if tool_msg:
            color = "#e06c75" if "FAILED" in tool_msg else "#98c379"
            self._tool_status.config(text=tool_msg, fg=color)
        else:
            self._tool_status.config(text="-- built-in / no tool", fg="#555")
        self._connect_btn.config(state="normal", text="Disconnect")
        self._home_btn.config(state="normal")
        self._recover_btn.config(state="normal")
        self._gripper_close_btn.config(state="normal")
        self._gripper_open_btn.config(state="normal")
        self._save_home_btn.config(state="normal")
        self._save_sort_btn.config(state="normal")
        if self._sort_joints is not None:
            self._go_sort_btn.config(state="normal")
        # Enable execute only if we already have a grasp
        if self._best_grasp_base is not None:
            self._execute_btn.config(state="normal")

    # ------------------------------------------------------------------
    # Execute grasp
    # ------------------------------------------------------------------
    def _highlight_grasp(self, idx: int, color):
        """Repaint a single grasp in meshcat.  Safe to call from any thread."""
        work = getattr(self, '_all_grasps_work', None)
        if self._vis is None or work is None:
            return
        if idx < 0 or idx >= len(work):
            return
        try:
            gripper_name = self._grasp_cfg.data.gripper_name
            visualize_grasp(self._vis, f"grasps/{idx:03d}/grasp",
                            work[idx],
                            color=color, gripper_name=gripper_name, linewidth=1.2)
        except Exception:
            pass

    def _on_execute(self):
        if not self._robot_connected or self._executing:
            return
        if self._best_grasp_base is None:
            messagebox.showwarning("No grasp", "Run GraspGen first.")
            return

        try:
            x, y, z, rx, ry, rz = self._grasp_base_to_robot_coords(
                self._best_grasp_base)
        except Exception as e:
            messagebox.showerror("Transform error", str(e)); return

        approach_mm = float(self._approach_var.get())
        speed = int(self._speed_var.get())
        tcp_z_offset = float(self._tcp_z_var.get())

        # ── Tool-Z approach: offset backwards along grasp Z-axis ─────────────
        approach_m = approach_mm / 1000.0
        tool_z = self._best_grasp_base[:3, 2]          # unit vector (approach dir)
        pre_pos_mm = (self._best_grasp_base[:3, 3] - approach_m * tool_z) * 1000.0
        x_pre, y_pre, z_pre = float(pre_pos_mm[0]), float(pre_pos_mm[1]), float(pre_pos_mm[2])
        # ── TCP Z offset: shift along gripper axis (+further in, -pull back) ──
        tcp_shift_mm = tcp_z_offset * tool_z           # world-frame shift vector
        x     += tcp_shift_mm[0]; y     += tcp_shift_mm[1]; z     += tcp_shift_mm[2]
        x_pre += tcp_shift_mm[0]; y_pre += tcp_shift_mm[1]; z_pre += tcp_shift_mm[2]
        # ─────────────────────────────────────────────────────────────────────

        # ── Reachability safety check ────────────────────────────────────────
        ok_g, joints, msg_g = self._check_pose_valid(x, y, z, rx, ry, rz)
        ok_a, _,      msg_a = self._check_pose_valid(x_pre, y_pre, z_pre, rx, ry, rz)
        self._log(f"[Reach] Grasp {'✓' if ok_g else '✗'} {msg_g}  "
                  f"Approach {'✓' if ok_a else '✗'} {msg_a}")
        if not ok_g or not ok_a:
            lines = ([f"• Grasp (X={x:.1f} Y={y:.1f} Z={z:.1f} mm): {msg_g}"]
                     if not ok_g else []) + \
                    ([f"• Approach (X={x_pre:.1f} Y={y_pre:.1f} Z={z_pre:.1f} mm): {msg_a}"]
                     if not ok_a else [])
            messagebox.showerror("Pose unreachable",
                                 "Robot cannot reach target:\n\n" + "\n".join(lines) +
                                 f"\n\nRx={rx:.1f} Ry={ry:.1f} Rz={rz:.1f} °")
            return
        # ────────────────────────────────────────────────────────────────────

        joints_str = (f"\nJoints: [{', '.join(f'{j:.1f}' for j in joints)}]°"
                      if joints else "")
        msg = (f"Execute grasp at:\n"
               f"  X={x:.1f}  Y={y:.1f}  Z={z:.1f} mm\n"
               f"  Rx={rx:.1f}  Ry={ry:.1f}  Rz={rz:.1f} °\n"
               f"  Approach: X={x_pre:.1f} Y={y_pre:.1f} Z={z_pre:.1f} mm{joints_str}\n\n"
               f"Speed: {speed}%\n\n"
               f"Make sure the workspace is clear before continuing!")
        if not messagebox.askyesno("Confirm Execute", msg):
            return

        self._executing = True
        self._execute_btn.config(state="disabled", text="Executing…")
        self._set_status("Executing grasp…")
        threading.Thread(target=self._execute_worker,
                          args=(x, y, z, x_pre, y_pre, z_pre, rx, ry, rz, speed),
                          daemon=True).start()

    def _try_single_grasp(self, robot, x, y, z, x_pre, y_pre, z_pre,
                           rx, ry, rz, speed):
        """Execute one full grasp sequence.  Raises on any robot error."""
        robot.set_speed(speed)

        self._log("[Execute] Vacuum OFF (open gripper)")
        self._set_status("Opening gripper…")
        robot.vacuum_off()
        time.sleep(0.3)

        self._log(f"[Execute] Pre-grasp  X={x_pre:.1f} Y={y_pre:.1f} Z={z_pre:.1f} mm")
        self._set_status("MovJ → pre-grasp…")
        # Use move_joint_nearest to avoid J1 full-rotation when IK picks a
        # configuration far from the current one (short Cartesian distance ≠
        # short joint distance for the Dobot solver).
        if hasattr(robot, "move_joint_nearest"):
            cmd_id = robot.move_joint_nearest(x_pre, y_pre, z_pre, rx, ry, rz)
        else:
            cmd_id = robot.move_linear(x_pre, y_pre, z_pre, rx, ry, rz)
        robot.wait_motion(cmd_id)

        self._log(f"[Execute] Grasp      X={x:.1f} Y={y:.1f} Z={z:.1f} mm")
        self._set_status("MovL → grasp…")
        cmd_id = robot.move_linear(x, y, z, rx, ry, rz)
        robot.wait_motion(cmd_id)

        self._log("[Execute] Vacuum ON — picking object")
        self._set_status("Vacuum ON…")
        robot.vacuum_on()
        time.sleep(0.8)

        self._log(f"[Execute] Retreat    X={x_pre:.1f} Y={y_pre:.1f} Z={z_pre:.1f} mm")
        self._set_status("MovL → retreat…")
        cmd_id = robot.move_linear(x_pre, y_pre, z_pre, rx, ry, rz)
        robot.wait_motion(cmd_id)

        sort_joints = self._sort_joints
        if sort_joints:
            self._log("[Execute] Going to sort position — dropping object")
            self._set_status("MovJ → sort…")
            cmd_id = robot.move_joint_angles(*sort_joints)
            robot.wait_motion(cmd_id)
            self._log("[Execute] Vacuum OFF — releasing at sort position")
            self._set_status("Releasing at sort…")
            robot.vacuum_off()
            time.sleep(0.5)
            # Return to home after drop
            home_joints = self._home_joints
            if home_joints:
                self._log("[Execute] Returning home (saved joints)")
                self._set_status("MovJ → home…")
                cmd_id = robot.move_joint_angles(*home_joints)
            else:
                hx, hy, hz, hrx, hry, hrz = HOME_POSE
                self._log("[Execute] Returning home (default)")
                self._set_status("MovL → home…")
                cmd_id = robot.move_linear(hx, hy, hz, hrx, hry, hrz)
            robot.wait_motion(cmd_id)
        else:
            home_joints = self._home_joints
            if home_joints:
                self._log("[Execute] Going home (saved joints)")
                self._set_status("MovJ → home…")
                cmd_id = robot.move_joint_angles(*home_joints)
            else:
                hx, hy, hz, hrx, hry, hrz = HOME_POSE
                self._log("[Execute] Going home (default pose)")
                self._set_status("MovL → home…")
                cmd_id = robot.move_linear(hx, hy, hz, hrx, hry, hrz)
            robot.wait_motion(cmd_id)
            self._log("[Execute] Vacuum OFF — releasing object")
            self._set_status("Releasing object…")
            robot.vacuum_off()
            time.sleep(0.5)

    def _recover_robot(self, robot):
        """Stop motion → vacuum off → clear alarm → re-enable (up to 3 retries) → go home.
        Returns True if robot reached an enabled/running state."""
        self._log("[Recover] Starting recovery sequence…")
        self._set_status("Recovering robot…")

        # 1. Stop any ongoing motion
        try:
            robot.stop()
            self._log("[Recover] StopRobot sent")
            time.sleep(0.4)
        except Exception as e:
            self._log(f"[Recover] stop error (ignored): {e}")

        # 2. Vacuum off — always safe
        try:
            robot.vacuum_off()
            self._log("[Recover] Vacuum OFF")
        except Exception as e:
            self._log(f"[Recover] vacuum_off error (ignored): {e}")

        # 3. Clear alarm + re-enable, up to 3 attempts
        recovered = False
        for attempt in range(1, 4):
            try:
                self._log(f"[Recover] Clear alarm + enable (attempt {attempt}/3)…")
                robot.clear_error()
                time.sleep(0.5)
                robot.enable()
                time.sleep(2.0)
                mode = robot.get_mode()
                self._log(f"[Recover] Robot mode after enable: {mode}")
                if mode in (robot.MODE_ENABLED, robot.MODE_RUNNING):
                    self._log("[Recover] Robot enabled successfully.")
                    recovered = True
                    break
                self._log(f"[Recover] Not yet enabled (mode={mode}), retrying…")
                time.sleep(1.5)
            except Exception as e:
                self._log(f"[Recover] Enable attempt {attempt} failed: {e}")
                time.sleep(1.5)

        if not recovered:
            self._log("[Recover] WARNING: robot did not reach enabled state after 3 attempts")

        # 4. Go home
        try:
            home_joints = self._home_joints
            if home_joints:
                self._log("[Recover] Returning home (saved joints)")
                cmd_id = robot.move_joint_angles(*home_joints)
            else:
                hx, hy, hz, hrx, hry, hrz = HOME_POSE
                self._log("[Recover] Returning home (default pose)")
                cmd_id = robot.move_linear(hx, hy, hz, hrx, hry, hrz)
            robot.wait_motion(cmd_id, timeout=45.0)
            self._log("[Recover] Back at home position.")
        except Exception as e:
            self._log(f"[Recover] WARNING: could not go home: {e}")

        return recovered

    def _execute_worker(self, x, y, z, x_pre, y_pre, z_pre, rx, ry, rz, speed):
        robot = self._robot
        try:
            if self._retry_grasps_var.get() and self._all_grasps_base is not None:
                # ── Auto-retry loop ────────────────────────────────────────────
                self._retry_stop_event.clear()
                self._cb_queue.put(
                    lambda: self._stop_retry_btn.config(state="normal"))

                start_idx = self._current_grasp_idx
                total = len(self._all_grasps_base)
                approach_m = float(self._approach_var.get()) / 1000.0
                tcp_z_offset = float(self._tcp_z_var.get())
                succeeded = False

                for idx in range(start_idx, total):
                    if self._retry_stop_event.is_set():
                        self._log("[Retry] Stopped by user.")
                        self._set_status("Retry stopped.")
                        break

                    grasp = self._all_grasps_base[idx]
                    try:
                        gx, gy, gz, grx, gry, grz = \
                            self._grasp_base_to_robot_coords(grasp)
                        tool_z = grasp[:3, 2]
                        pre_mm = (grasp[:3, 3] - approach_m * tool_z) * 1000.0
                        gx_pre = float(pre_mm[0])
                        gy_pre = float(pre_mm[1])
                        gz_pre = float(pre_mm[2])
                        tcp_shift_mm = tcp_z_offset * tool_z
                        gx     += tcp_shift_mm[0]; gy     += tcp_shift_mm[1]; gz     += tcp_shift_mm[2]
                        gx_pre += tcp_shift_mm[0]; gy_pre += tcp_shift_mm[1]; gz_pre += tcp_shift_mm[2]

                        self._log(f"[Retry] Trying grasp {idx + 1}/{total}  "
                                  f"X={gx:.1f} Y={gy:.1f} Z={gz:.1f} mm")
                        self._set_status(f"Retry {idx + 1}/{total}…")
                        self._cb_queue.put(lambda i=idx: self._select_grasp(i))

                        self._try_single_grasp(robot, gx, gy, gz,
                                               gx_pre, gy_pre, gz_pre,
                                               grx, gry, grz, speed)
                        self._log(f"[Retry] Grasp {idx + 1} succeeded!")
                        self._set_status("Done!")
                        succeeded = True
                        break

                    except Exception as e:
                        self._log(f"[Retry] Grasp {idx + 1} failed: {e}")
                        if idx + 1 < total and not self._retry_stop_event.is_set():
                            self._log("[Retry] Recovering and trying next grasp…")
                            self._recover_robot(robot)
                        # ──────────────────────────────────────────────────────

                self._cb_queue.put(
                    lambda: self._stop_retry_btn.config(state="disabled"))
                if not succeeded and not self._retry_stop_event.is_set():
                    self._log("[Retry] All grasps exhausted — none succeeded.")
                    self._set_status("All grasps tried — none succeeded.")

            else:
                # ── Single attempt (original behaviour) ───────────────────────
                self._try_single_grasp(robot, x, y, z, x_pre, y_pre, z_pre,
                                       rx, ry, rz, speed)
                self._log("[Execute] Done — object delivered to home position")
                self._set_status("Done!")

        except Exception as e:
            self._log(f"[Execute] ERROR: {e}")
            self._set_status(f"Execute error: {e}")
        finally:
            self._executing = False
            self._cb_queue.put(lambda: self._execute_btn.configure(
                state="normal", text="[>]  Execute Selected Grasp"))

    # ------------------------------------------------------------------
    # Sort position
    # ------------------------------------------------------------------
    def _on_save_sort(self):
        if not self._robot_connected:
            return
        threading.Thread(target=self._save_sort_worker, daemon=True).start()

    def _save_sort_worker(self):
        try:
            mode = self._robot.get_mode()
            if mode == self._robot.MODE_ERROR:
                self._robot.clear_error(); time.sleep(1.0)
                self._robot.enable();      time.sleep(1.5)
            joints = self._robot.get_angle()
            self._sort_joints = list(joints)
            self._save_positions()
            self._log(f"[Sort] Saved: "
                      f"J1={joints[0]:.2f} J2={joints[1]:.2f} J3={joints[2]:.2f} "
                      f"J4={joints[3]:.2f} J5={joints[4]:.2f} J6={joints[5]:.2f}")
            self._set_status(f"Sort saved: [{', '.join(f'{v:.1f}' for v in joints)}]")
            self._cb_queue.put(lambda j=list(joints): (
                self._sort_status.config(
                    text=f"Sort: [{', '.join(f'{v:.0f}' for v in j)}]",
                    fg="#98c379"),
                self._go_sort_btn.config(state="normal")))
        except Exception as e:
            self._log(f"[Sort] Could not save: {e}")

    def _on_go_sort(self):
        if not self._robot_connected or self._executing:
            return
        if self._sort_joints is None:
            from tkinter import messagebox as _mb
            _mb.showwarning("No sort position", "Save a sort position first.")
            return
        speed = int(self._speed_var.get())
        threading.Thread(target=self._go_sort_worker, args=(speed,), daemon=True).start()

    def _go_sort_worker(self, speed):
        try:
            self._ensure_enabled()
            self._robot.set_speed(speed)
            self._log(f"[Sort] Moving to sort: {[round(v,2) for v in self._sort_joints]}")
            self._set_status("Moving to sort position…")
            cmd_id = self._robot.move_joint_angles(*self._sort_joints)
            self._robot.wait_motion(cmd_id)
            self._log("[Sort] Done.")
            self._set_status("At sort position.")
        except Exception as e:
            self._log(f"[Sort] ERROR: {e}")

    # ------------------------------------------------------------------
    # Batch word-list execution
    # ------------------------------------------------------------------
    def _on_add_batch_word(self):
        word = self._batch_word_var.get().strip()
        if word:
            self._batch_listbox.insert("end", word)
            self._batch_word_var.set("")

    def _on_remove_batch_word(self):
        sel = self._batch_listbox.curselection()
        for i in reversed(sel):
            self._batch_listbox.delete(i)

    def _on_run_batch(self):
        if self._batch_running:
            return
        if self._inference_running:
            self._log("[Batch] Pipeline is busy — wait for it to finish.")
            return
        words = list(self._batch_listbox.get(0, "end"))
        if not words:
            self._log("[Batch] Word list is empty.")
            return
        if self._sampler is None:
            self._log("[Batch] No gripper config loaded.")
            return
        if not self._robot_connected or self._robot is None:
            self._log("[Batch] Robot not connected.")
            return
        if self._replay_frame is None and not self.camera._started \
                and self._current_frame is None:
            self._log("[Batch] No camera/frame available.")
            return
        speed = int(self._speed_var.get())
        self._batch_running = True
        self._batch_stop_event.clear()
        self._batch_run_btn.config(state="disabled", text="Running…")
        self._batch_stop_btn.config(state="normal")
        threading.Thread(target=self._batch_worker, args=(words, speed),
                         daemon=True).start()

    def _on_stop_batch(self):
        self._batch_stop_event.set()
        self._log("[Batch] Stop requested — will halt after current step.")
        self._batch_stop_btn.config(state="disabled")

    def _batch_worker(self, words, speed):
        """Loop through words top→bottom continuously until stopped.
        Each word: up to 3 attempts (capture → SAM3 → GraspGen → execute).
        On success or 3 failures, advance to next word.
        When the last word is reached, wrap back to the first."""
        self._log(f"[Batch] Starting continuous loop — {len(words)} word(s)")
        self._set_status("[Batch] Starting…")

        cycle = 0
        word_idx = 0
        while not self._batch_stop_event.is_set():
            word = words[word_idx]

            self._log(f"[Batch] ── cycle {cycle+1}  {word_idx+1}/{len(words)}: '{word}' ──")
            self._set_status(f"[Batch] cycle {cycle+1} · '{word}' ({word_idx+1}/{len(words)})")

            # Pre-check: recover if robot is in error state before attempting
            if self._robot_connected and self._robot is not None:
                try:
                    mode = self._robot.get_mode()
                    if mode == self._robot.MODE_ERROR:
                        self._log(f"[Batch] Robot in error state — recovering before '{word}'…")
                        self._recover_robot(self._robot)
                except Exception as _me:
                    self._log(f"[Batch] Could not check robot mode: {_me}")

            # Highlight current word in listbox
            self._cb_queue.put(lambda i=word_idx: (
                self._batch_listbox.selection_clear(0, "end"),
                self._batch_listbox.selection_set(i),
                self._batch_listbox.see(i)))

            success = False
            for attempt in range(1, 4):
                if self._batch_stop_event.is_set():
                    break

                self._log(f"[Batch]   Attempt {attempt}/3 — running pipeline…")
                self._set_status(
                    f"[Batch] '{word}' attempt {attempt}/3 — pipeline…")

                # Set prompt display
                self._cb_queue.put(lambda w=word: self._prompt_var.set(w))

                # Determine frame source
                if self._replay_frame is not None:
                    source = "replay"
                elif self.camera._started:
                    source = "camera"
                elif self._current_frame is not None:
                    source = "cached"
                else:
                    self._log("[Batch] No camera/frame — aborting.")
                    self._batch_running = False
                    self._cb_queue.put(lambda: self._batch_run_btn.config(
                        state="normal", text="[>]  Run Batch"))
                    self._cb_queue.put(lambda: self._batch_stop_btn.config(
                        state="disabled"))
                    return

                # Run the pipeline in a sub-thread, wait for completion
                pipeline_done = threading.Event()
                self._inference_running = True
                self._all_grasps_base = None
                self._best_grasp_base = None

                def _run_pipeline(src=source, prm=word):
                    try:
                        self._pipeline_worker(
                            src, prm, self._sampler, self._grasp_cfg)
                    except Exception as _pe:
                        self._log(f"[Batch] Pipeline exception: {_pe}")
                    finally:
                        pipeline_done.set()

                threading.Thread(target=_run_pipeline, daemon=True).start()
                if not pipeline_done.wait(timeout=300):
                    self._log(
                        f"[Batch]   Pipeline timed out — skipping attempt")
                    continue

                if self._all_grasps_base is None or \
                        len(self._all_grasps_base) == 0:
                    self._log(f"[Batch]   No grasps found — retrying")
                    continue

                # Execute the best grasp
                self._log(f"[Batch]   {len(self._all_grasps_base)} grasp(s) "
                          f"found — executing…")
                self._set_status(
                    f"[Batch] '{word}' attempt {attempt}/3 — executing…")

                approach_m = float(self._approach_var.get()) / 1000.0
                tcp_z_offset = float(self._tcp_z_var.get())
                grasp = self._all_grasps_base[0]
                try:
                    gx, gy, gz, grx, gry, grz = \
                        self._grasp_base_to_robot_coords(grasp)
                    tool_z = grasp[:3, 2]
                    pre_mm = (grasp[:3, 3] - approach_m * tool_z) * 1000.0
                    gx_pre = float(pre_mm[0])
                    gy_pre = float(pre_mm[1])
                    gz_pre = float(pre_mm[2])
                    tcp_shift_mm = tcp_z_offset * tool_z
                    gx     += tcp_shift_mm[0]; gy     += tcp_shift_mm[1]; gz     += tcp_shift_mm[2]
                    gx_pre += tcp_shift_mm[0]; gy_pre += tcp_shift_mm[1]; gz_pre += tcp_shift_mm[2]

                    ok_g, _, _ = self._check_pose_valid(
                        gx, gy, gz, grx, gry, grz)
                    ok_a, _, _ = self._check_pose_valid(
                        gx_pre, gy_pre, gz_pre, grx, gry, grz)
                    if not ok_g or not ok_a:
                        self._log(
                            f"[Batch]   Grasp pose unreachable — recovering and retrying")
                        try:
                            if self._robot.get_mode() == self._robot.MODE_ERROR:
                                self._recover_robot(self._robot)
                        except Exception:
                            pass
                        continue

                    self._executing = True
                    self._try_single_grasp(
                        self._robot, gx, gy, gz,
                        gx_pre, gy_pre, gz_pre,
                        grx, gry, grz, speed)
                    self._executing = False
                    self._log(
                        f"[Batch]   SUCCESS — '{word}' picked and placed!")
                    success = True
                    break

                except Exception as exec_e:
                    self._executing = False
                    self._log(
                        f"[Batch]   Execute failed: {exec_e}")
                    self._recover_robot(self._robot)

            if not success and not self._batch_stop_event.is_set():
                self._log(
                    f"[Batch] '{word}' — all 3 attempts exhausted, "
                    f"moving to next word.")

            # Advance to next word; wrap around at end of list
            word_idx += 1
            if word_idx >= len(words):
                word_idx = 0
                cycle += 1
                if not self._batch_stop_event.is_set():
                    self._log(f"[Batch] ── end of list — restarting cycle {cycle+1} ──")

        self._set_status("[Batch] Stopped.")

        self._batch_running = False
        self._cb_queue.put(lambda: self._batch_run_btn.config(
            state="normal", text="[>]  Run Batch"))
        self._cb_queue.put(lambda: self._batch_stop_btn.config(state="disabled"))

    # ------------------------------------------------------------------
    # Manual robot recovery
    # ------------------------------------------------------------------
    def _on_load_list(self):
        from tkinter import filedialog
        OBJECT_LISTS_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.askopenfilename(
            title="Load word list",
            initialdir=str(OBJECT_LISTS_DIR),
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            words = Path(path).read_text(encoding="utf-8").splitlines()
            words = [w.strip() for w in words if w.strip()]
        except Exception as e:
            self._log(f"[List] Failed to load '{path}': {e}")
            return
        self._batch_listbox.delete(0, "end")
        for w in words:
            self._batch_listbox.insert("end", w)
        self._last_list_path = path
        self._list_file_label.config(text=Path(path).name)
        self._save_positions()
        self._log(f"[List] Loaded {len(words)} word(s) from {Path(path).name}")

    def _on_save_list(self):
        from tkinter import filedialog
        words = list(self._batch_listbox.get(0, "end"))
        if not words:
            self._log("[List] Nothing to save — list is empty.")
            return
        OBJECT_LISTS_DIR.mkdir(parents=True, exist_ok=True)
        initial = Path(self._last_list_path).name if self._last_list_path else "word_list.txt"
        path = filedialog.asksaveasfilename(
            title="Save word list",
            initialdir=str(OBJECT_LISTS_DIR),
            initialfile=initial,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            Path(path).write_text("\n".join(words) + "\n", encoding="utf-8")
        except Exception as e:
            self._log(f"[List] Failed to save '{path}': {e}")
            return
        self._last_list_path = path
        self._list_file_label.config(text=Path(path).name)
        self._save_positions()
        self._log(f"[List] Saved {len(words)} word(s) to {Path(path).name}")

    # ------------------------------------------------------------------
    def _on_open_meshcat(self):
        try:
            url = self._vis.url() if self._vis is not None else "http://127.0.0.1:7000"
        except Exception:
            url = "http://127.0.0.1:7000"
        # Always print to terminal — guaranteed untruncated
        print(f"\n[Meshcat] {url}\n", flush=True)
        # Try xdg-open then webbrowser
        opened = False
        try:
            import subprocess
            subprocess.Popen(["xdg-open", url],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            opened = True
        except Exception:
            pass
        if not opened:
            try:
                opened = webbrowser.open(url)
            except Exception:
                pass
        if opened:
            self._log("[Meshcat] Browser opened.")
            return
        # No browser — show popup with selectable URL
        popup = tk.Toplevel(self.root)
        popup.title("Meshcat URL")
        popup.configure(bg="#1e1e1e")
        popup.resizable(False, False)
        tk.Label(popup, text="Open this URL in your browser:",
                 bg="#1e1e1e", fg="#abb2bf",
                 font=("Helvetica", 9)).pack(padx=16, pady=(12, 4))
        entry = tk.Entry(popup, font=("Courier", 11), width=30,
                         bg="#3a3a3a", fg="#98c379",
                         insertbackground="white", relief="flat", bd=4)
        entry.insert(0, url)
        entry.config(state="readonly")
        entry.pack(padx=16, pady=(0, 8))
        entry.select_range(0, "end")
        tk.Button(popup, text="Close", bg="#3a3a3a", fg="#aaa",
                  relief="flat", cursor="hand2", bd=0,
                  font=("Helvetica", 8),
                  command=popup.destroy).pack(pady=(0, 10))
        popup.transient(self.root)
        popup.grab_set()

    # ------------------------------------------------------------------
    def _on_recover_robot(self):
        if not self._robot_connected:
            return
        self._recover_btn.config(state="disabled", text="Stopping…")
        threading.Thread(target=self._recover_robot_worker, daemon=True).start()

    def _recover_robot_worker(self):
        try:
            # ── 1. Signal all running loops to stop ───────────────────────────
            self._retry_stop_event.set()
            self._batch_stop_event.set()
            self._executing = False

            # ── 2. Hard-stop robot motion ─────────────────────────────────────
            self._log("[Recover] Sending StopRobot…")
            self._cb_queue.put(lambda: self._recover_btn.config(text="Recovering…"))
            self._robot.stop()
            time.sleep(0.5)

            # ── 3. Full recovery (vacuum off → clear alarm → enable → home) ───
            ok = self._recover_robot(self._robot)
            self._set_status("Recovery done — robot enabled." if ok
                             else "Recovery attempted — check robot state.")
        except Exception as e:
            self._log(f"[Recover] Unexpected error: {e}")
        finally:
            self._cb_queue.put(lambda: self._recover_btn.config(
                state="normal", text="[!]  Recover Robot"))

    # ------------------------------------------------------------------
    # Utility buttons
    # ------------------------------------------------------------------
    def _on_save_home(self):
        if not self._robot_connected:
            return
        threading.Thread(target=self._save_home_worker, daemon=True).start()

    def _save_home_worker(self):
        try:
            mode = self._robot.get_mode()
            if mode == self._robot.MODE_ERROR:
                self._log("[Home] Robot in error — clearing before save…")
                self._robot.clear_error(); time.sleep(1.0)
                self._robot.enable();      time.sleep(1.5)
            joints = self._robot.get_angle()
            self._home_joints = list(joints)
            self._save_positions()
            self._log(f"[Home] Saved: "
                      f"J1={joints[0]:.2f} J2={joints[1]:.2f} J3={joints[2]:.2f} "
                      f"J4={joints[3]:.2f} J5={joints[4]:.2f} J6={joints[5]:.2f}")
            self._set_status(f"Home saved: [{', '.join(f'{v:.1f}' for v in joints)}]")
        except Exception as e:
            self._log(f"[Home] Could not read joint angles: {e}")

    def _on_home(self):
        if not self._robot_connected or self._executing:
            return
        speed = int(self._speed_var.get())
        threading.Thread(target=self._home_worker, args=(speed,), daemon=True).start()

    def _on_gripper_close(self):
        if not self._robot_connected:
            return
        def _worker():
            try:
                self._log("[Gripper] Closing…")
                self._robot.vacuum_on()
                self._log("[Gripper] Closed.")
            except Exception as e:
                self._log(f"[Gripper] Close error: {e}")
        threading.Thread(target=_worker, daemon=True).start()

    def _on_gripper_open(self):
        if not self._robot_connected:
            return
        def _worker():
            try:
                self._log("[Gripper] Opening…")
                self._robot.vacuum_off()
                self._log("[Gripper] Opened.")
            except Exception as e:
                self._log(f"[Gripper] Open error: {e}")
        threading.Thread(target=_worker, daemon=True).start()

    def _ensure_enabled(self):
        """Re-enable the robot if it slipped into error/disabled state."""
        mode = self._robot.get_mode()
        if mode == self._robot.MODE_ERROR:
            self._log("[Robot] Error state detected — clearing and re-enabling…")
            self._robot.clear_error(); time.sleep(0.5)
            self._robot.enable();      time.sleep(2.0)
        elif mode not in (self._robot.MODE_ENABLED, self._robot.MODE_RUNNING):
            self._log(f"[Robot] Unexpected mode {mode} — enabling…")
            self._robot.enable(); time.sleep(2.0)

    def _home_worker(self, speed):
        try:
            self._ensure_enabled()
            self._robot.set_speed(speed)
            joints = getattr(self, "_home_joints", None)
            if joints:
                self._log(f"[Home] Moving to saved joints: {[round(v,2) for v in joints]}")
                self._set_status("Moving to home (joint angles)…")
                cmd_id = self._robot.move_joint_angles(*joints)
            else:
                x, y, z, rx, ry, rz = HOME_POSE
                self._log(f"[Home] Moving to default home: {HOME_POSE}")
                self._set_status("Moving to home…")
                cmd_id = self._robot.move_linear(x, y, z, rx, ry, rz)
            self._robot.wait_motion(cmd_id)
            self._log("[Home] Done.")
            self._set_status("At home position.")
        except Exception as e:
            self._log(f"[Home] ERROR: {e}")



# ===========================================================================
# Configuration defaults
# ===========================================================================
class _Args:
    robot_ip           = ROBOT_IP_DEFAULT
    calib_file         = CALIB_FILE
    sam3_socket        = "/tmp/sam3_server.sock"
    sam3_device        = "cuda:0"
    sam3_no_fp16       = False
    num_grasps         = 200
    grasp_threshold    = -1.0
    topk_num_grasps    = 100
    collision_filter   = False
    collision_threshold= 0.02
    max_scene_points   = 8192
    max_object_points  = 12000
    scene_point_size   = 0.008
    object_point_size  = 0.012
    target_object_id   = 1


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Grasp Execute Pipeline")
    ap.add_argument("--robot-ip",  default=ROBOT_IP_DEFAULT)
    ap.add_argument("--calib-file",default=CALIB_FILE)
    ap.add_argument("--sam3-socket", default="/tmp/sam3_server.sock")
    cli = ap.parse_args()

    args = _Args()
    args.robot_ip    = cli.robot_ip
    args.calib_file  = cli.calib_file
    args.sam3_socket = cli.sam3_socket

    config_map = scan_checkpoints(CHECKPOINTS_DIR)

    # ── Splash / loading screen ──────────────────────────────────────────
    splash = tk.Tk()
    splash.title("AnySort — Loading")
    splash.configure(bg="#1e1e1e")
    splash.resizable(True, False)
    _sw, _sh = 440, 210
    splash.update_idletasks()
    _x = (splash.winfo_screenwidth() - _sw) // 2
    _y = (splash.winfo_screenheight() - _sh) // 2
    splash.geometry(f"{_sw}x{_sh}+{_x}+{_y}")

    _splash_closed = False

    def _splash_close():
        nonlocal _splash_closed
        _splash_closed = True
        splash.destroy()
        print("Startup cancelled by user.")
        sys.exit(0)

    splash.protocol("WM_DELETE_WINDOW", _splash_close)

    tk.Label(splash, text="AnySort", bg="#1e1e1e", fg="#61afef",
             font=("Helvetica", 22, "bold")).pack(pady=(24, 2))
    _step_var = tk.StringVar(value="Initialising…")
    tk.Label(splash, textvariable=_step_var, bg="#1e1e1e", fg="#abb2bf",
             font=("Helvetica", 9)).pack()

    _pbar = ttk.Progressbar(splash, mode="determinate", maximum=100)
    _pbar.pack(fill="x", padx=40, pady=(14, 6))

    _detail_var = tk.StringVar(value="")
    tk.Label(splash, textvariable=_detail_var, bg="#1e1e1e", fg="#666",
             font=("Courier", 8)).pack()

    _terminal_var = tk.StringVar(value="")
    tk.Label(splash, textvariable=_terminal_var, bg="#1e1e1e", fg="#22d0fc",
             font=("Courier", 8), wraplength=400, justify="left").pack(pady=(4, 0))

    splash.update()

    # Thread-safe queue for splash updates (no splash.after from threads)
    _splash_q = queue.Queue()
    _loaded = {}
    _loader_done = threading.Event()

    # Intercept stdout — forward to real stdout + push last line to splash
    class _SplashStdout:
        def __init__(self, real):
            self._real = real
        def write(self, s):
            self._real.write(s)
            line = s.strip()
            if line:
                _splash_q.put(("__terminal__", line, None))
        def flush(self):
            self._real.flush()
        def __getattr__(self, a):
            return getattr(self._real, a)

    _real_stdout = sys.stdout
    sys.stdout = _SplashStdout(_real_stdout)

    def _flush_splash():
        """Polled from main thread every 100ms."""
        if _splash_closed:
            return
        while not _splash_q.empty():
            step, detail, pct = _splash_q.get_nowait()
            if step == "__terminal__":
                _terminal_var.set(detail[:80])  # cap width
            else:
                _step_var.set(step)
                _detail_var.set(detail)
                if pct is not None:
                    _pbar["value"] = pct
        if _loader_done.is_set():
            splash.quit()
            return
        splash.after(100, _flush_splash)

    def _loader():
        def _set(step, detail="", pct=None):
            print(f"  [{pct:3d}%] {step}  {detail}" if pct is not None
                  else f"        {step}  {detail}")
            _splash_q.put((step, detail, pct))

        try:
            # 1 — Configs
            n = len(config_map)
            _set("Scanning gripper configs…",
                 f"{n} found" if n else "WARNING: none found", 10)

            # 2 — Meshcat
            _set("Starting Meshcat visualiser…", "ZMQ + HTTP server", 20)
            vis = None
            try:
                import meshcat as _mc
                vis = _mc.Visualizer()
                vis.delete()
                _set("Meshcat ready.", vis.url(), 35)
            except Exception as e:
                _set("Meshcat unavailable.", str(e), 35)
            _loaded["vis"] = vis

            # 3 — Camera
            _set("Opening camera…", "Orbbec Gemini 2", 45)
            camera = OrbbecCamera()
            try:
                camera.start()
                _set("Camera connected.", "", 55)
            except Exception as e:
                _set("Camera not found.", "connect later via UI", 55)
            _loaded["camera"] = camera

            # 4 — SAM3
            sock = args.sam3_socket
            import socket as _s
            try:
                s = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM)
                s.settimeout(2.0); s.connect(sock); s.close()
                _set("SAM3 server found.", "existing socket", 85)
                _loaded["sam3_proc"] = None
            except OSError:
                _set("Starting SAM3 server…",
                     "loading model weights (may take minutes)", 60)
                try:
                    proc = start_sam3_server(sock, args.sam3_device,
                                             not args.sam3_no_fp16)
                    _loaded["sam3_proc"] = proc
                    _set("SAM3 ready.", "", 85)
                except Exception as e:
                    _loaded["sam3_proc"] = None
                    _set("SAM3 FAILED.", str(e), 85)

            # 5 — Default gripper config / GraspGen weights
            if config_map:
                cfg_name = next(iter(config_map))
                cfg_path = config_map[cfg_name]
                _set("Loading GraspGen weights…", cfg_name, 90)
                try:
                    cfg = load_grasp_cfg(cfg_path)
                    sampler = GraspGenSampler(cfg)
                    _loaded["preloaded_config"] = {
                        "name": cfg_name, "cfg": cfg, "sampler": sampler}
                    _set("GraspGen ready.", cfg.data.gripper_name, 95)
                except Exception as e:
                    _set("GraspGen load failed.", str(e), 95)
                    _loaded["preloaded_config"] = None
            else:
                _loaded["preloaded_config"] = None

            # 6 — Done
            _set("All systems ready.", "Opening AnySort…", 100)
            import time as _t; _t.sleep(0.5)

        except Exception as e:
            print(f"[Loader] Error: {e}")
        finally:
            _loader_done.set()

    splash.after(100, _flush_splash)
    threading.Thread(target=_loader, daemon=True).start()
    splash.mainloop()
    sys.stdout = _real_stdout   # restore stdout
    if not _splash_closed:
        splash.destroy()

    # ── Main UI ─────────────────────────────────────────────────────────
    camera          = _loaded.get("camera",           OrbbecCamera())
    vis             = _loaded.get("vis",              None)
    sam3_proc       = _loaded.get("sam3_proc",        None)
    preloaded_config = _loaded.get("preloaded_config", None)

    root = tk.Tk()
    app_ref = [GraspExecuteApp(root, args, camera, config_map,
                               sam3_proc=sam3_proc, vis=vis,
                               preloaded_config=preloaded_config)]

    def _on_close():
        print("Shutting down…")
        app_ref[0]._debug_event.set()
        camera.stop()
        if app_ref[0].sam3_proc:
            try:
                app_ref[0].sam3_proc.terminate()
                app_ref[0].sam3_proc.wait(timeout=5)
            except Exception:
                try:
                    app_ref[0].sam3_proc.kill()
                except Exception:
                    pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
