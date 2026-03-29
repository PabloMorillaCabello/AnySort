#!/usr/bin/env python3
"""
Grasp Execute Pipeline
======================
Full pipeline: Camera → SAM3 → GraspGen → hand-eye transform → Dobot execution.

Extends demo_orbbec_gemini2_persistent_sam3.py with:
  - Loads hand-eye calibration (T_cam2base) from hand_eye_calibration.py output
  - Transforms the best GraspGen pose from camera frame → robot base frame
  - Sends the robot to the grasp position (pre-grasp → grasp → retreat)
  - Vacuum ON / OFF buttons for the vacuum tool

Usage — two-terminal workflow:
  # Terminal 1: start SAM3 server (sam3 Python env)
  /opt/sam3env/bin/python /ros2_ws/scripts/sam3_server.py

  # Terminal 2: run this UI (GraspGen Python env)
  python3 /ros2_ws/scripts/TEST/grasp_execute_pipeline.py

Meshcat visualisation: http://127.0.0.1:7000
"""

import gc
import json
import os
import queue
import re
import socket
import subprocess
import threading
import time
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
SAM3_SERVER_SCRIPT = "/ros2_ws/scripts/sam3_server.py"
CALIB_FILE       = "/ros2_ws/data/calibration/hand_eye_calib.npz"
RESULTS_DIR      = Path("/ros2_ws/results")

ROBOT_IP_DEFAULT = "192.168.5.1"
ROBOT_PORT       = 29999
VACUUM_DO_PORT   = 1        # Dobot digital output port for vacuum (1-based)
APPROACH_OFFSET  = 80       # mm above grasp position for pre-grasp approach
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


def _wait_for_sam3_socket(sock_path, timeout=120.0):
    import socket as _s
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM)
            s.settimeout(2.0); s.connect(sock_path); s.close(); return
        except OSError:
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
    _wait_for_sam3_socket(sock_path, timeout=120.0)
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


def _extract_intrinsics(depth_profile, depth_frame):
    for obj in [depth_frame, depth_profile]:
        if obj is None:
            continue
        for method in ["get_intrinsic", "get_camera_intrinsic", "get_intrinsics"]:
            if not hasattr(obj, method):
                continue
            intr = getattr(obj, method)()
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
    def __init__(self):
        from pyorbbecsdk import Config, OBSensorType, Pipeline, OBFormat
        self._OBFormat = OBFormat
        self._lock = threading.Lock()
        self._latest_rgb = None
        self._latest_depth_m = None
        self._latest_intrinsics = None
        self._running = False
        self._depth_profile = None

    def start(self):
        from pyorbbecsdk import Config, OBSensorType, Pipeline
        pipeline = Pipeline()
        config = Config()
        depth_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        self._depth_profile = depth_profile
        color_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        config.enable_stream(color_list.get_default_video_stream_profile())
        pipeline.start(config)
        self._pipeline = pipeline
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        OBFormat = self._OBFormat
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(100)
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
                dh, dw = df.get_height(), df.get_width()
                depth_raw = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(dh, dw)
                scale = float(df.get_depth_scale())
                depth_m = depth_raw.astype(np.float32) * scale
                valid = depth_m[(depth_m > 0.05) & np.isfinite(depth_m)]
                if valid.size > 0 and float(np.median(valid)) > 20.0:
                    depth_m /= 1000.0
                if rgb.shape[0] != dh or rgb.shape[1] != dw:
                    rgb = np.array(PILImage.fromarray(rgb).resize((dw, dh), PILImage.BILINEAR))
                intr = _extract_intrinsics(self._depth_profile, df)
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
            return self._latest_rgb.copy(), self._latest_depth_m.copy(), self._latest_intrinsics

    def stop(self):
        self._running = False
        try:
            self._pipeline.stop()
        except Exception:
            pass


# ===========================================================================
# Dobot Dashboard  — wraps the real DobotApiDashboard + DobotApiFeedBack
# ===========================================================================
try:
    import sys as _sys
    _sys.path.insert(0, "/opt/Dobot_hv")
    from dobot_api import DobotApiDashboard as _DobotApiDashboard
    from dobot_api import DobotApiFeedBack  as _DobotApiFeedBack
    _DOBOT_API_OK = True
except ImportError as _e:
    _DOBOT_API_OK = False
    print(f"[Dobot] WARNING: could not import dobot_api — robot disabled ({_e})")


def _parse_result_id(resp):
    """Parse dashboard response → [ErrorID, CommandID, ...]"""
    if resp is None:
        return [2]
    if "Not Tcp" in str(resp):
        return [1]
    nums = re.findall(r"-?\d+", str(resp))
    return [int(n) for n in nums] if nums else [2]


class DobotDashboard:
    """
    Thin wrapper around DobotApiDashboard (port 29999) + DobotApiFeedBack (port 30004).
    Exposes the same interface the rest of the pipeline expects while using the
    correct Dobot TCP API command format.
    """
    MODE_RUNNING = 7
    MODE_ERROR   = 9
    MODE_ENABLED = 5
    _FEEDBACK_PORT = 30004

    def __init__(self, ip, port=ROBOT_PORT):
        if not _DOBOT_API_OK:
            raise RuntimeError("dobot_api not available — cannot connect to robot")
        self._ip = ip
        self._dashboard = _DobotApiDashboard(ip, port)
        self._feed      = _DobotApiFeedBack(ip, self._FEEDBACK_PORT)
        self._lock      = threading.Lock()
        self._mode      = -1
        self._cmd_id    = -1
        self._speed     = 20  # current speed %
        # Start feedback thread
        self._feed_running = True
        threading.Thread(target=self._feed_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # Feedback loop (250 Hz)
    # ------------------------------------------------------------------
    def _feed_loop(self):
        while self._feed_running:
            try:
                info = self._feed.feedBackData()
                if info is not None and hex(int(info["TestValue"][0])) == "0x123456789abcdef":
                    with self._lock:
                        self._mode   = int(info["RobotMode"][0])
                        self._cmd_id = int(info["CurrentCommandId"][0])
            except Exception:
                pass
            time.sleep(0.004)

    def get_mode(self):
        with self._lock:
            return self._mode

    def _get_cmd_id(self):
        with self._lock:
            return self._cmd_id

    # ------------------------------------------------------------------
    # Basic commands
    # ------------------------------------------------------------------
    def enable(self):
        return self._dashboard.EnableRobot()

    def power_on(self):
        return self._dashboard.PowerOn()

    def clear_error(self):
        return self._dashboard.ClearError()

    def set_speed(self, p):
        self._speed = max(1, min(100, int(p)))
        return self._dashboard.SpeedFactor(self._speed)

    def vacuum_on(self, port=VACUUM_DO_PORT):
        return self._dashboard.ToolDO(port, 1)

    def vacuum_off(self, port=VACUUM_DO_PORT):
        return self._dashboard.ToolDO(port, 0)

    # ------------------------------------------------------------------
    # Position getters (parse numeric response)
    # ------------------------------------------------------------------
    def _nums(self, resp):
        return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(resp))]

    def get_pose(self):
        nums = self._nums(self._dashboard.GetPose())
        if len(nums) >= 7:
            return tuple(nums[1:7])
        if len(nums) >= 6:
            return tuple(nums[:6])
        raise ValueError(f"Cannot parse pose from GetPose response")

    def get_angle(self):
        """Returns (J1, J2, J3, J4, J5, J6) in degrees."""
        resp = self._dashboard.GetAngle()
        nums = self._nums(resp)
        if len(nums) >= 7:
            return tuple(nums[1:7])
        if len(nums) >= 6:
            return tuple(nums[:6])
        raise ValueError(f"Cannot parse joints — raw: {resp!r}")

    # ------------------------------------------------------------------
    # Motion commands (return CommandID for wait_motion)
    # ------------------------------------------------------------------
    def _send_motion(self, resp):
        """Parse response, return CommandID (or -1 on error)."""
        parsed = _parse_result_id(resp)
        if len(parsed) < 2 or parsed[0] != 0:
            raise RuntimeError(f"Move rejected (ErrorID={parsed[0] if parsed else '?'}): {resp!r}")
        return parsed[1]

    def move_linear(self, x, y, z, rx, ry, rz):
        resp = self._dashboard.MovL(
            x, y, z, rx, ry, rz,
            0,                  # coordinateMode=0 → pose (Cartesian)
            a=self._speed,
            v=self._speed
        )
        return self._send_motion(resp)

    def move_joint_angles(self, j1, j2, j3, j4, j5, j6):
        """Joint-space move with actual joint angles (degrees).
        Wraps each angle into [-360, 360] — the Dobot controller rejects values outside
        this range (e.g. J6 can accumulate beyond ±360 via continuous rotation).
        """
        def _wrap(a):
            # Bring into (-360, 360] via modulo, preserving sign
            a = a % 360.0
            if a > 180.0:
                a -= 360.0
            return a
        j1, j2, j3, j4, j5, j6 = (_wrap(a) for a in (j1, j2, j3, j4, j5, j6))
        resp = self._dashboard.MovJ(
            j1, j2, j3, j4, j5, j6,
            1,                  # coordinateMode=1 → joint
            a=self._speed,
            v=self._speed
        )
        return self._send_motion(resp)

    # ------------------------------------------------------------------
    # Wait for a specific command to finish
    # ------------------------------------------------------------------
    def wait_motion(self, cmd_id, timeout=90.0):
        """Block until the robot finishes executing command cmd_id."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                mode   = self._mode
                cur_id = self._cmd_id
            if cur_id > cmd_id or (mode == self.MODE_ENABLED and cur_id == cmd_id):
                return True
            if mode == self.MODE_ERROR:
                raise RuntimeError("Robot entered error state during motion")
            time.sleep(0.1)
        raise TimeoutError(f"Motion timeout after {timeout:.0f}s (waiting for cmd {cmd_id})")

    def wait_idle(self, timeout=90.0):
        """Legacy helper: just wait until mode is not RUNNING."""
        t0 = time.time()
        time.sleep(0.4)
        while time.time() - t0 < timeout:
            with self._lock:
                mode = self._mode
            if mode != self.MODE_RUNNING:
                return True
            time.sleep(0.15)
        return False

    def close(self):
        self._feed_running = False
        try:
            self._dashboard.close()
        except Exception:
            pass
        try:
            self._feed.close()
        except Exception:
            pass


# ===========================================================================
# Hand-eye calibration helpers
# ===========================================================================
def load_calibration(path=CALIB_FILE):
    """Load T_cam2base (4×4).  Translation stored in mm."""
    data = np.load(path)
    T = data["T_cam2base"]              # 4×4, t in mm
    K = data.get("camera_matrix", None)
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
    grasps_centered = np.array([t_center @ np.array(g) for g in grasps.tolist()])
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
                 config_map: dict, sam3_proc=None, vis=None):
        self.root       = root
        self.args       = args
        self.camera     = camera
        self.config_map = config_map
        self.sam3_proc  = sam3_proc
        self._vis       = vis

        # GraspGen state
        self._loaded_config_name = None
        self._grasp_cfg  = None
        self._sampler    = None
        self._last_mask  = None
        self._best_grasp_cam: np.ndarray = None   # 4×4, camera frame, meters
        self._best_grasp_info: dict = {}

        # Hand-eye calibration
        self._T_cam2base = None    # 4×4, t in mm
        self._calib_K    = None

        # Robot
        self._robot: DobotDashboard = None
        self._robot_connected = False

        # Flags
        self._inference_running = False
        self._config_loading    = False
        self._executing         = False
        self._show_mask         = tk.BooleanVar(value=True)

        self._log_queue = queue.Queue()
        self._cb_queue  = queue.Queue()

        self._build_ui()
        self._load_calibration()
        self._start_config_if_available()
        self._schedule_preview()
        self._schedule_flush()

    # ------------------------------------------------------------------
    # Startup helpers
    # ------------------------------------------------------------------
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
                text=f"✓  Calibration loaded  t=[{t[0]:.0f},{t[1]:.0f},{t[2]:.0f}]mm",
                fg="#98c379"))
        except Exception as e:
            self._log(f"[Calib] WARNING: {e}")
            self._cb_queue.put(lambda: self._calib_status.config(
                text="⚠  No calibration — run hand_eye_calibration.py first",
                fg="#e06c75"))

    def _start_config_if_available(self):
        if self.config_map:
            first = next(iter(self.config_map))
            self._config_combo.set(first)
            self._start_config_load(first)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.root.title("Grasp Execute Pipeline  —  SAM3 + GraspGen + Dobot")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(False, False)

        # ---- Left: camera preview ----
        left = tk.Frame(self.root, bg="#1e1e1e")
        left.grid(row=0, column=0, sticky="nsew", padx=(8,4), pady=8)

        self._canvas = tk.Canvas(left, width=_PREVIEW_W, height=_PREVIEW_H,
                                  bg="#111", highlightthickness=1, highlightbackground="#444")
        self._canvas.pack()
        self._canvas_img_id = self._canvas.create_image(0, 0, anchor="nw")
        self._no_cam_txt = self._canvas.create_text(
            _PREVIEW_W//2, _PREVIEW_H//2, text="Waiting for camera…",
            fill="#555", font=("Helvetica", 14))

        tk.Checkbutton(left, text="Show mask overlay", variable=self._show_mask,
                       bg="#1e1e1e", fg="#aaa", activebackground="#1e1e1e",
                       selectcolor="#2d2d2d").pack(anchor="w", padx=4, pady=(4,0))

        # ---- Middle: GraspGen controls ----
        mid = tk.Frame(self.root, bg="#2d2d2d", width=310)
        mid.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)
        mid.grid_propagate(False)

        tk.Label(mid, text="GraspGen", bg="#2d2d2d", fg="#61afef",
                 font=("Helvetica", 15, "bold")).pack(pady=(14,0))
        tk.Label(mid, text="SAM3  ·  Orbbec Gemini 2  ·  GraspGen",
                 bg="#2d2d2d", fg="#555", font=("Helvetica", 8)).pack(pady=(0,6))

        ttk.Separator(mid, orient="horizontal").pack(fill="x", padx=8, pady=2)

        # Calibration status
        self._calib_status = tk.Label(mid, text="Loading calibration…",
                                       bg="#2d2d2d", fg="#888",
                                       font=("Courier", 8), wraplength=280)
        self._calib_status.pack(anchor="w", padx=12, pady=(4,2))

        ttk.Separator(mid, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # Gripper config
        tk.Label(mid, text="Gripper / Tool:", bg="#2d2d2d", fg="#e5c07b",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(4,2))
        names = list(self.config_map.keys())
        self._config_combo = ttk.Combobox(mid, values=names, state="readonly",
                                           font=("Helvetica", 9), width=34)
        self._config_combo.pack(padx=12, fill="x")
        if not names:
            self._config_combo.set("(no configs found)")
            self._config_combo.configure(state="disabled")
        self._config_combo.bind("<<ComboboxSelected>>", self._on_config_change)
        self._config_hint = tk.Label(mid, text="", bg="#2d2d2d", fg="#555",
                                      font=("Courier", 7), wraplength=280, justify="left")
        self._config_hint.pack(anchor="w", padx=12)

        ttk.Separator(mid, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # Object prompt
        tk.Label(mid, text="Object prompt:", bg="#2d2d2d", fg="#e5c07b",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(4,2))
        self._prompt_var = tk.StringVar(value="")
        self._prompt_entry = tk.Entry(mid, textvariable=self._prompt_var,
                                       bg="#3a3a3a", fg="white", insertbackground="white",
                                       relief="flat", font=("Helvetica", 11), bd=4)
        self._prompt_entry.pack(padx=12, pady=(0,8), fill="x")
        self._prompt_entry.bind("<Return>", lambda _: self._on_run())

        self._run_btn = tk.Button(mid, text="▶  Capture & Run GraspGen",
                                   bg="#61afef", fg="#1e1e1e", activebackground="#4d9bd6",
                                   font=("Helvetica", 11, "bold"), relief="flat",
                                   cursor="hand2", bd=0, command=self._on_run)
        self._run_btn.pack(padx=12, pady=2, fill="x", ipady=10)

        tk.Button(mid, text="✕  Clear mask", bg="#3a3a3a", fg="#aaa",
                  activebackground="#444", font=("Helvetica", 9), relief="flat",
                  cursor="hand2", bd=0, command=lambda: setattr(self, "_last_mask", None)
                  ).pack(padx=12, pady=(2,6), fill="x", ipady=4)

        ttk.Separator(mid, orient="horizontal").pack(fill="x", padx=8, pady=2)

        tk.Label(mid, text="Status:", bg="#2d2d2d", fg="#666",
                 font=("Helvetica", 9)).pack(anchor="w", padx=12)
        self._status_var = tk.StringVar(value="Waiting for camera…")
        tk.Label(mid, textvariable=self._status_var, bg="#2d2d2d", fg="#98c379",
                 font=("Helvetica", 9), wraplength=280, justify="left"
                 ).pack(anchor="w", padx=12, pady=(2,6))

        ttk.Separator(mid, orient="horizontal").pack(fill="x", padx=8, pady=2)
        tk.Label(mid, text="Log:", bg="#2d2d2d", fg="#666",
                 font=("Helvetica", 9)).pack(anchor="w", padx=12)
        self._log_text = scrolledtext.ScrolledText(
            mid, width=34, height=12, bg="#1e1e1e", fg="#abb2bf",
            font=("Courier", 8), state="disabled", relief="flat")
        self._log_text.pack(padx=12, pady=(2,8), fill="both", expand=True)

        # ---- Right: Robot execution ----
        right = tk.Frame(self.root, bg="#252525", width=290)
        right.grid(row=0, column=2, sticky="nsew", padx=(4,8), pady=8)
        right.grid_propagate(False)

        tk.Label(right, text="Robot Execution", bg="#252525", fg="#c678dd",
                 font=("Helvetica", 14, "bold")).pack(pady=(14,0))
        tk.Label(right, text="Dobot CR  ·  Vacuum Tool",
                 bg="#252525", fg="#555", font=("Helvetica", 8)).pack(pady=(0,8))

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=2)

        # Connection
        tk.Label(right, text="Connection", bg="#252525", fg="#e5c07b",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(6,2))

        row = tk.Frame(right, bg="#252525"); row.pack(fill="x", padx=12, pady=2)
        tk.Label(row, text="IP:", bg="#252525", fg="#ccc",
                 font=("Helvetica", 9), width=4, anchor="w").pack(side="left")
        self._ip_var = tk.StringVar(value=self.args.robot_ip)
        tk.Entry(row, textvariable=self._ip_var, width=15, bg="#3a3a3a",
                 fg="white", insertbackground="white", relief="flat",
                 font=("Helvetica", 9)).pack(side="left", padx=(0,6))

        self._connect_btn = tk.Button(right, text="⚡  Connect Robot",
                                       bg="#4a5568", fg="white", activebackground="#5a6578",
                                       relief="flat", cursor="hand2", bd=0,
                                       font=("Helvetica", 9), command=self._on_connect)
        self._connect_btn.pack(padx=12, pady=4, fill="x", ipady=5)

        self._robot_status = tk.Label(right, text="● Disconnected",
                                       bg="#252525", fg="#e06c75",
                                       font=("Courier", 9))
        self._robot_status.pack(anchor="w", padx=12, pady=(0,4))

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # Motion settings
        tk.Label(right, text="Motion Settings", bg="#252525", fg="#e5c07b",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(4,2))

        row2 = tk.Frame(right, bg="#252525"); row2.pack(fill="x", padx=12, pady=2)
        tk.Label(row2, text="Speed %:", bg="#252525", fg="#ccc",
                 font=("Helvetica", 9), width=9, anchor="w").pack(side="left")
        self._speed_var = tk.StringVar(value="15")
        tk.Entry(row2, textvariable=self._speed_var, width=5, bg="#3a3a3a",
                 fg="white", insertbackground="white", relief="flat",
                 font=("Helvetica", 9)).pack(side="left")

        row3 = tk.Frame(right, bg="#252525"); row3.pack(fill="x", padx=12, pady=2)
        tk.Label(row3, text="Approach mm:", bg="#252525", fg="#ccc",
                 font=("Helvetica", 9), width=12, anchor="w").pack(side="left")
        self._approach_var = tk.StringVar(value=str(APPROACH_OFFSET))
        tk.Entry(row3, textvariable=self._approach_var, width=5, bg="#3a3a3a",
                 fg="white", insertbackground="white", relief="flat",
                 font=("Helvetica", 9)).pack(side="left")

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # Grasp target display
        tk.Label(right, text="Best Grasp  (Robot Frame)", bg="#252525", fg="#e5c07b",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(4,2))
        self._grasp_display = tk.Label(right, text="—  run GraspGen first",
                                        bg="#252525", fg="#666",
                                        font=("Courier", 8), wraplength=260, justify="left")
        self._grasp_display.pack(anchor="w", padx=12, pady=(0,6))

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # Execute
        tk.Label(right, text="Execute", bg="#252525", fg="#e5c07b",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(4,2))

        self._execute_btn = tk.Button(
            right, text="🤖  Execute Best Grasp",
            bg="#c678dd", fg="white", activebackground="#a85dc0",
            relief="flat", cursor="hand2", bd=0,
            font=("Helvetica", 11, "bold"), state="disabled",
            command=self._on_execute)
        self._execute_btn.pack(padx=12, pady=2, fill="x", ipady=10)

        self._home_btn = tk.Button(
            right, text="🏠  Move to Home",
            bg="#3a3a3a", fg="#ccc", activebackground="#444",
            relief="flat", cursor="hand2", bd=0,
            font=("Helvetica", 9), state="disabled",
            command=self._on_home)
        self._home_btn.pack(padx=12, pady=2, fill="x", ipady=5)

        self._save_home_btn = tk.Button(
            right, text="📌  Save Current Pos as Home",
            bg="#3a3a3a", fg="#e5c07b", activebackground="#444",
            relief="flat", cursor="hand2", bd=0,
            font=("Helvetica", 9), state="disabled",
            command=self._on_save_home)
        self._save_home_btn.pack(padx=12, pady=(0,2), fill="x", ipady=5)

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # Vacuum
        tk.Label(right, text="Vacuum Tool", bg="#252525", fg="#e5c07b",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(4,2))
        vac_row = tk.Frame(right, bg="#252525"); vac_row.pack(fill="x", padx=12, pady=4)
        self._vac_on_btn = tk.Button(vac_row, text="💨  ON",
                                      bg="#98c379", fg="#1e1e1e", activebackground="#7aad5b",
                                      relief="flat", cursor="hand2", bd=0,
                                      font=("Helvetica", 10, "bold"), state="disabled",
                                      command=self._on_vacuum_on)
        self._vac_on_btn.pack(side="left", expand=True, fill="x", ipady=6, padx=(0,4))
        self._vac_off_btn = tk.Button(vac_row, text="✕  OFF",
                                       bg="#e06c75", fg="white", activebackground="#c0505a",
                                       relief="flat", cursor="hand2", bd=0,
                                       font=("Helvetica", 10, "bold"), state="disabled",
                                       command=self._on_vacuum_off)
        self._vac_off_btn.pack(side="left", expand=True, fill="x", ipady=6)

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.columnconfigure(2, weight=0)
        self.root.rowconfigure(0, weight=1)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def _schedule_preview(self):
        self._update_preview()
        self.root.after(80, self._schedule_preview)

    def _update_preview(self):
        if not _PIL_OK:
            return
        rgb, _, _ = self.camera.get_latest()
        if rgb is None:
            return
        self._canvas.itemconfig(self._no_cam_txt, state="hidden")
        if self._status_var.get().startswith("Waiting"):
            self._status_var.set("Camera ready.")

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
        scale = min(_PREVIEW_W/w, _PREVIEW_H/h)
        nw, nh = int(w*scale), int(h*scale)
        resized = cv2.resize(display, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((_PREVIEW_H, _PREVIEW_W, 3), dtype=np.uint8)
        yo, xo = (_PREVIEW_H-nh)//2, (_PREVIEW_W-nw)//2
        padded[yo:yo+nh, xo:xo+nw] = resized

        pil = PILImage.fromarray(padded)
        tk_img = ImageTk.PhotoImage(pil)
        self._canvas.itemconfig(self._canvas_img_id, image=tk_img)
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
            self._cb_queue.put(lambda p=path: self._config_hint.configure(text=p))
            self._log(f"[Config] Ready: {cfg.data.gripper_name}")
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
            self._run_btn.configure(state="normal", text="▶  Capture & Run GraspGen")

    def _on_config_change(self, _=None):
        name = self._config_combo.get()
        if not name or name == self._loaded_config_name:
            return
        if self._config_loading or self._inference_running:
            self._config_combo.set(self._loaded_config_name or "")
            return
        self._start_config_load(name)

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
        rgb, depth_m, intrinsics = self.camera.get_latest()
        if rgb is None or depth_m is None:
            self._set_status("No camera frame yet.")
            return
        self._inference_running = True
        self._best_grasp_cam = None
        self._cb_queue.put(lambda: self._execute_btn.configure(state="disabled"))
        self._run_btn.configure(state="disabled", text="Running…")
        self._set_status("Running pipeline…")
        self._last_mask = None
        threading.Thread(target=self._pipeline_worker,
                          args=(rgb.copy(), depth_m.copy(), intrinsics, prompt,
                                self._sampler, self._grasp_cfg),
                          daemon=True).start()

    def _pipeline_worker(self, rgb, depth_m, intrinsics, prompt, sampler, grasp_cfg):
        args = self.args
        try:
            self._log(f"[SAM3] prompt='{prompt}'")
            t0 = time.time()
            mask = segment_with_sam3(rgb, prompt, args.sam3_socket)
            self._log(f"[SAM3] {time.time()-t0:.2f}s — {int(mask.sum())} px")

            if mask.shape != depth_m.shape:
                mask = cv2.resize(mask.astype(np.float32),
                                   (depth_m.shape[1], depth_m.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(np.uint8)

            self._last_mask = mask
            if mask.sum() < 50:
                raise RuntimeError(f"Only {mask.sum()} px — try a different prompt")

            fx, fy, cx, cy = intrinsics
            scene_pc, object_pc, scene_colors, object_colors = \
                depth_and_segmentation_to_point_clouds(
                    depth_image=depth_m, segmentation_mask=mask,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    rgb_image=rgb, target_object_id=args.target_object_id,
                    remove_object_from_scene=True)

            if len(object_pc) > args.max_object_points:
                idx = np.random.choice(len(object_pc), args.max_object_points, replace=False)
                object_pc = object_pc[idx]

            pc_torch = torch.from_numpy(object_pc)
            pc_filtered, _ = point_cloud_outlier_removal(pc_torch)
            pc_filtered = pc_filtered.numpy()
            if len(pc_filtered) == 0:
                raise RuntimeError("Point cloud empty after outlier removal")

            t_center = tra.translation_matrix(-pc_filtered.mean(axis=0))
            pc_centered = tra.transform_points(pc_filtered, t_center)
            scene_centered = tra.transform_points(scene_pc, t_center)

            if self._vis:
                self._vis.delete()
                make_frame(self._vis, "world", h=0.12, radius=0.004)
                _sc = scene_colors if scene_colors is not None else \
                    np.tile([[120,120,120]], (len(scene_pc),1)).astype(np.uint8)
                visualize_pointcloud(self._vis, "pc_scene", scene_centered, _sc,
                                     size=args.scene_point_size)
                visualize_pointcloud(self._vis, "pc_obj", pc_centered,
                                     np.tile([[255,255,255]],(len(pc_centered),1)).astype(np.uint8),
                                     size=args.object_point_size)

            self._log("[GraspGen] Running inference…")
            t1 = time.time()
            grasps, grasp_conf = GraspGenSampler.run_inference(
                pc_filtered, sampler,
                grasp_threshold=args.grasp_threshold,
                num_grasps=args.num_grasps,
                topk_num_grasps=args.topk_num_grasps)
            self._log(f"[GraspGen] {time.time()-t1:.2f}s — {len(grasps)} grasps")

            if len(grasps) == 0:
                raise RuntimeError("No grasps found")

            grasps_np = grasps.cpu().numpy()
            grasp_conf_np = grasp_conf.cpu().numpy()

            pc_centered, grasps_centered, scores, _ = _process_point_cloud(
                pc_filtered, grasps_np, grasp_conf_np)

            if self._vis:
                gripper_name = grasp_cfg.data.gripper_name
                for j, g in enumerate(grasps_centered):
                    visualize_grasp(self._vis, f"grasps/{j:03d}/grasp", g,
                                    color=scores[j], gripper_name=gripper_name, linewidth=1.2)

            best_info = _save_best_grasp(grasps_np, grasp_conf_np)
            best_idx = int(np.argmax(grasp_conf_np.ravel()))
            self._best_grasp_cam = grasps_np[best_idx].copy()   # camera frame, meters

            conf_min, conf_max = float(grasp_conf_np.min()), float(grasp_conf_np.max())
            best_conf = best_info.get("confidence", 0.0)
            summary = (f"{len(grasps_np)} grasps [{conf_min:.3f}–{conf_max:.3f}]\n"
                       f"Best conf: {best_conf:.4f}")
            self._log("─" * 36)
            self._log(summary)
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

    def _update_grasp_display(self):
        """Compute robot-frame pose and update the display label."""
        if self._best_grasp_cam is None or self._T_cam2base is None:
            return
        try:
            x, y, z, rx, ry, rz = transform_grasp_to_robot(
                self._best_grasp_cam, self._T_cam2base)
            txt = (f"X  = {x:+8.1f} mm\n"
                   f"Y  = {y:+8.1f} mm\n"
                   f"Z  = {z:+8.1f} mm\n"
                   f"Rx = {rx:+7.2f} °\n"
                   f"Ry = {ry:+7.2f} °\n"
                   f"Rz = {rz:+7.2f} °")
            self._log(f"[Transform] Robot frame: X={x:.1f} Y={y:.1f} Z={z:.1f} mm  "
                      f"Rx={rx:.1f} Ry={ry:.1f} Rz={rz:.1f} deg")
            self._cb_queue.put(lambda t=txt: (
                self._grasp_display.config(text=t, fg="#98c379"),
                self._execute_btn.configure(
                    state="normal" if self._robot_connected else "disabled")))
        except Exception as e:
            self._log(f"[Transform] ERROR: {e}")

    # ------------------------------------------------------------------
    # Robot connection
    # ------------------------------------------------------------------
    def _on_connect(self):
        if self._robot_connected:
            # Disconnect
            if self._robot:
                self._robot.close(); self._robot = None
            self._robot_connected = False
            self._robot_status.config(text="● Disconnected", fg="#e06c75")
            self._connect_btn.config(text="⚡  Connect Robot")
            for btn in (self._execute_btn, self._home_btn, self._save_home_btn,
                        self._vac_on_btn, self._vac_off_btn):
                btn.config(state="disabled")
            return
        ip = self._ip_var.get().strip()
        self._connect_btn.config(state="disabled", text="Connecting…")
        threading.Thread(target=self._connect_worker, args=(ip,), daemon=True).start()

    def _connect_worker(self, ip):
        try:
            robot = DobotDashboard(ip)
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
            if mode == DobotDashboard.MODE_ERROR:
                self._log("[Robot] Error state — clearing and re-enabling…")
                robot.clear_error(); time.sleep(1.0)
                robot.enable();      time.sleep(2.0)
            # Set default speed (low for safety)
            robot.set_speed(15)
            self._robot = robot
            self._robot_connected = True
            self._log(f"[Robot] Ready  mode={robot.get_mode()}")
            self._cb_queue.put(self._on_robot_connected_ui)
        except Exception as e:
            self._log(f"[Robot] Connection failed: {e}")
            self._cb_queue.put(lambda err=e: (
                self._robot_status.config(text=f"● Error: {err}", fg="#e06c75"),
                self._connect_btn.config(state="normal", text="⚡  Connect Robot")))

    def _on_robot_connected_ui(self):
        self._robot_status.config(text=f"● Connected  {self._ip_var.get()}", fg="#98c379")
        self._connect_btn.config(state="normal", text="⏏  Disconnect")
        self._home_btn.config(state="normal")
        self._save_home_btn.config(state="normal")
        self._vac_on_btn.config(state="normal")
        self._vac_off_btn.config(state="normal")
        # Enable execute only if we already have a grasp
        if self._best_grasp_cam is not None and self._T_cam2base is not None:
            self._execute_btn.config(state="normal")

    # ------------------------------------------------------------------
    # Execute grasp
    # ------------------------------------------------------------------
    def _on_execute(self):
        if not self._robot_connected or self._executing:
            return
        if self._best_grasp_cam is None:
            messagebox.showwarning("No grasp", "Run GraspGen first.")
            return
        if self._T_cam2base is None:
            messagebox.showerror("No calibration",
                                  "Hand-eye calibration not loaded.\n"
                                  "Run hand_eye_calibration.py first.")
            return

        try:
            x, y, z, rx, ry, rz = transform_grasp_to_robot(
                self._best_grasp_cam, self._T_cam2base)
        except Exception as e:
            messagebox.showerror("Transform error", str(e)); return

        approach_mm = float(self._approach_var.get())
        speed = int(self._speed_var.get())

        msg = (f"Execute grasp at:\n"
               f"  X={x:.1f}  Y={y:.1f}  Z={z:.1f} mm\n"
               f"  Rx={rx:.1f}  Ry={ry:.1f}  Rz={rz:.1f} °\n\n"
               f"Pre-grasp approach: {approach_mm:.0f} mm above\n"
               f"Speed: {speed}%\n\n"
               f"Make sure the workspace is clear before continuing!")
        if not messagebox.askyesno("Confirm Execute", msg):
            return

        self._executing = True
        self._execute_btn.config(state="disabled", text="Executing…")
        self._set_status("Executing grasp…")
        threading.Thread(target=self._execute_worker,
                          args=(x, y, z, rx, ry, rz, approach_mm, speed),
                          daemon=True).start()

    def _execute_worker(self, x, y, z, rx, ry, rz, approach_mm, speed):
        robot = self._robot
        try:
            robot.set_speed(speed)

            z_pre = z + approach_mm

            # 1. Pre-grasp approach (above target)
            self._log(f"[Execute] Pre-grasp  X={x:.1f} Y={y:.1f} Z={z_pre:.1f} mm")
            self._set_status(f"MovL → pre-grasp Z={z_pre:.0f}mm…")
            cmd_id = robot.move_linear(x, y, z_pre, rx, ry, rz)
            robot.wait_motion(cmd_id)

            # 2. Descend to grasp
            self._log(f"[Execute] Grasp      X={x:.1f} Y={y:.1f} Z={z:.1f} mm")
            self._set_status(f"MovL → grasp Z={z:.0f}mm…")
            cmd_id = robot.move_linear(x, y, z, rx, ry, rz)
            robot.wait_motion(cmd_id)

            # 3. Vacuum ON
            self._log("[Execute] Vacuum ON")
            robot.vacuum_on()
            time.sleep(0.8)

            # 4. Retreat
            self._log(f"[Execute] Retreat    Z={z_pre:.1f} mm")
            self._set_status("MovL → retreat…")
            cmd_id = robot.move_linear(x, y, z_pre, rx, ry, rz)
            robot.wait_motion(cmd_id)

            self._log("[Execute] Done — grasp executed successfully")
            self._set_status("Done!  Press Vacuum OFF to release.")

        except Exception as e:
            self._log(f"[Execute] ERROR: {e}")
            self._set_status(f"Execute error: {e}")
        finally:
            self._executing = False
            self._cb_queue.put(lambda: self._execute_btn.configure(
                state="normal", text="🤖  Execute Best Grasp"))

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
            if mode == DobotDashboard.MODE_ERROR:
                self._log("[Home] Robot in error — clearing before save…")
                self._robot.clear_error(); time.sleep(1.0)
                self._robot.enable();      time.sleep(1.5)
            joints = self._robot.get_angle()
            self._home_joints = list(joints)
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

    def _ensure_enabled(self):
        """Re-enable the robot if it slipped into error/disabled state."""
        mode = self._robot.get_mode()
        if mode == DobotDashboard.MODE_ERROR:
            self._log("[Robot] Error state detected — clearing and re-enabling…")
            self._robot.clear_error(); time.sleep(0.5)
            self._robot.enable();      time.sleep(2.0)
        elif mode not in (DobotDashboard.MODE_ENABLED, DobotDashboard.MODE_RUNNING):
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

    def _on_vacuum_on(self):
        if self._robot:
            self._robot.vacuum_on()
            self._log("[Vacuum] ON")

    def _on_vacuum_off(self):
        if self._robot:
            self._robot.vacuum_off()
            self._log("[Vacuum] OFF")


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
    if config_map:
        print(f"[Config] Found {len(config_map)} config(s):")
        for n in config_map:
            print(f"  {n}")
    else:
        print(f"[Config] WARNING: No .yml in '{CHECKPOINTS_DIR}'")

    # SAM3 server
    sam3_proc = None
    print(f"[SAM3] Checking {args.sam3_socket}…")
    try:
        import socket as _s
        s = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM)
        s.settimeout(2.0); s.connect(args.sam3_socket); s.close()
        print("[SAM3] Using existing server.")
    except OSError:
        print("[SAM3] Starting server…")
        sam3_proc = start_sam3_server(args.sam3_socket, args.sam3_device,
                                       not args.sam3_no_fp16)

    # Meshcat
    vis = None
    try:
        vis = create_visualizer()
        print(f"[Meshcat] {getattr(vis,'viewer_url','http://127.0.0.1:7000/static/')}")
    except Exception as e:
        print(f"[Meshcat] Not available: {e}")

    # Camera
    print("[Camera] Starting Orbbec pipeline…")
    camera = OrbbecCamera()
    camera.start()
    print("[Camera] Live stream running.")

    root = tk.Tk()
    GraspExecuteApp(root, args, camera, config_map, sam3_proc=sam3_proc, vis=vis)

    def _on_close():
        print("Shutting down…")
        camera.stop()
        if sam3_proc:
            try:
                sam3_proc.terminate(); sam3_proc.wait(timeout=5)
            except Exception:
                try:
                    sam3_proc.kill()
                except Exception:
                    pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    print("UI ready.  Select gripper, enter prompt, click Capture & Run.")
    root.mainloop()


if __name__ == "__main__":
    main()
