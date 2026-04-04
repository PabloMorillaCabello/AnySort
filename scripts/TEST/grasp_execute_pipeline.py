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
import sys
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

# Redirect OrbbecSDK C-level stderr to file — prevents timestamp-anomaly spam
# from flooding the terminal. Full log available at /tmp/orbbec_sdk.log
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orbbec_quiet  # noqa: E402

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
        self._lock = threading.Lock()
        self._latest_rgb = None
        self._latest_depth_m = None
        self._latest_intrinsics = None
        self._running = False
        self._started = False
        self._color_profile = None    # intrinsics come from colour (D2C target)
        self._depth_filters = []      # SDK post-processing filters

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
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        hw_d2c_ok = False
        for i in range(len(color_profiles)):
            cp = color_profiles[i]
            if cp.get_format() != OBFormat.RGB:
                continue
            try:
                d2c_list = pipeline.get_d2c_depth_profile_list(cp, OBAlignMode.HW_MODE)
            except Exception:
                continue
            if len(d2c_list) == 0:
                continue
            # Prefer Y16 — RLE depth is not supported by ThresholdFilter/pixel-size API
            dp = None
            for j in range(len(d2c_list)):
                if d2c_list[j].get_format() == OBFormat.Y16:
                    dp = d2c_list[j]
                    break
            if dp is None:
                dp = d2c_list[0]
            config.enable_stream(dp)
            config.enable_stream(cp)
            config.set_align_mode(OBAlignMode.HW_MODE)
            self._color_profile = cp
            hw_d2c_ok = True
            print(f"[Camera] HW D2C: color {cp.get_width()}x{cp.get_height()} "
                  f"depth {dp.get_width()}x{dp.get_height()}")
            break

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

        pipeline.start(config)

        # Re-apply the fd 2 redirect: SDK extensions loaded during start()
        # can reset C-level stderr back to the terminal.
        try:
            import orbbec_quiet
            orbbec_quiet.reapply()
        except Exception:
            pass

        # ── Depth post-processing: enable device-recommended filters ──
        try:
            device = pipeline.get_device()
            sensor = device.get_sensor(OBSensorType.DEPTH_SENSOR)
            filters = sensor.get_recommended_filters()
            for f in filters:
                f.enable(True)
                print(f"[Camera] Depth filter: {f.get_name()} (enabled)")
            self._depth_filters = filters
        except Exception as e:
            print(f"[Camera] Depth filters unavailable: {e}")
            self._depth_filters = []

        # Let the sensor stabilise (avoid early timestamp anomalies)
        print("[Camera] Stabilising sensor (15 frames)…")
        for _ in range(15):
            try:
                pipeline.wait_for_frames(200)
            except Exception:
                pass

        self._pipeline = pipeline
        self._running = True
        self._started = True
        threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------------------------------------------ #
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
                # Apply depth post-processing filters (temporal, spatial, etc.)
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
                # Auto-detect mm vs m: if median valid depth > 20, scale gave mm
                valid = depth_m[(depth_m > 0.05) & np.isfinite(depth_m)]
                if valid.size > 0 and float(np.median(valid)) > 20.0:
                    depth_m /= 1000.0

                # With HW D2C, depth is already aligned to colour resolution.
                # Resize colour only as safety fallback.
                if rgb.shape[0] != dh or rgb.shape[1] != dw:
                    rgb = np.array(PILImage.fromarray(rgb).resize(
                        (dw, dh), PILImage.BILINEAR))

                # Use colour intrinsics (the D2C alignment target)
                intr = _extract_intrinsics(self._color_profile)
            except Exception:
                continue
            with self._lock:
                self._latest_rgb = rgb
                self._latest_depth_m = depth_m
                self._latest_intrinsics = intr

    # ------------------------------------------------------------------ #
    def get_latest(self):
        with self._lock:
            if self._latest_rgb is None:
                return None, None, None
            return (self._latest_rgb.copy(),
                    self._latest_depth_m.copy(),
                    self._latest_intrinsics)

    def stop(self):
        self._running = False
        self._started = False
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
    # Reachability check (InverseKin)
    # ------------------------------------------------------------------
    def check_reachability(self, x, y, z, rx, ry, rz):
        """Query InverseKin to verify (x,y,z mm, rx,ry,rz deg) is reachable.

        Returns:
            reachable (bool)
            joints    (tuple of 6 floats in degrees, or None)
            message   (str)
        """
        try:
            resp = self._dashboard.InverseKin(x, y, z, rx, ry, rz)
            nums = self._nums(resp)
            if not nums:
                return False, None, f"No response: {resp!r}"
            err_id = int(nums[0])
            if err_id != 0:
                return False, None, f"Out of workspace (ErrorID={err_id})"
            joints = tuple(nums[1:7]) if len(nums) >= 7 else None
            return True, joints, "OK"
        except Exception as e:
            return False, None, f"InverseKin error: {e}"

    # ------------------------------------------------------------------
    # Motion commands (return CommandID for wait_motion)
    # ------------------------------------------------------------------
    def _send_motion(self, resp):
        """Parse response, return CommandID (or -1 on error)."""
        if resp is None or resp == b'' or resp == '':
            try:
                resp = self._dashboard.wait_reply()
            except Exception:
                pass
        parsed = _parse_result_id(resp)
        if len(parsed) < 2 or parsed[0] != 0:
            raise RuntimeError(f"Move rejected (ErrorID={parsed[0] if parsed else '?'}): {resp!r}")
        return parsed[1]

    def move_linear(self, x, y, z, rx, ry, rz):
        resp = self._dashboard.MovL(
            x, y, z, rx, ry, rz,
            0,                  # coordinateMode=0 → Cartesian
            a=self._speed, v=self._speed
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
            1,                  # coordinateMode=1 → joint angles
            a=self._speed, v=self._speed
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
                 config_map: dict, sam3_proc=None, vis=None):
        self.root       = root
        self.args       = args
        self.camera     = camera
        self.config_map = config_map
        self.sam3_proc  = sam3_proc
        self._vis       = vis

        # Replay mode — (rgb uint8, depth_m float32, intrinsics tuple) or None for live
        self._replay_frame = None
        self._replay_label_var = tk.StringVar(value="")

        # GraspGen state
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
        self._robot: DobotDashboard = None
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
        self._live_preview      = tk.BooleanVar(value=True)

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
        tk.Checkbutton(left, text="Live camera preview", variable=self._live_preview,
                       bg="#1e1e1e", fg="#aaa", activebackground="#1e1e1e",
                       selectcolor="#2d2d2d").pack(anchor="w", padx=4, pady=(2,0))

        # ── Camera connect / disconnect ──
        cam_row = tk.Frame(left, bg="#1e1e1e")
        cam_row.pack(fill="x", padx=4, pady=(8, 0))
        self._cam_connect_btn = tk.Button(
            cam_row, text="📷  Connect Camera",
            bg="#3a3a3a", fg="#61afef", activebackground="#4a4a4a",
            relief="flat", cursor="hand2", bd=0, font=("Helvetica", 9),
            command=self._on_camera_connect)
        self._cam_connect_btn.pack(side="left", ipady=4)
        self._cam_status_var = tk.StringVar(
            value="● Connected" if self.camera._started else "● No camera")
        self._cam_status_lbl = tk.Label(
            cam_row, textvariable=self._cam_status_var, bg="#1e1e1e",
            fg="#98c379" if self.camera._started else "#e06c75",
            font=("Helvetica", 8))
        self._cam_status_lbl.pack(side="left", padx=(8, 0))

        # ── Replay mode controls ──
        replay_row = tk.Frame(left, bg="#1e1e1e")
        replay_row.pack(fill="x", padx=4, pady=(6, 0))
        tk.Button(replay_row, text="📂  Load Saved Frame",
                  bg="#3a3a3a", fg="#e5c07b", activebackground="#4a4a4a",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 9),
                  command=self._on_load_replay).pack(side="left", ipady=4, padx=(0, 6))
        tk.Button(replay_row, text="✕ Live",
                  bg="#3a3a3a", fg="#aaa", activebackground="#4a4a4a",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 9),
                  command=self._on_clear_replay).pack(side="left", ipady=4)
        self._replay_info = tk.Label(left, textvariable=self._replay_label_var,
                                     bg="#1e1e1e", fg="#c678dd",
                                     font=("Courier", 7), wraplength=_PREVIEW_W - 8,
                                     justify="left", anchor="w")
        self._replay_info.pack(fill="x", padx=4, pady=(2, 0))

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

        # ── Pipeline options ──
        tk.Checkbutton(mid, text="Check collisions (filter colliding grasps)",
                       variable=self._collision_var,
                       bg="#2d2d2d", fg="#aaa", activebackground="#2d2d2d",
                       selectcolor="#3a3a3a", font=("Helvetica", 8)
                       ).pack(anchor="w", padx=12, pady=(4, 2))
        tk.Checkbutton(mid, text="Filter unreachable grasps (needs robot)",
                       variable=self._reach_var,
                       bg="#2d2d2d", fg="#aaa", activebackground="#2d2d2d",
                       selectcolor="#3a3a3a", font=("Helvetica", 8)
                       ).pack(anchor="w", padx=12, pady=(0, 2))
        tk.Checkbutton(mid, text="Process debug (step-by-step)",
                       variable=self._debug_var,
                       bg="#2d2d2d", fg="#aaa", activebackground="#2d2d2d",
                       selectcolor="#3a3a3a", font=("Helvetica", 8)
                       ).pack(anchor="w", padx=12, pady=(0, 4))

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

        self._continue_btn = tk.Button(
            mid, text="▶  Continue to next step",
            bg="#e5c07b", fg="#1e1e1e", activebackground="#c9a44e",
            font=("Helvetica", 10, "bold"), relief="flat", cursor="hand2", bd=0,
            state="disabled", command=self._on_debug_continue)
        self._continue_btn.pack(padx=12, pady=(2, 2), fill="x", ipady=8)

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

        row4 = tk.Frame(right, bg="#252525"); row4.pack(fill="x", padx=12, pady=2)
        tk.Label(row4, text="Num grasps:", bg="#252525", fg="#ccc",
                 font=("Helvetica", 9), width=12, anchor="w").pack(side="left")
        self._num_grasps_var = tk.StringVar(value=str(self.args.num_grasps))
        tk.Entry(row4, textvariable=self._num_grasps_var, width=5, bg="#3a3a3a",
                 fg="white", insertbackground="white", relief="flat",
                 font=("Helvetica", 9)).pack(side="left")

        row5 = tk.Frame(right, bg="#252525"); row5.pack(fill="x", padx=12, pady=2)
        tk.Label(row5, text="Top-K grasps:", bg="#252525", fg="#ccc",
                 font=("Helvetica", 9), width=12, anchor="w").pack(side="left")
        self._topk_grasps_var = tk.StringVar(value=str(self.args.topk_num_grasps))
        tk.Entry(row5, textvariable=self._topk_grasps_var, width=5, bg="#3a3a3a",
                 fg="white", insertbackground="white", relief="flat",
                 font=("Helvetica", 9)).pack(side="left")

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # Grasp target display
        tk.Label(right, text="Selected Grasp  (Robot Frame)", bg="#252525", fg="#e5c07b",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(4,2))
        self._grasp_display = tk.Label(right, text="—  run GraspGen first",
                                        bg="#252525", fg="#666",
                                        font=("Courier", 8), wraplength=260, justify="left")
        self._grasp_display.pack(anchor="w", padx=12, pady=(0,4))

        # Grasp navigation (prev / index label / next)
        nav_row = tk.Frame(right, bg="#252525")
        nav_row.pack(fill="x", padx=12, pady=(0, 6))
        self._prev_grasp_btn = tk.Button(
            nav_row, text="◀  Prev", width=8,
            bg="#3a3a3a", fg="#ccc", activebackground="#444",
            relief="flat", cursor="hand2", bd=0,
            font=("Helvetica", 9), state="disabled",
            command=self._on_prev_grasp)
        self._prev_grasp_btn.pack(side="left")
        self._grasp_idx_label = tk.Label(
            nav_row, text="— / —", bg="#252525", fg="#abb2bf",
            font=("Courier", 9, "bold"))
        self._grasp_idx_label.pack(side="left", expand=True)
        self._next_grasp_btn = tk.Button(
            nav_row, text="Next  ▶", width=8,
            bg="#3a3a3a", fg="#ccc", activebackground="#444",
            relief="flat", cursor="hand2", bd=0,
            font=("Helvetica", 9), state="disabled",
            command=self._on_next_grasp)
        self._next_grasp_btn.pack(side="right")

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # Execute
        tk.Label(right, text="Execute", bg="#252525", fg="#e5c07b",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(4,2))

        self._execute_btn = tk.Button(
            right, text="🤖  Execute Selected Grasp",
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
    # Camera connect / disconnect
    # ------------------------------------------------------------------
    def _on_camera_connect(self):
        if self.camera._started:
            # Disconnect
            self.camera.stop()
            self._cam_status_var.set("● No camera")
            self._cam_status_lbl.config(fg="#e06c75")
            self._cam_connect_btn.config(text="📷  Connect Camera")
            self._log("[Camera] Disconnected.")
        else:
            # Connect in background (Pipeline() can take a moment)
            self._cam_connect_btn.config(state="disabled", text="Connecting…")
            self._cam_status_var.set("● Connecting…")
            self._cam_status_lbl.config(fg="#e5c07b")
            threading.Thread(target=self._camera_connect_worker, daemon=True).start()

    def _camera_connect_worker(self):
        try:
            self.camera.start()
            self._log("[Camera] Connected — live stream running.")
            def _ok():
                self._cam_status_var.set("● Connected")
                self._cam_status_lbl.config(fg="#98c379")
                self._cam_connect_btn.config(state="normal", text="⏏  Disconnect Camera")
            self._cb_queue.put(_ok)
        except Exception as e:
            self._log(f"[Camera] Failed to connect: {e}")
            def _err():
                self._cam_status_var.set(f"● Error: {e}")
                self._cam_status_lbl.config(fg="#e06c75")
                self._cam_connect_btn.config(state="normal", text="📷  Connect Camera")
            self._cb_queue.put(_err)

    # ------------------------------------------------------------------
    # Replay mode
    # ------------------------------------------------------------------
    def _on_load_replay(self):
        from tkinter import filedialog
        rgb_path = filedialog.askopenfilename(
            title="Select RGB image",
            initialdir=str(Path(self.args.calib_file).parent.parent / "rgb"),
            filetypes=[("PNG images", "*.png"), ("All files", "*.*")])
        if not rgb_path:
            return
        try:
            rgb = np.array(__import__("PIL").Image.open(rgb_path).convert("RGB"))

            # Auto-detect matching depth_aligned .npy from timestamp in filename
            import re as _re
            ts_match = _re.search(r"(\d{8}_\d{6})", rgb_path)
            depth_m = None
            if ts_match:
                ts = ts_match.group(1)
                data_root = Path(rgb_path).parent.parent
                for candidate in [
                    data_root / "depth_aligned" / f"depth_aligned_{ts}.npy",
                    data_root / "depth"         / f"depth_{ts}.npy",
                ]:
                    if candidate.exists():
                        raw = np.load(str(candidate))          # uint16, mm
                        depth_m = raw.astype(np.float32) / 1000.0
                        self._log(f"[Replay] Depth: {candidate.name}")
                        break

            if depth_m is None:
                # Manual depth selection
                depth_path = filedialog.askopenfilename(
                    title="Select depth .npy (uint16 mm)",
                    initialdir=str(Path(rgb_path).parent.parent / "depth_aligned"),
                    filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")])
                if not depth_path:
                    return
                raw = np.load(depth_path)
                depth_m = raw.astype(np.float32) / 1000.0

            # Use calibrated intrinsics if available, else fall back to Gemini 2 defaults
            if self._calib_K is not None:
                K = self._calib_K
                intr = (float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2]))
            else:
                h, w = rgb.shape[:2]
                intr = (684.7, 685.9, w / 2.0, h / 2.0)

            self._replay_frame = (rgb, depth_m, intr)
            fname = Path(rgb_path).name
            self._replay_label_var.set(f"REPLAY: {fname}")
            self._log(f"[Replay] Loaded {fname}  rgb={rgb.shape}  "
                      f"depth={depth_m.shape} ({depth_m[depth_m>0].mean()*1000:.0f}mm avg)")
            # Show the replay frame in the preview immediately
            self._show_replay_in_canvas(rgb)
        except Exception as e:
            self._log(f"[Replay] Load error: {e}")
            import traceback; traceback.print_exc()

    def _on_clear_replay(self):
        self._replay_frame = None
        self._replay_label_var.set("")
        self._last_mask = None
        if not self.camera._started:
            self._canvas.itemconfig(self._no_cam_txt, state="normal")
        self._log("[Replay] Cleared — back to live camera")

    def _show_replay_in_canvas(self, rgb):
        if not _PIL_OK:
            return
        from PIL import Image as _PILImg, ImageTk as _ITk
        h, w = rgb.shape[:2]
        scale = min(_PREVIEW_W / w, _PREVIEW_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((_PREVIEW_H, _PREVIEW_W, 3), dtype=np.uint8)
        yo, xo = (_PREVIEW_H - nh) // 2, (_PREVIEW_W - nw) // 2
        padded[yo:yo+nh, xo:xo+nw] = resized
        tk_img = _ITk.PhotoImage(_PILImg.fromarray(padded))
        self._canvas.itemconfig(self._canvas_img_id, image=tk_img)
        self._canvas.itemconfig(self._no_cam_txt, state="hidden")
        self._canvas._tk_ref = tk_img

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def _schedule_preview(self):
        self._update_preview()
        self.root.after(80, self._schedule_preview)

    def _update_preview(self):
        if not _PIL_OK:
            return
        if not self._live_preview.get():
            return
        rgb, _, _ = self.camera.get_latest()
        if rgb is None:
            # No live frame — fall back to replay image so mask overlay still works
            if self._replay_frame is not None:
                rgb = self._replay_frame[0]
            else:
                return
        self._canvas.itemconfig(self._no_cam_txt, state="hidden")
        if self._status_var.get().startswith("Waiting"):
            self._status_var.set("Replay mode." if self._replay_frame is not None
                                 else "Camera ready.")

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
    # Debug step control
    # ------------------------------------------------------------------
    def _debug_pause(self, step_label):
        """Block the pipeline worker at a debug checkpoint until the user continues."""
        if not self._debug_var.get():
            return
        self._debug_event.clear()
        self._set_status(f"Debug — {step_label}  ▶ click Continue")
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
            rgb, depth_m, intrinsics = self._replay_frame
            self._log(f"[Source] Using saved frame (replay mode)")
        else:
            rgb, depth_m, intrinsics = self.camera.get_latest()
        if rgb is None or depth_m is None:
            self._set_status("No camera frame yet.")
            return
        self._inference_running = True
        self._best_grasp_base = None
        self._preview_was_on = self._live_preview.get()
        if not self._debug_var.get():
            self._live_preview.set(False)  # suppress live refresh during normal run
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

            # ── Debug step 1: SAM3 segmentation ─────────────────────────────
            # Preview timer picks up _last_mask and shows green overlay on canvas
            self._debug_pause(f"Step 1/7 — SAM3  ({int(mask.sum())} px masked)")
            # ────────────────────────────────────────────────────────────────

            # Prefer calibrated intrinsics over SDK defaults
            if self._calib_K is not None:
                K = self._calib_K
                fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
                self._log("[PC] Using calibrated camera intrinsics")
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


            scene_pc, object_pc, scene_colors, object_colors = \
                depth_and_segmentation_to_point_clouds(
                    depth_image=depth_m, segmentation_mask=mask,
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

            self._log("[GraspGen] Running inference…")
            t1 = time.time()
            grasps, grasp_conf = GraspGenSampler.run_inference(
                pc_centered, sampler,
                grasp_threshold=args.grasp_threshold,
                num_grasps=int(self._num_grasps_var.get()),
                topk_num_grasps=int(self._topk_grasps_var.get()))
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

            # Filter: keep only grasps approaching from above (Z-axis points down)
            approach_z = grasps_obj_np[:, 2, 2]
            top_down = approach_z < 0  # approach Z-axis pointing downward = from above
            if top_down.sum() == 0:
                self._log("[GraspGen] WARNING: no top-down grasps — keeping all")
            else:
                n_before = len(grasps_obj_np)
                grasps_obj_np = grasps_obj_np[top_down]
                grasp_conf_np = grasp_conf_np[top_down]
                n_topdown_removed = n_before - len(grasps_obj_np)
                self._log(f"[GraspGen] Top-down filter: {n_before} → "
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
                            ok_g, _, _ = self._robot.check_reachability(
                                _x, _y, _z, _rx, _ry, _rz)
                            ok_a, _, _ = self._robot.check_reachability(
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
            if getattr(self, '_preview_was_on', True):
                self._cb_queue.put(lambda: self._live_preview.set(True))
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

        # ── Tool-Z approach: offset backwards along grasp Z-axis ─────────────
        approach_m = approach_mm / 1000.0
        tool_z = self._best_grasp_base[:3, 2]          # unit vector (approach dir)
        pre_pos_mm = (self._best_grasp_base[:3, 3] - approach_m * tool_z) * 1000.0
        x_pre, y_pre, z_pre = float(pre_pos_mm[0]), float(pre_pos_mm[1]), float(pre_pos_mm[2])
        # ─────────────────────────────────────────────────────────────────────

        # ── Reachability safety check ────────────────────────────────────────
        ok_g, joints, msg_g = self._robot.check_reachability(x, y, z, rx, ry, rz)
        ok_a, _,      msg_a = self._robot.check_reachability(x_pre, y_pre, z_pre, rx, ry, rz)
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

    def _execute_worker(self, x, y, z, x_pre, y_pre, z_pre, rx, ry, rz, speed):
        robot = self._robot
        try:
            robot.set_speed(speed)

            # 1. Pre-grasp approach (along tool Z-axis, away from object)
            self._log(f"[Execute] Pre-grasp  X={x_pre:.1f} Y={y_pre:.1f} Z={z_pre:.1f} mm")
            self._set_status(f"MovL → pre-grasp…")
            cmd_id = robot.move_linear(x_pre, y_pre, z_pre, rx, ry, rz)
            robot.wait_motion(cmd_id)

            # 2. Move to grasp
            self._log(f"[Execute] Grasp      X={x:.1f} Y={y:.1f} Z={z:.1f} mm")
            self._set_status(f"MovL → grasp…")
            cmd_id = robot.move_linear(x, y, z, rx, ry, rz)
            robot.wait_motion(cmd_id)

            # 3. Vacuum ON
            self._log("[Execute] Vacuum ON")
            robot.vacuum_on()
            time.sleep(0.8)

            # 4. Retreat (back along tool Z-axis)
            self._log(f"[Execute] Retreat    X={x_pre:.1f} Y={y_pre:.1f} Z={z_pre:.1f} mm")
            self._set_status("MovL → retreat…")
            cmd_id = robot.move_linear(x_pre, y_pre, z_pre, rx, ry, rz)
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

    # Camera — optional, can connect later via the UI button
    print("[Camera] Starting Orbbec pipeline…")
    camera = OrbbecCamera()
    try:
        camera.start()
        print("[Camera] Live stream running.")
    except Exception as e:
        print(f"[Camera] No device found ({e}) — starting without camera. "
              f"Use '📷 Connect Camera' in the UI once the device is attached.")

    root = tk.Tk()
    app_ref = [GraspExecuteApp(root, args, camera, config_map, sam3_proc=sam3_proc, vis=vis)]

    def _on_close():
        print("Shutting down…")
        app_ref[0]._debug_event.set()   # unblock any waiting debug pause
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
