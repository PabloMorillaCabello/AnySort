#!/usr/bin/env python3
"""
Hand-Eye Calibration UI  —  Eye-to-Hand
=========================================
Camera fixed.  ChArUco board mounted on robot gripper (vacuum tool).

Single Tkinter window:
  - Live camera preview with real-time ChArUco detection overlay
  - Board parameter inputs (square size, marker size) + Generate Board button
  - Auto mode  : robot moves through pre-programmed poses automatically
  - Manual mode: you move the robot, click "Capture Pose" at each position
  - Solve button: runs cv2.calibrateHandEye() and saves results

Usage:
  python3 /ros2_ws/scripts/TEST/hand_eye_calibration.py
  python3 /ros2_ws/scripts/TEST/hand_eye_calibration.py --robot-ip 192.168.5.1
"""

import argparse
import gc
import json
import os
import queue
import re
import socket
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

try:
    from PIL import Image as PILImage, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False
    print("[WARN] Pillow not found — install with: pip install Pillow")

try:
    from scipy.spatial.transform import Rotation
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    _ORBBEC_OK = True
except ImportError:
    _ORBBEC_OK = False

# ===========================================================================
# Defaults / constants
# ===========================================================================
ROBOT_IP_DEFAULT = "192.168.5.1"
ROBOT_PORT       = 29999
OUTPUT_DIR       = Path("/ros2_ws/data/calibration")
MOVE_SPEED       = 15
SETTLE_TIME      = 2.5
MIN_CORNERS      = 6

_PREVIEW_W = 640
_PREVIEW_H = 480
_DEFAULT_FX, _DEFAULT_FY = 905.0, 905.0
_DEFAULT_CX, _DEFAULT_CY = 640.0, 360.0

# Pre-programmed calibration poses  [X_mm, Y_mm, Z_mm, Rx_deg, Ry_deg, Rz_deg]
CALIB_POSES = [
    [300, -100, 400,   0,  30,   0],
    [300, -100, 400,  20,  30,   0],
    [300, -100, 400, -20,  30,   0],
    [300, -100, 400,   0,  15,   0],
    [300, -100, 400,   0,  45,   0],
    [300, -100, 400,   0,  30,  20],
    [300, -100, 400,   0,  30, -20],
    [250, -150, 380,  10,  30,  10],
    [350,  -50, 420, -10,  30, -10],
    [280, -100, 350,  15,  25,   5],
    [320, -100, 450, -15,  35,  -5],
    [300, -120, 400,   5,  30,  15],
    [300,  -80, 400,  -5,  30, -15],
    [270, -110, 395,  10,  20,   0],
    [330,  -90, 405, -10,  40,   0],
]

# ===========================================================================
# Camera (background grab loop, exposes latest RGB)
# ===========================================================================
def _to_rgb(color_frame) -> Optional[np.ndarray]:
    h, w = color_frame.get_height(), color_frame.get_width()
    raw = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    fmt = color_frame.get_format()
    fmt_s = str(fmt).upper()
    if "RGB" in fmt_s and raw.size == h * w * 3:
        return raw.reshape(h, w, 3).copy()
    if "BGR" in fmt_s and raw.size == h * w * 3:
        return raw.reshape(h, w, 3)[:, :, ::-1].copy()
    if "MJPG" in fmt_s or "JPEG" in fmt_s:
        try:
            return np.array(PILImage.open(BytesIO(raw.tobytes())).convert("RGB"))
        except Exception:
            decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            return decoded[:, :, ::-1] if decoded is not None else None
    if "YUYV" in fmt_s and raw.size == h * w * 2:
        return cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUY2)
    if raw.size == h * w * 3:
        return raw.reshape(h, w, 3).copy()
    return None


class OrbbecCamera:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_rgb = None
        self._K = None
        self._running = False

    def start(self):
        pipeline = Pipeline()
        cfg = Config()
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        cfg.enable_stream(color_profile)
        pipeline.start(cfg)
        self._pipeline = pipeline
        # Read intrinsics
        try:
            cp = pipeline.get_camera_param()
            for attr in ("rgb_intrinsic", "color_intrinsic"):
                ci = getattr(cp, attr, None)
                if ci:
                    fx, fy = float(ci.fx), float(ci.fy)
                    cx_, cy_ = float(ci.cx), float(ci.cy)
                    if all(v > 0 for v in [fx, fy, cx_, cy_]):
                        self._K = np.array([[fx, 0, cx_], [0, fy, cy_], [0, 0, 1]], np.float64)
                        break
        except Exception:
            pass
        if self._K is None:
            self._K = np.array([[_DEFAULT_FX, 0, _DEFAULT_CX],
                                 [0, _DEFAULT_FY, _DEFAULT_CY],
                                 [0, 0, 1]], np.float64)
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            try:
                fs = self._pipeline.wait_for_frames(100)
            except Exception:
                continue
            if not fs:
                continue
            cf = fs.get_color_frame()
            if cf is None:
                continue
            rgb = _to_rgb(cf)
            if rgb is None:
                continue
            with self._lock:
                self._latest_rgb = rgb

    def get_latest_rgb(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_rgb is None else self._latest_rgb.copy()

    def intrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._K, np.zeros((5,), np.float64)

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
    sys.path.insert(0, "/opt/Dobot_hv")
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
    """
    MODE_RUNNING = 7
    MODE_ERROR   = 9
    MODE_ENABLED = 5
    _FEEDBACK_PORT = 30004

    def __init__(self, ip: str, port: int = ROBOT_PORT):
        if not _DOBOT_API_OK:
            raise RuntimeError("dobot_api not available — cannot connect to robot")
        self._ip        = ip
        self._dashboard = _DobotApiDashboard(ip, port)
        self._feed      = _DobotApiFeedBack(ip, self._FEEDBACK_PORT)
        self._lock      = threading.Lock()
        self._mode      = -1
        self._cmd_id    = -1
        self._speed     = 20
        self._feed_running = True
        threading.Thread(target=self._feed_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # Feedback loop
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

    def get_mode(self) -> int:
        with self._lock:
            return self._mode

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

    # ------------------------------------------------------------------
    # Position getter
    # ------------------------------------------------------------------
    def _nums(self, resp) -> List[float]:
        return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(resp))]

    def get_pose(self) -> tuple:
        resp = self._dashboard.GetPose()
        nums = self._nums(resp)
        if len(nums) >= 7:
            return tuple(nums[1:7])
        if len(nums) >= 6:
            return tuple(nums[:6])
        raise ValueError(f"Cannot parse pose — raw response: {resp!r}")

    # ------------------------------------------------------------------
    # Motion  (returns CommandID)
    # ------------------------------------------------------------------
    def _send_motion(self, resp):
        parsed = _parse_result_id(resp)
        if len(parsed) < 2 or parsed[0] != 0:
            raise RuntimeError(f"Move rejected (ErrorID={parsed[0] if parsed else '?'}): {resp!r}")
        return parsed[1]

    def move_joint(self, x, y, z, rx, ry, rz):
        """Move to Cartesian pose using joint-space motion (MovJ, coordinateMode=0)."""
        resp = self._dashboard.MovJ(
            x, y, z, rx, ry, rz,
            0,               # coordinateMode=0 → Cartesian pose
            a=self._speed,
            v=self._speed
        )
        return self._send_motion(resp)

    # ------------------------------------------------------------------
    # Wait helpers
    # ------------------------------------------------------------------
    def wait_motion(self, cmd_id, timeout=90.0):
        """Block until the robot finishes cmd_id."""
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
        raise TimeoutError(f"Motion timeout after {timeout:.0f}s")

    def wait_idle(self, timeout=90.0):
        """Block until mode is not RUNNING."""
        time.sleep(0.4)
        t0 = time.time()
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
# ChArUco helpers
# ===========================================================================
def make_board(sq_x: int, sq_y: int, sq_len: float, mk_len: float,
               dict_type=cv2.aruco.DICT_6X6_250):
    d = cv2.aruco.getPredefinedDictionary(dict_type)
    try:
        b = cv2.aruco.CharucoBoard((sq_x, sq_y), sq_len, mk_len, d)
    except TypeError:
        b = cv2.aruco.CharucoBoard_create(sq_x, sq_y, sq_len, mk_len, d)
    return d, b


def make_charuco_detector(board):
    if hasattr(cv2.aruco, "CharucoDetector"):
        return cv2.aruco.CharucoDetector(board)
    return None


def detect_charuco(frame_rgb: np.ndarray, aruco_dict, board, detector,
                   K: np.ndarray, d: np.ndarray
                   ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, int]:
    """
    Returns (rvec, tvec, annotated_bgr, num_corners).
    rvec/tvec are None if detection failed.
    """
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    annotated = bgr.copy()

    if detector is not None:
        # OpenCV 4.7+ API
        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            detector.detectBoard(gray)
        num = len(charuco_ids) if charuco_ids is not None else 0

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)

        if num >= MIN_CORNERS:
            cv2.aruco.drawDetectedCornersCharuco(annotated, charuco_corners, charuco_ids)
            obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
            if obj_pts is not None and len(obj_pts) >= 4:
                ret, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, d,
                                                flags=cv2.SOLVEPNP_ITERATIVE)
                if ret:
                    cv2.drawFrameAxes(annotated, K, d, rvec, tvec, 0.05)
                    return rvec, tvec, annotated, num
        return None, None, annotated, num

    else:
        # Legacy API
        params = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        if ids is None:
            return None, None, annotated, 0
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        n, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board,
                                                         cameraMatrix=K, distCoeffs=d)
        if n < MIN_CORNERS:
            return None, None, annotated, n
        cv2.aruco.drawDetectedCornersCharuco(annotated, cc, ci)
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(cc, ci, board, K, d, None, None)
        if ok:
            cv2.drawFrameAxes(annotated, K, d, rvec, tvec, 0.05)
            return rvec, tvec, annotated, n
        return None, None, annotated, n


def generate_board_image(sq_x, sq_y, sq_len, mk_len, output_path: Path,
                         ppm: int = 4000):
    _, board = make_board(sq_x, sq_y, sq_len, mk_len)
    pw = int(sq_x * sq_len * ppm)
    ph = int(sq_y * sq_len * ppm)
    try:
        img = board.generateImage((pw, ph), marginSize=20, borderBits=1)
    except AttributeError:
        img = board.draw((pw, ph), marginSize=20, borderBits=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return img, pw, ph


# ===========================================================================
# Hand-eye solver
# ===========================================================================
def euler_zyx_to_rmat(rx, ry, rz):
    if _SCIPY_OK:
        return Rotation.from_euler("ZYX", [rz, ry, rx], degrees=True).as_matrix()
    rx_r, ry_r, rz_r = np.radians([rx, ry, rz])
    Rx = np.array([[1,0,0],[0,np.cos(rx_r),-np.sin(rx_r)],[0,np.sin(rx_r),np.cos(rx_r)]])
    Ry = np.array([[np.cos(ry_r),0,np.sin(ry_r)],[0,1,0],[-np.sin(ry_r),0,np.cos(ry_r)]])
    Rz = np.array([[np.cos(rz_r),-np.sin(rz_r),0],[np.sin(rz_r),np.cos(rz_r),0],[0,0,1]])
    return Rz @ Ry @ Rx


def solve_hand_eye(robot_poses: list, board_poses: list) -> dict:
    n = len(robot_poses)
    R_b2g, t_b2g, R_t2c, t_t2c = [], [], [], []
    for (x, y, z, rx, ry, rz) in robot_poses:
        R = euler_zyx_to_rmat(rx, ry, rz)
        t = np.array([[x],[y],[z]], np.float64)
        Ri, ti = R.T, -R.T @ t
        R_b2g.append(Ri); t_b2g.append(ti)
    for (rvec, tvec) in board_poses:
        Rv, _ = cv2.Rodrigues(rvec.reshape(3,1))
        R_t2c.append(Rv)
        t_t2c.append(tvec.reshape(3,1) * 1000.0)   # m → mm

    methods = {
        "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
        "PARK":       cv2.CALIB_HAND_EYE_PARK,
        "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    results = {}
    for name, flag in methods.items():
        try:
            R_c2b, t_c2b = cv2.calibrateHandEye(R_b2g, t_b2g, R_t2c, t_t2c, method=flag)
            results[name] = (R_c2b, t_c2b)
        except Exception as e:
            results[name] = None

    primary = next((k for k in ("TSAI","PARK","HORAUD","ANDREFF","DANIILIDIS")
                    if results.get(k) is not None), None)
    if primary is None:
        raise RuntimeError("All hand-eye methods failed.")
    R_c2b, t_c2b = results[primary]
    T = np.eye(4); T[:3,:3] = R_c2b; T[:3,3] = t_c2b.ravel()
    return {"T": T, "R": R_c2b, "t": t_c2b, "primary": primary, "all": results}


def save_calibration(result: dict, K: np.ndarray, d: np.ndarray,
                     robot_poses: list, board_poses: list, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    data = dict(T_cam2base=result["T"], R_cam2base=result["R"], t_cam2base=result["t"],
                camera_matrix=K, dist_coeffs=d, n_poses=len(robot_poses),
                robot_poses=np.array(robot_poses),
                board_rvecs=np.array([p[0].ravel() for p in board_poses]),
                board_tvecs=np.array([p[1].ravel() for p in board_poses]))
    np.savez(str(output / "hand_eye_calib.npz"), **data)
    np.savez(str(output / f"hand_eye_calib_{ts}.npz"), **data)
    json_data = {
        "timestamp": ts, "n_poses": len(robot_poses), "primary_method": result["primary"],
        "T_cam2base": result["T"].tolist(),
        "t_cam2base_mm": result["t"].ravel().tolist(),
        "camera_matrix": K.tolist(),
        "all_methods": {
            k: {"R": R.tolist(), "t_mm": t.ravel().tolist()}
            for k, (R, t) in result["all"].items() if result["all"][k] is not None
        },
    }
    with open(output / "hand_eye_calib.json", "w") as f:
        json.dump(json_data, f, indent=2)
    return ts


# ===========================================================================
# Tkinter Application
# ===========================================================================
class HandEyeCalibApp:
    """Hand-Eye Calibration UI."""

    # App states
    IDLE      = "idle"
    DETECTING = "detecting"   # live detection, no robot
    MANUAL    = "manual"      # waiting for user to click Capture
    AUTO      = "auto"        # robot moving automatically
    SOLVING   = "solving"

    def __init__(self, root: tk.Tk, robot_ip: str):
        self.root = root
        self.robot_ip = robot_ip

        # Data
        self._robot_poses: list = []
        self._board_poses: list = []
        self._state = self.IDLE
        self._board = None
        self._aruco_dict = None
        self._detector = None
        self._camera: Optional[OrbbecCamera] = None
        self._robot: Optional[DobotDashboard] = None
        self._K = np.array([[_DEFAULT_FX,0,_DEFAULT_CX],[0,_DEFAULT_FY,_DEFAULT_CY],[0,0,1]], np.float64)
        self._d = np.zeros((5,), np.float64)
        self._last_det_result = (None, None, None, 0)  # rvec, tvec, bgr, n
        self._auto_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        self._cb_queue: queue.Queue = queue.Queue()
        self._log_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._rebuild_board()
        self._start_camera()
        self._schedule_preview()
        self._schedule_flush()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.root.title("Hand-Eye Calibration  —  Eye-to-Hand")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(False, False)

        # Left: camera preview
        left = tk.Frame(self.root, bg="#1e1e1e")
        left.grid(row=0, column=0, sticky="nsew", padx=(8,4), pady=8)

        self._canvas = tk.Canvas(left, width=_PREVIEW_W, height=_PREVIEW_H,
                                  bg="#111", highlightthickness=1, highlightbackground="#444")
        self._canvas.pack()
        self._canvas_img_id = self._canvas.create_image(0, 0, anchor="nw")
        self._no_cam_lbl = self._canvas.create_text(
            _PREVIEW_W//2, _PREVIEW_H//2, text="Waiting for camera…",
            fill="#555", font=("Helvetica", 14))

        self._det_status = tk.Label(left, text="Detection: —",
                                     bg="#1e1e1e", fg="#666", font=("Courier", 9))
        self._det_status.pack(anchor="w", padx=4, pady=(4,0))

        # Right: controls
        right = tk.Frame(self.root, bg="#2d2d2d", width=360)
        right.grid(row=0, column=1, sticky="nsew", padx=(4,8), pady=8)
        right.grid_propagate(False)

        tk.Label(right, text="Hand-Eye Calibration", bg="#2d2d2d", fg="#61afef",
                 font=("Helvetica", 15, "bold")).pack(pady=(14,0))
        tk.Label(right, text="Eye-to-Hand  |  Orbbec Gemini 2  +  Dobot CR",
                 bg="#2d2d2d", fg="#555", font=("Helvetica", 8)).pack(pady=(0,8))

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=2)

        # --- Board Parameters ---
        self._section(right, "ChArUco Board Parameters")

        row1 = tk.Frame(right, bg="#2d2d2d"); row1.pack(fill="x", padx=12, pady=2)
        tk.Label(row1, text="Squares X:", bg="#2d2d2d", fg="#ccc",
                 font=("Helvetica", 9), width=11, anchor="w").pack(side="left")
        self._sq_x_var = tk.StringVar(value="5")
        tk.Entry(row1, textvariable=self._sq_x_var, width=5,
                 bg="#3a3a3a", fg="white", insertbackground="white",
                 relief="flat", font=("Helvetica", 9)).pack(side="left", padx=(0,12))
        tk.Label(row1, text="Y:", bg="#2d2d2d", fg="#ccc",
                 font=("Helvetica", 9)).pack(side="left")
        self._sq_y_var = tk.StringVar(value="7")
        tk.Entry(row1, textvariable=self._sq_y_var, width=5,
                 bg="#3a3a3a", fg="white", insertbackground="white",
                 relief="flat", font=("Helvetica", 9)).pack(side="left")

        row2 = tk.Frame(right, bg="#2d2d2d"); row2.pack(fill="x", padx=12, pady=2)
        tk.Label(row2, text="Square (m):", bg="#2d2d2d", fg="#ccc",
                 font=("Helvetica", 9), width=11, anchor="w").pack(side="left")
        self._sq_len_var = tk.StringVar(value="0.030")
        tk.Entry(row2, textvariable=self._sq_len_var, width=8,
                 bg="#3a3a3a", fg="white", insertbackground="white",
                 relief="flat", font=("Helvetica", 9)).pack(side="left", padx=(0,8))
        tk.Label(row2, text="Marker (m):", bg="#2d2d2d", fg="#ccc",
                 font=("Helvetica", 9)).pack(side="left")
        self._mk_len_var = tk.StringVar(value="0.022")
        tk.Entry(row2, textvariable=self._mk_len_var, width=8,
                 bg="#3a3a3a", fg="white", insertbackground="white",
                 relief="flat", font=("Helvetica", 9)).pack(side="left")

        tk.Button(right, text="♟  Apply Parameters & Update Board",
                  bg="#3a3a3a", fg="#ccc", activebackground="#444",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 9),
                  command=self._on_apply_params).pack(padx=12, pady=(4,2), fill="x", ipady=4)

        tk.Button(right, text="🖨  Generate & Save Board Image",
                  bg="#3a3a3a", fg="#abb2bf", activebackground="#444",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 9),
                  command=self._on_generate_board).pack(padx=12, pady=(2,4), fill="x", ipady=4)

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # --- Robot ---
        self._section(right, "Robot")
        row3 = tk.Frame(right, bg="#2d2d2d"); row3.pack(fill="x", padx=12, pady=2)
        tk.Label(row3, text="IP:", bg="#2d2d2d", fg="#ccc",
                 font=("Helvetica", 9), width=7, anchor="w").pack(side="left")
        self._robot_ip_var = tk.StringVar(value=self.robot_ip)
        tk.Entry(row3, textvariable=self._robot_ip_var, width=16,
                 bg="#3a3a3a", fg="white", insertbackground="white",
                 relief="flat", font=("Helvetica", 9)).pack(side="left", padx=(0,10))
        tk.Label(row3, text="Speed %:", bg="#2d2d2d", fg="#ccc",
                 font=("Helvetica", 9)).pack(side="left")
        self._speed_var = tk.StringVar(value=str(MOVE_SPEED))
        tk.Entry(row3, textvariable=self._speed_var, width=4,
                 bg="#3a3a3a", fg="white", insertbackground="white",
                 relief="flat", font=("Helvetica", 9)).pack(side="left")

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # --- Calibration ---
        self._section(right, "Calibration")

        self._mode_var = tk.StringVar(value="auto")
        modes = tk.Frame(right, bg="#2d2d2d"); modes.pack(fill="x", padx=12, pady=(2,6))
        for val, lbl in [("auto","Auto (robot moves)"), ("manual","Manual (I move robot)")]:
            tk.Radiobutton(modes, text=lbl, variable=self._mode_var, value=val,
                           bg="#2d2d2d", fg="#ccc", selectcolor="#2d2d2d",
                           activebackground="#2d2d2d", font=("Helvetica", 9),
                           command=self._on_mode_change).pack(side="left", padx=(0,12))

        self._detect_btn = tk.Button(
            right, text="👁  Start Live Detection",
            bg="#4a5568", fg="white", activebackground="#5a6578",
            relief="flat", cursor="hand2", bd=0, font=("Helvetica", 10),
            command=self._on_toggle_detect)
        self._detect_btn.pack(padx=12, pady=2, fill="x", ipady=6)

        self._start_btn = tk.Button(
            right, text="▶  Start Auto Calibration",
            bg="#61afef", fg="#1e1e1e", activebackground="#4d9bd6",
            relief="flat", cursor="hand2", bd=0, font=("Helvetica", 10, "bold"),
            command=self._on_start)
        self._start_btn.pack(padx=12, pady=2, fill="x", ipady=8)

        self._capture_btn = tk.Button(
            right, text="📷  Capture Pose  (0 captured)",
            bg="#98c379", fg="#1e1e1e", activebackground="#7aad5b",
            relief="flat", cursor="hand2", bd=0, font=("Helvetica", 10, "bold"),
            command=self._on_capture_manual, state="disabled")
        self._capture_btn.pack(padx=12, pady=2, fill="x", ipady=8)

        self._stop_btn = tk.Button(
            right, text="■  Stop",
            bg="#e06c75", fg="white", activebackground="#c0505a",
            relief="flat", cursor="hand2", bd=0, font=("Helvetica", 9),
            command=self._on_stop, state="disabled")
        self._stop_btn.pack(padx=12, pady=2, fill="x", ipady=5)

        # Progress
        prog_row = tk.Frame(right, bg="#2d2d2d"); prog_row.pack(fill="x", padx=12, pady=(6,2))
        tk.Label(prog_row, text="Poses:", bg="#2d2d2d", fg="#666",
                 font=("Helvetica", 9)).pack(side="left")
        self._progress_var = tk.StringVar(value="0 / 0")
        tk.Label(prog_row, textvariable=self._progress_var, bg="#2d2d2d", fg="#98c379",
                 font=("Helvetica", 9, "bold")).pack(side="left", padx=6)
        self._clear_btn = tk.Button(prog_row, text="Clear", bg="#3a3a3a", fg="#aaa",
                                     activebackground="#444", relief="flat", cursor="hand2",
                                     font=("Helvetica", 8), command=self._on_clear_poses)
        self._clear_btn.pack(side="right")

        self._solve_btn = tk.Button(
            right, text="✓  Solve Calibration",
            bg="#c678dd", fg="white", activebackground="#a85dc0",
            relief="flat", cursor="hand2", bd=0, font=("Helvetica", 10, "bold"),
            command=self._on_solve, state="disabled")
        self._solve_btn.pack(padx=12, pady=(4,2), fill="x", ipady=8)

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # Status
        tk.Label(right, text="Status:", bg="#2d2d2d", fg="#666",
                 font=("Helvetica", 9)).pack(anchor="w", padx=12)
        self._status_var = tk.StringVar(value="Waiting for camera…")
        tk.Label(right, textvariable=self._status_var, bg="#2d2d2d", fg="#98c379",
                 font=("Helvetica", 9), wraplength=320, justify="left"
                 ).pack(anchor="w", padx=12, pady=(2,6))

        # Log
        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=2)
        tk.Label(right, text="Log:", bg="#2d2d2d", fg="#666",
                 font=("Helvetica", 9)).pack(anchor="w", padx=12)
        self._log_text = scrolledtext.ScrolledText(
            right, width=40, height=10,
            bg="#1e1e1e", fg="#abb2bf", font=("Courier", 8),
            state="disabled", relief="flat")
        self._log_text.pack(padx=12, pady=(2,10), fill="both", expand=True)

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

        self._on_mode_change()  # set initial button states

    def _section(self, parent, title: str):
        tk.Label(parent, text=title, bg="#2d2d2d", fg="#e5c07b",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(6,2))

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def _start_camera(self):
        if not _ORBBEC_OK:
            self._log("[Camera] pyorbbecsdk not available — no live feed")
            return
        try:
            self._camera = OrbbecCamera()
            self._camera.start()
            self._K, self._d = self._camera.intrinsics()
            self._log(f"[Camera] Started  K.fx={self._K[0,0]:.1f}")
        except Exception as e:
            self._log(f"[Camera] ERROR: {e}")
            self._camera = None

    # ------------------------------------------------------------------
    # Preview loop (main thread, ~12 fps)
    # ------------------------------------------------------------------
    def _schedule_preview(self):
        self._update_preview()
        self.root.after(80, self._schedule_preview)

    def _update_preview(self):
        if not _PIL_OK:
            return
        if self._camera is None:
            return
        rgb = self._camera.get_latest_rgb()
        if rgb is None:
            return

        self._canvas.itemconfig(self._no_cam_lbl, state="hidden")
        if self._status_var.get().startswith("Waiting for camera"):
            self._status_var.set("Camera ready. Choose mode and start.")

        # Run detection when active
        if self._state in (self.DETECTING, self.MANUAL, self.AUTO):
            rvec, tvec, bgr_ann, num = detect_charuco(
                rgb, self._aruco_dict, self._board, self._detector, self._K, self._d)
            self._last_det_result = (rvec, tvec, bgr_ann, num)
            display_rgb = bgr_ann[:, :, ::-1] if bgr_ann is not None else rgb
            det_txt = (f"Detection: {num} corners  t=[{tvec[0,0]:.3f},{tvec[1,0]:.3f},{tvec[2,0]:.3f}]m"
                       if rvec is not None else f"Detection: {num} corners (need {MIN_CORNERS})")
            color = "#98c379" if rvec is not None else "#e06c75"
            self._det_status.config(text=det_txt, fg=color)
        else:
            display_rgb = rgb
            self._det_status.config(text="Detection: —", fg="#666")

        h, w = display_rgb.shape[:2]
        scale = min(_PREVIEW_W / w, _PREVIEW_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(display_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((_PREVIEW_H, _PREVIEW_W, 3), dtype=np.uint8)
        yo, xo = (_PREVIEW_H - nh) // 2, (_PREVIEW_W - nw) // 2
        padded[yo:yo+nh, xo:xo+nw] = resized

        pil = PILImage.fromarray(padded)
        tk_img = ImageTk.PhotoImage(pil)
        self._canvas.itemconfig(self._canvas_img_id, image=tk_img)
        self._canvas._tk_ref = tk_img

    # ------------------------------------------------------------------
    # Log / callback queue
    # ------------------------------------------------------------------
    def _schedule_flush(self):
        self._flush_log()
        self._flush_cb()
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

    def _log(self, msg: str):
        print(msg, flush=True)
        self._log_queue.put(msg)

    def _set_status(self, msg: str):
        self._cb_queue.put(lambda m=msg: self._status_var.set(m))

    def _ui(self, fn):
        """Schedule a UI update on the main thread."""
        self._cb_queue.put(fn)

    # ------------------------------------------------------------------
    # Board helpers
    # ------------------------------------------------------------------
    def _rebuild_board(self):
        try:
            sq_x = int(self._sq_x_var.get())
            sq_y = int(self._sq_y_var.get())
            sq_len = float(self._sq_len_var.get())
            mk_len = float(self._mk_len_var.get())
        except ValueError:
            self._log("[Board] Invalid parameters — using defaults")
            sq_x, sq_y, sq_len, mk_len = 5, 7, 0.030, 0.022

        self._aruco_dict, self._board = make_board(sq_x, sq_y, sq_len, mk_len)
        self._detector = make_charuco_detector(self._board)
        self._log(f"[Board] {sq_x}×{sq_y}  sq={sq_len*100:.1f}cm  mk={mk_len*100:.1f}cm")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_apply_params(self):
        self._rebuild_board()
        self._set_status("Board parameters updated.")

    def _on_generate_board(self):
        try:
            sq_x = int(self._sq_x_var.get())
            sq_y = int(self._sq_y_var.get())
            sq_len = float(self._sq_len_var.get())
            mk_len = float(self._mk_len_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Board parameters must be valid numbers.")
            return
        out = OUTPUT_DIR / "charuco_board.png"
        try:
            img, pw, ph = generate_board_image(sq_x, sq_y, sq_len, mk_len, out)
            self._log(f"[Board] Saved {out}  ({pw}×{ph}px)")
            self._log(f"[Board] Print at {sq_x*sq_len*100:.0f}cm × {sq_y*sq_len*100:.0f}cm")
            self._set_status(f"Board saved → {out}")
            # Show board in preview canvas
            if _PIL_OK:
                self._ui(lambda i=img: self._show_board_in_canvas(i))
        except Exception as e:
            self._log(f"[Board] ERROR: {e}")
            messagebox.showerror("Error", str(e))

    def _show_board_in_canvas(self, board_bgr: np.ndarray):
        h, w = board_bgr.shape[:2]
        scale = min(_PREVIEW_W / w, _PREVIEW_H / h)
        nw, nh = int(w * scale), int(h * scale)
        res = cv2.resize(board_bgr, (nw, nh))
        padded = np.zeros((_PREVIEW_H, _PREVIEW_W), dtype=np.uint8)
        yo, xo = (_PREVIEW_H - nh) // 2, (_PREVIEW_W - nw) // 2
        padded[yo:yo+nh, xo:xo+nw] = res
        pil = PILImage.fromarray(padded, mode="L")
        tk_img = ImageTk.PhotoImage(pil)
        self._canvas.itemconfig(self._canvas_img_id, image=tk_img)
        self._canvas._tk_ref = tk_img

    def _on_mode_change(self, *_):
        mode = self._mode_var.get()
        if mode == "auto":
            self._start_btn.config(text="▶  Start Auto Calibration", state="normal")
            self._capture_btn.config(state="disabled")
        else:
            self._start_btn.config(text="▶  Start Manual Mode", state="normal")
            if self._state == self.MANUAL:
                self._capture_btn.config(state="normal")

    def _on_toggle_detect(self):
        if self._state == self.DETECTING:
            self._state = self.IDLE
            self._detect_btn.config(text="👁  Start Live Detection", bg="#4a5568")
            self._set_status("Detection stopped.")
        else:
            if self._state != self.IDLE:
                return
            self._state = self.DETECTING
            self._detect_btn.config(text="⬛  Stop Detection", bg="#5a6578")
            self._set_status("Live detection active — point board at camera.")

    def _on_start(self):
        if self._state not in (self.IDLE, self.DETECTING):
            return
        self._rebuild_board()
        mode = self._mode_var.get()
        if mode == "auto":
            self._start_auto()
        else:
            self._start_manual()

    def _start_manual(self):
        self._state = self.MANUAL
        self._set_status("Manual mode — move robot, click Capture Pose.")
        self._log("[Manual] Started. Move robot to each position and click Capture Pose.")
        self._capture_btn.config(state="normal",
                                  text=f"📷  Capture Pose  ({len(self._robot_poses)} captured)")
        self._stop_btn.config(state="normal")
        self._start_btn.config(state="disabled")
        self._detect_btn.config(state="disabled")

    def _start_auto(self):
        ip = self._robot_ip_var.get().strip()
        speed = self._speed_var.get().strip()
        try:
            speed = int(speed)
        except ValueError:
            speed = MOVE_SPEED

        self._stop_flag.clear()
        self._state = self.AUTO
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._detect_btn.config(state="disabled")
        self._capture_btn.config(state="disabled")
        self._set_status("Connecting to robot…")
        self._auto_thread = threading.Thread(
            target=self._auto_worker, args=(ip, speed), daemon=True)
        self._auto_thread.start()

    def _auto_worker(self, ip: str, speed: int):
        self._log(f"[Auto] Connecting to robot {ip}:{ROBOT_PORT} …")
        try:
            robot = DobotDashboard(ip)
            self._log("[Auto] Clearing errors…")
            robot.clear_error(); time.sleep(0.5)
            self._log("[Auto] PowerOn…")
            robot.power_on(); time.sleep(3.0)
            self._log("[Auto] Enabling robot…")
            robot.enable(); time.sleep(2.0)
            if robot.get_mode() == DobotDashboard.MODE_ERROR:
                self._log("[Auto] Error state — clearing and re-enabling…")
                robot.clear_error(); time.sleep(1.0); robot.enable(); time.sleep(2.0)
            robot.set_speed(speed)
            self._log(f"[Auto] Robot ready. Speed={speed}%  Poses={len(CALIB_POSES)}")
        except Exception as e:
            self._log(f"[Auto] Robot connection FAILED: {e}")
            self._set_status(f"Robot error: {e}")
            self._ui(self._reset_buttons)
            return

        new_r, new_b = [], []
        for idx, target in enumerate(CALIB_POSES):
            if self._stop_flag.is_set():
                self._log("[Auto] Stopped by user.")
                break
            x, y, z, rx, ry, rz = target
            self._log(f"[Auto] Pose {idx+1}/{len(CALIB_POSES)}: {target}")
            self._set_status(f"Moving to pose {idx+1}/{len(CALIB_POSES)}…")
            cmd_id = robot.move_joint(x, y, z, rx, ry, rz)
            robot.wait_motion(cmd_id, timeout=60.0)
            self._log(f"[Auto] Settling {SETTLE_TIME}s…")
            time.sleep(SETTLE_TIME)

            # Grab a few frames and pick the one with detection
            det = None
            for _ in range(4):
                if self._camera is None:
                    break
                rgb = self._camera.get_latest_rgb()
                if rgb is None:
                    time.sleep(0.3); continue
                rvec, tvec, _, num = detect_charuco(
                    rgb, self._aruco_dict, self._board, self._detector, self._K, self._d)
                if rvec is not None:
                    det = (rvec, tvec)
                    break
                time.sleep(0.4)

            if det is None:
                self._log(f"[Auto] Pose {idx+1}: detection FAILED — skipping")
                continue

            try:
                pose = robot.get_pose()
            except Exception as e:
                self._log(f"[Auto] GetPose error: {e} — using target")
                pose = tuple(target)

            new_r.append(pose)
            new_b.append(det)
            n = len(new_r)
            self._log(f"[Auto] Pose {idx+1}: OK  ({n} collected)")
            self._ui(lambda n=n: self._update_progress(n))

        robot.close()
        self._robot_poses.extend(new_r)
        self._board_poses.extend(new_b)
        total = len(self._robot_poses)
        self._log(f"[Auto] Done. Total collected: {total}")
        self._set_status(f"Auto done — {total} poses. Click Solve to calibrate.")
        self._ui(lambda t=total: self._post_collection(t))

    def _on_capture_manual(self):
        if self._state != self.MANUAL:
            return
        if self._camera is None:
            self._log("[Manual] No camera")
            return
        rgb = self._camera.get_latest_rgb()
        if rgb is None:
            self._log("[Manual] Frame not ready — try again")
            return
        rvec, tvec, _, num = detect_charuco(
            rgb, self._aruco_dict, self._board, self._detector, self._K, self._d)
        if rvec is None:
            self._log(f"[Manual] Detection failed ({num} corners) — reposition board")
            self._set_status("Board not detected — reposition and try again.")
            return

        # Connect robot for pose if not already connected
        if self._robot is None:
            ip = self._robot_ip_var.get().strip()
            try:
                self._log(f"[Manual] Connecting to {ip}…")
                self._robot = DobotDashboard(ip)
                # In manual mode the robot is already on — skip PowerOn.
                # Just clear any alarms and enable.
                self._robot.clear_error(); time.sleep(0.5)
                self._robot.enable();      time.sleep(1.5)
                # Wait until mode == ENABLED (5) or timeout
                deadline = time.time() + 8.0
                while time.time() < deadline:
                    if self._robot.get_mode() == DobotDashboard.MODE_ENABLED:
                        break
                    time.sleep(0.3)
                self._log(f"[Manual] Robot ready (mode={self._robot.get_mode()})")
            except Exception as e:
                self._log(f"[Manual] Robot connection failed: {e}")
                self._robot = None
                return

        try:
            pose = self._robot.get_pose()
            self._log(f"[Manual] Raw pose: {pose}")
        except Exception as e:
            self._log(f"[Manual] GetPose error: {e}")
            return

        self._robot_poses.append(pose)
        self._board_poses.append((rvec, tvec))
        n = len(self._robot_poses)
        self._log(f"[Manual] Capture {n}: pose=[{','.join(f'{v:.1f}' for v in pose)}]  "
                  f"t=[{tvec[0,0]:.3f},{tvec[1,0]:.3f},{tvec[2,0]:.3f}]m")
        self._update_progress(n)

    def _on_stop(self):
        self._stop_flag.set()
        if self._state == self.MANUAL and self._robot:
            self._robot.close(); self._robot = None
        self._state = self.IDLE
        self._reset_buttons()
        self._set_status("Stopped.")
        self._log("[Stop] Stopped.")

    def _on_clear_poses(self):
        self._robot_poses.clear()
        self._board_poses.clear()
        self._update_progress(0)
        self._log("[Clear] Pose data cleared.")

    def _on_solve(self):
        n = len(self._robot_poses)
        if n < 3:
            messagebox.showwarning("Not enough poses",
                                   f"Need at least 3 valid poses, have {n}.")
            return
        if n < 8:
            if not messagebox.askyesno("Few poses",
                                        f"Only {n} poses — calibration may be inaccurate.\n"
                                        "Recommend 10+. Continue anyway?"):
                return
        self._state = self.SOLVING
        self._solve_btn.config(state="disabled", text="Solving…")
        self._set_status("Running hand-eye solver…")
        threading.Thread(target=self._solve_worker, daemon=True).start()

    def _solve_worker(self):
        try:
            self._log(f"[Solve] Running with {len(self._robot_poses)} poses…")
            result = solve_hand_eye(self._robot_poses, self._board_poses)
            ts = save_calibration(result, self._K, self._d,
                                  self._robot_poses, self._board_poses, OUTPUT_DIR)
            T = result["T"]
            t_mm = result["t"].ravel()
            self._log(f"\n[Result] Primary method: {result['primary']}")
            self._log(f"[Result] t_cam2base = [{t_mm[0]:.1f}, {t_mm[1]:.1f}, {t_mm[2]:.1f}] mm")
            if _SCIPY_OK:
                e = Rotation.from_matrix(result["R"]).as_euler("ZYX", degrees=True)
                self._log(f"[Result] R ZYX = [{e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f}] deg")
            self._log(f"[Result] T_cam2base:\n{np.array2string(T, precision=3, suppress_small=True)}")
            self._log(f"[Result] Saved to {OUTPUT_DIR}/hand_eye_calib.npz + .json")
            self._set_status(f"Calibration done!  t=[{t_mm[0]:.1f},{t_mm[1]:.1f},{t_mm[2]:.1f}]mm")
            self._log("[All methods]")
            for name, val in result["all"].items():
                if val:
                    R, t = val
                    self._log(f"  {name:<12}: t=[{t[0,0]:.1f},{t[1,0]:.1f},{t[2,0]:.1f}]mm")
        except Exception as e:
            self._log(f"[Solve] ERROR: {e}")
            self._set_status(f"Solve failed: {e}")
        finally:
            self._state = self.IDLE
            self._ui(lambda: self._solve_btn.config(
                state="normal" if len(self._robot_poses) >= 3 else "disabled",
                text="✓  Solve Calibration"))

    # ------------------------------------------------------------------
    # UI state helpers
    # ------------------------------------------------------------------
    def _update_progress(self, n: int):
        total = len(CALIB_POSES)
        self._progress_var.set(f"{n} / {total}")
        self._capture_btn.config(text=f"📷  Capture Pose  ({n} captured)")
        self._solve_btn.config(state="normal" if n >= 3 else "disabled")

    def _post_collection(self, total: int):
        self._update_progress(total)
        self._reset_buttons()
        self._state = self.IDLE

    def _reset_buttons(self):
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._detect_btn.config(state="normal",
                                 text="👁  Start Live Detection", bg="#4a5568")
        self._on_mode_change()


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Hand-Eye Calibration UI")
    ap.add_argument("--robot-ip", default=ROBOT_IP_DEFAULT)
    args = ap.parse_args()

    if not _PIL_OK:
        print("ERROR: Pillow is required.  pip install Pillow")
        sys.exit(1)

    root = tk.Tk()
    app = HandEyeCalibApp(root, args.robot_ip)
    root.mainloop()


if __name__ == "__main__":
    main()
