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

# Redirect OrbbecSDK C-level stderr to file — prevents timestamp-anomaly spam
# from flooding the terminal. Full log available at /tmp/orbbec_sdk.log
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orbbec_quiet  # noqa: E402

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
OUTPUT_DIR       = Path("/ros2_ws/data/calibration")
SAVED_POSES_FILE = OUTPUT_DIR / "auto_calib_poses.json"
MOVE_SPEED       = 15
SETTLE_TIME      = 2.5
MIN_CORNERS      = 6

_PREVIEW_W = 640
_PREVIEW_H = 480
_DEFAULT_FX, _DEFAULT_FY = 905.0, 905.0
_DEFAULT_CX, _DEFAULT_CY = 640.0, 360.0

# Pre-programmed calibration poses  [X_mm, Y_mm, Z_mm, Rx_deg, Ry_deg, Rz_deg]
#
# Design rules for good rotation accuracy:
#   1. Board must appear at DIFFERENT places in the camera image → vary X, Y, Z
#   2. Rotation axes each need ≥ 45° total spread → large Rx/Rz range, Ry 0–45°
#   3. Avoid clustering many poses at the same position with only small angle changes
#
# !! BEFORE RUNNING: verify poses 1–4 in manual mode to confirm the board is
#    visible in the camera at each position, then run auto mode. !!
CALIB_POSES = [
    # ── Centre — sweep each rotation axis independently (big range) ──────
    [300, -100, 400,   0,   0,   0],   # neutral: board flat facing camera
    [300, -100, 400,  40,   0,   0],   # Rx +40
    [300, -100, 400, -40,   0,   0],   # Rx -40
    [300, -100, 400,   0,  40,   0],   # Ry +40 (tilt forward)
    [300, -100, 400,   0,   0,  45],   # Rz +45
    [300, -100, 400,   0,   0, -45],   # Rz -45
    # ── Centre — combined / diagonal rotations ───────────────────────────
    [300, -100, 400,  30,  20,  30],
    [300, -100, 400, -30,  20, -30],
    [300, -100, 400,  30,  20, -30],
    [300, -100, 400, -30,  20,  30],
    # ── Different XY — board at different corners of the camera FOV ──────
    [210, -200, 390,   0,  20,   0],   # far left / back
    [390, -200, 390,   0,  20,   0],   # far right / back
    [210,  -20, 390,   0,  20,   0],   # far left / front
    [390,  -20, 390,   0,  20,   0],   # far right / front
    # ── Different XY with rotations ──────────────────────────────────────
    [220, -190, 380,  25,  25,  25],
    [380, -190, 380, -25,  25, -25],
    [220,  -30, 380,  25,  25, -25],
    [380,  -30, 380, -25,  25,  25],
    # ── Z variation (different camera distances) ─────────────────────────
    [300, -100, 320,   0,  15,   0],   # closer
    [300, -100, 320,  30,  15,  30],
    [300, -100, 480,   0,  15,   0],   # further
    [300, -100, 480, -30,  15, -30],
    # ── Extreme rotations — needed to break rotation degeneracy ──────────
    [300, -100, 400,  40,   5,  40],
    [300, -100, 400, -40,   5, -40],
    [300, -100, 400,   0,  45,  45],
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
        # Read intrinsics from SDK
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
        # Try to load distortion from camera_calibration.py output
        calib_file = Path("/ros2_ws/data/calibration/camera_intrinsics.npz")
        if calib_file.exists():
            try:
                data = np.load(str(calib_file))
                d = data["dist_coeffs"].ravel().astype(np.float64)
                # Also use calibrated K if image size matches
                sz = tuple(data["image_size"].astype(int)) if "image_size" in data else None
                if sz is not None and self._K is not None:
                    # Check resolution match (within 10px tolerance)
                    cx_ok = abs(data["camera_matrix"][0, 2] - self._K[0, 2]) < 50
                    if cx_ok:
                        return data["camera_matrix"].astype(np.float64), d
                return self._K, d
            except Exception:
                pass
        return self._K, np.zeros(5, np.float64)

    def stop(self):
        self._running = False
        try:
            self._pipeline.stop()
        except Exception:
            pass


# ===========================================================================
# Modular robot drivers  — see app/robots/ for available backends
# ===========================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robots import RobotBase, get_driver_names, create_robot


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


def _handeye_residual(R_c2b, t_c2b, R_b2g, t_b2g, R_t2c, t_t2c):
    """
    Consistency residual for eye-to-hand: for a perfect calibration,
    T_gripper2base_i^-1 @ T_cam2base @ T_target2cam_i = T_target2gripper (constant).
    Returns (rms_rotation_deg, rms_translation_mm).
    """
    def T4(R, t):
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(t).ravel(); return M

    TX = T4(R_c2b, t_c2b)
    Ms = []
    for i in range(len(R_b2g)):
        # R_b2g[i] IS already T_base2gripper (FK was inverted in solve_hand_eye).
        # Eye-to-hand constraint: T_base2gripper @ T_cam2base @ T_target2cam = constant
        T_b2g = T4(R_b2g[i], t_b2g[i])
        T_t2c = T4(R_t2c[i], t_t2c[i])
        Ms.append(T_b2g @ TX @ T_t2c)

    ts = np.array([M[:3, 3] for M in Ms])
    t_rms = float(np.sqrt(np.mean(np.linalg.norm(ts - ts.mean(axis=0), axis=1) ** 2)))

    R_ref = Ms[0][:3, :3]
    rot_errs = []
    for M in Ms:
        cos_a = np.clip((np.trace(R_ref.T @ M[:3, :3]) - 1) / 2, -1.0, 1.0)
        rot_errs.append(np.degrees(np.arccos(cos_a)))
    r_rms = float(np.sqrt(np.mean(np.array(rot_errs) ** 2)))

    return r_rms, t_rms


def solve_hand_eye(robot_poses: list, board_poses: list) -> dict:
    n = len(robot_poses)
    R_b2g, t_b2g, R_t2c, t_t2c = [], [], [], []
    for (x, y, z, rx, ry, rz) in robot_poses:
        R = euler_zyx_to_rmat(rx, ry, rz)
        t = np.array([[x],[y],[z]], np.float64)
        Ri, ti = R.T, -R.T @ t          # eye-to-hand: invert FK → T_base2gripper
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
    residuals = {}
    for name, flag in methods.items():
        try:
            R_c2b, t_c2b = cv2.calibrateHandEye(R_b2g, t_b2g, R_t2c, t_t2c, method=flag)
            r_rms, t_rms = _handeye_residual(R_c2b, t_c2b, R_b2g, t_b2g, R_t2c, t_t2c)
            results[name] = (R_c2b, t_c2b)
            residuals[name] = (r_rms, t_rms)
        except Exception:
            results[name] = None
            residuals[name] = (float("inf"), float("inf"))

    # Pick method with lowest translation residual (most reliable metric in mm space)
    valid = [(k, residuals[k][1]) for k in results if results[k] is not None]
    if not valid:
        raise RuntimeError("All hand-eye methods failed.")
    primary = min(valid, key=lambda x: x[1])[0]

    R_c2b, t_c2b = results[primary]
    T = np.eye(4); T[:3,:3] = R_c2b; T[:3,3] = t_c2b.ravel()
    return {"T": T, "R": R_c2b, "t": t_c2b, "primary": primary,
            "all": results, "residuals": residuals}


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
        self._guide_idx: int = 0   # tracks which CALIB_POSES target to show next
        self._goto_chaining = False  # True while stepping through poses automatically
        self._state = self.IDLE
        self._board = None
        self._aruco_dict = None
        self._detector = None
        self._camera: Optional[OrbbecCamera] = None
        self._robot: Optional[RobotBase] = None
        self._robot_connected = False
        self._K = np.array([[_DEFAULT_FX,0,_DEFAULT_CX],[0,_DEFAULT_FY,_DEFAULT_CY],[0,0,1]], np.float64)
        self._d = np.zeros((5,), np.float64)
        self._last_det_result = (None, None, None, 0)  # rvec, tvec, bgr, n
        self._auto_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._auto_poses: List[List[float]] = []   # user-saved positions for auto mode

        self._cb_queue: queue.Queue = queue.Queue()
        self._log_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._rebuild_board()
        self._load_auto_poses()
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

        # ── Column 0: camera preview ──────────────────────────────────
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

        # ── Column 1: Board setup + Auto-Mode Positions ───────────────
        mid = tk.Frame(self.root, bg="#2d2d2d", width=300)
        mid.grid(row=0, column=1, sticky="nsew", padx=(4,2), pady=8)
        mid.grid_propagate(False)

        tk.Label(mid, text="Hand-Eye Calibration", bg="#2d2d2d", fg="#61afef",
                 font=("Helvetica", 13, "bold")).pack(pady=(14,0))
        tk.Label(mid, text="Eye-to-Hand  |  Orbbec Gemini 2  +  Dobot CR",
                 bg="#2d2d2d", fg="#555", font=("Helvetica", 8)).pack(pady=(0,8))

        ttk.Separator(mid, orient="horizontal").pack(fill="x", padx=8, pady=2)

        # --- Board Parameters ---
        self._section(mid, "ChArUco Board Parameters")

        row1 = tk.Frame(mid, bg="#2d2d2d"); row1.pack(fill="x", padx=12, pady=2)
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

        row2 = tk.Frame(mid, bg="#2d2d2d"); row2.pack(fill="x", padx=12, pady=2)
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

        tk.Button(mid, text="♟  Apply Parameters & Update Board",
                  bg="#3a3a3a", fg="#ccc", activebackground="#444",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 9),
                  command=self._on_apply_params).pack(padx=12, pady=(4,2), fill="x", ipady=4)

        self._board_active_lbl = tk.Label(
            mid, text="Active: sq=3.0cm  mk=2.2cm",
            bg="#2d2d2d", fg="#e5c07b", font=("Courier", 8), anchor="w")
        self._board_active_lbl.pack(anchor="w", padx=14, pady=(0,2))

        tk.Button(mid, text="🖨  Generate & Save Board Image",
                  bg="#3a3a3a", fg="#abb2bf", activebackground="#444",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 9),
                  command=self._on_generate_board).pack(padx=12, pady=(2,4), fill="x", ipady=4)

        ttk.Separator(mid, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # --- Auto-Mode Positions ---
        self._section(mid, "Auto-Mode Positions")

        poses_hdr = tk.Frame(mid, bg="#2d2d2d"); poses_hdr.pack(fill="x", padx=12, pady=(0,2))
        self._poses_count_lbl = tk.Label(
            poses_hdr, text="0 saved  (will use built-in 15 poses)",
            bg="#2d2d2d", fg="#666", font=("Helvetica", 8), anchor="w")
        self._poses_count_lbl.pack(side="left", fill="x", expand=True)

        lb_frame = tk.Frame(mid, bg="#2d2d2d"); lb_frame.pack(fill="x", padx=12, pady=(0,4))
        lb_scroll = tk.Scrollbar(lb_frame, orient="vertical")
        self._poses_listbox = tk.Listbox(
            lb_frame, height=6, yscrollcommand=lb_scroll.set,
            bg="#1e1e1e", fg="#abb2bf", font=("Courier", 7),
            selectbackground="#3a3a3a", selectforeground="white",
            relief="flat", activestyle="none", exportselection=False)
        lb_scroll.config(command=self._poses_listbox.yview)
        self._poses_listbox.pack(side="left", fill="x", expand=True)
        lb_scroll.pack(side="right", fill="y")

        poses_btns = tk.Frame(mid, bg="#2d2d2d"); poses_btns.pack(fill="x", padx=12, pady=(0,4))
        self._save_pose_btn = tk.Button(
            poses_btns, text="📌 Save Robot Position",
            bg="#98c379", fg="#1e1e1e", activebackground="#7aad5b",
            relief="flat", cursor="hand2", bd=0, font=("Helvetica", 8, "bold"),
            command=self._on_save_auto_pose)
        self._save_pose_btn.pack(side="left", fill="x", expand=True, ipady=4, padx=(0,2))
        tk.Button(
            poses_btns, text="✕ Remove",
            bg="#3a3a3a", fg="#e06c75", activebackground="#444",
            relief="flat", cursor="hand2", bd=0, font=("Helvetica", 8),
            command=self._on_remove_auto_pose).pack(side="left", ipady=4, padx=(0,2))
        tk.Button(
            poses_btns, text="Clear All",
            bg="#3a3a3a", fg="#aaa", activebackground="#444",
            relief="flat", cursor="hand2", bd=0, font=("Helvetica", 8),
            command=self._on_clear_saved_poses).pack(side="left", ipady=4)

        # ── Column 2: Robot + Vacuum + Calibration + Log ─────────────
        right = tk.Frame(self.root, bg="#252525", width=300)
        right.grid(row=0, column=2, sticky="nsew", padx=(2,8), pady=8)
        right.grid_propagate(False)

        # --- Robot ---
        self._section(right, "Robot")
        row3t = tk.Frame(right, bg="#252525"); row3t.pack(fill="x", padx=12, pady=2)
        tk.Label(row3t, text="Type:", bg="#252525", fg="#ccc",
                 font=("Helvetica", 9), width=7, anchor="w").pack(side="left")
        driver_names = get_driver_names()
        self._robot_type_var = tk.StringVar(value=driver_names[0] if driver_names else "")
        ttk.Combobox(row3t, textvariable=self._robot_type_var,
                     values=driver_names, state="readonly",
                     font=("Helvetica", 9), width=15).pack(side="left")

        row3 = tk.Frame(right, bg="#252525"); row3.pack(fill="x", padx=12, pady=2)
        tk.Label(row3, text="IP:", bg="#252525", fg="#ccc",
                 font=("Helvetica", 9), width=7, anchor="w").pack(side="left")
        self._robot_ip_var = tk.StringVar(value=self.robot_ip)
        tk.Entry(row3, textvariable=self._robot_ip_var, width=15,
                 bg="#3a3a3a", fg="white", insertbackground="white",
                 relief="flat", font=("Helvetica", 9)).pack(side="left", padx=(0,8))
        tk.Label(row3, text="Spd%:", bg="#252525", fg="#ccc",
                 font=("Helvetica", 9)).pack(side="left")
        self._speed_var = tk.StringVar(value=str(MOVE_SPEED))
        tk.Entry(row3, textvariable=self._speed_var, width=4,
                 bg="#3a3a3a", fg="white", insertbackground="white",
                 relief="flat", font=("Helvetica", 9)).pack(side="left")

        row3b = tk.Frame(right, bg="#252525"); row3b.pack(fill="x", padx=12, pady=2)
        tk.Label(row3b, text="Tool TCP:", bg="#252525", fg="#ccc",
                 font=("Helvetica", 9), width=7, anchor="w").pack(side="left")
        self._tool_idx_var = tk.StringVar(value="1")
        tk.Entry(row3b, textvariable=self._tool_idx_var, width=4,
                 bg="#3a3a3a", fg="white", insertbackground="white",
                 relief="flat", font=("Helvetica", 9)).pack(side="left")
        tk.Label(row3b, text="(0=flange  1=vacuum)",
                 bg="#252525", fg="#555", font=("Helvetica", 8)).pack(side="left", padx=(6,0))

        self._connect_btn = tk.Button(
            right, text="⚡  Connect Robot",
            bg="#4a5568", fg="white", activebackground="#5a6578",
            relief="flat", cursor="hand2", bd=0, font=("Helvetica", 9),
            command=self._on_connect)
        self._connect_btn.pack(padx=12, pady=(6,2), fill="x", ipady=5)

        self._robot_status = tk.Label(
            right, text="● Disconnected",
            bg="#252525", fg="#e06c75", font=("Courier", 9))
        self._robot_status.pack(anchor="w", padx=12, pady=(0,4))

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=2)

        # Vacuum buttons
        vac_row = tk.Frame(right, bg="#252525"); vac_row.pack(fill="x", padx=12, pady=(4,2))
        tk.Label(vac_row, text="Vacuum DO1:", bg="#252525", fg="#ccc",
                 font=("Helvetica", 9), width=10, anchor="w").pack(side="left")
        tk.Button(vac_row, text="ON",
                  bg="#e5c07b", fg="#1e1e1e", activebackground="#c9a55f",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 9, "bold"),
                  width=5, command=self._on_vacuum_on).pack(side="left", padx=(0,4), ipady=3)
        tk.Button(vac_row, text="OFF",
                  bg="#3a3a3a", fg="#ccc", activebackground="#444",
                  relief="flat", cursor="hand2", bd=0, font=("Helvetica", 9),
                  width=5, command=self._on_vacuum_off).pack(side="left", ipady=3)

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=6)

        # --- Calibration ---
        self._section(right, "Calibration")

        self._mode_var = tk.StringVar(value="auto")
        modes = tk.Frame(right, bg="#252525"); modes.pack(fill="x", padx=12, pady=(2,6))
        for val, lbl in [("auto","Auto (robot moves)"), ("manual","Manual (I move robot)")]:
            tk.Radiobutton(modes, text=lbl, variable=self._mode_var, value=val,
                           bg="#252525", fg="#ccc", selectcolor="#252525",
                           activebackground="#252525", font=("Helvetica", 9),
                           command=self._on_mode_change).pack(side="left", padx=(0,8))

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

        # Guided pose target display (manual mode only)
        self._guide_frame = tk.Frame(right, bg="#2a2a3a", relief="flat")
        self._guide_frame.pack(padx=12, pady=(0, 4), fill="x")
        tk.Label(self._guide_frame, text="Next target pose →",
                 bg="#2a2a3a", fg="#e5c07b", font=("Helvetica", 8, "bold")
                 ).pack(anchor="w", padx=6, pady=(4, 0))
        self._guide_var = tk.StringVar(value="—")
        tk.Label(self._guide_frame, textvariable=self._guide_var,
                 bg="#2a2a3a", fg="#61afef", font=("Courier", 9),
                 justify="left", anchor="w"
                 ).pack(anchor="w", padx=6, pady=(0, 2))
        self._goto_btn = tk.Button(
            self._guide_frame, text="🤖  Send robot to this pose",
            bg="#4a5568", fg="white", relief="flat", font=("Helvetica", 9),
            command=self._on_goto_target)
        self._goto_btn.pack(fill="x", padx=6, pady=(0, 6), ipady=4)

        self._stop_btn = tk.Button(
            right, text="■  Stop",
            bg="#e06c75", fg="white", activebackground="#c0505a",
            relief="flat", cursor="hand2", bd=0, font=("Helvetica", 9),
            command=self._on_stop, state="disabled")
        self._stop_btn.pack(padx=12, pady=2, fill="x", ipady=5)

        # Progress
        prog_row = tk.Frame(right, bg="#252525"); prog_row.pack(fill="x", padx=12, pady=(6,2))
        tk.Label(prog_row, text="Poses:", bg="#252525", fg="#666",
                 font=("Helvetica", 9)).pack(side="left")
        self._progress_var = tk.StringVar(value="0 / 0")
        tk.Label(prog_row, textvariable=self._progress_var, bg="#252525", fg="#98c379",
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
        tk.Label(right, text="Status:", bg="#252525", fg="#666",
                 font=("Helvetica", 9)).pack(anchor="w", padx=12)
        self._status_var = tk.StringVar(value="Waiting for camera…")
        tk.Label(right, textvariable=self._status_var, bg="#252525", fg="#98c379",
                 font=("Helvetica", 9), wraplength=270, justify="left"
                 ).pack(anchor="w", padx=12, pady=(2,6))

        # Log
        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=2)
        tk.Label(right, text="Log:", bg="#252525", fg="#666",
                 font=("Helvetica", 9)).pack(anchor="w", padx=12)
        self._log_text = scrolledtext.ScrolledText(
            right, width=34, height=10,
            bg="#1e1e1e", fg="#abb2bf", font=("Courier", 8),
            state="disabled", relief="flat")
        self._log_text.pack(padx=12, pady=(2,10), fill="both", expand=True)

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.columnconfigure(2, weight=0)
        self.root.rowconfigure(0, weight=1)

        self._on_mode_change()  # set initial button states

    def _section(self, parent, title: str, bg: str = None):
        if bg is None:
            bg = str(parent.cget("bg"))
        tk.Label(parent, text=title, bg=bg, fg="#e5c07b",
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
            d_loaded = any(v != 0 for v in self._d)
            self._log(f"[Camera] Started  K.fx={self._K[0,0]:.1f}  "
                      f"dist={'loaded from file' if d_loaded else 'zeros (run camera_calibration.py first)'}")
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
        self._log(f"[Board] {sq_x}×{sq_y}  sq={sq_len*100:.2f}cm  mk={mk_len*100:.2f}cm")
        lbl_txt = f"Active: sq={sq_len*100:.2f}cm  mk={mk_len*100:.2f}cm  ({sq_x}×{sq_y})"
        try:
            self._board_active_lbl.config(text=lbl_txt)
        except AttributeError:
            pass   # label not built yet (called from __init__ before _build_ui)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Auto-mode pose management
    # ------------------------------------------------------------------
    def _load_auto_poses(self):
        try:
            if SAVED_POSES_FILE.exists():
                with open(SAVED_POSES_FILE, "r") as f:
                    data = json.load(f)
                raw = data.get("poses", [])
                # Support both old format (plain list) and new format (dict with joints)
                self._auto_poses = []
                for p in raw:
                    if isinstance(p, dict):
                        self._auto_poses.append(p)
                    elif isinstance(p, list):
                        self._auto_poses.append({"cartesian": p, "joints": None})
                    else:
                        continue
                self._log(f"[Poses] Loaded {len(self._auto_poses)} saved positions")
            else:
                self._log(f"[Poses] No saved positions file — will use built-in {len(CALIB_POSES)} poses")
        except Exception as e:
            self._log(f"[Poses] Load error: {e}")
        self._update_poses_listbox()

    def _save_auto_poses(self):
        try:
            SAVED_POSES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SAVED_POSES_FILE, "w") as f:
                json.dump({"poses": self._auto_poses}, f, indent=2)
        except Exception as e:
            self._log(f"[Poses] Save error: {e}")

    def _update_poses_listbox(self):
        self._poses_listbox.delete(0, "end")
        for i, entry in enumerate(self._auto_poses):
            c = entry.get("cartesian") if isinstance(entry, dict) else entry
            has_j = isinstance(entry, dict) and entry.get("joints") is not None
            tag = "J" if has_j else "C"
            if c:
                self._poses_listbox.insert("end",
                    f"#{i+1:02d} [{tag}] X={c[0]:.0f} Y={c[1]:.0f} Z={c[2]:.0f}"
                    f"  Rx={c[3]:.1f} Ry={c[4]:.1f} Rz={c[5]:.1f}")
            else:
                self._poses_listbox.insert("end", f"#{i+1:02d} [J] (joint-only)")
        n = len(self._auto_poses)
        if n == 0:
            self._poses_count_lbl.config(
                text=f"0 saved  (will use built-in {len(CALIB_POSES)} poses)", fg="#666")
        else:
            self._poses_count_lbl.config(
                text=f"{n} saved  (auto mode will use these)", fg="#98c379")

    def _on_save_auto_pose(self):
        ip = self._robot_ip_var.get().strip()
        self._save_pose_btn.config(state="disabled", text="Reading…")
        threading.Thread(target=self._save_auto_pose_thread, args=(ip,), daemon=True).start()

    def _save_auto_pose_thread(self, ip: str):
        try:
            if not self._robot_connected or self._robot is None:
                raise RuntimeError("Robot not connected")
            pose = self._robot.get_pose()    # (x,y,z,rx,ry,rz)
            angles = self._robot.get_angle() # (j1..j6)

            cartesian = list(pose)
            joints = list(angles)
            entry = {"cartesian": cartesian, "joints": joints}
            self._auto_poses.append(entry)
            self._save_auto_poses()
            self._log(f"[Poses] Saved #{len(self._auto_poses)}: "
                      f"cart=[{', '.join(f'{v:.1f}' for v in cartesian)}]  "
                      f"J=[{', '.join(f'{v:.1f}' for v in joints)}]")
            self._cb_queue.put(self._update_poses_listbox)
        except Exception as e:
            self._log(f"[Poses] Error reading robot pose: {e}")
        finally:
            self._cb_queue.put(lambda: self._save_pose_btn.config(
                state="normal", text="📌 Save Robot Position"))

    def _on_remove_auto_pose(self):
        sel = self._poses_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self._auto_poses):
            removed = self._auto_poses.pop(idx)
            self._save_auto_poses()
            c = removed.get("cartesian") if isinstance(removed, dict) else removed
            self._log(f"[Poses] Removed #{idx+1}: [{', '.join(f'{v:.1f}' for v in c)}]")
            self._update_poses_listbox()

    def _on_clear_saved_poses(self):
        if not self._auto_poses:
            return
        if not messagebox.askyesno("Clear positions",
                                    f"Delete all {len(self._auto_poses)} saved positions?"):
            return
        self._auto_poses.clear()
        self._save_auto_poses()
        self._log("[Poses] Cleared all saved positions.")
        self._update_poses_listbox()

    # ------------------------------------------------------------------
    # Robot connect / disconnect
    # ------------------------------------------------------------------
    def _on_connect(self):
        if self._robot_connected:
            # Disconnect
            if self._robot:
                try:
                    self._robot.close()
                except Exception:
                    pass
                self._robot = None
            self._robot_connected = False
            self._robot_status.config(text="● Disconnected", fg="#e06c75")
            self._connect_btn.config(text="⚡  Connect Robot", bg="#4a5568")
            self._log("[Robot] Disconnected.")
            return
        ip = self._robot_ip_var.get().strip()
        self._connect_btn.config(state="disabled", text="Connecting…")
        self._robot_status.config(text="● Connecting…", fg="#e5c07b")
        threading.Thread(target=self._connect_worker, args=(ip,), daemon=True).start()

    def _connect_worker(self, ip: str):
        try:
            driver_name = self._robot_type_var.get()
            self._log(f"[Robot] Connecting to {ip} via {driver_name!r}...")
            robot = create_robot(driver_name, ip)
            robot.clear_error(); time.sleep(0.3)
            robot.enable()
            deadline = time.time() + 15.0
            while time.time() < deadline:
                m = robot.get_mode()
                if m == robot.MODE_ENABLED:
                    break
                if m == robot.MODE_ERROR:
                    robot.clear_error(); time.sleep(0.3); robot.enable()
                time.sleep(0.4)
            m = robot.get_mode()
            if m not in (robot.MODE_ENABLED, robot.MODE_RUNNING):
                raise RuntimeError(f"Robot not ready (mode={m})")
            try:
                tool_idx = int(self._tool_idx_var.get())
            except ValueError:
                tool_idx = 1
            robot.set_tool(tool_idx)
            robot.set_speed(int(self._speed_var.get()))
            self._robot = robot
            self._robot_connected = True
            self._log(f"[Robot] Connected  mode={m}  tool={tool_idx}")
            self._ui(self._on_robot_connected_ui)
        except Exception as e:
            self._log(f"[Robot] Connection failed: {e}")
            self._ui(lambda err=str(e): (
                self._robot_status.config(text=f"● Error: {err}", fg="#e06c75"),
                self._connect_btn.config(state="normal", text="⚡  Connect Robot",
                                         bg="#4a5568")))

    def _on_robot_connected_ui(self):
        ip = self._robot_ip_var.get()
        self._robot_status.config(text=f"● Connected  {ip}", fg="#98c379")
        self._connect_btn.config(state="normal", text="⏏  Disconnect", bg="#3a6048")

    # ------------------------------------------------------------------
    # Vacuum handlers
    # ------------------------------------------------------------------
    def _on_vacuum_on(self):
        ip = self._robot_ip_var.get().strip()
        threading.Thread(target=self._vacuum_cmd, args=(ip, True), daemon=True).start()

    def _on_vacuum_off(self):
        ip = self._robot_ip_var.get().strip()
        threading.Thread(target=self._vacuum_cmd, args=(ip, False), daemon=True).start()

    def _vacuum_cmd(self, ip: str, on: bool):
        try:
            if not (self._robot_connected and self._robot):
                raise RuntimeError("Robot not connected")
            resp = self._robot.vacuum_on() if on else self._robot.vacuum_off()
            self._log(f"[Vacuum] {'ON' if on else 'OFF'}  resp={resp}")
        except Exception as e:
            self._log(f"[Vacuum] Error: {e}")

    # ------------------------------------------------------------------
    # Board param handlers
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
            self._guide_frame.pack_forget()
        else:
            self._start_btn.config(text="▶  Start Manual Mode", state="normal")
            self._guide_frame.pack(padx=12, pady=(0, 4), fill="x",
                                   after=self._capture_btn)
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

    def _update_guide(self):
        """Update the guided-pose target label for manual mode."""
        targets = self._auto_poses if self._auto_poses else CALIB_POSES
        idx = self._guide_idx
        if idx >= len(targets):
            self._guide_var.set(f"All {len(targets)} targets done ✓\n(keep adding for more accuracy)")
            return
        entry = targets[idx]
        c = entry.get("cartesian") if isinstance(entry, dict) else list(entry)
        if c and len(c) >= 6:
            self._guide_var.set(
                f"#{idx+1}/{len(targets)}  "
                f"Rx={c[3]:+.0f}°  Ry={c[4]:+.0f}°  Rz={c[5]:+.0f}°\n"
                f"  aim for  X≈{c[0]:.0f}  Y≈{c[1]:.0f}  Z≈{c[2]:.0f} mm")
        else:
            self._guide_var.set(f"#{idx+1}/{len(targets)}")

    def _on_goto_target(self):
        if self._state != self.MANUAL:
            return
        if not self._robot_connected or self._robot is None:
            self._log("[Goto] Robot not connected — click ⚡ Connect Robot first")
            return
        targets = self._auto_poses if self._auto_poses else CALIB_POSES
        if self._guide_idx >= len(targets):
            self._log("[Goto] All targets done.")
            return
        entry = targets[self._guide_idx]
        c = entry.get("cartesian") if isinstance(entry, dict) else list(entry)
        joints = entry.get("joints") if isinstance(entry, dict) else None
        if not c and joints is None:
            self._log("[Goto] No cartesian position for this target.")
            return
        self._goto_btn.config(state="disabled", text="Moving…")
        self._capture_btn.config(state="disabled")
        threading.Thread(target=self._goto_worker,
                         args=(c, joints), daemon=True).start()

    def _goto_worker(self, cart, joints):
        try:
            robot = self._robot
            if joints is not None:
                cmd_id = robot.move_joint_angles(*joints)
            else:
                x, y, z, rx, ry, rz = cart
                cmd_id = robot.move_joint(x, y, z, rx, ry, rz)
            robot.wait_motion(cmd_id, timeout=60.0)
            self._log(f"[Goto] Arrived at pose #{self._guide_idx + 1}. "
                      f"Settling {SETTLE_TIME}s…")
            time.sleep(SETTLE_TIME)
            # Auto-capture after arriving (chaining enabled)
            self._goto_chaining = True
            self._ui(self._trigger_capture_from_goto)
        except Exception as e:
            self._log(f"[Goto] Move error: {e}")
            self._ui(lambda: (
                self._goto_btn.config(state="normal",
                                      text="🤖  Send robot to this pose"),
                self._capture_btn.config(
                    state="normal" if self._state == self.MANUAL else "disabled")))

    def _trigger_capture_from_goto(self):
        """Called on main thread after robot arrives — captures if board detected."""
        self._goto_btn.config(state="normal", text="🤖  Send robot to this pose")
        if self._camera is None:
            self._capture_btn.config(state="normal")
            return
        rgb = self._camera.get_latest_rgb()
        if rgb is None:
            self._capture_btn.config(state="normal")
            return
        rvec, tvec, _, num = detect_charuco(
            rgb, self._aruco_dict, self._board, self._detector, self._K, self._d)
        if rvec is not None:
            # Board detected — auto-capture
            threading.Thread(target=self._capture_manual_worker, daemon=True).start()
        else:
            self._log(f"[Goto] Board not detected ({num} corners) — "
                      f"reposition if needed, then click Capture manually.")
            self._capture_btn.config(state="normal")

    def _start_manual(self):
        self._state = self.MANUAL
        self._guide_idx = 0
        self._update_guide()
        targets = self._auto_poses if self._auto_poses else CALIB_POSES
        self._set_status("Manual mode — move robot, click Capture Pose.")
        self._log(f"[Manual] Started. {len(targets)} target poses to collect.")
        self._log("[Manual] Move robot to the shown orientation and click Capture Pose.")
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
        try:
            if self._robot_connected and self._robot is not None:
                robot = self._robot
                # Health-check: verify the socket is still alive
                try:
                    robot.get_pose()
                    self._log("[Auto] Using existing robot connection.")
                except Exception:
                    self._log("[Auto] Existing connection dead — reconnecting…")
                    try:
                        robot.close()
                    except Exception:
                        pass
                    self._robot = None
                    self._robot_connected = False
                    robot = None
            else:
                robot = None

            if robot is not None:
                robot.set_speed(speed)
                try:
                    tool_idx = int(self._tool_idx_var.get())
                except ValueError:
                    tool_idx = 1
                robot.set_tool(tool_idx)
                self._log(f"[Auto] Speed={speed}%  Tool={tool_idx}")
            else:
                driver_name = self._robot_type_var.get()
                self._log(f"[Auto] Connecting to robot {ip} via {driver_name!r}...")
                robot = create_robot(driver_name, ip)
                self._log("[Auto] Clearing errors...")
                robot.clear_error(); time.sleep(0.5)
                self._log("[Auto] PowerOn...")
                robot.power_on(); time.sleep(3.0)
                self._log("[Auto] Enabling robot...")
                robot.enable()
                deadline = time.time() + 20.0
                while time.time() < deadline:
                    m = robot.get_mode()
                    self._log(f"[Auto] Waiting for idle... mode={m}")
                    if m == robot.MODE_ENABLED:
                        break
                    if m == robot.MODE_ERROR:
                        robot.clear_error(); time.sleep(0.5); robot.enable()
                    time.sleep(0.5)
                final_mode = robot.get_mode()
                if final_mode != robot.MODE_ENABLED:
                    raise RuntimeError(f"Robot not idle after 20s (mode={final_mode}). "
                                       f"Check pendant -- may need manual enable.")
                self._log(f"[Auto] Robot idle (mode={final_mode}). Proceeding.")
                robot.set_speed(speed)
                try:
                    tool_idx = int(self._tool_idx_var.get())
                except ValueError:
                    tool_idx = 1
                robot.set_tool(tool_idx)
                self._log(f"[Auto] Tool TCP set to index {tool_idx}")
                self._robot = robot
                self._robot_connected = True
                self._ui(self._on_robot_connected_ui)
            poses_to_use = self._auto_poses if self._auto_poses else CALIB_POSES
            src = "saved" if self._auto_poses else "built-in"
            self._log(f"[Auto] Robot ready. Speed={speed}%  Poses={len(poses_to_use)} ({src})")
        except Exception as e:
            self._log(f"[Auto] Robot connection FAILED: {e}")
            self._set_status(f"Robot error: {e}")
            self._ui(self._reset_buttons)
            return

        # Log current robot mode so we can see if it's actually ready
        mode_now = robot.get_mode()
        self._log(f"[Auto] Robot mode before moves: {mode_now} "
                  f"(5=enabled, 7=running, 9=error)")

        new_r, new_b = [], []
        try:
            for idx, entry in enumerate(poses_to_use):
                if self._stop_flag.is_set():
                    self._log("[Auto] Stopped by user.")
                    break

                # Determine how to move: joint angles (saved) or Cartesian (built-in)
                if isinstance(entry, dict):
                    joints = entry.get("joints")
                    cart = entry.get("cartesian", joints)
                else:
                    joints = None
                    cart = list(entry)

                self._log(f"[Auto] Pose {idx+1}/{len(poses_to_use)}: {cart}")
                self._set_status(f"Moving to pose {idx+1}/{len(poses_to_use)}…")
                try:
                    if joints is not None:
                        cmd_id = robot.move_joint_angles(*joints)
                        self._log(f"[Auto] MovJ (joints) sent  cmd_id={cmd_id}")
                    else:
                        x, y, z, rx, ry, rz = cart
                        cmd_id = robot.move_joint(x, y, z, rx, ry, rz)
                        self._log(f"[Auto] MovJ (cartesian) sent  cmd_id={cmd_id}")
                    robot.wait_motion(cmd_id, timeout=60.0)
                except TimeoutError:
                    self._log(f"[Auto] TIMEOUT waiting for pose {idx+1} — robot mode={robot.get_mode()}")
                    self._set_status(f"Timeout at pose {idx+1}. Check robot state.")
                    break
                except RuntimeError as e:
                    self._log(f"[Auto] MOVE ERROR at pose {idx+1}: {e}")
                    self._set_status(f"Move error: {e}")
                    break
                self._log(f"[Auto] Arrived. Settling {SETTLE_TIME}s…")
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
                    self._log(f"[Auto] GetPose error: {e} — skipping pose {idx+1}")
                    continue

                new_r.append(pose)
                new_b.append(det)
                n = len(new_r)
                self._log(f"[Auto] Pose {idx+1}: OK  ({n} collected)")
                self._ui(lambda n=n: self._update_progress(n))

        except Exception as e:
            self._log(f"[Auto] Unexpected error: {e}")
            self._set_status(f"Auto error: {e}")
        finally:
            # Only close if we opened the connection ourselves
            if not self._robot_connected or self._robot is not robot:
                try:
                    robot.close()
                except Exception:
                    pass
            self._robot_poses.extend(new_r)
            self._board_poses.extend(new_b)
            total = len(self._robot_poses)
            self._log(f"[Auto] Done. Total collected: {total}")
            self._set_status(f"Auto done — {total} poses. Click Solve to calibrate.")
            self._ui(lambda t=total: self._post_collection(t))

    def _on_capture_manual(self):
        self._goto_chaining = False  # manual click breaks auto-chain
        if self._state != self.MANUAL:
            return
        if self._camera is None:
            self._log("[Manual] No camera")
            return
        self._capture_btn.config(state="disabled", text="Capturing…")
        threading.Thread(target=self._capture_manual_worker, daemon=True).start()

    def _capture_manual_worker(self):
        try:
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

            # Use persistent connection, or fail if not connected
            if not self._robot_connected or self._robot is None:
                self._log("[Manual] Robot not connected — click ⚡ Connect Robot first")
                self._set_status("Robot not connected.")
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
            self._guide_idx += 1
            chaining = self._goto_chaining
            def _post(n=n, c=chaining):
                self._update_progress(n)
                self._update_guide()
                if c:
                    targets = self._auto_poses if self._auto_poses else CALIB_POSES
                    if self._guide_idx < len(targets):
                        self.root.after(1500, self._on_goto_target)
                    else:
                        self._goto_chaining = False
                        self._log("[Goto] All poses captured — click Solve to calibrate.")
            self._ui(_post)
        finally:
            n = len(self._robot_poses)
            self._ui(lambda: self._capture_btn.config(
                state="normal" if self._state == self.MANUAL else "disabled",
                text=f"📷  Capture Pose  ({n} captured)"))

    def _on_stop(self):
        self._stop_flag.set()
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
            primary_r, primary_t = result["residuals"][result["primary"]]
            self._log(f"\n[Result] Primary method: {result['primary']}  "
                      f"(residual: rot={primary_r:.3f}°  t={primary_t:.2f}mm)")
            self._log(f"[Result] t_cam2base = [{t_mm[0]:.1f}, {t_mm[1]:.1f}, {t_mm[2]:.1f}] mm")
            if _SCIPY_OK:
                e = Rotation.from_matrix(result["R"]).as_euler("ZYX", degrees=True)
                self._log(f"[Result] R ZYX = [{e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f}] deg")
            self._log(f"[Result] T_cam2base:\n{np.array2string(T, precision=3, suppress_small=True)}")
            self._log(f"[Result] Saved to {OUTPUT_DIR}/hand_eye_calib.npz + .json")
            quality = "GOOD" if primary_t < 3.0 else ("ACCEPTABLE" if primary_t < 8.0 else "POOR — recollect poses")
            self._set_status(f"Calibration done! [{quality}]  residual={primary_t:.2f}mm  "
                             f"t=[{t_mm[0]:.1f},{t_mm[1]:.1f},{t_mm[2]:.1f}]mm")
            self._log("[All methods — translation residual (lower = better)]")
            for name, val in result["all"].items():
                if val:
                    R, t = val
                    r_res, t_res = result["residuals"][name]
                    marker = " ← selected" if name == result["primary"] else ""
                    self._log(f"  {name:<12}: t=[{t[0,0]:.1f},{t[1,0]:.1f},{t[2,0]:.1f}]mm  "
                              f"residual={t_res:.2f}mm{marker}")
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
        total = len(self._auto_poses) if self._auto_poses else len(CALIB_POSES)
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
