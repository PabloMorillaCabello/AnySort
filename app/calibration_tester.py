#!/usr/bin/env python3
"""
Calibration Quality Tester
===========================
Tests hand-eye calibration accuracy by:
  1. Detecting an ArUco marker (4X4_50, ID 0) with the Orbbec camera
  2. Unprojecting the marker centroid to 3D using the depth frame
  3. Transforming to robot base frame via T_cam2base

Two test modes
--------------
  Mode A — "Move to Predicted":
    Robot moves its TCP to the camera-predicted marker position.
    Visually inspect how far off the TCP lands.
    Actual TCP pose (from Dobot feedback) is compared vs predicted.

  Mode B — "Ground Truth at Current Pose":
    Jog the robot TCP to the marker manually, click "Set Ground Truth".
    Camera predicts where the marker is.
    Error = prediction − ground_truth.  No automatic robot motion needed.

6-DOF Correction Sliders
-------------------------
    ΔX / ΔY / ΔZ (mm) and ΔRoll / ΔPitch / ΔYaw (deg) are applied on top of
    T_cam2base in real-time.  The live preview shows the corrected prediction.
    Click "Save Corrected Calibration" to write a new .npz / .json.

Usage:
  python3 /ros2_ws/scripts/TEST/calibration_tester.py
  python3 /ros2_ws/scripts/TEST/calibration_tester.py --robot-ip 192.168.5.1
  python3 /ros2_ws/scripts/TEST/calibration_tester.py --calib /ros2_ws/data/calibration/hand_eye_calib.npz

ArUco marker:
  Print a 4X4_50 marker, ID 0 and place it flat in the workspace.
  The "Gen Marker" button saves aruco_id0.png to the calibration folder.
"""

import argparse
import gc
import json
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

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

# ===========================================================================
# Constants
# ===========================================================================
CALIB_FILE_DEFAULT  = "/ros2_ws/data/calibration/hand_eye_calib.npz"
CALIB_DIR           = Path("/ros2_ws/data/calibration")
ROBOT_IP_DEFAULT    = "192.168.5.1"
APPROACH_OFFSET_MM  = 80                         # mm above marker for Mode A
MOVE_SPEED          = 15                         # % speed for test moves

_PREVIEW_W = 640
_PREVIEW_H = 480
_DEFAULT_FX, _DEFAULT_FY = 905.0, 905.0
_DEFAULT_CX, _DEFAULT_CY = 640.0, 360.0

ARUCO_DICT_ID   = cv2.aruco.DICT_4X4_50
ARUCO_MARKER_ID = 0   # which marker ID to track


# ===========================================================================
# ArUco helpers
# ===========================================================================
def _build_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params     = cv2.aruco.DetectorParameters()
    try:                                    # OpenCV ≥ 4.7
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        def _detect(gray):
            return detector.detectMarkers(gray)
    except AttributeError:                  # OpenCV < 4.7
        def _detect(gray):
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
            return corners, ids, None
    return _detect


_aruco_detect = _build_aruco_detector()


def detect_aruco(rgb: np.ndarray) -> dict | None:
    """
    Detect ArUco marker with id ARUCO_MARKER_ID in rgb (H×W×3 uint8).
    Returns dict(corners, centroid_px, rvec, tvec) or None if not found.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _aruco_detect(gray)
    if ids is None or len(ids) == 0:
        return None
    ids_flat = ids.flatten()
    for i, mid in enumerate(ids_flat):
        if int(mid) == ARUCO_MARKER_ID:
            c = corners[i][0]  # (4, 2)
            cx = float(c[:, 0].mean())
            cy = float(c[:, 1].mean())
            return {"corners": c, "centroid_px": (cx, cy)}
    return None


def draw_aruco_overlay(rgb: np.ndarray, detection: dict | None,
                       predicted_px: tuple | None = None) -> np.ndarray:
    """Draw marker outline + centroid on a copy of rgb."""
    out = rgb.copy()
    if detection is not None:
        c = detection["corners"].astype(int)
        cv2.polylines(out, [c.reshape(-1, 1, 2)], True, (0, 220, 80), 2)
        u, v = int(detection["centroid_px"][0]), int(detection["centroid_px"][1])
        cv2.drawMarker(out, (u, v), (0, 220, 80), cv2.MARKER_CROSS, 20, 2)
    if predicted_px is not None:
        pu, pv = int(predicted_px[0]), int(predicted_px[1])
        cv2.drawMarker(out, (pu, pv), (255, 80, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
        cv2.putText(out, "predicted", (pu + 8, pv - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 0), 1)
    return out


def generate_aruco_image(size_px: int = 400) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    try:
        img = cv2.aruco.generateImageMarker(dictionary, ARUCO_MARKER_ID, size_px)
    except AttributeError:
        img = np.zeros((size_px, size_px), np.uint8)
        img = cv2.aruco.drawMarker(dictionary, ARUCO_MARKER_ID, size_px, img, 1)
    return img


# ===========================================================================
# 3-D unprojection helper
# ===========================================================================
def unproject_pixel(u: float, v: float, depth_m: np.ndarray,
                    fx: float, fy: float, cx: float, cy: float,
                    patch: int = 5) -> np.ndarray | None:
    """
    Given a pixel (u, v) and a depth map (in meters), return 3-D point in
    camera frame in MILLIMETRES using a median patch for robustness.
    Returns None if no valid depth found.
    """
    h, w = depth_m.shape
    u0, v0 = int(round(u)), int(round(v))
    r = patch // 2
    u1, u2 = max(0, u0 - r), min(w, u0 + r + 1)
    v1, v2 = max(0, v0 - r), min(h, v0 + r + 1)
    patch_d = depth_m[v1:v2, u1:u2]
    valid   = patch_d[(patch_d > 0.05) & np.isfinite(patch_d)]
    if valid.size == 0:
        return None
    Z_m   = float(np.median(valid))
    Z_mm  = Z_m * 1000.0
    X_mm  = (u - cx) * Z_mm / fx
    Y_mm  = (v - cy) * Z_mm / fy
    return np.array([X_mm, Y_mm, Z_mm], dtype=np.float64)


# ===========================================================================
# Calibration load / correction / save
# ===========================================================================
def load_calibration(path: str):
    """Returns T_cam2base (4×4, t in mm), K (3×3) or (None, None)."""
    data = np.load(path)
    T = data["T_cam2base"].copy()  # 4×4, translation in mm
    K = data["camera_matrix"].copy() if "camera_matrix" in data else None
    return T, K


def build_correction_matrix(dx_mm, dy_mm, dz_mm,
                             drx_deg, dry_deg, drz_deg) -> np.ndarray:
    """
    Build a 4×4 correction transform (in robot base frame units = mm).
    Pre-multiplied onto T_cam2base so that the correction is additive in base frame.
    """
    if _SCIPY_OK:
        R_corr = Rotation.from_euler("ZYX",
                                     [drz_deg, dry_deg, drx_deg],
                                     degrees=True).as_matrix()
    else:
        R_corr = np.eye(3)
    T_corr = np.eye(4)
    T_corr[:3, :3] = R_corr
    T_corr[:3,  3] = [dx_mm, dy_mm, dz_mm]
    return T_corr


def apply_T_cam2base(P_cam_mm: np.ndarray, T_cam2base: np.ndarray,
                     T_correction: np.ndarray | None = None) -> np.ndarray:
    """
    Transform a 3-D point from camera frame (mm) → robot base frame (mm).
    Optionally applies a correction pre-multiplied on T_cam2base.
    Returns xyz in mm.
    """
    T = T_cam2base.copy()
    if T_correction is not None:
        T = T_correction @ T
    p_h = np.array([P_cam_mm[0], P_cam_mm[1], P_cam_mm[2], 1.0])
    result = T @ p_h
    return result[:3]


def save_corrected_calibration(original_path: str,
                                T_corrected: np.ndarray,
                                T_correction: np.ndarray,
                                out_dir: Path):
    """Save a new .npz and .json with the corrected T_cam2base."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Load originals to preserve camera matrix etc.
    data = dict(np.load(original_path))
    data["T_cam2base"]   = T_corrected
    data["R_cam2base"]   = T_corrected[:3, :3]
    data["t_cam2base"]   = T_corrected[:3, 3]
    data["T_correction"] = T_correction
    npz_path = out_dir / f"hand_eye_calib_corrected_{ts}.npz"
    np.savez(str(npz_path), **data)
    json_path = out_dir / f"hand_eye_calib_corrected_{ts}.json"
    R = T_corrected[:3, :3]
    t = T_corrected[:3, 3]
    euler = Rotation.from_matrix(R).as_euler("ZYX", degrees=True).tolist() if _SCIPY_OK else []
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "T_cam2base": T_corrected.tolist(),
            "t_cam2base_mm": t.tolist(),
            "euler_ZYX_deg": euler,
            "T_correction": T_correction.tolist(),
        }, f, indent=2)
    return npz_path, json_path


# ===========================================================================
# Orbbec camera  (RGB + depth + intrinsics — background thread)
# ===========================================================================
def _to_rgb_array(color_frame, OBFormat):
    h, w = color_frame.get_height(), color_frame.get_width()
    raw  = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    fmt  = color_frame.get_format()
    fmt_s = str(fmt).upper()
    if hasattr(OBFormat, "RGB")  and fmt == OBFormat.RGB:
        return raw.reshape(h, w, 3).copy()
    if hasattr(OBFormat, "BGR")  and fmt == OBFormat.BGR:
        return raw.reshape(h, w, 3)[:, :, ::-1].copy()
    if (hasattr(OBFormat, "MJPG") and fmt == OBFormat.MJPG) or "MJPG" in fmt_s:
        return np.array(PILImage.open(BytesIO(raw.tobytes())).convert("RGB"))
    if (hasattr(OBFormat, "YUYV") and fmt == OBFormat.YUYV) or "YUYV" in fmt_s:
        return cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUY2)
    if raw.size == h * w * 3:
        return raw.reshape(h, w, 3).copy()
    raise RuntimeError(f"Unsupported color format: {fmt}")


def _extract_intrinsics(profile_or_frame):
    """Try to read (fx, fy, cx, cy) from a stream profile or frame object."""
    if profile_or_frame is None:
        raise RuntimeError("Cannot read intrinsics from None")
    for method in ["get_intrinsic", "get_camera_intrinsic", "get_intrinsics"]:
        if not hasattr(profile_or_frame, method):
            continue
        intr = getattr(profile_or_frame, method)()
        try:
            fx = float(next(getattr(intr, a) for a in ["fx", "focal_x"] if hasattr(intr, a)))
            fy = float(next(getattr(intr, a) for a in ["fy", "focal_y"] if hasattr(intr, a)))
            cx = float(next(getattr(intr, a) for a in ["cx", "ppx", "principal_x"] if hasattr(intr, a)))
            cy = float(next(getattr(intr, a) for a in ["cy", "ppy", "principal_y"] if hasattr(intr, a)))
            return fx, fy, cx, cy
        except StopIteration:
            continue
    raise RuntimeError("Cannot read camera intrinsics")


class OrbbecCamera:
    """Orbbec Gemini 2 camera with HW D2C alignment and frame sync.

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
        self._lock     = threading.Lock()
        self._rgb      = None
        self._depth_m  = None
        self._intr     = None
        self._running  = False
        self._color_profile = None
        self._depth_filters = []

    def start(self):
        from pyorbbecsdk import Config, OBAlignMode, OBFormat, OBSensorType, Pipeline
        pipeline = Pipeline()

        # Enable hardware frame sync
        try:
            pipeline.enable_frame_sync()
            print("[Camera] Frame sync enabled")
        except Exception as e:
            print(f"[Camera] Frame sync failed: {e}")

        config = Config()

        # Find an RGB colour profile and a matching HW-D2C depth profile
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
            print("[Camera] WARNING: HW D2C unavailable -- falling back to defaults")
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

        try:
            import orbbec_quiet
            orbbec_quiet.reapply()
        except Exception:
            pass

        # Depth post-processing filters
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

        # Let the sensor stabilise
        print("[Camera] Stabilising sensor (15 frames)...")
        for _ in range(15):
            try:
                pipeline.wait_for_frames(200)
            except Exception:
                pass

        self._pipeline = pipeline
        self._running  = True
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
                # Apply depth post-processing filters
                for filt in self._depth_filters:
                    if filt.is_enabled():
                        try:
                            df = filt.process(df)
                        except Exception:
                            pass
                if hasattr(df, "as_depth_frame"):
                    df = df.as_depth_frame()
                dh, dw = df.get_height(), df.get_width()
                depth_raw = np.frombuffer(df.get_data(),
                                          dtype=np.uint16).reshape(dh, dw)
                scale = float(df.get_depth_scale())
                depth_m = depth_raw.astype(np.float32) * scale
                # Auto-detect mm vs m
                valid = depth_m[(depth_m > 0.05) & np.isfinite(depth_m)]
                if valid.size > 0 and float(np.median(valid)) > 20.0:
                    depth_m /= 1000.0

                # With HW D2C, depth is already aligned to colour resolution
                if rgb.shape[0] != dh or rgb.shape[1] != dw:
                    rgb = np.array(PILImage.fromarray(rgb).resize(
                        (dw, dh), PILImage.BILINEAR))

                # Use colour intrinsics (the D2C alignment target)
                intr = _extract_intrinsics(self._color_profile)
            except Exception:
                continue
            with self._lock:
                self._rgb     = rgb
                self._depth_m = depth_m
                self._intr    = intr

    def get_latest(self):
        with self._lock:
            if self._rgb is None:
                return None, None, None
            return self._rgb.copy(), self._depth_m.copy(), self._intr

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
# Main Tkinter application
# ===========================================================================
class CalibTesterApp:

    # ------------------------------------------------------------------ init
    def __init__(self, root: tk.Tk, args):
        self.root   = root
        self._args  = args
        root.title("Hand-Eye Calibration Quality Tester")
        root.resizable(False, False)

        # State
        self._cam           = None
        self._robot         = None
        self._T_cam2base    = None   # loaded calibration (4×4, t in mm)
        self._K_calib       = None   # camera matrix from calibration file
        self._calib_path    = args.calib
        self._robot_ip      = args.robot_ip
        self._test_results  = []     # list of result dicts
        self._home_pose     = None   # [X,Y,Z,Rx,Ry,Rz] set by user
        self._ground_truth  = None   # (x,y,z) mm in robot frame
        self._last_det      = None   # last ArUco detection result
        self._last_pred_mm  = None   # last predicted robot coords (corrected)
        self._frame_q       = queue.Queue(maxsize=2)
        self._worker_busy   = False
        self._cam_connected = False
        self._robot_connected = False
        self._cam_dot_state  = "init"    # track last dot state to avoid redundant config
        self._robot_dot_state = "init"

        # 6-DOF correction variables
        self._cv = {k: tk.DoubleVar(value=0.0) for k in
                    ("dx", "dy", "dz", "drx", "dry", "drz")}

        self._build_ui()
        self._start_camera_loop()

    # ------------------------------------------------------------- UI layout
    def _build_ui(self):
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left  = tk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        right = tk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(1, weight=1)

        # --- Camera preview ---
        preview_lf = tk.LabelFrame(left, text="Camera Preview  (ArUco 4X4_50 ID=0)")
        preview_lf.pack(fill=tk.BOTH)
        self._canvas = tk.Canvas(preview_lf, width=_PREVIEW_W, height=_PREVIEW_H,
                                 bg="#1a1a1a")
        self._canvas.pack()
        self._status_lbl = tk.Label(left, text="No camera", anchor="w",
                                    fg="gray", font=("Courier", 9))
        self._status_lbl.pack(fill=tk.X)

        # --- Right panel stacked ---
        self._build_connection_frame(right)
        self._build_detection_frame(right)
        self._build_test_frame(right)
        self._build_correction_frame(right)
        self._build_log_frame(right)

    def _build_connection_frame(self, parent):
        lf = tk.LabelFrame(parent, text="Connections")
        lf.pack(fill=tk.X, pady=(0, 4))

        # Calibration file row — combo showing available files + Browse
        tk.Label(lf, text="Calibration:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self._calib_var = tk.StringVar(value=self._calib_path)
        self._calib_combo_t = ttk.Combobox(lf, textvariable=self._calib_var, width=33)
        self._calib_combo_t.grid(row=0, column=1, padx=2)
        self._refresh_calib_combo_t()
        tk.Button(lf, text="Browse…", command=self._browse_calib).grid(row=0, column=2, padx=2)
        tk.Button(lf, text="Load", command=self._load_calib,
                  bg="#3a7fc1", fg="white").grid(row=0, column=3, padx=2)

        # Robot type row
        tk.Label(lf, text="Robot:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        driver_names = get_driver_names()
        self._robot_type_var = tk.StringVar(value=driver_names[0] if driver_names else "")
        ttk.Combobox(lf, textvariable=self._robot_type_var,
                     values=driver_names, state="readonly",
                     width=14).grid(row=1, column=1, sticky="w", padx=2)

        # Robot IP row
        tk.Label(lf, text="IP:").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self._ip_var = tk.StringVar(value=self._robot_ip)
        tk.Entry(lf, textvariable=self._ip_var, width=18).grid(row=2, column=1, sticky="w", padx=2)
        self._connect_btn = tk.Button(lf, text="Connect Robot",
                                      command=self._connect_robot,
                                      bg="#2e7d32", fg="white")
        self._connect_btn.grid(row=2, column=2, columnspan=2, padx=2, pady=2)

        # Status dots row
        status_row = tk.Frame(lf)
        status_row.grid(row=3, column=0, columnspan=4, sticky="w", padx=4, pady=(0, 4))
        self._calib_dot  = tk.Label(status_row, text="● Calib: NOT loaded", fg="red",  font=("Courier", 9))
        self._calib_dot.pack(side=tk.LEFT, padx=(0, 12))
        self._robot_dot  = tk.Label(status_row, text="● Robot: disconnected", fg="red", font=("Courier", 9))
        self._robot_dot.pack(side=tk.LEFT)
        self._cam_dot    = tk.Label(status_row, text="● Camera: connecting…", fg="orange", font=("Courier", 9))
        self._cam_dot.pack(side=tk.LEFT, padx=(12, 0))

        # Utility buttons
        util = tk.Frame(lf)
        util.grid(row=4, column=0, columnspan=4, sticky="w", padx=4, pady=(0, 4))
        tk.Button(util, text="Gen Marker", command=self._gen_marker).pack(side=tk.LEFT, padx=2)
        tk.Button(util, text="Enable Robot", command=self._enable_robot).pack(side=tk.LEFT, padx=2)
        tk.Button(util, text="Clear Error", command=self._clear_error).pack(side=tk.LEFT, padx=2)
        tk.Button(util, text="Set Home\n(current pose)", command=self._set_home,
                  bg="#4e342e", fg="white").pack(side=tk.LEFT, padx=2)
        self._home_btn = tk.Button(util, text="Go Home", command=self._go_home,
                                   bg="#5d4037", fg="white", state="disabled")
        self._home_btn.pack(side=tk.LEFT, padx=2)

    def _build_detection_frame(self, parent):
        lf = tk.LabelFrame(parent, text="Live Detection")
        lf.pack(fill=tk.X, pady=(0, 4))

        self._det_info = tk.Label(lf, text="Marker: —\nCamera (mm): —\nRobot predicted (mm): —",
                                  anchor="w", justify=tk.LEFT, font=("Courier", 9))
        self._det_info.pack(fill=tk.X, padx=4, pady=4)

    def _build_test_frame(self, parent):
        lf = tk.LabelFrame(parent, text="Tests")
        lf.pack(fill=tk.X, pady=(0, 4))

        btn_row = tk.Frame(lf)
        btn_row.pack(fill=tk.X, padx=4, pady=4)

        tk.Button(btn_row, text="[A] Approach\n(+offset)",
                  command=self._test_a_move,
                  bg="#1565c0", fg="white", width=16).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="[A2] Descend\nto Marker",
                  command=self._test_a2_descend,
                  bg="#0d47a1", fg="white", width=16).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="[B] Set Ground Truth\n(current pose)",
                  command=self._test_b_set_gt,
                  bg="#6a1b9a", fg="white", width=22).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Clear\nLog",
                  command=self._clear_results).pack(side=tk.LEFT, padx=2)

        # Error table (Treeview)
        cols = ("#", "Pred X", "Pred Y", "Pred Z", "Actual X", "Actual Y", "Actual Z",
                "Err X", "Err Y", "Err Z", "Dist mm")
        self._tree = ttk.Treeview(lf, columns=cols, show="headings", height=4)
        for c in cols:
            self._tree.heading(c, text=c)
            w = 38 if c != "#" else 26
            self._tree.column(c, width=w, anchor="center")
        self._tree.pack(fill=tk.X, padx=4)

        self._stats_lbl = tk.Label(lf, text="Mean error: —", anchor="w",
                                   font=("Courier", 9), fg="#333")
        self._stats_lbl.pack(fill=tk.X, padx=4, pady=(2, 4))

    def _build_correction_frame(self, parent):
        lf = tk.LabelFrame(parent, text="6-DOF Correction  (applied on top of T_cam2base)")
        lf.pack(fill=tk.X, pady=(0, 4))

        sliders = [
            ("ΔX (mm)",    "dx",  -200, 200),
            ("ΔY (mm)",    "dy",  -200, 200),
            ("ΔZ (mm)",    "dz",  -200, 200),
            ("ΔRoll (°)",  "drx", -30,  30),
            ("ΔPitch (°)", "dry", -30,  30),
            ("ΔYaw (°)",   "drz", -30,  30),
        ]
        self._slider_lbls = {}
        for i, (label, key, lo, hi) in enumerate(sliders):
            row, col = divmod(i, 2)
            fr = tk.Frame(lf)
            fr.grid(row=row, column=col, padx=6, pady=2, sticky="ew")
            lf.columnconfigure(col, weight=1)
            tk.Label(fr, text=label, width=10, anchor="w").pack(side=tk.LEFT)
            sl = tk.Scale(fr, from_=lo, to=hi, resolution=0.5, orient=tk.HORIZONTAL,
                          variable=self._cv[key], length=160,
                          command=lambda _v, k=key: self._on_slider(k))
            sl.pack(side=tk.LEFT)
            val_lbl = tk.Label(fr, textvariable=self._cv[key], width=6,
                               font=("Courier", 9))
            val_lbl.pack(side=tk.LEFT)
            self._slider_lbls[key] = val_lbl

        btn_row = tk.Frame(lf)
        btn_row.grid(row=3, column=0, columnspan=2, padx=4, pady=4, sticky="w")
        tk.Button(btn_row, text="Reset All to 0", command=self._reset_correction).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Save Corrected Calibration",
                  command=self._save_correction,
                  bg="#2e7d32", fg="white").pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Auto-Fill From Errors",
                  command=self._autofill_correction,
                  bg="#e65100", fg="white").pack(side=tk.LEFT, padx=2)

    def _build_log_frame(self, parent):
        lf = tk.LabelFrame(parent, text="Log")
        lf.pack(fill=tk.BOTH, expand=True)
        self._log = scrolledtext.ScrolledText(lf, height=7, font=("Courier", 9),
                                              state=tk.DISABLED, wrap=tk.WORD)
        self._log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------- log
    def _log_msg(self, msg: str):
        def _do():
            self._log.config(state=tk.NORMAL)
            self._log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            self._log.see(tk.END)
            self._log.config(state=tk.DISABLED)
        self.root.after(0, _do)

    def _set_status(self, msg: str):
        self.root.after(0, lambda: self._status_lbl.config(text=msg))

    # -------------------------------------------------------- calibration IO
    def _refresh_calib_combo_t(self):
        """Populate the calibration combobox with available .npz files."""
        files = []
        if CALIB_DIR.is_dir():
            for f in sorted(CALIB_DIR.glob("hand_eye_calib*.npz")):
                files.append(str(f))
        try:
            self._calib_combo_t["values"] = files
        except Exception:
            pass

    def _browse_calib(self):
        path = filedialog.askopenfilename(
            title="Select calibration .npz",
            filetypes=[("NumPy archive", "*.npz"), ("All", "*.*")])
        if path:
            self._calib_var.set(path)

    def _load_calib(self):
        path = self._calib_var.get().strip()
        if not os.path.isfile(path):
            messagebox.showerror("Error", f"File not found:\n{path}")
            return
        try:
            T, K = load_calibration(path)
            self._T_cam2base = T
            self._calib_path = path
            if K is not None:
                self._K_calib = K
            t = T[:3, 3]
            self._log_msg(f"Loaded calibration: {path}")
            self._log_msg(f"  t_cam2base = [{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}] mm")
            if _SCIPY_OK:
                e = Rotation.from_matrix(T[:3, :3]).as_euler("ZYX", degrees=True)
                self._log_msg(f"  R ZYX = [{e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f}] deg")
            self.root.after(0, lambda: self._calib_dot.config(
                text="● Calib: loaded", fg="green"))
        except Exception as ex:
            messagebox.showerror("Load error", str(ex))

    # --------------------------------------------------------- robot control
    def _connect_robot(self):
        ip = self._ip_var.get().strip()
        if not ip:
            messagebox.showerror("Error", "Enter a robot IP address")
            return

        def _thread():
            try:
                driver_name = self._robot_type_var.get()
                self._log_msg(f"Connecting to robot @ {ip} via {driver_name!r}...")
                robot = create_robot(driver_name, ip)
                self._robot = robot
                self._robot_connected = True
                self._log_msg(f"Robot connected: {ip}")
            except Exception as ex:
                self._log_msg(f"[ERROR] Robot connection: {ex}")

        threading.Thread(target=_thread, daemon=True).start()

    def _disconnect_robot(self):
        if self._robot:
            try:
                self._robot.close()
            except Exception:
                pass
            self._robot = None
        self._robot_connected = False
        self._connect_btn.config(text="Connect Robot", command=self._connect_robot,
                                 bg="#2e7d32")
        self._log_msg("Robot disconnected.")

    def _enable_robot(self):
        if not self._robot:
            self._log_msg("[WARN] Robot not connected"); return
        try:
            self._robot.enable()
            self._log_msg("Enable sent.")
        except Exception as ex:
            self._log_msg(f"[ERROR] Enable: {ex}")

    def _clear_error(self):
        if not self._robot:
            self._log_msg("[WARN] Robot not connected"); return
        try:
            self._robot.clear_error()
            self._log_msg("ClearError sent.")
        except Exception as ex:
            self._log_msg(f"[ERROR] ClearError: {ex}")

    def _set_home(self):
        if not self._robot:
            self._log_msg("[WARN] Robot not connected"); return
        try:
            pose = self._robot.get_pose()
            self._home_pose = list(pose)
            self._home_btn.config(state="normal")
            self._log_msg(f"Home set: {[f'{v:.1f}' for v in self._home_pose]}")
        except Exception as ex:
            self._log_msg(f"[ERROR] Set home: {ex}")

    def _go_home(self):
        if not self._robot:
            self._log_msg("[WARN] Robot not connected"); return
        if not self._home_pose:
            self._log_msg("[WARN] Home not set — click 'Set Home' first"); return
        def _thread():
            try:
                self._log_msg(f"Going home: {[f'{v:.1f}' for v in self._home_pose]}")
                cmd = self._robot.move_linear(*self._home_pose)
                self._robot.wait_motion(cmd)
                self._log_msg("Home reached.")
            except Exception as ex:
                self._log_msg(f"[ERROR] Home: {ex}")
        threading.Thread(target=_thread, daemon=True).start()

    # --------------------------------------------------------- marker helper
    def _gen_marker(self):
        img  = generate_aruco_image(400)
        path = CALIB_DIR / "aruco_id0_4x4_50.png"
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        self._log_msg(f"Marker saved: {path}  (print at ~10×10 cm)")

    # ------------------------------------------------------- camera pipeline
    def _start_camera_loop(self):
        """Try to open Orbbec camera in a background thread."""
        def _init():
            try:
                cam = OrbbecCamera()
                cam.start()
                self._cam = cam
                self._cam_connected = True
                self._log_msg("Camera started.")
            except Exception as ex:
                self._cam_connected = False
                self._log_msg(f"[WARN] Camera not available: {ex}")
        threading.Thread(target=_init, daemon=True).start()
        self.root.after(50, self._ui_loop)

    def _ui_loop(self):
        """Main Tkinter tick — fetch camera frame, run detection, update UI."""
        try:
            self._tick()
        except Exception as ex:
            self._log_msg(f"[ERROR] UI loop: {ex}")
        self.root.after(40, self._ui_loop)   # ~25 fps

    def _tick(self):
        # Update status dots from state flags (safe — main thread only)
        cam_state = "ok" if self._cam_connected else "off"
        if cam_state != self._cam_dot_state:
            self._cam_dot_state = cam_state
            if cam_state == "ok":
                self._cam_dot.config(text="● Camera: OK", fg="green")
            else:
                self._cam_dot.config(text="● Camera: disconnected", fg="red")

        robot_state = "ok" if self._robot_connected else "off"
        if robot_state != self._robot_dot_state:
            self._robot_dot_state = robot_state
            if robot_state == "ok":
                ip = self._ip_var.get().strip()
                self._robot_dot.config(text=f"● Robot: {ip}", fg="green")
                self._connect_btn.config(text="Disconnect",
                                         command=self._disconnect_robot, bg="#c62828")
            else:
                self._robot_dot.config(text="● Robot: disconnected", fg="red")

        if self._cam is None:
            return
        rgb, depth_m, intr = self._cam.get_latest()
        if rgb is None:
            return

        # Use calibration K if camera didn't provide good intrinsics
        if intr is not None:
            fx, fy, cx, cy = intr
        elif self._K_calib is not None:
            K = self._K_calib
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        else:
            fx, fy, cx, cy = _DEFAULT_FX, _DEFAULT_FY, _DEFAULT_CX, _DEFAULT_CY

        # Resize for display
        disp_rgb = cv2.resize(rgb, (_PREVIEW_W, _PREVIEW_H))

        # Detect ArUco
        det = detect_aruco(disp_rgb)
        self._last_det = det

        # Compute prediction if all data available
        pred_robot_mm  = None
        cam_point_mm   = None
        pred_px        = None

        if det is not None and self._T_cam2base is not None and depth_m is not None:
            u_orig = det["centroid_px"][0] * rgb.shape[1] / _PREVIEW_W
            v_orig = det["centroid_px"][1] * rgb.shape[0] / _PREVIEW_H
            p_cam = unproject_pixel(u_orig, v_orig, depth_m, fx, fy, cx, cy)
            if p_cam is not None:
                cam_point_mm = p_cam
                T_corr = self._get_correction_T()
                pred_robot_mm = apply_T_cam2base(p_cam, self._T_cam2base, T_corr)
                self._last_pred_mm = pred_robot_mm

        # Draw overlay
        overlay = draw_aruco_overlay(disp_rgb, det, pred_px)
        self._update_canvas(overlay)

        # Update detection info label
        if det is None:
            info = "Marker: NOT DETECTED\nCamera (mm): —\nRobot predicted (mm): —"
        else:
            u, v = det["centroid_px"]
            if cam_point_mm is not None:
                cx_s = f"[{cam_point_mm[0]:.1f}, {cam_point_mm[1]:.1f}, {cam_point_mm[2]:.1f}]"
            else:
                cx_s = "no depth"
            if pred_robot_mm is not None:
                rb_s = f"[{pred_robot_mm[0]:.1f}, {pred_robot_mm[1]:.1f}, {pred_robot_mm[2]:.1f}]"
            else:
                rb_s = "—  (load calibration)"
            info = (f"Marker @ px ({u:.0f}, {v:.0f})\n"
                    f"Camera (mm):  {cx_s}\n"
                    f"Robot pred (mm): {rb_s}")
        self._det_info.config(text=info)

    def _update_canvas(self, rgb: np.ndarray):
        if not _PIL_OK:
            return
        img = PILImage.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)
        self._canvas.create_image(0, 0, anchor="nw", image=photo)
        self._canvas._photo = photo  # keep reference

    # ------------------------------------------------------ correction utils
    def _get_correction_T(self) -> np.ndarray:
        return build_correction_matrix(
            self._cv["dx"].get(),  self._cv["dy"].get(),  self._cv["dz"].get(),
            self._cv["drx"].get(), self._cv["dry"].get(), self._cv["drz"].get()
        )

    def _on_slider(self, _key):
        # Nothing extra needed — _tick() reads sliders every frame
        pass

    def _reset_correction(self):
        for v in self._cv.values():
            v.set(0.0)
        self._log_msg("Correction reset to zero.")

    def _autofill_correction(self):
        """Set correction sliders to negative mean error across all test results."""
        if not self._test_results:
            messagebox.showinfo("No data", "Run some tests first to accumulate errors.")
            return
        errs = np.array([[r["err_x"], r["err_y"], r["err_z"]]
                         for r in self._test_results])
        mean_err = errs.mean(axis=0)
        # Correction = negate mean error (shift robot coords to compensate)
        self._cv["dx"].set(round(-float(mean_err[0]), 1))
        self._cv["dy"].set(round(-float(mean_err[1]), 1))
        self._cv["dz"].set(round(-float(mean_err[2]), 1))
        self._log_msg(f"Auto-fill correction: dx={-mean_err[0]:.1f}, "
                      f"dy={-mean_err[1]:.1f}, dz={-mean_err[2]:.1f} mm")
        self._log_msg("  (rotation correction not auto-filled — adjust manually if needed)")

    def _save_correction(self):
        if self._T_cam2base is None:
            messagebox.showerror("Error", "Load a calibration file first."); return
        T_corr = self._get_correction_T()
        T_corrected = T_corr @ self._T_cam2base
        try:
            npz_path, json_path = save_corrected_calibration(
                self._calib_path, T_corrected, T_corr, CALIB_DIR)
            self._log_msg("Saved corrected calibration:")
            self._log_msg(f"  {npz_path}")
            self._log_msg(f"  {json_path}")

            # Ask whether to overwrite the main calibration so all tools pick it up
            overwrite = messagebox.askyesno(
                "Overwrite main calibration?",
                f"Overwrite the main calibration file used by the pipeline?\n\n"
                f"  {self._calib_path}\n\n"
                "Yes → pipeline will use the corrected calibration immediately.\n"
                "No  → only the timestamped backup is saved.")
            if overwrite:
                import shutil
                shutil.copy2(str(npz_path), self._calib_path)
                # Also update the .json sidecar (same name, .json extension)
                json_main = str(Path(self._calib_path).with_suffix(".json"))
                shutil.copy2(str(json_path), json_main)
                self._log_msg(f"Main calibration overwritten: {self._calib_path}")
                # Reload so the tester itself uses the corrected matrix
                self._T_cam2base = T_corrected
        except Exception as ex:
            messagebox.showerror("Save error", str(ex))

    # ---------------------------------------------------------------- Test A
    def _test_a_move(self):
        """Move robot TCP to camera-predicted marker position (+Z offset)."""
        if self._robot is None:
            messagebox.showerror("Error", "Robot not connected."); return
        if self._T_cam2base is None:
            messagebox.showerror("Error", "Calibration not loaded."); return
        if self._last_pred_mm is None:
            messagebox.showinfo("No detection", "No marker detected — aim camera at marker."); return
        if self._worker_busy:
            return

        pred = self._last_pred_mm.copy()
        # Keep current robot orientation but move to predicted XY, Z + offset
        try:
            current_pose = self._robot.get_pose()
            rx, ry, rz   = current_pose[3], current_pose[4], current_pose[5]
        except Exception:
            rx, ry, rz = 0.0, 0.0, 0.0

        target = (float(pred[0]), float(pred[1]),
                  float(pred[2]) + APPROACH_OFFSET_MM,
                  rx, ry, rz)

        self._log_msg(f"[Test A] Predicted marker @ [{pred[0]:.1f}, {pred[1]:.1f}, {pred[2]:.1f}] mm")
        self._log_msg(f"[Test A] Moving to approach pose: {[f'{v:.1f}' for v in target]}")

        def _thread():
            self._worker_busy = True
            try:
                cmd = self._robot.move_linear(*target)
                self._robot.wait_motion(cmd)
                actual = self._robot.get_pose()
                # Error between predicted XY and actual XY (Z offset removed)
                err_x = actual[0] - pred[0]
                err_y = actual[1] - pred[1]
                err_z = actual[2] - (pred[2] + APPROACH_OFFSET_MM)
                dist  = float(np.linalg.norm([err_x, err_y]))
                self.root.after(0, lambda: self._log_msg(
                    f"[Test A] Actual pose: [{actual[0]:.1f}, {actual[1]:.1f}, {actual[2]:.1f}] mm"))
                self.root.after(0, lambda: self._log_msg(
                    f"[Test A] XY error: [{err_x:.1f}, {err_y:.1f}] mm  |XY dist|={dist:.1f} mm"))
                result = {
                    "mode": "A",
                    "pred_x": pred[0], "pred_y": pred[1], "pred_z": pred[2],
                    "actual_x": actual[0], "actual_y": actual[1], "actual_z": actual[2],
                    "err_x": err_x, "err_y": err_y, "err_z": err_z,
                    "dist": dist,
                }
                self.root.after(0, lambda: self._add_result(result))
            except Exception as ex:
                self.root.after(0, lambda: self._log_msg(f"[ERROR] Test A: {ex}"))
            finally:
                self._worker_busy = False

        threading.Thread(target=_thread, daemon=True).start()

    # ---------------------------------------------------------------- Test A2
    def _test_a2_descend(self):
        """Approach + descend to the predicted marker position, record error, then retreat."""
        if self._robot is None:
            messagebox.showerror("Error", "Robot not connected."); return
        if self._T_cam2base is None:
            messagebox.showerror("Error", "Calibration not loaded."); return
        if self._last_pred_mm is None:
            messagebox.showinfo("No detection", "No marker detected — aim camera at marker."); return
        if self._worker_busy:
            return

        pred = self._last_pred_mm.copy()
        try:
            current_pose = self._robot.get_pose()
            rx, ry, rz   = current_pose[3], current_pose[4], current_pose[5]
        except Exception:
            rx, ry, rz = 0.0, 0.0, 0.0

        approach = (float(pred[0]), float(pred[1]),
                    float(pred[2]) + APPROACH_OFFSET_MM,
                    rx, ry, rz)
        target   = (float(pred[0]), float(pred[1]), float(pred[2]),
                    rx, ry, rz)

        self._log_msg(f"[A2] Predicted @ [{pred[0]:.1f}, {pred[1]:.1f}, {pred[2]:.1f}] mm")

        def _thread():
            self._worker_busy = True
            try:
                # 1 — approach
                self._log_msg(f"[A2] → approach Z+{APPROACH_OFFSET_MM}mm")
                cmd = self._robot.move_linear(*approach)
                self._robot.wait_motion(cmd)
                # 2 — descend
                self._log_msg(f"[A2] → descend to marker")
                cmd = self._robot.move_linear(*target)
                self._robot.wait_motion(cmd)
                # 3 — record
                actual = self._robot.get_pose()
                err_x = actual[0] - pred[0]
                err_y = actual[1] - pred[1]
                err_z = actual[2] - pred[2]
                dist  = float(np.linalg.norm([err_x, err_y, err_z]))
                self.root.after(0, lambda: self._log_msg(
                    f"[A2] Actual: [{actual[0]:.1f}, {actual[1]:.1f}, {actual[2]:.1f}] mm  "
                    f"err=[{err_x:+.1f}, {err_y:+.1f}, {err_z:+.1f}]  |3D|={dist:.1f} mm"))
                result = {
                    "mode": "A2",
                    "pred_x": pred[0], "pred_y": pred[1], "pred_z": pred[2],
                    "actual_x": actual[0], "actual_y": actual[1], "actual_z": actual[2],
                    "err_x": err_x, "err_y": err_y, "err_z": err_z,
                    "dist": dist,
                }
                self.root.after(0, lambda: self._add_result(result))
                # 4 — retreat
                self._log_msg(f"[A2] → retreat")
                cmd = self._robot.move_linear(*approach)
                self._robot.wait_motion(cmd)
            except Exception as ex:
                self.root.after(0, lambda: self._log_msg(f"[ERROR] Test A2: {ex}"))
            finally:
                self._worker_busy = False

        threading.Thread(target=_thread, daemon=True).start()

    # ---------------------------------------------------------------- Test B
    def _test_b_set_gt(self):
        """
        Record current robot TCP pose as ground truth for the marker position.
        Then compare with camera prediction.
        (User must have jogged TCP to the marker beforehand.)
        """
        if self._robot is None:
            messagebox.showerror("Error", "Robot not connected."); return
        if self._T_cam2base is None:
            messagebox.showerror("Error", "Calibration not loaded."); return
        if self._last_pred_mm is None:
            messagebox.showinfo("No detection", "No marker detected — aim camera at marker."); return

        try:
            gt = self._robot.get_pose()  # (x, y, z, rx, ry, rz)
        except Exception as ex:
            self._log_msg(f"[ERROR] GetPose: {ex}"); return

        gt_xyz   = np.array([gt[0], gt[1], gt[2]])
        pred_xyz = self._last_pred_mm.copy()

        err_x = pred_xyz[0] - gt_xyz[0]
        err_y = pred_xyz[1] - gt_xyz[1]
        err_z = pred_xyz[2] - gt_xyz[2]
        dist  = float(np.linalg.norm([err_x, err_y, err_z]))

        self._log_msg(f"[Test B] Ground truth (TCP): [{gt_xyz[0]:.1f}, {gt_xyz[1]:.1f}, {gt_xyz[2]:.1f}] mm")
        self._log_msg(f"[Test B] Prediction:         [{pred_xyz[0]:.1f}, {pred_xyz[1]:.1f}, {pred_xyz[2]:.1f}] mm")
        self._log_msg(f"[Test B] Error (pred−GT):    [{err_x:.1f}, {err_y:.1f}, {err_z:.1f}] mm  |3D dist|={dist:.1f} mm")

        result = {
            "mode": "B",
            "pred_x": pred_xyz[0], "pred_y": pred_xyz[1], "pred_z": pred_xyz[2],
            "actual_x": gt_xyz[0], "actual_y": gt_xyz[1], "actual_z": gt_xyz[2],
            "err_x": err_x, "err_y": err_y, "err_z": err_z,
            "dist": dist,
        }
        self._add_result(result)

    # ------------------------------------------------------------- results
    def _add_result(self, r: dict):
        self._test_results.append(r)
        n   = len(self._test_results)
        tag = f"{'A' if r['mode'] == 'A' else 'B'}"
        row = (
            f"#{n}{tag}",
            f"{r['pred_x']:.1f}",   f"{r['pred_y']:.1f}",   f"{r['pred_z']:.1f}",
            f"{r['actual_x']:.1f}", f"{r['actual_y']:.1f}", f"{r['actual_z']:.1f}",
            f"{r['err_x']:+.1f}",   f"{r['err_y']:+.1f}",   f"{r['err_z']:+.1f}",
            f"{r['dist']:.1f}",
        )
        self._tree.insert("", tk.END, values=row)
        self._tree.yview_moveto(1.0)
        self._update_stats()

    def _update_stats(self):
        if not self._test_results:
            self._stats_lbl.config(text="Mean error: —")
            return
        errs = np.array([[r["err_x"], r["err_y"], r["err_z"]]
                         for r in self._test_results])
        mean = errs.mean(axis=0)
        std  = errs.std(axis=0)
        dists = np.linalg.norm(errs, axis=1)
        self._stats_lbl.config(
            text=(f"n={len(errs)}  |  Mean error: "
                  f"X={mean[0]:+.1f}  Y={mean[1]:+.1f}  Z={mean[2]:+.1f} mm  |  "
                  f"Mean 3D dist: {dists.mean():.1f} mm  |  "
                  f"Std: [{std[0]:.1f}, {std[1]:.1f}, {std[2]:.1f}]"))

    def _clear_results(self):
        self._test_results.clear()
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._stats_lbl.config(text="Mean error: —")
        self._log_msg("Test log cleared.")

    # ---------------------------------------------------------------- close
    def on_close(self):
        if self._cam:
            try: self._cam.stop()
            except Exception: pass
        if self._robot:
            try: self._robot.close()
            except Exception: pass
        self.root.destroy()


# ===========================================================================
# Entry point
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Hand-eye calibration quality tester")
    p.add_argument("--robot-ip", default=ROBOT_IP_DEFAULT,
                   help=f"Dobot IP (default: {ROBOT_IP_DEFAULT})")
    p.add_argument("--calib",    default=CALIB_FILE_DEFAULT,
                   help=f"Path to hand_eye_calib.npz (default: {CALIB_FILE_DEFAULT})")
    return p.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    app  = CalibTesterApp(root, args)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    # Auto-load calibration if file exists at startup
    if os.path.isfile(args.calib):
        root.after(500, app._load_calib)

    root.mainloop()


if __name__ == "__main__":
    main()
