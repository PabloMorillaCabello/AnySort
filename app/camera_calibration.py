#!/usr/bin/env python3
"""
camera_calibration.py
=====================
Intrinsic calibration for the Orbbec Gemini 2 using the same ChArUco board
that is used in hand_eye_calibration.py.

Saves camera_intrinsics.npz and camera_intrinsics.json to OUTPUT_DIR.
hand_eye_calibration.py automatically loads these on startup.

Usage:
    python3 /ros2_ws/scripts/TEST/camera_calibration.py

Workflow:
    1. Hold the ChArUco board at different distances/angles in front of the camera.
    2. Click "Capture Frame" (or enable Auto-Capture) when the board is well detected.
       Aim for 20+ frames covering the full field of view.
    3. Click "Calibrate" once enough frames are collected.
    4. Check the reprojection error — should be < 0.5 px for good results.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orbbec_quiet  # noqa: E402 — suppress OrbbecSDK C-level stderr spam

import gc
import json
import queue
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

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    _ORBBEC_OK = True
except ImportError:
    _ORBBEC_OK = False

# ===========================================================================
# Constants
# ===========================================================================
OUTPUT_DIR   = Path("/ros2_ws/data/calibration")
_PREVIEW_W   = 640
_PREVIEW_H   = 480
MIN_FRAMES   = 15       # warn if below this
GOOD_FRAMES  = 25       # recommended
MIN_CORNERS  = 6        # corners needed per frame to accept it
AUTO_INTERVAL = 1.5     # seconds between auto-captures

_DEFAULT_FX, _DEFAULT_FY = 905.0, 905.0
_DEFAULT_CX, _DEFAULT_CY = 640.0, 360.0

# ===========================================================================
# Camera
# ===========================================================================
def _to_rgb(color_frame) -> Optional[np.ndarray]:
    h, w = color_frame.get_height(), color_frame.get_width()
    raw = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    fmt = str(color_frame.get_format()).upper()
    if "RGB" in fmt and raw.size == h * w * 3:
        return raw.reshape(h, w, 3).copy()
    if "BGR" in fmt and raw.size == h * w * 3:
        return raw.reshape(h, w, 3)[:, :, ::-1].copy()
    if "MJPG" in fmt or "JPEG" in fmt:
        try:
            return np.array(PILImage.open(BytesIO(raw.tobytes())).convert("RGB"))
        except Exception:
            decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            return decoded[:, :, ::-1] if decoded is not None else None
    if "YUYV" in fmt and raw.size == h * w * 2:
        return cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUY2)
    if raw.size == h * w * 3:
        return raw.reshape(h, w, 3).copy()
    return None


class OrbbecCamera:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_rgb = None
        self._image_size = None   # (w, h) of the actual stream
        self._K_sdk = None        # intrinsics from SDK (may be None)
        self._running = False

    def start(self):
        pipeline = Pipeline()
        cfg = Config()
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        cfg.enable_stream(color_profile)
        pipeline.start(cfg)
        self._pipeline = pipeline
        self._image_size = (color_profile.get_width(), color_profile.get_height())
        # Read SDK intrinsics as initial estimate (will be overwritten by calibration)
        try:
            cp = pipeline.get_camera_param()
            for attr in ("rgb_intrinsic", "color_intrinsic"):
                ci = getattr(cp, attr, None)
                if ci:
                    fx, fy = float(ci.fx), float(ci.fy)
                    cx_, cy_ = float(ci.cx), float(ci.cy)
                    if all(v > 0 for v in [fx, fy, cx_, cy_]):
                        self._K_sdk = np.array(
                            [[fx, 0, cx_], [0, fy, cy_], [0, 0, 1]], np.float64)
                        break
        except Exception:
            pass
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
                if self._image_size is None:
                    h, w = rgb.shape[:2]
                    self._image_size = (w, h)

    def get_latest_rgb(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_rgb is None else self._latest_rgb.copy()

    def image_size(self) -> Optional[Tuple[int, int]]:
        with self._lock:
            return self._image_size

    def sdk_K(self) -> Optional[np.ndarray]:
        return self._K_sdk

    def stop(self):
        self._running = False
        try:
            self._pipeline.stop()
        except Exception:
            pass


# ===========================================================================
# ChArUco helpers  (mirror of hand_eye_calibration.py)
# ===========================================================================
def make_board(sq_x, sq_y, sq_len, mk_len,
               dict_type=cv2.aruco.DICT_6X6_250):
    d = cv2.aruco.getPredefinedDictionary(dict_type)
    try:
        b = cv2.aruco.CharucoBoard((sq_x, sq_y), sq_len, mk_len, d)
    except TypeError:
        b = cv2.aruco.CharucoBoard_create(sq_x, sq_y, sq_len, mk_len, d)
    return d, b


def detect_charuco(frame_rgb, aruco_dict, board, detector
                   ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, int]:
    """Returns (charuco_corners, charuco_ids, annotated_bgr, num_corners).
    Uses zero distortion — intrinsics are not needed for corner detection."""
    bgr  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    ann  = bgr.copy()

    if detector is not None:
        cc, ci, mc, mi = detector.detectBoard(gray)
        num = len(ci) if ci is not None else 0
        if mi is not None and len(mi) > 0:
            cv2.aruco.drawDetectedMarkers(ann, mc, mi)
        if num >= MIN_CORNERS:
            cv2.aruco.drawDetectedCornersCharuco(ann, cc, ci)
        return cc, ci, ann, num
    else:
        params = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        if ids is None:
            return None, None, ann, 0
        cv2.aruco.drawDetectedMarkers(ann, corners, ids)
        n, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if n >= MIN_CORNERS:
            cv2.aruco.drawDetectedCornersCharuco(ann, cc, ci)
        return (cc, ci, ann, n) if n >= MIN_CORNERS else (None, None, ann, n)


# ===========================================================================
# Calibration
# ===========================================================================
def run_calibration(all_corners: list, all_ids: list,
                    board, image_size: Tuple[int, int],
                    K_init: Optional[np.ndarray] = None
                    ) -> dict:
    """
    Run cv2.calibrateCamera with ChArUco object/image points.
    Works with OpenCV 4.7+ (calibrateCameraCharuco was removed in 4.8).
    image_size: (width, height)
    """
    obj_points = []
    img_points = []
    valid_idx  = []
    for i, (corners, ids) in enumerate(zip(all_corners, all_ids)):
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < 4:
            continue
        obj_points.append(obj_pts.astype(np.float32))
        img_points.append(img_pts.astype(np.float32))
        valid_idx.append(i)

    if len(obj_points) < 4:
        raise RuntimeError(
            f"Only {len(obj_points)} usable frames after matchImagePoints — need at least 4.")

    flags = 0
    K_guess = K_init.copy() if K_init is not None else None
    if K_guess is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS

    rms, K, d, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, K_guess, None, flags=flags)

    # Per-frame reprojection errors
    per_frame = []
    for j, (obj_pts, img_pts, rvec, tvec) in enumerate(
            zip(obj_points, img_points, rvecs, tvecs)):
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, d)
        err = float(np.sqrt(np.mean((img_pts - proj.reshape(-1, 1, 2)) ** 2)))
        per_frame.append((valid_idx[j], err))

    return {
        "rms": float(rms),
        "K": K,
        "d": d,
        "image_size": image_size,
        "n_frames": len(obj_points),
        "per_frame_errors": per_frame,
    }


def save_intrinsics(result: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    np.savez(str(output / "camera_intrinsics.npz"),
             camera_matrix=result["K"],
             dist_coeffs=result["d"],
             image_size=np.array(result["image_size"]),
             rms=np.array(result["rms"]))
    np.savez(str(output / f"camera_intrinsics_{ts}.npz"),
             camera_matrix=result["K"],
             dist_coeffs=result["d"],
             image_size=np.array(result["image_size"]),
             rms=np.array(result["rms"]))
    json_data = {
        "timestamp": ts,
        "rms_px": result["rms"],
        "n_frames": result["n_frames"],
        "image_size_wh": list(result["image_size"]),
        "camera_matrix": result["K"].tolist(),
        "dist_coeffs": result["d"].ravel().tolist(),
    }
    with open(output / "camera_intrinsics.json", "w") as f:
        json.dump(json_data, f, indent=2)


# ===========================================================================
# Tkinter Application
# ===========================================================================
class CameraCalibApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self._camera: Optional[OrbbecCamera] = None
        self._board = None
        self._aruco_dict = None
        self._detector = None

        self._all_corners: List[np.ndarray] = []
        self._all_ids: List[np.ndarray] = []

        self._auto_var   = tk.BooleanVar(value=False)
        self._last_auto  = 0.0
        self._cb_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._rebuild_board()
        self._start_camera()
        self._schedule_preview()
        self._schedule_flush()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.root.title("Camera Intrinsic Calibration  —  Orbbec Gemini 2")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(False, False)

        left = tk.Frame(self.root, bg="#1e1e1e")
        left.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)

        self._canvas = tk.Canvas(left, width=_PREVIEW_W, height=_PREVIEW_H,
                                  bg="#111", highlightthickness=1,
                                  highlightbackground="#444")
        self._canvas.pack()
        self._canvas.create_image(0, 0, anchor="nw")
        self._no_cam_lbl = self._canvas.create_text(
            _PREVIEW_W // 2, _PREVIEW_H // 2,
            text="Waiting for camera…", fill="#555", font=("Helvetica", 14))

        self._det_lbl = tk.Label(left, text="Detection: —",
                                  bg="#1e1e1e", fg="#666", font=("Courier", 9))
        self._det_lbl.pack(anchor="w", padx=4, pady=(4, 0))

        # ── Right panel ────────────────────────────────────────────────
        right = tk.Frame(self.root, bg="#2d2d2d", width=300)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        right.grid_propagate(False)

        tk.Label(right, text="Camera Calibration", bg="#2d2d2d", fg="#61afef",
                 font=("Helvetica", 13, "bold")).pack(pady=(14, 0))
        tk.Label(right, text="ChArUco board  ·  Orbbec Gemini 2",
                 bg="#2d2d2d", fg="#555", font=("Helvetica", 8)).pack(pady=(0, 8))
        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=2)

        # Board params
        self._lbl("ChArUco Board Parameters", right)
        r1 = tk.Frame(right, bg="#2d2d2d"); r1.pack(fill="x", padx=12, pady=2)
        tk.Label(r1, text="Squares X:", bg="#2d2d2d", fg="#ccc",
                 font=("Helvetica", 9), width=11, anchor="w").pack(side="left")
        self._sq_x = tk.StringVar(value="5")
        self._entry(r1, self._sq_x, 5)
        tk.Label(r1, text="Y:", bg="#2d2d2d", fg="#ccc",
                 font=("Helvetica", 9)).pack(side="left")
        self._sq_y = tk.StringVar(value="7")
        self._entry(r1, self._sq_y, 5)

        r2 = tk.Frame(right, bg="#2d2d2d"); r2.pack(fill="x", padx=12, pady=2)
        tk.Label(r2, text="Square (m):", bg="#2d2d2d", fg="#ccc",
                 font=("Helvetica", 9), width=11, anchor="w").pack(side="left")
        self._sq_len = tk.StringVar(value="0.030")
        self._entry(r2, self._sq_len, 8)
        tk.Label(r2, text="Marker (m):", bg="#2d2d2d", fg="#ccc",
                 font=("Helvetica", 9)).pack(side="left")
        self._mk_len = tk.StringVar(value="0.022")
        self._entry(r2, self._mk_len, 8)

        tk.Button(right, text="↺  Rebuild Board",
                  command=self._rebuild_board,
                  bg="#4a5568", fg="white", relief="flat",
                  font=("Helvetica", 9)).pack(fill="x", padx=12, pady=(4, 8))
        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=2)

        # Capture controls
        self._lbl("Capture", right)
        self._frame_lbl = tk.Label(right, text="Frames captured: 0",
                                    bg="#2d2d2d", fg="#98c379",
                                    font=("Helvetica", 11, "bold"))
        self._frame_lbl.pack(pady=(4, 2))

        tk.Checkbutton(right, text="  Auto-capture every 1.5 s when board detected",
                       variable=self._auto_var,
                       bg="#2d2d2d", fg="#ccc", selectcolor="#3a3a3a",
                       font=("Helvetica", 9), activebackground="#2d2d2d").pack(
                           anchor="w", padx=12, pady=2)

        self._capture_btn = tk.Button(right, text="📷  Capture Frame",
                                       command=self._on_capture,
                                       bg="#2e7d32", fg="white", relief="flat",
                                       font=("Helvetica", 10, "bold"), height=2)
        self._capture_btn.pack(fill="x", padx=12, pady=(4, 2))

        self._clear_btn = tk.Button(right, text="✕  Clear Frames",
                                     command=self._on_clear,
                                     bg="#4a5568", fg="white", relief="flat",
                                     font=("Helvetica", 9))
        self._clear_btn.pack(fill="x", padx=12, pady=2)
        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # Calibrate
        self._lbl("Calibration", right)
        self._calib_btn = tk.Button(right, text="✓  Calibrate",
                                     command=self._on_calibrate,
                                     bg="#1565c0", fg="white", relief="flat",
                                     font=("Helvetica", 10, "bold"), height=2,
                                     state="disabled")
        self._calib_btn.pack(fill="x", padx=12, pady=(4, 2))
        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # Status
        self._status_var = tk.StringVar(value="Waiting for camera…")
        tk.Label(right, textvariable=self._status_var, bg="#2d2d2d", fg="#e5c07b",
                 font=("Helvetica", 9), wraplength=270, justify="left").pack(
                     anchor="w", padx=12, pady=(0, 4))

        # Log
        self._log_box = scrolledtext.ScrolledText(
            right, height=10, bg="#1e1e1e", fg="#abb2bf",
            font=("Courier", 8), state="disabled", relief="flat")
        self._log_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _lbl(self, text, parent):
        tk.Label(parent, text=text, bg=str(parent.cget("bg")), fg="#e5c07b",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(6, 2))

    def _entry(self, parent, var, width):
        tk.Entry(parent, textvariable=var, width=width,
                 bg="#3a3a3a", fg="white", insertbackground="white",
                 relief="flat", font=("Helvetica", 9)).pack(side="left", padx=(0, 8))

    # ------------------------------------------------------------------
    # Board
    # ------------------------------------------------------------------
    def _rebuild_board(self):
        try:
            sq_x = int(self._sq_x.get())
            sq_y = int(self._sq_y.get())
            sq_len = float(self._sq_len.get())
            mk_len = float(self._mk_len.get())
        except Exception:
            return
        self._aruco_dict, self._board = make_board(sq_x, sq_y, sq_len, mk_len)
        if hasattr(cv2.aruco, "CharucoDetector"):
            self._detector = cv2.aruco.CharucoDetector(self._board)
        else:
            self._detector = None
        self._log(f"Board rebuilt: {sq_x}×{sq_y}, sq={sq_len}m, mk={mk_len}m")

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def _start_camera(self):
        if not _ORBBEC_OK:
            self._log("[Camera] pyorbbecsdk not available")
            return
        try:
            self._camera = OrbbecCamera()
            self._camera.start()
            sz = self._camera.image_size()
            K = self._camera.sdk_K()
            self._log(f"[Camera] Started  size={sz}  SDK K.fx={K[0,0]:.1f}" if K is not None
                      else f"[Camera] Started  size={sz}  (no SDK intrinsics)")
            self._status_var.set("Camera ready. Capture frames from many angles.")
        except Exception as e:
            self._log(f"[Camera] ERROR: {e}")
            self._camera = None

    # ------------------------------------------------------------------
    # Preview loop
    # ------------------------------------------------------------------
    def _schedule_preview(self):
        self._update_preview()
        self.root.after(80, self._schedule_preview)

    def _update_preview(self):
        if not _PIL_OK or self._camera is None:
            return
        rgb = self._camera.get_latest_rgb()
        if rgb is None:
            return
        self._canvas.itemconfig(self._no_cam_lbl, state="hidden")

        cc, ci, bgr_ann, num = detect_charuco(
            rgb, self._aruco_dict, self._board, self._detector)

        if num >= MIN_CORNERS:
            color = "#98c379"
            det_txt = f"Detection: {num} corners  ✓"
        else:
            color = "#e06c75"
            det_txt = f"Detection: {num} corners (need {MIN_CORNERS})"
        self._det_lbl.config(text=det_txt, fg=color)

        # Auto-capture
        if (self._auto_var.get() and cc is not None
                and time.time() - self._last_auto >= AUTO_INTERVAL):
            self._do_capture(cc, ci)

        # Show preview
        display = bgr_ann[:, :, ::-1]
        h, w = display.shape[:2]
        scale = min(_PREVIEW_W / w, _PREVIEW_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(display, (nw, nh))
        img = PILImage.fromarray(resized)
        self._tk_img = ImageTk.PhotoImage(img)
        self._canvas.itemconfig(1, image=self._tk_img)

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def _on_capture(self):
        if self._camera is None:
            return
        rgb = self._camera.get_latest_rgb()
        if rgb is None:
            self._log("No frame available"); return
        cc, ci, _, num = detect_charuco(
            rgb, self._aruco_dict, self._board, self._detector)
        if cc is None or num < MIN_CORNERS:
            self._log(f"Detection failed ({num} corners) — reposition board")
            self._status_var.set(f"Board not detected ({num} corners). Try again.")
            return
        self._do_capture(cc, ci)

    def _do_capture(self, cc, ci):
        self._all_corners.append(cc)
        self._all_ids.append(ci)
        self._last_auto = time.time()
        n = len(self._all_corners)
        self._frame_lbl.config(text=f"Frames captured: {n}")
        self._calib_btn.config(
            state="normal" if n >= 4 else "disabled")
        quality = "GOOD" if n >= GOOD_FRAMES else ("OK" if n >= MIN_FRAMES else "need more")
        self._log(f"Frame {n} captured  [{quality}]")
        if n >= MIN_FRAMES:
            self._status_var.set(f"{n} frames — ready to calibrate (more is better).")

    def _on_clear(self):
        self._all_corners.clear()
        self._all_ids.clear()
        self._frame_lbl.config(text="Frames captured: 0")
        self._calib_btn.config(state="disabled")
        self._status_var.set("Frames cleared.")
        self._log("All frames cleared.")

    # ------------------------------------------------------------------
    # Calibrate
    # ------------------------------------------------------------------
    def _on_calibrate(self):
        if len(self._all_corners) < 4:
            messagebox.showwarning("Not enough frames",
                                   "Need at least 4 frames to calibrate.")
            return
        if len(self._all_corners) < MIN_FRAMES:
            if not messagebox.askyesno(
                    "Few frames",
                    f"Only {len(self._all_corners)} frames captured.\n"
                    f"Recommend {MIN_FRAMES}+. Continue anyway?"):
                return
        self._calib_btn.config(state="disabled", text="Calibrating…")
        self._status_var.set("Running calibration…")
        threading.Thread(target=self._calibrate_worker, daemon=True).start()

    def _calibrate_worker(self):
        try:
            sz = self._camera.image_size() if self._camera else None
            if sz is None:
                # Infer from first captured frame
                rgb = self._camera.get_latest_rgb() if self._camera else None
                if rgb is not None:
                    h, w = rgb.shape[:2]
                    sz = (w, h)
                else:
                    sz = (1280, 720)

            K_init = self._camera.sdk_K() if self._camera else None
            result = run_calibration(
                self._all_corners, self._all_ids,
                self._board, sz, K_init)

            self._ui(lambda r=result: self._show_result(r))
        except Exception as e:
            self._ui(lambda e=e: (
                self._log(f"[Calibrate] ERROR: {e}"),
                self._status_var.set(f"Calibration failed: {e}"),
                self._calib_btn.config(state="normal", text="✓  Calibrate"),
            ))

    def _show_result(self, result):
        rms = result["rms"]
        K   = result["K"]
        d   = result["d"]
        self._log(f"\n[Result] RMS reprojection error: {rms:.4f} px")
        self._log(f"[Result] fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  "
                  f"cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
        self._log(f"[Result] dist = {d.ravel().round(6).tolist()}")

        quality = ("EXCELLENT" if rms < 0.3
                   else "GOOD" if rms < 0.5
                   else "ACCEPTABLE" if rms < 1.0
                   else "POOR — recollect with more varied angles")

        self._log(f"[Result] Quality: {quality}")

        # Per-frame errors — flag outliers
        bad = [(i, e) for i, e in result["per_frame_errors"] if e > 1.5]
        if bad:
            self._log(f"[Result] Outlier frames (err > 1.5 px): "
                      f"{[f'#{i}({e:.2f}px)' for i, e in bad]}")
            self._log("  → Consider clearing and recapturing without those poses.")

        save_intrinsics(result, OUTPUT_DIR)
        self._log(f"[Result] Saved to {OUTPUT_DIR}/camera_intrinsics.npz + .json")
        self._status_var.set(
            f"Calibration done! [{quality}]  RMS={rms:.3f}px  "
            f"fx={K[0,0]:.1f} fy={K[1,1]:.1f}")
        self._calib_btn.config(state="normal", text="✓  Calibrate")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ui(self, fn):
        self._cb_queue.put(fn)

    def _schedule_flush(self):
        self._flush_queue()
        self.root.after(50, self._schedule_flush)

    def _flush_queue(self):
        try:
            while True:
                fn = self._cb_queue.get_nowait()
                fn()
        except queue.Empty:
            pass

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self._log_box.config(state="normal")
        self._log_box.insert("end", line)
        self._log_box.see("end")
        self._log_box.config(state="disabled")


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    if not _PIL_OK:
        print("ERROR: Pillow required.  pip install Pillow")
        sys.exit(1)
    root = tk.Tk()
    CameraCalibApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
