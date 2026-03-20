#!/usr/bin/env python3
"""
Real-time viewer for Orbbec Gemini 2 camera — pure Python, no ROS2.

Displays RGB, Depth (colorized), and optionally IR streams side by side.
Uses pyorbbecsdk (Orbbec SDK v2 Python bindings) directly.

Usage:
    python3 /ros2_ws/scripts/view_camera.py
    python3 /ros2_ws/scripts/view_camera.py --no-depth
    python3 /ros2_ws/scripts/view_camera.py --ir
    python3 /ros2_ws/scripts/view_camera.py --align     # align depth to color

Controls:
    s  — Save current frames to /ros2_ws/data/
    q  — Quit

Prerequisites:
    pip install pyorbbecsdk opencv-python numpy
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

try:
    from pyorbbecsdk import (
        Pipeline,
        Config,
        OBSensorType,
        OBFormat,
        VideoFrame,
        FrameSet,
        OBAlignMode,
    )
except ImportError:
    print("ERROR: pyorbbecsdk not installed.")
    print("Install with: pip install pyorbbecsdk")
    print("Or: uv pip install pyorbbecsdk")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Frame conversion helpers (based on pyorbbecsdk examples/utils.py)
# ---------------------------------------------------------------------------

def frame_to_bgr(frame: VideoFrame) -> Optional[np.ndarray]:
    """Convert an Orbbec color/IR VideoFrame to a BGR numpy array."""
    if frame is None:
        return None
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if fmt == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    elif fmt == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif fmt == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    elif fmt == OBFormat.NV12:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    elif fmt == OBFormat.NV21:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    elif fmt == OBFormat.I420:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    else:
        print(f"Unsupported color format: {fmt}")
        return None


def depth_frame_to_array(frame: VideoFrame) -> Optional[np.ndarray]:
    """Convert an Orbbec depth frame to a uint16 numpy array (raw mm values)."""
    if frame is None:
        return None
    width = frame.get_width()
    height = frame.get_height()
    data = np.frombuffer(frame.get_data(), dtype=np.uint16)
    return data.reshape((height, width))


def colorize_depth(depth: np.ndarray, max_distance_mm: int = 5000) -> np.ndarray:
    """Colorize a 16-bit depth image (mm) using TURBO colormap."""
    depth_clipped = np.clip(depth, 0, max_distance_mm)
    depth_norm = (depth_clipped.astype(np.float32) / max_distance_mm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    colored[depth == 0] = [0, 0, 0]  # invalid pixels = black
    return colored


def ir_frame_to_bgr(frame: VideoFrame) -> Optional[np.ndarray]:
    """Convert an Orbbec IR frame to a displayable BGR image."""
    if frame is None:
        return None
    width = frame.get_width()
    height = frame.get_height()
    data = np.frombuffer(frame.get_data(), dtype=np.uint16)
    data = data.reshape((height, width))
    norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# FPS tracker
# ---------------------------------------------------------------------------

class FPSCounter:
    def __init__(self):
        self.count = 0
        self.fps = 0.0
        self.t0 = time.time()

    def tick(self):
        self.count += 1
        elapsed = time.time() - self.t0
        if elapsed >= 1.0:
            self.fps = self.count / elapsed
            self.count = 0
            self.t0 = time.time()


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-time Orbbec Gemini 2 camera viewer (pure Python, no ROS2)"
    )
    parser.add_argument("--no-depth", action="store_true", help="Disable depth stream")
    parser.add_argument("--ir", action="store_true", help="Enable IR stream")
    parser.add_argument("--align", action="store_true",
                        help="Align depth to color (hardware D2C alignment)")
    parser.add_argument("--max-depth", type=int, default=5000,
                        help="Max depth for colormap in mm (default: 5000)")
    parser.add_argument("--save-dir", default="/ros2_ws/data",
                        help="Directory to save frames (default: /ros2_ws/data)")
    args = parser.parse_args()

    # ---- Set up pipeline ----
    pipeline = Pipeline()
    config = Config()

    # Get device info
    device = pipeline.get_device()
    device_info = device.get_device_info()
    print(f"\nDevice: {device_info.get_name()}")
    print(f"Serial: {device_info.get_serial_number()}")
    print(f"FW:     {device_info.get_firmware_version()}")

    # Enable color stream
    try:
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        print(f"Color:  {color_profile.get_width()}x{color_profile.get_height()} "
              f"@ {color_profile.get_fps()} fps ({color_profile.get_format()})")
    except Exception as e:
        print(f"WARNING: Could not enable color stream: {e}")

    # Enable depth stream
    if not args.no_depth:
        try:
            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profiles.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            print(f"Depth:  {depth_profile.get_width()}x{depth_profile.get_height()} "
                  f"@ {depth_profile.get_fps()} fps ({depth_profile.get_format()})")
        except Exception as e:
            print(f"WARNING: Could not enable depth stream: {e}")

    # Enable IR stream
    if args.ir:
        try:
            ir_profiles = pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
            ir_profile = ir_profiles.get_default_video_stream_profile()
            config.enable_stream(ir_profile)
            print(f"IR:     {ir_profile.get_width()}x{ir_profile.get_height()} "
                  f"@ {ir_profile.get_fps()} fps ({ir_profile.get_format()})")
        except Exception as e:
            print(f"WARNING: Could not enable IR stream: {e}")

    # Enable depth-to-color alignment if requested
    if args.align:
        try:
            config.set_align_mode(OBAlignMode.HW_MODE)
            print("Align:  depth → color (hardware)")
        except Exception:
            try:
                config.set_align_mode(OBAlignMode.SW_MODE)
                print("Align:  depth → color (software)")
            except Exception as e:
                print(f"WARNING: Alignment not supported: {e}")

    # Start the pipeline
    pipeline.start(config)

    print("\n========================================")
    streams = ["RGB"]
    if not args.no_depth:
        streams.append("Depth")
    if args.ir:
        streams.append("IR")
    print(f"  Streaming: {' + '.join(streams)}")
    print("  Controls:  's' = save, 'q' = quit")
    print("========================================\n")

    # FPS counters
    color_fps = FPSCounter()
    depth_fps = FPSCounter()
    ir_fps = FPSCounter()

    # Latest raw frames for saving
    last_color = None
    last_depth_raw = None
    last_ir_raw = None

    try:
        while True:
            # Wait for a frameset (100ms timeout)
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            panels = []

            # ---- Color ----
            color_frame = frames.get_color_frame()
            if color_frame is not None:
                color_image = frame_to_bgr(color_frame)
                if color_image is not None:
                    last_color = color_image.copy()
                    color_fps.tick()
                    cv2.putText(color_image, f"RGB ({color_fps.fps:.1f} fps)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    panels.append(color_image)

            # ---- Depth ----
            if not args.no_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame is not None:
                    depth_raw = depth_frame_to_array(depth_frame)
                    if depth_raw is not None:
                        last_depth_raw = depth_raw.copy()
                        depth_fps.tick()
                        depth_vis = colorize_depth(depth_raw, args.max_depth)
                        # Resize to match color if needed
                        if last_color is not None and depth_vis.shape[:2] != last_color.shape[:2]:
                            depth_vis = cv2.resize(
                                depth_vis, (last_color.shape[1], last_color.shape[0])
                            )
                        cv2.putText(depth_vis, f"Depth ({depth_fps.fps:.1f} fps)",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        panels.append(depth_vis)

            # ---- IR ----
            if args.ir:
                ir_frame = frames.get_ir_frame()
                if ir_frame is not None:
                    ir_image = ir_frame_to_bgr(ir_frame)
                    if ir_image is not None:
                        last_ir_raw = ir_image.copy()
                        ir_fps.tick()
                        if last_color is not None and ir_image.shape[:2] != last_color.shape[:2]:
                            ir_image = cv2.resize(
                                ir_image, (last_color.shape[1], last_color.shape[0])
                            )
                        cv2.putText(ir_image, f"IR ({ir_fps.fps:.1f} fps)",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        panels.append(ir_image)

            # ---- Display ----
            if not panels:
                waiting = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(waiting, "Waiting for camera data...",
                            (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow("Orbbec Gemini 2", waiting)
            else:
                display = np.hstack(panels)
                # Scale down if too wide
                max_width = 1920
                if display.shape[1] > max_width:
                    scale = max_width / display.shape[1]
                    display = cv2.resize(display, None, fx=scale, fy=scale)
                cv2.imshow("Orbbec Gemini 2", display)

            # ---- Keyboard ----
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord("s"):
                save_frames(args.save_dir, last_color, last_depth_raw, last_ir_raw)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped.")


def save_frames(save_dir, color, depth_raw, ir):
    """Save current frames to disk."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if color is not None:
        d = os.path.join(save_dir, "rgb")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"rgb_{ts}.png")
        cv2.imwrite(path, color)
        print(f"  Saved: {path}")

    if depth_raw is not None:
        d = os.path.join(save_dir, "depth")
        os.makedirs(d, exist_ok=True)
        # 16-bit PNG (lossless, mm values preserved)
        path = os.path.join(d, f"depth_{ts}.png")
        cv2.imwrite(path, depth_raw)
        print(f"  Saved: {path}")
        # Also .npy for easy numpy loading
        npy_path = os.path.join(d, f"depth_{ts}.npy")
        np.save(npy_path, depth_raw)
        print(f"  Saved: {npy_path}")

    if ir is not None:
        d = os.path.join(save_dir, "depth")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"ir_{ts}.png")
        cv2.imwrite(path, ir)
        print(f"  Saved: {path}")

    if color is None and depth_raw is None and ir is None:
        print("  No frames to save yet.")


if __name__ == "__main__":
    main()
