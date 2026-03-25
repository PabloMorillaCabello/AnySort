#!/usr/bin/env python3
"""
Real-time viewer for Orbbec Gemini 2 camera — pure Python, no ROS2.
[Fixed: robust to corrupted frames + aligned pointcloud generation]

Displays RGB, Depth (colorized), and optionally IR streams side by side.
Uses pyorbbecsdk (Orbbec SDK v2 Python bindings) directly.

Usage:
  python3 /ros2_ws/scripts/view_camera.py
  python3 /ros2_ws/scripts/view_camera.py --no-depth
  python3 /ros2_ws/scripts/view_camera.py --ir
  python3 /ros2_ws/scripts/view_camera.py --align # align depth to color + pointcloud

Controls:
  s — Save current frames to /ros2_ws/data/ (+ aligned pointcloud if --align)
  q — Quit
  r — Reset drop counters

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
except ImportError as e:
    print(f"ERROR: Failed to import pyorbbecsdk: {e}")
    print("Install with: uv pip install pyorbbecsdk")
    print("If already installed, you may need the Orbbec SDK .deb:")
    print("  dpkg -i OrbbecSDK_v2.7.6_amd64.deb")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Frame conversion helpers (robustified)
# ---------------------------------------------------------------------------


def frame_to_bgr(frame: VideoFrame) -> Optional[np.ndarray]:
    """Convert an Orbbec color/IR VideoFrame to a BGR numpy array."""
    if frame is None:
        return None
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.asarray(frame.get_data())
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
    """Convert an Orbbec depth frame to a uint16 numpy array (raw mm values).

    Handles both raw uint16 and compressed formats (e.g. RLE).
    For compressed formats the SDK returns the decompressed pixel data
    via get_data(), but the buffer length may not match width*height*2.
    We use np.asarray which handles the SDK's internal conversion.
    """
    if frame is None:
        return None
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.asarray(frame.get_data())
    n_pixels = width * height

    # For compressed formats (RLE, LOSSLESS, etc.) the raw buffer size
    # won't match width*height*2.  The SDK already decompresses for us,
    # so just validate pixel count after reinterpretation.
    if data.dtype == np.uint16 and data.size == n_pixels:
        return data.reshape((height, width))

    # Raw uint16 delivered as uint8 bytes
    if data.dtype == np.uint8 and data.size == n_pixels * 2:
        return data.view(np.uint16).reshape((height, width))

    # Compressed / RLE: try reinterpreting whatever the SDK gave us
    if data.dtype == np.uint8:
        # Not enough bytes for full frame — truly compressed, can't decode here
        if data.size < n_pixels * 2:
            # Try the scale_conversion provided by the SDK
            try:
                depth = np.frombuffer(frame.get_data(), dtype=np.uint16)
                if depth.size == n_pixels:
                    return depth.reshape((height, width))
            except Exception:
                pass
            return None
        # More than enough bytes — take the first n_pixels uint16 values
        return data[:n_pixels * 2].view(np.uint16).reshape((height, width))

    # Fallback: try direct reshape
    try:
        return data.reshape((height, width)).astype(np.uint16)
    except Exception:
        return None


def colorize_depth(depth: np.ndarray, max_distance_mm: int = 5000) -> np.ndarray:
    """Colorize a 16-bit depth image (mm) using TURBO colormap."""
    depth_clipped = np.clip(depth, 0, max_distance_mm)
    depth_norm = (depth_clipped.astype(np.float32) / max_distance_mm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    colored[depth == 0] = [0, 0, 0]  # invalid pixels = black
    return colored


def ir_frame_to_bgr(frame: VideoFrame) -> Optional[np.ndarray]:
    """Convert an Orbbec IR frame to a displayable BGR image.

    Handles both Y8 (uint8, 1 byte/pixel) and Y16 (uint16, 2 bytes/pixel) formats.
    """
    if frame is None:
        return None
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.asarray(frame.get_data())
    n_pixels = width * height

    if fmt == OBFormat.Y8 or (data.dtype == np.uint8 and data.size == n_pixels):
        # Y8: 1 byte per pixel, already uint8
        gray = data[:n_pixels].reshape((height, width))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif fmt == OBFormat.Y16 or (data.dtype == np.uint16 and data.size == n_pixels):
        # Y16: 2 bytes per pixel
        if data.dtype == np.uint16:
            ir16 = data[:n_pixels].reshape((height, width))
        else:
            # uint8 buffer holding uint16 data
            if data.size < n_pixels * 2:
                return None
            ir16 = data[:n_pixels * 2].view(np.uint16).reshape((height, width))
        norm = cv2.normalize(ir16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    else:
        # Fallback: try treating as uint8 grayscale
        try:
            gray = data[:n_pixels].reshape((height, width)).astype(np.uint8)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        except Exception:
            return None


def depth_color_to_pointcloud(depth: np.ndarray, color: np.ndarray, fx: float = 615.0, fy: float = 615.0, cx: float = 320.0, cy: float = 240.0) -> np.ndarray:
    """
    Convert aligned depth + color to colored pointcloud (XYZRGB).
    
    Args:
        depth: uint16 depth image (mm) aligned to color - shape (H,W)
        color: RGB image (H,W,C uint8) 
        fx,fy,cx,cy: Color camera intrinsics (Gemini 2 defaults)
    
    Returns:
        N x 6 numpy array (X,Y,Z,R,G,B) - only valid points (depth > 0)
    """
    height, width = depth.shape
    
    # ✅ Verificar dimensiones coinciden (crítico para aligned data)
    if color.shape[:2] != (height, width):
        raise ValueError(f"Dimension mismatch: depth {depth.shape}, color {color.shape}")
    
    # Meshgrid de coordenadas píxel
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Máscara de puntos válidos
    valid_mask = depth > 0
    
    # Backprojection (depth en metros)
    Z = depth[valid_mask].astype(np.float32) / 1000.0
    X = (x_coords[valid_mask] - cx) * Z / fx
    Y = (y_coords[valid_mask] - cy) * Z / fy
    
    # ✅ FIXED: Extraer colores directamente de píxeles válidos
    valid_colors = color[valid_mask]  # Shape: (N_valid, 3)
    
    # Stack: X,Y,Z,R,G,B
    points = np.column_stack([X, Y, Z, valid_colors[:,0], valid_colors[:,1], valid_colors[:,2]])
    
    return points


# ---------------------------------------------------------------------------
# FPS + Drop tracker
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


class DropCounter:
    def __init__(self):
        self.drops = 0

    def increment(self):
        self.drops += 1

    def reset(self):
        self.drops = 0

    def get_text(self):
        return f"Dropped: {self.drops}" if self.drops > 0 else ""


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
                        help="Align depth to color (hardware D2C) + save pointcloud")
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
        print(f"Color:  {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()} fps ({color_profile.get_format()})")
    except Exception as e:
        print(f"WARNING: Could not enable color stream: {e}")

    # Enable depth stream
    if not args.no_depth:
        try:
            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profiles.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            print(f"Depth:  {depth_profile.get_width()}x{depth_profile.get_height()} @ {depth_profile.get_fps()} fps ({depth_profile.get_format()})")
        except Exception as e:
            print(f"WARNING: Could not enable depth stream: {e}")

    # Enable IR stream
    if args.ir:
        try:
            ir_profiles = pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
            ir_profile = ir_profiles.get_default_video_stream_profile()
            config.enable_stream(ir_profile)
            print(f"IR:     {ir_profile.get_width()}x{ir_profile.get_height()} @ {ir_profile.get_fps()} fps ({ir_profile.get_format()})")
        except Exception as e:
            print(f"WARNING: Could not enable IR stream: {e}")

    # Enable depth-to-color alignment if requested
    alignment_active = False
    if args.align:
        try:
            config.set_align_mode(OBAlignMode.HW_MODE)
            print("\nAlign:  depth → color (hardware)")
            alignment_active = True
        except Exception:
            try:
                config.set_align_mode(OBAlignMode.SW_MODE)
                print("Align:  depth → color (software)")
                alignment_active = True
            except Exception as e:
                print(f"WARNING: Alignment not supported: {e}")

    # Start the pipeline
    pipeline.start(config)

    print("\n" + "="*50)
    streams = ["RGB"]
    if not args.no_depth:
        streams.append("Depth")
    if args.ir:
        streams.append("IR")
    if alignment_active:
        streams.append("Pointcloud")
    print(f"  Streaming: {' + '.join(streams)}")
    print("  Controls:  's' = save, 'q' = quit, 'r' = reset drops")
    print("="*50)

    # Counters
    color_fps = FPSCounter()
    depth_fps = FPSCounter()
    ir_fps = FPSCounter()
    depth_drops = DropCounter()
    ir_drops = DropCounter()
    stats_time = time.time()

    # Latest raw frames for saving
    last_color = None
    last_depth_raw = None
    last_ir_raw = None
    last_depth_aligned = None  # Depth aligned to color (for pointcloud)

    try:
        while True:
            # Wait for a frameset (200ms timeout for robustness)
            frames: FrameSet = pipeline.wait_for_frames(200)
            if frames is None:
                continue

            panels = []

            # ---- Color ----
            color_frame = frames.get_color_frame()
            if color_frame is not None:
                try:
                    color_image = frame_to_bgr(color_frame)
                    if color_image is not None:
                        last_color = color_image.copy()
                        color_fps.tick()
                        cv2.putText(color_image, f"RGB ({color_fps.fps:.1f} fps)",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        panels.append(color_image)
                except Exception as e:
                    print(f"WARNING: Color frame error: {e}")

            # ---- Depth ----
            if not args.no_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame is not None:
                    try:
                        depth_raw = depth_frame_to_array(depth_frame)
                        if depth_raw is not None:
                            last_depth_raw = depth_raw.copy()
                            depth_fps.tick()
                            
                            # Si alignment activo, el depth_frame YA está alineado con color
                            if alignment_active:
                                last_depth_aligned = depth_raw.copy()
                            
                            depth_vis = colorize_depth(depth_raw, args.max_depth)
                            # Resize to match color if needed (for display only)
                            if last_color is not None and depth_vis.shape[:2] != last_color.shape[:2]:
                                depth_vis = cv2.resize(
                                    depth_vis, (last_color.shape[1], last_color.shape[0]))
                            drop_text = depth_drops.get_text()
                            cv2.putText(depth_vis, f"Depth ({depth_fps.fps:.1f} fps) {drop_text}",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            panels.append(depth_vis)
                    except Exception as e:
                        depth_drops.increment()
                        print(f"WARNING: Depth frame skipped: {e}")

            # ---- IR ----
            if args.ir:
                ir_frame = frames.get_ir_frame()
                if ir_frame is not None:
                    try:
                        ir_image = ir_frame_to_bgr(ir_frame)
                        if ir_image is not None:
                            last_ir_raw = ir_image.copy()
                            ir_fps.tick()
                            if last_color is not None and ir_image.shape[:2] != last_color.shape[:2]:
                                ir_image = cv2.resize(
                                    ir_image, (last_color.shape[1], last_color.shape[0]))
                            drop_text = ir_drops.get_text()
                            cv2.putText(ir_image, f"IR ({ir_fps.fps:.1f} fps) {drop_text}",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            panels.append(ir_image)
                    except Exception as e:
                        ir_drops.increment()
                        print(f"WARNING: IR frame skipped: {e}")

            # Periodic stats
            if time.time() - stats_time > 10:
                if depth_drops.drops > 0 or ir_drops.drops > 0:
                    print(f"Stats: Depth drops={depth_drops.drops}, IR drops={ir_drops.drops}")
                stats_time = time.time()

            # ---- Display ----
            if not panels:
                waiting = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(waiting, "Waiting for camera data...", (80, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
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
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                save_frames(args.save_dir, last_color, last_depth_raw, last_ir_raw, 
                           last_depth_aligned, last_color, args.align)
            elif key == ord('r'):
                depth_drops.reset()
                ir_drops.reset()
                print("Drop counters reset.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped.")


def save_frames(save_dir, color, depth_raw, ir, depth_aligned, color_for_pc, save_pc):
    """Save current frames + aligned pointcloud if available."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # RGB
    if color is not None:
        d = os.path.join(save_dir, "rgb")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"rgb_{ts}.png")
        cv2.imwrite(path, color)
        print(f" Saved: {path}")

    # Depth raw
    if depth_raw is not None:
        d = os.path.join(save_dir, "depth")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"depth_{ts}.png")
        cv2.imwrite(path, depth_raw)
        print(f" Saved: {path}")
        npy_path = os.path.join(d, f"depth_{ts}.npy")
        np.save(npy_path, depth_raw)
        print(f" Saved: {npy_path}")

    # Depth aligned (solo si alignment activo)
    if depth_aligned is not None and color_for_pc is not None:
        d = os.path.join(save_dir, "depth_aligned")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"depth_aligned_{ts}.png")
        cv2.imwrite(path, depth_aligned)
        print(f" Saved: {path}")
        npy_path = os.path.join(d, f"depth_aligned_{ts}.npy")
        np.save(npy_path, depth_aligned)
        print(f" Saved: {npy_path}")

    # Pointcloud (solo si alignment activo)
    if save_pc and depth_aligned is not None and color_for_pc is not None:
        try:
            print(" Generating aligned pointcloud...")
            pointcloud = depth_color_to_pointcloud(depth_aligned, color_for_pc)
            
            d = os.path.join(save_dir, "pointcloud")
            os.makedirs(d, exist_ok=True)
            
            # .ply (estándar Open3D/PCL)
            ply_path = os.path.join(d, f"pointcloud_aligned_{ts}.ply")
            with open(ply_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(pointcloud)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                for pt in pointcloud:
                    f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {int(pt[3])} {int(pt[4])} {int(pt[5])}\n")
            print(f" Saved: {ply_path}")
            
            # .npy (numpy rápido)
            npy_path = os.path.join(d, f"pointcloud_aligned_{ts}.npy")
            np.save(npy_path, pointcloud)
            print(f" Saved: {npy_path} ({len(pointcloud):,} points)")
            
        except Exception as e:
            print(f"❌ Pointcloud generation failed: {e}")

    # IR
    if ir is not None:
        d = os.path.join(save_dir, "ir")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"ir_{ts}.png")
        cv2.imwrite(path, ir)
        print(f" Saved: {path}")

    if all(x is None for x in [color, depth_raw, ir, depth_aligned]):
        print(" No frames to save yet.")


if __name__ == "__main__":
    main()
