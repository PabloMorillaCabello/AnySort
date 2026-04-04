#!/usr/bin/env python3
"""
Real-time viewer for Orbbec Gemini 2 camera — pure Python, no ROS2.

Displays RGB, Depth (colorized), and optionally IR streams side by side.
Optionally opens a live 3D Open3D point cloud window (--pointcloud).
Uses pyorbbecsdk (Orbbec SDK v2 Python bindings) directly.

Usage:
  python3 /ros2_ws/scripts/view_camera.py
  python3 /ros2_ws/scripts/view_camera.py --no-depth
  python3 /ros2_ws/scripts/view_camera.py --ir
  python3 /ros2_ws/scripts/view_camera.py --align          # align depth to color
  python3 /ros2_ws/scripts/view_camera.py --pointcloud     # depth-only 3D view
  python3 /ros2_ws/scripts/view_camera.py --pointcloud --align  # colored 3D view

Controls:
  s — Save current frames to /ros2_ws/data/  (+ pointcloud .ply if --align)
  q — Quit
  r — Reset drop counters
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional, Tuple

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
    sys.exit(1)

# ---------------------------------------------------------------------------
# Default Gemini 2 intrinsics (1280×800 depth stream, approximate)
# Used as fallback when the SDK does not expose intrinsics via API.
# ---------------------------------------------------------------------------
_DEFAULT_FX = 891.5
_DEFAULT_FY = 891.5
_DEFAULT_CX = 640.0
_DEFAULT_CY = 400.0

# How often (in camera frames) to rebuild the point cloud in the O3D window.
_PCD_UPDATE_EVERY = 15
# Maximum points sent to Open3D per update (keeps it interactive).
_PCD_MAX_PTS = 10_000_000


# ---------------------------------------------------------------------------
# Intrinsics helpers
# ---------------------------------------------------------------------------

def try_get_intrinsics(pipeline) -> Tuple[float, float, float, float]:
    """Try to read depth intrinsics from the SDK; fall back to Gemini 2 defaults."""
    try:
        cam_param = pipeline.get_camera_param()
        di = cam_param.depth_intrinsic
        fx, fy = float(di.fx), float(di.fy)
        cx, cy = float(di.cx), float(di.cy)
        if all(v > 0 for v in [fx, fy, cx, cy]):
            return fx, fy, cx, cy
    except Exception:
        pass
    return _DEFAULT_FX, _DEFAULT_FY, _DEFAULT_CX, _DEFAULT_CY


# ---------------------------------------------------------------------------
# Frame conversion helpers
# ---------------------------------------------------------------------------

def frame_to_bgr(frame: VideoFrame) -> Optional[np.ndarray]:
    """Convert an Orbbec color VideoFrame to a BGR numpy array."""
    if frame is None:
        return None
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.asarray(frame.get_data())
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.BGR:
        return data.reshape(height, width, 3)
    elif fmt == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUYV)
    elif fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif fmt == OBFormat.UYVY:
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY)
    elif fmt == OBFormat.NV12:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12)
    elif fmt == OBFormat.NV21:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV21)
    elif fmt == OBFormat.I420:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_I420)
    else:
        print(f"Unsupported color format: {fmt}")
        return None


def depth_frame_to_array(frame: VideoFrame) -> Optional[np.ndarray]:
    """Convert an Orbbec depth frame to a uint16 numpy array (mm values).

    Handles raw uint16 and compressed formats (e.g. RLE).  The SDK decompresses
    internally; np.asarray() on get_data() gives us the pixel values directly.
    """
    if frame is None:
        return None
    width, height = frame.get_width(), frame.get_height()
    n_pixels = width * height
    data = np.asarray(frame.get_data())

    # Already uint16 with correct pixel count
    if data.dtype == np.uint16 and data.size == n_pixels:
        return data.reshape((height, width))

    # uint8 buffer carrying uint16 values (raw, uncompressed)
    if data.dtype == np.uint8 and data.size == n_pixels * 2:
        return data.view(np.uint16).reshape((height, width))

    # Compressed / RLE: buffer smaller than full frame — try frombuffer directly
    if data.dtype == np.uint8 and data.size < n_pixels * 2:
        try:
            arr = np.frombuffer(bytes(frame.get_data()), dtype=np.uint16)
            if arr.size == n_pixels:
                return arr.reshape((height, width))
        except Exception:
            pass
        return None

    # Oversized buffer: take first n_pixels uint16 words
    if data.dtype == np.uint8 and data.size >= n_pixels * 2:
        return data[: n_pixels * 2].view(np.uint16).reshape((height, width))

    # Generic fallback
    try:
        return data.reshape((height, width)).astype(np.uint16)
    except Exception:
        return None


def ir_frame_to_bgr(frame: VideoFrame) -> Optional[np.ndarray]:
    """Convert an Orbbec IR frame to a displayable BGR image.

    Handles Y8 (1 byte/pixel) and Y16 (2 bytes/pixel) formats.
    The Gemini 2 IR stream uses Y8, so 1280×800 → 1 024 000 bytes.
    """
    if frame is None:
        return None
    width, height = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    n_pixels = width * height
    data = np.asarray(frame.get_data())

    # --- Y8: 1 byte per pixel ---
    is_y8 = (hasattr(OBFormat, "Y8") and fmt == OBFormat.Y8) or "Y8" in str(fmt).upper()
    if is_y8 or (data.dtype == np.uint8 and data.size == n_pixels):
        gray = data[:n_pixels].reshape((height, width))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # --- Y16: 2 bytes per pixel ---
    is_y16 = (hasattr(OBFormat, "Y16") and fmt == OBFormat.Y16) or "Y16" in str(fmt).upper()
    if is_y16 or data.dtype == np.uint16:
        if data.dtype == np.uint16 and data.size == n_pixels:
            ir16 = data.reshape((height, width))
        elif data.dtype == np.uint8 and data.size == n_pixels * 2:
            ir16 = data.view(np.uint16).reshape((height, width))
        else:
            return None
        norm = cv2.normalize(ir16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    # Fallback: treat as uint8 grayscale
    try:
        gray = data[:n_pixels].reshape((height, width)).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    except Exception:
        return None


def colorize_depth(depth: np.ndarray, max_distance_mm: int = 5000) -> np.ndarray:
    """Colorize a 16-bit depth image (mm) using TURBO colormap."""
    depth_clipped = np.clip(depth, 0, max_distance_mm)
    depth_norm = (depth_clipped.astype(np.float32) / max_distance_mm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    colored[depth == 0] = [0, 0, 0]
    return colored


# ---------------------------------------------------------------------------
# Point cloud helpers
# ---------------------------------------------------------------------------

def build_open3d_pcd(depth_raw: np.ndarray,
                     color_bgr: Optional[np.ndarray],
                     fx: float, fy: float, cx: float, cy: float,
                     max_depth_mm: int = 5000,
                     max_pts: int = _PCD_MAX_PTS):
    """Build an Open3D PointCloud from uint16 depth (mm) + optional BGR color.

    Color is only applied when depth and color have the same resolution
    (i.e. when --align is active).  Otherwise the cloud is rendered in grey.
    """
    import open3d as o3d

    height, width = depth_raw.shape
    ys, xs = np.mgrid[0:height, 0:width]
    z_mm = depth_raw.astype(np.float32)
    valid = (z_mm > 0) & (z_mm < max_depth_mm)

    xv = xs[valid].astype(np.float32)
    yv = ys[valid].astype(np.float32)
    zv = z_mm[valid] / 1000.0          # mm → m

    # Camera frame: X right, Y down, Z forward (depth)
    # World frame (Z up): X right, Y forward (depth), Z up (-cam_Y)
    X_cam = (xv - cx) * zv / fx
    Y_cam = (yv - cy) * zv / fy
    Z_cam = zv
    pts = np.column_stack([X_cam,    # world X = cam X (right)
                            Z_cam,   # world Y = cam Z (depth / forward)
                            -Y_cam]) # world Z = -cam Y (up)

    # Downsample so Open3D stays interactive
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
        xv = xv[idx]
        yv = yv[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Colors only when resolutions match (aligned mode)
    if (color_bgr is not None
            and color_bgr.shape[0] == height
            and color_bgr.shape[1] == width):
        yi = yv.astype(np.int32)
        xi = xv.astype(np.int32)
        rgb = color_bgr[yi, xi][:, ::-1].astype(np.float32) / 255.0  # BGR→RGB [0,1]
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


def depth_color_to_pointcloud(depth: np.ndarray, color: np.ndarray,
                               fx: float = 615.0, fy: float = 615.0,
                               cx: float = 320.0, cy: float = 240.0) -> np.ndarray:
    """Convert aligned depth + color to XYZRGB pointcloud (Nx6).

    Used for saving .ply files.  Only valid for aligned streams.
    """
    height, width = depth.shape
    if color.shape[:2] != (height, width):
        raise ValueError(f"Dimension mismatch: depth {depth.shape}, color {color.shape}")

    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    valid_mask = depth > 0

    Z = depth[valid_mask].astype(np.float32) / 1000.0
    X = -(x_coords[valid_mask] - cx) * Z / fx  # negate X: depth stream is horizontally mirrored
    Y = (y_coords[valid_mask] - cy) * Z / fy
    valid_colors = color[valid_mask]

    return np.column_stack([X, Y, Z,
                             valid_colors[:, 0],
                             valid_colors[:, 1],
                             valid_colors[:, 2]])


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
                        help="Align depth to color (hardware D2C) + enables colored point cloud")
    parser.add_argument("--pointcloud", action="store_true",
                        help="Open a live 3D Open3D point cloud window (requires depth)")
    parser.add_argument("--max-depth", type=int, default=5000,
                        help="Max depth for colormap and point cloud in mm (default: 5000)")
    parser.add_argument("--save-dir", default="/ros2_ws/data",
                        help="Directory to save frames (default: /ros2_ws/data)")
    args = parser.parse_args()

    # Point cloud requires depth
    if args.pointcloud:
        args.no_depth = False

    # ---- Set up pipeline ----
    pipeline = Pipeline()
    config = Config()

    device = pipeline.get_device()
    device_info = device.get_device_info()
    print(f"\nDevice: {device_info.get_name()}")
    print(f"Serial: {device_info.get_serial_number()}")
    print(f"FW:     {device_info.get_firmware_version()}")

    # Enable color stream
    color_profile = None
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

    # Enable depth-to-color alignment
    alignment_active = False
    if args.align:
        try:
            config.set_align_mode(OBAlignMode.HW_MODE)
            print("Align:  depth → color (hardware)")
            alignment_active = True
        except Exception:
            try:
                config.set_align_mode(OBAlignMode.SW_MODE)
                print("Align:  depth → color (software)")
                alignment_active = True
            except Exception as e:
                print(f"WARNING: Alignment not supported: {e}")

    pipeline.start(config)

    # Read camera intrinsics (for point cloud)
    fx, fy, cx, cy = try_get_intrinsics(pipeline)

    streams = ["RGB"]
    if not args.no_depth:
        streams.append("Depth")
    if args.ir:
        streams.append("IR")
    if alignment_active:
        streams.append("Aligned")
    if args.pointcloud:
        streams.append("3D-PointCloud")

    print("\n" + "=" * 50)
    print(f"  Streaming: {' + '.join(streams)}")
    print("  Controls:  's' = save, 'q' = quit, 'r' = reset drops")
    print("=" * 50)

    # ---- Open3D point cloud window ----
    pcd_vis = None
    pcd_geom = None
    pcd_counter = 0
    pcd_view_reset = False

    if args.pointcloud:
        try:
            import open3d as o3d
            pcd_vis = o3d.visualization.Visualizer()
            pcd_vis.create_window("Point Cloud 3D — Orbbec Gemini 2", width=960, height=640)
            pcd_geom = o3d.geometry.PointCloud()
            pcd_vis.add_geometry(pcd_geom)
            # Add origin coordinate frame (X=red, Y=green, Z=blue), size 0.3 m
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            pcd_vis.add_geometry(coord_frame)
            opt = pcd_vis.get_render_option()
            opt.point_size = 2.0
            opt.background_color = np.array([0.08, 0.08, 0.12])
            print(f"Open3D window created  |  intrinsics: "
                  f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
            print("  Point cloud updates every "
                  f"{_PCD_UPDATE_EVERY} frames, max {_PCD_MAX_PTS:,} pts")
        except ImportError:
            print("WARNING: open3d not installed — point cloud window disabled.")
            print("         Install with:  uv pip install open3d")
            pcd_vis = None
        except Exception as e:
            print(f"WARNING: Could not create Open3D window: {e}")
            pcd_vis = None

    # Counters & state
    color_fps = FPSCounter()
    depth_fps = FPSCounter()
    ir_fps = FPSCounter()
    depth_drops = DropCounter()
    ir_drops = DropCounter()
    stats_time = time.time()

    last_color = None
    last_depth_raw = None
    last_ir_raw = None
    last_depth_aligned = None

    try:
        while True:
            frames: FrameSet = pipeline.wait_for_frames(200)
            if frames is None:
                if pcd_vis is not None:
                    pcd_vis.poll_events()
                    pcd_vis.update_renderer()
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
                            if alignment_active:
                                last_depth_aligned = depth_raw.copy()
                            depth_vis = colorize_depth(depth_raw, args.max_depth)
                            if last_color is not None and depth_vis.shape[:2] != last_color.shape[:2]:
                                depth_vis = cv2.resize(
                                    depth_vis, (last_color.shape[1], last_color.shape[0]))
                            drop_text = depth_drops.get_text()
                            cv2.putText(depth_vis,
                                        f"Depth ({depth_fps.fps:.1f} fps) {drop_text}",
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
                            cv2.putText(ir_image,
                                        f"IR ({ir_fps.fps:.1f} fps) {drop_text}",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            panels.append(ir_image)
                    except Exception as e:
                        ir_drops.increment()
                        print(f"WARNING: IR frame skipped: {e}")

            # ---- Open3D point cloud update ----
            if pcd_vis is not None and last_depth_raw is not None:
                pcd_counter += 1
                if pcd_counter % _PCD_UPDATE_EVERY == 0:
                    try:
                        color_for_pcd = last_color if alignment_active else None
                        new_pcd = build_open3d_pcd(
                            last_depth_raw, color_for_pcd,
                            fx, fy, cx, cy,
                            max_depth_mm=args.max_depth,
                        )
                        pcd_geom.points = new_pcd.points
                        pcd_geom.colors = new_pcd.colors
                        pcd_vis.update_geometry(pcd_geom)
                        if not pcd_view_reset:
                            pcd_vis.reset_view_point(True)
                            pcd_view_reset = True
                    except Exception as e:
                        print(f"WARNING: Point cloud update error: {e}")
                pcd_vis.poll_events()
                pcd_vis.update_renderer()

            # ---- Periodic stats ----
            if time.time() - stats_time > 10:
                if depth_drops.drops > 0 or ir_drops.drops > 0:
                    print(f"Stats: Depth drops={depth_drops.drops}, "
                          f"IR drops={ir_drops.drops}")
                stats_time = time.time()

            # ---- Display ----
            if not panels:
                waiting = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(waiting, "Waiting for camera data...", (80, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow("Orbbec Gemini 2", waiting)
            else:
                display = np.hstack(panels)
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
                save_frames(args.save_dir, last_color, last_depth_raw, last_ir_raw,
                            last_depth_aligned, last_color, alignment_active,
                            fx, fy, cx, cy)
            elif key == ord("r"):
                depth_drops.reset()
                ir_drops.reset()
                print("Drop counters reset.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if pcd_vis is not None:
            try:
                pcd_vis.destroy_window()
            except Exception:
                pass
        print("Pipeline stopped.")


# ---------------------------------------------------------------------------
# Frame saving
# ---------------------------------------------------------------------------

def save_frames(save_dir, color, depth_raw, ir, depth_aligned, color_for_pc,
                save_pc, fx, fy, cx, cy):
    """Save current frames + aligned coloured point cloud if available."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if color is not None:
        d = os.path.join(save_dir, "rgb")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"rgb_{ts}.png")
        cv2.imwrite(path, color)
        print(f" Saved: {path}")

    if depth_raw is not None:
        d = os.path.join(save_dir, "depth")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, f"depth_{ts}.png"), depth_raw)
        np.save(os.path.join(d, f"depth_{ts}.npy"), depth_raw)
        print(f" Saved: {os.path.join(d, f'depth_{ts}.npy')}")

    if depth_aligned is not None and color_for_pc is not None:
        d = os.path.join(save_dir, "depth_aligned")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, f"depth_aligned_{ts}.png"), depth_aligned)
        np.save(os.path.join(d, f"depth_aligned_{ts}.npy"), depth_aligned)
        print(f" Saved: {os.path.join(d, f'depth_aligned_{ts}.npy')}")

    if save_pc and depth_aligned is not None and color_for_pc is not None:
        try:
            print(" Generating aligned point cloud...")
            pointcloud = depth_color_to_pointcloud(depth_aligned, color_for_pc, fx, fy, cx, cy)
            d = os.path.join(save_dir, "pointcloud")
            os.makedirs(d, exist_ok=True)

            ply_path = os.path.join(d, f"pointcloud_aligned_{ts}.ply")
            with open(ply_path, "w") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {len(pointcloud)}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write("end_header\n")
                for pt in pointcloud:
                    f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} "
                            f"{int(pt[3])} {int(pt[4])} {int(pt[5])}\n")
            print(f" Saved: {ply_path} ({len(pointcloud):,} pts)")

            np.save(os.path.join(d, f"pointcloud_aligned_{ts}.npy"), pointcloud)
            print(f" Saved: {os.path.join(d, f'pointcloud_aligned_{ts}.npy')}")
        except Exception as e:
            print(f" Pointcloud generation failed: {e}")

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
