#!/usr/bin/env python3
"""
Test: Full pipeline integration test (without real hardware).
Publishes synthetic RGB-D data and verifies that:
  1. SAM3 segmentation produces a mask
  2. GraspGen produces grasp poses from the mask + depth
  3. Topics are connected end-to-end

Run inside the Docker container after building the workspace.

Usage:
  # Terminal 1: Launch pipeline in sim mode
  ros2 launch graspgen_pipeline full_pipeline.launch.py \
      use_sim:=true launch_camera:=false text_prompt:="object"

  # Terminal 2: Run this test
  python3 scripts/test_full_pipeline.py
"""
import time
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray
from std_msgs.msg import String
from cv_bridge import CvBridge


class PipelineTester(Node):
    """Publishes synthetic data and checks pipeline outputs."""

    def __init__(self):
        super().__init__("pipeline_tester")
        self.bridge = CvBridge()

        # Publishers (simulate camera)
        self.rgb_pub = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/depth/camera_info", 10)

        # Subscribers (check outputs)
        self.mask_received = False
        self.grasps_received = False
        self.grasps_count = 0

        self.mask_sub = self.create_subscription(
            Image, "/segmentation/mask", self._mask_cb, 10
        )
        self.grasps_sub = self.create_subscription(
            PoseArray, "/grasp_gen/grasp_poses", self._grasps_cb, 10
        )

    def _mask_cb(self, msg):
        mask = self.bridge.imgmsg_to_cv2(msg, "mono8")
        n_pixels = np.sum(mask > 0)
        self.get_logger().info(f"[RECV] Segmentation mask: {mask.shape}, {n_pixels} masked pixels")
        self.mask_received = True

    def _grasps_cb(self, msg):
        self.grasps_count = len(msg.poses)
        self.get_logger().info(f"[RECV] Grasp poses: {self.grasps_count} grasps")
        if self.grasps_count > 0:
            best = msg.poses[0]
            self.get_logger().info(
                f"  Best grasp: pos=({best.position.x:.3f}, "
                f"{best.position.y:.3f}, {best.position.z:.3f})"
            )
        self.grasps_received = True

    def publish_synthetic_data(self):
        """Publish a synthetic RGB-D frame with a visible object."""
        # RGB: blue background with a red rectangle (the "object")
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[:, :] = [40, 40, 120]  # dark blue background
        rgb[150:350, 200:450] = [200, 50, 50]  # red "object"

        # Depth: object at ~0.5m, background at 1.5m
        depth = np.full((480, 640), 1500, dtype=np.uint16)  # 1500mm background
        depth[150:350, 200:450] = 500  # 500mm = 0.5m for the object

        # Camera info (typical 640x480 RGB camera intrinsics)
        info = CameraInfo()
        info.header.stamp = self.get_clock().now().to_msg()
        info.header.frame_id = "camera_depth_optical_frame"
        info.width = 640
        info.height = 480
        info.k = [615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0]
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.p = [615.0, 0.0, 320.0, 0.0, 0.0, 615.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        # Publish
        now = self.get_clock().now().to_msg()
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, "bgr8")
        rgb_msg.header.stamp = now
        rgb_msg.header.frame_id = "camera_color_optical_frame"

        depth_msg = self.bridge.cv2_to_imgmsg(depth, "16UC1")
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = "camera_depth_optical_frame"

        info.header.stamp = now

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info)

        self.get_logger().info("[PUB] Synthetic RGB-D frame published")


def main():
    rclpy.init()
    tester = PipelineTester()

    print("=" * 50)
    print("  Full Pipeline Integration Test")
    print("  (synthetic data, no hardware needed)")
    print("=" * 50)
    print()
    print("Make sure the pipeline is running:")
    print("  ros2 launch graspgen_pipeline full_pipeline.launch.py \\")
    print("      use_sim:=true launch_camera:=false text_prompt:=\"object\"")
    print()

    # Publish synthetic data multiple times (give pipeline time to initialize)
    for i in range(30):
        tester.publish_synthetic_data()
        rclpy.spin_once(tester, timeout_sec=1.0)

        if tester.grasps_received:
            break

    # Report results
    print()
    print("=" * 50)
    print("  Results")
    print("=" * 50)

    if tester.mask_received:
        print("  [PASS] Segmentation mask received")
    else:
        print("  [FAIL] No segmentation mask received")

    if tester.grasps_received:
        print(f"  [PASS] Grasp poses received ({tester.grasps_count} grasps)")
    else:
        print("  [FAIL] No grasp poses received")

    success = tester.mask_received and tester.grasps_received
    print()
    print("  Pipeline test:", "PASSED" if success else "FAILED")

    tester.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
