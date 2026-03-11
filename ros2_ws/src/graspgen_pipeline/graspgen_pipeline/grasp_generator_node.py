#!/usr/bin/env python3
"""
Grasp Generator Node - Takes depth image + segmentation mask, builds a
masked point cloud, and runs GraspGen inference to produce 6-DOF grasp poses.

GraspGen API (from NVlabs/GraspGen):
  from grasp_gen.sampler import GraspGenSampler
  from grasp_gen.utils import load_grasp_cfg
  grasps, confidences = GraspGenSampler.run_inference(point_cloud, sampler, ...)

GraspGen expects a point cloud (N x 3 numpy array) as input and returns
6-DOF grasp poses + confidence scores.
"""
import os
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import message_filters

from grasp_gen.sampler import GraspGenSampler
from grasp_gen.utils import load_grasp_cfg


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def depth_to_pointcloud(depth_img, mask, camera_info):
    """Convert masked depth image to 3D point cloud using camera intrinsics."""
    fx = camera_info.k[0]
    fy = camera_info.k[4]
    cx = camera_info.k[2]
    cy = camera_info.k[5]

    # Get valid points inside mask
    rows, cols = np.where(mask > 0)
    depths = depth_img[rows, cols].astype(np.float32) / 1000.0  # mm → meters

    # Filter out zero/invalid depths
    valid = depths > 0.01
    rows, cols, depths = rows[valid], cols[valid], depths[valid]

    # Back-project to 3D
    x = (cols - cx) * depths / fx
    y = (rows - cy) * depths / fy
    z = depths

    points = np.stack([x, y, z], axis=-1)  # (N, 3)
    return points


class GraspGeneratorNode(Node):
    """GraspGen grasp pose generator ROS2 node."""

    def __init__(self):
        super().__init__("grasp_generator_node")

        # Parameters
        self.declare_parameter("gripper_config", "/opt/models/graspgen/robotiq_2f140.yml")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("num_grasps", 50)
        self.declare_parameter("topk_num_grasps", 10)
        self.declare_parameter("grasp_threshold", 0.7)
        self.declare_parameter("remove_outliers", False)

        gripper_config = self.get_parameter("gripper_config").value
        self.num_grasps = self.get_parameter("num_grasps").value
        self.topk = self.get_parameter("topk_num_grasps").value
        self.threshold = self.get_parameter("grasp_threshold").value
        self.remove_outliers = self.get_parameter("remove_outliers").value

        self.bridge = CvBridge()
        self._camera_info = None

        # Load GraspGen model
        self.get_logger().info(f"Loading GraspGen with config: {gripper_config}")
        grasp_cfg = load_grasp_cfg(gripper_config)
        self.grasp_sampler = GraspGenSampler(grasp_cfg)
        self.get_logger().info("GraspGen model loaded successfully.")

        # Camera info subscriber
        self.info_sub = self.create_subscription(
            CameraInfo, "/camera/depth/camera_info", self._info_cb, 10
        )

        # Synchronized depth + mask subscriber
        self.depth_sub = message_filters.Subscriber(self, Image, "/camera/depth/image_raw")
        self.mask_sub = message_filters.Subscriber(self, Image, "/segmentation/mask")
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.mask_sub], queue_size=10, slop=0.5
        )
        self.sync.registerCallback(self._synced_callback)

        # Publishers
        self.grasps_pub = self.create_publisher(PoseArray, "/grasp_gen/grasp_poses", 10)
        self.confidence_pub = self.create_publisher(
            Float32MultiArray, "/grasp_gen/confidences", 10
        )

        self.get_logger().info("GraspGen node ready. Waiting for depth + mask...")

    def _info_cb(self, msg):
        self._camera_info = msg

    def _synced_callback(self, depth_msg, mask_msg):
        """When depth + mask are available, generate grasps."""
        if self._camera_info is None:
            self.get_logger().warn("No camera_info yet, skipping.", throttle_duration_sec=5.0)
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            mask = self.bridge.imgmsg_to_cv2(mask_msg, "mono8")

            # Build masked point cloud
            pc = depth_to_pointcloud(depth, mask, self._camera_info)

            if pc.shape[0] < 100:
                self.get_logger().warn(
                    f"Too few points ({pc.shape[0]}), skipping.", throttle_duration_sec=5.0
                )
                return

            # Run GraspGen inference
            grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
                pc,
                self.grasp_sampler,
                grasp_threshold=self.threshold,
                num_grasps=self.num_grasps,
                topk_num_grasps=self.topk,
                remove_outliers=self.remove_outliers,
            )

            # Convert to PoseArray
            pose_array = PoseArray()
            pose_array.header.stamp = self.get_clock().now().to_msg()
            pose_array.header.frame_id = depth_msg.header.frame_id or "camera_depth_optical_frame"

            confidences = Float32MultiArray()

            for i, (grasp, conf) in enumerate(zip(grasps_inferred, grasp_conf_inferred)):
                pose = Pose()
                # GraspGen returns 4x4 homogeneous transforms
                position = grasp[:3, 3]
                rotation = grasp[:3, :3]
                quat = rotation_matrix_to_quaternion(rotation)

                pose.position.x = float(position[0])
                pose.position.y = float(position[1])
                pose.position.z = float(position[2])
                pose.orientation.x = float(quat[0])
                pose.orientation.y = float(quat[1])
                pose.orientation.z = float(quat[2])
                pose.orientation.w = float(quat[3])

                pose_array.poses.append(pose)
                confidences.data.append(float(conf))

            self.grasps_pub.publish(pose_array)
            self.confidence_pub.publish(confidences)

            self.get_logger().info(
                f"Published {len(pose_array.poses)} grasps "
                f"(best conf: {max(confidences.data):.3f})"
            )

        except Exception as e:
            self.get_logger().error(f"Grasp generation failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = GraspGeneratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
