#!/usr/bin/env python3
"""
Camera Node - Thin relay that remaps Orbbec camera topics if needed.
The actual camera driver is the OrbbecSDK_ROS2 package (launched separately
via gemini2.launch.py). This node subscribes to the Orbbec topics and
republishes with synchronized timestamps + optional depth-to-RGB alignment.

If OrbbecSDK_ROS2 is running with default settings, this node is optional —
downstream nodes can subscribe directly to /camera/color/image_raw etc.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters


class CameraNode(Node):
    """Orbbec RGB-D camera relay / synchronizer node."""

    def __init__(self):
        super().__init__("camera_node")

        # Parameters
        self.declare_parameter("rgb_topic_in", "/camera/color/image_raw")
        self.declare_parameter("depth_topic_in", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic_in", "/camera/color/camera_info")
        self.declare_parameter("sync_slop", 0.1)  # seconds tolerance for sync

        rgb_topic = self.get_parameter("rgb_topic_in").value
        depth_topic = self.get_parameter("depth_topic_in").value
        info_topic = self.get_parameter("camera_info_topic_in").value
        sync_slop = self.get_parameter("sync_slop").value

        self.bridge = CvBridge()
        self._camera_info = None

        # Camera info subscriber (single)
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self._info_cb, 10
        )

        # Approximate time synchronizer for RGB + Depth
        self.rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=sync_slop
        )
        self.sync.registerCallback(self._synced_callback)

        # Republish synchronized frames
        self.rgb_pub = self.create_publisher(Image, "/pipeline/rgb", 10)
        self.depth_pub = self.create_publisher(Image, "/pipeline/depth", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/pipeline/camera_info", 10)

        self.get_logger().info(
            f"Camera relay node ready. Subscribing to {rgb_topic} and {depth_topic}"
        )

    def _info_cb(self, msg):
        self._camera_info = msg

    def _synced_callback(self, rgb_msg, depth_msg):
        """Republish synchronized RGB-D pair with unified timestamp."""
        now = self.get_clock().now().to_msg()
        rgb_msg.header.stamp = now
        depth_msg.header.stamp = now

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)

        if self._camera_info is not None:
            info = self._camera_info
            info.header.stamp = now
            self.info_pub.publish(info)


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
