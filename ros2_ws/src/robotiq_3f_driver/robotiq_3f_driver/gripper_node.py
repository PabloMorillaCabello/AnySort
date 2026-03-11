#!/usr/bin/env python3
"""
ROS2 node for controlling the Robotiq 3-Finger Adaptive Gripper.
Communicates via Modbus RTU over serial.
"""
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Float32


class Robotiq3FGripperNode(Node):
    """Robotiq 3F Gripper ROS2 driver node."""

    def __init__(self):
        super().__init__("robotiq_3f_gripper")

        # Parameters
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("slave_id", 9)

        self.port = self.get_parameter("port").value
        self.baudrate = self.get_parameter("baudrate").value
        self.slave_id = self.get_parameter("slave_id").value

        # Services
        self.srv_activate = self.create_service(
            Trigger, "~/activate", self.activate_callback
        )
        self.srv_open = self.create_service(
            Trigger, "~/open", self.open_callback
        )
        self.srv_close = self.create_service(
            Trigger, "~/close", self.close_callback
        )

        # Subscriber for position commands (0.0 = open, 1.0 = closed)
        self.sub_command = self.create_subscription(
            Float32, "~/command", self.command_callback, 10
        )

        # Publisher for gripper status
        self.pub_status = self.create_publisher(Float32, "~/status", 10)

        self.get_logger().info(
            f"Robotiq 3F Gripper node started on {self.port} @ {self.baudrate}"
        )

        # TODO: Initialize Modbus connection
        # self._init_modbus()

    def activate_callback(self, request, response):
        """Activate the gripper."""
        self.get_logger().info("Activating gripper...")
        # TODO: Send activation command via Modbus
        response.success = True
        response.message = "Gripper activated"
        return response

    def open_callback(self, request, response):
        """Fully open the gripper."""
        self.get_logger().info("Opening gripper...")
        # TODO: Send open command
        response.success = True
        response.message = "Gripper opened"
        return response

    def close_callback(self, request, response):
        """Fully close the gripper."""
        self.get_logger().info("Closing gripper...")
        # TODO: Send close command
        response.success = True
        response.message = "Gripper closed"
        return response

    def command_callback(self, msg):
        """Set gripper position (0.0=open, 1.0=closed)."""
        position = max(0.0, min(1.0, msg.data))
        self.get_logger().info(f"Gripper command: {position:.2f}")
        # TODO: Send position command via Modbus


def main(args=None):
    rclpy.init(args=args)
    node = Robotiq3FGripperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
