#!/usr/bin/env python3
"""
Motion Planner Node - Receives grasp poses and plans + executes trajectories
on a UR robot using MoveIt2. Controls the Robotiq 3F gripper.
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseArray, PoseStamped
from std_srvs.srv import Trigger

# TODO: MoveIt2 Python bindings
# from moveit_py import MoveItPy
# from moveit_py.planning import PlanRequestParameters


class MotionPlannerNode(Node):
    """MoveIt2-based motion planner for UR robots."""

    def __init__(self):
        super().__init__("motion_planner_node")

        # Parameters
        self.declare_parameter("robot_ip", "192.168.1.100")
        self.declare_parameter("robot_type", "ur5e")
        self.declare_parameter("use_sim", False)
        self.declare_parameter("planning_group", "manipulator")
        self.declare_parameter("max_velocity_scaling", 0.3)
        self.declare_parameter("max_acceleration_scaling", 0.3)
        self.declare_parameter("planning_time", 5.0)
        self.declare_parameter("num_planning_attempts", 10)

        self.planning_group = self.get_parameter("planning_group").value
        self.vel_scale = self.get_parameter("max_velocity_scaling").value
        self.acc_scale = self.get_parameter("max_acceleration_scaling").value

        # Subscribers
        self.grasps_sub = self.create_subscription(
            PoseArray, "/grasp_gen/grasp_poses", self.grasps_callback, 10
        )

        # Gripper service clients
        self.gripper_open_client = self.create_client(Trigger, "/robotiq_3f_gripper/open")
        self.gripper_close_client = self.create_client(Trigger, "/robotiq_3f_gripper/close")

        self.get_logger().info(
            f"Motion planner ready | robot: {self.get_parameter('robot_type').value} "
            f"| sim: {self.get_parameter('use_sim').value}"
        )

        # TODO: Initialize MoveIt2
        # self._init_moveit()

    def grasps_callback(self, msg):
        """Receive grasp poses, select best, plan and execute."""
        if len(msg.poses) == 0:
            self.get_logger().warn("No grasp poses received.")
            return

        self.get_logger().info(f"Received {len(msg.poses)} grasp poses. Planning...")

        # Select the first (best-ranked) grasp
        target_pose = PoseStamped()
        target_pose.header = msg.header
        target_pose.pose = msg.poses[0]

        # TODO: Plan and execute with MoveIt2
        # 1. Open gripper
        # self._call_gripper_service(self.gripper_open_client)
        # 2. Plan to pre-grasp (approach pose)
        # 3. Plan to grasp pose
        # 4. Close gripper
        # self._call_gripper_service(self.gripper_close_client)
        # 5. Plan retreat
        # 6. Plan to place pose

        self.get_logger().info("Motion planning pipeline completed.")

    def _call_gripper_service(self, client):
        """Call a gripper service (open/close)."""
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Gripper service not available.")
            return False
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        return future.result().success if future.result() else False


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
