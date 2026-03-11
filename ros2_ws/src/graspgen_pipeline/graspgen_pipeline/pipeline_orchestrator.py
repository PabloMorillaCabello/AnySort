#!/usr/bin/env python3
"""
Pipeline Orchestrator - Coordinates the full grasp generation pipeline:
  Camera -> Segmentation -> Grasp Generation -> Motion Planning -> Execution

Provides a service interface for triggering the pipeline and monitoring status.
"""
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import String


class PipelineOrchestrator(Node):
    """Orchestrates the full pick pipeline."""

    def __init__(self):
        super().__init__("pipeline_orchestrator")

        # Parameters
        self.declare_parameter("auto_execute", False)
        self.declare_parameter("save_results", True)
        self.declare_parameter("results_dir", "/ros2_ws/data/grasp_poses")

        self.auto_execute = self.get_parameter("auto_execute").value

        # Service to trigger a single pick cycle
        self.trigger_srv = self.create_service(
            Trigger, "~/trigger_pick", self.trigger_pick_callback
        )

        # Status publisher
        self.status_pub = self.create_publisher(String, "~/status", 10)

        self._publish_status("IDLE")
        self.get_logger().info("Pipeline orchestrator ready. Call ~/trigger_pick to start.")

    def trigger_pick_callback(self, request, response):
        """Trigger a full pick cycle."""
        self.get_logger().info("=== Starting pick cycle ===")
        self._publish_status("RUNNING")

        try:
            # TODO: Implement the orchestration logic:
            # 1. Request camera capture
            # 2. Wait for segmentation result
            # 3. Wait for grasp generation
            # 4. If auto_execute: send to motion planner
            # 5. Save results if configured

            response.success = True
            response.message = "Pick cycle completed"
            self._publish_status("IDLE")

        except Exception as e:
            response.success = False
            response.message = f"Pick cycle failed: {e}"
            self._publish_status("ERROR")
            self.get_logger().error(str(e))

        return response

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PipelineOrchestrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
