#!/usr/bin/env python3
"""
Segmentation Node — ROS2 thin client for the SAM3 inference server.

Architecture:
  SAM3 requires Python 3.12 but ROS2 Humble runs on Python 3.10.
  This node does NOT import SAM3 directly. Instead, it communicates with
  a persistent sam3_server.py process (running under /opt/sam3env/bin/python)
  via a Unix domain socket.

Protocol (per request):
  1. Send JSON header: {"width": W, "height": H, "prompt": "text", "size": N}
  2. Send N bytes of raw RGB uint8 (H*W*3)
  3. Receive JSON header: {"ok": true, "width": W, "height": H, "size": N}
  4. Receive N bytes of mask uint8 (H*W, values 0 or 255)
"""
import json
import socket
import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class SegmentationNode(Node):
    """SAM3 text-prompted segmentation ROS2 node (socket client)."""

    def __init__(self):
        super().__init__("segmentation_node")

        # Parameters
        self.declare_parameter("model_path", "/opt/models/sam3")
        self.declare_parameter("text_prompt", "object")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("mask_threshold", 0.5)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("use_half_precision", False)
        self.declare_parameter("use_transformers_api", True)
        self.declare_parameter("sam3_socket", "/tmp/sam3_server.sock")
        self.declare_parameter("sam3_server_script", "/ros2_ws/scripts/sam3_server.py")
        self.declare_parameter("sam3_python", "/opt/sam3env/bin/python")
        self.declare_parameter("auto_start_server", True)

        self.text_prompt = self.get_parameter("text_prompt").value
        self.sock_path = self.get_parameter("sam3_socket").value
        self.bridge = CvBridge()
        self._conn = None
        self._server_proc = None

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.rgb_callback, 10
        )
        self.prompt_sub = self.create_subscription(
            String, "/segmentation/set_prompt", self.prompt_callback, 10
        )

        # Publishers
        self.mask_pub = self.create_publisher(Image, "/segmentation/mask", 10)
        self.viz_pub = self.create_publisher(Image, "/segmentation/visualization", 10)

        # Auto-start the SAM3 server if requested
        if self.get_parameter("auto_start_server").value:
            self._start_server()

        # Connect to SAM3 server
        self._connect()

        self.get_logger().info(
            f"Segmentation node ready | prompt: '{self.text_prompt}' | "
            f"server: {self.sock_path}"
        )

    # ------------------------------------------------------------------
    # Server management
    # ------------------------------------------------------------------
    def _start_server(self):
        """Launch the SAM3 server as a subprocess (Python 3.12 venv)."""
        python_bin = self.get_parameter("sam3_python").value
        script = self.get_parameter("sam3_server_script").value
        device = self.get_parameter("device").value
        use_tf = self.get_parameter("use_transformers_api").value
        use_half = self.get_parameter("use_half_precision").value
        confidence = self.get_parameter("confidence_threshold").value
        mask_thresh = self.get_parameter("mask_threshold").value

        cmd = [
            python_bin, script,
            "--socket", self.sock_path,
            "--device", device,
            "--confidence", str(confidence),
            "--mask-threshold", str(mask_thresh),
        ]
        if use_tf:
            cmd.append("--use-transformers")
        else:
            cmd.append("--no-transformers")
        if use_half:
            cmd.append("--fp16")
        else:
            cmd.append("--no-fp16")

        self.get_logger().info(f"Starting SAM3 server: {' '.join(cmd)}")
        self._server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def _connect(self, retries: int = 30, delay: float = 2.0):
        """Connect to the SAM3 server socket with retries."""
        for attempt in range(retries):
            try:
                self._conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self._conn.connect(self.sock_path)
                self.get_logger().info("Connected to SAM3 server.")
                return
            except (FileNotFoundError, ConnectionRefusedError):
                if attempt < retries - 1:
                    self.get_logger().info(
                        f"Waiting for SAM3 server... ({attempt + 1}/{retries})"
                    )
                    time.sleep(delay)
                    self._conn = None
                else:
                    self.get_logger().error(
                        "Could not connect to SAM3 server after "
                        f"{retries * delay:.0f}s. Is sam3_server.py running?"
                    )

    def _reconnect(self):
        """Attempt to reconnect after a broken pipe."""
        self.get_logger().warn("Lost connection to SAM3 server, reconnecting...")
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        self._connect(retries=5, delay=2.0)

    # ------------------------------------------------------------------
    # Socket protocol helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _recv_exactly(conn: socket.socket, n: int) -> bytes:
        chunks = []
        received = 0
        while received < n:
            chunk = conn.recv(min(n - received, 65536))
            if not chunk:
                raise ConnectionError("SAM3 server disconnected")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    @staticmethod
    def _recv_json_line(conn: socket.socket) -> dict:
        buf = b""
        while True:
            byte = conn.recv(1)
            if not byte:
                raise ConnectionError("SAM3 server disconnected")
            if byte == b"\n":
                break
            buf += byte
        return json.loads(buf.decode("utf-8"))

    @staticmethod
    def _send_json_line(conn: socket.socket, obj: dict):
        data = (json.dumps(obj) + "\n").encode("utf-8")
        conn.sendall(data)

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def prompt_callback(self, msg):
        """Update text prompt dynamically via ROS topic."""
        self.text_prompt = msg.data
        self.get_logger().info(f"Text prompt updated to: '{self.text_prompt}'")

    def rgb_callback(self, msg):
        """Process incoming RGB image through SAM3 server."""
        if self._conn is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            h, w = cv_image.shape[:2]
            rgb_bytes = cv_image.tobytes()

            # Send request to SAM3 server
            self._send_json_line(self._conn, {
                "width": w,
                "height": h,
                "prompt": self.text_prompt,
                "size": len(rgb_bytes),
            })
            self._conn.sendall(rgb_bytes)

            # Receive response
            resp = self._recv_json_line(self._conn)
            if not resp.get("ok", False):
                self.get_logger().warn(
                    f"SAM3 error: {resp.get('error', 'unknown')}",
                    throttle_duration_sec=5.0,
                )
                return

            mask_bytes = self._recv_exactly(self._conn, resp["size"])
            mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape((h, w))

            # Check if mask is all zeros (no detection)
            if mask.max() == 0:
                self.get_logger().warn(
                    f"No objects found for prompt: '{self.text_prompt}'",
                    throttle_duration_sec=5.0,
                )
                return

            # Publish binary mask
            mask_msg = self.bridge.cv2_to_imgmsg(mask, "mono8")
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)

            # Publish visualization overlay
            viz = cv_image.copy()
            mask_bool = mask > 0
            viz[mask_bool] = (
                viz[mask_bool] * 0.5 + np.array([0, 255, 0]) * 0.5
            ).astype(np.uint8)
            viz_msg = self.bridge.cv2_to_imgmsg(viz, "rgb8")
            viz_msg.header = msg.header
            self.viz_pub.publish(viz_msg)

        except ConnectionError:
            self._reconnect()
        except Exception as e:
            self.get_logger().error(
                f"Segmentation failed: {e}", throttle_duration_sec=5.0
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def destroy_node(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        if self._server_proc:
            self._server_proc.terminate()
            self._server_proc.wait(timeout=5)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
