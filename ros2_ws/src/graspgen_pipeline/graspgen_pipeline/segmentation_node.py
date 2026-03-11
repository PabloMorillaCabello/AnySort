#!/usr/bin/env python3
"""
Segmentation Node - Uses SAM3 (Segment Anything Model 3) to segment objects
from RGB images based on a text prompt. Publishes binary masks.

SAM3 API reference (from facebook/sam3 repo):
  - build_sam3_image_model(bpe_path=...) or from_pretrained via transformers
  - Sam3Processor: set_image(), set_text_prompt()
  - Also available via: transformers.Sam3Model / transformers.Sam3Processor
"""
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from PIL import Image as PILImage


class SegmentationNode(Node):
    """SAM3 text-prompted segmentation ROS2 node."""

    def __init__(self):
        super().__init__("segmentation_node")

        # Parameters
        self.declare_parameter("model_path", "/opt/models/sam3")
        self.declare_parameter("text_prompt", "object")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("mask_threshold", 0.5)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("use_transformers_api", True)

        self.model_path = self.get_parameter("model_path").value
        self.text_prompt = self.get_parameter("text_prompt").value
        self.confidence = self.get_parameter("confidence_threshold").value
        self.mask_threshold = self.get_parameter("mask_threshold").value
        self.device = self.get_parameter("device").value
        self.use_transformers = self.get_parameter("use_transformers_api").value

        self.bridge = CvBridge()
        self.model = None
        self.processor = None

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

        # Load model
        self._load_model()

        self.get_logger().info(
            f"Segmentation node ready | prompt: '{self.text_prompt}' | device: {self.device}"
        )

    def _load_model(self):
        """Load the SAM3 model."""
        self.get_logger().info("Loading SAM3 model...")

        if self.use_transformers:
            # ---------- Transformers API (recommended) ----------
            from transformers import Sam3Processor as TFSam3Processor
            from transformers import Sam3Model

            self.processor = TFSam3Processor.from_pretrained("facebook/sam3")
            self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
            self.model.eval()
            self.get_logger().info("SAM3 loaded via Transformers API.")
        else:
            # ---------- Native SAM3 API ----------
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor as NativeSam3Processor

            self.model = build_sam3_image_model(load_from_HF=True)
            self.model.to(self.device)
            self.model.eval()
            self.processor = NativeSam3Processor(
                self.model, confidence_threshold=self.confidence
            )
            self.get_logger().info("SAM3 loaded via native API.")

    def prompt_callback(self, msg):
        """Update text prompt dynamically via ROS topic."""
        self.text_prompt = msg.data
        self.get_logger().info(f"Text prompt updated to: '{self.text_prompt}'")

    def rgb_callback(self, msg):
        """Process incoming RGB image with SAM3 text-prompted segmentation."""
        if self.model is None:
            return

        try:
            # Convert ROS Image → PIL Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            pil_image = PILImage.fromarray(cv_image)

            if self.use_transformers:
                mask = self._infer_transformers(pil_image)
            else:
                mask = self._infer_native(pil_image)

            if mask is not None:
                # Publish binary mask (0 or 255)
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_msg = self.bridge.cv2_to_imgmsg(mask_uint8, "mono8")
                mask_msg.header = msg.header
                self.mask_pub.publish(mask_msg)

                # Publish visualization overlay
                viz = cv_image.copy()
                viz[mask > 0] = (viz[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
                viz_msg = self.bridge.cv2_to_imgmsg(viz, "rgb8")
                viz_msg.header = msg.header
                self.viz_pub.publish(viz_msg)

        except Exception as e:
            self.get_logger().error(f"Segmentation failed: {e}", throttle_duration_sec=5.0)

    def _infer_transformers(self, pil_image):
        """Run SAM3 inference via HuggingFace Transformers API."""
        inputs = self.processor(
            images=pil_image, text=self.text_prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.confidence,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        # Combine all instance masks into a single binary mask
        if len(results["segments_info"]) == 0:
            self.get_logger().warn(
                f"No objects found for prompt: '{self.text_prompt}'",
                throttle_duration_sec=5.0,
            )
            return None

        combined_mask = np.zeros(
            (pil_image.height, pil_image.width), dtype=np.uint8
        )
        segmentation = results["segmentation"].cpu().numpy()
        for seg_info in results["segments_info"]:
            combined_mask[segmentation == seg_info["id"]] = 1

        return combined_mask

    def _infer_native(self, pil_image):
        """Run SAM3 inference via native sam3 API."""
        state = self.processor.set_image(pil_image)
        self.processor.set_text_prompt(state=state, prompt=self.text_prompt)

        # Extract masks from state
        # The native API stores results in the inference state
        masks = state.get("masks", None)
        if masks is None or len(masks) == 0:
            return None

        # Combine all masks
        combined = np.zeros_like(masks[0], dtype=np.uint8)
        for m in masks:
            combined = np.maximum(combined, m.astype(np.uint8))
        return combined


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
