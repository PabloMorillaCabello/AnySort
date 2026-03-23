#!/usr/bin/env python3
"""
SAM3 single-shot segmentation helper — runs under Python 3.12 SAM3 venv.

Called by demo_orbbec_gemini2.py (Python 3.10) via subprocess to bridge the
Python version gap between GraspGen (3.10) and SAM3 (3.12).

Usage:
  /opt/sam3env/bin/python scripts/sam3_segment_once.py \
      --image /path/to/image.png \
      --prompt "red mug" \
      --mask_out /path/to/mask.npy

Output:
  Binary mask saved as .npy uint8 array (H, W) with values {0, 1}.
  Exits 0 on success, 1 on failure.
"""

import argparse
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 single-shot segmentation")
    parser.add_argument("--image", required=True, help="Path to input RGB image (PNG/JPG)")
    parser.add_argument("--prompt", required=True, help="Text prompt for SAM3 segmentation")
    parser.add_argument("--mask_out", required=True, help="Path to save output mask (.npy)")
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0)")
    parser.add_argument("--no-transformers", dest="use_transformers", action="store_false",
                        default=True, help="Use native SAM3 API instead of Transformers")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", default=True,
                        help="Use FP32 instead of FP16")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for detections")
    return parser.parse_args()


def load_rgb(image_path: str) -> np.ndarray:
    """Load image as RGB uint8 numpy array (H, W, 3)."""
    from PIL import Image as PILImage
    img = PILImage.open(image_path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def run_inference(rgb: np.ndarray, prompt: str, device: str,
                  use_transformers: bool, fp16: bool,
                  confidence: float) -> np.ndarray:
    """Run SAM3 inference and return binary mask (H, W) uint8 {0, 1}."""
    import torch
    from PIL import Image as PILImage

    pil_image = PILImage.fromarray(rgb)
    h, w = rgb.shape[:2]

    if use_transformers:
        from transformers import Sam3Processor, Sam3Model

        processor = Sam3Processor.from_pretrained("facebook/sam3")
        dtype = torch.float16 if fp16 else torch.float32
        model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=dtype).to(device)
        model.eval()

        inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=confidence,
            target_sizes=[(h, w)],
        )[0]

        masks = results.get("masks", None)
        scores = results.get("scores", None)

        combined = np.zeros((h, w), dtype=np.uint8)
        if masks is not None and len(masks) > 0:
            for i, mask in enumerate(masks):
                score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
                if score >= confidence:
                    combined[mask.cpu().numpy().astype(bool)] = 1
        return combined

    else:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor as NativeProc

        model = build_sam3_image_model(load_from_HF=True)
        model.to(device)
        model.eval()
        processor = NativeProc(model, confidence_threshold=confidence)

        state = processor.set_image(pil_image)
        processor.set_text_prompt(state=state, prompt=prompt)
        masks = state.get("masks", None)

        combined = np.zeros((h, w), dtype=np.uint8)
        if masks is not None and len(masks) > 0:
            for m in masks:
                combined = np.maximum(combined, (m > 0).astype(np.uint8))
        return combined


def main():
    args = parse_args()

    print(f"[sam3_segment_once] image={args.image}", flush=True)
    print(f"[sam3_segment_once] prompt='{args.prompt}'", flush=True)
    print(f"[sam3_segment_once] mask_out={args.mask_out}", flush=True)

    try:
        rgb = load_rgb(args.image)
    except Exception as e:
        print(f"[sam3_segment_once] ERROR loading image: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[sam3_segment_once] Image loaded: {rgb.shape}", flush=True)

    try:
        mask = run_inference(
            rgb,
            prompt=args.prompt,
            device=args.device,
            use_transformers=args.use_transformers,
            fp16=args.fp16,
            confidence=args.confidence,
        )
    except Exception as e:
        print(f"[sam3_segment_once] ERROR during inference: {e}", file=sys.stderr)
        sys.exit(1)

    n_pixels = int(np.sum(mask > 0))
    print(f"[sam3_segment_once] Mask: {mask.shape}, {n_pixels} masked pixels", flush=True)

    if n_pixels == 0:
        print("[sam3_segment_once] WARNING: no pixels detected for the given prompt",
              file=sys.stderr)

    try:
        np.save(args.mask_out, mask)
        print(f"[sam3_segment_once] Mask saved to {args.mask_out}", flush=True)
    except Exception as e:
        print(f"[sam3_segment_once] ERROR saving mask: {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
