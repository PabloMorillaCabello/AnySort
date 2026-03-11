#!/usr/bin/env python3
"""
Test: SAM3 model loading and inference on a sample image.
Run inside the Docker container using the SAM3 Python 3.12 venv.

Usage:
  /opt/sam3env/bin/python scripts/test_sam3.py
  /opt/sam3env/bin/python scripts/test_sam3.py --image path/to/test.jpg --prompt "cup"

NOTE: This script must be run with /opt/sam3env/bin/python (Python 3.12),
      NOT with the system python3 (Python 3.10).
"""
import argparse
import sys
import time
import numpy as np


def check_python_version():
    """Ensure we are running in the correct Python environment."""
    v = sys.version_info
    if v.major == 3 and v.minor >= 12:
        print(f"  [PASS] Python {v.major}.{v.minor}.{v.micro} (>= 3.12 required for SAM3)")
    else:
        print(f"  [FAIL] Python {v.major}.{v.minor}.{v.micro} — SAM3 requires Python 3.12+")
        print(f"         Run this script with: /opt/sam3env/bin/python {sys.argv[0]}")
        sys.exit(1)


def test_imports():
    """Test that SAM3 packages can be imported."""
    print("[TEST] Importing SAM3 packages...")
    try:
        from sam3 import build_sam3_image_model
        print("  [PASS] sam3.build_sam3_image_model")
    except ImportError as e:
        print(f"  [FAIL] sam3 native API: {e}")

    try:
        from transformers import Sam3Processor, Sam3Model
        print("  [PASS] transformers Sam3Model + Sam3Processor")
    except ImportError as e:
        print(f"  [WARN] transformers SAM3: {e}")


def test_model_loading():
    """Test loading the SAM3 model."""
    import torch
    print(f"\n[TEST] Loading SAM3 model (device: cuda:{0 if torch.cuda.is_available() else 'cpu'})...")

    try:
        from transformers import Sam3Processor, Sam3Model

        t0 = time.time()
        processor = Sam3Processor.from_pretrained("facebook/sam3")
        model = Sam3Model.from_pretrained("facebook/sam3")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        load_time = time.time() - t0

        print(f"  [PASS] Model loaded in {load_time:.1f}s")
        print(f"  [INFO] Device: {device}")
        print(f"  [INFO] Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        return processor, model, device
    except Exception as e:
        print(f"  [FAIL] Model loading failed: {e}")
        return None, None, None


def test_inference(processor, model, device, image_path=None, text_prompt="object"):
    """Test SAM3 inference on a sample image."""
    import torch
    from PIL import Image

    print(f"\n[TEST] Running inference with prompt: '{text_prompt}'...")

    if image_path:
        img = Image.open(image_path).convert("RGB")
        print(f"  [INFO] Using image: {image_path} ({img.size})")
    else:
        # Create a synthetic test image (colored rectangles)
        img_array = np.zeros((480, 640, 3), dtype=np.uint8)
        img_array[100:300, 200:400] = [255, 0, 0]   # Red rectangle
        img_array[150:350, 300:500] = [0, 255, 0]    # Green rectangle
        img = Image.fromarray(img_array)
        print("  [INFO] Using synthetic test image (640x480)")

    try:
        t0 = time.time()
        inputs = processor(images=img, text=text_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        inference_time = time.time() - t0
        n_segments = len(results["segments_info"])
        print(f"  [PASS] Inference completed in {inference_time:.2f}s")
        print(f"  [INFO] Found {n_segments} segment(s)")

        for i, seg in enumerate(results["segments_info"]):
            score = seg.get("score", "N/A")
            label = seg.get("label_id", "N/A")
            print(f"  [INFO]   Segment {i}: score={score}, label_id={label}")

        return True
    except Exception as e:
        print(f"  [FAIL] Inference failed: {e}")
        return False


def test_gpu_memory():
    """Report GPU memory usage."""
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"\n[INFO] GPU Memory: {allocated:.1f}GB allocated, "
              f"{reserved:.1f}GB reserved, {total:.1f}GB total")


def main():
    parser = argparse.ArgumentParser(description="Test SAM3 model")
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    parser.add_argument("--prompt", type=str, default="object", help="Text prompt")
    args = parser.parse_args()

    print("=" * 50)
    print("  SAM3 Model Test (Python 3.12 venv)")
    print("=" * 50)

    check_python_version()
    test_imports()
    processor, model, device = test_model_loading()

    if model is not None:
        success = test_inference(processor, model, device, args.image, args.prompt)
        test_gpu_memory()
        sys.exit(0 if success else 1)
    else:
        print("\n[SKIP] Skipping inference (model not loaded)")
        sys.exit(1)


if __name__ == "__main__":
    main()
