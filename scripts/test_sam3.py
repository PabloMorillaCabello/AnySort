#!/usr/bin/env python3
"""
Test: SAM3 model loading and inference with visual output.
Run inside the Docker container using the SAM3 Python 3.12 venv.

Displays the original image side-by-side with the segmentation mask overlay
using matplotlib (requires X11 forwarding, e.g. XLaunch on Windows).

Usage:
  /opt/sam3env/bin/python scripts/test_sam3.py
  /opt/sam3env/bin/python scripts/test_sam3.py --image path/to/test.jpg --prompt "cup"
  /opt/sam3env/bin/python scripts/test_sam3.py --no-display   # skip visualization

NOTE: This script must be run with /opt/sam3env/bin/python (Python 3.12),
      NOT with the system python3 (Python 3.10).
"""
import argparse
import sys
import time
import numpy as np


# ---------------------------------------------------------------------------
# Colormap for overlaying multiple masks with distinct colors
# ---------------------------------------------------------------------------
MASK_COLORS = [
    (0.12, 0.47, 0.71, 0.50),   # blue
    (1.00, 0.50, 0.05, 0.50),   # orange
    (0.17, 0.63, 0.17, 0.50),   # green
    (0.84, 0.15, 0.16, 0.50),   # red
    (0.58, 0.40, 0.74, 0.50),   # purple
    (0.55, 0.34, 0.29, 0.50),   # brown
    (0.89, 0.47, 0.76, 0.50),   # pink
    (0.50, 0.50, 0.50, 0.50),   # grey
    (0.74, 0.74, 0.13, 0.50),   # olive
    (0.09, 0.75, 0.81, 0.50),   # cyan
]


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
        from transformers import Sam3Processor, Sam3Model
        print("  [PASS] transformers Sam3Model + Sam3Processor")
    except ImportError as e:
        print(f"  [FAIL] transformers SAM3: {e}")
        sys.exit(1)


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


def test_inference(processor, model, device, image_path=None, text_prompt="object",
                   show_display=True):
    """Test SAM3 inference and optionally visualize results."""
    import torch
    from PIL import Image

    print(f"\n[TEST] Running inference with prompt: '{text_prompt}'...")

    if image_path:
        img = Image.open(image_path).convert("RGB")
        print(f"  [INFO] Using image: {image_path} ({img.size})")
    else:
        # Create a synthetic test image (colored rectangles on grey background)
        img_array = np.full((480, 640, 3), 40, dtype=np.uint8)
        img_array[100:300, 200:400] = [255, 0, 0]    # Red rectangle
        img_array[150:350, 300:500] = [0, 255, 0]     # Green rectangle
        img_array[50:150, 50:180] = [0, 100, 255]     # Blue rectangle
        img = Image.fromarray(img_array)
        print("  [INFO] Using synthetic test image (640x480)")

    try:
        t0 = time.time()
        h, w = img.size[1], img.size[0]  # PIL size is (W, H)
        img_np = np.array(img)

        inputs = processor(images=img, text=text_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # API returns: {"scores": [...], "boxes": [...], "masks": Tensor(N, H, W)}
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            target_sizes=[(h, w)],
        )[0]

        inference_time = time.time() - t0
        masks_tensor = results.get("masks", [])
        scores = results.get("scores", [])
        boxes = results.get("boxes", [])
        n_segments = len(masks_tensor)
        print(f"  [PASS] Inference completed in {inference_time:.2f}s")
        print(f"  [INFO] Found {n_segments} segment(s)")

        # Convert masks to numpy for display
        masks_np = []
        for i in range(n_segments):
            score = float(scores[i]) if i < len(scores) else 0.0
            box = boxes[i].tolist() if i < len(boxes) else []
            mask_i = masks_tensor[i].cpu().numpy().astype(bool)
            mask_pixels = int(mask_i.sum())
            masks_np.append(mask_i)
            print(f"  [INFO]   Segment {i}: score={score:.3f}, "
                  f"box={[f'{v:.0f}' for v in box]}, mask_pixels={mask_pixels}")

        # ---- Visualization ----
        if show_display and n_segments > 0:
            visualize(img_np, masks_np, scores, boxes, text_prompt, inference_time)

        return True
    except Exception as e:
        print(f"  [FAIL] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize(img_np, masks, scores, boxes, prompt, inference_time):
    """Display original image and mask overlay side by side.

    Uses matplotlib Agg backend to render to file, then cv2.imshow()
    for X11 display (no tkinter needed).
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (always works)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    h, w = img_np.shape[:2]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"SAM3 Segmentation — prompt: \"{prompt}\"  ({inference_time:.2f}s)",
                 fontsize=14, fontweight="bold")

    # --- Panel 1: Original Image ---
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    # --- Panel 2: Mask Overlay on Image ---
    overlay = img_np.astype(np.float32) / 255.0
    for i, mask in enumerate(masks):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        rgba = np.array(color)
        overlay[mask] = overlay[mask] * (1 - rgba[3]) + rgba[:3] * rgba[3]

    axes[1].imshow(np.clip(overlay, 0, 1))
    axes[1].set_title(f"Mask Overlay ({len(masks)} segments)", fontsize=12)
    axes[1].axis("off")

    # Draw bounding boxes
    for i in range(len(boxes)):
        box = boxes[i].tolist() if hasattr(boxes[i], "tolist") else boxes[i]
        if len(box) == 4:
            x1, y1, x2, y2 = box
            color_rgb = MASK_COLORS[i % len(MASK_COLORS)][:3]
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color_rgb, facecolor="none"
            )
            axes[1].add_patch(rect)
            score = float(scores[i]) if i < len(scores) else 0.0
            axes[1].text(x1, y1 - 4, f"{score:.2f}", color=color_rgb,
                         fontsize=9, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

    # --- Panel 3: Masks Only (binary, colored per instance) ---
    mask_canvas = np.zeros((h, w, 3), dtype=np.float32)
    for i, mask in enumerate(masks):
        color = MASK_COLORS[i % len(MASK_COLORS)][:3]
        mask_canvas[mask] = color

    axes[2].imshow(mask_canvas)
    axes[2].set_title("Instance Masks", fontsize=12)
    axes[2].axis("off")

    # Legend
    legend_patches = []
    for i in range(len(masks)):
        score = float(scores[i]) if i < len(scores) else 0.0
        color_rgb = MASK_COLORS[i % len(MASK_COLORS)][:3]
        legend_patches.append(
            mpatches.Patch(color=color_rgb, label=f"Seg {i} ({score:.2f})")
        )
    if legend_patches:
        axes[2].legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()

    # Save to file (results/ is volume-mounted to host)
    import os
    import subprocess
    os.makedirs("/ros2_ws/results", exist_ok=True)
    save_path = "/ros2_ws/results/sam3_test_result.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [INFO] Result saved to {save_path}")
    print(f"  [INFO] Open from your host at: results/sam3_test_result.png")

    # Display with feh (lightweight pure-X11 viewer, no Qt/Tk issues)
    try:
        print(f"  [INFO] Opening display window (close it to continue)...")
        subprocess.run(["feh", "--auto-zoom", "--title", "SAM3 Result", save_path],
                       check=True)
    except FileNotFoundError:
        print(f"  [WARN] feh not installed. Install with: apt-get install feh")
    except subprocess.CalledProcessError:
        print(f"  [WARN] Could not open display. Check DISPLAY={os.environ.get('DISPLAY', 'NOT SET')}")
        print(f"         Make sure XLaunch is running with 'Disable access control' checked.")


def test_gpu_memory():
    """Report GPU memory usage."""
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n[INFO] GPU Memory: {allocated:.1f}GB allocated, "
              f"{reserved:.1f}GB reserved, {total:.1f}GB total")


def main():
    parser = argparse.ArgumentParser(description="Test SAM3 model")
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    parser.add_argument("--prompt", type=str, default="Red rectangle", help="Text prompt")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip visualization (no X11 needed)")
    args = parser.parse_args()

    print("=" * 50)
    print("  SAM3 Model Test (Python 3.12 venv)")
    print("=" * 50)

    check_python_version()
    test_imports()
    processor, model, device = test_model_loading()

    if model is not None:
        success = test_inference(
            processor, model, device,
            args.image, args.prompt,
            show_display=not args.no_display,
        )
        test_gpu_memory()
        sys.exit(0 if success else 1)
    else:
        print("\n[SKIP] Skipping inference (model not loaded)")
        sys.exit(1)


if __name__ == "__main__":
    main()
