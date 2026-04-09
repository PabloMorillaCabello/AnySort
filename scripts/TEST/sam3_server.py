#!/usr/bin/env python3
"""
SAM3 Persistent Server — runs under Python 3.12 SAM3 venv.

Loads model.safetensors ONCE at startup, then handles repeated inference
requests via stdin/stdout JSON protocol.

Protocol:
  Request  (stdin,  one JSON line):  {"image": "/path/img.png", "prompt": "red mug", "mask_out": "/path/mask.npy"}
  Response (stdout, one JSON line):  {"status": "ok",    "mask": "/path/mask.npy"}
                                 or  {"status": "error", "error": "message"}

After model is loaded, prints:  {"status": "ready"}
This signals the parent process it can start sending requests.

Usage:
  /opt/sam3env/bin/python3 scripts/TEST/sam3_server.py [--device cuda:0] [--no-fp16] [--no-transformers]
"""

import argparse
import json
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 persistent inference server")
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0)")
    parser.add_argument("--no-transformers", dest="use_transformers", action="store_false",
                        default=True, help="Use native SAM3 API instead of HuggingFace Transformers")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", default=True,
                        help="Use FP32 instead of FP16")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for detections")
    return parser.parse_args()


def load_model(device: str, use_transformers: bool, fp16: bool):
    """Load SAM3 model and return (model, processor) tuple. Called once at startup."""
    import torch
    dtype = torch.float16 if fp16 else torch.float32

    if use_transformers:
        import os
        from transformers import Sam3Processor, Sam3Model
        _local = "/opt/models/sam3"
        _model_id = _local if os.path.isdir(_local) else "facebook/sam3"
        print(f"[sam3_server] Loading Sam3Processor from {_model_id}...", file=sys.stderr, flush=True)
        processor = Sam3Processor.from_pretrained(_model_id, local_files_only=os.path.isdir(_local))
        print(f"[sam3_server] Loading Sam3Model from {_model_id}...", file=sys.stderr, flush=True)
        model = Sam3Model.from_pretrained(_model_id, torch_dtype=dtype, local_files_only=os.path.isdir(_local)).to(device)
        model.eval()
        print(f"[sam3_server] Model loaded on {device} dtype={dtype}", file=sys.stderr, flush=True)
        return model, processor, "transformers"

    else:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor as NativeProc
        print("[sam3_server] Loading SAM3 native model...", file=sys.stderr, flush=True)
        model = build_sam3_image_model(load_from_HF=True)
        model.to(device)
        model.eval()
        processor = NativeProc(model)
        print(f"[sam3_server] Native model loaded on {device}", file=sys.stderr, flush=True)
        return model, processor, "native"


def run_inference(model, processor, backend: str, rgb: np.ndarray,
                  prompt: str, device: str, confidence: float) -> np.ndarray:
    """Run inference with the already-loaded model. Returns binary mask (H,W) uint8."""
    import torch
    from PIL import Image as PILImage

    pil_image = PILImage.fromarray(rgb)
    h, w = rgb.shape[:2]

    if backend == "transformers":
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

    else:  # native
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

    # Load model once — this is the slow part
    try:
        model, processor, backend = load_model(args.device, args.use_transformers, args.fp16)
    except Exception as e:
        # Signal failure so parent doesn't hang waiting for "ready"
        sys.stdout.write(json.dumps({"status": "load_error", "error": str(e)}) + "\n")
        sys.stdout.flush()
        sys.exit(1)

    # Signal parent: ready to receive requests
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()
    print("[sam3_server] Ready. Waiting for requests on stdin...", file=sys.stderr, flush=True)

    # Main request loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(json.dumps({"status": "error", "error": f"JSON decode: {e}"}) + "\n")
            sys.stdout.flush()
            continue

        image_path = req.get("image", "")
        prompt = req.get("prompt", "")
        mask_out = req.get("mask_out", "")

        if not image_path or not prompt or not mask_out:
            sys.stdout.write(json.dumps({"status": "error", "error": "Missing image/prompt/mask_out"}) + "\n")
            sys.stdout.flush()
            continue

        print(f"[sam3_server] Request: prompt='{prompt}' image={image_path}", file=sys.stderr, flush=True)

        try:
            from PIL import Image as PILImage
            rgb = np.array(PILImage.open(image_path).convert("RGB"), dtype=np.uint8)
        except Exception as e:
            sys.stdout.write(json.dumps({"status": "error", "error": f"Image load: {e}"}) + "\n")
            sys.stdout.flush()
            continue

        try:
            mask = run_inference(model, processor, backend, rgb, prompt, args.device, args.confidence)
        except Exception as e:
            sys.stdout.write(json.dumps({"status": "error", "error": f"Inference: {e}"}) + "\n")
            sys.stdout.flush()
            continue

        try:
            np.save(mask_out, mask)
        except Exception as e:
            sys.stdout.write(json.dumps({"status": "error", "error": f"Mask save: {e}"}) + "\n")
            sys.stdout.flush()
            continue

        n_pixels = int(np.sum(mask > 0))
        print(f"[sam3_server] Mask saved: {mask.shape}, {n_pixels} pixels", file=sys.stderr, flush=True)
        sys.stdout.write(json.dumps({"status": "ok", "mask": mask_out, "pixels": n_pixels}) + "\n")
        sys.stdout.flush()

    print("[sam3_server] stdin closed, exiting.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
