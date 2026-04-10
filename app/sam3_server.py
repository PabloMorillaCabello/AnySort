#!/usr/bin/env python3
"""
SAM3 Persistent Unix-Socket Server — runs under Python 3.12 SAM3 venv.

Loads model weights ONCE at startup, then handles repeated inference
requests over a Unix domain socket.

Protocol (per connection):
  Client → Server:
    1. JSON header line:  {"width": W, "height": H, "prompt": "text", "size": N}\n
    2. N bytes of raw RGB24 pixel data (H×W×3, uint8, row-major)

  Server → Client:
    1. JSON response line: {"ok": true, "size": M, "num_masks": K}\n   (M = K*H*W bytes)
                       or: {"ok": false, "error": "message"}\n
    2. M bytes of mask data (K×H×W uint8, concatenated)

After model is loaded, the socket file appears at --socket path.
This signals the parent process it can start sending requests.

Usage:
  /opt/sam3env/bin/python3.12 app/sam3_server.py [--socket /tmp/sam3_server.sock]
                                                  [--device cuda:0]
                                                  [--no-fp16]
                                                  [--no-transformers]
                                                  [--confidence 0.5]
"""

import argparse
import json
import os
import socket
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 persistent Unix-socket inference server")
    parser.add_argument("--socket", default="/tmp/sam3_server.sock",
                        help="Unix socket path (default: /tmp/sam3_server.sock)")
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0)")
    parser.add_argument("--no-transformers", dest="use_transformers", action="store_false",
                        default=True, help="Use native SAM3 API instead of HuggingFace Transformers")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", default=True,
                        help="Use FP32 instead of FP16")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for detections (default: 0.5)")
    return parser.parse_args()


def load_model(device: str, use_transformers: bool, fp16: bool):
    """Load SAM3 model once at startup. Returns (model, processor, backend_str)."""
    import torch
    dtype = torch.float16 if fp16 else torch.float32

    if use_transformers:
        from transformers import Sam3Processor, Sam3Model
        _local = "/opt/models/sam3"
        _model_id = _local if os.path.isdir(_local) else "facebook/sam3"
        print(f"[sam3_server] Loading Sam3Processor from {_model_id}...", file=sys.stderr, flush=True)
        processor = Sam3Processor.from_pretrained(_model_id, local_files_only=os.path.isdir(_local))
        print(f"[sam3_server] Loading Sam3Model from {_model_id}...", file=sys.stderr, flush=True)
        model = Sam3Model.from_pretrained(
            _model_id, torch_dtype=dtype, local_files_only=os.path.isdir(_local)
        ).to(device)
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


def run_inference(model, processor, backend: str, rgb, prompt: str,
                  device: str, confidence: float):
    """Run segmentation. Returns list of (H,W) uint8 mask arrays."""
    import numpy as np
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

        masks_t = results.get("masks", None)
        scores   = results.get("scores", None)

        out = []
        if masks_t is not None and len(masks_t) > 0:
            for i, m in enumerate(masks_t):
                score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
                if score >= confidence:
                    out.append(m.cpu().numpy().astype(np.uint8))
        return out

    else:  # native
        state = processor.set_image(pil_image)
        processor.set_text_prompt(state=state, prompt=prompt)
        masks = state.get("masks", None)
        if masks is None:
            return []
        return [(m > 0).astype(np.uint8) for m in masks]


def _recv_exactly(conn, n):
    """Read exactly n bytes from a socket connection."""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(n - len(buf), 65536))
        if not chunk:
            raise ConnectionError("Client disconnected mid-stream")
        buf.extend(chunk)
    return bytes(buf)


def _recv_line(conn):
    """Read one newline-terminated line from socket."""
    buf = bytearray()
    while True:
        b = conn.recv(1)
        if not b:
            raise ConnectionError("Client disconnected")
        if b == b"\n":
            break
        buf.extend(b)
    return buf.decode()


def handle_client(conn, model, processor, backend, device, confidence):
    """Handle one client connection: read request, run inference, send masks."""
    import numpy as np

    try:
        line = _recv_line(conn)
        req = json.loads(line)
        w, h = req["width"], req["height"]
        prompt = req["prompt"]
        size = req["size"]

        rgb_bytes = _recv_exactly(conn, size)
        rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(h, w, 3)

        print(f"[sam3_server] Request: prompt='{prompt}' size=({w}x{h})", file=sys.stderr, flush=True)

        masks = run_inference(model, processor, backend, rgb, prompt, device, confidence)

        if not masks:
            resp = json.dumps({"ok": True, "size": 0, "num_masks": 0}) + "\n"
            conn.sendall(resp.encode())
            print(f"[sam3_server] No masks found for prompt='{prompt}'", file=sys.stderr, flush=True)
            return

        mask_data = np.concatenate([m.reshape(1, h, w) for m in masks], axis=0).astype(np.uint8)
        flat = mask_data.tobytes()
        resp = json.dumps({"ok": True, "size": len(flat), "num_masks": len(masks)}) + "\n"
        conn.sendall(resp.encode())
        conn.sendall(flat)
        print(f"[sam3_server] Sent {len(masks)} mask(s), {len(flat)} bytes", file=sys.stderr, flush=True)

    except Exception as e:
        print(f"[sam3_server] Error handling client: {e}", file=sys.stderr, flush=True)
        try:
            err = json.dumps({"ok": False, "error": str(e)}) + "\n"
            conn.sendall(err.encode())
        except Exception:
            pass
    finally:
        conn.close()


def main():
    args = parse_args()

    # Load model — slow on first run
    try:
        model, processor, backend = load_model(args.device, args.use_transformers, args.fp16)
    except Exception as e:
        print(f"[sam3_server] FATAL: model load failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Remove stale socket file if it exists
    sock_path = args.socket
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    # Create Unix socket server — socket file appearing signals "ready" to parent
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(5)
    print(f"[sam3_server] Ready. Listening on {sock_path}", file=sys.stderr, flush=True)

    try:
        while True:
            conn, _ = server.accept()
            handle_client(conn, model, processor, backend, args.device, args.confidence)
    except KeyboardInterrupt:
        print("[sam3_server] Interrupted, shutting down.", file=sys.stderr, flush=True)
    finally:
        server.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)


if __name__ == "__main__":
    main()
