#!/usr/bin/env python3
"""
SAM3 Inference Server — Runs under Python 3.12 venv (/opt/sam3env/bin/python).

Bridges the Python version gap:
  - ROS2 nodes run on Python 3.10 (system, Ubuntu 22.04 packages)
  - SAM3 requires Python 3.12+

Communication: Unix domain socket at /tmp/sam3_server.sock
Protocol (per request):
  1. Client sends JSON header (utf-8, newline-terminated):
       {"width": W, "height": H, "prompt": "text", "size": N}
     where N = byte length of the raw image payload that follows.
  2. Client sends N bytes of raw RGB uint8 data (H*W*3).
  3. Server replies with JSON header (newline-terminated):
       {"ok": true, "width": W, "height": H, "size": N}
     followed by N bytes of raw uint8 mask (H*W, values 0 or 255).
     On error: {"ok": false, "error": "message"}

Usage:
  /opt/sam3env/bin/python scripts/sam3_server.py [--socket /tmp/sam3_server.sock]
"""
import argparse
import json
import os
import signal
import socket
import struct
import sys
import threading
import numpy as np

# ---------------------------------------------------------------------------
# Globals (set in main after model load)
# ---------------------------------------------------------------------------
model = None
processor = None
device = None
use_transformers = True
use_half = True
confidence_threshold = 0.5
mask_threshold = 0.5


def load_model():
    """Load SAM3 model into GPU."""
    global model, processor
    import torch

    print(f"[sam3_server] Loading SAM3 on {device} "
          f"(transformers={use_transformers}, fp16={use_half}) ...")

    if use_transformers:
        from transformers import Sam3Processor as TFProc, Sam3Model

        processor = TFProc.from_pretrained("facebook/sam3")
        dtype = torch.float16 if use_half else torch.float32
        model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=dtype).to(device)
        model.eval()
        print("[sam3_server] SAM3 loaded via Transformers API.")
    else:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor as NativeProc

        model = build_sam3_image_model(load_from_HF=True)
        model.to(device)
        model.eval()
        processor = NativeProc(model, confidence_threshold=confidence_threshold)
        print("[sam3_server] SAM3 loaded via native API.")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def infer(rgb_array: np.ndarray, prompt: str) -> np.ndarray | None:
    """Run SAM3 on an RGB image with a text prompt.

    Returns a uint8 mask (H, W) with values 0 or 255, or None if nothing found.
    """
    import torch
    from PIL import Image as PILImage

    pil_image = PILImage.fromarray(rgb_array)

    if use_transformers:
        inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        h, w = rgb_array.shape[:2]
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=confidence_threshold,
            target_sizes=[(h, w)],
        )[0]
        # API returns: {"scores": [...], "boxes": [...], "masks": Tensor(N, H, W)}
        masks = results.get("masks", None)
        scores = results.get("scores", None)
        if masks is None or len(masks) == 0:
            return None
        # Filter by confidence and combine all masks into one
        combined = np.zeros((h, w), dtype=np.uint8)
        for i, mask in enumerate(masks):
            score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
            if score >= confidence_threshold:
                m = mask.cpu().numpy().astype(bool)
                combined[m] = 255
        return combined
    else:
        state = processor.set_image(pil_image)
        processor.set_text_prompt(state=state, prompt=prompt)
        masks = state.get("masks", None)
        if masks is None or len(masks) == 0:
            return None
        combined = np.zeros_like(masks[0], dtype=np.uint8)
        for m in masks:
            combined = np.maximum(combined, (m * 255).astype(np.uint8))
        return combined


# ---------------------------------------------------------------------------
# Socket protocol helpers
# ---------------------------------------------------------------------------
def recv_exactly(conn: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from a socket."""
    chunks = []
    received = 0
    while received < n:
        chunk = conn.recv(min(n - received, 65536))
        if not chunk:
            raise ConnectionError("Client disconnected")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def recv_json_line(conn: socket.socket) -> dict:
    """Receive a newline-terminated JSON object."""
    buf = b""
    while True:
        byte = conn.recv(1)
        if not byte:
            raise ConnectionError("Client disconnected")
        if byte == b"\n":
            break
        buf += byte
    return json.loads(buf.decode("utf-8"))


def send_json_line(conn: socket.socket, obj: dict):
    """Send a newline-terminated JSON object."""
    data = (json.dumps(obj) + "\n").encode("utf-8")
    conn.sendall(data)


# ---------------------------------------------------------------------------
# Client handler
# ---------------------------------------------------------------------------
def handle_client(conn: socket.socket, addr):
    """Handle a single client connection (may send multiple requests)."""
    try:
        while True:
            # 1. Read JSON header
            try:
                header = recv_json_line(conn)
            except (ConnectionError, json.JSONDecodeError):
                break

            w = header["width"]
            h = header["height"]
            prompt = header["prompt"]
            payload_size = header["size"]

            # 2. Read raw RGB bytes
            rgb_bytes = recv_exactly(conn, payload_size)
            rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((h, w, 3))

            # 3. Run inference
            try:
                mask = infer(rgb, prompt)
            except Exception as e:
                send_json_line(conn, {"ok": False, "error": str(e)})
                continue

            if mask is None:
                # No detections — send empty mask (all zeros)
                mask = np.zeros((h, w), dtype=np.uint8)

            mask_bytes = mask.tobytes()
            send_json_line(conn, {
                "ok": True,
                "width": w,
                "height": h,
                "size": len(mask_bytes),
            })
            conn.sendall(mask_bytes)

    except Exception as e:
        print(f"[sam3_server] Client error: {e}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global device, use_transformers, use_half, confidence_threshold, mask_threshold

    parser = argparse.ArgumentParser(description="SAM3 inference server (Python 3.12)")
    parser.add_argument("--socket", default="/tmp/sam3_server.sock",
                        help="Unix domain socket path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-transformers", action="store_true", default=True)
    parser.add_argument("--no-transformers", dest="use_transformers", action="store_false")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = args.device
    use_transformers = args.use_transformers
    use_half = args.fp16
    confidence_threshold = args.confidence
    mask_threshold = args.mask_threshold

    # Load model
    load_model()

    # Clean up old socket file
    sock_path = args.socket
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    # Create Unix domain socket
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(2)
    os.chmod(sock_path, 0o666)

    print(f"[sam3_server] Listening on {sock_path}")
    sys.stdout.flush()

    # Graceful shutdown
    def shutdown(signum, frame):
        print("\n[sam3_server] Shutting down...")
        server.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Accept connections (one thread per client, but typically only one)
    while True:
        try:
            conn, addr = server.accept()
            print("[sam3_server] Client connected.")
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
        except OSError:
            break


if __name__ == "__main__":
    main()
