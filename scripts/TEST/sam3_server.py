#!/usr/bin/env python3
"""
SAM3 Persistent Server — runs under Python 3.12 SAM3 venv.

Loads model.safetensors ONCE at startup, then handles repeated inference
requests via a Unix domain socket.

Protocol (per connection):
  Client → Server  (one JSON line):  {"width": W, "height": H, "prompt": "...", "size": N}\n
                   followed by N raw RGB bytes (uint8, H×W×3)
  Server → Client  (one JSON line):  {"ok": true,  "size": M}\n
                   followed by M raw mask bytes (uint8, H×W, values {0,1})
               or  {"ok": false, "error": "message"}\n

After model is loaded, the server prints to stdout:  {"status": "ready"}
This signals the parent process it can start sending requests.

Usage:
  /opt/sam3env/bin/python3 scripts/TEST/sam3_server.py \\
      [--socket /tmp/sam3_server.sock] [--device cuda:0] [--no-fp16] [--no-transformers]
"""

import argparse
import json
import os
import socket as _socket
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 persistent inference server")
    parser.add_argument("--socket", type=str, default="/tmp/sam3_server.sock",
                        help="Unix socket path to listen on (default: /tmp/sam3_server.sock)")
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0)")
    parser.add_argument("--no-transformers", dest="use_transformers", action="store_false",
                        default=True, help="Use native SAM3 API instead of HuggingFace Transformers")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", default=True,
                        help="Use FP32 instead of FP16")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for detections")
    parser.add_argument("--max_masks", type=int, default=10,
                        help="Maximum number of instance masks to return (default: 10)")
    return parser.parse_args()


def load_model(device: str, use_transformers: bool, fp16: bool):
    """Load SAM3 model and return (model, processor, backend) tuple. Called once at startup."""
    import torch
    dtype = torch.float16 if fp16 else torch.float32

    if use_transformers:
        from transformers import Sam3Processor, Sam3Model
        print("[sam3_server] Loading Sam3Processor...", file=sys.stderr, flush=True)
        processor = Sam3Processor.from_pretrained("facebook/sam3")
        print("[sam3_server] Loading Sam3Model (this loads model.safetensors)...", file=sys.stderr, flush=True)
        model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=dtype).to(device)
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
                  prompt: str, device: str, confidence: float,
                  max_masks: int = 10) -> tuple:
    """Run inference. Returns (stacked_masks, scores) where:
      - stacked_masks: np.ndarray (N, H, W) uint8, sorted by score descending
      - scores: list of float, same order
    N=0 means no detections above threshold.
    """
    import torch
    from PIL import Image as PILImage

    pil_image = PILImage.fromarray(rgb)
    h, w = rgb.shape[:2]

    all_masks = []
    all_scores = []

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

        if masks is not None and len(masks) > 0:
            if scores is not None and len(scores) > 0:
                for mask, score in zip(masks, scores):
                    s = float(score)
                    if s >= confidence:
                        all_masks.append(mask.cpu().numpy().astype(np.uint8))
                        all_scores.append(s)
            else:
                # No scores — return first mask only
                all_masks.append(masks[0].cpu().numpy().astype(np.uint8))
                all_scores.append(1.0)

    else:  # native
        state = processor.set_image(pil_image)
        processor.set_text_prompt(state=state, prompt=prompt)
        masks = state.get("masks", None)
        scores = state.get("scores", None)

        if masks is not None and len(masks) > 0:
            if scores is not None and len(scores) > 0:
                for mask, score in zip(masks, scores):
                    s = float(score)
                    if s >= confidence:
                        all_masks.append((mask > 0).astype(np.uint8))
                        all_scores.append(s)
            else:
                all_masks.append((masks[0] > 0).astype(np.uint8))
                all_scores.append(1.0)

    if not all_masks:
        print(f"[sam3_server] No detections above threshold={confidence}", file=sys.stderr, flush=True)
        return np.zeros((0, h, w), dtype=np.uint8), []

    # Sort by score descending, limit to max_masks
    pairs = sorted(zip(all_scores, all_masks), key=lambda x: x[0], reverse=True)[:max_masks]
    all_scores, all_masks = zip(*pairs)
    all_scores = list(all_scores)

    print(
        f"[sam3_server] {len(all_masks)} mask(s) detected (scores: "
        + ", ".join(f"{s:.4f}" for s in all_scores) + ")",
        file=sys.stderr, flush=True,
    )

    return np.stack(all_masks), all_scores


def _recv_exactly(conn: _socket.socket, n: int) -> bytes:
    chunks, received = [], 0
    while received < n:
        chunk = conn.recv(min(n - received, 65536))
        if not chunk:
            raise ConnectionError("Client disconnected mid-transfer")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def _recv_json_line(conn: _socket.socket) -> dict:
    buf = b""
    while True:
        byte = conn.recv(1)
        if not byte:
            raise ConnectionError("Client disconnected before sending header")
        if byte == b"\n":
            break
        buf += byte
    return json.loads(buf.decode("utf-8"))


def _send_json_line(conn: _socket.socket, obj: dict):
    conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def handle_connection(conn: _socket.socket, model, processor, backend: str,
                      device: str, confidence: float, max_masks: int = 10):
    """Handle a single client connection: read request → infer → send masks."""
    try:
        header = _recv_json_line(conn)
        w = int(header["width"])
        h = int(header["height"])
        prompt = header["prompt"]
        size = int(header["size"])

        if size != h * w * 3:
            _send_json_line(conn, {"ok": False, "error": f"Expected {h*w*3} bytes, got {size}"})
            return

        rgb_bytes = _recv_exactly(conn, size)
        rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(h, w, 3).copy()

        print(f"[sam3_server] Request: prompt='{prompt}' image={w}x{h}", file=sys.stderr, flush=True)

        stacked, scores = run_inference(model, processor, backend, rgb, prompt, device, confidence, max_masks)
        num_masks = stacked.shape[0]
        mask_bytes = np.ascontiguousarray(stacked, dtype=np.uint8).tobytes() if num_masks > 0 else b""

        _send_json_line(conn, {
            "ok": True,
            "num_masks": num_masks,
            "size": len(mask_bytes),
            "scores": scores,
        })
        if mask_bytes:
            conn.sendall(mask_bytes)

        total_pixels = int(np.sum(stacked > 0))
        print(f"[sam3_server] Sent {num_masks} mask(s), {total_pixels} total pixels", file=sys.stderr, flush=True)

    except Exception as e:
        try:
            _send_json_line(conn, {"ok": False, "error": str(e)})
        except Exception:
            pass
        print(f"[sam3_server] Connection error: {e}", file=sys.stderr, flush=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main():
    args = parse_args()

    # Load model once — this is the slow part
    try:
        model, processor, backend = load_model(args.device, args.use_transformers, args.fp16)
    except Exception as e:
        sys.stdout.write(json.dumps({"status": "load_error", "error": str(e)}) + "\n")
        sys.stdout.flush()
        sys.exit(1)

    # Create Unix socket server
    sock_path = args.socket
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass

    server = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(5)
    try:
        os.chmod(sock_path, 0o666)
    except Exception:
        pass

    # Signal parent: ready to receive requests
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()
    print(f"[sam3_server] Ready. Listening on '{sock_path}'…", file=sys.stderr, flush=True)

    try:
        while True:
            try:
                conn, _ = server.accept()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[sam3_server] Accept error: {e}", file=sys.stderr, flush=True)
                continue
            handle_connection(conn, model, processor, backend, args.device, args.confidence, args.max_masks)
    finally:
        try:
            server.close()
        except Exception:
            pass
        try:
            os.unlink(sock_path)
        except Exception:
            pass
        print("[sam3_server] Exiting.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
