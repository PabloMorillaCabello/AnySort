# SAM3 — Segment Anything 3

> Segmentation model that isolates target objects from RGB frames.
> **File:** `app/sam3_server.py`

---

## What It Does

Given an RGB frame + prompt (click point or text), produces a **binary mask** of the target object. Used to isolate the object point cloud before grasp generation.

---

## Architecture

SAM3 runs as a **persistent Unix socket server** to avoid reloading model weights on every query (model load takes ~2–10 minutes on first run due to HuggingFace download).

```
anysort.py
  → subprocess.Popen(sam3_server.py)   ← auto-started at splash
  → waits up to 600s for socket
  → /tmp/sam3_server.sock (Unix socket)
  ↕ JSON protocol
sam3_server.py
  → loads facebook/sam3 weights
  → listens for segmentation requests
  → returns mask arrays
```

---

## Python Environment

| Property | Value |
|---|---|
| Venv path | `/opt/sam3env/` |
|Python | 3.12 (pip) |
| Key deps | PyTorch 2.7, HuggingFace transformers |
| Activate | `sam3_activate` (container alias) |

**Why isolated?** SAM3 requires PyTorch 2.7 + Python 3.12, which conflicts with GraspGen's deps.

---

## Auto-Start in AnySort

```python
# In anysort.py:
_find_sam3_python()               # /opt/sam3env/bin/python3.12
subprocess.Popen([python, sam3_server.py])
_wait_for_sam3_socket(sock_path, timeout=600)
```

No manual pre-start needed. The socket appears only after model is fully loaded.

---

## Communication Protocol

Unix socket at `/tmp/sam3_server.sock`. One connection per request.

### Client → Server
1. **JSON header line** (newline-terminated):
   ```json
   {"width": 1280, "height": 720, "prompt": "blue circle", "size": 2764800}
   ```
2. **Raw RGB24 bytes** — `H×W×3` uint8, row-major (`size` bytes)

### Server → Client
1. **JSON response line**:
   ```json
   {"ok": true, "size": 737280, "num_masks": 1}
   ```
   or: `{"ok": false, "error": "message"}`
2. **Mask bytes** — `K×H×W` uint8 concatenated (`size` bytes total)

### Client-Side Post-Processing
`segment_with_sam3()` in pipeline:
- Takes first mask (`all_masks[0]`)
- Runs `_select_largest_component()` (connected components) to clean noise
- Returns single `H×W` uint8 binary mask

### Server Args
```bash
sam3_server.py --socket /tmp/sam3_server.sock
               --device cuda:0
               --no-fp16          # use FP32
               --no-transformers  # native SAM3 API instead of HuggingFace
               --confidence 0.5   # detection threshold
```

---

## Test Script

```bash
python3 /ros2_ws/scripts/test_sam3.py
```

---

## Links
- [[../Pipeline/Pipeline Flow|Pipeline Flow]] — SAM3 in data path
- [[../Pipeline/AnySort Pipeline|AnySort Pipeline]] — auto-start logic
- [[../Infrastructure/Python Environments|Python Environments]] — isolated sam3env
- [[../Infrastructure/Docker Setup|Docker Setup]] — build stage 6 (SAM3)
