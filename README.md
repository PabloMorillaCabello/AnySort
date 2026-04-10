# AnySort — Robotic Grasping Pipeline

**Master Thesis** — Text-Prompted Object Sorting with RGB-D Perception for Robotic Arms

An end-to-end pipeline that combines:
- **Orbbec Gemini 2** RGB-D depth camera
- **SAM3** (Segment Anything 3) text-prompted segmentation
- **GraspGen** (NVIDIA) 6-DOF grasp pose generation
- **Hand-eye calibration** for accurate robot manipulation
- **Dobot CR** (or other) robot arm with vacuum gripper

Everything runs inside Docker with CUDA 12.6, ROS2 Humble, and WSL2 on Windows. The main workflow is the **AnySort Tkinter application** — a graphical interface for capturing scenes, segmenting objects, generating grasps, and executing picks with automatic retry and batch processing.

---

## Quick Start (5 minutes)

### Prerequisites
- NVIDIA GPU (CUDA 12.6+) with Docker + NVIDIA Container Toolkit
- HuggingFace account with access to `facebook/sam3` and `adithyamurali/GraspGenModels`
- Orbbec Gemini 2 USB 3.0 camera
- Dobot CR robot (or UR robot) on network
- Windows 11 with WSL2 + X11 server (VcXsrv/X410) or Linux

### 1. Clone and configure

```bash
git clone <repo-url>
cd GraspGen_Thesis_Repo

# Copy .env template and set your secrets
cp docker/.env.example docker/.env

# Edit docker/.env — set HF_TOKEN, TORCH_CUDA_ARCH_LIST
# HF_TOKEN: generate at https://huggingface.co/settings/tokens
# TORCH_CUDA_ARCH_LIST: 8.6 (RTX 3060/3070/3080/3090) or 8.9 (RTX 4060/4070/4080/4090)
nano docker/.env
```

### 2. Build Docker image

```bash
# From repo root (build context is repo root, not docker/)
docker compose -f docker/docker-compose.yml build

# Takes ~30-45 minutes first time
# Installs: CUDA 12.6, ROS2 Humble, GraspGen (uv venv), SAM3 (Python 3.12 venv),
# Dobot API, OrbbecSDK v2.7.6, PointNet++ CUDA extensions
```

### 3. Start container

```bash
docker compose -f docker/docker-compose.yml up -d
```

### 4. USB camera passthrough (WSL2, once per session)

In PowerShell (Administrator):
```powershell
usbipd list              # Find Orbbec bus ID (VID 2bc5, PID 0670)
usbipd bind --busid <ID>
usbipd attach --wsl --busid <ID>
```

See [USB_WSL_Docker_Guide.md](USB_WSL_Docker_Guide.md) for detailed setup.

### 5. Launch AnySort

**From Windows desktop (recommended):**
```
Double-click AnySort.vbs
```

No terminal window. Automatically:
1. Starts container if not running
2. Launches `grasp_execute_pipeline.py`
3. Loads Meshcat visualiser
4. Initializes camera and models

**From terminal (inside container):**
```bash
docker compose -f docker/docker-compose.yml exec graspgen bash
cd app
python grasp_execute_pipeline.py
```

---

## AnySort UI Overview

A **5-column layout** with status log bar:

```
┌─────────────┬──────────────┬────────────┬──────────────┬──────────────┐
│   CAMERA    │  GRASPGEN    │   ROBOT    │  WORD LIST   │  EXECUTION   │
├─────────────┼──────────────┼────────────┼──────────────┼──────────────┤
│ • Connect   │ • Calibration│ • IP entry │ • Add/Remove │ • Status     │
│ • Capture   │ • Gripper    │ • Speed    │ • Load/Save  │ • Run single │
│ • Save/Load │ • Prompt     │ • Recover  │ • Auto-load  │ • Batch mode │
│ • ROI select│ • Options    │ • Home/Go  │              │ • Stop/Retry │
│ • Mask view │ • Grasps (N) │ • Sort/Go  │              │ • Meshcat    │
└─────────────┴──────────────┴────────────┴──────────────┴──────────────┘
└────────────────────────────── LOG BAR ─────────────────────────────────┘
```

### Column: Camera
- **Connect** — Initialize Orbbec camera
- **Capture** — Snapshot current frame
- **Save/Load scene** — Save as `.npz` for later analysis
- **Mask & ROI** — Show segmentation overlay, define 4-point polygon ROI

### Column: GraspGen
- **Calibration status** — Load hand-eye calib from file
- **Gripper dropdown** — Auto-scans `/opt/GraspGen/GraspGenModels/checkpoints/` for available models
- **Object prompt** — Enter text description (e.g., "red cylinder")
- **Options** — Collision filter, reachability filter, debug step-by-step
- **Grasps** — Shows N candidates, best N returned by confidence
- **Grasp navigator** — Previous/Next buttons

### Column: Robot
- **IP entry & Connect** — Dobot TCP/IP address (default `192.168.5.1`)
- **Speed slider** — Motion speed percentage
- **Recover Robot** — Reset alarm, re-enable, go home
- **Save/Go Home** — Define and return to home position
- **Save/Go Sort** — Define drop-off location

### Column: Word List
- **Editable list** — Object names for batch mode
- **Add/Remove** — Manage list items
- **Load/Save** — Persist lists to `data/object_lists/`
- **Auto-reload** — Remember last-used list on restart

### Column: Execution
- **Status display** — Pipeline state and errors
- **Capture & Run** — Single grasp attempt
- **Execute Selected** — Move to highlighted grasp
- **Auto-Retry** — 3 attempts per object before advancing
- **Run/Stop Batch** — Loop through word list continuously
- **Clear Mask** — Reset segmentation
- **Meshcat Viewer** — Open 3D viz (http://127.0.0.1:7000)

---

## Batch / Sorting Mode

Full end-to-end automatic picking:

1. **Build word list** — Add object names (e.g., `bottle`, `cup`, `box`)
2. **Teach Home** — Robot position between picks (click "Save Home")
3. **Teach Sort** — Drop-off location (click "Save Sort")
4. **Click "Run Batch"** — Continuous loop:
   - Capture frame → SAM3 segment → GraspGen → pick → deliver to Sort → return Home
   - 3 retries per word; on failure → advance to next word
   - Robot error → auto-recover
   - Wraps back to top when list exhausted

Expected cycle time: ~10–15 seconds per grasp (camera + SAM3 + GraspGen + motion).

---

## Project Structure

```
GraspGen_Thesis_Repo/
├── AnySort.vbs                   # Windows launcher (no terminal)
├── AnySort.cmd                   # Batch launcher
├── README.md
├── USB_WSL_Docker_Guide.md       # WSL2 USB passthrough setup
│
├── app/                          # PRIMARY: AnySort application
│   ├── grasp_execute_pipeline.py # Main Tkinter app with robot control
│   ├── hand_eye_calibration.py   # ChArUco calibration UI
│   ├── calibration_tester.py     # Calibration validation + correction
│   ├── camera_calibration.py     # Camera intrinsics (OpenCV)
│   ├── sam3_server.py            # Persistent SAM3 Unix socket server
│   ├── orbbec_quiet.py           # Suppresses OrbbecSDK C-level stderr
│   ├── pipeline_positions.json   # Saved Home/Sort positions
│   └── pipeline_roi.json         # Saved ROI polygon
│
├── docker/
│   ├── Dockerfile               # CUDA 12.6 + ROS2 Humble + all Python envs
│   ├── docker-compose.yml       # GPU, USB cgroup, port/volume mounts
│   ├── entrypoint.sh            # Auto build-workspace, venv activation
│   ├── requirements.txt          # pip packages (numpy<2 LAST)
│   ├── .env.example             # HF_TOKEN, TORCH_CUDA_ARCH_LIST, DISPLAY
│   └── patches/
│       └── fix_dobot_feedback.py # Patches Dobot TCP/Tkinter bugs
│
├── scripts/                      # Utilities and tests
│   ├── view_camera.py           # Live RGB+Depth+IR viewer (no ROS2)
│   ├── test_environment.sh      # Full environment check
│   ├── test_graspgen.py         # GraspGen + CUDA extensions test
│   ├── test_sam3.py             # SAM3 model load test
│   ├── test_camera.sh           # Orbbec ROS2 connectivity
│   ├── test_full_pipeline.py    # End-to-end integration (synthetic)
│   ├── build_workspace.sh       # colcon build helper
│   ├── reattach.ps1             # USB passthrough re-attachment (WSL2)
│   └── download_models.sh       # Manual model weight download
│
├── data/
│   ├── calibration/             # Hand-eye calib outputs + ChArUco board
│   │   ├── hand_eye_calib.npz   # Binary calibration matrix
│   │   ├── hand_eye_calib.json  # Human-readable JSON
│   │   └── auto_calib_poses.json # 26 pre-programmed robot poses
│   ├── object_lists/            # .txt word lists for batch mode
│   ├── rgb/, depth/             # Captured frames (gitignored)
│   └── OrbbecSDK_v2.7.6_amd64.deb
│
├── results/                     # Grasp JSONs (runtime, gitignored)
└── docs/                        # Additional documentation
```

---

## Python Environments (Inside Container)

The container hosts **three isolated Python environments**:

| Env | Path | Python | Used for |
|-----|------|--------|----------|
| **GraspGen** (main) | `/opt/GraspGen/.venv/` | 3.10 (uv) | AnySort pipeline, GraspGen, pyorbbecsdk, Dobot API |
| **SAM3** | `/opt/sam3env/` | 3.12 (pip) | Segmentation model server |
| **System** | `/usr/bin/python3` | 3.10 | ROS2 system packages |

Default entry uses the GraspGen venv. Container aliases for manual switching:
```bash
graspgen_activate  # Switch to GraspGen venv
sam3_activate      # Switch to SAM3 venv
```

**Model weights:** Cached in Docker named volume `model_cache` → `/opt/models` inside container (persisted across restarts).

---

## Hand-Eye Calibration

Before executing grasps, the robot must know where the camera is relative to its base frame.

### Workflow

**Step 1: Capture calibration poses**
```bash
cd app
python hand_eye_calibration.py --robot-ip 192.168.5.1
```

- Prints ChArUco board image (save/print it)
- Mount board on robot gripper
- **Auto mode** — robot moves through 26 pre-programmed poses automatically
- **Manual mode** — you move robot, click "Capture Pose" at each position
- Collect **≥10 poses** (≥20 recommended)

**Step 2: Solve**
- Click "Solve" → runs `cv2.calibrateHandEye()`
- Typical good error: < 5 mm
- Saves:
  - `data/calibration/hand_eye_calib.npz` (binary, used by app)
  - `data/calibration/hand_eye_calib.json` (human-readable)

**Step 3: Validate (optional but recommended)**
```bash
python calibration_tester.py --robot-ip 192.168.5.1
```

Tests calibration accuracy by:
1. Running test points with robot
2. Measuring predicted vs. actual error
3. Correcting systematic offsets with 6-DOF sliders
4. Saving corrected calibration if needed

---

## Camera Calibration

Camera intrinsics (focal length, principal point) must be calibrated once per camera setup:

```bash
cd app
python camera_calibration.py
```

- Prompts to print checkerboard pattern
- Capture 15+ images at different angles
- Saves to `data/calibration/camera_intrinsics.npz`

Current reference: **1280×720**, fx=684.7, fy=685.9, cx=655.3, cy=357.0, RMS=0.20 px

---

## Testing

All tests run inside the container. Enter with:
```bash
docker compose -f docker/docker-compose.yml exec graspgen bash
```

### Full environment check
```bash
bash scripts/test_environment.sh
```

Verifies: Python versions, CUDA, PyTorch, GraspGen imports, SAM3 imports, Dobot API, ROS2, model weights.

### GraspGen (grasp generation + CUDA)
```bash
python3 scripts/test_graspgen.py
python3 scripts/test_graspgen.py --no-display  # Headless
```

Tests: PointNet++ CUDA extensions, model weights, GPU inference.

### SAM3 (segmentation model)
```bash
/opt/sam3env/bin/python scripts/test_sam3.py
/opt/sam3env/bin/python scripts/test_sam3.py --image <path> --prompt "object"
/opt/sam3env/bin/python scripts/test_sam3.py --no-display  # Headless
```

Output: `results/sam3_test_result.png`

### Orbbec Gemini 2 Camera

**ROS2-based test:**
```bash
ros2 launch orbbec_camera gemini2.launch.py
ros2 topic hz /camera/color/image_raw  # Verify publishing
```

**Pure Python viewer (no ROS2 required):**
```bash
python3 scripts/view_camera.py                    # RGB + depth
python3 scripts/view_camera.py --ir --align       # With IR + alignment
python3 scripts/view_camera.py --pointcloud       # 3D point cloud
```

Press 's' to save, 'q' to quit.

### Dobot Robot API

```bash
# Check import
python3 -c "from dobot_api import DobotApiDashboard, DobotApiFeedBack; print('OK')"

# Tkinter GUI (requires X11 forwarding)
python3 /opt/Dobot_hv/ui.py
```

### End-to-end pipeline test (synthetic, no hardware)

```bash
python3 scripts/test_full_pipeline.py
```

This tests the core pipeline without requiring actual robot or camera hardware.

| Test | Script | Hardware needed |
|------|--------|-----------------|
| Environment check | `scripts/test_environment.sh` | None |
| GraspGen | `scripts/test_graspgen.py` | GPU only |
| SAM3 | `scripts/test_sam3.py` | GPU only |
| Camera | `scripts/view_camera.py` | Orbbec Gemini 2 camera |
| Dobot API | `python3 -c "from dobot_api import DobotApiDashboard"` | None |
| Full pipeline | `scripts/test_full_pipeline.py` | GPU only |

---

## Docker Reference

### Build
```bash
# From repo root (build context is repo root):
docker compose -f docker/docker-compose.yml build

# Rebuild from scratch (clear cache):
docker compose -f docker/docker-compose.yml build --no-cache
```

### Start / Stop
```bash
docker compose -f docker/docker-compose.yml up -d      # Start detached
docker compose -f docker/docker-compose.yml down       # Stop + remove
```

### Shell access
```bash
docker compose -f docker/docker-compose.yml exec graspgen bash
```

### View logs
```bash
docker compose -f docker/docker-compose.yml logs -f graspgen
```

### Download models (if not done at build)
```bash
docker compose -f docker/docker-compose.yml exec graspgen bash -c \
  "export HF_TOKEN=hf_xxx && /opt/sam3env/bin/python -c \
  \"from huggingface_hub import snapshot_download; \
  snapshot_download('facebook/sam3', local_dir='/opt/models/sam3')\""
```

---

## Exposed Ports

| Port | Service |
|------|---------|
| 7000 | Meshcat 3D visualization (grasp poses + point cloud) |
| 7860 | Viser / web UI |
| 8080 | Viser alternate port |
| 6000 | General use |
| 29999 | Dobot dashboard (TCP/IP commands) |
| 30004 | Dobot real-time feedback (1440-byte packets) |

Access Meshcat from host: `http://localhost:7000`

---

## Key Technical Notes

### numpy < 2
PyTorch is compiled against numpy 1.x. Keep `numpy<2` pinned. The Dockerfile installs it **last** via pip to prevent conflicts.

### Tkinter Updates (Main Thread Only)
Background threads updating Tkinter widgets cause segfaults on Linux. The app uses `root.after(0, callback)` to dispatch UI updates to the main thread.

### Dobot TCP Partial Reads
The Dobot feedback socket sends 1440-byte packets. Python's `socket.recv()` doesn't guarantee a full buffer. The Dockerfile patches this with a byte-accumulation loop.

### SAM3 Model Loading
- Cached in Docker named volume `model_cache` → `/opt/models/sam3/` inside container
- Falls back to HuggingFace API if cache is empty
- First inference loads model (~2–5 minutes), then cached in memory for session

### Meshcat Visualization
Self-hosted via `meshcat.Visualizer()` — no external `meshcat-server` needed.

### Build Context
Docker build context is the **repository root**, not `docker/`. All `COPY` paths in Dockerfile use `docker/` or `data/` prefixes.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `HF_TOKEN not set` | Set in `docker/.env` and rebuild |
| `CUDA out of memory` | Reduce `--num_grasps` in app or use smaller model |
| `ChArUco board not detected` | Ensure good overhead lighting, print larger |
| `Calibration error > 10 mm` | Collect more poses (≥20), ensure board fully visible |
| `Grasp off-target by 50+ mm` | Use `calibration_tester.py` to measure/correct systematic bias |
| `Camera not found (USB)` | Run `usbipd attach` again in PowerShell (WSL2) |
| `Meshcat not opening` | Ensure port 7000 is exposed; check firewall; try `http://127.0.0.1:7000` |
| `Dobot connection refused` | Verify IP (default `192.168.5.1`), ensure same network subnet |
| `Docker model cache empty` | Models auto-download on first use via HuggingFace API; ensure `HF_TOKEN` set in `docker/.env` |

---

## Upstream Repositories

- [GraspGen (NVlabs)](https://github.com/NVlabs/GraspGen) — Grasp generation
- [SAM3 (Meta)](https://github.com/facebookresearch/sam3) — Segmentation
- [Dobot_hv (TCP/IP API)](https://github.com/dauken85/Dobot_hv) — Robot control
- OrbbecSDK v2.7.6 — Depth camera driver
- [OrbbecSDK_ROS2](https://github.com/orbbec/OrbbecSDK_ROS2) — ROS2 integration

---

## Author

**Pablo Morilla Cabello** — Master's Thesis, Robotic Grasping and Sorting
