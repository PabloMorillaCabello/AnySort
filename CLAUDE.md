# CLAUDE.md — GraspGen Thesis Repository

## Project Overview

Master Thesis robotic grasping pipeline integrating **GraspGen** (NVIDIA grasp generation), **SAM3** (Segment Anything 3), **Dobot CR series robot arm** (TCP/IP), **Orbbec Gemini 2** depth camera — Docker, ROS2 Humble, WSL2.

Pipeline: Orbbec RGB-D → SAM3 segment → GraspGen poses → Dobot execute.

## Repository Structure

```
GraspGen_Thesis_Repo/
├── docker/
│   ├── Dockerfile              # Multi-stage: GraspGen + SAM3 + Dobot + Orbbec + ROS2 (CUDA 12.6)
│   ├── docker-compose.yml      # Container orchestration (cgroup rules, GPU, volumes, ports)
│   ├── entrypoint.sh           # venv activation, workspace sourcing, startup banner
│   ├── requirements.txt        # Python deps (opencv, open3d, trimesh, scipy, etc.)
│   ├── Bash.cmd                # Windows batch helper for container entry
│   └── patches/
│       ├── fix_dobot_feedback.py   # Patches Dobot ui.py at build time (TCP + Tkinter fixes)
│       └── fix_dobot_feedback.patch
├── data/
│   ├── OrbbecSDK_v2.7.6_amd64.deb  # Orbbec SDK installer (not in git)
│   ├── calibration/
│   │   ├── camera_intrinsics.json/.npz     # 1280×720, fx≈685, fy≈686, RMS 0.20px (31 frames)
│   │   ├── hand_eye_calib.json/.npz        # T_cam2base + Z-correction (-25 mm)
│   │   ├── hand_eye_calib_corrected_*.json/.npz  # Corrected calibrations (best: 20260403)
│   │   └── auto_calib_poses.json           # 26 pre-programmed calibration robot poses
│   ├── depth/, depth_aligned/              # Depth captures
│   ├── rgb/                                # RGB captures
│   └── pointcloud/                         # .npy and .ply point clouds
├── app/                        # *** FINAL SOLUTION (AnySort pipeline) ***
│   ├── grasp_execute_pipeline.py  # AnySort: full pipeline + robot execution (Tkinter UI)
│   ├── hand_eye_calibration.py    # ChArUco hand-eye calibration UI
│   ├── calibration_tester.py      # Calibration validator + 6-DOF correction
│   ├── camera_calibration.py      # Camera intrinsics calibration
│   ├── sam3_server.py             # Persistent SAM3 Unix socket server (auto-started by AnySort)
│   └── orbbec_quiet.py            # Suppresses OrbbecSDK C-level stderr spam
├── scripts/
│   ├── view_camera.py          # Pure Python real-time RGB+Depth+IR+PointCloud viewer
│   ├── reattach.ps1            # Windows USB passthrough via usbipd-win (run as admin)
│   ├── test_full_pipeline.py   # End-to-end pipeline test
│   ├── test_graspgen.py        # GraspGen loading/inference test
│   ├── test_sam3.py            # SAM3 loading test
│   ├── test_environment.sh     # Environment validation
│   ├── test_camera.sh          # ROS2 camera connectivity test
│   ├── build_workspace.sh      # colcon build helper
│   ├── download_models.sh      # Model weight downloader
│   └── setup_orbbec.sh         # Orbbec SDK host setup
├── orbbec_examples/                        # Orbbec SDK reference examples (not in git)
├── results/                                # Grasp result JSONs (not in git)
├── config/                                 # Global config overrides
├── docs/                                   # Documentation
├── AnySort.cmd                 # Windows one-click launcher (cmd version)
├── AnySort.vbs                 # Windows one-click launcher (no terminal window)
├── CLAUDE.md                               # This file
├── README.md                               # Project documentation
└── USB_WSL_Docker_Guide.md                # WSL2 USB passthrough setup guide
```

## Primary Workflow (app/)

**AnySort** (`app/grasp_execute_pipeline.py`) is the single entry point — starts SAM3 server, Meshcat, camera, and loads GraspGen weights automatically via a splash screen.

### Launch options:
```bash
# From Windows (no terminal window):
AnySort.vbs   # double-click

# From Windows (cmd):
AnySort.cmd

# From inside container:
cd /ros2_ws/app && python grasp_execute_pipeline.py

# Calibration tools (from inside container):
python /ros2_ws/app/hand_eye_calibration.py --robot-ip 192.168.5.1
python /ros2_ws/app/calibration_tester.py
python /ros2_ws/app/camera_calibration.py

# Meshcat visualization (URL shown in AnySort UI):
# http://127.0.0.1:<port>   (self-hosted, port printed at startup)
```

### Suppressing OrbbecSDK stderr spam:
```python
# Import orbbec_quiet BEFORE pyorbbecsdk in any script
import sys, os
sys.path.insert(0, "/ros2_ws/app")
import orbbec_quiet  # redirects C-level stderr to /tmp/orbbec_sdk.log
```

## Docker Architecture

### Build Context

Build context: **repo root** (not `docker/`). All `COPY` paths use `docker/` or `data/` prefixes.

```bash
# Build
docker compose -f docker/docker-compose.yml build --no-cache

# Run
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml exec graspgen /bin/bash
```

### Dockerfile Sections (10 stages)

1. **Base image** — CUDA 12.6 + ROS2 Humble (Python 3.10) + system packages
2. **System deps** — Python 3.12, build tools, OpenGL, Tkinter
3. **GraspGen** — `uv` venv at `/opt/GraspGen/.venv/` (Python 3.10), PointNet++ CUDA extensions
4. **Python packages** — `requirements.txt` + pymodbus/pyserial + **numpy<2 pinned LAST**
5. **Dobot API** — cloned from GitHub, patched with `fix_dobot_feedback.py`
6. **SAM3** — venv at `/opt/sam3env/` (Python 3.12), PyTorch 2.7 + HuggingFace weights
7. **Orbbec SDK v2.7.6** — installed from `.deb` + `pyorbbecsdk` built from source
8. **ROS2 workspace** — OrbbecSDK_ROS2 + pipeline packages built with `colcon build`
9. **Environment** — `.bashrc` aliases, PYTHONPATH
10. **Entrypoint** — `docker/entrypoint.sh`

### Python Environments

| Environment | Path | Python | Purpose |
|---|---|---|---|
| GraspGen (main) | `/opt/GraspGen/.venv/` | 3.10 (uv) | Main pipeline: torch, pyorbbecsdk, graspgen, pymodbus |
| SAM3 | `/opt/sam3env/` | 3.12 (pip) | Segmentation model (isolated deps) |
| System | `/usr/bin/python3` | 3.10 | ROS2 packages, build tools |

Container aliases: `graspgen_activate`, `sam3_activate`.

### Container Runtime (docker-compose.yml)

- No `privileged: true` — uses `device_cgroup_rules`:
  - `c 189:* rwm` (USB), `c 180:* rwm` (USB serial), `c 81:* rwm` (Video)
- Volumes: `/dev:/dev`, `/dev/bus/usb:/dev/bus/usb`, live-linked `app/` + `scripts/` + ROS2 packages
- Ports: `29999` (Dobot dashboard), `30004` (Dobot feedback), `7860`/`8080` (Viser), `7000`/`6000` (Meshcat)
- Env: `DISPLAY=host.docker.internal:0.0`, `PYTHONPATH=/opt/GraspGen:/opt/Dobot_hv`, `ROS_DOMAIN_ID=42`

## Key Components

### Dobot Robot Arm

- Protocol: TCP/IP ports 29999 (commands), 30004 (real-time feedback, 1440-byte packets)
- API: `/opt/Dobot_hv/` — `dobot_api.py` (DobotApiDashboard, DobotApiFeedBack), `ui.py` (Tkinter GUI)
- Network: `192.168.5.1` (typical), robot on `192.168.X.X` subnet
- Tool: vacuum gripper
- Home pose: `[300, 0, 450, 0, 0, 0]` (X, Y, Z, Rx, Ry, Rz in mm/deg)
- Approach offset: 80 mm above grasp target
- **Known bugs fixed** (via `docker/patches/fix_dobot_feedback.py`):
  1. **TCP partial read** — `socket.recv()` doesn't guarantee full 1440-byte packet; fixed with byte accumulation loop
  2. **Tkinter thread-safety** — background feedback thread updating widgets causes segfault on Linux; fixed with `self.root.after(0, callback)`

### Orbbec Gemini 2 Camera

- VID: `2bc5`, PID: `0670`, USB 3.0
- SDK: OrbbecSDK v2.7.6 (native `.deb`) + `pyorbbecsdk` built from source
- ROS2: OrbbecSDK_ROS2 driver, topics: `/camera/color/image_raw`, `/camera/depth/image_raw`, `/camera/ir/image_raw`
- **Pure Python viewer**: `scripts/view_camera.py` (no ROS2 dep) — supports `--align`, `--ir`, `--pointcloud`
- **Known issue**: Default MJPG color format unsupported in ROS2 callback — use `color_format:=RGB`
- **Known issue**: `pyorbbecsdk` 1.3.2 PyPI wheel has packaging bug (installs darwin `.so` on Linux) — built from source in Dockerfile

### Hand-Eye Calibration

- Type: Eye-to-hand (camera fixed, ChArUco board on gripper)
- Method: `cv2.calibrateHandEye()` on 26+ robot poses
- Output: `T_cam2base` (4×4 homogeneous transform) + Z-correction offset
- Best calibration: `data/calibration/hand_eye_calib_corrected_20260403_150244.npz`
- Camera intrinsics: 1280×720, fx=684.7, fy=685.9, cx=655.3, cy=357.0 (RMS 0.20 px)
- Z-correction: −25 mm after initial solve (stored as `T_correction` in JSON)
- Pre-programmed poses: `data/calibration/auto_calib_poses.json` (26 poses, full rotation space)

### USB Passthrough (WSL2)

Windows → WSL2 via `usbipd-win` → Docker via device mounts + cgroup rules.
Use `scripts/reattach.ps1` (PowerShell, admin) to bind/attach Orbbec device.

## Common Commands

```bash
# Inside container — camera viewer (pure Python)
python3 /ros2_ws/scripts/view_camera.py
python3 /ros2_ws/scripts/view_camera.py --ir --align --pointcloud

# Inside container — AnySort (primary app, single command)
cd /ros2_ws/app && python grasp_execute_pipeline.py

# Inside container — calibration tools
python /ros2_ws/app/hand_eye_calibration.py --robot-ip 192.168.5.1
python /ros2_ws/app/calibration_tester.py
python /ros2_ws/app/camera_calibration.py

# Inside container — Dobot UI
python3 /opt/Dobot_hv/main_UI.py

# Inside container — run tests
bash /ros2_ws/scripts/test_environment.sh
python3 /ros2_ws/scripts/test_graspgen.py
python3 /ros2_ws/scripts/test_sam3.py
```

## Critical Constraints

- **numpy must stay <2** — PyTorch compiled against numpy 1.x; numpy 2.x breaks it. Install `numpy<2` LAST.
- **Tkinter updates from main thread only** — background thread touching widgets segfaults on Linux. Use `widget.after(0, callback)`.
- **TCP packet accumulation** — Dobot feedback socket needs exactly 1440 bytes before `np.frombuffer(data, dtype=MyType)`.
- **No `privileged: true`** — use `device_cgroup_rules` for USB/video access.
- **Build context is repo root** — all Dockerfile `COPY` paths must use `docker/`, `data/` prefixes.
- **SAM3 server auto-starts** — `grasp_execute_pipeline.py` (AnySort) launches `app/sam3_server.py` automatically during splash loading. No manual pre-start needed.

## Dobot Feedback Fix Details

### Bug #1: TCP Partial Reads
```python
# WRONG: recv() doesn't guarantee full buffer
data = self.client_feed.socket_dobot.recv(14400)[0:1440]
a = np.frombuffer(data, dtype=MyType)  # ValueError if incomplete

# CORRECT: accumulate until exactly 1440 bytes
data = bytes()
while len(data) < PACKET_SIZE:
    remaining = PACKET_SIZE - len(data)
    chunk = self.client_feed.socket_dobot.recv(remaining)
    if not chunk:
        break
    data += chunk
```

### Bug #2: Tkinter Thread-Safety
```python
# WRONG: direct widget update from background thread
def feed_back(self):
    while True:
        self.label_feed_speed["text"] = ...  # Segfault on Linux

# CORRECT: dispatch to main thread via after()
def feed_back(self):
    while True:
        self.root.after(0, self._update_feed_ui, data)

def _update_feed_ui(self, a):
    """Runs on main thread - safe to update widgets"""
    self.label_feed_speed["text"] = a["SpeedScaling"][0]
```

## Upstream Repositories

- **GraspGen**: https://github.com/NVlabs/GraspGen
- **Dobot_hv**: https://github.com/dauken85/Dobot_hv (TCP-IP-CR-Python-V4)
- **OrbbecSDK_ROS2**: Orbbec official ROS2 driver
- **SAM3**: Segment Anything 3

## Current Status (as of 2026-04-10)

- AnySort UI fully functional: splash loading, batch word-list mode, robot error recovery
- Hand-eye calibration done + validated (RMS 0.20 px)
- Active grasping tests in progress
- Repo reorganized: final scripts moved from `scripts/TEST/` to `app/`
- Single-command Windows launch: `AnySort.vbs` (no terminal window)
