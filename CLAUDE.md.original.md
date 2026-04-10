# CLAUDE.md — GraspGen Thesis Repository

## Project Overview

Master Thesis robotic grasping pipeline that integrates **GraspGen** (NVIDIA grasp generation), **SAM3** (Segment Anything 3), a **Dobot CR series robot arm** (TCP/IP control), and an **Orbbec Gemini 2** depth camera — all running inside a Docker container on ROS2 Humble over WSL2.

The pipeline captures RGB-D frames from the Orbbec camera, segments objects with SAM3, generates grasp poses with GraspGen, and executes them on the Dobot arm.

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
├── scripts/
│   ├── view_camera.py          # Pure Python real-time RGB+Depth+IR+PointCloud viewer (669 lines)
│   ├── demo_orbbec_gemini2.py  # Legacy Orbbec demo (725 lines)
│   ├── sam3_server.py          # SAM3 Unix socket server (root-level version)
│   ├── sam3_segment_once.py    # Single-frame segmentation test
│   ├── reattach.ps1            # Windows USB passthrough via usbipd-win (run as admin)
│   ├── test_full_pipeline.py   # End-to-end pipeline test (159 lines)
│   ├── test_graspgen.py        # GraspGen loading/inference test (308 lines)
│   ├── test_sam3.py            # SAM3 loading test (294 lines)
│   ├── test_environment.sh     # Environment validation
│   ├── test_camera.sh          # ROS2 camera connectivity test
│   ├── build_workspace.sh      # colcon build helper
│   ├── download_models.sh      # Model weight downloader
│   ├── setup_orbbec.sh         # Orbbec SDK host setup
│   └── TEST/                   # *** PRIMARY WORKFLOW SCRIPTS (Tkinter UIs) ***
│       ├── demo_orbbec_gemini2_persistent_sam3.py  # Main UI: camera+SAM3+GraspGen+Meshcat (1102 lines)
│       ├── grasp_execute_pipeline.py               # Full pipeline + robot execution (1836 lines)
│       ├── hand_eye_calibration.py                 # ChArUco hand-eye calibration UI (1822 lines)
│       ├── calibration_tester.py                   # Calibration validator + 6-DOF correction (1335 lines)
│       ├── camera_calibration.py                   # Camera intrinsics calibration (656 lines)
│       ├── sam3_server.py                          # Persistent SAM3 Unix socket server (181 lines)
│       └── orbbec_quiet.py                         # Suppresses OrbbecSDK C-level stderr spam (86 lines)
├── ros2_ws/
│   └── src/
│       ├── graspgen_pipeline/
│       │   ├── graspgen_pipeline/
│       │   │   ├── camera_node.py           # RGB-D sync relay (message_filters)
│       │   │   ├── segmentation_node.py     # SAM3 integration node
│       │   │   ├── grasp_generator_node.py  # GraspGen inference node
│       │   │   ├── motion_planner_node.py   # MoveIt2 planning node
│       │   │   └── pipeline_orchestrator.py # Workflow coordinator
│       │   └── launch/
│       │       ├── full_pipeline.launch.py      # Orchestrates all 7 ROS2 nodes (145 lines)
│       │       └── orbbec_camera.launch.py      # Camera driver wrapper (57 lines)
│       └── robotiq_3f_driver/              # Gripper Modbus driver node
├── orbbec_examples/                        # Orbbec SDK reference examples (40+ scripts)
├── results/                                # Grasp result JSONs (not in git)
├── models/                                 # Model weights (not in git, populated at runtime)
├── config/                                 # Global config overrides
├── docs/                                   # Documentation
├── CLAUDE.md                               # This file
├── README.md                               # Project documentation
└── USB_WSL_Docker_Guide.md                # WSL2 USB passthrough setup guide
```

## Primary Workflow (scripts/TEST/)

The **Tkinter UIs in `scripts/TEST/`** are the primary way to use this system — not the ROS2 launch files.

### Typical session:
```bash
# Terminal 1 — SAM3 server (must start first, loads model once)
/opt/sam3env/bin/python /ros2_ws/scripts/TEST/sam3_server.py

# Terminal 2 — Main UI (camera + SAM3 + GraspGen + Meshcat viz)
python /ros2_ws/scripts/TEST/demo_orbbec_gemini2_persistent_sam3.py

# Terminal 2 alt — Full pipeline with robot execution
python /ros2_ws/scripts/TEST/grasp_execute_pipeline.py --robot-ip 192.168.5.1

# Meshcat visualization
# http://127.0.0.1:7000
```

### Suppressing OrbbecSDK stderr spam:
```python
# Import orbbec_quiet BEFORE pyorbbecsdk in any script
import sys, os
sys.path.insert(0, "/ros2_ws/scripts/TEST")
import orbbec_quiet  # redirects C-level stderr to /tmp/orbbec_sdk.log
```

## Docker Architecture

### Build Context

Build context is the **repo root** (not `docker/`). All `COPY` paths in the Dockerfile use `docker/` or `data/` prefixes accordingly.

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

Aliases in container: `graspgen_activate`, `sam3_activate`.

### Container Runtime (docker-compose.yml)

- **No `privileged: true`** — uses `device_cgroup_rules` instead:
  - `c 189:* rwm` (USB), `c 180:* rwm` (USB serial), `c 81:* rwm` (Video)
- Volumes: `/dev:/dev`, `/dev/bus/usb:/dev/bus/usb`, live-linked scripts and ROS2 packages
- Ports: `29999` (Dobot dashboard), `30004` (Dobot feedback), `7860`/`8080` (Viser), `7000`/`6000` (Meshcat)
- Environment: `DISPLAY=host.docker.internal:0.0`, `PYTHONPATH=/opt/GraspGen:/opt/Dobot_hv`, `ROS_DOMAIN_ID=42`

## Key Components

### Dobot Robot Arm

- Protocol: TCP/IP on ports 29999 (commands) and 30004 (real-time feedback at 1440-byte packets)
- API: `/opt/Dobot_hv/` — `dobot_api.py` (DobotApiDashboard, DobotApiFeedBack), `ui.py` (Tkinter GUI)
- Network: `192.168.5.1` (typical), robot on `192.168.X.X` subnet
- Tool: vacuum gripper
- Home pose: `[300, 0, 450, 0, 0, 0]` (X, Y, Z, Rx, Ry, Rz in mm/deg)
- Approach offset: 80 mm above grasp target
- **Known bugs fixed** (applied via `docker/patches/fix_dobot_feedback.py`):
  1. **TCP partial read** — `socket.recv()` doesn't guarantee full 1440-byte packet; fixed with byte accumulation loop
  2. **Tkinter thread-safety** — background feedback thread updating widgets causes segfault on Linux; fixed with `self.root.after(0, callback)`

### Orbbec Gemini 2 Camera

- VID: `2bc5`, PID: `0670`, USB 3.0
- SDK: OrbbecSDK v2.7.6 (native `.deb`) + `pyorbbecsdk` built from source
- ROS2: OrbbecSDK_ROS2 driver, topics: `/camera/color/image_raw`, `/camera/depth/image_raw`, `/camera/ir/image_raw`
- **Pure Python viewer**: `scripts/view_camera.py` (no ROS2 dependency) — supports `--align`, `--ir`, `--pointcloud`
- **Known issue**: Default MJPG color format unsupported in ROS2 callback — use `color_format:=RGB`
- **Known issue**: `pyorbbecsdk` 1.3.2 PyPI wheel has packaging bug (installs darwin `.so` on Linux) — built from source in Dockerfile

### Hand-Eye Calibration

- Type: Eye-to-hand (camera fixed, ChArUco board on gripper)
- Method: `cv2.calibrateHandEye()` on 26+ robot poses
- Output: `T_cam2base` (4×4 homogeneous transform) + Z-correction offset
- Best calibration: `data/calibration/hand_eye_calib_corrected_20260403_150244.npz`
- Camera intrinsics: 1280×720, fx=684.7, fy=685.9, cx=655.3, cy=357.0 (RMS 0.20 px)
- Z-correction: −25 mm applied after initial solve (stored as `T_correction` in JSON)
- Pre-programmed poses: `data/calibration/auto_calib_poses.json` (26 poses covering full rotation space)

### USB Passthrough (WSL2)

Windows host → WSL2 via `usbipd-win` → Docker via device mounts + cgroup rules.
Use `scripts/reattach.ps1` (PowerShell, admin) to bind and attach the Orbbec device.

## Common Commands

```bash
# Inside container — camera viewer (pure Python)
python3 /ros2_ws/scripts/view_camera.py
python3 /ros2_ws/scripts/view_camera.py --ir --align --pointcloud

# Inside container — primary pipeline UIs
python /ros2_ws/scripts/TEST/demo_orbbec_gemini2_persistent_sam3.py
python /ros2_ws/scripts/TEST/grasp_execute_pipeline.py --robot-ip 192.168.5.1
python /ros2_ws/scripts/TEST/hand_eye_calibration.py --robot-ip 192.168.5.1
python /ros2_ws/scripts/TEST/calibration_tester.py

# Inside container — ROS2 camera node
ros2 launch orbbec_camera gemini2.launch.py color_format:=RGB

# Inside container — Dobot UI
python3 /opt/Dobot_hv/main_UI.py

# Inside container — run tests
bash /ros2_ws/scripts/test_environment.sh
python3 /ros2_ws/scripts/test_graspgen.py
python3 /ros2_ws/scripts/test_sam3.py
```

## Critical Constraints

- **numpy must stay <2** — PyTorch is compiled against numpy 1.x; any numpy 2.x breaks it. Always install `numpy<2` as the LAST pip command.
- **Tkinter updates from main thread only** — any background thread touching Tkinter widgets will segfault on Linux. Use `widget.after(0, callback)`.
- **TCP packet accumulation** — Dobot feedback socket requires collecting exactly 1440 bytes before parsing with `np.frombuffer(data, dtype=MyType)`.
- **No `privileged: true`** — use `device_cgroup_rules` for USB/video access.
- **Build context is repo root** — all Dockerfile `COPY` paths must account for this (`docker/`, `data/` prefixes).
- **SAM3 server must run first** — `demo_orbbec_gemini2_persistent_sam3.py` and `grasp_execute_pipeline.py` require the SAM3 Unix socket server running before launching.

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

## Current Status (as of 2026-04-04)

- Hand-eye calibration completed and validated (branch: FxingCenteredObject)
- First grasping attempts in progress — centering/alignment being refined
- Camera intrinsics calibrated at 1280×720 with RMS 0.20 px
- All primary Tkinter UIs functional
