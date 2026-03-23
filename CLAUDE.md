# CLAUDE.md — GraspGen Thesis Repository

## Project Overview

Master Thesis robotic grasping pipeline that integrates **GraspGen** (NVIDIA grasp generation), **SAM3** (Segment Anything 3), a **Dobot CR series robot arm** (TCP/IP control), and an **Orbbec Gemini 2** depth camera — all running inside a Docker container on ROS2 Humble over WSL2.

The pipeline captures RGB-D frames from the Orbbec camera, segments objects with SAM3, generates grasp poses with GraspGen, and executes them on the Dobot arm.

## Repository Structure

```
GraspGen_Thesis_Repo/
├── docker/
│   ├── Dockerfile              # Multi-stage: GraspGen + SAM3 + Dobot + Orbbec + ROS2
│   ├── docker-compose.yml      # Container orchestration (cgroup rules, volumes, ports)
│   ├── entrypoint.sh           # venv activation, workspace sourcing, startup banner
│   ├── requirements.txt        # Python deps (pymodbus, pyserial, pyorbbecsdk)
│   └── patches/
│       └── fix_dobot_feedback.py  # Patches Dobot ui.py at build time
├── data/
│   ├── ui.py                   # Patched Dobot UI (replaces /opt/Dobot_hv/ui.py)
│   ├── OrbbecSDK_v2.7.6_amd64.deb  # Orbbec SDK installer (not in git)
│   ├── depth/                  # Depth data
│   ├── rgb/                    # RGB images
│   ├── masks/                  # Segmentation masks
│   └── grasp_poses/            # Generated grasp poses
├── scripts/
│   ├── view_camera.py          # Pure Python real-time RGB+Depth+IR viewer (pyorbbecsdk)
│   ├── reattach.ps1            # Windows USB passthrough via usbipd-win
│   ├── capture_rgbd.py         # RGBD capture utility
│   ├── test_full_pipeline.py   # End-to-end pipeline test
│   ├── test_graspgen.py        # GraspGen unit test
│   ├── test_sam3.py            # SAM3 unit test
│   ├── test_environment.sh     # Environment validation
│   ├── test_camera.sh          # Camera connectivity test
│   ├── build_workspace.sh      # ROS2 workspace builder
│   ├── download_models.sh      # Model weight downloader
│   └── setup_orbbec.sh         # Orbbec SDK host setup
├── ros2_ws/
│   └── src/
│       └── graspgen_pipeline/
│           └── launch/
│               └── orbbec_camera.launch.py  # ROS2 launch wrapper for Gemini 2
├── models/                     # Model weights (not in git)
├── config/                     # Configuration files
├── docs/                       # Documentation
├── results/                    # Results output
├── CLAUDE.md                   # This file (context for Claude CLI)
├── README.md                   # Project documentation
├── USB_WSL_Docker_Guide.md     # Setup guide
└── .gitignore
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

### Dockerfile Sections (11 total)

1. **Base image** — `osrf/ros:humble-desktop-full` + system packages + usbutils
2. **System CUDA PyTorch 2.7** (optional)
3. **GraspGen** — `uv` venv at `/opt/GraspGen/.venv/` (Python 3.10)
4. **Python packages** — `requirements.txt` + pymodbus/pyserial/pyorbbecsdk + **numpy<2 pinned LAST**
5. **Dobot_hv** — cloned from GitHub, patched with `fix_dobot_feedback.py`
6. **SAM3** — separate venv at `/opt/sam3/.venv/` (Python 3.9)
7. **Orbbec SDK v2.7.6** — installed from `.deb`
8. **ROS2 workspace** — OrbbecSDK_ROS2 built with `colcon build` at image time
9. **Tkinter** — system fallback for GUI
10. **Environment** — `.bashrc` aliases, PYTHONPATH
11. **Entrypoint** — `docker/entrypoint.sh`

### Container Runtime (docker-compose.yml)

- **No `privileged: true`** — uses `device_cgroup_rules` instead:
  - `c 189:* rwm` (USB), `c 180:* rwm` (USB serial), `c 81:* rwm` (Video)
- Volumes: `/dev:/dev`, `/dev/bus/usb:/dev/bus/usb`, live-linked scripts and ROS2 packages
- Ports: `29999` (Dobot dashboard), `30004` (Dobot feedback), `7860` (Viser)
- Environment: `DISPLAY`, `PYTHONPATH=/opt/GraspGen:/opt/Dobot_hv:/opt/sam3:/ros2_ws/src`

## Python Environments

| Environment | Path | Python | Purpose |
|---|---|---|---|
| GraspGen | `/opt/GraspGen/.venv/` | 3.10 (uv) | Main pipeline: torch, pyorbbecsdk, graspgen, pymodbus |
| SAM3 | `/opt/sam3/.venv/` | 3.9 (pip) | Segmentation model (isolated deps) |
| System | `/usr/bin/python3` | 3.10 | ROS2 packages, build tools |

Aliases in container: `graspgen_activate`, `sam3_activate`.

## Key Components

### Dobot Robot Arm

- Protocol: TCP/IP on ports 29999 (commands) and 30004 (real-time feedback at 1440-byte packets)
- API: `/opt/Dobot_hv/` — `dobot_api.py` (DobotApiDashboard, DobotApiFeedBack), `ui.py` (Tkinter GUI)
- Network: Robot on `192.168.X.X` subnet
- **Known bugs fixed** (applied via `docker/patches/fix_dobot_feedback.py`):
  1. **TCP partial read** — `socket.recv()` doesn't guarantee full 1440-byte packet; fixed with byte accumulation loop
  2. **Tkinter thread-safety** — background feedback thread updating widgets causes segfault on Linux; fixed with `self.root.after(0, callback)`

### Orbbec Gemini 2 Camera

- VID: `2bc5`, PID: `0670`, USB 3.0
- SDK: OrbbecSDK v2.7.6 (native `.deb`) + `pyorbbecsdk` (Python bindings)
- ROS2: OrbbecSDK_ROS2 driver, topics: `/color/image_raw`, `/depth/image_raw`, `/ir/image_raw`
- **Pure Python viewer**: `scripts/view_camera.py` (no ROS2 dependency)
- **Known issue**: Default MJPG color format unsupported in ROS2 callback — use `color_format:=RGB`
- **Known issue**: `pyorbbecsdk` 1.3.2 PyPI wheel has packaging bug (installs `cpython-311-darwin.so` even on Linux) — may need building from source

### USB Passthrough (WSL2)

Windows host → WSL2 via `usbipd-win` → Docker via device mounts + cgroup rules.
Use `scripts/reattach.ps1` (PowerShell, admin) to bind and attach the Orbbec device.

## Common Commands

```bash
# Inside container — camera viewer (pure Python)
python3 /ros2_ws/scripts/view_camera.py
python3 /ros2_ws/scripts/view_camera.py --ir --align

# Inside container — ROS2 camera node
ros2 launch orbbec_camera gemini2.launch.py color_format:=RGB

# Inside container — Dobot UI
python3 /opt/Dobot_hv/main_UI.py

# Inside container — run tests
bash /ros2_ws/scripts/test_environment.sh
python3 /ros2_ws/scripts/test_graspgen.py
```

## Critical Constraints

- **numpy must stay <2** — PyTorch is compiled against numpy 1.x; any numpy 2.x breaks it. Always install `numpy<2` as the LAST pip command.
- **Tkinter updates from main thread only** — any background thread touching Tkinter widgets will segfault on Linux. Use `widget.after(0, callback)`.
- **TCP packet accumulation** — Dobot feedback socket requires collecting exactly 1440 bytes before parsing with `np.frombuffer(data, dtype=MyType)`.
- **No `privileged: true`** — use `device_cgroup_rules` for USB/video access.
- **Build context is repo root** — all Dockerfile `COPY` paths must account for this (`docker/`, `data/` prefixes).

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

## Pending Work

- Fix `pyorbbecsdk` platform wheel issue (consider building from source)
- Fix OrbbecSDK_ROS2 MJPG format error (workaround: `color_format:=RGB`)
- Full Docker rebuild with all Dockerfile changes (`docker compose build --no-cache`)
- End-to-end pipeline integration test (camera → segmentation → grasp → execution)

## How Claude CLI Uses This File

Claude Code (Claude CLI) automatically searches for `CLAUDE.md` in the project root and uses it as context for understanding:
- Project structure and architecture
- Key constraints and gotchas
- Integration points between components
- Critical bug fixes and workarounds
- Common commands and workflows

When using `claude` to work on this project, it will have automatic context about:
- Docker architecture and build process
- Python environment setup
- Dobot robot integration
- Orbbec camera integration
- Known issues and their fixes
