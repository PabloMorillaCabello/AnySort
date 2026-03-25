# GraspGen - Automated Grasp Generation Pipeline

> **Master Thesis** — Text-Prompted Object Grasping with RGB-D Perception for Robotic Arms

An end-to-end robotic grasping pipeline that combines RGB-D perception (Orbbec Gemini 2), text-prompted segmentation ([SAM3](https://github.com/facebookresearch/sam3)), learned grasp pose generation ([GraspGen](https://github.com/NVlabs/GraspGen)), and motion planning (MoveIt2) to autonomously pick objects with a UR robot and Robotiq 3F gripper — all orchestrated through ROS2 Humble. Dobot robots are also supported via the [Dobot TCP/IP API](https://github.com/dauken85/Dobot_hv).

## Pipeline Overview

Two execution modes are available:

**Primary (Standalone Python):**
```
Orbbec Gemini 2 ──> SAM3 Segmentation ──> GraspGen ──> Python Executor ──> Dobot / UR Robot
    (RGB-D)         (text prompt)       (point cloud     (6-DOF poses)      + Robotiq 3F
                                         → 6DOF grasps)
```

**Alternative (ROS2 + MoveIt2):**
```
Orbbec ROS2 topics ──> SAM3 node ──> GraspGen node ──> MoveIt2 Planner ──> UR Robot Driver
                      (segmentation)   (grasp generation)  (trajectory)      (motion execution)
```

## Environment Stack

| Component | Version | Python | Location / Notes |
|-----------|---------|--------|------------------|
| CUDA | 12.6 | — | Unified across all environments |
| ROS2 | Humble | 3.10 (system) | Binary packages for Ubuntu 22.04 |
| GraspGen | latest | 3.10 (uv venv) | `/opt/GraspGen/.venv` — installed via `uv pip install -e .` |
| SAM3 | Transformers API | 3.12 (venv) | `/opt/sam3env` — HuggingFace `facebook/sam3`; called via subprocess (`sam3_segment_once.py`) or socket server (`sam3_server.py`) |
| Dobot API | V4 | 3.10 (venv) | `/opt/Dobot_hv` — TCP/IP robot control (ports 29999, 30004) |
| Orbbec SDK | v2.7.6 | 3.10 | SDK .deb + OrbbecSDK_ROS2 (v2-main) for Gemini 2 |
| GraspGen Models | — | — | `/opt/GraspGen/GraspGenModels` (git-lfs) |
| SAM3 Models | — | — | `/opt/models/sam3` (HuggingFace) |

## Container Directory Layout

Inside the container there are two key areas. Understanding this avoids path confusion:

- **`/opt/`** — Third-party software baked into the Docker image during build. You don't edit these; they come from upstream repos.
  - `/opt/GraspGen/` — NVlabs GraspGen repo + `.venv/` (uv Python 3.10) + `GraspGenModels/` (checkpoints)
  - `/opt/sam3env/` — Python 3.12 venv for SAM3 (torch + transformers + numpy + Pillow + matplotlib)
  - `/opt/models/sam3/` — SAM3 model weights (pre-downloaded from HuggingFace at build time)
  - `/opt/Dobot_hv/` — Dobot TCP/IP API

- **`/ros2_ws/`** — The ROS2 workspace and **your working directory** (where you land when entering the container). Your repo folders are volume-mounted here:
  - `/ros2_ws/scripts/` ← `scripts/` from your repo (test scripts, utilities)
  - `/ros2_ws/src/graspgen_pipeline/` ← `ros2_ws/src/graspgen_pipeline/` (your pipeline code)
  - `/ros2_ws/src/robotiq_3f_driver/` ← `ros2_ws/src/robotiq_3f_driver/`
  - `/ros2_ws/data/`, `results/`, `config/` ← volume-mounted from your repo
  - `/ros2_ws/src/OrbbecSDK_ROS2/` — cloned during build (not from your repo)
  - `/ros2_ws/install/` — built by colcon on first run

All commands in this README assume you are at `/ros2_ws/` (the default). Commands that need GraspGen's own files (its bundled demos/tests) explicitly `cd /opt/GraspGen` first.

## Repository Structure

```
GraspGen_Thesis_Repo/
├── docker/                           # Containerized dev environment
│   ├── Dockerfile                    # CUDA 12.6 + ROS2 + GraspGen (uv) + SAM3 + Dobot
│   ├── docker-compose.yml            # One-command startup with GPU
│   ├── entrypoint.sh                 # Auto-build + venv activation + env banner
│   ├── requirements.txt              # Additional Python deps (installed into GraspGen venv)
│   ├── patches/                      # Upstream bug fixes applied at build time
│   │   └── fix_dobot_feedback.py     # Fix Dobot UI TCP partial-read crash
│   └── .env.example                  # Template for HF_TOKEN + CUDA arch + DISPLAY
├── ros2_ws/src/
│   ├── graspgen_pipeline/            # Main pipeline ROS2 package
│   │   ├── graspgen_pipeline/
│   │   │   ├── camera_node.py            # RGB-D sync relay (Orbbec topics)
│   │   │   ├── segmentation_node.py      # SAM3 text-prompted segmentation
│   │   │   ├── grasp_generator_node.py   # GraspGen: depth+mask → grasp poses
│   │   │   ├── motion_planner_node.py    # MoveIt2 planning + execution
│   │   │   └── pipeline_orchestrator.py  # Coordinates full pick cycle
│   │   ├── launch/
│   │   │   ├── full_pipeline.launch.py   # Launches everything
│   │   │   └── orbbec_camera.launch.py   # Orbbec Gemini 2 camera node
│   │   └── config/
│   │       └── pipeline_params.yaml      # All tuneable parameters
│   └── robotiq_3f_driver/            # Robotiq 3F gripper Modbus driver
├── scripts/                          # Utility & test scripts
│   ├── demo_orbbec_gemini2.py        # **PRIMARY DEMO**: capture → SAM3 → GraspGen → execute (standalone Python)
│   ├── view_camera.py                # Real-time RGB+Depth+IR viewer (pyorbbecsdk, no ROS2)
│   ├── sam3_segment_once.py          # One-shot SAM3 helper called by demo (subprocess bridge)
│   ├── sam3_server.py                # SAM3 inference server (socket IPC alternative)
│   ├── download_models.sh            # Download SAM3 + GraspGen weights from HuggingFace
│   ├── build_workspace.sh            # colcon build helper
│   ├── test_environment.sh           # Verify full environment setup
│   ├── test_sam3.py                  # Test SAM3 loading + inference
│   ├── test_graspgen.py              # Test GraspGen loading + inference
│   ├── test_camera.sh                # Test Orbbec Gemini 2 via ROS2
│   └── test_full_pipeline.py         # End-to-end integration test (synthetic data)
├── config/                           # Global config overrides (currently empty)
├── data/                             # Captured data (git-ignored)
├── results/                          # Experiment results (git-ignored)
└── docs/                             # Thesis notes & documentation (currently empty)
```

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 12.6+ support (RTX 3060/3090/4090, A100, etc.)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- HuggingFace account with access to:
  - [facebook/sam3](https://huggingface.co/facebook/sam3) (request access)
  - [adithyamurali/GraspGenModels](https://huggingface.co/adithyamurali/GraspGenModels)
- Orbbec Gemini 2 camera (USB)
- UR robot (UR3e/UR5e/UR10e/UR16e) on the network, or Dobot robot in TCP/IP mode
- Robotiq 3F gripper (serial)

### Step 1: Clone and configure

```bash
git clone git@github.com:YOUR_USERNAME/GraspGen_Thesis_Repo.git
cd GraspGen_Thesis_Repo

# Set your secrets
cp docker/.env.example docker/.env
nano docker/.env   # Add your HF_TOKEN, set TORCH_CUDA_ARCH_LIST, and DISPLAY
```

### Step 2: Build the Docker image

```bash
cd docker

# Build (this takes ~30-45 min on first run):
#   - Installs CUDA 12.6 + ROS2 Humble
#   - Installs GraspGen via uv (Python 3.10 venv with PointNet++ CUDA extensions)
#   - Clones GraspGen model checkpoints via git-lfs
#   - Creates SAM3 Python 3.12 venv (torch 2.7 + transformers, no git clone needed)
#   - Clones Dobot TCP/IP API
#   - Clones OrbbecSDK_ROS2
docker compose build

# Start the container in detached mode
docker compose up -d

# Enter the dev environment (container runs with sleep infinity to stay alive)
docker compose exec graspgen bash
```

### Step 3: Setup camera on host (once)

Install the Orbbec udev rules on the **host machine** (WSL2, not inside Docker):

```bash
# Copy udev rule so the camera is accessible without root
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2bc5", MODE="0666", GROUP="plugdev"' \
    | sudo tee /etc/udev/rules.d/99-orbbec.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

If you're on **Windows with WSL2**, attach the USB camera to WSL2 via `usbipd-win`. Run in PowerShell (as Administrator):

```powershell
# One-time: install usbipd-win
winget install usbipd

# List devices and find the Orbbec camera bus ID (e.g. 2-3)
usbipd list

# Bind once, then attach each session:
usbipd bind --busid 2-3
usbipd attach --wsl --busid 2-3
```

Verify the camera is visible in WSL2: `lsusb | grep Orbbec`

### Step 4: Verify the environment

```bash
# Inside the container — run the full environment check first:
./scripts/test_environment.sh
```

Then test individual components as needed (see [Testing Each Component](#testing-each-component) below).

### Step 5a: Run the standalone Python pipeline (recommended)

This is the primary workflow — runs the full capture → segmentation → grasp generation → execution cycle without ROS2 overhead. Useful for quick experiments and debugging.

```bash
# Basic demo: depth-based auto-mask (no SAM3)
python3 /ros2_ws/scripts/demo_orbbec_gemini2.py \
    --gripper_config /opt/GraspGen/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml

# With SAM3 text-prompted segmentation:
python3 /ros2_ws/scripts/demo_orbbec_gemini2.py \
    --gripper_config /opt/GraspGen/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml \
    --sam3_use \
    --sam3_prompt "red mug"

# Interactive mode (press Enter to capture, q to quit):
python3 /ros2_ws/scripts/demo_orbbec_gemini2.py \
    --gripper_config /opt/GraspGen/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml \
    --sam3_use --sam3_prompt "bottle" \
    --collision_filter \
    --keypress
```

SAM3 segmentation is handled by `sam3_segment_once.py` running in the Python 3.12 venv (subprocess call), bridging the version gap between GraspGen (3.10) and SAM3 (3.12).

### Step 5b: Run the ROS2-based pipeline (optional, for UR robot integration)

If you are integrating with a UR robot and MoveIt2 motion planning, use the full ROS2 pipeline instead:

```bash
# Build ROS2 workspace (first time only, or use entrypoint auto-build)
./scripts/build_workspace.sh

# Launch the full pipeline
ros2 launch graspgen_pipeline full_pipeline.launch.py \
    text_prompt:="cup" \
    robot_ip:="192.168.1.100" \
    use_sim:=false

# Or without hardware (for development):
ros2 launch graspgen_pipeline full_pipeline.launch.py \
    use_sim:=true \
    launch_camera:=false \
    text_prompt:="object"
```

To trigger a pick cycle (in another terminal):

```bash
docker compose exec graspgen bash
ros2 service call /pipeline_orchestrator/trigger_pick std_srvs/srv/Trigger

# Change the text prompt at runtime:
ros2 topic pub --once /segmentation/set_prompt std_msgs/String "data: 'bottle'"
```

## Dobot Robot Control

The [Dobot TCP/IP API](https://github.com/dauken85/Dobot_hv) is included at `/opt/Dobot_hv` and available on `PYTHONPATH`. It provides control of Dobot robots (CR series) via TCP/IP protocol.

```python
from dobot_api import DobotApiDashboard, DobotApiFeedBack

# Connect to robot (must be in TCP/IP mode, same network segment 192.168.X.X)
dashboard = DobotApiDashboard("192.168.1.6", 29999)
feedback = DobotApiFeedBack("192.168.1.6", 30004)

# Enable robot and move
dashboard.EnableRobot()
dashboard.MovJ(x, y, z, rx, ry, rz)
```

A tkinter-based GUI is also available: `python /opt/Dobot_hv/ui.py` (requires X11 forwarding via XLaunch).

## Environment Switching

Inside the container, the GraspGen Python 3.10 venv is activated by default. To switch environments:

```bash
# Switch to SAM3 (Python 3.12)
sam3_activate

# Switch back to GraspGen (Python 3.10)
graspgen_activate
```

## Testing Each Component

All tests run inside the container (`docker compose exec graspgen bash`). Make sure you are at `/ros2_ws/` first (type `cd /ros2_ws` if unsure). All commands below use absolute paths so they work from anywhere, but relative paths like `./scripts/...` only work from `/ros2_ws/`.

### 1. Full Environment Check

Runs a quick pass/fail sweep across every dependency: Python versions, CUDA, PyTorch, GraspGen imports, SAM3 imports, Dobot API, ROS2 packages, model weights, and key Python libraries.

```bash
/ros2_ws/scripts/test_environment.sh
```

Expected output: a summary line like `Results: 25 passed, 0 failed, 2 warnings`. Any `[FAIL]` lines indicate something that needs fixing before the pipeline will work.

### 2. GraspGen (grasp pose generation)

Tests all GraspGen dependencies (23+ packages), deep-imports every `grasp_gen` submodule, verifies PointNet++ CUDA extensions compile and run, checks model weights are present, and runs a GPU matmul sanity check.

```bash
python3 /ros2_ws/scripts/test_graspgen.py
```

To skip visualization (headless / no X11):

```bash
python3 /ros2_ws/scripts/test_graspgen.py --no-display
```

You can also run GraspGen's own bundled tests (these live inside `/opt/GraspGen/`, not in your repo):

```bash
cd /opt/GraspGen
python tests/test_inference_installation.py
```

And the bundled demo on real scene data:

```bash
cd /opt/GraspGen
python scripts/demo_scene_pc.py \
  --filter_collisions \
  --sample_data_dir GraspGenModels/sample_data/real_scene_pc \
  --gripper_config GraspGenModels/checkpoints/graspgen_franka_panda.yml
```

To return to your working directory afterwards: `cd /ros2_ws`

### 3. SAM3 (text-prompted segmentation)

Tests SAM3 in its Python 3.12 venv: verifies the HuggingFace Transformers API imports (`Sam3Processor`, `Sam3Model`), loads the model, runs inference on a synthetic image with a text prompt, and saves a 3-panel visualization.

```bash
/opt/sam3env/bin/python /ros2_ws/scripts/test_sam3.py
```

With a custom image and prompt:

```bash
/opt/sam3env/bin/python /ros2_ws/scripts/test_sam3.py --image /path/to/image.jpg --prompt "cup"
```

Headless (no X11 display needed):

```bash
/opt/sam3env/bin/python /ros2_ws/scripts/test_sam3.py --no-display
```

Output is saved to `/ros2_ws/results/sam3_test_result.png`.

### 4. Orbbec Gemini 2 Camera

Tests the camera hardware chain: USB device detection, udev rules, ROS2 `orbbec_camera` package, topic publishing, and data flow frequency. Requires the camera to be physically plugged in and attached to WSL2.

```bash
# Quick test script
/ros2_ws/scripts/test_camera.sh

# Launch the camera node (publishes RGB + depth topics)
ros2 launch orbbec_camera gemini2.launch.py

# Or use the project wrapper with custom defaults:
ros2 launch graspgen_pipeline orbbec_camera.launch.py

# Verify topics are publishing:
ros2 topic list | grep camera
ros2 topic hz /camera/color/image_raw
```

For a live video feed directly from the Orbbec (no ROS2 required):

```bash
# RGB + aligned depth viewer — press 's' to save, 'q' to quit
python3 /ros2_ws/scripts/view_camera.py

# With IR stream and depth-to-color alignment + pointcloud output:
python3 /ros2_ws/scripts/view_camera.py --ir --align
```

### 5. Dobot Robot API

The Dobot API is a pure-Python TCP/IP library — no compiled extensions to test. Verify it's importable:

```bash
python3 -c "from dobot_api import DobotApiDashboard, DobotApiFeedBack; print('Dobot API OK')"
```

To test with a real Dobot robot (must be on the same 192.168.X.X network and in TCP/IP mode):

```bash
python3 /opt/Dobot_hv/main.py
```

Or launch the tkinter GUI (requires X11 forwarding via XLaunch):

```bash
python3 /opt/Dobot_hv/ui.py
```

### 6. ROS2 Workspace

Verify ROS2 is sourced and key packages are available:

```bash
ros2 pkg list | grep -E "orbbec_camera|ur_robot_driver|moveit"
```

If packages are missing, build the workspace:

```bash
/ros2_ws/scripts/build_workspace.sh
```

### 7. End-to-End Pipeline (no hardware)

Integration test that publishes synthetic RGB-D data, runs it through SAM3 segmentation and GraspGen, and verifies grasp poses are produced. Requires two terminals.

Terminal 1 — launch the pipeline in simulation:

```bash
ros2 launch graspgen_pipeline full_pipeline.launch.py \
    use_sim:=true launch_camera:=false text_prompt:="object"
```

Terminal 2 — run the integration test:

```bash
python3 /ros2_ws/scripts/test_full_pipeline.py
```

Reports whether segmentation masks and grasp poses were received, along with the best grasp position.

### Test Summary

| Test | Script | What it verifies | Hardware needed |
|------|--------|-----------------|-----------------|
| Environment | `test_environment.sh` | All deps, imports, model weights | None |
| GraspGen | `test_graspgen.py` | GraspGen deps, PointNet++ CUDA, GPU | GPU only |
| SAM3 | `test_sam3.py` | SAM3 model load + inference | GPU only |
| Camera (ROS2) | `test_camera.sh` | Orbbec USB + ROS2 topics | Camera |
| Camera (viewer) | `view_camera.py` | Live RGB+Depth feed, no ROS2 needed | Camera |
| Dobot | `python3 -c "import dobot_api"` | API importable | None (robot for full test) |
| ROS2 | `ros2 pkg list` | Workspace packages built | None |
| Full pipeline | `test_full_pipeline.py` | End-to-end with synthetic data | GPU only |

## Docker Commands Reference

```bash
# Build image
cd docker && docker compose build

# Start/stop container
docker compose up -d
docker compose down

# Enter running container
docker compose exec graspgen bash

# Rebuild after Dockerfile changes
docker compose build --no-cache

# View logs
docker compose logs -f graspgen

# If models weren't downloaded at build time:
docker compose exec graspgen bash -c "export HF_TOKEN=hf_xxx && ./scripts/download_models.sh"
```

## Exposed Ports

| Port | Service |
|------|---------|
| 8080 | Viser web UI |
| 7860 | Viser alternate |
| 7000 | General use |
| 6000 | General use |
| 29999 | Dobot dashboard (TCP/IP control) |
| 30004 | Dobot real-time feedback |

## ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB from Orbbec Gemini 2 |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Depth from Orbbec Gemini 2 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics |
| `/segmentation/mask` | `sensor_msgs/Image` | Binary mask from SAM3 |
| `/segmentation/visualization` | `sensor_msgs/Image` | RGB overlay with mask |
| `/segmentation/set_prompt` | `std_msgs/String` | Update text prompt at runtime |
| `/grasp_gen/grasp_poses` | `geometry_msgs/PoseArray` | 6-DOF grasp poses |
| `/grasp_gen/confidences` | `std_msgs/Float32MultiArray` | Confidence per grasp |
| `/pipeline_orchestrator/status` | `std_msgs/String` | Pipeline state |

## Configuration

All tuneable parameters are in `ros2_ws/src/graspgen_pipeline/config/pipeline_params.yaml`:

- **SAM3**: confidence threshold, mask threshold
- **GraspGen**: gripper config, number of grasp candidates, quality threshold
- **Motion planner**: velocity/acceleration limits, planning time, approach/retreat distance
- **Pipeline**: auto-execute mode, result saving

## Author

**Pablo Morilla Cabello** — pablomorillacabello@gmail.com
