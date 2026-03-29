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
| SAM3 | latest | 3.9 (venv) | `/opt/sam3env` — persistent socket server (sam3_server.py) for inference |
| Dobot API | V4 | 3.10 (venv) | `/opt/Dobot_hv` — TCP/IP robot control (ports 29999, 30004) |
| Orbbec SDK | v2.7.6 | 3.10 | SDK .deb + OrbbecSDK_ROS2 (v2-main) for Gemini 2 |
| GraspGen Models | — | — | `/opt/GraspGen/GraspGenModels` (git-lfs) |
| SAM3 Models | — | — | `/opt/models/sam3` (HuggingFace) |

## Container Directory Layout

Inside the container there are two key areas. Understanding this avoids path confusion:

- **`/opt/`** — Third-party software baked into the Docker image during build. You don't edit these; they come from upstream repos.
  - `/opt/GraspGen/` — NVlabs GraspGen repo + `.venv/` (uv Python 3.10) + `GraspGenModels/` (checkpoints)
  - `/opt/sam3/` — Facebook SAM3 repo
  - `/opt/sam3env/` — Python venv for SAM3 (separate from GraspGen)
  - `/opt/models/sam3/` — SAM3 model weights
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
│   ├── view_camera.py                # Real-time RGB+Depth+IR viewer (pyorbbecsdk, no ROS2)
│   ├── sam3_server.py                # **SAM3 PERSISTENT SERVER**: loads model once, serves requests via Unix socket
│   ├── download_models.sh            # Download SAM3 + GraspGen weights from HuggingFace
│   ├── build_workspace.sh            # colcon build helper
│   ├── test_environment.sh           # Verify full environment setup
│   ├── test_sam3.py                  # Test SAM3 loading + inference
│   ├── test_graspgen.py              # Test GraspGen loading + inference
│   ├── test_camera.sh                # Test Orbbec Gemini 2 via ROS2
│   ├── test_full_pipeline.py         # End-to-end integration test (synthetic data)
│   ├── demo_orbbec_gemini2.py        # Legacy demo (reference, use TEST scripts instead)
│   └── TEST/                         # **PRIMARY WORKFLOW SCRIPTS** (Tkinter UIs)
│       ├── demo_orbbec_gemini2_persistent_sam3.py  # Main UI: live preview → segmentation → grasp generation
│       ├── grasp_execute_pipeline.py               # Extended UI: adds hand-eye calibration → robot execution
│       ├── hand_eye_calibration.py                 # Calibration UI: ChArUco board capture & solve
│       └── calibration_tester.py                   # Calibration validation: test accuracy with ArUco markers
├── config/                           # Global config overrides (currently empty)
├── data/                             # Captured data and calibration (git-ignored)
│   ├── calibration/                  # Hand-eye calibration outputs
│   │   ├── hand_eye_calib.npz        # Calibration matrix (binary)
│   │   ├── hand_eye_calib.json       # Calibration matrix (JSON)
│   │   └── aruco_id0_4x4_50.png      # Printed ArUco marker (ID 0) for testing
│   ├── rgb/                          # Captured RGB frames
│   ├── depth/                        # Captured depth frames
│   └── masks/                        # SAM3 segmentation masks
├── results/                          # Experiment results (git-ignored)
│   └── best_grasp_<timestamp>.json   # Generated grasp poses per execution
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
#   - Installs SAM3 in a separate Python 3.12 venv
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

### Step 5a: Run the primary Tkinter UI (recommended)

This is the primary workflow — a graphical interface that captures live RGB-D frames, applies SAM3 segmentation, generates grasps, and visualizes results in real-time.

```bash
# Terminal 1: Start the SAM3 persistent server (loads model once)
python3 /ros2_ws/scripts/sam3_server.py

# Terminal 2 (or add --sam3_autostart flag to avoid Terminal 1):
python3 /ros2_ws/scripts/TEST/demo_orbbec_gemini2_persistent_sam3.py
```

Or run both in one terminal with automatic server startup:

```bash
python3 /ros2_ws/scripts/TEST/demo_orbbec_gemini2_persistent_sam3.py --sam3_autostart
```

The UI provides:
- **Live RGB preview** with depth overlay
- **Gripper selection dropdown** — auto-scans `/opt/GraspGen/GraspGenModels/checkpoints/` for available models
- **SAM3 text prompt input** — enter object description (e.g., "red mug", "bottle")
- **Capture & Run button** — captures current frame and runs full pipeline
- **Green mask overlay** on RGB after SAM3 inference
- **Meshcat visualization** at `http://127.0.0.1:7000` — shows 3D point cloud and grasp poses
- **Results saved** to `/ros2_ws/results/best_grasp_<timestamp>.json` after each inference
- **Grasp sampler cached per gripper** — reloads only when gripper selection changes (faster subsequent runs)

### Step 5b: Run extended pipeline with robot execution (optional)

For full robot integration with hand-eye calibration and execution:

```bash
# Terminal 1: Start the SAM3 persistent server
python3 /ros2_ws/scripts/sam3_server.py

# Terminal 2: Run the extended pipeline with execution
python3 /ros2_ws/scripts/TEST/grasp_execute_pipeline.py
```

This extends the main UI with:
- **Hand-eye calibration matrix loading** from `data/calibration/hand_eye_calib.npz`
- **6-DOF pose transformation** — converts camera frame grasps to robot base frame
- **Pre-grasp → Grasp → Retreat motion** planning and execution
- **Vacuum tool control** — ON/OFF buttons (requires robot to support vacuum actuator)
- **Robot IP configuration** — set via command-line flag

See [Hand-Eye Calibration](#hand-eye-calibration) section below for calibration workflow.

### Step 5c: Run the ROS2-based pipeline (optional, for UR robot integration)

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

**Note:** The ROS2 pipeline is fully functional but the primary workflow is the standalone Tkinter UI (Step 5a). ROS2 mode is recommended for UR robot + MoveIt2 integration; Dobot and vacuum-based systems use the Tkinter pipeline.

## TEST Scripts (Primary Workflow)

All primary scripts are Tkinter-based graphical applications located in `scripts/TEST/`. These are the recommended entry points for the pipeline.

### 1. demo_orbbec_gemini2_persistent_sam3.py

**Main segmentation and grasp generation UI.**

Features:
- Live RGB preview with depth overlay
- Automatic gripper detection dropdown (scans `/opt/GraspGen/GraspGenModels/checkpoints/`)
- SAM3 text prompt input (e.g., "red mug")
- "Capture & Run" button to execute the full pipeline
- Green mask overlay after SAM3 segmentation
- Meshcat 3D visualization at `http://127.0.0.1:7000`
- Results saved to `/ros2_ws/results/best_grasp_<timestamp>.json`
- Grasp sampler cached per gripper (only reloaded on gripper change)

**Usage:**

```bash
# With manual SAM3 server (2 terminals)
# Terminal 1:
python3 /ros2_ws/scripts/sam3_server.py

# Terminal 2:
python3 /ros2_ws/scripts/TEST/demo_orbbec_gemini2_persistent_sam3.py

# Or auto-start SAM3 server (single terminal)
python3 /ros2_ws/scripts/TEST/demo_orbbec_gemini2_persistent_sam3.py --sam3_autostart
```

### 2. grasp_execute_pipeline.py

**Extended UI with hand-eye calibration and robot execution.**

Extends the main demo with:
- Hand-eye calibration matrix loading from `data/calibration/hand_eye_calib.npz`
- 6-DOF pose transformation (camera frame → robot base frame)
- Pre-grasp → Grasp → Retreat motion planning and execution
- Vacuum tool control (ON/OFF buttons)
- Robot IP configuration via command-line flag

**Usage:**

```bash
# Terminal 1:
python3 /ros2_ws/scripts/sam3_server.py

# Terminal 2 (requires calibration in place):
python3 /ros2_ws/scripts/TEST/grasp_execute_pipeline.py --robot_ip 192.168.5.1
```

### 3. hand_eye_calibration.py

**Interactive hand-eye calibration UI using ChArUco boards.**

Workflow:
1. Print a ChArUco board (generated automatically by the script)
2. Mount it on the robot gripper (or end-effector)
3. In **Auto mode**: robot moves through pre-programmed poses automatically
4. In **Manual mode**: you move the robot manually, click "Capture Pose" at each position
5. Collect ≥10 poses, then click "Solve" to run `cv2.calibrateHandEye()`
6. Results saved to:
   - `data/calibration/hand_eye_calib.npz` (binary calibration matrix)
   - `data/calibration/hand_eye_calib.json` (human-readable)

**Usage:**

```bash
python3 /ros2_ws/scripts/TEST/hand_eye_calibration.py --robot_ip 192.168.5.1
```

Flags:
- `--robot_ip <IP>` — Dobot robot IP (default: 192.168.1.6)
- `--auto_mode` — Run in automatic mode (robot moves itself)

### 4. calibration_tester.py

**Hand-eye calibration quality assessment and correction UI.**

Tests the accuracy of an existing hand-eye calibration using ArUco markers. Supports correction of systematic offsets (XYZ translation, Roll/Pitch/Yaw rotation).

**Setup:**
1. Calibration must exist at `data/calibration/hand_eye_calib.npz` (from `hand_eye_calibration.py`)
2. Print an ArUco marker (4X4_50, ID 0) — script can auto-generate with "Gen Marker" button
3. Place marker in the workspace at a fixed, known location

**Workflow:**

**Test A (Move to Predicted):**
- Robot moves TCP to the camera-predicted marker position (using calibration)
- Logs XY error automatically
- Tests if the calibration transforms correctly

**Test B (Ground Truth):**
- Camera detects marker position
- You manually jog the TCP to the marker center (using controller)
- Click button to capture actual TCP pose
- Computes error: predicted vs. actual pose

**Results table shows per-test-point:**
- Predicted XYZ / Actual XYZ / Error XYZ / 3D distance

**Correction workflow:**
1. Run multiple test points to find systematic bias
2. Use 6-DOF sliders (ΔX/Y/Z ±200mm, ΔRoll/Pitch/Yaw ±30°) to dial in corrections
3. Click "Auto-Fill From Errors" — fills sliders with negative mean error (zero out bias)
4. Click "Save Corrected Calibration" — writes new `.npz`/`.json` for use in execution

**Usage:**

```bash
python3 /ros2_ws/scripts/TEST/calibration_tester.py \
    --robot_ip 192.168.5.1 \
    --calib data/calibration/hand_eye_calib.npz
```

Flags:
- `--robot_ip <IP>` — Dobot robot IP
- `--calib <path>` — Path to calibration `.npz` file (default: `data/calibration/hand_eye_calib.npz`)

---

## Hand-Eye Calibration

Hand-eye calibration is the process of finding the transformation matrix between the camera frame and the robot base frame. This is critical for accurate grasp execution.

### Calibration Workflow

**Step 1: Capture calibration poses**

```bash
# Generate and print a ChArUco board:
python3 /ros2_ws/scripts/TEST/hand_eye_calibration.py --robot_ip 192.168.5.1
```

The script will display a window with a ChArUco board image and print its location:
```
ChArUco board saved to: data/calibration/charuco_board.png
Print and mount on robot gripper
```

Mount the printed board on the robot's gripper or end-effector.

**Step 2: Solve calibration**

In the UI, select Auto or Manual mode:
- **Auto**: Robot moves through pre-programmed poses automatically
- **Manual**: You move the robot, click "Capture Pose" at each position

Collect **≥10 poses** (more is better, ≥20 recommended). The camera must see the ChArUco board clearly in each capture.

Click **Solve** — this runs `cv2.calibrateHandEye()` and displays the reprojection error. Typical good error: < 5 mm.

Output files:
```
data/calibration/hand_eye_calib.npz    # Binary (used by grasp_execute_pipeline.py)
data/calibration/hand_eye_calib.json   # Human-readable
```

**Step 3: Test calibration quality (optional but recommended)**

```bash
python3 /ros2_ws/scripts/TEST/calibration_tester.py --robot_ip 192.168.5.1
```

This validates the calibration by:
1. Running test points with the robot
2. Measuring predicted vs. actual error
3. Allowing you to correct systematic offsets
4. Saving a corrected calibration if needed

### Troubleshooting Calibration

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Cannot detect ChArUco board" | Poor lighting or board too small | Ensure good overhead lighting, print board larger |
| Reprojection error > 10 mm | Not enough poses or board out of frame | Collect more poses (≥20), ensure board fully visible |
| Grasp execution off by 50+ mm | Systematic bias in camera extrinsics | Use `calibration_tester.py` to measure and correct |
| Robot hits table/gripper collision | Calibration matrix transposed/inverted | Verify calibration with hand-eye solver output; re-calibrate if needed |

---

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
# Switch to SAM3 (Python 3.9)
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

Tests SAM3 in its Python 3.9 venv: verifies the package imports (native API and HuggingFace Transformers API), loads the model, runs inference on a synthetic image with a text prompt, and saves a 3-panel visualization.

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

**SAM3 Server (persistent inference):**

For repeated inference calls (e.g., in the Tkinter UI), use the persistent socket server to load the model once:

```bash
# Terminal 1: Start server
python3 /ros2_ws/scripts/sam3_server.py
# Server listens on /tmp/sam3_server.sock and loads model once

# Terminal 2: Run UI (connects to server)
python3 /ros2_ws/scripts/TEST/demo_orbbec_gemini2_persistent_sam3.py
```

The server accepts JSON + raw RGB bytes over a Unix socket and returns segmentation masks efficiently.

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
| 7000 | Meshcat 3D visualization (grasp poses, point clouds) |
| 8080 | Viser web UI |
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

- **SAM3**: confidence threshold, mask threshold, Transformers vs native API
- **GraspGen**: gripper config, number of grasp candidates, quality threshold
- **Motion planner**: velocity/acceleration limits, planning time, approach/retreat distance
- **Pipeline**: auto-execute mode, result saving

## Author

**Pablo Morilla Cabello** — pablomorillacabello@gmail.com
