# AnySort — Robotic Grasping Pipeline

**Master Thesis** — Text-Prompted Object Sorting with RGB-D Perception for Robotic Arms

An end-to-end pipeline that combines:
- **Orbbec Gemini 2** RGB-D depth camera
- **SAM3** (Segment Anything 3) text-prompted segmentation
- **GraspGen** (NVIDIA) 6-DOF grasp pose generation
- **Hand-eye calibration** for accurate robot manipulation
- **Multi-robot support** — Dobot CR with vacuum gripper, UR10 with OnRobot RG gripper (or add your own)

Everything runs inside Docker with CUDA 12.6, ROS2 Humble, and WSL2 on Windows. The main workflow is the **AnySort Tkinter application** — a graphical interface for capturing scenes, segmenting objects, generating grasps, and executing picks with automatic retry and batch processing.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation & Platform Setup](#installation--platform-setup)
  - [Windows + WSL2](#platform-windows--wsl2)
  - [Native Linux](#platform-native-linux)
- [Verify Your Installation](#verify-your-installation)
- [AnySort UI Overview](#anysort-ui-overview)
- [Multi-Robot Support](#multi-robot-support)
- [Batch / Sorting Mode](#batch--sorting-mode)
- [Project Structure](#project-structure)
- [Python Environments](#python-environments-inside-container)
- [Hand-Eye Calibration](#hand-eye-calibration)
- [Camera Calibration](#camera-calibration)
- [Testing](#testing)
- [Docker Reference](#docker-reference)
- [Exposed Ports](#exposed-ports)
- [Key Technical Notes](#key-technical-notes)
- [Troubleshooting](#troubleshooting)
- [Upstream Repositories](#upstream-repositories)

---

## Quick Start

Choose your platform below for detailed setup instructions:

- **[Windows + WSL2](#platform-windows--wsl2)** (current default) — Docker Desktop with WSL2 backend, X11 via VcXsrv/X410, USB camera via usbipd-win
- **[Native Linux](#platform-native-linux)** — Docker Engine with native GPU support, standard X11

Both paths take **30–45 minutes** for the initial Docker build, then launch is **single-click** (Windows) or **one command** (Linux).

---

## Installation & Platform Setup

The project runs in Docker with support for **Windows + WSL2** (current default) and **Native Linux**. Platform-specific setup differs slightly.

### Platform: Windows + WSL2

**Prerequisites:**
- Windows 11 with WSL2 installed and Docker Desktop with WSL2 backend
- NVIDIA GPU with CUDA 12.6+ support
- nvidia-container-toolkit installed in WSL2 (install via `wsl.exe -d <distro> sudo apt install nvidia-docker2`)
- X11 server for Windows (VcXsrv or X410)
- PowerShell with admin privileges (for USB attachment)
- HuggingFace account with **approved access** to [`facebook/sam3`](https://huggingface.co/facebook/sam3) and [`adithyamurali/GraspGenModels`](https://huggingface.co/adithyamurali/GraspGenModels) — request access on each model page before building (approval can take minutes to hours)
- **Orbbec SDK `.deb`** placed at `data/OrbbecSDK_v2.7.6_amd64.deb` — download from [Orbbec Developer Center](https://www.orbbec.com/developers/) (file is not in git)

**Setup Steps:**

1. **Clone the repository (in WSL2 terminal)**
   ```bash
   git clone https://github.com/PabloMorillaCabello/AnySort.git
   cd AnySort
   ```

2. **Copy and edit the env file**

   WSL2 / Git Bash:
   ```bash
   cp docker/.env.example docker/.env
   nano docker/.env
   ```
   PowerShell:
   ```powershell
   Copy-Item docker\.env.example docker\.env
   notepad docker\.env
   ```
   CMD:
   ```cmd
   copy docker\.env.example docker\.env
   notepad docker\.env
   ```
   Set these values:
   - `HF_TOKEN`: Generate at https://huggingface.co/settings/tokens
   - `TORCH_CUDA_ARCH_LIST`: Find your GPU's compute capability:
     ```bash
     nvidia-smi --query-gpu=compute_cap --format=csv,noheader
     # e.g. output "8.6" → set TORCH_CUDA_ARCH_LIST=8.6
     ```
     Common values: `8.6` (RTX 30-series / A500 / A2000), `8.9` (RTX 40-series), `7.5` (RTX 20-series)
   - `DISPLAY`: Leave as `host.docker.internal:0.0`

3. **Place Orbbec SDK file**
   Download `OrbbecSDK_v2.7.6_amd64.deb` from [Orbbec Developer Center](https://www.orbbec.com/developers/) and place it at:
   ```
   AnySort/data/OrbbecSDK_v2.7.6_amd64.deb
   ```

4. **Start X11 server on Windows**
   - Launch VcXsrv or X410
   - Enable "Disable access control" option

5. **Build Docker image (from repo root in WSL2 terminal)**
   ```bash
   docker compose -f docker/docker-compose.yml --env-file docker/.env build
   ```
   Takes 30–45 minutes on first build.

6. **Start container**
   ```bash
   docker compose -f docker/docker-compose.yml --env-file docker/.env up -d
   ```

7. **Attach Orbbec camera via USB (PowerShell as admin)**
   ```powershell
   # List USB devices (find Orbbec: VID 2bc5, PID 0670)
   usbipd list

   # Bind and attach (replace <ID> with bus ID from previous step)
   usbipd bind --busid <ID>
   usbipd attach --wsl --busid <ID>
   ```
   See [USB_WSL_Docker_Guide.md](USB_WSL_Docker_Guide.md) for detailed USB setup and troubleshooting.

8. **Launch AnySort**

   Double-click `AnySort.vbs` (no terminal window)

   Or from WSL2 terminal:
   ```bash
   docker compose -f docker/docker-compose.yml exec graspgen bash -c \
     "source /opt/GraspGen/.venv/bin/activate && cd /ros2_ws/app && python grasp_execute_pipeline.py"
   ```

**docker-compose.yml notes (Windows+WSL2 — already configured):**
- `DISPLAY=host.docker.internal:0.0` — X server runs on Windows side
- `extra_hosts: host.docker.internal` — enables container to resolve Windows host
- `/tmp/.X11-unix` volume — NOT mounted (not needed on Windows)

### Platform: Native Linux

**Prerequisites:**
- Docker Engine + Docker Compose v2
  ```bash
  # Install Docker Engine (Ubuntu/Debian):
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker $USER  # log out and back in after this
  ```
- NVIDIA GPU with CUDA 12.6+ and nvidia-container-toolkit on host:
  ```bash
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt update && sudo apt install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- X11 display server (default on most Linux desktops)
- HuggingFace account with **approved access** to [`facebook/sam3`](https://huggingface.co/facebook/sam3) and [`adithyamurali/GraspGenModels`](https://huggingface.co/adithyamurali/GraspGenModels) — request access on each model page before building (approval can take minutes to hours)
- **Orbbec SDK `.deb`** placed at `data/OrbbecSDK_v2.7.6_amd64.deb` — download from [Orbbec Developer Center](https://www.orbbec.com/developers/) (file is not in git)

**Setup Steps:**

1. **Clone the repository**
   ```bash
   git clone https://github.com/PabloMorillaCabello/AnySort.git
   cd AnySort
   ```

2. **Copy and edit the env file**
   ```bash
   cp docker/.env.example docker/.env
   ```
   > **Note:** `.env.example` starts with a dot — hidden file. Use `ls -la docker/` to confirm it's there, or press `Ctrl+H` in your file manager.
   ```bash
   nano docker/.env
   ```
   Set these values:
   - `HF_TOKEN`: Generate at https://huggingface.co/settings/tokens
   - `TORCH_CUDA_ARCH_LIST`: Find your GPU's compute capability:
     ```bash
     nvidia-smi --query-gpu=compute_cap --format=csv,noheader
     # e.g. output "8.6" → set TORCH_CUDA_ARCH_LIST=8.6
     ```
   - `DISPLAY`: Leave blank — overridden at runtime by `${DISPLAY}`

3. **Place Orbbec SDK file**
   Download `OrbbecSDK_v2.7.6_amd64.deb` from [Orbbec Developer Center](https://www.orbbec.com/developers/) and place it at:
   ```
   AnySort/data/OrbbecSDK_v2.7.6_amd64.deb
   ```

4. **Modify docker-compose.yml for Linux**
   Edit `docker/docker-compose.yml` (comments already guide you):
   ```yaml
   environment:
     # Comment out this line:
     # - DISPLAY=host.docker.internal:0.0
     # Uncomment this line:
     - DISPLAY=${DISPLAY}

   volumes:
     # Uncomment this line:
     - /tmp/.X11-unix:/tmp/.X11-unix

   # Comment out the entire extra_hosts block:
   # extra_hosts:
   #   - "host.docker.internal:host-gateway"
   ```

5. **Verify GPU access**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
   ```
   Should show GPU info without error.

6. **Build Docker image**
   ```bash
   docker compose -f docker/docker-compose.yml --env-file docker/.env build
   ```
   Takes 30–45 minutes on first build.

7. **Start container**
   ```bash
   docker compose -f docker/docker-compose.yml --env-file docker/.env up -d
   ```

8. **Plug in Orbbec camera (USB 3.0)**
   No special setup — works natively on Linux. Verify with:
   ```bash
   lsusb | grep 2bc5
   ```

9. **Launch AnySort**

   **Option A — shell script (terminal, handles xhost automatically):**
   ```bash
   chmod +x AnySort.sh   # only needed once
   ./AnySort.sh
   ```

   **Option B — double-click (no terminal):**
   ```bash
   chmod +x AnySort.desktop   # only needed once
   ```
   Then double-click `AnySort.desktop` in your file manager. If prompted, choose "Run" or "Trust and Launch".

   > **Note:** `xhost +local:docker` is required each reboot for X11 display. `AnySort.sh` and `AnySort.desktop` run it automatically. If launching manually, run it first:
   > ```bash
   > xhost +local:docker
   > ```

---

## Verify Your Installation

Run these checks in order after a fresh install. Enter the container first:
```bash
docker compose -f docker/docker-compose.yml exec graspgen bash
```

### Step 1 — Full environment check (no hardware needed)
```bash
bash scripts/test_environment.sh
```
Verifies: Python versions, CUDA, PyTorch, GraspGen imports, SAM3 imports, Dobot API, ROS2, model weights.
All lines should print `OK` or `PASS`. Fix any failures before continuing.

### Step 2 — GPU + GraspGen inference
```bash
python3 scripts/test_graspgen.py --no-display
```
Expected: PointNet++ CUDA extensions load, model weights found, inference runs without error.

### Step 3 — SAM3 segmentation model
```bash
/opt/sam3env/bin/python scripts/test_sam3.py --no-display
```
Expected: model loads from cache, inference runs. Output saved to `results/sam3_test_result.png`.
> First run downloads model weights (~2 GB) — takes a few minutes if not cached.

### Step 4 — Camera (requires Orbbec plugged in)
```bash
python3 scripts/view_camera.py
```
RGB and depth windows should open. Press `s` to save a frame, `q` to quit.
If camera not detected, verify USB connection:
- **Windows**: run `usbipd attach` again in PowerShell (admin)
- **Linux**: run `lsusb | grep 2bc5`

### Step 5 — End-to-end pipeline (no hardware needed)
```bash
python3 scripts/test_full_pipeline.py
```
Runs a synthetic scene through the full pipeline (camera → SAM3 → GraspGen). No robot or camera required.

### Quick reference

| Test | Command | Hardware |
|------|---------|----------|
| Environment | `bash scripts/test_environment.sh` | None |
| GraspGen | `python3 scripts/test_graspgen.py --no-display` | GPU only |
| SAM3 | `/opt/sam3env/bin/python scripts/test_sam3.py --no-display` | GPU only |
| Camera | `python3 scripts/view_camera.py` | Orbbec camera |
| Full pipeline | `python3 scripts/test_full_pipeline.py` | GPU only |

---

## AnySort UI Overview

A **5-column layout** with status log bar:

```
┌─────────────┬──────────────┬────────────┬──────────────┬──────────────┐
│   CAMERA    │  GRASPGEN    │   ROBOT    │  WORD LIST   │  EXECUTION   │
├─────────────┼──────────────┼────────────┼──────────────┼──────────────┤
│ • Connect   │ • Calibration│ • Type sel.│ • Add/Remove │ • Status     │
│ • Capture   │ • Calib file │ • IP entry │ • Load/Save  │ • Run single │
│ • Save/Load │ • Tool type  │ • Connect  │ • Auto-load  │ • Batch mode │
│ • ROI select│ • Prompt     │ • Speed    │              │ • Stop/Retry │
│ • Mask view │ • Options    │ • Recover  │              │ • Meshcat    │
│             │ • Grasps (N) │ • Home/Go  │              │              │
│             │ • Navigator  │ • Sort/Go  │              │              │
└─────────────┴──────────────┴────────────┴──────────────┴──────────────┘
└────────────────────────────── LOG BAR ─────────────────────────────────┘
```

### Column: Camera
- **Connect** — Initialize Orbbec camera
- **Capture** — Snapshot current frame
- **Save/Load scene** — Save as `.npz` for later analysis
- **Mask & ROI** — Show segmentation overlay, define 4-point polygon ROI

### Column: GraspGen
- **Calibration status label** — Shows loaded calibration file or status
- **Calibration file dropdown** — Browse and switch between available `.npz` files in `data/calibration/` without restart
- **Tool dropdown** — Select end-effector type (e.g., Dobot Vacuum, OnRobot RG gripper)
- **Object prompt** — Enter text description (e.g., "red cylinder")
- **Options** — Collision filter, reachability filter, debug step-by-step
- **Grasps** — Shows N candidates, best N returned by confidence
- **Grasp navigator** — Previous/Next buttons

### Column: Robot
- **Robot type dropdown** — Select robot model (DobotCR, UR10, or custom)
- **IP entry & Connect** — Robot TCP/IP address (Dobot default: `192.168.5.1`, UR10: check pendant)
- **Motion parameters** — Speed %, Approach offset (mm), TCP Z offset (mm)
- **Recover Robot** — Reset alarm, re-enable, go home
- **Save/Go Home** — Define and return to home position
- **Save/Go Sort** — Define drop-off location
- **Actions** — Emergency stop, motion test buttons

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

## Multi-Robot Support

The pipeline supports multiple robot types through a modular architecture. Currently implemented:

- **Dobot CR** — TCP/IP control on ports 29999 (commands) and 30004 (real-time feedback), vacuum gripper via digital output
- **UR10** — `ur_rtde` real-time interface, OnRobot RG gripper via Dashboard program execution

### Adding a New Robot

Create a new driver in `app/robots/`:

1. Copy `TEMPLATE.py` to `my_robot.py`
2. Implement the `RobotBase` abstract interface:
   ```python
   from app.robots.base import RobotBase

   class MyRobotDriver(RobotBase):
       def connect(self, ip: str) -> bool: ...
       def disconnect(self) -> bool: ...
       def move_to(self, pose: list, speed_pct: float) -> bool: ...
       def set_home(self, pose: list) -> None: ...
       def go_home(self) -> bool: ...
       def recover(self) -> bool: ...
   ```
3. Register in `app/robots/__init__.py`
4. Select from UI dropdown

### Adding a New End-Effector / Tool

Create a new tool driver in `app/tools/`:

1. Copy `TEMPLATE.py` to `my_tool.py`
2. Implement the `ToolBase` abstract interface:
   ```python
   from app.tools.base import ToolBase

   class MyTool(ToolBase):
       def initialize(self) -> bool: ...
       def grasp(self, duration_s: float = 1.0) -> bool: ...
       def release(self, duration_s: float = 1.0) -> bool: ...
       def cleanup(self) -> None: ...
   ```
3. Register in `app/tools/__init__.py`
4. Select from UI dropdown

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
│   ├── hand_eye_calibration.py   # ChArUco calibration UI (named pose sets & calib saves)
│   ├── calibration_tester.py     # Calibration validation + correction (calib file selector)
│   ├── camera_calibration.py     # Camera intrinsics (OpenCV)
│   ├── sam3_server.py            # Persistent SAM3 Unix socket server
│   ├── orbbec_quiet.py           # Suppresses OrbbecSDK C-level stderr
│   ├── pipeline_positions.json   # Saved Home/Sort positions
│   ├── pipeline_roi.json         # Saved ROI polygon
│   │
│   ├── robots/                   # Multi-robot drivers (modular)
│   │   ├── __init__.py          # Robot registry
│   │   ├── base.py              # RobotBase abstract class
│   │   ├── dobot_cr.py          # Dobot CR TCP/IP driver
│   │   ├── ur10.py              # UR10 ur_rtde driver
│   │   └── TEMPLATE.py          # Template for new robots
│   │
│   └── tools/                    # End-effector drivers (modular)
│       ├── __init__.py          # Tool registry
│       ├── base.py              # ToolBase abstract class
│       ├── dobot_vacuum.py      # Dobot vacuum (digital output)
│       ├── onrobot_urscript.py  # OnRobot RG gripper (Dashboard program execution)
│       └── TEMPLATE.py          # Template for new tools
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
│   │   ├── hand_eye_calib.npz   # Default binary calibration matrix
│   │   ├── hand_eye_calib.json  # Default human-readable JSON
│   │   ├── hand_eye_calib_{name}.npz      # Named calibration saves
│   │   ├── hand_eye_calib_{name}.json
│   │   ├── hand_eye_calib_{ts}.npz        # Timestamped backups (auto-created)
│   │   ├── auto_calib_poses.json          # Default pose set (26 pre-programmed poses)
│   │   ├── auto_calib_poses_{name}.json   # Named pose sets (e.g., ur10, dobot)
│   │   ├── camera_intrinsics.npz
│   │   └── camera_intrinsics.json
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

Before executing grasps, the robot must know where the camera is relative to its base frame. The system now supports **named calibration pose sets** and **named calibration saves** for managing multiple robot configurations.

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

### Named Pose Sets

Save and load different sets of calibration poses (e.g., one per robot type):

- **Save as** field → Save current captured poses as `auto_calib_poses_{name}.json`
- Example: `auto_calib_poses_ur10.json`, `auto_calib_poses_dobot.json`
- On next run, load a named pose set via the UI combobox
- Empty "Save as" field overwrites the default `auto_calib_poses.json`

**Step 2: Solve**
- Click "Solve" → runs `cv2.calibrateHandEye()`
- Typical good error: < 5 mm

### Named Calibration Saves

Save calibration results by name for different robot/gripper combinations:

- **Save as** field (GraspGen panel) → Save calibration as `hand_eye_calib_{name}.npz`
- Example: `hand_eye_calib_ur10.npz`, `hand_eye_calib_dobot_v2.npz`
- Timestamped backup always created: `hand_eye_calib_{timestamp}.npz` (auto-created)
- Empty name overwrites default: `hand_eye_calib.npz`
- Saved calibrations appear in **Calibration file dropdown** in main AnySort UI

Files saved:
- `data/calibration/hand_eye_calib_{name}.npz` (binary, used by app)
- `data/calibration/hand_eye_calib_{name}.json` (human-readable)

### Load Calibration in Main App

In the **GraspGen column**, use the **Calibration file dropdown** to:
- Browse available `.npz` files in `data/calibration/`
- Switch calibrations without restarting the app
- Display currently loaded calibration in status label

**Step 3: Validate (optional but recommended)**
```bash
python calibration_tester.py --robot-ip 192.168.5.1
```

Tests calibration accuracy by:
1. Running test points with robot
2. Measuring predicted vs. actual error
3. Correcting systematic offsets with 6-DOF sliders
4. Saving corrected calibration if needed

The calibration file combobox lists all available `.npz` files; use the Browse button to load a specific file.

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

## OnRobot RG Gripper Setup (UR10)

The OnRobot RG gripper is controlled via UR Dashboard program execution. Programs must be created on the robot pendant and saved as `.urp` files.

### Create Programs on the Pendant

1. **Power on UR10** and access the pendant (teach pad)
2. **Create a new program** (or modify existing):
   - Open **Program** menu → New
   - Name it `gripper_open`
3. **Add the RG gripper control nodes**:
   - From menu, insert: **Robotics** → **OnRobot** → **RG** → **Open RG6** (or your model)
   - Save the program as `gripper_open.urp`
4. **Repeat for close**:
   - Create program `gripper_close`
   - Insert: **Robotics** → **OnRobot** → **RG** → **Close RG6**
   - Save as `gripper_close.urp`

### Gripper Timing

The `onrobot_urscript.py` driver waits for program completion + settle times:

- `grasp_wait_s = 2.0` — time after "Close" program finishes before next action
- `release_wait_s = 1.5` — time after "Open" program finishes before next action

**Better approach (recommended):** Add a **Wait node** inside the `.urp` programs on the pendant:
1. In program, after the gripper command, insert **Flow** → **Wait**
2. Set duration to match gripper settle time
3. Save the program
4. Set `grasp_wait_s = 0.2` and `release_wait_s = 0.2` in the code (just for handshake)

This ensures gripper timing is built into the program logic, not dependent on container delays.

### Testing OnRobot Gripper

From inside the container:
```bash
cd app
python -c "from tools.onrobot_urscript import OnRobotRGGripper; g = OnRobotRGGripper('192.168.X.X'); g.initialize(); g.grasp(2.0); g.release(1.5)"
```

Verify:
- UR pendant shows program execution starting
- Gripper physically opens/closes
- No timeout errors in log

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
| `403 / access denied on HuggingFace during build` | Three checks: (1) request access at huggingface.co/facebook/sam3 and wait for approval email; (2) ensure build uses `--env-file docker/.env` so token is passed; (3) token must have **read** scope for gated repos (check at huggingface.co/settings/tokens) |
| `COPY failed: data/OrbbecSDK*.deb not found` | Download `OrbbecSDK_v2.7.6_amd64.deb` from Orbbec Developer Center and place at `data/OrbbecSDK_v2.7.6_amd64.deb` |
| `docker/.env not found` after cloning | `.env` is gitignored — run `cp docker/.env.example docker/.env` then fill in your `HF_TOKEN` |
| `CUDA out of memory` | Reduce `--num_grasps` in app or use smaller model |
| `ChArUco board not detected` | Ensure good overhead lighting, print larger |
| `Calibration error > 10 mm` | Collect more poses (≥20), ensure board fully visible |
| `Grasp off-target by 50+ mm` | Use `calibration_tester.py` to measure/correct systematic bias |
| `Camera not found (USB)` | Run `usbipd attach` again in PowerShell (WSL2) |
| `Meshcat not opening` | Ensure port 7000 is exposed; check firewall; try `http://127.0.0.1:7000` |
| `Dobot connection refused` | Verify IP (default `192.168.5.1`), ensure same network subnet |
| `UR10 connection refused` | Verify IP, ensure robot is in Manual mode (pendant), check firewall on robot controller |
| `OnRobot gripper not responding` | Verify programs `gripper_open.urp` and `gripper_close.urp` exist on pendant; check Dashboard server enabled (robot settings); test gripper manually on pendant first |
| `Gripper timing too slow / collisions during grasp` | Add Wait nodes to `.urp` programs on pendant; adjust `grasp_wait_s` and `release_wait_s` in `tools/onrobot_urscript.py` |
| `Calibration file dropdown empty` | Ensure at least one `.npz` file exists in `data/calibration/`; run `hand_eye_calibration.py` to create one |
| `Cannot switch calibration in main app` | Restart AnySort; verify calibration file readable; check file is valid `.npz` (not corrupted) |
| `Robot not recognized in dropdown` | Verify robot driver imported in `app/robots/__init__.py`; check for syntax errors in driver file |
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
