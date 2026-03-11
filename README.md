# GraspGen - Automated Grasp Generation Pipeline

> **Master Thesis** — Text-Prompted Object Grasping with RGB-D Perception for UR Robots

An end-to-end robotic grasping pipeline that combines RGB-D perception (Orbbec Gemini 2), text-prompted segmentation ([SAM3](https://github.com/facebookresearch/sam3)), learned grasp pose generation ([GraspGen](https://github.com/NVlabs/GraspGen)), and motion planning (MoveIt2) to autonomously pick objects with a UR robot and Robotiq 3F gripper — all orchestrated through ROS2 Humble.

## Pipeline Overview

```
Orbbec Gemini 2 ──> SAM3 Segmentation ──> GraspGen ──> MoveIt2 Planner ──> UR Robot
    (RGB-D)         (text prompt)       (point cloud     (trajectory)     + Robotiq 3F
                                         → 6DOF grasps)
```

## Environment Stack

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA | 12.6 | Unified for SAM3 + GraspGen |
| Python | 3.12 | Required by SAM3 |
| PyTorch | 2.7.0 | Required by SAM3, compatible with GraspGen |
| ROS2 | Humble | On Ubuntu 22.04 |
| SAM3 | latest | `facebook/sam3` from HuggingFace |
| GraspGen | latest | `adithyamurali/GraspGenModels` from HuggingFace |
| Orbbec SDK | v2-main | OrbbecSDK_ROS2 for Gemini 2 |

## Repository Structure

```
GraspGen_Thesis_Repo/
├── docker/                           # Containerized dev environment
│   ├── Dockerfile                    # CUDA 12.6 + ROS2 Humble + SAM3 + GraspGen
│   ├── docker-compose.yml            # One-command startup with GPU
│   ├── entrypoint.sh                 # Auto-build + environment banner
│   ├── requirements.txt              # Additional Python deps
│   └── .env.example                  # Template for HF_TOKEN + CUDA arch
├── models/                           # Model weights (git-ignored, downloaded at build)
│   ├── sam3/                         # facebook/sam3 checkpoints
│   └── graspgen/                     # adithyamurali/GraspGenModels checkpoints
├── ros2_ws/src/
│   ├── graspgen_pipeline/            # Main pipeline ROS2 package
│   │   ├── graspgen_pipeline/
│   │   │   ├── camera_node.py            # RGB-D sync relay (Orbbec topics)
│   │   │   ├── segmentation_node.py      # SAM3 text-prompted segmentation
│   │   │   ├── grasp_generator_node.py   # GraspGen: depth+mask → grasp poses
│   │   │   ├── motion_planner_node.py    # MoveIt2 planning + execution
│   │   │   └── pipeline_orchestrator.py  # Coordinates full pick cycle
│   │   ├── launch/
│   │   │   └── full_pipeline.launch.py   # Launches everything
│   │   └── config/
│   │       └── pipeline_params.yaml      # All tuneable parameters
│   └── robotiq_3f_driver/            # Robotiq 3F gripper Modbus driver
├── scripts/                          # Utility & test scripts
│   ├── download_models.sh            # Download SAM3 + GraspGen weights
│   ├── setup_orbbec.sh               # Host udev rules for camera
│   ├── build_workspace.sh            # colcon build helper
│   ├── test_environment.sh           # Verify full environment setup
│   ├── test_sam3.py                  # Test SAM3 loading + inference
│   ├── test_graspgen.py              # Test GraspGen loading + inference
│   ├── test_camera.sh                # Test Orbbec Gemini 2 via ROS2
│   └── test_full_pipeline.py         # End-to-end integration test
├── config/                           # Global config overrides
├── data/                             # Captured data (git-ignored)
├── results/                          # Experiment results (git-ignored)
└── docs/                             # Thesis notes & documentation
```

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 12.6+ support (RTX 3090/4090, A100, etc.)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- HuggingFace account with access to:
  - [facebook/sam3](https://huggingface.co/facebook/sam3) (request access)
  - [adithyamurali/GraspGenModels](https://huggingface.co/adithyamurali/GraspGenModels)
- Orbbec Gemini 2 camera (USB)
- UR robot (UR3e/UR5e/UR10e/UR16e) on the network
- Robotiq 3F gripper (serial)

### Step 1: Clone and configure

```bash
git clone git@github.com:YOUR_USERNAME/GraspGen_Thesis_Repo.git
cd GraspGen_Thesis_Repo

# Set your secrets
cp docker/.env.example docker/.env
nano docker/.env   # Add your HF_TOKEN and set TORCH_CUDA_ARCH_LIST for your GPU
```

### Step 2: Build the Docker image

```bash
cd docker

# Build (this takes ~30-45 min on first run):
#   - Installs CUDA 12.6 + ROS2 Humble
#   - Clones and installs SAM3 from source
#   - Clones and installs GraspGen from source
#   - Compiles PointNet++ CUDA extensions
#   - Clones OrbbecSDK_ROS2
#   - Downloads model weights (if HF_TOKEN is set)
docker compose build

# Start the container
docker compose up -d

# Enter the dev environment
docker compose exec graspgen bash
```

### Step 3: Setup camera on host (once)

```bash
# On the HOST machine (not inside Docker):
./scripts/setup_orbbec.sh
```

### Step 4: Verify the environment

```bash
# Inside the container:
./scripts/test_environment.sh       # Check all dependencies
python3 scripts/test_sam3.py        # Test SAM3 model
python3 scripts/test_graspgen.py    # Test GraspGen model
./scripts/test_camera.sh            # Test Orbbec camera (needs camera plugged in)
```

### Step 5: Build and run the pipeline

```bash
# Build ROS2 workspace (first time only)
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

### Step 6: Trigger a pick cycle

```bash
# In another terminal:
docker compose exec graspgen bash
ros2 service call /pipeline_orchestrator/trigger_pick std_srvs/srv/Trigger

# Change the text prompt at runtime:
ros2 topic pub --once /segmentation/set_prompt std_msgs/String "data: 'bottle'"
```

### Step 7: Integration test (no hardware)

```bash
# Terminal 1: pipeline in sim mode
ros2 launch graspgen_pipeline full_pipeline.launch.py \
    use_sim:=true launch_camera:=false

# Terminal 2: run integration test with synthetic data
python3 scripts/test_full_pipeline.py
```

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
