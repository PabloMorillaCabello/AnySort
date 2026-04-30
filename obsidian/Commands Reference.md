# Commands Reference

> Quick reference for all common commands.

---

## Launch AnySort (Primary)

```bash
# Windows — no terminal window (preferred)
AnySort.vbs

# Windows — with terminal
AnySort.cmd

# Inside container
cd /ros2_ws/app && python anysort.py
```

---

## Docker

```bash
# Build
docker compose -f docker/docker-compose.yml build --no-cache

# Start container
docker compose -f docker/docker-compose.yml up -d

# Enter container (bash)
docker compose -f docker/docker-compose.yml exec graspgen /bin/bash

# Windows shortcut
docker/Bash.cmd

# Stop container
docker compose -f docker/docker-compose.yml down
```

---

## Camera

```bash
# Live viewer (no ROS2)
python3 /ros2_ws/scripts/view_camera.py
python3 /ros2_ws/scripts/view_camera.py --align
python3 /ros2_ws/scripts/view_camera.py --ir --align --pointcloud

# ROS2 camera launch (use RGB format, not MJPG)
ros2 launch orbbec_camera gemini2.launch.py color_format:=RGB
```

---

## Calibration

```bash
# Camera intrinsics
python /ros2_ws/app/camera_calibration.py

# Hand-eye calibration
python /ros2_ws/app/hand_eye_calibration.py --robot-ip 192.168.5.1

# Validate calibration
python /ros2_ws/app/calibration_tester.py
```

---

## Tests

```bash
# Environment check
bash /ros2_ws/scripts/test_environment.sh

# GraspGen model load test
python3 /ros2_ws/scripts/test_graspgen.py

# SAM3 load test
python3 /ros2_ws/scripts/test_sam3.py

# Camera connectivity
bash /ros2_ws/scripts/test_camera.sh

# Full pipeline end-to-end
python3 /ros2_ws/scripts/test_full_pipeline.py

# Gripper test
python3 /ros2_ws/scripts/test_gripper.py
```

---

## Robot (Dobot)

```bash
# Dobot UI (standalone)
python3 /opt/Dobot_hv/main_UI.py
```

Python snippet to connect:
```python
from robots import create_robot
robot = create_robot("Dobot CR", ip="192.168.5.1")
robot.enable()
pose = robot.get_pose()   # (x, y, z, rx, ry, rz)
```

---

## Robot (UR10)

```python
from robots import create_robot
robot = create_robot("UR10", ip="192.168.X.X")
robot.enable()
pose = robot.get_pose()
```

---

## USB Passthrough (Windows PowerShell, Admin)

```powershell
# List USB devices
usbipd list

# Bind device (first time)
usbipd bind --busid <BUSID>

# Attach to WSL2
usbipd attach --wsl --busid <BUSID>

# Or use the script:
.\scripts\reattach.ps1
```

---

## Python Environments (inside container)

```bash
graspgen_activate   # /opt/GraspGen/.venv/ (Python 3.10)
sam3_activate       # /opt/sam3env/ (Python 3.12)
```

---

## ROS2 Workspace

```bash
# Build
bash /ros2_ws/scripts/build_workspace.sh
# or
cd /ros2_ws && colcon build --symlink-install

# Source
source /ros2_ws/install/setup.bash
```

---

## Download Models

```bash
bash /ros2_ws/scripts/download_models.sh
```

---

## Links
- [[Pipeline/AnySort Pipeline|AnySort Pipeline]] — what the launch command runs
- [[Infrastructure/Docker Setup|Docker Setup]] — container details
- [[Infrastructure/USB Passthrough|USB Passthrough]] — camera forwarding
