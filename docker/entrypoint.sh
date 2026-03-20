#!/bin/bash
set -e

# Source ROS2
source /opt/ros/humble/setup.bash

# Activate GraspGen Python 3.10 venv (uv-managed)
source /opt/GraspGen/.venv/bin/activate

# GraspGen + Dobot API on PYTHONPATH
# Note: SAM3 is NOT on system PYTHONPATH — it runs in /opt/sam3env (Python 3.12)
export PYTHONPATH="/opt/GraspGen:/opt/Dobot_hv:${PYTHONPATH}"
export CUDA_HOME=/usr/local/cuda

# Source ROS2 workspace (OrbbecSDK_ROS2 + project packages)
source /ros2_ws/install/setup.bash 2>/dev/null || true

# Rebuild workspace if new packages were mounted but not yet built
if [ -d "/ros2_ws/src/graspgen_pipeline" ] && [ ! -d "/ros2_ws/install/graspgen_pipeline" ]; then
    echo "==> New packages detected, rebuilding workspace..."
    cd /ros2_ws
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
        --parallel-workers "$(nproc)"
    source /ros2_ws/install/setup.bash
    echo "==> Workspace rebuilt."
fi

# Copy sam3_server.py into workspace for easy access
if [ -f "/ros2_ws/src/graspgen_pipeline/scripts/sam3_server.py" ]; then
    cp /ros2_ws/src/graspgen_pipeline/scripts/sam3_server.py /ros2_ws/scripts/sam3_server.py 2>/dev/null || true
fi

# Verify key components
echo "============================================"
echo "  GraspGen Thesis Environment"
echo "--------------------------------------------"
echo "  ROS2:      Humble"
echo "  CUDA:      $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',')"
echo "  Python:    $(python3 --version 2>&1 | awk '{print $2}') (GraspGen venv / uv-managed)"
echo "  Python:    $(/opt/sam3env/bin/python --version 2>&1 | awk '{print $2}') (SAM3 venv)"
echo "  PyTorch:   $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "  GraspGen:  $(python3 -c 'import grasp_gen; print("OK")' 2>/dev/null || echo 'NOT FOUND')"
echo "  SAM3:      $(/opt/sam3env/bin/python -c 'import sam3; print("OK")' 2>/dev/null || echo 'NOT FOUND')"
echo "  Dobot API: $(python3 -c 'import dobot_api; print("OK")' 2>/dev/null || echo 'NOT FOUND')"
echo "  Orbbec:    $(ros2 pkg list 2>/dev/null | grep -q orbbec_camera && echo 'OK' || echo 'NOT FOUND')"
echo "  GPU:       $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")' 2>/dev/null)"
echo "  Camera:    $(lsusb 2>/dev/null | grep -i 'orbbec\|2bc5' | head -1 || echo 'NOT DETECTED')"
echo "  DISPLAY:   ${DISPLAY:-NOT SET}"
echo "--------------------------------------------"
echo "  Camera:       ros2 launch orbbec_camera gemini2.launch.py"
echo "  SAM3 server:  /opt/sam3env/bin/python scripts/sam3_server.py"
echo "  Switch envs:  sam3_activate / graspgen_activate"
echo "============================================"

exec "$@"
