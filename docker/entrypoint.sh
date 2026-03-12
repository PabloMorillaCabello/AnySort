#!/bin/bash
set -e

# Source ROS2
source /opt/ros/humble/setup.bash

# GraspGen on system Python 3.10 PYTHONPATH
# Note: SAM3 is NOT on system PYTHONPATH — it runs in /opt/sam3env (Python 3.12)
export PYTHONPATH="/opt/GraspGen:${PYTHONPATH}"
export CUDA_HOME=/usr/local/cuda

# Build workspace if not built yet
if [ ! -d "/ros2_ws/install" ]; then
    echo "==> First run: building ROS2 workspace..."
    cd /ros2_ws
    colcon build --symlink-install \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
        --parallel-workers "$(nproc)"
    echo "==> Workspace built successfully."
fi

# Source workspace
source /ros2_ws/install/setup.bash 2>/dev/null || true

# Copy sam3_server.py into workspace for easy access
if [ -f "/ros2_ws/src/graspgen_pipeline/scripts/sam3_server.py" ]; then
    cp /ros2_ws/src/graspgen_pipeline/scripts/sam3_server.py /ros2_ws/scripts/sam3_server.py 2>/dev/null || true
elif [ -f "/ros2_ws/scripts/sam3_server.py" ]; then
    true  # already in place
fi

# Verify key components
echo "============================================"
echo "  GraspGen Thesis Environment"
echo "--------------------------------------------"
echo "  ROS2:      Humble"
echo "  CUDA:      $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',')"
echo "  Python:    $(python3 --version 2>&1 | awk '{print $2}') (system / ROS2 / GraspGen)"
echo "  Python:    $(/opt/sam3env/bin/python --version 2>&1 | awk '{print $2}') (SAM3 venv)"
echo "  PyTorch:   $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "  SAM3:      $(/opt/sam3env/bin/python -c 'import sam3; print("OK")' 2>/dev/null || echo 'NOT FOUND')"
echo "  GraspGen:  $(python3 -c 'import grasp_gen; print("OK")' 2>/dev/null || echo 'NOT FOUND')"
echo "  GPU:       $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")' 2>/dev/null)"
echo "  DISPLAY:   ${DISPLAY:-NOT SET}"
echo "--------------------------------------------"
echo "  SAM3 server: /opt/sam3env/bin/python scripts/sam3_server.py"
echo "  Activate SAM3 venv: source /opt/sam3env/bin/activate"
echo "============================================"

exec "$@"
