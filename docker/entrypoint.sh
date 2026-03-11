#!/bin/bash
set -e

# Source ROS2
source /opt/ros/humble/setup.bash

# Set PYTHONPATH for SAM3 and GraspGen
export PYTHONPATH="/opt/sam3:/opt/GraspGen:${PYTHONPATH}"
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

# Verify key components
echo "============================================"
echo "  GraspGen Thesis Environment"
echo "--------------------------------------------"
echo "  ROS2:     Humble"
echo "  CUDA:     $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',')"
echo "  Python:   $(python3 --version 2>&1 | awk '{print $2}')"
echo "  PyTorch:  $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "  SAM3:     $(python3 -c 'import sam3; print("OK")' 2>/dev/null || echo 'NOT FOUND')"
echo "  GraspGen: $(python3 -c 'import grasp_gen; print("OK")' 2>/dev/null || echo 'NOT FOUND')"
echo "  GPU:      $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")' 2>/dev/null)"
echo "============================================"

exec "$@"
