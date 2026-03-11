#!/bin/bash
# =============================================================================
# Build the ROS2 workspace (run inside the Docker container)
# =============================================================================
set -e

source /opt/ros/humble/setup.bash
cd /ros2_ws

echo "=== Building ROS2 workspace ==="
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

echo "=== Sourcing workspace ==="
source install/setup.bash

echo "=== Build complete ==="
echo "Packages built:"
colcon list --names-only
