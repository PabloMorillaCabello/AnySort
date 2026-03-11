#!/bin/bash
# =============================================================================
# Test: Verify the full development environment is set up correctly.
# Run inside the Docker container.
#
# Usage: ./scripts/test_environment.sh
# =============================================================================
set -e

PASS=0
FAIL=0
WARN=0

pass() { echo "  [PASS] $1"; PASS=$((PASS+1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
warn() { echo "  [WARN] $1"; WARN=$((WARN+1)); }

echo "============================================"
echo "  Environment Verification Tests"
echo "============================================"

# --- 1. Python version ---
echo ""
echo "--- Python ---"
PY_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 12 ]; then
    pass "Python $PY_VERSION (>= 3.12 required)"
else
    fail "Python $PY_VERSION (>= 3.12 required)"
fi

# --- 2. CUDA ---
echo ""
echo "--- CUDA ---"
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
    pass "CUDA $CUDA_VER"
else
    fail "nvcc not found"
fi

if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    pass "PyTorch CUDA available ($GPU_NAME)"
else
    fail "PyTorch cannot see GPU"
fi

# --- 3. PyTorch ---
echo ""
echo "--- PyTorch ---"
PT_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ -n "$PT_VER" ]; then
    pass "PyTorch $PT_VER"
else
    fail "PyTorch not importable"
fi

TV_VER=$(python3 -c "import torchvision; print(torchvision.__version__)" 2>/dev/null)
if [ -n "$TV_VER" ]; then
    pass "TorchVision $TV_VER"
else
    fail "TorchVision not importable"
fi

# --- 4. SAM3 ---
echo ""
echo "--- SAM3 ---"
if python3 -c "from sam3 import build_sam3_image_model; print('OK')" 2>/dev/null; then
    pass "sam3 package importable (native API)"
else
    fail "sam3 package NOT importable"
fi

if python3 -c "from transformers import Sam3Model; print('OK')" 2>/dev/null; then
    pass "Sam3Model importable (Transformers API)"
else
    warn "Sam3Model not in transformers (may need newer version)"
fi

# --- 5. GraspGen ---
echo ""
echo "--- GraspGen ---"
if python3 -c "from grasp_gen.sampler import GraspGenSampler; print('OK')" 2>/dev/null; then
    pass "grasp_gen.sampler importable"
else
    fail "grasp_gen.sampler NOT importable"
fi

if python3 -c "from grasp_gen.utils import load_grasp_cfg; print('OK')" 2>/dev/null; then
    pass "grasp_gen.utils importable"
else
    fail "grasp_gen.utils NOT importable"
fi

if python3 -c "import pointnet2_ops; print('OK')" 2>/dev/null; then
    pass "pointnet2_ops CUDA extensions compiled"
else
    fail "pointnet2_ops NOT available (run install_pointnet.sh)"
fi

if python3 -c "import spconv; print('OK')" 2>/dev/null; then
    pass "spconv importable"
else
    fail "spconv NOT importable"
fi

if python3 -c "import torch_geometric; print('OK')" 2>/dev/null; then
    pass "torch_geometric importable"
else
    fail "torch_geometric NOT importable"
fi

# --- 6. ROS2 ---
echo ""
echo "--- ROS2 ---"
if command -v ros2 &>/dev/null; then
    ROS_DISTRO=$(ros2 --version 2>&1 | head -1 || echo "unknown")
    pass "ROS2 CLI available"
else
    fail "ros2 command not found"
fi

if [ -n "$ROS_DISTRO" ] || [ -f "/opt/ros/humble/setup.bash" ]; then
    pass "ROS2 Humble installed"
else
    fail "ROS2 Humble not found"
fi

# --- 7. ROS2 Packages ---
echo ""
echo "--- ROS2 Packages ---"
for pkg in orbbec_camera ur_robot_driver moveit; do
    if ros2 pkg list 2>/dev/null | grep -q "$pkg"; then
        pass "ros2 pkg: $pkg"
    else
        warn "ros2 pkg: $pkg not found (may need workspace build)"
    fi
done

# --- 8. Model Weights ---
echo ""
echo "--- Model Weights ---"
if [ -d "/opt/models/sam3" ] && [ "$(ls -A /opt/models/sam3 2>/dev/null)" ]; then
    pass "SAM3 weights present in /opt/models/sam3"
else
    warn "SAM3 weights NOT found (run: ./scripts/download_models.sh)"
fi

if [ -d "/opt/models/graspgen" ] && [ "$(ls -A /opt/models/graspgen 2>/dev/null)" ]; then
    pass "GraspGen weights present in /opt/models/graspgen"
else
    warn "GraspGen weights NOT found (run: ./scripts/download_models.sh)"
fi

# --- 9. Key Python packages ---
echo ""
echo "--- Additional Python Packages ---"
for pkg in cv2 open3d trimesh pymodbus serial hydra; do
    if python3 -c "import $pkg" 2>/dev/null; then
        pass "Python: $pkg"
    else
        fail "Python: $pkg NOT importable"
    fi
done

# --- Summary ---
echo ""
echo "============================================"
echo "  Results: $PASS passed, $FAIL failed, $WARN warnings"
echo "============================================"

if [ $FAIL -gt 0 ]; then
    echo "  Some tests FAILED. Fix the issues above before running the pipeline."
    exit 1
else
    echo "  Environment is ready!"
    exit 0
fi
