#!/bin/bash
# =============================================================================
# Test: Verify the full development environment is set up correctly.
# Run inside the Docker container.
#
# Architecture:
#   - System Python 3.10: ROS2 Humble + GraspGen + common packages
#   - SAM3 venv Python 3.12: /opt/sam3env/bin/python (SAM3 only)
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

# --- 1. Python versions (dual architecture) ---
echo ""
echo "--- Python (Dual Architecture) ---"
PY_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MINOR" -eq 10 ]; then
    pass "System Python $PY_VERSION (3.10 required for ROS2 Humble)"
else
    fail "System Python $PY_VERSION (expected 3.10 for ROS2 Humble)"
fi

if [ -x "/opt/sam3env/bin/python" ]; then
    SAM3_PY=$(/opt/sam3env/bin/python --version 2>&1 | awk '{print $2}')
    SAM3_MINOR=$(echo "$SAM3_PY" | cut -d. -f2)
    if [ "$SAM3_MINOR" -ge 12 ]; then
        pass "SAM3 venv Python $SAM3_PY (>= 3.12 required)"
    else
        fail "SAM3 venv Python $SAM3_PY (>= 3.12 required)"
    fi
else
    fail "SAM3 venv not found at /opt/sam3env/bin/python"
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

# --- 3. PyTorch (both environments) ---
echo ""
echo "--- PyTorch ---"
PT_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ -n "$PT_VER" ]; then
    pass "PyTorch $PT_VER (system Python 3.10)"
else
    fail "PyTorch not importable (system Python 3.10)"
fi

SAM3_PT=$(/opt/sam3env/bin/python -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ -n "$SAM3_PT" ]; then
    pass "PyTorch $SAM3_PT (SAM3 venv Python 3.12)"
else
    fail "PyTorch not importable (SAM3 venv)"
fi

TV_VER=$(python3 -c "import torchvision; print(torchvision.__version__)" 2>/dev/null)
if [ -n "$TV_VER" ]; then
    pass "TorchVision $TV_VER"
else
    fail "TorchVision not importable"
fi

# --- 4. SAM3 (tested in Python 3.12 venv) ---
echo ""
echo "--- SAM3 (Python 3.12 venv) ---"
if /opt/sam3env/bin/python -c "from sam3 import build_sam3_image_model; print('OK')" 2>/dev/null; then
    pass "sam3 package importable (native API)"
else
    fail "sam3 package NOT importable in venv"
fi

if /opt/sam3env/bin/python -c "from transformers import Sam3Processor, Sam3Model; print('OK')" 2>/dev/null; then
    pass "Sam3Model importable (Transformers API)"
else
    warn "Sam3Model not in transformers (may need newer version)"
fi

# --- 5. GraspGen (system Python 3.10) ---
echo ""
echo "--- GraspGen (System Python 3.10) ---"
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
    fail "pointnet2_ops NOT available"
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
    pass "ROS2 CLI available"
else
    fail "ros2 command not found"
fi

if [ -f "/opt/ros/humble/setup.bash" ]; then
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

# --- 9. SAM3 Server Bridge ---
echo ""
echo "--- SAM3 Server Bridge ---"
if [ -f "/ros2_ws/scripts/sam3_server.py" ]; then
    pass "sam3_server.py accessible at /ros2_ws/scripts/sam3_server.py"
else
    warn "sam3_server.py not found (should be volume-mounted)"
fi

# --- 10. Key Python packages (system) ---
echo ""
echo "--- Additional Python Packages (System) ---"
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
echo ""
echo "  Architecture:"
echo "    System (Python 3.10): ROS2 + GraspGen + common packages"
echo "    SAM3 venv (Python 3.12): /opt/sam3env/bin/python"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "  Some tests FAILED. Fix the issues above before running the pipeline."
    exit 1
else
    echo "  Environment is ready!"
    exit 0
fi
