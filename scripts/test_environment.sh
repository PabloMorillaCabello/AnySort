#!/bin/bash
# =============================================================================
# Test: Verify the full development environment is set up correctly.
# Run inside the Docker container.
#
# Architecture:
#   - GraspGen venv Python 3.10: /opt/GraspGen/.venv/bin/python (uv-managed)
#   - SAM3 venv Python 3.12: /opt/sam3env/bin/python (SAM3 only)
#   - Dobot API: /opt/Dobot_hv (on PYTHONPATH)
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

# --- 1. Python versions ---
echo ""
echo "--- Python ---"
PY_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MINOR" -eq 10 ]; then
    pass "GraspGen venv Python $PY_VERSION (3.10 required)"
else
    fail "GraspGen venv Python $PY_VERSION (expected 3.10)"
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
    pass "PyTorch $PT_VER (GraspGen venv Python 3.10)"
else
    fail "PyTorch not importable (GraspGen venv Python 3.10)"
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
if /opt/sam3env/bin/python -c "from transformers import Sam3Processor, Sam3Model; print('OK')" 2>/dev/null; then
    pass "SAM3 via Transformers API (Sam3Processor, Sam3Model)"
else
    fail "SAM3 Transformers API not importable (check transformers version)"
fi

# --- 5. GraspGen (uv-managed venv Python 3.10) ---
echo ""
echo "--- GraspGen (uv-managed venv Python 3.10) ---"
if python3 -c "import grasp_gen; print('OK')" 2>/dev/null; then
    pass "grasp_gen package importable"
else
    fail "grasp_gen package NOT importable"
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

# --- 6. Dobot TCP/IP API ---
echo ""
echo "--- Dobot Robot API ---"
if [ -d "/opt/Dobot_hv" ]; then
    pass "Dobot_hv repository present at /opt/Dobot_hv"
else
    fail "Dobot_hv NOT found at /opt/Dobot_hv"
fi

if python3 -c "import dobot_api; print('OK')" 2>/dev/null; then
    pass "dobot_api importable"
else
    warn "dobot_api NOT importable (check PYTHONPATH includes /opt/Dobot_hv)"
fi

# --- 7. ROS2 ---
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

# --- 8. ROS2 Packages ---
echo ""
echo "--- ROS2 Packages ---"
for pkg in orbbec_camera ur_robot_driver moveit; do
    if ros2 pkg list 2>/dev/null | grep -q "$pkg"; then
        pass "ros2 pkg: $pkg"
    else
        warn "ros2 pkg: $pkg not found (may need workspace build)"
    fi
done

# --- 9. Model Weights ---
echo ""
echo "--- Model Weights ---"
if [ -d "/opt/models/sam3" ] && [ "$(ls -A /opt/models/sam3 2>/dev/null)" ]; then
    pass "SAM3 weights present in /opt/models/sam3"
else
    warn "SAM3 weights NOT found (run: ./scripts/download_models.sh)"
fi

if [ -d "/opt/GraspGen/GraspGenModels" ] && [ "$(ls -A /opt/GraspGen/GraspGenModels 2>/dev/null)" ]; then
    pass "GraspGen weights present in /opt/GraspGen/GraspGenModels"
else
    warn "GraspGen weights NOT found (run: cd /opt/GraspGen && git clone https://huggingface.co/adithyamurali/GraspGenModels)"
fi

# --- 10. SAM3 Server Bridge ---
echo ""
echo "--- SAM3 Server Bridge ---"
if [ -f "/ros2_ws/scripts/sam3_server.py" ]; then
    pass "sam3_server.py accessible at /ros2_ws/scripts/sam3_server.py"
else
    warn "sam3_server.py not found (should be volume-mounted)"
fi

# --- 11. Key Python packages (GraspGen venv) ---
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
echo ""
echo "  Architecture:"
echo "    GraspGen venv (Python 3.10): /opt/GraspGen/.venv/bin/python (uv-managed)"
echo "    SAM3 venv (Python 3.12): /opt/sam3env/bin/python"
echo "    Dobot API: /opt/Dobot_hv (on PYTHONPATH)"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "  Some tests FAILED. Fix the issues above before running the pipeline."
    exit 1
else
    echo "  Environment is ready!"
    exit 0
fi
