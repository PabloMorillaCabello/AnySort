#!/bin/bash
# =============================================================================
# Test: Orbbec Gemini 2 camera via ROS2.
# Run inside the Docker container with the camera plugged in.
#
# Usage: ./scripts/test_camera.sh
# =============================================================================
set -e

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash 2>/dev/null || true

echo "============================================"
echo "  Orbbec Gemini 2 Camera Test"
echo "============================================"

# --- 1. Check USB device ---
echo ""
echo "[TEST] Checking for Orbbec USB device..."
if lsusb | grep -qi "orbbec\|2bc5"; then
    echo "  [PASS] Orbbec device detected on USB"
    lsusb | grep -i "orbbec\|2bc5" | sed 's/^/  [INFO] /'
else
    echo "  [FAIL] No Orbbec device found on USB"
    echo "  [INFO] Make sure the camera is plugged in and the container has --privileged"
    exit 1
fi

# --- 2. Check udev rules ---
echo ""
echo "[TEST] Checking udev rules..."
if [ -f /etc/udev/rules.d/99-orbbec-camera.rules ] || [ -f /etc/udev/rules.d/99-orbbec.rules ]; then
    echo "  [PASS] Orbbec udev rules installed"
else
    echo "  [WARN] Orbbec udev rules not found (may cause permission issues)"
    echo "  [INFO] Run: cd /ros2_ws/src/OrbbecSDK_ROS2/orbbec_camera/scripts && bash install_udev_rules.sh"
fi

# --- 3. Check ROS2 package ---
echo ""
echo "[TEST] Checking orbbec_camera ROS2 package..."
if ros2 pkg list 2>/dev/null | grep -q "orbbec_camera"; then
    echo "  [PASS] orbbec_camera package found"
else
    echo "  [FAIL] orbbec_camera package not built"
    echo "  [INFO] Run: cd /ros2_ws && colcon build --packages-select orbbec_camera"
    exit 1
fi

# --- 4. Launch camera briefly and check topics ---
echo ""
echo "[TEST] Launching Gemini 2 camera (10 second test)..."
ros2 launch orbbec_camera gemini2.launch.py &
LAUNCH_PID=$!

# Wait for topics to appear
sleep 8

echo ""
echo "[TEST] Checking published topics..."
TOPICS=$(ros2 topic list 2>/dev/null)

check_topic() {
    if echo "$TOPICS" | grep -q "$1"; then
        echo "  [PASS] Topic: $1"
        # Check if data is flowing
        HZ=$(timeout 3 ros2 topic hz "$1" 2>/dev/null | head -1 || echo "")
        if [ -n "$HZ" ]; then
            echo "  [INFO]   $HZ"
        fi
    else
        echo "  [FAIL] Topic: $1 NOT found"
    fi
}

check_topic "/camera/color/image_raw"
check_topic "/camera/depth/image_raw"
check_topic "/camera/color/camera_info"
check_topic "/camera/depth/camera_info"

# Clean up
echo ""
echo "[INFO] Shutting down camera..."
kill $LAUNCH_PID 2>/dev/null
wait $LAUNCH_PID 2>/dev/null || true

echo ""
echo "============================================"
echo "  Camera test complete"
echo "============================================"
