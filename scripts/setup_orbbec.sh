#!/bin/bash
# =============================================================================
# Setup Orbbec Gemini 2 camera on the HOST machine.
# Run ONCE on your host (not inside Docker).
# The Docker container accesses the camera via --privileged + /dev mount.
#
# Usage: ./scripts/setup_orbbec.sh
# =============================================================================
set -e

echo "============================================"
echo "  Orbbec Gemini 2 - Host Setup"
echo "============================================"

# --- 1. Udev rules for Orbbec cameras ---
echo ""
echo "[1/3] Installing udev rules..."
UDEV_RULE='SUBSYSTEM=="usb", ATTR{idVendor}=="2bc5", MODE="0666", GROUP="plugdev"'
echo "$UDEV_RULE" | sudo tee /etc/udev/rules.d/99-orbbec.rules > /dev/null

# If OrbbecSDK_ROS2 is cloned, use their more complete rules
if [ -f "ros2_ws/src/OrbbecSDK_ROS2/orbbec_camera/scripts/install_udev_rules.sh" ]; then
    echo "  -> Also installing OrbbecSDK_ROS2 udev rules..."
    cd ros2_ws/src/OrbbecSDK_ROS2/orbbec_camera/scripts
    sudo bash install_udev_rules.sh
    cd - > /dev/null
fi

sudo udevadm control --reload-rules
sudo udevadm trigger
echo "  [DONE] Udev rules installed"

# --- 2. Allow X11 forwarding for Docker ---
echo ""
echo "[2/3] Allowing X11 forwarding for Docker (for RViz, Meshcat)..."
xhost +local:docker 2>/dev/null || echo "  [SKIP] xhost not available (headless system?)"
echo "  [DONE] X11 forwarding configured"

# --- 3. Check camera ---
echo ""
echo "[3/3] Checking for Orbbec camera on USB..."
if lsusb | grep -qi "orbbec\|2bc5"; then
    echo "  [PASS] Orbbec camera detected:"
    lsusb | grep -i "orbbec\|2bc5" | sed 's/^/    /'
else
    echo "  [WARN] No Orbbec camera detected on USB."
    echo "    Make sure the Gemini 2 is plugged in before starting Docker."
fi

echo ""
echo "============================================"
echo "  Host setup complete!"
echo "  Next: cd docker && docker compose up -d"
echo "============================================"
