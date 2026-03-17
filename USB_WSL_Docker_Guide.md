# USB Devices on Windows + WSL + Docker: Complete Setup Guide

A step-by-step guide to connect any USB device (cameras, serial adapters, robots) from Windows through WSL to Docker containers.

---

## Overview

The flow is always the same:
1. **Windows** → Find and share the USB device with `usbipd`
2. **WSL** → Attach the device so Linux can see it
3. **Docker** → Map the Linux device into your container
4. **Restart** → Reattach after reboot (binding usually persists, but attachment doesn't)

---

## Prerequisites

- Windows 11 with WSL 2 installed
- `usbipd-win` installed on Windows ([install guide](https://github.com/dorssel/usbipd-win/releases))
- Docker Desktop running on WSL 2
- USB device connected to Windows

---

## Step 1: List Connected USB Devices

### On Windows (PowerShell)

Open **PowerShell** (not necessarily as Admin yet):

```powershell
usbipd list
```

**Example output:**
```
Connected:
BUSID  VID:PID    DEVICE                                STATE
2-10   5986:211b  HD Webcam                             Not shared
2-14   8087:0026  Intel(R) Wireless Bluetooth(R)        Not shared
3-1    0bda:8153  Realtek USB GbE Family Controller     Not shared
```

**What it means:**
- `BUSID` = identifier you use in `bind` and `attach` commands (e.g., `2-10`)
- `VID:PID` = vendor and product IDs
- `STATE` = `Not shared` means not yet available to WSL
- `STATE` = `Shared` means already bound but may not be attached yet

**Note your device's BUSID** — you will need it in the next steps.

---

## Step 2: Share the Device from Windows

### On Windows (PowerShell as Administrator)

Open **PowerShell as Administrator** and run:

```powershell
usbipd bind --busid <BUSID>
```

**Example:**
```powershell
usbipd bind --busid 2-10
```

**What it does:**
- Shares the USB device so WSL can access it
- Changes the Windows driver mode to allow USB/IP passthrough
- Only needs to be done once per device (persists across reboots usually)

**If it fails with "Device busy":**
- Close any Windows app using the device (Camera app, Teams, browser tabs, OBS, vendor software, DISCORD !!)
- Retry the bind command

**If you need to force it:**
```powershell
usbipd bind --busid <BUSID> --force
```
Note: `--force` may require a reboot afterward.

---

## Step 3: Attach to WSL

### On Windows (PowerShell)

Run:
```powershell
usbipd attach --wsl --busid <BUSID>
```

**Example:**
```powershell
usbipd attach --wsl --busid 2-10
```

**What it does:**
- Makes the device available inside your WSL 2 distribution
- Creates a virtual USB connection through USB/IP protocol
- **Not persistent** — you usually need to reattach after reboot or device reconnection

**Success output:**
```
usbipd: info: Using WSL distribution 'Ubuntu' to attach; the device will be available in all WSL 2 distributions.
usbipd: info: Loading vhci_hcd module.
usbipd: info: Detected networking mode 'nat'.
usbipd: info: Using IP address 172.17.96.1 to reach the host.
```

**If it fails with "Device busy":**
- Close Windows apps using the device
- Try again

**If it still fails after closing apps:**
- The device may still be claimed by Windows drivers
- Try `usbipd bind --busid <BUSID> --force`, then reboot, then attach again

---

## Step 4: Verify Device in WSL

### In WSL/Ubuntu

Open WSL terminal and check what Linux sees:

```bash
lsusb
```

**Example output for a camera:**
```
Bus 001 Device 002: ID 5986:211b Acer, Inc HD Webcam
```

**For cameras specifically:**
```bash
ls /dev/video*
```

**For serial/USB adapters:**
```bash
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

**For generic device info:**
```bash
dmesg | tail -n 50
```

**If device doesn't appear:**
- Check that `usbipd attach` completed successfully in PowerShell
- Verify with `usbipd list` on Windows that state shows attachment
- In WSL, try: `sudo modprobe usbip_host`
- Rerun attach in PowerShell

---

## Step 5: Query Device Capabilities (Optional)

### For Cameras

```bash
v4l2-ctl --device=/dev/video0 --all
v4l2-ctl --device=/dev/video0 --list-formats-ext
v4l2-ctl --device=/dev/video0 --list-ctrls
```

### For USB Serial Adapters

```bash
udevadm info -a -n /dev/ttyUSB0
```

### For Generic USB Info

```bash
lsusb -v -d <VID>:<PID>
```

Example for the Acer camera (VID=5986, PID=211b):
```bash
lsusb -v -d 5986:211b
```

---

## Step 6: Add Device to Docker Container

### Modify docker-compose.yml

Add the `devices:` section to your service. The device path is the Linux path from WSL (e.g., `/dev/video0`).

**Example for a camera:**

```yaml
version: '3.9'

services:
  graspgen:
    image: your-image:latest
    devices:
      - "/dev/video0:/dev/video0"
    # ... rest of your service config
```

**Example for a serial device:**

```yaml
services:
  robot-controller:
    image: your-image:latest
    devices:
      - "/dev/ttyUSB0:/dev/ttyUSB0"
    # ... rest of your service config
```

**Example for multiple devices:**

```yaml
services:
  my-app:
    image: your-image:latest
    devices:
      - "/dev/video0:/dev/video0"
      - "/dev/video1:/dev/video1"
      - "/dev/ttyUSB0:/dev/ttyUSB0"
    # ... rest of your service config
```

### Apply the Changes

```powershell
docker compose down
docker compose up -d
```

### Verify Inside Container

```powershell
docker compose exec <service_name> bash
```

Inside the container:
```bash
ls /dev/video0
lsusb
```

---

## Step 7: Test Device Access

### For Cameras

```bash
ffmpeg -f v4l2 -input_format mjpeg -video_size 640x480 -framerate 30 -i /dev/video0 -frames:v 1 -y test.jpg
```

Or with OpenCV:
```python
import cv2
import time

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
time.sleep(1.0)

ret, frame = cap.read()
print("opened =", cap.isOpened(), "ret =", ret, "shape =", None if frame is None else frame.shape)
if ret:
    cv2.imwrite("test.jpg", frame)
cap.release()
```

### For Serial Devices

```bash
stty -F /dev/ttyUSB0 115200
cat /dev/ttyUSB0
```

---

## Reboot Recovery: Quick Checklist

After you restart your computer, USB devices detach automatically. Use this quick checklist to reconnect:

### 1. In Windows PowerShell

```powershell
usbipd list
```

Check if your device shows `Shared` or `Not shared`.

**If `Not shared`:**
```powershell
usbipd bind --busid <BUSID>
```

**Always run (regardless of state):**
```powershell
usbipd attach --wsl --busid <BUSID>
```

### 2. In WSL

```bash
lsusb
ls /dev/video* 2>/dev/null  # for cameras
ls /dev/ttyUSB* 2>/dev/null # for serial
```

### 3. In Docker

```powershell
docker compose down
docker compose up -d
```

---

## Common Issues & Solutions

### Issue: `usbipd: error: Device busy (exported)`

**Cause:** Windows app or service still using the device.

**Solution:**
1. Close Camera app, Teams, Zoom, browser tabs, OBS, vendor software
2. Check Task Manager for any process using the camera
3. Retry: `usbipd attach --wsl --busid <BUSID>`

### Issue: Device shows in `lsusb` but no `/dev/video0` or `/dev/ttyUSB0`

**Cause:** Device recognized at USB level but kernel driver not loaded or negotiated.

**Solution:**
```bash
sudo apt update
sudo apt install -y v4l-utils usbutils
sudo modprobe usbip_host
```

Then detach and reattach in PowerShell.

### Issue: `docker compose exec` says device not found

**Cause:** Device path in `docker-compose.yml` doesn't exist in WSL.

**Solution:**
1. Inside container: `ls /dev/video*` to confirm what device is actually there
2. Update `devices:` in `docker-compose.yml` to match
3. Run `docker compose up -d` again

### Issue: Camera works in `ffmpeg` but not in OpenCV

**Cause:** OpenCV V4L2 backend needs explicit format settings.

**Solution:** Use this pattern:
```python
import cv2
import time

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
time.sleep(1.0)  # warmup time

ret, frame = cap.read()
cap.release()
```

### Issue: High resolution (1280x720) doesn't work but 640x480 does

**Cause:** Passthrough path is unstable at higher bandwidth.

**Solution:**
- Use `640x480` as your stable mode
- Test if the camera works at 720p on native Windows or Linux (without WSL passthrough)
- If it works elsewhere, the limitation is the USB/IP passthrough layer, not the camera
- Consider FFmpeg or ROS camera driver as alternative capture methods

---

## Device Examples

### Example 1: Acer HD Webcam (Camera)

**Windows:**
```powershell
usbipd list          # Find: 2-10   5986:211b  HD Webcam
usbipd bind --busid 2-10
usbipd attach --wsl --busid 2-10
```

**WSL:**
```bash
lsusb                # Should show: ID 5986:211b Acer, Inc HD Webcam
ls /dev/video*       # Should show: /dev/video0 /dev/video1 /dev/media0
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

**docker-compose.yml:**
```yaml
services:
  graspgen:
    devices:
      - "/dev/video0:/dev/video0"
```

**Test in container:**
```bash
ffmpeg -f v4l2 -input_format mjpeg -video_size 640x480 -framerate 30 -i /dev/video0 -frames:v 1 -y test.jpg
```

### Example 2: USB Serial Adapter

**Windows:**
```powershell
usbipd list          # Find: 3-1   0403:6001  USB Serial Device
usbipd bind --busid 3-1
usbipd attach --wsl --busid 3-1
```

**WSL:**
```bash
lsusb                # Should show the serial device
ls /dev/ttyUSB*      # Should show: /dev/ttyUSB0 (or higher number)
```

**docker-compose.yml:**
```yaml
services:
  robot_controller:
    devices:
      - "/dev/ttyUSB0:/dev/ttyUSB0"
```

**Test in container:**
```bash
stty -F /dev/ttyUSB0 115200  # Set baud rate
cat /dev/ttyUSB0             # Read data
```

### Example 3: Orbbec RGB-D Camera

**Windows:**
```powershell
usbipd list          # Find Orbbec devices
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

**WSL:**
```bash
lsusb                # Should show Orbbec device
ls /dev/video*       # Multiple video nodes expected
```

**docker-compose.yml:**
```yaml
services:
  vision_app:
    devices:
      - "/dev/video0:/dev/video0"
      - "/dev/video1:/dev/video1"
      - "/dev/media0:/dev/media0"
```

---

## Reference: Complete Reusable Command Set

### Windows PowerShell (Administrator)

```powershell
# List all devices
usbipd list

# Share a device (one-time, persists usually)
usbipd bind --busid 2-10

# Attach to WSL (repeat after reboot)
usbipd attach --wsl --busid 2-10

# Force bind if stuck
usbipd bind --busid 2-10 --force

# After --force, reboot Windows, then:
# usbipd list
# usbipd attach --wsl --busid 2-10
```

### WSL/Ubuntu

```bash
# List USB devices
lsusb

# List video devices (cameras)
ls /dev/video*

# List serial devices
ls /dev/ttyUSB*

# Camera capabilities
v4l2-ctl --device=/dev/video0 --all
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Kernel messages
dmesg | tail -n 50

# Device attributes
udevadm info -a -n /dev/video0
```

### Docker

```bash
# Down and up with new device mappings
docker compose down
docker compose up -d

# Access container
docker compose exec <service> bash

# Inside container, verify device
ls /dev/video0
lsusb
```

---

## Tips for Your Thesis Setup

1. **Always use explicit device paths** in Docker, not indices (e.g., `/dev/video0` instead of `0`)
2. **Set camera formats explicitly** in code (FOURCC, width, height, fps) to avoid timeouts
3. **Use FFmpeg for initial testing** if OpenCV times out
4. **Document your working modes** (e.g., "640x480 MJPEG @ 30 fps is stable, 1280x720 times out")
5. **Create a shell script** to automate bind + attach after reboots
6. **Avoid unplugging devices** mid-session; detach in Windows first

---

## Creating an Automation Script

### For Windows (PowerShell)

Create a file `attach_usb.ps1`:

```powershell
# attach_usb.ps1
# Usage: .\attach_usb.ps1 2-10

param(
    [string]$BusId = "2-10"
)

Write-Host "Binding USB device $BusId..."
usbipd bind --busid $BusId

Write-Host "Attaching to WSL..."
usbipd attach --wsl --busid $BusId

Write-Host "Verifying in WSL..."
wsl lsusb

Write-Host "Done!"
```

Run after reboot:
```powershell
.\attach_usb.ps1 2-10
```

---

## Additional Resources

- [Microsoft WSL USB Docs](https://learn.microsoft.com/en-us/windows/wsl/connect-usb)
- [usbipd-win GitHub](https://github.com/dorssel/usbipd-win)
- [v4l2 Webcam Setup](https://www.marcusfolkesson.se/blog/capture-a-picture-with-v4l2/)
- [Docker Compose Device Mapping](https://docs.docker.com/compose/compose-file/compose-file-v3/#devices)

---

**Last updated:** March 17, 2026

**For your robotics thesis:** Use this guide as a template. Update BUSID and device paths for your specific hardware, and save this alongside your project documentation.
