# Orbbec Gemini 2 Camera

> RGB-D depth camera — primary perception sensor.

---

## Hardware

| Property | Value |
|---|---|
| Model | Orbbec Gemini 2 |
| USB VID/PID | `2bc5:0670` |
| Interface | USB 3.0 |
| Color resolution | 1280×720 |
| Depth resolution | aligned to color |

---

## Software Stack

```
OrbbecSDK v2.7.6 (.deb)          ← native C SDK
       ↓
pyorbbecsdk                       ← Python bindings (built from source)
       ↓
app/orbbec_quiet.py               ← suppress C-level stderr spam
       ↓
anysort.py                        ← RGB + depth frames
       ↓
OrbbecSDK_ROS2 driver             ← optional ROS2 interface
```

---

## Usage in Pipeline

```python
# Always import orbbec_quiet BEFORE pyorbbecsdk
import sys
sys.path.insert(0, "/ros2_ws/app")
import orbbec_quiet   # redirects C stderr → /tmp/orbbec_sdk.log

import pyorbbecsdk
```

### Frame acquisition:
- **Color** → RGB array → displayed in Tkinter preview + sent to SAM3
- **Depth** → aligned depth array → combined with SAM3 mask → point cloud

---

## Pure Python Viewer

No ROS2 required:
```bash
python3 /ros2_ws/scripts/view_camera.py
python3 /ros2_ws/scripts/view_camera.py --ir --align --pointcloud
```

Flags:
- `--align` — align depth to color frame
- `--ir` — show infrared channel
- `--pointcloud` — live 3D point cloud view

---

## ROS2 Topics

| Topic | Type | Note |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | Use `color_format:=RGB` |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Raw depth |
| `/camera/ir/image_raw` | `sensor_msgs/Image` | Infrared |

---

## Known Issues

### MJPG Format Unsupported in ROS2
Default color format `MJPG` fails in ROS2 callback. Fix:
```bash
ros2 launch orbbec_camera gemini2.launch.py color_format:=RGB
```

### pyorbbecsdk PyPI Wheel Bug
v1.3.2 on PyPI installs a macOS `.so` on Linux. Fix: **built from source** in Dockerfile stage 7.

### Stderr Spam
OrbbecSDK floods stderr with C-level debug output. Fix: `app/orbbec_quiet.py` redirects to `/tmp/orbbec_sdk.log`.

---

## USB Passthrough (WSL2)

Camera attached to Windows host must be forwarded to WSL2:
```powershell
# Run as admin — scripts/reattach.ps1
usbipd bind --busid <ID>
usbipd attach --wsl --busid <ID>
```
See [[../Infrastructure/USB Passthrough|USB Passthrough]].

---

## Camera Intrinsics

Calibrated values (see [[../Calibration/Camera Intrinsics|Camera Intrinsics]]):
```
fx = 684.7,  fy = 685.9
cx = 655.3,  cy = 357.0
Resolution: 1280×720
RMS: 0.20 px
```

---

## Links
- [[../Pipeline/Pipeline Flow|Pipeline Flow]] — how frames feed into pipeline
- [[../Calibration/Camera Intrinsics|Camera Intrinsics]] — focal length / principal point
- [[../Calibration/Hand-Eye Calibration|Hand-Eye Calibration]] — camera-to-robot transform
- [[../Infrastructure/USB Passthrough|USB Passthrough]] — WSL2 USB forwarding
- [[../Known Issues & Fixes|Known Issues]] — Orbbec quirks
