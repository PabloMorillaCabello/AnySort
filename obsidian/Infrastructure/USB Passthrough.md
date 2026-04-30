# USB Passthrough (WSL2 → Docker)

> Forwarding the Orbbec camera from Windows host → WSL2 → Docker container.
> **Guide:** `USB_WSL_Docker_Guide.md`

---

## Stack

```
Orbbec Gemini 2
      ↓ USB 3.0
Windows Host
      ↓ usbipd-win (bind + attach)
WSL2 Linux kernel (usbip client)
      ↓ /dev/bus/usb
Docker container (device mounts + cgroup rules)
      ↓
pyorbbecsdk / OrbbecSDK_ROS2
```

---

## Quick Attach

Run as **Administrator** in PowerShell:
```powershell
# scripts/reattach.ps1
usbipd list                      # find Orbbec device (VID 2bc5, PID 0670)
usbipd bind --busid <ID>         # first time only
usbipd attach --wsl --busid <ID>
```

Or manually:
```powershell
usbipd wsl attach --busid <BUSID>
```

---

## Verify Inside WSL2

```bash
lsusb | grep 2bc5   # should show Orbbec device
ls /dev/bus/usb/    # check USB bus
```

---

## Verify Inside Container

```bash
lsusb
python3 /ros2_ws/scripts/view_camera.py   # live preview
```

---

## Docker Device Access

No `privileged: true`. Uses explicit rules in `docker-compose.yml`:
```yaml
device_cgroup_rules:
  - "c 189:* rwm"   # USB (major 189)
  - "c 180:* rwm"   # USB serial (major 180)
  - "c 81:* rwm"    # Video (major 81)
volumes:
  - /dev:/dev
  - /dev/bus/usb:/dev/bus/usb
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Device not found in WSL2 | usbipd not bound/attached | Run `reattach.ps1` as admin |
| Camera found in WSL2 but not container | Cgroup rules missing | Check `docker-compose.yml` |
| `pyorbbecsdk` can't open device | Wrong USB permissions | Add user to `plugdev` group in WSL2 |
| Camera disconnects randomly | USB suspend | Disable USB selective suspend in Windows power settings |

---

## Links
- [[../Components/Camera - Orbbec Gemini 2|Orbbec Camera]] — the device being forwarded
- [[Docker Setup]] — container device configuration
