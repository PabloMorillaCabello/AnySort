# Docker Setup

> Multi-stage Dockerfile + docker-compose orchestration.
> **Files:** `docker/Dockerfile`, `docker/docker-compose.yml`

---

## Build

```bash
# Build context is REPO ROOT (not docker/)
docker compose -f docker/docker-compose.yml build --no-cache

# Start
docker compose -f docker/docker-compose.yml up -d

# Enter container
docker compose -f docker/docker-compose.yml exec graspgen /bin/bash
# or use the Windows helper:
docker/Bash.cmd
```

---

## 10 Build Stages

| Stage | Content |
|---|---|
| 1. Base | CUDA 12.6 + ROS2 Humble (Python 3.10) + system packages |
| 2. System deps | Python 3.12, build tools, OpenGL, Tkinter |
| 3. GraspGen | `uv` venv at `/opt/GraspGen/.venv/` (Python 3.10), PointNet++ CUDA extensions |
| 4. Python packages | `requirements.txt` + pymodbus/pyserial + **numpy<2 pinned LAST** |
| 5. Dobot API | Cloned from GitHub, patched with `fix_dobot_feedback.py` |
| 6. SAM3 | Venv at `/opt/sam3env/` (Python 3.12), PyTorch 2.7 + HuggingFace |
| 7. Orbbec SDK | v2.7.6 from `.deb` + `pyorbbecsdk` built from source |
| 8. ROS2 workspace | OrbbecSDK_ROS2 + pipeline packages via `colcon build` |
| 9. Environment | `.bashrc` aliases, PYTHONPATH |
| 10. Entrypoint | `docker/entrypoint.sh` |

---

## Python Environments

See [[Python Environments]] for full detail.

| Env | Path | Python |
|---|---|---|
| GraspGen (main) | `/opt/GraspGen/.venv/` | 3.10 |
| SAM3 | `/opt/sam3env/` | 3.12 |
| System | `/usr/bin/python3` | 3.10 |

---

## Container Runtime

### Security: No `privileged: true`
Uses `device_cgroup_rules` instead:
```yaml
device_cgroup_rules:
  - "c 189:* rwm"   # USB devices
  - "c 180:* rwm"   # USB serial
  - "c 81:* rwm"    # Video devices
```

### Volumes
```yaml
volumes:
  - /dev:/dev
  - /dev/bus/usb:/dev/bus/usb
  - ./app:/ros2_ws/app        # live-linked
  - ./scripts:/ros2_ws/scripts
```

### Ports
| Port | Service |
|---|---|
| `29999` | Dobot dashboard / UR dashboard |
| `30004` | Dobot feedback / UR RTDE |
| `7860`, `8080` | Viser visualization |
| `7000`, `6000` | Meshcat visualization |

### Environment
```yaml
DISPLAY: host.docker.internal:0.0
PYTHONPATH: /opt/GraspGen:/opt/Dobot_hv
ROS_DOMAIN_ID: 42
```

---

## Dobot Patch

`docker/patches/fix_dobot_feedback.py` applied at build stage 5:
- Fixes TCP partial read bug
- Fixes Tkinter thread-safety bug

See [[../Known Issues & Fixes|Known Issues]] for details.

---

## Entrypoint

`docker/entrypoint.sh`:
- Activates GraspGen venv
- Sources ROS2 workspace (`source /ros2_ws/install/setup.bash`)
- Shows startup banner

---

## Build Context Note

Build context is **repo root**, not `docker/`. All Dockerfile `COPY` paths use `docker/` or `data/` prefixes:
```dockerfile
COPY docker/requirements.txt /tmp/
COPY data/OrbbecSDK_v2.7.6_amd64.deb /tmp/
```

---

## Links
- [[Python Environments]] — venv details
- [[USB Passthrough]] — USB device forwarding
- [[../Known Issues & Fixes|Known Issues]] — Dobot patch, numpy constraint
