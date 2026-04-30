# Python Environments

> Three isolated Python environments inside the container.

---

## Overview

| Env | Path | Python | Manager | Alias |
|---|---|---|---|---|
| **GraspGen** (main) | `/opt/GraspGen/.venv/` | 3.10 | uv | `graspgen_activate` |
| **SAM3** | `/opt/sam3env/` | 3.12 | pip | `sam3_activate` |
| **System** | `/usr/bin/python3` | 3.10 | apt | — |

---

## GraspGen Env (Main Pipeline)

**Active for:** `anysort.py` and all pipeline code.

Key packages:
- PyTorch (with CUDA 12.6)
- `pyorbbecsdk` (built from source)
- `graspgen` (from NVlabs/GraspGen)
- `pymodbus`, `pyserial`
- `opencv-python`, `open3d`, `trimesh`, `scipy`
- `numpy<2` (**pinned last** — critical constraint)
- `ur-rtde` (for UR10 support)

```bash
graspgen_activate   # activates /opt/GraspGen/.venv/
```

---

## SAM3 Env

**Active for:** `sam3_server.py` (auto-started by AnySort as subprocess).

Key packages:
- PyTorch 2.7
- HuggingFace `transformers`
- `facebook/sam3` weights (downloaded on first run to HuggingFace cache)

```bash
sam3_activate   # activates /opt/sam3env/
```

**Why isolated?** SAM3 requires PyTorch 2.7 + Python 3.12, incompatible with GraspGen's PyTorch version.

---

## System Python

**Active for:** ROS2 packages, `colcon build`, build tools.

Not used for pipeline execution.

---

## Critical Constraint: numpy < 2

```
numpy must stay <2 in the GraspGen env.
PyTorch is compiled against numpy 1.x. numpy 2.x breaks it.
numpy<2 is installed LAST in the Dockerfile (stage 4) to prevent
any subsequent package from upgrading it.
```

If `numpy 2.x` appears after container rebuild → GraspGen will break.

---

## PYTHONPATH

Set in container environment:
```
PYTHONPATH=/opt/GraspGen:/opt/Dobot_hv
```

Allows importing `grasp_gen.*` and `dobot_api.*` without explicit path manipulation.

Pipeline also does:
```python
sys.path.insert(0, "/ros2_ws/app")  # for orbbec_quiet, robots, tools
```

---

## Links
- [[Docker Setup]] — build stages that create these envs
- [[../Components/GraspGen|GraspGen]] — uses GraspGen env
- [[../Components/SAM3 Segmentation|SAM3]] — uses SAM3 env
- [[../Known Issues & Fixes|Known Issues]] — numpy<2 constraint
