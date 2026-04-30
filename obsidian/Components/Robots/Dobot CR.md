# Dobot CR Robot Driver

> CR-series robot arm driver (CR3 / CR5 / CR10 / CR16).
> **File:** `app/robots/dobot_cr.py`

---

## Connection

| Port | Use |
|---|---|
| `29999` | Dashboard — commands (MovL, MovJ, DO, etc.) |
| `30004` | Feedback — 1440-byte real-time status at 250 Hz |

```python
robot = create_robot("Dobot CR", ip="192.168.5.1")
# Internally:
self._dashboard = DobotApiDashboard(ip, 29999)
self._feed      = DobotApiFeedBack(ip, 30004)
```

---

## Mode Codes

| Constant | Value | Meaning |
|---|---|---|
| `MODE_RUNNING` | `7` | Executing motion |
| `MODE_ERROR` | `9` | Error / protective stop |
| `MODE_ENABLED` | `5` | Idle, ready |

---

## Vacuum Gripper

Controlled directly via dashboard `ToolDO` command (no tool attachment needed — but `DobotVacuum` tool is the clean way):

```python
from tools import create_tool
tool = create_tool("Dobot Vacuum", robot=robot)
robot.attach_tool(tool)
robot.vacuum_on()   # → DobotVacuum.grasp() → ToolDO(1, 1)
robot.vacuum_off()  # → DobotVacuum.release() → ToolDO(1, 0)
```

---

## Feedback Loop

Runs in a background daemon thread at 250 Hz:
- Reads 1440-byte packets from port 30004
- Parses `RobotMode`, `SpeedScaling`, joint angles, Cartesian pose
- Stores in instance variables under `self._lock`

**Critical bug fixed:** TCP `recv()` doesn't guarantee 1440 bytes. See [[../../Known Issues & Fixes|Known Issues]].

---

## Motion Commands

```python
cmd_id = robot.move_linear(x, y, z, rx, ry, rz)   # MovL (Cartesian linear)
cmd_id = robot.move_joint(x, y, z, rx, ry, rz)    # MovJ (joint-interpolated)
cmd_id = robot.move_joint_angles(j1,j2,j3,j4,j5,j6)  # JointMovJ
robot.wait_motion(cmd_id, timeout=90.0)
```

Dashboard response parsing: `_parse_result_id(resp)` extracts `[ErrorID, CommandID, ...]`.

---

## Error Recovery

```python
robot.clear_error()   # → dashboard.ClearError()
robot.enable()        # → dashboard.EnableRobot()
```

AnySort UI monitors `get_mode()` and shows recovery dialog on `MODE_ERROR`.

---

## Underlying API

From `/opt/Dobot_hv/` (Dobot_hv repo):
- `DobotApiDashboard` — command socket wrapper
- `DobotApiFeedBack` — feedback socket + numpy struct parser

Both patched at Docker build time via `docker/patches/fix_dobot_feedback.py`.

---

## Home Pose

```python
HOME_POSE = [300, 0, 450, 0, 0, 0]  # X, Y, Z mm + Rx, Ry, Rz deg
```

---

## Links
- [[Robot Architecture]] — base class + registry
- [[../../Tools/Dobot Vacuum|Dobot Vacuum]] — vacuum end-effector
- [[../../Known Issues & Fixes|Known Issues]] — TCP partial read + Tkinter fixes
- [[../../Infrastructure/Docker Setup|Docker Setup]] — Dobot API patch (stage 5)
