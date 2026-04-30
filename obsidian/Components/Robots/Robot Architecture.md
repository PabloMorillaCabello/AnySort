# Robot Architecture

> Modular robot driver system — abstract base class + registry pattern.
> **Files:** `app/robots/base.py`, `app/robots/__init__.py`

---

## Design

The pipeline calls **only** `RobotBase` methods. No robot-specific code leaks into `anysort.py`. Robot is selected at runtime from UI dropdown.

```
RobotBase (ABC)
├── DobotCR          ← robots/dobot_cr.py
├── UR10             ← robots/ur10.py
└── [YourRobot]      ← robots/my_robot.py (add new robot here)
```

---

## Registry (`robots/__init__.py`)

```python
ROBOT_DRIVERS = {
    "Dobot CR": ("robots.dobot_cr", "DobotCR"),
    "UR10":     ("robots.ur10",     "UR10"),
}

# Create robot by name:
robot = create_robot("Dobot CR", ip="192.168.5.1")
```

`get_available_drivers()` — returns only drivers whose deps are importable (graceful degradation if e.g. ur-rtde not installed).

---

## RobotBase Interface

### Mode Constants
```python
MODE_RUNNING = -1   # executing motion (subclass overrides with real codes)
MODE_ERROR   = -2   # error / protective stop
MODE_ENABLED = -3   # idle, ready
```
Each subclass maps its real controller state codes to these constants.

### Lifecycle
| Method | Description |
|---|---|
| `__init__(ip, **kwargs)` | Connect to robot. Raise on failure |
| `enable()` | Servo on / brake release |
| `power_on()` | Power controller |
| `clear_error()` | Clear alarms / protective stops |
| `stop()` | Emergency decelerate |
| `close()` | Disconnect, free resources |

### Status
| Method | Returns |
|---|---|
| `get_mode()` | Current mode (MODE_* constant) |
| `get_pose()` | `(x, y, z, rx, ry, rz)` mm / deg |
| `get_angle()` | `(j1..j6)` degrees |

### Motion
| Method | Returns |
|---|---|
| `set_speed(percent)` | — |
| `move_linear(x,y,z,rx,ry,rz)` | `cmd_id` |
| `move_joint_angles(j1..j6)` | `cmd_id` |
| `move_joint(x,y,z,rx,ry,rz)` | `cmd_id` (default = move_linear) |
| `wait_motion(cmd_id, timeout)` | `True` on success |
| `wait_idle(timeout)` | `True` when not RUNNING |

### End-Effector (Tool Delegation)
```python
robot.attach_tool(tool)    # ToolBase instance
robot.vacuum_on()          # → tool.grasp()
robot.vacuum_off()         # → tool.release()
```

### Reachability Check
```python
reachable, joints, msg = robot.check_reachability(x, y, z, rx, ry, rz)
```
Default: always `True`. Override with IK solver.

---

## Adding a New Robot

1. Create `app/robots/my_robot.py`
2. `class MyRobot(RobotBase)` — implement all `@abstractmethod` methods
3. Set `MODE_RUNNING`, `MODE_ERROR`, `MODE_ENABLED` to real controller codes
4. Register in `ROBOT_DRIVERS` dict in `robots/__init__.py`
5. Robot appears in AnySort UI dropdown automatically

---

## Links
- [[Dobot CR]] — TCP/IP CR-series implementation
- [[UR10]] — RTDE register-based implementation
- [[../../Pipeline/AnySort Pipeline|AnySort Pipeline]] — how robot is used
- [[../../Pipeline/Pipeline Flow|Pipeline Flow]] — execution sequence
- [[../Tools/Tool Architecture|Tool Architecture]] — end-effector layer
