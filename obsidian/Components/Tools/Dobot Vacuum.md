# Dobot Vacuum Gripper

> Vacuum end-effector for Dobot CR — controlled via digital output.
> **File:** `app/tools/dobot_vacuum.py`

---

## How It Works

Uses Dobot dashboard `ToolDO` command to toggle a digital output port:
- `ToolDO(port, 1)` → vacuum ON (picks object)
- `ToolDO(port, 0)` → vacuum OFF (releases object)

Default DO port: **1**

---

## Usage

```python
from tools import create_tool

tool = create_tool("Dobot Vacuum", robot=robot)
robot.attach_tool(tool)

robot.vacuum_on()   # → DobotVacuum.grasp() → ToolDO(1, 1)
robot.vacuum_off()  # → DobotVacuum.release() → ToolDO(1, 0)
```

Or directly:
```python
from tools.dobot_vacuum import DobotVacuum
tool = DobotVacuum(dashboard=robot._dashboard, do_port=1)
```

---

## Constructor

```python
DobotVacuum(dashboard, do_port=1)
```
- `dashboard` — accepts either `DobotApiDashboard` **or** a `DobotCR` robot instance (auto-extracts `._dashboard`)
- `do_port` — 1-based digital output port number

---

## Links
- [[Tool Architecture]] — base class + registry
- [[../Robots/Dobot CR|Dobot CR]] — the robot this tool attaches to
