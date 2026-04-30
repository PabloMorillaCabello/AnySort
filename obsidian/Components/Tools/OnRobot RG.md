# OnRobot RG Gripper

> OnRobot RG-series gripper for UR10 — controlled via UR Dashboard program execution.
> **File:** `app/tools/onrobot_urscript.py`

---

## How It Works

The UR Dashboard Server (port 29999) loads and plays `.urp` programs saved on the teach pendant. Two programs are required:

| Program name | Action |
|---|---|
| `gripper_open` | RG6 node — width=max (open) |
| `gripper_close` | RG6 node — width=0, desired force (close) |

This avoids the unreliable `rg_grip()` URScript path (which fails when called outside the URCap program context).

---

## Why Not URScript `rg_grip()`?

ur_rtde keeps its own daemon running. The URCap's `rg_grip()` communicates via `127.0.0.1:54321` to a Java process managed by the URCap. When called from a detached secondary script, that socket call can fail silently because the URCap server only services the **active program context**. Running a saved program avoids this entirely.

---

## Setup (one-time on teach pendant)

1. New program → add "RG6 Grip" node → set width/force → save as `gripper_close`
2. New program → add "RG6 Grip" node → set width=max → save as `gripper_open`
3. Programs must be in `/programs/` on the controller

---

## Usage

```python
from tools.onrobot_urscript import OnRobotRGURScript

tool = OnRobotRGURScript(robot=robot)
robot.attach_tool(tool)

robot.vacuum_on()   # → gripper_close.urp runs
robot.vacuum_off()  # → gripper_open.urp runs
```

Custom program names:
```python
tool = OnRobotRGURScript(robot=robot,
                          open_program="my_open",
                          close_program="my_close")
```

---

## Timing Parameters

| Constant | Value | Purpose |
|---|---|---|
| `POLL_INTERVAL` | 0.15 s | Between `programState` polls |
| `PLAY_TIMEOUT` | 15.0 s | Max wait for program to finish |
| `ACTION_WAIT_S` | 5.0 s | Wait after each gripper action |

---

## Connection

Reuses the robot's dashboard socket (port 29999). No additional connection needed.

---

## Links
- [[Tool Architecture]] — base class + registry
- [[../Robots/UR10|UR10]] — the robot this tool attaches to
- [[../../Known Issues & Fixes|Known Issues]] — OnRobot timing notes
