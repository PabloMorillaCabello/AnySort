# ABB IRB Robot Driver

> ABB IRB (IRC5 controller, RobotWare 6.xx) driver via Robot Web Services + a RAPID motion-server program on the controller.
> **File:** `app/robots/abb_irb.py`
> **Setup guide:** [[../../../docs/abb_setup|docs/abb_setup.md]]

---

## Connection

| Port | Use |
|---|---|
| `80` (HTTP) | RWS — state, IO, motion-program execution |

```python
robot = create_robot("ABB IRB", ip="192.168.125.1")
# Internally opens:
abb_motion_program_exec.MotionProgramExecClient(base_url=f"http://{ip}")
abb_robot_client.rws.RWS(base_url=f"http://{ip}")
```

Default credentials: `Default User` / `robotics`. Override via `username=`, `password=` kwargs.

---

## Protocol Design

Two libraries cooperate over RWS:

| Library | Used for |
|---|---|
| `abb_motion_program_exec` | Build `MotionProgram` with `MoveL` / `MoveJ` / `MoveAbsJ`, run via RAPID motion-server |
| `abb_robot_client.rws` | `get_robtarget`, `get_jointtarget`, `set_digital_io`, `set_controller_state`, `get_operation_mode`, `resetpp`, `stop` |

Motion is asynchronous via a single-worker `ThreadPoolExecutor` — `move_*` submits a motion program, returns a `cmd_id`, `wait_motion(cmd_id)` blocks on the future.

---

## Controller-side prerequisites

The IRC5 needs a one-time setup (full guide in `docs/abb_setup.md`):

1. Load vendored RAPID modules from `docker/abb_rapid/HOME/` into the controller `HOME` folder
2. Load `docker/abb_rapid/config_params/EIO.cfg` and `SYS.cfg`
3. Define `DO_VACUUM` digital output signal
4. Set controller to AUTO + motors on (motion-server task starts)

Vendored upstream of `rpiRobotics/abb_motion_program_exec`.

---

## Mode Constants (RobotBase mapping)

| Constant | Value | Meaning |
|---|---|---|
| `MODE_RUNNING` | `1` | `get_execution_state().ctrlexecstate == "running"` |
| `MODE_ERROR` | `2` | Controller in emergency/guard-stop/sysfail |
| `MODE_ENABLED` | `0` | Motors on, idle |

`get_operation_mode()` (driver-specific, not in RobotBase) returns `'AUTO'` / `'MANR'` / `'MANF'` — the physical key-switch state.

---

## Unit Conversions

| Direction | Conversion |
|---|---|
| `move_linear(x,y,z,rx,ry,rz)` | mm passed as-is, ZYX Euler deg → ABB quaternion `(w,x,y,z)` |
| `move_joint_angles(j1..j6)` | degrees passed as-is to `jointtarget` |
| `get_pose()` | ABB quaternion → ZYX Euler deg via `scipy.spatial.transform.Rotation` (fallback: math-only) |
| `get_angle()` | `jointtarget.robax` already degrees |

ABB quaternion convention is `(w, x, y, z)` (scalar first); scipy expects `(x, y, z, w)` — handled inside the converter helpers.

---

## Gripper Control (Digital Output)

Vacuum is mapped to a controller digital output, same model as Dobot CR:

```python
robot.vacuum_on()    # RWS set_digital_io("DO_VACUUM", 1)
robot.vacuum_off()   # RWS set_digital_io("DO_VACUUM", 0)
```

Signal name overridable per-instance via `do_vacuum_signal` kwarg. Define the signal in **I/O System → Signal** in RobotStudio and wire it to the vacuum solenoid.

---

## Freedrive

**Not implemented.** Classic IRB has no hand-guide. Hand-eye calibration's freedrive button stays disabled for ABB (gated on `hasattr(robot, "freedrive_start")`). Manual mode = user jogs from the FlexPendant, then clicks 📷 Capture Pose; the driver's `get_pose()` works in MANUAL mode over RWS.

If using a collaborative ABB (GoFa CRB, YuMi), extend the driver with `freedrive_start/stop` mapped to ABB Lead-Through.

---

## Configuration Data (`confdata`)

ABB `MoveL` to a `robtarget` checks the robot configuration (`cf1`, `cf4`, `cf6`, `cfx`). If the IK solution doesn't match the current configuration, `MoveL` fails with `ConfL`.

The driver reads the current `robtarget.robconf` before each motion and reuses it — so consecutive moves near the current pose succeed without manual config tuning. For large jumps (e.g. GraspGen returns a pose in a different IK branch), you may need to interpose a `MoveAbsJ` to a known waypoint first.

Alternative: disable config checking in the motion server (`ConfL\Off`) — not currently exposed by the driver.

---

## Speed Scaling

`set_speed(percent)` maps 1-100 → linear scale of:

```python
MAX_TCP_SPEED_MMPS = 1000.0   # at 100%
MAX_ORI_SPEED_DEGS = 500.0    # at 100%
```

Each motion uses `abb.speeddata(v_tcp, v_ori, 5000.0, 1000.0)` built from the current scale.

---

## Dependencies

```python
import abb_motion_program_exec
from abb_robot_client.rws import RWS
```

pip package: `abb-motion-program-exec` (pulls `abb-robot-client`). Added to `docker/requirements.txt`.

---

## Links
- [[Robot Architecture]] — base class + registry
- [[Dobot CR]] — comparable DO-vacuum driver
- [[UR10]] — comparable async-motion driver
- [[../../../docs/abb_setup|ABB Setup Guide]] — full IRC5 setup
