# ABB IRB (IRC5) Setup Guide

End-to-end setup to use an ABB IRB robot with an IRC5 controller from this repo's AnySort pipeline and hand-eye calibration tool.

Target: classic IRB on **IRC5** with **RobotWare 6.xx**, vacuum gripper on a digital output.

---

## 1. What's involved

Two cooperating pieces:

| Side | What runs there |
|---|---|
| **Python (inside Docker container)** | `app/robots/abb_irb.py` driver → talks to controller via HTTP/REST (RWS) |
| **Controller (IRC5)** | `motion_program_exec` RAPID module → receives motion programs from Python and executes them |

The Python library `abb-motion-program-exec` and its dependency `abb-robot-client` are installed by the container (already added to `docker/requirements.txt`). The RAPID module has to be loaded onto the controller manually one time.

---

## 2. Controller-side setup (one-time, via RobotStudio)

### 2.1 RobotWare and licence

Required:
- IRC5 controller
- RobotWare 6.xx (any minor version)
- No paid options needed (no EGM, no Lead-Through)

Open RobotStudio → connect to the controller (Ethernet, port 80/443). Confirm the **Controller** tab shows the IRC5 online.

### 2.2 Copy the RAPID modules to the controller

The required files are vendored in this repo at `docker/abb_rapid/HOME/`:

```
motion_program_exec.mod        ← main motion server (REQUIRED)
motion_program_logger.mod      ← logging task (REQUIRED)
motion_program_shared.sys      ← shared data types (REQUIRED)
error_reporter.mod             ← error event reporter (REQUIRED)
motion_program_exec_egm.mod    ← EGM support (OPTIONAL — only if you license EGM)
```

In RobotStudio:

1. **Controller** tab → **File Transfer** → expand `HOME` on the controller.
2. Drag the four required `.mod`/`.sys` files from `docker/abb_rapid/HOME/` into the controller's `HOME` folder.
3. (Optional) Drag `motion_program_exec_egm.mod` too, only if you have the EGM option.

### 2.3 Load the I/O signal configuration

Apply the EIO configuration that defines the internal signals the motion server uses (`motion_program_executing`, `motion_program_seqno`, ...):

1. **Controller** tab → **Configuration Editor** → **I/O System** → **Load Parameters**.
2. Choose `docker/abb_rapid/config_params/EIO.cfg`.
3. Pick "Load parameters and replace duplicates" → OK.
4. Repeat for `docker/abb_rapid/config_params/SYS.cfg` (defines the RAPID tasks).
5. Restart the controller (X-start or warm-start as RobotStudio prompts).

### 2.4 Add the `DO_VACUUM` signal

The vacuum solenoid is wired to a DSQC io board (or equivalent). Define the signal:

1. **Configuration Editor** → **I/O System** → **Signal** → **Add**.
2. Name: `DO_VACUUM`. Type: `Digital Output`. Device + device-map: whichever physical channel the vacuum solenoid is wired to.
3. Apply and restart the controller.

If your installation already exposes the vacuum under a different name, you can keep that name and pass it to the driver via the `do_vacuum_signal` kwarg — see [section 4.2](#42-driver-kwargs).

### 2.5 Set the motion-server task to auto-start

The `motion_program_exec` module runs in a background task that must be on at boot:

1. **Configuration Editor** → **Controller** → **Task** → confirm `T_ROB1` exists.
2. The vendored `SYS.cfg` already declares the logger task. Verify under **Task** that `motion_program_logger` is set to **Type = NORMAL**, **TrustLevel = NoSafety**, **Main entry = main**.
3. Set the controller to **AUTO** mode (key-switch on the FlexPendant) and press **Motors On**. The motion server is now listening.

### 2.6 Network

Two networking choices on the IRC5:
- **Service port** (yellow, default `192.168.125.1/24`) — quick, point-to-point, good for first bring-up.
- **WAN port** — plug into your lab subnet, configure via the FlexPendant under **Control Panel → Configuration → Communication**.

Note whichever IP you'll use. The driver expects HTTP on port 80 (default RWS port). Verify from the host:

```bash
curl -u "Default User:robotics" http://<ABB_IP>/rw/system
```

A successful response is XML with the controller's serial and RobotWare version. A `401` means wrong credentials; a `timeout` means wrong IP or network.

---

## 3. Container-side setup

The Python deps are already declared in `docker/requirements.txt`. Rebuild the image to pick them up:

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml exec graspgen /bin/bash
```

Quick import smoke test inside the container:

```bash
graspgen_activate
python -c "from robots import get_driver_names; print(get_driver_names())"
# expected: ['Dobot CR', 'UR10', 'ABB IRB']
```

---

## 4. Using the driver

### 4.1 From a Python REPL (smoke test)

```python
from robots import create_robot

robot = create_robot("ABB IRB", "192.168.125.1")
print("mode :", robot.get_mode())          # MODE_ENABLED if motors on
print("pose :", robot.get_pose())          # (x, y, z, rx, ry, rz) mm / deg
print("joints:", robot.get_angle())        # (j1..j6) deg
print("op   :", robot.get_operation_mode()) # 'AUTO' or 'MANR' / 'MANF'
robot.close()
```

If `get_mode()` returns `MODE_ERROR`, check that motors are on and there are no active guard-stops on the pendant.

### 4.2 Driver kwargs

```python
create_robot(
    "ABB IRB", "192.168.125.1",
    username="Default User",        # RWS auth
    password="robotics",
    do_vacuum_signal="DO_VACUUM",   # override if your signal has a different name
    tool="tool0",                   # active TCP for pose reads / motion
    wobj="wobj0",                   # active work object
    mechunit="ROB_1",               # mechanical unit identifier
)
```

### 4.3 Hand-eye calibration

```bash
python /ros2_ws/app/hand_eye_calibration.py --robot-ip <ABB_IP>
```

- Pick **ABB IRB** from the robot dropdown.
- For **Manual mode**: switch the controller key-switch to MANUAL on the pendant. Jog the robot to each calibration target, then click 📷 **Capture Pose**. Pose reads work in MANUAL — only motion programs are gated.
- For **Auto mode**: keep the controller in AUTO with motors on. The robot moves to each target via `MoveL`. Use the **Save current pose** button to build a per-robot pose list (e.g. `auto_calib_poses_ABB.json`).
- The **Freedrive** button is intentionally disabled — classic IRBs have no hand-guide. If you have a GoFa/YuMi/CRB collaborative arm, contact the maintainer to extend the driver.

### 4.4 AnySort

```bash
AnySort.vbs    # Windows one-click launcher
```

Pick **ABB IRB** from the robot dropdown and the appropriate tool. The pipeline calls `move_linear`, `move_joint_angles`, `vacuum_on/off`, `wait_motion` — all routed through the RAPID motion server.

---

## 5. AUTO vs MANUAL mode

The ABB controller has a physical key-switch on the FlexPendant with three positions:

| Position | What the driver can do |
|---|---|
| **AUTO** | Full pipeline — `move_*`, `vacuum_*`, `wait_motion`. Required for AnySort and calibration's Auto mode. |
| **MANR / MANF** (Manual) | Pose reads (`get_pose`, `get_angle`, `get_mode`) still work. Motion programs are rejected by the controller. Use this for calibration's **Manual** mode where the user jogs from the pendant. |

`ABBIRB.get_operation_mode()` exposes the current key-switch state if you want to surface it in your own UI.

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ConnectionError: [ABBIRB] connection failed` | Wrong IP, unreachable, RWS disabled | `curl http://<ip>/rw/system` — must return 200/401. Check Service vs WAN port. |
| `401 Unauthorized` | Default credentials changed | Pass `username=...`, `password=...` to `create_robot`. |
| `Motion failed: ...ConfL...` | Robot configuration changed mid-task, IK can't reach | Driver reads current `confdata` per move — but if it's been a long time since the last move, briefly do `get_pose()` to refresh, then retry. |
| Motion programs silently never start | Controller in MANUAL or motors off | Switch to AUTO, press Motors On, check **System Output** on pendant for guard-stop events. |
| `vacuum_on()` does nothing | `DO_VACUUM` signal not defined or wired | Confirm signal in **I/O System → Signal**, test from pendant: **I/O Inputs/Outputs → Set DO_VACUUM = 1**. |
| Hand-eye calibration "Capture Pose" reads stale pose | Manual mode is jogging via virtual robot (RobotStudio simulation) | Confirm RobotStudio is connected to the **real** controller, not a station. |

---

## 7. Files reference

| Path | Purpose |
|---|---|
| `app/robots/abb_irb.py` | Driver |
| `app/robots/__init__.py` | Registry — `"ABB IRB": ("robots.abb_irb", "ABBIRB")` |
| `docker/requirements.txt` | `abb-motion-program-exec` pip dep |
| `docker/abb_rapid/HOME/` | Vendored RAPID modules — copy these to controller HOME |
| `docker/abb_rapid/config_params/` | Vendored EIO/SYS configs — load via Configuration Editor |

Upstream source for vendored files: <https://github.com/rpiRobotics/abb_motion_program_exec> (re-fetch from `robot/` directory if you need a newer version).
