# AnySort Pipeline

> Single entry point for the full grasping pipeline.
> **File:** `app/anysort.py`

---

## What It Does

AnySort is the **main application**. On startup it:
1. Shows a Tkinter splash screen
2. Auto-starts the [[../Components/SAM3 Segmentation|SAM3 server]] via Unix socket
3. Starts Meshcat visualizer (URL printed at startup)
4. Connects to the [[../Components/Camera - Orbbec Gemini 2|Orbbec camera]]
5. Loads [[../Components/GraspGen|GraspGen]] weights

Then enters interactive loop for grasping.

---

## Launch Commands

```bash
# Windows — no terminal window (preferred)
AnySort.vbs           # double-click at repo root

# Windows — with terminal
AnySort.cmd

# Inside container
cd /ros2_ws/app && python anysort.py
```

---

## Key Constants

| Constant | Value | Purpose |
|---|---|---|
| `CHECKPOINTS_DIR` | `/opt/GraspGen/GraspGenModels/checkpoints` | GraspGen model weights |
| `SAM3_SERVER_SCRIPT` | `/ros2_ws/app/sam3_server.py` | SAM3 auto-start path |
| `CALIB_FILE` | `/ros2_ws/data/calibration/hand_eye_calib.npz` | Hand-eye transform |
| `ROBOT_IP_DEFAULT` | `192.168.5.1` | Default robot IP |
| `APPROACH_OFFSET` | `40 mm` | Pre-grasp approach height |
| `HOME_POSE` | `[300, 0, 450, 0, 0, 0]` | Safe home position (mm/deg) |

---

## Pipeline Modes

### Batch Word-List Mode
- Loads a `.txt` word list from `data/object_lists/`
- Each word → SAM3 segment → GraspGen poses → robot execute
- Config saved in `app/pipeline_positions.json`:
  - `home_joints` — robot home joint config
  - `sort_joints` — sort drop-off joint config
  - `tcp_z_offset` — TCP height correction
  - `word_list` — last used object targets

### Interactive Mode
- Click on camera preview to select object
- SAM3 segments clicked region
- GraspGen generates grasp poses
- User confirms → robot executes

---

## Robot Error Recovery
- Monitors robot mode in background thread
- On error/protective stop → UI shows recovery dialog
- Auto-retries after `clear_error()` + `enable()`

---

## SAM3 Auto-Start Flow

```
anysort.py
  → _find_sam3_python()      # finds /opt/sam3env/bin/python3.12
  → subprocess.Popen(sam3_server.py)
  → _wait_for_sam3_socket()  # blocks up to 600s (model download)
  → Unix socket connected at /tmp/sam3_server.sock
```

---

## Visualization

- **Meshcat** — 3D point cloud + grasp poses (browser, port shown at startup)
- **Tkinter UI** — live RGB+depth preview, controls, status log

---

## Links
- [[Pipeline Flow]] — data path diagram
- [[../Components/Robots/Robot Architecture|Robot Architecture]] — how robots plug in
- [[../Components/Tools/Tool Architecture|Tool Architecture]] — how tools plug in
- [[../Calibration/Hand-Eye Calibration|Hand-Eye Calibration]] — transform used
- [[../Known Issues & Fixes|Known Issues]] — Tkinter thread-safety, TCP reads
