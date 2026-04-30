# Hand-Eye Calibration

> Calibrates the transform between camera frame and robot base frame.
> **Files:** `data/calibration/hand_eye_calib*.json/.npz`
> **Scripts:** `app/hand_eye_calibration.py`, `app/calibration_tester.py`

---

## Calibration Type

**Eye-to-hand** — camera is fixed (mounted externally), ChArUco board attached to robot gripper.

```
Camera (fixed)
    ↓ sees board
ChArUco board on gripper
    ↑ mounted to
Robot arm (moves through poses)
```

---

## Method

`cv2.calibrateHandEye()` — AX = XB solver.

Input:
- 26+ robot poses (from `data/calibration/auto_calib_poses.json`)
- Corresponding ChArUco board detections from camera

Output:
- `T_cam2base` — 4×4 homogeneous transform (camera frame → robot base frame)
- `T_correction` — additional Z-offset correction

---

## Calibration Files

| File | Content |
|---|---|
| `hand_eye_calib.json/.npz` | Base calibration |
| `hand_eye_calib_corrected_20260403_150244.json/.npz` | **Best calibration** (corrected) |
| `auto_calib_poses.json` | 26 pre-programmed robot poses |

**Active file used by pipeline:**
```python
CALIB_FILE = "/ros2_ws/data/calibration/hand_eye_calib.npz"
```

---

## Z-Correction

After initial solve, a −25 mm Z offset was applied as `T_correction`. This accounts for systematic error in the initial solve (tool geometry not perfectly modeled).

---

## Pre-Programmed Poses

26 poses in `auto_calib_poses.json` — cover full rotation space of the robot workspace. Used for automated calibration collection (robot moves through poses automatically).

---

## Run Calibration

```bash
# Interactive ChArUco hand-eye calibration UI
python /ros2_ws/app/hand_eye_calibration.py --robot-ip 192.168.5.1

# Validate current calibration
python /ros2_ws/app/calibration_tester.py
```

---

## Usage in Pipeline

```python
calib = np.load(CALIB_FILE)
T_cam2base   = calib["T_cam2base"]    # 4×4 float64
T_correction = calib.get("T_correction", np.eye(4))

# Transform grasp pose from camera → base:
pose_base = T_cam2base @ T_correction @ pose_camera
```

---

## Validation

`calibration_tester.py`:
- Moves robot to known poses
- Measures actual vs predicted position
- Reports 6-DOF correction errors

---

## Links
- [[Camera Intrinsics]] — required for board detection
- [[../Components/Camera - Orbbec Gemini 2|Camera]] — the calibrated sensor
- [[../Components/Robots/Dobot CR|Dobot CR]] — robot used for calibration
- [[../Pipeline/Pipeline Flow|Pipeline Flow]] — transform in data path
