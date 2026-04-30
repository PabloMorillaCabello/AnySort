# Camera Intrinsics

> Calibration of Orbbec Gemini 2 lens parameters.
> **Files:** `data/calibration/camera_intrinsics.json`, `camera_intrinsics.npz`
> **Script:** `app/camera_calibration.py`

---

## Calibrated Values

| Parameter | Value |
|---|---|
| Resolution | 1280 × 720 |
| fx | 684.7 |
| fy | 685.9 |
| cx (principal point X) | 655.3 |
| cy (principal point Y) | 357.0 |
| **RMS reprojection error** | **0.20 px** |
| Frames used | 31 |

---

## Camera Matrix

```
K = [[684.7,   0,   655.3],
     [  0,   685.9, 357.0],
     [  0,     0,     1  ]]
```

---

## Calibration Process

Run from inside container:
```bash
python /ros2_ws/app/camera_calibration.py
```

Steps:
1. Print ChArUco board
2. Capture 30+ frames from different angles
3. Script detects corners, runs `cv2.calibrateCamera()`
4. Saves to `data/calibration/camera_intrinsics.json/.npz`

---

## Usage in Pipeline

Intrinsics fed to `depth_and_segmentation_to_point_clouds()`:
```python
points = depth_and_segmentation_to_point_clouds(
    depth_frame, segmentation_mask,
    fx=684.7, fy=685.9, cx=655.3, cy=357.0
)
```

---

## Links
- [[Hand-Eye Calibration]] — uses these intrinsics
- [[../Components/Camera - Orbbec Gemini 2|Camera]] — the calibrated device
- [[../Pipeline/Pipeline Flow|Pipeline Flow]] — where intrinsics are used
