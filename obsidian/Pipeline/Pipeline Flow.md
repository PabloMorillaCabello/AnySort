# Pipeline Flow

> End-to-end data path from sensor to robot action.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    PERCEPTION                           │
│                                                         │
│  Orbbec Gemini 2                                        │
│  ┌──────────┐    RGB frame (1280×720)                   │
│  │  Camera  │ ──────────────────────────► SAM3 Server  │
│  │  RGB-D   │    Depth frame (aligned)  ◄─── mask ──── │
│  └──────────┘                                           │
│       │                                                 │
│       │ depth + segmentation mask                       │
│       ▼                                                 │
│  depth_and_segmentation_to_point_clouds()               │
│       │                                                 │
│       │ segmented point cloud (N×3)                     │
│       ▼                                                 │
│  GraspGen (GraspGenSampler)                             │
│       │                                                 │
│       │ grasp poses [4×4 transforms] + scores          │
│       ▼                                                 │
│  filter_colliding_grasps()  ← collision check          │
│  point_cloud_outlier_removal()                          │
└─────────────────────────────────────────────────────────┘
                        │
                        │ best grasp pose (camera frame)
                        ▼
┌─────────────────────────────────────────────────────────┐
│                 COORDINATE TRANSFORM                    │
│                                                         │
│  T_cam2base  (from hand_eye_calib.npz)                  │
│                                                         │
│  pose_camera ──► T_cam2base @ pose_camera ──► pose_base │
│                                                         │
│  Z-correction: −25 mm offset (T_correction)             │
└─────────────────────────────────────────────────────────┘
                        │
                        │ grasp pose (robot base frame)
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  ROBOT EXECUTION                        │
│                                                         │
│  1. robot.move_linear(approach_pose)   ← 40 mm above   │
│  2. robot.wait_motion(cmd_id)                           │
│  3. robot.move_linear(grasp_pose)      ← at object     │
│  4. robot.wait_motion(cmd_id)                           │
│  5. robot.vacuum_on()  → tool.grasp()                   │
│  6. robot.move_linear(approach_pose)   ← lift up        │
│  7. robot.move_linear(home_pose)       ← go home        │
│  8. robot.vacuum_off() → tool.release()                 │
└─────────────────────────────────────────────────────────┘
```

---

## Key Transforms

| Step | Transform | Source |
|---|---|---|
| Depth → point cloud | Camera intrinsics (fx, fy, cx, cy) | `camera_intrinsics.json` |
| Camera frame → base frame | `T_cam2base` (4×4) | `hand_eye_calib.npz` |
| Z correction | `T_correction` (−25 mm) | `hand_eye_calib.npz` |
| Grasp rotation | `trimesh.transformations` | GraspGen output |

---

## GraspGen Output Format

Each grasp pose is a **4×4 homogeneous transform** in camera frame:
```
[R | t]   R = 3×3 rotation, t = 3×1 translation (meters)
[0 | 1]
```
- Score: 0–1 (higher = better)
- Visualized in Meshcat with color mapped to score

---

## ROI (Region of Interest)

Saved in `app/pipeline_roi.json` — polygon mask within camera frame to limit point cloud processing. Structure: `{"poly": [[x1,y1], [x2,y2], ...]}` (null when not set). Configured via UI.

---

## Links
- [[AnySort Pipeline]] — the orchestrator
- [[../Components/Camera - Orbbec Gemini 2|Camera]] — point cloud source
- [[../Components/SAM3 Segmentation|SAM3]] — segmentation
- [[../Components/GraspGen|GraspGen]] — grasp poses
- [[../Calibration/Hand-Eye Calibration|Hand-Eye Calibration]] — frame transform
- [[../Components/Robots/Robot Architecture|Robot]] — execution layer
