# GraspGen

> NVIDIA grasp pose generation model.
> **Upstream:** https://github.com/NVlabs/GraspGen

---

## What It Does

Takes a **segmented point cloud** and outputs ranked **6-DOF grasp poses** (4×4 homogeneous transforms) with confidence scores.

---

## Python Environment

| Property | Value |
|---|---|
| Venv path | `/opt/GraspGen/.venv/` |
| Python | 3.10 (uv) |
| Key deps | PyTorch, PointNet++ CUDA extensions |
| Activate | `graspgen_activate` (container alias) |

---

## Key Imports in Pipeline

```python
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.robot import get_gripper_info
from grasp_gen.utils.meshcat_utils import (
    create_visualizer, get_color_from_score, make_frame,
    visualize_grasp, visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import (
    depth_and_segmentation_to_point_clouds,
    filter_colliding_grasps,
    point_cloud_outlier_removal,
)
```

---

## Model Weights

- Location: `/opt/GraspGen/GraspGenModels/checkpoints`
- Downloaded via: `scripts/download_models.sh`
- Loaded during AnySort splash screen

---

## Grasp Output

Each pose:
- **Transform:** 4×4 homogeneous matrix (camera frame, meters)
- **Score:** float 0–1 (higher = better grasp quality)
- Post-processing:
  - `filter_colliding_grasps()` — removes poses that collide with scene
  - `point_cloud_outlier_removal()` — cleans point cloud before inference

---

## Visualization

Meshcat (browser-based 3D viewer):
- Point cloud rendered as colored dots
- Grasp poses shown as coordinate frames
- Color mapped to score (via `get_color_from_score`)
- URL printed at AnySort startup

---

## CUDA Requirement

- CUDA 12.6 (from base Docker image)
- PointNet++ extensions compiled at Docker build time
- **numpy must stay <2** — PyTorch compiled against numpy 1.x

---

## Test Script

```bash
python3 /ros2_ws/scripts/test_graspgen.py
```

---

## Links
- [[../Pipeline/Pipeline Flow|Pipeline Flow]] — where GraspGen fits
- [[../Pipeline/AnySort Pipeline|AnySort Pipeline]] — orchestrator
- [[../Infrastructure/Docker Setup|Docker Setup]] — build stage 3 (GraspGen env)
- [[../Infrastructure/Python Environments|Python Environments]] — venv details
- [[../Known Issues & Fixes|Known Issues]] — numpy<2 constraint
