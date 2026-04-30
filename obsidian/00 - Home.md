# GraspGen Thesis — Vault Index

> Master Thesis: Robotic grasping pipeline integrating GraspGen + SAM3 + robot arms + Orbbec depth camera.
> **Status (2026-04-18):** Active grasping tests. AnySort UI fully functional.

---

## Pipeline

- [[Pipeline/AnySort Pipeline|AnySort Pipeline]] — Single-command entry point + full flow
- [[Pipeline/Pipeline Flow|Pipeline Flow]] — Step-by-step data path (Camera → SAM3 → GraspGen → Robot)

---

## Components

### Perception
- [[Components/Camera - Orbbec Gemini 2|Orbbec Gemini 2]] — RGB-D camera, SDK, ROS2 driver
- [[Components/SAM3 Segmentation|SAM3]] — Segment Anything 3 server (auto-started)
- [[Components/GraspGen|GraspGen]] — NVIDIA grasp pose generation

### Robots
- [[Components/Robots/Robot Architecture|Robot Architecture]] — Abstract base class + registry
- [[Components/Robots/Dobot CR|Dobot CR]] — TCP/IP CR-series driver
- [[Components/Robots/UR10|UR10]] — RTDE register-based driver

### End-Effectors
- [[Components/Tools/Tool Architecture|Tool Architecture]] — Abstract base class + registry
- [[Components/Tools/Dobot Vacuum|Dobot Vacuum]] — Digital output vacuum gripper
- [[Components/Tools/OnRobot RG|OnRobot RG]] — Dashboard program gripper (UR)

---

## Infrastructure

- [[Infrastructure/Docker Setup|Docker Setup]] — Multi-stage Dockerfile, compose, 10 build stages
- [[Infrastructure/Python Environments|Python Environments]] — 3 envs (GraspGen/SAM3/System)
- [[Infrastructure/USB Passthrough|USB Passthrough]] — WSL2 → Docker via usbipd-win

---

## Calibration

- [[Calibration/Camera Intrinsics|Camera Intrinsics]] — 1280×720, RMS 0.20 px
- [[Calibration/Hand-Eye Calibration|Hand-Eye Calibration]] — Eye-to-hand ChArUco, T_cam2base

---

## Reference

- [[Known Issues & Fixes]] — Dobot TCP/Tkinter bugs, Orbbec quirks, OnRobot timing
- [[Commands Reference]] — Quick commands inside/outside container

---

## Repo Map

```
GraspGen_Thesis_Repo/
├── app/                      ← FINAL PIPELINE (AnySort)
│   ├── anysort.py                  ← main entry point
│   ├── robots/               ← robot drivers (base + Dobot CR + UR10)
│   ├── tools/                ← end-effector drivers (base + vacuum + OnRobot)
│   ├── sam3_server.py        ← persistent SAM3 socket server
│   ├── hand_eye_calibration.py
│   ├── calibration_tester.py
│   └── camera_calibration.py
├── docker/                   ← Dockerfile + compose + patches
├── data/calibration/         ← camera intrinsics + hand-eye transforms
├── scripts/                  ← tests + utilities
├── AnySort.vbs               ← Windows one-click launcher
└── CLAUDE.md                 ← project instructions for Claude
```
