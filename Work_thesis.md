# Chapter 4 — Design and Implementation

This chapter describes the design and implementation of the final solution, covering each module of the pipeline, the design decisions behind them, and how they interconnect.

---

## 4.1 System Architecture Overview

The system comprises four core modules: a perception module to acquire RGB-D data, a segmentation module to isolate the target object, a grasp generation module to estimate grasping poses, and a robot module to execute the motion. In addition, a set of calibration tools was developed to simplify deployment in new environments.

The complete system is orchestrated by a single entry-point application (`app/anysort.py`), referred to as **AnySort**. On startup it initialises all subsystems sequentially — launching the segmentation server, the 3D visualiser, the camera, and loading the grasp model weights — then presents the main interface to the operator.

The pipeline operates as follows: the Orbbec Gemini 2 captures a colour image and an aligned depth frame. The colour image is sent to the SAM3 segmentation server, which returns a binary mask of the target object. The depth frame combined with the mask produces a 3D point cloud, which GraspGen processes to output a ranked set of 6-DoF grasp poses. These poses are transformed from the camera frame to the robot base frame via the hand-eye calibration matrix, filtered for feasibility, and the best candidate is executed by the robot.

```
Orbbec Gemini 2 (RGB-D)
        |
        |--- RGB frame (1280x720) ---> SAM3 Segmentation Server
        |                                      |
        |<--- binary mask (H x W) ------------|
        |
        |--- depth + mask --> Point Cloud Construction
                                      |
                                      v
                              GraspGen Inference
                                      |
                                      v
                         Coordinate Frame Transform (T_cam2base)
                                      |
                                      v
                           Grasp Filtering & Ranking
                                      |
                                      v
                           Robot Execution (Dobot CR / UR10)
                                      |
                                      v
                           End-Effector Actuation (Vacuum / Gripper)
```

---

## 4.2 RGB-D Perception Module

The perception module acquires synchronised colour and depth data using the **Orbbec Gemini 2**, a structured-light RGB-D camera (USB 3.0, 1280×720). Its hardware depth-to-colour alignment ensures each depth pixel corresponds directly to its colour counterpart without software reprojection, allowing the SAM3 segmentation mask to be applied to the depth map directly. The camera is mounted in a fixed, elevated position angled downward towards the robot work surface.

The software interface uses OrbbecSDK v2.7.6 with Python bindings built from source inside the container. The camera pipeline is started fresh per capture and stopped immediately afterwards to avoid a memory leak present in long WSL2 sessions. Ten warmup frames are discarded at each start to allow auto-exposure and depth processing to converge. The spatial noise filter is enabled; temporal and hole-filling filters are disabled as they maintain state that produces incorrect results across restarts.

The OrbbecSDK emits C-level diagnostic messages to `stderr` that cannot be suppressed via Python. The module `app/orbbec_quiet.py` redirects Unix file descriptor 2 to `/tmp/orbbec_sdk.log` via `os.dup2()` before the SDK is imported, keeping the terminal clean.

---

## 4.3 Segmentation Module

Object segmentation is performed in the 2D colour image domain. The resulting binary mask is then applied to the depth frame to isolate the object's 3D geometry. This approach was chosen because 2D segmentation is a mature field with powerful zero-shot models, whereas direct 3D point cloud segmentation is significantly less mature.

**SAM3 (Segment Anything Model 3)** was selected for its zero-shot generalisation to arbitrary, unseen object categories and its support for both text and click prompts — essential for a sorting system handling an open-ended set of objects.

SAM3 runs as a **persistent Unix-socket server** (`app/sam3_server.py`) to avoid reloading its weights on every query (model load takes up to several minutes). The server is launched once as a daemon subprocess at startup, loads the weights into GPU memory, and handles repeated inference requests with sub-second latency. It runs in an isolated Python 3.12 virtual environment (`/opt/sam3env/`) because its dependencies (PyTorch 2.7) conflict with GraspGen's environment (Python 3.10).

Each inference transaction over the socket follows a simple framing protocol:

- **Request:** JSON header line `{"width", "height", "prompt", "size"}` followed by the raw RGB24 image bytes.
- **Response:** JSON line `{"ok", "size", "num_masks"}` followed by the raw mask bytes (`K×H×W uint8`).

On the client side, the first returned mask is selected and a largest-connected-component filter removes isolated noise pixels, yielding a single `H×W` binary mask.

---

## 4.4 Calibration System

A precise camera-to-robot transform is essential: any calibration error manifests directly as a grasping position error. Three dedicated tools were built to establish and validate this transform.

### 4.4.1 Camera Intrinsics

Although most cameras provide factory intrinsic values, small per-unit manufacturing differences can exist between the nominal specification and the actual hardware. A dedicated module (`app/camera_calibration.py`) was developed to measure the intrinsic parameters — focal lengths (`fx`, `fy`) and principal point (`cx`, `cy`) — directly from the physical camera using a ChArUco board. The operator captures 30 or more frames from different angles; the module detects sub-pixel corners in each frame and estimates the parameters via `cv2.calibrateCamera()`. For the Orbbec Gemini 2 used in this work, calibration over 31 frames yielded an RMS reprojection error of 0.20 px:

```
K = [[684.7,   0,   655.3],
     [  0,   685.9, 357.0],
     [  0,     0,     1  ]]
```

Undistortion is applied only to the colour image; the depth frame is left unchanged to avoid flying-pixel artefacts at depth discontinuities.

### 4.4.2 Hand-Eye Calibration

The main calibration step estimates **T_cam2base**, the 4×4 rigid transform from the camera frame to the robot base frame. A graphical module (`app/hand_eye_calibration.py`) was developed to make this process reproducible and robot-agnostic. Its key features are: robot model selection, editable target pose lists, ArUco marker generation, and automated calibration with multi-method comparison.

The calibration follows an eye-to-hand configuration — the camera is fixed and a ChArUco board is mounted rigidly on the robot end-effector. The operator teaches at least 15 poses (a set of 26 pre-programmed poses covering the full rotation space is provided), then triggers the automated collection: the robot moves to each pose, pauses 2.5 s for mechanical stabilisation, and the camera captures the board only if a sufficient number of corners are detected. Once all poses are collected, five solver methods (`TSAI`, `PARK`, `HORAUD`, `ANDREFF`, `DANIILIDIS`) are evaluated via `cv2.calibrateHandEye()`; the method with the lowest reprojection error is selected automatically and all results are reported to the operator.

### 4.4.3 Calibration Correction

In practice, hand-eye calibration always retains some residual error. The module `app/calibration_tester.py` provides an empirical way to quantify and correct it. An ArUco marker is placed on the robot work surface; the system detects it in the camera image, transforms its centroid to the robot base frame using the current calibration, and commands the robot to that position. Any discrepancy between the predicted and actual position directly reveals the calibration error.

To correct it, the module offers two options: six-axis correction sliders (ΔX/ΔY/ΔZ in mm, ΔRoll/ΔPitch/ΔYaw in degrees) for fine adjustment, or manual robot guidance in freedrive mode to the exact marker position. The resulting correction is composed into the calibration transform and saved as a timestamped `.npz` file. For this system a −25 mm Z correction was applied after the initial solve.

---

## 4.5 Coordinate Frame Transformations

This module is the mathematical bridge between the perception subsystem and the robot. It transforms every grasp pose from the camera coordinate frame into the robot base frame that the controller understands. The full chain has three steps.

**Step 1 — Depth to point cloud.** Each pixel `(u, v)` within the SAM3 mask with depth `d` is back-projected into 3D camera-frame coordinates using the pinhole model:

```
X = (u - cx) * d / fx
Y = (v - cy) * d / fy
Z = d
```

A 4-pixel erosion is applied to the mask boundary beforehand to remove flying-pixel artefacts at depth edges.

**Step 2 — Camera frame to robot base frame.** The point cloud and all grasp poses output by GraspGen are transformed to the robot base frame using the hand-eye calibration result:

```
G_base = T_cam2base @ T_correction @ G_cam
```

where `T_cam2base` is the calibrated transform and `T_correction` is the empirical correction from Section 4.4.3.

**Step 3 — Convention alignment.** GraspGen expects a Z-up world frame (Z pointing away from gravity). Because the camera is mounted looking downward, its optical axis points toward the ground, which conflicts with this convention. The pipeline detects this at startup and temporarily rotates the coordinate frame so that GraspGen always receives a consistent Z-up representation. The inverse rotation is applied to all output poses afterwards, restoring them to the correct robot base frame orientation before execution.

---

## 4.6 Grasp Generation Module

**GraspGen** (NVIDIA Research) takes a centred, metric point cloud of the target object and outputs N ranked 6-DoF grasp poses as 4×4 homogeneous matrices with confidence scores in [0, 1]. It runs in a dedicated Python 3.10 virtual environment with compiled PointNet++ CUDA extensions.

Before inference, the point cloud undergoes statistical K-NN outlier removal (`point_cloud_outlier_removal()`), and colours are re-associated to the filtered cloud via cKDTree nearest-neighbour lookup.

After inference, three sequential filters are applied:

1. **Top-down filter** — retains only grasps whose approach vector aligns with gravity (approach from above), discarding side and bottom approaches infeasible on a table-top.
2. **Collision filter** — removes grasps where the gripper mesh intersects the scene point cloud.
3. **Reachability filter** — discards grasps for which the robot IK solver finds no valid solution at the grasp or pre-grasp pose.

The surviving grasps are sorted by confidence score and made available for visualisation and execution.

---

## 4.7 Robot Abstraction Layer

A core design goal of this system is robot-agnosticism: the pipeline should work with any robot arm without modifications to the core logic. This is achieved through a modular driver architecture based on an abstract base class `RobotBase` (`app/robots/base.py`). Every robot driver implements this interface, and the pipeline calls only its methods — no robot-specific code appears anywhere in `anysort.py`.

Adding support for a new robot requires only creating a new driver file that implements `RobotBase` and registering it by name. The new robot then appears automatically in the application's UI dropdown, ready to use. Two drivers are currently implemented: Dobot CR and UR10.

The interface covers four areas: lifecycle (`enable`, `clear_error`, `stop`, `close`), status (`get_mode`, `get_pose`, `get_angle`), motion (`move_linear`, `move_joint_angles`, `wait_motion`), and end-effector (`attach_tool`, `vacuum_on`, `vacuum_off`). Mode states (`MODE_RUNNING`, `MODE_ERROR`, `MODE_ENABLED`) are normalised across all drivers so the pipeline's error monitoring works identically regardless of the robot platform.

### 4.7.1 Dobot CR Driver

The Dobot CR driver communicates over two TCP/IP connections: port 29999 for text-based dashboard commands (`MovL`, `MovJ`, `EnableRobot`, etc.) and port 30004 for 1440-byte binary feedback packets at 250 Hz. A daemon thread reads the feedback stream continuously and stores state under a lock.

Two bugs in the upstream Dobot API were patched at Docker build time:

- **TCP partial reads:** `socket.recv()` may return fewer than 1440 bytes. The fix accumulates chunks in a buffer until the full packet is received before parsing.
- **Tkinter thread-safety:** Direct widget updates from the feedback thread cause segfaults on Linux. The fix dispatches all UI updates to the main thread via `widget.after(0, callback)`.

### 4.7.2 UR10 Driver

The UR10 driver deploys a custom URScript command server (`ur10_cmd_server.script`) to the robot controller at startup via port 30002, then communicates via RTDE integer and double registers. The pipeline writes a command code and target pose to input registers; the URScript loop on the controller executes the motion and writes status back to output registers. This avoids the limitations of real-time URScript parsing and secondary-socket conflicts with the `ur_rtde` daemon. All units are converted transparently: the pipeline uses mm and Euler degrees; the UR controller expects metres and axis-angle radians.

---

## 4.8 End-Effector Abstraction

End-effectors implement the two-method `ToolBase` interface (`grasp()`, `release()`) and are attached to the robot at runtime. The pipeline calls `robot.vacuum_on()` / `vacuum_off()`, which delegate to the attached tool.

**Dobot Vacuum:** sends `ToolDO(port, 1/0)` via the dashboard to activate/deactivate a vacuum solenoid valve.

**OnRobot RG (UR10):** loads and plays pre-saved `.urp` programs on the UR teach pendant (`gripper_close.urp` / `gripper_open.urp`). Direct `rg_grip()` URScript calls are unreliable outside the URCap program context; running a saved program avoids this entirely. Completion is detected by polling `programState` every 0.15 s, followed by a 5 s settle delay.

---

## 4.9 Execution Sequence

1. **Release tool** — open gripper/vacuum off (0.3 s settle).
2. **MovJ → pre-grasp** — joint motion to a point `approach_offset` mm above the grasp target.
3. **MovL → grasp pose** — linear descent to the grasp position.
4. **Activate tool** — vacuum on / gripper close (0.8 s settle).
5. **MovL → pre-grasp** — linear retreat with object.
6. **MovJ → sort position** — joint motion to the drop-off bin.
7. **Release tool** — drop object.
8. **MovJ → home** — return to home configuration, ready for next cycle.

On failure, the system retries with the next-ranked grasp after a recovery sequence (stop → clear error → enable → home).

---

## 4.10 Batch Processing Mode

In batch mode, a word list defines a sequence of object categories to sort. For each word, the system performs capture → SAM3 segmentation (using the word as text prompt) → GraspGen inference → robot execution, with up to three attempts before advancing. The loop runs indefinitely until stopped by the operator.

---

## 4.11 Visualisation

A **Meshcat** browser-based 3D viewer displays the scene point cloud, coordinate frames, and all generated grasp poses as gripper meshes colour-coded by confidence (best = yellow). The operator can click any grasp to select it for execution, overriding the automatic ranking. The viewer is self-hosted inside the container and its URL is printed at startup.

---

## 4.12 Portability and Reproducibility

One research aim of this thesis is to produce a system that can be deployed on a new machine without manual dependency resolution or environment configuration. This is non-trivial: the stack combines components with deeply incompatible requirements — GraspGen (Python 3.10, CUDA 12.6), SAM3 (Python 3.12, PyTorch 2.7), ROS2 Humble, and native C libraries for the Orbbec SDK. Resolving these conflicts on a bare host is error-prone and hard to reproduce.

The solution is to encode the entire environment as a **Docker image**. The Dockerfile uses a 10-stage multi-stage build, where each stage has a clearly scoped responsibility: base OS and CUDA drivers, system dependencies, GraspGen environment, Python packages, Dobot API (with automatic bug patching), SAM3 environment, Orbbec SDK, ROS2 workspace, and entrypoint. All dependency conflicts are resolved once inside the image definition and are identical on every machine that builds it. A new deployment requires only cloning the repository and running:

```bash
docker compose -f docker/docker-compose.yml build
```

**Windows and Linux support** is handled through the same image. The container runs a Linux environment (Ubuntu + ROS2 Humble) regardless of the host OS. On Linux, Docker accesses hardware directly. On Windows, Docker runs inside WSL2; USB devices such as the Orbbec camera must be forwarded from the Windows host into WSL2 using `usbipd-win`, which is automated by the provided `scripts/reattach.ps1` script. On Windows, the system can be launched with a double-click on `AnySort.vbs` — no terminal required.

The **GitHub repository** is structured so that everything needed to build, run, and calibrate the system is self-contained. Source code in `app/` is live-mounted into the container, so code edits on the host take effect immediately without rebuilding the image. Calibration files, configuration, and launcher scripts are all versioned alongside the code, ensuring that any collaborator who clones the repository starts from an identical, known-good state.
