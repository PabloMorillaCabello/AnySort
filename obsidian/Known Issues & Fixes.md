# Known Issues & Fixes

> All known bugs, quirks, and their resolutions.

---

## Dobot: TCP Partial Read

**Bug:** `socket.recv(14400)` doesn't guarantee 1440 bytes. If packet is incomplete, `np.frombuffer()` raises `ValueError`.

**Symptom:** Random crashes in feedback thread, `ValueError: buffer size must be a multiple of...`

**Fix** (in `docker/patches/fix_dobot_feedback.py`):
```python
# WRONG
data = self.client_feed.socket_dobot.recv(14400)[0:1440]
a = np.frombuffer(data, dtype=MyType)  # fails if incomplete

# CORRECT — accumulate until exactly 1440 bytes
data = bytes()
while len(data) < 1440:
    remaining = 1440 - len(data)
    chunk = self.client_feed.socket_dobot.recv(remaining)
    if not chunk:
        break
    data += chunk
a = np.frombuffer(data, dtype=MyType)
```

Applied automatically at Docker build time (stage 5).

---

## Dobot: Tkinter Thread-Safety

**Bug:** Background feedback thread updates Tkinter widgets directly → segfault on Linux.

**Symptom:** Random segfault when robot feedback is running.

**Fix:**
```python
# WRONG — direct update from background thread
def feed_back(self):
    while True:
        self.label_feed_speed["text"] = ...  # SEGFAULT on Linux

# CORRECT — dispatch to main thread
def feed_back(self):
    while True:
        self.root.after(0, self._update_feed_ui, data)

def _update_feed_ui(self, a):
    """Runs on main thread — safe"""
    self.label_feed_speed["text"] = a["SpeedScaling"][0]
```

**Rule:** Never update Tkinter widgets from any thread other than the main thread. Always use `widget.after(0, callback)`.

---

## numpy >= 2 Breaks PyTorch

**Bug:** numpy 2.x changes C array interface in a way that breaks PyTorch compiled against numpy 1.x.

**Symptom:** `ImportError` or silent wrong results in GraspGen.

**Fix:** Pin `numpy<2` in `requirements.txt` and install it **last** in Dockerfile stage 4 (overrides any dependency that tries to pull numpy 2.x).

```dockerfile
# At the END of pip install stage:
RUN pip install "numpy<2"
```

---

## Orbbec: MJPG Format in ROS2

**Bug:** Default color format `MJPG` is not supported in the ROS2 image callback.

**Symptom:** No color images published, or callback errors.

**Fix:**
```bash
ros2 launch orbbec_camera gemini2.launch.py color_format:=RGB
```

---

## Orbbec: PyPI Wheel Bug

**Bug:** `pyorbbecsdk` v1.3.2 on PyPI installs a macOS `.so` file on Linux (packaging error).

**Symptom:** `ImportError: ... .so: invalid ELF header`

**Fix:** Build `pyorbbecsdk` from source in Dockerfile stage 7 instead of using PyPI wheel.

---

## Orbbec: Stderr Spam

**Bug:** OrbbecSDK floods stderr with C-level debug messages, making logs unreadable.

**Fix:** Import `orbbec_quiet` **before** `pyorbbecsdk`:
```python
import sys
sys.path.insert(0, "/ros2_ws/app")
import orbbec_quiet  # redirects C stderr → /tmp/orbbec_sdk.log
import pyorbbecsdk
```

---

## OnRobot RG: Silent Failure via URScript

**Bug:** `rg_grip()` called from secondary URScript port fails silently — the URCap Java process only services the active program context.

**Symptom:** Gripper doesn't move, no error reported.

**Fix:** Use UR Dashboard to run saved `.urp` programs (`gripper_open.urp`, `gripper_close.urp`) instead of direct URScript calls. See [[Components/Tools/OnRobot RG|OnRobot RG]].

---

## UR10: RTDE Register Conflicts

**Note:** All RTDE registers used by the UR10 driver are within user-accessible ranges (int registers 18–19, double registers 18–25, output int registers 12–13). Avoid using these ranges in other programs running on the controller.

---

## Links
- [[Components/Robots/Dobot CR|Dobot CR]] — robot affected by TCP/Tkinter fixes
- [[Components/Tools/OnRobot RG|OnRobot RG]] — gripper affected by URScript issue
- [[Components/Camera - Orbbec Gemini 2|Camera]] — Orbbec quirks
- [[Components/GraspGen|GraspGen]] — numpy constraint
- [[Infrastructure/Docker Setup|Docker Setup]] — patches applied at build time
