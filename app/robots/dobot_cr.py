"""
Dobot CR-series robot driver (TCP/IP protocol).

Wraps ``dobot_api.DobotApiDashboard`` (port 29999) and
``dobot_api.DobotApiFeedBack`` (port 30004).
"""

import re
import threading
import time

from robots.base import RobotBase

# ── Try to import Dobot API ──────────────────────────────────────────────
try:
    import sys
    sys.path.insert(0, "/opt/Dobot_hv")
    from dobot_api import DobotApiDashboard as _DobotApiDashboard
    from dobot_api import DobotApiFeedBack  as _DobotApiFeedBack
    _DOBOT_API_OK = True
except ImportError as _e:
    _DOBOT_API_OK = False

# ── Defaults ─────────────────────────────────────────────────────────────
DEFAULT_PORT     = 29999
FEEDBACK_PORT    = 30004
VACUUM_DO_PORT   = 1   # Dobot digital output port for vacuum (1-based)


def _parse_result_id(resp):
    """Parse dashboard response -> [ErrorID, CommandID, ...]"""
    if resp is None:
        return [2]
    if "Not Tcp" in str(resp):
        return [1]
    nums = re.findall(r"-?\d+", str(resp))
    return [int(n) for n in nums] if nums else [2]


class DobotCR(RobotBase):
    """Dobot CR-series driver (CR3 / CR5 / CR10 / CR16)."""

    MODE_RUNNING = 7
    MODE_ERROR   = 9
    MODE_ENABLED = 5

    def __init__(self, ip: str, port: int = DEFAULT_PORT, **kwargs):
        if not _DOBOT_API_OK:
            raise ImportError("dobot_api not available — install Dobot TCP-IP-CR-Python-V4")
        self._ip = ip
        self._dashboard = _DobotApiDashboard(ip, port)
        self._feed      = _DobotApiFeedBack(ip, FEEDBACK_PORT)
        self._lock      = threading.Lock()
        self._mode      = -1
        self._cmd_id    = -1
        self._speed     = 20
        self._feed_running = True
        threading.Thread(target=self._feed_loop, daemon=True).start()

    # ── Feedback loop (250 Hz) ───────────────────────────────────────────
    def _feed_loop(self):
        while self._feed_running:
            try:
                info = self._feed.feedBackData()
                if info is not None and hex(int(info["TestValue"][0])) == "0x123456789abcdef":
                    with self._lock:
                        self._mode   = int(info["RobotMode"][0])
                        self._cmd_id = int(info["CurrentCommandId"][0])
            except Exception:
                pass
            time.sleep(0.004)

    # ── Lifecycle ────────────────────────────────────────────────────────
    def enable(self):
        return self._dashboard.EnableRobot()

    def power_on(self):
        return self._dashboard.PowerOn()

    def clear_error(self):
        return self._dashboard.ClearError()

    def stop(self):
        try:
            return self._dashboard.StopRobot()
        except Exception as e:
            print(f"[DobotCR] stop() error (ignored): {e}", flush=True)

    def close(self):
        self._feed_running = False
        # dobot_api.py prints "Error while closing socket: [Errno 9]" internally
        # when closing already-dead connections — suppress stdout for these calls.
        import io, contextlib
        _sink = io.StringIO()
        with contextlib.redirect_stdout(_sink):
            try:
                self._dashboard.close()
            except Exception:
                pass
            try:
                self._feed.close()
            except Exception:
                pass

    # ── Status ───────────────────────────────────────────────────────────
    def get_mode(self) -> int:
        with self._lock:
            return self._mode

    def _get_cmd_id(self):
        with self._lock:
            return self._cmd_id

    def _nums(self, resp):
        return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(resp))]

    def get_pose(self) -> tuple:
        nums = self._nums(self._dashboard.GetPose())
        if len(nums) >= 7:
            return tuple(nums[1:7])
        if len(nums) >= 6:
            return tuple(nums[:6])
        raise ValueError("Cannot parse pose from GetPose response")

    def get_angle(self) -> tuple:
        resp = self._dashboard.GetAngle()
        nums = self._nums(resp)
        if len(nums) >= 7:
            return tuple(nums[1:7])
        if len(nums) >= 6:
            return tuple(nums[:6])
        raise ValueError(f"Cannot parse joints — raw: {resp!r}")

    def get_error_id(self) -> str:
        """Return the raw GetErrorID response string for diagnostics."""
        try:
            return str(self._dashboard.GetErrorID())
        except Exception as e:
            return f"(GetErrorID failed: {e})"

    # ── Reachability ─────────────────────────────────────────────────────
    def check_reachability(self, x, y, z, rx, ry, rz):
        # Dobot InverseKin returns ErrorID=-1 for negative Z even when valid
        # (robot mounted above working surface).  Use XY radius as primary check.
        r = float((x ** 2 + y ** 2) ** 0.5)
        if r < 100.0:
            return False, None, f"Too close to base (r={r:.1f} < 100 mm)"
        if r > 820.0:
            return False, None, f"Beyond max reach (r={r:.1f} > 820 mm)"
        try:
            resp = self._dashboard.InverseKin(x, y, z, rx, ry, rz)
            nums = self._nums(resp)
            if not nums:
                return False, None, f"No response: {resp!r}"
            err_id = int(nums[0])
            if err_id != 0:
                return False, None, f"Out of workspace (ErrorID={err_id})"
            joints = tuple(nums[1:7]) if len(nums) >= 7 else None
            return True, joints, "OK"
        except Exception as e:
            return False, None, f"InverseKin error: {e}"

    # ── Motion ───────────────────────────────────────────────────────────
    def set_speed(self, percent: int):
        self._speed = max(1, min(100, int(percent)))
        return self._dashboard.SpeedFactor(self._speed)

    def _send_motion(self, resp):
        if resp is None or resp == b'' or resp == '':
            try:
                resp = self._dashboard.wait_reply()
            except Exception:
                pass
        parsed = _parse_result_id(resp)
        if len(parsed) < 2 or parsed[0] != 0:
            raise RuntimeError(f"Move rejected (ErrorID={parsed[0] if parsed else '?'}): {resp!r}")
        return parsed[1]

    def set_tool(self, index: int):
        """Select active tool TCP by index (0 = base flange, 1+ = user-defined)."""
        return self._dashboard.Tool(index)

    def move_linear(self, x, y, z, rx, ry, rz) -> int:
        resp = self._dashboard.MovL(
            x, y, z, rx, ry, rz,
            0,                  # coordinateMode=0 → Cartesian
            a=self._speed, v=self._speed
        )
        return self._send_motion(resp)

    def move_joint(self, x, y, z, rx, ry, rz) -> int:
        """Cartesian-target joint-space move (MovJ coordinateMode=0)."""
        resp = self._dashboard.MovJ(
            x, y, z, rx, ry, rz,
            0,                  # coordinateMode=0 → Cartesian
            a=self._speed, v=self._speed
        )
        return self._send_motion(resp)

    def move_joint_angles(self, j1, j2, j3, j4, j5, j6) -> int:
        def _wrap(a):
            a = a % 360.0
            if a > 180.0:
                a -= 360.0
            return a
        j1, j2, j3, j4, j5, j6 = (_wrap(a) for a in (j1, j2, j3, j4, j5, j6))
        resp = self._dashboard.MovJ(
            j1, j2, j3, j4, j5, j6,
            1,                  # coordinateMode=1 → joint angles
            a=self._speed, v=self._speed
        )
        return self._send_motion(resp)

    def move_joint_angles_nearest(self, j1, j2, j3, j4, j5, j6) -> int:
        """MovJ to joint-space target, normalising each axis to minimise travel.

        ``move_joint_angles`` wraps each angle to [-180, 180] independently,
        which can cause a ~360° spin when saved angles straddle the ±180°
        boundary (e.g. stored as +183° → wraps to -177°, while the next pose
        is stored as +177° → stays +177°, giving a 354° J1 rotation for what
        is effectively a tiny move).

        This method instead picks the equivalent angle closest to the current
        joint position, guaranteeing the shortest arc per axis.
        """
        current = self.get_angle()
        targets = (j1, j2, j3, j4, j5, j6)
        nearest = []
        for cur, tgt in zip(current, targets):
            diff = ((tgt - cur) + 180.0) % 360.0 - 180.0
            nearest.append(cur + diff)
        fmt = lambda v: f"{v:+.2f}"
        print(f"[DobotCR] nearest_angles  cur={[fmt(v) for v in current]}", flush=True)
        print(f"[DobotCR] nearest_angles  tgt={[fmt(v) for v in targets]}", flush=True)
        print(f"[DobotCR] nearest_angles  →  {[fmt(v) for v in nearest]}", flush=True)
        resp = self._dashboard.MovJ(
            nearest[0], nearest[1], nearest[2],
            nearest[3], nearest[4], nearest[5],
            1,          # coordinateMode=1 → joint angles
            a=self._speed, v=self._speed
        )
        return self._send_motion(resp)

    def move_joint_nearest(self, x, y, z, rx, ry, rz) -> int:
        """MovJ to a Cartesian target, choosing the IK solution that minimises
        joint travel from the current configuration.

        The Dobot IK solver is deterministic and ignores the current joint state,
        so it can return J1=+170° when the robot is at J1=-170°, causing a 340°
        spin for a tiny Cartesian displacement.  This method:
          1. Reads current joint angles.
          2. Calls InverseKin for the target.
          3. Normalises each IK joint to the nearest equivalent angle relative to
             the current joints (shortest-path per axis).
          4. Executes MovJ in joint-angle mode with the normalised angles.
        Falls back to a standard Cartesian MovJ if InverseKin fails.
        """
        ok, ik_joints, _ = self.check_reachability(x, y, z, rx, ry, rz)
        if not ok or ik_joints is None:
            return self.move_joint(x, y, z, rx, ry, rz)

        current = self.get_angle()

        # For each axis, pick the equivalent angle closest to current position.
        # diff is in (-180, +180], so cur+diff is always the short-arc target.
        nearest = []
        for cur, tgt in zip(current, ik_joints):
            diff = ((tgt - cur) + 180.0) % 360.0 - 180.0
            nearest.append(cur + diff)

        resp = self._dashboard.MovJ(
            nearest[0], nearest[1], nearest[2],
            nearest[3], nearest[4], nearest[5],
            1,          # coordinateMode=1 → joint angles
            a=self._speed, v=self._speed
        )
        return self._send_motion(resp)

    def wait_motion(self, cmd_id: int, timeout: float = 90.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                mode   = self._mode
                cur_id = self._cmd_id
            if cur_id > cmd_id or (mode == self.MODE_ENABLED and cur_id == cmd_id):
                return True
            if mode == self.MODE_ERROR:
                err = self.get_error_id()
                raise RuntimeError(f"Robot entered error state during motion — GetErrorID: {err}")
            time.sleep(0.1)
        raise TimeoutError(f"Motion timeout after {timeout:.0f}s (waiting for cmd {cmd_id})")

    def wait_idle(self, timeout: float = 90.0) -> bool:
        t0 = time.time()
        time.sleep(0.4)
        while time.time() - t0 < timeout:
            with self._lock:
                mode = self._mode
            if mode != self.MODE_RUNNING:
                return True
            time.sleep(0.15)
        return False

    # ── End-effector ─────────────────────────────────────────────────────
    def vacuum_on(self, port: int = VACUUM_DO_PORT):
        resp = self._dashboard.ToolDO(port, 1)
        print(f"[DobotCR] vacuum_on  ToolDO({port},1) → {resp!r}", flush=True)
        return resp

    def vacuum_off(self, port: int = VACUUM_DO_PORT):
        resp = self._dashboard.ToolDO(port, 0)
        print(f"[DobotCR] vacuum_off ToolDO({port},0) → {resp!r}", flush=True)
        return resp
