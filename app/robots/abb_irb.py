"""
ABB IRB driver (IRC5 controller) via Robot Web Services (RWS).

Two libraries cooperate:
  - ``abb_motion_program_exec``  → MoveL / MoveJ / MoveAbsJ execution
  - ``abb_robot_client.rws``     → state polling, digital IO, motors on/off

Controller prerequisites (one-time setup on the IRC5):
  1. RobotWare 6.xx with RWS enabled (default).
  2. Load the ``abb_motion_program_exec`` RAPID module onto the controller
     (https://github.com/rpiRobotics/abb_motion_program_exec, see
     ``abb_motion_program_exec_RAPID/`` directory).
  3. Configure a digital-output signal named ``DO_VACUUM`` mapped to the
     vacuum-solenoid relay (override the name via the ``do_vacuum_signal``
     kwarg if your signal is named differently).
  4. AUTO mode + motors on for automatic runs. MANUAL mode is fine for
     hand-eye calibration's "Manual (I move robot)" workflow — pose reads
     still work, only motion commands are gated by the controller key-switch.

Install:  pip install abb-motion-program-exec
"""

import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutTimeout

from robots.base import RobotBase

try:
    import abb_motion_program_exec as abb
    from abb_robot_client.rws import RWS
    _ABB_OK = True
except ImportError:
    _ABB_OK = False

try:
    from scipy.spatial.transform import Rotation as _Rotation
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


DEFAULT_USERNAME   = "Default User"
DEFAULT_PASSWORD   = "robotics"
DEFAULT_DO_VACUUM  = "DoValve1"
DEFAULT_TOOL       = "tool0"
DEFAULT_WOBJ       = "wobj0"
DEFAULT_MECHUNIT   = "ROB_1"
MAX_TCP_SPEED_MMPS = 1000.0   # at speed=100%
MAX_ORI_SPEED_DEGS = 500.0    # at speed=100%


def _abb_quat_to_zyx_euler_deg(w, x, y, z):
    """ABB quaternion (w, x, y, z) → intrinsic ZYX Euler (rx, ry, rz) in degrees."""
    if _SCIPY_OK:
        rz, ry, rx = _Rotation.from_quat([x, y, z, w]).as_euler("ZYX", degrees=True)
        return float(rx), float(ry), float(rz)
    # Fallback: direct conversion (matches UR10's ZYX convention).
    R00 = 1 - 2 * (y * y + z * z)
    R10 = 2 * (x * y + z * w)
    R11 = 1 - 2 * (x * x + z * z)
    R12 = 2 * (y * z - x * w)
    R20 = 2 * (x * z - y * w)
    R21 = 2 * (y * z + x * w)
    R22 = 1 - 2 * (x * x + y * y)
    sy = max(-1.0, min(1.0, -R20))
    ry = math.asin(sy)
    if abs(math.cos(ry)) > 1e-6:
        rx = math.atan2(R21, R22)
        rz = math.atan2(R10, R00)
    else:
        rx = math.atan2(-R12, R11)
        rz = 0.0
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def _zyx_euler_deg_to_abb_quat(rx_deg, ry_deg, rz_deg):
    """Intrinsic ZYX Euler (rx, ry, rz) degrees → ABB quaternion (w, x, y, z)."""
    if _SCIPY_OK:
        x, y, z, w = _Rotation.from_euler("ZYX", [rz_deg, ry_deg, rx_deg],
                                          degrees=True).as_quat()
        return (float(w), float(x), float(y), float(z))
    rxr, ryr, rzr = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(rxr / 2), math.sin(rxr / 2)
    cy, sy = math.cos(ryr / 2), math.sin(ryr / 2)
    cz, sz = math.cos(rzr / 2), math.sin(rzr / 2)
    w = cz * cy * cx + sz * sy * sx
    x = cz * cy * sx - sz * sy * cx
    y = cz * sy * cx + sz * cy * sx
    z = sz * cy * cx - cz * sy * sx
    return (w, x, y, z)


class ABBIRB(RobotBase):
    """ABB IRB driver (IRC5 + RobotWare 6.xx) via RWS."""

    # Internal mode codes (mapped from ABB controller state strings in get_mode()).
    MODE_RUNNING = 1
    MODE_ERROR   = 2
    MODE_ENABLED = 0

    def __init__(self, ip: str, *,
                 username: str = DEFAULT_USERNAME,
                 password: str = DEFAULT_PASSWORD,
                 do_vacuum_signal: str = DEFAULT_DO_VACUUM,
                 tool: str = DEFAULT_TOOL,
                 wobj: str = DEFAULT_WOBJ,
                 mechunit: str = DEFAULT_MECHUNIT,
                 **kwargs):
        if not _ABB_OK:
            raise ImportError(
                "abb_motion_program_exec / abb_robot_client not available — "
                "pip install abb-motion-program-exec")
        self._ip          = ip
        self._tool_name   = tool
        self._wobj_name   = wobj
        self._mechunit    = mechunit
        self._do_vacuum   = do_vacuum_signal
        self._speed_pct   = 20
        self._lock        = threading.Lock()
        self._cmd_counter = 0
        self._futures: dict[int, Future] = {}

        base_url = f"http://{ip}"
        print(f"[ABBIRB] Connecting to {base_url} (user={username!r})…", flush=True)
        try:
            self._mp = abb.MotionProgramExecClient(
                base_url=base_url, username=username, password=password)
            self._rws = RWS(base_url=base_url, username=username, password=password)
            _ = self._rws.get_controller_state()  # fail fast on bad connection
        except Exception as e:
            raise ConnectionError(
                f"[ABBIRB] connection failed: {e}\n"
                "  Possible causes:\n"
                "  - IP unreachable or wrong\n"
                "  - RWS disabled on controller\n"
                "  - Wrong username/password (defaults: 'Default User' / 'robotics')\n"
                "  - abb_motion_program_exec RAPID module not loaded on controller"
            ) from e
        print(f"[ABBIRB] Connected. controller_state={self._rws.get_controller_state()!r}",
              flush=True)

        # Single worker serialises motion-program execution (the RAPID server
        # only runs one motion program at a time anyway).
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="abb-motion")

    # ── Lifecycle ────────────────────────────────────────────────────────
    def get_controller_state_raw(self) -> str:
        """Return raw controller state string (e.g. 'motoron', 'emergencystop')."""
        try:
            return str(self._rws.get_controller_state()).lower()
        except Exception:
            return "unknown"

    def is_auto_mode(self) -> bool:
        """Return True if the key-switch is in AUTO (motors-on commands allowed)."""
        return self.get_operation_mode().upper().startswith("AUTO")

    def is_emergency_stopped(self) -> bool:
        """Return True if the controller is in an emergency-stop state."""
        return "emergency" in self.get_controller_state_raw()

    def is_guard_stopped(self) -> bool:
        """Return True if the controller is in a guard stop (safety latch requires
        physical Motors ON / Reset on the FlexPendant to clear)."""
        return "guardstop" in self.get_controller_state_raw()

    def _motoron_blocked(self) -> bool:
        """Return True if set_controller_state('motoron') would be rejected by RWS."""
        state = self.get_controller_state_raw()
        return ("emergency" in state or "guardstop" in state
                or "sysfail" in state or not self.is_auto_mode())

    def enable(self):
        """Set controller motors on. No-op (with log) when motoron would be rejected."""
        if self._motoron_blocked():
            state = self.get_controller_state_raw()
            op = self.get_operation_mode()
            print(f"[ABBIRB] enable skipped — state={state!r} mode={op!r} "
                  "(requires AUTO + no guard/emergency stop)", flush=True)
            return
        return self._rws.set_controller_state("motoron")

    def power_on(self):
        """IRC5 has no separate power button — motors-on + reset PP.
        Motors-on is skipped when the controller would reject it."""
        if self._motoron_blocked():
            state = self.get_controller_state_raw()
            op = self.get_operation_mode()
            print(f"[ABBIRB] power_on motoron skipped — state={state!r} mode={op!r}",
                  flush=True)
        else:
            try:
                self._rws.set_controller_state("motoron")
            except Exception as e:
                print(f"[ABBIRB] power_on motoron failed: {e}", flush=True)
        try:
            self._rws.resetpp()
        except Exception as e:
            print(f"[ABBIRB] power_on resetpp failed (ignored): {e}", flush=True)

    def clear_error(self):
        """Reset program pointer + motors on.
        Motors-on is skipped when the controller would reject it."""
        try:
            self._rws.resetpp()
        except Exception as e:
            print(f"[ABBIRB] clear_error resetpp failed: {e}", flush=True)
        if self._motoron_blocked():
            state = self.get_controller_state_raw()
            op = self.get_operation_mode()
            print(f"[ABBIRB] clear_error motoron skipped — state={state!r} mode={op!r}",
                  flush=True)
            return
        try:
            self._rws.set_controller_state("motoron")
        except Exception as e:
            print(f"[ABBIRB] clear_error motoron failed: {e}", flush=True)

    def stop(self):
        try:
            return self._rws.stop()
        except Exception as e:
            print(f"[ABBIRB] stop() error (ignored): {e}", flush=True)

    def close(self):
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    # ── Status ───────────────────────────────────────────────────────────
    def get_mode(self) -> int:
        try:
            ctrl = str(self._rws.get_controller_state()).lower()
        except Exception:
            return self.MODE_ERROR
        if any(k in ctrl for k in ("emergency", "guard", "sysfail", "fail")):
            return self.MODE_ERROR
        try:
            exec_state = self._rws.get_execution_state()
            # RAPIDExecutionState is a named-tuple-like; .ctrlexecstate is the field.
            running = str(getattr(exec_state, "ctrlexecstate", exec_state)).lower() == "running"
        except Exception:
            running = False
        if running:
            return self.MODE_RUNNING
        return self.MODE_ENABLED if "motoron" in ctrl else self.MODE_ERROR

    def get_pose(self) -> tuple:
        """Return (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg) ZYX Euler."""
        rt = self._rws.get_robtarget(mechunit=self._mechunit,
                                     tool=self._tool_name,
                                     wobj=self._wobj_name)
        x, y, z = float(rt.trans[0]), float(rt.trans[1]), float(rt.trans[2])
        q = rt.rot  # ABB convention: (w, x, y, z)
        rx, ry, rz = _abb_quat_to_zyx_euler_deg(
            float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        return (x, y, z, rx, ry, rz)

    def get_angle(self) -> tuple:
        """Return joint angles (j1..j6) in degrees."""
        jt = self._rws.get_jointtarget(mechunit=self._mechunit)
        return tuple(float(j) for j in jt.robax)

    def get_operation_mode(self) -> str:
        """Return ABB operation mode string ('AUTO', 'MANR', 'MANF', ...).

        Not part of the RobotBase contract — useful for UI to warn the user
        when the controller key-switch is in MANUAL.
        """
        try:
            return str(self._rws.get_operation_mode())
        except Exception:
            return "UNKNOWN"

    # ── Motion ───────────────────────────────────────────────────────────
    def set_speed(self, percent: int):
        self._speed_pct = max(1, min(100, int(percent)))

    def _speeddata(self):
        scale = self._speed_pct / 100.0
        v_tcp = MAX_TCP_SPEED_MMPS * scale
        v_ori = MAX_ORI_SPEED_DEGS * scale
        return abb.speeddata(v_tcp, v_ori, 5000.0, 1000.0)

    def _next_cmd_id(self) -> int:
        with self._lock:
            self._cmd_counter += 1
            return self._cmd_counter

    def _current_confdata(self):
        """Reuse the current robot configuration so new robtargets don't
        trigger ConfL/ConfJ errors when the IK has multiple solutions."""
        try:
            rt = self._rws.get_robtarget(mechunit=self._mechunit,
                                         tool=self._tool_name,
                                         wobj=self._wobj_name)
            c = rt.robconf
            return abb.confdata(int(c[0]), int(c[1]), int(c[2]), int(c[3]))
        except Exception:
            return abb.confdata(0, 0, 0, 0)

    def _build_robtarget(self, x, y, z, rx, ry, rz):
        w, qx, qy, qz = _zyx_euler_deg_to_abb_quat(rx, ry, rz)
        return abb.robtarget(
            [float(x), float(y), float(z)],
            [w, qx, qy, qz],
            self._current_confdata(),
            [0.0] * 6,
        )

    def _exec_linear(self, x, y, z, rx, ry, rz):
        mp = abb.MotionProgram()
        mp.MoveL(self._build_robtarget(x, y, z, rx, ry, rz),
                 self._speeddata(), abb.fine)
        return self._mp.execute_motion_program(mp)

    def _exec_joint_cart(self, x, y, z, rx, ry, rz):
        mp = abb.MotionProgram()
        mp.MoveJ(self._build_robtarget(x, y, z, rx, ry, rz),
                 self._speeddata(), abb.fine)
        return self._mp.execute_motion_program(mp)

    def _exec_joint_abs(self, j1, j2, j3, j4, j5, j6):
        mp = abb.MotionProgram()
        jt = abb.jointtarget(
            [float(j1), float(j2), float(j3),
             float(j4), float(j5), float(j6)],
            [0.0] * 6,
        )
        mp.MoveAbsJ(jt, self._speeddata(), abb.fine)
        return self._mp.execute_motion_program(mp)

    def move_linear(self, x, y, z, rx, ry, rz) -> int:
        cmd_id = self._next_cmd_id()
        self._futures[cmd_id] = self._executor.submit(
            self._exec_linear, x, y, z, rx, ry, rz)
        return cmd_id

    def move_joint(self, x, y, z, rx, ry, rz) -> int:
        cmd_id = self._next_cmd_id()
        self._futures[cmd_id] = self._executor.submit(
            self._exec_joint_cart, x, y, z, rx, ry, rz)
        return cmd_id

    def move_joint_angles(self, j1, j2, j3, j4, j5, j6) -> int:
        cmd_id = self._next_cmd_id()
        self._futures[cmd_id] = self._executor.submit(
            self._exec_joint_abs, j1, j2, j3, j4, j5, j6)
        return cmd_id

    def wait_motion(self, cmd_id: int, timeout: float = 90.0) -> bool:
        fut = self._futures.get(cmd_id)
        if fut is None:
            raise ValueError(f"[ABBIRB] unknown cmd_id {cmd_id}")
        try:
            fut.result(timeout=timeout)
        except _FutTimeout:
            raise TimeoutError(f"[ABBIRB] motion timeout after {timeout:.0f}s "
                               f"(cmd {cmd_id})")
        except Exception as e:
            raise RuntimeError(f"[ABBIRB] motion failed (cmd {cmd_id}): {e}") from e
        finally:
            self._futures.pop(cmd_id, None)
        return True

    # ── End-effector (DO-controlled vacuum) ──────────────────────────────
    def vacuum_on(self, port: int = 0):
        if self._tool is not None:
            return self._tool.grasp()
        resp = self._rws.set_digital_io(self._do_vacuum, 1)
        print(f"[ABBIRB] vacuum_on  set_digital_io({self._do_vacuum},1)", flush=True)
        return resp

    def vacuum_off(self, port: int = 0):
        if self._tool is not None:
            return self._tool.release()
        resp = self._rws.set_digital_io(self._do_vacuum, 0)
        print(f"[ABBIRB] vacuum_off set_digital_io({self._do_vacuum},0)", flush=True)
        return resp
