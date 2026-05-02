"""
Universal Robots UR10 driver via ``ur_rtde``.

Uses RTDEControlInterface for motion commands (moveL, moveJ, teachMode) and
the Dashboard Server (port 29999) for lifecycle commands and gripper programs.

Gripper control uses saved .urp programs on the teach pendant:
  - gripper_close.urp  (RG6 node: width=0, force=40)
  - gripper_open.urp   (RG6 node: width=110, force=40)
The driver stops the control script, loads/plays the gripper program, waits,
then reconnects rtde_control automatically.

Install:  pip install ur_rtde
Docs:     https://sdurobotics.gitlab.io/ur_rtde/
"""

import math
import socket
import threading
import time

from robots.base import RobotBase

try:
    import rtde_control
    import rtde_receive
    _UR_RTDE_OK = True
except ImportError:
    _UR_RTDE_OK = False

DASHBOARD_PORT      = 29999
DEFAULT_ACCEL       = 0.5      # m/s^2
DEFAULT_VEL         = 0.25     # m/s
DEFAULT_JOINT_VEL   = 1.05     # rad/s
DEFAULT_JOINT_ACCEL = 1.4      # rad/s^2

GRIPPER_CLOSE_PROG  = "gripper_close"
GRIPPER_OPEN_PROG   = "gripper_open"
GRIPPER_TIMEOUT     = 15.0     # s
GRIPPER_SETTLE_S    = 1.5      # s after gripper program finishes


class UR10(RobotBase):
    """Universal Robots driver (UR3/5/10/10e/16e/20/30) via ur_rtde."""

    MODE_RUNNING = 1
    MODE_ERROR   = 9
    MODE_ENABLED = 0

    def __init__(self, ip: str, **kwargs):
        if not _UR_RTDE_OK:
            raise ImportError("ur_rtde not available — pip install ur_rtde")
        self._ip = ip
        self._speed_pct = 20
        self._lock = threading.Lock()
        self._cmd_counter = 0
        self._freedrive_active = False

        # ── 1. Connect RTDE control + receive ────────────────────────────
        print(f"[UR10] Connecting rtde_control to {ip}…", flush=True)
        try:
            self._rtde_c = rtde_control.RTDEControlInterface(ip)
        except Exception as e:
            raise ConnectionError(
                f"[UR10] rtde_control failed: {e}\n"
                "  Possible causes:\n"
                "  - Robot not in Remote Control mode (check teach pendant)\n"
                "  - Robot powered off or E-stopped\n"
                "  - Another RTDE client already connected\n"
                "  - Network unreachable"
            ) from e
        print("[UR10] rtde_control connected.", flush=True)

        print(f"[UR10] Connecting rtde_receive to {ip}…", flush=True)
        try:
            self._rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        except Exception as e:
            self._rtde_c.disconnect()
            raise ConnectionError(
                f"[UR10] rtde_receive failed: {e}") from e
        print("[UR10] rtde_receive connected.", flush=True)

        # ── 2. Dashboard socket (for power, enable, gripper programs) ────
        self._dash_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._dash_sock.settimeout(5.0)
        try:
            self._dash_sock.connect((ip, DASHBOARD_PORT))
        except OSError as e:
            self._rtde_c.disconnect()
            self._rtde_r.disconnect()
            raise ConnectionError(
                f"[UR10] Dashboard connect failed ({ip}:{DASHBOARD_PORT}): {e}"
            ) from e
        # consume welcome banner
        self._dash_recv_line()
        print("[UR10] Dashboard connected.", flush=True)

        # ── 3. Verify robot state ───────────────────────────────────────
        mode = self._rtde_r.getRobotMode()
        safety = self._rtde_r.getSafetyMode()
        print(f"[UR10] robotMode={mode}  safetyMode={safety}", flush=True)
        if safety in (3, 6, 7, 8, 9, 10, 11):
            print("[UR10] WARNING: Robot in safety/protective stop — "
                  "call clear_error() + enable() before moving.", flush=True)

        print("[UR10] Ready.", flush=True)

    # ── Dashboard helpers ────────────────────────────────────────────────
    def _dash_recv_line(self) -> "str | None":
        """Read one \\n-terminated response. Returns None if socket closed."""
        data = b""
        while b"\n" not in data:
            try:
                chunk = self._dash_sock.recv(1024)
            except socket.timeout:
                break
            if not chunk:
                return None
            data += chunk
        return data.decode(errors="replace").strip()

    def _reconnect_dashboard(self):
        """Re-open dashboard socket. Call with self._lock held."""
        print("[UR10] Dashboard reconnecting…", flush=True)
        try:
            self._dash_sock.close()
        except Exception:
            pass
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((self._ip, DASHBOARD_PORT))
        self._dash_sock = s
        self._dash_recv_line()  # consume banner
        print("[UR10] Dashboard reconnected.", flush=True)

    def _dash_cmd(self, cmd: str) -> str:
        """Send dashboard command, return response. Auto-reconnects on drop."""
        with self._lock:
            try:
                self._dash_sock.sendall((cmd + "\n").encode())
            except OSError:
                self._reconnect_dashboard()
                self._dash_sock.sendall((cmd + "\n").encode())
            resp = self._dash_recv_line()
            if resp is None:
                self._reconnect_dashboard()
                self._dash_sock.sendall((cmd + "\n").encode())
                resp = self._dash_recv_line()
            return resp if resp is not None else ""

    # ── Helpers ──────────────────────────────────────────────────────────
    def _next_cmd_id(self) -> int:
        self._cmd_counter += 1
        return self._cmd_counter

    def _linear_vel(self) -> float:
        return DEFAULT_VEL * (self._speed_pct / 100.0)

    def _linear_acc(self) -> float:
        return DEFAULT_ACCEL * (self._speed_pct / 100.0)

    def _joint_vel(self) -> float:
        return DEFAULT_JOINT_VEL * (self._speed_pct / 100.0)

    def _joint_acc(self) -> float:
        return DEFAULT_JOINT_ACCEL * (self._speed_pct / 100.0)

    # ── Lifecycle ────────────────────────────────────────────────────────
    def power_on(self):
        self._dash_cmd("power on")
        time.sleep(2.0)
        return self._dash_cmd("brake release")

    def enable(self):
        resp = self._dash_cmd("brake release")
        time.sleep(2.5)  # wait for robot to reach running mode before reuploading
        try:
            self._rtde_c.reuploadScript()
            print("[UR10] Control script re-uploaded after enable.", flush=True)
        except Exception as e:
            print(f"[UR10] reuploadScript failed ({e}) — full rtde_control reconnect…",
                  flush=True)
            try:
                self._rtde_c.disconnect()
            except Exception:
                pass
            time.sleep(0.5)
            self._rtde_c = rtde_control.RTDEControlInterface(self._ip)
        return resp

    def clear_error(self):
        resp = self._dash_cmd("close safety popup")
        self._dash_cmd("unlock protective stop")
        return resp

    def stop(self):
        try:
            self._rtde_c.stopJ(2.0)
        except Exception as e:
            print(f"[UR10] stop() error (ignored): {e}", flush=True)

    def close(self):
        for fn in [
            lambda: self._rtde_c.stopScript(),
            lambda: self._rtde_c.disconnect(),
            lambda: self._rtde_r.disconnect(),
            lambda: self._dash_sock.close(),
        ]:
            try:
                fn()
            except Exception:
                pass

    # ── Status ───────────────────────────────────────────────────────────
    def get_mode(self) -> int:
        try:
            safety = self._rtde_r.getSafetyMode()
            if safety in (3, 6, 7, 8, 9, 10, 11):
                return self.MODE_ERROR
            return self.MODE_RUNNING if self._rtde_r.getRobotMode() == 7 else self.MODE_ENABLED
        except Exception:
            return self.MODE_ERROR

    def get_pose(self) -> tuple:
        """Return (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg) ZYX Euler."""
        pose = self._rtde_r.getActualTCPPose()
        x_mm = pose[0] * 1000.0
        y_mm = pose[1] * 1000.0
        z_mm = pose[2] * 1000.0
        # axis-angle (Rodrigues) → rotation matrix → ZYX Euler
        ax, ay, az = pose[3], pose[4], pose[5]
        angle = math.sqrt(ax*ax + ay*ay + az*az)
        if angle < 1e-9:
            return (x_mm, y_mm, z_mm, 0.0, 0.0, 0.0)
        ux, uy, uz = ax/angle, ay/angle, az/angle
        c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
        R00 = t*ux*ux+c;     R01 = t*ux*uy-s*uz;  R02 = t*ux*uz+s*uy
        R10 = t*ux*uy+s*uz;  R11 = t*uy*uy+c;     R12 = t*uy*uz-s*ux
        R20 = t*ux*uz-s*uy;  R21 = t*uy*uz+s*ux;  R22 = t*uz*uz+c
        sy = max(-1.0, min(1.0, -R20))
        ry = math.asin(sy)
        if abs(math.cos(ry)) > 1e-6:
            rx = math.atan2(R21, R22)
            rz = math.atan2(R10, R00)
        else:
            rx = math.atan2(-R12, R11)
            rz = 0.0
        return (x_mm, y_mm, z_mm, math.degrees(rx), math.degrees(ry), math.degrees(rz))

    def get_angle(self) -> tuple:
        """Return joint angles (j1..j6) in degrees."""
        return tuple(math.degrees(a) for a in self._rtde_r.getActualQ())

    # ── Motion ───────────────────────────────────────────────────────────
    def set_speed(self, percent: int):
        self._speed_pct = max(1, min(100, int(percent)))

    def set_tool(self, index: int):
        pass

    def _euler_to_axis_angle(self, rx_deg, ry_deg, rz_deg):
        """ZYX Euler (degrees) → axis-angle (Rodrigues vector)."""
        rxr, ryr, rzr = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
        cx, sx = math.cos(rxr), math.sin(rxr)
        cy, sy = math.cos(ryr), math.sin(ryr)
        cz, sz = math.cos(rzr), math.sin(rzr)
        R00 = cz*cy;  R01 = cz*sy*sx - sz*cx;  R02 = cz*sy*cx + sz*sx
        R10 = sz*cy;  R11 = sz*sy*sx + cz*cx;  R12 = sz*sy*cx - cz*sx
        R20 = -sy;    R21 = cy*sx;              R22 = cy*cx
        cos_a = max(-1.0, min(1.0, (R00 + R11 + R22 - 1.0) / 2.0))
        angle = math.acos(cos_a)
        if angle < 1e-9:
            return 0.0, 0.0, 0.0
        s2 = 2.0 * math.sin(angle)
        return ((R21-R12)/s2*angle, (R02-R20)/s2*angle, (R10-R01)/s2*angle)

    def move_linear(self, x, y, z, rx, ry, rz) -> int:
        """Cartesian linear move (mm, degrees) → async. Returns cmd_id."""
        ax, ay, az = self._euler_to_axis_angle(rx, ry, rz)
        pose = [x/1000.0, y/1000.0, z/1000.0, ax, ay, az]
        cmd_id = self._next_cmd_id()
        self._rtde_c.moveL(pose, self._linear_vel(), self._linear_acc(),
                           asynchronous=True)
        return cmd_id

    def move_joint_angles(self, j1, j2, j3, j4, j5, j6) -> int:
        """Joint-space move (degrees) → async. Returns cmd_id."""
        q = [math.radians(a) for a in (j1, j2, j3, j4, j5, j6)]
        cmd_id = self._next_cmd_id()
        self._rtde_c.moveJ(q, self._joint_vel(), self._joint_acc(),
                           asynchronous=True)
        return cmd_id

    def wait_motion(self, cmd_id: int, timeout: float = 90.0) -> bool:
        """Wait for async motion to complete."""
        t0 = time.time()
        time.sleep(0.1)
        while time.time() - t0 < timeout:
            safety = self._rtde_r.getSafetyMode()
            if safety in (3, 6, 7, 8, 9, 10, 11):
                raise RuntimeError(
                    f"Robot entered protective/safety stop (safetyMode={safety})")
            try:
                if self._rtde_c.isSteady():
                    return True
            except Exception:
                pass
            time.sleep(0.05)
        raise TimeoutError(f"Motion timeout after {timeout:.0f}s")

    # ── Freedrive ────────────────────────────────────────────────────────
    def freedrive_start(self):
        """Enable freedrive (hand-guide) mode."""
        self._rtde_c.teachMode()
        self._freedrive_active = True

    def freedrive_stop(self):
        """Disable freedrive mode."""
        self._rtde_c.endTeachMode()
        self._freedrive_active = False

    # ── Gripper (via Dashboard programs) ─────────────────────────────────
    def _run_gripper_program(self, program_name: str):
        """Stop control script, run a saved .urp gripper program, then
        reconnect rtde_control.

        Requires programs saved on the teach pendant:
          /programs/gripper_close.urp
          /programs/gripper_open.urp
        """
        # 1. Stop rtde_control's script so Dashboard can load a program
        print(f"[UR10] Gripper: stopping control script…", flush=True)
        try:
            self._rtde_c.stopScript()
        except Exception:
            pass
        time.sleep(0.3)

        # 2. Load and play the gripper program
        path = f"/programs/{program_name}.urp"
        resp = self._dash_cmd(f"load {path}")
        if "error" in resp.lower() or "file not found" in resp.lower():
            print(f"[UR10] WARNING: load {path} failed: {resp!r}", flush=True)
            print("[UR10] Re-uploading control script…", flush=True)
            self._rtde_c.reuploadScript()
            raise RuntimeError(
                f"Gripper program '{program_name}.urp' not found on robot.\n"
                "Create it on the teach pendant: New Program → RG6 Grip node → Save."
            )
        print(f"[UR10] load → {resp!r}", flush=True)

        resp = self._dash_cmd("play")
        print(f"[UR10] play → {resp!r}", flush=True)

        # 3. Wait for program to finish
        t0 = time.time()
        while time.time() - t0 < GRIPPER_TIMEOUT:
            state = self._dash_cmd("programState").lower()
            if "stopped" in state or "idle" in state or "paused" in state:
                break
            time.sleep(0.15)
        else:
            print(f"[UR10] WARNING: gripper program did not finish within {GRIPPER_TIMEOUT}s",
                  flush=True)

        # 4. Settle time for gripper to physically finish
        time.sleep(GRIPPER_SETTLE_S)

        # 5. Reconnect rtde_control (re-uploads its script)
        print("[UR10] Re-uploading control script…", flush=True)
        try:
            self._rtde_c.reuploadScript()
        except Exception:
            # Full reconnect as fallback
            try:
                self._rtde_c.disconnect()
            except Exception:
                pass
            self._rtde_c = rtde_control.RTDEControlInterface(self._ip)
        print("[UR10] Control script restored.", flush=True)

    def vacuum_on(self, port: int = 0):
        """Close gripper (run gripper_close.urp)."""
        # If a tool is attached, delegate to it instead
        if self._tool is not None:
            self._tool.grasp()
            return
        self._run_gripper_program(GRIPPER_CLOSE_PROG)
        print("[UR10] Gripper closed.", flush=True)

    def vacuum_off(self, port: int = 0):
        """Open gripper (run gripper_open.urp)."""
        if self._tool is not None:
            self._tool.release()
            return
        self._run_gripper_program(GRIPPER_OPEN_PROG)
        print("[UR10] Gripper opened.", flush=True)

    # ── Reachability ─────────────────────────────────────────────────────
    def check_reachability(self, x, y, z, rx, ry, rz):
        """IK + safety-limit check via ur_rtde."""
        ax, ay, az = self._euler_to_axis_angle(rx, ry, rz)
        pose = [x/1000.0, y/1000.0, z/1000.0, ax, ay, az]
        try:
            # Safety planes / speed limits configured on the pendant
            if not self._rtde_r.isPoseWithinSafetyLimits(pose):
                return False, None, "Outside pendant safety limits"
            # Kinematic reachability — returns [] on failure
            result = self._rtde_c.getInverseKinematics(pose)
            if not result or len(result) != 6:
                return False, None, "IK returned no solution"
            if any(not math.isfinite(j) for j in result):
                return False, None, "IK returned non-finite joints"
            joints_deg = tuple(math.degrees(a) for a in result)
            return True, joints_deg, "OK"
        except Exception as e:
            return False, None, f"IK error: {e}"
