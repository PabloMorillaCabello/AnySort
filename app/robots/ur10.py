"""
Universal Robots UR10 driver via ``ur_rtde``.

Uses the ``ur_rtde`` library (pip install ur_rtde) for real-time control.
Falls back to UR Dashboard Server (port 29999) for power/enable/error commands.

Install:  pip install ur_rtde
Docs:     https://sdurobotics.gitlab.io/ur_rtde/

Communication ports:
  - 30004  RTDE (real-time data exchange) — used by ur_rtde
  - 29999  Dashboard Server — power on, brake release, safety status
  - 30003  Real-time client interface (alternative)
"""

import math
import socket
import threading
import time

from robots.base import RobotBase

# ── Try to import ur_rtde ────────────────────────────────────────────────
try:
    import rtde_control
    import rtde_receive
    _UR_RTDE_OK = True
except ImportError:
    _UR_RTDE_OK = False

# ── Constants ────────────────────────────────────────────────────────────
DEFAULT_PORT     = 30004
DASHBOARD_PORT   = 29999
DEFAULT_ACCEL    = 0.5   # m/s²
DEFAULT_VEL      = 0.25  # m/s
DEFAULT_JOINT_VEL   = 1.05  # rad/s
DEFAULT_JOINT_ACCEL = 1.4   # rad/s²


class UR10(RobotBase):
    """Universal Robots UR10 driver (also works with UR5, UR3, UR10e, UR16e, UR20, UR30)."""

    MODE_RUNNING = 1
    MODE_ERROR   = 9    # protective stop / fault
    MODE_ENABLED = 0    # idle / normal

    def __init__(self, ip: str, **kwargs):
        if not _UR_RTDE_OK:
            raise ImportError(
                "ur_rtde not available — pip install ur_rtde")
        self._ip = ip
        self._speed_pct = 20
        self._lock = threading.Lock()
        self._cmd_counter = 0

        # Connect RTDE interfaces
        self._rtde_c = rtde_control.RTDEControlInterface(ip)
        self._rtde_r = rtde_receive.RTDEReceiveInterface(ip)

        # Dashboard socket for power/enable commands
        self._dash_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._dash_sock.settimeout(5.0)
        self._dash_sock.connect((ip, DASHBOARD_PORT))
        self._dash_sock.recv(1024)  # read welcome banner

    # ── Dashboard helpers ────────────────────────────────────────────────
    def _dash_cmd(self, cmd: str) -> str:
        """Send a command to the UR Dashboard Server and return the response."""
        with self._lock:
            self._dash_sock.sendall((cmd + "\n").encode())
            return self._dash_sock.recv(1024).decode().strip()

    def _next_cmd_id(self) -> int:
        self._cmd_counter += 1
        return self._cmd_counter

    # ── Speed helpers ────────────────────────────────────────────────────
    def _linear_vel(self) -> float:
        """Current linear velocity in m/s (scaled from percentage)."""
        return DEFAULT_VEL * (self._speed_pct / 100.0)

    def _joint_vel(self) -> float:
        """Current joint velocity in rad/s (scaled from percentage)."""
        return DEFAULT_JOINT_VEL * (self._speed_pct / 100.0)

    # ── Lifecycle ────────────────────────────────────────────────────────
    def power_on(self):
        self._dash_cmd("power on")
        time.sleep(2.0)
        return self._dash_cmd("brake release")

    def enable(self):
        # UR uses "brake release" to enable
        return self._dash_cmd("brake release")

    def clear_error(self):
        resp = self._dash_cmd("close safety popup")
        self._dash_cmd("unlock protective stop")
        return resp

    def stop(self):
        try:
            self._rtde_c.stopJ(2.0)  # decel = 2 rad/s²
        except Exception as e:
            print(f"[UR10] stop() error (ignored): {e}", flush=True)

    def close(self):
        try:
            self._rtde_c.stopScript()
        except Exception:
            pass
        try:
            self._rtde_c.disconnect()
        except Exception:
            pass
        try:
            self._rtde_r.disconnect()
        except Exception:
            pass
        try:
            self._dash_sock.close()
        except Exception:
            pass

    # ── Status ───────────────────────────────────────────────────────────
    def get_mode(self) -> int:
        try:
            safety = self._rtde_r.getSafetyMode()
            # safety modes: 1=Normal, 2=Reduced, 3=ProtectiveStop,
            #               6=SafeguardStop, 7=SystemEmergencyStop, etc.
            if safety in (3, 6, 7, 8, 9, 10, 11):
                return self.MODE_ERROR

            robot_mode = self._rtde_r.getRobotMode()
            # robot modes: 7=Running, 5=Idle
            if robot_mode == 7:  # RUNNING
                return self.MODE_RUNNING
            return self.MODE_ENABLED
        except Exception:
            return self.MODE_ERROR

    def get_pose(self) -> tuple:
        """Return (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg) in ZYX Euler convention.

        ur_rtde returns TCP pose as [x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]
        where the rotation is an axis-angle (Rodrigues) vector, NOT Euler angles.
        We convert: axis-angle → rotation matrix → ZYX Euler angles (degrees),
        which is the convention expected by the hand-eye calibration solver.
        """
        pose = self._rtde_r.getActualTCPPose()
        x_mm = pose[0] * 1000.0
        y_mm = pose[1] * 1000.0
        z_mm = pose[2] * 1000.0

        # axis-angle → rotation matrix (Rodrigues)
        ax, ay, az = pose[3], pose[4], pose[5]
        angle = math.sqrt(ax*ax + ay*ay + az*az)
        if angle < 1e-9:
            return (x_mm, y_mm, z_mm, 0.0, 0.0, 0.0)
        ux, uy, uz = ax / angle, ay / angle, az / angle
        c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
        R00 = t*ux*ux + c;       R01 = t*ux*uy - s*uz;  R02 = t*ux*uz + s*uy
        R10 = t*ux*uy + s*uz;    R11 = t*uy*uy + c;     R12 = t*uy*uz - s*ux
        R20 = t*ux*uz - s*uy;    R21 = t*uy*uz + s*ux;  R22 = t*uz*uz + c

        # rotation matrix → ZYX Euler (R = Rz @ Ry @ Rx)
        sy = max(-1.0, min(1.0, -R20))
        ry = math.asin(sy)
        if abs(math.cos(ry)) > 1e-6:
            rx = math.atan2(R21, R22)
            rz = math.atan2(R10, R00)
        else:                          # gimbal lock
            rx = math.atan2(-R12, R11)
            rz = 0.0
        return (x_mm, y_mm, z_mm, math.degrees(rx), math.degrees(ry), math.degrees(rz))

    def get_angle(self) -> tuple:
        """Return joint angles (j1..j6) in degrees."""
        q = self._rtde_r.getActualQ()
        return tuple(math.degrees(a) for a in q)

    # ── Motion ───────────────────────────────────────────────────────────
    def set_speed(self, percent: int):
        self._speed_pct = max(1, min(100, int(percent)))

    def set_tool(self, index: int):
        """UR uses setTcp() — but tool selection is typically done via
        the teach pendant or script. No-op here; override if needed."""
        pass

    def move_linear(self, x, y, z, rx, ry, rz) -> int:
        """Cartesian linear move.

        Pipeline provides (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg) ZYX Euler.
        ur_rtde expects (x_m, y_m, z_m, ax, ay, az) axis-angle (Rodrigues vector).
        We convert: ZYX Euler → rotation matrix → axis-angle.
        """
        # ZYX Euler → rotation matrix (R = Rz @ Ry @ Rx)
        rxr, ryr, rzr = math.radians(rx), math.radians(ry), math.radians(rz)
        cx, sx = math.cos(rxr), math.sin(rxr)
        cy, sy = math.cos(ryr), math.sin(ryr)
        cz, sz = math.cos(rzr), math.sin(rzr)
        R00 = cz*cy;  R01 = cz*sy*sx - sz*cx;  R02 = cz*sy*cx + sz*sx
        R10 = sz*cy;  R11 = sz*sy*sx + cz*cx;  R12 = sz*sy*cx - cz*sx
        R20 = -sy;    R21 = cy*sx;              R22 = cy*cx

        # rotation matrix → axis-angle (Rodrigues)
        cos_a = max(-1.0, min(1.0, (R00 + R11 + R22 - 1.0) / 2.0))
        angle = math.acos(cos_a)
        if angle < 1e-9:
            ax = ay = az = 0.0
        else:
            s2 = 2.0 * math.sin(angle)
            ax = (R21 - R12) / s2 * angle
            ay = (R02 - R20) / s2 * angle
            az = (R10 - R01) / s2 * angle

        pose = [x / 1000.0, y / 1000.0, z / 1000.0, ax, ay, az]
        cmd_id = self._next_cmd_id()
        vel = self._linear_vel()
        acc = DEFAULT_ACCEL * (self._speed_pct / 100.0)
        self._rtde_c.moveL(pose, vel, acc, asynchronous=True)
        return cmd_id

    def move_joint_angles(self, j1, j2, j3, j4, j5, j6) -> int:
        """Joint-space move.  Angles in degrees → converted to radians for ur_rtde."""
        q = [math.radians(a) for a in (j1, j2, j3, j4, j5, j6)]
        cmd_id = self._next_cmd_id()
        vel = self._joint_vel()
        acc = DEFAULT_JOINT_ACCEL * (self._speed_pct / 100.0)
        self._rtde_c.moveJ(q, vel, acc, asynchronous=True)
        return cmd_id

    def wait_motion(self, cmd_id: int, timeout: float = 90.0) -> bool:
        """Wait for async motion to complete by polling isSteady()."""
        t0 = time.time()
        time.sleep(0.1)
        while time.time() - t0 < timeout:
            try:
                # Check safety first
                safety = self._rtde_r.getSafetyMode()
                if safety in (3, 6, 7, 8, 9, 10, 11):
                    raise RuntimeError(
                        f"Robot entered protective/safety stop (safetyMode={safety})")

                # Check if motion is done
                if not self._rtde_c.isSteady():
                    time.sleep(0.05)
                    continue
                return True
            except RuntimeError:
                raise
            except Exception:
                time.sleep(0.1)
        raise TimeoutError(f"Motion timeout after {timeout:.0f}s")

    # ── End-effector ─────────────────────────────────────────────────────
    # vacuum_on / vacuum_off are inherited from RobotBase and delegate to
    # the attached tool (e.g. OnRobotGripper).
    # Call robot.attach_tool(tool) before executing a grasp.
    #
    # Example:
    #   from tools.onrobot import OnRobotGripper
    #   robot.attach_tool(OnRobotGripper(ip="192.168.1.1", model="RG2"))

    # ── Freedrive (teach / hand-guide) ───────────────────────────────────
    def freedrive_start(self):
        """Enable freedrive (hand-guide) mode — robot can be moved by hand."""
        self._rtde_c.teachMode()

    def freedrive_stop(self):
        """Disable freedrive mode — robot returns to normal control."""
        self._rtde_c.endTeachMode()

    # ── Reachability ─────────────────────────────────────────────────────
    def check_reachability(self, x, y, z, rx, ry, rz):
        """Use ur_rtde inverse kinematics to check reachability."""
        pose = [
            x / 1000.0, y / 1000.0, z / 1000.0,
            math.radians(rx), math.radians(ry), math.radians(rz),
        ]
        try:
            result = self._rtde_c.getInverseKinematics(pose)
            if result and len(result) == 6:
                joints_deg = tuple(math.degrees(a) for a in result)
                return True, joints_deg, "OK"
            return False, None, "IK returned no solution"
        except Exception as e:
            return False, None, f"IK error: {e}"
