"""
Abstract base class for robot drivers.

Every robot backend must subclass ``RobotBase`` and implement all abstract
methods.  The pipeline calls *only* these methods — no robot-specific code
leaks into the main application.

Adding a new robot
------------------
1. Create ``app/robots/my_robot.py``
2. ``class MyRobot(RobotBase): ...`` — implement every ``@abstractmethod``
3. Register in ``app/robots/__init__.py`` → ``ROBOT_DRIVERS``
"""

from abc import ABC, abstractmethod


class RobotBase(ABC):
    """Interface that every robot driver must implement."""

    # ── Robot mode constants ─────────────────────────────────────────────
    # Subclasses must set these to match the robot controller's state codes.
    MODE_RUNNING: int = -1   # robot is executing a motion
    MODE_ERROR:   int = -2   # robot is in an error / protective-stop state
    MODE_ENABLED: int = -3   # robot is idle and ready to accept commands

    @abstractmethod
    def __init__(self, ip: str, **kwargs):
        """Connect to the robot at *ip*.  Raise on failure."""

    # ── Lifecycle ────────────────────────────────────────────────────────
    @abstractmethod
    def enable(self):
        """Enable the robot (servo on / brake release)."""

    @abstractmethod
    def power_on(self):
        """Power on the controller (if applicable)."""

    @abstractmethod
    def clear_error(self):
        """Clear alarms / protective stops."""

    @abstractmethod
    def stop(self):
        """Emergency-decelerate and stop all motion."""

    @abstractmethod
    def close(self):
        """Disconnect and free resources."""

    # ── Status ───────────────────────────────────────────────────────────
    @abstractmethod
    def get_mode(self) -> int:
        """Return the current robot mode (use the MODE_* constants)."""

    @abstractmethod
    def get_pose(self) -> tuple:
        """Return ``(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg)``."""

    @abstractmethod
    def get_angle(self) -> tuple:
        """Return joint angles ``(j1, j2, ..., j6)`` in degrees."""

    # ── Motion ───────────────────────────────────────────────────────────
    @abstractmethod
    def set_speed(self, percent: int):
        """Set motion speed as a percentage (1–100)."""

    @abstractmethod
    def move_linear(self, x, y, z, rx, ry, rz) -> int:
        """Cartesian linear move.  Returns a *command_id* for ``wait_motion``."""

    @abstractmethod
    def move_joint_angles(self, j1, j2, j3, j4, j5, j6) -> int:
        """Joint-space move.  Returns a *command_id* for ``wait_motion``."""

    def move_joint(self, x, y, z, rx, ry, rz) -> int:
        """Cartesian-target joint-space move (MovJ to Cartesian pose).

        Default: same as ``move_linear``.  Override if robot has a distinct
        joint-interpolated Cartesian command (e.g. Dobot MovJ with coordinateMode=0).
        """
        return self.move_linear(x, y, z, rx, ry, rz)

    def set_tool(self, index: int):
        """Select active tool/TCP by index.  No-op by default."""
        pass

    @abstractmethod
    def wait_motion(self, cmd_id: int, timeout: float = 90.0) -> bool:
        """Block until the motion identified by *cmd_id* finishes.

        Returns ``True`` on success.
        Raises ``RuntimeError`` on robot error, ``TimeoutError`` on timeout.
        """

    # ── End-effector ─────────────────────────────────────────────────────
    @abstractmethod
    def vacuum_on(self, port: int = 0):
        """Activate the vacuum / gripper."""

    @abstractmethod
    def vacuum_off(self, port: int = 0):
        """Deactivate the vacuum / gripper."""

    # ── Optional helpers (default implementations) ───────────────────────
    def wait_idle(self, timeout: float = 90.0) -> bool:
        """Wait until robot is no longer in RUNNING mode."""
        import time
        t0 = time.time()
        time.sleep(0.4)
        while time.time() - t0 < timeout:
            if self.get_mode() != self.MODE_RUNNING:
                return True
            time.sleep(0.15)
        return False

    def check_reachability(self, x, y, z, rx, ry, rz):
        """Check if a Cartesian pose is reachable.

        Returns ``(reachable: bool, joints: tuple|None, message: str)``.
        Default: always returns True (override for IK-based checking).
        """
        return True, None, "OK (no IK check)"
