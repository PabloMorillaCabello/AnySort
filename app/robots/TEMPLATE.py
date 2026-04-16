"""
Template for adding a new robot driver.

Steps:
  1. Copy this file → ``my_robot.py`` in the same directory
  2. Implement every method marked with ``raise NotImplementedError``
  3. Register in ``__init__.py`` → add to ``ROBOT_DRIVERS`` dict:
       "My Robot": ("robots.my_robot", "MyRobot"),
  4. (Optional) Add your pip dependency to ``docker/requirements.txt``
  5. Test:  from robots import create_robot
           r = create_robot("My Robot", "192.168.1.100")
"""

import time

from robots.base import RobotBase


class MyRobot(RobotBase):
    """Short description of your robot driver."""

    # Map these to your robot controller's actual state codes
    MODE_RUNNING = 1
    MODE_ERROR   = 2
    MODE_ENABLED = 0

    def __init__(self, ip: str, **kwargs):
        # Connect to your robot here
        # Example: self._sock = socket.create_connection((ip, 12345))
        raise NotImplementedError

    def enable(self):
        raise NotImplementedError

    def power_on(self):
        raise NotImplementedError

    def clear_error(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def get_mode(self) -> int:
        raise NotImplementedError

    def get_pose(self) -> tuple:
        # Must return (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg)
        raise NotImplementedError

    def get_angle(self) -> tuple:
        # Must return (j1_deg, j2_deg, ..., j6_deg)
        raise NotImplementedError

    def set_speed(self, percent: int):
        raise NotImplementedError

    def move_linear(self, x, y, z, rx, ry, rz) -> int:
        # x,y,z in mm — rx,ry,rz in degrees
        # Return a command ID (int) for wait_motion()
        raise NotImplementedError

    def move_joint_angles(self, j1, j2, j3, j4, j5, j6) -> int:
        # All angles in degrees
        # Return a command ID (int) for wait_motion()
        raise NotImplementedError

    def wait_motion(self, cmd_id: int, timeout: float = 90.0) -> bool:
        # Block until motion cmd_id completes
        # Return True on success
        # Raise RuntimeError on robot error
        # Raise TimeoutError on timeout
        raise NotImplementedError

    # ── End-effector ─────────────────────────────────────────────────────
    # Option A (recommended): rely on RobotBase delegation.
    #   Attach a tool with robot.attach_tool(tool) — vacuum_on/off are
    #   handled automatically via tool.grasp() / tool.release().
    #
    # Option B: override directly (e.g. robot has built-in IO for gripper):
    #
    # def vacuum_on(self, port: int = 0):
    #     raise NotImplementedError
    #
    # def vacuum_off(self, port: int = 0):
    #     raise NotImplementedError
