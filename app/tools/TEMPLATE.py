"""
Template for adding a new end-effector tool.

Steps:
  1. Copy this file → ``my_tool.py`` in the same directory
  2. Implement ``grasp()`` and ``release()``
  3. Register in ``tools/__init__.py`` → add to ``TOOL_DRIVERS``:
       "My Tool": ("tools.my_tool", "MyTool"),
  4. (Optional) Add your pip dependency to ``docker/requirements.txt``
  5. Attach to a robot:
       from tools import create_tool
       robot.attach_tool(create_tool("My Tool", ip="192.168.1.x"))

Communication patterns to follow:
  - Digital output  → see dobot_vacuum.py   (write a port high/low)
  - Modbus TCP      → see onrobot.py        (pymodbus holding registers)
  - Serial / RS-485 → use pyserial (already in requirements.txt)
  - TCP socket      → use stdlib socket
"""

from tools.base import ToolBase


class MyTool(ToolBase):
    """Short description of your tool."""

    def __init__(self, ip: str, **kwargs):
        # Connect to your tool here.
        # Store any state you need for grasp/release.
        raise NotImplementedError

    def grasp(self):
        """Close gripper / activate suction — called at pick."""
        raise NotImplementedError

    def release(self):
        """Open gripper / deactivate suction — called at place."""
        raise NotImplementedError

    def close(self):
        """Disconnect — called when the pipeline shuts down."""
        pass  # implement if your tool needs cleanup

    @property
    def tool_name(self) -> str:
        return "MyTool"
