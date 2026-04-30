"""
Dobot vacuum gripper tool.

Controls the vacuum via the Dobot dashboard ``ToolDO`` command (digital output).
Pass the ``DobotApiDashboard`` instance (``robot._dashboard``) or the full
``DobotCR`` robot object — both work.

Usage::

    from tools import create_tool
    tool = create_tool("Dobot Vacuum", robot=robot)
    robot.attach_tool(tool)

Or directly::

    from tools.dobot_vacuum import DobotVacuum
    tool = DobotVacuum(dashboard=robot._dashboard, do_port=1)
    robot.attach_tool(tool)
"""

from tools.base import ToolBase

# Digital output port used by the vacuum on standard Dobot setups
DEFAULT_DO_PORT = 1


class DobotVacuum(ToolBase):
    """Dobot vacuum gripper — on/off via ``ToolDO`` digital output.

    Parameters
    ----------
    dashboard:
        ``DobotApiDashboard`` instance **or** a ``DobotCR`` robot instance
        (its ``._dashboard`` attribute is used automatically).
    do_port:
        Digital output port number (1-based, default 1).
    """

    def __init__(self, dashboard, do_port: int = DEFAULT_DO_PORT):
        # Accept either a raw DobotApiDashboard or a DobotCR robot object
        if hasattr(dashboard, "_dashboard"):
            dashboard = dashboard._dashboard
        self._dashboard = dashboard
        self._port = do_port

    def grasp(self):
        """Activate vacuum (ToolDO port → 1)."""
        resp = self._dashboard.ToolDO(self._port, 1)
        print(f"[DobotVacuum] grasp  ToolDO({self._port}, 1) → {resp!r}", flush=True)
        return resp

    def release(self):
        """Deactivate vacuum (ToolDO port → 0)."""
        resp = self._dashboard.ToolDO(self._port, 0)
        print(f"[DobotVacuum] release ToolDO({self._port}, 0) → {resp!r}", flush=True)
        return resp

    @property
    def tool_name(self) -> str:
        return f"DobotVacuum(DO={self._port})"
