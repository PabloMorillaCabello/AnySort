"""
End-effector tool registry — modular tool backend for AnySort pipeline.

To add a new tool:
  1. Create a new file in this directory (e.g. ``my_tool.py``)
  2. Subclass ``ToolBase`` from ``tools.base``
  3. Implement ``grasp()`` and ``release()``
  4. Register your driver in ``TOOL_DRIVERS`` below

The pipeline attaches a tool to the robot via:
    robot.attach_tool(create_tool("OnRobot Gripper", ip="192.168.1.1", model="RG2"))
After that ``robot.vacuum_on()`` → ``tool.grasp()`` and
``robot.vacuum_off()`` → ``tool.release()`` automatically.
"""

from tools.base import ToolBase  # noqa: F401

# ── Tool registry ─────────────────────────────────────────────────────────────
# key   = human-readable name shown in UI dropdown
# value = (module_path, class_name)
TOOL_DRIVERS: dict[str, tuple[str, str]] = {
    "Dobot Vacuum":           ("tools.dobot_vacuum",      "DobotVacuum"),
    "OnRobot RG (URCap)":     ("tools.onrobot_urscript",  "OnRobotRGURScript"),
}


def get_available_tools() -> list[str]:
    """Return tool names whose dependencies are importable."""
    available = []
    for name, (mod_path, _cls_name) in TOOL_DRIVERS.items():
        try:
            __import__(mod_path, fromlist=[_cls_name])
            available.append(name)
        except ImportError:
            pass
    return available


def get_tool_names() -> list[str]:
    """Return ALL registered tool names (even if deps missing)."""
    return list(TOOL_DRIVERS.keys())


def create_tool(driver_name: str, **kwargs) -> "ToolBase":
    """Instantiate a tool driver by its registry name.

    All kwargs are forwarded to the tool constructor.

    Raises ``KeyError`` if driver_name is not registered.
    Raises ``ImportError`` if the tool's dependencies are missing.

    Examples::

        # Dobot vacuum (pass the robot or its dashboard):
        tool = create_tool("Dobot Vacuum", dashboard=robot, do_port=1)

        # OnRobot RG2 via Compute Box:
        tool = create_tool("OnRobot Gripper", ip="192.168.1.1", model="RG2")

        # OnRobot with manual register config:
        tool = create_tool(
            "OnRobot Gripper",
            ip="192.168.1.1", port=502, slave_id=65,
            open_width_01mm=1100, close_width_01mm=0, force_01n=200,
        )
    """
    if driver_name not in TOOL_DRIVERS:
        raise KeyError(
            f"Unknown tool driver: {driver_name!r}. "
            f"Available: {list(TOOL_DRIVERS.keys())}"
        )
    mod_path, cls_name = TOOL_DRIVERS[driver_name]
    mod = __import__(mod_path, fromlist=[cls_name])
    cls = getattr(mod, cls_name)
    return cls(**kwargs)
