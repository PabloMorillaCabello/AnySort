"""
Abstract base class for end-effector tools.

Every tool must subclass ``ToolBase`` and implement ``grasp()`` and ``release()``.
The pipeline calls ``robot.vacuum_on()`` / ``robot.vacuum_off()``, which delegate
to the attached tool's ``grasp()`` / ``release()`` when a tool is attached.

Adding a new tool
-----------------
1. Copy ``TEMPLATE.py`` → ``my_tool.py`` in this directory
2. ``class MyTool(ToolBase): ...`` — implement ``grasp()``, ``release()``
3. Register in ``tools/__init__.py`` → ``TOOL_DRIVERS``
4. Attach to robot: ``robot.attach_tool(create_tool("My Tool", ...))``
"""

from abc import ABC, abstractmethod


class ToolBase(ABC):
    """Interface that every end-effector tool must implement."""

    @abstractmethod
    def grasp(self):
        """Close gripper / activate vacuum — pick the object."""

    @abstractmethod
    def release(self):
        """Open gripper / deactivate vacuum — release the object."""

    def close(self):
        """Disconnect and free resources.  No-op by default."""
        pass

    @property
    def tool_name(self) -> str:
        """Human-readable name.  Override to customise."""
        return type(self).__name__
