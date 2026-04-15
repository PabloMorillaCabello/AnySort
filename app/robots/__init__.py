"""
Robot driver registry — modular robot backend for AnySort pipeline.

To add a new robot:
  1. Create a new file in this directory (e.g. ``my_robot.py``)
  2. Subclass ``RobotBase`` from ``robots.base``
  3. Implement all abstract methods
  4. Register your driver in ``ROBOT_DRIVERS`` below

The pipeline discovers available drivers at startup via ``get_available_drivers()``.
"""

from robots.base import RobotBase  # noqa: F401

# ── Driver registry ──────────────────────────────────────────────────────
# key   = human-readable name shown in UI dropdown
# value = (module_path, class_name)
ROBOT_DRIVERS: dict[str, tuple[str, str]] = {
    "Dobot CR": ("robots.dobot_cr", "DobotCR"),
    "UR10":     ("robots.ur10",     "UR10"),
}


def get_available_drivers() -> list[str]:
    """Return driver names whose dependencies are importable."""
    available = []
    for name, (mod_path, _cls_name) in ROBOT_DRIVERS.items():
        try:
            __import__(mod_path, fromlist=[_cls_name])
            available.append(name)
        except ImportError:
            pass
    return available


def get_driver_names() -> list[str]:
    """Return ALL registered driver names (even if deps missing)."""
    return list(ROBOT_DRIVERS.keys())


def create_robot(driver_name: str, ip: str, **kwargs) -> "RobotBase":
    """Instantiate a robot driver by its registry name.

    Raises ``KeyError`` if driver_name is not registered.
    Raises ``ImportError`` if the driver's dependencies are missing.
    """
    if driver_name not in ROBOT_DRIVERS:
        raise KeyError(f"Unknown robot driver: {driver_name!r}. "
                       f"Available: {list(ROBOT_DRIVERS.keys())}")
    mod_path, cls_name = ROBOT_DRIVERS[driver_name]
    mod = __import__(mod_path, fromlist=[cls_name])
    cls = getattr(mod, cls_name)
    return cls(ip, **kwargs)
