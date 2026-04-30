# Tool Architecture

> Modular end-effector driver system — abstract base class + registry pattern.
> **Files:** `app/tools/base.py`, `app/tools/__init__.py`

---

## Design

Tools are attached to robots at runtime. The pipeline calls `robot.vacuum_on()` / `robot.vacuum_off()`, which delegate to the tool's `grasp()` / `release()`.

```
ToolBase (ABC)
├── DobotVacuum        ← tools/dobot_vacuum.py  (Dobot digital output)
├── OnRobotRGURScript  ← tools/onrobot_urscript.py  (UR dashboard programs)
└── [YourTool]         ← tools/my_tool.py (add new tool here)
```

---

## ToolBase Interface

```python
class ToolBase(ABC):
    @abstractmethod
    def grasp(self):    # close gripper / activate vacuum

    @abstractmethod
    def release(self):  # open gripper / deactivate vacuum

    def close(self):    # disconnect — no-op by default

    @property
    def tool_name(self) -> str:  # human-readable name
```

---

## Registry (`tools/__init__.py`)

```python
TOOL_DRIVERS = {
    "Dobot Vacuum":           ("tools.dobot_vacuum",      "DobotVacuum"),
    "OnRobot RG (URCap)":     ("tools.onrobot_urscript",  "OnRobotRGURScript"),
}

# Create and attach:
tool = create_tool("Dobot Vacuum", dashboard=robot, do_port=1)
robot.attach_tool(tool)
```

---

## How Robot Uses Tool

```python
# In RobotBase:
def vacuum_on(self, port=0):
    self._tool.grasp()

def vacuum_off(self, port=0):
    self._tool.release()
```

If no tool is attached and `vacuum_on()` is called → `RuntimeError`.

Dobot CR overrides `vacuum_on()`/`vacuum_off()` directly via `ToolDO` (but `DobotVacuum` is the clean abstraction).

---

## Adding a New Tool

1. Copy `app/tools/TEMPLATE.py` → `my_tool.py`
2. `class MyTool(ToolBase)` — implement `grasp()` and `release()`
3. Register in `TOOL_DRIVERS` dict in `tools/__init__.py`
4. Attach: `robot.attach_tool(create_tool("My Tool", ...))`

---

## Links
- [[Dobot Vacuum]] — digital output vacuum (Dobot)
- [[OnRobot RG]] — dashboard program gripper (UR10)
- [[../Robots/Robot Architecture|Robot Architecture]] — how tools attach to robots
