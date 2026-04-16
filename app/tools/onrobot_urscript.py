"""
OnRobot RG gripper — UR Dashboard program execution.

HOW IT WORKS
------------
The UR Dashboard Server (port 29999) can load and play .urp programs saved on
the teach pendant.  Create two simple programs on the pendant:

  gripper_open.urp   — one RG6 node set to open (width=max, any force)
  gripper_close.urp  — one RG6 node set to close (width=0, desired force)

This tool connects to the dashboard, runs the appropriate program, and waits
for it to finish.  Because the program runs natively on the controller with the
URCap in its proper context, the gripper always moves reliably.

WHY NOT URScript / rg_grip()
-----------------------------
ur_rtde keeps its own daemon running.  The URCap's ``rg_grip()`` function
communicates via a local socket (127.0.0.1:54321) to a Java process that the
URCap manages.  When called from a detached secondary script, that socket call
can fail silently because the URCap server only services the active program
context.  Running a saved program avoids this entirely.

Setup (one-time, on the teach pendant)
---------------------------------------
1. New program → add one "RG6 Grip" node → set width/force → save as
   ``gripper_close`` (exact name, no spaces, lowercase).
2. New program → add one "RG6 Grip" node → set width=max → save as
   ``gripper_open``.
3. Programs live in /programs/ on the controller.

Usage::

    tool = OnRobotRGURScript(robot=robot)   # reuses robot's dashboard socket
    robot.attach_tool(tool)

    # Custom program names (if you saved them differently):
    tool = OnRobotRGURScript(robot=robot,
                             open_program="my_open",
                             close_program="my_close")
"""

import socket
import time

from tools.base import ToolBase

DASHBOARD_PORT  = 29999
SCRIPT_PORT     = 30002       # fallback only
POLL_INTERVAL   = 0.15        # s between programState polls
PLAY_TIMEOUT    = 15.0        # s max wait for program to finish


class OnRobotRGURScript(ToolBase):
    """OnRobot RG via UR Dashboard — load + play saved .urp programs.

    Parameters
    ----------
    robot :
        Connected ``UR10`` robot object (provides ``_ip`` and ``_dash_sock``).
    robot_ip : str or None
        IP of the UR controller — used if ``robot`` is None.
    open_program : str
        Name of the saved program for opening (without .urp).
    close_program : str
        Name of the saved program for closing (without .urp).
    grasp_wait_s : float
        Seconds to wait after the close program finishes (gripper settle time).
        Default 2.0 s — the gripper may still be physically closing after the
        UR program reports as stopped.
    release_wait_s : float
        Seconds to wait after the open program finishes.  Default 1.5 s.
    """

    def __init__(
        self,
        robot=None,
        robot_ip: str | None = None,
        open_program: str  = "gripper_open",
        close_program: str = "gripper_close",
        grasp_wait_s: float = 2.0,
        release_wait_s: float = 1.5,
    ):
        if robot is not None and hasattr(robot, "_ip"):
            self._ip = robot._ip
        elif robot_ip:
            self._ip = robot_ip
        else:
            raise ValueError("Pass robot= (UR10 object) or robot_ip=")

        self._open_program  = open_program
        self._close_program = close_program
        self._grasp_wait    = grasp_wait_s
        self._release_wait  = release_wait_s

        # Open a dedicated dashboard socket for gripper commands so we don't
        # interfere with the UR10 robot driver's own dashboard socket.
        self._dash = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._dash.settimeout(5.0)
        try:
            self._dash.connect((self._ip, DASHBOARD_PORT))
            banner = self._dash.recv(1024).decode(errors="replace").strip()
            print(f"[OnRobotRGURScript] Dashboard connected: {banner}", flush=True)
        except OSError as e:
            raise ConnectionError(
                f"[OnRobotRGURScript] Cannot reach dashboard {self._ip}:{DASHBOARD_PORT} — {e}"
            )

        print(
            f"[OnRobotRGURScript] Ready  "
            f"open='{open_program}.urp'  close='{close_program}.urp'",
            flush=True,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _dash_cmd(self, cmd: str) -> str:
        """Send one Dashboard command, return the response line."""
        self._dash.sendall((cmd + "\n").encode())
        try:
            resp = self._dash.recv(1024).decode(errors="replace").strip()
        except socket.timeout:
            resp = ""
        return resp

    def _run_program(self, name: str, settle_s: float = 0.0):
        """Load, play, and wait for a saved .urp program to finish.

        After the UR program reports as stopped, waits an additional *settle_s*
        seconds for the gripper to physically finish moving — the program may
        report "stopped" before the mechanical motion is fully complete.
        """
        path = f"/programs/{name}.urp"

        resp = self._dash_cmd(f"load {path}")
        if "error" in resp.lower() or "failed" in resp.lower():
            raise RuntimeError(f"[OnRobotRGURScript] load failed: {resp!r}")
        print(f"[OnRobotRGURScript] load {name} → {resp!r}", flush=True)

        resp = self._dash_cmd("play")
        print(f"[OnRobotRGURScript] play → {resp!r}", flush=True)

        # Poll until program stops running
        t0 = time.time()
        while time.time() - t0 < PLAY_TIMEOUT:
            state = self._dash_cmd("programState").lower()
            if "stopped" in state or "idle" in state or "paused" in state:
                break
            time.sleep(POLL_INTERVAL)

        # Extra settle time — gripper may still be physically moving
        if settle_s > 0:
            print(f"[OnRobotRGURScript] waiting {settle_s:.1f}s for gripper to settle", flush=True)
            time.sleep(settle_s)

    # ── ToolBase interface ────────────────────────────────────────────────────
    def grasp(self):
        print(f"[OnRobotRGURScript] grasp  → running '{self._close_program}'", flush=True)
        self._run_program(self._close_program, settle_s=self._grasp_wait)

    def release(self):
        print(f"[OnRobotRGURScript] release → running '{self._open_program}'", flush=True)
        self._run_program(self._open_program, settle_s=self._release_wait)

    def close(self):
        try:
            self._dash.close()
        except Exception:
            pass

    @property
    def tool_name(self) -> str:
        return f"OnRobotRG-Dashboard({self._close_program}/{self._open_program})"
