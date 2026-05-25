"""
ABB IRB driver (IRC5 controller) via a custom RAPID TCP socket server.

No external libraries — stdlib socket only.

Controller prerequisites (one-time, via RobotStudio):
  1. Copy docker/abb_rapid/HOME/abb_tcp_server.mod into the controller's HOME
     folder and load it into T_ROB1 as the main program.
  2. Set controller to AUTO mode + Motors On from the FlexPendant.
  3. The RAPID server listens on TCP port 10100 by default.

Protocol: line-oriented ASCII, newline-terminated.
  Commands  → PING | GETPOSE | GETJOINTS | STOP
               MOVEL    x y z rx ry rz vtcp vori
               MOVEJ    x y z rx ry rz vtcp vori
               MOVEABSJ j1 j2 j3 j4 j5 j6 vtcp vori
               SETDO    signal_name 0|1
  Responses → OK [payload...]  |  ERR message

Motion commands block until RAPID confirms the move is complete, then reply OK.
"""

import socket
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutTimeout

from robots.base import RobotBase

DEFAULT_PORT       = 80
DEFAULT_DO_VACUUM  = "DO_VACUUM"
MAX_TCP_SPEED_MMPS = 1000.0
MAX_ORI_SPEED_DEGS = 500.0
MOTION_SOCK_TIMEOUT_S = 120.0   # covers slow moves; raised from default 10 s


class ABBTCP(RobotBase):
    """ABB IRB driver via raw TCP socket (abb_tcp_server.mod on controller)."""

    MODE_RUNNING = 1
    MODE_ERROR   = 2
    MODE_ENABLED = 0

    def __init__(self, ip: str, *,
                 port: int = DEFAULT_PORT,
                 do_vacuum_signal: str = DEFAULT_DO_VACUUM,
                 **kwargs):
        self._ip        = ip
        self._port      = port
        self._do_vacuum = do_vacuum_signal
        self._speed_pct = 20
        self._lock      = threading.Lock()
        self._cmd_counter = 0
        self._futures: dict[int, Future] = {}
        self._moving    = False
        self._closed    = False

        print(f"[ABBTCP] Connecting to {ip}:{port}…", flush=True)
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(10.0)
            self._sock.connect((ip, port))
            self._sock.settimeout(MOTION_SOCK_TIMEOUT_S)
            self._file = self._sock.makefile("r", encoding="ascii")
        except OSError as e:
            raise ConnectionError(
                f"[ABBTCP] connection failed — {e}\n"
                "  Check: IP address, port 10100 open, "
                "abb_tcp_server.mod running in T_ROB1"
            ) from e

        resp = self._send_recv("PING")
        if not resp.startswith("OK"):
            raise ConnectionError(f"[ABBTCP] PING failed: {resp!r}")
        print(f"[ABBTCP] Connected ({ip}:{port})", flush=True)

        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="abbtcp-motion")

    # ── Low-level I/O ─────────────────────────────────────────────────────
    def _send_recv(self, line: str) -> str:
        with self._lock:
            self._sock.sendall((line.rstrip() + "\n").encode("ascii"))
            return self._file.readline().rstrip("\r\n")

    def _send_recv_check(self, line: str) -> str:
        resp = self._send_recv(line)
        if resp.startswith("ERR"):
            raise RuntimeError(f"[ABBTCP] controller error: {resp}")
        return resp

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def enable(self):
        print("[ABBTCP] enable(): set controller to AUTO + Motors On on FlexPendant",
              flush=True)

    def power_on(self):
        self.enable()

    def clear_error(self):
        try:
            self._send_recv("STOP")
        except Exception as e:
            print(f"[ABBTCP] clear_error: {e}", flush=True)

    def stop(self):
        try:
            self._send_recv("STOP")
        except Exception as e:
            print(f"[ABBTCP] stop(): {e}", flush=True)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            self._file.close()
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass

    # ── Status ────────────────────────────────────────────────────────────
    def get_mode(self) -> int:
        if self._moving:
            return self.MODE_RUNNING
        if self._closed:
            return self.MODE_ERROR
        try:
            resp = self._send_recv("PING")
            return self.MODE_ENABLED if resp.startswith("OK") else self.MODE_ERROR
        except Exception:
            return self.MODE_ERROR

    def get_pose(self) -> tuple:
        """Return (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg) ZYX Euler."""
        resp = self._send_recv_check("GETPOSE")
        return tuple(float(v) for v in resp[3:].split())

    def get_angle(self) -> tuple:
        """Return (j1..j6) in degrees."""
        resp = self._send_recv_check("GETJOINTS")
        return tuple(float(v) for v in resp[3:].split())

    # ── Motion ────────────────────────────────────────────────────────────
    def set_speed(self, percent: int):
        self._speed_pct = max(1, min(100, int(percent)))

    def _speeddata(self):
        scale = self._speed_pct / 100.0
        return MAX_TCP_SPEED_MMPS * scale, MAX_ORI_SPEED_DEGS * scale

    def _next_cmd_id(self) -> int:
        with self._lock:
            self._cmd_counter += 1
            return self._cmd_counter

    def _exec_motion(self, cmd_line: str):
        self._moving = True
        try:
            self._send_recv_check(cmd_line)
        finally:
            self._moving = False

    def move_linear(self, x, y, z, rx, ry, rz) -> int:
        vtcp, vori = self._speeddata()
        cmd = (f"MOVEL {x:.2f} {y:.2f} {z:.2f} "
               f"{rx:.3f} {ry:.3f} {rz:.3f} {vtcp:.1f} {vori:.1f}")
        cmd_id = self._next_cmd_id()
        self._futures[cmd_id] = self._executor.submit(self._exec_motion, cmd)
        return cmd_id

    def move_joint(self, x, y, z, rx, ry, rz) -> int:
        vtcp, vori = self._speeddata()
        cmd = (f"MOVEJ {x:.2f} {y:.2f} {z:.2f} "
               f"{rx:.3f} {ry:.3f} {rz:.3f} {vtcp:.1f} {vori:.1f}")
        cmd_id = self._next_cmd_id()
        self._futures[cmd_id] = self._executor.submit(self._exec_motion, cmd)
        return cmd_id

    def move_joint_angles(self, j1, j2, j3, j4, j5, j6) -> int:
        vtcp, vori = self._speeddata()
        cmd = (f"MOVEABSJ {j1:.2f} {j2:.2f} {j3:.2f} "
               f"{j4:.2f} {j5:.2f} {j6:.2f} {vtcp:.1f} {vori:.1f}")
        cmd_id = self._next_cmd_id()
        self._futures[cmd_id] = self._executor.submit(self._exec_motion, cmd)
        return cmd_id

    def wait_motion(self, cmd_id: int, timeout: float = 90.0) -> bool:
        fut = self._futures.get(cmd_id)
        if fut is None:
            raise ValueError(f"[ABBTCP] unknown cmd_id {cmd_id}")
        try:
            fut.result(timeout=timeout)
        except _FutTimeout:
            raise TimeoutError(
                f"[ABBTCP] motion timeout after {timeout:.0f}s (cmd {cmd_id})")
        except Exception as e:
            raise RuntimeError(
                f"[ABBTCP] motion failed (cmd {cmd_id}): {e}") from e
        finally:
            self._futures.pop(cmd_id, None)
        return True

    # ── End-effector ──────────────────────────────────────────────────────
    def vacuum_on(self, port: int = 0):
        if self._tool is not None:
            return self._tool.grasp()
        self._send_recv_check(f"SETDO {self._do_vacuum} 1")
        print(f"[ABBTCP] vacuum_on  ({self._do_vacuum})", flush=True)

    def vacuum_off(self, port: int = 0):
        if self._tool is not None:
            return self._tool.release()
        self._send_recv_check(f"SETDO {self._do_vacuum} 0")
        print(f"[ABBTCP] vacuum_off ({self._do_vacuum})", flush=True)
