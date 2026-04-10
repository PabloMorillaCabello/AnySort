"""
orbbec_quiet.py
Redirect OrbbecSDK C-level stderr to a log file so timestamp-anomaly spam
does not flood the terminal or crash the system.  Nothing is discarded —
all SDK messages are preserved in LOG_PATH for debugging.

Import BEFORE pyorbbecsdk (add near the top of your script):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import orbbec_quiet  # noqa

After Pipeline.start() (which loads extensions that can reset fd 2), call:
    orbbec_quiet.reapply()

To watch SDK output live:
    tail -f /tmp/orbbec_sdk.log
"""
import os
import sys
import threading

LOG_PATH     = "/tmp/orbbec_sdk.log"
MAX_LOG_SIZE = 50 * 1024 * 1024   # 50 MB — rotate (truncate) when exceeded

# Saved write-end of the pipe so reapply() can re-point fd 2 at it
# after SDK extensions reset fd 2 back to the terminal.
_pipe_write_fd: int = -1


def reapply():
    """Re-apply the fd 2 → pipe redirect.

    OrbbecSDK loads extensions (e.g. from /ros2_ws/install/orbbec_camera/)
    during Pipeline.start().  Those extensions can reset C-level fd 2 back
    to the original terminal.  Call this function immediately after
    pipeline.start(config) to restore the redirect.
    """
    if _pipe_write_fd >= 0:
        os.dup2(_pipe_write_fd, 2)


def _install():
    global _pipe_write_fd

    # Open the log file (append mode)
    log_file = open(LOG_PATH, "a", buffering=1, errors="replace")

    r_fd, w_fd = os.pipe()

    # Save the real stderr fd before we redirect fd 2
    real_fd     = os.dup(2)
    real_stderr = os.fdopen(real_fd, "w", buffering=1)

    # Keep a dup of the write end so reapply() can re-point fd 2 later
    _pipe_write_fd = os.dup(w_fd)

    # Point C-level fd 2 (where OrbbecSDK writes) at the pipe
    os.dup2(w_fd, 2)
    os.close(w_fd)

    # Keep Python sys.stderr aimed at the real terminal
    sys.stderr = real_stderr

    def _reader():
        with os.fdopen(r_fd, "r", errors="replace") as pipe_r:
            for line in pipe_r:
                try:
                    # Rotate log file if it has grown too large
                    if log_file.tell() > MAX_LOG_SIZE:
                        log_file.seek(0)
                        log_file.truncate()
                        log_file.write("=== log rotated (size limit reached) ===\n")

                    log_file.write(line)
                    log_file.flush()
                except Exception:
                    pass

    t = threading.Thread(target=_reader, name="orbbec-stderr-sink", daemon=True)
    t.start()

    print(f"[orbbec_quiet] OrbbecSDK stderr → {LOG_PATH}  (max {MAX_LOG_SIZE//1024//1024} MB)")


_install()
del _install
