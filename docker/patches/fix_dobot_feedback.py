#!/usr/bin/env python3
"""
Patch: Fix Dobot UI feedback thread crash and thread-safety issue.

Bug 1 - TCP partial reads:
  TCP recv() can return fewer bytes than the 1440-byte Dobot feedback packet.
  The original code slices temp[0:1440] without checking actual length, causing
  np.frombuffer to fail with: ValueError: buffer size must be a multiple of
  element size.

Bug 2 - Tkinter thread-safety:
  The original code updates tkinter widgets directly from the feedback thread.
  Tkinter is not thread-safe; this causes segfaults on Linux.

Fix:
  1. Accumulate exactly 1440 bytes in a loop before parsing.
  2. Use self.root.after() to schedule UI updates on the main thread.

Usage (called from Dockerfile):
    python3 /tmp/patches/fix_dobot_feedback.py /opt/Dobot_hv/ui.py
"""
import sys, re

def patch(path):
    with open(path, "r") as f:
        src = f.read()

    # ---- old block (the entire broken feed_back method) ----
    old = (
        "    def feed_back(self):\n"
        "        while True:\n"
        '            print("self.global_state(connect)", self.global_state["connect"])\n'
        '            if not self.global_state["connect"]:\n'
        "                break\n"
        "\n"
        "            self.client_feed.socket_dobot.setblocking(True)  # 设置为阻塞模式\n"
        "            data = bytes()\n"
        "            temp = self.client_feed.socket_dobot.recv(144000)\n"
        "            if len(temp) > 1440:\n"
        "                temp = self.client_feed.socket_dobot.recv(144000)\n"
        "            data = temp[0:1440]\n"
        "        \n"
        "\n"
        "            a = np.frombuffer(data, dtype=MyType)\n"
        '            print("robot_mode:", a["RobotMode"][0])\n'
        '            print("TestValue:", hex((a[\'TestValue\'][0])))\n'
        "            if hex((a['TestValue'][0])) == '0x123456789abcdef':\n"
        "                # print('tool_vector_actual',\n"
        "                #       np.around(a['tool_vector_actual'], decimals=4))\n"
        "                # print('QActual', np.around(a['q_aQActualctual'], decimals=4))\n"
        "\n"
        "                # Refresh Properties\n"
        '                self.label_feed_speed["text"] = a["SpeedScaling"][0]\n'
        '                self.label_robot_mode["text"] = LABEL_ROBOT_MODE[a["RobotMode"][0]]\n'
        '                self.label_di_input["text"] = bin(a["DigitalInputs"][0])[\n'
        "                    2:].rjust(64, '0')\n"
        '                self.label_di_output["text"] = bin(a["DigitalOutputs"][0])[\n'
        "                    2:].rjust(64, '0')\n"
        "\n"
        "                # Refresh coordinate points\n"
        '                self.set_feed_joint(LABEL_JOINT, a["QActual"])\n'
        '                self.set_feed_joint(LABEL_COORD, a["ToolVectorActual"])\n'
        "\n"
        "                # check alarms\n"
        '                if a["RobotMode"] == 9:\n'
        "                    self.display_error_info()"
    )

    # ---- new block (TCP accumulation + thread-safe UI via root.after) ----
    new = (
        "    def feed_back(self):\n"
        "        PACKET_SIZE = 1440\n"
        "        while True:\n"
        '            if not self.global_state["connect"]:\n'
        "                break\n"
        "            try:\n"
        "                self.client_feed.socket_dobot.setblocking(True)\n"
        "                data = bytes()\n"
        "                while len(data) < PACKET_SIZE:\n"
        "                    remaining = PACKET_SIZE - len(data)\n"
        "                    chunk = self.client_feed.socket_dobot.recv(remaining)\n"
        "                    if not chunk:\n"
        "                        break\n"
        "                    data += chunk\n"
        "                if len(data) < PACKET_SIZE:\n"
        "                    time.sleep(0.5)\n"
        "                    continue\n"
        "                a = np.frombuffer(data, dtype=MyType)\n"
        "                if hex((a['TestValue'][0])) != '0x123456789abcdef':\n"
        "                    continue\n"
        "                # Schedule UI updates on the main thread (tkinter is not thread-safe)\n"
        "                self.root.after(0, self._update_feed_ui, a)\n"
        "            except Exception as e:\n"
        '                print(f"[feedback] error: {e}")\n'
        "                time.sleep(0.5)\n"
        "                continue\n"
        "\n"
        '    def _update_feed_ui(self, a):\n'
        '        """Update UI labels with feedback data. Runs on the main thread."""\n'
        "        try:\n"
        '            self.label_feed_speed["text"] = a["SpeedScaling"][0]\n'
        '            self.label_robot_mode["text"] = LABEL_ROBOT_MODE[a["RobotMode"][0]]\n'
        '            self.label_di_input["text"] = bin(a["DigitalInputs"][0])[\n'
        "                2:].rjust(64, '0')\n"
        '            self.label_di_output["text"] = bin(a["DigitalOutputs"][0])[\n'
        "                2:].rjust(64, '0')\n"
        '            self.set_feed_joint(LABEL_JOINT, a["QActual"])\n'
        '            self.set_feed_joint(LABEL_COORD, a["ToolVectorActual"])\n'
        '            if a["RobotMode"] == 9:\n'
        "                self.display_error_info()\n"
        "        except Exception as e:\n"
        '            print(f"[feedback UI] error: {e}")'
    )

    if old not in src:
        # Try a more lenient match (strip trailing whitespace per line)
        print("WARNING: exact match not found, trying lenient match...")
        old_norm = re.sub(r'[ \t]+\n', '\n', old)
        src_norm = re.sub(r'[ \t]+\n', '\n', src)
        if old_norm not in src_norm:
            print("ERROR: Could not find the feed_back method to patch!")
            print("The upstream code may have changed. Manual patching required.")
            sys.exit(1)
        src = src_norm.replace(old_norm, new)
    else:
        src = src.replace(old, new)

    with open(path, "w") as f:
        f.write(src)
    print(f"SUCCESS: Patched {path} — feed_back() now uses TCP accumulation + thread-safe UI")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-ui.py>")
        sys.exit(1)
    patch(sys.argv[1])
