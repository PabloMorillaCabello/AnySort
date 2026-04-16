import socket
import time

ROBOT_IP = "192.168.1.100"
DASHBOARD_PORT = 29999

def dashboard_cmd(cmd, wait=0.3):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ROBOT_IP, DASHBOARD_PORT))
        hello = s.recv(1024).decode(errors="ignore")
        s.sendall((cmd + "\n").encode())
        time.sleep(wait)
        reply = s.recv(4096).decode(errors="ignore")
        return hello.strip(), reply.strip()

def run_program(program_name):
    print(dashboard_cmd("is in remote control"))
    print(dashboard_cmd(f"load {program_name}"))
    time.sleep(2.0)
    print(dashboard_cmd("get loaded program"))
    print(dashboard_cmd("play"))

if __name__ == "__main__":
    run_program("gripper_open.urp")
    time.sleep(4)
    run_program("gripper_close.urp")