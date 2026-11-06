import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
HOST = os.environ.get("OVMS_STUDIO_HOST", "127.0.0.1")
PORT = int(os.environ.get("OVMS_STUDIO_PORT", "18080"))
URL = f"http://{HOST}:{PORT}"


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def start_backend_if_needed():
    if is_port_open(HOST, PORT):
        return

    py_exe = sys.executable
    # Prefer pythonw.exe so no console window stays open
    pyw = os.path.join(os.path.dirname(py_exe), "pythonw.exe")
    runner = pyw if os.path.exists(pyw) else py_exe

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if os.name == "nt" else 0
    log_file = open(os.path.join(STUDIO_DIR, "studio_web.log"), "a", encoding="utf-8", errors="ignore")
    subprocess.Popen(
        [runner, "-m", "uvicorn", "server:app", "--host", HOST, "--port", str(PORT), "--no-access-log"],
        cwd=STUDIO_DIR,
        stdout=log_file,
        stderr=log_file,
        creationflags=creationflags,
    )

    for _ in range(30):
        if is_port_open(HOST, PORT):
            break
        time.sleep(0.2)


def open_app_window():
    edge_candidates = [
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
    ]
    for edge in edge_candidates:
        if os.path.exists(edge):
            subprocess.Popen([edge, f"--app={URL}", "--window-size=1380,880"])
            return
    webbrowser.open(URL)


if __name__ == "__main__":
    start_backend_if_needed()
    if "--no-window" not in sys.argv and "--server-only" not in sys.argv:
        open_app_window()
