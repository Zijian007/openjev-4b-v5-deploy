"""Start/stop this deployment without sudo or Docker."""
import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIDFILE = ROOT / "runtime/service.pid"
LOGFILE = ROOT / "logs/service.log"
WORKER_PIDFILE = ROOT / "runtime/model-worker.pid"
PYTHON = ROOT / ".venv/bin/python"
HOST = "127.0.0.1"
BIND_HOST = "0.0.0.0"
PORT = 31000
ENV_FILE = ROOT / "service.env"
if ENV_FILE.is_file():
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

STOP_TIMEOUT = float(os.environ.get("OPENJEV_STOP_TIMEOUT", "10"))
START_TIMEOUT = float(os.environ.get("OPENJEV_START_TIMEOUT", "180"))


def pid():
    try:
        value = int(PIDFILE.read_text())
        proc = Path(f"/proc/{value}")
        if (proc / "stat").read_text().split(")", 1)[1].split()[0] == "Z":
            return None
        cmd = (proc / "cmdline").read_bytes()
        if (proc / "cwd").resolve() == ROOT and b"uvicorn" in cmd and (b"server:app" in cmd or b"openjev.server:app" in cmd):
            return value
    except (FileNotFoundError, ValueError, PermissionError, ProcessLookupError):
        pass
    return None


def health():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{HOST}:{PORT}/health", timeout=2) as r:
        return json.load(r)


def alive(value):
    try:
        return Path(f"/proc/{value}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def worker_pid():
    try:
        value = int(WORKER_PIDFILE.read_text())
        proc = Path(f"/proc/{value}")
        if alive(value) and (proc / "cwd").resolve() == ROOT and b"spawn_main" in (proc / "cmdline").read_bytes():
            return value
    except (FileNotFoundError, ValueError, PermissionError, ProcessLookupError):
        pass
    return None


def start():
    current = pid()
    if current:
        print(f"Already running: PID {current}", flush=True)
        try:
            print(json.dumps(health()), flush=True)
        except Exception:
            raise SystemExit("Process exists but is not ready; use restart to recover")
        return
    if not PYTHON.is_file():
        raise SystemExit("Missing .venv/bin/python")
    with socket.socket() as s:
        s.settimeout(1)
        if s.connect_ex((HOST, PORT)) == 0:
            raise SystemExit(f"Port {PORT} is already used by another service")
    env = os.environ.copy()
    python_path = f"{ROOT / 'src'}:{ROOT / 'vendor/openjev'}"
    if "PYTHONPATH" in env and env["PYTHONPATH"]:
        python_path = f"{python_path}:{env['PYTHONPATH']}"
    env.update({
        "CUDA_VISIBLE_DEVICES": env.get("OPENJEV_GPU", "0"),
        "PYTHONPATH": python_path,
        "HF_HOME": str(ROOT / "runtime/hf-cache"),
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
    })
    LOGFILE.parent.mkdir(exist_ok=True)
    PIDFILE.parent.mkdir(exist_ok=True)
    with LOGFILE.open("ab") as log:
        child = subprocess.Popen(
            [str(PYTHON), "-m", "uvicorn", "openjev.server:app", "--host", BIND_HOST, "--port", str(PORT), "--workers", "1",
             "--timeout-graceful-shutdown", "5", "--limit-concurrency", "32", "--backlog", "64"],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=log, start_new_session=True,
        )
    PIDFILE.write_text(str(child.pid))
    print(f"Started PID {child.pid}, GPU {env['CUDA_VISIBLE_DEVICES']}; waiting for readiness", flush=True)
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise SystemExit(f"Server exited; inspect {LOGFILE}")
        try:
            print(json.dumps(health()), flush=True)
            return
        except Exception:
            time.sleep(0.5)
    stop()
    raise SystemExit(f"Startup exceeded {START_TIMEOUT:g}s; stopped unsuccessful startup; inspect {LOGFILE}")


def stop():
    current = pid()
    worker = worker_pid()
    if not current and not worker:
        PIDFILE.unlink(missing_ok=True)
        WORKER_PIDFILE.unlink(missing_ok=True)
        print("Not running", flush=True)
        return
    owned_group = current is not None and os.getpgid(current) == current
    tracked = [value for value in (current, worker) if value is not None]
    print(f"Stopping service PID {current}, model PID {worker}; graceful wait up to {STOP_TIMEOUT:g}s", flush=True)
    for value in tracked:
        try:
            os.kill(value, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + STOP_TIMEOUT
    while time.monotonic() < deadline:
        if not any(alive(value) for value in tracked):
            PIDFILE.unlink(missing_ok=True)
            WORKER_PIDFILE.unlink(missing_ok=True)
            print("Stopped", flush=True)
            return
        time.sleep(0.1)
    if any(alive(value) for value in tracked):
        print("Graceful stop timed out; terminating this service and its model worker", flush=True)
        if owned_group:
            try:
                os.killpg(current, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for value in tracked:
            if alive(value):
                try:
                    os.kill(value, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not any(alive(value) for value in tracked):
            PIDFILE.unlink(missing_ok=True)
            WORKER_PIDFILE.unlink(missing_ok=True)
            print("Stopped", flush=True)
            return
        time.sleep(0.1)
    raise SystemExit("Process did not exit after SIGKILL; inspect host/GPU driver")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "stop", "restart", "status", "logs"])
    args = ap.parse_args()
    guard = None
    if args.action in {"start", "stop", "restart"}:
        PIDFILE.parent.mkdir(exist_ok=True)
        guard = (ROOT / "runtime/service-control.lock").open("a")
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another start/stop/restart is running; check status or wait for that command")
    if args.action == "restart":
        stop()
        start()
    elif args.action == "start":
        start()
    elif args.action == "stop":
        stop()
    elif args.action == "logs":
        subprocess.run(["tail", "-n", "100", str(LOGFILE)], check=True)
    else:
        current = pid()
        if not current:
            print("Not running")
            raise SystemExit(1)
        print(f"PID {current}")
        try:
            print(json.dumps(health()))
        except Exception:
            print(f"Not ready; inspect {LOGFILE}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
