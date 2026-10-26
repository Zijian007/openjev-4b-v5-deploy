"""Manage only this checkout's original OpenJev SGLang process group (Linux)."""
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

ROOT = Path(__file__).resolve().parents[2]
PIDFILE = ROOT / "runtime/sglang-service.json"
LOGFILE = ROOT / "logs/sglang.log"
PYTHON = ROOT / ".venv-sglang/bin/python"
PORT = 30000
START_TIMEOUT = float(os.environ.get("SGLANG_START_TIMEOUT", "600"))
STOP_TIMEOUT = float(os.environ.get("SGLANG_STOP_TIMEOUT", "10"))


def process_info(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pgid": int(fields[2]), "started": fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def group_members(pgid):
    return [int(p.name) for p in Path("/proc").iterdir()
            if p.name.isdigit() and (info := process_info(int(p.name))) and info["pgid"] == pgid]


def record():
    try:
        value = json.loads(PIDFILE.read_text())
        pid = value["pid"]
        if type(pid) is not int or pid <= 1 or not isinstance(value["started"], str):
            raise ValueError("Invalid process identity")
        return value
    except FileNotFoundError:
        return None


def owned_members(value):
    pid = value["pid"]
    members = group_members(pid)
    if not members:
        return []
    info = process_info(pid)
    if info and (info["started"] != value["started"] or info["pgid"] != pid):
        raise SystemExit("PID identity changed; refusing to signal another process")
    for member in members:
        try:
            if Path(f"/proc/{member}/cwd").resolve() != ROOT:
                raise SystemExit("Process group is outside this checkout; refusing to stop")
        except FileNotFoundError:
            continue
    if info:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        if b"sglang.launch_server" not in command or str(PYTHON).encode() not in command:
            raise SystemExit("Process is not this checkout's SGLang server; refusing to stop")
    return members


def health():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{PORT}/health", timeout=3) as response:
        if response.status != 200:
            raise RuntimeError("SGLang not ready")


def stop():
    value = record()
    if value is None:
        print("Not running (no process record)", flush=True)
        return
    members = owned_members(value)
    if members:
        print(f"Stopping SGLang PID {value['pid']} and its model workers", flush=True)
        try:
            os.killpg(value["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + STOP_TIMEOUT
        while time.monotonic() < deadline and group_members(value["pid"]):
            time.sleep(0.1)
        if owned_members(value):
            try:
                os.killpg(value["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and group_members(value["pid"]):
            time.sleep(0.1)
        if group_members(value["pid"]):
            raise SystemExit("SGLang did not exit; inspect GPU driver/process state")
    PIDFILE.unlink(missing_ok=True)
    print("Stopped", flush=True)


def start():
    value = record()
    if value and owned_members(value):
        if not process_info(value["pid"]):
            raise SystemExit("Model workers remain; use stop before starting")
        health()
        print(f"Already running: PID {value['pid']}", flush=True)
        return
    if not PYTHON.is_file() or not (ROOT / "model-overlay/config.json").is_file():
        raise SystemExit("Install the SGLang environment and run download_model.py first")
    with socket.socket() as connection:
        connection.settimeout(1)
        if connection.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit("Port 30000 occupied; refusing duplicate startup")
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": env.get("SGLANG_GPU", "0"), "PY": str(PYTHON),
        "MEM_FRACTION": env.get("SGLANG_MEM_FRACTION", "0.60"),
        "HF_HOME": str(ROOT / "runtime/hf-cache"), "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    })
    command = ["bash", str(ROOT / "vendor/openjev/serve_sglang.sh"),
               str(ROOT / "model-overlay"), str(PORT), "--context-length", "8192",
               "--max-running-requests", "4"]
    LOGFILE.parent.mkdir(exist_ok=True)
    with LOGFILE.open("ab") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True)
    info = process_info(child.pid)
    if info is None:
        raise SystemExit(f"SGLang exited; inspect {LOGFILE}")
    PIDFILE.write_text(json.dumps({"pid": child.pid, "started": info["started"]}) + "\n")
    print(f"Started PID {child.pid}, GPU {env['CUDA_VISIBLE_DEVICES']}; waiting for readiness", flush=True)
    try:
        deadline = time.monotonic() + START_TIMEOUT
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f"SGLang exited; inspect {LOGFILE}")
            try:
                health()
                print("SGLang ready on port 30000", flush=True)
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError(f"Startup exceeded {START_TIMEOUT:g}s; inspect {LOGFILE}")
    except BaseException:
        stop()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "restart", "status", "logs"))
    args = parser.parse_args()
    guard = None
    if args.action in ("start", "stop", "restart"):
        PIDFILE.parent.mkdir(exist_ok=True)
        guard = (ROOT / "runtime/sglang-control.lock").open("a")
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another SGLang control command is running")
    if args.action == "restart":
        stop()
        start()
    elif args.action == "start":
        start()
    elif args.action == "stop":
        stop()
    elif args.action == "logs":
        if LOGFILE.exists():
            subprocess.run(["tail", "-n", "100", str(LOGFILE)], check=True)
        else:
            print("No SGLang log yet")
    else:
        value = record()
        if not value or not owned_members(value) or not process_info(value["pid"]):
            print("Not running (use stop if model workers remain)")
            raise SystemExit(1)
        health()
        print(f"Ready: PID {value['pid']}, port {PORT}")


if __name__ == "__main__":
    main()
