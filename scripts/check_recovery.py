"""Exercise real client disconnects and frozen worker/manager recovery."""
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:31000"
BODY = {"state": "A man is playing a guitar.", "questions": {
    "activity": {"type": "choice", "instructions": "What is the man doing?", "criteria": {"A": "Making music", "B": "Swimming"}}}}
records = []


def record(name, result):
    records.append({"name": name, "result": result})
    print("PASS", name, json.dumps(result), flush=True)


def health():
    response = requests.get(URL + "/health", timeout=2)
    return response.status_code, response.json()


def ready(timeout=60, previous_worker=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            code, result = health()
            if code == 200 and result["status"] == "ok" and (previous_worker is None or result["worker_pid"] != previous_worker):
                return result
        except requests.RequestException:
            pass
        time.sleep(0.1)
    raise AssertionError("Service did not recover within deadline")


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def freeze_worker():
    value = ready()["worker_pid"]
    proc = Path(f"/proc/{value}")
    assert (proc / "cwd").resolve() == ROOT
    assert "spawn_main" in (proc / "cmdline").read_text()
    os.kill(value, signal.SIGSTOP)
    return value


def client():
    source = "import requests; requests.post('" + URL + "/v1/systemone',json=" + repr(BODY) + ",timeout=60).raise_for_status()"
    return subprocess.Popen([sys.executable, "-c", source], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def active(timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, result = health()
        if "active_request_seconds" in result:
            return result
        time.sleep(0.05)
    raise AssertionError("Request was not dispatched")


def normal():
    response = requests.post(URL + "/v1/systemone", json=BODY, timeout=30)
    assert response.status_code == 200, response.text
    assert response.json()["answers"]["activity"]["choice"] == "A", response.text
    return response.json()


def control(action, env=None, timeout=90):
    start = time.perf_counter()
    result = subprocess.run([sys.executable, "service.py", action], cwd=ROOT,
                            env=env, timeout=timeout, capture_output=True, text=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return {"seconds": round(time.perf_counter() - start, 3), "output": result.stdout.strip()}


# 1. A frozen GPU worker must not freeze /health. Ctrl+C must abandon the job.
old_worker = freeze_worker()
caller = client()
try:
    active()
    start = time.perf_counter()
    code, state = health()
    elapsed = time.perf_counter() - start
    assert code == 200 and elapsed < 1, (code, elapsed)
    record("health responds while worker is frozen", {"seconds": round(elapsed, 3)})
    caller.send_signal(signal.SIGINT)
    caller.wait(timeout=5)
    state = ready(previous_worker=old_worker)
    assert not alive(old_worker), old_worker
    record("Ctrl+C cancels running job and reloads worker", {"old_worker": old_worker, "new_worker": state["worker_pid"]})
    record("ordinary request works after cancellation", normal())
finally:
    if caller.poll() is None:
        caller.kill()
        caller.wait(timeout=5)

# 2. Stop while there is an unresponsive calculation; then start again.
old_worker = freeze_worker()
caller = client()
try:
    active()
    result = control("stop", timeout=20)
    assert result["seconds"] < 15, result
    assert not alive(old_worker), old_worker
    record("stop completes during frozen calculation", result)
finally:
    if caller.poll() is None:
        caller.kill()
    caller.wait(timeout=5)
record("start after cancelling frozen calculation", control("start"))
record("ordinary request works after start", normal())

# 3. Test the hard fallback with even the HTTP manager itself suspended.
parent = int((ROOT / "runtime/service.pid").read_text())
worker = ready()["worker_pid"]
assert Path(f"/proc/{parent}/cwd").resolve() == ROOT
os.kill(parent, signal.SIGSTOP)
result = control("stop", timeout=20)
assert result["seconds"] < 15, result
assert not alive(parent) and not alive(worker), (parent, worker)
record("forced stop of frozen manager removes GPU child", result)

# 4. Use short TEST limits and freeze the worker, proving real timeouts.
test_env = os.environ.copy()
test_env.update(OPENJEV_REQUEST_TIMEOUT="2", OPENJEV_QUEUE_TIMEOUT="0.5")
record("start with short test limits", control("start", env=test_env))
worker = freeze_worker()
with ThreadPoolExecutor(max_workers=1) as pool:
    running = pool.submit(requests.post, URL + "/v1/systemone", json=BODY, timeout=15)
    active()
    response = requests.post(URL + "/v1/systemone", json=BODY, timeout=5)
    assert response.status_code == 503, response.text
    assert "0.5s" in response.json()["error"]["message"], response.text
    record("queued request has bounded wait", response.json())
    response = running.result(timeout=15)
    assert response.status_code == 504, response.text
    record("frozen calculation times out clearly", response.json())
ready(previous_worker=worker)
assert not alive(worker), worker
record("ordinary request works after timeout recovery", normal())

# Restore production defaults, not the short test configuration.
record("restart with production limits", control("restart"))
state = ready()
assert state["request_timeout_seconds"] == 180 and state["queue_timeout_seconds"] == 30, state
record("ordinary request works after restart", normal())
output = {"passed": True, "checks": records, "final_health": state}
(ROOT / "verification-recovery.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
print("ALL RECOVERY CHECKS PASSED", flush=True)
