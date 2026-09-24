#!/usr/bin/env python3
"""Start, inspect, and stop a Classifast verification instance."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[4]
STATE_NAME = "instance.json"
LOCAL_REDIS_HOST = "127.0.0.1"
LOCAL_REDIS_PORT = 16379


def local_redis_ready() -> bool:
    try:
        with socket.create_connection(
            (LOCAL_REDIS_HOST, LOCAL_REDIS_PORT), timeout=2
        ) as conn:
            conn.settimeout(2)
            conn.sendall(b"*1\r\n$4\r\nPING\r\n")
            return conn.makefile("rb").readline() == b"+PONG\r\n"
    except OSError:
        return False


def process_start(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(") ", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def request(url: str) -> tuple[int, str]:
    try:
        with urlopen(url, timeout=8) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


def own_instance(state: dict) -> bool:
    return process_start(state["pid"]) == state["process_start"]


def load_state(run_dir: Path) -> dict:
    return json.loads((run_dir / STATE_NAME).read_text())


def inspect(state: dict) -> dict:
    if not own_instance(state):
        raise RuntimeError("The recorded server process is gone or its PID was reused")
    base = state["url"]
    page_status, page = request(f"{base}/mapping/")
    health_status, health = request(f"{base}/health")
    page_ready = page_status == 200 and "Mapping tables for cross-referencing" in page
    health_ready = health_status == 200 and '"healthy"' in health
    redis_ready = local_redis_ready() if state["mode"] == "full" else None
    if not page_ready or (
        state["mode"] == "full" and not (health_ready and redis_ready)
    ):
        raise RuntimeError(
            f"Doctor failed: mapping page={page_status}, health={health_status}, "
            f"local Redis ready={redis_ready}, expected mode={state['mode']}"
        )
    return {
        "url": base,
        "pid": state["pid"],
        "mode": state["mode"],
        "revision_at_launch": state["revision"],
        "mapping_page_ready": page_ready,
        "health_status": health_status,
        "health_gate_passed": health_ready,
        "local_redis_ready": redis_ready,
    }


def stop(state: dict) -> None:
    if not own_instance(state):
        return
    os.kill(state["pid"], signal.SIGTERM)
    for _ in range(50):
        if not own_instance(state):
            return
        time.sleep(0.1)
    if own_instance(state):
        os.kill(state["pid"], signal.SIGKILL)


def launch(run_dir: Path, mode: str) -> dict:
    state_path = run_dir / STATE_NAME
    if state_path.exists():
        raise RuntimeError(f"Instance state already exists: {state_path}")
    for asset in ("app/static/js/common.js", "app/static/css/styles.css"):
        if not (ROOT / asset).is_file():
            raise RuntimeError(f"Missing {asset}; run npm run build first")
    if mode == "full" and not local_redis_ready():
        raise RuntimeError(
            f"Start disposable Redis at {LOCAL_REDIS_HOST}:{LOCAL_REDIS_PORT} before full verification"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence = run_dir / "evidence"
    evidence.mkdir(exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    with (evidence / "server.log").open("wb") as log:
        server_env = os.environ.copy()
        if mode == "full":
            server_env.update(
                REDIS_HOST=LOCAL_REDIS_HOST,
                REDIS_PORT=str(LOCAL_REDIS_PORT),
                REDIS_USERNAME="",
                REDIS_PASSWORD="",
            )
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--lifespan",
                "on" if mode == "full" else "off",
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=server_env,
        )
    state = {
        "pid": server.pid,
        "process_start": process_start(server.pid),
        "url": f"http://127.0.0.1:{port}",
        "mode": mode,
        "revision": revision,
    }
    try:
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError(
                    f"Server exited with code {server.returncode}; see {evidence / 'server.log'}"
                )
            try:
                result = inspect(state)
                state_path.write_text(json.dumps(state, indent=2) + "\n")
                return result
            except (RuntimeError, URLError, TimeoutError):
                time.sleep(0.2)
        raise RuntimeError(
            f"Server did not become ready; see {evidence / 'server.log'}"
        )
    except BaseException:
        stop(state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "doctor", "cleanup"))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--mode", choices=("public", "full"), default="full")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.action == "launch":
        result = launch(run_dir, args.mode)
    elif args.action == "doctor":
        result = inspect(load_state(run_dir))
    else:
        state = load_state(run_dir)
        stop(state)
        (run_dir / STATE_NAME).unlink()
        result = {"stopped_pid": state["pid"], "evidence": str(run_dir / "evidence")}
    result["run_dir"] = str(run_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
