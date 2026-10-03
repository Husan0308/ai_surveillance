"""Single supervised camera-only application owner (F3); no AI is implied."""
from __future__ import annotations

from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx

from services.shared.deployment import load_deployment
from services.shared.runtime_python import preflight_python

ROOT = Path(__file__).resolve().parents[1]


def check_ports(deployment) -> None:
    for host, port in ((deployment.api_host, deployment.api_port), (deployment.ml_host, deployment.ml_port)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
            except OSError as exc:
                raise RuntimeError(f"service address {host}:{port} is occupied; no process was stopped") from exc


def main() -> int:
    python = preflight_python(ROOT)
    deployment = load_deployment()  # Resolve before credential dotenv loading.
    check_ports(deployment)
    from services.ml_service.app.config import load_settings
    from services.camera_v11.source_ownership import SourceOwnership
    cameras = load_settings().cameras
    if tuple(c.camera_id for c in cameras) != tuple(f"CAM-{i:02d}" for i in range(1, 7)):
        raise ValueError("full stack requires the canonical six enabled cameras")
    if any(not c.username or not c.password for c in cameras):
        raise ValueError("authorized camera credentials are missing")
    locks = ROOT / ".runtime/camera-owner-locks"
    with SourceOwnership(tuple(c.camera_id for c in cameras), locks):
        pass  # No RTSP is opened during preflight; producers acquire lifetime locks.
    platform = json.loads((ROOT / "config/deepstream-platform.json").read_text())
    subprocess.run(["docker", "image", "inspect", platform["image"]], stdout=subprocess.DEVNULL, check=True)
    if not os.getenv("DISPLAY") and not os.getenv("WAYLAND_DISPLAY"):
        raise RuntimeError("a real display is required for the production wall")
    lock_file = open("/tmp/ai_surveillance_full_live_stack.lock", "a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another full stack supervisor is running") from exc
    output = Path(os.getenv("FULL_STACK_OUT", str(ROOT / ".runtime" / time.strftime("full-stack-%Y%m%dT%H%M%SZ", time.gmtime())))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    (output / "room-pair").mkdir()  # Empty session root; never expose old replay identities.
    started = time.monotonic_ns()
    env = dict(os.environ, **deployment.environment(), SURVEILLANCE_ANALYTICS_ENABLED="0",
        MV3DT_ROOM_RUNTIME_ROOT=str(output / "room-pair"),
        V11_MONITORING_TELEMETRY_PATH=str(output / "monitoring.json"),
        FRONTEND_USE_V11_SHARED_MEMORY="1", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    (output / "launch.json").write_text(json.dumps({"deployment": asdict(deployment),
        "session_start_ns": started, "session_id": output.name, "python": str(python),
        "analytics_enabled": False, "port_8000_owner": "AutoMagicCalib",
        "surveillance_api_port": deployment.api_port, "surveillance_ml_port": deployment.ml_port,
        "reason": "existing AMC ownership preserved"}, indent=2) + "\n")
    children: dict[str, subprocess.Popen] = {}
    logs = []
    running = True

    def stop(*_):
        nonlocal running
        running = False

    def spawn(name: str, args: list[str]) -> None:
        log = (output / "logs" / f"{name}.log").open("x")
        logs.append(log)
        children[name] = subprocess.Popen([str(python), "-B", "-u", *args], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, close_fds=True)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    result = 0
    try:
        spawn("preview", ["-m", "scripts.run_preview_foundation", "--output", str(output / "preview")])
        spawn("telemetry", ["-m", "services.camera_v11.preview_monitoring_bridge", "--stats",
            str(output / "preview/camera-health.json"), "--output", str(output / "monitoring.json"),
            "--session-start-ns", str(started), "--session-id", output.name])
        spawn("ml", ["-m", "services.ml_service.app.main"])
        spawn("api", ["-m", "services.api_service.app.main"])
        # UI starts while connecting; it does not wait on analytics or camera readiness.
        if os.getenv("FULL_STACK_UI_EVIDENCE"):
            spawn("frontend", ["-m", "scripts.validate_production_wall", "--output", str(output / "ui.jsonl")])
        else:
            spawn("frontend", ["-m", "services.frontend.app.main"])
        (output / "processes.json").write_text(json.dumps({k: p.pid for k, p in children.items()}, indent=2))
        deadline = time.monotonic() + 90
        ready_written = False
        while running:
            failed = [k for k, p in children.items() if p.poll() is not None]
            if failed:
                if failed == ["frontend"] and children["frontend"].returncode == 0:
                    break
                raise RuntimeError(f"owned child exited: {failed}")
            if not ready_written:
                try:
                    ml = httpx.get(deployment.ml_service_url + "/health", timeout=1).json()
                    api = httpx.get(deployment.frontend_api_base_url + "/health", timeout=1).json()
                    if ml.get("status") == api.get("status") == "ok" and ml.get("online_camera_count") == 6:
                        (output / "ready.json").write_text(json.dumps({"monotonic_ns": time.monotonic_ns(),
                            "ml": ml, "api": api}, indent=2))
                        ready_written = True
                except (httpx.HTTPError, ValueError):
                    pass
                if not ready_written and time.monotonic() > deadline:
                    raise RuntimeError("six fresh advancing producers / service health did not become ready")
            time.sleep(0.2)
    except Exception as exc:
        print(f"FULL_STACK FAIL: {exc}", file=sys.stderr, flush=True)
        result = 1
    finally:
        # Stop consumers first. TERM reaches docker's attached process and its
        # worker, which owns and closes each decoder before releasing locks.
        for name in ("frontend", "api", "ml", "telemetry", "preview"):
            proc = children.get(name)
            if proc is None or proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                result = 1
        for log in logs:
            log.close()
        (output / "shutdown.json").write_text(json.dumps({"status": "PASS" if result == 0 else "FAIL",
            "children": {k: p.poll() for k, p in children.items()}, "monotonic_ns": time.monotonic_ns()}, indent=2))
        lock_file.close()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
