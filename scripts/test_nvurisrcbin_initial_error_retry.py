#!/usr/bin/env python3
"""Prove pinned nvurisrcbin does not retry an initial localhost RTSP error."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18671)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / ".runtime") or output.exists():
        raise ValueError("output must be a new directory under repository .runtime")
    if not 1024 <= args.port <= 65535:
        raise ValueError("port must be unprivileged")
    output.mkdir(parents=True)
    runner_log = output / "nvurisrcbin.log"
    fixture_log = output / "fixture.log"
    ss = subprocess.run(["ss", "-Hlnpt", f"( sport = :{args.port} )"], capture_output=True, text=True, check=True)
    if ss.stdout.strip():
        raise RuntimeError(f"loopback test port {args.port} is already listening")

    uid, gid = os.getuid(), os.getgid()
    uri = f"rtsp://127.0.0.1:{args.port}/synthetic"
    command = [
        "docker", "run", "--rm", "--pull=never", "--network=host", "--gpus", "all",
        "--user", f"{uid}:{gid}", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "-v", f"{ROOT}:/workspace:ro", IMAGE, "/usr/bin/python3",
        "/workspace/scripts/inspect_nvurisrcbin_reconnect.py", uri,
        "--duration-sec", "20", "--attempts", "0",
    ]
    fixture = None
    client = None
    started = time.monotonic()
    with runner_log.open("x", encoding="utf-8") as log:
        client = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            # Wait until the probe actually reports the initial connection error,
            # then start the fixture. This cannot accidentally become an initial
            # success followed by an inactivity-reconnect test.
            deadline = time.monotonic() + 12
            initial_error_observed = False
            while time.monotonic() < deadline:
                current = runner_log.read_text(errors="replace")
                if "BUS_ERROR" in current:
                    initial_error_observed = True
                    break
                if client.poll() is not None:
                    break
                time.sleep(0.1)
            with fixture_log.open("x", encoding="utf-8") as fixture_out:
                fixture = subprocess.Popen(
                    ["/usr/bin/python3", str(ROOT / "scripts/local_rtsp_test_fixture.py"),
                     "--port", str(args.port), "--pattern", "ball"],
                    cwd=ROOT, stdout=fixture_out, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if fixture.poll() is not None:
                        raise RuntimeError("localhost RTSP fixture exited before ready")
                    if f"LOCAL_RTSP_FIXTURE_READY port={args.port}" in fixture_log.read_text(errors="replace"):
                        break
                    time.sleep(0.05)
                else:
                    raise TimeoutError("localhost RTSP fixture did not become ready")
            try:
                client_exit = client.wait(timeout=30)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("nvurisrcbin probe did not complete") from exc
        finally:
            if client.poll() is None:
                client.send_signal(signal.SIGINT)
                try:
                    client.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    client.kill()
                    client.wait(timeout=5)
            if fixture is not None and fixture.poll() is None:
                fixture.send_signal(signal.SIGTERM)
                fixture.wait(timeout=5)

    plugin_text = runner_log.read_text(errors="replace")
    fixture_text = fixture_log.read_text(errors="replace") if fixture_log.exists() else ""
    result_line = next((line.removeprefix("RESULT ") for line in plugin_text.splitlines()
                        if line.startswith("RESULT ")), None)
    probe_result = json.loads(result_line) if result_line else {}
    effective = probe_result.get("effective_reconnect_properties", {})
    client_connections = fixture_text.count("RTSP_CLIENT_CONNECTED")
    status = "PASS" if (
        initial_error_observed and client_exit == 0 and
        effective.get("rtsp-reconnect-interval") == 0 and
        effective.get("init-rtsp-reconnect-interval") == 0 and
        effective.get("rtsp-reconnect-attempts") == -1 and
        client_connections == 0
    ) else "FAIL"
    result = {
        "status": status,
        "scope": "pinned nvurisrcbin isolated initial-connection-error test; localhost only",
        "port": args.port,
        "initial_connection_error_observed_before_fixture_start": initial_error_observed,
        "initial_bus_error_line": next((line for line in plugin_text.splitlines()
                                         if line.startswith("BUS_ERROR")), None),
        "fixture_started_after_initial_error": True,
        "fixture_client_connections_after_start": client_connections,
        "effective_gobject_properties": effective,
        "probe_exit_code": client_exit,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "plugin_probe_log": runner_log.name,
        "fixture_log": fixture_log.name,
        "physical_camera_connections": 0,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
