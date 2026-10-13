#!/usr/bin/env python3
"""Exercise a verified six-camera candidate against loopback-only RTSP feeds.

This is an offline integration test. It never reads camera credentials and
never opens a connection to a physical camera.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PORTS = tuple(range(18554, 18560))
PATTERNS = ("smpte", "snow", "circular", "smpte75", "zone-plate", "pinwheel")
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sources(path: Path) -> list[str]:
    uris = [f"rtsp://127.0.0.1:{port}/synthetic" for port in PORTS]
    lines: list[str] = []
    for index, uri in enumerate(uris, 1):
        lines.extend((
            f"[CAM-{index:02d}]",
            f"uri={base64.b64encode(uri.encode()).decode()}",
            "username=", "password=", "latency_ms=0", "drop_on_latency=false",
            "decoder_low_latency_mode=false", "decoder_extra_surfaces=4", "",
        ))
    lines.extend(("[validation]", "interrupt_camera=none", "interrupt_at=20",
                  "interrupt_seconds=12", "preview_enabled=false", "preview_port=5600", ""))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write("\n".join(lines))
    return uris


def run_capture(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, text=True, capture_output=True, **kwargs)


def used_gpu_memory_mib() -> int:
    result = run_capture([
        "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
    ])
    return int(result.stdout.strip().splitlines()[0])


def established_to(port: int) -> list[str]:
    result = subprocess.run(
        ["ss", "-Htnp", "state", "established", f"( dport = :{port} )"],
        check=True, capture_output=True, text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def read_monitoring(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


FPS_COUNTERS = {
    "source_input": "analytics_input_frames_total",
    "decoded": "decoded_frames",
    "pgie": "pgie_frames_total",
    "tracker": "canonical_tracker_updates_total",
    "preview": "preview_frames_total",
}


def summarize_steady_state_fps(observations: list[dict], cameras: tuple[str, ...],
                               start_sec: float, end_sec: float,
                               min_span_sec: float = 20.0) -> dict:
    """Measure sustained throughput from counter deltas while the candidate is live.

    Do not use ``source_fps`` from the final telemetry snapshot: orderly shutdown
    stops frame counters before the process exits, so that instantaneous sample
    is intentionally stale and can be arbitrarily low. Planned outage intervals
    remain represented by the separate outage result; this window proves recovery
    to normal sustained throughput after those outages.
    """
    selected = [row for row in observations
                if start_sec <= float(row.get("monotonic_sec", -1)) <= end_sec]
    result: dict[str, dict] = {}
    for camera in cameras:
        rows = [row for row in selected if camera in row.get("cameras", {})]
        rows.sort(key=lambda row: float(row["monotonic_sec"]))
        if len(rows) < 2:
            result[camera] = {"status": "FAIL", "reason": "insufficient_active_window_samples",
                              "sample_count": len(rows)}
            continue
        first, last = rows[0], rows[-1]
        span = float(last["monotonic_sec"]) - float(first["monotonic_sec"])
        camera_rows = [row["cameras"][camera] for row in rows]
        all_fresh_online = all(
            row.get("online") is True
            and row.get("current_state") == "ONLINE"
            and all(float(row.get(field, float("inf"))) <= 2000.0 for field in (
                "age_since_last_decoded_frame_ms", "age_since_last_pgie_frame_ms",
                "age_since_last_tracker_output_ms", "age_since_last_preview_ms"))
            for row in camera_rows
        )
        rates: dict[str, float | None] = {}
        for stage, field in FPS_COUNTERS.items():
            start_value = first["cameras"][camera].get(field)
            end_value = last["cameras"][camera].get(field)
            if start_value is None or end_value is None or span <= 0 or end_value < start_value:
                rates[stage] = None
            else:
                rates[stage] = round((int(end_value) - int(start_value)) / span, 3)
        rates_pass = span >= min_span_sec and all(
            value is not None and 18.0 <= value <= 22.0 for value in rates.values())
        result[camera] = {
            "status": "PASS" if all_fresh_online and rates_pass else "FAIL",
            "sample_count": len(rows),
            "window_sec": [round(float(first["monotonic_sec"]), 3),
                           round(float(last["monotonic_sec"]), 3)],
            "window_span_sec": round(span, 3),
            "all_samples_fresh_online": all_fresh_online,
            "stage_fps": rates,
        }
    return {
        "window_sec": [start_sec, end_sec],
        "minimum_span_sec": min_span_sec,
        "per_camera": result,
        "pass": len(result) == len(cameras) and all(
            row.get("status") == "PASS" for row in result.values()),
    }


def parse_fixture_outage_plan(value: str, duration_sec: int) -> list[dict[str, int | str]]:
    """Parse loopback RTSP server outages: CAM-XX@START+DOWN_SECONDS."""
    if not value:
        return []
    plan: list[dict[str, int | str]] = []
    previous_second = -1
    previous_by_camera: dict[str, tuple[int, int]] = {}
    for item in value.split(","):
        match = re.fullmatch(r"(CAM-0[1-6])@([0-9]+)\+([0-9]+)", item)
        if not match:
            raise ValueError("fixture outage entries must be CAM-XX@START+DOWN_SECONDS")
        camera_id, raw_start, raw_down = match.groups()
        start_sec, down_sec = int(raw_start), int(raw_down)
        if start_sec < 20 or down_sec < 1 or down_sec > 1200:
            raise ValueError("fixture outage start must be >=20s and downtime 1..1200s")
        if start_sec < previous_second:
            raise ValueError("fixture outages must be ordered by start time")
        previous = previous_by_camera.get(camera_id)
        if previous and start_sec < previous[0] + previous[1] + 30:
            raise ValueError("same-camera fixture outages require 30s recovery space")
        plan.append({"camera_id": camera_id, "at_sec": start_sec, "down_sec": down_sec})
        previous_second = start_sec
        previous_by_camera[camera_id] = (start_sec, down_sec)
    # A long outage can accumulate the bounded 60s retry backoff; the test
    # needs room for that retry plus the 15s all-stage recovery confirmation.
    if plan[-1]["at_sec"] + plan[-1]["down_sec"] + 75 > duration_sec:
        raise ValueError("test duration must include up to 60s retry backoff and 15s confirmation after the final outage")
    return plan


def run(candidate: Path, output: Path, duration: int, plan: str,
        fixture_outage_plan: str = "") -> int:
    if not output.resolve().is_relative_to(ROOT / ".runtime"):
        raise ValueError("output must be under .runtime")
    if output.exists():
        raise FileExistsError(output)
    if duration < 30:
        raise ValueError("duration must be at least 30 seconds")
    candidate = candidate.resolve()
    from scripts.run_six_camera_v13_wall_candidate import (
        CAMERAS, IMAGE as PINNED_IMAGE, docker_command, parse_source_stall_test_plan, validate_candidate,
        validate_yolo_context_verifier,
    )
    from services.camera_v11.source_ownership import SourceOwnership

    if IMAGE != PINNED_IMAGE:
        raise RuntimeError("test image differs from the candidate launcher's pinned image")
    manifest = validate_candidate(candidate)
    validate_yolo_context_verifier(candidate, manifest)
    plan_rows = parse_source_stall_test_plan(plan, "1")
    outage_rows = parse_fixture_outage_plan(fixture_outage_plan, duration)
    if plan_rows and outage_rows:
        raise ValueError("choose either source-state stalls or loopback RTSP fixture outages, not both")
    expected_cycle = ["TEST_SOURCE_STALL_BEGIN", "SOURCE_STALL_RECOVERY_BEGIN",
                      "SOURCE_STALL_OLD_OWNER_REMOVED", "SOURCE_STALL_RECOVERY_RECONNECTED",
                      "SOURCE_STALL_RECOVERY_PASS"]
    binary = candidate / "bin/cam-six-yolo-local6-posttracker"
    binary_hash = sha256(binary)
    image_id = run_capture(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"]).stdout.strip()
    if image_id != IMAGE.replace("nvcr.io/nvidia/deepstream@", ""):
        raise RuntimeError("local DeepStream image ID does not match the pinned digest")

    output.mkdir(parents=True)
    cfg = output / "six-synthetic.ini"
    uris = write_sources(cfg)
    fixture_logs = [(output / f"fixture-{port}.log").open("x") for port in PORTS]
    fixture_processes: list[subprocess.Popen | None] = [None] * len(PORTS)
    supervisor_log_path = output / "native.log"
    monitor_path = output / "monitoring.json"
    resource_path = output / "resource-samples.jsonl"
    event_path = output / "recovery-events.jsonl"
    started: list[dict] = []
    max_gpu_mib = 0
    max_port_owners = {port: 0 for port in PORTS}
    native_offset = 0
    events: list[dict] = []
    observations: list[dict] = []
    result_code = 1
    stop_reason = "native candidate did not complete"
    container = f"surv-synthetic-{output.name}"[:120]

    for port in PORTS:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"synthetic RTSP fixture port {port} is occupied") from exc
    if run_capture(["ss", "-Htnp", "state", "established", "( dport = :554 )"]).stdout.strip():
        raise RuntimeError("physical RTSP sessions are active; refusing isolated test")
    for port in (8000, 8100, 8101):
        if run_capture(["ss", "-Hlnpt", f"( sport = :{port} )"]).stdout.strip() == "" and port == 8000:
            raise RuntimeError("AMC listener on port 8000 is not present")
    for port in (8100, 8101):
        if run_capture(["ss", "-Hlnpt", f"( sport = :{port} )"]).stdout.strip():
            raise RuntimeError(f"surveillance port {port} is occupied before isolated test")

    env_before = os.environ.copy()
    env_before.update({
        "V11_SOURCE_STALL_RECOVERY": "1",
        "V11_TEST_SOURCE_STALL_PLAN": plan,
        "V11_VERBOSE_EVIDENCE": "1",
    })
    from scripts.run_source_stall_recovery_acceptance import recovery_verifier_environment
    env_before.update(recovery_verifier_environment(manifest))
    old_environment = os.environ.copy()
    os.environ.update(env_before)
    try:
        docker_argv = docker_command(candidate, output, cfg, duration, yolo_context_verifier=True)
    finally:
        os.environ.clear()
        os.environ.update(old_environment)
    # Keep the test shared-memory region private to this container. Do not
    # unlink or overwrite the normal host UI's six shared-memory slots.
    private_shm_mount = docker_argv.index("type=bind,src=/dev/shm,dst=/dev/shm")
    del docker_argv[private_shm_mount - 1:private_shm_mount + 1]
    docker_argv[docker_argv.index("--network=host") + 1:docker_argv.index("--network=host") + 1] = ["--shm-size=512m"]
    docker_argv[docker_argv.index("-e"):docker_argv.index("-e")] = [
        "-e", "V11_MONITORING_TELEMETRY_PATH=/work/monitoring.json",
    ]
    docker_argv[docker_argv.index("--name") + 1] = container
    write_json = lambda p, obj: p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    write_json(output / "preflight.json", {
        "test_scope": "offline synthetic RTSP only; no physical sources",
        "binary_sha256": binary_hash,
        "source_snapshot_tree_sha256": manifest["source_snapshot"]["tree_sha256"],
        "source_manifest_sha256": manifest["source_snapshot"]["manifest_sha256"],
        "deepstream_image_id": image_id,
        "duration_sec": duration,
        "planned_stall": plan,
        "planned_fixture_outages": outage_rows,
        "sources": [{"camera_id": camera, "uri": uri, "port": port, "pattern": PATTERNS[index]}
                    for index, (camera, uri, port) in enumerate(zip(CAMERAS, uris, PORTS))],
        "gpu_memory_before_mib": used_gpu_memory_mib(),
        "surveillance_ports_8100_8101": "free",
        "amc_port_8000": "listening; not modified",
        "rtsp_port_554_sessions_before": 0,
    })
    with SourceOwnership(CAMERAS, ROOT / ".runtime/camera-owner-locks"):
        try:
            def start_fixture(index: int) -> subprocess.Popen:
                proc = subprocess.Popen(
                    ["/usr/bin/python3", str(ROOT / "scripts/local_rtsp_test_fixture.py"),
                     "--port", str(PORTS[index]), "--pattern", PATTERNS[index]],
                    cwd=ROOT, stdout=fixture_logs[index], stderr=subprocess.STDOUT,
                )
                fixture_processes[index] = proc
                return proc

            def wait_fixture_ready(index: int, timeout: float = 10.0) -> None:
                proc = fixture_processes[index]
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if proc is None or proc.poll() is not None:
                        raise RuntimeError(f"loopback RTSP fixture {PORTS[index]} exited before ready")
                    listener = subprocess.run(
                        ["ss", "-Hlnpt", f"( sport = :{PORTS[index]} )"],
                        check=True, capture_output=True, text=True,
                    ).stdout
                    if listener.strip():
                        return
                    time.sleep(0.05)
                raise TimeoutError(f"loopback RTSP fixture {PORTS[index]} did not become ready")

            for index, port in enumerate(PORTS):
                start_fixture(index)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if all(proc.poll() is None for proc in fixture_processes) and all(
                    f"LOCAL_RTSP_FIXTURE_READY port={port}" in (output / f"fixture-{port}.log").read_text(errors="replace")
                    for port in PORTS
                ):
                    break
                if any(proc.poll() is not None for proc in fixture_processes):
                    raise RuntimeError("a loopback RTSP fixture failed to start; inspect fixture logs")
                time.sleep(0.1)
            else:
                raise TimeoutError("not all six loopback RTSP fixtures became ready")

            with supervisor_log_path.open("x") as native_log:
                process = subprocess.Popen(
                    docker_argv, cwd=ROOT, stdout=native_log, stderr=subprocess.STDOUT,
                )
                run_start = time.monotonic()
                with resource_path.open("x") as resource_out, event_path.open("x") as event_out:
                    while time.monotonic() - run_start < duration + 90:
                        if process.poll() is not None:
                            result_code = process.returncode
                            stop_reason = "candidate process exited"
                            break
                        now = time.monotonic()
                        elapsed = now - run_start
                        for outage in outage_rows:
                            camera_id = str(outage["camera_id"])
                            index = CAMERAS.index(camera_id)
                            scheduled = int(outage["at_sec"])
                            end_at = scheduled + int(outage["down_sec"])
                            if elapsed >= scheduled and "stop_elapsed_sec" not in outage:
                                proc = fixture_processes[index]
                                if proc is None or proc.poll() is not None:
                                    raise RuntimeError(f"fixture {camera_id} was not running at outage start")
                                all_rows = {item.get("camera_id"): item for item in
                                            (read_monitoring(monitor_path) or {}).get("cameras", [])}
                                row = all_rows.get(camera_id, {})
                                outage["baseline_source_epoch"] = int(row.get("source_epoch") or 0)
                                outage["baseline_generation"] = int(row.get("source_generation") or 0)
                                outage["baseline_decoded_frames"] = int(row.get("decoded_frames") or 0)
                                outage["baseline_all_cameras"] = {
                                    name: {key: int(values.get(key) or 0) for key in (
                                        "analytics_input_frames_total", "decoded_frames",
                                        "pgie_frames_total", "canonical_tracker_updates_total",
                                        "preview_frames_total")}
                                    for name, values in all_rows.items()
                                }
                                for counter in ("pgie_frames_total", "canonical_tracker_updates_total",
                                                "preview_frames_total"):
                                    outage[f"baseline_{counter}"] = int(row.get(counter) or 0)
                                proc.send_signal(signal.SIGTERM)
                                try:
                                    proc.wait(timeout=5)
                                except subprocess.TimeoutExpired as exc:
                                    raise RuntimeError(f"fixture {camera_id} failed to stop gracefully") from exc
                                outage["stop_elapsed_sec"] = round(elapsed, 3)
                                event_out.write(json.dumps({
                                    "camera_id": camera_id,
                                    "event": "LOOPBACK_RTSP_FIXTURE_STOPPED",
                                    "observed_monotonic_sec": round(elapsed, 3),
                                    "down_sec": outage["down_sec"],
                                }, sort_keys=True) + "\n")
                                event_out.flush()
                            elif ("stop_elapsed_sec" in outage and "restart_elapsed_sec" not in outage
                                  and elapsed >= end_at):
                                start_fixture(index)
                                wait_fixture_ready(index)
                                outage["restart_elapsed_sec"] = round(elapsed, 3)
                                event_out.write(json.dumps({
                                    "camera_id": camera_id,
                                    "event": "LOOPBACK_RTSP_FIXTURE_RESTARTED",
                                    "observed_monotonic_sec": round(elapsed, 3),
                                }, sort_keys=True) + "\n")
                                event_out.flush()
                        gpu_mib = used_gpu_memory_mib()
                        max_gpu_mib = max(max_gpu_mib, gpu_mib)
                        if gpu_mib > 10_500:
                            process.send_signal(signal.SIGINT)
                            stop_reason = f"aborted: GPU memory safety ceiling exceeded ({gpu_mib} MiB)"
                            try:
                                process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                stop_reason += "; graceful stop timed out"
                            result_code = 1
                            break
                        monitoring = read_monitoring(monitor_path)
                        by_camera = {row.get("camera_id"): row for row in (monitoring or {}).get("cameras", [])}
                        socket_rows = {port: established_to(port) for port in PORTS}
                        for port, rows in socket_rows.items():
                            max_port_owners[port] = max(max_port_owners[port], len(rows))
                        observation = {
                            "monotonic_sec": round(now - run_start, 3),
                            "gpu_memory_mib": gpu_mib,
                            "monitor_sequence": (monitoring or {}).get("sequence"),
                            "cameras": {camera: {
                                key: by_camera.get(camera, {}).get(key)
                                for key in ("analytics_input_frames_total", "decoded_frames",
                                            "pgie_frames_total", "canonical_tracker_updates_total",
                                            "preview_frames_total", "queue_depth", "queue_high_water",
                                            "pipeline_errors", "reconnects", "tracking_epoch",
                                            "source_epoch", "source_generation", "current_state",
                                        "online", "age_since_last_decoded_frame_ms",
                                        "age_since_last_pgie_frame_ms", "age_since_last_tracker_output_ms",
                                        "age_since_last_preview_ms", "source_fps", "input_pts_resets",
                                        "decoded_pts_resets")
                            } for camera in CAMERAS},
                            "established_client_sockets_by_fixture_port": {
                                str(port): len(rows) for port, rows in socket_rows.items()},
                        }
                        observations.append(observation)
                        resource_out.write(json.dumps(observation, sort_keys=True) + "\n")
                        resource_out.flush()
                        if monitoring:
                            for camera, row in by_camera.items():
                                if camera not in started and row.get("decoded_frames", 0) > 0:
                                    started.append({"camera_id": camera, "first_decoded_monotonic_sec": observation["monotonic_sec"]})
                        log_path = output / "native.log"
                        if log_path.exists():
                            with log_path.open("r", errors="replace") as stream:
                                stream.seek(native_offset)
                                additions = stream.readlines()
                                native_offset = stream.tell()
                            for line in additions:
                                parts = line.strip().split(maxsplit=2)
                                event_name = parts[1] if len(parts) >= 2 else ""
                                # The native source-local lifecycle emits
                                # SOURCE_RECOVERY_BEGIN; normalize that exact
                                # event into this test report's stable name.
                                if event_name == "SOURCE_RECOVERY_BEGIN":
                                    event_name = "SOURCE_STALL_RECOVERY_BEGIN"
                                if event_name in set(expected_cycle):
                                    row = {"camera_id": parts[0], "event": event_name,
                                           "detail": parts[2] if len(parts) == 3 else "",
                                           "observed_monotonic_sec": observation["monotonic_sec"]}
                                    events.append(row)
                                    event_out.write(json.dumps(row, sort_keys=True) + "\n")
                                    event_out.flush()
                        time.sleep(0.25)
                    else:
                        process.send_signal(signal.SIGINT)
                        stop_reason = "candidate exceeded duration+90s timeout"
                        try:
                            result_code = process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            stop_reason += "; graceful stop timed out"
                            result_code = 124
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        stop_reason += "; graceful stop timed out"
                        result_code = 124

            log_text = supervisor_log_path.read_text(errors="replace")
            cycles_by_camera: dict[str, list[list[str]]] = {}
            for camera in CAMERAS:
                camera_events = [row["event"] for row in events if row["camera_id"] == camera]
                cycles: list[list[str]] = []
                current: list[str] = []
                for name in camera_events:
                    if name == "TEST_SOURCE_STALL_BEGIN" and current:
                        cycles.append(current)
                        current = []
                    current.append(name)
                if current:
                    cycles.append(current)
                cycles_by_camera[camera] = cycles
            planned_counts = {camera: sum(1 for row in plan_rows if row["camera_id"] == camera)
                              for camera in CAMERAS}
            observed_counts = {camera: len(cycles_by_camera[camera]) for camera in CAMERAS}
            native_source_cycle_pass = (observed_counts == planned_counts and
                                        all(cycle == expected_cycle for cycles in cycles_by_camera.values()
                                            for cycle in cycles))
            stall_order_pass = native_source_cycle_pass
            observed_order = [row["event"] for row in events if row["camera_id"] == "CAM-05"]
            end_stats = {}
            import re
            for line in log_text.splitlines():
                match = re.search(r"CAM-(\d\d) (DETECT|TRACK|PREVIEW)\b.*", line)
                if match:
                    end_stats.setdefault(f"CAM-{match.group(1)}", {})[match.group(2).lower()] = line
            final_monitor = read_monitoring(monitor_path) or {}
            final_by_camera = {row.get("camera_id"): row for row in final_monitor.get("cameras", [])}
            counter_pass = all(
                final_by_camera.get(camera, {}).get("decoded_frames", 0) > 0
                and final_by_camera.get(camera, {}).get("pgie_frames_total", 0) > 0
                and final_by_camera.get(camera, {}).get("canonical_tracker_updates_total", 0) > 0
                and final_by_camera.get(camera, {}).get("preview_frames_total", 0) > 0
                for camera in CAMERAS
            )
            fps_window = summarize_steady_state_fps(
                observations, CAMERAS, max(0.0, duration - 30.0),
                max(0.0, duration - 5.0))
            camera_fps = {
                camera: fps_window["per_camera"].get(camera, {}).get("stage_fps", {})
                for camera in CAMERAS
            }
            fps_pass = bool(fps_window["pass"])
            queue_high_water = {
                camera: max((row["cameras"].get(camera, {}).get("queue_high_water") or 0
                             for row in observations), default=0)
                for camera in CAMERAS
            }
            bounded_queues_pass = all(value <= 12 for value in queue_high_water.values())
            errors_pass = all(
                final_by_camera.get(camera, {}).get("pipeline_errors", 1) == 0
                and final_by_camera.get(camera, {}).get("infer_errors", 1) == 0
                and final_by_camera.get(camera, {}).get("preview_errors", 1) == 0
                for camera in CAMERAS
            )
            sync_pass = all(
                "sync_mismatches=0" in end_stats.get(camera, {}).get("preview", "")
                and "errors=0" in end_stats.get(camera, {}).get("preview", "")
                for camera in CAMERAS
            )
            per_source_owner_pass = all(count <= 1 for count in max_port_owners.values())
            early_events = [row for row in [*plan_rows, *outage_rows]
                            if 19 <= int(row["at_sec"]) <= 40]
            early_affected = {str(row["camera_id"]) for row in early_events}
            early_affected_times = [int(row["at_sec"]) for row in early_events]
            window_start = max(0, min(early_affected_times, default=20) - 1)
            early_window_end = max(
                [40, *(int(row["at_sec"]) + int(row.get("down_sec", 0)) + 20
                       for row in early_events)]
            )
            window_end = early_window_end
            recovery_rows = [row for row in observations
                             if window_start <= row["monotonic_sec"] <= window_end]
            unaffected_cameras = [camera for camera in CAMERAS if camera not in early_affected]
            unaffected_pass = True
            for camera in unaffected_cameras:
                values = [row["cameras"].get(camera, {}).get("analytics_input_frames_total")
                          for row in recovery_rows]
                values = [value for value in values if value is not None]
                if len(values) < 2 or values[-1] <= values[0]:
                    unaffected_pass = False
            outage_rows_validation: list[dict] = []
            for outage in outage_rows:
                camera = str(outage["camera_id"])
                stop_at = float(outage.get("stop_elapsed_sec", -1))
                restart_at = float(outage.get("restart_elapsed_sec", -1))
                actual_down = restart_at - stop_at
                next_same_camera = min(
                    (float(row.get("stop_elapsed_sec", row["at_sec"])) for row in outage_rows
                     if row["camera_id"] == camera and int(row["at_sec"]) > int(outage["at_sec"])
                     and row.get("stop_elapsed_sec") is not None),
                    default=float(duration) + 1,
                )
                followup = [
                    row for row in observations
                    if restart_at >= 0 and restart_at <= row["monotonic_sec"] < next_same_camera
                ]
                baseline_fields = {
                    "decoded_frames": "baseline_decoded_frames",
                    "pgie_frames_total": "baseline_pgie_frames_total",
                    "canonical_tracker_updates_total": "baseline_canonical_tracker_updates_total",
                    "preview_frames_total": "baseline_preview_frames_total",
                }
                stage_row = next((sample["cameras"].get(camera, {}) for sample in followup
                    if all((sample["cameras"].get(camera, {}).get(field) or 0) >
                           int(outage.get(baseline, 0)) for field, baseline in baseline_fields.items())
                    and (sample["cameras"].get(camera, {}).get("source_epoch") or 0) >
                        int(outage.get("baseline_source_epoch", 0))
                    and (sample["cameras"].get(camera, {}).get("source_generation") or 0) >
                        int(outage.get("baseline_generation", 0))
                    and sample["cameras"].get(camera, {}).get("current_state") == "ONLINE"), None)
                outage_observations = [
                    sample for sample in observations
                    if stop_at >= 0 and stop_at <= sample["monotonic_sec"] <= restart_at
                ]
                unaffected = [
                    name for name in CAMERAS
                    if name != camera and not any(
                        other["camera_id"] == name
                        and float(other.get("stop_elapsed_sec", -1)) <= restart_at
                        and float(other.get("restart_elapsed_sec", -1)) >= stop_at
                        for other in outage_rows
                    )
                ]
                healthy_peer_progress = bool(outage_observations) and all(
                    len(outage_observations) >= 2 and
                    (outage_observations[-1]["cameras"].get(name, {}).get(
                        "analytics_input_frames_total") or 0) >
                    int(outage.get("baseline_all_cameras", {}).get(name, {}).get(
                        "analytics_input_frames_total", 0))
                    for name in unaffected
                )
                matching_pass_events = [
                    row for row in events
                    if row["camera_id"] == camera and row["event"] == "SOURCE_STALL_RECOVERY_PASS"
                    and row["observed_monotonic_sec"] >= stop_at
                    and row["observed_monotonic_sec"] >= restart_at
                    and int(re.search(r"source_epoch=(\d+)", row["detail"]).group(1))
                        > int(outage.get("baseline_source_epoch", 0))
                    if re.search(r"source_epoch=(\d+)", row["detail"])
                ]
                outage_status = (
                    stop_at >= 0 and restart_at >= 0
                    and abs(actual_down - int(outage["down_sec"])) <= 1.5
                    and stage_row is not None and bool(matching_pass_events)
                    and healthy_peer_progress
                )
                outage["actual_down_sec"] = round(actual_down, 3) if restart_at >= 0 else None
                outage["recovery_event_count_after_restore"] = len(matching_pass_events)
                outage["recovered_stage_snapshot"] = stage_row
                outage["unaffected_cameras"] = unaffected
                outage["unaffected_cameras_advanced_during_outage"] = healthy_peer_progress
                outage["status"] = "PASS" if outage_status else "FAIL"
                outage_rows_validation.append(outage)
            fixture_outages_pass = bool(outage_rows_validation) and all(
                row["status"] == "PASS" for row in outage_rows_validation
            )
            if outage_rows:
                stall_order_pass = fixture_outages_pass
                # Source-local GStreamer error counts are expected for a server outage;
                # shared errors and downstream analytics errors remain forbidden.
                unaffected_error_pass = all(
                    final_by_camera.get(camera, {}).get("pipeline_errors", 1) == 0
                    for camera in CAMERAS if camera not in {str(row["camera_id"]) for row in outage_rows}
                )
                errors_pass = (unaffected_error_pass and
                    all(final_by_camera.get(camera, {}).get("infer_errors", 1) == 0 and
                        final_by_camera.get(camera, {}).get("preview_errors", 1) == 0
                        for camera in CAMERAS) and
                    "shared_errors=0" in (next((line for line in log_text.splitlines()
                       if line.startswith("GROUP END")), "")))
            summary = {
                "status": "PASS" if (result_code == 0 and stall_order_pass and counter_pass
                                      and errors_pass and sync_pass and per_source_owner_pass
                                      and unaffected_pass and fps_pass and bounded_queues_pass) else "FAIL",
                "exit_code": result_code,
                "stop_reason": stop_reason,
                "binary_sha256": binary_hash,
                "source_snapshot_tree_sha256": manifest["source_snapshot"]["tree_sha256"],
                "planned_stall": plan,
                "planned_fixture_outages": outage_rows,
                "recovery_event_order": observed_order,
                "planned_recovery_cycles": planned_counts,
                "observed_recovery_cycles": observed_counts,
                "recovery_cycles_by_camera": cycles_by_camera,
                "stall_order_pass": stall_order_pass,
                "native_source_state_stall_cycle_pass": native_source_cycle_pass,
                "loopback_rtsp_outage_plan": outage_rows_validation,
                "loopback_rtsp_outage_recovery_pass": (fixture_outages_pass if outage_rows else None),
                "all_six_decode_pgie_tracker_preview_advanced": counter_pass,
                "camera_source_fps": camera_fps,
                "steady_state_fps_window": fps_window,
                "camera_fps_18_to_22_pass": fps_pass,
                "camera_queue_high_water": queue_high_water,
                "camera_queues_bounded_pass": bounded_queues_pass,
                "zero_pipeline_infer_preview_errors": errors_pass,
                "preview_sync_mismatches_zero": sync_pass,
                "per_fixture_rtsp_owner_max_one": per_source_owner_pass,
                "max_client_owners_by_port": {str(k): v for k, v in max_port_owners.items()},
                "unaffected_cameras_advanced_during_early_recovery_window": unaffected_pass,
                "unaffected_cameras_checked": unaffected_cameras,
                "affected_cameras_in_early_recovery_window": sorted(early_affected),
                "early_recovery_observation_window_sec": [window_start, window_end],
                "peak_gpu_memory_mib": max_gpu_mib,
                "camera_first_decoded": started,
                "final_monitoring": final_monitor,
                "native_final_stats": end_stats,
                "native_group_end": next((line for line in log_text.splitlines() if line.startswith("GROUP END")), None),
                "native_errors": [line for line in log_text.splitlines() if "ERROR" in line or "FATAL" in line],
            }
            write_json(output / "result.json", summary)
            return 0 if summary["status"] == "PASS" else 1
        finally:
            for process in fixture_processes:
                if process is not None and process.poll() is None:
                    process.send_signal(signal.SIGTERM)
            for process in fixture_processes:
                if process is None:
                    continue
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in fixture_logs:
                stream.close()
            write_json(output / "cleanup.json", {
                "fixture_processes_exited": all(process is None or process.poll() is not None
                                                 for process in fixture_processes),
                "fixture_connections_after_cleanup": {str(port): len(established_to(port)) for port in PORTS},
                "amc_port_8000_listener_preserved": bool(run_capture(["ss", "-Hlnpt", "( sport = :8000 )"]).stdout.strip()),
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=50)
    parser.add_argument("--plan", default="CAM-05@20")
    parser.add_argument("--fixture-outage-plan", default="",
                        help="loopback RTSP server outages CAM-XX@START+DOWN_SECONDS")
    args = parser.parse_args()
    try:
        return run(args.candidate, args.output, args.duration, args.plan,
                   args.fixture_outage_plan)
    except Exception as exc:
        print(f"SYNTHETIC_SIX_CAMERA_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
