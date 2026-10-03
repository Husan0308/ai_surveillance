"""Opt-in F5 real six-camera application acceptance, no production promotion."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import httpx

# Capture the existing frontend settings before a camera credential loader can
# inject a legacy shared .env. Explicit caller overrides still take precedence.
from services.frontend.app.config import load_settings as load_frontend_settings
FRONTEND_SETTINGS = load_frontend_settings()

from scripts.full_stack_runtime import check_ports
from scripts.freeze_app_contract import DEPLOYMENT, websocket_samples
from scripts.run_room_candidate import ROOT, save
from scripts.run_v13_room_candidate import protected_v13, sidecar_environment, V13_SHA
from scripts.v13_monitoring import MixedMonitoring, OTHER
from services.camera_v11.monitoring_telemetry_ipc_v1 import MonitoringTelemetryWriter, offline_snapshot
from services.camera_v11.source_ownership import SourceOwnership
from services.shared.runtime_python import preflight_python


def percentile(values, q):
    ordered = sorted(values)
    at = (len(ordered) - 1) * q / 100
    low, high = int(at), min(int(at) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


async def current_websocket_samples(url, bridge, writer, collector=websocket_samples, interval=.25):
    """Keep the actual producer observer running during the websocket check."""
    async def publish():
        while True:
            writer.publish(bridge.snapshot())
            await asyncio.sleep(interval)

    task = asyncio.create_task(publish())
    try:
        return await collector(url)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def clean_child_shutdown(name, code, pid, log, forced=False):
    if forced:
        return False
    if code == 0:
        return True
    # Uvicorn completes its lifespan then re-raises the captured SIGTERM. Check
    # the owned PID's completion, never treat an arbitrary signal exit as clean.
    return (name in {"api", "ml"} and code == -signal.SIGTERM
            and "Application shutdown complete." in log
            and f"Finished server process [{pid}]" in log)


def retention_accepted(retention, gap):
    # An empty live room does not measure retention. Never turn 0/0 into 100%.
    return isinstance(retention, (int, float)) and 98 <= retention <= 100 and gap <= 10


def latency(path: Path) -> dict:
    rows, invalid = [], []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
            ts = [row[k] for k in ("t0_decoder_reference_monotonic_ns", "t1_decoder_out_monotonic_ns",
                "t6_ipc_publish_monotonic_ns", "t7_ui_receive_monotonic_ns", "t8_ui_paint_monotonic_ns")]
            if not all(isinstance(t, int) and t > 0 for t in ts) or ts != sorted(ts):
                raise ValueError("invalid same-frame chain")
            rows.append((row, ts))
        except (ValueError, KeyError, TypeError) as exc:
            invalid.append({"line": number, "reason": str(exc)})
    result = {"invalid_rows": invalid, "cameras": {}}
    for camera in (f"CAM-{i:02d}" for i in range(1, 7)):
        selected = [(r, ts) for r, ts in rows if r["camera_id"] == camera]
        if not selected:
            result["cameras"][camera] = {"samples": 0, "pass": False}
            continue
        seq = [r["sequence"] for r, _ in selected]
        durations = [(b[1][-1] - a[1][-1]) / 1e6 for a, b in zip(selected, selected[1:])]
        total = [(t[-1] - t[0]) / 1e6 for _, t in selected]
        stages = {"total": total, "decoder": [(t[1]-t[0])/1e6 for _, t in selected],
            "out_to_publish": [(t[2]-t[1])/1e6 for _, t in selected],
            "publish_to_receive": [(t[3]-t[2])/1e6 for _, t in selected],
            "receive_to_paint": [(t[4]-t[3])/1e6 for _, t in selected], "paint_gap": durations}
        result["cameras"][camera] = {"samples": len(selected),
            "timing_ms": {s: {"p50": percentile(v, 50), "p95": percentile(v, 95),
                              "p99": percentile(v, 99), "max": max(v)} for s, v in stages.items() if v},
            "ui_skipped": sum(r.get("ui_skipped_preview_frames", 0) for r, _ in selected),
            "sequence_backsteps": sum(b <= a for a, b in zip(seq, seq[1:])),
            "painted_fps": (len(selected) - 1) * 1e9 / (selected[-1][1][-1] - selected[0][1][-1]),
            "pass": len(selected) >= 200 and percentile(total, 95) < 40 and all(b > a for a, b in zip(seq, seq[1:]))}
    result["status"] = "PASS" if not invalid and all(c["pass"] for c in result["cameras"].values()) else "FAIL"
    return result


def integration_checks(output: Path) -> dict:
    native_root = output / "room-pair/live-current"
    result = json.loads((native_root / "result.json").read_text())
    stats = json.loads((output / "preview/camera-health.json").read_text())
    native_samples = json.loads((native_root / "samples.json").read_text())
    log = (native_root / "logs/deepstream.log").read_text(errors="replace")
    checks = {"native_runtime": result["status"] == "PASS",
        "exact_v13_loaded": result["loaded_tracker_sha256"] == V13_SHA,
        "four_preview_only_owners": set(stats) == set(OTHER),
        "preview_bus_clean": all(r["bus_errors"] == 0 for r in stats.values()),
        "preview_reconnects_zero": all(r["reconnects"] == 0 for r in stats.values()),
        "preview_queues_bounded": all(r["queue_high_water"] <= 1 for r in stats.values()),
        "four_single_readers_decoders": all(r["runtime_graph"]["factories"].get("rtspsrc") == r["runtime_graph"]["factories"].get("nvv4l2decoder") == 1 for r in stats.values())}
    report = {"checks": checks, "native": {}, "preview_only": stats}
    for index, camera in enumerate(("CAM-01", "CAM-04")):
        decoded = [d for s in native_samples for d in s.get("decoder") or [] if d["camera_id"] == camera]
        ready = [(s["mono_ns"], r) for s in native_samples for r in (s.get("readiness") or {}).get("sources", [])
                 if r["source_id"] == index and (s.get("readiness") or {}).get("ready")]
        retention = result["retention"]["cameras"][camera]
        gap = max((w["frames"] for w in retention["association_deficit_windows"]), default=0)
        stage_fps = {stage: (ready[-1][1][stage + "_frames"] - ready[0][1][stage + "_frames"]) * 1e9 / (ready[-1][0] - ready[0][0])
                     for stage in ("mux", "pgie", "tracker")} if len(ready) >= 2 else {}
        checks[camera + "_retention"] = retention_accepted(retention["detector_associated_retention_percent"], gap)
        checks[camera + "_one_decoder"] = log.count(f"exact decoder pads instrumented camera_index={index}") == 1
        checks[camera + "_zero_reconnects"] = bool(ready) and max(r["reconnect_attempts"] for _, r in ready) == 0
        checks[camera + "_bounded_queue"] = bool(decoded) and all(d["analytics_queue_level"] <= d["analytics_queue_limit"] for d in decoded)
        checks[camera + "_no_queue_growth"] = bool(decoded) and max(d["analytics_queue_level"] for d in decoded[-30:]) < decoded[-1]["analytics_queue_limit"]
        checks[camera + "_fps"] = len(stage_fps) == 3 and all(19 <= v <= 21 for v in stage_fps.values())
        checks[camera + "_timestamps"] = bool(decoded) and all(d["preview_pts_misses"] == d["input_pts_backsteps"] == d["output_pts_backsteps"] == 0 for d in decoded)
        report["native"][camera] = {"retention_percent": retention["detector_associated_retention_percent"],
            "maximum_proposal_gap_frames": gap, "analytics_fps": stage_fps,
            "queue_high_water_sampled": max((d["analytics_queue_level"] for d in decoded), default=None),
            "queue_limit": decoded[-1]["analytics_queue_limit"] if decoded else None,
            "queue_overrun_count": decoded[-1]["analytics_queue_overruns"] if decoded else None}
    save(output / "integration-checks.json", report)
    return checks


def run(output: Path, duration: float, shutdown_debug=False, native_build=None) -> int:
    if duration < 300:
        raise ValueError("F5 full-stack latency acceptance requires at least five continuous minutes")
    python = preflight_python(ROOT)
    # Frozen F3 captures deployment before legacy credential dotenv imports.
    deployment = DEPLOYMENT
    check_ports(deployment)
    if not os.getenv("DISPLAY") and not os.getenv("WAYLAND_DISPLAY"):
        raise RuntimeError("real PySide6 display required")
    with SourceOwnership(tuple(f"CAM-{i:02d}" for i in range(1, 7)), ROOT / ".runtime/camera-owner-locks"):
        pass
    if subprocess.check_output(["ss", "-Htnp", "state", "established", "( dport = :554 )"], text=True).strip():
        raise RuntimeError("existing RTSP owner; no process was stopped")
    output = output.resolve()
    output.relative_to(ROOT / ".runtime")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    (output / "room-pair").mkdir()
    save(output / "protected-before.json", protected_v13())
    lock = open("/tmp/ai_surveillance_full_live_stack.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic_ns()
    save(output / "launch.json", {"deployment": asdict(deployment), "session_start_ns": started,
        "port_8000_owner": "AutoMagicCalib", "reason": "existing AMC ownership preserved",
        "surveillance_api_port": deployment.api_port, "surveillance_ml_port": deployment.ml_port,
        "experimental": True, "production_promoted": False, "python": str(python),
        "frontend_settings": asdict(FRONTEND_SETTINGS)})
    env = dict(sidecar_environment(), **deployment.environment(), SURVEILLANCE_ANALYTICS_ENABLED="1",
        MV3DT_ROOM_RUNTIME_ROOT=str(output / "room-pair"), V11_MONITORING_TELEMETRY_PATH=str(output / "monitoring.json"),
        MV3DT_PREVIEW_LATENCY_LOG=str(output / "paint.jsonl"), FRONTEND_USE_V11_SHARED_MEMORY="1",
        FRONTEND_FRAME_REFRESH_INTERVAL_MS=str(FRONTEND_SETTINGS.frame_refresh_interval_ms),
        FRONTEND_REFRESH_INTERVAL_MS=str(FRONTEND_SETTINGS.refresh_interval_ms))
    bridge = MixedMonitoring(output, started)
    writer = MonitoringTelemetryWriter(output / "monitoring.json")
    children, logs, samples, checks, forced = {}, [], [], {}, set()
    running = True

    def stop(*_):
        nonlocal running
        running = False

    def spawn(name, args):
        log = (output / "logs" / f"{name}.log").open("x")
        logs.append(log)
        children[name] = subprocess.Popen([str(python), "-B", "-u", *args], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    error = None
    try:
        spawn("native", ["-m", "scripts.run_v13_room_candidate", "--dataset", "live", "--duration", str(duration + 150),
            "--output", str(output / "room-pair/live-current"), *(["--shutdown-debug"] if shutdown_debug else []),
            *(["--native-build", str(native_build.resolve())] if native_build else [])])
        spawn("preview", ["-m", "scripts.run_preview_foundation", "--cameras", ",".join(OTHER), "--output", str(output / "preview")])
        spawn("ml", ["-m", "services.ml_service.app.main"])
        spawn("api", ["-m", "services.api_service.app.main"])
        spawn("frontend", ["-m", "scripts.validate_production_wall", "--output", str(output / "ui.jsonl")])
        save(output / "processes.json", {name: proc.pid for name, proc in children.items()})
        deadline, ready_at = time.monotonic() + 110, None
        with httpx.Client(timeout=3) as client:
            while running:
                if any(p.poll() is not None for p in children.values()):
                    raise RuntimeError("owned child exited before acceptance completed")
                snapshot = bridge.snapshot()
                writer.publish(snapshot)
                now = time.monotonic()
                if ready_at is not None and now - ready_at >= duration:
                    break
                row = {"mono_ns": time.monotonic_ns(), "monitoring": snapshot}
                try:
                    row["ml"] = client.get(deployment.ml_service_url + "/health").json()
                    row["api"] = client.get(deployment.frontend_api_base_url + "/health").json()
                    row["api_snapshot"] = client.get(deployment.frontend_api_base_url + "/api/v1/monitoring/snapshot").json()
                    healthy = row["ml"]["status"] == row["api"]["status"] == "ok" and row["ml"]["online_camera_count"] == 6
                except (httpx.HTTPError, ValueError, KeyError):
                    healthy = False
                row["rtsp"] = subprocess.check_output(["ss", "-Htnp", "state", "established", "( dport = :554 )"], text=True).splitlines()
                row["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True).strip()
                row["host_cpu_ticks"] = Path("/proc/stat").read_text().splitlines()[0]
                row["host_memory"] = [l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith(("MemTotal:", "MemAvailable:"))]
                samples.append(row)
                if ready_at is None and healthy and len(row["rtsp"]) == 6:
                    ready_at = now
                    save(output / "ready.json", row)
                elif ready_at is not None and (not healthy or len(row["rtsp"]) != 6):
                    raise RuntimeError("steady six-camera health / ownership regression")
                if ready_at is None and now > deadline:
                    raise RuntimeError("native READY and six current previews did not become healthy")
                time.sleep(.75)
            if not running:
                raise RuntimeError("acceptance interrupted")
            cameras = client.get(deployment.frontend_api_base_url + "/api/v1/cameras").json()
            checks["canonical_schema"] = len(cameras["cameras"]) == 6 and all("camera_id" in c and "id" not in c for c in cameras["cameras"])
            messages = asyncio.run(current_websocket_samples(deployment.monitoring_ws_url, bridge, writer))
            save(output / "websocket.json", messages)
            checks["websocket_current"] = len({r["sequence"] for r in messages}) == 3 and all(all(c["online"] for c in r["cameras"]) for r in messages)
            save(output / "identity-api.json", client.get(deployment.frontend_api_base_url + "/api/v1/room-pair/identity").json())
            ui = [json.loads(line) for line in (output / "ui.jsonl").read_text().splitlines()]
            live = [r for r in ui if len(r["cameras"]) == 6 and all(c["connected"] for c in r["cameras"].values())]
            checks["real_ui_advances"] = len(live) > 400 and all(live[-1]["cameras"][c]["sequence"] - live[0]["cameras"][c]["sequence"] > 5000 for c in live[0]["cameras"])
            checks["ui_no_network_reader"] = all(not c["network_reader"] for r in ui for c in r["cameras"].values())
            checks["fullscreen"] = any(r["fullscreen_checked"] for r in live)
    except Exception as exc:
        error = str(exc)
    finally:
        # Close consumers before their owned producers, never stop AMC/brokers.
        for name in ("frontend", "api", "ml", "preview", "native"):
            proc = children.get(name)
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=120 if name == "native" else 20)
                except subprocess.TimeoutExpired:
                    error = error or f"{name} failed clean shutdown"
                    forced.add(name)
                    proc.kill()
                    proc.wait(timeout=10)
        bridge.close()
        writer.publish({"runtime": dict(offline_snapshot()["runtime"], status="stopped"), "cameras": offline_snapshot()["cameras"]})
        for log in logs:
            log.close()
        lock.close()
        save(output / "samples.json", samples)
        save(output / "shutdown.json", {name: p.poll() for name, p in children.items()})
        after = protected_v13()
        save(output / "protected-after.json", after)
        checks["frozen_stages_unchanged"] = json.loads((output / "protected-before.json").read_text()) == after
        checks["no_rtsp_after_shutdown"] = not subprocess.check_output(["ss", "-Htnp", "state", "established", "( dport = :554 )"], text=True).strip()
        shutdown = {name: clean_child_shutdown(name, p.poll(), p.pid,
                    (output / "logs" / f"{name}.log").read_text(errors="replace"), name in forced)
                    for name, p in children.items()}
        save(output / "shutdown-verification.json", {"clean": shutdown, "forced": sorted(forced)})
        checks["clean_shutdown"] = all(shutdown.values())
        try:
            check_ports(deployment)
            checks["surveillance_ports_released"] = True
        except RuntimeError:
            checks["surveillance_ports_released"] = False
    if (output / "paint.jsonl").exists():
        timing = latency(output / "paint.jsonl")
        save(output / "latency.json", timing)
        checks["six_camera_latency"] = timing["status"] == "PASS"
    try:
        checks.update(integration_checks(output))
    except (OSError, ValueError, KeyError) as exc:
        error = error or f"missing integration evidence: {exc}"
    report = {"status": "PASS" if not error and checks and all(checks.values()) else "FAIL", "error": error, "checks": checks,
        "duration_requested_sec": duration, "samples": len(samples)}
    save(output / "result.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--shutdown-debug", action="store_true")
    parser.add_argument("--native-build", type=Path)
    parser.add_argument("--audit-finished", action="store_true",
                        help="post-mortem only; cannot replace a failed/missing live acceptance")
    args = parser.parse_args()
    if args.audit_finished:
        checks = integration_checks(args.output)
        timing = json.loads((args.output / "latency.json").read_text())
        checks["six_camera_latency"] = timing["status"] == "PASS"
        checks["clean_shutdown"] = all(json.loads((args.output / "shutdown-verification.json").read_text())["clean"].values())
        messages = json.loads((args.output / "websocket.json").read_text())
        checks["websocket_current"] = len({r["sequence"] for r in messages}) == 3 and all(
            all(c["online"] for c in r["cameras"]) for r in messages)
        target = args.output / "post-audit.json"
        if target.exists():
            raise ValueError("post-mortem evidence already exists")
        save(target, {"status": "FAIL", "scope": "POST_MORTEM_ONLY; not a replacement live acceptance",
                      "checks": checks, "original_harness_result_exists": (args.output / "result.json").exists()})
        raise SystemExit(1)
    raise SystemExit(run(args.output, args.duration, args.shutdown_debug, args.native_build))


if __name__ == "__main__":
    main()
