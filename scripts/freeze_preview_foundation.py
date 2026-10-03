#!/usr/bin/env python3
"""F2 live evidence campaign: no inference, tracking, or service-port changes."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
from urllib.parse import urlsplit

from scripts.freeze_repo_baseline import file_record
from scripts.run_preview_foundation import CONTAINER, ROOT
from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader
from services.ml_service.app.config import load_settings
from services.shared.runtime_python import preflight_python

CAMERAS = tuple(f"CAM-{i:02d}" for i in range(1, 7))


def save(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def protected() -> dict:
    baseline = json.loads((ROOT / ".runtime/freeze/F0-repo-baseline/baseline.json").read_text())
    f1 = json.loads((ROOT / ".runtime/freeze/F1-runtime-foundation/commit.json").read_text())
    paths = {Path(r["path"]) for group in ("protected_hashes", "artifact_hashes")
             for r in baseline[group].values()}
    paths.update(ROOT / p for p in f1["source_hashes"])
    paths.update(ROOT / p for p in subprocess.check_output(["git", "ls-files", "config/"], cwd=ROOT, text=True).splitlines())
    for stage in ("F0-repo-baseline", "F1-runtime-foundation"):
        paths.update(p for p in (ROOT / ".runtime/freeze" / stage).rglob("*") if p.is_file())
    return {str(p): file_record(p) for p in sorted(paths)}


def inspect_container() -> dict | None:
    result = subprocess.run(["docker", "inspect", CONTAINER], capture_output=True, text=True)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def rtsp_sockets() -> list[str]:
    result = subprocess.run(["ss", "-Htnp", "state", "established", "( dport = :554 )"],
                            capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def finalize(output: Path, accepted: Path, python: Path) -> int:
    """Run unchanged previous-freeze tests after live owners have shut down."""
    if inspect_container() or rtsp_sockets():
        raise RuntimeError("stop live camera owners before isolated unit regression")
    result = json.loads((accepted / "camera-health.json").read_text())
    if result["status"] != "PASS":
        raise RuntimeError("cannot finalize a failed live campaign")
    commands = [
        [str(python), "-B", "-m", "pytest", "-q", "tests", f"--junitxml={output / 'unit-final.xml'}"],
        ["bash", "tests/native/run_bbox_history_lifecycle_tests.sh"],
        ["bash", "scripts/start_full_live_stack.sh", "--preflight"],
        [str(python), "-I", "-B", "-m", "pip", "check"],
        [str(python), "-B", "scripts/freeze_repo_baseline.py", "--output", ".runtime/freeze/F0-repo-baseline", "--verify"],
    ]
    checks = []
    log_path = output / "test-results.txt"
    with log_path.open("x") as log:
        for cmd in commands:
            completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                       env=dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", QT_QPA_PLATFORM="offscreen"))
            log.write(json.dumps(cmd) + "\n" + completed.stdout + completed.stderr + "\n")
            checks.append({"command": cmd, "exit": completed.returncode})
        paths = sorted(set(subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, text=True).splitlines()))
        syntax_errors, python_files, shell_files = [], 0, 0
        for name in paths:
            path = ROOT / name
            if name.endswith(".py"):
                python_files += 1
                try:
                    ast.parse(path.read_text(), filename=name)
                except (SyntaxError, UnicodeError) as exc:
                    syntax_errors.append(f"{name}: {exc}")
            if name.endswith(".sh"):
                shell_files += 1
                check = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
                if check.returncode:
                    syntax_errors.append(f"{name}: {check.stderr}")
        whitespace = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
        log.write(json.dumps({"python_syntax": python_files, "shell_syntax": shell_files,
                              "syntax_errors": syntax_errors, "whitespace_exit": whitespace.returncode}) + "\n")
    before, after = json.loads((accepted / "protected-before.json").read_text()), protected()
    unchanged = before == after
    passed = unchanged and not syntax_errors and whitespace.returncode == 0 and all(c["exit"] == 0 for c in checks)
    save(output / "final-acceptance.json", {"status": "PASS" if passed else "FAIL", "accepted_campaign": str(accepted),
                                           "checks": checks, "protected_unchanged": unchanged,
                                           "python_syntax": python_files, "shell_syntax": shell_files,
                                           "syntax_errors": syntax_errors, "runtime": result})
    if passed:
        # Preserve the initial failed attempt's aggregate before publishing the
        # successful campaign at the canonical evidence location. Its original
        # raw logs and samples are untouched, and the report names every attempt.
        for name in ("camera-health.json", "preview-freshness.json", "protected-before.json", "protected-after.json", "source-ownership.json"):
            old = output / name
            archive = output / f"attempt1-{name}"
            if old.exists():
                if archive.exists():
                    raise FileExistsError(f"refuse to overwrite {archive}")
                shutil.copy2(old, archive)
            shutil.copy2(accepted / name, old)
    print(json.dumps({"final_status": "PASS" if passed else "FAIL", "protected_unchanged": unchanged}), flush=True)
    return 0 if passed else 1


def run_case(output: Path, label: str, duration: float, python: Path,
             reconnect: bool = False) -> dict:
    directory = output / label
    readers = {c: PreviewFrameReader(f"/dev/shm/v11_ui_preview_{c.lower().replace('-', '')}_v1.bin") for c in CAMERAS}
    seen = {c: [] for c in CAMERAS}
    started_ns = time.monotonic_ns()
    cmd = [str(python), "-B", "-m", "scripts.run_preview_foundation",
           "--output", str(directory), "--duration", str(duration + 45)]
    if reconnect:
        cmd.extend(["--test-reconnect-camera", "CAM-05", "--test-reconnect-at", "25"])
    samples, socket_checks, gpu = [], [], []
    container_evidence = None
    with (output / "runtime-logs" / f"{label}.log").open("w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1"))
        last_sample, all_ready_at, last_gpu = 0.0, None, 0.0
        try:
            while proc.poll() is None:
                now = time.monotonic()
                for camera, reader in readers.items():
                    frame = reader.read_latest(metadata_only=True)
                    if frame and frame.timestamp_ns >= started_ns:
                        history = seen[camera]
                        if not history or history[-1]["sequence"] != frame.sequence:
                            seen[camera].append({"wall_ns": time.monotonic_ns(), "sequence": frame.sequence,
                                                 "publish_ns": frame.timestamp_ns, "pts_ns": frame.pts_ns,
                                                 "t0": frame.decoder_reference_ns, "t1": frame.decoder_out_ns,
                                                 "source_frame_num": frame.source_frame_num})
                if now - last_sample >= 1:
                    last_sample = now
                    stats_path = directory / "camera-health.json"
                    try:
                        stats = json.loads(stats_path.read_text())
                        samples.append({"mono_ns": time.monotonic_ns(), "cameras": stats})
                    except (FileNotFoundError, json.JSONDecodeError):
                        stats = {}
                    if all(len(seen[c]) > 1 and stats.get(c, {}).get("state") == "LIVE" for c in CAMERAS):
                        all_ready_at = all_ready_at or now
                        if container_evidence is None:
                            info = inspect_container()
                            if info:
                                container_evidence = {"id": info["Id"], "image": info["Image"],
                                                      "pid": info["State"]["Pid"], "args": info["Args"]}
                        socket_checks.append({"mono_ns": time.monotonic_ns(), "sockets": rtsp_sockets()})
                    if now - last_gpu >= 5:
                        last_gpu = now
                        result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.decoder,memory.used",
                                                 "--format=csv,noheader,nounits"], capture_output=True, text=True)
                        gpu.append({"mono_ns": time.monotonic_ns(), "value": result.stdout.strip(), "exit": result.returncode})
                    if all_ready_at and now - all_ready_at >= duration:
                        break
                if now - started_ns / 1e9 > duration + 30:
                    break
                time.sleep(0.1)
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Stop only the exact container created by this campaign.
                info = inspect_container()
                if info and container_evidence and info["Id"] == container_evidence["id"]:
                    subprocess.run(["docker", "stop", "--time", "10", info["Id"]], check=True)
                proc.wait(timeout=15)
            time.sleep(1.3)
            stale_rejected = {c: r.read_latest(metadata_only=True) is None for c, r in readers.items()}
            for reader in readers.values():
                reader.close()
    save(directory / "observed-frames.json", seen)
    save(directory / "health-samples.json", samples)
    save(directory / "socket-checks.json", socket_checks)
    save(directory / "gpu-samples.json", gpu)
    save(directory / "container.json", container_evidence)
    final = json.loads((directory / "camera-health.json").read_text())
    failures = []
    for camera in CAMERAS:
        rows = [r["cameras"][camera] for r in samples if r["cameras"].get(camera, {}).get("state") == "LIVE"]
        expected_reconnects = 1 if reconnect and camera == "CAM-05" else 0
        if len(seen[camera]) < 100:
            failures.append(f"{camera}: insufficient advancing frames")
        if final[camera]["reconnects"] != expected_reconnects:
            failures.append(f"{camera}: unexpected reconnect count")
        if final[camera]["bus_errors"] or final[camera]["timing_invalid"]:
            failures.append(f"{camera}: bus/timing errors")
        if final[camera]["queue_high_water"] > 1:
            failures.append(f"{camera}: queue growth")
        live_samples = [r for r in samples if r["cameras"].get(camera, {}).get("state") == "LIVE"]
        # Measure recovered steady state, separately from the deliberate outage.
        stable = live_samples[-20:]
        measured_fps = 0.0
        if len(stable) >= 2:
            measured_fps = (stable[-1]["cameras"][camera]["frames"] - stable[0]["cameras"][camera]["frames"]) / ((stable[-1]["mono_ns"] - stable[0]["mono_ns"]) / 1e9)
        final[camera]["stable_measured_fps"] = measured_fps
        if not rows or measured_fps < 19:
            failures.append(f"{camera}: source FPS below 19")
        if not stale_rejected[camera]:
            failures.append(f"{camera}: preview still fresh after shutdown")
        graph = final[camera]["runtime_graph"]
        factories = graph.get("factories", {})
        if (factories.get("rtspsrc") != 1 or factories.get("nvv4l2decoder") != 1
                or factories.get("nvinfer", 0) or factories.get("nvtracker", 0)
                or graph.get("queue_max_buffers") != 1 or graph.get("queue_leaky") != 2
                or graph.get("appsink_max_buffers") != 1 or graph.get("appsink_drop") is not True
                or "deepstream-7.1" in graph.get("decoder_plugin", "")):
            failures.append(f"{camera}: incorrect runtime graph/decoder contract")
    counts = {len(r["sockets"]) for r in socket_checks}
    if not counts or not counts.issubset({5, 6} if reconnect else {6}) or 6 not in counts:
        failures.append("RTSP socket ownership not exactly six")
    actual_duration = time.monotonic() - all_ready_at if all_ready_at else 0
    if actual_duration < duration:
        failures.append("required simultaneous live duration was not reached")
    if proc.returncode != 0 or inspect_container() is not None:
        failures.append("unclean process/container shutdown")
    result = {"label": label, "status": "PASS" if not failures else "FAIL", "failures": failures,
              "exit": proc.returncode, "duration_sec": actual_duration, "cameras": final,
              "stale_rejected_after_shutdown": stale_rejected,
              "rtsp_counts": sorted({len(r["sockets"]) for r in socket_checks}),
              "preview_observations": {c: len(r) for c, r in seen.items()}}
    save(directory / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize", type=Path, help="accepted campaign; run offline regressions only")
    args = parser.parse_args()
    python = preflight_python(ROOT)
    output = args.output.resolve()
    if args.finalize:
        return finalize(output, args.finalize.resolve(), python)
    output.mkdir(parents=True, exist_ok=False)
    (output / "runtime-logs").mkdir()
    save(output / "protected-before.json", protected())
    settings = load_settings()
    save(output / "source-ownership.json", {c.camera_id: {
        "process": CONTAINER, "uri_owner": "preview_only_runtime", "preview_producer": "PreviewFrameWriter",
        "host": urlsplit(c.uri).hostname, "channel": urlsplit(c.uri).path,
        "uri_sha256": hashlib.sha256(c.uri.encode()).hexdigest(), "decoder": "nvv4l2decoder",
        "effective_latency_ms": c.effective_latency_ms(settings.deepstream.latency_ms),
        "decoder_low_latency_mode": c.decoder_low_latency_mode,
    } for c in settings.cameras})
    results = []
    for label, duration, reconnect in (("smoke", 30, False), ("stability", 300, False), ("reconnect", 65, True)):
        result = run_case(output, label, duration, python, reconnect)
        results.append(result)
        print(json.dumps({"case": label, "status": result["status"], "failures": result["failures"]}), flush=True)
        if result["status"] != "PASS":
            break
    after = protected()
    save(output / "protected-after.json", after)
    unchanged = after == json.loads((output / "protected-before.json").read_text())
    status = "PASS" if len(results) == 3 and all(r["status"] == "PASS" for r in results) and unchanged else "FAIL"
    save(output / "camera-health.json", {"status": status, "cases": results, "protected_unchanged": unchanged})
    save(output / "preview-freshness.json", {r["label"]: r["stale_rejected_after_shutdown"] for r in results})
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
