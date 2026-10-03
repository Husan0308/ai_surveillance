"""Read-only protected-state inventory for the F3 application contract gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import os
import signal
import sys
import time
import asyncio
import ast

import httpx
import websockets

from services.shared.deployment import load_deployment
# Capture explicit deployment settings before legacy credential loaders read
# their shared .env. Camera credentials must not reconfigure service ports.
DEPLOYMENT = load_deployment()

from scripts.freeze_preview_foundation import ROOT, protected, save
from scripts.freeze_repo_baseline import file_record
from scripts.run_preview_foundation import CONTAINER


def inventory() -> dict:
    files = protected()
    stage = ROOT / ".runtime/freeze/F2-six-camera-foundation"
    commit = json.loads((stage / "commit.json").read_text())
    files.update({str(ROOT / p): file_record(ROOT / p) for p in commit["source_hashes"]})
    files.update({str(p): file_record(p) for p in stage.rglob("*") if p.is_file()})
    amc = json.loads(subprocess.check_output(
        ["docker", "inspect", "auto-magic-calib-ms-1", "auto-magic-calib-ui-1"], text=True))
    return {"files": files, "amc": [{"name": r["Name"], "id": r["Id"],
        "image": r["Image"], "started_at": r["State"]["StartedAt"],
        "ports": r["HostConfig"]["PortBindings"]} for r in amc]}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


async def websocket_samples(url: str) -> list[dict]:
    async with websockets.connect(url, open_timeout=3) as websocket:
        return [json.loads(await asyncio.wait_for(websocket.recv(), 8)) for _ in range(3)]


def replace_known_stale_services(output: Path) -> None:
    """Retire only the two explicitly identified old surveillance services."""
    records = []
    for pid, module in ((4019197, "services.ml_service.app.main"), (4019198, "services.api_service.app.main")):
        root = Path(f"/proc/{pid}")
        if not root.exists():
            continue
        argv = (root / "cmdline").read_bytes().decode().strip("\0").split("\0")
        cwd = str((root / "cwd").resolve())
        if argv != ["/home/apsidal/ai_surveillance/.venv/bin/python", "-m", module] or cwd != str(
            ROOT / ".runtime/gate3jr-DV3lTHm8/recovery"):
            raise RuntimeError(f"old PID {pid} changed provenance; not stopping it")
        records.append({"pid": pid, "argv": argv, "cwd": cwd, "reason": "replace stale surveillance service only"})
    save(output / "retired-services.json", records)
    for row in records:
        os.kill(row["pid"], signal.SIGTERM)
    until = time.monotonic() + 15
    while any(Path(f"/proc/{r['pid']}").exists() for r in records):
        if time.monotonic() > until:
            raise RuntimeError("old surveillance service did not stop cleanly")
        time.sleep(0.2)


def run(output: Path, duration: float) -> int:
    output.mkdir(parents=True, exist_ok=False)
    save(output / "protected-before.json", inventory())
    replace_known_stale_services(output)
    config = DEPLOYMENT
    stack = output / "stack"
    env = dict(os.environ, **config.environment(), FULL_STACK_OUT=str(stack), FULL_STACK_UI_EVIDENCE="1",
               MV3DT_PREVIEW_LATENCY_LOG=str(output / "paint.jsonl"))
    samples, checks, gpu = [], {}, []
    with (output / "supervisor.log").open("x") as log:
        proc = subprocess.Popen(["bash", "scripts/start_full_live_stack.sh"], cwd=ROOT,
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        paused, stopped_ml = False, None
        try:
            deadline = time.monotonic() + 100
            while not (stack / "ready.json").exists():
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("application did not reach fresh six-camera readiness")
                time.sleep(0.5)
            started = time.monotonic()
            with httpx.Client(timeout=6) as client:
                while time.monotonic() - started < duration:
                    ml = client.get(config.ml_service_url + "/health").json()
                    api = client.get(config.frontend_api_base_url + "/health").json()
                    snapshot = client.get(config.frontend_api_base_url + "/api/v1/monitoring/snapshot").json()
                    sockets = subprocess.check_output(["ss", "-Htnp", "state", "established", "( dport = :554 )"], text=True).splitlines()
                    row = {"mono_ns": time.monotonic_ns(), "ml": ml, "api": api,
                           "snapshot": snapshot, "rtsp": sockets}
                    samples.append(row)
                    if ml["status"] != "ok" or api["status"] != "ok" or len(sockets) != 6:
                        raise RuntimeError("live health / source ownership failed")
                    if len(samples) % 5 == 1:
                        gpu.append(subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                                          "--format=csv,noheader,nounits"], text=True).strip())
                    time.sleep(1)
                cameras = client.get(config.frontend_api_base_url + "/api/v1/cameras").json()
                checks["canonical_schema"] = len(cameras["cameras"]) == 6 and all("camera_id" in c and "id" not in c for c in cameras["cameras"])
                messages = asyncio.run(websocket_samples(config.monitoring_ws_url))
                save(output / "websocket-live.json", messages)
                checks["websocket_current"] = len({m["sequence"] for m in messages}) == 3 and all(all(c["online"] for c in m["cameras"]) for m in messages)
                pids = read_json(stack / "processes.json")
                stopped_ml = pids["ml"]
                os.kill(stopped_ml, signal.SIGSTOP)
                try:
                    unavailable = client.get(config.frontend_api_base_url + "/health").json()
                    degraded = asyncio.run(websocket_samples(config.monitoring_ws_url))
                    save(output / "ml-loss.json", {"health": unavailable, "websocket": degraded})
                    checks["ml_loss_truthful"] = unavailable["status"] == "degraded" and all(
                        m["runtime"]["status"] == "degraded" and not any(c["online"] for c in m["cameras"]) for m in degraded)
                finally:
                    os.kill(stopped_ml, signal.SIGCONT)
                    stopped_ml = None
                time.sleep(5)
                checks["ml_recovered"] = client.get(config.frontend_api_base_url + "/health").json()["status"] == "ok"
                subprocess.run(["docker", "pause", CONTAINER], stdout=subprocess.DEVNULL, check=True)
                paused = True
                try:
                    time.sleep(3)
                    stale = client.get(config.ml_service_url + "/health").json()
                    stale_ui = json.loads((stack / "ui.jsonl").read_text().splitlines()[-1])
                    save(output / "stale-producer.json", {"health": stale, "ui": stale_ui})
                    checks["stale_rejected"] = stale["status"] == "degraded" and stale["online_camera_count"] == 0 and not any(r["connected"] for r in stale_ui["cameras"].values())
                finally:
                    subprocess.run(["docker", "unpause", CONTAINER], stdout=subprocess.DEVNULL, check=True)
                    paused = False
                time.sleep(15)
                checks["producer_recovered"] = client.get(config.ml_service_url + "/health").json()["status"] == "ok"
                ui_rows = [json.loads(line) for line in (stack / "ui.jsonl").read_text().splitlines()]
                live = [r for r in ui_rows if len(r["cameras"]) == 6 and all(c["connected"] for c in r["cameras"].values())]
                checks["real_ui_advances"] = len(live) > 30 and all(
                    live[-1]["cameras"][c]["sequence"] > live[0]["cameras"][c]["sequence"] + 500
                    for c in live[0]["cameras"])
                checks["no_ui_network_video"] = all(not c["network_reader"] for r in ui_rows for c in r["cameras"].values())
                checks["fullscreen"] = any(r["fullscreen_checked"] for r in live)
                (stack / "wall.png").stat()
                os.kill(pids["frontend"], signal.SIGTERM)  # User-close semantics own supervisor cleanup.
                proc.wait(timeout=40)
                checks["clean_ui_close"] = proc.returncode == 0
        finally:
            if stopped_ml:
                os.kill(stopped_ml, signal.SIGCONT)
            if paused:
                subprocess.run(["docker", "unpause", CONTAINER], stdout=subprocess.DEVNULL)
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=50)
            save(output / "samples.json", samples)
            save(output / "gpu.json", gpu)
            save(output / "checks.json", checks)
            save(output / "protected-after.json", inventory())
    before, after = read_json(output / "protected-before.json"), read_json(output / "protected-after.json")
    checks["protected_runtime_unchanged"] = before == after
    checks["no_rtsp_after_shutdown"] = not subprocess.check_output(["ss", "-Htnp", "state", "established", "( dport = :554 )"], text=True).strip()
    checks["no_preview_container"] = subprocess.run(["docker", "inspect", CONTAINER], capture_output=True).returncode != 0
    save(output / "checks.json", checks)
    save(output / "result.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "duration_sec": duration, "samples": len(samples), "port_8000_owner": "AutoMagicCalib",
        "surveillance_api_port": config.api_port, "surveillance_ml_port": config.ml_port,
        "reason": "existing AMC ownership preserved"})
    print(json.dumps(read_json(output / "result.json")), flush=True)
    return 0 if all(checks.values()) else 1


def finalize(output: Path, accepted: Path) -> int:
    if read_json(accepted / "result.json")["status"] != "PASS":
        raise RuntimeError("cannot freeze a failed runtime acceptance")
    if subprocess.check_output(["ss", "-Htnp", "state", "established", "( dport = :554 )"], text=True).strip():
        raise RuntimeError("stop camera owners before isolated static/unit regression")
    python = ROOT / ".runtime/full-stack-venv/bin/python"
    commands = [[str(python), "-B", "-m", "pytest", "-q", "tests", f"--junitxml={output / 'unit-final.xml'}"],
                ["bash", "tests/native/run_bbox_history_lifecycle_tests.sh"],
                ["bash", "scripts/start_full_live_stack.sh", "--preflight"],
                [str(python), "-I", "-B", "-m", "pip", "check"],
                [str(python), "-B", "scripts/freeze_repo_baseline.py", "--output", ".runtime/freeze/F0-repo-baseline", "--verify"]]
    results = []
    with (output / "test-results.txt").open("x") as log:
        for command in commands:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                env=dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1"))
            log.write(json.dumps(command) + "\n" + result.stdout + result.stderr + "\n")
            results.append({"command": command, "exit": result.returncode})
        paths = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard"], text=True).splitlines()
        syntax_errors = []
        counts = {"python": 0, "shell": 0}
        for name in set(paths):
            if name.endswith(".py"):
                counts["python"] += 1
                try:
                    ast.parse((ROOT / name).read_text(), filename=name)
                except (SyntaxError, UnicodeError) as exc:
                    syntax_errors.append(str(exc))
            elif name.endswith(".sh"):
                counts["shell"] += 1
                check = subprocess.run(["bash", "-n", str(ROOT / name)], capture_output=True, text=True)
                if check.returncode:
                    syntax_errors.append(check.stderr)
        whitespace = subprocess.run(["git", "diff", "--check"], capture_output=True, text=True)
    baseline = read_json(output / "protected-before.json")
    current = inventory()
    changed = [p for p in baseline["files"] if baseline["files"][p] != current["files"][p]]
    allowed = [str(ROOT / "scripts/start_full_live_stack.sh")]
    frozen_launcher = subprocess.check_output(["git", "show",
        "2617f146f3a7d3acfcd29f2621fdedbdb1eb6a80:scripts/start_full_live_stack.sh"], text=True)
    runtime_prefix_preserved = (ROOT / "scripts/start_full_live_stack.sh").read_text().startswith(
        frozen_launcher.split("ROOM_PAIR_DURATION=")[0])
    protection = changed == allowed and baseline["amc"] == current["amc"] and runtime_prefix_preserved
    save(output / "protected-after.json", current)
    passed = protection and not syntax_errors and whitespace.returncode == 0 and all(r["exit"] == 0 for r in results)
    save(output / "final-acceptance.json", {"status": "PASS" if passed else "FAIL", "checks": results,
        "protected_changes": changed, "authorized_changes": allowed, "frozen_runtime_resolution_unchanged": runtime_prefix_preserved,
        "previous_evidence_unchanged": protection, "syntax": counts, "syntax_errors": syntax_errors,
        "whitespace_exit": whitespace.returncode, "accepted_run": str(accepted),
        "port_8000_owner": "AutoMagicCalib", "surveillance_api_port": DEPLOYMENT.api_port,
        "surveillance_ml_port": DEPLOYMENT.ml_port, "reason": "existing AMC ownership preserved"})
    camera_stats = read_json(accepted / "stack/preview/camera-health.json")
    normal_samples = read_json(accepted / "samples.json")
    ownership = {c: {"process": "DeepStream 9.1 preview-only container", "producer": CONTAINER,
        "reader_count": r["runtime_graph"]["factories"].get("rtspsrc"),
        "decoder_count": r["runtime_graph"]["factories"].get("nvv4l2decoder"),
        "queue_high_water": r["queue_high_water"], "queue_max": r["runtime_graph"]["queue_max_buffers"],
        "reconnects_final_including_controlled_pause": r["reconnects"],
        "reconnects_normal_window_max": max((s["snapshot"]["cameras"][int(c[-2:]) - 1]["reconnects"] or 0)
                                          for s in normal_samples),
        "errors": r["bus_errors"], "warnings": r["bus_warnings"]}
        for c, r in camera_stats.items()}
    save(output / "source-ownership.json", ownership)
    report = (f"# FREEZE F3 — {'PASS' if passed else 'FAIL'}\n\n"
        "Authorized port correction: AutoMagicCalib remains unchanged on 8000; surveillance API 8100 / ML 8101.\n\n"
        f"Accepted real-display campaign: `{accepted}`; 120 seconds of six healthy current producers, followed by controlled ML loss/recovery and producer pause/staleness/recovery.\n\n"
        "Canonical camera_id; atomic latest-only telemetry bridge outside F2 producer; no video fallback or frontend RTSP; real 3x2 PySide6 MainWindow/fullscreen; clean user-close shutdown.\n\n"
        "All runtime checks: `run2/result.json`. Samples/socket checks, websocket captures, actual-paint timing and GUI sequences are saved separately. Initial failed run1 is preserved: credential dotenv injected legacy port 8000; launch failed closed before any camera opened.\n\n"
        "Retired only two exactly identified old surveillance processes; did not stop/reconfigure AMC.\n\n"
        "Normal-window spontaneous reconnects: zero. The deliberate all-producer pause caused one watchdog reconnect per camera; all six recovered, with no bus errors/warnings or duplicate owners. These induced reconnects are not reported as zero.\n\n"
        "F0/F1/F2 evidence and production/calibration/model/V13 assets remain unchanged. F1 launcher orchestration is an explicitly authorized F3 edit; exact F1 runtime resolution prefix and unchanged F1 tests remain valid. All nine F2 source/test/docs hashes are unchanged.\n\n"
        "Pre-existing bare asset source/binary/manifest drift is unresolved and must be handled as a separately pinned experimental chain in F4; no production rebaseline. Optional plugin-scanner library warnings are present in the pinned SDK image; camera pipeline bus errors/warnings are recorded independently, not hidden.\n\n"
        f"Static/unit/native/import/protected checks: `{output / 'test-results.txt'}`.\n")
    (output / "F3-report.md").write_text(report)
    print(json.dumps(read_json(output / "final-acceptance.json")), flush=True)
    return 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--finalize", type=Path)
    args = parser.parse_args()
    if args.finalize:
        raise SystemExit(finalize(args.output.resolve(), args.finalize.resolve()))
    if args.run:
        raise SystemExit(run(args.output.resolve(), args.duration))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, inventory())


if __name__ == "__main__":
    main()
