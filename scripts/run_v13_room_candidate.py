"""F5 exact V13 live/replay integration, isolated from frozen production assets."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from kafka.admin import KafkaAdminClient, NewTopic

from scripts.build_room_candidate import ROOT, verify_build
from scripts.dev_room_mv3dt import run_room_pair as room
from scripts.dev_room_mv3dt.verify_validated_assets import sha256
from scripts.freeze_repo_baseline import file_record
from scripts.run_room_candidate import command, stage_profile, section_value, save, protected
from scripts.yolo26m_person.runtime import validate_artifacts
from services.camera_v11.source_ownership import SourceOwnership
from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader
from services.shared.runtime_python import preflight_python

V13 = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml"
V13_SHA = "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834"
F4 = ROOT / ".runtime/freeze/F4-yolo26m-live"
ARCHIVE = ROOT / ".runtime/final-detector-ab-20261002-uSpPqjLY"
AUDITOR = ROOT / ".runtime/gate3jr-DV3lTHm8/recovery/scripts/dev_room_mv3dt/audit_detection_path.py"


def protected_v13() -> dict:
    state = protected()
    commit = json.loads((F4 / "commit.json").read_text())
    state["files"].update({str(ROOT / p): file_record(ROOT / p) for p in commit["source_hashes"]})
    state["files"].update({str(p): file_record(p) for p in F4.rglob("*") if p.is_file()})
    return state


def stage_v13(output: Path, mode: str) -> Path:
    if sha256(V13) != V13_SHA:
        raise ValueError("frozen V13 hash mismatch")
    stage = stage_profile(output, mode, "production-reference")
    (stage / "config_tracker.yml").write_bytes(V13.read_bytes())
    if sha256(stage / "config_tracker.yml") != V13_SHA:
        raise ValueError("staged tracker must be byte-identical V13")
    return stage


def replay_inputs(dataset: str) -> dict:
    inputs = json.loads((ARCHIVE / "inputs.json").read_text())[dataset]
    for camera, item in inputs.items():
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"{camera} input hash mismatch")
        # Count encoded access units, not a different decoder's error concealment.
        # The archived HEVC empty-room clip has software-decoder slice warnings;
        # acceptance below independently requires EVERY frame at native PGIE and
        # tracker, matching the previously validated NVIDIA decode convention.
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,nb_read_packets,duration", "-of", "json", str(path)], text=True))["streams"][0]
        verify_video_contract(dataset, probe)
        item["verified_probe"] = probe
    return inputs


def verify_video_contract(dataset: str, probe: dict) -> None:
    if dataset != "empty-room" and (probe["width"], probe["height"], probe["avg_frame_rate"], probe["nb_frames"], probe["nb_read_packets"]) != (1920, 1080, "20/1", "2400", "2400"):
        raise ValueError("acceptance requires the complete synchronized 2400-frame pair")
    if dataset == "empty-room" and (int(probe["nb_frames"]), int(probe["nb_read_packets"])) != (12031, 12031):
        raise ValueError("empty-room safety requires the complete archived input")


def complete_native_frames(report: dict, dataset: str) -> bool:
    expected = 12031 if dataset == "empty-room" else 2400
    return set(report["cameras"]) == {"CAM-01", "CAM-04"} and all(
        row["pgie_frames"] == row["tracker_frames"] == expected for row in report["cameras"].values())


def audit_shutdown_drain(path: Path) -> dict:
    required = ("shutdown_requested", "source_admission_stopped", "drain_completed",
                "flush_start", "destroy_pipeline", "native_exit")
    try:
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        return {"status": "FAIL", "reason": f"missing or malformed drain log: {error}", "events": []}
    names = [row.get("event") for row in events]
    positions = {name: names.index(name) for name in required if name in names}
    if set(positions) != set(required) or [positions[name] for name in required] != sorted(positions.values()):
        return {"status": "FAIL", "reason": "required drain/flush/destroy lifecycle events missing or out of order",
                "events": events}
    terminal = events[positions["native_exit"]]
    sources = terminal.get("sources") or []
    def source_drained(row: dict) -> bool:
        last_pts = row.get("last_admitted_pts")
        return (row.get("blocked") is True and last_pts not in (None, 0)
            and row.get("mux_last_pts") == last_pts
            and row.get("pgie_last_pts") == last_pts
            and row.get("tracker_last_pts") == last_pts
            and row.get("pgie_frames") == row.get("tracker_frames"))
    source_results = [dict(row, drained=source_drained(row)) for row in sources]
    if len(source_results) != 2 or not all(row["drained"] for row in source_results):
        return {"status": "FAIL", "reason": "terminal counters or PTS do not prove full drain for both sources",
                "events": events, "sources": source_results}
    times = [events[positions[name]].get("mono_ns", 0) for name in required]
    if not all(isinstance(value, int) and value > 0 for value in times) or times != sorted(times):
        return {"status": "FAIL", "reason": "monotonic lifecycle timestamps are missing or inconsistent",
                "events": events, "sources": source_results}
    return {"status": "PASS", "events": events, "sources": source_results,
            "terminal_tracker_equals_pgie_per_source": all(
                row["tracker_frames"] == row["pgie_frames"] for row in source_results)}


def verify_native_build(path: Path | None) -> dict:
    if path is None:
        from scripts.build_v13_shutdown_candidate import verify_candidate
        return verify_candidate(F4 / "candidate-audit-build/build.json")
    record = json.loads(path.read_text())
    if record.get("candidate") == "F5_DRAIN_BEFORE_FLUSH":
        from scripts.build_v13_drain_shutdown_candidate import verify_build as verify_drain_build
        return verify_drain_build(path)
    from scripts.build_v13_shutdown_candidate import verify_candidate
    return verify_candidate(path)


def shutdown_drain_required(dataset: str, native: dict) -> bool:
    """Drain telemetry applies only to RTSP-live teardown; replay teardown is unchanged."""
    return dataset == "live" and native.get("candidate") == "F5_DRAIN_BEFORE_FLUSH"


def verify_pose(path: Path) -> dict:
    record = json.loads(path.read_text())
    if sha256(Path(record["engine"])) != record["engine_sha256"]:
        raise ValueError("staged pose engine changed")
    model = Path(record["engine"]).parent / "bodypose3dnet_accuracy.onnx"
    if sha256(model) != record["model_sha256"] or any(sha256(Path(p)) != s for p, s in record["original"].items()):
        raise ValueError("staged/production pose provenance changed")
    return record


def sidecar_environment() -> dict:
    # Existing OSNet helper contains a legacy sibling-module import. Resolve it
    # explicitly from the frozen checkout, never a global/Trash Python path.
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "services/mv3dt_room"))))
    env.pop("PYTHONHOME", None)
    return env


def audit_retention(output: Path) -> dict:
    spec = importlib.util.spec_from_file_location("unchanged_frozen_retention_auditor", AUDITOR)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    report, proposals = auditor.audit(output)
    report["auditor_sha256"] = sha256(AUDITOR)
    save(output / "retention.json", report)
    with (output / "proposal-association.jsonl").open("x") as handle:
        for row in proposals:
            handle.write(json.dumps(row) + "\n")
    return report


def run(output: Path, dataset: str, duration: float, pose_path: Path, shutdown_debug=False, native_build=None) -> dict:
    output = output.resolve()
    output.relative_to(ROOT / ".runtime")
    from scripts.build_v13_shutdown_candidate import verify_candidate
    native = verify_native_build(native_build)
    validate_artifacts(ROOT)
    pose = verify_pose(pose_path.resolve())
    python = preflight_python(ROOT, "identity")
    preflight_python(ROOT, "kafka")
    subprocess.run([str(python), "-B", "-m", "services.mv3dt_room.live_identity_worker", "--help"],
                   cwd=ROOT, env=sidecar_environment(), stdout=subprocess.DEVNULL, check=True)
    inputs = replay_inputs(dataset) if dataset != "live" else None
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    save(output / "protected-before.json", protected_v13())
    mode = "live" if dataset == "live" else "replay"
    stage = stage_v13(output, mode)
    topic = f"freeze-f5-{time.time_ns()}-{os.getpid()}"
    admin = KafkaAdminClient(bootstrap_servers="localhost:9092")
    admin.create_topics([NewTopic(topic, 1, 1)])
    admin.close()
    app = stage / "config_deepstream.txt"
    app.write_text(section_value(section_value(app.read_text(), "sink3", "topic", topic),
                                 "sink3", "msg-broker-conn-str", f"localhost;9092;{topic}"))
    # Unique transport topics prevent retained/shared previous-run MV3DT state.
    peer = stage / "pub_sub_info_config_0.yml"
    peer.write_text(peer.read_text().replace("/trck/cam", f"/{topic}/cam"))
    session = output.name
    identity = output / "identity-live"
    identity.mkdir()
    gallery = identity / "gallery.sqlite"
    if gallery.exists():
        raise RuntimeError("fresh gallery required")
    alias = Path("/tmp") / f"f5-crops-{os.getpid()}-{time.time_ns()}"
    alias.symlink_to(stage / "logs", target_is_directory=True)
    capture_log = output / "logs/kafka_current.jsonl"
    done = output / "logs/deepstream.done"
    container = f"ai-surveillance-v13-candidate-{os.getpid()}"
    cmd = command(stage, native, mode, container)
    if native.get("candidate") == "F5_DRAIN_BEFORE_FLUSH":
        image = json.loads((ROOT / "config/deepstream-platform.json").read_text())["image"]
        at = cmd.index(image)
        cmd[at:at] = [
            "-e", "MV3DT_SHUTDOWN_DRAIN_LOG=/workspace/experiments/logs/probe/shutdown-drain.jsonl",
            "-e", "MV3DT_SHUTDOWN_DRAIN_TIMEOUT_SEC=10",
        ]
    if shutdown_debug:
        from scripts.native_shutdown_diagnostics import debugger_command
        image = json.loads((ROOT / "config/deepstream-platform.json").read_text())["image"]
        cmd = debugger_command(cmd, image)
    old_models = f"{Path(room.load_profile()['runtime']['models_root'])}:/workspace/models:ro"
    cmd[cmd.index(old_models)] = f"{pose_path.resolve().parent / 'models'}:/workspace/models:ro"
    if inputs:
        old_videos = next(x for x in cmd if x.endswith(":/workspace/inputs/videos:ro"))
        at = cmd.index(old_videos)
        del cmd[at-1:at+1]
        image = json.loads((ROOT / "config/deepstream-platform.json").read_text())["image"]
        at = cmd.index(image)
        cmd[at:at] = sum((["-v", f"{item['path']}:/workspace/inputs/videos/cam_{i:02d}.mp4:ro"]
                         for i, item in enumerate(inputs.values())), [])
    hashes = {p.name: sha256(p) for p in stage.iterdir() if p.is_file()}
    save(output / "loaded-config-manifest.json", {"native": native, "detector": json.loads((ROOT / ".runtime/models/yolo26m/engine_raw_otm.json").read_text()),
        "tracker_path": str(V13), "tracker_sha256": sha256(stage / "config_tracker.yml"), "configs": hashes,
        "pose": pose, "dataset": dataset, "inputs": inputs, "gallery": str(gallery), "gallery_preexisting": False,
        "identity_manager_sha256": sha256(ROOT / "services/mv3dt_room/global_identity_manager.py"),
        "identity_worker_sha256": sha256(ROOT / "services/mv3dt_room/live_identity_worker.py"),
        "kafka_topic": topic, "production_promoted": False, "shutdown_debugger_enabled": shutdown_debug})
    save(output / "command.json", cmd)
    (output / "running").touch()
    (output / mode).touch()
    save(output / "source_mode.json", {"source_mode": mode, "session_id": session, "started_at_epoch": time.time()})
    processes, handles, samples = [], [], []
    readers = {c: PreviewFrameReader(f"/dev/shm/v11_ui_preview_{c.lower().replace('-', '')}_v1.bin") for c in ("CAM-01", "CAM-04")}
    stopping, started_ns = False, time.monotonic_ns()

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def spawn(name, args, stdout=None):
        out = (output / "logs" / f"{name}.log").open("x") if stdout is None else stdout.open("x")
        err = (output / "logs" / f"{name}.err").open("x")
        handles.extend([out, err])
        process = subprocess.Popen(args, cwd=ROOT, env=sidecar_environment(), stdout=out, stderr=err)
        processes.append(process)
        return process

    lock = SourceOwnership(readers, ROOT / ".runtime/camera-owner-locks") if mode == "live" else nullcontext()
    try:
        with lock:
            capture = spawn("kafka_capture", [str(python), "-B", "-m", "scripts.capture_candidate_kafka", "--topic", topic], capture_log)
            spawn("mqtt", ["mosquitto_sub", "-h", "127.0.0.1", "-t", f"/{topic}/#"], output / "logs/mqtt_peer.raw")
            deadline = time.monotonic() + 15
            while not capture_log.stat().st_size:
                if capture.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("fresh topic capture did not subscribe")
                time.sleep(.1)
            sidecar = spawn("identity", [str(python), "-B", "-m", "services.mv3dt_room.live_identity_worker",
                "--kafka", str(capture_log), "--output-dir", str(identity), "--osnet-model", room.load_profile()["runtime"]["osnet_model"],
                "--duration", str(duration + 120), "--done-file", str(done), "--source-mode", mode, "--session-id", session,
                "--crop-socket", str(alias / "crops.sock"), "--gallery-db", str(gallery)])
            native_process = spawn("deepstream", cmd)
            start = time.monotonic()
            while native_process.poll() is None and not stopping and time.monotonic() - start < duration:
                if sidecar.poll() is not None or capture.poll() is not None:
                    raise RuntimeError("candidate sidecar/capture stopped unexpectedly")
                row = {"mono_ns": time.monotonic_ns(), "preview": {}}
                for camera, reader in readers.items():
                    frame = reader.read_latest(metadata_only=True)
                    if frame and frame.timestamp_ns >= started_ns:
                        row["preview"][camera] = {"sequence": frame.sequence, "timestamp_ns": frame.timestamp_ns,
                            "t0": frame.decoder_reference_ns, "t1": frame.decoder_out_ns, "t6": frame.timestamp_ns,
                            "source_frame": frame.source_frame_num, "pts": frame.pts_ns}
                try:
                    row["readiness"] = json.loads((stage / "logs/probe/readiness.json").read_text())
                except (OSError, ValueError):
                    row["readiness"] = None
                diagnostics = stage / "logs/probe/preview_diagnostics.jsonl"
                if diagnostics.exists():
                    try:
                        row["decoder"] = [json.loads(l) for l in diagnostics.read_text().splitlines()[-2:]]
                    except ValueError:
                        row["decoder"] = None
                row["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True).strip()
                row["rtsp_count"] = len(subprocess.check_output(["ss", "-Htnp", "state", "established", "( dport = :554 )"], text=True).splitlines())
                samples.append(row)
                save(output / "live-health.json", row)
                time.sleep(1)
            active_duration = time.monotonic() - start
            stop_offset = (output / "logs/deepstream.log").stat().st_size
            timed_out = native_process.poll() is None and mode == "replay"
            if native_process.poll() is None:
                subprocess.run(["docker", "kill", "--signal=SIGINT", container], capture_output=True)
                try:
                    native_process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    if shutdown_debug:
                        from scripts.native_shutdown_diagnostics import shutdown_backtrace
                        save(output / "shutdown-debugger.json", shutdown_backtrace(container, output / "logs/shutdown-backtrace.log"))
                        try:
                            native_process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            room.stop_deepstream_container(container)
                            native_process.wait(timeout=20)
                    else:
                        room.stop_deepstream_container(container)
                        native_process.wait(timeout=20)
            time.sleep(2)  # Drain broker before closing capture.
            capture.terminate()
            capture.wait(timeout=15)
            done.touch()
            sidecar.wait(timeout=90)
    finally:
        if 'native_process' in locals() and native_process.poll() is None:
            room.stop_deepstream_container(container)
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=20)
        for handle in handles:
            handle.close()
        for reader in readers.values():
            reader.close()
        alias.unlink(missing_ok=True)
        (output / "running").unlink(missing_ok=True)
        (output / "done").touch()
        save(output / "samples.json", samples)
        save(output / "protected-after.json", protected_v13())
    report = audit_retention(output)
    drain_path = stage / "logs/probe/shutdown-drain.jsonl"
    drain_report = audit_shutdown_drain(drain_path) if shutdown_drain_required(dataset, native) else None
    if drain_report is not None:
        save(output / "shutdown-drain-audit.json", drain_report)
    gpu = [list(map(float, s["gpu"].split(","))) for s in samples]
    save(output / "resource_metrics.json", {"gpu_utilization_percent": {"mean": sum(g[0] for g in gpu) / len(gpu), "peak": max(g[0] for g in gpu)},
        "gpu_memory_mib": {"mean": sum(g[1] for g in gpu) / len(gpu), "peak": max(g[1] for g in gpu)},
        "host_cpu_busy_percent": {}, "host_memory_free_mib": {}, "note": "GPU observed once per second; no inferred CPU figures"})
    log = (output / "logs/deepstream.log").read_bytes()
    stderr = (output / "logs/deepstream.err").read_text(errors="replace")
    error_lines = [l for l in (log.decode(errors="replace") + "\n" + stderr).splitlines() if "ERROR" in l]
    expected = [l for l in log[stop_offset:].decode(errors="replace").splitlines() if "<_intr_handler:" in l and "User Interrupted.." in l]
    faults = [l for l in error_lines if l not in expected]
    ids = sorted({r.get("global_person_id") for r in map(json.loads, (identity / "global_identity.jsonl").read_text().splitlines())
                  if r.get("global_person_id", "Unknown") != "Unknown"})
    result = {"dataset": dataset, "duration_sec": active_duration, "native_exit": native_process.returncode,
        "identity_exit": sidecar.returncode, "capture_exit": capture.returncode, "replay_timeout": timed_out,
        "pipeline_errors": faults, "all_error_level_messages": error_lines,
        "stderr_warnings": [l for l in stderr.splitlines() if "WARNING" in l or "WARN" in l], "canonical_ids": ids,
        "retention": report, "loaded_tracker_sha256": sha256(stage / "config_tracker.yml"),
        "shutdown_drain": drain_report,
        "staged_configs_unchanged": hashes == {p.name: sha256(p) for p in stage.iterdir() if p.is_file()},
        "protected_unchanged": json.loads((output / "protected-before.json").read_text()) == protected_v13()}
    result["complete_native_frame_processing"] = mode == "live" or complete_native_frames(report, dataset)
    drain_ok = drain_report is None or drain_report["status"] == "PASS"
    result["status"] = "PASS" if not faults and not timed_out and result["native_exit"] == result["identity_exit"] == 0 and result["protected_unchanged"] and result["staged_configs_unchanged"] and report["audit_integrity_pass"] and result["complete_native_frame_processing"] and drain_ok else "FAIL"
    save(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("live", "canonical", "person-present", "empty-room"), required=True)
    parser.add_argument("--duration", type=float, default=180)
    parser.add_argument("--pose", type=Path, default=ROOT / ".runtime/freeze/F5-v13-live/pose-cache/build.json")
    parser.add_argument("--shutdown-debug", action="store_true")
    parser.add_argument("--native-build", type=Path)
    args = parser.parse_args()
    result = run(args.output, args.dataset, args.duration, args.pose, args.shutdown_debug, args.native_build)
    print(json.dumps({k: v for k, v in result.items() if k != "retention"}), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
