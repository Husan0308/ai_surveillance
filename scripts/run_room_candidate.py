"""Isolated experimental room profile. Never rebases accepted production hashes."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from scripts.build_room_candidate import ROOT, verify_build
from scripts.dev_room_mv3dt import run_room_pair as room
from scripts.dev_room_mv3dt.verify_validated_assets import sha256, verify
from scripts.freeze_app_contract import inventory
from scripts.freeze_repo_baseline import file_record
from scripts.yolo26m_person.runtime import validate_artifacts
from services.camera_v11.source_ownership import SourceOwnership
from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader


def protected() -> dict:
    state = inventory()
    freeze = ROOT / ".runtime/freeze/F3-ml-api-ui/authorized-810x"
    commit = json.loads((freeze / "commit.json").read_text())
    state["files"].update({str(ROOT / p): file_record(ROOT / p) for p in commit["source_hashes"]})
    state["files"].update({str(p): file_record(p) for p in freeze.rglob("*") if p.is_file()})
    return state


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def section_value(text: str, section: str, key: str, value: str) -> str:
    pattern = rf"(?ms)(^\[{re.escape(section)}\]\n)(.*?)(?=^\[|\Z)"
    found = re.search(pattern, text)
    if not found:
        raise ValueError(f"missing [{section}]")
    body, count = re.subn(rf"(?m)^{re.escape(key)}=.*$", f"{key}={value}", found[2])
    if count != 1:
        raise ValueError(f"expected exactly one {section}.{key}")
    return text[:found.start()] + found[1] + body + text[found.end():]


def two_source_pgie(text: str) -> str:
    # Dynamic engine is pinned to min=1, opt=max=6. Only requested batch differs.
    if text.count("batch-size=6") != 1:
        raise ValueError("unexpected detector batch config")
    return text.replace("batch-size=6", "batch-size=2", 1)


def stage_profile(output: Path, mode: str, tracker: str) -> Path:
    stage = output / "run"
    room.copy_profile(stage)
    if mode == "live":
        room.live_config(stage, room.camera_uris())
    else:
        room.replay_config(stage)
    (stage / "config_pgie.txt").write_text(two_source_pgie(
        (ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt").read_text()))
    path = stage / "config_deepstream.txt"
    text = section_value(path.read_text(), "primary-gie", "model-engine-file", "/models/yolo26m_raw_otm_b6_fp16.engine")
    text = section_value(text, "tracker", "enable", "0" if tracker == "off" else "1")
    # Detector-only diagnostics must not publish untracked production identities.
    text = section_value(text, "sink3", "enable", "0" if tracker == "off" else "1")
    path.write_text(text)
    path.chmod(0o600)  # Live URIs contain authorized credentials; never print them.
    return stage


def command(stage: Path, build: dict, mode: str, container: str) -> list[str]:
    env_names = {"MV3DT_FRAME_AUDIT_LOG": "1", "MV3DT_UI_PREVIEW_DIAGNOSTICS": "1"}
    old = {k: os.environ.get(k) for k in env_names}
    os.environ.update(env_names)
    try:
        runtime_image = json.loads((ROOT / "config/deepstream-platform.json").read_text())["image"]
        cmd = room.deepstream_command(stage, Path(build["binary"]), runtime_image, mode, container)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    # Keep bbox audit mode identical to accepted replay when tracker is enabled.
    at = cmd.index(runtime_image)
    cmd[at:at] = ["--pull=never", "--user", f"{os.getuid()}:{os.getgid()}",
                 "-v", f"{ROOT / '.runtime/models/yolo26m'}:/models:ro"]
    return cmd


def summarize(output: Path, samples: list, duration: float, tracker: str, stop_offset: int) -> dict:
    rows = [json.loads(line) for line in (output / "run/logs/probe/frame_path_audit.jsonl").read_text().splitlines()]
    cameras = {}
    stages = sorted({r.get("stage") for r in rows if r.get("stage")})
    for camera in ("CAM-01", "CAM-04"):
        found = [r for r in rows if r.get("mapped_camera_id") == camera]
        counts = Counter(r["stage"] for r in found)
        pgie = [r for r in found if r["stage"] == "pgie"]
        proposals = [o for r in pgie for o in r["objects"] if o["class_id"] == 0]
        boxes = [o.get("bbox", []) for o in proposals]
        invalid_boxes = sum(len(b) != 4 or b[2] <= 0 or b[3] <= 0 or any(not (-1e6 < x < 1e6) for x in b) for b in boxes)
        previews = [s["preview"][camera] for s in samples if camera in s["preview"]]
        if len(previews) > 1:
            seconds = (previews[-1]["wall_ns"] - previews[0]["wall_ns"]) / 1e9
            publish_fps = (previews[-1]["sequence"] - previews[0]["sequence"]) / seconds
            pgie_fps = (counts.get("pgie", 0) / duration)
        else:
            publish_fps = pgie_fps = 0
        cameras[camera] = {"stage_counts": dict(counts), "pgie_proposals": len(proposals),
            "pgie_person_frames": sum(any(o["class_id"] == 0 for o in r["objects"]) for r in pgie),
            "confidence_min": min((o["confidence"] for o in proposals), default=None),
            "confidence_max": max((o["confidence"] for o in proposals), default=None),
            "published_fps": publish_fps, "pgie_run_average_fps": pgie_fps,
            "invalid_boxes": invalid_boxes, "preview_samples": len(previews),
            "fresh": bool(previews) and previews[-1]["age_ms"] < 1000}
    log = (output / "deepstream.log").read_text(errors="replace")
    errors = [l for l in log.splitlines() if re.search(r"\bERROR\b|Failed to create|Failed to initialize|Failed to open", l)]
    # Sample app logs its own clean SIGINT handler at ERROR level. Classify only
    # that exact handler, only after our recorded shutdown; retain all messages.
    expected_shutdown = [l for l in (output / "deepstream.log").read_bytes()[stop_offset:].decode(errors="replace").splitlines()
                         if re.fullmatch(r"\*\* ERROR: <_intr_handler:\d+>: User Interrupted\.\.\s*", l)]
    faults = [l for l in errors if l not in expected_shutdown]
    health = output / "run/logs/probe/source_health.jsonl"
    events = [json.loads(l) for l in health.read_text().splitlines()] if health.exists() else []
    reconnects = sum(r.get("event") == "reconnect_attempt" for r in events)
    result = {"tracker": tracker, "duration_sec": duration, "stages": stages, "cameras": cameras,
        "errors": faults, "all_error_level_messages": errors, "expected_owned_shutdown": expected_shutdown,
        "reconnects": reconnects, "source_health_events": len(events),
        "protected_unchanged": json.loads((output / "protected-before.json").read_text()) == protected()}
    result["status"] = "PASS" if result["protected_unchanged"] and not faults and reconnects == 0 and all(
        c["fresh"] and c["invalid_boxes"] == 0 and c["pgie_proposals"] > 0 and 19 <= c["published_fps"] <= 21 and c["pgie_run_average_fps"] >= 18.5
        for c in cameras.values()) else "FAIL"
    return result


def run(output: Path, build_path: Path, duration: float, tracker: str, mode: str) -> dict:
    output = output.resolve()
    output.relative_to(ROOT / ".runtime")
    build = verify_build(build_path.resolve())
    validate_artifacts(ROOT)
    experimental_flags = {k: v for k, v in os.environ.items() if k.startswith("MV3DT_TEST_") and v not in {"", "0"}}
    if experimental_flags:
        raise RuntimeError("obsolete experimental preview flags must not be enabled")
    if subprocess.check_output(["ss", "-Htnp", "state", "established", "( dport = :554 )"], text=True).strip():
        raise RuntimeError("camera owners already running; refusing duplicate reader")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "protected-before.json", protected())
    stage = stage_profile(output, mode, tracker)
    hashes = {p.name: sha256(p) for p in stage.iterdir() if p.is_file()}
    with redirect_stdout(io.StringIO()):
        bare = verify(ROOT, Path(room.load_profile()["runtime"]["binary"]))
    save(output / "loaded-config-manifest.json", {"build": build, "configs": hashes,
        "yolo": json.loads((ROOT / ".runtime/models/yolo26m/engine_raw_otm.json").read_text()),
        "bare_production_asset_check": bare,
        "staging_hash_override": "EXACT_PINNED_CANDIDATE_ONLY", "production_promoted": False})
    container = f"ai-surveillance-room-candidate-{os.getpid()}"
    cmd = command(stage, build, mode, container)
    save(output / "command.json", cmd)
    readers = {c: PreviewFrameReader(f"/dev/shm/v11_ui_preview_{c.lower().replace('-', '')}_v1.bin") for c in ("CAM-01", "CAM-04")}
    started_ns, samples, stopping = time.monotonic_ns(), [], False
    snapshots = []
    (output / "visual").mkdir()

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    lock = SourceOwnership(readers, ROOT / ".runtime/camera-owner-locks") if mode == "live" else nullcontext()
    with lock, (output / "deepstream.log").open("x") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        start = time.monotonic()
        try:
            while process.poll() is None and time.monotonic() - start < duration and not stopping:
                previews = {}
                for camera, reader in readers.items():
                    frame = reader.read_latest(metadata_only=True)
                    if frame and frame.timestamp_ns >= started_ns:
                        previews[camera] = {"sequence": frame.sequence, "wall_ns": time.monotonic_ns(),
                            "age_ms": (time.monotonic_ns() - frame.timestamp_ns) / 1e6,
                            "pts": frame.pts_ns, "t0": frame.decoder_reference_ns,
                            "t1": frame.decoder_out_ns, "t6": frame.timestamp_ns}
                samples.append({"mono_ns": time.monotonic_ns(), "preview": previews,
                    "rtsp_count": len(subprocess.check_output(["ss", "-Htnp", "state", "established", "( dport = :554 )"], text=True).splitlines()),
                    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True).strip()})
                diagnostics = stage / "logs/probe/preview_diagnostics.jsonl"
                if diagnostics.exists():
                    try:
                        samples[-1]["decoder"] = [json.loads(line) for line in diagnostics.read_text().splitlines()[-2:]]
                    except json.JSONDecodeError:
                        samples[-1]["decoder"] = None  # next sample retries a partial producer row
                if len(samples) % 15 == 0:
                    from PIL import Image
                    for camera, reader in readers.items():
                        frame = reader.read_latest()
                        if frame and frame.timestamp_ns >= started_ns:
                            path = output / "visual" / f"{camera}-{frame.sequence}.jpg"
                            Image.frombytes("RGBA", (frame.width, frame.height), frame.payload,
                                            "raw", "BGRA", frame.stride).convert("RGB").save(path)
                            snapshots.append({"camera_id": camera, "pts": frame.pts_ns, "path": str(path)})
                time.sleep(1)
        finally:
            active_duration = time.monotonic() - start
            stop_offset = (output / "deepstream.log").stat().st_size
            if process.poll() is None:
                subprocess.run(["docker", "kill", "--signal=SIGINT", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    room.stop_deepstream_container(container)
                    process.wait(timeout=20)
            for reader in readers.values():
                reader.close()
    save(output / "samples.json", samples)
    save(output / "visual/snapshots.json", snapshots)
    save(output / "protected-after.json", protected())
    result = summarize(output, samples, active_duration, tracker, stop_offset)
    result["exit_code"] = process.returncode
    result["staged_configs_unchanged"] = hashes == {p.name: sha256(p) for p in stage.iterdir() if p.is_file()}
    if not result["staged_configs_unchanged"] or process.returncode != 0:
        result["status"] = "FAIL"
    save(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--tracker", choices=("off", "production-reference"), default="off")
    parser.add_argument("--mode", choices=("live", "replay"), default="live")
    args = parser.parse_args()
    result = run(args.output, args.build, args.duration, args.tracker, args.mode)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
