#!/usr/bin/env python3
"""Run Gate A NvDCF topology diagnostics on exact direct-MP4 sources (never RTSP)."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / ".runtime/gate-a-direct-file-20261005-r3"
CANDIDATE = OUT_ROOT / "candidate"
CONFIG_DIR = OUT_ROOT / "config"
INPUT_DIR = ROOT / ".runtime/amc-transformer-personpresent-MtbZPbuh/amc_inputs/videos"
INPUTS = {
    "CAM-01": (INPUT_DIR / "cam_00.mp4", "1e2818e022b58f2f6608b9af24682f9dbc0d628ee80c7754f0c95a7c943f70f1"),
    "CAM-04": (INPUT_DIR / "cam_01.mp4", "2b73ecfd4c9361c02442b67966fe78e17aa2352e7b6f39eff0f0258de2ff425f"),
}
NATIVE_SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
ENGINE_SOURCE = ROOT / ".runtime/gate-a-tracker-dim-candidate-20261005-r2/models/yolo26m_raw_otm_b6_fp16.engine"
MODEL_DIR = ENGINE_SOURCE.parent
PGIE_SOURCE = ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt"
TRACKER_SOURCE = ROOT / "config/deepstream/config_tracker_NvDCF_v13_local6.yml"
PARSER_SOURCE = MODEL_DIR / "libyolo26rawperson.so"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
EXPECTED = {
    "engine": "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16",
    "parser": "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19",
    "pgie": "fbfa196338b6781b7d65c3058d4f02069e9116963804d0c3b23d1bf2301e4113",
    "tracker": "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355",
}
EXPECTED_FRAMES = 2400
EXPECTED_PTS_STEP_NS = 50_000_000
PTS_STEP_TOLERANCE_NS = 1_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(NATIVE_SOURCE.iterdir()) if path.is_file()}


def verify_inputs() -> dict[str, dict]:
    result = {}
    for camera, (path, expected_hash) in INPUTS.items():
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"{camera} pinned input missing or hash mismatch: {path}")
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames",
            "-show_entries", "format=duration", "-of", "json", str(path),
        ], text=True, capture_output=True, check=True)
        metadata = json.loads(probe.stdout)
        stream = metadata["streams"][0]
        # ffprobe returns a rational such as "20/1". Parse data, never eval it.
        fps = float(Fraction(stream["avg_frame_rate"]))
        duration = float(metadata["format"]["duration"])
        contract = (int(stream["width"]), int(stream["height"]), float(fps), int(stream["nb_frames"]), duration)
        if contract[:4] != (1920, 1080, 20.0, EXPECTED_FRAMES) or abs(duration - 120.0) > 0.01:
            raise RuntimeError(f"{camera} video contract mismatch: {contract}")
        result[camera] = {
            "path": str(path), "sha256": expected_hash, "codec": stream["codec_name"],
            "width": contract[0], "height": contract[1], "fps": contract[2],
            "frames": contract[3], "duration_sec": duration,
        }
    return result


def build_candidate() -> dict:
    if sha256(PGIE_SOURCE) != EXPECTED["pgie"]:
        raise RuntimeError("frozen YOLO PGIE config hash mismatch")
    if sha256(TRACKER_SOURCE) != EXPECTED["tracker"]:
        raise RuntimeError("local NvDCF diagnostic profile hash mismatch")
    if sha256(ENGINE_SOURCE) != EXPECTED["engine"] or sha256(PARSER_SOURCE) != EXPECTED["parser"]:
        raise RuntimeError("frozen YOLO engine/parser hash mismatch")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CANDIDATE.mkdir(parents=True, exist_ok=True)
    (CANDIDATE / "bin").mkdir(exist_ok=True)
    CONFIG_DIR.mkdir(exist_ok=True)
    for batch in (2, 6):
        staged = CONFIG_DIR / f"pgie-batch{batch}.txt"
        original = PGIE_SOURCE.read_text()
        changed, count = re.subn(r"(?m)^batch-size=\d+$", f"batch-size={batch}", original, count=1)
        if count != 1:
            raise RuntimeError("expected one nvinfer batch-size field")
        if staged.exists() and staged.read_text() != changed:
            raise RuntimeError(f"refusing to overwrite staged PGIE config: {staged}")
        if not staged.exists():
            staged.write_text(changed)
    tracker_staged = CONFIG_DIR / "config_tracker_NvDCF_v13_local6.yml"
    if tracker_staged.exists() and sha256(tracker_staged) != EXPECTED["tracker"]:
        raise RuntimeError(f"refusing to overwrite staged tracker config: {tracker_staged}")
    if not tracker_staged.exists():
        tracker_staged.write_bytes(TRACKER_SOURCE.read_bytes())

    binary = CANDIDATE / "bin/gate-a-direct-file"
    digest = source_hashes()
    manifest_path = CANDIDATE / "build-manifest.json"
    if binary.exists() or manifest_path.exists():
        if not binary.is_file() or not manifest_path.is_file():
            raise RuntimeError("incomplete direct-file candidate already exists; refusing to overwrite")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("source_sha256") != digest or manifest.get("binary_sha256") != sha256(binary):
            raise RuntimeError("existing direct-file candidate does not match current source; choose a fresh output root")
        return manifest

    build_script = (
        "g++ -std=c++17 -O2 -DYOLO26_PERSON -DYOLO26_TRACKER -DPGIE_ISOLATION "
        "-I/opt/nvidia/deepstream/deepstream/sources/includes "
        "$(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0) "
        "/src/main.cpp -o /work/bin/gate-a-direct-file "
        "$(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0) "
        "-L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -ldl -pthread "
        "-Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    command = [
        "docker", "run", "--rm", "--pull=never", "--user", f"{os.getuid()}:{os.getgid()}",
        "--mount", f"type=bind,src={NATIVE_SOURCE},dst=/src,readonly",
        "--mount", f"type=bind,src={CANDIDATE},dst=/work",
        IMAGE, "bash", "-lc", build_script,
    ]
    build_log = OUT_ROOT / "build.log"
    with build_log.open("x") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    manifest = {
        "deepstream_image": IMAGE,
        "compile_defines": ["YOLO26_PERSON", "YOLO26_TRACKER", "PGIE_ISOLATION"],
        "source_sha256": digest,
        "binary_sha256": sha256(binary),
        "engine_sha256": EXPECTED["engine"],
        "parser_sha256": EXPECTED["parser"],
        "pgie_source_sha256": EXPECTED["pgie"],
        "tracker_config_sha256": EXPECTED["tracker"],
        "role": "diagnostic-only; direct MP4; tracker raw outputs; no RTSP, identity, MV3DT, UI preview, or production config mutation",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def camera_config(case: str) -> tuple[str, list[str]]:
    if case == "R1":
        cameras = ["CAM-01", "CAM-04"]
        clips = ["CAM-01", "CAM-04"]
    elif case == "R2":
        cameras = [f"CAM-{index:02d}" for index in range(1, 7)]
        clips = ["CAM-01"] * 3 + ["CAM-04"] * 3
    else:
        raise ValueError("case must be R1 or R2")
    lines = []
    for camera, clip in zip(cameras, clips):
        path = INPUTS[clip][0]
        uri = f"file:///workspace/inputs/videos/{path.name}"
        lines.extend((f"[{camera}]", f"uri={base64.b64encode(uri.encode()).decode()}",
                      "username=", "password=", "latency_ms=0", "drop_on_latency=false",
                      "decoder_low_latency_mode=false", "decoder_extra_surfaces=4", ""))
    lines.extend(("[validation]", "interrupt_camera=none", "interrupt_at=20",
                  "interrupt_seconds=12", "preview_enabled=false", "preview_port=5600", ""))
    return "\n".join(lines), cameras


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed JSON at {path}:{line_number}: {exc}") from exc
    return rows


def validate_source_frame_rows(rows: list[dict], source_ids: list[int], expected_frames: int = EXPECTED_FRAMES) -> dict:
    by_source: dict[int, list[dict]] = {source_id: [] for source_id in source_ids}
    for row in rows:
        source_id = int(row["source_id"])
        if source_id not in by_source:
            raise ValueError(f"unexpected decoded source ID {source_id}")
        by_source[source_id].append(row)
    report = {}
    for source_id, source_rows in by_source.items():
        local = [int(row["source_local_frame_num"]) for row in source_rows]
        sequences = [int(row["decoded_sequence"]) for row in source_rows]
        pts = [int(row["source_pts_ns"]) for row in source_rows]
        expected = list(range(expected_frames))
        if len(source_rows) != expected_frames:
            raise ValueError(f"source {source_id}: expected {expected_frames} decoded frames, got {len(source_rows)}")
        if local != expected:
            raise ValueError(f"source {source_id}: source-local frame restart/gap/repeat: {local[:3]} ... {local[-3:]}")
        if sequences != list(range(1, expected_frames + 1)):
            raise ValueError(f"source {source_id}: decoded sequence restart/gap/repeat")
        diffs = [b - a for a, b in zip(pts, pts[1:])]
        if not all(delta > 0 for delta in diffs):
            raise ValueError(f"source {source_id}: source PTS restarted or repeated")
        if not all(abs(delta - EXPECTED_PTS_STEP_NS) <= PTS_STEP_TOLERANCE_NS for delta in diffs):
            raise ValueError(f"source {source_id}: unexpected source PTS cadence; possible dropped/repeated source frame")
        report[str(source_id)] = {
            "decoded_frames": len(source_rows), "source_local_first": local[0],
            "source_local_last": local[-1], "unique_local_frames": len(set(local)),
            "first_pts_ns": pts[0], "last_pts_ns": pts[-1],
            "min_pts_step_ns": min(diffs), "max_pts_step_ns": max(diffs),
            "pts_restarts": 0, "frame_restarts": 0,
        }
    return report


def write_normalized_rows(run_dir: Path, camera_ids: list[str], source_frame_rows: list[dict]) -> dict:
    source_map: dict[tuple[int, int], dict] = {}
    frame_sources: dict[int, str] = {}
    clips = ["CAM-01", "CAM-04"] if len(camera_ids) == 2 else ["CAM-01"] * 3 + ["CAM-04"] * 3
    id_to_camera = {index: clip for index, clip in enumerate(clips)}
    for row in source_frame_rows:
        key = (int(row["source_id"]), int(row["source_pts_ns"]))
        if key in source_map:
            raise ValueError(f"duplicate decoded source PTS row: {key}")
        source_map[key] = row
        frame_sources[key[0]] = id_to_camera[key[0]]

    frames = read_jsonl(run_dir / "frames.jsonl")
    if len(frames) != len(camera_ids) * EXPECTED_FRAMES:
        raise ValueError(f"PGIE frame-record count mismatch: {len(frames)}")
    batch_pts = sorted({int(row["batch_pts_ns"]) for row in frames})
    pipeline_index = {pts: index for index, pts in enumerate(batch_pts)}
    normalized_frames = []
    seen = set()
    normalized_by_source_local: dict[tuple[int, int], dict] = {}
    for row in frames:
        sid, pts = int(row["source_id"]), int(row["pts_ns"])
        source = source_map.get((sid, pts))
        if source is None:
            raise ValueError(f"PGIE frame has no decoded-source mapping: source={sid} pts={pts}")
        key = (sid, int(source["source_local_frame_num"]))
        if key in seen:
            raise ValueError(f"PGIE source frame repeated: {key}")
        seen.add(key)
        normalized = {
            "camera_id": frame_sources[sid], "source_id": sid,
            "pipeline_frame_index": pipeline_index[int(row["batch_pts_ns"])],
            "pipeline_batch_pts_ns": int(row["batch_pts_ns"]),
            "nvds_frame_num": int(row["frame"]),
            "source_local_frame_num": int(source["source_local_frame_num"]),
            "source_pts_ns": pts, "batch_id": int(row["batch_id"]),
            "persons": int(row["persons"]), "source_width": int(row["source_width"]),
            "source_height": int(row["source_height"]), "mux_width": int(row["pipeline_width"]),
            "mux_height": int(row["pipeline_height"]),
        }
        normalized_frames.append(normalized)
        normalized_by_source_local[key] = normalized
    for sid in range(len(camera_ids)):
        local_frames = sorted(row["source_local_frame_num"] for row in normalized_frames if row["source_id"] == sid)
        if local_frames != list(range(EXPECTED_FRAMES)):
            raise ValueError(f"PGIE frames are not exactly source-local 0..2399 for source {sid}")

    for filename, source_pts_key in (("detections.jsonl", "pts_ns"), ("tracks.jsonl", "pts_ns")):
        normalized_path = run_dir / f"normalized-{filename}"
        with normalized_path.open("x") as stream:
            for row in read_jsonl(run_dir / filename):
                sid, pts = int(row["source_id"]), int(row[source_pts_key])
                source = source_map.get((sid, pts))
                if source is None:
                    raise ValueError(f"{filename} row has no decoded source mapping: {sid}/{pts}")
                local_frame = int(source["source_local_frame_num"])
                batch_row = normalized_by_source_local.get((sid, local_frame))
                if batch_row is None:
                    raise ValueError(f"{filename} row has no PGIE frame record: {sid}/{local_frame}")
                stream.write(json.dumps({
                    **row,
                    "camera_id": frame_sources[sid],
                    "pipeline_frame_index": batch_row["pipeline_frame_index"],
                    "pipeline_batch_pts_ns": batch_row["pipeline_batch_pts_ns"],
                    "nvds_frame_num": int(row["frame"]),
                    "source_local_frame_num": int(source["source_local_frame_num"]),
                    "source_pts_ns": pts,
                }, separators=(",", ":")) + "\n")
    with (run_dir / "normalized-frames.jsonl").open("x") as stream:
        for row in normalized_frames:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {
        "frame_records": len(normalized_frames), "pipeline_frame_indices": len(batch_pts),
        "source_local_unique_frames_per_source": EXPECTED_FRAMES,
        "all_pgie_frames_mapped_to_decoded_source": True,
        "all_detector_and_tracker_rows_have_source_pts_mapping": True,
    }


def run_case(case: str, timeout_sec: int = 240) -> dict:
    manifest = build_candidate()
    source_files = verify_inputs()
    config, camera_ids = camera_config(case)
    source_count = len(camera_ids)
    run_dir = OUT_ROOT / case
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite Gate A run evidence: {run_dir}")
    run_dir.mkdir()
    source_config_path = run_dir / "sources.ini"
    source_config_path.write_text(config)
    source_config_path.chmod(0o600)
    command = [
        "docker", "run", "--rm", "--pull=never", "--init", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "-e", f"PGIE_ISOLATION_SOURCE_COUNT={source_count}",
        "-e", f"PGIE_ISOLATION_MUX_BATCH={source_count}",
        "-e", "PGIE_ISOLATION_MUX_WIDTH=1920", "-e", "PGIE_ISOLATION_MUX_HEIGHT=1080",
        "-e", "PGIE_ISOLATION_LIVE_SOURCE=0", "-e", "PGIE_ISOLATION_BATCHED_PUSH_TIMEOUT_US=-1",
        "-e", "PGIE_ISOLATION_MUX_MEMORY_TYPE=0",
        "-e", f"PGIE_ISOLATION_CONFIG=/config/pgie-batch{source_count}.txt",
        "-e", "PGIE_ISOLATION_CAPTURE_FRAMES=435",
        "-e", "V11_TRACKER_WIDTH=1920", "-e", "V11_TRACKER_HEIGHT=1080",
        "-e", "V11_TRACKER_DECISION_DIAGNOSTICS=1", "-e", "V11_TRACKER_NAN_DIAGNOSTICS=1",
        "--mount", f"type=bind,src={CANDIDATE / 'bin'},dst=/candidate,readonly",
        "--mount", f"type=bind,src={CONFIG_DIR},dst=/config,readonly",
        "--mount", f"type=bind,src={MODEL_DIR},dst=/models,readonly",
        "--mount", f"type=bind,src={INPUT_DIR},dst=/workspace/inputs/videos,readonly",
        "--mount", f"type=bind,src={run_dir},dst=/work",
        IMAGE, "/candidate/gate-a-direct-file", "/work/sources.ini", "600",
    ]
    runtime_overrides = {}
    for key in ("PGIE_ISOLATION_COMPUTE_HW", "PGIE_ISOLATION_ID_RESET_MODE"):
        value = os.environ.get(key)
        if value is not None:
            if not value.isdigit():
                raise RuntimeError(f"{key} must be an integer diagnostic override")
            runtime_overrides[key] = int(value)
    runtime_env_args = [arg for key, value in runtime_overrides.items()
                        for arg in ("-e", f"{key}={value}")]
    container_index = command.index(IMAGE)
    command[container_index:container_index] = runtime_env_args
    launch = {
        "case": case,
        "source_mode": "direct file:/// MP4 via nvurisrcbin; no local RTSP server, no RTSP connection, no loop",
        "camera_ids": camera_ids,
        "source_mapping": {str(index): clip for index, clip in enumerate(
            ["CAM-01", "CAM-04"] if case == "R1" else ["CAM-01"] * 3 + ["CAM-04"] * 3)},
        "input_videos": source_files,
        "mux": {"width": 1920, "height": 1080, "batch_size": source_count,
                "live_source": 0, "batched_push_timeout_us": -1, "sync_inputs": False},
        "pgie": {"batch_size": source_count, "source_config_sha256": EXPECTED["pgie"],
                 "staged_config_sha256": sha256(CONFIG_DIR / f"pgie-batch{source_count}.txt"),
                 "engine_sha256": EXPECTED["engine"], "parser_sha256": EXPECTED["parser"]},
        "tracker": {"profile": str(TRACKER_SOURCE.relative_to(ROOT)), "sha256": EXPECTED["tracker"],
                    "width": 1920, "height": 1080, "decision_diagnostics_only": True,
                    "runtime_property_overrides": runtime_overrides},
        "candidate_binary_sha256": manifest["binary_sha256"],
        "candidate_source_sha256": manifest["source_sha256"],
        "network_mode": "Docker default bridge is unused; RTSP/network input disabled",
        "identity_enabled": False, "mv3dt_enabled": False, "preview_enabled": False,
        "command_redacted": ["docker", "run", "--gpus all", "<direct MP4 config>", "DeepStream 9.1", "duration=600s"],
    }
    (run_dir / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    start = time.monotonic()
    with (run_dir / "native.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            return_code = process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            process.send_signal(signal.SIGINT)
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    return_code = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    return_code = process.wait(timeout=5)
            result = {"status": "FAIL", "reason": "direct-file replay timeout", "return_code": return_code}
            (run_dir / "exit.json").write_text(json.dumps(result, indent=2) + "\n")
            raise RuntimeError(f"{case} timed out; diagnostic shutdown evidence retained at {run_dir}")
    elapsed = time.monotonic() - start
    native_log = (run_dir / "native.log").read_text(errors="replace")
    if return_code != 0 or "GROUP END" not in native_log or "fatal=0" not in native_log:
        failure = {
            "status": "FAIL", "return_code": return_code, "elapsed_sec": elapsed,
            "reason": "native pipeline did not exit cleanly; source/frame audit not evaluated",
            "group_end": [line for line in native_log.splitlines() if line.startswith("GROUP END")],
        }
        (run_dir / "exit.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise RuntimeError(f"{case} native pipeline failed; inspect {run_dir / 'native.log'}")
    source_rows = read_jsonl(run_dir / "source-frames.jsonl")
    source_integrity = validate_source_frame_rows(source_rows, list(range(source_count)))
    normalized = write_normalized_rows(run_dir, camera_ids, source_rows)
    direct_source_lines = [line for line in native_log.splitlines() if "DIRECT_FILE_SOURCE" in line]
    if len(direct_source_lines) != source_count:
        raise RuntimeError(f"{case}: expected {source_count} direct-file source property records, got {len(direct_source_lines)}")
    if "PGIE_ISOLATION EOS all file sources drained" not in native_log:
        raise RuntimeError(f"{case}: normal file-source EOS was not observed")
    if "GROUP END" not in native_log or "fatal=0" not in native_log:
        raise RuntimeError(f"{case}: native pipeline did not end without fatal error")
    if return_code != 0:
        raise RuntimeError(f"{case}: native process exit={return_code}; inspect {run_dir / 'native.log'}")
    exit_record = {
        "status": "PASS", "return_code": return_code, "elapsed_sec": elapsed,
        "source_frame_integrity": source_integrity, "normalized_frame_accounting": normalized,
        "eos_observed": True, "file_loop": False, "rtsp_reconnect": "not_applicable_file_uri",
        "rtsp_connections_opened": 0, "native_direct_source_records": len(direct_source_lines),
    }
    (run_dir / "exit.json").write_text(json.dumps(exit_record, indent=2) + "\n")
    return exit_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("R1", "R2"), required=True)
    parser.add_argument("--timeout-sec", type=int, default=240)
    args = parser.parse_args()
    print(json.dumps(run_case(args.case, args.timeout_sec), indent=2))


if __name__ == "__main__":
    main()
