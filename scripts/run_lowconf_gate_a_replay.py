#!/usr/bin/env python3
"""Run exact direct-file Gate A replays against an isolated verifier candidate."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_gate_a_direct_file_replay import (
    IMAGE, INPUTS, NATIVE_SOURCE, EXPECTED_FRAMES, sha256,
    validate_source_frame_rows,
)

DEFAULT_CANDIDATE = ROOT / ".runtime/gate-a-realtime-verifier-20261006/candidate-native-r1"
REFERENCE = ROOT / ".runtime/approved-bbox-reference-v5-amended-20261006-r5/reference"
SOURCE_SLOTS = ["CAM-01", "CAM-01", "CAM-01", "CAM-04", "CAM-04", "CAM-04"]
EXPECTED_ARTIFACTS = {
    "engine": "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16",
    "parser": "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19",
    "primary": "fbfa196338b6781b7d65c3058d4f02069e9116963804d0c3b23d1bf2301e4113",
    "tracker": "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355",
    "verifier_engine": "0f08176e640738caf1f94720019057f984ce0663e9363fa614051f113cef5140",
    "verifier_onnx": "a6aa31618f8476261807b5ba26b9841c684708c66a5f0be460a25a972f8ee7ad",
}


def inputs_manifest() -> dict:
    rows = {}
    for camera, (path, expected) in INPUTS.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"pinned video hash mismatch: {camera}")
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames",
            "-show_entries", "format=duration", "-of", "json", str(path),
        ], text=True, capture_output=True, check=True)
        info = json.loads(probe.stdout)
        stream = info["streams"][0]
        actual = (int(stream["width"]), int(stream["height"]), stream["avg_frame_rate"],
                  int(stream["nb_frames"]), float(info["format"]["duration"]))
        if actual[:4] != (1920, 1080, "20/1", 2400) or abs(actual[4] - 120.0) > .01:
            raise RuntimeError(f"unexpected pinned video contract for {camera}: {actual}")
        rows[camera] = {"path": str(path), "sha256": expected, "codec": stream["codec_name"],
            "width": actual[0], "height": actual[1], "fps": 20, "frames": actual[3],
            "duration_sec": actual[4]}
    return rows


def candidate_artifacts(candidate: Path) -> dict:
    checks = {
        "engine": candidate / "models/yolo26m_raw_otm_b6_fp16.engine",
        "parser": candidate / "models/libyolo26rawperson.so",
        "primary": candidate / "config/config_infer_primary_yolo26m_raw_otm.txt",
        "tracker": candidate / "config/config_tracker_NvDCF_v13_local6.yml",
        "verifier_engine": candidate / "models/peopleNet/benchmarks/engines/peoplenet_dynamic_b2.engine",
        "verifier_onnx": candidate / "models/peopleNet/model/resnet34_peoplenet.onnx",
    }
    actual = {name: sha256(path) for name, path in checks.items()}
    if actual != EXPECTED_ARTIFACTS:
        raise RuntimeError(f"staged candidate artifact provenance mismatch: {actual}")
    if sha256(candidate / "config/config_infer_primary_yolo26m_raw_otm_lowconf023.txt") == EXPECTED_ARTIFACTS["primary"]:
        raise RuntimeError("low-confidence PGIE profile unexpectedly equals the frozen production profile")
    return actual


def compile_direct(candidate: Path) -> dict:
    binary = candidate / "bin/cam-six-yolo-local6-direct-gatea"
    source_hashes = {p.name: sha256(p) for p in sorted(NATIVE_SOURCE.iterdir()) if p.is_file()}
    command = [
        "docker", "run", "--rm", "--pull=never", "--network=none",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{NATIVE_SOURCE}:/src:ro", "-v", f"{candidate}:/work",
        IMAGE, "bash", "-lc",
        "g++ -std=c++17 -O2 -DYOLO26_PERSON -DYOLO26_TRACKER -DPGIE_ISOLATION "
        "-I/opt/nvidia/deepstream/deepstream/sources/includes "
        "$(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0) "
        "/src/main.cpp -o /work/bin/cam-six-yolo-local6-direct-gatea "
        "$(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0) "
        "-L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta -ldl -pthread "
        "-Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return {"binary": str(binary), "binary_sha256": sha256(binary), "source_sha256": source_hashes,
            "defines": ["YOLO26_PERSON", "YOLO26_TRACKER", "PGIE_ISOLATION"]}


def source_config() -> str:
    lines: list[str] = []
    for source_id, camera in enumerate(SOURCE_SLOTS):
        path = INPUTS[camera][0]
        uri = f"file:///workspace/inputs/videos/{path.name}"
        lines.extend((f"[CAM-{source_id + 1:02d}]", f"uri={base64.b64encode(uri.encode()).decode()}",
            "username=", "password=", "latency_ms=0", "drop_on_latency=false",
            "decoder_low_latency_mode=false", "decoder_extra_surfaces=4", ""))
    lines.extend(("[validation]", "interrupt_camera=none", "interrupt_at=20",
        "interrupt_seconds=12", "preview_enabled=false", "preview_port=5600", ""))
    return "\n".join(lines)


def docker_command(candidate: Path, run_dir: Path, mode: str) -> list[str]:
    lowconf = mode in {"rescue", "yolo-context"}
    config_name = "config_infer_primary_yolo26m_raw_otm_lowconf023.txt" if lowconf else "config_infer_primary_yolo26m_raw_otm.txt"
    command = ["docker", "run", "--rm", "--pull=never", "--init", "--gpus", "all",
        "--network=none", "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "-e", "GST_REGISTRY=/tmp/gate-a-lowconf-registry.bin",
        "-e", "PGIE_ISOLATION_SOURCE_COUNT=6", "-e", "PGIE_ISOLATION_MUX_BATCH=6",
        "-e", "PGIE_ISOLATION_MUX_WIDTH=1920", "-e", "PGIE_ISOLATION_MUX_HEIGHT=1080",
        "-e", "PGIE_ISOLATION_LIVE_SOURCE=0", "-e", "PGIE_ISOLATION_BATCHED_PUSH_TIMEOUT_US=-1",
        # Match the recorded 20 FPS operating rate. Unpaced decode reaches
        # ~51 FPS/source and can outrun the deliberately bounded publisher.
        "-e", "PGIE_ISOLATION_REALTIME_PACE=1",
        "-e", "PGIE_ISOLATION_MUX_MEMORY_TYPE=0", "-e", f"PGIE_ISOLATION_CONFIG=/config/{config_name}",
        "-e", "V11_TRACKER_WIDTH=1920", "-e", "V11_TRACKER_HEIGHT=1080",
        "-e", "V11_TRACKER_DECISION_DIAGNOSTICS=1", "-e", "V11_TRACKER_NAN_DIAGNOSTICS=1",
        # The candidate feed is production shared memory. Do not redirect its
        # per-frame temp-write/rename IPC path onto the host-mounted evidence
        # directory; canonical-objects.jsonl below is the replay evidence.
        "-e", "V11_TRACKED_OBJECT_FEED_DIR=/dev/shm",
        "--mount", f"type=bind,src={candidate / 'bin'},dst=/candidate,readonly",
        "--mount", f"type=bind,src={candidate / 'config'},dst=/config,readonly",
        "--mount", f"type=bind,src={candidate / 'models'},dst=/models,readonly",
        "--mount", f"type=bind,src={INPUTS['CAM-01'][0].parent},dst=/workspace/inputs/videos,readonly",
        "--mount", f"type=bind,src={run_dir},dst=/work",]
    if lowconf:
        command.extend(["-e", "V11_LOWCONF_VERIFIER_ENABLE=1",
            "-e", "V11_LOWCONF_PRIMARY_CONFIG=/config/config_infer_primary_yolo26m_raw_otm_lowconf023.txt"])
        if mode == "yolo-context":
            command.extend(["-e", "V11_ACTIVE_TRACK_VERIFIER_ENABLE=1",
                "-e", "V11_LOWCONF_VERIFIER_CONFIG=/config/yolo26m_context_verifier.txt",
                "-e", "V11_CONTEXT_VERIFIER_NAME=YOLO26m-640-FP16",
                "-e", "V11_CONTEXT_VERIFIER_EVIDENCE=1"])
        else:
            command.extend(["-e", "V11_LOWCONF_VERIFIER_CONFIG=/config/peoplenet_lowconf_verifier.txt"])
    command.extend([IMAGE, "/candidate/cam-six-yolo-local6-direct-gatea", "/work/sources.ini", "600"])
    return command


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    result = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                if line.strip(): result.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return result


def normalize_source_map(rows: list[dict]) -> dict[tuple[int, int], dict]:
    result = {}
    for row in rows:
        key = (int(row["source_id"]), int(row["source_pts_ns"]))
        if key in result: raise RuntimeError(f"duplicate source PTS mapping: {key}")
        result[key] = row
    return result


def normalized_high_detections(run_dir: Path, source_map: dict) -> list[tuple]:
    rows = []
    for row in read_rows(run_dir / "detections.jsonl"):
        source_id, pts = int(row["source_id"]), int(row["pts_ns"])
        source = source_map.get((source_id, pts))
        if source is None: raise RuntimeError(f"detector row lacks exact source frame: {source_id}/{pts}")
        if source_id not in (0, 3): continue
        rows.append((source_id, int(source["source_local_frame_num"]), int(row["class_id"]),
            round(float(row["confidence"]), 6), *(round(float(v), 3) for v in row["box"])))
    return sorted(rows)


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    den = a[2]*a[3] + b[2]*b[3] - inter
    return inter/den if den > 0 else 0.0


def evaluate_reference(run_dir: Path, source_map: dict) -> dict:
    canonical = read_rows(run_dir / "canonical-objects.jsonl")
    by_camera_frame: dict[tuple[str, int], list[dict]] = {}
    camera_source = {"CAM-01": 0, "CAM-04": 3}
    canonical_frame_mismatches = []
    for row in canonical:
        # The integrated canonical feed is camera-keyed (camera_id), while
        # detector/tracker evidence remains source-id keyed. Do not silently
        # assume the feed schema contains a source_id field.
        camera = row.get("camera_id")
        if camera not in ("CAM-01", "CAM-04"): continue
        local_frame = int(row["source_frame_num"])
        source_id = camera_source[camera]
        source = source_map.get((source_id, int(row["pts_ns"])))
        if source is None or int(source["source_local_frame_num"]) != local_frame:
            canonical_frame_mismatches.append({"camera_id": camera, "frame": local_frame,
                "pts_ns": int(row["pts_ns"]), "source_mapping": source})
        width, height = int(row["source_frame_width"]), int(row["source_frame_height"])
        for item in row.get("objects", []):
            box = item.get("bbox", [])
            if len(box) != 4 or not all(float(v) == float(v) and abs(float(v)) != float("inf") for v in box):
                raise RuntimeError(f"non-finite/invalid canonical bbox: {camera}/{local_frame}/{item}")
            if not (0 <= float(box[0]) < float(box[2]) <= width and
                    0 <= float(box[1]) < float(box[3]) <= height):
                raise RuntimeError(f"out-of-frame canonical bbox: {camera}/{local_frame}/{item}")
        by_camera_frame[(camera, local_frame)] = row.get("objects", [])

    expected: dict[str, dict[int, list[dict]]] = {"CAM-01": {}, "CAM-04": {}}
    for camera in expected:
        with (REFERENCE / f"{camera.replace('-', '')}-per-frame.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                value = row["chosen_bbox"].strip()
                if not value: continue
                box = [float(v) for v in json.loads(value)]
                expected[camera].setdefault(int(row["frame"]), []).append({
                    "bbox": box, "source": row["chosen_source"],
                    "native_id": row["chosen_native_track_id"],
                })
    report = {}
    for camera in expected:
        matched = missed = extras = 0
        minimum_iou = 1.0
        frame_ids = set(expected[camera])
        for frame in frame_ids:
            exp = expected[camera][frame]
            got_rows = by_camera_frame.get((camera, frame), [])
            got = []
            for item in got_rows:
                xyxy = [float(v) for v in item["bbox"]]
                got.append({"bbox": [xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]],
                    "source": item.get("bbox_source"), "native_id": item.get("native_track_id")})
            pairs = sorted(((iou(a["bbox"], b["bbox"]), ei, gi)
                for ei, a in enumerate(exp) for gi, b in enumerate(got)), reverse=True)
            used_e, used_g = set(), set()
            for score, ei, gi in pairs:
                if score < .80 or ei in used_e or gi in used_g: continue
                used_e.add(ei); used_g.add(gi); matched += 1; minimum_iou = min(minimum_iou, score)
            missed += len(exp) - len(used_e)
            extras += len(got) - len(used_g)
        report[camera] = {"expected_boxes": sum(map(len, expected[camera].values())),
            "compared_frames": len(frame_ids), "matched": matched, "missed": missed,
            "unmatched_candidate_boxes_in_scoped_frames": extras,
            "minimum_matched_iou": minimum_iou if matched else None}
    misses = {int(row["frame"]) for row in json.loads((REFERENCE.parent / "visible-person-misses.json").read_text())
              if row["camera_id"] == "CAM-04"}
    recovered = {}
    for frame in sorted(misses):
        objects = by_camera_frame.get(("CAM-04", frame), [])
        recovered[frame] = [row for row in objects if row.get("bbox_source") == "PGIE_VERIFIED_LOWCONF_FALLBACK"]
    return {"reference_id": "approved-bbox-reference-v5-amended-20261006-r5",
        "scoped_box_parity": report,
        "canonical_source_frame_pts_mismatches": canonical_frame_mismatches,
        "adjudicated_visible_person_misses": {"CAM-04": {str(frame): {
            "verified_lowconf_boxes": len(rows), "recovered": len(rows) == 1,
            "boxes": rows} for frame, rows in recovered.items()},
            "all_recovered": bool(recovered) and all(len(rows) == 1 for rows in recovered.values())},
        "transition_frame_435_boxes": [row for row in by_camera_frame.get(("CAM-01", 435), [])],
        "lowconf_fallback_rows": len(read_rows(run_dir / "lowconf-fallbacks.jsonl"))}


def run_mode(candidate: Path, output_root: Path, mode: str, input_meta: dict,
             direct_build: dict) -> dict:
    run_dir = output_root / mode
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "tracked-feed").mkdir()
    config = source_config()
    (run_dir / "sources.ini").write_text(config)
    (run_dir / "sources.ini").chmod(0o600)
    provenance = {"mode": mode, "source_mode": "six direct MP4 sources; three independent readers per unique clip for batch-6 diagnostic only",
        "unique_historical_inputs": input_meta, "source_slots": SOURCE_SLOTS,
        "source_frame_contract": "each decoder must produce exactly local frames 0..2399 once and EOS; no loop/restart",
        "mux": {"batch_size": 6, "size": [1920,1080], "live_source": False, "timeout_us": -1},
        "tracker": {"config_sha256": EXPECTED_ARTIFACTS["tracker"], "dimensions": [1920,1080],
                    "bbox_correction": "OFF", "identity": False},
        "yolo": {"engine_sha256": EXPECTED_ARTIFACTS["engine"], "parser_sha256": EXPECTED_ARTIFACTS["parser"],
                 "mode": "FP16 640x640 batch6", "highconf_threshold": .25,
                 "candidate_extraction_floor": .23 if mode in {"rescue", "yolo-context"} else .25},
        "verifier": (({"model": "YOLO26m frozen context crop",
                        "engine_sha256": "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16",
                        "parser_sha256": EXPECTED_ARTIFACTS["parser"], "batch": 1,
                        "input": [3,640,640], "threshold": .20,
                        "active_track_birth_gate": True}
                       if mode == "yolo-context" else
                       {"model": "PeopleNet 2.6.3", "engine_sha256": EXPECTED_ARTIFACTS["verifier_engine"],
                        "onnx_sha256": EXPECTED_ARTIFACTS["verifier_onnx"], "batch": 1,
                        "input": [3,544,960], "threshold": .20})
                      if mode in {"rescue", "yolo-context"} else None),
        "verifier_publication": ("per-camera ordered async publisher; exact-current-frame results only; low-confidence proposals remain outside NvDCF; new active-track canonical output waits for strict candidate-local verification; 26ms deadline; late results withheld; PGIE/NvDCF callback never waits"
                                 if mode == "yolo-context" else
                                 "per-camera ordered async publisher; waits only when exact-frame low-confidence candidate already has a unique continuous INACTIVE-shadow match; 26ms absolute deadline from sidecar submission; late results fail closed; PGIE/NvDCF callback never waits"
                                 if mode == "rescue" else "disabled"),
        "replay_pacing": {"enabled": True,
            "mechanism": "file-only GStreamer identity sync=1 before decoded tee",
            "preserves_source_pts": True, "source_rate_fps": 20,
            "reason": "match recorded/live rate and avoid unpaced benchmark overload of bounded publisher"},
        "direct_binary": direct_build}
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    command = docker_command(candidate, run_dir, mode)
    (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.monotonic()
    with (run_dir / "native.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            process.send_signal(signal.SIGINT)
            try: rc = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                try: rc = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); rc = process.wait(timeout=5)
            raise RuntimeError(f"direct replay timed out; evidence retained: {run_dir}")
    elapsed = time.monotonic() - started
    log_text = (run_dir / "native.log").read_text(errors="replace")
    if rc != 0 or "PGIE_ISOLATION EOS all file sources drained" not in log_text or "fatal=0" not in log_text:
        raise RuntimeError(f"{mode} direct replay failed rc={rc}; see {run_dir/'native.log'}")
    source_rows = read_rows(run_dir / "source-frames.jsonl")
    frame_audit = validate_source_frame_rows(source_rows, list(range(6)))
    source_map = normalize_source_map(source_rows)
    detectors = normalized_high_detections(run_dir, source_map)
    result = {"status": "PASS", "return_code": rc, "elapsed_sec": elapsed,
        "source_frame_audit": frame_audit,
        "direct_pgie_frames": len(read_rows(run_dir / "frames.jsonl")),
        "high_confidence_detections": len(detectors),
        "canonical_frames": len(read_rows(run_dir / "canonical-objects.jsonl")),
        "lowconf_candidate_rows": len(read_rows(run_dir / "lowconf-candidates.jsonl")),
        "lowconf_verifier_rows": len(read_rows(run_dir / "lowconf-verifier.jsonl")),
        "lowconf_fallback_rows": len(read_rows(run_dir / "lowconf-fallbacks.jsonl")),
        "reference_evaluation": evaluate_reference(run_dir, source_map)
            if mode in {"rescue", "yolo-context"} else None,
        "active_birth_verifier_rows": sum(1 for row in read_rows(run_dir / "lowconf-verifier.jsonl")
            if row.get("verification_kind") == "ACTIVE_TRACK_BIRTH"),
        "active_birth_confirmed": sum(1 for row in read_rows(run_dir / "lowconf-verifier.jsonl")
            if row.get("verification_kind") == "ACTIVE_TRACK_BIRTH" and row.get("status") == "CONFIRMED"),
        "active_birth_unconfirmed_or_late": sum(1 for row in read_rows(run_dir / "lowconf-verifier.jsonl")
            if row.get("verification_kind") == "ACTIVE_TRACK_BIRTH" and row.get("status") != "CONFIRMED")}
    (run_dir / "high-conf-detections.normalized.json").write_text(json.dumps(detectors, separators=(",", ":")) + "\n")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"run_dir": str(run_dir), "result": result, "high_detections": detectors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "rescue", "yolo-context", "both", "both-yolo"), default="both")
    args = parser.parse_args()
    candidate, output = args.candidate.resolve(), args.output.resolve()
    if not candidate.is_relative_to(ROOT / ".runtime") or not output.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate and evidence must remain inside .runtime")
    if not REFERENCE.is_dir(): raise FileNotFoundError(REFERENCE)
    if output.exists(): raise FileExistsError(f"refusing to reuse evidence directory: {output}")
    input_meta = inputs_manifest()
    artifacts = candidate_artifacts(candidate)
    direct_build = compile_direct(candidate)
    output.mkdir(parents=True)
    (output / "input-and-artifact-manifest.json").write_text(json.dumps({
        "inputs": input_meta, "artifacts": artifacts, "direct_build": direct_build,
        "reference_manifest_sha256": sha256(REFERENCE / "reference-manifest.json"),
    }, indent=2) + "\n")
    modes = (("baseline", "rescue") if args.mode == "both" else
             ("baseline", "yolo-context") if args.mode == "both-yolo" else (args.mode,))
    runs = {}
    for mode in modes:
        runs[mode] = run_mode(candidate, output, mode, input_meta, direct_build)
    high_parity = None
    candidate_mode = "yolo-context" if "yolo-context" in runs else "rescue"
    if "baseline" in runs and candidate_mode in runs:
        a, b = runs["baseline"]["high_detections"], runs[candidate_mode]["high_detections"]
        high_parity = {"exact": a == b, "baseline_rows": len(a), "rescue_rows": len(b),
            "first_difference_index": next((i for i, pair in enumerate(zip(a,b)) if pair[0] != pair[1]), None)}
    summary = {"status": "PASS" if high_parity is None or high_parity["exact"] else "FAIL",
        "high_confidence_pgie_input_unchanged": high_parity,
        "runs": {mode: value["result"] for mode, value in runs.items()}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
