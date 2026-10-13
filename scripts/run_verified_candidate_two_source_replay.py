#!/usr/bin/env python3
"""Replay two unique MP4 sources through an immutable candidate's direct binary."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cam05_candidate_direct_replay import ffprobe_video, read_jsonl, stage_pts_normalized_stream_copy
from scripts.run_six_camera_v13_wall_candidate import validate_candidate, validate_yolo_context_verifier

IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
RUNTIME_SOURCE_SECTIONS = ("CAM-01", "CAM-04")
# PGIE_ISOLATION assigns source IDs 0 and 1 in load order; canonical telemetry
# labels those IDs CAM-01 and CAM-02 (source_id + 1), irrespective of the INI
# section labels. Keep this distinction explicit in every replay artifact.
RUNTIME_CAMERA_ALIASES = ("CAM-01", "CAM-02")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def two_source_config(first: Path, second: Path) -> str:
    lines = []
    for camera, path in zip(RUNTIME_SOURCE_SECTIONS, (first, second)):
        uri = f"file:///workspace/inputs/{path.name}"
        lines.extend((f"[{camera}]", f"uri={base64.b64encode(uri.encode()).decode()}",
                      "username=", "password=", "latency_ms=0", "drop_on_latency=false",
                      "decoder_low_latency_mode=false", "decoder_extra_surfaces=4", ""))
    lines.extend(("[validation]", "interrupt_camera=none", "interrupt_at=20",
                  "interrupt_seconds=12", "preview_enabled=false", "preview_port=5600", ""))
    return "\n".join(lines)


def physical_camera_row(row: dict, source_id_to_camera: dict[int, str],
                        alias_to_camera: dict[str, str]) -> dict:
    """Return a derived row with physical IDs while retaining runtime aliases."""
    mapped = dict(row)
    source_id = mapped.get("source_id")
    physical = None
    if isinstance(source_id, int) and source_id in source_id_to_camera:
        physical = source_id_to_camera[source_id]
    runtime_camera = mapped.get("camera_id")
    if runtime_camera in alias_to_camera:
        alias_physical = alias_to_camera[runtime_camera]
        if physical is not None and physical != alias_physical:
            raise RuntimeError(
                f"source/camera alias conflict: source_id={source_id} "
                f"maps to {physical}, but runtime label {runtime_camera} maps to {alias_physical}"
            )
        physical = alias_physical
    if physical is None:
        return mapped
    if runtime_camera is not None:
        mapped["runtime_camera_id"] = runtime_camera
    mapped["camera_id"] = physical
    mapped["physical_camera_id"] = physical
    if isinstance(mapped.get("objects"), list):
        objects = []
        for item in mapped["objects"]:
            obj = dict(item)
            if obj.get("camera_id") is not None:
                obj["runtime_camera_id"] = obj["camera_id"]
            obj["camera_id"] = physical
            obj["physical_camera_id"] = physical
            objects.append(obj)
        mapped["objects"] = objects
    if isinstance(mapped.get("detector_proposals"), list):
        mapped["detector_proposals"] = [
            {**item, "runtime_camera_id": item.get("camera_id"),
             "camera_id": physical, "physical_camera_id": physical}
            for item in mapped["detector_proposals"]
        ]
    return mapped


def write_physical_camera_evidence(output: Path, source_id_to_camera: dict[int, str]) -> dict[str, int]:
    """Write camera-ID-normalized copies; original native evidence stays immutable."""
    target = output / "physical-camera-mapped"
    target.mkdir(exist_ok=False)
    alias_to_camera = dict(zip(RUNTIME_CAMERA_ALIASES, source_id_to_camera.values()))
    filenames = (
        "canonical-objects.jsonl", "detections.jsonl", "frames.jsonl",
        "source-frames.jsonl", "tracks.jsonl", "tracker_decisions.jsonl",
        "tracker-detection-join.jsonl", "shadow_fallbacks.jsonl",
        "lowconf-fallbacks.jsonl", "lowconf-verifier.jsonl",
        "lowconf-candidates.jsonl", "lowconf-assignments.jsonl",
        "proposal-duplicate-suppressions.jsonl", "invalid_shadow_geometry.jsonl",
    )
    counts: dict[str, int] = {}
    for filename in filenames:
        source = output / filename
        if not source.is_file():
            continue
        rows = read_jsonl(source)
        mapped = [physical_camera_row(row, source_id_to_camera, alias_to_camera) for row in rows]
        destination = target / filename
        with destination.open("x", encoding="utf-8") as stream:
            for row in mapped:
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        counts[filename] = len(mapped)
    return counts


def batch_two_config(source: Path, target: Path) -> None:
    text = source.read_text(encoding="utf-8")
    if text.count("batch-size=6") != 1:
        raise RuntimeError(f"expected exactly one batch-size=6 in {source.name}")
    target.write_text(text.replace("batch-size=6", "batch-size=2", 1), encoding="utf-8")
    target.chmod(0o444)


def run(candidate: Path, first: Path, second: Path, output: Path,
        timeout_sec: int = 600, assert_jacket_fixture: bool = False,
        first_camera_id: str = "CAM-02", second_camera_id: str = "CAM-05",
        tracker_nan_diagnostics: bool = False) -> dict:
    candidate, first, second, output = map(Path.resolve, (candidate, first, second, output))
    if not candidate.is_relative_to(ROOT / ".runtime") or not output.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate and output must remain under .runtime")
    if output.exists():
        raise FileExistsError(f"refusing to reuse replay evidence directory: {output}")
    if not first.is_file() or not second.is_file() or sha256(first) == sha256(second):
        raise ValueError("two distinct existing video files are required")
    valid_camera_ids = {f"CAM-{index:02d}" for index in range(1, 7)}
    if first_camera_id not in valid_camera_ids or second_camera_id not in valid_camera_ids or first_camera_id == second_camera_id:
        raise ValueError("two distinct valid physical camera IDs are required")
    manifest = validate_candidate(candidate)
    validate_yolo_context_verifier(candidate, manifest)
    first_media, second_media = ffprobe_video(first), ffprobe_video(second)
    output.mkdir(parents=True)
    mount_dir = output / "inputs"
    mount_dir.mkdir()
    first_staged, second_staged = mount_dir / first.name, mount_dir / second.name
    first_stage = stage_pts_normalized_stream_copy(first, first_staged)
    second_stage = stage_pts_normalized_stream_copy(second, second_staged)
    if first_staged.name == second_staged.name:
        raise ValueError("input basenames must be distinct for the mounted replay")

    config_dir = output / "config"
    config_dir.mkdir()
    pgie = config_dir / "config-primary-b2.txt"
    auxiliary = config_dir / "config-aux768-b2.txt"
    batch_two_config(candidate / "config/config_infer_primary_yolo26m_raw_otm_lowconf023.txt", pgie)
    batch_two_config(candidate / "config/config_infer_aux_yolo26m_768.txt", auxiliary)
    metamux_text = (candidate / "config/config_metamux_yolo26m_aux768.txt").read_text()
    for key in ("src-ids-model-1", "src-ids-model-3"):
        line_prefix = key + "="
        old = [line for line in metamux_text.splitlines() if line.startswith(line_prefix)]
        if len(old) != 1 or old[0].split("=", 1)[1] != "0;1;2;3;4;5":
            raise RuntimeError(f"unexpected metamuX source mapping for {key}")
        metamux_text = metamux_text.replace(old[0], line_prefix + "0;1", 1)
    metamux = config_dir / "config-metamux-b2.txt"
    metamux.write_text(metamux_text, encoding="utf-8")
    metamux.chmod(0o444)
    source_id_to_camera = {0: first_camera_id, 1: second_camera_id}
    source_file = output / "sources.ini"
    source_file.write_text(two_source_config(first_staged, second_staged), encoding="utf-8")
    source_file.chmod(0o600)

    env = [
        ("NVIDIA_DRIVER_CAPABILITIES", "compute,utility,video"),
        ("PGIE_ISOLATION_SOURCE_COUNT", "2"), ("PGIE_ISOLATION_MUX_BATCH", "2"),
        ("PGIE_ISOLATION_MUX_WIDTH", "1920"), ("PGIE_ISOLATION_MUX_HEIGHT", "1080"),
        ("PGIE_ISOLATION_MUX_MEMORY_TYPE", "0"), ("PGIE_ISOLATION_LIVE_SOURCE", "0"),
        ("PGIE_ISOLATION_BATCHED_PUSH_TIMEOUT_US", "1000"),
        ("PGIE_ISOLATION_CONFIG", "/config/config_infer_primary_yolo26m_raw_otm.txt"),
        ("V11_LOWCONF_PRIMARY_CONFIG", "/work/config/config-primary-b2.txt"),
        ("V11_TRACKER_WIDTH", "1920"), ("V11_TRACKER_HEIGHT", "1080"),
        ("V11_VERBOSE_EVIDENCE", "1"), ("V11_SOURCE_STALL_RECOVERY", "0"),
        ("V11_LOWCONF_VERIFIER_ENABLE", "1"),
        ("V11_ACTIVE_TRACK_VERIFIER_ENABLE", "1"),
        ("V11_ACTIVE_TRACK_VERIFIER_FAIL_CLOSED", "1"),
        ("V11_LOWCONF_VERIFIER_CONFIG", "/config/yolo26m_context_verifier_448.txt"),
        ("V11_CONTEXT_VERIFIER_BATCH_SIZE", "2"),
        ("V11_CONTEXT_VERIFIER_NAME", "YOLO26m-448-FP16"),
        ("V11_CONTEXT_VERIFIER_EVIDENCE", "1"),
        ("V11_AUX_YOLO_ENABLE", "1"),
        ("V11_AUX_YOLO_CONFIG", "/work/config/config-aux768-b2.txt"),
        ("V11_AUX_YOLO_METAMUX_CONFIG", "/work/config/config-metamux-b2.txt"),
        ("V11_AUX_YOLO_INPUT_SIZE", "768"),
        ("V11_TRACKER_DECISION_DIAGNOSTICS", "1"),
    ]
    if tracker_nan_diagnostics:
        env.append(("V11_TRACKER_NAN_DIAGNOSTICS", "1"))
    command = ["docker", "run", "--rm", "--pull=never", "--init", "--gpus", "all",
        "--network=none", "--user", f"{os.getuid()}:{os.getgid()}"]
    for key, value in env:
        command.extend(("-e", f"{key}={value}"))
    command.extend((
        "--mount", f"type=bind,src={candidate / 'bin'},dst=/candidate,readonly",
        "--mount", f"type=bind,src={candidate / 'config'},dst=/config,readonly",
        "--mount", f"type=bind,src={candidate / 'models'},dst=/models,readonly",
        "--mount", f"type=bind,src={mount_dir},dst=/workspace/inputs,readonly",
        "--mount", f"type=bind,src={output},dst=/work",
        IMAGE, "/candidate/cam-six-yolo-local6-direct-gatea", "/work/sources.ini",
        str(int(max(first_media["duration_sec"], second_media["duration_sec"]) + 60)),
    ))
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "candidate_binary_sha256": manifest["direct_file_binary_sha256"],
        "candidate_live_binary_sha256": manifest["binary_sha256"],
        "candidate_source_tree_sha256": manifest["source_snapshot"]["tree_sha256"],
        "source_topology": "two unique direct-file sources; no repeated slots or looping",
        "runtime_source_sections": {"source_id_0": "CAM-01", "source_id_1": "CAM-04"},
        "runtime_camera_aliases": {"source_id_0": RUNTIME_CAMERA_ALIASES[0],
                                   "source_id_1": RUNTIME_CAMERA_ALIASES[1]},
        "physical_camera_ids": {"source_id_0": first_camera_id, "source_id_1": second_camera_id},
        "camera_id_resolution": "native direct-file telemetry uses slot aliases; physical-camera-mapped/*.jsonl contains normalized copies; original evidence is retained unchanged",
        "batch": 2,
        "mux": {"width": 1920, "height": 1080, "live_source": 0, "timeout_us": 1000},
        "source_id_mapping": {"source_id_0": first_camera_id,
                              "source_id_1": second_camera_id},
        "inputs": [{"camera_id": first_camera_id, "path": str(first), "sha256": sha256(first),
                    "media": first_media, "staged": first_stage},
                   {"camera_id": second_camera_id, "path": str(second), "sha256": sha256(second),
                    "media": second_media, "staged": second_stage}],
        "primary_engine_sha256": manifest["model_sha256"]["yolo26m_raw_otm_b6_fp16.engine"],
        "primary_parser_sha256": manifest["model_sha256"]["libyolo26rawperson.so"],
        "auxiliary_engine_sha256": manifest["model_sha256"]["yolo26m_raw_otm_768_b6_fp16.engine"],
        "verifier_engine_sha256": manifest["model_sha256"][manifest["yolo_context_verifier"]["engine"]],
        "calibration_enabled": False,
        "identity_enabled": False,
        "tracker_nan_diagnostics": tracker_nan_diagnostics,
    }, indent=2) + "\n")
    started = time.monotonic()
    with (output / "native.log").open("x") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout_sec, check=False)
    native_log = (output / "native.log").read_text(errors="replace")
    if completed.returncode != 0 or "PGIE_ISOLATION EOS all file sources drained" not in native_log or "fatal=0" not in native_log:
        raise RuntimeError(f"two-source candidate replay failed; inspect {output / 'native.log'}")
    rows = read_jsonl(output / "source-frames.jsonl")
    by_source = {0: [], 1: []}
    for row in rows:
        by_source[int(row["source_id"])].append(row)
    for source_id, source_rows, media in ((0, by_source[0], first_media), (1, by_source[1], second_media)):
        local = [int(row["source_local_frame_num"]) for row in source_rows]
        pts = [int(row["source_pts_ns"]) for row in source_rows]
        if len(local) != media["frames"] or local != list(range(media["frames"])) or \
                len(set(pts)) != len(pts) or any(right <= left for left, right in zip(pts, pts[1:])):
            raise RuntimeError(f"source {source_id} did not produce exact unique local frame coverage")
    canonical_rows = read_jsonl(output / "canonical-objects.jsonl")
    verifier_rows = read_jsonl(output / "lowconf-verifier.jsonl")
    physical_evidence_rows = write_physical_camera_evidence(output, source_id_to_camera)
    def canonical_frame(camera: str, frame: int) -> dict | None:
        return next((row for row in canonical_rows
                     if row.get("camera_id") == camera and
                     int(row.get("source_frame_num", -1)) == frame), None)
    jacket_f0 = canonical_frame("CAM-01", 0)
    jacket_f25 = canonical_frame("CAM-01", 25)
    distinct_f25 = canonical_frame("CAM-02", 25)
    jacket_rejections = {
        str(frame): [row for row in verifier_rows
            if row.get("camera_id") == "CAM-01" and int(row.get("frame_num", -1)) == frame
            and row.get("verification_kind") == "ACTIVE_TRACK_BIRTH"
            and row.get("status") == "NO_LOCAL_PERSON"]
        for frame in (0, 25)
    }
    invalid_geometry_rows = sum(1 for row in canonical_rows
        for obj in row.get("objects", [])
        if not isinstance(obj.get("bbox"), list) or len(obj["bbox"]) != 4
        or not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in obj["bbox"])
        or obj["bbox"][0] < 0 or obj["bbox"][1] < 0
        or obj["bbox"][2] <= obj["bbox"][0] or obj["bbox"][3] <= obj["bbox"][1]
        or obj["bbox"][2] > obj.get("bbox_coordinate_width", 0) + 0.1
        or obj["bbox"][3] > obj.get("bbox_coordinate_height", 0) + 0.1)
    jacket_counts = {
        "frame_0_person_boxes": len(jacket_f0.get("objects", [])) if jacket_f0 else None,
        "frame_25_person_boxes": len(jacket_f25.get("objects", [])) if jacket_f25 else None,
        "frame_0_active_birth_rejections": len(jacket_rejections["0"]),
        "frame_25_active_birth_rejections": len(jacket_rejections["25"]),
    }
    distinct_count = len(distinct_f25.get("objects", [])) if distinct_f25 else None
    result = {
        "status": "PASS_DIAGNOSTIC_REPLAY",
        "candidate_live_binary_sha256": manifest["binary_sha256"],
        "candidate_direct_binary_sha256": manifest["direct_file_binary_sha256"],
        "source_rows": {str(source): len(source_rows) for source, source_rows in by_source.items()},
        "source_id_to_physical_camera_id": {str(key): value for key, value in source_id_to_camera.items()},
        "physical_camera_mapped_evidence_rows": physical_evidence_rows,
        "physical_canonical_output": "physical-camera-mapped/canonical-objects.jsonl",
        "unique_source_frame_coverage": True,
        "eos": True,
        "return_code": completed.returncode,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "canonical_rows": len(canonical_rows),
        "verifier_rows": len(verifier_rows),
        "jacket_fixture": jacket_counts,
        "distinct_people_frame_25_box_count": distinct_count,
        "invalid_geometry_rows": invalid_geometry_rows,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if invalid_geometry_rows or (assert_jacket_fixture and jacket_counts != {
            "frame_0_person_boxes": 2, "frame_25_person_boxes": 2,
            "frame_0_active_birth_rejections": 1, "frame_25_active_birth_rejections": 1}) or \
            (assert_jacket_fixture and distinct_count != 3):
        raise RuntimeError("jacket or distinct-person acceptance assertions failed; inspect result.json and metadata rows")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--assert-jacket-fixture", action="store_true",
                        help="assert exactly two people and a verifier rejection at the saved jacket fixture's frames 0/25, plus three distinct people at source 1 frame 25")
    parser.add_argument("--first-camera-id", default="CAM-02")
    parser.add_argument("--second-camera-id", default="CAM-05")
    parser.add_argument("--tracker-nan-diagnostics", action="store_true",
                        help="capture per-frame invalid shadow geometry from the candidate tracker boundary")
    args = parser.parse_args()
    print(json.dumps(run(args.candidate, args.first, args.second, args.output,
                         args.timeout_sec, args.assert_jacket_fixture,
                         args.first_camera_id, args.second_camera_id,
                         args.tracker_nan_diagnostics), indent=2))


if __name__ == "__main__":
    main()
