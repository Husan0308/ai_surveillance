#!/usr/bin/env python3
"""Validate a finalized native-tee calibration recording session."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOM_CAMERAS = {
    "entrance": (("CAM-02", "Entrance 2"), ("CAM-05", "Entrance 1")),
    "main_room": (("CAM-03", "Main Room 1"), ("CAM-06", "Main Room 2")),
}
SYNC_METHODS = {
    "common monotonic session gate; per-frame source PTS and host monotonic timestamps",
    "shared monotonic start; each camera stops at the earlier of host-duration or first-source-PTS duration; included source PTS retained",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_read_frames",
         "-show_entries", "format=duration,size", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    if len(data.get("streams", [])) != 1:
        raise ValueError(f"expected one video stream in {path.name}")
    return {**data["streams"][0], **data.get("format", {})}


def read_rows(path: Path, stage: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed {path.name}:{line_no}: {exc}") from exc
            if row.get("stage") == stage:
                rows.append(row)
    return rows


def rate(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def validate(session_dir: Path) -> dict:
    metadata_path = session_dir / "recording_metadata.json"
    if not metadata_path.is_file():
        raise ValueError("native recorder did not atomically finalize recording_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    requested_duration = float(metadata.get("requested_duration_sec", 120))
    if requested_duration not in (120.0, 180.0):
        raise ValueError(f"unsupported requested duration: {requested_duration}")
    expected = ROOM_CAMERAS.get(metadata.get("room"))
    if not expected:
        raise ValueError(f"unsupported room in metadata: {metadata.get('room')!r}")
    duration = float(metadata.get("actual_duration_sec", 0))
    issues: list[str] = []
    if metadata.get("status") != "COMPLETE":
        issues.append(f"native recorder status is {metadata.get('status')!r}")
    if abs(duration - requested_duration) > 0.25:
        issues.append(f"recorded monotonic duration {duration:.6f}s is outside {requested_duration:.0f}s ±0.25s")
    actual_cameras = metadata.get("cameras", [])
    if [row.get("camera_id") for row in actual_cameras] != [pair[0] for pair in expected]:
        issues.append("recorded camera IDs/order do not match the configured room pair")
    cameras = []
    for ordinal, (camera_id, camera_name) in enumerate(expected, 1):
        native = next((row for row in actual_cameras if row.get("camera_id") == camera_id), None)
        video = session_dir / f"camera_{ordinal}.mp4"
        rows_path = session_dir / f"frames-{camera_id}.jsonl"
        if native is None:
            issues.append(f"missing native metadata for {camera_id}")
            continue
        if not video.is_file() or video.stat().st_size == 0:
            issues.append(f"{camera_id} MP4 missing or empty")
            continue
        if not rows_path.is_file():
            issues.append(f"{camera_id} per-frame timestamp rows missing")
            continue
        details = probe(video)
        decoded = read_rows(rows_path, "decoded_source")
        encoded = read_rows(rows_path, "encoded_access_unit")
        decoded_pts = [int(row["pts_ns"]) for row in decoded]
        encoded_pts = [int(row["pts_ns"]) for row in encoded]
        fps = rate(details["avg_frame_rate"])
        file_duration = float(details["duration"])
        decoded_count = int(native.get("decoded_input_frames", -1))
        encoded_count = int(native.get("encoded_access_units", -1))
        stage_counts = {
            "decoded_input": decoded_count,
            "queue_output": int(native.get("queue_output_frames", -1)),
            "converter_output": int(native.get("converted_frames", -1)),
            "encoder_input": int(native.get("encoder_input_frames", -1)),
            "encoder_output": int(native.get("encoder_output_frames", -1)),
            "parsed_access_unit": encoded_count,
        }
        if (int(details.get("width", 0)), int(details.get("height", 0))) != (
                int(native.get("width", -1)), int(native.get("height", -1))):
            issues.append(f"{camera_id} output dimensions differ from decoded source caps")
        if decoded_count != encoded_count or decoded_count != len(decoded) or encoded_count != len(encoded):
            issues.append(f"{camera_id} decoded/encoded/per-frame row counts differ")
        if any(count < 0 for count in stage_counts.values()):
            issues.append(f"{camera_id} native recorder did not provide all frame-stage counters")
        elif len(set(stage_counts.values())) != 1:
            ordered = list(stage_counts.items())
            first_difference = next((f"{left}={left_count} -> {right}={right_count}"
                                     for (left, left_count), (right, right_count)
                                     in zip(ordered, ordered[1:]) if left_count != right_count),
                                    "unknown stage")
            issues.append(f"{camera_id} recorder first frame loss boundary: {first_difference}")
        if int(native.get("queue_overruns", -1)) != 0:
            issues.append(f"{camera_id} recording branch queue overran")
        if decoded_count <= 0 or encoded_count <= 0:
            issues.append(f"{camera_id} recorded no frames")
        if abs(file_duration - requested_duration) > 0.25:
            issues.append(f"{camera_id} container duration {file_duration:.6f}s is outside {requested_duration:.0f}s ±0.25s")
        if not decoded_pts or any(right <= left for left, right in zip(decoded_pts, decoded_pts[1:])):
            issues.append(f"{camera_id} decoded source PTS is empty, repeated, or non-monotonic")
        if not encoded_pts or any(right <= left for left, right in zip(encoded_pts, encoded_pts[1:])):
            issues.append(f"{camera_id} encoded access-unit PTS is empty, repeated, or non-monotonic")
        if int(details.get("nb_read_frames", -1)) != encoded_count:
            issues.append(f"{camera_id} ffprobe decoded frame count differs from encoded access-unit count")
        cameras.append({
            "camera_id": camera_id, "camera_name": camera_name,
            "video": str(video.resolve()), "frame_rows": str(rows_path.resolve()),
            "sha256": sha256(video), "codec": details.get("codec_name"),
            "width": int(details.get("width", 0)), "height": int(details.get("height", 0)),
            "fps": fps, "duration_sec": file_duration, "bytes": int(details.get("size", 0)),
            "decoded_input_frames": decoded_count, "encoded_access_units": encoded_count,
            "frame_stage_counts": stage_counts,
            "container_decoded_frames": int(details.get("nb_read_frames", -1)),
            "source_pts_monotonic": bool(decoded_pts and all(b > a for a, b in zip(decoded_pts, decoded_pts[1:]))),
            "encoded_pts_monotonic": bool(encoded_pts and all(b > a for a, b in zip(encoded_pts, encoded_pts[1:]))),
            "first_source_pts_ns": decoded_pts[0] if decoded_pts else None,
            "last_source_pts_ns": decoded_pts[-1] if decoded_pts else None,
            "first_encoded_pts_ns": encoded_pts[0] if encoded_pts else None,
            "last_encoded_pts_ns": encoded_pts[-1] if encoded_pts else None,
            "queue_overruns": int(native.get("queue_overruns", -1)),
        })
    synchronization = metadata.get("synchronization", {})
    if synchronization.get("method") not in SYNC_METHODS:
        issues.append("synchronization method metadata is missing or unexpected")
    result = {
        "schema_version": 1,
        "room": metadata.get("room"),
        "session_id": metadata.get("session_id"),
        "status": "PASS" if not issues else "FAIL",
        "session_start_utc": metadata.get("start_utc"),
        "session_end_utc": metadata.get("end_utc"),
        "monotonic_duration_sec": duration,
        "requested_duration_sec": requested_duration,
        "max_pair_frame_arrival_skew_ms": synchronization.get("max_pair_frame_arrival_skew_ms"),
        "cross_camera_identity": metadata.get("cross_camera_identity"),
        "cameras": cameras,
        "issues": issues,
    }
    destination = session_dir / "recording_validation.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.session_dir.resolve())
    except Exception as exc:
        print(f"CALIBRATION_RECORDING_VALIDATION_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
