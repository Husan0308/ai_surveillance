"""Finalize opt-in six-camera stage snapshots after the pipeline has stopped."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from scripts.cam_six_validation.credential_redaction import (
    CredentialRedactor,
    child_environment_without_camera_secrets,
)


SAMPLE_SECONDS = (5, 30, 55)
PTS_NONE = (1 << 64) - 1


def _pts(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return None if result < 0 or result >= PTS_NONE else result


def compare_source_to_mosaic(source: dict, mosaics: list[dict]) -> dict:
    """The tiled output has no trustworthy per-camera frame identity.

    Do not interpret its synthetic NvDsFrameMeta, buffer offsets, or source-0
    PTS-0 records as a camera crosswalk, even in older evidence files.
    """
    result = {
        "relation": "uncorrelated",
        "camera_id": source.get("camera_id"),
        "source_id": source.get("source_id"),
        "source_filename": source.get("filename"),
        "reason": "per-camera frame identity is unavailable after tiling",
    }
    source_id = source.get("source_id")
    elapsed = source.get("elapsed_monotonic_us")
    target = source.get("target_sec")
    if (not isinstance(source_id, int) or isinstance(source_id, bool)
            or source_id not in range(6) or type(elapsed) is not int
            or target not in SAMPLE_SECONDS
            or not target * 1_000_000 <= elapsed <= (target + 2) * 1_000_000):
        return result
    candidates = [row for row in mosaics
                  if row.get("target_sec") == source.get("target_sec")
                  and type(row.get("elapsed_monotonic_us")) is int
                  and target * 1_000_000 <= row["elapsed_monotonic_us"] <= (target + 2) * 1_000_000
                  and row.get("status") == "saved"]
    if not candidates:
        return result
    delta = min(abs(row["elapsed_monotonic_us"] - elapsed) for row in candidates)
    nearest = [row for row in candidates if abs(row["elapsed_monotonic_us"] - elapsed) == delta]
    if len(nearest) != 1 or delta > 2_000_000:
        result["reason"] = "ambiguous or distant monotonic sampling times; no frame crosswalk"
        return result
    mosaic = nearest[0]
    result.update({
        "relation": "time_adjacent_only",
        "mosaic_filename": mosaic.get("filename"),
        "delta_monotonic_us": delta,
        "source_pts_ns": _pts(source.get("buffer_pts_ns")),
        "mosaic_pts_ns": _pts(mosaic.get("buffer_pts_ns")),
        "note": "Proximity of capture callbacks does not prove that this source frame appears in the mosaic.",
    })
    return result


def compare_mosaic_to_decoded(mosaic: dict, decoded_frames: list[dict]) -> dict:
    mosaic_pts = _pts(mosaic.get("buffer_pts_ns"))
    candidates = []
    for frame in decoded_frames:
        if frame.get("target_sec") != mosaic.get("target_sec"):
            continue
        if frame.get("status") != "saved":
            continue
        decoded_pts = _pts(frame.get("pts_ns"))
        if decoded_pts is not None and mosaic_pts is not None:
            candidates.append((abs(decoded_pts - mosaic_pts), frame))
    if not candidates:
        return {
            "relation": "unmatched",
            "mosaic_filename": mosaic.get("filename"),
            "mosaic_pts_ns": mosaic_pts,
            "reason": "no decoded MKV timestamp available",
        }
    delta = min(item[0] for item in candidates)
    nearest = [frame for value, frame in candidates if value == delta]
    if len(nearest) != 1:
        return {
            "relation": "ambiguous_duplicate_pts",
            "mosaic_filename": mosaic.get("filename"),
            "mosaic_pts_ns": mosaic_pts,
            "candidate_count": len(nearest),
            "delta_ns": delta,
        }
    frame = nearest[0]
    if frame.get("timestamp_candidate_count") is None:
        return {
            "relation": "uncorrelated",
            "mosaic_filename": mosaic.get("filename"),
            "reason": "decoded timestamp uniqueness was not established",
        }
    if frame["timestamp_candidate_count"] != 1:
        return {
            "relation": "ambiguous_duplicate_pts",
            "mosaic_filename": mosaic.get("filename"),
            "mosaic_pts_ns": mosaic_pts,
            "candidate_count": frame["timestamp_candidate_count"],
            "delta_ns": delta,
        }
    return {
        "relation": "exact_mosaic_pts" if delta == 0 else "time_adjacent_only",
        "mosaic_filename": mosaic.get("filename"),
        "mkv_image": frame.get("image"),
        "mosaic_pts_ns": mosaic_pts,
        "mkv_pts_ns": frame.get("pts_ns"),
        "delta_ns": delta,
        "mkv_frame_index": frame.get("frame_index"),
    }


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"kind": "invalid_record", "line": len(rows) + 1})
    return rows


def _make_nv12_preview(record: dict, directory: Path, ffmpeg: str | None) -> str | None:
    filename = record.get("filename")
    if not filename or not ffmpeg:
        return None
    raw = directory / filename
    width, height = record.get("width"), record.get("height")
    if not raw.is_file() or not isinstance(width, int) or not isinstance(height, int):
        return None
    if (record.get("packed_layout") != "NV12_Y_UV_tight"
            or width <= 0 or height <= 0 or width % 2 or height % 2
            or record.get("packed_row_bytes") != [width, width]
            or record.get("raw_plane_width") != [width, width // 2]
            or record.get("raw_plane_height") != [height, height // 2]):
        return None
    expected = width * height * 3 // 2
    if expected > 64 * 1024 * 1024 or raw.stat().st_size != expected:
        return None
    raw_width, raw_height = width, height
    image = raw.with_suffix(".png")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
        "-pixel_format", "nv12", "-video_size", f"{raw_width}x{raw_height}",
        "-i", str(raw), "-vf", f"crop={width}:{height}:0:0",
        "-frames:v", "1", "-y", str(image),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            env=child_environment_without_camera_secrets(),
        )
    except (OSError, subprocess.TimeoutExpired):
        image.unlink(missing_ok=True)
        return None
    if result.returncode == 0 and image.is_file():
        image.chmod(0o600)
    else:
        image.unlink(missing_ok=True)
    return image.name if result.returncode == 0 and image.is_file() else None


def _decode_mkv_samples(recording: Path, directory: Path, ffmpeg: str | None,
                        ffprobe: str | None, mosaics: list[dict],
                        redact_text: Callable[[str], str] | None = None) -> tuple[list[dict], dict]:
    if not recording.is_file():
        return [], {"status": "recording_missing"}
    if not ffmpeg or not ffprobe:
        return [], {"status": "ffmpeg_or_ffprobe_unavailable"}
    try:
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_streams",
             "-show_entries", "frame=best_effort_timestamp,best_effort_timestamp_time,width,height:stream=time_base",
             "-of", "json", str(recording)],
            capture_output=True, text=True, timeout=60,
            env=child_environment_without_camera_secrets(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return [], {"status": "ffprobe_execution_failed"}
    if probe.returncode != 0:
        error_text = probe.stderr[-500:]
        if redact_text is not None:
            error_text = redact_text(error_text)
        return [], {"status": "ffprobe_failed", "returncode": probe.returncode,
                    "stderr": error_text}
    try:
        probe_data = json.loads(probe.stdout)
        frames = probe_data.get("frames", [])
        stream_time_base = Fraction(probe_data["streams"][0]["time_base"])
    except (json.JSONDecodeError, KeyError, IndexError, ValueError, ZeroDivisionError):
        return [], {"status": "ffprobe_metadata_invalid"}
    pts_values = []
    for index, frame in enumerate(frames):
        try:
            if frame.get("best_effort_timestamp") not in (None, "N/A"):
                pts_ns = round(int(frame["best_effort_timestamp"]) * stream_time_base * 1_000_000_000)
            else:
                pts_ns = round(float(frame["best_effort_timestamp_time"]) * 1_000_000_000)
        except (KeyError, TypeError, ValueError):
            continue
        pts_values.append((index, pts_ns, frame))
    if not pts_values:
        return [], {"status": "no_decoded_frame_pts"}

    first_pts = pts_values[0][1]
    selected = []
    for target in SAMPLE_SECONDS:
        mosaic = next((row for row in mosaics if row.get("target_sec") == target), None)
        target_pts = _pts(mosaic.get("buffer_pts_ns")) if mosaic else None
        wanted_pts = target_pts if target_pts is not None else first_pts + target * 1_000_000_000
        nearest_delta = min(abs(item[1] - wanted_pts) for item in pts_values)
        nearest_frames = [item for item in pts_values if abs(item[1] - wanted_pts) == nearest_delta]
        frame_index, pts_ns, frame_meta = nearest_frames[0]
        image_name = f"mkv-decoded-t{target:02d}s.png"
        image_path = directory / image_name
        # Select by decoded frame index; the stored PTS comes from ffprobe for that frame.
        vf = f"select=eq(n\\,{frame_index})"
        try:
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(recording),
                 "-map", "0:v:0", "-vf", vf, "-vsync", "0", "-frames:v", "1",
                 "-y", str(image_path)],
                capture_output=True, text=True, timeout=60,
                env=child_environment_without_camera_secrets(),
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None
        saved = result is not None and result.returncode == 0 and image_path.is_file()
        if saved:
            image_path.chmod(0o600)
        else:
            image_path.unlink(missing_ok=True)
        selected.append({
            "stage": "decoded_saved_mkv",
            "target_sec": target,
            "status": "saved" if saved else "decode_failed",
            "image": image_name if saved else None,
            "frame_index": frame_index,
            "timestamp_candidate_count": len(nearest_frames),
            "pts_ns": pts_ns,
            "width": frame_meta.get("width"),
            "height": frame_meta.get("height"),
            "relative_to_first_pts_ns": pts_ns - first_pts,
            "selection_basis": "nearest_pre_nvenc_pts" if target_pts is not None else "target_relative_to_first_mkv_pts",
            "pre_nvenc_pts_ns": target_pts,
            "pre_nvenc_delta_ns": None if target_pts is None else abs(pts_ns - target_pts),
        })
    return selected, {
        "status": "pass" if all(row["status"] == "saved" for row in selected) else "partial",
        "probed_frame_count": len(pts_values),
        "selected_frame_count": sum(row["status"] == "saved" for row in selected),
    }


def finalize_stage_capture(directory: Path, recording: Path,
                           redact_text: Callable[[str], str] | None = None) -> Path:
    """Create PNG review samples and an explicit exact-vs-adjacent manifest offline."""
    if redact_text is None:
        # Even non-camera callers must not serialize a complete RTSP URL from
        # a diagnostic subprocess error. The validator supplies its
        # credential-aware sanitizer for separately configured user/pass.
        redact_text = CredentialRedactor().sanitize
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    rows = _read_jsonl(directory / "stage-frames.jsonl")
    snapshots = [r for r in rows if r.get("kind") == "snapshot"]
    mosaics = [r for r in snapshots if r.get("stage") == "pre_nvenc" and r.get("status") == "saved"]
    sources = [r for r in snapshots if r.get("stage") == "source_decoded" and r.get("status") == "saved"]

    for record in snapshots:
        if record.get("status") == "saved":
            record["preview_image"] = _make_nv12_preview(record, directory, ffmpeg)

    decoded, decoded_status = _decode_mkv_samples(
        recording, directory, ffmpeg, ffprobe, mosaics, redact_text=redact_text
    )
    source_joins = [compare_source_to_mosaic(row, mosaics) for row in sources]
    mosaic_joins = [compare_mosaic_to_decoded(row, decoded) for row in mosaics]
    capture_session_complete = any(row.get("kind") == "summary" for row in rows)
    queue_dropped = next(
        (row.get("queue_dropped") for row in rows if row.get("kind") == "summary"),
        None,
    )
    manifest = {
        "schema": 2,
        "capture_session_complete": capture_session_complete,
        "queue_dropped": queue_dropped,
        "sample_targets_sec": list(SAMPLE_SECONDS),
        "stages": [
            "decoded camera source output (NVMM NV12 raw snapshot)",
            "pre-NVENC 3x2 mosaic after tiler/conversion/optional OSD",
            "offline-decoded saved MKV output",
        ],
        "direct_comparison_rules": {
            "source_to_pre_nvenc": (
                "No exact camera-frame join is available after tiling. Synthetic metadata is ignored. "
                "Same-target monotonic samples within 2 seconds are time_adjacent_only; "
                "all other comparisons are uncorrelated. Source PTS and mosaic PTS are not assumed equivalent."
            ),
            "pre_nvenc_to_decoded_mkv": (
                "Exact only when pre-NVENC GstBuffer PTS equals the decoded MKV best-effort PTS; "
                "otherwise the nearest timestamp is labeled time_adjacent_only."
            ),
            "visual_comparison": "Source frames and mosaic crops have different dimensions/layout; they are not pixel-identical images.",
        },
        "limitations": [
            "capture_session_complete is true only when the native worker drained and wrote its final summary after the pipeline reached verified NULL.",
            "Samples are first buffers within 2 seconds after each monotonic target, not synchronized exposure triggers.",
            "No exact-frame claim is made when source IDs or timestamps are absent or differ.",
            "Tiled NvDsFrameMeta does not establish contributing camera-frame identity; no source-to-mosaic exact join is claimed.",
        ],
        "capture_records": rows,
        "decoded_mkv_samples": decoded,
        "decoded_mkv_status": decoded_status,
        "source_to_pre_nvenc": source_joins,
        "pre_nvenc_to_decoded_mkv": mosaic_joins,
    }
    if redact_text is not None:
        manifest = _sanitize_manifest(manifest, redact_text)
    path = directory / "stage-capture-comparability.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _sanitize_manifest(value: Any, redact_text: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_sanitize_manifest(item, redact_text) for item in value]
    if isinstance(value, dict):
        return {redact_text(key) if isinstance(key, str) else key:
                _sanitize_manifest(item, redact_text) for key, item in value.items()}
    return value
