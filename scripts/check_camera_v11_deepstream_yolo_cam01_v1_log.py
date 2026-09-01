#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

STATS_PREFIX = "CAMERA_V11_DS_YOLO_CAM01 "
ARCH_PREFIX = "CAMERA_V11_DS_YOLO_CAM01_ARCH "
POLICY_PREFIX = "CAMERA_V11_DS_YOLO_CAM01_POLICY "
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def parse_kv(line: str) -> dict[str, str]:
    return {key: value for key, value in KV_RE.findall(line)}


def as_float(row: dict[str, str], name: str, fallback: float = -1.0) -> float:
    try:
        return float(row.get(name, fallback))
    except (TypeError, ValueError):
        return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CAM-01 DeepStream + isolated TRT8.6 YOLO live log")
    parser.add_argument("--log", type=Path, default=Path("/tmp/CAMERA_V11_DS_YOLO_CAM01.log"))
    parser.add_argument("--require-person", action="store_true")
    parser.add_argument("--require-stale-expiry", action="store_true")
    parser.add_argument("--detector-disabled", action="store_true")
    parser.add_argument("--allow-warnings", action="store_true")
    parser.add_argument("--min-render-fps", type=float, default=8.0)
    parser.add_argument("--min-infer-hz", type=float, default=0.5)
    parser.add_argument("--max-queue", type=int, default=1)
    parser.add_argument("--min-runtime-stats", type=int, default=2)
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"V11_DS_YOLO_CAM01_CHECK RESULT=FAIL reason=log_missing path={args.log}")
        return 1

    text = args.log.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    reasons: list[str] = []
    arch_lines = [line for line in lines if line.startswith(ARCH_PREFIX)]
    policy_lines = [line for line in lines if line.startswith(POLICY_PREFIX)]
    arch = parse_kv(arch_lines[-1]) if arch_lines else {}
    policy = parse_kv(policy_lines[-1]) if policy_lines else {}
    if not arch:
        reasons.append("arch_marker_missing")
    if not policy:
        reasons.append("policy_marker_missing")
    if "CAMERA_V11_DS_YOLO_CAM01_START " not in text:
        reasons.append("start_marker_missing")
    if "CAMERA_V11_DS_YOLO_CAM01_ERROR " in text:
        reasons.append("pipeline_error_marker")
    if "401 Unauthorized" in text or "Unauthorized (401)" in text:
        reasons.append("rtsp_unauthorized")
    if "Traceback (most recent call last)" in text or "KeyboardInterrupt" in text:
        reasons.append("runtime_traceback")

    required_arch = {
        "rtsp_sources": "1",
        "decode": "deepstream-nvdec",
        "source": "nvurisrcbin",
        "detector": "trt86-sidecar",
        "detector_rtsp": "0",
        "detector_queue": "latest1",
        "detector_thread": "dedicated",
        "gst_nvinfer": "0",
        "second_rtsp": "0",
        "opencv": "0",
        "ffmpeg": "0",
    }
    for key, expected in required_arch.items():
        if arch.get(key) != expected:
            reasons.append(f"arch_{key}={arch.get(key, 'missing')}!={expected}")

    expected_enabled = "0" if args.detector_disabled else "1"
    if policy.get("enabled") != expected_enabled:
        reasons.append(f"policy_enabled={policy.get('enabled', 'missing')}!={expected_enabled}")
    if not args.detector_disabled and "CAMERA_V11_DS_YOLO_CAM01_DETECTOR_THREAD state=START" not in text:
        reasons.append("detector_thread_start_missing")

    rows = [parse_kv(line) for line in lines if line.startswith(STATS_PREFIX)]
    if len(rows) < args.min_runtime_stats:
        reasons.append(f"stats_rows={len(rows)}<{args.min_runtime_stats}")
    steady = [row for row in rows if as_float(row, "source_fps") > 0 and as_float(row, "render_fps") > 0]
    latest = steady[-1] if steady else (rows[-1] if rows else {})
    if not steady:
        reasons.append("steady_stats_missing")

    source_fps = as_float(latest, "source_fps")
    render_fps = as_float(latest, "render_fps")
    infer_hz = as_float(latest, "infer_hz")
    infer_buffers = int(as_float(latest, "infer_buffers", -1))
    latest_boxes = int(as_float(latest, "latest_boxes", -1))
    detector_thread_alive = int(as_float(latest, "detector_thread_alive", -1))
    worker_alive = int(as_float(latest, "worker_alive", -1))

    def max_metric(name: str, fallback: float = -1.0) -> float:
        values = [as_float(row, name, fallback) for row in rows]
        return max(values) if values else fallback

    queue = int(max_metric("queue", 999))
    source_queue = int(max_metric("source_queue", 999))
    display_queue = int(max_metric("display_queue", 999))
    detector_queue = int(max_metric("detector_queue", 999))
    positive_buffers = int(max_metric("positive_buffers"))
    detections_total = int(max_metric("detections_total"))
    max_objects = int(max_metric("max_objects"))
    metadata_added = int(max_metric("metadata_added"))
    result_clears = int(max_metric("result_clears", 0))
    stale_expirations = int(max_metric("stale_expirations", 0))
    warnings = int(max_metric("warnings", 999))
    copy_errors = int(max_metric("copy_errors", 999))
    infer_errors = int(max_metric("infer_errors", 999))
    meta_errors = int(max_metric("meta_errors", 999))
    pipeline_errors = int(max_metric("pipeline_errors", 999))

    if rows:
        if source_fps <= 0:
            reasons.append(f"source_fps={source_fps:.2f}")
        if render_fps < args.min_render_fps:
            reasons.append(f"render_fps={render_fps:.2f}<{args.min_render_fps:.2f}")
        for name, value in (
            ("queue", queue),
            ("source_queue", source_queue),
            ("display_queue", display_queue),
            ("detector_queue", detector_queue),
        ):
            if value > args.max_queue:
                reasons.append(f"{name}={value}>{args.max_queue}")
        if copy_errors != 0:
            reasons.append(f"copy_errors={copy_errors}")
        if infer_errors != 0:
            reasons.append(f"infer_errors={infer_errors}")
        if meta_errors != 0:
            reasons.append(f"meta_errors={meta_errors}")
        if pipeline_errors != 0:
            reasons.append(f"pipeline_errors={pipeline_errors}")
        if warnings != 0 and not args.allow_warnings:
            reasons.append(f"warnings={warnings}")
        if not args.detector_disabled:
            if infer_hz < args.min_infer_hz:
                reasons.append(f"infer_hz={infer_hz:.2f}<{args.min_infer_hz:.2f}")
            if infer_buffers <= 0:
                reasons.append(f"infer_buffers={infer_buffers}")
            if detector_thread_alive != 1:
                reasons.append(f"detector_thread_alive={detector_thread_alive}")
            if worker_alive != 1:
                reasons.append(f"worker_alive={worker_alive}")
        if args.require_person and (
            positive_buffers <= 0
            or detections_total <= 0
            or max_objects <= 0
            or metadata_added <= 0
        ):
            reasons.append(
                "person_not_detected_or_drawn "
                f"positive={positive_buffers} detections={detections_total} "
                f"max_objects={max_objects} metadata_added={metadata_added}"
            )
        if args.require_stale_expiry and (
            positive_buffers <= 0
            or latest_boxes != 0
            or result_clears + stale_expirations <= 0
        ):
            reasons.append(
                "stale_expiry_missing "
                f"positive={positive_buffers} latest={latest_boxes} "
                f"clears={result_clears} stale={stale_expirations}"
            )

    details = (
        f"stats_rows={len(rows)} source_fps={source_fps:.2f} render_fps={render_fps:.2f} "
        f"infer_hz={infer_hz:.2f} queue={queue}/{source_queue}/{display_queue}/{detector_queue} "
        f"infer_buffers={infer_buffers} positive_buffers={positive_buffers} "
        f"detections_total={detections_total} max_objects={max_objects} latest_boxes={latest_boxes} "
        f"result_clears={result_clears} stale_expirations={stale_expirations} "
        f"metadata_added={metadata_added} thread={detector_thread_alive} worker={worker_alive} "
        f"warnings={warnings} copy_errors={copy_errors} infer_errors={infer_errors} "
        f"meta_errors={meta_errors} pipeline_errors={pipeline_errors}"
    )
    if reasons:
        print(f"V11_DS_YOLO_CAM01_CHECK RESULT=FAIL reasons={';'.join(reasons)} {details}")
        return 1
    print(f"V11_DS_YOLO_CAM01_CHECK RESULT=PASS {details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
