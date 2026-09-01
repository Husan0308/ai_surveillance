#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

PREFIX = "CAMERA_V11_DS_YOLO_CAM01 "
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def parse_kv(line: str) -> dict[str, str]:
    return {key: value for key, value in KV_RE.findall(line)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CAM-01 DeepStream + isolated TRT8.6 YOLO live log")
    parser.add_argument("--log", type=Path, default=Path("/tmp/CAMERA_V11_DS_YOLO_CAM01.log"))
    parser.add_argument("--require-person", action="store_true")
    parser.add_argument("--min-render-fps", type=float, default=8.0)
    parser.add_argument("--min-infer-hz", type=float, default=0.5)
    parser.add_argument("--max-queue", type=int, default=1)
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"V11_DS_YOLO_CAM01_CHECK RESULT=FAIL reason=log_missing path={args.log}")
        return 1

    text = args.log.read_text(encoding="utf-8", errors="replace")
    reasons: list[str] = []
    if "CAMERA_V11_DS_YOLO_CAM01_ARCH " not in text:
        reasons.append("arch_marker_missing")
    if "CAMERA_V11_DS_YOLO_CAM01_START " not in text:
        reasons.append("start_marker_missing")
    if "CAMERA_V11_DS_YOLO_CAM01_ERROR " in text:
        reasons.append("pipeline_error_marker")

    rows = [parse_kv(line) for line in text.splitlines() if line.startswith(PREFIX)]
    latest = rows[-1] if rows else {}
    if not rows:
        reasons.append("stats_marker_missing")

    def number(name: str, fallback: float = -1.0) -> float:
        try:
            return float(latest.get(name, fallback))
        except (TypeError, ValueError):
            return fallback

    source_fps = number("source_fps")
    render_fps = number("render_fps")
    infer_hz = number("infer_hz")
    queue = int(number("queue", 999))
    infer_buffers = int(number("infer_buffers", -1))
    positive_buffers = int(number("positive_buffers", -1))
    detections_total = int(number("detections_total", -1))
    max_objects = int(number("max_objects", -1))
    metadata_added = int(number("metadata_added", -1))
    infer_errors = int(number("infer_errors", 999))
    meta_errors = int(number("meta_errors", 999))
    errors = int(number("errors", 999))

    if rows:
        if source_fps <= 0:
            reasons.append(f"source_fps={source_fps:.2f}")
        if render_fps < args.min_render_fps:
            reasons.append(f"render_fps={render_fps:.2f}<{args.min_render_fps:.2f}")
        if infer_hz < args.min_infer_hz:
            reasons.append(f"infer_hz={infer_hz:.2f}<{args.min_infer_hz:.2f}")
        if queue > args.max_queue:
            reasons.append(f"queue={queue}>{args.max_queue}")
        if infer_buffers <= 0:
            reasons.append(f"infer_buffers={infer_buffers}")
        if infer_errors != 0:
            reasons.append(f"infer_errors={infer_errors}")
        if meta_errors != 0:
            reasons.append(f"meta_errors={meta_errors}")
        if errors != 0:
            reasons.append(f"errors={errors}")
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

    details = (
        f"source_fps={source_fps:.2f} render_fps={render_fps:.2f} infer_hz={infer_hz:.2f} "
        f"queue={queue} infer_buffers={infer_buffers} positive_buffers={positive_buffers} "
        f"detections_total={detections_total} max_objects={max_objects} metadata_added={metadata_added} "
        f"infer_errors={infer_errors} meta_errors={meta_errors} errors={errors}"
    )
    if reasons:
        print(f"V11_DS_YOLO_CAM01_CHECK RESULT=FAIL reasons={';'.join(reasons)} {details}")
        return 1
    print(f"V11_DS_YOLO_CAM01_CHECK RESULT=PASS {details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
