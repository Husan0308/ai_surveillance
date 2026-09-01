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
    parser = argparse.ArgumentParser(description="Check CAM-01 direct DeepStream YOLO live log")
    parser.add_argument("--log", type=Path, default=Path("/tmp/CAMERA_V11_DS_YOLO_CAM01.log"))
    parser.add_argument("--require-person", action="store_true")
    parser.add_argument("--min-render-fps", type=float, default=8.0)
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
    if not rows:
        reasons.append("stats_marker_missing")
        latest: dict[str, str] = {}
    else:
        latest = rows[-1]

    def number(name: str, fallback: float = -1.0) -> float:
        try:
            return float(latest.get(name, fallback))
        except (TypeError, ValueError):
            return fallback

    render_fps = number("render_fps")
    source_fps = number("source_fps")
    queue = int(number("queue", 999))
    infer_buffers = int(number("infer_buffers", -1))
    positive_buffers = int(number("positive_buffers", -1))
    detections_total = int(number("detections_total", -1))
    max_objects = int(number("max_objects", -1))
    meta_errors = int(number("meta_errors", 999))
    errors = int(number("errors", 999))

    if rows:
        if source_fps <= 0:
            reasons.append(f"source_fps={source_fps:.2f}")
        if render_fps < args.min_render_fps:
            reasons.append(f"render_fps={render_fps:.2f}<{args.min_render_fps:.2f}")
        if queue > args.max_queue:
            reasons.append(f"queue={queue}>{args.max_queue}")
        if infer_buffers <= 0:
            reasons.append(f"infer_buffers={infer_buffers}")
        if meta_errors != 0:
            reasons.append(f"meta_errors={meta_errors}")
        if errors != 0:
            reasons.append(f"errors={errors}")
        if args.require_person and (positive_buffers <= 0 or detections_total <= 0 or max_objects <= 0):
            reasons.append(
                f"person_not_detected positive={positive_buffers} detections={detections_total} max_objects={max_objects}"
            )

    details = (
        f"source_fps={source_fps:.2f} render_fps={render_fps:.2f} queue={queue} "
        f"infer_buffers={infer_buffers} positive_buffers={positive_buffers} "
        f"detections_total={detections_total} max_objects={max_objects} "
        f"meta_errors={meta_errors} errors={errors}"
    )
    if reasons:
        print(f"V11_DS_YOLO_CAM01_CHECK RESULT=FAIL reasons={';'.join(reasons)} {details}")
        return 1
    print(f"V11_DS_YOLO_CAM01_CHECK RESULT=PASS {details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
