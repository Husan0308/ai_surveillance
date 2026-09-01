#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

ARCH_PREFIX = "CAMERA_V11_DS_YOLO_MULTI_ARCH "
STAT_PREFIX = "CAMERA_V11_DS_YOLO_MULTI "
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def parse_kv(line: str) -> dict[str, str]:
    return {key: value for key, value in KV_RE.findall(line)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check V11 CAM-01/CAM-02 DeepStream + shared TRT8.6 log")
    parser.add_argument("--log", type=Path, default=Path("/tmp/CAMERA_V11_DS_YOLO_CAM01_CAM02.log"))
    parser.add_argument("--required-cameras", default="CAM-01,CAM-02")
    parser.add_argument("--min-runtime-stats", type=int, default=5)
    parser.add_argument("--min-render-fps", type=float, default=15.0)
    parser.add_argument("--min-source-fps", type=float, default=8.0)
    parser.add_argument("--min-infer-hz", type=float, default=1.20)
    parser.add_argument("--max-infer-skew-hz", type=float, default=0.75)
    parser.add_argument("--max-queue", type=int, default=1)
    parser.add_argument("--require-person-camera", action="append", default=[])
    parser.add_argument("--require-stale-expiry-camera", action="append", default=[])
    args = parser.parse_args()

    required = [x.strip() for x in args.required_cameras.split(",") if x.strip()]
    if not required:
        print("V11_DS_YOLO_MULTI_CHECK RESULT=FAIL reason=no_required_cameras")
        return 1
    if not args.log.is_file():
        print(f"V11_DS_YOLO_MULTI_CHECK RESULT=FAIL reason=log_missing path={args.log}")
        return 1

    text = args.log.read_text(encoding="utf-8", errors="replace")
    reasons: list[str] = []
    if "Traceback (most recent call last)" in text:
        reasons.append("runtime_traceback")
    if "CAMERA_V11_DS_YOLO_MULTI_ERROR " in text:
        reasons.append("pipeline_error_marker")
    if "CAMERA_V11_DS_YOLO_MULTI_DETECTOR_THREAD state=START" not in text:
        reasons.append("detector_thread_start_missing")
    if "CAMERA_V11_DS_YOLO_MULTI_START " not in text:
        reasons.append("start_marker_missing")

    arch_lines = [line for line in text.splitlines() if line.startswith(ARCH_PREFIX)]
    if not arch_lines:
        reasons.append("arch_marker_missing")
        arch: dict[str, str] = {}
    else:
        arch = parse_kv(arch_lines[-1])
        expected = len(required)
        if arch.get("rtsp_sources") != str(expected):
            reasons.append(f"arch_rtsp_sources={arch.get('rtsp_sources')}!={expected}")
        if arch.get("rtsp_sessions") != str(expected):
            reasons.append(f"arch_rtsp_sessions={arch.get('rtsp_sessions')}!={expected}")
        if arch.get("rtsp_per_camera") != "1":
            reasons.append("arch_rtsp_per_camera")
        if arch.get("detector_workers") != "1":
            reasons.append("arch_detector_workers")
        if arch.get("detector_rtsp") != "0":
            reasons.append("arch_detector_rtsp")
        if arch.get("second_rtsp") != "0":
            reasons.append("arch_second_rtsp")
        if arch.get("detector_queue") != "latest1-per-camera":
            reasons.append("arch_detector_queue")
        if arch.get("scheduler") != "round-robin":
            reasons.append("arch_scheduler")
        if arch.get("gst_nvinfer") != "0":
            reasons.append("arch_gst_nvinfer")
        if arch.get("opencv") != "0" or arch.get("ffmpeg") != "0":
            reasons.append("arch_extra_capture_backend")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line in text.splitlines():
        if not line.startswith(STAT_PREFIX):
            continue
        row = parse_kv(line)
        camera = row.get("camera", "")
        if camera:
            grouped[camera].append(row)

    def number(row: dict[str, str], name: str, fallback: float = -1.0) -> float:
        try:
            return float(row.get(name, fallback))
        except (TypeError, ValueError):
            return fallback

    details: list[str] = []
    latest_by_camera: dict[str, dict[str, str]] = {}
    infer_hz_by_camera: dict[str, float] = {}
    for cid in required:
        rows = grouped.get(cid, [])
        if len(rows) < args.min_runtime_stats:
            reasons.append(f"{cid}:stats={len(rows)}<{args.min_runtime_stats}")
            continue
        tail = rows[-min(3, len(rows)):]
        latest = rows[-1]
        latest_by_camera[cid] = latest
        source_min = min(number(row, "source_fps") for row in tail)
        render_min = min(number(row, "render_fps") for row in tail)
        infer_min = min(number(row, "infer_hz") for row in tail)
        infer_latest = number(latest, "infer_hz")
        infer_hz_by_camera[cid] = infer_latest
        queue_max = max(int(number(row, "queue", 999)) for row in tail)
        infer_count = int(number(latest, "infer_count", -1))
        detector_drops = int(number(latest, "detector_drops", -1))
        positive = int(number(latest, "positive_inferences", -1))
        detections = int(number(latest, "detections_total", -1))
        metadata = int(number(latest, "metadata_added", -1))
        stale = int(number(latest, "stale_expirations", -1))
        result_clears = int(number(latest, "result_clears", -1))
        worker_alive = int(number(latest, "worker_alive", 0))
        thread_alive = int(number(latest, "detector_thread_alive", 0))
        copy_errors = int(number(latest, "copy_errors", 999))
        infer_errors = int(number(latest, "infer_errors", 999))
        meta_errors = int(number(latest, "meta_errors", 999))
        warnings = int(number(latest, "warnings", 999))
        pipeline_errors = int(number(latest, "pipeline_errors", 999))
        infer_p95 = number(latest, "infer_p95_ms")

        if source_min < args.min_source_fps:
            reasons.append(f"{cid}:source_fps_min={source_min:.2f}<{args.min_source_fps:.2f}")
        if render_min < args.min_render_fps:
            reasons.append(f"{cid}:render_fps_min={render_min:.2f}<{args.min_render_fps:.2f}")
        if infer_min < args.min_infer_hz:
            reasons.append(f"{cid}:infer_hz_min={infer_min:.2f}<{args.min_infer_hz:.2f}")
        if queue_max > args.max_queue:
            reasons.append(f"{cid}:queue={queue_max}>{args.max_queue}")
        if infer_count <= 0:
            reasons.append(f"{cid}:infer_count={infer_count}")
        if worker_alive != 1:
            reasons.append(f"{cid}:worker_alive={worker_alive}")
        if thread_alive != 1:
            reasons.append(f"{cid}:detector_thread_alive={thread_alive}")
        if copy_errors != 0:
            reasons.append(f"{cid}:copy_errors={copy_errors}")
        if infer_errors != 0:
            reasons.append(f"{cid}:infer_errors={infer_errors}")
        if meta_errors != 0:
            reasons.append(f"{cid}:meta_errors={meta_errors}")
        if warnings != 0:
            reasons.append(f"{cid}:warnings={warnings}")
        if pipeline_errors != 0:
            reasons.append(f"{cid}:pipeline_errors={pipeline_errors}")

        details.append(
            f"{cid}:source_min={source_min:.2f},render_min={render_min:.2f},"
            f"infer={infer_latest:.2f},queue={queue_max},infer_count={infer_count},"
            f"drops={detector_drops},positive={positive},detections={detections},"
            f"metadata={metadata},stale={stale},infer_p95={infer_p95:.1f}ms"
        )

    if len(infer_hz_by_camera) == len(required) and len(required) > 1:
        skew = max(infer_hz_by_camera.values()) - min(infer_hz_by_camera.values())
        if skew > args.max_infer_skew_hz:
            reasons.append(f"infer_skew={skew:.2f}>{args.max_infer_skew_hz:.2f}")

    for cid in args.require_person_camera:
        latest = latest_by_camera.get(cid)
        if latest is None:
            reasons.append(f"{cid}:person_check_no_stats")
            continue
        positive = int(number(latest, "positive_inferences", -1))
        detections = int(number(latest, "detections_total", -1))
        metadata = int(number(latest, "metadata_added", -1))
        if positive <= 0 or detections <= 0 or metadata <= 0:
            reasons.append(
                f"{cid}:person_not_detected positive={positive} detections={detections} metadata={metadata}"
            )

    for cid in args.require_stale_expiry_camera:
        latest = latest_by_camera.get(cid)
        if latest is None:
            reasons.append(f"{cid}:stale_check_no_stats")
            continue
        stale = int(number(latest, "stale_expirations", -1))
        clears = int(number(latest, "result_clears", -1))
        if stale <= 0 and clears <= 0:
            reasons.append(f"{cid}:stale_expiry_missing")

    suffix = " | ".join(details)
    if reasons:
        print(f"V11_DS_YOLO_MULTI_CHECK RESULT=FAIL reasons={';'.join(reasons)} details={suffix}")
        return 1
    print(f"V11_DS_YOLO_MULTI_CHECK RESULT=PASS required={','.join(required)} details={suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
