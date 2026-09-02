#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

STATS_RE = re.compile(
    r"CAMERA_V11_DS_YOLO_MULTI camera=(?P<camera>CAM-\d{2}) .*?"
    r"source_fps=(?P<source>[0-9.]+) render_fps=(?P<render>[0-9.]+) "
    r"infer_hz=(?P<infer>[0-9.]+) queue=(?P<queue>\d+) .*?"
    r"infer_errors=(?P<infer_errors>\d+) meta_errors=(?P<meta_errors>\d+) .*?"
    r"pipeline_errors=(?P<pipeline_errors>\d+)"
)
CORRECTION_RE = re.compile(
    r"CAMERA_V11_BBOX_NVDCF_CORRECTION camera=(?P<camera>CAM-\d{2}) "
    r"sequence=(?P<seq>\d+) .*?corrections=(?P<corr>\d+)"
)
TRACK_RE = re.compile(
    r"CAMERA_V11_BBOX_NVDCF_TRACK camera=(?P<camera>CAM-\d{2}) "
    r"visible=(?P<visible>\d+) tracker_frames=(?P<frames>\d+) visible_max=(?P<max>\d+)"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/tmp/CAMERA_V11_BBOX_NVDCF_CAM01_CAM02_CAM03.log")
    ap.add_argument("--required-cameras", default="CAM-01,CAM-02,CAM-03")
    ap.add_argument("--min-stats", type=int, default=5)
    ap.add_argument("--require-visible", action="store_true")
    args = ap.parse_args()

    required = [c.strip() for c in args.required_cameras.split(",") if c.strip()]
    path = Path(args.log)
    if not path.is_file():
        print(f"V11_BBOX_NVDCF_CAM01_CAM02_CAM03_CHECK RESULT=FAIL reasons=log_missing:{path}")
        return 1
    text = path.read_text(encoding="utf-8", errors="replace")
    reasons: list[str] = []

    if "CAMERA_V11_BBOX_NVDCF_ARCH cameras=CAM-01,CAM-02,CAM-03" not in text:
        reasons.append("arch_marker_missing")
    if "tracker=nvdcf-local-only" not in text:
        reasons.append("nvdcf_marker_missing")
    if "detector_metadata=once-per-sequence" not in text:
        reasons.append("once_per_sequence_marker_missing")
    if "reid=0" not in text:
        reasons.append("reid_not_disabled")

    stats_by_cam: dict[str, list[dict[str, str]]] = defaultdict(list)
    for match in STATS_RE.finditer(text):
        stats_by_cam[match.group("camera")].append(match.groupdict())

    corr_by_cam: dict[str, list[re.Match[str]]] = defaultdict(list)
    for match in CORRECTION_RE.finditer(text):
        corr_by_cam[match.group("camera")].append(match)

    track_by_cam: dict[str, list[re.Match[str]]] = defaultdict(list)
    for match in TRACK_RE.finditer(text):
        track_by_cam[match.group("camera")].append(match)

    details: list[str] = []
    for camera in required:
        rows = stats_by_cam.get(camera, [])
        if len(rows) < args.min_stats:
            reasons.append(f"{camera}:stats={len(rows)}<{args.min_stats}")
            continue
        recent = rows[-args.min_stats:]
        source_min = min(float(r["source"]) for r in recent)
        render_min = min(float(r["render"]) for r in recent)
        infer_min = min(float(r["infer"]) for r in recent)
        queue_max = max(int(r["queue"]) for r in recent)
        if source_min < 15.0:
            reasons.append(f"{camera}:source_fps_min={source_min:.2f}<15")
        if render_min < 15.0:
            reasons.append(f"{camera}:render_fps_min={render_min:.2f}<15")
        if infer_min < 1.2:
            reasons.append(f"{camera}:infer_hz_min={infer_min:.2f}<1.2")
        if queue_max > 1:
            reasons.append(f"{camera}:queue_max={queue_max}>1")
        for key in ("infer_errors", "meta_errors", "pipeline_errors"):
            if max(int(r[key]) for r in recent) != 0:
                reasons.append(f"{camera}:{key}>0")

        corrections = corr_by_cam.get(camera, [])
        if not corrections:
            reasons.append(f"{camera}:detector_corrections=0")
            corr_count = 0
            unique_sequences = 0
        else:
            sequences = [int(m.group("seq")) for m in corrections]
            if len(sequences) != len(set(sequences)):
                reasons.append(f"{camera}:repeated_detector_sequence_injection")
            corr_count = int(corrections[-1].group("corr"))
            unique_sequences = len(set(sequences))

        tracks = track_by_cam.get(camera, [])
        tracker_frames = max((int(m.group("frames")) for m in tracks), default=0)
        visible_max = max((int(m.group("max")) for m in tracks), default=0)
        if tracker_frames <= 0:
            reasons.append(f"{camera}:tracker_frames=0")
        if args.require_visible and visible_max <= 0:
            reasons.append(f"{camera}:visible_max=0")

        details.append(
            f"{camera}:stats={len(rows)},source_min={source_min:.2f},render_min={render_min:.2f},"
            f"infer_min={infer_min:.2f},queue_max={queue_max},corrections={corr_count},"
            f"unique_sequences={unique_sequences},tracker_frames={tracker_frames},visible_max={visible_max}"
        )

    if "CAMERA_V11_BBOX_NVDCF_META" in text:
        reasons.append("nvdcf_meta_error")
    if "CAMERA_V11_BBOX_NVDCF_TRACKER" in text:
        reasons.append("nvdcf_tracker_error")
    if "CAMERA_V11_DS_YOLO_MULTI_ERROR" in text:
        reasons.append("pipeline_error_marker")

    if reasons:
        print(
            "V11_BBOX_NVDCF_CAM01_CAM02_CAM03_CHECK RESULT=FAIL reasons="
            + ";".join(reasons)
            + (" details=" + " | ".join(details) if details else "")
        )
        return 1

    print(
        "V11_BBOX_NVDCF_CAM01_CAM02_CAM03_CHECK RESULT=PASS required="
        + ",".join(required)
        + " details="
        + " | ".join(details)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
