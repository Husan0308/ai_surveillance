#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


STATS_RE = re.compile(
    r"CAMERA_V11_DS_YOLO_MULTI camera=CAM-01 .*?source_fps=(?P<source>[0-9.]+) "
    r"render_fps=(?P<render>[0-9.]+) infer_hz=(?P<infer>[0-9.]+) queue=(?P<queue>\d+) .*?"
    r"infer_errors=(?P<infer_errors>\d+) meta_errors=(?P<meta_errors>\d+) .*?pipeline_errors=(?P<pipeline_errors>\d+)"
)
CORRECTION_RE = re.compile(
    r"CAMERA_V11_BBOX_NVDCF_CORRECTION camera=CAM-01 sequence=(?P<seq>\d+) .*?corrections=(?P<corr>\d+)"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/tmp/CAMERA_V11_BBOX_NVDCF_CAM01.log")
    ap.add_argument("--min-stats", type=int, default=5)
    args = ap.parse_args()

    path = Path(args.log)
    if not path.is_file():
        print(f"V11_BBOX_NVDCF_CAM01_CHECK RESULT=FAIL reasons=log_missing:{path}")
        return 1
    text = path.read_text(encoding="utf-8", errors="replace")
    reasons: list[str] = []

    if "CAMERA_V11_BBOX_NVDCF_ARCH cameras=CAM-01" not in text:
        reasons.append("arch_marker_missing")
    if "tracker=nvdcf-local-only" not in text:
        reasons.append("nvdcf_marker_missing")
    if "detector_metadata=once-per-sequence" not in text:
        reasons.append("once_per_sequence_marker_missing")
    if "reid=0" not in text:
        reasons.append("reid_not_disabled")

    stats = [m.groupdict() for m in STATS_RE.finditer(text)]
    if len(stats) < args.min_stats:
        reasons.append(f"stats={len(stats)}<{args.min_stats}")
    else:
        recent = stats[-args.min_stats :]
        source_min = min(float(row["source"]) for row in recent)
        render_min = min(float(row["render"]) for row in recent)
        infer_min = min(float(row["infer"]) for row in recent)
        queue_max = max(int(row["queue"]) for row in recent)
        if source_min < 15.0:
            reasons.append(f"source_fps_min={source_min:.2f}<15")
        if render_min < 15.0:
            reasons.append(f"render_fps_min={render_min:.2f}<15")
        if infer_min < 1.2:
            reasons.append(f"infer_hz_min={infer_min:.2f}<1.2")
        if queue_max > 1:
            reasons.append(f"queue_max={queue_max}>1")
        for key in ("infer_errors", "meta_errors", "pipeline_errors"):
            if max(int(row[key]) for row in recent) != 0:
                reasons.append(f"{key}>0")

    corrections = list(CORRECTION_RE.finditer(text))
    if not corrections:
        reasons.append("detector_corrections=0")
    sequences = [int(m.group("seq")) for m in corrections]
    if len(sequences) != len(set(sequences)):
        reasons.append("repeated_detector_sequence_injection")

    if "CAMERA_V11_BBOX_NVDCF_META" in text:
        reasons.append("nvdcf_meta_error")
    if "CAMERA_V11_DS_YOLO_MULTI_ERROR" in text:
        reasons.append("pipeline_error_marker")

    if reasons:
        print("V11_BBOX_NVDCF_CAM01_CHECK RESULT=FAIL reasons=" + ";".join(reasons))
        return 1

    latest_corr = int(corrections[-1].group("corr")) if corrections else 0
    print(
        "V11_BBOX_NVDCF_CAM01_CHECK RESULT=PASS "
        f"stats={len(stats)} corrections={latest_corr} unique_sequences={len(set(sequences))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
