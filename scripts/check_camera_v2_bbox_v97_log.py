#!/usr/bin/env python3
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

BASELINE = {
    "CAM-01": 450.0,
    "CAM-02": 450.0,
    "CAM-03": 500.0,
    "CAM-04": 500.0,
    "CAM-05": 400.0,
    "CAM-06": 200.0,
}


def parse_last(text: str) -> dict[str, tuple[float, float]]:
    found: dict[str, tuple[float, float]] = {}
    pattern = re.compile(
        r"CAMERA_V95_PTS camera=(CAM-\d+) .*?source_minus_display_p95=([0-9.]+)ms .*?source_minus_tracker_p95=([0-9.]+)ms"
    )
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            found[match.group(1)] = (float(match.group(2)), float(match.group(3)))
    return found


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v2_bbox_v97_log.py /tmp/CAMERA_BBOX_V97.log")
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V97_ARCH" not in text:
        print("V97_DISPLAY FAIL missing=CAMERA_V97_ARCH")
        return 2
    rows = parse_last(text)
    missing = [cid for cid in BASELINE if cid not in rows]
    if missing:
        print("V97_DISPLAY FAIL missing_cameras=" + ",".join(missing))
        return 2

    display = []
    reductions = []
    print("V97_DISPLAY per_camera:")
    for cid in BASELINE:
        value, tracker = rows[cid]
        baseline = BASELINE[cid]
        reduction = (baseline - value) / baseline * 100.0 if baseline > 0 else 0.0
        display.append(value)
        reductions.append(reduction)
        print(
            f"  {cid} source_display_p95={value:.0f}ms baseline_v96={baseline:.0f}ms "
            f"reduction={reduction:.0f}% source_tracker_p95={tracker:.0f}ms"
        )

    median_display = statistics.median(display)
    max_display = max(display)
    avg_reduction = statistics.mean(reductions)
    passed = median_display <= 300.0 and max_display <= 400.0 and avg_reduction >= 20.0
    status = "PASS" if passed else "FAIL"
    print(
        f"V97_DISPLAY {status} median_source_display_p95={median_display:.0f}ms "
        f"max_source_display_p95={max_display:.0f}ms avg_reduction={avg_reduction:.0f}%"
    )
    if passed:
        print("V97_DISPLAY next=display pool contributes materially; finish display freshness before tracker changes")
    else:
        print("V97_DISPLAY next=pool depth is not enough; localize pre-mux vs mux latency before another display change")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
