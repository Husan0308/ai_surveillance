#!/usr/bin/env python3
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

TRACKER_BASELINE = {
    "CAM-01": 300.0,
    "CAM-02": 400.0,
    "CAM-03": 300.0,
    "CAM-04": 200.0,
    "CAM-05": 200.0,
    "CAM-06": 250.0,
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
        print("usage: check_camera_v2_bbox_v98_log.py /tmp/CAMERA_BBOX_V98.log")
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V98_ARCH" not in text:
        print("V98_TRACKER FAIL missing=CAMERA_V98_ARCH")
        return 2
    rows = parse_last(text)
    missing = [cid for cid in TRACKER_BASELINE if cid not in rows]
    if missing:
        print("V98_TRACKER FAIL missing_cameras=" + ",".join(missing))
        return 2

    tracker_values = []
    display_values = []
    reductions = []
    print("V98_TRACKER per_camera:")
    for cid in TRACKER_BASELINE:
        display, tracker = rows[cid]
        baseline = TRACKER_BASELINE[cid]
        reduction = (baseline - tracker) / baseline * 100.0 if baseline > 0 else 0.0
        tracker_values.append(tracker)
        display_values.append(display)
        reductions.append(reduction)
        print(
            f"  {cid} source_tracker_p95={tracker:.0f}ms baseline_v97={baseline:.0f}ms "
            f"reduction={reduction:.0f}% source_display_p95={display:.0f}ms"
        )

    median_tracker = statistics.median(tracker_values)
    max_tracker = max(tracker_values)
    median_display = statistics.median(display_values)
    avg_reduction = statistics.mean(reductions)
    passed = (
        median_tracker <= 225.0
        and max_tracker <= 325.0
        and avg_reduction >= 15.0
        and median_display <= 300.0
    )
    status = "PASS" if passed else "FAIL"
    print(
        f"V98_TRACKER {status} median_source_tracker_p95={median_tracker:.0f}ms "
        f"max_source_tracker_p95={max_tracker:.0f}ms avg_reduction={avg_reduction:.0f}% "
        f"median_source_display_p95={median_display:.0f}ms"
    )
    if passed:
        print("V98_TRACKER next=tracker pool materially reduced stale-frame depth; proceed to PTS-aligned overlay")
    else:
        print("V98_TRACKER next=pool depth alone is insufficient; localize pre-mux vs mux/NvDCF latency before another tracker change")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
