#!/usr/bin/env python3
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v2_bbox_v105_log.py /tmp/CAMERA_BBOX_V105.log")
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V105_ARCH" not in text:
        print("V105_MUX_ARRIVAL FAIL missing=CAMERA_V105_ARCH")
        return 2

    source_pat = re.compile(
        r"CAMERA_V105_SOURCE camera=(CAM-\d+) input_count=(\d+) input_vs_max=([0-9.]+)% "
        r"arrival_p50=([0-9.]+)ms arrival_p95=([0-9.]+)ms arrival_p99=([0-9.]+)ms "
        r"mux_latest_lag_p50=([0-9.]+)ms mux_latest_lag_p95=([0-9.]+)ms "
        r"samples=arrival:(\d+),mux:(\d+)"
    )
    latest: dict[str, tuple[float, float, float, float, int]] = {}
    for m in source_pat.finditer(text):
        latest[m.group(1)] = (
            float(m.group(3)),
            float(m.group(5)),
            float(m.group(6)),
            float(m.group(8)),
            int(m.group(9)),
        )
    if len(latest) < 6:
        print(f"V105_MUX_ARRIVAL FAIL sources={len(latest)}/6")
        return 2

    mux_pat = re.compile(
        r"CAMERA_V104_MUX samples=(\d+) target=(\d+) input_passes=(\d+) .*?full_pct=([0-9.]+)"
    )
    mux_matches = list(mux_pat.finditer(text))
    raw_full = float(mux_matches[-1].group(4)) if mux_matches else 0.0

    ratios = [v[0] for v in latest.values()]
    arrival95 = [v[1] for v in latest.values()]
    arrival99 = [v[2] for v in latest.values()]
    stale95 = [v[3] for v in latest.values()]
    samples = [v[4] for v in latest.values()]

    if min(samples) < 200:
        diagnosis = "INSUFFICIENT_SAMPLES"
        next_step = "run V10.5 for at least 45-60 seconds"
    elif min(ratios) < 85.0 or max(arrival95) > 120.0:
        diagnosis = "SOURCE_ARRIVAL_JITTER"
        next_step = "fix the slow/jittery tracker input source path only; do not tune mux or NvDCF yet"
    elif statistics.median(stale95) > 100.0 or max(stale95) > 160.0:
        diagnosis = "MUX_QUEUES_OLD_FRAMES"
        next_step = "fix nvstreammux input freshness/backlog only; newest same-source frames already exist before stale rows are emitted"
    elif raw_full < 75.0:
        diagnosis = "MUX_FORMATION_PARTIAL_WITH_FRESH_INPUTS"
        next_step = "inputs are fresh but six-source batches still miss; tune mux formation policy only"
    else:
        diagnosis = "ARRIVAL_PATH_HEALTHY"
        next_step = "tracker mux input path is healthy; move to selected-batch policy or downstream NvDCF latency"

    print(
        "V105_MUX_ARRIVAL RESULT "
        f"diagnosis={diagnosis} raw_full_pct={raw_full:.1f} "
        f"input_vs_max_min={min(ratios):.1f}% "
        f"arrival_p95_median={statistics.median(arrival95):.0f}ms arrival_p95_max={max(arrival95):.0f}ms "
        f"arrival_p99_max={max(arrival99):.0f}ms "
        f"mux_latest_lag_p95_median={statistics.median(stale95):.0f}ms "
        f"mux_latest_lag_p95_max={max(stale95):.0f}ms"
    )
    for cid in sorted(latest):
        ratio, a95, a99, lag95, n = latest[cid]
        print(
            f"V105_MUX_ARRIVAL source={cid} input_vs_max={ratio:.1f}% "
            f"arrival_p95={a95:.0f}ms arrival_p99={a99:.0f}ms "
            f"mux_latest_lag_p95={lag95:.0f}ms samples={n}"
        )
    print(f"V105_MUX_ARRIVAL next={next_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
