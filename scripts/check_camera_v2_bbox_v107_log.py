#!/usr/bin/env python3
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

BASELINE_RATIO = 86.6
BASELINE_P95 = 188.0
BASELINE_P99 = 299.0
BASELINE_MUX_LAG = 250.0
BASELINE_TRACKER_DT_P95 = 350.0
BASELINE_DISPLAY_TRACKER_P95 = 449.0
BASELINE_SOURCE_TRACKER_P95 = 400.0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v2_bbox_v107_log.py /tmp/CAMERA_BBOX_V107.log")
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V107_ARCH" not in text:
        print("V107_CAM02_LATENCY FAIL missing=CAMERA_V107_ARCH")
        return 2
    if not re.search(
        r"CAMERA_V107_SOURCE camera=CAM-02 source_latency_ms=120\b.*property_set=1\b",
        text,
    ):
        print("V107_CAM02_LATENCY FAIL latency120_not_confirmed=1")
        return 2

    source_pat = re.compile(
        r"CAMERA_V105_SOURCE camera=(CAM-\d+) input_count=(\d+) input_vs_max=([0-9.]+)% "
        r"arrival_p50=([0-9.]+)ms arrival_p95=([0-9.]+)ms arrival_p99=([0-9.]+)ms "
        r"mux_latest_lag_p50=([0-9.]+)ms mux_latest_lag_p95=([0-9.]+)ms "
        r"samples=arrival:(\d+),mux:(\d+)"
    )
    latest_source: dict[str, tuple[float, float, float, float, int]] = {}
    for m in source_pat.finditer(text):
        latest_source[m.group(1)] = (
            float(m.group(3)),
            float(m.group(5)),
            float(m.group(6)),
            float(m.group(8)),
            int(m.group(9)),
        )
    if "CAM-02" not in latest_source:
        print("V107_CAM02_LATENCY FAIL missing=CAMERA_V105_SOURCE_CAM02")
        return 2

    pts_pat = re.compile(
        r"CAMERA_V95_PTS camera=CAM-02 tracker_pts_hz=([0-9.]+) "
        r"tracker_dt_p50=([0-9.]+)ms tracker_dt_p95=([0-9.]+)ms "
        r"display_minus_tracker_p50=([0-9.]+)ms display_minus_tracker_p95=([0-9.]+)ms "
        r"source_minus_display_p95=([0-9.]+)ms source_minus_tracker_p95=([0-9.]+)ms"
    )
    pts_matches = list(pts_pat.finditer(text))
    if not pts_matches:
        print("V107_CAM02_LATENCY FAIL missing=CAMERA_V95_PTS_CAM02")
        return 2

    ratio, p95, p99, lag95, samples = latest_source["CAM-02"]
    p = pts_matches[-1]
    tracker_hz = float(p.group(1))
    tracker_dt95 = float(p.group(3))
    display_tracker95 = float(p.group(5))
    source_tracker95 = float(p.group(7))

    peers = [
        row[1]
        for cid, row in latest_source.items()
        if cid != "CAM-02"
    ]
    peer_p95_median = statistics.median(peers) if peers else 0.0

    if samples < 300:
        diagnosis = "INSUFFICIENT_SAMPLES"
        next_step = "run V10.7 for at least 60-90 seconds"
    else:
        wins = sum(
            (
                ratio >= 90.0,
                p95 <= 170.0,
                lag95 <= 200.0,
                tracker_dt95 <= 300.0,
            )
        )
        latency_regression = display_tracker95 > 500.0 or source_tracker95 > 500.0
        source_regression = p95 > BASELINE_P95 + 20.0 or ratio < BASELINE_RATIO - 3.0
        near_peers = peer_p95_median > 0.0 and p95 <= peer_p95_median + 25.0

        if wins >= 3 and not latency_regression and not source_regression:
            diagnosis = "PASS" if near_peers else "PARTIAL"
            next_step = (
                "keep CAM-02 latency=120 and run a longer soak"
                if diagnosis == "PASS"
                else "120ms helps but CAM-02 is still an outlier; do not tune NvDCF yet"
            )
        elif latency_regression or source_regression:
            diagnosis = "FAIL_REGRESSION"
            next_step = "revert CAM-02 to 60ms; next source-only A/B should be transport/buffer policy, not mux/NvDCF"
        else:
            diagnosis = "FAIL_NO_BENEFIT"
            next_step = "120ms did not materially fix CAM-02; revert it before the next source-only A/B"

    print(
        "V107_CAM02_LATENCY RESULT "
        f"diagnosis={diagnosis} input_vs_max={ratio:.1f}% arrival_p95={p95:.0f}ms "
        f"arrival_p99={p99:.0f}ms mux_latest_lag_p95={lag95:.0f}ms "
        f"peer_arrival_p95_median={peer_p95_median:.0f}ms tracker_hz={tracker_hz:.2f} "
        f"tracker_dt_p95={tracker_dt95:.0f}ms display_minus_tracker_p95={display_tracker95:.0f}ms "
        f"source_minus_tracker_p95={source_tracker95:.0f}ms samples={samples}"
    )
    print(
        "V107_CAM02_LATENCY baseline="
        f"ratio:{BASELINE_RATIO:.1f}%,arrival_p95:{BASELINE_P95:.0f},arrival_p99:{BASELINE_P99:.0f},"
        f"mux_lag:{BASELINE_MUX_LAG:.0f},tracker_dt:{BASELINE_TRACKER_DT_P95:.0f},"
        f"display_tracker:{BASELINE_DISPLAY_TRACKER_P95:.0f},source_tracker:{BASELINE_SOURCE_TRACKER_P95:.0f}"
    )
    print(f"V107_CAM02_LATENCY next={next_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
