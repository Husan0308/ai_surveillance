#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

BASELINE_RATIO = 86.6
BASELINE_P95 = 188.0
BASELINE_P99 = 299.0
BASELINE_MUX_LAG = 250.0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v2_bbox_v106_log.py /tmp/CAMERA_BBOX_V106.log")
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V106_ARCH" not in text:
        print("V106_CAM02_TCP_TS FAIL missing=CAMERA_V106_ARCH")
        return 2
    if not re.search(r"CAMERA_V106_RTSP camera=CAM-02 tcp_timestamp=0\b", text):
        print("V106_CAM02_TCP_TS FAIL tcp_timestamp_off_not_confirmed=1")
        return 2

    pat = re.compile(
        r"CAMERA_V105_SOURCE camera=CAM-02 input_count=(\d+) input_vs_max=([0-9.]+)% "
        r"arrival_p50=([0-9.]+)ms arrival_p95=([0-9.]+)ms arrival_p99=([0-9.]+)ms "
        r"mux_latest_lag_p50=([0-9.]+)ms mux_latest_lag_p95=([0-9.]+)ms "
        r"samples=arrival:(\d+),mux:(\d+)"
    )
    matches = list(pat.finditer(text))
    if not matches:
        print("V106_CAM02_TCP_TS FAIL missing=CAMERA_V105_SOURCE_CAM02")
        return 2
    m = matches[-1]
    ratio = float(m.group(2))
    p95 = float(m.group(4))
    p99 = float(m.group(5))
    lag95 = float(m.group(7))
    samples = int(m.group(8))

    if samples < 300:
        diagnosis = "INSUFFICIENT_SAMPLES"
        next_step = "run 60-90 seconds"
    else:
        improved = sum(
            (
                ratio >= 91.0,
                p95 <= 165.0,
                p99 <= 265.0,
                lag95 <= 220.0,
            )
        )
        regressed = p95 > BASELINE_P95 + 20.0 or ratio < BASELINE_RATIO - 3.0
        if improved >= 3 and not regressed:
            diagnosis = "PASS"
            next_step = "keep CAM-02 tcp-timestamp off and do a longer soak before closing source jitter"
        elif regressed or improved == 0:
            diagnosis = "FAIL_NO_BENEFIT"
            next_step = "revert this single change; next A/B is bounded CAM-02 RTSP latency, not mux or NvDCF"
        else:
            diagnosis = "PARTIAL"
            next_step = "do not close yet; compare source arrival directly before choosing bounded latency"

    print(
        "V106_CAM02_TCP_TS RESULT "
        f"diagnosis={diagnosis} input_vs_max={ratio:.1f}% "
        f"arrival_p95={p95:.0f}ms arrival_p99={p99:.0f}ms "
        f"mux_latest_lag_p95={lag95:.0f}ms samples={samples} "
        f"baseline=ratio:{BASELINE_RATIO:.1f}%,p95:{BASELINE_P95:.0f},p99:{BASELINE_P99:.0f},lag:{BASELINE_MUX_LAG:.0f}"
    )
    print(f"V106_CAM02_TCP_TS next={next_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
