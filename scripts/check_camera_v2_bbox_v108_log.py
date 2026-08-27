#!/usr/bin/env python3
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

BASELINE = {
    "ratio": 86.6,
    "arrival_p95": 188.0,
    "arrival_p99": 299.0,
    "mux_lag": 250.0,
    "tracker_dt": 350.0,
    "display_tracker": 449.0,
    "source_tracker": 400.0,
}

SOURCE_RE = re.compile(
    r"CAMERA_V105_SOURCE camera=(CAM-\d+) input_count=(\d+) input_vs_max=([0-9.]+)% "
    r"arrival_p50=([0-9.]+)ms arrival_p95=([0-9.]+)ms arrival_p99=([0-9.]+)ms "
    r"mux_latest_lag_p50=([0-9.]+)ms mux_latest_lag_p95=([0-9.]+)ms "
    r"samples=arrival:(\d+),mux:(\d+)"
)
PTS_RE = re.compile(
    r"CAMERA_V95_PTS camera=(CAM-\d+) tracker_pts_hz=([0-9.]+) "
    r"tracker_dt_p50=([0-9.]+)ms tracker_dt_p95=([0-9.]+)ms "
    r"display_minus_tracker_p50=(-?[0-9.]+)ms display_minus_tracker_p95=(-?[0-9.]+)ms "
    r"source_minus_display_p95=([0-9.]+)ms source_minus_tracker_p95=([0-9.]+)ms"
)


def latest_by_camera(pattern: re.Pattern[str], text: str) -> dict[str, re.Match[str]]:
    out: dict[str, re.Match[str]] = {}
    for m in pattern.finditer(text):
        out[m.group(1)] = m
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v2_bbox_v108_log.py /tmp/CAMERA_BBOX_V108.log")
        return 2

    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V108_ARCH" not in text:
        print("V108_CAM02_UDP FAIL missing=CAMERA_V108_ARCH")
        return 2
    if not re.search(r"CAMERA_V108_SOURCE camera=CAM-02 select_rtp_protocol=0\b", text):
        print("V108_CAM02_UDP FAIL nvurisrcbin_udp_enable_not_confirmed=1")
        return 2
    if not re.search(r"CAMERA_V108_RTSP camera=CAM-02 protocols=1\b", text):
        print("V108_CAM02_UDP FAIL rtspsrc_udp_not_confirmed=1")
        return 2

    sources = latest_by_camera(SOURCE_RE, text)
    pts = latest_by_camera(PTS_RE, text)
    if "CAM-02" not in sources or "CAM-02" not in pts:
        print("V108_CAM02_UDP FAIL missing=CAM-02-metrics")
        return 2

    s = sources["CAM-02"]
    p = pts["CAM-02"]
    ratio = float(s.group(3))
    arrival_p95 = float(s.group(5))
    arrival_p99 = float(s.group(6))
    mux_lag = float(s.group(8))
    samples = int(s.group(9))
    tracker_hz = float(p.group(2))
    tracker_dt = float(p.group(4))
    display_tracker = float(p.group(6))
    source_tracker = float(p.group(8))

    peer_p95 = [
        float(m.group(5))
        for cid, m in sources.items()
        if cid != "CAM-02"
    ]
    peer_median = statistics.median(peer_p95) if peer_p95 else float("nan")

    if samples < 300:
        diagnosis = "INSUFFICIENT_SAMPLES"
        next_step = "run 60-90 seconds"
    else:
        improved = sum(
            (
                ratio >= 91.0,
                arrival_p95 <= 160.0,
                arrival_p99 <= 260.0,
                mux_lag <= 180.0,
                tracker_dt <= 275.0,
                display_tracker <= 375.0,
                source_tracker <= 350.0,
            )
        )
        regressed = (
            ratio < 82.0
            or arrival_p95 > 220.0
            or arrival_p99 > 360.0
            or tracker_hz < 8.5
            or source_tracker > 475.0
            or "CAMERA_CLEAN_GST ERROR" in text
        )
        if improved >= 4 and not regressed:
            diagnosis = "PASS"
            next_step = "keep CAM-02 UDP and soak longer before closing source jitter"
        elif regressed or improved <= 1:
            diagnosis = "FAIL_NO_BENEFIT"
            next_step = "revert CAM-02 to TCP; inspect network/NVR channel behavior before more mux or NvDCF tuning"
        else:
            diagnosis = "PARTIAL"
            next_step = "do not close yet; compare UDP loss/jitter and source cadence before choosing transport"

    print(
        "V108_CAM02_UDP RESULT "
        f"diagnosis={diagnosis} input_vs_max={ratio:.1f}% "
        f"arrival_p95={arrival_p95:.0f}ms arrival_p99={arrival_p99:.0f}ms "
        f"mux_latest_lag_p95={mux_lag:.0f}ms peer_arrival_p95_median={peer_median:.0f}ms "
        f"tracker_hz={tracker_hz:.2f} tracker_dt_p95={tracker_dt:.0f}ms "
        f"display_minus_tracker_p95={display_tracker:.0f}ms source_minus_tracker_p95={source_tracker:.0f}ms "
        f"samples={samples}"
    )
    print(
        "V108_CAM02_UDP baseline="
        f"ratio:{BASELINE['ratio']:.1f}%,arrival_p95:{BASELINE['arrival_p95']:.0f},"
        f"arrival_p99:{BASELINE['arrival_p99']:.0f},mux_lag:{BASELINE['mux_lag']:.0f},"
        f"tracker_dt:{BASELINE['tracker_dt']:.0f},display_tracker:{BASELINE['display_tracker']:.0f},"
        f"source_tracker:{BASELINE['source_tracker']:.0f}"
    )
    print(f"V108_CAM02_UDP next={next_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
