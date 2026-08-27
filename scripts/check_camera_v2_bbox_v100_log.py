#!/usr/bin/env python3
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

CAMERAS = [f"CAM-{i:02d}" for i in range(1, 7)]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v2_bbox_v100_log.py /tmp/CAMERA_BBOX_V100.log")
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V100_ARCH" not in text:
        print("V100_TRACKER_TIMEOUT FAIL missing=CAMERA_V100_ARCH")
        return 2

    stage_pat = re.compile(
        r"CAMERA_V99_TRACK_STAGE camera=(CAM-\d+) .*?"
        r"source_gate_p50=([0-9.]+)ms source_gate_p95=([0-9.]+)ms .*?"
        r"gate_mux_p50=([0-9.]+)ms gate_mux_p95=([0-9.]+)ms .*?"
        r"source_mux_p95=([0-9.]+)ms .*?"
        r"mux_nvdcf_p50=([0-9.]+)ms mux_nvdcf_p95=([0-9.]+)ms"
    )
    pts_pat = re.compile(
        r"CAMERA_V95_PTS camera=(CAM-\d+) .*?source_minus_display_p95=([0-9.]+)ms "
        r"source_minus_tracker_p95=([0-9.]+)ms"
    )
    stage = {}
    pts = {}
    for line in text.splitlines():
        m = stage_pat.search(line)
        if m:
            stage[m.group(1)] = tuple(float(x) for x in m.groups()[1:])
        p = pts_pat.search(line)
        if p:
            pts[p.group(1)] = (float(p.group(2)), float(p.group(3)))

    missing = [cid for cid in CAMERAS if cid not in stage or cid not in pts]
    if missing:
        print("V100_TRACKER_TIMEOUT FAIL missing_cameras=" + ",".join(missing))
        return 2

    gm95 = []
    sm95 = []
    mn95 = []
    tracker95 = []
    display95 = []
    print("V100_TRACKER_TIMEOUT per_camera:")
    for cid in CAMERAS:
        sg50, sg95, gm50, gm, sm, mn50, mn = stage[cid]
        disp, trk = pts[cid]
        gm95.append(gm); sm95.append(sm); mn95.append(mn); tracker95.append(trk); display95.append(disp)
        print(
            f"  {cid} gate_mux_p95={gm:.0f}ms source_mux_p95={sm:.0f}ms "
            f"mux_nvdcf_p95={mn:.0f}ms source_tracker_p95={trk:.0f}ms source_display_p95={disp:.0f}ms"
        )

    med_gm = statistics.median(gm95)
    max_gm = max(gm95)
    med_sm = statistics.median(sm95)
    med_mn = statistics.median(mn95)
    max_mn = max(mn95)
    med_trk = statistics.median(tracker95)
    max_trk = max(tracker95)
    med_disp = statistics.median(display95)

    passed = (
        med_gm <= 100.0
        and max_gm <= 160.0
        and med_sm <= 150.0
        and med_trk <= 225.0
        and max_trk <= 350.0
        and max_mn <= 180.0
        and med_disp <= 300.0
    )
    status = "PASS" if passed else "FAIL"
    print(
        f"V100_TRACKER_TIMEOUT {status} median_gate_mux_p95={med_gm:.0f}ms "
        f"max_gate_mux_p95={max_gm:.0f}ms median_source_mux_p95={med_sm:.0f}ms "
        f"median_mux_nvdcf_p95={med_mn:.0f}ms max_mux_nvdcf_p95={max_mn:.0f}ms "
        f"median_source_tracker_p95={med_trk:.0f}ms max_source_tracker_p95={max_trk:.0f}ms "
        f"median_source_display_p95={med_disp:.0f}ms"
    )
    if passed:
        print("V100_TRACKER_TIMEOUT next=tracker mux wait materially reduced; close tracker freshness and move to exact display-tracker PTS alignment")
    else:
        print("V100_TRACKER_TIMEOUT next=40ms timeout insufficient or increases downstream cost; do not tune NvDCF yet, inspect batch-size/partial-batch behavior")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
