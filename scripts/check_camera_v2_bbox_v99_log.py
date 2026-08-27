#!/usr/bin/env python3
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

CAMERAS = [f"CAM-{i:02d}" for i in range(1, 7)]


def parse(text: str):
    rows = {}
    pat = re.compile(
        r"CAMERA_V99_TRACK_STAGE camera=(CAM-\d+) .*?"
        r"source_gate_p50=([0-9.]+)ms source_gate_p95=([0-9.]+)ms .*?"
        r"gate_mux_p50=([0-9.]+)ms gate_mux_p95=([0-9.]+)ms .*?"
        r"source_mux_p95=([0-9.]+)ms .*?"
        r"mux_nvdcf_p50=([0-9.]+)ms mux_nvdcf_p95=([0-9.]+)ms"
    )
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            rows[m.group(1)] = tuple(float(x) for x in m.groups()[1:])
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v2_bbox_v99_log.py /tmp/CAMERA_BBOX_V99.log")
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V99_ARCH" not in text:
        print("V99_TRACK_STAGE FAIL missing=CAMERA_V99_ARCH")
        return 2
    rows = parse(text)
    missing = [cid for cid in CAMERAS if cid not in rows]
    if missing:
        print("V99_TRACK_STAGE FAIL missing_cameras=" + ",".join(missing))
        return 2

    sg95, gm95, sm95, mn95 = [], [], [], []
    print("V99_TRACK_STAGE per_camera:")
    for cid in CAMERAS:
        sg50, sg, gm50, gm, sm, mn50, mn = rows[cid]
        sg95.append(sg); gm95.append(gm); sm95.append(sm); mn95.append(mn)
        print(
            f"  {cid} source_gate_p95={sg:.0f}ms gate_mux_p95={gm:.0f}ms "
            f"source_mux_p95={sm:.0f}ms mux_nvdcf_p95={mn:.0f}ms"
        )

    med_sg = statistics.median(sg95)
    max_sg = max(sg95)
    med_gm = statistics.median(gm95)
    max_gm = max(gm95)
    med_mn = statistics.median(mn95)
    max_mn = max(mn95)
    med_sm = statistics.median(sm95)

    # Diagnose the first stage that is materially stale.  Thresholds are
    # intentionally loose for a 10 Hz tracker: >150 ms is already >1.5 tracker periods.
    if med_sg > 150.0 or max_sg > 250.0:
        diagnosis = "PRE_MUX_STALE"
        next_step = "fix tracker branch admission/freshness before tracker_mux only"
    elif med_gm > 150.0 or max_gm > 250.0:
        diagnosis = "MUX_BATCH_WAIT"
        next_step = "fix tracker_mux batch formation timeout/freshness only"
    elif med_mn > 120.0 or max_mn > 220.0:
        diagnosis = "NVDCF_STAGE_LATENCY"
        next_step = "profile/fix NvDCF processing stage only; keep mux/display unchanged"
    else:
        diagnosis = "NO_SINGLE_STAGE_BACKLOG"
        next_step = "move to display-tracker exact PTS alignment; stage latencies are bounded"

    print(
        f"V99_TRACK_STAGE RESULT diagnosis={diagnosis} "
        f"median_source_gate_p95={med_sg:.0f}ms max_source_gate_p95={max_sg:.0f}ms "
        f"median_gate_mux_p95={med_gm:.0f}ms max_gate_mux_p95={max_gm:.0f}ms "
        f"median_source_mux_p95={med_sm:.0f}ms "
        f"median_mux_nvdcf_p95={med_mn:.0f}ms max_mux_nvdcf_p95={max_mn:.0f}ms"
    )
    print(f"V99_TRACK_STAGE next={next_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
