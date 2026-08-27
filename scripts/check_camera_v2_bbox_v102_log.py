#!/usr/bin/env python3
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

CAMERAS = [f"CAM-{i:02d}" for i in range(1, 7)]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v2_bbox_v102_log.py /tmp/CAMERA_BBOX_V102.log")
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V102_ARCH" not in text:
        print("V102_BATCH4 FAIL missing=CAMERA_V102_ARCH")
        return 2

    batch_pat = re.compile(
        r"CAMERA_V102_BATCH samples=(\d+) target=(\d+) .*?"
        r"size_p50=([0-9.]+) size_p95=([0-9.]+) .*?"
        r"unique_p50=([0-9.]+) unique_p95=([0-9.]+) .*?"
        r"full_pct=([0-9.]+) partial_pct=([0-9.]+) .*?"
        r"pts_spread_p50=([0-9.]+)ms pts_spread_p95=([0-9.]+)ms .*?"
        r"output_dt_p50=([0-9.]+)ms output_dt_p95=([0-9.]+)ms .*?source_hit=(.*)$"
    )
    stage_pat = re.compile(
        r"CAMERA_V99_TRACK_STAGE camera=(CAM-\d+) .*?"
        r"gate_mux_p50=([0-9.]+)ms gate_mux_p95=([0-9.]+)ms .*?"
        r"source_mux_p95=([0-9.]+)ms .*?"
        r"mux_nvdcf_p50=([0-9.]+)ms mux_nvdcf_p95=([0-9.]+)ms"
    )
    pts_pat = re.compile(
        r"CAMERA_V95_PTS camera=(CAM-\d+) .*?source_minus_display_p95=([0-9.]+)ms "
        r"source_minus_tracker_p95=([0-9.]+)ms"
    )

    batch = None
    stage = {}
    pts = {}
    for line in text.splitlines():
        b = batch_pat.search(line)
        if b:
            batch = b.groups()
        s = stage_pat.search(line)
        if s:
            stage[s.group(1)] = tuple(float(x) for x in s.groups()[1:])
        p = pts_pat.search(line)
        if p:
            pts[p.group(1)] = (float(p.group(2)), float(p.group(3)))

    if batch is None:
        print("V102_BATCH4 FAIL missing=CAMERA_V102_BATCH")
        return 2
    missing = [cid for cid in CAMERAS if cid not in stage or cid not in pts]
    if missing:
        print("V102_BATCH4 FAIL missing_cameras=" + ",".join(missing))
        return 2

    samples = int(batch[0]); target = int(batch[1])
    size50 = float(batch[2]); size95 = float(batch[3])
    uniq50 = float(batch[4]); uniq95 = float(batch[5])
    full_pct = float(batch[6]); partial_pct = float(batch[7])
    spread50 = float(batch[8]); spread95 = float(batch[9])
    out50 = float(batch[10]); out95 = float(batch[11])
    hit_text = batch[12]
    hit_vals = []
    for part in hit_text.split("/"):
        if ":" in part:
            try:
                hit_vals.append(float(part.rsplit(":", 1)[1].rstrip("%")))
            except ValueError:
                pass

    gm95 = []
    mn95 = []
    trk95 = []
    disp95 = []
    for cid in CAMERAS:
        gm50, gm, sm, mn50, mn = stage[cid]
        disp, trk = pts[cid]
        gm95.append(gm); mn95.append(mn); trk95.append(trk); disp95.append(disp)

    med_gm = statistics.median(gm95)
    max_gm = max(gm95)
    med_mn = statistics.median(mn95)
    max_mn = max(mn95)
    med_trk = statistics.median(trk95)
    max_trk = max(trk95)
    med_disp = statistics.median(disp95)
    min_hit = min(hit_vals) if hit_vals else 0.0
    max_hit = max(hit_vals) if hit_vals else 0.0

    print(
        "V102_BATCH4 metrics "
        f"samples={samples} target={target} size_p50={size50:.0f} size_p95={size95:.0f} "
        f"full_pct={full_pct:.1f} partial_pct={partial_pct:.1f} "
        f"pts_spread_p50={spread50:.0f}ms pts_spread_p95={spread95:.0f}ms "
        f"output_dt_p50={out50:.0f}ms output_dt_p95={out95:.0f}ms "
        f"source_hit_min={min_hit:.0f}% source_hit_max={max_hit:.0f}%"
    )
    print(
        "V102_BATCH4 latency "
        f"median_gate_mux_p95={med_gm:.0f}ms max_gate_mux_p95={max_gm:.0f}ms "
        f"median_mux_nvdcf_p95={med_mn:.0f}ms max_mux_nvdcf_p95={max_mn:.0f}ms "
        f"median_source_tracker_p95={med_trk:.0f}ms max_source_tracker_p95={max_trk:.0f}ms "
        f"median_source_display_p95={med_disp:.0f}ms"
    )

    passed = (
        target == 4
        and full_pct >= 70.0
        and min_hit >= 50.0
        and spread95 <= 450.0
        and med_gm <= 110.0
        and max_gm <= 150.0
        and med_trk <= 225.0
        and max_trk <= 350.0
        and max_mn <= 180.0
        and med_disp <= 300.0
    )
    if passed:
        print("V102_BATCH4 PASS diagnosis=batch-target-4-removes-most-mux-tail next=close-tracker-mux-wait-and-move-to-exact-display-tracker-PTS-alignment")
        return 0

    if full_pct < 70.0:
        diagnosis = "STILL_PARTIAL"
        next_step = "batch-size=4 still waits; inspect per-source arrival phase before changing timeout again"
    elif spread95 > 450.0:
        diagnosis = "PTS_SKEW_REMAINS"
        next_step = "batch completion improved but frames are too far apart in PTS; fix timestamp alignment/freshness before NvDCF tuning"
    elif med_gm > 110.0 or max_gm > 150.0:
        diagnosis = "MUX_TAIL_REMAINS"
        next_step = "batch target alone did not remove mux tail; inspect source phase/round-robin behavior"
    elif max_mn > 180.0:
        diagnosis = "NVDCF_COST_INCREASED"
        next_step = "do not reduce batch/timeout further; smaller batches increased NvDCF cost"
    else:
        diagnosis = "TRACKER_END_TO_END_REMAINS"
        next_step = "mux improved but source->tracker remains stale; use exact display-tracker PTS alignment next"
    print(f"V102_BATCH4 FAIL diagnosis={diagnosis} next={next_step}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
