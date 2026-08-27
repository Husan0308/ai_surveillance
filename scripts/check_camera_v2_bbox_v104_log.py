#!/usr/bin/env python3
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

CAMERAS = [f"CAM-{i:02d}" for i in range(1, 7)]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v2_bbox_v104_log.py /tmp/CAMERA_BBOX_V104.log")
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V104_ARCH" not in text:
        print("V104_POSTMUX FAIL missing=CAMERA_V104_ARCH")
        return 2

    mux_pat = re.compile(
        r"CAMERA_V104_MUX samples=(\d+) target=(\d+) input_passes=(\d+) "
        r"size_p50=([0-9.]+) size_p95=([0-9.]+) "
        r"unique_p50=([0-9.]+) unique_p95=([0-9.]+) "
        r"full_pct=([0-9.]+) raw_dt_p50=([0-9.]+)ms raw_dt_p95=([0-9.]+)ms "
        r"selected=(\d+) dropped=(\d+) source_hit=([^\s]+)"
    )
    selected_pat = re.compile(
        r"CAMERA_V101_BATCH samples=(\d+) target=(\d+) .*?"
        r"size_p50=([0-9.]+) size_p95=([0-9.]+) .*?"
        r"full_pct=([0-9.]+) partial_pct=([0-9.]+)"
    )
    stage_pat = re.compile(
        r"CAMERA_V99_TRACK_STAGE camera=(CAM-\d+) .*?"
        r"gate_mux_p50=([0-9.]+)ms gate_mux_p95=([0-9.]+)ms .*?"
        r"mux_nvdcf_p50=([0-9.]+)ms mux_nvdcf_p95=([0-9.]+)ms"
    )
    pts_pat = re.compile(
        r"CAMERA_V95_PTS camera=(CAM-\d+) .*?"
        r"source_minus_display_p95=([0-9.]+)ms source_minus_tracker_p95=([0-9.]+)ms"
    )

    mux = None
    selected = None
    stage = {}
    pts = {}
    for line in text.splitlines():
        m = mux_pat.search(line)
        if m:
            mux = m.groups()
        s = selected_pat.search(line)
        if s:
            selected = s.groups()
        a = stage_pat.search(line)
        if a:
            stage[a.group(1)] = tuple(float(x) for x in a.groups()[1:])
        p = pts_pat.search(line)
        if p:
            pts[p.group(1)] = (float(p.group(2)), float(p.group(3)))

    if mux is None or selected is None:
        print("V104_POSTMUX FAIL missing=mux-or-selected-stats")
        return 2
    missing = [cid for cid in CAMERAS if cid not in stage or cid not in pts]
    if missing:
        print("V104_POSTMUX FAIL missing_cameras=" + ",".join(missing))
        return 2

    raw_samples = int(mux[0]); target = int(mux[1]); input_passes = int(mux[2])
    raw_size50 = float(mux[3]); raw_size95 = float(mux[4])
    raw_full = float(mux[7]); raw_dt50 = float(mux[8]); raw_dt95 = float(mux[9])
    post_selected = int(mux[10]); post_dropped = int(mux[11])
    source_hit_text = mux[12]

    sel_samples = int(selected[0]); sel_target = int(selected[1])
    sel_size50 = float(selected[2]); sel_size95 = float(selected[3])
    sel_full = float(selected[4]); sel_partial = float(selected[5])

    hit_vals = []
    for part in source_hit_text.split('/'):
        try:
            hit_vals.append(float(part.rsplit(':', 1)[1].rstrip('%')))
        except Exception:
            pass

    gate95 = []
    nv95 = []
    tracker95 = []
    display95 = []
    for cid in CAMERAS:
        _gm50, gm95, _mn50, mn95 = stage[cid]
        disp95, trk95 = pts[cid]
        gate95.append(gm95); nv95.append(mn95)
        tracker95.append(trk95); display95.append(disp95)

    med_gate = statistics.median(gate95)
    max_gate = max(gate95)
    med_nv = statistics.median(nv95)
    max_nv = max(nv95)
    med_trk = statistics.median(tracker95)
    max_trk = max(tracker95)
    med_disp = statistics.median(display95)
    min_hit = min(hit_vals) if hit_vals else 0.0

    print(
        "V104_POSTMUX mux "
        f"raw_samples={raw_samples} target={target} input_passes={input_passes} "
        f"raw_size_p50={raw_size50:.0f} raw_size_p95={raw_size95:.0f} raw_full_pct={raw_full:.1f} "
        f"raw_dt_p50={raw_dt50:.0f}ms raw_dt_p95={raw_dt95:.0f}ms "
        f"selected={post_selected} dropped={post_dropped} source_hit_min={min_hit:.0f}%"
    )
    print(
        "V104_POSTMUX selected "
        f"samples={sel_samples} target={sel_target} size_p50={sel_size50:.0f} size_p95={sel_size95:.0f} "
        f"full_pct={sel_full:.1f} partial_pct={sel_partial:.1f}"
    )
    print(
        "V104_POSTMUX latency "
        f"median_gate_mux_p95={med_gate:.0f}ms max_gate_mux_p95={max_gate:.0f}ms "
        f"median_mux_nvdcf_p95={med_nv:.0f}ms max_mux_nvdcf_p95={max_nv:.0f}ms "
        f"median_source_tracker_p95={med_trk:.0f}ms max_source_tracker_p95={max_trk:.0f}ms "
        f"median_source_display_p95={med_disp:.0f}ms"
    )

    passed = (
        target == 6 and sel_target == 6
        and raw_samples >= 100
        and raw_full >= 70.0
        and sel_full >= 70.0
        and min_hit >= 80.0
        and med_gate <= 90.0
        and max_gate <= 120.0
        and med_trk <= 200.0
        and max_trk <= 300.0
        and max_nv <= 180.0
    )
    if passed:
        print("V104_POSTMUX PASS diagnosis=pre-mux-sparsification-was-main-mux-latency-source next=exact-display-tracker-PTS-alignment")
        return 0

    if raw_full < 70.0:
        diagnosis = "RAW_MUX_STILL_PARTIAL"
        next_step = "continuous inputs are not enough; measure per-source mux arrival jitter and source starvation before changing NvDCF"
    elif sel_full + 10.0 < raw_full:
        diagnosis = "POSTMUX_SELECTION_BIAS"
        next_step = "raw mux is healthy but slot selection catches partial batches; select full/fresh batch within each slot"
    elif med_gate > 90.0 or max_gate > 120.0:
        diagnosis = "MUX_QUEUE_TAIL_REMAINS"
        next_step = "batch fullness improved but mux queuing remains; inspect queue depth/backpressure, not bbox smoothing"
    elif max_nv > 180.0:
        diagnosis = "NVDCF_DOMINANT"
        next_step = "mux is healthy; optimize NvDCF workload/config only"
    else:
        diagnosis = "DISPLAY_TRACKER_ALIGNMENT_REMAINS"
        next_step = "tracker path is materially fresher; move to exact display-frame versus tracker-history PTS selection"
    print(f"V104_POSTMUX FAIL diagnosis={diagnosis} next={next_step}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
