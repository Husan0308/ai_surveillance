#!/usr/bin/env python3
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

CAMERAS = [f"CAM-{i:02d}" for i in range(1, 7)]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v2_bbox_v103_log.py /tmp/CAMERA_BBOX_V103.log")
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V103_ARCH" not in text:
        print("V103_SHARED_PHASE FAIL missing=CAMERA_V103_ARCH")
        return 2

    batch_pat = re.compile(
        r"CAMERA_V101_BATCH samples=(\d+) target=(\d+) "
        r"size_p50=([0-9.]+) size_p95=([0-9.]+) "
        r"unique_p50=([0-9.]+) unique_p95=([0-9.]+) "
        r"full_pct=([0-9.]+) partial_pct=([0-9.]+) "
        r"pts_spread_p50=([0-9.]+)ms pts_spread_p95=([0-9.]+)ms "
        r"output_dt_p50=([0-9.]+)ms output_dt_p95=([0-9.]+)ms "
        r"source_hit=([^\s]+)"
    )
    phase_pat = re.compile(
        r"CAMERA_V103_PHASE accepts=(\d+) drops=(\d+) "
        r"phase_p50=([0-9.]+)ms phase_p95=([0-9.]+)ms"
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
    phase = None
    stages = {}
    pts = {}
    for line in text.splitlines():
        m = batch_pat.search(line)
        if m:
            batch = m.groups()
        m = phase_pat.search(line)
        if m:
            phase = m.groups()
        m = stage_pat.search(line)
        if m:
            stages[m.group(1)] = tuple(float(x) for x in m.groups()[1:])
        m = pts_pat.search(line)
        if m:
            pts[m.group(1)] = (float(m.group(2)), float(m.group(3)))

    if batch is None or phase is None:
        print("V103_SHARED_PHASE FAIL missing=batch_or_phase_stats")
        return 2
    missing = [cid for cid in CAMERAS if cid not in stages or cid not in pts]
    if missing:
        print("V103_SHARED_PHASE FAIL missing_cameras=" + ",".join(missing))
        return 2

    samples = int(batch[0]); target = int(batch[1])
    size50 = float(batch[2]); size95 = float(batch[3])
    full_pct = float(batch[6]); partial_pct = float(batch[7])
    spread50 = float(batch[8]); spread95 = float(batch[9])
    out50 = float(batch[10]); out95 = float(batch[11])
    hits = []
    for part in batch[12].split('/'):
        try:
            hits.append(float(part.rsplit(':', 1)[1].rstrip('%')))
        except Exception:
            pass
    min_hit = min(hits) if hits else 0.0

    accepts = int(phase[0]); drops = int(phase[1])
    phase50 = float(phase[2]); phase95 = float(phase[3])

    gm95 = []
    mn95 = []
    trk95 = []
    disp95 = []
    for cid in CAMERAS:
        _gm50, gm, _sm, _mn50, mn = stages[cid]
        disp, trk = pts[cid]
        gm95.append(gm); mn95.append(mn); trk95.append(trk); disp95.append(disp)

    med_gm = statistics.median(gm95)
    max_gm = max(gm95)
    med_mn = statistics.median(mn95)
    max_mn = max(mn95)
    med_trk = statistics.median(trk95)
    max_trk = max(trk95)
    med_disp = statistics.median(disp95)

    print(
        "V103_SHARED_PHASE metrics "
        f"samples={samples} target={target} size_p50={size50:.0f} size_p95={size95:.0f} "
        f"full_pct={full_pct:.1f} partial_pct={partial_pct:.1f} "
        f"output_dt_p50={out50:.0f}ms output_dt_p95={out95:.0f}ms "
        f"source_hit_min={min_hit:.0f}% gate_phase_p50={phase50:.0f}ms gate_phase_p95={phase95:.0f}ms "
        f"accepts={accepts} drops={drops}"
    )
    print(
        "V103_SHARED_PHASE latency "
        f"median_gate_mux_p95={med_gm:.0f}ms max_gate_mux_p95={max_gm:.0f}ms "
        f"median_mux_nvdcf_p95={med_mn:.0f}ms max_mux_nvdcf_p95={max_mn:.0f}ms "
        f"median_source_tracker_p95={med_trk:.0f}ms max_source_tracker_p95={max_trk:.0f}ms "
        f"median_source_display_p95={med_disp:.0f}ms"
    )
    print(
        "V103_SHARED_PHASE note=cross-source raw buf_pts spread is informational only "
        f"pts_spread_p50={spread50:.0f}ms pts_spread_p95={spread95:.0f}ms"
    )

    passed = (
        target == 6
        and samples >= 50
        and full_pct >= 65.0
        and size50 >= 5.0
        and min_hit >= 80.0
        and med_gm <= 100.0
        and max_gm <= 140.0
        and med_trk <= 225.0
        and max_trk <= 350.0
        and max_mn <= 180.0
        and med_disp <= 300.0
    )
    if passed:
        print("V103_SHARED_PHASE PASS diagnosis=independent-gate-phase-was-mux-tail-root next=close-tracker-mux-freshness-and-move-to-display-tracker-exact-PTS-alignment")
        return 0

    if full_pct < 65.0 or size50 < 5.0 or min_hit < 80.0:
        diagnosis = "PHASE_ALIGNMENT_INSUFFICIENT"
        next_step = "measure accepted-frame arrival span per global slot; do not lower timeout or batch-size again"
    elif med_gm > 100.0 or max_gm > 140.0:
        diagnosis = "MUX_TAIL_REMAINS"
        next_step = "shared gate improved formation but mux still queues; inspect mux output/backpressure only"
    elif max_mn > 180.0:
        diagnosis = "NVDCF_STAGE_DOMINATES"
        next_step = "mux formation is acceptable; close mux work and isolate NvDCF stage next"
    else:
        diagnosis = "TRACKER_END_TO_END_REMAINS"
        next_step = "tracker batching is improved enough; move to exact display-frame/tracker-frame PTS alignment"
    print(f"V103_SHARED_PHASE FAIL diagnosis={diagnosis} next={next_step}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
