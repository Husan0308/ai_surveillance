#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v2_bbox_v101_log.py /tmp/CAMERA_BBOX_V101.log")
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V101_ARCH" not in text:
        print("V101_BATCH_AUDIT FAIL missing=CAMERA_V101_ARCH")
        return 2

    pat = re.compile(
        r"CAMERA_V101_BATCH samples=(\d+) target=(\d+) "
        r"size_p50=([0-9.]+) size_p95=([0-9.]+) "
        r"unique_p50=([0-9.]+) unique_p95=([0-9.]+) "
        r"full_pct=([0-9.]+) partial_pct=([0-9.]+) "
        r"pts_spread_p50=([0-9.]+)ms pts_spread_p95=([0-9.]+)ms "
        r"output_dt_p50=([0-9.]+)ms output_dt_p95=([0-9.]+)ms "
        r"source_hit=([^\s]+)"
    )
    matches = list(pat.finditer(text))
    if not matches:
        print("V101_BATCH_AUDIT FAIL missing=CAMERA_V101_BATCH")
        return 2
    m = matches[-1]
    samples = int(m.group(1))
    target = int(m.group(2))
    size50 = float(m.group(3)); size95 = float(m.group(4))
    uniq50 = float(m.group(5)); uniq95 = float(m.group(6))
    full_pct = float(m.group(7)); partial_pct = float(m.group(8))
    spread50 = float(m.group(9)); spread95 = float(m.group(10))
    out50 = float(m.group(11)); out95 = float(m.group(12))
    source_hit_text = m.group(13)

    ratios = []
    for part in source_hit_text.split('/'):
        try:
            ratios.append(float(part.rsplit(':', 1)[1].rstrip('%')))
        except Exception:
            pass
    min_hit = min(ratios) if ratios else 0.0
    max_hit = max(ratios) if ratios else 0.0

    if samples < 50:
        diagnosis = "INSUFFICIENT_SAMPLES"
        next_step = "run at least 45-60 seconds"
    elif min_hit + 10.0 < max_hit:
        diagnosis = "SOURCE_IMBALANCE"
        next_step = "fix per-source tracker mux fairness/admission only"
    elif full_pct < 50.0 or size50 < target:
        diagnosis = "MOSTLY_PARTIAL"
        next_step = "partial batches dominate; tune batch formation for low latency instead of waiting for six"
    elif full_pct >= 80.0 and out95 > 100.0:
        diagnosis = "FULL_BUT_BACKPRESSURED"
        next_step = "batches are already full; locate downstream mux->NvDCF backpressure before more timeout changes"
    elif spread95 > 150.0:
        diagnosis = "BATCH_PTS_SKEW"
        next_step = "fix cross-source PTS skew/synchronization policy only"
    else:
        diagnosis = "MIXED_BATCHING"
        next_step = "use batch-size and per-source inclusion data to choose one low-latency mux policy"

    print(
        "V101_BATCH_AUDIT RESULT "
        f"diagnosis={diagnosis} samples={samples} target={target} "
        f"size_p50={size50:.0f} size_p95={size95:.0f} unique_p50={uniq50:.0f} unique_p95={uniq95:.0f} "
        f"full_pct={full_pct:.1f} partial_pct={partial_pct:.1f} "
        f"pts_spread_p50={spread50:.0f}ms pts_spread_p95={spread95:.0f}ms "
        f"output_dt_p50={out50:.0f}ms output_dt_p95={out95:.0f}ms "
        f"source_hit_min={min_hit:.0f}% source_hit_max={max_hit:.0f}%"
    )
    print(f"V101_BATCH_AUDIT source_hit={source_hit_text}")
    print(f"V101_BATCH_AUDIT next={next_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
