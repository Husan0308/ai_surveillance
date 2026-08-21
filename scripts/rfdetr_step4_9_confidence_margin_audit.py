#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rfdetr_step4_8_residual_fp_audit import _find_unmatched, _nearest_reference


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step 4.9: diagnostic-only confidence separation audit after the bottom-fragment filter. "
            "Classifies the residual third box using Step 4.8 matching, then checks whether a global "
            "confidence threshold can remove it without dropping either of the two matched people."
        )
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/rfdetr_step4/trt_live_fragment_filtered/live_report.json"),
    )
    p.add_argument("--width", type=float, default=800.0)
    p.add_argument("--height", type=float, default=448.0)
    p.add_argument("--expected-persons", type=int, default=2)
    return p.parse_args()


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = int(round((len(ordered) - 1) * q))
    pos = max(0, min(len(ordered) - 1, pos))
    return float(ordered[pos])


def main() -> int:
    args = _args()
    if not args.report.is_file():
        raise SystemExit(f"STEP4_9_FAIL report_not_found={args.report}")
    if args.expected_persons != 2:
        raise SystemExit("STEP4_9_FAIL expected_persons_must_be_2")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    frames = list(report.get("frames") or [])
    if not frames:
        raise SystemExit("STEP4_9_FAIL no_frames")

    two_indices = [i for i, f in enumerate(frames) if len(f.get("persons") or []) == 2]
    three_indices = [i for i, f in enumerate(frames) if len(f.get("persons") or []) == 3]
    other = [i for i, f in enumerate(frames) if len(f.get("persons") or []) not in (2, 3)]
    if not two_indices or not three_indices:
        raise SystemExit(
            f"STEP4_9_FAIL need_both_2_and_3_person_frames got2={len(two_indices)} got3={len(three_indices)}"
        )

    fp_keys: set[tuple[int, int]] = set()
    fp_conf: list[float] = []
    real_conf: list[float] = []

    # Frames already at two detections are treated as the two reference people.
    for idx in two_indices:
        for person in frames[idx].get("persons") or []:
            real_conf.append(float(person.get("confidence", 0.0)))

    # In residual 3-person frames, reuse Step 4.8's nearest-frame geometric assignment.
    for idx in three_indices:
        current = list(frames[idx].get("persons") or [])
        ref_idx = _nearest_reference(idx, two_indices)
        reference = list(frames[ref_idx].get("persons") or [])
        unmatched_idx, _cost, _pairs = _find_unmatched(
            current, reference, float(args.width), float(args.height)
        )
        for person_idx, person in enumerate(current):
            conf = float(person.get("confidence", 0.0))
            if person_idx == unmatched_idx:
                fp_conf.append(conf)
                fp_keys.add((idx, person_idx))
            else:
                real_conf.append(conf)

    if not fp_conf or not real_conf:
        raise SystemExit("STEP4_9_FAIL classification_empty")

    fp_max = max(fp_conf)
    real_min = min(real_conf)
    margin = real_min - fp_max
    suggested = (fp_max + real_min) * 0.5 if margin > 0.0 else None

    thresholds = sorted(
        set(
            [0.18, 0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.49, 0.50, 0.52, 0.55, 0.60]
            + ([round(float(suggested), 4)] if suggested is not None else [])
        )
    )
    sweep = []
    for threshold in thresholds:
        counts = []
        for frame in frames:
            persons = list(frame.get("persons") or [])
            counts.append(sum(1 for p in persons if float(p.get("confidence", 0.0)) >= threshold))
        expected_frames = sum(1 for count in counts if count == args.expected_persons)
        under = sum(1 for count in counts if count < args.expected_persons)
        over = sum(1 for count in counts if count > args.expected_persons)
        sweep.append(
            {
                "threshold": threshold,
                "expected_frames": expected_frames,
                "under_frames": under,
                "over_frames": over,
                "min_count": min(counts),
                "max_count": max(counts),
            }
        )

    best = max(
        sweep,
        key=lambda row: (row["expected_frames"], -row["under_frames"], -row["over_frames"], -row["threshold"]),
    )

    output = {
        "stage": "4.9",
        "source_report": str(args.report),
        "frames": len(frames),
        "two_person_frames": len(two_indices),
        "three_person_frames": len(three_indices),
        "other_frames": len(other),
        "classification": "Step4.8 nearest 2-person frame geometric assignment",
        "real_confidence": {
            "count": len(real_conf),
            "min": real_min,
            "p01": _pct(real_conf, 0.01),
            "p05": _pct(real_conf, 0.05),
            "mean": statistics.mean(real_conf),
            "max": max(real_conf),
        },
        "residual_fp_confidence": {
            "count": len(fp_conf),
            "min": min(fp_conf),
            "mean": statistics.mean(fp_conf),
            "max": fp_max,
        },
        "separation_margin": margin,
        "safe_midpoint_threshold": suggested,
        "sweep": sweep,
        "best_sweep": best,
    }
    out = args.report.parent / "confidence_margin_audit.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(
        "STEP4_9_RESULT "
        f"frames={len(frames)} real_samples={len(real_conf)} fp_samples={len(fp_conf)} "
        f"real_conf={real_min:.3f}-{max(real_conf):.3f} real_p01={_pct(real_conf,0.01):.3f} "
        f"real_p05={_pct(real_conf,0.05):.3f} fp_conf={min(fp_conf):.3f}-{fp_max:.3f} "
        f"margin={margin:.3f} safe_midpoint={suggested if suggested is not None else 'none'}",
        flush=True,
    )
    print(
        "STEP4_9_BEST "
        f"threshold={best['threshold']:.4f} expected={best['expected_frames']}/{len(frames)} "
        f"under={best['under_frames']} over={best['over_frames']} "
        f"count_range={best['min_count']}-{best['max_count']}",
        flush=True,
    )
    for row in sweep:
        if row["threshold"] >= 0.45:
            print(
                "STEP4_9_SWEEP "
                f"threshold={row['threshold']:.4f} expected={row['expected_frames']}/{len(frames)} "
                f"under={row['under_frames']} over={row['over_frames']} range={row['min_count']}-{row['max_count']}",
                flush=True,
            )
    print(f"STEP4_9_JSON={out}", flush=True)
    print("STEP4_9_PASS diagnostic_only=true", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
