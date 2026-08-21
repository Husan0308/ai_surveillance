#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step 4.6: audit TensorRT live person boxes for a systematic bottom-border "
            "short/wide fragment. Diagnostic only; does not modify detector output."
        )
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/rfdetr_step4/trt_live/live_report.json"),
    )
    p.add_argument("--width", type=float, default=800.0)
    p.add_argument("--height", type=float, default=448.0)
    p.add_argument("--bottom-edge", type=float, default=0.985)
    p.add_argument("--max-height-ratio", type=float, default=0.12)
    p.add_argument("--min-aspect", type=float, default=1.60)
    p.add_argument("--expected-persons", type=int, default=2)
    return p.parse_args()


def _metrics(person: dict, width: float, height: float) -> dict:
    x1, y1, x2, y2 = [float(v) for v in person["xyxy"]]
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return {
        "w": w,
        "h": h,
        "height_ratio": h / height,
        "width_ratio": w / width,
        "aspect": w / h,
        "bottom_ratio": y2 / height,
        "cx_ratio": ((x1 + x2) * 0.5) / width,
        "cy_ratio": ((y1 + y2) * 0.5) / height,
    }


def main() -> int:
    args = _args()
    if not args.report.is_file():
        raise SystemExit(f"STEP4_6_FAIL report_not_found={args.report}")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("STEP4_6_FAIL invalid_frame_shape")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    frames = list(report.get("frames") or [])
    if not frames:
        raise SystemExit("STEP4_6_FAIL no_frames")

    raw_counts: list[int] = []
    corrected_counts: list[int] = []
    candidate_counts: list[int] = []
    candidates: list[dict] = []

    for frame in frames:
        persons = list(frame.get("persons") or [])
        frame_candidates: list[dict] = []
        for index, person in enumerate(persons, start=1):
            m = _metrics(person, args.width, args.height)
            is_candidate = (
                m["bottom_ratio"] >= float(args.bottom_edge)
                and m["height_ratio"] <= float(args.max_height_ratio)
                and m["aspect"] >= float(args.min_aspect)
            )
            if is_candidate:
                row = {
                    "frame": int(frame.get("n", 0)),
                    "index": index,
                    "confidence": float(person.get("confidence", 0.0)),
                    "xyxy": [float(v) for v in person["xyxy"]],
                    **m,
                }
                frame_candidates.append(row)
                candidates.append(row)

        raw_counts.append(len(persons))
        candidate_counts.append(len(frame_candidates))
        corrected_counts.append(len(persons) - len(frame_candidates))

    raw_mode, raw_mode_frames = Counter(raw_counts).most_common(1)[0]
    corrected_mode, corrected_mode_frames = Counter(corrected_counts).most_common(1)[0]
    candidate_mode, candidate_mode_frames = Counter(candidate_counts).most_common(1)[0]
    exact_one_candidate = sum(1 for value in candidate_counts if value == 1)
    expected_corrected = sum(1 for value in corrected_counts if value == args.expected_persons)

    if candidates:
        confs = [row["confidence"] for row in candidates]
        hratios = [row["height_ratio"] for row in candidates]
        aspects = [row["aspect"] for row in candidates]
        bottoms = [row["bottom_ratio"] for row in candidates]
        cx = [row["cx_ratio"] for row in candidates]
        candidate_stats = {
            "count": len(candidates),
            "confidence_mean": statistics.mean(confs),
            "confidence_min": min(confs),
            "confidence_max": max(confs),
            "height_ratio_mean": statistics.mean(hratios),
            "aspect_mean": statistics.mean(aspects),
            "bottom_ratio_mean": statistics.mean(bottoms),
            "cx_ratio_mean": statistics.mean(cx),
            "cx_ratio_min": min(cx),
            "cx_ratio_max": max(cx),
        }
    else:
        candidate_stats = {"count": 0}

    output = {
        "stage": "4.6",
        "source_report": str(args.report),
        "frame_shape": [int(args.width), int(args.height)],
        "diagnostic_rule": {
            "bottom_ratio_gte": float(args.bottom_edge),
            "height_ratio_lte": float(args.max_height_ratio),
            "aspect_gte": float(args.min_aspect),
        },
        "frames": len(frames),
        "raw": {
            "mode": raw_mode,
            "mode_frames": raw_mode_frames,
            "min": min(raw_counts),
            "max": max(raw_counts),
        },
        "candidate_count": {
            "mode": candidate_mode,
            "mode_frames": candidate_mode_frames,
            "exactly_one_frames": exact_one_candidate,
        },
        "diagnostic_corrected": {
            "mode": corrected_mode,
            "mode_frames": corrected_mode_frames,
            "min": min(corrected_counts),
            "max": max(corrected_counts),
            "expected_persons": int(args.expected_persons),
            "expected_frames": expected_corrected,
        },
        "candidate_stats": candidate_stats,
        "candidate_examples": candidates[:20],
    }

    output_path = args.report.parent / "fp_geometry_audit.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(
        "STEP4_6_RESULT "
        f"frames={len(frames)} raw_mode={raw_mode}({raw_mode_frames}/{len(frames)}) "
        f"candidate_mode={candidate_mode}({candidate_mode_frames}/{len(frames)}) "
        f"exact_one_candidate={exact_one_candidate}/{len(frames)} "
        f"corrected_mode={corrected_mode}({corrected_mode_frames}/{len(frames)}) "
        f"corrected_range={min(corrected_counts)}-{max(corrected_counts)} "
        f"expected={args.expected_persons}:{expected_corrected}/{len(frames)}",
        flush=True,
    )
    if candidates:
        print(
            "STEP4_6_CANDIDATE "
            f"count={len(candidates)} "
            f"conf={candidate_stats['confidence_min']:.3f}-{candidate_stats['confidence_max']:.3f} "
            f"mean_conf={candidate_stats['confidence_mean']:.3f} "
            f"mean_height_ratio={candidate_stats['height_ratio_mean']:.3f} "
            f"mean_aspect={candidate_stats['aspect_mean']:.2f} "
            f"mean_bottom_ratio={candidate_stats['bottom_ratio_mean']:.4f} "
            f"cx_range={candidate_stats['cx_ratio_min']:.3f}-{candidate_stats['cx_ratio_max']:.3f}",
            flush=True,
        )
    print(f"STEP4_6_JSON={output_path}", flush=True)
    print("STEP4_6_PASS diagnostic_only=true", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
