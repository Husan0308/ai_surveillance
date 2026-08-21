#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step 4.8: diagnostic-only audit of residual 3-person frames after the "
            "bottom-fragment filter. Each 3-box frame is matched against the nearest "
            "2-box frame; the unmatched box is summarized as a residual FP candidate."
        )
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/rfdetr_step4/trt_live_fragment_filtered/live_report.json"),
    )
    p.add_argument("--expected-persons", type=int, default=2)
    p.add_argument("--width", type=float, default=800.0)
    p.add_argument("--height", type=float, default=448.0)
    return p.parse_args()


def _geom(person: dict, fw: float, fh: float) -> dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in person["xyxy"]]
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return {
        "cx": ((x1 + x2) * 0.5) / fw,
        "cy": ((y1 + y2) * 0.5) / fh,
        "wr": w / fw,
        "hr": h / fh,
        "aspect": w / h,
        "bottom": y2 / fh,
        "area": (w * h) / (fw * fh),
    }


def _match_cost(a: dict, b: dict, fw: float, fh: float) -> float:
    ga = _geom(a, fw, fh)
    gb = _geom(b, fw, fh)
    center = math.hypot(ga["cx"] - gb["cx"], ga["cy"] - gb["cy"])
    size = abs(math.log(max(1e-6, ga["wr"]) / max(1e-6, gb["wr"])))
    size += abs(math.log(max(1e-6, ga["hr"]) / max(1e-6, gb["hr"])))
    return center + 0.12 * size


def _nearest_reference(index: int, reference_indices: list[int]) -> int:
    return min(reference_indices, key=lambda j: (abs(j - index), j))


def _find_unmatched(
    current: list[dict], reference: list[dict], fw: float, fh: float
) -> tuple[int, float, list[tuple[int, int]]]:
    if len(current) != 3 or len(reference) != 2:
        raise ValueError("expected current=3 and reference=2")
    best_cost = float("inf")
    best_pairs: list[tuple[int, int]] = []
    best_used: set[int] = set()
    for chosen in itertools.combinations(range(3), 2):
        for perm in itertools.permutations(chosen):
            pairs = [(perm[0], 0), (perm[1], 1)]
            cost = sum(_match_cost(current[a], reference[b], fw, fh) for a, b in pairs)
            if cost < best_cost:
                best_cost = cost
                best_pairs = pairs
                best_used = set(perm)
    unmatched = next(i for i in range(3) if i not in best_used)
    return unmatched, best_cost, best_pairs


def _fmt_range(values: list[float], digits: int = 3) -> str:
    if not values:
        return "n/a"
    return f"{min(values):.{digits}f}-{max(values):.{digits}f}"


def main() -> int:
    args = _args()
    if not args.report.is_file():
        raise SystemExit(f"STEP4_8_FAIL report_not_found={args.report}")
    if args.expected_persons != 2:
        raise SystemExit("STEP4_8_FAIL current_audit_contract_expected_persons_must_be_2")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    frames = list(report.get("frames") or [])
    if not frames:
        raise SystemExit("STEP4_8_FAIL no_frames")

    two_indices = [i for i, f in enumerate(frames) if len(f.get("persons") or []) == 2]
    three_indices = [i for i, f in enumerate(frames) if len(f.get("persons") or []) == 3]
    other_indices = [
        i for i, f in enumerate(frames) if len(f.get("persons") or []) not in (2, 3)
    ]
    if not two_indices:
        raise SystemExit("STEP4_8_FAIL no_reference_two_person_frames")
    if not three_indices:
        raise SystemExit("STEP4_8_FAIL no_residual_three_person_frames")

    candidates: list[dict] = []
    match_costs: list[float] = []
    query_counter: Counter[int] = Counter()

    for idx in three_indices:
        frame = frames[idx]
        current = list(frame.get("persons") or [])
        ref_idx = _nearest_reference(idx, two_indices)
        reference = list(frames[ref_idx].get("persons") or [])
        unmatched_idx, cost, pairs = _find_unmatched(
            current, reference, float(args.width), float(args.height)
        )
        person = current[unmatched_idx]
        g = _geom(person, float(args.width), float(args.height))
        query = int(person.get("query", -1))
        query_counter[query] += 1
        match_costs.append(cost)
        candidates.append(
            {
                "frame": int(frame.get("n", idx + 1)),
                "reference_frame": int(frames[ref_idx].get("n", ref_idx + 1)),
                "unmatched_index": unmatched_idx + 1,
                "query": query,
                "confidence": float(person.get("confidence", 0.0)),
                "xyxy": [float(v) for v in person["xyxy"]],
                "match_cost": float(cost),
                "matched_pairs": [[a + 1, b + 1] for a, b in pairs],
                **g,
            }
        )

    confs = [c["confidence"] for c in candidates]
    cxs = [c["cx"] for c in candidates]
    cys = [c["cy"] for c in candidates]
    hrs = [c["hr"] for c in candidates]
    wrs = [c["wr"] for c in candidates]
    aspects = [c["aspect"] for c in candidates]
    bottoms = [c["bottom"] for c in candidates]
    areas = [c["area"] for c in candidates]

    top_queries = query_counter.most_common(8)
    dominant_query = top_queries[0] if top_queries else (-1, 0)

    output = {
        "stage": "4.8",
        "source_report": str(args.report),
        "frames": len(frames),
        "reference_two_person_frames": len(two_indices),
        "residual_three_person_frames": len(three_indices),
        "other_count_frames": len(other_indices),
        "matching": "nearest 2-person frame + exhaustive 2-of-3 geometric assignment",
        "candidate_stats": {
            "count": len(candidates),
            "confidence_min": min(confs),
            "confidence_max": max(confs),
            "confidence_mean": statistics.mean(confs),
            "cx_min": min(cxs),
            "cx_max": max(cxs),
            "cy_min": min(cys),
            "cy_max": max(cys),
            "height_ratio_min": min(hrs),
            "height_ratio_max": max(hrs),
            "width_ratio_min": min(wrs),
            "width_ratio_max": max(wrs),
            "aspect_min": min(aspects),
            "aspect_max": max(aspects),
            "bottom_min": min(bottoms),
            "bottom_max": max(bottoms),
            "area_min": min(areas),
            "area_max": max(areas),
            "match_cost_mean": statistics.mean(match_costs),
        },
        "query_counts": [{"query": q, "count": n} for q, n in top_queries],
        "candidate_examples": candidates[:20],
    }
    output_path = args.report.parent / "residual_fp_audit.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(
        "STEP4_8_RESULT "
        f"frames={len(frames)} correct2={len(two_indices)} residual3={len(three_indices)} "
        f"other={len(other_indices)} candidates={len(candidates)} "
        f"dominant_query={dominant_query[0]}:{dominant_query[1]}/{len(candidates)} "
        f"conf={_fmt_range(confs)} mean_conf={statistics.mean(confs):.3f} "
        f"cx={_fmt_range(cxs)} cy={_fmt_range(cys)}",
        flush=True,
    )
    print(
        "STEP4_8_GEOMETRY "
        f"height_ratio={_fmt_range(hrs)} width_ratio={_fmt_range(wrs)} "
        f"aspect={_fmt_range(aspects,2)} bottom={_fmt_range(bottoms,4)} "
        f"area={_fmt_range(areas,4)} match_cost_mean={statistics.mean(match_costs):.4f}",
        flush=True,
    )
    print(
        "STEP4_8_QUERIES "
        + " ".join(f"q{q}={n}" for q, n in top_queries),
        flush=True,
    )
    for row in candidates[:5]:
        print(
            "STEP4_8_EXAMPLE "
            f"frame={row['frame']} ref={row['reference_frame']} query={row['query']} "
            f"conf={row['confidence']:.3f} xyxy={[round(v,1) for v in row['xyxy']]} "
            f"cx={row['cx']:.3f} cy={row['cy']:.3f} hr={row['hr']:.3f} "
            f"aspect={row['aspect']:.2f} bottom={row['bottom']:.4f}",
            flush=True,
        )
    print(f"STEP4_8_JSON={output_path}", flush=True)
    print("STEP4_8_PASS diagnostic_only=true", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
