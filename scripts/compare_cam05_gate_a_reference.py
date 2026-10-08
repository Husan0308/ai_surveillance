#!/usr/bin/env python3
"""Compare isolated candidate canonical boxes with amended human reference v5."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _box(value):
    if isinstance(value, str):
        value = json.loads(value) if value else None
    if not value or len(value) != 4:
        return None
    values = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in values) or values[2] <= 0 or values[3] <= 0:
        return None
    return values


def _canonical_xyxy_to_xywh(value):
    xyxy = _box(value)
    if not xyxy:
        return None
    left, top, right, bottom = xyxy
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def _iou(a, b):
    x = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    intersection = x * y
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0.0


def _read_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def compare(run_dir: Path, reference_dir: Path) -> dict:
    results = {}
    canonical = defaultdict(lambda: defaultdict(list))
    for row in _read_jsonl(run_dir / "canonical-objects.jsonl"):
        for obj in row.get("objects", []):
            bbox = _canonical_xyxy_to_xywh(obj.get("bbox"))
            if bbox:
                canonical[row["camera_id"]][int(row["source_frame_num"])].append({
                    "bbox": bbox,
                    "source": obj.get("bbox_source", "UNKNOWN"),
                    "native_track_id": obj.get("native_track_id"),
                })

    for camera in ("CAM-01", "CAM-04"):
        expected = defaultdict(list)
        pts = {}
        with (reference_dir / f"{camera.replace('-', '')}-per-frame.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                bbox = _box(row.get("chosen_bbox"))
                if bbox:
                    frame = int(row["frame"])
                    expected[frame].append({"bbox": bbox, "source": row.get("chosen_source", "")})
                    pts[frame] = int(row["pts"])

        misses, extras, ious = [], [], []
        source_counts = Counter()
        per_frame = {}
        for frame in sorted(expected):
            candidates = canonical[camera].get(frame, [])
            pairs = sorted(
                ((_iou(ref["bbox"], cand["bbox"]), ri, ci)
                 for ri, ref in enumerate(expected[frame])
                 for ci, cand in enumerate(candidates)), reverse=True)
            used_ref, used_cand = set(), set()
            for score, ri, ci in pairs:
                if score <= 0 or ri in used_ref or ci in used_cand:
                    continue
                used_ref.add(ri)
                used_cand.add(ci)
                ious.append(score)
                source_counts[candidates[ci]["source"]] += 1
            missed_here = [i for i in range(len(expected[frame])) if i not in used_ref]
            extra_here = [i for i in range(len(candidates)) if i not in used_cand]
            misses.extend((frame, expected[frame][i]) for i in missed_here)
            extras.extend((frame, candidates[i]) for i in extra_here)
            per_frame[str(frame)] = {"expected": len(expected[frame]), "rendered": len(candidates),
                                     "missed": len(missed_here), "extra": len(extra_here),
                                     "pts_ns": pts[frame]}

        def pct(values, p):
            if not values:
                return None
            ordered = sorted(values)
            return ordered[round((len(ordered) - 1) * p)]

        results[camera] = {
            "reference_frame_count": len(expected),
            "reference_box_count": sum(map(len, expected.values())),
            "candidate_boxes_on_reviewed_frames": sum(len(canonical[camera].get(f, [])) for f in expected),
            "missed_reference_boxes": len(misses),
            "missed_examples": [{"frame": frame, **item} for frame, item in misses[:25]],
            "unmatched_candidate_boxes": len(extras),
            "unmatched_examples": [{"frame": frame, **item} for frame, item in extras[:25]],
            "matched_iou_p05_p50_p95_min": [pct(ious, .05), pct(ious, .50), pct(ious, .95), min(ious) if ious else None],
            "matched_box_source_counts": dict(source_counts),
            "per_frame": per_frame,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.run, args.reference)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
