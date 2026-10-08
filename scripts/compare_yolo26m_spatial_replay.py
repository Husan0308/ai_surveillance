#!/usr/bin/env python3
"""Frame-align and compare frozen 640 PGIE rows with an isolated spatial candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def load_run(directory: Path, source_map: dict[int, str]) -> tuple[dict, dict]:
    frame_map: dict[tuple[str, int], dict] = {}
    for row in rows(directory / "frames.jsonl"):
        sid = int(row["source_id"])
        if sid not in source_map:
            continue
        key = source_map[sid], int(row["frame"])
        if key in frame_map:
            raise ValueError(f"duplicate frame record {key} in {directory}")
        frame_map[key] = row
    box_map: dict[tuple[str, int], list[dict]] = {}
    for row in rows(directory / "detections.jsonl"):
        sid = int(row["source_id"])
        if sid not in source_map or int(row["class_id"]) != 0:
            continue
        key = source_map[sid], int(row["frame"])
        if key not in frame_map:
            raise ValueError(f"detection has no same-frame metadata {key} in {directory}")
        frame = frame_map[key]
        px, py = int(frame["pipeline_width"]), int(frame["pipeline_height"])
        sx, sy = int(frame["source_width"]), int(frame["source_height"])
        x, y, w, h = map(float, row["box"])
        box_map.setdefault(key, []).append({
            "bbox": [x * sx / px, y * sy / py, w * sx / px, h * sy / py],
            "confidence": float(row["confidence"]), "pts_ns": int(row["pts_ns"]),
        })
    return frame_map, box_map


def match(left: list[dict], right: list[dict]) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates = sorted(((iou(a["bbox"], b["bbox"]), i, j)
                         for i, a in enumerate(left) for j, b in enumerate(right)), reverse=True)
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for score, i, j in candidates:
        if score < 0.10:
            break
        if i not in used_left and j not in used_right:
            used_left.add(i); used_right.add(j); pairs.append((i, j, score))
    return pairs, set(range(len(left))) - used_left, set(range(len(right))) - used_right


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--baseline-name", default="frozen-640")
    ap.add_argument("--candidate-name", default="spatial-candidate")
    args = ap.parse_args()
    baseline, candidate, output = args.baseline.resolve(), args.candidate.resolve(), args.output.resolve()
    if output.exists() or not output.is_relative_to(Path.cwd() / ".runtime"):
        raise ValueError("output must be a new directory under .runtime")
    if not all((p / name).is_file() for p in (baseline, candidate) for name in ("frames.jsonl", "detections.jsonl")):
        raise FileNotFoundError("both runs must contain frame and detection JSONL")
    base_frames, base_boxes = load_run(baseline, {0: "CAM-01", 1: "CAM-04"})
    cand_frames, cand_boxes = load_run(candidate, {0: "CAM-01", 1: "CAM-04"})
    output.mkdir(parents=True)
    report: dict = {"status": "DIAGNOSTIC_ONLY", "baseline": str(baseline.relative_to(Path.cwd())),
        "candidate": str(candidate.relative_to(Path.cwd())), "camera_results": {}}
    delta_path = output / "unmatched-candidate-proposals.jsonl"
    summary_path = output / "comparison.json"
    ious_all: dict[str, list[float]] = {"CAM-01": [], "CAM-04": []}
    with delta_path.open("x") as delta:
        for camera in ("CAM-01", "CAM-04"):
            base_keys = sorted(key for key in base_frames if key[0] == camera)
            cand_keys = sorted(key for key in cand_frames if key[0] == camera)
            if len(base_keys) != 2400 or len(cand_keys) != 2400 or [k[1] for k in cand_keys] != list(range(2400)):
                raise ValueError(f"source-local frame accounting must be 0..2399 for {camera}")
            if [k[1] for k in base_keys] != list(range(2400)):
                raise ValueError(f"baseline source-local frame accounting must be 0..2399 for {camera}")
            if any(base_frames[k]["pts_ns"] != cand_frames[k]["pts_ns"] for k in base_keys):
                raise ValueError(f"same frame index has different source PTS for {camera}")
            matched = missing = added = 0
            frame_delta_counts: list[dict] = []
            for key in base_keys:
                old = base_boxes.get(key, [])
                new = cand_boxes.get(key, [])
                pairs, old_only, new_only = match(old, new)
                matched += len(pairs); missing += len(old_only); added += len(new_only)
                ious_all[camera].extend(score for _, _, score in pairs)
                if old_only or new_only:
                    frame_delta_counts.append({"frame": key[1], "baseline_count": len(old),
                        "candidate_count": len(new), "baseline_only": len(old_only), "candidate_only": len(new_only)})
                for idx in sorted(new_only):
                    delta.write(json.dumps({"camera_id": camera, "frame": key[1],
                        "pts_ns": cand_frames[key]["pts_ns"], **new[idx]}) + "\n")
            values = ious_all[camera]
            report["camera_results"][camera] = {
                "baseline_unique_frames": len(base_keys), "candidate_unique_frames": len(cand_keys),
                "baseline_detections": sum(len(base_boxes.get(k, [])) for k in base_keys),
                "candidate_detections": sum(len(cand_boxes.get(k, [])) for k in base_keys),
                "matched_by_iou_ge_0_10": matched, "baseline_only": missing,
                "candidate_only": added, "frames_with_any_difference": len(frame_delta_counts),
                "matched_iou_p05": sorted(values)[max(0, int(len(values) * .05) - 1)] if values else None,
                "matched_iou_median": statistics.median(values) if values else None,
                "matched_iou_p95": sorted(values)[min(len(values)-1, int(len(values) * .95))] if values else None,
                "differing_frame_counts": frame_delta_counts,
            }
    report["provenance"] = {
        "baseline_detections_sha256": sha(baseline / "detections.jsonl"),
        "baseline_frames_sha256": sha(baseline / "frames.jsonl"),
        "candidate_detections_sha256": sha(candidate / "detections.jsonl"),
        "candidate_frames_sha256": sha(candidate / "frames.jsonl"),
        "box_space": "normalized to same source-frame dimensions using each run's frame metadata",
        "matching": "greedy maximum-IoU one-to-one matching with IoU >= 0.10; candidate-only rows are retained for review",
        "pts": "same source-local frame index required to have identical source PTS",
    }
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["camera_results"], indent=2))


if __name__ == "__main__":
    main()
