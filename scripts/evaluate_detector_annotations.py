#!/usr/bin/env python3
"""Score human-verified person boxes against saved detector proposals.

The input format is the private Gate 3.1 review JSON. A frame contributes only
when ``annotation_complete`` is true. Boxes are XYXY in the original 1920x1080
rectified frame coordinates. Matching maximizes the number of IoU-qualified
one-to-one matches first, then total IoU; model proposals are never labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        raise ValueError("boxes must be XYXY sequences of four numbers")
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        raise ValueError("boxes must have positive width and height")
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    return intersection / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection)


def _hungarian(cost: list[list[float]]) -> list[int]:
    """Return the selected column for each row of a square cost matrix."""
    n = len(cost)
    if n == 0:
        return []
    if any(len(row) != n for row in cost):
        raise ValueError("assignment matrix must be square")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, n + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment


def match_iou_matrix(iou_matrix: Sequence[Sequence[float]], threshold: float = 0.5) -> list[tuple[int, int, float]]:
    """Maximum-cardinality one-to-one matches above threshold, then max IoU."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("IoU threshold must be between zero and one")
    n_gt = len(iou_matrix)
    n_pred = len(iou_matrix[0]) if n_gt else 0
    if any(len(row) != n_pred for row in iou_matrix):
        raise ValueError("IoU matrix must be rectangular")
    if not n_gt or not n_pred:
        return []
    size = n_gt + n_pred
    cardinality_bonus = min(n_gt, n_pred) + 1.0
    forbidden_cost = 1e6
    costs = [[0.0] * size for _ in range(size)]
    for i in range(n_gt):
        for j in range(n_pred):
            value = float(iou_matrix[i][j])
            if not 0.0 <= value <= 1.0:
                raise ValueError("IoU values must be between zero and one")
            costs[i][j] = -(cardinality_bonus + value) if value >= threshold else forbidden_cost
    assignment = _hungarian(costs)
    matches = []
    for i in range(n_gt):
        j = assignment[i]
        if 0 <= j < n_pred and float(iou_matrix[i][j]) >= threshold:
            matches.append((i, j, float(iou_matrix[i][j])))
    return matches


def evaluate_frame(gt_boxes: Sequence[Sequence[float]], pred_boxes: Sequence[Sequence[float]],
                   threshold: float = 0.5) -> dict:
    matrix = [[box_iou(gt, pred) for pred in pred_boxes] for gt in gt_boxes]
    matches = match_iou_matrix(matrix, threshold)
    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "matches": matches}


def evaluate_review(data: dict, detector: str, threshold: float = 0.5) -> dict:
    totals = {"tp": 0, "fp": 0, "fn": 0, "frames": 0}
    by_camera: dict[str, dict[str, int]] = {}
    for frame in data.get("images", []):
        if not frame.get("annotation_complete", False):
            continue
        camera = str(frame["camera"])
        ground_truth = frame.get("human_verified_annotations")
        if ground_truth is None:
            raise ValueError(f"completed frame has no human annotations: {frame.get('image')}")
        proposals = frame.get("proposals", {}).get(detector)
        if proposals is None:
            raise ValueError(f"missing detector proposals {detector!r}: {frame.get('image')}")
        result = evaluate_frame(
            [item["bbox_xyxy"] for item in ground_truth],
            [item["xyxy"] for item in proposals],
            threshold,
        )
        bucket = by_camera.setdefault(camera, {"tp": 0, "fp": 0, "fn": 0, "frames": 0})
        for key in ("tp", "fp", "fn"):
            bucket[key] += result[key]
            totals[key] += result[key]
        bucket["frames"] += 1
        totals["frames"] += 1
    if totals["frames"] == 0:
        raise ValueError("no frames marked annotation_complete; no accuracy metrics computed")
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "detector": detector,
        "iou_threshold": threshold,
        "scope": "only frames explicitly marked annotation_complete by a human reviewer",
        "totals": {**totals, "precision": precision, "recall": recall, "f1": f1},
        "by_camera": by_camera,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--detector", choices=("YOLO26m", "PeopleNet Transformer"), required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    data = json.loads(args.annotations.read_text())
    try:
        result = evaluate_review(data, args.detector, args.iou_threshold)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
