#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_camera_v11_trt86_person_quality import Runner, iou, load_ppm_rgb, resolve


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be label=engine_path")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("candidate label is empty")
    return label, resolve(path.strip())


def match_gt(
    preds: list[list[float]],
    gt: list[list[float]],
    conf: float,
    gate: float,
) -> dict[int, tuple[list[float], float]]:
    used_gt: set[int] = set()
    matched: dict[int, tuple[list[float], float]] = {}
    for pred in sorted((row for row in preds if row[4] >= conf), key=lambda row: row[4], reverse=True):
        best_idx = -1
        best_iou = gate
        for gt_idx, box in enumerate(gt):
            if gt_idx in used_gt:
                continue
            value = iou(pred[:4], box)
            if value >= best_iou:
                best_iou = value
                best_idx = gt_idx
        if best_idx >= 0:
            used_gt.add(best_idx)
            matched[best_idx] = (pred, best_iou)
    return matched


def best_candidate_for_gt(
    preds: list[list[float]], box: list[float]
) -> tuple[list[float] | None, float]:
    best_pred = None
    best_iou = -1.0
    for pred in preds:
        value = iou(pred[:4], box)
        if value > best_iou:
            best_iou = value
            best_pred = pred
    return best_pred, max(0.0, best_iou)


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diagnose FP32->mixed person recall loss as score drift vs bbox/ranking drift"
    )
    ap.add_argument(
        "--fp32",
        default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine",
    )
    ap.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="repeatable label=engine_path",
    )
    ap.add_argument(
        "--quality-dir",
        default="artifacts/yolo26s_trt86/person_final_holdout_b1",
    )
    ap.add_argument("--fp32-conf", type=float, default=0.18)
    ap.add_argument("--candidate-conf", type=float, default=0.16)
    ap.add_argument("--iou", type=float, default=0.50)
    args = ap.parse_args()

    fp32_path = resolve(args.fp32)
    quality_dir = resolve(args.quality_dir)
    manifest_path = quality_dir / "person_gt.json"
    candidates = [parse_candidate(value) for value in args.candidate]

    for path in [fp32_path, manifest_path, *[path for _, path in candidates]]:
        if not path.is_file():
            raise SystemExit(f"V11_OUTPUT_DRIFT FAIL missing={path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("images") or []
    if len(entries) < 100:
        raise SystemExit(f"V11_OUTPUT_DRIFT FAIL images={len(entries)} expected>=100")

    print(
        "V11_OUTPUT_DRIFT_START "
        f"images={len(entries)} fp32={fp32_path.name} fp32_conf={args.fp32_conf:.3f} "
        f"candidate_conf={args.candidate_conf:.3f} iou={args.iou:.2f} "
        f"candidates={','.join(label for label, _ in candidates)}",
        flush=True,
    )

    fp32_runner = Runner(fp32_path)
    baseline: list[tuple[list[list[float]], list[list[float]], dict[int, tuple[list[float], float]]]] = []
    try:
        for idx, entry in enumerate(entries, 1):
            rgb = load_ppm_rgb(quality_dir / entry["file"])
            preds = fp32_runner.infer(rgb, 0.0)
            gt = entry["person_boxes"]
            matched = match_gt(preds, gt, float(args.fp32_conf), float(args.iou))
            baseline.append((preds, gt, matched))
            if idx <= 3 or idx % 100 == 0 or idx == len(entries):
                print(
                    f"V11_OUTPUT_DRIFT_BASE_PROGRESS image={idx}/{len(entries)} matched={sum(len(x[2]) for x in baseline)}",
                    flush=True,
                )
    finally:
        fp32_runner.close()

    total_gt = sum(len(gt) for _, gt, _ in baseline)
    baseline_tp = sum(len(matched) for _, _, matched in baseline)
    print(
        "V11_OUTPUT_DRIFT_BASELINE "
        f"gt={total_gt} matched={baseline_tp} recall={baseline_tp/max(1,total_gt):.4f}",
        flush=True,
    )

    for label, path in candidates:
        runner = Runner(path)
        retained = lost = gained = score_loss = geometry_loss = 0
        common_ious: list[float] = []
        common_score_delta: list[float] = []
        lost_best_iou: list[float] = []
        lost_best_score: list[float] = []
        candidate_tp = 0
        try:
            for idx, entry in enumerate(entries, 1):
                rgb = load_ppm_rgb(quality_dir / entry["file"])
                candidate_preds = runner.infer(rgb, 0.0)
                _, gt, fp32_matched = baseline[idx - 1]
                cand_matched = match_gt(
                    candidate_preds,
                    gt,
                    float(args.candidate_conf),
                    float(args.iou),
                )
                candidate_tp += len(cand_matched)
                fp32_ids = set(fp32_matched)
                cand_ids = set(cand_matched)
                common = fp32_ids & cand_ids
                fp32_only = fp32_ids - cand_ids
                cand_only = cand_ids - fp32_ids
                retained += len(common)
                lost += len(fp32_only)
                gained += len(cand_only)

                for gt_idx in common:
                    fp_pred, _ = fp32_matched[gt_idx]
                    cand_pred, cand_iou = cand_matched[gt_idx]
                    common_ious.append(cand_iou)
                    common_score_delta.append(float(cand_pred[4]) - float(fp_pred[4]))

                for gt_idx in fp32_only:
                    best_pred, best_iou = best_candidate_for_gt(candidate_preds, gt[gt_idx])
                    lost_best_iou.append(best_iou)
                    best_score = float(best_pred[4]) if best_pred is not None else 0.0
                    lost_best_score.append(best_score)
                    if best_iou >= float(args.iou) and best_score < float(args.candidate_conf):
                        score_loss += 1
                    else:
                        geometry_loss += 1

                if idx <= 3 or idx % 100 == 0 or idx == len(entries):
                    print(
                        "V11_OUTPUT_DRIFT_PROGRESS "
                        f"label={label} image={idx}/{len(entries)} candidate_tp={candidate_tp} "
                        f"lost={lost} score_loss={score_loss} geometry_loss={geometry_loss}",
                        flush=True,
                    )
        finally:
            runner.close()

        if lost != score_loss + geometry_loss:
            raise SystemExit(
                f"V11_OUTPUT_DRIFT FAIL accounting label={label} lost={lost} "
                f"score={score_loss} geometry={geometry_loss}"
            )

        dominant = "score" if score_loss > geometry_loss else "geometry" if geometry_loss > score_loss else "balanced"
        print(
            "V11_OUTPUT_DRIFT_RESULT "
            f"label={label} file={path.name} candidate_tp={candidate_tp} retained={retained} "
            f"lost={lost} gained={gained} score_loss={score_loss} geometry_loss={geometry_loss} "
            f"dominant={dominant} common_iou_p50={pct(common_ious,50):.4f} "
            f"common_iou_p10={pct(common_ious,10):.4f} score_delta_p50={pct(common_score_delta,50):.4f} "
            f"score_delta_p10={pct(common_score_delta,10):.4f} lost_best_iou_p50={pct(lost_best_iou,50):.4f} "
            f"lost_best_score_p50={pct(lost_best_score,50):.4f}",
            flush=True,
        )

    print("V11_OUTPUT_DRIFT_DONE status=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
