#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_camera_v11_trt86_person_quality import (
    Runner,
    load_ppm_rgb,
    metrics,
    resolve,
    score_image,
)


def frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("step must be > 0")
    values: list[float] = []
    value = start
    while value <= stop + 1e-9:
        values.append(round(value, 6))
        value += step
    return values


def evaluate_engine(
    runner: Runner,
    entries: list[dict],
    base: Path,
    conf: float,
    iou_gate: float,
) -> tuple[int, int, int, float, float, float]:
    tp = fp = fn = 0
    for idx, entry in enumerate(entries, 1):
        rgb = load_ppm_rgb(base / entry["file"])
        preds = runner.infer(rgb, conf)
        a, b, c = score_image(preds, entry["person_boxes"], iou_gate)
        tp += a
        fp += b
        fn += c
        if idx <= 3 or idx % 100 == 0 or idx == len(entries):
            print(
                f"V11_CONF_SWEEP_PROGRESS engine={runner.path.name} conf={conf:.3f} "
                f"image={idx}/{len(entries)} tp={tp} fp={fp} fn={fn}",
                flush=True,
            )
    precision, recall, f1 = metrics(tp, fp, fn)
    return tp, fp, fn, precision, recall, f1


def cache_predictions(
    runner: Runner,
    entries: list[dict],
    base: Path,
    min_conf: float,
) -> list[tuple[list[list[float]], list[list[float]]]]:
    cached: list[tuple[list[list[float]], list[list[float]]]] = []
    for idx, entry in enumerate(entries, 1):
        rgb = load_ppm_rgb(base / entry["file"])
        preds = runner.infer(rgb, min_conf)
        cached.append((preds, entry["person_boxes"]))
        if idx <= 3 or idx % 100 == 0 or idx == len(entries):
            print(
                f"V11_CONF_SWEEP_CACHE image={idx}/{len(entries)} preds={len(preds)} min_conf={min_conf:.3f}",
                flush=True,
            )
    return cached


def score_cached(
    cached: list[tuple[list[list[float]], list[list[float]]]],
    conf: float,
    iou_gate: float,
) -> tuple[int, int, int, float, float, float]:
    tp = fp = fn = 0
    for preds, gt in cached:
        filtered = [row for row in preds if row[4] >= conf]
        a, b, c = score_image(filtered, gt, iou_gate)
        tp += a
        fp += b
        fn += c
    precision, recall, f1 = metrics(tp, fp, fn)
    return tp, fp, fn, precision, recall, f1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diagnostic confidence sweep for V11 mixed-precision person detector"
    )
    ap.add_argument(
        "--fp32",
        default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine",
    )
    ap.add_argument(
        "--candidate",
        default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-mixed-headfp32-trt86.engine",
    )
    ap.add_argument("--quality-dir", default="artifacts/yolo26s_trt86/person_quality_b1")
    ap.add_argument("--fp32-conf", type=float, default=0.18)
    ap.add_argument("--min-conf", type=float, default=0.10)
    ap.add_argument("--max-conf", type=float, default=0.18)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--max-drop", type=float, default=0.03)
    args = ap.parse_args()

    fp32_path = resolve(args.fp32)
    candidate_path = resolve(args.candidate)
    quality = resolve(args.quality_dir)
    manifest_path = quality / "person_gt.json"
    for path in (fp32_path, candidate_path, manifest_path):
        if not path.is_file():
            raise SystemExit(f"V11_CONF_SWEEP FAIL missing={path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("images") or []
    if len(entries) < 100:
        raise SystemExit(f"V11_CONF_SWEEP FAIL images={len(entries)} expected>=100")

    thresholds = frange(float(args.min_conf), float(args.max_conf), float(args.step))
    if not thresholds:
        raise SystemExit("V11_CONF_SWEEP FAIL no thresholds")

    print(
        "V11_CONF_SWEEP_START "
        f"images={len(entries)} fp32={fp32_path.name} candidate={candidate_path.name} "
        f"fp32_conf={args.fp32_conf:.3f} range={thresholds[0]:.3f}..{thresholds[-1]:.3f} "
        f"step={args.step:.3f} iou={args.iou:.2f} max_drop={args.max_drop:.3f} diagnostic=1",
        flush=True,
    )

    fp32_runner = Runner(fp32_path)
    try:
        fp32 = evaluate_engine(
            fp32_runner,
            entries,
            quality,
            float(args.fp32_conf),
            float(args.iou),
        )
    finally:
        fp32_runner.close()

    print(
        "V11_CONF_SWEEP_BASELINE "
        f"conf={args.fp32_conf:.3f} tp={fp32[0]} fp={fp32[1]} fn={fp32[2]} "
        f"precision={fp32[3]:.4f} recall={fp32[4]:.4f} f1={fp32[5]:.4f}",
        flush=True,
    )

    candidate_runner = Runner(candidate_path)
    try:
        cached = cache_predictions(candidate_runner, entries, quality, thresholds[0])
    finally:
        candidate_runner.close()

    limit = max(0.0, float(args.max_drop))
    passing: list[tuple[float, tuple[int, int, int, float, float, float], float, float, float]] = []
    all_rows: list[tuple[float, tuple[int, int, int, float, float, float], float, float, float]] = []
    for conf in thresholds:
        result = score_cached(cached, conf, float(args.iou))
        p_drop = fp32[3] - result[3]
        r_drop = fp32[4] - result[4]
        f1_drop = fp32[5] - result[5]
        row = (conf, result, p_drop, r_drop, f1_drop)
        all_rows.append(row)
        passed = p_drop <= limit and r_drop <= limit and f1_drop <= limit
        if passed:
            passing.append(row)
        print(
            "V11_CONF_SWEEP_RESULT "
            f"conf={conf:.3f} tp={result[0]} fp={result[1]} fn={result[2]} "
            f"precision={result[3]:.4f} recall={result[4]:.4f} f1={result[5]:.4f} "
            f"precision_drop={p_drop:.4f} recall_drop={r_drop:.4f} f1_drop={f1_drop:.4f} "
            f"gate={'PASS' if passed else 'FAIL'}",
            flush=True,
        )

    if passing:
        best = max(passing, key=lambda row: (row[1][5], row[0]))
        print(
            "V11_CONF_SWEEP_BEST status=PASS diagnostic=1 "
            f"conf={best[0]:.3f} precision={best[1][3]:.4f} recall={best[1][4]:.4f} f1={best[1][5]:.4f} "
            f"precision_drop={best[2]:.4f} recall_drop={best[3]:.4f} f1_drop={best[4]:.4f}",
            flush=True,
        )
        return 0

    best = max(all_rows, key=lambda row: row[1][5])
    print(
        "V11_CONF_SWEEP_BEST status=FAIL diagnostic=1 "
        f"conf={best[0]:.3f} precision={best[1][3]:.4f} recall={best[1][4]:.4f} f1={best[1][5]:.4f} "
        f"precision_drop={best[2]:.4f} recall_drop={best[3]:.4f} f1_drop={best[4]:.4f}",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
