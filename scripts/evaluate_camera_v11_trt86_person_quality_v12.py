#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from evaluate_camera_v11_trt86_person_quality import Runner, evaluate, metrics, resolve


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One identical FP32/FP16/INT8 TensorRT 8.6 person-quality gate"
    )
    parser.add_argument(
        "--fp32",
        default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine",
    )
    parser.add_argument(
        "--fp16",
        default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp16-trt86.engine",
    )
    parser.add_argument(
        "--int8",
        default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-int8-trt86.engine",
    )
    parser.add_argument(
        "--quality-dir",
        default="artifacts/yolo26s_trt86/person_final_holdout_b1",
    )
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-drop", type=float, default=0.03)
    args = parser.parse_args()

    engines = [
        ("fp32", resolve(args.fp32)),
        ("fp16", resolve(args.fp16)),
        ("int8", resolve(args.int8)),
    ]
    quality = resolve(args.quality_dir)
    manifest = quality / "person_gt.json"
    for _label, path in engines:
        if not path.is_file():
            raise SystemExit(f"V11_PERSON_QUALITY_V12 FAIL missing={path}")
    if not manifest.is_file():
        raise SystemExit(f"V11_PERSON_QUALITY_V12 FAIL missing={manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries = payload.get("images") or []
    if len(entries) < 100:
        raise SystemExit(f"V11_PERSON_QUALITY_V12 FAIL images={len(entries)} expected>=100")

    results: dict[str, tuple[int, int, int, float, float, float]] = {}
    for label, path in engines:
        runner = Runner(path)
        try:
            tp, fp, fn = evaluate(runner, entries, quality, float(args.conf), float(args.iou))
        finally:
            runner.close()
        precision, recall, f1 = metrics(tp, fp, fn)
        results[label] = (tp, fp, fn, precision, recall, f1)
        print(
            "V11_PERSON_QUALITY_V12_RESULT "
            f"engine={label} file={path.name} images={len(entries)} conf={args.conf:.2f} "
            f"iou={args.iou:.2f} tp={tp} fp={fp} fn={fn} precision={precision:.4f} "
            f"recall={recall:.4f} f1={f1:.4f}",
            flush=True,
        )

    baseline = results["fp32"]
    limit = max(0.0, float(args.max_drop))
    failed = False
    for label in ("fp16", "int8"):
        candidate = results[label]
        drops = (baseline[3] - candidate[3], baseline[4] - candidate[4], baseline[5] - candidate[5])
        reasons = []
        for metric, drop in zip(("precision", "recall", "f1"), drops):
            if drop > limit:
                reasons.append(f"{metric}_drop={drop:.4f}")
        failed = failed or bool(reasons)
        print(
            "V11_PERSON_QUALITY_V12_GATE "
            f"candidate={label} status={'FAIL' if reasons else 'PASS'} max_drop={limit:.4f} "
            f"precision_drop={drops[0]:.4f} recall_drop={drops[1]:.4f} "
            f"f1_drop={drops[2]:.4f} reasons={';'.join(reasons) if reasons else 'none'}",
            flush=True,
        )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
