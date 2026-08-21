#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1.2: suppress duplicate RF-DETR person boxes without changing detector inference."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/rfdetr_step1/person_detection.json"),
        help="Step 1 JSON report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rfdetr_step1_dedupe"),
    )
    parser.add_argument("--iou", type=float, default=0.62)
    parser.add_argument("--containment", type=float, default=0.90)
    parser.add_argument("--center", type=float, default=0.35)
    return parser.parse_args()


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def _iou(a: list[float], b: list[float]) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _containment(a: list[float], b: list[float]) -> float:
    inter = _intersection(a, b)
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0.0 else 0.0


def _center_distance(a: list[float], b: list[float]) -> float:
    acx = (a[0] + a[2]) * 0.5
    acy = (a[1] + a[3]) * 0.5
    bcx = (b[0] + b[2]) * 0.5
    bcy = (b[1] + b[3]) * 0.5
    aw = max(1.0, a[2] - a[0])
    ah = max(1.0, a[3] - a[1])
    bw = max(1.0, b[2] - b[0])
    bh = max(1.0, b[3] - b[1])
    scale = max(20.0, math.hypot(aw, ah), math.hypot(bw, bh))
    return math.hypot(acx - bcx, acy - bcy) / scale


def _dedupe(persons: list[dict], iou_gate: float, containment_gate: float, center_gate: float):
    ordered = sorted(persons, key=lambda row: float(row["confidence"]), reverse=True)
    kept: list[dict] = []
    rejected: list[dict] = []

    for candidate in ordered:
        cbox = [float(v) for v in candidate["xyxy"]]
        duplicate_of = None
        duplicate_metrics = None
        for existing in kept:
            ebox = [float(v) for v in existing["xyxy"]]
            pair_iou = _iou(cbox, ebox)
            pair_containment = _containment(cbox, ebox)
            pair_center = _center_distance(cbox, ebox)
            duplicate = pair_iou >= iou_gate or (
                pair_containment >= containment_gate and pair_center <= center_gate
            )
            if duplicate:
                duplicate_of = existing
                duplicate_metrics = {
                    "iou": round(pair_iou, 4),
                    "containment": round(pair_containment, 4),
                    "center_distance": round(pair_center, 4),
                }
                break

        if duplicate_of is None:
            kept.append(candidate)
        else:
            rejected.append(
                {
                    **candidate,
                    "duplicate_of_confidence": float(duplicate_of["confidence"]),
                    "duplicate_metrics": duplicate_metrics,
                }
            )

    return kept, rejected


def main() -> int:
    args = _parse_args()
    if not args.report.is_file():
        raise SystemExit(f"STEP1_2_FAIL report_not_found={args.report}")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    persons = list(report.get("persons") or [])
    source_image = Path(str(report.get("source_image") or ""))
    if not source_image.is_file():
        raise SystemExit(f"STEP1_2_FAIL source_image_not_found={source_image}")

    kept, rejected = _dedupe(
        persons,
        float(args.iou),
        float(args.containment),
        float(args.center),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(source_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()

    for index, row in enumerate(kept, start=1):
        x1, y1, x2, y2 = [float(v) for v in row["xyxy"]]
        conf = float(row["confidence"])
        draw.rectangle((x1, y1, x2, y2), outline=(255, 196, 64), width=4)
        label = f"person {index}  {conf:.2f}"
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_h = max(16, text_box[3] - text_box[1] + 6)
        text_w = max(40, text_box[2] - text_box[0] + 8)
        label_y = max(0.0, y1 - text_h)
        draw.rectangle((x1, label_y, x1 + text_w, label_y + text_h), fill=(255, 196, 64))
        draw.text((x1 + 4, label_y + 3), label, fill=(15, 18, 22), font=font)

    image_out = args.output_dir / "person_detection_deduped.jpg"
    json_out = args.output_dir / "person_detection_deduped.json"
    image.save(image_out, quality=95, subsampling=0)

    output = {
        "stage": "1.2",
        "source_report": str(args.report),
        "source_image": str(source_image),
        "gates": {
            "iou": float(args.iou),
            "containment": float(args.containment),
            "center_distance": float(args.center),
        },
        "input_persons": len(persons),
        "kept_persons": len(kept),
        "rejected_duplicates": len(rejected),
        "persons": kept,
        "rejected": rejected,
    }
    json_out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    rejected_text = " ".join(
        f"conf={row['confidence']:.2f}->keep={row['duplicate_of_confidence']:.2f}"
        f"/iou={row['duplicate_metrics']['iou']:.3f}"
        f"/contain={row['duplicate_metrics']['containment']:.3f}"
        f"/center={row['duplicate_metrics']['center_distance']:.3f}"
        for row in rejected
    ) or "none"
    print(
        "STEP1_2_RESULT "
        f"input={len(persons)} kept={len(kept)} rejected={len(rejected)} "
        f"duplicates=[{rejected_text}]",
        flush=True,
    )
    print(f"STEP1_2_IMAGE={image_out}", flush=True)
    print(f"STEP1_2_JSON={json_out}", flush=True)
    print("STEP1_2_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
