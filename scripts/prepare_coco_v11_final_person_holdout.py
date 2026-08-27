#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TARGET_W = 672
TARGET_H = 384
PERSON_CATEGORY_ID = 1
VAL_SENTINEL = "000000000139.jpg"


def resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def find_image_root(data_root: Path) -> Path:
    matches = list(data_root.rglob(VAL_SENTINEL))
    if not matches:
        raise SystemExit(f"V11_FINAL_HOLDOUT FAIL sentinel_missing={VAL_SENTINEL} root={data_root}")
    matches.sort(key=lambda p: len(p.parts))
    return matches[0].parent


def letterbox_rgb(img: Image.Image) -> tuple[Image.Image, float, int, int]:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(TARGET_W / w, TARGET_H / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
    left = (TARGET_W - nw) // 2
    top = (TARGET_H - nh) // 2
    out = Image.new("RGB", (TARGET_W, TARGET_H), (114, 114, 114))
    out.paste(resized, (left, top))
    return out, scale, left, top


def transform_box(box: list[float], scale: float, left: int, top: int) -> list[float]:
    x, y, w, h = (float(v) for v in box)
    x1 = max(0.0, min(TARGET_W - 1.0, x * scale + left))
    y1 = max(0.0, min(TARGET_H - 1.0, y * scale + top))
    x2 = max(0.0, min(TARGET_W - 1.0, (x + w) * scale + left))
    y2 = max(0.0, min(TARGET_H - 1.0, (y + h) * scale + top))
    return [x1, y1, x2, y2]


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare a final COCO person holdout disjoint from tuning and calibration")
    ap.add_argument("--work-dir", default="artifacts/coco2017_v11")
    ap.add_argument("--quality-dir", default="artifacts/yolo26s_trt86/person_quality_b1")
    ap.add_argument("--person-calibration-dir", default="artifacts/yolo26s_trt86/int8_calibration_person_b1")
    ap.add_argument("--output-dir", default="artifacts/yolo26s_trt86/person_final_holdout_b1")
    ap.add_argument("--max-images", type=int, default=0, help="0 means use every remaining person image")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    work = resolve(args.work_dir)
    data_root = work / "data"
    ann_path = data_root / "annotations" / "instances_val2017.json"
    if not ann_path.is_file():
        raise SystemExit(f"V11_FINAL_HOLDOUT FAIL annotations_missing={ann_path}")
    image_root = find_image_root(data_root)

    quality_manifest = resolve(args.quality_dir) / "person_gt.json"
    calibration_manifest = resolve(args.person_calibration_dir) / "manifest.json"
    for path in (quality_manifest, calibration_manifest):
        if not path.is_file():
            raise SystemExit(f"V11_FINAL_HOLDOUT FAIL missing_manifest={path}")

    quality_payload = json.loads(quality_manifest.read_text(encoding="utf-8"))
    calibration_payload = json.loads(calibration_manifest.read_text(encoding="utf-8"))
    quality_ids = {int(row["image_id"]) for row in quality_payload.get("images", [])}
    calibration_ids = {int(v) for v in calibration_payload.get("image_ids", [])}
    if quality_ids & calibration_ids:
        raise SystemExit(
            f"V11_FINAL_HOLDOUT FAIL quality_calibration_overlap={len(quality_ids & calibration_ids)}"
        )

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in payload["images"]}
    person_boxes: dict[int, list[list[float]]] = {}
    for row in payload["annotations"]:
        if int(row.get("category_id", -1)) != PERSON_CATEGORY_ID or int(row.get("iscrowd", 0)):
            continue
        box = row.get("bbox")
        if not box or float(box[2]) <= 1.0 or float(box[3]) <= 1.0:
            continue
        image_id = int(row["image_id"])
        person_boxes.setdefault(image_id, []).append([float(v) for v in box])

    excluded = quality_ids | calibration_ids
    remaining = sorted(image_id for image_id in person_boxes if image_id not in excluded)
    if int(args.max_images) > 0:
        remaining = remaining[: int(args.max_images)]
    if len(remaining) < 100:
        raise SystemExit(
            f"V11_FINAL_HOLDOUT FAIL remaining_person_images={len(remaining)} expected>=100 "
            f"quality={len(quality_ids)} calibration={len(calibration_ids)}"
        )

    out_dir = resolve(args.output_dir)
    if args.reset:
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    gt_instances = 0
    print(
        "V11_FINAL_HOLDOUT_START "
        f"quality_excluded={len(quality_ids)} calibration_excluded={len(calibration_ids)} "
        f"selected={len(remaining)} geometry={TARGET_W}x{TARGET_H} letterbox114=1 image_root={image_root}",
        flush=True,
    )

    for idx, image_id in enumerate(remaining, 1):
        row = images[image_id]
        src = image_root / row["file_name"]
        if not src.is_file():
            raise SystemExit(f"V11_FINAL_HOLDOUT FAIL missing_image={src}")
        with Image.open(src) as img:
            out, scale, left, top = letterbox_rgb(img)
            dst = out_dir / f"{image_id:012d}.ppm"
            out.save(dst, format="PPM")
        boxes = [transform_box(box, scale, left, top) for box in person_boxes[image_id]]
        boxes = [box for box in boxes if box[2] > box[0] and box[3] > box[1]]
        gt_instances += len(boxes)
        entries.append({"image_id": image_id, "file": dst.name, "person_boxes": boxes})
        if idx <= 5 or idx % 100 == 0 or idx == len(remaining):
            print(
                f"V11_FINAL_HOLDOUT_PROGRESS image={idx}/{len(remaining)} image_id={image_id} persons={len(boxes)}",
                flush=True,
            )

    manifest = out_dir / "person_gt.json"
    manifest.write_text(
        json.dumps(
            {
                "geometry": [TARGET_H, TARGET_W],
                "letterbox_fill": 114,
                "iou_gate": 0.5,
                "source": "coco-val2017-final-disjoint",
                "excluded_quality_ids": len(quality_ids),
                "excluded_calibration_ids": len(calibration_ids),
                "images": entries,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    selected_ids = {int(row["image_id"]) for row in entries}
    overlap_quality = len(selected_ids & quality_ids)
    overlap_calibration = len(selected_ids & calibration_ids)
    if overlap_quality or overlap_calibration:
        raise SystemExit(
            f"V11_FINAL_HOLDOUT FAIL overlap_quality={overlap_quality} overlap_calibration={overlap_calibration}"
        )

    print(
        "V11_FINAL_HOLDOUT_RESULT "
        f"status=PASS images={len(entries)} person_instances={gt_instances} "
        f"overlap_quality=0 overlap_calibration=0 output_dir={out_dir} manifest={manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
