#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TARGET_W = 672
TARGET_H = 384
PERSON_CATEGORY_ID = 1
VAL_SENTINEL = "000000000139.jpg"


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def find_image_root(data_root: Path) -> Path:
    matches = list(data_root.rglob(VAL_SENTINEL))
    if not matches:
        raise SystemExit(f"V11_PERSON_CALIB FAIL sentinel_missing={VAL_SENTINEL} root={data_root}")
    if len(matches) > 1:
        matches.sort(key=lambda p: len(p.parts))
    return matches[0].parent


def letterbox_rgb(img: Image.Image) -> Image.Image:
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
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare held-out-safe person-focused COCO calibration set")
    ap.add_argument("--work-dir", default="artifacts/coco2017_v11")
    ap.add_argument("--quality-dir", default="artifacts/yolo26s_trt86/person_quality_b1")
    ap.add_argument("--calibration-dir", default="artifacts/yolo26s_trt86/int8_calibration_person_b1")
    ap.add_argument("--calibration-images", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    work = resolve(args.work_dir)
    data_root = work / "data"
    ann_path = data_root / "annotations" / "instances_val2017.json"
    if not ann_path.is_file():
        raise SystemExit(f"V11_PERSON_CALIB FAIL annotations_missing={ann_path}")

    image_root = find_image_root(data_root)
    quality_dir = resolve(args.quality_dir)
    quality_manifest = quality_dir / "person_gt.json"
    if not quality_manifest.is_file():
        raise SystemExit(f"V11_PERSON_CALIB FAIL quality_manifest_missing={quality_manifest}")

    quality_payload = json.loads(quality_manifest.read_text(encoding="utf-8"))
    held_out = {int(row["image_id"]) for row in quality_payload.get("images", [])}
    if len(held_out) < 100:
        raise SystemExit(f"V11_PERSON_CALIB FAIL held_out={len(held_out)} expected>=100")

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in payload["images"]}
    person_ids: set[int] = set()
    for row in payload["annotations"]:
        if int(row.get("category_id", -1)) != PERSON_CATEGORY_ID:
            continue
        if int(row.get("iscrowd", 0)):
            continue
        box = row.get("bbox")
        if not box or float(box[2]) <= 1.0 or float(box[3]) <= 1.0:
            continue
        image_id = int(row["image_id"])
        if image_id not in held_out:
            person_ids.add(image_id)

    candidates = sorted(person_ids)
    rng = random.Random(int(args.seed))
    rng.shuffle(candidates)
    requested = max(500, int(args.calibration_images))
    selected = candidates[: min(requested, len(candidates))]
    if len(selected) < 1000:
        raise SystemExit(
            f"V11_PERSON_CALIB FAIL available_person_images={len(candidates)} selected={len(selected)} expected>=1000"
        )

    calib_root = resolve(args.calibration_dir)
    shard = calib_root / "CAM-COCO-PERSON"
    if args.reset:
        shutil.rmtree(calib_root, ignore_errors=True)
    shard.mkdir(parents=True, exist_ok=True)

    print(
        "V11_PERSON_CALIB_START "
        f"person_candidates={len(candidates)} held_out_quality={len(held_out)} selected={len(selected)} "
        f"geometry={TARGET_W}x{TARGET_H} letterbox114=1 seed={args.seed} image_root={image_root}",
        flush=True,
    )

    for idx, image_id in enumerate(selected, 1):
        row = images[image_id]
        src = image_root / row["file_name"]
        if not src.is_file():
            raise SystemExit(f"V11_PERSON_CALIB FAIL missing_image={src}")
        dst = shard / f"{image_id:012d}.ppm"
        with Image.open(src) as img:
            letterbox_rgb(img).save(dst, format="PPM")
        if idx <= 5 or idx % 100 == 0 or idx == len(selected):
            print(f"V11_PERSON_CALIB_PROGRESS frame={idx}/{len(selected)} image_id={image_id}", flush=True)

    manifest = calib_root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "mode": "person-focused",
                "seed": int(args.seed),
                "geometry": [TARGET_H, TARGET_W],
                "held_out_quality_images": len(held_out),
                "image_ids": selected,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    print(
        "V11_PERSON_CALIB_RESULT "
        f"status=PASS images={len(selected)} held_out_quality={len(held_out)} "
        f"calibration_dir={calib_root} manifest={manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
