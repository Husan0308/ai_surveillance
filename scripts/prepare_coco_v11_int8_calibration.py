#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VAL_URL = "https://huggingface.co/datasets/baohao/coco2017/resolve/main/coco_val2017.zip?download=true"
VAL_SHA256 = "6cb37504e5c5106585bfb82b109de3a321f406892ec06c1c547d39310a26bf92"
ANN_URL = "https://huggingface.co/datasets/LibreYOLO/coco2017/resolve/main/instances_val2017.json?download=true"
ANN_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
TARGET_W = 672
TARGET_H = 384
PERSON_CATEGORY_ID = 1
VAL_SENTINEL = "000000000139.jpg"


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_verified(url: str, dst: Path, expected_sha256: str) -> None:
    if dst.is_file() and dst.stat().st_size > 0:
        actual = sha256_file(dst)
        if actual == expected_sha256:
            print(
                f"V11_COCO_DOWNLOAD status=HIT file={dst} bytes={dst.stat().st_size} sha256={actual}",
                flush=True,
            )
            return
        print(
            f"V11_COCO_DOWNLOAD status=STALE file={dst} sha256={actual} expected={expected_sha256}",
            flush=True,
        )
        dst.unlink()

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    print(f"V11_COCO_DOWNLOAD status=START url={url} file={dst}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ai-surveillance-v11-coco-prep/1.0"})
    h = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=60) as src, tmp.open("wb") as out:
        while True:
            chunk = src.read(8 << 20)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)

    actual = h.hexdigest()
    if actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"V11_COCO_DOWNLOAD checksum mismatch file={dst.name} sha256={actual} expected={expected_sha256}"
        )
    tmp.replace(dst)
    print(
        f"V11_COCO_DOWNLOAD status=DONE file={dst} bytes={dst.stat().st_size} sha256={actual}",
        flush=True,
    )


def discover_val_dir(root: Path) -> Path | None:
    matches = list(root.rglob(VAL_SENTINEL)) if root.exists() else []
    if not matches:
        return None
    parents = sorted({p.parent for p in matches}, key=lambda p: str(p))
    if len(parents) == 1:
        return parents[0]
    # Defensive mirror handling: choose the candidate containing the most jpg files.
    ranked = sorted(
        ((sum(1 for _ in parent.glob("*.jpg")), parent) for parent in parents),
        key=lambda row: row[0],
        reverse=True,
    )
    return ranked[0][1]


def ensure_extracted_val_dir(archive: Path, dst: Path) -> Path:
    existing = discover_val_dir(dst)
    if existing is not None:
        print(f"V11_COCO_EXTRACT status=HIT image_root={existing}", flush=True)
        return existing

    print(f"V11_COCO_EXTRACT status=START archive={archive}", flush=True)
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"V11_COCO_EXTRACT corrupt_member={bad}")
        zf.extractall(dst)
    print(f"V11_COCO_EXTRACT status=DONE archive={archive}", flush=True)

    discovered = discover_val_dir(dst)
    if discovered is None:
        sample_roots = sorted({str(p.parent) for p in dst.rglob("*.jpg")})[:10]
        raise RuntimeError(
            "V11_COCO_EXTRACT could not discover COCO val image root "
            f"sentinel={VAL_SENTINEL} sample_roots={sample_roots}"
        )
    print(f"V11_COCO_EXTRACT status=DISCOVERED image_root={discovered}", flush=True)
    return discovered


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


def save_ppm(src: Path, dst: Path) -> tuple[float, int, int]:
    with Image.open(src) as img:
        out, scale, left, top = letterbox_rgb(img)
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst, format="PPM")
    return scale, left, top


def transform_box(box: list[float], scale: float, left: int, top: int) -> list[float]:
    x, y, w, h = (float(v) for v in box)
    x1 = max(0.0, min(TARGET_W - 1.0, x * scale + left))
    y1 = max(0.0, min(TARGET_H - 1.0, y * scale + top))
    x2 = max(0.0, min(TARGET_W - 1.0, (x + w) * scale + left))
    y2 = max(0.0, min(TARGET_H - 1.0, (y + h) * scale + top))
    return [x1, y1, x2, y2]


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare COCO val2017 for V11 TRT8.6 INT8 calibration and person quality gate")
    ap.add_argument("--work-dir", default="artifacts/coco2017_v11")
    ap.add_argument("--calibration-dir", default="artifacts/yolo26s_trt86/int8_calibration_b1")
    ap.add_argument("--quality-dir", default="artifacts/yolo26s_trt86/person_quality_b1")
    ap.add_argument("--calibration-images", type=int, default=1000)
    ap.add_argument("--quality-images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    work = resolve(args.work_dir)
    calib_root = resolve(args.calibration_dir)
    calib = calib_root / "CAM-COCO"
    quality = resolve(args.quality_dir)
    val_zip = work / "coco_val2017.zip"
    ann_download = work / "instances_val2017.json"
    data_root = work / "data"
    ann_path = data_root / "annotations" / "instances_val2017.json"

    download_verified(VAL_URL, val_zip, VAL_SHA256)
    download_verified(ANN_URL, ann_download, ANN_SHA256)
    val_dir = ensure_extracted_val_dir(val_zip, data_root)
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    if not ann_path.is_file() or sha256_file(ann_path) != ANN_SHA256:
        shutil.copy2(ann_download, ann_path)
    if sha256_file(ann_path) != ANN_SHA256:
        raise RuntimeError("V11_COCO_PREP annotations checksum mismatch after install")

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in payload["images"]}
    person_boxes: dict[int, list[list[float]]] = {}
    for row in payload["annotations"]:
        if int(row.get("category_id", -1)) != PERSON_CATEGORY_ID or int(row.get("iscrowd", 0)):
            continue
        box = row.get("bbox")
        if not box or float(box[2]) <= 1.0 or float(box[3]) <= 1.0:
            continue
        person_boxes.setdefault(int(row["image_id"]), []).append([float(v) for v in box])

    all_ids = sorted(images)
    rng = random.Random(int(args.seed))
    rng.shuffle(all_ids)
    n_calib = max(500, min(len(all_ids), int(args.calibration_images)))
    calib_ids = all_ids[:n_calib]
    remaining = set(all_ids[n_calib:])
    person_ids = [image_id for image_id in sorted(person_boxes) if image_id in remaining]
    rng.shuffle(person_ids)
    n_quality = min(len(person_ids), max(100, int(args.quality_images)))
    quality_ids = person_ids[:n_quality]

    if len(calib_ids) < 500:
        raise SystemExit(f"V11_COCO_PREP FAIL calibration_images={len(calib_ids)} expected>=500")
    if len(quality_ids) < 100:
        raise SystemExit(f"V11_COCO_PREP FAIL quality_person_images={len(quality_ids)} expected>=100")

    if args.reset:
        shutil.rmtree(calib_root, ignore_errors=True)
        shutil.rmtree(quality, ignore_errors=True)
    calib.mkdir(parents=True, exist_ok=True)
    quality.mkdir(parents=True, exist_ok=True)

    print(
        "V11_COCO_PREP_START "
        f"val_images={len(images)} calibration={len(calib_ids)} quality_person={len(quality_ids)} "
        f"geometry={TARGET_W}x{TARGET_H} letterbox114=1 seed={args.seed} calibration_shard=CAM-COCO "
        f"source=hf-verified image_root={val_dir}",
        flush=True,
    )

    for idx, image_id in enumerate(calib_ids, 1):
        row = images[image_id]
        src = val_dir / row["file_name"]
        if not src.is_file():
            raise RuntimeError(f"V11_COCO_PREP missing_image={src}")
        dst = calib / f"{image_id:012d}.ppm"
        save_ppm(src, dst)
        if idx <= 5 or idx % 100 == 0 or idx == len(calib_ids):
            print(f"V11_COCO_CALIB frame={idx}/{len(calib_ids)} image_id={image_id}", flush=True)

    quality_manifest: list[dict] = []
    gt_instances = 0
    for idx, image_id in enumerate(quality_ids, 1):
        row = images[image_id]
        src = val_dir / row["file_name"]
        if not src.is_file():
            raise RuntimeError(f"V11_COCO_PREP missing_quality_image={src}")
        dst = quality / f"{image_id:012d}.ppm"
        scale, left, top = save_ppm(src, dst)
        boxes = [transform_box(box, scale, left, top) for box in person_boxes[image_id]]
        boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
        gt_instances += len(boxes)
        quality_manifest.append({"image_id": image_id, "file": dst.name, "person_boxes": boxes})
        if idx <= 5 or idx % 100 == 0 or idx == len(quality_ids):
            print(f"V11_COCO_QUALITY frame={idx}/{len(quality_ids)} image_id={image_id} persons={len(boxes)}", flush=True)

    manifest = quality / "person_gt.json"
    manifest.write_text(
        json.dumps(
            {
                "geometry": [TARGET_H, TARGET_W],
                "letterbox_fill": 114,
                "iou_gate": 0.5,
                "images": quality_manifest,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    print(
        "V11_COCO_PREP_RESULT "
        f"status=PASS calibration={len(calib_ids)} quality_images={len(quality_ids)} "
        f"quality_person_instances={gt_instances} calibration_dir={calib_root} quality_dir={quality} manifest={manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
