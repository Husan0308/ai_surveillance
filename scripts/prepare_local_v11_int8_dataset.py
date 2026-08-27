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
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def letterbox_rgb(src: Path) -> Image.Image:
    with Image.open(src) as img:
        img = img.convert('RGB')
        w, h = img.size
        scale = min(TARGET_W / w, TARGET_H / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
        left = (TARGET_W - nw) // 2
        top = (TARGET_H - nh) // 2
        out = Image.new('RGB', (TARGET_W, TARGET_H), (114, 114, 114))
        out.paste(resized, (left, top))
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description='Prepare local images for V11 TRT8.6 INT8 calibration and held-out agreement gate')
    ap.add_argument('--source-dir', required=True)
    ap.add_argument('--calibration-dir', default='artifacts/yolo26s_trt86/int8_calibration_b1')
    ap.add_argument('--quality-dir', default='artifacts/yolo26s_trt86/person_quality_b1')
    ap.add_argument('--calibration-images', type=int, default=1000)
    ap.add_argument('--quality-images', type=int, default=300)
    ap.add_argument('--seed', type=int, default=26)
    ap.add_argument('--reset', action='store_true')
    args = ap.parse_args()

    source = resolve(args.source_dir)
    calib = resolve(args.calibration_dir)
    quality = resolve(args.quality_dir)
    if not source.is_dir():
        raise SystemExit(f'V11_LOCAL_PREP FAIL source_missing={source}')

    files = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
    if len(files) < 600:
        raise SystemExit(f'V11_LOCAL_PREP FAIL images={len(files)} expected>=600')

    rng = random.Random(int(args.seed))
    rng.shuffle(files)
    n_calib = min(max(500, int(args.calibration_images)), len(files) - 100)
    remaining = len(files) - n_calib
    n_quality = min(max(100, int(args.quality_images)), remaining)
    calib_files = files[:n_calib]
    quality_files = files[n_calib:n_calib + n_quality]

    if args.reset:
        shutil.rmtree(calib, ignore_errors=True)
        shutil.rmtree(quality, ignore_errors=True)
    calib.mkdir(parents=True, exist_ok=True)
    quality.mkdir(parents=True, exist_ok=True)

    print(
        'V11_LOCAL_PREP_START '
        f'source={source} total_images={len(files)} calibration={len(calib_files)} '
        f'quality={len(quality_files)} geometry={TARGET_W}x{TARGET_H} letterbox114=1 seed={args.seed}',
        flush=True,
    )

    for idx, src in enumerate(calib_files, 1):
        dst = calib / f'calib_{idx:05d}.ppm'
        letterbox_rgb(src).save(dst, format='PPM')
        if idx <= 5 or idx % 100 == 0 or idx == len(calib_files):
            print(f'V11_LOCAL_CALIB frame={idx}/{len(calib_files)} src={src.name}', flush=True)

    manifest_rows = []
    for idx, src in enumerate(quality_files, 1):
        dst = quality / f'quality_{idx:05d}.ppm'
        letterbox_rgb(src).save(dst, format='PPM')
        manifest_rows.append({'file': dst.name, 'source': str(src)})
        if idx <= 5 or idx % 100 == 0 or idx == len(quality_files):
            print(f'V11_LOCAL_QUALITY frame={idx}/{len(quality_files)} src={src.name}', flush=True)

    manifest = quality / 'agreement_manifest.json'
    manifest.write_text(
        json.dumps(
            {
                'geometry': [TARGET_H, TARGET_W],
                'letterbox_fill': 114,
                'source_dir': str(source),
                'images': manifest_rows,
            },
            separators=(',', ':'),
        ),
        encoding='utf-8',
    )

    print(
        'V11_LOCAL_PREP_RESULT '
        f'status=PASS total={len(files)} calibration={len(calib_files)} quality={len(quality_files)} '
        f'calibration_dir={calib} quality_dir={quality} manifest={manifest}',
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
