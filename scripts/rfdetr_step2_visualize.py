#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Step 2 RF-DETR sequence detections without rerunning inference."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/rfdetr_step2_sequence/sequence_report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rfdetr_step2_sequence/visual"),
    )
    parser.add_argument("--thumb-width", type=int, default=720)
    return parser.parse_args()


def _font(size: int = 18):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _annotate(image: Image.Image, frame: dict) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font(18)
    persons = list(frame.get("persons") or [])

    for index, row in enumerate(persons, start=1):
        x1, y1, x2, y2 = [float(v) for v in row["xyxy"]]
        conf = float(row["confidence"])
        draw.rectangle((x1, y1, x2, y2), outline=(255, 196, 64), width=5)
        label = f"person {index}  {conf:.2f}"
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = max(60, bbox[2] - bbox[0] + 10)
        text_h = max(22, bbox[3] - bbox[1] + 8)
        label_y = max(0.0, y1 - text_h)
        draw.rectangle((x1, label_y, x1 + text_w, label_y + text_h), fill=(255, 196, 64))
        draw.text((x1 + 5, label_y + 4), label, fill=(15, 18, 22), font=font)

    header = (
        f"frame {int(frame.get('index', 0)):02d} | "
        f"unique={int(frame.get('unique_persons', len(persons)))} | "
        f"raw={int(frame.get('raw_persons', 0))} | "
        f"dup={int(frame.get('duplicates', 0))} | "
        f"infer={float(frame.get('infer_ms', 0.0)):.1f}ms"
    )
    hb = draw.textbbox((0, 0), header, font=font)
    hw = hb[2] - hb[0] + 16
    hh = hb[3] - hb[1] + 12
    draw.rectangle((0, 0, hw, hh), fill=(20, 22, 26))
    draw.text((8, 6), header, fill=(245, 245, 245), font=font)
    return out


def main() -> int:
    args = _parse_args()
    if not args.report.is_file():
        raise SystemExit(f"STEP2_VIS_FAIL report_not_found={args.report}")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    frames = list(report.get("frames") or [])
    if not frames:
        raise SystemExit("STEP2_VIS_FAIL no_frames_in_report")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.output_dir / "frames"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    tiles: list[Image.Image] = []
    thumb_w = max(320, int(args.thumb_width))
    tile_gap = 12
    caption_h = 0

    for frame in frames:
        source = Path(str(frame.get("image") or ""))
        if not source.is_file():
            raise SystemExit(f"STEP2_VIS_FAIL frame_not_found={source}")
        image = Image.open(source).convert("RGB")
        annotated = _annotate(image, frame)

        index = int(frame.get("index", len(tiles) + 1))
        out_path = annotated_dir / f"frame_{index:03d}.jpg"
        annotated.save(out_path, quality=95, subsampling=0)

        ratio = thumb_w / float(annotated.width)
        thumb_h = max(1, int(round(annotated.height * ratio)))
        thumb = annotated.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        thumb = ImageOps.expand(thumb, border=2, fill=(70, 70, 70))
        tiles.append(thumb)

    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    sheet_w = columns * tile_w + (columns - 1) * tile_gap
    sheet_h = rows * tile_h + (rows - 1) * tile_gap + caption_h
    sheet = Image.new("RGB", (sheet_w, sheet_h), (8, 10, 14))

    for idx, tile in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        x = col * (tile_w + tile_gap)
        y = row * (tile_h + tile_gap)
        sheet.paste(tile, (x, y))

    contact = args.output_dir / "contact_sheet.jpg"
    sheet.save(contact, quality=94, subsampling=0)

    summary = report.get("summary") or {}
    print(
        "STEP2_VIS_RESULT "
        f"frames={len(frames)} "
        f"mode_unique={summary.get('mode_unique_persons', 'n/a')} "
        f"stable_ratio={summary.get('stable_ratio', 'n/a')} "
        f"contact={contact}",
        flush=True,
    )
    print(f"STEP2_VIS_FRAMES={annotated_dir}", flush=True)
    print("STEP2_VIS_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
