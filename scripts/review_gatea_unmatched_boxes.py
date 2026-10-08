#!/usr/bin/env python3
"""Render source-frame evidence for candidate boxes absent from Gate A rows.

This is a visual-review helper only; it performs no detector inference and
does not alter the Gate A candidate or reference.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from scripts.run_lowconf_gate_a_replay import INPUTS, REFERENCE, iou, read_rows


def load_expected(camera: str) -> dict[int, list[list[float]]]:
    result: dict[int, list[list[float]]] = {}
    path = REFERENCE / f"{camera.replace('-', '')}-per-frame.csv"
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            raw = row["chosen_bbox"].strip()
            if raw:
                result.setdefault(int(row["frame"]), []).append(json.loads(raw))
    return result


def xyxy(row: dict) -> list[float]:
    x1, y1, x2, y2 = map(float, row["bbox"])
    return [x1, y1, x2 - x1, y2 - y1]


def unmatched_rows(run_dir: Path) -> list[dict]:
    snapshots = read_rows(run_dir / "canonical-objects.jsonl")
    grouped: dict[tuple[str, int], list[dict]] = {}
    for snapshot in snapshots:
        camera = snapshot.get("camera_id")
        if camera not in {"CAM-01", "CAM-04"}:
            continue
        frame = int(snapshot["source_frame_num"])
        grouped[(camera, frame)] = snapshot.get("objects", [])
    output = []
    for camera in ("CAM-01", "CAM-04"):
        expected = load_expected(camera)
        for (frame_camera, frame_num), boxes in sorted(grouped.items()):
            if frame_camera != camera:
                continue
            if frame_num not in expected:
                # The Gate A comparison is scoped to adjudicated source frames;
                # unreviewed frames are not mislabeled as false positives.
                continue
            exp = expected.get(frame_num, [])
            got = boxes
            pairs = sorted(((iou(a, xyxy(b)), ei, gi)
                for ei, a in enumerate(exp) for gi, b in enumerate(got)), reverse=True)
            used_expected, used_got = set(), set()
            for score, ei, gi in pairs:
                if score < .80 or ei in used_expected or gi in used_got:
                    continue
                used_expected.add(ei)
                used_got.add(gi)
            for gi, box in enumerate(got):
                if gi not in used_got:
                    output.append({"camera_id": camera, "frame": frame_num,
                        "pts_ns": int(box["pts_ns"]), "bbox_xyxy": box["bbox"],
                        "bbox_source": box.get("bbox_source"),
                        "native_track_id": box.get("native_track_id"),
                        "detector_confidence": box.get("detector_confidence"),
                        "tracker_confidence": box.get("tracker_confidence"),
                        "expected_boxes_in_frame": len(exp), "overlap_to_expected": [
                            round(iou(xyxy(box), expected_box), 4) for expected_box in exp]})
    return output


def extract(camera: str, frames: list[int], directory: Path) -> dict[int, Image.Image]:
    path, _ = INPUTS[camera]
    ordered = sorted(set(frames))
    pattern = directory / f"{camera.lower()}-%04d.jpg"
    selection = "+".join(f"eq(n\\,{frame})" for frame in ordered)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vf", f"select={selection}", "-fps_mode", "vfr", "-q:v", "2", str(pattern)],
        check=True)
    result = {}
    for index, frame in enumerate(ordered, 1):
        image_path = directory / f"{camera.lower()}-{index:04d}.jpg"
        if not image_path.is_file():
            raise RuntimeError(f"ffmpeg failed to extract {camera} source frame {frame}")
        result[frame] = Image.open(image_path).convert("RGB")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows = unmatched_rows(args.run.resolve())
    manifest = {"source": "exact pinned CAM-01/CAM-04 MP4 frames",
        "matching_rule": "Gate A greedy IoU >= 0.80, one-to-one",
        "candidate_rows": rows, "counts": {camera: sum(r["camera_id"] == camera for r in rows)
            for camera in ("CAM-01", "CAM-04")}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    crops = []
    for camera in ("CAM-01", "CAM-04"):
        selected = [r for r in rows if r["camera_id"] == camera]
        frames = extract(camera, [r["frame"] for r in selected], args.output)
        for row in selected:
            image = frames[row["frame"]]
            x1, y1, x2, y2 = map(float, row["bbox_xyxy"])
            box_w, box_h = x2 - x1, y2 - y1
            crop_w = min(image.width, max(420, int(box_w * 4)))
            crop_h = min(image.height, max(320, int(box_h * 4)))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            left = max(0, min(image.width - crop_w, int(cx - crop_w / 2)))
            top = max(0, min(image.height - crop_h, int(cy - crop_h / 2)))
            crop = image.crop((left, top, left + crop_w, top + crop_h))
            draw = ImageDraw.Draw(crop)
            draw.rectangle((x1-left, y1-top, x2-left, y2-top), outline=(255, 40, 40), width=4)
            label = (f"{camera} frame {row['frame']} {row['bbox_source']} "
                     f"native:{row['native_track_id']} conf:{row['detector_confidence']}")
            crops.append((label, crop.copy()))
    cell_w, cell_h = 650, 420
    cols = 2
    sheet = Image.new("RGB", (cols * cell_w, ((len(crops)+cols-1)//cols) * cell_h), "#202020")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, crop) in enumerate(crops):
        x, y = (index % cols) * cell_w, (index // cols) * cell_h
        draw.text((x + 8, y + 6), label, fill="white", font=font)
        crop.thumbnail((cell_w - 16, cell_h - 40), Image.Resampling.LANCZOS)
        sheet.paste(crop, (x + 8, y + 28))
    sheet.save(args.output / "unmatched-candidate-contact-sheet.jpg", quality=94)
    print(json.dumps({"output": str(args.output), "counts": manifest["counts"],
        "contact_sheet": str(args.output / "unmatched-candidate-contact-sheet.jpg")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
