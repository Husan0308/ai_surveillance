#!/usr/bin/env python3
"""Prepare same-frame YOLO960 supplemental proposals for DeepStream context-only evaluation.

OpenCV is used only to decode source video and crop pixels. All model inference
is performed by the existing DeepStream/nvinfer diagnostic runner.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / ".runtime/amc-transformer-personpresent-MtbZPbuh/amc_inputs/videos"
VIDEOS = {0: VIDEO_DIR / "cam_00.mp4", 1: VIDEO_DIR / "cam_01.mp4"}
YOLO960 = ROOT / ".runtime/f5-cam05-safety-20261007-r1/replay-960-exact/detections.jsonl"
YOLO640 = ROOT / ".runtime/gate-a-direct-file-20261005-r3/R1/detections.jsonl"
CARDBOARD640 = ROOT / ".runtime/gate-a-pgie-forensics-20261006-r1/runs/threshold-020-r2/detections.jsonl"
CAM05_960 = ROOT / ".runtime/cam05-supplemental-diagnostic-20261007-r1/run-yolo960-b2-r2/detections.jsonl"
CAM05_640 = ROOT / ".runtime/cam05-supplemental-diagnostic-20261007-r1/run-h264420/detections.jsonl"
CAM05_CLIP = ROOT / ".runtime/cam05-supplemental-diagnostic-20261007-r1/cam05-miss-clip-h264420.mp4"
DEFAULT_OUT = ROOT / ".runtime/f5-cam05-safety-20261007-r1/secondary-context-eval"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def candidate_only(row: dict, normal: list[dict]) -> bool:
    box = [float(v) for v in row["box"]]
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    for base in normal:
        other = [float(v) for v in base["box"]]
        if iou(box, other) >= 0.30:
            return False
        if other[0] <= cx <= other[0] + other[2] and other[1] <= cy <= other[1] + other[3]:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--supplemental-detections", type=Path, default=YOLO960,
        help="same-frame full-frame person proposals to verify against 640-frame detections")
    ap.add_argument("--cam05-supplemental-detections", type=Path, default=CAM05_960,
        help="same-frame CAM-05 clip proposals; source_id 0 is used")
    ap.add_argument("--cam05-supplemental-frames", type=Path,
        help="optional frame metadata used to normalize candidate boxes into decoded source coordinates")
    ap.add_argument("--candidate-label", default="YOLO960",
        help="provenance label for the supplemental detector, e.g. YOLO896")
    ap.add_argument("--cam05-only", action="store_true",
        help="prepare only the 41-frame CAM-05 seated-person miss crop set")
    ap.add_argument("--cam05-right-half-tiles", action="store_true",
        help="diagnostic: evaluate a generic 540x540 right-half tile on each CAM-05 miss frame")
    ap.add_argument("--cardboard-right-half-tiles", action="store_true",
        help="diagnostic: evaluate a generic 1080x1080 right-side tile on reviewed CAM-04 cardboard frames")
    args = ap.parse_args()
    if args.cam05_right_half_tiles and not args.cam05_only:
        raise ValueError("--cam05-right-half-tiles requires --cam05-only")
    if args.cardboard_right_half_tiles and args.cam05_only:
        raise ValueError("cardboard tile diagnostic and CAM-05-only mode are mutually exclusive")
    if args.cardboard_right_half_tiles and args.cam05_right_half_tiles:
        raise ValueError("choose only one diagnostic tile mode")
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("output must remain under .runtime")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    crop_dir = out / "crop-frames"
    crop_dir.mkdir()

    baseline: dict[tuple[int, int], list[dict]] = {}
    for row in read_jsonl(YOLO640):
        baseline.setdefault((int(row["source_id"]), int(row["frame"])), []).append(row)
    supplemental_path = args.supplemental_detections.resolve()
    cam05_supplemental_path = args.cam05_supplemental_detections.resolve()
    supplemental = [] if (args.cam05_only or args.cardboard_right_half_tiles) else [row for row in read_jsonl(supplemental_path)
        if int(row["source_id"]) in VIDEOS and int(row["class_id"]) == 0
        and candidate_only(row, baseline.get((int(row["source_id"]), int(row["frame"])), []))]
    cam05_baseline: dict[tuple[int, int], list[dict]] = {}
    for row in read_jsonl(CAM05_640):
        cam05_baseline.setdefault((int(row["source_id"]), int(row["frame"])), []).append(row)
    if args.cam05_right_half_tiles:
        cam05_supplemental = [{"source_id": 0, "frame": frame, "pts_ns": 100_000_000 + frame * 50_000_000,
            "class_id": 0, "confidence": 0.0, "box": [480.0, 60.0, 90.0, 120.0]}
            for frame in range(41)]
    elif args.cardboard_right_half_tiles:
        cam05_supplemental = []
    else:
        candidate_rows = read_jsonl(cam05_supplemental_path)
        frame_metadata = {}
        if args.cam05_supplemental_frames:
            for frame_row in read_jsonl(args.cam05_supplemental_frames.resolve()):
                frame_metadata[(int(frame_row["source_id"]), int(frame_row["frame"]))] = frame_row
        normalized_candidates = []
        for row in candidate_rows:
            if int(row["source_id"]) != 0 or int(row["class_id"]) != 0:
                continue
            normalized = dict(row)
            if args.cam05_supplemental_frames:
                metadata = frame_metadata.get((0, int(row["frame"])))
                if metadata is None:
                    raise RuntimeError(f"missing same-frame metadata for candidate {row['frame']}")
                px, py = int(metadata["pipeline_width"]), int(metadata["pipeline_height"])
                sx, sy = int(metadata["source_width"]), int(metadata["source_height"])
                x, y, w, h = map(float, row["box"])
                normalized["box"] = [x * sx / px, y * sy / py, w * sx / px, h * sy / py]
            if candidate_only(normalized, cam05_baseline.get((0, int(row["frame"])), [])):
                normalized_candidates.append(normalized)
        cam05_supplemental = normalized_candidates

    cardboard_rows = [row for row in read_jsonl(CARDBOARD640)
        if int(row["source_id"]) == 3 and int(row["class_id"]) == 0
        and float(row["confidence"]) < 0.25]
    cardboard = []
    for row in ([] if (args.cam05_only or not args.cardboard_right_half_tiles) else cardboard_rows):
        x, y, w, h = map(float, row["box"])
        cx, cy = x + w / 2, y + h / 2
        # This is a previously human-reviewed diagnostic set, not an inference rule.
        if 1120 <= cx <= 1270 and 0 <= cy <= 210:
            cardboard.append(row)

    sources = {}
    for source_id, path in ({} if args.cam05_only else VIDEOS).items():
        if not path.is_file():
            raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source video: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if (width, height, count) != (1920, 1080, 2400) or abs(fps - 20.0) > .01:
            raise RuntimeError(f"unexpected pinned source contract {path}: {(width,height,fps,count)}")
        sources[source_id] = {"path": path, "cap": cap, "width": width, "height": height,
                              "count": count}
    cam05_cap = cv2.VideoCapture(str(CAM05_CLIP))
    if not cam05_cap.isOpened():
        raise RuntimeError(f"cannot open CAM-05 miss clip: {CAM05_CLIP}")
    cam05_width = int(cam05_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam05_height = int(cam05_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam05_count = int(cam05_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cam05_fps = cam05_cap.get(cv2.CAP_PROP_FPS)
    if (cam05_width, cam05_height, cam05_count) != (960, 540, 41) or abs(cam05_fps - 20.0) > .01:
        raise RuntimeError(f"unexpected CAM-05 clip contract: {(cam05_width,cam05_height,cam05_fps,cam05_count)}")
    sources[2] = {"path": CAM05_CLIP, "cap": cam05_cap, "width": cam05_width,
                  "height": cam05_height, "count": cam05_count}

    all_candidates = []
    for row in supplemental:
        all_candidates.append({"kind": f"{args.candidate_label}_SUPPLEMENTAL", "source_id": int(row["source_id"]),
            "frame": int(row["frame"]), "pts_ns": int(row["pts_ns"]),
            "confidence": float(row["confidence"]), "bbox": list(map(float, row["box"]))})
    for row in cam05_supplemental:
        all_candidates.append({"kind": f"CAM05_{args.candidate_label}_SUPPLEMENTAL", "source_id": 2,
            "frame": int(row["frame"]), "pts_ns": int(row["pts_ns"]),
            "confidence": float(row["confidence"]), "bbox": list(map(float, row["box"]))})
    for row in cardboard:
        # threshold-r2 six-source topology repeats CAM-04 in source slots 3,4,5;
        # source slot 3 is used once to keep this reviewed negative set unique.
        all_candidates.append({"kind": "REVIEWED_CARDBOARD_640_020", "source_id": 1,
            "frame": int(row["frame"]), "pts_ns": int(row["pts_ns"]),
            "confidence": float(row["confidence"]), "bbox": list(map(float, row["box"]))})

    dedup = {}
    for item in all_candidates:
        key = (item["kind"], item["source_id"], item["frame"], tuple(round(x, 2) for x in item["bbox"]))
        dedup[key] = item
    all_candidates = sorted(dedup.values(), key=lambda x: (x["source_id"], x["frame"], x["kind"]))

    rows_out = []
    current_source, current_frame, frame_image = None, -1, None
    for next_index, item in enumerate(all_candidates):
        source_id, frame_num = item["source_id"], item["frame"]
        source = sources[source_id]
        if source_id != current_source:
            current_source, current_frame, frame_image = source_id, -1, None
        if frame_num != current_frame:
            source["cap"].set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, frame_image = source["cap"].read()
            if not ok or frame_image is None:
                raise RuntimeError(f"failed exact frame decode: source={source_id}, frame={frame_num}")
            if frame_image.shape[:2] != (source["height"], source["width"]):
                raise RuntimeError(f"decoded dimensions changed at {source_id}/{frame_num}: {frame_image.shape}")
            current_frame = frame_num
        width, height = source["width"], source["height"]
        side = min(int(math.floor(width / 3.0 + .5)), height) & ~1
        x, y, w, h = item["bbox"]
        cx, cy = x + w / 2.0, y + h / 2.0
        crop_x = int(math.floor(cx - side / 2 + .5))
        crop_y = int(math.floor(cy - side / 2 + .5))
        if args.cam05_right_half_tiles or args.cardboard_right_half_tiles:
            side = source["height"]
            crop_x, crop_y = source["width"] - side, 0
        crop_x = max(0, min(width - side, crop_x)) & ~1
        crop_y = max(0, min(height - side, crop_y)) & ~1
        crop = frame_image[crop_y:crop_y + side, crop_x:crop_x + side]
        if crop.shape[:2] != (side, side):
            raise RuntimeError(f"crop geometry invalid at {source_id}/{frame_num}")
        if side != 640:
            crop = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_LINEAR)
        filename = f"crop-{next_index:06d}.png"
        if not cv2.imwrite(str(crop_dir / filename), crop):
            raise RuntimeError(f"failed crop image write: {filename}")
        item.update({"crop_index": next_index, "crop_file": filename,
            "crop_x": crop_x, "crop_y": crop_y, "crop_side": side,
            "expected_center_in_crop": [((cx - crop_x) * 640 / side), ((cy - crop_y) * 640 / side)]})
        rows_out.append(item)

    for source in sources.values():
        source["cap"].release()
    (out / "candidate-map.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows_out))
    manifest = {
        "purpose": "diagnostic only; pixel crops prepared by OpenCV, all inference via DeepStream nvinfer",
        "secondary_run_detections": str(supplemental_path.relative_to(ROOT)),
        "cam05_miss_clip_secondary_detections": str(cam05_supplemental_path.relative_to(ROOT)),
        "cam05_miss_clip_primary_detections": str(CAM05_640.relative_to(ROOT)),
        "normal_run_detections": str(YOLO640.relative_to(ROOT)),
        "reviewed_cardboard_candidates": str(CARDBOARD640.relative_to(ROOT)),
        "sources": {str(k): {"path": str(v["path"].relative_to(ROOT)), "width": v["width"],
            "height": v["height"], "frames": v["count"], "fps": 20} for k,v in sources.items()},
        "candidate_counts": {f"{args.candidate_label.lower()}_unmatched_by_high_conf_640": len(supplemental),
            "cam05_miss_clip_supplemental": len(cam05_supplemental),
            "reviewed_cardboard_640_0.20_candidates": len(cardboard), "total_crops": len(rows_out)},
        "crop_contract": "same square context crop geometry as lowconf_prepare_nv12_crop; crop side=min(round(frame_width/3),frame_height), centered on candidate and clamped to even pixel coordinates; resized to 640 only when side differs",
        "crop_inference_contract": "DeepStream 9.1 nvinfer with frozen YOLO26m FP16 640 engine/parser, batch-2 direct MP4; not external Python/OpenCV inference",
        "crop_variant": ("CAM-05 right-half tile diagnostic" if args.cam05_right_half_tiles else
            "reviewed cardboard right-half tile diagnostic" if args.cardboard_right_half_tiles else
            "candidate-centered context crop"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
