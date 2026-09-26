#!/usr/bin/env python3
"""Causal, sparse full-body OSNet extraction from corrected MV3DT metadata."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from reid_embedder import OsnetCpuEmbedder

CAMS = {"CAM-01": "cam_00.mp4", "CAM-04": "cam_01.mp4"}
W, H = 1920, 1080


def load_frames(path: Path) -> dict[tuple[str, int], dict]:
    latest = {}
    with path.open() as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame = event.get("frame")
            if not frame or frame.get("sensorId") not in CAMS:
                continue
            latest[(frame["sensorId"], int(frame["id"]))] = frame
    return latest


def bbox_of(obj: dict) -> tuple[float, float, float, float]:
    b = obj["bbox"]
    return float(b.get("leftX", 0.0)), float(b.get("topY", 0.0)), float(b.get("rightX", 0.0)), float(b.get("bottomY", 0.0))


def crop_quality(box: tuple[float, float, float, float]) -> tuple[bool, dict]:
    l, t, r, b = box
    width, height = r - l, b - t
    clipped_l, clipped_t = max(0.0, l), max(0.0, t)
    clipped_r, clipped_b = min(float(W), r), min(float(H), b)
    visible_area = max(0.0, clipped_r - clipped_l) * max(0.0, clipped_b - clipped_t)
    area = max(width * height, 1.0)
    inside = visible_area / area
    aspect = width / max(height, 1.0)
    border_truncated = l <= 2.0 or t <= 2.0 or r >= W - 2.0 or b >= H - 2.0
    useful = (
        width >= 28.0
        and height >= 96.0
        and area >= W * H * 0.003
        and 0.16 <= aspect <= 0.95
        and inside >= 0.97
        and not border_truncated
    )
    return useful, {
        "width": width,
        "height": height,
        "area_fraction": area / (W * H),
        "inside_fraction": inside,
        "aspect": aspect,
        "border_truncated": border_truncated,
    }


def pct(values, q):
    return float(np.quantile(values, q)) if values else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kafka", required=True, type=Path)
    ap.add_argument("--video-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--experiment-root", required=True, type=Path)
    ap.add_argument("--interval", type=int, default=10)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(args.kafka)
    by_camera = defaultdict(dict)
    for (cam, frame_num), payload in frames.items():
        by_camera[cam][frame_num] = payload

    config = {
        "model_name": "osnet_ain_x1_0",
        "model_path": "models/reid/osnet_ain_x1_0_msmt17.pth",
        "model_sha256": "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f",
        "download_if_missing": False,
        "input_height": 256,
        "input_width": 128,
        "cpu_threads": 2,
    }
    embedder = OsnetCpuEmbedder(config, args.experiment_root)

    embeddings = []
    records = []
    latency_ms = []
    crop_dims = []
    rejection = defaultdict(int)
    extracted_by_cam = defaultdict(int)
    last_extract = {}

    started = time.perf_counter()
    for cam, video_name in CAMS.items():
        capture = cv2.VideoCapture(str(args.video_dir / video_name))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {video_name}")
        frame_num = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            payload = by_camera[cam].get(frame_num)
            if payload:
                crops, pending = [], []
                for obj in payload.get("objects", []):
                    if obj.get("type", "").lower() != "person":
                        continue
                    native_id = int(obj["id"])
                    box = bbox_of(obj)
                    useful, quality = crop_quality(box)
                    key = (cam, native_id)
                    due = frame_num - last_extract.get(key, -10**9) >= args.interval
                    if not due:
                        rejection["interval"] += 1
                        continue
                    if not useful:
                        rejection["quality"] += 1
                        continue
                    l, t, r, b = box
                    il, it = max(0, int(l)), max(0, int(t))
                    ir, ib = min(W, int(np.ceil(r))), min(H, int(np.ceil(b)))
                    crop = image[it:ib, il:ir]
                    if crop.size == 0:
                        rejection["empty"] += 1
                        continue
                    crops.append(crop)
                    pending.append((obj, box, quality, (ir - il, ib - it)))
                    last_extract[key] = frame_num

                if crops:
                    tic = time.perf_counter()
                    batch = embedder.embed_batch(crops)
                    batch_ms = (time.perf_counter() - tic) * 1000.0
                    each_ms = batch_ms / len(crops)
                    for vector, item in zip(batch, pending):
                        obj, box, quality, dimensions = item
                        index = len(embeddings)
                        embeddings.append(vector)
                        latency_ms.append(each_ms)
                        crop_dims.append(dimensions)
                        extracted_by_cam[cam] += 1
                        xyz = obj.get("bbox3d", {}).get("coordinates", [])
                        records.append({
                            "embedding_index": index,
                            "camera_id": cam,
                            "frame": frame_num,
                            "timestamp": payload.get("timestamp"),
                            "native_track_id": int(obj["id"]),
                            "bbox": list(box),
                            "confidence": float(obj.get("confidence", 0.0)),
                            "world": [float(xyz[0]), float(xyz[1])] if len(xyz) >= 2 else None,
                            "visibility": float(obj.get("info", {}).get("visibility", 0.0)),
                            "crop_width": dimensions[0],
                            "crop_height": dimensions[1],
                            "quality": quality,
                            "batch_latency_ms": batch_ms,
                            "per_crop_latency_ms": each_ms,
                        })
            frame_num += 1
        capture.release()

    matrix = np.stack(embeddings).astype(np.float32) if embeddings else np.empty((0, 512), np.float32)
    np.save(args.output_dir / "embeddings.npy", matrix)
    with (args.output_dir / "embedding_records.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    elapsed = time.perf_counter() - started
    report = {
        "model": embedder.metrics(),
        "causal": True,
        "cross_camera_input": False,
        "face_input": False,
        "future_frame_input": False,
        "extraction_interval_frames": args.interval,
        "embeddings_extracted": len(records),
        "extracted_by_camera": dict(extracted_by_cam),
        "rejections": dict(rejection),
        "wall_seconds": elapsed,
        "effective_source_fps_per_camera": 2400.0 / elapsed if elapsed else None,
        "latency_ms": {
            "mean": statistics.fmean(latency_ms) if latency_ms else None,
            "p95": pct(latency_ms, 0.95),
            "max": max(latency_ms) if latency_ms else None,
        },
        "crop_width": {
            "median": statistics.median([x[0] for x in crop_dims]) if crop_dims else None,
            "p10": pct([x[0] for x in crop_dims], 0.10),
            "p90": pct([x[0] for x in crop_dims], 0.90),
        },
        "crop_height": {
            "median": statistics.median([x[1] for x in crop_dims]) if crop_dims else None,
            "p10": pct([x[1] for x in crop_dims], 0.10),
            "p90": pct([x[1] for x in crop_dims], 0.90),
        },
    }
    (args.output_dir / "extraction_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
