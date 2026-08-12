#!/usr/bin/env python3
"""Benchmark frozen detector variants against saved confirmed-track miss targets."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1]) + max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1]) - intersection
    return intersection / union if union else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/detection_diagnostic/miss_model_benchmark.json")
    args = parser.parse_args()
    samples = []
    for metadata_path in sorted((ROOT / "data/detection_miss_debug").glob("*/*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        image_path = metadata_path.parent / metadata["full_frame_file"]
        image = cv2.imread(str(image_path))
        if image is not None: samples.append((metadata, image))
    configurations = (("models/yolo26n.pt", (512, 896), False),
                      ("models/yolo26m.pt", (384, 640), False),
                      ("models/yolo26m.pt", (448, 800), False),
                      ("models/yolo26m.pt", (384, 640), True))
    report = []
    for model_path, size, half in configurations:
        model = YOLO(str(ROOT / model_path)).to("cuda:0")
        recovered = strong = small_recovered = small_total = 0
        latencies = []; per_camera = {}; peak_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for offset in range(0, len(samples), 6):
            group = samples[offset:offset + 6]; torch.cuda.synchronize(); started = time.perf_counter()
            outputs = model.predict([image for _, image in group], classes=[0], conf=.05, iou=.45,
                                    max_det=50, imgsz=size, device="cuda:0", half=half, verbose=False)
            torch.cuda.synchronize(); latencies.append((time.perf_counter() - started) * 1000)
            for (metadata, _), output in zip(group, outputs):
                target = metadata["bbox_xyxy"]; candidates = []
                for box, confidence in zip(output.boxes.xyxy.cpu(), output.boxes.conf.cpu()):
                    candidates.append((iou(target, [float(x) for x in box]), float(confidence)))
                best_iou, best_conf = max(candidates, default=(0.0, 0.0))
                hit = best_iou >= .10 and best_conf >= .05; high = hit and best_conf >= .28
                is_small = float(metadata.get("bbox_height_ratio", 1)) < .20 or float(metadata.get("bbox_area_ratio", 1)) < .025
                recovered += hit; strong += high; small_total += is_small; small_recovered += bool(hit and is_small)
                camera = per_camera.setdefault(metadata["camera_id"], {"samples": 0, "recovered": 0, "strong": 0})
                camera["samples"] += 1; camera["recovered"] += hit; camera["strong"] += high
        report.append({"model": model_path, "size": list(size), "precision": "fp16" if half else "fp32",
            "samples": len(samples), "recovered": recovered, "strong_recovered": strong,
            "small_samples": small_total, "small_recovered": small_recovered,
            "batch_p50_ms": round(float(np.percentile(latencies, 50)), 3),
            "batch_p95_ms": round(float(np.percentile(latencies, 95)), 3),
            "vram_peak_mib": round((torch.cuda.max_memory_allocated() - peak_before) / 1048576, 1),
            "per_camera": per_camera})
        del model; torch.cuda.empty_cache()
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
