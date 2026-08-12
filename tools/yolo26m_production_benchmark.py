#!/usr/bin/env python3
"""Compare YOLO26n/m on the same bounded real-camera samples."""
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


def percentile(values, value):
    return round(float(np.percentile(values, value)), 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", default="data/detection_diagnostic/yolo26m_benchmark.json")
    args = parser.parse_args()
    paths = sorted((ROOT / "data/detection_diagnostic/current").glob("CAM-??_A_original.jpg"))
    frames = [cv2.imread(str(path)) for path in paths]
    if len(frames) != 6 or any(frame is None for frame in frames):
        raise SystemExit("Expected six saved real-camera frames")
    results = []
    for model_path in ("models/yolo26n.pt", "models/yolo26m.pt"):
        for size in ((384, 640), (448, 800)):
            for half in (False, True):
                model = YOLO(str(ROOT / model_path)).to("cuda:0")
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                timings = []; outputs = None
                for index in range(args.warmup + args.iterations):
                    torch.cuda.synchronize(); started = time.perf_counter()
                    outputs = model.predict(frames, classes=[0], conf=.05, iou=.45, max_det=50,
                                            imgsz=size, device="cuda:0", half=half, verbose=False)
                    torch.cuda.synchronize(); elapsed = (time.perf_counter() - started) * 1000
                    if index >= args.warmup: timings.append(elapsed)
                detections = []
                for path, output in zip(paths, outputs):
                    detections.append({"camera_id": path.name[:6], "count": len(output.boxes),
                        "confidences": [round(float(x), 5) for x in output.boxes.conf.cpu()],
                        "boxes": [[round(float(v), 2) for v in row] for row in output.boxes.xyxy.cpu()]})
                results.append({"model": model_path, "size": list(size), "precision": "fp16" if half else "fp32",
                    "batch": 6, "forward_calls": args.warmup + args.iterations,
                    "wall_p50_ms": percentile(timings, 50), "wall_p95_ms": percentile(timings, 95),
                    "vram_peak_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1), "detections": detections})
                del model; torch.cuda.empty_cache()
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
