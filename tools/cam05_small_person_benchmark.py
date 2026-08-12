#!/usr/bin/env python3
"""Compare bounded CAM-05 small-person recovery strategies on saved miss frames."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import torch
from ultralytics import YOLO


def iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = ((left[2] - left[0]) * (left[3] - left[1])
             + (right[2] - right[0]) * (right[3] - right[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def predict(model, images, size, confidence=.01):
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = model.predict(images, classes=[0], conf=confidence, iou=.45, max_det=50,
                            imgsz=size, device="cuda:0", half=False, verbose=False)
    torch.cuda.synchronize()
    return outputs, (time.perf_counter() - started) * 1000


def boxes(output, offset=(0, 0)):
    ox, oy = offset
    return [([float(row[0]) + ox, float(row[1]) + oy,
              float(row[2]) + ox, float(row[3]) + oy], float(confidence))
            for row, confidence in zip(output.boxes.xyxy.cpu(), output.boxes.conf.cpu())]


def deduplicate(candidates, threshold=.5):
    result = []
    for candidate in sorted(candidates, key=lambda value: value[1], reverse=True):
        if all(iou(candidate[0], existing[0]) < threshold for existing in result):
            result.append(candidate)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/detection_diagnostic/cam05_small_person_benchmark.json")
    args = parser.parse_args()
    samples = []
    for metadata_path in sorted((ROOT / "data/detection_miss_debug/CAM-05").glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        image = cv2.imread(str(metadata_path.parent / metadata["full_frame_file"]))
        if image is not None:
            samples.append((metadata_path.stem, metadata, image))
    if not samples:
        raise SystemExit("No CAM-05 saved miss samples found")

    model = YOLO(str(ROOT / "models/yolo26m.pt")).to("cuda:0")
    model.predict([samples[0][2]], classes=[0], conf=.01, imgsz=(384, 640),
                  device="cuda:0", verbose=False)
    configurations = {}

    for name, size, confidence in (("A_current_full", (384, 640), .05),
                                   ("B_larger_full", (448, 800), .05),
                                   ("E_lower_conf_full", (384, 640), .01)):
        outputs, latency = predict(model, [sample[2] for sample in samples], size, confidence)
        configurations[name] = ([boxes(output) for output in outputs], latency, size, confidence)

    tiled_candidates = [[] for _ in samples]
    tiled_latency = 0.0
    for index, (_, _, image) in enumerate(samples):
        height, width = image.shape[:2]
        crops, offsets = [], []
        for y1, y2 in ((0, height // 2), (height // 2, height)):
            for x1, x2 in ((0, width // 2), (width // 2, width)):
                crops.append(image[y1:y2, x1:x2]); offsets.append((x1, y1))
        outputs, latency = predict(model, crops, (384, 640), .05)
        tiled_latency += latency
        tiled_candidates[index] = deduplicate([item for output, offset in zip(outputs, offsets)
                                                for item in boxes(output, offset)])
    configurations["C_four_tiles"] = (tiled_candidates, tiled_latency, (384, 640), .05)

    roi_candidates, roi_latency = [], 0.0
    for _, _, image in samples:
        height, width = image.shape[:2]
        x1, x2, y1, y2 = int(width * .30), int(width * .75), 0, int(height * .40)
        outputs, latency = predict(model, [image[y1:y2, x1:x2]], (384, 640), .05)
        roi_latency += latency
        roi_candidates.append(boxes(outputs[0], (x1, y1)))
    configurations["D_top_center_roi"] = (roi_candidates, roi_latency, (384, 640), .05)

    report = {"camera_id": "CAM-05", "model": "models/yolo26m.pt", "samples": len(samples), "strategies": {}}
    for name, (all_candidates, latency, size, confidence) in configurations.items():
        details, recovered, unmatched = [], 0, 0
        for (sample_id, metadata, image), candidates in zip(samples, all_candidates):
            target = metadata["bbox_xyxy"]
            ranked = sorted(((iou(target, box), conf, box) for box, conf in candidates), reverse=True)
            best_iou, best_conf, best_box = ranked[0] if ranked else (0.0, 0.0, None)
            hit = best_iou >= .10
            recovered += int(hit)
            unmatched += sum(iou(target, box) < .10 for box, _ in candidates)
            height, width = image.shape[:2]
            if name=="C_four_tiles":scale=min(size[1]/(width/2),size[0]/(height/2))
            elif name=="D_top_center_roi":scale=min(size[1]/(width*.45),size[0]/(height*.40))
            else:scale=min(size[1]/width,size[0]/height)
            target_w, target_h = target[2] - target[0], target[3] - target[1]
            details.append({"sample": sample_id, "hit": hit, "best_iou": round(best_iou, 4),
                            "best_confidence": round(best_conf, 4), "best_bbox": best_box,
                            "target_bbox": target, "effective_target_px": [round(target_w * scale, 1), round(target_h * scale, 1)],
                            "person_candidates": len(candidates)})
        report["strategies"][name] = {"recovered": recovered, "total": len(samples),
            "recall": round(recovered / len(samples), 4), "batch_or_total_latency_ms": round(latency, 3),
            "mean_per_frame_ms": round(latency / len(samples), 3), "source_images_per_second":round(len(samples)/max(latency/1000,1e-9),2), "unmatched_person_candidates": unmatched,
            "confidence_floor": confidence, "imgsz": list(size), "details": details}

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
