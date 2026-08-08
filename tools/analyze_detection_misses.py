#!/usr/bin/env python3
"""Summarize bounded YOLO miss evidence without modifying pipeline state."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def pct(count, total):
    return 100.0 * count / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="data/detection_miss_debug")
    args = parser.parse_args()
    root = Path(args.root)
    samples = []
    for path in root.glob("*/*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            item["_path"] = path
            samples.append(item)
        except (OSError, ValueError):
            continue
    if not samples:
        print(f"No miss evidence found under {root}")
        return

    by_camera = defaultdict(list)
    for sample in samples:
        by_camera[sample.get("camera_id", "UNKNOWN")].append(sample)

    for camera_id in sorted(by_camera):
        camera_samples = by_camera[camera_id]
        latest_by_track = {}
        for sample in sorted(camera_samples, key=lambda x: x["_path"].stat().st_mtime):
            latest_by_track[sample.get("track_id")] = sample
        tracks = list(latest_by_track.values())
        detections = sum(int(x.get("detections", 0)) for x in tracks)
        tracker_only = sum(int(x.get("tracker_only_frames", 0)) for x in tracks)
        opportunities = detections + tracker_only
        recalls = [float(x.get("detector_recall", 0.0)) for x in tracks]
        small = sum(
            float(x.get("bbox_height_ratio", 1.0)) < 0.20
            or float(x.get("bbox_area_ratio", 1.0)) < 0.025
            for x in camera_samples
        )
        low_light = sum(float(x.get("mean_brightness", 255.0)) < 65.0 for x in camera_samples)
        blurry = sum(float(x.get("blur_laplacian_variance", 9999.0)) < 80.0 for x in camera_samples)
        edge = sum(bool(x.get("near_frame_edge", False)) for x in camera_samples)
        candidate_confidences = [
            float(x["yolo_candidate_confidence"])
            for x in camera_samples
            if x.get("yolo_candidate_confidence") is not None
        ]
        low_conf = sum(value < 0.22 for value in candidate_confidences)
        print(
            f"{camera_id} samples:{len(camera_samples)} tracks:{len(tracks)} "
            f"average_detector_recall:{pct(sum(recalls), len(recalls)):.1f}% "
            f"worst_track_recall:{pct(min(recalls) if recalls else 0.0, 1):.1f}% "
            f"tracker_only_percentage:{pct(tracker_only, opportunities):.1f}% "
            f"lost/recovered:{sum(int(x.get('lost_count', 0)) for x in tracks)}/"
            f"{sum(int(x.get('recovered_count', 0)) for x in tracks)}"
        )
        print(
            f"  miss-correlations small:{pct(small, len(camera_samples)):.1f}% "
            f"low_light:{pct(low_light, len(camera_samples)):.1f}% "
            f"blur:{pct(blurry, len(camera_samples)):.1f}% "
            f"edge:{pct(edge, len(camera_samples)):.1f}% "
            f"candidate_conf<0.22:{pct(low_conf, len(candidate_confidences)):.1f}% "
            f"candidate_conf_available:{len(candidate_confidences)}/{len(camera_samples)}"
        )
    print("Seated/chair/monitor occlusion requires manual review of *_context.jpg samples.")


if __name__ == "__main__":
    main()
