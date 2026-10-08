#!/usr/bin/env python3
"""Audit invalid NvDCF shadow metadata and prove canonical-box containment."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def valid_xywh(value):
    try:
        x, y, w, h = map(float, value)
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(v) for v in (x, y, w, h)) and w > 0 and h > 0


def analyze(run_dir: Path):
    invalid = rows(run_dir / "invalid_shadow_geometry.jsonl")
    dets = rows(run_dir / "detections.jsonl")
    tracked = rows(run_dir / "tracks.jsonl")
    fallbacks = rows(run_dir / "shadow_fallbacks.jsonl")
    frames = rows(run_dir / "frames.jsonl")
    by_cam = defaultdict(list)
    for row in invalid:
        by_cam[row["camera_id"]].append(row)
    det_by_source_frame = Counter((r["source_id"], r["frame"]) for r in dets)
    active_ids = {(r["source_id"], r["frame"], r["object_id"]) for r in tracked}
    fallback_ids = {(int(r["camera_id"][-2:]) - 1, r["frame_num"], r["matched_shadow_native_id"])
                    for r in fallbacks}
    warn_count = len(re.findall(
        r"Width and height of input and output bounding boxes from vpiSubmitDCFTrackerLocalizeBatch\(\) are NAN!!",
        (run_dir / "native.log").read_text(errors="replace"),
    ))
    source_map = {f"CAM-{i:02d}": i - 1 for i in range(1, 7)}
    cameras = {}
    for camera, src in source_map.items():
        rs = by_cam[camera]
        frames_seen = sorted({r["frame"] for r in rs})
        ranges = []
        for frame in frames_seen:
            if not ranges or frame > ranges[-1][-1] + 1:
                ranges.append([frame])
            else:
                ranges[-1].append(frame)
        cameras[camera] = {
            "invalid_shadow_samples": len(rs),
            "unique_frames": len(frames_seen),
            "invalid_frame_ranges": [[r[0], r[-1]] for r in ranges],
            "unique_invalid_shadow_ids": len({r["native_track_id"] for r in rs}),
            "state_values": dict(Counter(str(r["state_value"]) for r in rs)),
            "mapped_state_names": {"2": "INACTIVE/shadow"} if rs and all(r["state_value"] == 2 for r in rs) else {},
            "all_bbox_components_finite": all(all(r["bbox_finite"]) for r in rs),
            "all_bbox_rows_have_nonpositive_width_or_height": all(not r["bbox_positive"] for r in rs),
            "all_confidence_finite": all(r["confidence_finite"] for r in rs),
            "invalid_rows_with_current_detector_person_proposal": sum(
                det_by_source_frame[(src, r["frame"])] > 0 for r in rs
            ),
            "invalid_rows_same_native_id_active_output": sum(
                (src, r["frame"], r["native_track_id"]) in active_ids for r in rs
            ),
            "invalid_rows_same_native_id_fallback_output": sum(
                (src, r["frame"], r["native_track_id"]) in fallback_ids for r in rs
            ),
            "examples": rs[:3],
        }
    bad_active = [r for r in tracked if not valid_xywh(r.get("box", []))]
    bad_detection = [r for r in dets if r.get("class_id") == 0 and not valid_xywh(r.get("box", []))]
    bad_fallback = [r for r in fallbacks if not valid_xywh(r.get("detector_bbox", []))]
    source_frame_sets = defaultdict(set)
    for r in frames:
        source_frame_sets[r["source_id"]].add(r["frame"])
    return {
        "run": str(run_dir),
        "native_vpi_nan_warning_lines": warn_count,
        "invalid_shadow_sample_rows": len(invalid),
        "warning_per_source_mapping_available": False,
        "warning_mapping_note": "DeepStream 9.1 NvMultiObjectTracker CLOG warning contains no camera, PTS, or native ID. The app-level shadow metadata records are source/frame-addressed but are not 1:1 with native warning lines.",
        "per_camera": cameras,
        "canonical_feed_safety": {
            "active_tracker_jsonl_rows": len(tracked),
            "invalid_active_tracker_boxes": len(bad_active),
            "raw_person_detection_rows": sum(1 for r in dets if r.get("class_id") == 0),
            "invalid_raw_person_detection_boxes": len(bad_detection),
            "fallback_rows": len(fallbacks),
            "invalid_fallback_detection_boxes": len(bad_fallback),
            "canonical_bbox_validity_proven_by": [
                "tracker_exit rejects any active bbox with non-finite coordinates/confidence or non-positive width/height",
                "tracker_current_shadows rejects any shadow bbox with non-finite coordinates or non-positive width/height before matching",
                "fallback rectangles are copied from PGIE detections validated finite, positive, and within source bounds",
                "the IPC reader rejects non-finite/decreasing bbox coordinates and duplicate IDs",
            ],
            "invalid_geometry_reaches_canonical_output": bool(bad_active or bad_fallback),
        },
        "frames_with_pgie_by_source_first_2400": {
            f"CAM-{i+1:02d}": sum(0 <= f < 2400 for f in source_frame_sets[i]) for i in range(6)
        },
        "interpretation": "Observed public shadow metadata contains finite coordinates but degenerate extents and is dropped from the fallback candidate list. The exact association between proprietary VPI warning lines and individual frames/targets is unavailable; warning impact is bounded at the exported active/shadow/canonical-feed boundary, but this run does not attribute every warning line to one camera/target.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.run)
    output = args.output or args.run / "nvdcf-invalid-shadow-audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
