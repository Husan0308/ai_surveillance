#!/usr/bin/env python3
"""Record explicit human visual review for the YOLO26m person gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--pass-review", action="store_true",
                    help="confirm live/recorded person boxes were visually checked")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    if not args.pass_review:
        ap.error("--pass-review is required only after you actually inspect the preview/recording")

    recording = args.directory / "CAM-01_CAM-02_CAM-03_CAM-04_CAM-05_CAM-06.mkv"
    overlap = args.directory / "overlap_evidence.json"
    if not recording.exists():
        raise RuntimeError(f"missing recording: {recording}")
    if not overlap.exists():
        raise RuntimeError("run check_gate.py first so overlap_evidence.json exists")

    overlap_data = json.loads(overlap.read_text())
    if overlap_data.get("pairs_over_nms_threshold", 1) != 0:
        raise RuntimeError("cannot mark visual review PASS while duplicate-box NMS evidence is failing")

    report = {
        "status": "PASS",
        "recording_sha256": sha256(recording),
        "reviewed": {
            "all_six_tiles_advancing": True,
            "person_boxes_visually_aligned": True,
            "cam04_double_bbox_not_observed": True,
            "no_obvious_duplicate_person_boxes": True,
        },
        "automated_overlap": {
            "nms_iou_threshold": overlap_data.get("nms_iou_threshold"),
            "pairs_over_nms_threshold": overlap_data.get("pairs_over_nms_threshold"),
            "pairs_iou_gte_090": overlap_data.get("pairs_iou_gte_090"),
            "pairs_iou_gte_095": overlap_data.get("pairs_iou_gte_095"),
            "max_person_iou": overlap_data.get("max_person_iou"),
        },
        "notes": args.notes,
    }
    path = args.directory / "visual_review.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
