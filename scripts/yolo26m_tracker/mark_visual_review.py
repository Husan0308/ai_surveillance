#!/usr/bin/env python3
"""Record explicit human visual review for the NvDCF tracker gate."""
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
    ap.add_argument("--pass-review", action="store_true")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    if not args.pass_review:
        ap.error("--pass-review is required only after actually inspecting the live/recorded IDs")

    gate = args.directory / "tracker_gate.json"
    recording = args.directory / "CAM-01_CAM-02_CAM-03_CAM-04_CAM-05_CAM-06.mkv"
    if not gate.exists():
        raise RuntimeError("run tracker check_gate.py first so tracker evidence exists")
    if not recording.exists():
        raise RuntimeError(f"missing recording: {recording}")

    automated = json.loads(gate.read_text())
    disallowed = [
        f for f in automated.get("failures", [])
        if f not in {"Missing separate NvDCF visual ID-stability review"}
    ]
    if disallowed:
        raise RuntimeError(
            "cannot mark tracker visual review PASS while automated tracker evidence is failing: "
            + "; ".join(disallowed)
        )

    report = {
        "status": "PASS",
        "recording_sha256": sha256(recording),
        "reviewed": {
            "all_six_tiles_advancing": True,
            "one_visible_person_keeps_same_id_while_continuously_visible": True,
            "nearby_people_keep_separate_ids": True,
            "no_obvious_id_flicker_or_rapid_switching": True,
            "person_boxes_remain_visually_aligned": True,
        },
        "notes": args.notes,
    }
    path = args.directory / "tracker_visual_review.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
