#!/usr/bin/env python3
"""Patch archived detector A/B runner to v8 tentative-survival diagnostic.

v8 starts from the successful v6 target-creation candidate, restores
minTrackerConfidence=0.20 if the local runner is currently on v7, and changes
one probationary lifecycle parameter only:
  TargetManagement.earlyTerminationAge: 1 -> 2

The archived runner is under .runtime. Production configs are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = ROOT / ".runtime/final-detector-ab-20261002-uSpPqjLY/run_candidate.py"

V6 = (
    "YOLO_DELTA = {"
    "'TargetManagement.minTrackerConfidence': .20, "
    "'TargetManagement.probationAge': 2, "
    "'TargetManagement.minIouDiff4NewTarget': .35, "
    "'DataAssociator.tentativeDetectorConfidence': .25, "
    "'DataAssociator.minMatchingScore4Overall': .35, "
    "'DataAssociator.minMatchingScore4Iou': .10, "
    "'DataAssociator.minMatchingScore4SizeSimilarity': .30"
    "}"
)
V7 = (
    "YOLO_DELTA = {"
    "'TargetManagement.minTrackerConfidence': .15, "
    "'TargetManagement.probationAge': 2, "
    "'TargetManagement.minIouDiff4NewTarget': .35, "
    "'DataAssociator.tentativeDetectorConfidence': .25, "
    "'DataAssociator.minMatchingScore4Overall': .35, "
    "'DataAssociator.minMatchingScore4Iou': .10, "
    "'DataAssociator.minMatchingScore4SizeSimilarity': .30"
    "}"
)
V8 = (
    "YOLO_DELTA = {"
    "'TargetManagement.minTrackerConfidence': .20, "
    "'TargetManagement.probationAge': 2, "
    "'TargetManagement.earlyTerminationAge': 2, "
    "'TargetManagement.minIouDiff4NewTarget': .35, "
    "'DataAssociator.tentativeDetectorConfidence': .25, "
    "'DataAssociator.minMatchingScore4Overall': .35, "
    "'DataAssociator.minMatchingScore4Iou': .10, "
    "'DataAssociator.minMatchingScore4SizeSimilarity': .30"
    "}"
)

PRODUCTION_TRACKER = ROOT / "config/mv3dt_dev_room/config_tracker.yml"
CANDIDATE_V8 = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v8_early2.yml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    args = ap.parse_args()

    runner = args.runner.resolve()
    runtime_root = (ROOT / ".runtime").resolve()
    try:
        runner.relative_to(runtime_root)
    except ValueError as exc:
        raise SystemExit(f"refusing to edit non-runtime file: {runner}") from exc
    if not runner.is_file():
        raise SystemExit(f"runner not found: {runner}")

    production_before = sha256(PRODUCTION_TRACKER)

    candidate = CANDIDATE_V8.read_text()
    required = (
        "minIouDiff4NewTarget: 0.35",
        "minTrackerConfidence: 0.20",
        "probationAge: 2",
        "earlyTerminationAge: 2",
        "maxShadowTrackingAge: 162",
        "tentativeDetectorConfidence: 0.25",
        "minMatchingScore4Overall: 0.35",
        "minMatchingScore4Iou: 0.10",
        "minMatchingScore4SizeSimilarity: 0.30",
        "minMatchingScore4VisualSimilarity: 0.0520394823204932",
        "minPeerTrackletMatchScore: 0.48",
    )
    missing = [item for item in required if item not in candidate]
    if missing:
        raise SystemExit(f"tracked v8 candidate drifted; missing: {missing}")

    before = runner.read_text()
    if V8 in before:
        print("runner already patched to v8")
        return 0
    if before.count(V7) == 1:
        after = before.replace(V7, V8, 1)
        migrated_from = "v7"
    elif before.count(V6) == 1:
        after = before.replace(V6, V8, 1)
        migrated_from = "v6"
    else:
        raise SystemExit("archived runner contract changed; expected v7 or v6 YOLO_DELTA")

    policy = (
        "retention-v8 early2 diagnostic; v6 target creation held fixed; "
        "trackerconf020 restored; not production accepted"
    )
    for stale in (
        "retention-v7 trackerconf015 diagnostic; v6 target creation held fixed; not production accepted",
        "retention-v6 newtarget035 diagnostic; shadow162 restored; v4 association held fixed; not production accepted",
    ):
        after = after.replace(stale, policy)

    backup = runner.with_suffix(runner.suffix + ".pre-nvdcf-retention-v8")
    if not backup.exists():
        shutil.copy2(runner, backup)

    before_hash = sha256(runner)
    runner.write_text(after)
    after_hash = sha256(runner)

    production_after = sha256(PRODUCTION_TRACKER)
    if production_after != production_before:
        shutil.copy2(backup, runner)
        raise SystemExit("production tracker changed during runtime patch; restored runner")

    manifest = {
        "runner": str(runner),
        "backup": str(backup),
        "migrated_from": migrated_from,
        "runner_sha256_before": before_hash,
        "runner_sha256_after": after_hash,
        "production_tracker_sha256": production_after,
        "tracked_candidate_v8_sha256": sha256(CANDIDATE_V8),
        "diagnostic_hypothesis": (
            "remaining CAM-01 unmatched real-looking detections die during tentative "
            "probation after a single association/confidence miss; earlyTerminationAge=2 "
            "should let them survive one miss"
        ),
        "yolo_tracker_delta": {
            "TargetManagement.minTrackerConfidence": 0.20,
            "TargetManagement.probationAge": 2,
            "TargetManagement.earlyTerminationAge": 2,
            "TargetManagement.minIouDiff4NewTarget": 0.35,
            "DataAssociator.tentativeDetectorConfidence": 0.25,
            "DataAssociator.minMatchingScore4Overall": 0.35,
            "DataAssociator.minMatchingScore4Iou": 0.10,
            "DataAssociator.minMatchingScore4SizeSimilarity": 0.30,
        },
        "held_fixed": {
            "TargetManagement.maxShadowTrackingAge": 162,
            "DataAssociator.minMatchingScore4VisualSimilarity": 0.0520394823204932,
            "MultiViewAssociator.minPeerTrackletMatchScore": 0.48,
        },
        "production_modified": False,
        "production_accepted": False,
    }
    out = runner.parent / "nvdcf-retention-v8-early2-runner-patch.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
