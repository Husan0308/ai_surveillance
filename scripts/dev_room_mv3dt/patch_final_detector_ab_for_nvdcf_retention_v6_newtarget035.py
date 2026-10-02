#!/usr/bin/env python3
"""Patch archived detector A/B runner from v5 shadow diagnostic to v6 target-creation diagnostic.

v6 restores maxShadowTrackingAge=162 and changes one target-creation control:
  TargetManagement.minIouDiff4NewTarget: 0.22656630527418112 -> 0.35

The archived runner lives under .runtime and is gitignored. Production configs
are never modified. This remains diagnostic only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = ROOT / ".runtime/final-detector-ab-20261002-uSpPqjLY/run_candidate.py"

V4 = (
    "YOLO_DELTA = {"
    "'TargetManagement.minTrackerConfidence': .20, "
    "'TargetManagement.probationAge': 2, "
    "'DataAssociator.tentativeDetectorConfidence': .25, "
    "'DataAssociator.minMatchingScore4Overall': .35, "
    "'DataAssociator.minMatchingScore4Iou': .10, "
    "'DataAssociator.minMatchingScore4SizeSimilarity': .30"
    "}"
)
V5 = (
    "YOLO_DELTA = {"
    "'TargetManagement.minTrackerConfidence': .20, "
    "'TargetManagement.probationAge': 2, "
    "'TargetManagement.maxShadowTrackingAge': 49, "
    "'DataAssociator.tentativeDetectorConfidence': .25, "
    "'DataAssociator.minMatchingScore4Overall': .35, "
    "'DataAssociator.minMatchingScore4Iou': .10, "
    "'DataAssociator.minMatchingScore4SizeSimilarity': .30"
    "}"
)
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

PRODUCTION_TRACKER = ROOT / "config/mv3dt_dev_room/config_tracker.yml"
CANDIDATE_V6 = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v6_newtarget035.yml"


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

    candidate = CANDIDATE_V6.read_text()
    required = (
        "minIouDiff4NewTarget: 0.35",
        "minTrackerConfidence: 0.20",
        "probationAge: 2",
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
        raise SystemExit(f"tracked v6 candidate drifted; missing: {missing}")

    before = runner.read_text()
    if V6 in before:
        print("runner already patched to v6")
        return 0
    if before.count(V5) == 1:
        after = before.replace(V5, V6, 1)
        migrated_from = "v5"
    elif before.count(V4) == 1:
        after = before.replace(V4, V6, 1)
        migrated_from = "v4"
    else:
        raise SystemExit("archived runner contract changed; expected v5 or v4 YOLO_DELTA")

    policy = (
        "retention-v6 newtarget035 diagnostic; shadow162 restored; "
        "v4 association held fixed; not production accepted"
    )
    for stale in (
        "retention-v5 shadow49 diagnostic; v4 association held fixed; not production accepted",
        "retention-v4 lifecycle020-2-025 overall035 iou010 size030; no identity tuning; not production accepted",
    ):
        after = after.replace(stale, policy)

    backup = runner.with_suffix(runner.suffix + ".pre-nvdcf-retention-v6")
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
        "tracked_candidate_v6_sha256": sha256(CANDIDATE_V6),
        "diagnostic_hypothesis": (
            "retain long shadow window but allow unmatched detector object to escape "
            "stale-shadow duplicate suppression sooner"
        ),
        "yolo_tracker_delta": {
            "TargetManagement.minTrackerConfidence": 0.20,
            "TargetManagement.probationAge": 2,
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
    out = runner.parent / "nvdcf-retention-v6-newtarget035-runner-patch.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
