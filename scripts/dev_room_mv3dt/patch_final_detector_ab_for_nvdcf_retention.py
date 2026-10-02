#!/usr/bin/env python3
"""Patch the archived final detector A/B runner to stage the NvDCF retention candidate.

This only edits the gitignored .runtime experiment runner. Production configs are
never modified. The original runner is backed up before any change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = ROOT / ".runtime/final-detector-ab-20261002-uSpPqjLY/run_candidate.py"

OLD = "YOLO_DELTA = {'DataAssociator.tentativeDetectorConfidence': .40}"
NEW = (
    "YOLO_DELTA = {"
    "'TargetManagement.minTrackerConfidence': .20, "
    "'TargetManagement.probationAge': 2, "
    "'DataAssociator.tentativeDetectorConfidence': .25"
    "}"
)

PRODUCTION_TRACKER = ROOT / "config/mv3dt_dev_room/config_tracker.yml"
CANDIDATE_TRACKER = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention.yml"


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
    runtime = (ROOT / ".runtime").resolve()
    try:
        runner.relative_to(runtime)
    except ValueError as exc:
        raise SystemExit(f"refusing to edit non-runtime file: {runner}") from exc
    if not runner.is_file():
        raise SystemExit(f"runner not found: {runner}")

    production_before = sha256(PRODUCTION_TRACKER)
    candidate_text = CANDIDATE_TRACKER.read_text()
    required_candidate = (
        "minTrackerConfidence: 0.20",
        "probationAge: 2",
        "tentativeDetectorConfidence: 0.25",
    )
    missing = [x for x in required_candidate if x not in candidate_text]
    if missing:
        raise SystemExit(f"tracked retention candidate drifted; missing: {missing}")

    before = runner.read_text()
    if NEW in before:
        print("runner already patched")
        return 0
    if before.count(OLD) != 1:
        raise SystemExit(
            "archived runner contract changed; expected exactly one original YOLO_DELTA"
        )

    backup = runner.with_suffix(runner.suffix + ".pre-nvdcf-retention")
    if not backup.exists():
        shutil.copy2(runner, backup)

    before_hash = sha256(runner)
    runner.write_text(before.replace(OLD, NEW, 1))
    after_hash = sha256(runner)

    if sha256(PRODUCTION_TRACKER) != production_before:
        shutil.copy2(backup, runner)
        raise SystemExit("production tracker changed during runtime patch; restored runner")

    manifest = {
        "runner": str(runner),
        "backup": str(backup),
        "runner_sha256_before": before_hash,
        "runner_sha256_after": after_hash,
        "production_tracker_sha256": production_before,
        "tracked_candidate_sha256": sha256(CANDIDATE_TRACKER),
        "yolo_tracker_delta": {
            "TargetManagement.minTrackerConfidence": 0.20,
            "TargetManagement.probationAge": 2,
            "DataAssociator.tentativeDetectorConfidence": 0.25,
        },
        "production_modified": False,
    }
    manifest_path = runner.parent / "nvdcf-retention-runner-patch.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
