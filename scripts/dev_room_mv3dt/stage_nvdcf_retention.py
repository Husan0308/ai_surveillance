#!/usr/bin/env python3
"""Stage the YOLO26m NvDCF retention candidate without touching production files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = (ROOT / ".runtime").resolve()
CANDIDATE = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention.yml"
PRODUCTION_TRACKER = ROOT / "config/mv3dt_dev_room/config_tracker.yml"
YOLO_CONFIG = ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_under_runtime(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(RUNTIME_ROOT)
    except ValueError as exc:
        raise SystemExit(f"refusing to stage outside {RUNTIME_ROOT}: {resolved}") from exc
    return resolved


def verify_detector_contract() -> None:
    text = YOLO_CONFIG.read_text()
    required = (
        "network-mode=2",
        "interval=0",
        "infer-dims=3;640;640",
        "pre-cluster-threshold=0.25",
        "nms-iou-threshold=0.45",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise SystemExit(f"YOLO26m detector contract changed; missing: {missing}")


def enable_shadow_diagnostics(tracker: Path, app_config: Path) -> None:
    text = tracker.read_text()
    needle = "  outputShadowTracks: 0"
    if needle not in text:
        raise SystemExit("candidate tracker does not contain outputShadowTracks: 0")
    tracker.write_text(text.replace(needle, "  outputShadowTracks: 1", 1))

    if not app_config.exists():
        raise SystemExit(f"diagnostics require staged app config: {app_config}")
    app = app_config.read_text()
    if "shadow-track-output-dir=" not in app:
        marker = "[application]\n"
        if marker not in app:
            raise SystemExit("staged config_deepstream.txt has no [application] section")
        app = app.replace(marker, marker + "shadow-track-output-dir=shadow-track-dump\n", 1)
        app_config.write_text(app)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True, type=Path)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()

    verify_detector_contract()
    stage = require_under_runtime(args.stage_dir)
    stage.mkdir(parents=True, exist_ok=True)

    production_before = sha256(PRODUCTION_TRACKER)
    candidate_hash = sha256(CANDIDATE)

    staged_tracker = stage / "config_tracker.yml"
    shutil.copy2(CANDIDATE, staged_tracker)

    app_config = stage / "config_deepstream.txt"
    if args.diagnostics:
        enable_shadow_diagnostics(staged_tracker, app_config)

    production_after = sha256(PRODUCTION_TRACKER)
    if production_before != production_after:
        raise SystemExit("production tracker changed while staging; abort")

    manifest = {
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": candidate_hash,
        "staged_tracker": str(staged_tracker),
        "staged_tracker_sha256": sha256(staged_tracker),
        "production_tracker": str(PRODUCTION_TRACKER.relative_to(ROOT)),
        "production_tracker_sha256": production_after,
        "diagnostics": bool(args.diagnostics),
        "fixed_detector_contract": {
            "model": "YOLO26m",
            "network_mode": "FP16",
            "input": "640x640",
            "confidence": 0.25,
            "nms_iou": 0.45,
            "interval": 0,
        },
        "tracker_delta": {
            "minTrackerConfidence": 0.20,
            "probationAge": 2,
            "tentativeDetectorConfidence": 0.25,
        },
        "held_fixed": {
            "earlyTerminationAge": 1,
            "maxShadowTrackingAge": 162,
            "minIouDiff4NewTarget": 0.22656630527418112,
            "dataAssociationOverallMinimum": 0.4,
        },
    }
    (stage / "nvdcf_retention_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
