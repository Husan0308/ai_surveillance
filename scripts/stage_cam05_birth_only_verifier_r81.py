#!/usr/bin/env python3
"""Stage r81 with the optimized r80 binary and optional shadow refresh disabled."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-final-20261008-r80-fast-verifier-current-source"
OUT = ROOT / ".runtime/sixcam-cam05-final-20261008-r81-birth-only-verifier"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {OUT}")
    if not OUT.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate must stay under .runtime")
    parent_path = PARENT / "build-manifest.json"
    manifest = json.loads(parent_path.read_text())
    for binary, key in (
        ("cam-six-yolo-local6-posttracker", "binary_sha256"),
        ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256"),
    ):
        if sha256(PARENT / "bin" / binary) != manifest.get(key):
            raise RuntimeError(f"parent binary mismatch: {binary}")
    shutil.copytree(PARENT, OUT)
    manifest["candidate_label"] = OUT.name + ": exact r80 source/binaries with diagnostic shadow refresh disabled"
    manifest["shadow_context_refresh_enabled"] = False
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "change": "disable optional per-frame shadow-context refresh; preserve verifier for active births and low-confidence sidecars",
        "invariants": [
            "binary, primary/auxiliary models/configs, parser, and tracker are byte-identical to r80",
            "normal >=0.25 primary detections and NvDCF input unchanged",
            "active track birth verifier remains enabled by runtime harness",
            "shadow geometry is not rendered and no identity is added",
        ],
    }
    (OUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "STAGED_ISOLATED_DIAGNOSTIC_PROFILE",
        "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "shadow_context_refresh_enabled": False,
        "primary_engine_sha256": manifest["detector_profile"]["primary_engine_sha256"],
        "tracker_config_sha256": manifest["tracker_config_sha256"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
