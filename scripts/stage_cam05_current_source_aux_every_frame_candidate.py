#!/usr/bin/env python3
"""Stage a current-source, interval-0 auxiliary replay candidate.

This is a diagnostic profile-only derivative: the already rebuilt r7 binaries,
640 primary, parsers, engines, local NvDCF profile, and verifier are preserved.
Only the manifested 896 auxiliary nvinfer config changes interval 2 -> 0.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r7-current-source"
REFERENCE = ROOT / ".runtime/sixcam-cam05-final-20261008-r49-aux896-every-frame"
OUT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r8-aux896-every-frame"
CONFIG_NAME = "config_infer_aux_yolo26m_896_everyframe.txt"
EXPECTED_ENGINE = "44ae13c7bc8388b530509d68fb34b74c85fc2c5d1f7248f937ef506947523fd4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite isolated candidate: {OUT}")
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    reference = json.loads((REFERENCE / "build-manifest.json").read_text())
    profile = parent.get("detector_profile", {})
    ref_profile = reference.get("detector_profile", {})
    if (profile.get("aux_interval") != 2 or profile.get("aux_input") != [896, 896]
            or profile.get("aux_engine_sha256") != EXPECTED_ENGINE
            or ref_profile.get("aux_interval") != 0
            or ref_profile.get("aux_engine_sha256") != EXPECTED_ENGINE):
        raise RuntimeError("parent/reference auxiliary detector contracts are not the expected 896 interval variants")
    for name, key in (("cam-six-yolo-local6-posttracker", "binary_sha256"),
                      ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256")):
        if sha256(PARENT / "bin" / name) != parent.get(key):
            raise RuntimeError(f"current-source parent binary hash mismatch: {name}")
    source_config = REFERENCE / "config" / CONFIG_NAME
    if not source_config.is_file() or sha256(source_config) != reference["config_sha256"][CONFIG_NAME]:
        raise RuntimeError("interval-0 auxiliary config does not match its preserved reference manifest")
    if "interval=0" not in source_config.read_text():
        raise RuntimeError("interval-0 auxiliary config has no interval=0 setting")

    shutil.copytree(PARENT, OUT)
    target_config = OUT / "config" / CONFIG_NAME
    shutil.copy2(source_config, target_config)
    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = sha256(target_config)
    manifest["config_sha256"][CONFIG_NAME] = digest
    manifest["detector_profile"]["aux_config"] = CONFIG_NAME
    manifest["detector_profile"]["aux_interval"] = 0
    manifest["detector_profile"]["aux_config_sha256"] = digest
    manifest["detector_profile"]["cadence_note"] = (
        "Diagnostic every-frame 896 auxiliary inference; primary YOLO26m 640 output and NvDCF input are unchanged."
    )
    manifest["candidate_label"] = f"{OUT.name}: current-source r7 binaries, 896 auxiliary interval=0 diagnostic"
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "reference_aux_config": str((REFERENCE / "config" / CONFIG_NAME).relative_to(ROOT)),
        "change": "896 auxiliary nvinfer interval 2 -> 0 only",
        "runtime_binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "primary_engine_parser_threshold_nvdcf_and_verifier_unchanged": True,
        "diagnostic_only": True,
    }
    (OUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "STAGED_DIAGNOSTIC_ONLY",
        "candidate": str(OUT.relative_to(ROOT)),
        "runtime_binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "aux_interval": 0,
        "aux_config_sha256": digest,
        "aux_engine_sha256": EXPECTED_ENGINE,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
