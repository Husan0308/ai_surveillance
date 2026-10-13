#!/usr/bin/env python3
"""Stage an isolated r5-derived sparse 768 auxiliary detector profile.

This candidate keeps the exact r5 native binaries, 640 primary detector,
NvDCF, canonical policy, verifier, and preview path. Only the supplemental
detector artifact/input cadence changes to the previously reviewed 768 model
running once every 20 mux batches.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r5-ui-coordinate-contract"
AUX_SOURCE = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r17-aux768-sparse20"
OUT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r6b-aux768-sparse20"

EXPECTED_PARENT = {
    "binary_sha256": "3f76f807a8401e35fa33a3811ef612f70af78f87dd51101580c4399c9e409a07",
    "direct_file_binary_sha256": "1990d66c7dabf1fa7f51296b9465ee992c752ad6f14478758d0bbb3d4f8ec4d7",
    "primary_engine_sha256": "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16",
    "primary_parser_sha256": "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19",
    "tracker_sha256": "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355",
    "v13_sha256": "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834",
}
AUX_FILES = (
    "yolo26m_raw_otm_768.onnx",
    "yolo26m_raw_otm_768_b6_fp16.engine",
    "libyolo26rawperson_768.so",
)
AUX_CONFIG = "config_infer_aux_yolo26m_768.txt"
METAMUX_CONFIG = "config_metamux_yolo26m_aux768.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hashes(paths: list[Path]) -> dict[str, str]:
    return {path.name: sha256(path) for path in paths}


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite staged candidate: {OUT}")
    if not OUT.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate must remain under .runtime")

    parent = json.loads((PARENT / "build-manifest.json").read_text())
    profile = parent.get("detector_profile", {})
    actual = {
        "binary_sha256": sha256(PARENT / "bin/cam-six-yolo-local6-posttracker"),
        "direct_file_binary_sha256": sha256(PARENT / "bin/cam-six-yolo-local6-direct-gatea"),
        "primary_engine_sha256": profile.get("primary_engine_sha256"),
        "primary_parser_sha256": profile.get("primary_parser_sha256"),
        "tracker_sha256": parent.get("tracker_config_sha256"),
        "v13_sha256": parent.get("frozen_v13_reference_sha256"),
    }
    if actual != EXPECTED_PARENT:
        raise RuntimeError(f"r5 protected identity mismatch: {actual}")
    if profile.get("aux_input") != [896, 896] or profile.get("aux_interval") != 2:
        raise RuntimeError("r5 parent is not the expected interval-2 896 profile")

    aux_manifest = json.loads((AUX_SOURCE / "build-manifest.json").read_text())
    aux_profile = aux_manifest.get("detector_profile", {})
    if aux_profile.get("aux_input") != [768, 768] or aux_profile.get("aux_interval") != 19:
        raise RuntimeError("reviewed 768 auxiliary source is not the every-20-batch profile")
    if (aux_profile.get("primary_engine_sha256") != EXPECTED_PARENT["primary_engine_sha256"]
            or aux_profile.get("primary_parser_sha256") != EXPECTED_PARENT["primary_parser_sha256"]):
        raise RuntimeError("768 auxiliary source does not preserve the frozen primary detector")
    for filename in AUX_FILES:
        source = AUX_SOURCE / "models" / filename
        expected = aux_manifest.get("model_sha256", {}).get(filename)
        if not source.is_file() or sha256(source) != expected:
            raise RuntimeError(f"reviewed auxiliary artifact hash mismatch: {filename}")

    for filename, expected_values in (
        (AUX_CONFIG, ("batch-size=6", "infer-dims=3;768;768", "interval=19",
                      "gie-unique-id=3", "pre-cluster-threshold=0.25",
                      "nms-iou-threshold=0.45", "maintain-aspect-ratio=1",
                      "symmetric-padding=1")),
        (METAMUX_CONFIG, ("enable=1", "pts-tolerance=0")),
    ):
        source = AUX_SOURCE / "config" / filename
        if not source.is_file() or any(value not in source.read_text() for value in expected_values):
            raise RuntimeError(f"reviewed auxiliary configuration contract mismatch: {filename}")

    shutil.copytree(PARENT, OUT)
    models = OUT / "models"
    config = OUT / "config"
    for filename in AUX_FILES:
        shutil.copyfile(AUX_SOURCE / "models" / filename, models / filename)
    for filename in (AUX_CONFIG, METAMUX_CONFIG):
        shutil.copyfile(AUX_SOURCE / "config" / filename, config / filename)

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    staged_profile = dict(profile)
    staged_profile.update({
        "name": "yolo26m-fp16-640-primary-768-aux-interval-19-context-verified",
        "aux_config": AUX_CONFIG,
        "aux_engine": AUX_FILES[1],
        "aux_onnx": AUX_FILES[0],
        "aux_parser": AUX_FILES[2],
        "aux_engine_sha256": sha256(models / AUX_FILES[1]),
        "aux_onnx_sha256": sha256(models / AUX_FILES[0]),
        "aux_parser_sha256": sha256(models / AUX_FILES[2]),
        "aux_input": [768, 768],
        "aux_interval": 19,
        "aux_unique_id": 3,
        "aux_threshold": 0.25,
        "aux_nms_iou": 0.45,
        "aux_config_sha256": sha256(config / AUX_CONFIG),
        "metamux_config": METAMUX_CONFIG,
        "metamux_pts_tolerance_us": 0,
        "canonical_birth_safety": (
            "newly born 640-primary and 768-auxiliary native targets require the frozen-640 "
            "current-frame context verifier before canonical publication, without changing NvDCF input"
        ),
        "cadence_note": (
            "one supplemental YOLO26m 768 batch-6 inference per 20 mux batches; "
            "frozen 640 primary, >=0.25 NvDCF input, and exact-frame sidecar policy unchanged"
        ),
    })
    manifest["detector_profile"] = staged_profile
    manifest["config_sha256"].update(hashes(list(config.iterdir())))
    manifest["model_sha256"].update({
        str(path.relative_to(models)): sha256(path)
        for path in models.rglob("*") if path.is_file()
    })
    manifest["candidate_label"] = (
        f"{OUT.name}: exact r5 native binaries and 640 primary with reviewed 768 auxiliary every 20 batches"
    )
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": EXPECTED_PARENT["binary_sha256"],
        "change": "supplemental detector only: 896 interval=2 replaced by the reviewed 768 interval=19 model/config",
        "invariants": [
            "exact r5 live/direct-file native binaries retained",
            "frozen YOLO26m 640 primary engine/parser/config/threshold/preprocessing unchanged",
            "normal >=0.25 primary NvDCF input unchanged",
            "NvDCF, canonical policy, verifier, preview geometry, and source recovery code unchanged",
            "auxiliary output remains exact-frame sidecar evidence and never replaces primary proposals",
            "frozen V13 and calibration unchanged",
        ],
    }
    manifest["recovery"] = {
        "required_environment": "V11_SOURCE_STALL_RECOVERY=1",
        "candidate_binary_sha256": EXPECTED_PARENT["binary_sha256"],
        "validated_cycles": 0,
        "status": "REQUIRES_PROFILE_REVALIDATION",
        "note": "the exact r5 binary passed 3x CAM-02 + 1x CAM-03 before this isolated auxiliary profile change",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "STAGED_DIAGNOSTIC_CANDIDATE",
        "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": EXPECTED_PARENT["binary_sha256"],
        "direct_file_binary_sha256": EXPECTED_PARENT["direct_file_binary_sha256"],
        "primary_engine_sha256": staged_profile["primary_engine_sha256"],
        "aux_engine_sha256": staged_profile["aux_engine_sha256"],
        "aux_input": staged_profile["aux_input"],
        "aux_interval": staged_profile["aux_interval"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
