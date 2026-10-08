#!/usr/bin/env python3
"""Stage an isolated candidate with a low-duty-cycle supplemental 896 scan.

The frozen 640 PGIE and its NvDCF input are unchanged. The supplemental GIE
continues to run in the same DeepStream pipeline and remains exact-frame
joined; it scans once per 20 batched frames to lower contention while keeping
the already reviewed CAM-05 small-person recovery path available.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r14-exact-sidecar"
OUT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r15-aux-sparse20"
PARENT_BINARY_SHA256 = "76550d7dba4cc57025129724a1d2d8804211e2d18a3ba440f0d648fa09b29ed4"
PRIMARY_ENGINE_SHA256 = "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"
PRIMARY_PARSER_SHA256 = "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19"
LOCAL_TRACKER_SHA256 = "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355"
FROZEN_V13_SHA256 = "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834"
AUX_INTERVAL = 19  # DeepStream interval counts skipped batches; scan every 20.
AUX_CONFIG_NAME = "config_infer_aux_yolo26m_896.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def set_aux_interval(config_text: str, interval: int) -> str:
    if not 0 <= interval <= 29:
        raise ValueError("auxiliary interval must be in the validated DeepStream range 0..29")
    old = "interval=4"
    if config_text.count(old) != 1:
        raise ValueError("expected exactly one r14 interval=4 setting")
    return config_text.replace(old, f"interval={interval}", 1)


def updated_profile(parent_profile: dict, interval: int) -> dict:
    profile = dict(parent_profile)
    if (profile.get("primary_engine_sha256") != PRIMARY_ENGINE_SHA256
            or profile.get("primary_parser_sha256") != PRIMARY_PARSER_SHA256
            or profile.get("normal_nvdcf_input_threshold") != 0.25
            or profile.get("aux_input") != [896, 896]
            or profile.get("aux_interval") != 4
            or profile.get("aux_join_mode") != "exact-key-sidecar-v1"
            or profile.get("metamux_pts_tolerance_us") != 0):
        raise ValueError("r14 parent detector contract is not the expected frozen profile")
    if not 0 <= interval <= 29:
        raise ValueError("auxiliary interval must be in the validated DeepStream range 0..29")
    profile["name"] = "yolo26m-fp16-640-primary-896-aux-interval-19-context-verified"
    profile["aux_interval"] = interval
    profile["cadence_note"] = (
        "one supplemental 896x896 batch-6 inference per 20 source batches; the frozen 640 "
        "PGIE still runs every frame, >=0.25 detections remain the only primary NvDCF input, "
        "auxiliary results remain exact source/frame/PTS joined and fail closed when late"
    )
    return profile


def stage_candidate() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite isolated candidate: {OUT}")
    parent_manifest_path = PARENT / "build-manifest.json"
    parent = json.loads(parent_manifest_path.read_text())
    if parent.get("binary_sha256") != PARENT_BINARY_SHA256:
        raise ValueError("r14 candidate binary does not match the validated parent SHA256")
    if sha256(PARENT / "bin/cam-six-yolo-local6-posttracker") != PARENT_BINARY_SHA256:
        raise ValueError("r14 native binary does not match its recorded hash")
    direct_hash = parent.get("direct_file_binary_sha256")
    if not direct_hash or sha256(PARENT / "bin/cam-six-yolo-local6-direct-gatea") != direct_hash:
        raise ValueError("r14 direct-file binary does not match its recorded hash")
    if parent.get("tracker_config_sha256") != LOCAL_TRACKER_SHA256:
        raise ValueError("local NvDCF profile changed")
    if parent.get("frozen_v13_reference_sha256") != FROZEN_V13_SHA256:
        raise ValueError("frozen V13 reference changed")

    shutil.copytree(PARENT, OUT)
    aux_config = OUT / "config" / AUX_CONFIG_NAME
    aux_config.write_text(set_aux_interval(aux_config.read_text(), AUX_INTERVAL))
    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    profile = updated_profile(manifest["detector_profile"], AUX_INTERVAL)
    manifest["detector_profile"] = profile
    manifest.setdefault("config_sha256", {})[AUX_CONFIG_NAME] = sha256(aux_config)
    manifest["candidate_label"] = (
        "r15 candidate: exact r14 runtime/binary and approved verifier policy; "
        "only supplemental 896 GIE cadence changes from every 5 to every 20 source batches"
    )
    manifest["build_provenance"] = {
        **manifest.get("build_provenance", {}),
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "change": "auxiliary inference interval only: 4 -> 19; primary PGIE, NvDCF input, "
                  "exact join, canonical policy, preview, and recovery implementation unchanged",
        "native_binary_sha256_unchanged": True,
        "primary_nvdcf_input_changed": False,
        "recovery_code_changed": False,
    }
    manifest["recovery"] = {
        "required_environment": "V11_SOURCE_STALL_RECOVERY=1",
        "candidate_binary_sha256": manifest["binary_sha256"],
        "parent_binary_sha256": PARENT_BINARY_SHA256,
        "validated_cycles": 4,
        "status": "PARENT_4_OF_4_REUSED; SOURCE_RECOVERY_CODE_AND_BINARY_IDENTICAL",
        "note": "reuse is valid only for the unchanged recovery binary/code; profile change is auxiliary inference cadence only",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": direct_hash,
        "aux_interval": profile["aux_interval"],
        "primary_engine_sha256": profile["primary_engine_sha256"],
        "primary_parser_sha256": profile["primary_parser_sha256"],
        "aux_engine_sha256": profile["aux_engine_sha256"],
        "tracker_config_sha256": manifest["tracker_config_sha256"],
        "frozen_v13_reference_sha256": manifest["frozen_v13_reference_sha256"],
    }


if __name__ == "__main__":
    print(json.dumps(stage_candidate(), indent=2))
