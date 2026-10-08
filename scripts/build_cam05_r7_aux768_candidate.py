#!/usr/bin/env python3
"""Stage r7 using the existing batch-6 768px auxiliary engine at r6 cadence.

This is an isolated latency/recall candidate.  It copies r6 and swaps only the
supplemental detector assets/configuration from the previously validated r50
diagnostic package; primary YOLO, V13/local NvDCF, verifier, and native binary
remain byte-identical.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r6-verifier-admission"
AUX768_SOURCE = ROOT / ".runtime/sixcam-cam05-eligible-lowconf-aux768-interval1-20261008-r50"
OUT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r7b-aux768-interval2"


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
    aux_source = json.loads((AUX768_SOURCE / "build-manifest.json").read_text())
    parent_profile = parent["detector_profile"]
    source_profile = aux_source["detector_profile"]

    # This A/B is deliberately limited to auxiliary input geometry.  Both
    # profiles use the same frozen primary and the same 448 context verifier.
    if parent_profile.get("aux_interval") != source_profile.get("aux_interval") + 1:
        raise RuntimeError("r6 aux interval must be exactly one frame slower than r50 (2 vs 1)")
    for key in ("primary_engine_sha256", "primary_parser_sha256", "primary_threshold",
                "normal_nvdcf_input_threshold", "aux_unique_id", "aux_threshold",
                "aux_nms_iou", "batch_size", "metamux_pts_tolerance_us"):
        if parent_profile.get(key) != source_profile.get(key):
            raise RuntimeError(f"primary/aux contract differs between parents: {key}")
    if parent_profile.get("aux_interval") != 2:
        raise RuntimeError("r7 must retain the existing auxiliary inference cadence")
    if (parent.get("yolo_context_verifier") != aux_source.get("yolo_context_verifier")
            and parent.get("yolo_context_verifier", {}).get("engine_sha256")
            != aux_source.get("yolo_context_verifier", {}).get("engine_sha256")):
        raise RuntimeError("the 448 verifier package must remain identical")

    shutil.copytree(PARENT, OUT)
    config_name = "config_infer_aux_yolo26m_768.txt"
    metamux_name = "config_metamux_yolo26m_aux768.txt"
    engine_name = "yolo26m_raw_otm_768_b6_fp16.engine"
    onnx_name = "yolo26m_raw_otm_768.onnx"
    parser_name = "libyolo26rawperson_768.so"
    assets = (
        (AUX768_SOURCE / "config" / config_name, OUT / "config" / config_name),
        (AUX768_SOURCE / "config" / metamux_name, OUT / "config" / metamux_name),
        (AUX768_SOURCE / "models" / engine_name, OUT / "models" / engine_name),
        (AUX768_SOURCE / "models" / onnx_name, OUT / "models" / onnx_name),
        (AUX768_SOURCE / "models" / parser_name, OUT / "models" / parser_name),
    )
    for source, target in assets:
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)
    aux_config_path = OUT / "config" / config_name
    aux_config_text = aux_config_path.read_text()
    if aux_config_text.count("interval=1\n") != 1:
        raise RuntimeError("768 source config cadence is not the expected interval=1")
    # Match r6 exactly: DeepStream interval=2 (infer one, skip two).
    aux_config_path.write_text(aux_config_text.replace("interval=1\n", "interval=2\n", 1))

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    profile = dict(parent_profile)
    profile.update({
        "name": "yolo26m-fp16-640-primary-768-aux-interval-2-context-verified",
        "candidate_only": True,
        "aux_config": config_name,
        "aux_engine_sha256": sha256(OUT / "models" / engine_name),
        "aux_parser_sha256": sha256(OUT / "models" / parser_name),
        "aux_input": [768, 768],
        "aux_interval": 2,
        "metamux_config": metamux_name,
        "aux_config_sha256": sha256(OUT / "config" / config_name),
        "aux_engine": engine_name,
        "aux_onnx": onnx_name,
        "aux_parser": parser_name,
        "aux_onnx_sha256": sha256(OUT / "models" / onnx_name),
        "aux_environment_contract": "generic-v1",
        "canonical_birth_safety": (
            "all newly born 640-primary and 768-auxiliary native targets are context checked "
            "using the frozen-640 YOLO crop verifier before publication; rejected targets are "
            "withheld without changing NvDCF input"
        ),
    })
    manifest["candidate_label"] = (
        "sixcam-cam05-verified-track-20261008-r7-aux768-interval2: "
        "r6 behavior with only the supplemental detector spatial input reduced to 768px"
    )
    manifest["detector_profile"] = profile
    manifest.setdefault("config_sha256", {})[config_name] = sha256(aux_config_path)
    manifest["config_sha256"][metamux_name] = sha256(OUT / "config" / metamux_name)
    manifest.setdefault("model_sha256", {})[engine_name] = sha256(OUT / "models" / engine_name)
    manifest["model_sha256"][onnx_name] = sha256(OUT / "models" / onnx_name)
    manifest["model_sha256"][parser_name] = sha256(OUT / "models" / parser_name)
    manifest["build_provenance"] = {
        **manifest.get("build_provenance", {}),
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "auxiliary_assets_source": str(AUX768_SOURCE.relative_to(ROOT)),
        "change": "replace only supplemental 896x896 FP16 engine with existing 768x768 FP16 engine; preserve interval=2 and every primary/tracker/verifier behavior",
        "primary_nvdcf_input_changed": False,
        "tracker_changed": False,
        "verifier_changed": False,
        "native_binary_sha256_unchanged": manifest["binary_sha256"],
    }
    manifest["recovery"] = {
        **manifest.get("recovery", {}),
        "candidate_binary_sha256": manifest["binary_sha256"],
        "status": "REQUIRES_REVALIDATION_ON_EXACT_BINARY_AND_PROFILE",
        "validated_cycles": 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "STAGED_DIAGNOSTIC_CANDIDATE",
        "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": manifest["binary_sha256"],
        "primary_engine_sha256": profile["primary_engine_sha256"],
        "aux_engine_sha256": profile["aux_engine_sha256"],
        "aux_interval": profile["aux_interval"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
