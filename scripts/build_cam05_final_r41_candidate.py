#!/usr/bin/env python3
"""Build an isolated CAM-05/duplicate-safe candidate from the current r23 profile.

The frozen 640 primary remains the only input to normal NvDCF. A denser 896
supplemental pass and the existing exact-frame 448 verifier recover the reviewed
CAM-05 seated-person interval; the generic same-GIE nested-duplicate guard runs
before NvDCF. All artifacts are staged under .runtime and prior runs are kept.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r23-final-aux768-sparse"
AUX_PARENT = ROOT / ".runtime/sixcam-cam05-aux448-no-shadow-refresh-20261008-r40"
OUT = ROOT / ".runtime/sixcam-cam05-final-20261008-r42-aux896-interval2-dedupe"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"

PRIMARY_ENGINE = "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"
PRIMARY_PARSER = "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19"
LOCAL_TRACKER = "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355"
V13_REFERENCE = "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834"
CONTEXT_ENGINE = "5bf1c545363b5cc7606c701eabae580011b5a9fd5d27011507fd3bc565aebd83"
CONTEXT_CONFIG = "fb423f88c6dde3ec3b8f3ad274e7093898124b4d078aa65dfa7dbb70c5a0e807"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {p.name: sha256(p) for p in sorted(SOURCE.iterdir()) if p.is_file()}


def compile_binary(name: str, *, direct_file: bool) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER -DYOLO26_AUXILIARY_DETECTOR"
    if direct_file:
        macros += " -DPGIE_ISOLATION"
    compile_command = (
        "g++ -std=c++17 -O2 " + macros
        + " -I/host-cuda-include -I/opt/nvidia/deepstream/deepstream/sources/includes"
        + " $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " /src/main.cpp -o /work/bin/" + name
        + " $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta"
        + " -lnvbufsurface -lnvbufsurftransform -lnvds_infer -lnvinfer"
        + " -lnvinfer_plugin -L/usr/local/cuda/lib64 -lcudart -ldl -pthread"
        + " -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    subprocess.run([
        "docker", "run", "--rm", "--pull=never", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{OUT}:/work",
        "-v", "/usr/local/cuda/include:/host-cuda-include:ro", IMAGE,
        "bash", "-lc", compile_command,
    ], cwd=ROOT, check=True)


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {OUT}")
    if not OUT.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate must be isolated under .runtime")
    parent_manifest = json.loads((PARENT / "build-manifest.json").read_text())
    aux_manifest = json.loads((AUX_PARENT / "build-manifest.json").read_text())
    if parent_manifest.get("deepstream_image") != IMAGE or aux_manifest.get("deepstream_image") != IMAGE:
        raise RuntimeError("candidate parent does not use pinned DeepStream 9.1 image")
    for filename, field in (
        ("cam-six-yolo-local6-posttracker", "binary_sha256"),
        ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256"),
    ):
        if sha256(PARENT / "bin" / filename) != parent_manifest.get(field):
            raise RuntimeError(f"r23 binary does not match manifest: {filename}")
    if (parent_manifest["detector_profile"]["primary_engine_sha256"] != PRIMARY_ENGINE
            or parent_manifest["detector_profile"]["primary_parser_sha256"] != PRIMARY_PARSER
            or parent_manifest["tracker_config_sha256"] != LOCAL_TRACKER
            or parent_manifest["frozen_v13_reference_sha256"] != V13_REFERENCE):
        raise RuntimeError("r23 protected detector/tracker provenance differs")
    if (aux_manifest["detector_profile"].get("aux_interval") != 2
            or aux_manifest["yolo_context_verifier"].get("engine_sha256") != CONTEXT_ENGINE
            or aux_manifest["yolo_context_verifier"].get("config_sha256") != CONTEXT_CONFIG):
        raise RuntimeError("r40 auxiliary/verifier profile differs from reviewed candidate")
    if not (SOURCE / "person_proposal_dedupe.hpp").is_file():
        raise RuntimeError("current native source is missing the tested duplicate guard")

    shutil.copytree(PARENT, OUT)
    models = OUT / "models"
    configs = OUT / "config"
    aux_config_name = "config_infer_aux_yolo26m_896.txt"
    metamux_name = "config_metamux_yolo26m_aux896.txt"
    for filename in ("yolo26m_raw_otm_896.onnx", "yolo26m_raw_otm_896_b6_fp16.engine",
                     "libyolo26rawperson_896.so"):
        source = AUX_PARENT / "models" / filename
        expected = aux_manifest["model_sha256"].get(filename)
        if not source.is_file() or sha256(source) != expected:
            raise RuntimeError(f"reviewed 896 artifact mismatch: {filename}")
        shutil.copyfile(source, models / filename)
    for filename in (aux_config_name, metamux_name):
        source = AUX_PARENT / "config" / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, configs / filename)
    if "interval=2" not in (configs / aux_config_name).read_text():
        raise RuntimeError("supplemental inference cadence is not the reviewed interval-2 profile")
    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True)

    manifest = json.loads((OUT / "build-manifest.json").read_text())
    profile = dict(manifest["detector_profile"])
    profile.update({
        "name": "yolo26m-fp16-640-primary-896-aux-interval-2-context-verified-deduped",
        "aux_config": aux_config_name,
        "aux_engine": "yolo26m_raw_otm_896_b6_fp16.engine",
        "aux_onnx": "yolo26m_raw_otm_896.onnx",
        "aux_parser": "libyolo26rawperson_896.so",
        "aux_engine_sha256": aux_manifest["detector_profile"]["aux_engine_sha256"],
        "aux_onnx_sha256": aux_manifest["model_sha256"]["yolo26m_raw_otm_896.onnx"],
        "aux_parser_sha256": aux_manifest["detector_profile"]["aux_parser_sha256"],
        "aux_input": [896, 896],
        "aux_interval": 2,
        "aux_unique_id": 3,
        "batch_size": 6,
        "metamux_config": metamux_name,
        "metamux_pts_tolerance_us": 0,
        "aux_environment_contract": "generic-v1",
        "aux_preprocess_gate": {"enabled": False, "reason": "interval-2 nvinfer cadence; no branch-local drop gate"},
        "canonical_birth_safety": (
            "newly born 640-primary and 896-auxiliary native targets require the frozen-640 "
            "current-frame context verifier before canonical publication, without changing NvDCF input"
        ),
    })
    manifest["candidate_label"] = OUT.name + ": current r23 source + interval-2 896 recall + 448 verifier + strict same-GIE nested dedupe"
    manifest["detector_profile"] = profile
    manifest["same_component_nested_duplicate_policy"] = {
        "enabled": True,
        "placement": "after per-GIE NMS and before NvDCF input",
        "component_scope": "same class-0 unique_component_id only",
        "iou_min": 0.30,
        "smaller_box_containment_min": 0.95,
        "center_distance_over_max_height_max": 0.25,
        "width_ratio_max": 2.0,
        "height_ratio_max": 2.0,
        "confidence_margin_min": 0.05,
        "unique_higher_confidence_match_required": True,
        "ambiguous_match_policy": "retain all candidates",
        "retained_proposal": "higher detector confidence",
    }
    manifest["source_sha256"] = source_hashes()
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["config_sha256"] = {p.name: sha256(p) for p in sorted(configs.iterdir()) if p.is_file()}
    manifest["model_sha256"] = {
        str(p.relative_to(models)): sha256(p) for p in sorted(models.rglob("*")) if p.is_file()
    }
    manifest["shadow_context_refresh_enabled"] = False
    manifest["runtime"] = (
        "DeepStream 9.1; frozen YOLO26m FP16 640 batch-6 primary; 896 supplemental batch-6 interval=2; "
        "only primary >=0.25 enters normal NvDCF; 448 current-frame context verification gates canonical "
        "birth/low-confidence output; exact source/frame/PTS joins; same-GIE nested duplicate guard before "
        "NvDCF; local NvDCF; no MV3DT/calibration"
    )
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": parent_manifest["binary_sha256"],
        "auxiliary_profile_source": str(AUX_PARENT.relative_to(ROOT)),
        "change": "use the reviewed interval-2 896 supplemental profile on the current r23 native source, including its already-tested generic same-GIE nested-duplicate guard; disable optional shadow refresh in the replay/live runner",
        "invariants": [
            "primary YOLO engine/parser/config and >=0.25 NvDCF input semantics unchanged",
            "local NvDCF and frozen V13 hashes unchanged",
            "auxiliary results remain exact source/frame/PTS sidecar metadata and never replace primary proposals",
            "duplicate guard only removes a unique strictly nested same-component lower-confidence pair",
            "ambiguous/close-person pairs fail open",
            "preview geometry, exact image/metadata join, and recovery source code retained",
        ],
    }
    manifest["recovery"] = {
        "required_environment": "V11_SOURCE_STALL_RECOVERY=1",
        "candidate_binary_sha256": manifest["binary_sha256"],
        "validated_cycles": 0,
        "status": "REQUIRES_REVALIDATION_ON_EXACT_BINARY_AND_PROFILE",
    }
    (OUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_ISOLATED_CANDIDATE_REQUIRES_VALIDATION",
        "candidate": str(OUT.relative_to(ROOT)),
        "live_binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "primary_engine_sha256": PRIMARY_ENGINE,
        "primary_parser_sha256": PRIMARY_PARSER,
        "auxiliary_engine_sha256": profile["aux_engine_sha256"],
        "context_engine_sha256": CONTEXT_ENGINE,
        "tracker_sha256": LOCAL_TRACKER,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
