#!/usr/bin/env python3
"""Build a fresh 768-pixel auxiliary A/B from the exact r15 candidate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r15-aux-sparse20"
OUT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r17-aux768-sparse20"
ARTIFACTS = ROOT / ".runtime/f5-cam05-safety-20261007-r1/yolo768-candidate-r2"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
EXPECTED = {
    "engine": "84832420e6e3ad68e1664b2ceb04b38474459960990c00d76987ca8bbda153be",
    "onnx": "eef24113344008c26747ca65d7d4b9637b46193c80dadffb37396bbdbee57ace",
    "parser": "32766d0cc5855d243f9ec9c46ea959a3f88f250b6ca024b293f1b6b7cd7a949f",
}
INTERVAL = 19


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compile_binary(name: str, *, direct_file: bool) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER -DYOLO26_AUXILIARY_DETECTOR"
    if direct_file:
        macros += " -DPGIE_ISOLATION"
    command = (
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
        "bash", "-lc", command,
    ], cwd=ROOT, check=True)


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite isolated candidate: {OUT}")
    if not OUT.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate must remain under .runtime")
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    if parent.get("deepstream_image") != IMAGE:
        raise RuntimeError("r15 parent does not use the pinned DeepStream image")
    for binary_name, manifest_key in (
        ("cam-six-yolo-local6-posttracker", "binary_sha256"),
        ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256"),
    ):
        if sha256(PARENT / "bin" / binary_name) != parent.get(manifest_key):
            raise RuntimeError(f"r15 parent binary hash mismatch: {binary_name}")
    parent_profile = parent.get("detector_profile", {})
    if (parent_profile.get("aux_interval") != 19
            or parent_profile.get("aux_input") != [896, 896]
            or parent_profile.get("primary_engine_sha256")
            != "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"):
        raise RuntimeError("r15 parent does not match expected 640+896 interval-19 contract")

    for name, expected in EXPECTED.items():
        filename = {
            "engine": "yolo26m_raw_otm_768_b6_fp16.engine",
            "onnx": "yolo26m_raw_otm_768.onnx",
            "parser": "libyolo26rawperson_768.so",
        }[name]
        if sha256(ARTIFACTS / "models" / filename) != expected:
            raise RuntimeError(f"reviewed 768 artifact hash mismatch: {filename}")

    shutil.copytree(PARENT, OUT)
    models = OUT / "models"
    config_dir = OUT / "config"
    for filename in (
        "yolo26m_raw_otm_768.onnx",
        "yolo26m_raw_otm_768_b6_fp16.engine",
        "libyolo26rawperson_768.so",
    ):
        shutil.copyfile(ARTIFACTS / "models" / filename, models / filename)
    config_name = "config_infer_aux_yolo26m_768.txt"
    source_config = (ARTIFACTS / "config/config_infer_yolo26m_768_b6.txt").read_text()
    for text in ("batch-size=6", "infer-dims=3;768;768", "interval=0",
                 "pre-cluster-threshold=0.25", "nms-iou-threshold=0.45",
                 "maintain-aspect-ratio=1", "symmetric-padding=1"):
        if source_config.count(text) != 1:
            raise RuntimeError(f"reviewed 768 source config is missing unique field: {text}")
    source_config = source_config.replace("gie-unique-id=1", "gie-unique-id=3", 1)
    source_config = source_config.replace("interval=0", f"interval={INTERVAL}", 1)
    (config_dir / config_name).write_text(source_config)
    metamux_name = "config_metamux_yolo26m_aux768.txt"
    metamux_text = (PARENT / "config/config_metamux_yolo26m_aux896.txt").read_text()
    (config_dir / metamux_name).write_text(metamux_text)

    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True)

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    profile = dict(parent_profile)
    profile.update({
        "name": "yolo26m-fp16-640-primary-768-aux-interval-19-context-verified",
        "aux_config": config_name,
        "aux_engine": "yolo26m_raw_otm_768_b6_fp16.engine",
        "aux_onnx": "yolo26m_raw_otm_768.onnx",
        "aux_parser": "libyolo26rawperson_768.so",
        "aux_engine_sha256": sha256(models / "yolo26m_raw_otm_768_b6_fp16.engine"),
        "aux_onnx_sha256": sha256(models / "yolo26m_raw_otm_768.onnx"),
        "aux_parser_sha256": sha256(models / "libyolo26rawperson_768.so"),
        "aux_input": [768, 768],
        "aux_interval": INTERVAL,
        "aux_unique_id": 3,
        "aux_threshold": 0.25,
        "aux_nms_iou": 0.45,
        "batch_size": 6,
        "metamux_config": metamux_name,
        "metamux_pts_tolerance_us": 0,
        "aux_environment_contract": "generic-v1",
        "canonical_birth_safety": (
            "newly born 640-primary and 768-auxiliary native targets require the frozen-640 "
            "current-frame context verifier before canonical publication, without changing NvDCF input"
        ),
        "aux_config_sha256": sha256(config_dir / config_name),
        "cadence_note": "one supplemental 768x768 batch-6 inference per 20 source batches; frozen 640 primary and NvDCF input are unchanged",
    })
    manifest["detector_profile"] = profile
    manifest["source_sha256"] = {
        path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()
    }
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["config_sha256"] = {
        path.name: sha256(path) for path in sorted(config_dir.iterdir()) if path.is_file()
    }
    manifest["model_sha256"] = {
        str(path.relative_to(models)): sha256(path)
        for path in sorted(models.rglob("*")) if path.is_file()
    }
    manifest["runtime"] = (
        "DeepStream 9.1; frozen YOLO26m FP16 640x640 batch-6 primary unchanged; "
        "YOLO26m 768 supplemental batch-6 interval-19 exact-frame sidecar; 448 context verifier; "
        "local NvDCF; no MV3DT/calibration"
    )
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_live_binary_sha256": parent["binary_sha256"],
        "parent_direct_file_binary_sha256": parent["direct_file_binary_sha256"],
        "change": "replace only the 896 supplemental detector engine/config with the reviewed 768 variant at the same interval; compile the exact current source",
        "invariants": [
            "frozen YOLO26m 640 primary config/engine/parser/threshold/preprocessing unchanged",
            "normal primary detections >=0.25 remain unchanged before NvDCF",
            "local NvDCF input policy unchanged",
            "exact source/frame/PTS sidecar and verifier policy unchanged",
            "frozen V13 and calibration unchanged",
        ],
    }
    manifest["recovery"] = {
        "required_environment": "V11_SOURCE_STALL_RECOVERY=1",
        "candidate_binary_sha256": manifest["binary_sha256"],
        "validated_cycles": 0,
        "status": "REQUIRES_REVALIDATION_ON_EXACT_BINARY",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_DIAGNOSTIC_CANDIDATE",
        "candidate": str(OUT.relative_to(ROOT)),
        "live_binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "primary_engine_sha256": profile["primary_engine_sha256"],
        "aux_engine_sha256": profile["aux_engine_sha256"],
        "aux_interval": INTERVAL,
        "tracker_config_sha256": manifest["tracker_config_sha256"],
        "verifier_engine_sha256": manifest["yolo_context_verifier"]["engine_sha256"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
