#!/usr/bin/env python3
"""Build an isolated 896x896 YOLO candidate with the frozen local NvDCF stack."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from scripts.build_six_camera_v13_wall_candidate import IMAGE, ROOT, SOURCE

BASE = Path(os.getenv("CAM05_RECALL_BASE_CANDIDATE", str(
    ROOT / ".runtime/sixcam-approved-bbox-20261006-r12-profile"))).resolve()
DETECTOR_896 = ROOT / ".runtime/f5-cam05-safety-20261007-r1/yolo896-candidate-r1"
OUTPUT = Path(os.getenv("CAM05_RECALL_CANDIDATE", str(
    ROOT / ".runtime/sixcam-cam05-recall-896-verified-20261007-r1"))).resolve()
FROZEN_PRIMARY = ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt"
EXPECTED = {
    "frozen_640_engine": "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16",
    "frozen_640_parser": "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19",
    "frozen_primary_config": "fbfa196338b6781b7d65c3058d4f02069e9116963804d0c3b23d1bf2301e4113",
    "tracker_config": "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355",
    "v13_reference": "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.name: sha(path) for path in sorted(SOURCE.iterdir()) if path.is_file()}


def stage() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite experimental candidate: {OUTPUT}")
    if not BASE.is_dir() or not DETECTOR_896.is_dir():
        raise FileNotFoundError("validated 640 base candidate or 896 detector candidate is missing")
    if sha(FROZEN_PRIMARY) != EXPECTED["frozen_primary_config"]:
        raise RuntimeError("frozen production PGIE config changed")
    base_manifest = json.loads((BASE / "build-manifest.json").read_text())
    source_manifest = json.loads((DETECTOR_896 / "candidate-manifest.json").read_text())
    detector_models = DETECTOR_896 / "models"
    engine = detector_models / "yolo26m_raw_otm_896_b6_fp16.engine"
    onnx = detector_models / "yolo26m_raw_otm_896.onnx"
    parser = detector_models / "libyolo26rawperson_896.so"
    if any(not path.is_file() for path in (engine, onnx, parser)):
        raise FileNotFoundError("896 engine, ONNX, or parser is missing")
    if (sha(BASE / "models/yolo26m_raw_otm_b6_fp16.engine") != EXPECTED["frozen_640_engine"]
            or sha(BASE / "models/libyolo26rawperson.so") != EXPECTED["frozen_640_parser"]
            or sha(BASE / "config/config_tracker_NvDCF_v13_local6.yml") != EXPECTED["tracker_config"]
            or sha(ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml") != EXPECTED["v13_reference"]):
        raise RuntimeError("protected 640/V13 input hash changed")
    if source_manifest.get("engine_sha256") != sha(engine):
        raise RuntimeError("896 engine disagrees with its candidate manifest")
    shutil.copytree(BASE, OUTPUT, copy_function=os.link)

    models = OUTPUT / "models"
    config = OUTPUT / "config/config_infer_primary_yolo26m_raw_otm.txt"
    build_manifest_path = OUTPUT / "build-manifest.json"
    binary = OUTPUT / "bin/cam-six-yolo-local6-posttracker"
    for path in (config, build_manifest_path, binary):
        path.unlink()
    (models / engine.name).write_bytes(engine.read_bytes())
    (models / onnx.name).write_bytes(onnx.read_bytes())
    (models / parser.name).write_bytes(parser.read_bytes())

    config_text = FROZEN_PRIMARY.read_text()
    replacements = {
        "model-engine-file=/models/yolo26m_raw_otm_b6_fp16.engine":
            "model-engine-file=/models/yolo26m_raw_otm_896_b6_fp16.engine",
        "custom-lib-path=/models/libyolo26rawperson.so":
            "custom-lib-path=/models/libyolo26rawperson_896.so",
        "infer-dims=3;640;640": "infer-dims=3;896;896",
    }
    for old, new in replacements.items():
        if config_text.count(old) != 1:
            raise RuntimeError(f"frozen PGIE setting not found exactly once: {old}")
        config_text = config_text.replace(old, new)
    config.write_text(config_text)

    build_script = (
        "g++ -std=c++17 -O2 -DYOLO26_PERSON -DYOLO26_TRACKER "
        "-I/opt/nvidia/deepstream/deepstream/sources/includes "
        "$(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0) "
        "/src/main.cpp -o /work/bin/cam-six-yolo-local6-posttracker "
        "$(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0) "
        "-L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta -ldl -pthread "
        "-Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    command = ["docker", "run", "--rm", "--pull=never", "--user", f"{os.getuid()}:{os.getgid()}",
        "--mount", f"type=bind,src={SOURCE},dst=/src,readonly",
        "--mount", f"type=bind,src={OUTPUT},dst=/work", IMAGE, "bash", "-lc", build_script]
    with (OUTPUT / "build.log").open("x") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    config_hashes = {path.name: sha(path) for path in sorted((OUTPUT / "config").iterdir()) if path.is_file()}
    model_hashes = {str(path.relative_to(models)): sha(path)
                    for path in sorted(models.rglob("*")) if path.is_file()}
    profile = {
        "name": "yolo26m-fp16-896-context-verified",
        "candidate_only": True,
        "input": [896, 896],
        "engine_sha256": sha(models / engine.name),
        "onnx_sha256": sha(models / onnx.name),
        "parser_sha256": sha(models / parser.name),
        "config_sha256": sha(config),
        "confidence": 0.25,
        "nms_iou": 0.45,
        "batch_size": 6,
        "aspect_ratio": 1,
        "symmetric_padding": 1,
        "interval": 0,
        "shared_weights_sha256": source_manifest["source_weights_sha256"],
        "frozen_640_baseline_engine_sha256": EXPECTED["frozen_640_engine"],
        "frozen_640_baseline_parser_sha256": EXPECTED["frozen_640_parser"],
        "false_track_policy": "new canonical target withheld until same-frame frozen-640 context crop positively verifies its center; NvDCF input/state remain normal candidate detections",
    }
    manifest = dict(base_manifest)
    manifest.update({
        "binary_sha256": sha(binary),
        "source_sha256": source_hashes(),
        "config_sha256": config_hashes,
        "model_sha256": model_hashes,
        "detector_profile": profile,
        "runtime": "DeepStream 9.1; candidate YOLO26m FP16 896x896 batch=6; frozen YOLO26m 640 crop verifier for first-publication gate; local NvDCF; no MV3DT/calibration",
    })
    build_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"candidate": str(OUTPUT), "binary_sha256": manifest["binary_sha256"],
        "detector_profile": profile, "protected_frozen_640_sha256": EXPECTED,
        "config_sha256": config_hashes, "model_sha256": model_hashes}


if __name__ == "__main__":
    print(json.dumps(stage(), indent=2))
