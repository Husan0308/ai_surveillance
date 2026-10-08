#!/usr/bin/env python3
"""Build an isolated 768 YOLO26m TensorRT engine/config from the staged dynamic graph."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".runtime/f5-cam05-safety-20261007-r1/yolo768-candidate-r2"
MODELS = OUT / "models"
ENGINE = MODELS / "yolo26m_raw_otm_768_b6_fp16.engine"
PARSER = MODELS / "libyolo26rawperson_768.so"
LABELS = MODELS / "labels.txt"
SOURCE_CONFIG = ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt"
EXPECTED_CONFIG = "fbfa196338b6781b7d65c3058d4f02069e9116963804d0c3b23d1bf2301e4113"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
CUDA_HEADERS = Path("/home/apsidal/ai_surveillance/.runtime/build-deps/cuda13.2")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest_path = OUT / "candidate-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("candidate_onnx_sha256") != sha(MODELS / "yolo26m_raw_otm_768.onnx"):
        raise RuntimeError("candidate ONNX provenance mismatch")
    if not PARSER.is_file() or not SOURCE_CONFIG.is_file() or not CUDA_HEADERS.is_dir():
        raise FileNotFoundError("candidate parser, frozen config, or CUDA headers missing")
    if sha(SOURCE_CONFIG) != EXPECTED_CONFIG:
        raise RuntimeError("frozen primary PGIE config changed")
    if ENGINE.exists():
        raise FileExistsError(ENGINE)
    if not LABELS.exists():
        shutil.copy2(ROOT / ".runtime/freeze/F5R-six-camera-yolo-resolution-20261003/models/960/labels.txt", LABELS)
    text = SOURCE_CONFIG.read_text()
    replacements = {
        "onnx-file=/models/yolo26m_raw_otm.onnx": "onnx-file=/models/yolo26m_raw_otm_768.onnx",
        "model-engine-file=/models/yolo26m_raw_otm_b6_fp16.engine": "model-engine-file=/models/yolo26m_raw_otm_768_b6_fp16.engine",
        "custom-lib-path=/models/libyolo26rawperson.so": "custom-lib-path=/models/libyolo26rawperson_768.so",
        "infer-dims=3;640;640": "infer-dims=3;768;768",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError(f"expected one frozen candidate field: {old}")
        text = text.replace(old, new)
    config_dir = OUT / "config"
    config_dir.mkdir(exist_ok=True)
    config = config_dir / "config_infer_yolo26m_768_b6.txt"
    config.write_text(text)
    command = ["docker", "run", "--rm", "--pull=never", "--network=none", "--gpus", "device=0",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "-v", f"{MODELS}:/models:rw",
        "-v", f"{CUDA_HEADERS}:/cuda-headers:ro", "--entrypoint", "/usr/src/tensorrt/bin/trtexec",
        IMAGE, "--onnx=/models/yolo26m_raw_otm_768.onnx", "--saveEngine=/models/yolo26m_raw_otm_768_b6_fp16.engine",
        "--fp16", "--minShapes=images:1x3x768x768", "--optShapes=images:6x3x768x768",
        "--maxShapes=images:6x3x768x768", "--memPoolSize=workspace:2048",
        "--profilingVerbosity=detailed", "--skipInference"]
    log = MODELS / "trtexec-build.log"
    with log.open("x") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    log_text = log.read_text(errors="replace")
    if "Engine built in" not in log_text or "[TRT]" not in log_text:
        raise RuntimeError("TensorRT engine build did not report completion")
    manifest.update({"engine_sha256": sha(ENGINE), "parser_sha256": sha(PARSER),
        "config_sha256": sha(config), "config_path": str(config.relative_to(ROOT)),
        "engine_path": str(ENGINE.relative_to(ROOT)), "engine_build_log_sha256": sha(log),
        "input_dimensions": [768, 768], "batch": {"min": 1, "opt": 6, "max": 6},
        "precision": "FP16", "confidence": .25, "nms_iou": .45,
        "maintain_aspect_ratio": 1, "symmetric_padding": 1,
        "frozen_primary_config_sha256": EXPECTED_CONFIG})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "shape_conversion"}, indent=2))


if __name__ == "__main__":
    main()
