#!/usr/bin/env python3
"""Build an isolated native-context 320px verifier candidate from r64."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-context448-direct-context-20261008-r64"
OUT = ROOT / ".runtime/sixcam-cam05-context320-direct-context-20261008-r65"
SHAPE = ROOT / ".runtime/cam05-verifier-shapes-20261008-b6/320"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
ENGINE_NAME = "yolo26m_context_verifier_320_b6_fp16.engine"
ONNX_NAME = "yolo26m_context_320.onnx"
CONFIG_NAME = "yolo26m_context_verifier_320_b6.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()}


def build_binary(output: Path, *, direct: bool) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER -DYOLO26_AUXILIARY_DETECTOR"
    if direct:
        macros += " -DPGIE_ISOLATION"
    cmd = (
        "g++ -std=c++17 -O2 " + macros
        + " -I/opt/nvidia/deepstream/deepstream/sources/includes"
        + " $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " /src/main.cpp -o /work/bin/" + output.name
        + " $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta"
        + " -lnvbufsurface -lnvbufsurftransform -lnvds_infer -ldl -pthread"
        + " -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    subprocess.run([
        "docker", "run", "--rm", "--pull=never", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{OUT}:/work", IMAGE,
        "bash", "-lc", cmd,
    ], cwd=ROOT, check=True)


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite isolated candidate: {OUT}")
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    for name, field in (("cam-six-yolo-local6-posttracker", "binary_sha256"),
                        ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256")):
        if sha256(PARENT / "bin" / name) != parent[field]:
            raise RuntimeError(f"r64 parent binary does not match manifest: {name}")
    source_engine = SHAPE / "models/yolo26m_context_320_b6_fp16.engine"
    source_onnx = SHAPE / "models/yolo26m_context_320.onnx"
    source_config = SHAPE / "config/context-320-batch2.txt"
    if not all(path.is_file() for path in (source_engine, source_onnx, source_config)):
        raise FileNotFoundError("previously evaluated 320px engine/config artifacts are incomplete")
    if sha256(source_engine) != "e20c159594b17eee2a32479ba237dfaa06998f4ce1a89f970b2cf5ce26b00bc3":
        raise RuntimeError("320px engine differs from the already evaluated artifact")

    shutil.copytree(PARENT, OUT)
    models, configs = OUT / "models", OUT / "config"
    shutil.copy2(source_engine, models / ENGINE_NAME)
    shutil.copy2(source_onnx, models / ONNX_NAME)
    text = source_config.read_text()
    parser = parent["yolo_context_verifier"]["parser"]
    old_engine = "/models/yolo26m_context_320_b2_fp16.engine"
    if text.count(old_engine) != 1 or text.count("custom-lib-path=/models/libyolo26rawperson.so") != 1:
        raise RuntimeError("320px source config has unexpected engine/parser fields")
    text = text.replace(old_engine, f"/models/{ENGINE_NAME}", 1)
    text = text.replace("custom-lib-path=/models/libyolo26rawperson.so",
                        f"custom-lib-path=/models/{parser}", 1)
    text = text.replace("gie-unique-id=1", "gie-unique-id=2", 1)
    checks = ("batch-size=6", "infer-dims=3;320;320", "pre-cluster-threshold=0.20",
              "nms-iou-threshold=0.45", "maintain-aspect-ratio=1", "symmetric-padding=1")
    if any(text.count(item) != 1 for item in checks):
        raise RuntimeError("320px verifier config changed batch/preprocess/threshold semantics")
    (configs / CONFIG_NAME).write_text(text)

    build_binary(OUT / "bin/cam-six-yolo-local6-posttracker", direct=False)
    build_binary(OUT / "bin/cam-six-yolo-local6-direct-gatea", direct=True)

    manifest = json.loads((OUT / "build-manifest.json").read_text())
    context_parser = parent["yolo_context_verifier"]["parser"]
    engine_hash, config_hash = sha256(models / ENGINE_NAME), sha256(configs / CONFIG_NAME)
    parser_hash, onnx_hash = sha256(models / context_parser), sha256(models / ONNX_NAME)
    manifest["yolo_context_verifier"] = {
        **parent["yolo_context_verifier"],
        "config": CONFIG_NAME, "config_sha256": config_hash,
        "engine": ENGINE_NAME, "engine_sha256": engine_hash,
        "runtime_batch_size": 6, "network_input": [3, 320, 320],
        "engine_batch_profile": {"min": 1, "opt": 6, "max": 6},
        "diagnostic_only": True,
        "scheduling": {
            "worker_contexts": 1, "camera_workers": 6,
            "queue_capacity_per_camera": 4, "global_pending_limit": 32,
            "policy": "persistent NvDsInferContext dynamic batch<=6; bounded ready-crop queue; exact source frame/PTS; 26ms deadline; late results fail closed",
        },
    }
    manifest.setdefault("yolo_context_verifier_variants", {})[CONFIG_NAME] = {
        "config": CONFIG_NAME, "config_sha256": config_hash,
        "engine": ENGINE_NAME, "engine_sha256": engine_hash,
        "engine_profile": {"min": [1, 3, 320, 320], "opt": [6, 3, 320, 320],
                            "max": [6, 3, 320, 320]},
        "onnx": ONNX_NAME, "onnx_sha256": onnx_hash,
        "parser": context_parser, "parser_sha256": parser_hash,
        "input": [3, 320, 320], "batch": {"runtime": 6, "engine_min": 1,
            "engine_opt": 6, "engine_max": 6},
        "precision": "FP16", "threshold": .20, "nms_iou": .45,
        "maintain_aspect_ratio": 1, "symmetric_padding": 1,
        "purpose": "diagnostic smaller context verifier only; primary detector and NvDCF input unchanged",
    }
    manifest["config_sha256"] = {p.name: sha256(p) for p in sorted(configs.iterdir()) if p.is_file()}
    manifest["model_sha256"] = {str(p.relative_to(models)): sha256(p)
                                for p in sorted(models.rglob("*")) if p.is_file()}
    manifest["source_sha256"] = source_hashes()
    live_hash = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    direct_hash = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["binary_sha256"] = live_hash
    manifest["direct_file_binary_sha256"] = direct_hash
    manifest["candidate_label"] = f"{OUT.name}: isolated 320px diagnostic verifier, primary and NvDCF unchanged"
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": parent["binary_sha256"],
        "change": "context verifier only: 448px to previously evaluated 320px dynamic FP16 engine; persistent native NvDsInferContext batch up to six",
        "source_sha256": manifest["source_sha256"],
        "binary_sha256": live_hash, "direct_file_binary_sha256": direct_hash,
        "invariants": ["YOLO26m 640 primary config/engine/parser unchanged",
                       "NvDCF local6 config unchanged", "lowconf remains sidecar-only",
                       "source/frame/PTS and strict confirmation policy unchanged",
                       "late verifier result fails closed"],
    }
    (OUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"status": "BUILT_DIAGNOSTIC_ONLY", "candidate": str(OUT.relative_to(ROOT)),
            "live_binary_sha256": live_hash, "direct_file_binary_sha256": direct_hash,
            "engine_sha256": engine_hash, "config_sha256": config_hash}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
