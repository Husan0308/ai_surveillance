#!/usr/bin/env python3
"""Build an isolated current-source candidate with a shared batched TRT verifier.

Primary YOLO26m, its NvDCF input, and local NvDCF are inherited byte-for-byte
from r9. Only the secondary context-verifier execution/scheduling is changed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r9-frame-state-diagnostic"
SHAPES = ROOT / ".runtime/cam05-verifier-shapes-20261008-448-b6/448"
TRT_PARENT = ROOT / ".runtime/sixcam-cam05-context448-direct-trt-20261008-r73"
OUT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r16-batched-aux"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
ENGINE_NAME = "yolo26m_context_verifier_448_b6_fp16.engine"
SOURCE_ENGINE_NAME = "yolo26m_context_448_b6_fp16.engine"
CONFIG_NAME = "yolo26m_context_verifier_448_b6.txt"
PREPROCESSOR_NAME = "libv11_context_preprocess.so"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()}


def build_binary(name: str, *, direct_file: bool, auxiliary_detector: bool = False) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER"
    if auxiliary_detector:
        macros += " -DYOLO26_AUXILIARY_DETECTOR"
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
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    for name, field in (("cam-six-yolo-local6-posttracker", "binary_sha256"),
                        ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256")):
        if sha256(PARENT / "bin" / name) != parent[field]:
            raise RuntimeError(f"r9 parent binary differs from its manifest: {name}")

    engine = SHAPES / "models" / SOURCE_ENGINE_NAME
    onnx = SHAPES / "models/yolo26m_context_448.onnx"
    parser = PARENT / "models/libyolo26rawperson_context_dynamic.so"
    template = SHAPES / "config/context-448-batch6.txt"
    preprocessor = TRT_PARENT / "lib" / PREPROCESSOR_NAME
    if any(not path.is_file() for path in (engine, onnx, parser, template, preprocessor)):
        raise FileNotFoundError("448 batch-6 verifier artifacts are incomplete")
    if sha256(parser) != parent["yolo_context_verifier"]["parser_sha256"]:
        raise RuntimeError("context parser differs from the verified r9 parser")
    if sha256(onnx) != parent["yolo_context_verifier_variants"][
            "yolo26m_context_verifier_448.txt"]["onnx_sha256"]:
        raise RuntimeError("448 ONNX differs from the verified r9 graph")
    if sha256(engine) != "f597e1b4ab9bc3a195344f005da6dda8a448b63aa9a6b776d5f04dd4f536732f":
        raise RuntimeError("448 batch-6 engine differs from the previously reviewed artifact")
    if sha256(preprocessor) != "ebb0ce91b61be4088e83dc10ee0d3663d4fcf96da39b2a597be4b359c9c13fbb":
        raise RuntimeError("preprocessor differs from the previously reviewed CUDA crop implementation")

    shutil.copytree(PARENT, OUT)
    (OUT / "lib").mkdir(exist_ok=True)
    shutil.copy2(engine, OUT / "models" / ENGINE_NAME)
    shutil.copy2(onnx, OUT / "models" / onnx.name)
    shutil.copy2(preprocessor, OUT / "lib" / PREPROCESSOR_NAME)
    config_text = template.read_text()
    if config_text.count("gie-unique-id=1") != 1:
        raise RuntimeError("verifier template must contain exactly one unique component id")
    config_text = config_text.replace("gie-unique-id=1", "gie-unique-id=2", 1)
    config_text = config_text.replace(
        f"/models/{SOURCE_ENGINE_NAME}", f"/models/{ENGINE_NAME}", 1)
    config_text = config_text.replace(
        "custom-lib-path=/models/libyolo26rawperson.so",
        "custom-lib-path=/models/libyolo26rawperson_context_dynamic.so", 1)
    # The single persistent execution context is fed bounded microbatches of
    # up to six crops, matching the verified engine's optimum profile.
    expected = ("batch-size=6", "infer-dims=3;448;448",
                "pre-cluster-threshold=0.20", "nms-iou-threshold=0.45",
                "maintain-aspect-ratio=1", "symmetric-padding=1")
    if any(config_text.count(item) != 1 for item in expected):
        raise RuntimeError("verifier config does not match the reviewed 448/batch-6 contract")
    (OUT / "config" / CONFIG_NAME).write_text(config_text)

    build_binary("cam-six-yolo-local6-posttracker", direct_file=False,
                 auxiliary_detector=True)
    build_binary("cam-six-yolo-local6-direct-gatea", direct_file=True,
                 auxiliary_detector=True)

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = source_hashes()
    config_sha, engine_sha, onnx_sha, parser_sha = (
        sha256(OUT / "config" / CONFIG_NAME), sha256(OUT / "models" / ENGINE_NAME),
        sha256(OUT / "models" / onnx.name), sha256(OUT / "models" / parser.name))
    preprocessor_sha = sha256(OUT / "lib" / PREPROCESSOR_NAME)
    manifest["native_context_backend"] = False
    manifest["native_trt_runtime_backend"] = True
    manifest["per_camera_context_backend"] = False
    manifest["native_preprocessed_context_backend"] = True
    manifest["gpu_context_crop"] = True
    manifest["shadow_context_refresh_enabled"] = True
    manifest["auxiliary_detector_compiled"] = True
    manifest["context_preprocessor"] = {
        "library": f"/contextlib/{PREPROCESSOR_NAME}",
        "library_sha256": preprocessor_sha,
        "source": "context_preprocess.cu",
        "source_sha256": source["context_preprocess.cu"],
        "input": "current-frame RGB NVMM crop to normalized FP32 NCHW TensorRT input",
        "batch_capacity": 6,
        "diagnostic_only": True,
    }
    manifest["yolo_context_verifier"] = {
        **manifest["yolo_context_verifier"],
        "config": CONFIG_NAME,
        "runtime_batch_size": 6,
        "engine_batch_profile": {"min": 1, "opt": 6, "max": 6},
        "config_sha256": config_sha,
        "engine": ENGINE_NAME,
        "engine_sha256": engine_sha,
        "onnx": onnx.name,
        "onnx_sha256": onnx_sha,
        "parser": parser.name,
        "parser_sha256": parser_sha,
        "network_input": [3, 448, 448],
        "diagnostic_only": True,
        "scheduling": {
            "execution_contexts": 1,
            "camera_workers": 6,
            "gpu_crop_streams": 1,
            "queue_capacity_per_camera": 8,
            "policy": "one persistent FP16 TensorRT context; 750us opportunistic microbatch up to six crops; current-frame crop; exact frame/PTS; fixed fail-closed 26ms result deadline; late result never blocks a streaming callback",
        },
    }
    manifest.setdefault("yolo_context_verifier_variants", {})[CONFIG_NAME] = {
        "config": CONFIG_NAME, "config_sha256": config_sha,
        "engine": ENGINE_NAME, "engine_sha256": engine_sha,
        "engine_profile": {"min": [1, 3, 448, 448], "opt": [6, 3, 448, 448],
                            "max": [6, 3, 448, 448]},
        "onnx": onnx.name, "onnx_sha256": onnx_sha,
        "parser": parser.name, "parser_sha256": parser_sha,
        "input": [3, 448, 448], "precision": "FP16",
        "batch": {"runtime": 6, "engine_min": 1, "engine_opt": 6,
                  "engine_max": 6, "per_camera_execution": 0},
        "threshold": 0.20, "nms_iou": 0.45,
        "maintain_aspect_ratio": 1, "symmetric_padding": 1,
        "purpose": "candidate-local second-stage verifier only; output remains gated by exact-frame native-track/shadow association",
    }
    manifest["config_sha256"] = {
        path.name: sha256(path) for path in sorted((OUT / "config").iterdir()) if path.is_file()
    }
    manifest["model_sha256"] = {
        str(path.relative_to(OUT / "models")): sha256(path)
        for path in sorted((OUT / "models").rglob("*")) if path.is_file()
    }
    manifest["source_sha256"] = source
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["candidate_label"] = (
        f"{OUT.name}: diagnostic shared batch-6 448px warm TRT verifier; primary/NvDCF unchanged"
    )
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": parent["binary_sha256"],
        "change": "secondary verifier only: reviewed 448 model/parser executed by one persistent direct TensorRT context with opportunistic microbatching; existing 896 auxiliary proposal path retained to test recall",
        "purpose": "reduce exact-frame verifier scheduling latency while preserving the fixed fail-closed deadline and all candidate-local gates",
        "source_sha256": source,
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "invariants": [
            "YOLO26m 640 primary engine/parser/config and >=0.25 NvDCF input are unchanged",
            "frozen local NvDCF profile and calibration are unchanged",
            "same 448 context model/parser/threshold/NMS as previously reviewed",
            "low-confidence candidates stay out of NvDCF",
            "existing 896 proposals remain candidate-local and subject to the same birth verifier/duplicate safety policy",
            "exact frame/PTS and strict same-frame shadow matching are unchanged",
            "26ms verifier deadline remains fixed and fail-closed",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_DIAGNOSTIC_ONLY",
        "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "engine_sha256": engine_sha,
        "parser_sha256": parser_sha,
        "preprocessor_sha256": preprocessor_sha,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
