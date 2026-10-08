#!/usr/bin/env python3
"""Build an isolated grouped pre-tracker shadow-context candidate.

The frozen primary detector, supplemental detector, and local NvDCF profile
are copied unchanged from r42. Nearby verified targets share one current-frame
448px TensorRT context crop only when their full prior boxes fit in its exact
aligned geometry; output matching remains mutual, unique, and one-to-one.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-final-20261008-r48-lane-crop-stream"
TRT_PARENT = ROOT / ".runtime/sixcam-cam05-context448-timing-noaux-20261008-r77"
OUT = ROOT / ".runtime/sixcam-cam05-final-20261008-r49-aux896-every-frame"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
PINNED_IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"

PRIMARY_ENGINE = "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"
PRIMARY_PARSER = "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19"
LOCAL_TRACKER = "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355"
V13_REFERENCE = "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834"
CONTEXT_ENGINE = "f597e1b4ab9bc3a195344f005da6dda8a448b63aa9a6b776d5f04dd4f536732f"
CONTEXT_CONFIG = "2203998f46039d3390559d5d4447a1e5078a053a5308b884d4865f7a6c3632ae"
CONTEXT_PREPROCESSOR = "ebb0ce91b61be4088e83dc10ee0d3663d4fcf96da39b2a597be4b359c9c13fbb"
ENGINE_NAME = "yolo26m_context_verifier_448_b6_fp16.engine"
CONFIG_NAME = "yolo26m_context_verifier_448_b6.txt"
PARSER_NAME = "libyolo26rawperson_context_dynamic.so"
ONNX_NAME = "yolo26m_context_448.onnx"
PREPROCESSOR_NAME = "libv11_context_preprocess.so"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()}


def build_binary(name: str, *, direct_file: bool) -> None:
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
        "-v", "/usr/local/cuda/include:/host-cuda-include:ro", PINNED_IMAGE,
        "bash", "-lc", command,
    ], cwd=ROOT, check=True)


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {OUT}")
    if not OUT.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate must remain under .runtime")

    parent = json.loads((PARENT / "build-manifest.json").read_text())
    trt_parent = json.loads((TRT_PARENT / "build-manifest.json").read_text())
    if parent.get("deepstream_image") != PINNED_IMAGE or trt_parent.get("deepstream_image") != PINNED_IMAGE:
        raise RuntimeError("candidate parents must use the pinned DeepStream 9.1 image")
    for name, field in (("cam-six-yolo-local6-posttracker", "binary_sha256"),
                        ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256")):
        if sha256(PARENT / "bin" / name) != parent.get(field):
            raise RuntimeError(f"r42 parent binary differs from its manifest: {name}")
    profile = parent["detector_profile"]
    if (profile.get("primary_engine_sha256") != PRIMARY_ENGINE
            or profile.get("primary_parser_sha256") != PRIMARY_PARSER
            or parent.get("tracker_config_sha256") != LOCAL_TRACKER
            or parent.get("frozen_v13_reference_sha256") != V13_REFERENCE):
        raise RuntimeError("r42 protected detector/tracker provenance differs")

    context = trt_parent["yolo_context_verifier"]
    context_variant = trt_parent["yolo_context_verifier_variants"][CONFIG_NAME]
    engine = TRT_PARENT / "models" / ENGINE_NAME
    config = TRT_PARENT / "config" / CONFIG_NAME
    parser = PARENT / "models" / PARSER_NAME
    onnx = TRT_PARENT / "models" / ONNX_NAME
    preprocessor = TRT_PARENT / "lib" / PREPROCESSOR_NAME
    for path in (engine, config, parser, onnx, preprocessor):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (sha256(engine) != CONTEXT_ENGINE or sha256(config) != CONTEXT_CONFIG
            or sha256(preprocessor) != CONTEXT_PREPROCESSOR
            or sha256(parser) != context["parser_sha256"]
            or sha256(onnx) != context_variant["onnx_sha256"]):
        raise RuntimeError("reviewed context verifier artifact hash mismatch")
    config_text = config.read_text()
    for required in ("batch-size=6", "infer-dims=3;448;448",
                     f"model-engine-file=/models/{ENGINE_NAME}",
                     f"custom-lib-path=/models/{PARSER_NAME}",
                     "pre-cluster-threshold=0.20", "nms-iou-threshold=0.45",
                     "maintain-aspect-ratio=1", "symmetric-padding=1"):
        if config_text.count(required) != 1:
            raise RuntimeError(f"context verifier config contract mismatch: {required}")

    shutil.copytree(PARENT, OUT)
    shutil.copy2(engine, OUT / "models" / ENGINE_NAME)
    shutil.copy2(config, OUT / "config" / CONFIG_NAME)
    shutil.copy2(onnx, OUT / "models" / ONNX_NAME)
    (OUT / "lib").mkdir(exist_ok=True)
    shutil.copy2(preprocessor, OUT / "lib" / PREPROCESSOR_NAME)
    # Candidate-only cadence test: retain the exact 896 engine/parser and all
    # thresholds, but run the existing supplemental branch on every input
    # frame so it can keep low-confidence seated targets associated locally.
    aux_source = OUT / "config" / "config_infer_aux_yolo26m_896.txt"
    aux_text = aux_source.read_text()
    if aux_text.count("interval=2") != 1:
        raise RuntimeError("expected the isolated r48 supplemental interval=2 profile")
    aux_every_frame = OUT / "config" / "config_infer_aux_yolo26m_896_everyframe.txt"
    aux_every_frame.write_text(aux_text.replace("interval=2", "interval=0", 1))
    build_binary("cam-six-yolo-local6-posttracker", direct_file=False)
    build_binary("cam-six-yolo-local6-direct-gatea", direct_file=True)

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["detector_profile"]["aux_config"] = aux_every_frame.name
    manifest["detector_profile"]["aux_interval"] = 0
    manifest["detector_profile"]["experimental_cadence_change"] = {
        "from": 2, "to": 0,
        "purpose": "diagnostic every-frame CAM-05 seated-person association test",
        "primary_detector_unchanged": True,
    }
    parser_sha = sha256(OUT / "models" / PARSER_NAME)
    engine_sha = sha256(OUT / "models" / ENGINE_NAME)
    config_sha = sha256(OUT / "config" / CONFIG_NAME)
    onnx_sha = sha256(OUT / "models" / ONNX_NAME)
    preprocessor_sha = sha256(OUT / "lib" / PREPROCESSOR_NAME)
    manifest["yolo_context_verifier"] = {
        **manifest["yolo_context_verifier"],
        "config": CONFIG_NAME,
        "config_sha256": config_sha,
        "engine": ENGINE_NAME,
        "engine_sha256": engine_sha,
        "parser": PARSER_NAME,
        "parser_sha256": parser_sha,
        "onnx": ONNX_NAME,
        "runtime_batch_size": 6,
        "network_input": [3, 448, 448],
        "engine_batch_profile": {"min": 1, "opt": 6, "max": 6},
        "threshold": 0.20,
        "nms_iou": 0.45,
        "scheduling": {
            "worker_contexts": 6, "camera_workers": 6,
            "queue_capacity_per_camera": 4, "global_pending_limit": 32,
            "policy": "six persistent preprocessed FP16 TensorRT 448 batch-1 camera contexts and CUDA streams; exact-frame/PTS; bounded per-camera admission; late result fails closed",
        },
    }
    manifest.setdefault("yolo_context_verifier_variants", {})[CONFIG_NAME] = {
        "config": CONFIG_NAME, "config_sha256": config_sha,
        "engine": ENGINE_NAME, "engine_sha256": engine_sha,
        "parser": PARSER_NAME, "parser_sha256": parser_sha,
        "onnx": ONNX_NAME, "onnx_sha256": onnx_sha,
        "input": [3, 448, 448], "threshold": 0.20, "nms_iou": 0.45,
        "batch": {"runtime": 6, "engine_min": 1, "engine_opt": 6, "engine_max": 6},
        "precision": "FP16", "maintain_aspect_ratio": 1, "symmetric_padding": 1,
        "purpose": "candidate-local same-frame person verification for already verified auxiliary shadow targets; never enters NvDCF",
    }
    manifest["native_context_backend"] = False
    manifest["native_trt_runtime_backend"] = True
    manifest["native_preprocessed_context_backend"] = True
    manifest["per_camera_context_backend"] = True
    manifest["gpu_context_crop"] = True
    manifest["context_preprocessor"] = {
        "library": f"/contextlib/{PREPROCESSOR_NAME}",
        "library_sha256": preprocessor_sha,
        "source": "context_preprocess.cu",
        "source_sha256": source_hashes()["context_preprocess.cu"],
        "input": "RGB HWC NVMM surface to normalized FP32 NCHW TensorRT binding",
        "batch_capacity": 6,
        "diagnostic_only": True,
    }
    manifest["model_sha256"].update({
        ENGINE_NAME: engine_sha,
        ONNX_NAME: onnx_sha,
        PARSER_NAME: parser_sha,
    })
    manifest["config_sha256"][CONFIG_NAME] = config_sha
    manifest["shadow_context_refresh_enabled"] = True
    manifest["source_sha256"] = source_hashes()
    manifest["model_sha256"] = {
        str(path.relative_to(OUT / "models")): sha256(path)
        for path in sorted((OUT / "models").rglob("*")) if path.is_file()
    }
    manifest["config_sha256"] = {
        path.name: sha256(path) for path in sorted((OUT / "config").iterdir()) if path.is_file()
    }
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["candidate_label"] = OUT.name + ": r48 isolated verifier lanes plus diagnostic interval=0 896 supplemental detector cadence"
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": parent["binary_sha256"],
        "context_verifier_parent": str(TRT_PARENT.relative_to(ROOT)),
        "change": "preserve r48 verifier changes and run the unchanged 896 supplemental YOLO branch with interval=0 instead of interval=2",
        "reason": "the confirmed CAM-05 seated track was only intermittently associated by a supplemental detector running every third frame; test whether per-frame supplemental proposals remove multi-second gaps and reduce shadow-verifier work without altering the primary detector or thresholds",
        "invariants": [
            "YOLO primary, 896 auxiliary artifacts, parser and thresholds unchanged",
            ">=0.25 primary inputs to NvDCF unchanged",
            "NvDCF config and all tuning unchanged",
            "pretracker hint must be the immediately preceding frame and belong to a previously appearance-verified auxiliary target",
            "same-frame detector proposal suppresses redundant shadow-context submission",
            "recent primary-associated targets skip redundant crop verification for two frames",
            "targets share a crop only when their complete prior-frame hint rectangles fit inside the exact crop computed by the same production crop helper",
            "each visible fallback requires mutual unique-best one-to-one matching between current INACTIVE shadows and current verifier proposals with existing IoU, center, size, and 0.10 margin gates",
            "canonical output still requires current same-ID INACTIVE shadow with immediate continuity and exact frame/PTS",
            "visible box uses current verifier detector bbox only; shadow geometry remains hint-only",
            "late verifier results fail closed",
        ],
    }
    manifest["recovery"] = {
        "required_environment": "V11_SOURCE_STALL_RECOVERY=1",
        "candidate_binary_sha256": manifest["binary_sha256"],
        "validated_cycles": 0,
        "status": "REQUIRES_REVALIDATION_ON_EXACT_BINARY_AND_PROFILE",
    }
    manifest["context_preprocessor"]["library_sha256"] = preprocessor_sha
    manifest["context_preprocessor"]["diagnostic_only"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_ISOLATED_CANDIDATE_REQUIRES_VALIDATION",
        "candidate": str(OUT.relative_to(ROOT)),
        "live_binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "primary_engine_sha256": PRIMARY_ENGINE,
        "primary_parser_sha256": PRIMARY_PARSER,
        "auxiliary_engine_sha256": profile["aux_engine_sha256"],
        "context_engine_sha256": engine_sha,
        "context_config_sha256": config_sha,
        "context_preprocessor_sha256": preprocessor_sha,
        "tracker_sha256": LOCAL_TRACKER,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
