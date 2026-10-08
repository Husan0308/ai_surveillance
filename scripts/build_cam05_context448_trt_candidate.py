#!/usr/bin/env python3
"""Build isolated 448px direct-TensorRT verifier candidate from reviewed r64 artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-context448-direct-context-20261008-r64"
OUT = ROOT / ".runtime/sixcam-cam05-context448-direct-trt-20261008-r73"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
LIBRARY_NAME = "libv11_context_preprocess.so"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()}


def container(command: str) -> None:
    subprocess.run([
        "docker", "run", "--rm", "--pull=never", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{OUT}:/work",
        "-v", "/usr/local/cuda/include:/host-cuda-include:ro", IMAGE,
        "bash", "-lc", command,
    ], cwd=ROOT, check=True)


def build_binary(name: str, *, direct: bool) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER -DYOLO26_AUXILIARY_DETECTOR"
    if direct:
        macros += " -DPGIE_ISOLATION"
    command = (
        "g++ -std=c++17 -O2 " + macros
        + " -I/host-cuda-include"
        + " -I/opt/nvidia/deepstream/deepstream/sources/includes"
        + " $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " /src/main.cpp -o /work/bin/" + name
        + " $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta"
        + " -lnvbufsurface -lnvbufsurftransform -lnvds_infer -lnvinfer"
        + " -lnvinfer_plugin -L/usr/local/cuda/lib64 -lcudart -ldl -pthread"
        + " -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    container(command)


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite isolated candidate: {OUT}")
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    for name, field in (("cam-six-yolo-local6-posttracker", "binary_sha256"),
                        ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256")):
        if sha256(PARENT / "bin" / name) != parent[field]:
            raise RuntimeError(f"r64 parent binary differs from its manifest: {name}")
    context = parent["yolo_context_verifier"]
    engine = PARENT / "models" / context["engine"]
    parser = PARENT / "models" / context["parser"]
    config = PARENT / "config" / context["config"]
    if any(not path.is_file() for path in (engine, parser, config)):
        raise FileNotFoundError("reviewed 448 verifier artifacts are incomplete")
    if (sha256(engine) != "f597e1b4ab9bc3a195344f005da6dda8a448b63aa9a6b776d5f04dd4f536732f"
            or sha256(config) != "2203998f46039d3390559d5d4447a1e5078a053a5308b884d4865f7a6c3632ae"
            or context.get("network_input") != [3, 448, 448]
            or int(context.get("runtime_batch_size", 0)) != 6):
        raise RuntimeError("r71 must use the exact previously reviewed 448px batch-6 engine/config")

    shutil.copytree(PARENT, OUT)
    (OUT / "lib").mkdir(exist_ok=True)
    (OUT / "build").mkdir(exist_ok=True)
    host_nvcc = Path("/usr/local/cuda/bin/nvcc")
    if not host_nvcc.is_file():
        raise RuntimeError("host CUDA compiler unavailable; refusing any toolkit installation")
    subprocess.run([
        str(host_nvcc), "-std=c++17", "-O2", "-arch=sm_86", "-Xcompiler", "-fPIC",
        "-c", str(SOURCE / "context_preprocess.cu"),
        "-o", str(OUT / "build/context_preprocess.o"),
    ], cwd=ROOT, check=True)
    container("g++ -shared -o /work/lib/" + LIBRARY_NAME
        + " /work/build/context_preprocess.o -L/usr/local/cuda/lib64 -lcudart")
    build_binary("cam-six-yolo-local6-posttracker", direct=False)
    build_binary("cam-six-yolo-local6-direct-gatea", direct=True)

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    library = OUT / "lib" / LIBRARY_NAME
    library_hash = sha256(library)
    sources = source_hashes()
    manifest["native_context_backend"] = False
    manifest["native_preprocessed_context_backend"] = True
    manifest["native_trt_runtime_backend"] = True
    manifest["context_preprocessor"] = {
        "library": f"/contextlib/{LIBRARY_NAME}",
        "library_sha256": library_hash,
        "source": "context_preprocess.cu",
        "source_sha256": sources["context_preprocess.cu"],
        "input": "RGB HWC device surface -> normalized FP32 NCHW; TensorRT engine input binding is FP32 and internal compute is FP16",
        "batch_capacity": 6,
        "diagnostic_only": True,
    }
    manifest["source_sha256"] = sources
    manifest["config_sha256"] = {
        path.name: sha256(path) for path in sorted((OUT / "config").iterdir()) if path.is_file()
    }
    manifest["model_sha256"] = {
        str(path.relative_to(OUT / "models")): sha256(path)
        for path in sorted((OUT / "models").rglob("*")) if path.is_file()
    }
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["candidate_label"] = f"{OUT.name}: isolated 448px direct TensorRT context-verifier diagnostic"
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": parent["binary_sha256"],
        "change": "same reviewed 448px verifier artifacts; replace NvDsInferContext wrapper with persistent TensorRT execution context and the same staged YOLO parser/NMS thresholds",
        "source_sha256": sources,
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "context_preprocessor_sha256": library_hash,
        "build_toolchain": {
            "device_compiler": str(host_nvcc),
            "device_compiler_version": "CUDA 12.6, sm_86 cubin",
            "compile_headers": "host CUDA 12.6 headers mounted read-only",
            "link_runtime": "pinned DeepStream container CUDA 13.2 libcudart",
        },
        "invariants": [
            "frozen YOLO26m 640 PGIE/config/parser unchanged",
            "frozen local NvDCF config unchanged",
            "low-confidence candidates remain out of NvDCF",
            "same-frame PTS and strict candidate-local gates retained",
            "late verifier output fails closed",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_DIAGNOSTIC_ONLY",
        "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "engine_sha256": context["engine_sha256"],
        "parser_sha256": context["parser_sha256"],
        "preprocessor_sha256": library_hash,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
