#!/usr/bin/env python3
"""Build an isolated CAM-05 verifier candidate using a native preprocessed tensor."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-context320-direct-context-20261008-r65"
OUT = ROOT / ".runtime/sixcam-cam05-context320-preprocessed-20261008-r70"
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


def run_container(command: str) -> None:
    subprocess.run([
        "docker", "run", "--rm", "--pull=never", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{OUT}:/work", IMAGE,
        "bash", "-lc", command,
    ], cwd=ROOT, check=True)


def build_binary(name: str, *, direct: bool) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER -DYOLO26_AUXILIARY_DETECTOR"
    if direct:
        macros += " -DPGIE_ISOLATION"
    command = (
        "g++ -std=c++17 -O2 " + macros
        + " -I/opt/nvidia/deepstream/deepstream/sources/includes"
        + " $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " /src/main.cpp -o /work/bin/" + name
        + " $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta"
        + " -lnvbufsurface -lnvbufsurftransform -lnvds_infer -ldl -pthread"
        + " -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    run_container(command)


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite isolated candidate: {OUT}")
    parent_manifest_path = PARENT / "build-manifest.json"
    if not parent_manifest_path.is_file():
        raise FileNotFoundError(parent_manifest_path)
    parent = json.loads(parent_manifest_path.read_text())
    for name, field in (("cam-six-yolo-local6-posttracker", "binary_sha256"),
                        ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256")):
        if sha256(PARENT / "bin" / name) != parent[field]:
            raise RuntimeError(f"r65 parent binary does not match manifest: {name}")
    context = parent["yolo_context_verifier"]
    engine = PARENT / "models" / context["engine"]
    config = PARENT / "config" / context["config"]
    if sha256(engine) != context["engine_sha256"] or sha256(config) != context["config_sha256"]:
        raise RuntimeError("r65 verifier engine/config do not match manifest")
    if context.get("network_input") != [3, 320, 320] or int(context.get("runtime_batch_size", 0)) != 6:
        raise RuntimeError("r66 requires the already evaluated 320x320, batch-6 verifier profile")

    shutil.copytree(PARENT, OUT)
    (OUT / "lib").mkdir(exist_ok=True)
    host_nvcc = Path("/usr/local/cuda/bin/nvcc")
    if not host_nvcc.is_file():
        raise RuntimeError("host CUDA compiler is unavailable; refusing to install or upgrade CUDA")
    (OUT / "build").mkdir(exist_ok=True)
    subprocess.run([
        str(host_nvcc), "-std=c++17", "-O2", "-arch=sm_86",
        "-Xcompiler", "-fPIC", "-c", str(SOURCE / "context_preprocess.cu"),
        "-o", str(OUT / "build/context_preprocess.o"),
    ], cwd=ROOT, check=True)
    # Compile device code with the already installed host toolkit, then link
    # against the candidate's pinned DeepStream CUDA 13.2 runtime. This avoids
    # requiring nvcc in the production runtime image and avoids a CUDA 12
    # runtime dependency inside the container.
    run_container(
        "g++ -shared -o /work/lib/" + LIBRARY_NAME
        + " /work/build/context_preprocess.o -L/usr/local/cuda/lib64 -lcudart"
    )
    build_binary("cam-six-yolo-local6-posttracker", direct=False)
    build_binary("cam-six-yolo-local6-direct-gatea", direct=True)

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    lib = OUT / "lib" / LIBRARY_NAME
    library_sha = sha256(lib)
    source_digest = source_hashes()
    manifest["native_preprocessed_context_backend"] = True
    manifest["context_preprocessor"] = {
        "library": f"/contextlib/{LIBRARY_NAME}",
        "library_sha256": library_sha,
        "source": "context_preprocess.cu",
        "source_sha256": source_digest["context_preprocess.cu"],
        "input": "RGB HWC device surface -> normalized FP32 NCHW (engine input binding is FLOAT; engine compute remains FP16)",
        "batch_capacity": 6,
        "diagnostic_only": True,
    }
    manifest["source_sha256"] = source_digest
    manifest["config_sha256"] = {
        path.name: sha256(path) for path in sorted((OUT / "config").iterdir()) if path.is_file()
    }
    manifest["model_sha256"] = {
        str(path.relative_to(OUT / "models")): sha256(path)
        for path in sorted((OUT / "models").rglob("*")) if path.is_file()
    }
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["candidate_label"] = f"{OUT.name}: isolated preprocessed-tensor verifier diagnostic"
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": parent["binary_sha256"],
        "change": "r65 320px diagnostic verifier plus opt-in CUDA RGB-HWC to normalized FP16 NCHW preprocessing; primary detector and NvDCF inputs unchanged",
        "source_sha256": source_digest,
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "context_preprocessor_sha256": library_sha,
        "build_toolchain": {
            "device_compiler": str(host_nvcc),
            "device_compiler_version": "CUDA 12.6, sm_86 cubin",
            "link_runtime": "pinned DeepStream container CUDA 13.2 libcudart",
        },
        "invariants": [
            "YOLO26m 640 primary config/engine/parser unchanged",
            "NvDCF local6 config unchanged",
            "lowconf proposals remain sidecar-only",
            "normal >=0.25 NvDCF proposal input unchanged",
            "verifier matches exact source/frame/PTS and late result fails closed",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_DIAGNOSTIC_ONLY",
        "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "engine_sha256": context["engine_sha256"],
        "preprocessor_sha256": library_sha,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
