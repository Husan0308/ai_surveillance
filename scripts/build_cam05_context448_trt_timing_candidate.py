#!/usr/bin/env python3
"""Build a diagnostic r73-derived candidate with TensorRT timing probes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-context448-direct-trt-20261008-r73"
OUT = ROOT / ".runtime/sixcam-cam05-context448-trt-timing-20261008-r75"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
LIBRARY_NAME = "libv11_context_preprocess.so"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parent_manifest = json.loads((PARENT / "build-manifest.json").read_text())
    for name, field in (("cam-six-yolo-local6-posttracker", "binary_sha256"),
                        ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256")):
        if sha256(PARENT / "bin" / name) != parent_manifest[field]:
            raise RuntimeError(f"r73 parent binary differs from manifest: {name}")
    shutil.copytree(PARENT, OUT)
    build_binary("cam-six-yolo-local6-posttracker", direct=False)
    build_binary("cam-six-yolo-local6-direct-gatea", direct=True)
    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_hashes = {path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()}
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["source_sha256"] = source_hashes
    manifest["diagnostic_timing_probes"] = {
        "source": "context_trt_runtime.hpp",
        "source_sha256": source_hashes["context_trt_runtime.hpp"],
        "no_detection_policy_change": True,
        "measurements": ["set_shape", "enqueue_host", "stream_sync_wait", "gpu_stream", "parser_cpu"],
    }
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "change": "diagnostic wall/GPU event timing around persistent TensorRT enqueue and parser; no inference or threshold changes",
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "source_sha256": source_hashes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"candidate": str(OUT.relative_to(ROOT)),
            "binary_sha256": manifest["binary_sha256"],
            "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
            "status": "BUILT_DIAGNOSTIC_ONLY"}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
