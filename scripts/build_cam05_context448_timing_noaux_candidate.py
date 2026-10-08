#!/usr/bin/env python3
"""Build r73-derived no-aux diagnostic to isolate GPU contention."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-context448-trt-timing-20261008-r75"
OUT = ROOT / ".runtime/sixcam-cam05-context448-timing-noaux-20261008-r77"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite isolated candidate: {OUT}")
    parent_manifest = json.loads((PARENT / "build-manifest.json").read_text())
    if sha256(PARENT / "bin/cam-six-yolo-local6-direct-gatea") != parent_manifest["direct_file_binary_sha256"]:
        raise RuntimeError("r75 parent binary differs from its manifest")
    shutil.copytree(PARENT, OUT)
    command = (
        "g++ -std=c++17 -O2 -DYOLO26_PERSON -DYOLO26_TRACKER -DPGIE_ISOLATION"
        " -I/host-cuda-include -I/opt/nvidia/deepstream/deepstream/sources/includes"
        " $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        " /src/main.cpp -o /work/bin/cam-six-yolo-local6-direct-gatea"
        " $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        " -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta"
        " -lnvbufsurface -lnvbufsurftransform -lnvds_infer -lnvinfer"
        " -lnvinfer_plugin -L/usr/local/cuda/lib64 -lcudart -ldl -pthread"
        " -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    subprocess.run([
        "docker", "run", "--rm", "--pull=never", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{OUT}:/work",
        "-v", "/usr/local/cuda/include:/host-cuda-include:ro", IMAGE,
        "bash", "-lc", command,
    ], cwd=ROOT, check=True)
    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    direct_hash = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["direct_file_binary_sha256"] = direct_hash
    manifest["gpu_contention_isolation"] = {
        "auxiliary_branch_compiled": False,
        "primary640_and_tracker_unchanged": True,
        "context_verifier_448_unchanged": True,
        "diagnostic_only": True,
        "parent_binary_sha256": parent_manifest["direct_file_binary_sha256"],
        "source_sha256": {path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"candidate": str(OUT.relative_to(ROOT)), "direct_file_binary_sha256": direct_hash,
            "status": "BUILT_DIAGNOSTIC_ONLY", "auxiliary_branch_compiled": False}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
