#!/usr/bin/env python3
"""Build an isolated r20 candidate with opt-in per-frame pipeline stage timing."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r19-aux768-sparse-exact-join"
candidate_name = os.getenv(
    "V11_STAGE_TIMING_CANDIDATE_NAME",
    "sixcam-cam05-verified-track-20261008-r20-stage-timing",
)
if not re.fullmatch(r"[A-Za-z0-9_.-]+", candidate_name):
    raise ValueError("V11_STAGE_TIMING_CANDIDATE_NAME must be one safe directory name")
OUT = ROOT / ".runtime" / candidate_name
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compile_binary(name: str, *, direct_file: bool) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER -DYOLO26_AUXILIARY_DETECTOR -DYOLO26_AUXILIARY_SPARSE_PREPROCESS"
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
        "docker", "run", "--rm", "--pull=never", "--network=none", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{OUT}:/work",
        "-v", "/usr/local/cuda/include:/host-cuda-include:ro", IMAGE,
        "bash", "-lc", command,
    ], cwd=ROOT, check=True)


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite isolated candidate: {OUT}")
    parent_manifest = json.loads((PARENT / "build-manifest.json").read_text())
    if parent_manifest.get("deepstream_image") != IMAGE:
        raise RuntimeError("r19 parent does not match the pinned DeepStream image")
    shutil.copytree(PARENT, OUT)
    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True)
    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha256"] = {
        path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()
    }
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_live_binary_sha256": parent_manifest["binary_sha256"],
        "parent_direct_file_binary_sha256": parent_manifest["direct_file_binary_sha256"],
        "change": "opt-in per-frame timestamps at mux out, PGIE in/out, and NvDCF in/out; no runtime decision or inference behavior changed",
        "invariants": [
            "diagnostic only when V11_PIPELINE_STAGE_TIMING=1",
            "YOLO/NvDCF/verifier/preview behavior and all configs copied byte-identically from r19",
            "protected engine, parser, tracker, and calibration artifacts unchanged",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_DIAGNOSTIC_ONLY",
        "candidate": str(OUT.relative_to(ROOT)),
        "live_binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "parent_live_binary_sha256": parent_manifest["binary_sha256"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
