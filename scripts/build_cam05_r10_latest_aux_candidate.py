#!/usr/bin/env python3
"""Build a fresh r10 candidate with non-blocking latest-only auxiliary queues."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r9-sparse-aux4"
OUT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r10-latest-aux"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()}


def compile_binary(name: str, *, direct_file: bool) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER -DYOLO26_AUXILIARY_896"
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
        raise FileExistsError(f"refusing to overwrite candidate: {OUT}")
    parent_path = PARENT / "build-manifest.json"
    parent = json.loads(parent_path.read_text())
    if parent.get("deepstream_image") != IMAGE:
        raise RuntimeError("r9 parent does not use the pinned DeepStream image")
    if sha256(PARENT / "bin/cam-six-yolo-local6-posttracker") != parent.get("binary_sha256"):
        raise RuntimeError("r9 live binary hash mismatch")
    if sha256(PARENT / "bin/cam-six-yolo-local6-direct-gatea") != parent.get("direct_file_binary_sha256"):
        raise RuntimeError("r9 direct-file binary hash mismatch")
    profile = parent.get("detector_profile", {})
    if (profile.get("primary_engine_sha256") !=
            "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"
            or profile.get("primary_parser_sha256") !=
            "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19"
            or profile.get("aux_interval") != 4
            or profile.get("normal_nvdcf_input_threshold") != 0.25):
        raise RuntimeError("r9 detector profile differs from the accepted candidate contract")
    shutil.copytree(PARENT, OUT)
    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True)

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha256"] = source_hashes()
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["auxiliary_queue_policy"] = {
        "primary_input_and_metadata_queues": "blocking, bounded at 2 buffers; unchanged",
        "auxiliary_input_and_metadata_queues": "downstream-leaky, latest-only, bounded at 1 buffer",
        "metamux": "active primary sink_0; exact PTS tolerance 0; missing aux frame is fail-closed",
        "primary_proposals_dropped": False,
        "auxiliary_overrun_counters": ["input", "metadata"],
        "diagnostic_reason": "r9 exact-PTS supplemental GIE branch caused upstream backpressure under live six-camera load",
    }
    manifest["recovery"] = {
        "required_environment": "V11_SOURCE_STALL_RECOVERY=1",
        "candidate_binary_sha256": manifest["binary_sha256"],
        "validated_cycles": 0,
        "status": "REQUIRES_REVALIDATION_ON_EXACT_BINARY",
    }
    manifest["candidate_label"] = (
        f"{OUT.name}: r9 candidate with supplemental-branch-only latest-only queue isolation"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "primary_engine_sha256": profile["primary_engine_sha256"],
        "tracker_config_sha256": manifest["tracker_config_sha256"],
        "aux_engine_sha256": profile["aux_engine_sha256"],
        "aux_interval": profile["aux_interval"],
        "source_sha256": manifest["source_sha256"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
