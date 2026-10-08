#!/usr/bin/env python3
"""Build a fresh current-source candidate with reviewed sparse YOLO-896 assist.

The frozen 640 PGIE, NvDCF profile, context verifier, and bbox policy are copied
unchanged from the r6 candidate. Only the candidate-only 896 auxiliary
inference cadence is changed to the already replay-validated interval 4.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r6-verifier-admission"
OUT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r9-sparse-aux4"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
FROZEN_ENGINE = "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"
FROZEN_PARSER = "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19"
FROZEN_TRACKER = "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355"
FROZEN_V13 = "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834"
AUX_ENGINE = "44ae13c7bc8388b530509d68fb34b74c85fc2c5d1f7248f937ef506947523fd4"
CONTEXT_ENGINE = "5bf1c545363b5cc7606c701eabae580011b5a9fd5d27011507fd3bc565aebd83"


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
    parent_manifest_path = PARENT / "build-manifest.json"
    parent = json.loads(parent_manifest_path.read_text())
    if parent.get("deepstream_image") != IMAGE:
        raise RuntimeError("parent does not use the pinned DeepStream image")
    for filename, manifest_key in (
        ("cam-six-yolo-local6-posttracker", "binary_sha256"),
        ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256"),
    ):
        if sha256(PARENT / "bin" / filename) != parent.get(manifest_key):
            raise RuntimeError(f"parent binary hash mismatch: {filename}")
    profile = parent.get("detector_profile", {})
    if (profile.get("primary_engine_sha256") != FROZEN_ENGINE
            or profile.get("primary_parser_sha256") != FROZEN_PARSER
            or profile.get("aux_engine_sha256") != AUX_ENGINE
            or profile.get("aux_interval") != 2
            or profile.get("normal_nvdcf_input_threshold") != 0.25):
        raise RuntimeError("r6 parent detector/profile differs from the proven candidate")
    tracker = ROOT / "config/deepstream/config_tracker_NvDCF_v13_local6.yml"
    v13 = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml"
    if sha256(tracker) != FROZEN_TRACKER or sha256(v13) != FROZEN_V13:
        raise RuntimeError("frozen NvDCF artifacts changed")
    if sha256(PARENT / "models/yolo26m_context_verifier_448_b2_fp16.engine") != CONTEXT_ENGINE:
        raise RuntimeError("the strict context-verifier engine differs from the tested r6 engine")

    shutil.copytree(PARENT, OUT)
    aux_config = OUT / "config/config_infer_aux_yolo26m_896.txt"
    config_text = aux_config.read_text()
    if config_text.count("interval=2") != 1:
        raise RuntimeError("parent auxiliary config is not the expected interval-2 profile")
    aux_config.write_text(config_text.replace("interval=2", "interval=4", 1))

    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True)

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    profile = manifest["detector_profile"]
    profile["name"] = "yolo26m-fp16-640-primary-896-aux-interval-4-context-verified"
    profile["aux_interval"] = 4
    profile["cadence_note"] = (
        "sparse 896 auxiliary cadence interval=4 (approximately 4 inference frames/sec per source); "
        "direct replay showed CAM-05 seated-person target creation and jacket rejection"
    )
    profile["config_sha256"] = None  # Auxiliary config hash is recorded in the manifest map below.
    manifest["config_sha256"] = {
        path.name: sha256(path) for path in sorted((OUT / "config").iterdir()) if path.is_file()
    }
    profile.pop("config_sha256", None)
    manifest["model_sha256"] = {
        str(path.relative_to(OUT / "models")): sha256(path)
        for path in sorted((OUT / "models").rglob("*")) if path.is_file()
    }
    manifest["source_sha256"] = source_hashes()
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["recovery"] = {
        "required_environment": "V11_SOURCE_STALL_RECOVERY=1",
        "candidate_binary_sha256": manifest["binary_sha256"],
        "validated_cycles": 0,
        "status": "REQUIRES_REVALIDATION_ON_EXACT_BINARY",
    }
    manifest["candidate_label"] = (
        f"{OUT.name}: r6 verified-track path with only the replay-tested 896 auxiliary cadence changed to interval 4"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "aux_interval": profile["aux_interval"],
        "aux_engine_sha256": profile["aux_engine_sha256"],
        "context_verifier_engine_sha256": manifest["yolo_context_verifier"]["engine_sha256"],
        "source_sha256": manifest["source_sha256"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
