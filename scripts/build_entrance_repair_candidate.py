#!/usr/bin/env python3
"""Build a reversible source-updated clone of the active Entrance r6 candidate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
PARENT = ROOT / ".runtime/sixcam-calibration-recording-20261009-r6-eos-fix"
OUT = ROOT / ".runtime/sixcam-entrance-repairs-20261009-r1"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
PARENT_BINARY_SHA256 = "6d3bf5613bebb0322c7c924b354f5f3c6097aa7b9750467c2d5c83ae58f817bd"
PROTECTED = {
    "primary_engine": ("models/yolo26m_raw_otm_b6_fp16.engine",
                       "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"),
    "primary_parser": ("models/libyolo26rawperson.so",
                       "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19"),
    "auxiliary_engine": ("models/yolo26m_raw_otm_768_b6_fp16.engine",
                         "84832420e6e3ad68e1664b2ceb04b38474459960990c00d76987ca8bbda153be"),
    "local_tracker": ("config/config_tracker_NvDCF_v13_local6.yml",
                      "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355"),
}
FROZEN_V13 = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml"
FROZEN_V13_SHA256 = "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def compile_binary(name: str, *, direct_file: bool) -> None:
    # Match the active runtime's graph: auxiliary 768, infer interval 19, with
    # the same compiled auxiliary branch and no sparse-preprocess gate.
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
        "docker", "run", "--rm", "--pull=never", "--network=none", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{OUT}:/work",
        "-v", "/usr/local/cuda/include:/host-cuda-include:ro", IMAGE,
        "bash", "-lc", command,
    ], cwd=ROOT, check=True)


def main() -> dict:
    if not PARENT.is_relative_to(ROOT / ".runtime") or not OUT.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate paths must remain under .runtime")
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {OUT}")
    if sha256(PARENT / "bin/cam-six-yolo-local6-posttracker") != PARENT_BINARY_SHA256:
        raise RuntimeError("rollback parent binary hash mismatch")
    parent_manifest_path = PARENT / "build-manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("deepstream_image") != IMAGE:
        raise RuntimeError("rollback parent DeepStream image provenance mismatch")
    if parent_manifest.get("detector_profile", {}).get("name") != (
            "yolo26m-fp16-640-primary-768-aux-interval-19-context-verified"):
        raise RuntimeError("rollback parent detector profile mismatch")
    if parent_manifest.get("detector_profile", {}).get("aux_interval") != 19:
        raise RuntimeError("rollback parent auxiliary interval mismatch")
    if sha256(FROZEN_V13) != FROZEN_V13_SHA256:
        raise RuntimeError("frozen V13 changed")

    protected_before = {}
    for name, (relative, expected) in PROTECTED.items():
        actual = sha256(PARENT / relative)
        if actual != expected:
            raise RuntimeError(f"protected parent asset mismatch: {name}={actual}")
        protected_before[name] = actual
    configs_before = tree_hashes(PARENT / "config")
    models_before = tree_hashes(PARENT / "models")

    shutil.copytree(PARENT, OUT)
    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True)

    configs_after = tree_hashes(OUT / "config")
    models_after = tree_hashes(OUT / "models")
    if configs_before != configs_after or models_before != models_after:
        raise RuntimeError("candidate build changed detector/tracker config or model assets")
    protected_after = {name: sha256(OUT / relative)
                       for name, (relative, _) in PROTECTED.items()}
    if protected_before != protected_after:
        raise RuntimeError("candidate build changed protected inference/tracker artifacts")

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_sha256"] = {
        str(path.relative_to(SOURCE)): sha256(path)
        for path in sorted(SOURCE.rglob("*")) if path.is_file()
    }
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": PARENT_BINARY_SHA256,
        "deepstream_compile_macros": ["YOLO26_PERSON", "YOLO26_TRACKER", "YOLO26_AUXILIARY_896"],
        "changes": [
            "suppress duplicate same-native-ID NvDCF output before canonical publication only when one unique immediate-frame continuation is proven",
            "count native nvurisrcbin RTSP retry warnings separately from watchdog recoveries",
            "advance per-source epoch on reconnect/recovery and reject stale cross-epoch frame/verifier joins",
        ],
        "runtime_profile_unchanged": True,
        "protected_hashes_before_and_after": protected_before,
        "calibration_changed": False,
        "recovery_validation": "NOT_RUN; no production camera fault injection",
    }
    manifest["recovery"] = {
        **manifest.get("recovery", {}),
        "candidate_binary_sha256": manifest["binary_sha256"],
        "validated_cycles": 0,
        "status": "REQUIRES_VALIDATION_ON_EXACT_BINARY",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "BUILT_ISOLATED_CANDIDATE_NOT_YET_RUNTIME_VALIDATED",
        "candidate": str(OUT.relative_to(ROOT)),
        "live_binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "rollback_candidate": str(PARENT.relative_to(ROOT)),
        "rollback_binary_sha256": PARENT_BINARY_SHA256,
        "protected_hashes": protected_after,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
