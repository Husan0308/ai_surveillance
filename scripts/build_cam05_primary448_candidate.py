"""Build a fresh 640-primary, no-896, 448-context-verifier candidate.

This is an isolated diagnostic/live candidate. It keeps the frozen primary
YOLO and local NvDCF configuration while replacing the 512/640 crop verifier
with the reviewed 448 engine and omitting the unrelated 896 auxiliary GIE.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-cascade-20261008-r38-shadow-context-448-b2"
OUT = Path(os.getenv(
    "CAM05_PRIMARY448_OUT",
    str(ROOT / ".runtime/sixcam-cam05-primary448-20261008-r39"),
)).resolve()
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


def _compile(output: Path, *, direct_file: bool) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER"
    if direct_file:
        macros += " -DPGIE_ISOLATION"
    binary_name = "cam-six-yolo-local6-direct-gatea" if direct_file else "cam-six-yolo-local6-posttracker"
    compile_script = (
        "g++ -std=c++17 -O2 " + macros
        + " -I/opt/nvidia/deepstream/deepstream/sources/includes"
        + " $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " /src/main.cpp -o /work/bin/" + binary_name
        + " $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta -ldl -pthread"
        + " -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    command = [
        "docker", "run", "--rm", "--pull=never", "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{output}:/work", IMAGE,
        "bash", "-lc", compile_script,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def build() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {OUT}")
    parent_manifest_path = PARENT / "build-manifest.json"
    parent = json.loads(parent_manifest_path.read_text())
    if parent.get("deepstream_image") != IMAGE:
        raise RuntimeError("448 parent was not built against the pinned DeepStream image")
    if source_hashes() != parent.get("source_sha256"):
        raise RuntimeError("native source changed after the reviewed 448 parent was built")
    if sha256(PARENT / "bin/cam-six-yolo-local6-posttracker") != parent.get("binary_sha256"):
        raise RuntimeError("448 parent binary does not match its manifest")

    shutil.copytree(PARENT, OUT)
    # The new binary has no YOLO26_AUXILIARY_896 compile flag. Remove the
    # auxiliary-only assets from this fresh candidate so the runtime manifest
    # cannot imply that an unbuilt branch is part of the accepted graph.
    for name in ("config_infer_aux_yolo26m_896.txt", "config_metamux_yolo26m_aux896.txt"):
        (OUT / "config" / name).unlink(missing_ok=True)
    for name in ("yolo26m_raw_otm_896.onnx", "yolo26m_raw_otm_896_b6_fp16.engine",
                 "libyolo26rawperson_896.so"):
        (OUT / "models" / name).unlink(missing_ok=True)

    _compile(OUT, direct_file=False)
    _compile(OUT, direct_file=True)

    manifest = json.loads((OUT / "build-manifest.json").read_text())
    profile = manifest["detector_profile"]
    manifest.update({
        "candidate_label": f"{OUT.name}: frozen 640 primary + local NvDCF + 448 current-frame context verifier; no 896 auxiliary GIE",
        "source_sha256": source_hashes(),
        "binary_sha256": sha256(OUT / "bin/cam-six-yolo-local6-posttracker"),
        "direct_file_binary_sha256": sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea"),
        "config_sha256": {p.name: sha256(p) for p in sorted((OUT / "config").iterdir()) if p.is_file()},
        "model_sha256": {str(p.relative_to(OUT / "models")): sha256(p)
                         for p in sorted((OUT / "models").rglob("*")) if p.is_file()},
        "runtime": "DeepStream 9.1; frozen YOLO26m FP16 640x640 batch=6 primary; primary extraction floor 0.23 with only >=0.25 entering NvDCF; bounded persistent 448 FP16 context verifier gates canonical births/low-confidence continuity only; local NvDCF type2; no 896 auxiliary GIE, MV3DT, or calibration",
        "recovery": {"required_environment": "V11_SOURCE_STALL_RECOVERY=1",
                     "candidate_binary_sha256": sha256(OUT / "bin/cam-six-yolo-local6-posttracker")},
        "auxiliary_yolo": None,
    })
    profile.clear()
    profile.update({
        "name": "yolo26m-fp16-640-primary-448-context-verified",
        "candidate_only": True,
        "primary_config": "config_infer_primary_yolo26m_raw_otm_lowconf023.txt",
        "primary_engine_sha256": parent["detector_profile"]["primary_engine_sha256"],
        "primary_parser_sha256": parent["detector_profile"]["primary_parser_sha256"],
        "primary_threshold": 0.23,
        "normal_nvdcf_input_threshold": 0.25,
        "batch_size": 6,
        "auxiliary_detector": False,
        "context_verifier_mode": "yolo26m-current-frame-crop-confirmation",
        "canonical_birth_safety": "640-primary target publication is context checked without changing NvDCF input",
    })
    verifier = manifest["yolo_context_verifier"]
    verifier["use"] = "current-frame local person confirmation for 0.23-0.25 sidecar and verified target births; never changes primary NvDCF input"
    verifier["scheduling"] = {
        "worker_contexts": 3, "camera_workers": 6,
        "queue_capacity_per_camera": 4, "global_pending_limit": 32,
        "policy": "three persistent batch-2 TensorRT contexts; bounded per-camera crop queues; exact source frame/PTS; 26ms deadline; late result fails closed",
    }
    manifest["frozen_v13_reference_sha256"] = parent["frozen_v13_reference_sha256"]
    manifest["tracker_config_sha256"] = parent["tracker_config_sha256"]
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": parent["binary_sha256"],
        "macro_contract": ["YOLO26_PERSON", "YOLO26_TRACKER"],
        "excluded_macro": "YOLO26_AUXILIARY_896",
        "reason": "the 896 auxiliary graph is not needed for this targeted CAM-05 context-verifier fix and caused verifier deadline contention in the prior diagnostic",
    }
    (OUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
