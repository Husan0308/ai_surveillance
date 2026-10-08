#!/usr/bin/env python3
"""Build an isolated r19 candidate with immediate exact empty joins on skipped aux batches."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r18-aux768-preprocess-sparse20"
OUT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r19-aux768-sparse-exact-join"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
PRIMARY_ENGINE_SHA256 = "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"
PRIMARY_PARSER_SHA256 = "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19"
TRACKER_SHA256 = "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355"
AUX_ENGINE_SHA256 = "84832420e6e3ad68e1664b2ceb04b38474459960990c00d76987ca8bbda153be"
GATE_PERIOD = 20
GATE_PHASE = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compile_binary(name: str, *, direct_file: bool) -> None:
    macros = (
        "-DYOLO26_PERSON -DYOLO26_TRACKER -DYOLO26_AUXILIARY_DETECTOR "
        "-DYOLO26_AUXILIARY_SPARSE_PREPROCESS"
    )
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
    if not OUT.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate must remain under .runtime")
    parent_manifest = json.loads((PARENT / "build-manifest.json").read_text())
    if parent_manifest.get("deepstream_image") != IMAGE:
        raise RuntimeError("r18 parent is not the pinned DeepStream image")
    profile = parent_manifest.get("detector_profile", {})
    if (profile.get("primary_engine_sha256") != PRIMARY_ENGINE_SHA256
            or profile.get("primary_parser_sha256") != PRIMARY_PARSER_SHA256
            or profile.get("aux_engine_sha256") != AUX_ENGINE_SHA256
            or profile.get("aux_interval") != 0
            or profile.get("aux_preprocess_gate", {}).get("period_batches") != GATE_PERIOD):
        raise RuntimeError("r18 parent does not match the reviewed 640+768 sparse profile")
    for binary_name, manifest_key in (
        ("cam-six-yolo-local6-posttracker", "binary_sha256"),
        ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256"),
    ):
        if sha256(PARENT / "bin" / binary_name) != parent_manifest.get(manifest_key):
            raise RuntimeError(f"r18 parent binary hash mismatch: {binary_name}")

    shutil.copytree(PARENT, OUT)
    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True)

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    profile = dict(profile)
    profile.update({
        "name": "yolo26m-fp16-640-primary-768-aux-sparse-exact-empty-join-context-verified",
        "aux_preprocess_gate": {
            "enabled": True,
            "gate": "branch-local mux-batch sampler before auxiliary nvvideoconvert",
            "period_batches": GATE_PERIOD,
            "phase": GATE_PHASE,
            "inference_batches_per_20_mux_batches": 1,
            "primary_input_unchanged": True,
            "exact_sidecar_join_unchanged": True,
            "skipped_batch_result": "publish empty values for every exact source/frame/PTS key before dropping conversion branch buffer",
        },
        "cadence_note": (
            "Auxiliary preprocessing admits mux batch 0 then every 20th batch; skipped batches "
            "publish immediate empty exact-key results so the primary path does not wait for "
            "the auxiliary join timeout. Admitted batches retain interval=0 and the existing "
            "auxiliary inference schedule."
        ),
    })
    manifest["detector_profile"] = profile
    manifest["source_sha256"] = {
        path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()
    }
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["config_sha256"] = {
        path.name: sha256(path) for path in sorted((OUT / "config").iterdir()) if path.is_file()
    }
    manifest["model_sha256"] = {
        str(path.relative_to(OUT / "models")): sha256(path)
        for path in sorted((OUT / "models").rglob("*")) if path.is_file()
    }
    manifest["runtime"] = (
        "DeepStream 9.1; exact frozen YOLO26m 640 primary and local NvDCF; "
        "768 supplemental YOLO on one in 20 mux batches with exact empty-join completion "
        "for skipped batches; 448 context verifier; exact frame/PTS joins; no MV3DT/calibration"
    )
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_live_binary_sha256": parent_manifest["binary_sha256"],
        "parent_direct_file_binary_sha256": parent_manifest["direct_file_binary_sha256"],
        "change": "complete skipped auxiliary frames with immediate empty exact-key sidecars before dropping them ahead of NVMM conversion",
        "invariants": [
            "primary 640 engine/parser/config/threshold and PGIE inputs unchanged",
            "NvDCF configuration and input unchanged",
            "auxiliary sample cadence unchanged from r18",
            "auxiliary outputs still require exact source/frame/PTS joins",
            "skipped auxiliary frames complete with exact-key empty result; no stale/nearest-frame match",
            "context verifier/fallback and canonical object policy unchanged",
            "frozen V13 and calibration unchanged",
        ],
    }
    manifest["recovery"] = {
        "required_environment": "V11_SOURCE_STALL_RECOVERY=1",
        "candidate_binary_sha256": manifest["binary_sha256"],
        "validated_cycles": 0,
        "status": "REQUIRES_REVALIDATION_ON_EXACT_BINARY_AND_PROFILE",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_DIAGNOSTIC_CANDIDATE",
        "candidate": str(OUT.relative_to(ROOT)),
        "live_binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "primary_engine_sha256": PRIMARY_ENGINE_SHA256,
        "aux_engine_sha256": AUX_ENGINE_SHA256,
        "tracker_config_sha256": manifest["tracker_config_sha256"],
        "auxiliary_preprocess_gate": profile["aux_preprocess_gate"],
        "verifier_engine_sha256": manifest["yolo_context_verifier"]["engine_sha256"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
