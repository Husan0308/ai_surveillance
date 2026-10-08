"""Build an isolated 640-primary + 768 supplemental CAM-05 recall candidate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-aux448-every-frame-20261008-r42"
OUT = (ROOT / os.getenv("SIXCAM_AUX_768_CANDIDATE",
                        ".runtime/sixcam-cam05-aux768-context448-20261008-r43")).resolve()
ARTIFACTS = ROOT / ".runtime/f5-cam05-safety-20261007-r1/yolo768-candidate-r2"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
AUX_INTERVAL = int(os.getenv("SIXCAM_AUX_768_INTERVAL", "1"))
EXPECTED = {
    "engine": "84832420e6e3ad68e1664b2ceb04b38474459960990c00d76987ca8bbda153be",
    "onnx": "eef24113344008c26747ca65d7d4b9637b46193c80dadffb37396bbdbee57ace",
    "parser": "32766d0cc5855d243f9ec9c46ea959a3f88f250b6ca024b293f1b6b7cd7a949f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()}


def build_binary(target: Path, direct_file: bool) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER -DYOLO26_AUXILIARY_DETECTOR"
    if direct_file:
        macros += " -DPGIE_ISOLATION"
    command_text = (
        "g++ -std=c++17 -O2 " + macros
        + " -I/opt/nvidia/deepstream/deepstream/sources/includes"
        + " $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " /src/main.cpp -o /work/bin/" + target.name
        + " $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta -ldl -pthread"
        + " -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    subprocess.run([
        "docker", "run", "--rm", "--pull=never", "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{OUT}:/work", IMAGE,
        "bash", "-lc", command_text,
    ], cwd=ROOT, check=True)


def main() -> dict:
    if not OUT.is_relative_to(ROOT / ".runtime"):
        raise ValueError("isolated candidate must remain under .runtime")
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {OUT}")
    if not 0 <= AUX_INTERVAL <= 29:
        raise ValueError("SIXCAM_AUX_768_INTERVAL must be in 0..29")
    parent_manifest = json.loads((PARENT / "build-manifest.json").read_text())
    for name, expected_key in (("cam-six-yolo-local6-posttracker", "binary_sha256"),
                               ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256")):
        path = PARENT / "bin" / name
        if sha256(path) != parent_manifest.get(expected_key):
            raise RuntimeError(f"r42 parent binary does not match manifest: {name}")
    for artifact, expected in EXPECTED.items():
        path = ARTIFACTS / "models" / {
            "engine": "yolo26m_raw_otm_768_b6_fp16.engine",
            "onnx": "yolo26m_raw_otm_768.onnx",
            "parser": "libyolo26rawperson_768.so",
        }[artifact]
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"768 supplemental artifact hash mismatch: {path.name}")

    shutil.copytree(PARENT, OUT)
    models = OUT / "models"
    config_dir = OUT / "config"
    for name in ("yolo26m_raw_otm_768.onnx", "yolo26m_raw_otm_768_b6_fp16.engine",
                 "libyolo26rawperson_768.so"):
        shutil.copyfile(ARTIFACTS / "models" / name, models / name)

    aux_config_name = "config_infer_aux_yolo26m_768.txt"
    metamux_name = "config_metamux_yolo26m_aux768.txt"
    config_text = (ARTIFACTS / "config/config_infer_yolo26m_768_b6.txt").read_text()
    if (config_text.count("batch-size=6") != 1
            or config_text.count("gie-unique-id=1") != 1
            or config_text.count("interval=0") != 1
            or "infer-dims=3;768;768" not in config_text
            or "pre-cluster-threshold=0.25" not in config_text
            or "nms-iou-threshold=0.45" not in config_text):
        raise RuntimeError("source 768 PGIE config violates the frozen supplemental contract")
    config_text = config_text.replace("gie-unique-id=1", "gie-unique-id=3", 1)
    config_text = config_text.replace("interval=0", f"interval={AUX_INTERVAL}", 1)
    (config_dir / aux_config_name).write_text(config_text)
    (config_dir / metamux_name).write_text(
        (PARENT / "config/config_metamux_yolo26m_aux896.txt").read_text())

    live_binary = OUT / "bin/cam-six-yolo-local6-posttracker"
    direct_binary = OUT / "bin/cam-six-yolo-local6-direct-gatea"
    build_binary(live_binary, direct_file=False)
    build_binary(direct_binary, direct_file=True)

    manifest = json.loads((OUT / "build-manifest.json").read_text())
    profile = manifest["detector_profile"]
    profile.update({
        "name": f"yolo26m-fp16-640-primary-768-aux-interval-{AUX_INTERVAL}-context-verified",
        "aux_config": aux_config_name,
        "aux_engine": "yolo26m_raw_otm_768_b6_fp16.engine",
        "aux_onnx": "yolo26m_raw_otm_768.onnx",
        "aux_parser": "libyolo26rawperson_768.so",
        "aux_engine_sha256": sha256(models / "yolo26m_raw_otm_768_b6_fp16.engine"),
        "aux_parser_sha256": sha256(models / "libyolo26rawperson_768.so"),
        "aux_onnx_sha256": sha256(models / "yolo26m_raw_otm_768.onnx"),
        "aux_input": [768, 768],
        "aux_interval": AUX_INTERVAL,
        "aux_unique_id": 3,
        "aux_threshold": 0.25,
        "aux_nms_iou": 0.45,
        "batch_size": 6,
        "metamux_config": metamux_name,
        "metamux_pts_tolerance_us": 0,
        "aux_environment_contract": "generic-v1",
        "canonical_birth_safety": (
            "newly born 640-primary and 768-auxiliary native targets require the frozen-640 "
            "current-frame context verifier before canonical publication, without changing NvDCF input"
        ),
    })
    profile["aux_config_sha256"] = sha256(config_dir / aux_config_name)
    manifest["deepstream_image"] = IMAGE
    manifest["source_sha256"] = source_hashes()
    manifest["binary_sha256"] = sha256(live_binary)
    manifest["direct_file_binary_sha256"] = sha256(direct_binary)
    manifest["config_sha256"] = {
        path.name: sha256(path) for path in sorted(config_dir.iterdir()) if path.is_file()
    }
    manifest["model_sha256"] = {
        str(path.relative_to(models)): sha256(path)
        for path in sorted(models.rglob("*")) if path.is_file()
    }
    manifest["runtime"] = (
        "DeepStream 9.1; frozen YOLO26m FP16 640x640 batch=6 primary unchanged; "
        f"YOLO26m 768 supplemental batch-6 GIE interval={AUX_INTERVAL}, exact-PTS merged; "
        "only verified supplemental target births publish; frozen 448 context verifier; local NvDCF; "
        "no MV3DT/calibration"
    )
    manifest["parallel_inference_metadata_isolation"] = {
        "branch": "supplemental YOLO26m-768 only",
        "mechanism": "out-of-place NVMM conversion then branch-local object metadata stripping before nvinfer",
        "primary_branch_changed": False,
    }
    manifest["recovery"] = {
        "required_environment": "V11_SOURCE_STALL_RECOVERY=1",
        "candidate_binary_sha256": manifest["binary_sha256"],
        "validated_cycles": 0,
        "status": "REQUIRES_REVALIDATION_ON_EXACT_BINARY",
    }
    manifest["candidate_label"] = (
        f"{OUT.name}: frozen 640 primary + 768 supplemental interval={AUX_INTERVAL} "
        "with fail-closed 448 context verification"
    )
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_live_binary_sha256": parent_manifest["binary_sha256"],
        "parent_direct_file_binary_sha256": parent_manifest["direct_file_binary_sha256"],
        "source_sha256": manifest["source_sha256"],
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "change": "replace 896 supplemental GIE with 768 supplemental GIE; generalize auxiliary runtime config contract",
        "invariants": [
            "frozen YOLO26m 640 primary config/engine/parser/threshold/preprocessing unchanged",
            "normal primary detections >=0.25 remain unchanged before NvDCF",
            "auxiliary unique component ID=3 and exact PTS=0 merge retained",
            "new auxiliary births remain fail-closed behind frozen-640 context verification",
            "frozen V13 and calibration unchanged",
        ],
    }
    (OUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_EXPERIMENTAL",
        "candidate": str(OUT.relative_to(ROOT)),
        "live_binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "primary_engine_sha256": profile["primary_engine_sha256"],
        "auxiliary": {"input": profile["aux_input"], "interval": profile["aux_interval"],
                      "engine_sha256": profile["aux_engine_sha256"],
                      "parser_sha256": profile["aux_parser_sha256"]},
        "context_engine_sha256": manifest["yolo_context_verifier"]["engine_sha256"],
        "stall_recovery": manifest["recovery"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
