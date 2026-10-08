"""Build an isolated 640-primary + 896-auxiliary CAM-05 recall candidate."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CANDIDATE = Path(os.getenv(
    "SIXCAM_BBOX_CANDIDATE",
    str(ROOT / ".runtime/sixcam-cam05-cascade-20261007-r2"),
)).resolve()
os.environ["SIXCAM_BBOX_CANDIDATE"] = str(CANDIDATE)

from scripts.build_six_camera_v13_wall_candidate import (  # noqa: E402
    EXPECTED,
    IMAGE,
    SOURCE,
    sha256,
    source_hashes,
    stage_exact_inputs,
)

RECALL_CANDIDATE = ROOT / ".runtime/sixcam-cam05-recall-896-verified-20261007-r1"
# 29 is the previously tested sparse diagnostic cadence. Allow an isolated
# candidate build to set 0 (every frame) so a measured miss can be tested
# without changing the frozen 640 primary or overwriting an existing profile.
AUX_INTERVAL = int(os.getenv("SIXCAM_AUX_INTERVAL", "29"))
if AUX_INTERVAL < 0:
    raise ValueError("SIXCAM_AUX_INTERVAL must be >= 0")
CONFIG_AUX = "config_infer_aux_yolo26m_896.txt"
CONFIG_METAMUX = "config_metamux_yolo26m_aux896.txt"


def _copy_verified(source: Path, target: Path, expected_hash: str) -> None:
    if not source.is_file() or sha256(source) != expected_hash:
        raise RuntimeError(f"auxiliary artifact hash mismatch: {source}")
    if target.exists():
        if sha256(target) != expected_hash:
            raise RuntimeError(f"refusing to replace divergent candidate artifact: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _aux_config() -> str:
    return f"""[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file=/models/yolo26m_raw_otm_896.onnx
model-engine-file=/models/yolo26m_raw_otm_896_b6_fp16.engine
labelfile-path=/models/labels.txt
custom-lib-path=/models/libyolo26rawperson_896.so
parse-bbox-func-name=NvDsInferParseYolo26RawPerson
batch-size=6
network-mode=2
num-detected-classes=80
interval={AUX_INTERVAL}
gie-unique-id=3
process-mode=1
network-type=0
warmup-engine=1
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
infer-dims=3;896;896
output-blob-names=output0

[class-attrs-all]
pre-cluster-threshold=0.25
nms-iou-threshold=0.45
topk=300
"""


def _metamux_config() -> str:
    return """[property]
enable=1
active-pad=sink_0
pts-tolerance=0

[user-configs]

[group-0]
src-ids-model-1=0;1;2;3;4;5
src-ids-model-3=0;1;2;3;4;5
"""


def _write_exact(path: Path, content: str) -> None:
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"refusing to overwrite divergent candidate config: {path}")
    if not path.exists():
        path.write_text(content)


def _build_binary(target: Path, direct_file: bool) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER -DYOLO26_AUXILIARY_896"
    if direct_file:
        macros += " -DPGIE_ISOLATION"
    compile_script = (
        "g++ -std=c++17 -O2 " + macros
        + " -I/opt/nvidia/deepstream/deepstream/sources/includes"
        + " $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " /src/main.cpp -o /work/bin/" + target.name
        + " $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta -ldl -pthread"
        + " -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    command = [
        "docker", "run", "--rm", "--pull=never", "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{CANDIDATE}:/work", IMAGE,
        "bash", "-lc", compile_script,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def build() -> dict:
    stage_exact_inputs()
    models = CANDIDATE / "models"
    configs = CANDIDATE / "config"
    context_config = configs / "yolo26m_context_verifier.txt"
    recalled_models = RECALL_CANDIDATE / "models"
    recall_manifest = json.loads((RECALL_CANDIDATE / "build-manifest.json").read_text())
    recall_hashes = recall_manifest["model_sha256"]
    for name in ("yolo26m_raw_otm_896.onnx", "yolo26m_raw_otm_896_b6_fp16.engine",
                 "libyolo26rawperson_896.so"):
        _copy_verified(recalled_models / name, models / name, recall_hashes[name])

    _write_exact(configs / CONFIG_AUX, _aux_config())
    _write_exact(configs / CONFIG_METAMUX, _metamux_config())
    binary = CANDIDATE / "bin/cam-six-yolo-local6-posttracker"
    direct_binary = CANDIDATE / "bin/cam-six-yolo-local6-direct-gatea"
    _build_binary(binary, direct_file=False)
    _build_binary(direct_binary, direct_file=True)

    configs_sha = {p.name: sha256(p) for p in sorted(configs.iterdir()) if p.is_file()}
    model_sha = {str(p.relative_to(models)): sha256(p)
                 for p in sorted(models.rglob("*")) if p.is_file()}
    manifest = {
        "deepstream_image": IMAGE,
        "source_sha256": source_hashes(),
        "binary_sha256": sha256(binary),
        "direct_file_binary_sha256": sha256(direct_binary),
        "config_sha256": configs_sha,
        "model_sha256": model_sha,
        "frozen_v13_reference_sha256": EXPECTED["frozen_v13_reference.yml"],
        "tracker_config": "config_tracker_NvDCF_v13_local6.yml",
        "tracker_config_sha256": EXPECTED["config_tracker_NvDCF_v13_local6.yml"],
        "detector_profile": {
            "name": f"yolo26m-fp16-640-primary-896-aux-interval-{AUX_INTERVAL}-context-verified",
            "candidate_only": True,
            "primary_config": "config_infer_primary_yolo26m_raw_otm_lowconf023.txt",
            "primary_engine_sha256": EXPECTED["yolo26m_raw_otm_b6_fp16.engine"],
            "primary_parser_sha256": EXPECTED["libyolo26rawperson.so"],
            "primary_threshold": 0.23,
            "normal_nvdcf_input_threshold": 0.25,
            "aux_config": CONFIG_AUX,
            "aux_engine_sha256": model_sha["yolo26m_raw_otm_896_b6_fp16.engine"],
            "aux_parser_sha256": model_sha["libyolo26rawperson_896.so"],
            "aux_input": [896, 896],
            "aux_interval": AUX_INTERVAL,
            "aux_unique_id": 3,
            "aux_threshold": 0.25,
            "aux_nms_iou": 0.45,
            "batch_size": 6,
            "metamux_config": CONFIG_METAMUX,
            "metamux_pts_tolerance_us": 0,
            "aux_duplicate_policy": "withhold any auxiliary person box matching a primary box under the existing spatial/size gate; never remove a primary proposal",
            "canonical_birth_safety": "all newly born 640-primary and 896-auxiliary native targets are context checked using the frozen 640 YOLO crop verifier; rejected targets are hidden from canonical/UI output without changing NvDCF input",
        },
        "yolo_context_verifier": {
            "config": context_config.name,
            "config_sha256": sha256(context_config),
            "engine_sha256": EXPECTED["yolo26m_raw_otm_b6_fp16.engine"],
            "parser_sha256": EXPECTED["libyolo26rawperson.so"],
            "runtime_batch_size": 1,
            "network_input": [3, 640, 640],
            "threshold": 0.20,
            "nms_iou": 0.45,
            "use": "isolated crop verification for low-confidence sidecar and auxiliary-origin track births; never changes normal primary NvDCF input",
            "scheduling": {
                "worker_contexts": 6,
                "queue_capacity_per_camera": 4,
                "global_pending_limit": 32,
                "policy": "one bounded worker per camera; no camera can queue behind another camera's crop; same-frame result still fails closed at the existing deadline",
            },
        },
        "runtime": f"DeepStream 9.1; frozen 640 YOLO26m remains per-frame primary; 896 auxiliary interval={AUX_INTERVAL}; auxiliary branch strips inherited object metadata after conversion and before inference; exact-PTS metamux; local NvDCF; per-camera bounded async frozen-640 context-verifier workers gate new primary/auxiliary target publication; no MV3DT/calibration",
        "parallel_inference_metadata_isolation": {
            "branch": "auxiliary YOLO26m-896 only",
            "mechanism": "out-of-place NVMM conversion followed by an auxiliary-only pad probe that strips inherited object metadata before auxiliary nvinfer",
            "reason": "DeepStream's metadata transform can carry primary object metadata into the converted auxiliary branch, causing nvdsmetamux to merge primary detections twice",
            "primary_branch_changed": False,
        },
        "recovery": {"required_environment": "V11_SOURCE_STALL_RECOVERY=1", "candidate_binary_sha256": sha256(binary)},
        "preview_output": {
            "tile": {"width": 960, "height": 540, "format": "BGRx"},
            "fullscreen": {"width": 1920, "height": 1080, "format": "BGRx",
                "camera_branches_open_simultaneously": 1},
            "memory": "latest-only version-5 shared memory with embedded exact-frame tracker JSON",
            "nvvideoconvert_interpolation_method": 4,
            "nvvideoconvert_interpolation_name": "Algo-3 GPU Lanczos",
            "preparation": "source-tee parallel to AI/tracker; exact decoded source-frame sequence+PTS join",
            "fullscreen_resize": "native decoded source directly to 1920x1080 in one controlled conversion",
            "join_policy": "fail closed: no shared publish unless image and tracker source-frame sequence and PTS match; 32-frame bounded per-source cache covers measured parallel branch skew; no nearest-frame fallback"},
        "preview_semantics": "parallel GPU preview preparation; tracker metadata and image are stored together under one shared-memory lock; UI renders only matching source-frame sequence and PTS",
    }
    manifest_path = CANDIDATE / "build-manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError(f"refusing to replace divergent build manifest: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
