"""Build an isolated DS 9.1 six-camera YOLO26m + local NvDCF candidate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = Path(os.getenv("SIXCAM_BBOX_CANDIDATE",
                           str(ROOT / ".runtime/sixcam-approved-bbox-20261006-r12-profile"))).resolve()
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
YOLO = ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt"
V13 = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml"
LOCAL_TRACKER = ROOT / "config/deepstream/config_tracker_NvDCF_v13_local6.yml"
MODEL_SOURCE = ROOT / ".runtime/freeze/F5R-six-camera-yolo-resolution-20261003/models/640"
PARSER_SOURCE = ROOT / ".runtime/detection-ab-peoplenet-yolo26m-20261002-pI5KEcrC/models/yolo26m/parser/libyolo26rawperson.so"
PEOPLENET_SOURCE = ROOT / ".runtime/detection-ab-peoplenet-yolo26m-20261002-pI5KEcrC/models/peopleNet"
EXPECTED = {
    "config_infer_primary_yolo26m_raw_otm.txt": "fbfa196338b6781b7d65c3058d4f02069e9116963804d0c3b23d1bf2301e4113",
    "config_tracker_NvDCF_v13_local6.yml": "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355",
    "frozen_v13_reference.yml": "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834",
    "yolo26m_raw_otm.onnx": "d3aa1ab159dba902eea9983df297cd83a56e39f62138082de7f0642958f13f33",
    "yolo26m_raw_otm_b6_fp16.engine": "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16",
    "libyolo26rawperson.so": "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19",
    "labels.txt": "bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a",
    "peoplenet_onnx": "a6aa31618f8476261807b5ba26b9841c684708c66a5f0be460a25a972f8ee7ad",
    "peoplenet_engine": "0f08176e640738caf1f94720019057f984ce0663e9363fa614051f113cef5140",
    "peoplenet_labels": "d58fbd3accfafe6676ec9243f1c8612bd4220c5df9ecb28fa7f2ac1da6bc7411",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()}


def stage_exact_inputs() -> None:
    config_dir = CANDIDATE / "config"
    model_dir = CANDIDATE / "models"
    (CANDIDATE / "bin").mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    if sha256(V13) != EXPECTED["frozen_v13_reference.yml"]:
        raise RuntimeError("frozen V13 reference hash mismatch")
    sources = {"config_infer_primary_yolo26m_raw_otm.txt": YOLO,
               "config_tracker_NvDCF_v13_local6.yml": LOCAL_TRACKER}
    for name, source in sources.items():
        if name in EXPECTED and sha256(source) != EXPECTED[name]:
            raise RuntimeError(f"protected configuration hash mismatch: {name}")
        destination = config_dir / name
        if destination.exists() and sha256(destination) != sha256(source):
            raise RuntimeError(f"refusing to replace divergent staged config: {destination}")
        if not destination.exists():
            destination.write_bytes(source.read_bytes())
    for name in ("yolo26m_raw_otm.onnx", "yolo26m_raw_otm_b6_fp16.engine",
                 "libyolo26rawperson.so", "labels.txt"):
        source = PARSER_SOURCE if name == "libyolo26rawperson.so" else MODEL_SOURCE / name
        if sha256(source) != EXPECTED[name]:
            raise RuntimeError(f"protected model artifact hash mismatch: {name}")
        destination = model_dir / name
        if destination.exists() and sha256(destination) != EXPECTED[name]:
            raise RuntimeError(f"refusing to replace divergent staged model: {destination}")
        if not destination.exists():
            destination.write_bytes(source.read_bytes())
    people_files = {
        "model/resnet34_peoplenet.onnx": (PEOPLENET_SOURCE / "model/resnet34_peoplenet.onnx", EXPECTED["peoplenet_onnx"]),
        "benchmarks/engines/peoplenet_dynamic_b2.engine": (PEOPLENET_SOURCE / "benchmarks/engines/peoplenet_dynamic_b2.engine", EXPECTED["peoplenet_engine"]),
        "config/labels.txt": (PEOPLENET_SOURCE / "config/labels.txt", EXPECTED["peoplenet_labels"]),
    }
    people_root = model_dir / "peopleNet"
    for relative, (source, expected_hash) in people_files.items():
        if sha256(source) != expected_hash:
            raise RuntimeError(f"PeopleNet verifier artifact hash mismatch: {relative}")
        destination = people_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) != expected_hash:
            raise RuntimeError(f"refusing to replace divergent staged verifier artifact: {destination}")
        if not destination.exists():
            destination.write_bytes(source.read_bytes())

    yolo_candidate = config_dir / "config_infer_primary_yolo26m_raw_otm_lowconf023.txt"
    base_config = YOLO.read_text()
    if base_config.count("pre-cluster-threshold=0.25") != 1:
        raise RuntimeError("frozen primary config threshold contract changed")
    lowconf_yolo_config = base_config.replace("pre-cluster-threshold=0.25", "pre-cluster-threshold=0.23")
    if yolo_candidate.exists() and yolo_candidate.read_text() != lowconf_yolo_config:
        raise RuntimeError(f"refusing to overwrite divergent candidate PGIE config: {yolo_candidate}")
    if not yolo_candidate.exists():
        yolo_candidate.write_text(lowconf_yolo_config)

    verifier_config = config_dir / "peoplenet_lowconf_verifier.txt"
    verifier_text = """[property]
gpu-id=0
batch-size=1
net-scale-factor=0.0039215697906911373
offsets=0.0;0.0;0.0
labelfile-path=/models/peopleNet/config/labels.txt
onnx-file=/models/peopleNet/model/resnet34_peoplenet.onnx
model-engine-file=/models/peopleNet/benchmarks/engines/peoplenet_dynamic_b2.engine
gie-unique-id=2
network-type=0
network-mode=2
num-detected-classes=3
filter-out-class-ids=1;2
infer-dims=3;544;960
model-color-format=0
maintain-aspect-ratio=1
symmetric-padding=1
output-tensor-meta=0
cluster-mode=2

[class-attrs-all]
pre-cluster-threshold=0.20
nms-iou-threshold=0.5
topk=20
"""
    if verifier_config.exists() and verifier_config.read_text() != verifier_text:
        raise RuntimeError(f"refusing to overwrite divergent verifier config: {verifier_config}")
    if not verifier_config.exists():
        verifier_config.write_text(verifier_text)

    yolo_context_config = config_dir / "yolo26m_context_verifier.txt"
    yolo_context_text = """[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file=/models/yolo26m_raw_otm.onnx
model-engine-file=/models/yolo26m_raw_otm_b6_fp16.engine
labelfile-path=/models/labels.txt
custom-lib-path=/models/libyolo26rawperson.so
parse-bbox-func-name=NvDsInferParseYolo26RawPerson
batch-size=1
network-mode=2
num-detected-classes=80
interval=0
gie-unique-id=2
process-mode=1
network-type=0
warmup-engine=1
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
infer-dims=3;640;640
output-blob-names=output0

[class-attrs-all]
pre-cluster-threshold=0.20
nms-iou-threshold=0.45
topk=300
"""
    if yolo_context_config.exists() and yolo_context_config.read_text() != yolo_context_text:
        raise RuntimeError(f"refusing to overwrite divergent context verifier config: {yolo_context_config}")
    if not yolo_context_config.exists():
        yolo_context_config.write_text(yolo_context_text)


def build() -> dict:
    stage_exact_inputs()
    yolo_context_config = CANDIDATE / "config/yolo26m_context_verifier.txt"
    source_digest = source_hashes()
    binary = CANDIDATE / "bin/cam-six-yolo-local6-posttracker"
    command = [
        "docker", "run", "--rm", "--pull=never", "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{CANDIDATE}:/work",
        IMAGE, "bash", "-lc",
        "g++ -std=c++17 -O2 -DYOLO26_PERSON -DYOLO26_TRACKER "
        "-I/opt/nvidia/deepstream/deepstream/sources/includes "
        "$(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0) "
        "/src/main.cpp -o /work/bin/cam-six-yolo-local6-posttracker "
        "$(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0) "
        "-L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta -ldl -pthread "
        "-Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    manifest = {
        "deepstream_image": IMAGE,
        "source_sha256": source_digest,
        "binary_sha256": sha256(binary),
        "config_sha256": {path.name: sha256(path) for path in sorted((CANDIDATE / "config").iterdir())},
        "model_sha256": {str(path.relative_to(CANDIDATE / "models")): sha256(path)
                         for path in sorted((CANDIDATE / "models").rglob("*")) if path.is_file()},
        "lowconf_verifier": {
            "enabled_by_default": True,
            "engine_sha256": EXPECTED["peoplenet_engine"],
            "onnx_sha256": EXPECTED["peoplenet_onnx"],
            "labels_sha256": EXPECTED["peoplenet_labels"],
            "primary_extraction_threshold": 0.23,
            "normal_nvdcf_input_threshold": 0.25,
            "network_input": [3, 544, 960],
            "runtime_batch_size": 1,
            "candidate_local_match": {"iou_min": 0.30, "center_distance_px_max": 50,
                "width_ratio_max": 2, "height_ratio_max": 2, "exactly_one_match": True},
            "async_policy": "persistent sidecar inference; per-camera ordered canonical publisher; only a uniquely continuous current INACTIVE-shadow candidate waits for its exact-frame verifier result; deadline=26ms from sidecar submission based on paced replay P99 20.759ms/max 23.882ms; late/queue-full/ambiguous fails closed; PGIE/NvDCF streaming callbacks never wait",
        },
        "yolo_context_verifier": {
            "config": yolo_context_config.name,
            "config_sha256": sha256(yolo_context_config),
            "engine_sha256": EXPECTED["yolo26m_raw_otm_b6_fp16.engine"],
            "parser_sha256": EXPECTED["libyolo26rawperson.so"],
            "runtime_batch_size": 1,
            "network_input": [3, 640, 640],
            "threshold": 0.20,
            "nms_iou": 0.45,
            "use": "isolated context confirmation only; never enters primary NvDCF",
        },
        "runtime": "DeepStream 9.1; nvinfer YOLO26m FP16 640x640 batch=6; optional isolated PeopleNet FP16 batch-1 low-confidence context verifier; calibration-free local NvDCF type2 with public shadow output; no MV3DT/calibration",
        "frozen_v13_reference_sha256": sha256(V13),
        "tracker_config_sha256": sha256(LOCAL_TRACKER),
        "tracker_config": LOCAL_TRACKER.name,
        "preview_output": {
            "tile": {"width": 960, "height": 540, "format": "BGRx"},
            "fullscreen": {"width": 1920, "height": 1080, "format": "BGRx",
                "camera_branches_open_simultaneously": 1},
            "memory": "latest-only version-5 shared memory with embedded exact-frame tracker JSON",
            "nvvideoconvert_interpolation_method": 4,
            "nvvideoconvert_interpolation_name": "Algo-3 GPU Lanczos",
            "preparation": "source-tee parallel to AI/tracker; exact decoded source-frame sequence+PTS join",
            "fullscreen_resize": "native decoded source directly to 1920x1080 in one controlled conversion",
            "join_policy": "fail closed: no shared publish unless image and tracker source-frame sequence and PTS match"},
        "preview_semantics": "parallel GPU preview preparation; tracker metadata and image are stored together under one shared-memory lock; UI renders only matching source-frame sequence and PTS",
    }
    (CANDIDATE / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
