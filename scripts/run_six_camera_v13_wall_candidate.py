"""Run the single-owner six-camera YOLO26m + NvDCF V13 UI candidate."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import yaml

from services.camera_v11.source_ownership import SourceOwnership
from services.camera_v11.tracked_object_ipc_v1 import read_camera_snapshot
from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader
from services.ml_service.app.config import load_settings
from services.shared.runtime_python import preflight_python

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = ROOT / ".runtime/sixcam-approved-bbox-20261006-r5"
CAMERAS = tuple(f"CAM-{index:02d}" for index in range(1, 7))
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
V13_CONFIG = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml"
EXPECTED = {
    "config_infer_primary_yolo26m_raw_otm.txt": "fbfa196338b6781b7d65c3058d4f02069e9116963804d0c3b23d1bf2301e4113",
    "config_tracker_NvDCF_v13_local6.yml": "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355",
    "frozen_v13_reference.yml": "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834",
    "yolo26m_raw_otm.onnx": "d3aa1ab159dba902eea9983df297cd83a56e39f62138082de7f0642958f13f33",
    "yolo26m_raw_otm_b6_fp16.engine": "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16",
    "libyolo26rawperson.so": "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19",
    "labels.txt": "bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_source_stall_test_plan(value: str, recovery_enabled: str) -> list[dict[str, int | str]]:
    """Validate explicit test-only source interruptions; production default is empty."""
    if not value:
        return []
    if recovery_enabled != "1":
        raise ValueError("V11_TEST_SOURCE_STALL_PLAN requires V11_SOURCE_STALL_RECOVERY=1")
    plan: list[dict[str, int | str]] = []
    previous_second = -1
    for item in value.split(","):
        match = re.fullmatch(r"(CAM-0[1-6])@([0-9]+)", item)
        if not match:
            raise ValueError("V11_TEST_SOURCE_STALL_PLAN entries must be CAM-XX@SECONDS")
        camera_id, raw_second = match.groups()
        second = int(raw_second)
        if not 20 <= second <= 86400:
            raise ValueError("source-stall test times must be within 20..86400 seconds")
        if previous_second >= 0 and second - previous_second < 75:
            raise ValueError("source-stall test entries must be at least 75 seconds apart")
        plan.append({"camera_id": camera_id, "at_sec": second})
        previous_second = second
    return plan


def validate_auxiliary_yolo_profile(candidate: Path, manifest: dict) -> dict:
    """Validate exact 640 primary plus a dimension-declared auxiliary YOLO profile."""
    profile = manifest.get("detector_profile", {})
    aux_input = profile.get("aux_input")
    if (not isinstance(aux_input, list) or len(aux_input) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in aux_input)
            or aux_input[0] != aux_input[1] or not 320 <= aux_input[0] <= 2048):
        raise RuntimeError("auxiliary detector input must declare one square spatial size")
    side = aux_input[0]
    aux_config_name = profile.get("aux_config")
    metamux_name = profile.get("metamux_config") or f"config_metamux_yolo26m_aux{side}.txt"
    engine_name = profile.get("aux_engine") or f"yolo26m_raw_otm_{side}_b6_fp16.engine"
    onnx_name = profile.get("aux_onnx") or f"yolo26m_raw_otm_{side}.onnx"
    parser_name = profile.get("aux_parser") or f"libyolo26rawperson_{side}.so"
    names = (aux_config_name, metamux_name, engine_name, onnx_name, parser_name)
    if any(not isinstance(name, str) or Path(name).name != name for name in names):
        raise RuntimeError("auxiliary detector artifact names must be plain filenames")
    aux_config = candidate / "config" / aux_config_name
    metamux_config = candidate / "config" / metamux_name
    aux_engine = candidate / "models" / engine_name
    aux_onnx = candidate / "models" / onnx_name
    aux_parser = candidate / "models" / parser_name
    primary_config = candidate / "config/config_infer_primary_yolo26m_raw_otm.txt"
    for path in (aux_config, metamux_config, aux_engine, aux_onnx, aux_parser, primary_config):
        if not path.is_file():
            raise RuntimeError(f"auxiliary detector artifact is missing: {path.name}")
    aux_interval = profile.get("aux_interval")
    if not isinstance(aux_interval, int) or isinstance(aux_interval, bool) or not 0 <= aux_interval <= 29:
        raise RuntimeError("auxiliary interval must be an integer in 0..29")
    expected_profile = {
        "primary_engine_sha256": EXPECTED["yolo26m_raw_otm_b6_fp16.engine"],
        "primary_parser_sha256": EXPECTED["libyolo26rawperson.so"],
        "normal_nvdcf_input_threshold": 0.25,
        "aux_input": [side, side],
        "aux_interval": aux_interval,
        "aux_unique_id": 3,
        "aux_threshold": 0.25,
        "aux_nms_iou": 0.45,
        "batch_size": 6,
        "metamux_pts_tolerance_us": 0,
    }
    if any(profile.get(key) != value for key, value in expected_profile.items()):
        raise RuntimeError("auxiliary detector profile violates its reviewed contract")
    birth_policy = profile.get("canonical_birth_safety", "")
    if (f"newly born 640-primary and {side}-auxiliary native targets" not in birth_policy
            or "without changing NvDCF input" not in birth_policy):
        raise RuntimeError("cascade candidate does not preserve the reviewed CAM-02 primary-track verifier safeguard")
    configs = manifest.get("config_sha256", {})
    models = manifest.get("model_sha256", {})
    if (sha256(primary_config) != EXPECTED["config_infer_primary_yolo26m_raw_otm.txt"]
            or sha256(aux_config) != configs.get(aux_config.name)
            or sha256(metamux_config) != configs.get(metamux_config.name)
            or sha256(aux_engine) != profile.get("aux_engine_sha256")
            or sha256(aux_engine) != models.get(aux_engine.name)
            or sha256(aux_onnx) != models.get(aux_onnx.name)
            or sha256(aux_parser) != profile.get("aux_parser_sha256")
            or sha256(aux_parser) != models.get(aux_parser.name)):
        raise RuntimeError("auxiliary detector artifact hashes disagree with the build manifest")
    aux_text = aux_config.read_text()
    required_aux = (
        "batch-size=6", f"infer-dims=3;{side};{side}", "pre-cluster-threshold=0.25",
        "nms-iou-threshold=0.45", "maintain-aspect-ratio=1", "symmetric-padding=1",
        f"interval={aux_interval}", "gie-unique-id=3",
        f"model-engine-file=/models/{engine_name}",
        f"custom-lib-path=/models/{parser_name}",
    )
    metamux_text = metamux_config.read_text()
    join_mode = profile.get("aux_join_mode", "nvdsmetamux-v1")
    if join_mode == "exact-key-sidecar-v1":
        timeout_ms = profile.get("exact_join_timeout_ms")
        if (not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool)
                or not 1 <= timeout_ms <= 100
                or profile.get("metamux_loaded", False) is not False):
            raise RuntimeError("exact sidecar join requires a bounded timeout and no runtime MetaMux")
    elif ("active-pad=sink_0" not in metamux_text
            or "pts-tolerance=0" not in metamux_text
            or "src-ids-model-1=0;1;2;3;4;5" not in metamux_text
            or "src-ids-model-3=0;1;2;3;4;5" not in metamux_text):
        raise RuntimeError("auxiliary nvinfer/metamux configuration violates exact-frame batch-6 contract")
    if any(aux_text.count(field) != 1 for field in required_aux):
        raise RuntimeError("auxiliary nvinfer configuration violates its batch-6 contract")
    return profile


def camera_config(settings) -> str:
    rows = {camera.camera_id: camera for camera in settings.cameras}
    if set(CAMERAS) - rows.keys() or any(not rows[c].username or not rows[c].password for c in CAMERAS):
        raise RuntimeError("all six configured cameras and authorized credentials are required")
    lines: list[str] = []
    for camera_id in CAMERAS:
        camera = rows[camera_id]
        values = {
            "uri": camera.uri,
            "username": camera.username,
            "password": camera.password,
        }
        lines.append(f"[{camera_id}]")
        for key, value in values.items():
            lines.append(f"{key}=" + base64.b64encode(value.encode()).decode())
        lines.append(f"latency_ms={camera.effective_latency_ms(settings.deepstream.latency_ms)}")
        lines.append(f"drop_on_latency={str(settings.deepstream.drop_on_latency).lower()}")
        lines.append(f"decoder_low_latency_mode={str(camera.decoder_low_latency_mode).lower()}")
        surfaces = camera.decoder_extra_surfaces
        if surfaces is None:
            surfaces = settings.deepstream.decoder_extra_surfaces
        lines.append(f"decoder_extra_surfaces={surfaces}")
        lines.append("")
    lines.extend(("[validation]", "interrupt_camera=none", "interrupt_at=20",
                  "interrupt_seconds=12", "preview_enabled=false", "preview_port=5600"))
    return "\n".join(lines) + "\n"


def has_auxiliary_yolo_profile(profile: dict | None) -> bool:
    """Identify an isolated 640+auxiliary cascade by declared artifacts, not name/cadence."""
    aux_input = profile.get("aux_input") if profile else None
    return bool(
        profile
        and isinstance(profile.get("aux_config"), str)
        and profile.get("aux_config", "").startswith("config_infer_aux_yolo26m_")
        and isinstance(aux_input, list) and len(aux_input) == 2
        and aux_input[0] == aux_input[1]
        and profile.get("aux_unique_id") == 3
        and profile.get("aux_engine_sha256")
        and profile.get("aux_parser_sha256")
    )


def has_auxiliary_personnet_profile(profile: dict | None) -> bool:
    """Identify the isolated PeopleNet supplemental-detector candidate by artifacts."""
    return bool(
        profile
        and profile.get("auxiliary_model") == "PeopleNet-2.6.3"
        and profile.get("aux_config") == "config_infer_aux_peoplenet_b2.txt"
        and profile.get("aux_engine_sha256")
        and profile.get("aux_unique_id") == 3
        and profile.get("aux_batch_size") == 2
        and profile.get("aux_input") == [960, 544]
    )


def has_auxiliary_detector_profile(profile: dict | None) -> bool:
    """Return true only for a fully declared supported supplemental GIE profile."""
    return has_auxiliary_yolo_profile(profile) or has_auxiliary_personnet_profile(profile)


def supports_yolo_context_profile(profile: dict | None) -> bool:
    """Return true only for explicitly manifested isolated context candidates."""
    return bool(
        profile
        and (
            profile.get("name") == "yolo26m-fp16-896-context-verified"
            or has_auxiliary_detector_profile(profile)
            or profile.get("context_verifier_mode") == "yolo26m-current-frame-crop-confirmation"
            or profile.get("person_context_verifier_mode") == "peoplenet-current-frame-crop-confirmation"
        )
    )


def auxiliary_detector_environment(profile: dict, *, config_root: str = "/config") -> list[str]:
    """Render the declared auxiliary profile while retaining old-binary compatibility."""
    if has_auxiliary_personnet_profile(profile):
        return [
            "V11_AUX_YOLO_ENABLE=1",
            f"V11_AUX_YOLO_CONFIG={config_root}/{profile['aux_config']}",
            f"V11_AUX_YOLO_METAMUX_CONFIG={config_root}/{profile['metamux_config']}",
            "V11_AUX_YOLO_INPUT_SIZE=960",
        ]
    if not has_auxiliary_yolo_profile(profile):
        return []
    side = int(profile["aux_input"][0])
    config = profile["aux_config"]
    metamux = profile.get("metamux_config") or f"config_metamux_yolo26m_aux{side}.txt"
    if profile.get("aux_join_mode") == "exact-key-sidecar-v1":
        return [
            "V11_AUX_YOLO_ENABLE=1",
            f"V11_AUX_YOLO_CONFIG={config_root}/{config}",
            f"V11_AUX_YOLO_INPUT_SIZE={side}",
            f"V11_AUX_EXACT_JOIN_TIMEOUT_MS={int(profile['exact_join_timeout_ms'])}",
        ]
    if profile.get("aux_environment_contract") == "generic-v1":
        environment = [
            "V11_AUX_YOLO_ENABLE=1",
            f"V11_AUX_YOLO_CONFIG={config_root}/{config}",
            f"V11_AUX_YOLO_METAMUX_CONFIG={config_root}/{metamux}",
            f"V11_AUX_YOLO_INPUT_SIZE={side}",
        ]
        if "metamux_latency_ms" in profile:
            environment.append(f"V11_AUX_METAMUX_LATENCY_MS={int(profile['metamux_latency_ms'])}")
        return environment
    # Historical 896 candidates use these names; the current native binary
    # also accepts them so prior diagnostic candidates remain reproducible.
    environment = [
        "V11_AUX_YOLO896_ENABLE=1",
        f"V11_AUX_YOLO896_CONFIG={config_root}/{config}",
        f"V11_AUX_YOLO896_METAMUX_CONFIG={config_root}/{metamux}",
    ]
    if "metamux_latency_ms" in profile:
        environment.append(f"V11_AUX_METAMUX_LATENCY_MS={int(profile['metamux_latency_ms'])}")
    return environment


def validate_auxiliary_personnet_profile(candidate: Path, manifest: dict) -> dict:
    """Validate the candidate-only PeopleNet full-frame GIE and exact-PTS merge."""
    from scripts.build_six_camera_v13_wall_candidate import EXPECTED as BUILD_EXPECTED

    profile = manifest.get("detector_profile", {})
    if not has_auxiliary_personnet_profile(profile):
        raise RuntimeError("supplemental PeopleNet profile is incomplete")
    config_name = profile["aux_config"]
    metamux_name = profile.get("metamux_config")
    if not isinstance(metamux_name, str) or Path(metamux_name).name != metamux_name:
        raise RuntimeError("supplemental PeopleNet metamux config is not a filename")
    config = candidate / "config" / config_name
    metamux = candidate / "config" / metamux_name
    engine = candidate / "models/peopleNet/benchmarks/engines/peoplenet_dynamic_b2.engine"
    onnx = candidate / "models/peopleNet/model/resnet34_peoplenet.onnx"
    labels = candidate / "models/peopleNet/config/labels.txt"
    if any(not path.is_file() for path in (config, metamux, engine, onnx, labels)):
        raise RuntimeError("supplemental PeopleNet config or model artifact is missing")
    hashes = manifest.get("config_sha256", {})
    models = manifest.get("model_sha256", {})
    if (sha256(config) != profile.get("aux_config_sha256")
            or hashes.get(config_name) != profile.get("aux_config_sha256")
            or sha256(metamux) != hashes.get(metamux_name)
            or sha256(engine) != profile.get("aux_engine_sha256")
            or profile.get("aux_engine_sha256") != BUILD_EXPECTED["peoplenet_engine"]
            or models.get("peopleNet/benchmarks/engines/peoplenet_dynamic_b2.engine") != profile.get("aux_engine_sha256")
            or sha256(onnx) != BUILD_EXPECTED["peoplenet_onnx"]
            or sha256(labels) != BUILD_EXPECTED["peoplenet_labels"]):
        raise RuntimeError("supplemental PeopleNet artifact hashes disagree with the candidate manifest")
    text = config.read_text()
    required = (
        "batch-size=2", "gie-unique-id=3", "infer-dims=3;544;960",
        "model-engine-file=/models/peopleNet/benchmarks/engines/peoplenet_dynamic_b2.engine",
        "filter-out-class-ids=1;2", "pre-cluster-threshold=0.23", "interval=0",
    )
    metamux_text = metamux.read_text()
    if (any(text.count(field) != 1 for field in required)
            or "active-pad=sink_0" not in metamux_text
            or "pts-tolerance=0" not in metamux_text
            or "src-ids-model-1=0;1;2;3;4;5" not in metamux_text
            or "src-ids-model-3=0;1;2;3;4;5" not in metamux_text):
        raise RuntimeError("supplemental PeopleNet GIE violates its batch-6 exact-PTS contract")
    if profile.get("aux_interval") != 0 or profile.get("batch_size") != 6:
        raise RuntimeError("supplemental PeopleNet must run every frame on the six-source path")
    policy = profile.get("canonical_birth_safety", "")
    if ("newly born 640-primary and PeopleNet auxiliary native targets" not in policy
            or "without changing NvDCF input" not in policy):
        raise RuntimeError("supplemental PeopleNet target-birth safety policy is missing")
    return profile


def validate_candidate(candidate: Path) -> dict:
    manifest_path = candidate / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("deepstream_image") != IMAGE:
        raise RuntimeError("candidate was not built against the pinned DeepStream 9.1 image")
    binary = candidate / "bin/cam-six-yolo-local6-posttracker"
    if sha256(binary) != manifest.get("binary_sha256"):
        raise RuntimeError("candidate binary hash does not match its build manifest")
    profile = manifest.get("detector_profile")
    profile_name = profile.get("name") if profile else ""
    candidate_detector = profile_name == "yolo26m-fp16-896-context-verified"
    cascade_detector = has_auxiliary_yolo_profile(profile)
    for name, expected in EXPECTED.items():
        if name == "frozen_v13_reference.yml":
            if sha256(V13_CONFIG) != expected:
                raise RuntimeError("frozen V13 reference config hash changed")
            continue
        if candidate_detector and name == "config_infer_primary_yolo26m_raw_otm.txt":
            continue
        path = ((candidate / "config" / name) if name.startswith("config_") else (candidate / "models" / name))
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"candidate frozen input hash mismatch: {name}")
    if manifest.get("source_sha256") != __import__(
            "scripts.build_six_camera_v13_wall_candidate", fromlist=["source_hashes"]
    ).source_hashes():
        raise RuntimeError("native source changed after candidate build")
    if manifest.get("frozen_v13_reference_sha256") != EXPECTED["frozen_v13_reference.yml"]:
        raise RuntimeError("candidate does not pin the exact frozen V13 reference")
    if manifest.get("tracker_config_sha256") != EXPECTED["config_tracker_NvDCF_v13_local6.yml"]:
        raise RuntimeError("candidate local tracker hash is incorrect")
    if candidate_detector:
        primary = candidate / "config/config_infer_primary_yolo26m_raw_otm.txt"
        engine = candidate / "models/yolo26m_raw_otm_896_b6_fp16.engine"
        onnx = candidate / "models/yolo26m_raw_otm_896.onnx"
        parser = candidate / "models/libyolo26rawperson_896.so"
        profile_config_hash = profile.get("config_sha256")
        expected_profile = {
            "input": [896, 896], "confidence": 0.25, "nms_iou": 0.45,
            "batch_size": 6, "aspect_ratio": 1, "symmetric_padding": 1, "interval": 0,
            "frozen_640_baseline_engine_sha256": EXPECTED["yolo26m_raw_otm_b6_fp16.engine"],
            "frozen_640_baseline_parser_sha256": EXPECTED["libyolo26rawperson.so"],
        }
        if any(profile.get(key) != value for key, value in expected_profile.items()):
            raise RuntimeError("candidate 896 detector profile violates the reviewed detector contract")
        if any(not path.is_file() for path in (primary, engine, onnx, parser)):
            raise RuntimeError("candidate 896 primary detector artifact is missing")
        hashes = manifest.get("config_sha256", {})
        models = manifest.get("model_sha256", {})
        if (sha256(primary) != profile_config_hash
                or hashes.get(primary.name) != profile_config_hash
                or sha256(engine) != profile.get("engine_sha256")
                or models.get(engine.name) != profile.get("engine_sha256")
                or sha256(onnx) != profile.get("onnx_sha256")
                or models.get(onnx.name) != profile.get("onnx_sha256")
                or sha256(parser) != profile.get("parser_sha256")
                or models.get(parser.name) != profile.get("parser_sha256")):
            raise RuntimeError("candidate 896 detector hashes disagree with its profile manifest")
        text = primary.read_text()
        required = (
            "batch-size=6", "infer-dims=3;896;896", "pre-cluster-threshold=0.25",
            "nms-iou-threshold=0.45", "maintain-aspect-ratio=1", "symmetric-padding=1",
            "interval=0", "model-engine-file=/models/yolo26m_raw_otm_896_b6_fp16.engine",
            "custom-lib-path=/models/libyolo26rawperson_896.so",
        )
        if any(text.count(field) != 1 for field in required):
            raise RuntimeError("candidate 896 primary config does not preserve the exact detector contract")
    if cascade_detector:
        profile = validate_auxiliary_yolo_profile(candidate, manifest)
    if has_auxiliary_personnet_profile(profile):
        profile = validate_auxiliary_personnet_profile(candidate, manifest)
    preview = manifest.get("preview_output", {})
    if (preview.get("tile", {}).get("width") != 960
            or preview.get("fullscreen", {}).get("width") != 1920
            or preview.get("join_policy", "").find("fail closed") < 0):
        raise RuntimeError("candidate does not stage the exact-frame dual-resolution preview contract")
    return manifest


def validate_lowconf_verifier_artifacts(candidate: Path, manifest: dict) -> None:
    """Fail closed unless the isolated verifier matches its frozen stage manifest."""
    from scripts.build_six_camera_v13_wall_candidate import EXPECTED as BUILD_EXPECTED

    files = {
        "primary": candidate / "config/config_infer_primary_yolo26m_raw_otm_lowconf023.txt",
        "verifier_config": candidate / "config/peoplenet_lowconf_verifier.txt",
        "verifier_engine": candidate / "models/peopleNet/benchmarks/engines/peoplenet_dynamic_b2.engine",
        "verifier_onnx": candidate / "models/peopleNet/model/resnet34_peoplenet.onnx",
        "verifier_labels": candidate / "models/peopleNet/config/labels.txt",
    }
    for label, path in files.items():
        if not path.is_file():
            raise RuntimeError(f"low-confidence verifier artifact is missing: {label}")

    config_hashes = manifest.get("config_sha256", {})
    model_hashes = manifest.get("model_sha256", {})
    config_names = {
        "primary": "config_infer_primary_yolo26m_raw_otm_lowconf023.txt",
        "verifier_config": "peoplenet_lowconf_verifier.txt",
    }
    model_names = {
        "verifier_engine": "peopleNet/benchmarks/engines/peoplenet_dynamic_b2.engine",
        "verifier_onnx": "peopleNet/model/resnet34_peoplenet.onnx",
        "verifier_labels": "peopleNet/config/labels.txt",
    }
    for label, name in config_names.items():
        if sha256(files[label]) != config_hashes.get(name):
            raise RuntimeError(f"low-confidence verifier config hash mismatch: {name}")
    for label, name in model_names.items():
        if sha256(files[label]) != model_hashes.get(name):
            raise RuntimeError(f"low-confidence verifier model hash mismatch: {name}")

    expected_hashes = {
        "verifier_engine": BUILD_EXPECTED["peoplenet_engine"],
        "verifier_onnx": BUILD_EXPECTED["peoplenet_onnx"],
        "verifier_labels": BUILD_EXPECTED["peoplenet_labels"],
    }
    for label, expected in expected_hashes.items():
        if sha256(files[label]) != expected:
            raise RuntimeError(f"low-confidence verifier frozen artifact mismatch: {label}")

    primary_text = files["primary"].read_text()
    verifier_text = files["verifier_config"].read_text()
    if (primary_text.count("pre-cluster-threshold=0.23") != 1
            or primary_text.count("batch-size=6") != 1
            or "model-engine-file=/models/yolo26m_raw_otm_b6_fp16.engine" not in primary_text
            or "pre-cluster-threshold=0.25" in primary_text):
        raise RuntimeError("low-confidence primary extraction config violates the 0.23/batch-6 contract")
    if ("batch-size=1" not in verifier_text
            or "infer-dims=3;544;960" not in verifier_text
            or "pre-cluster-threshold=0.20" not in verifier_text
            or "filter-out-class-ids=1;2" not in verifier_text):
        raise RuntimeError("PeopleNet verifier config violates the reviewed person-only contract")
    verifier_manifest = manifest.get("lowconf_verifier", {})
    if (verifier_manifest.get("engine_sha256") != BUILD_EXPECTED["peoplenet_engine"]
            or verifier_manifest.get("normal_nvdcf_input_threshold") != 0.25
            or verifier_manifest.get("primary_extraction_threshold") != 0.23):
        raise RuntimeError("low-confidence verifier provenance contract is invalid")


def validate_yolo_context_verifier(candidate: Path, manifest: dict) -> None:
    """Prove the isolated persistent verifier matches its staged artifact manifest."""
    lowconf_primary = candidate / "config/config_infer_primary_yolo26m_raw_otm_lowconf023.txt"
    context = manifest.get("yolo_context_verifier", {})
    config_name = context.get("config", "yolo26m_context_verifier.txt")
    config_path = candidate / "config" / config_name
    if not config_path.is_file():
        raise RuntimeError("YOLO context verifier config is missing")
    if not lowconf_primary.is_file():
        raise RuntimeError("0.23 extraction candidate config is missing")
    text = config_path.read_text()
    lowconf_text = lowconf_primary.read_text()
    config_hashes = manifest.get("config_sha256", {})
    model_hashes = manifest.get("model_sha256", {})
    network_input = context.get("network_input", [3, 640, 640])
    if len(network_input) != 3 or network_input[1] != network_input[2]:
        raise RuntimeError("YOLO context verifier must use a square staged input")
    verifier_side = int(network_input[1])
    verifier_batch = int(context.get("runtime_batch_size", 1))
    if verifier_batch not in {1, 2, 6}:
        raise RuntimeError("YOLO context verifier runtime batch size must be 1, 2, or 6")
    if verifier_batch == 6:
        profile = context.get("engine_batch_profile", {})
        if (int(profile.get("min", 0)) > 1 or int(profile.get("opt", 0)) != 6
                or int(profile.get("max", 0)) != 6):
            raise RuntimeError("batch-6 context verifier requires a manifested TensorRT profile 1..6, opt 6")
    verifier_engine = context.get("engine", "yolo26m_raw_otm_b6_fp16.engine")
    verifier_parser = context.get("parser", "libyolo26rawperson.so")
    if sha256(config_path) != config_hashes.get(config_path.name):
        raise RuntimeError("YOLO context verifier config hash mismatch")
    if (text.count(f"batch-size={verifier_batch}") != 1 or text.count("gie-unique-id=2") != 1
            or text.count(f"infer-dims=3;{verifier_side};{verifier_side}") != 1
            or text.count(f"model-engine-file=/models/{verifier_engine}") != 1
            or text.count(f"custom-lib-path=/models/{verifier_parser}") != 1
            or "parse-bbox-func-name=NvDsInferParseYolo26RawPerson" not in text
            or "pre-cluster-threshold=0.20" not in text
            or "nms-iou-threshold=0.45" not in text):
        raise RuntimeError("YOLO context verifier config violates its isolated batch contract")
    if (context.get("config_sha256") != sha256(config_path)
            or model_hashes.get(verifier_engine) != context.get("engine_sha256")
            or model_hashes.get(verifier_parser) != context.get("parser_sha256")
            or sha256(candidate / "models" / verifier_engine) != context.get("engine_sha256")
            or sha256(candidate / "models" / verifier_parser) != context.get("parser_sha256")):
        raise RuntimeError("YOLO context verifier model/config provenance is invalid")
    if manifest.get("native_preprocessed_context_backend") is True:
        preprocessor = manifest.get("context_preprocessor", {})
        library_name = Path(str(preprocessor.get("library", ""))).name
        library_path = candidate / "lib" / library_name
        if (not library_name or not str(preprocessor.get("library", "")).startswith("/contextlib/")
                or not library_path.is_file()
                or sha256(library_path) != preprocessor.get("library_sha256")):
            raise RuntimeError("native verifier preprocessor library provenance is invalid")
    if verifier_side == 640 and (context.get("engine_sha256") != EXPECTED["yolo26m_raw_otm_b6_fp16.engine"]
            or context.get("parser_sha256") != EXPECTED["libyolo26rawperson.so"]):
        raise RuntimeError("default 640 verifier must remain the frozen engine/parser")
    if (sha256(lowconf_primary) != config_hashes.get(lowconf_primary.name)
            or lowconf_text.count("pre-cluster-threshold=0.23") != 1
            or lowconf_text.count("batch-size=6") != 1
            or lowconf_text.count("infer-dims=3;640;640") != 1
            or "model-engine-file=/models/yolo26m_raw_otm_b6_fp16.engine" not in lowconf_text
            or "custom-lib-path=/models/libyolo26rawperson.so" not in lowconf_text
            or "nms-iou-threshold=0.45" not in lowconf_text
            or "maintain-aspect-ratio=1" not in lowconf_text
            or "symmetric-padding=1" not in lowconf_text):
        raise RuntimeError("low-confidence extraction config does not preserve frozen 640 detector semantics")


def established_rtsp_connections() -> list[str]:
    output = subprocess.check_output(
        ["ss", "-Htnp", "state", "established", "( dport = :554 )"], text=True
    )
    return [line for line in output.splitlines() if line.strip()]


def docker_command(candidate: Path, output: Path, camera_file: Path, duration: int,
                   lowconf_verifier: bool = False,
                   yolo_context_verifier: bool = False,
                   active_track_verifier: bool = False) -> list[str]:
    name = re.sub(r"[^A-Za-z0-9_.-]", "-", f"surv-sixcam-{output.name}")[:120]
    verbose_evidence = os.getenv("V11_VERBOSE_EVIDENCE", "0")
    if verbose_evidence not in {"0", "1"}:
        raise ValueError("V11_VERBOSE_EVIDENCE must be 0 or 1")
    source_recovery = os.getenv("V11_SOURCE_STALL_RECOVERY", "0")
    if source_recovery not in {"0", "1"}:
        raise ValueError("V11_SOURCE_STALL_RECOVERY must be 0 or 1")
    test_stall_plan = os.getenv("V11_TEST_SOURCE_STALL_PLAN", "")
    parsed_stall_plan = parse_source_stall_test_plan(test_stall_plan, source_recovery)
    stage_timing = os.getenv("V11_PIPELINE_STAGE_TIMING", "0")
    if stage_timing not in {"0", "1"}:
        raise ValueError("V11_PIPELINE_STAGE_TIMING must be 0 or 1")
    command = [
        "docker", "run", "--rm", "--pull=never", "--init", "--name", name,
        "--network=host", "--gpus", "all", "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "-e", "PYTHONNOUSERSITE=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", f"V11_VERBOSE_EVIDENCE={verbose_evidence}",
        "-e", f"V11_SOURCE_STALL_RECOVERY={source_recovery}",
        "-e", "V11_TRACKED_OBJECT_FEED_DIR=/dev/shm",
        "-e", "V11_MONITORING_TELEMETRY_PATH=/work/monitoring.json",
        "--mount", f"type=bind,src={candidate / 'bin'},dst=/candidate/bin,readonly",
        "--mount", f"type=bind,src={candidate / 'config'},dst=/config,readonly",
        "--mount", f"type=bind,src={candidate / 'models'},dst=/models,readonly",
        "--mount", f"type=bind,src={camera_file},dst=/run/camera-six-secret.ini,readonly",
        "--mount", f"type=bind,src={output},dst=/work",
        "--mount", "type=bind,src=/dev/shm,dst=/dev/shm",
        IMAGE, "/candidate/bin/cam-six-yolo-local6-posttracker", "/run/camera-six-secret.ini", str(duration),
    ]
    manifest_path = candidate / "build-manifest.json"
    profile = (json.loads(manifest_path.read_text()).get("detector_profile", {})
               if manifest_path.is_file() else {})
    if has_auxiliary_yolo_profile(profile):
        command[command.index("--mount"):command.index("--mount")] = [
            *[item for setting in auxiliary_detector_environment(profile)
              for item in ("-e", setting)],
        ]
    if parsed_stall_plan:
        command[command.index("--mount"):command.index("--mount")] = [
            "-e", f"V11_TEST_SOURCE_STALL_PLAN={test_stall_plan}"
        ]
    if stage_timing == "1":
        command[command.index("--mount"):command.index("--mount")] = [
            "-e", "V11_PIPELINE_STAGE_TIMING=1"
        ]
    mux_timeout = os.getenv("V11_MUX_BATCHED_PUSH_TIMEOUT_US", "1000")
    parsed = int(mux_timeout)
    if not 1000 <= parsed <= 50000:
        raise ValueError("V11_MUX_BATCHED_PUSH_TIMEOUT_US must be in 1000..50000")
    command[command.index("--mount"):command.index("--mount")] = [
        "-e", f"V11_MUX_BATCHED_PUSH_TIMEOUT_US={parsed}"
    ]
    if os.getenv("V11_TRACKER_DECISION_DIAGNOSTICS") == "1":
        command[command.index("--mount"):command.index("--mount")] = [
            "-e", "V11_TRACKER_DECISION_DIAGNOSTICS=1"
        ]
    tracker_nan_diagnostics = os.getenv("V11_TRACKER_NAN_DIAGNOSTICS", "0")
    if tracker_nan_diagnostics not in {"0", "1"}:
        raise ValueError("V11_TRACKER_NAN_DIAGNOSTICS must be 0 or 1")
    if tracker_nan_diagnostics == "1":
        command[command.index("--mount"):command.index("--mount")] = [
            "-e", "V11_TRACKER_NAN_DIAGNOSTICS=1"
        ]
    for key in ("V11_PREVIEW_CAPTURE_DIR", "V11_PREVIEW_CAPTURE_SOURCE_FRAME"):
        value = os.getenv(key)
        if value:
            command[command.index("--mount"):command.index("--mount")] = ["-e", f"{key}={value}"]
    for key in ("V11_MUX_WIDTH", "V11_MUX_HEIGHT", "V11_TRACKER_WIDTH", "V11_TRACKER_HEIGHT"):
        value = os.getenv(key)
        if value:
            number = int(value)
            if not 320 <= number <= 4096:
                raise ValueError(f"{key} must be in 320..4096")
            command[command.index("--mount"):command.index("--mount")] = ["-e", f"{key}={number}"]
    if lowconf_verifier:
        command[command.index("--mount"):command.index("--mount")] = [
            "-e", "V11_LOWCONF_VERIFIER_ENABLE=1",
            "-e", "V11_LOWCONF_PRIMARY_CONFIG=/config/config_infer_primary_yolo26m_raw_otm_lowconf023.txt",
            "-e", "V11_LOWCONF_VERIFIER_CONFIG=/config/peoplenet_lowconf_verifier.txt",
        ]
    if yolo_context_verifier:
        manifest = json.loads((candidate / "build-manifest.json").read_text())
        context = manifest.get("yolo_context_verifier", {})
        context_config = context.get("config", "yolo26m_context_verifier.txt")
        context_side = int((context.get("network_input") or [3, 640, 640])[1])
        command[command.index("--mount"):command.index("--mount")] = [
            "-e", "V11_LOWCONF_VERIFIER_ENABLE=1",
            "-e", "V11_ACTIVE_TRACK_VERIFIER_ENABLE=1",
            "-e", "V11_LOWCONF_PRIMARY_CONFIG=/config/config_infer_primary_yolo26m_raw_otm_lowconf023.txt",
            "-e", f"V11_LOWCONF_VERIFIER_CONFIG=/config/{context_config}",
            "-e", f"V11_CONTEXT_VERIFIER_NAME=YOLO26m-{context_side}-FP16",
            "-e", f"V11_CONTEXT_VERIFIER_BATCH_SIZE={int(context.get('runtime_batch_size', 1))}",
            "-e", "V11_CONTEXT_VERIFIER_EVIDENCE=1",
        ]
        if manifest.get("shadow_context_refresh_enabled") is True:
            command[command.index("--mount"):command.index("--mount")] = [
                "-e", "V11_SHADOW_CONTEXT_REFRESH_ENABLE=1"]
        if manifest.get("gpu_context_crop") is True:
            command[command.index("--mount"):command.index("--mount")] = [
                "-e", "V11_CONTEXT_VERIFIER_GPU_CROP=1"]
        if manifest.get("native_trt_runtime_backend") is True:
            command[command.index("--mount"):command.index("--mount")] = [
                "-e", "V11_CONTEXT_VERIFIER_BACKEND=trt-runtime",
                "-e", "V11_CONTEXT_VERIFIER_TRACE=1"]
        elif manifest.get("native_context_backend") is True:
            command[command.index("--mount"):command.index("--mount")] = [
                "-e", "V11_CONTEXT_VERIFIER_BACKEND=nvdsinfer-context"]
        if manifest.get("per_camera_context_backend") is True:
            command[command.index("--mount"):command.index("--mount")] = [
                "-e", "V11_CONTEXT_VERIFIER_PER_CAMERA=1"]
        if manifest.get("native_preprocessed_context_backend") is True:
            preprocessor = manifest["context_preprocessor"]
            command[command.index("--mount"):command.index("--mount")] = [
                "-e", "V11_CONTEXT_VERIFIER_PREPROCESSED=1",
                "-e", f"V11_CONTEXT_PREPROCESS_LIBRARY={preprocessor['library']}",
                "--mount", f"type=bind,src={candidate / 'lib'},dst=/contextlib,readonly",
            ]
    if active_track_verifier:
        command[command.index("--mount"):command.index("--mount")] = [
            "-e", "V11_ACTIVE_TRACK_VERIFIER_ENABLE=1",
            "-e", "V11_ACTIVE_TRACK_VERIFIER_FAIL_CLOSED=1",
            "-e", "V11_LOWCONF_VERIFIER_CONFIG=/config/yolo26m_context_verifier.txt",
            "-e", "V11_CONTEXT_VERIFIER_NAME=YOLO26m-640-FP16",
            "-e", "V11_CONTEXT_VERIFIER_EVIDENCE=1",
        ]
    return command


def wait_for_preview_progress(timeout_sec: float = 90.0) -> dict:
    readers = {camera: PreviewFrameReader(
        f"/dev/shm/v11_ui_preview_{camera.lower().replace('-', '')}_v1.bin"
    ) for camera in CAMERAS}
    seen: dict[str, list[int]] = {camera: [] for camera in CAMERAS}
    tracked_seen: dict[str, list[int]] = {camera: [] for camera in CAMERAS}
    deadline = time.monotonic() + timeout_sec
    try:
        while time.monotonic() < deadline:
            for camera, reader in readers.items():
                frame = reader.read_latest(max_age_sec=1.0)
                if frame and frame.timestamp_ns >= 1:
                    if (frame.tracker_snapshot is None
                            or int(frame.tracker_snapshot.get("pts_ns", -1)) != frame.pts_ns
                            or int(frame.tracker_snapshot.get("source_frame_num", -1)) != frame.source_frame_num):
                        continue
                    if not seen[camera] or frame.sequence != seen[camera][-1]:
                        seen[camera].append(frame.sequence)
                tracked = read_camera_snapshot(camera, max_age_sec=1.0)
                if tracked is not None:
                    frame_num = int(tracked["frame_num"])
                    if not tracked_seen[camera] or tracked_seen[camera][-1] != frame_num:
                        tracked_seen[camera].append(frame_num)
            if all(len(values) >= 2 for values in seen.values()) and all(
                    len(values) >= 2 for values in tracked_seen.values()):
                return {camera: {"preview_sequences": seen[camera][-2:],
                                 "tracker_frame_nums": tracked_seen[camera][-2:],
                                 "fps": readers[camera].read_latest().fps,
                                 "preview_size": [
                                     readers[camera].read_latest().width,
                                     readers[camera].read_latest().height,
                                 ],
                                 "exact_embedded_metadata": True}
                        for camera in CAMERAS}
            time.sleep(0.1)
        missing = [camera for camera in CAMERAS if len(seen[camera]) < 2 or len(tracked_seen[camera]) < 2]
        raise TimeoutError("preview/canonical tracker feed did not advance for: " + ", ".join(missing))
    finally:
        for reader in readers.values():
            reader.close()


def run(candidate: Path, output: Path, duration: int, lowconf_verifier: bool = False,
        yolo_context_verifier: bool = False,
        active_track_verifier: bool = False) -> int:
    preflight_python(ROOT)
    candidate = candidate.resolve()
    output = output.resolve()
    if not candidate.is_relative_to(ROOT / ".runtime") or not output.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate and run output must stay under .runtime")
    if duration < 20:
        raise ValueError("native candidate duration must be at least 20 seconds")
    manifest = validate_candidate(candidate)
    selected_modes = sum((lowconf_verifier, yolo_context_verifier, active_track_verifier))
    if selected_modes > 1:
        raise ValueError("choose only one context-verifier runtime mode")
    if yolo_context_verifier or active_track_verifier:
        profile = manifest.get("detector_profile", {})
        if not supports_yolo_context_profile(profile):
            raise ValueError("YOLO context verification requires an explicitly manifested isolated context candidate")
        validate_yolo_context_verifier(candidate, manifest)
    if has_auxiliary_yolo_profile(manifest.get("detector_profile")) and not yolo_context_verifier:
        raise ValueError("the 640+896 cascade requires both low-confidence rescue and fail-closed auxiliary birth verification")
    source_recovery = os.getenv("V11_SOURCE_STALL_RECOVERY", "0")
    if source_recovery not in {"0", "1"}:
        raise ValueError("V11_SOURCE_STALL_RECOVERY must be 0 or 1")
    parsed_stall_plan = parse_source_stall_test_plan(
        os.getenv("V11_TEST_SOURCE_STALL_PLAN", ""), source_recovery)
    settings = load_settings(ROOT / "config/cameras.yaml")
    output.mkdir(parents=True, exist_ok=False)
    camera_file = output / "camera-six-secret.ini"
    fd = os.open(camera_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(camera_config(settings))
    lock_dir = ROOT / ".runtime/camera-owner-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    command = docker_command(candidate, output, camera_file, duration,
                             lowconf_verifier, yolo_context_verifier, active_track_verifier)
    detector_profile = manifest.get("detector_profile", {})
    runtime_pgie_config = (
        detector_profile.get("primary_config")
        if (lowconf_verifier or yolo_context_verifier) and detector_profile.get("primary_config")
        else "config_infer_primary_yolo26m_raw_otm.txt"
    )
    (output / "launch.json").write_text(json.dumps({
        "command": ["<redacted-candidate-binary>", "<private-camera-config>", duration],
        "camera_ids": list(CAMERAS), "rtsp_uri_count": 6,
        "detector_profile": manifest.get("detector_profile", {}).get("name", "yolo26m-fp16-640-frozen"),
        "detector_config": runtime_pgie_config,
        "detector_config_sha256": manifest.get("config_sha256", {}).get(runtime_pgie_config),
        "tracker_config": "config_tracker_NvDCF_v13_local6.yml",
        "tracker_config_sha256": EXPECTED["config_tracker_NvDCF_v13_local6.yml"],
        "frozen_v13_reference_sha256": EXPECTED["frozen_v13_reference.yml"],
        "engine_sha256": (manifest.get("detector_profile", {}).get("engine_sha256")
                          or manifest.get("detector_profile", {}).get("primary_engine_sha256")
                          or EXPECTED["yolo26m_raw_otm_b6_fp16.engine"]),
        "parser_sha256": (manifest.get("detector_profile", {}).get("parser_sha256")
                          or manifest.get("detector_profile", {}).get("primary_parser_sha256")
                          or EXPECTED["libyolo26rawperson.so"]),
        "auxiliary_detector": ({
            "enabled": True,
            "engine_sha256": manifest.get("detector_profile", {}).get("aux_engine_sha256"),
            "parser_sha256": manifest.get("detector_profile", {}).get("aux_parser_sha256"),
            "interval": manifest.get("detector_profile", {}).get("aux_interval"),
            "unique_component_id": manifest.get("detector_profile", {}).get("aux_unique_id"),
            "metamux_pts_tolerance_us": manifest.get("detector_profile", {}).get("metamux_pts_tolerance_us"),
            "join_mode": manifest.get("detector_profile", {}).get("aux_join_mode", "nvdsmetamux-v1"),
            "exact_join_timeout_ms": manifest.get("detector_profile", {}).get("exact_join_timeout_ms"),
            "metamux_loaded": manifest.get("detector_profile", {}).get("metamux_loaded", True),
        } if has_auxiliary_yolo_profile(manifest.get("detector_profile")) else {"enabled": False}),
        "binary_sha256": manifest["binary_sha256"], "batch_size": 6,
        "source_stall_recovery_enabled": source_recovery == "1",
        "mux_batched_push_timeout_us": int(os.getenv("V11_MUX_BATCHED_PUSH_TIMEOUT_US", "1000")),
        "mux_width": int(os.getenv("V11_MUX_WIDTH", "2560")),
        "mux_height": int(os.getenv("V11_MUX_HEIGHT", "1440")),
        "tracker_width": int(os.getenv("V11_TRACKER_WIDTH", "960")),
        "tracker_height": int(os.getenv("V11_TRACKER_HEIGHT", "544")),
        "mv3dt_enabled": False,
        "calibration_required": False,
        "tracker_bbox_source": "current ACTIVE NvDCF bbox; or current PGIE bbox after same-frame verified unique continuous INACTIVE-shadow association; shadow geometry is never rendered",
        "preview_overlay_stage": "Qt draws only canonical tracker IPC rows over clean shared-memory frame",
        "preview_width": 960, "preview_height": 540,
        "preview_fullscreen_width": 1920, "preview_fullscreen_height": 1080,
        "preview_preparation": "parallel source-tee branches; tile 960x540 all cameras; fullscreen 1920x1080 only selected camera",
        "preview_join": "exact source frame sequence + PTS; 32-frame bounded per-source cache; snapshot embedded with image in one flock-protected slot; no nearest-frame fallback",
        "preview_interpolation_method": "nvvideoconvert Algo-3 GPU Lanczos (enum 4)",
        "tracked_object_feed": "six atomic latest-only /dev/shm/v11_tracked_objects_camXX_v1.json snapshots",
        "verified_lowconf_rescue": {
            "enabled": lowconf_verifier or yolo_context_verifier,
            "primary_extraction_floor": 0.23 if (lowconf_verifier or yolo_context_verifier) else 0.25,
            "nvdcf_input_floor": 0.25,
            "verifier_engine_sha256": (
                manifest.get("yolo_context_verifier", {}).get("engine_sha256")
                if yolo_context_verifier else
                manifest.get("lowconf_verifier", {}).get("engine_sha256") if lowconf_verifier else None),
            "fail_closed_same_frame": True,
        },
        "active_track_birth_verification": {
            "enabled": yolo_context_verifier or active_track_verifier,
            "fail_closed_until_positive": yolo_context_verifier or active_track_verifier,
            "canonical_only": True,
            "primary_640_nvdcf_input_preserved": True,
            "auxiliary_proposals_added": has_auxiliary_yolo_profile(manifest.get("detector_profile")),
            "requires_exact_same_frame_detector_crop": True,
            "native_track_result_cached_until_track_and_shadow_expire": True,
        },
        "verbose_per_frame_evidence": os.getenv("V11_VERBOSE_EVIDENCE", "0") == "1",
        "tracker_nan_diagnostics": os.getenv("V11_TRACKER_NAN_DIAGNOSTICS", "0") == "1",
        "tracker_decision_diagnostics": os.getenv("V11_TRACKER_DECISION_DIAGNOSTICS", "0") == "1",
        "controlled_source_stall_test_plan": parsed_stall_plan,
    }, indent=2) + "\n")
    # Lifetime locks and the socket check prevent this owner from being layered
    # over any of the existing six preview readers.
    with SourceOwnership(CAMERAS, lock_dir):
        current = established_rtsp_connections()
        if current:
            raise RuntimeError("existing RTSP connections detected; refusing duplicate camera ownership")
        # Remove only these exact known prior-session slots while holding all
        # cooperative camera-owner locks; sequence zero is never live.
        for camera in CAMERAS:
            Path(f"/dev/shm/v11_ui_preview_{camera.lower().replace('-', '')}_v1.bin").unlink(missing_ok=True)
            Path(f"/dev/shm/v11_tracked_objects_cam{camera[-2:]}_v1.json").unlink(missing_ok=True)
        Path("/dev/shm/v11_ui_preview_fullscreen_v1.bin").unlink(missing_ok=True)
        Path("/dev/shm/v11_ui_fullscreen_request_v1").unlink(missing_ok=True)
        log_path = output / "native.log"
        with log_path.open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            (output / "native-pid.json").write_text(json.dumps({"pid": process.pid}) + "\n")
            received: list[int] = []

            def stop(signum, _frame):
                received.append(signum)
                if process.poll() is None:
                    process.send_signal(signum)

            previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
            try:
                progress = wait_for_preview_progress(timeout_sec=min(90.0, duration))
                (output / "preview-ready.json").write_text(json.dumps(progress, indent=2) + "\n")
                return_code = process.wait()
                (output / "exit.json").write_text(json.dumps({
                    "return_code": return_code, "signal_received": received,
                    "native_exit_monotonic_ns": time.monotonic_ns(),
                }, indent=2) + "\n")
                return return_code
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                camera_file.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=86400)
    parser.add_argument("--lowconf-verifier", action="store_true",
                        help="Enable the isolated, bounded PeopleNet context verifier; low-confidence proposals remain outside NvDCF.")
    parser.add_argument("--yolo-context-verifier", action="store_true",
                        help="Enable frozen-YOLO context confirmation for low-confidence fallback and new active-track canonical publication.")
    parser.add_argument("--active-track-verifier", action="store_true",
                        help="Enable positive-only frozen-640 context validation for new canonical track publication; does not change the 896 PGIE/NvDCF input.")
    args = parser.parse_args()
    try:
        return run(args.candidate, args.output, args.duration,
                   args.lowconf_verifier, args.yolo_context_verifier, args.active_track_verifier)
    except Exception as exc:
        print(f"SIX_CAMERA_V13_CANDIDATE_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
