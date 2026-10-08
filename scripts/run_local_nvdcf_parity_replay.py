"""Run local6 NvDCF on the frozen CAM-01/CAM-04 replay, no identity worker."""
from __future__ import annotations

import hashlib
import json
import os
import argparse
from pathlib import Path
import subprocess
import time

from scripts.dev_room_mv3dt import run_room_pair as room
from scripts.run_room_candidate import stage_profile, section_value

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".runtime/sixcam-local6-20261004-a/replay-local6-exact"
PAIR = ROOT / ".runtime/amc-transformer-personpresent-MtbZPbuh/amc_inputs/videos"
INPUTS = {
    "CAM-01": (PAIR / "cam_00.mp4", "1e2818e022b58f2f6608b9af24682f9dbc0d628ee80c7754f0c95a7c943f70f1"),
    "CAM-04": (PAIR / "cam_01.mp4", "2b73ecfd4c9361c02442b67966fe78e17aa2352e7b6f39eff0f0258de2ff425f"),
}
BASELINE_AUDIT = ROOT / ".runtime/freeze/F5-v13-live/person-present/run/logs/probe/frame_path_audit.jsonl"
BASELINE_BINARY = ROOT / ".runtime/freeze/F4-yolo26m-live/candidate-audit-build/bin/deepstream-test5-pn263-global-id-proto"
BASELINE_BINARY_SHA = "b42701d714610a25a1785e5daff0f39e01b500279a7564050fa95e6cba5237bf"
LOCAL_CONFIG = ROOT / "config/deepstream/config_tracker_NvDCF_v13_local6.yml"
V13_CONFIG = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml"
POSE_MODELS = ROOT / ".runtime/freeze/F5-v13-live/pose-cache/models"
POSE_ENGINE = POSE_MODELS / "BodyPose3DNet/bodypose3dnet_accuracy.onnx_b1_gpu0_fp16.engine"
POSE_ENGINE_SHA256 = "d4c7570dff27016aa0c488323a30ef8f6c2f7ed98b92cc1e4b8a2c11a4120b8b"
IMAGE = json.loads((ROOT / "config/deepstream-platform.json").read_text())["image"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_diagnostic_profile(source: Path, state_estimator: int | None, shadow_output: bool) -> bytes:
    text = source.read_text()
    if state_estimator is not None:
        import re
        pattern = r"(?m)^(  stateEstimatorType: )\d+(\s*)$"
        text, count = re.subn(pattern, rf"\g<1>{state_estimator}\g<2>", text, count=1)
        if count != 1:
            raise RuntimeError("expected exactly one NvDCF stateEstimatorType")
    if shadow_output:
        import re
        pattern = r"(?m)^(  outputShadowTracks: )\d+(\s*)$"
        text, count = re.subn(pattern, rf"\g<1>1\g<2>", text, count=1)
        if count != 1:
            raise RuntimeError("expected exactly one outputShadowTracks setting")
    return text.encode()


def enable_shadow_dump(app_config: Path) -> None:
    text = app_config.read_text()
    marker = "[application]\n"
    if marker not in text:
        raise RuntimeError("staged DeepStream config has no [application] section")
    if "shadow-track-output-dir=" not in text:
        text = text.replace(marker,
                            marker + "shadow-track-output-dir=/workspace/experiments/logs/probe/shadow\n",
                            1)
    app_config.write_text(text)


def run(output: Path, tracker_profile: str, state_estimator: int | None = None,
        shadow_output: bool = False, diagnostic_binary: Path | None = None) -> dict:
    output = output.resolve()
    output.relative_to(ROOT / ".runtime")
    if output.exists():
        raise FileExistsError(f"refusing to reuse replay output: {output}")
    selected_binary = BASELINE_BINARY
    binary_sha = BASELINE_BINARY_SHA
    if diagnostic_binary is not None:
        selected_binary = diagnostic_binary.resolve()
        selected_binary.relative_to(ROOT / ".runtime")
        build_manifest = selected_binary.parent.parent / "build.json"
        if not build_manifest.is_file():
            raise RuntimeError("diagnostic binary lacks its isolated build manifest")
        build_data = json.loads(build_manifest.read_text())
        if build_data.get("binary") != str(selected_binary) or build_data.get("production_accepted") is not False:
            raise RuntimeError("diagnostic binary manifest does not pin an experimental candidate")
        binary_sha = sha256(selected_binary)
        if binary_sha != build_data.get("binary_sha256"):
            raise RuntimeError("diagnostic binary SHA does not match its build manifest")
    elif sha256(BASELINE_BINARY) != BASELINE_BINARY_SHA:
        raise RuntimeError("the visual-review DeepStream binary is not the pinned V13 binary")
    if not BASELINE_AUDIT.is_file():
        raise FileNotFoundError(BASELINE_AUDIT)
    for camera, (path, expected) in INPUTS.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"{camera} video path/hash does not match the approved replay")

    OUT = output
    OUT.mkdir(parents=True)
    stage = stage_profile(OUT, "replay", "production-reference")
    tracker_path = stage / "config_tracker.yml"
    tracker_config = V13_CONFIG if tracker_profile == "v13" else LOCAL_CONFIG
    if tracker_profile == "v13":
        if state_estimator is not None:
            raise RuntimeError("state-estimator override is forbidden for frozen V13")
        tracker_bytes = (tracker_config.read_bytes() if not shadow_output else
                         make_diagnostic_profile(V13_CONFIG, None, True))
    else:
        tracker_bytes = make_diagnostic_profile(LOCAL_CONFIG, state_estimator, shadow_output)
    tracker_path.write_bytes(tracker_bytes)
    tracker_path.chmod(0o600)
    deepstream_config = stage / "config_deepstream.txt"
    deepstream_config.write_text(section_value(deepstream_config.read_text(), "sink3", "enable", "0"))
    if shadow_output:
        enable_shadow_dump(deepstream_config)
        (stage / "logs/probe/shadow").mkdir(parents=True, exist_ok=True)
    deepstream_config.chmod(0o600)
    (OUT / "inputs.json").write_text(json.dumps({
        camera: {"path": str(path), "sha256": sha256(path), "frames": 2400,
                 "width": 1920, "height": 1080, "fps": 20}
        for camera, (path, _) in INPUTS.items()
    }, indent=2) + "\n")
    (OUT / "candidate.json").write_text(json.dumps({
        "reference_binary_sha256": binary_sha,
        "diagnostic_binary": str(selected_binary),
        "frozen_v13_config_sha256": "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834",
        "selected_tracker_profile": tracker_profile,
        "selected_tracker_config_sha256": hashlib.sha256(tracker_bytes).hexdigest(),
        "local_tracker_config_sha256": sha256(LOCAL_CONFIG),
        "state_estimator_type_override": state_estimator,
        "shadow_output_enabled": shadow_output,
        "yolo_engine_sha256": "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16",
        "yolo_parser_sha256": "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19",
        "pair_calibration_loaded_by_tracker": tracker_profile == "v13",
        "identity_enabled": False,
        "sink3_enabled": False,
        "mode": "recorded replay, two-source batch, no MV3DT topology",
    }, indent=2) + "\n")

    env_names = {"MV3DT_FRAME_AUDIT_LOG": "1", "MV3DT_UI_PREVIEW_DIR": ""}
    previous = {key: os.environ.get(key) for key in env_names}
    os.environ.update(env_names)
    container = f"local6-parity-{os.getpid()}"
    command = room.deepstream_command(stage, selected_binary, IMAGE, "replay", container)
    video_mount = f"{PAIR}:/workspace/inputs/videos:ro"
    found_video_mount = False
    for index in range(len(command) - 1):
        if command[index] == "-v" and command[index + 1].endswith(":/workspace/inputs/videos:ro"):
            command[index + 1] = video_mount
            found_video_mount = True
            break
    if not found_video_mount:
        raise RuntimeError("could not replace legacy replay mount with verified person-present inputs")
    if tracker_profile == "local6":
        filtered: list[str] = []
        index = 0
        while index < len(command):
            if command[index] == "-v" and index + 1 < len(command) and (
                "/workspace/inputs/camInfo:ro" in command[index + 1]
                or "/workspace/models:ro" in command[index + 1]
            ):
                index += 2
                continue
            filtered.append(command[index])
            index += 1
        command = filtered
    at = command.index(IMAGE)
    if shadow_output and diagnostic_binary is not None:
        command[at:at] = ["-e", "PN263_SHADOW_META_LOG=/workspace/experiments/logs/probe/tracker_shadow_meta.csv"]
        at += 2
    command[at:at] = ["--pull=never", "-v", f"{ROOT / '.runtime/models/yolo26m'}:/models:ro"]
    if tracker_profile == "v13":
        if not POSE_ENGINE.is_file() or sha256(POSE_ENGINE) != POSE_ENGINE_SHA256:
            raise RuntimeError("frozen V13 pose engine is missing or its hash changed")
        model_mount = None
        for index in range(len(command) - 1):
            if command[index] == "-v" and command[index + 1].endswith(":/workspace/models:ro"):
                model_mount = index + 1
                break
        if model_mount is None:
            raise RuntimeError("V13 command has no /workspace/models mount")
        command[model_mount] = f"{POSE_MODELS}:/workspace/models:ro"
    (OUT / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    start = time.monotonic_ns()
    try:
        result = subprocess.run(command, stdout=(OUT / "deepstream.log").open("w"),
                                stderr=subprocess.STDOUT, timeout=180, check=False)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    (OUT / "exit.json").write_text(json.dumps({
        "return_code": result.returncode,
        "elapsed_seconds": (time.monotonic_ns() - start) / 1e9,
    }, indent=2) + "\n")
    if result.returncode != 0:
        raise RuntimeError(f"local tracker replay exited {result.returncode}; see {OUT / 'deepstream.log'}")
    audit = stage / "logs/probe/frame_path_audit.jsonl"
    if not audit.is_file():
        raise RuntimeError("DeepStream completed but produced no frame-path audit")
    (OUT / "audit-path.txt").write_text(str(audit) + "\n")
    return {"output": str(OUT), "audit": str(audit), "return_code": result.returncode,
            "tracker_profile": tracker_profile}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=("v13", "local6"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-estimator-type", type=int, choices=(1, 2))
    parser.add_argument("--shadow-output", action="store_true")
    parser.add_argument("--diagnostic-binary", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.tracker, args.state_estimator_type,
                         args.shadow_output, args.diagnostic_binary), indent=2))
