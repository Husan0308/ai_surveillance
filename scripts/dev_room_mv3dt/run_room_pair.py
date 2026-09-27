#!/usr/bin/env python3
"""Run the production-scoped CAM-01/CAM-04 MV3DT profile.

Replay and live use one detector/tracker/projection/crop/identity/BEV data path.
Only the staged source configuration changes between synchronized files and
configured RTSP cameras.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

import yaml
from dotenv import dotenv_values

try:
    from .verify_validated_assets import sha256, verify
except ImportError:  # Direct script execution.
    from verify_validated_assets import sha256, verify


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config/mv3dt_dev_room"
ACCEPTED_EXP = Path(
    "/home/apsidal/nvidia/DeepStream/src/apps/reference_apps/"
    "deepstream-tracker-3d-multi-view/experiments/deepstream/"
    "dev-room-pn263-global-identity-20260921"
)
KAFKA_PYTHON = Path(
    "/home/apsidal/nvidia/DeepStream/src/apps/reference_apps/"
    "deepstream-tracker-3d-multi-view/mv3dt_venv/bin/python"
)
OSNET_PYTHON = Path("/home/apsidal/.local/share/Trash/files/ai_surveillance.2/venv/bin/python")


def load_profile() -> dict:
    with (PROFILE / "runtime.yaml").open() as handle:
        return yaml.safe_load(handle)


def replay_video_dir(profile: dict) -> Path:
    """Resolve the replay directory, with an explicit runtime-only test override."""
    configured = os.getenv("MV3DT_REPLAY_VIDEO_DIR") or profile["replay"]["video_dir"]
    video_dir = Path(configured)
    if not video_dir.is_absolute():
        video_dir = ROOT / video_dir
    video_dir = video_dir.resolve()
    missing = [name for name in ("cam_00.mp4", "cam_01.mp4") if not (video_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"replay video directory {video_dir} is missing: {', '.join(missing)}")
    return video_dir


def camera_uris() -> dict[str, str]:
    with (ROOT / "config/cameras.yaml").open() as handle:
        rows = yaml.safe_load(handle)["cameras"]
    env_file = {key: value for key, value in dotenv_values(ROOT / ".env").items() if value is not None}
    wanted = {"CAM-01", "CAM-04"}
    result: dict[str, str] = {}
    for row in rows:
        camera = str(row["id"])
        if camera not in wanted:
            continue
        username = os.getenv(f"{camera.replace('-', '_')}_RTSP_USERNAME") or env_file.get(f"{camera.replace('-', '_')}_RTSP_USERNAME") or os.getenv("SURVEILLANCE_RTSP_USERNAME") or env_file.get("SURVEILLANCE_RTSP_USERNAME") or str(row.get("username", ""))
        password = os.getenv(f"{camera.replace('-', '_')}_RTSP_PASSWORD") or env_file.get(f"{camera.replace('-', '_')}_RTSP_PASSWORD") or os.getenv("SURVEILLANCE_RTSP_PASSWORD") or env_file.get("SURVEILLANCE_RTSP_PASSWORD") or str(row.get("password", ""))
        uri = str(row["uri"])
        if username and password and "@" not in uri.split("://", 1)[-1].split("/", 1)[0]:
            uri = f"rtsp://{quote(username, safe='')}:{quote(password, safe='')}@{uri.split('://', 1)[1]}"
        if not username or not password:
            raise RuntimeError(f"{camera}: RTSP credentials were not resolved")
        result[camera] = uri
    return result


def run_checked(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("[room-pair]", " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def copy_profile(stage: Path) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    for path in PROFILE.iterdir():
        if path.name in {"runtime.yaml", "camInfo"}:
            continue
        if path.is_file():
            shutil.copy2(path, stage / path.name)
    (stage / "logs/probe").mkdir(parents=True, exist_ok=True)


def replay_config(stage: Path) -> None:
    """Pace synchronized files by the pipeline clock at their native 20 FPS."""
    path = stage / "config_deepstream.txt"
    text = path.read_text()
    sink0 = re.search(r"(?ms)^\[sink0\]\n(.*?)(?=^\[|\Z)", text)
    if sink0 is None or "sync=0" not in sink0.group(0):
        raise RuntimeError("replay sink0 sync setting was not found")
    paced = sink0.group(0).replace("sync=0", "sync=1", 1)
    path.write_text(text[:sink0.start()] + paced + text[sink0.end():])


def camera_latency_values() -> dict[str, int]:
    """Resolve the per-camera RTSP jitter latency for the Dev Room pair.

    DeepStream's source-config parser defaults an omitted source latency to
    100 ms. Keep the live MV3DT readers aligned with config/cameras.yaml rather
    than silently using that parser default.
    """
    with (ROOT / "config/cameras.yaml").open() as handle:
        raw = yaml.safe_load(handle) or {}
    fallback = int((raw.get("deepstream") or {}).get("latency_ms", 150))
    wanted = {"CAM-01", "CAM-04"}
    values = {}
    for row in raw.get("cameras") or []:
        camera = str(row.get("id", ""))
        if camera not in wanted:
            continue
        value = int(row["latency_ms"]) if row.get("latency_ms") is not None else fallback
        if value < 1:
            raise ValueError(f"{camera}: effective RTSP latency must be >= 1 ms")
        values[camera] = value
    if set(values) != wanted:
        raise RuntimeError("could not resolve CAM-01/CAM-04 RTSP latency values")
    return values


def apply_test_camera_latency_overrides(values: dict[str, int]) -> dict[str, int]:
    """Apply a numeric, experiment-only CAM-04 source-latency override."""
    raw = os.getenv("MV3DT_TEST_CAM04_RTSP_LATENCY_MS", "").strip()
    if not raw:
        return values
    if not re.fullmatch(r"\d{1,3}", raw) or not 1 <= int(raw) <= 150:
        raise ValueError("MV3DT_TEST_CAM04_RTSP_LATENCY_MS must be 1..150")
    result = dict(values)
    result["CAM-04"] = int(raw)
    return result


def apply_test_streammux_timeout(stage: Path) -> int | None:
    """Apply an opt-in test-only timeout to the staged live streammux config."""
    raw = os.getenv("MV3DT_TEST_STREAMMUX_BATCHED_PUSH_TIMEOUT", "").strip()
    if not raw:
        return None
    if not re.fullmatch(r"\d{1,5}", raw) or not 1 <= int(raw) <= 50_000:
        raise ValueError("MV3DT_TEST_STREAMMUX_BATCHED_PUSH_TIMEOUT must be 1..50000 us")
    path = stage / "config_deepstream.txt"
    text = path.read_text()
    match = re.search(r"(?ms)^\[streammux\]\n(.*?)(?=^\[|\Z)", text)
    if match is None:
        raise RuntimeError("missing [streammux] in staged config")
    block = match.group(0)
    key = "batched-push-timeout"
    if re.search(rf"(?m)^{re.escape(key)}=", block):
        updated, count = re.subn(
            rf"(?m)^{re.escape(key)}=.*$",
            f"{key}={int(raw)}",
            block,
            count=1,
        )
        if count != 1:
            raise RuntimeError("could not uniquely replace staged streammux timeout")
    else:
        updated = block.rstrip() + f"\n{key}={int(raw)}\n"
    path.write_text(text[:match.start()] + updated + text[match.end():])
    return int(raw)


def record_test_streammux_timeout_override(
    run_root: Path,
    timeout_us: int | None,
    experiment: str | None,
    tracker_lifecycle_diagnostics: bool,
) -> None:
    if timeout_us is None:
        return
    override_path = run_root / "experiment_overrides.json"
    if override_path.exists():
        overrides = json.loads(override_path.read_text())
    else:
        overrides = {
            "experiment": experiment,
            "production_profile_modified": False,
            "tracker_lifecycle_diagnostics": tracker_lifecycle_diagnostics,
        }
    overrides["streammux_batched_push_timeout_us_test"] = timeout_us
    overrides["production_profile_modified"] = False
    override_path.write_text(json.dumps(overrides, indent=2))


def live_config(stage: Path, uris: dict[str, str], latency_values: dict[str, int]) -> None:
    path = stage / "config_deepstream.txt"
    text = path.read_text()
    text = text.replace("type=3\nenable=1\ncudadec-memtype=0\ngpu-id=0\nnum-sources=1\nuri=file:///workspace/inputs/videos/cam_00.mp4",
                        f"type=4\nenable=1\ncudadec-memtype=0\ngpu-id=0\nnum-sources=1\nuri={uris['CAM-01']}\nrtsp-reconnect-interval-sec=5\ninit-rtsp-reconnect-interval-sec=5\nrtsp-reconnect-attempts=-1")
    text = text.replace("type=3\nenable=1\ncudadec-memtype=0\ngpu-id=0\nnum-sources=1\nuri=file:///workspace/inputs/videos/cam_01.mp4",
                        f"type=4\nenable=1\ncudadec-memtype=0\ngpu-id=0\nnum-sources=1\nuri={uris['CAM-04']}\nrtsp-reconnect-interval-sec=5\ninit-rtsp-reconnect-interval-sec=5\nrtsp-reconnect-attempts=-1")
    for source_id, camera_id in enumerate(("CAM-01", "CAM-04")):
        match = re.search(
            rf"(?ms)^\[source{source_id}\]\n(.*?)(?=^\[|\Z)", text
        )
        if match is None:
            raise RuntimeError(f"missing [source{source_id}] for {camera_id}")
        block = match.group(0)
        setting = f"latency={int(latency_values[camera_id])}"
        if re.search(r"(?m)^latency=", block):
            updated = re.sub(r"(?m)^latency=.*$", setting, block, count=1)
        else:
            updated = block.rstrip() + f"\n{setting}\n"
        drop_setting = "drop-on-latency=1"
        if re.search(r"(?m)^drop-on-latency=", updated):
            updated = re.sub(
                r"(?m)^drop-on-latency=.*$", drop_setting, updated, count=1
            )
        else:
            updated = updated.rstrip() + f"\n{drop_setting}\n"
        text = text[:match.start()] + updated + text[match.end():]
    text = text.replace("live-source=0", "live-source=1", 1)
    path.write_text(text)


RECALL_EXPERIMENTS = {
    "recall-003": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
    },
    "recall-004": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.50,
    },
    "recall-005": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.50,
        "early_termination_age": 2,
    },
    "recall-006": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.50,
        "early_termination_age": 2,
        "probation_age": 3,
    },
    "recall-007": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.30,
        "early_termination_age": 2,
        "probation_age": 3,
    },
    "recall-008": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.50,
        "early_termination_age": 2,
        "probation_age": 2,
    },
    "recall-009": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.50,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
    },
    "recall-010": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.50,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.045,
    },
    "recall-011": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.50,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.035,
    },
    "recall-012": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.45,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
    },
    "recall-013": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.46,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
    },
    "recall-014": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.46,
        "early_termination_age": 2,
        "probation_age": 3,
        "min_tracker_confidence": 0.03,
    },
    "recall-015": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.25,
        "min_iou_diff_new_target": 0.46,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
    },
    "recall-016": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.15,
        "min_iou_diff_new_target": 0.46,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
    },
    "recall-017": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.46,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.02,
    },
    "recall-018": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.46,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
        "min_detector_confidence": 0.04,
    },
    "recall-019": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.46,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
        "min_detector_confidence": 0.036,
    },
    "recall-020": {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.47,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
    },
}


def apply_experiment_overrides(
    stage: Path,
    run_root: Path,
    experiment: str | None,
    tracker_lifecycle_diagnostics: bool = False,
) -> None:
    """Apply experiment-only config overlays to the staged runtime copy.

    Production profile files under config/mv3dt_dev_room are never modified.
    """
    if not experiment:
        if tracker_lifecycle_diagnostics:
            raise ValueError("tracker lifecycle diagnostics require an experiment-only staged run")
        return
    if experiment not in RECALL_EXPERIMENTS:
        raise ValueError(f"unsupported experiment: {experiment}")
    spec = RECALL_EXPERIMENTS[experiment]

    pgie_path = stage / "config_pgie.txt"
    pgie = pgie_path.read_text()
    class_header = "[class-attrs-0]"
    setting = f"pre-cluster-threshold={spec['pre_cluster_threshold']}"
    if class_header in pgie:
        block_match = re.search(r"(?ms)^\[class-attrs-0\]\n(.*?)(?=^\[|\Z)", pgie)
        if block_match is None:
            raise RuntimeError("could not parse class-attrs-0 block")
        block = block_match.group(0)
        if re.search(r"(?m)^pre-cluster-threshold=", block):
            updated = re.sub(r"(?m)^pre-cluster-threshold=.*$", setting, block, count=1)
        else:
            updated = block.rstrip() + "\n" + setting + "\n"
        pgie = pgie[:block_match.start()] + updated + pgie[block_match.end():]
    else:
        pgie = pgie.rstrip() + f"\n\n{class_header}\n{setting}\n"
    pgie_path.write_text(pgie)

    tracker_path = stage / "config_tracker.yml"
    tracker = tracker_path.read_text()
    if "min_detector_confidence" in spec:
        base_match = re.search(r"(?ms)^BaseConfig:\n(.*?)(?=^[A-Za-z][^\n]*:\n|\Z)", tracker)
        if base_match is None:
            raise RuntimeError("could not parse BaseConfig block for detector confidence override")
        base = base_match.group(0)
        base, nb = re.subn(
            r"(?m)^(\s*minDetectorConfidence:)\s*[^\n]+$",
            rf"\1 {spec['min_detector_confidence']}",
            base,
            count=1,
        )
        if nb != 1:
            raise RuntimeError("recall experiment minDetectorConfidence was not uniquely resolved")
        tracker = tracker[:base_match.start()] + base + tracker[base_match.end():]

    assoc_match = re.search(r"(?ms)^DataAssociator:\n(.*?)(?=^[A-Za-z][^\n]*:\n|\Z)", tracker)
    if assoc_match is None:
        raise RuntimeError("could not parse DataAssociator block")
    assoc = assoc_match.group(0)
    assoc, n1 = re.subn(
        r"(?m)^(\s*tentativeDetectorConfidence:)\s*[^\n]+$",
        rf"\1 {spec['tentative_detector_confidence']}",
        assoc,
        count=1,
    )
    assoc, n2 = re.subn(
        r"(?m)^(\s*minMatchingScore4Overall:)\s*[^\n]+$",
        rf"\1 {spec['data_associator_min_matching_score']}",
        assoc,
        count=1,
    )
    if n1 != 1 or n2 != 1:
        raise RuntimeError("recall experiment tracker keys were not uniquely resolved")
    tracker = tracker[:assoc_match.start()] + assoc + tracker[assoc_match.end():]
    if "min_iou_diff_new_target" in spec:
        target_match = re.search(r"(?ms)^TargetManagement:\n(.*?)(?=^[A-Za-z][^\n]*:\n|\Z)", tracker)
        if target_match is None:
            raise RuntimeError("could not parse TargetManagement block")
        target = target_match.group(0)
        target, n3 = re.subn(
            r"(?m)^(\s*minIouDiff4NewTarget:)\s*[^\n]+$",
            rf"\1 {spec['min_iou_diff_new_target']}",
            target,
            count=1,
        )
        if n3 != 1:
            raise RuntimeError("recall experiment minIouDiff4NewTarget was not uniquely resolved")
        tracker = tracker[:target_match.start()] + target + tracker[target_match.end():]

    if "min_tracker_confidence" in spec:
        target_match = re.search(r"(?ms)^TargetManagement:\n(.*?)(?=^[A-Za-z][^\n]*:\n|\Z)", tracker)
        if target_match is None:
            raise RuntimeError("recall experiment minTrackerConfidence block was not found")
        target = target_match.group(0)
        target, n3 = re.subn(
            r"(?m)^(\s*minTrackerConfidence:)\s*[^\n]+$",
            rf"\1 {spec['min_tracker_confidence']}",
            target,
            count=1,
        )
        if n3 != 1:
            raise RuntimeError("recall experiment minTrackerConfidence was not uniquely resolved")
        tracker = tracker[:target_match.start()] + target + tracker[target_match.end():]

    if "early_termination_age" in spec:
        target_match = re.search(r"(?ms)^TargetManagement:\n(.*?)(?=^[A-Za-z][^\n]*:\n|\Z)", tracker)
        if target_match is None:
            raise RuntimeError("could not parse TargetManagement block for early termination override")
        target = target_match.group(0)
        target, n4 = re.subn(
            r"(?m)^(\s*earlyTerminationAge:)\s*[^\n]+$",
            rf"\1 {spec['early_termination_age']}",
            target,
            count=1,
        )
        if n4 != 1:
            raise RuntimeError("recall experiment earlyTerminationAge was not uniquely resolved")
        tracker = tracker[:target_match.start()] + target + tracker[target_match.end():]

    if "probation_age" in spec:
        target_match = re.search(r"(?ms)^TargetManagement:\n(.*?)(?=^[A-Za-z][^\n]*:\n|\Z)", tracker)
        if target_match is None:
            raise RuntimeError("could not parse TargetManagement block for probation override")
        target = target_match.group(0)
        target, n5 = re.subn(
            r"(?m)^(\s*probationAge:)\s*[^\n]+$",
            rf"\1 {spec['probation_age']}",
            target,
            count=1,
        )
        if n5 != 1:
            raise RuntimeError("recall experiment probationAge was not uniquely resolved")
        tracker = tracker[:target_match.start()] + target + tracker[target_match.end():]

    if tracker_lifecycle_diagnostics:
        target_match = re.search(r"(?ms)^TargetManagement:\n(.*?)(?=^[A-Za-z][^\n]*:\n|\Z)", tracker)
        if target_match is None:
            raise RuntimeError("could not parse TargetManagement block for lifecycle diagnostics")
        target = target_match.group(0)
        if re.search(r"(?m)^\s*outputShadowTracks:", target):
            target, n1 = re.subn(
                r"(?m)^(\s*outputShadowTracks:)\s*[^\n]+$",
                r"\1 1",
                target,
                count=1,
            )
        else:
            target = target.rstrip() + "\n  outputShadowTracks: 1\n"
            n1 = 1
        if n1 != 1:
            raise RuntimeError("outputShadowTracks was not uniquely resolved")
        tracker = tracker[:target_match.start()] + target + tracker[target_match.end():]

    tracker_path.write_text(tracker)

    (run_root / "experiment_overrides.json").write_text(json.dumps({
        "experiment": experiment,
        "production_profile_modified": False,
        "tracker_lifecycle_diagnostics": tracker_lifecycle_diagnostics,
        **spec,
    }, indent=2))


def resource_summary(run_root: Path) -> None:
    gpu_path = run_root / "logs/gpu_metrics.csv"
    gpu_values: list[float] = []
    gpu_mem: list[float] = []
    gpu_temp: list[float] = []
    try:
        with gpu_path.open() as handle:
            for row in csv.reader(handle):
                if not row or row[0].lower().startswith("timestamp"):
                    continue
                numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", ",".join(row[1:]))
                if len(numbers) >= 4:
                    gpu_values.append(float(numbers[0]))
                    gpu_mem.append(float(numbers[1]))
                    gpu_temp.append(float(numbers[3]))
    except OSError:
        pass
    vmstat = run_root / "logs/host_vmstat.log"
    cpu_busy: list[float] = []
    memory_free: list[float] = []
    try:
        total_memory_mib = 0.0
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total_memory_mib = float(line.split()[1]) / 1024.0
                break
        lines = vmstat.read_text().splitlines()
        for line in lines:
            fields = line.split()
            # vmstat -SM columns end with us sy id wa st gu.
            if len(fields) >= 18 and fields[0].isdigit():
                cpu_busy.append(100.0 - float(fields[14]))
                if total_memory_mib:
                    memory_free.append(float(fields[3]))
    except OSError:
        pass
    result = {
        "gpu_utilization_percent": {"mean": statistics.fmean(gpu_values) if gpu_values else None,
                                    "p95": statistics.quantiles(gpu_values, n=20)[18] if len(gpu_values) >= 2 else (gpu_values[0] if gpu_values else None),
                                    "peak": max(gpu_values) if gpu_values else None},
        "gpu_memory_mib": {"mean": statistics.fmean(gpu_mem) if gpu_mem else None,
                           "p95": statistics.quantiles(gpu_mem, n=20)[18] if len(gpu_mem) >= 2 else (gpu_mem[0] if gpu_mem else None),
                           "peak": max(gpu_mem) if gpu_mem else None},
        "gpu_temperature_c": {"peak": max(gpu_temp) if gpu_temp else None},
        "host_cpu_busy_percent": {"mean": statistics.fmean(cpu_busy) if cpu_busy else None,
                                   "peak": max(cpu_busy) if cpu_busy else None},
        "host_memory_free_mib": {"mean": statistics.fmean(memory_free) if memory_free else None,
                                  "minimum": min(memory_free) if memory_free else None,
                                  "total_mib": total_memory_mib if 'total_memory_mib' in locals() else None},
    }
    (run_root / "resource_metrics.json").write_text(json.dumps(result, indent=2))


def max_kafka_person_frame(path: Path) -> int:
    maximum = -1
    try:
        with path.open() as handle:
            for line in handle:
                try:
                    frame = json.loads(line).get("frame")
                except ValueError:
                    continue
                if frame and frame.get("sensorId") in {"CAM-01", "CAM-04"} and frame.get("objects"):
                    maximum = max(maximum, int(frame.get("id", -1)))
    except OSError:
        pass
    return maximum


def max_identity_frame(path: Path) -> int:
    maximum = -1
    try:
        with path.open() as handle:
            for line in handle:
                try:
                    maximum = max(maximum, int(json.loads(line).get("frame", -1)))
                except (ValueError, TypeError):
                    continue
    except OSError:
        pass
    return maximum


def wait_for_offline_sidecar_drain(kafka_log: Path, identity_dir: Path, sidecar: subprocess.Popen) -> None:
    target = max_kafka_person_frame(kafka_log)
    deadline = time.monotonic() + 240.0
    while time.monotonic() < deadline and sidecar.poll() is None:
        if target < 0 or max_identity_frame(identity_dir / "global_identity.jsonl") >= target:
            return
        time.sleep(0.5)
    if sidecar.poll() is not None:
        raise RuntimeError("offline identity sidecar exited before Kafka drain completed")
    raise TimeoutError(f"offline identity sidecar did not drain through frame {target}")


def start_monitors(run_root: Path) -> list[subprocess.Popen]:
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    commands = [
        (["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv", "-l", "1"], logs / "gpu_metrics.csv"),
        (["vmstat", "-SM", "1"], logs / "host_vmstat.log"),
        (["mosquitto_sub", "-h", "localhost", "-p", "1883", "-t", "/trck/#", "-v"], logs / "mqtt_peer.raw"),
    ]
    processes = []
    for command, output in commands:
        handle = output.open("w")
        processes.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
    return processes


def stop_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def audit_camera_mapping(stage: Path) -> str:
    """Derive source-id aliases from the staged DeepStream/msgconv config.

    The source-id aliases are authoritative for the runtime audit. Pair aliases
    are included only as an observed pad/source cross-check; the report marks
    any pair not seen at runtime as unmapped rather than guessing.
    """
    source_text = (stage / "config_deepstream.txt").read_text()
    msgconv_text = (stage / "config_msgconv.txt").read_text()
    source_ids = sorted({int(match) for match in re.findall(r"^\[source(\d+)\]$", source_text, re.MULTILINE)})
    sensors = {}
    current = None
    for line in msgconv_text.splitlines():
        section = re.fullmatch(r"\[sensor(\d+)\]", line.strip())
        if section:
            current = int(section.group(1))
            continue
        if current is not None:
            match = re.fullmatch(r"id=(.+)", line.strip())
            if match:
                sensors[current] = match.group(1).strip()
    aliases = []
    for source_id in source_ids:
        camera = sensors.get(source_id)
        if camera:
            aliases.extend((f"source:{source_id}={camera}", f"{source_id}/{source_id}={camera}"))
    if not aliases:
        raise RuntimeError("could not derive camera mapping from staged DeepStream/msgconv config")
    return ";".join(aliases)


def deepstream_command(stage: Path, binary: Path, image: str, source_mode: str, container_name: str) -> list[str]:
    live = source_mode == "live"
    command = [
        "docker", "run", "--name", container_name, "--rm",
        "--network", "host", "--runtime=nvidia",
        "-e", "PN263_BBOX_MODE=active",
        "-e", "PN263_BBOX_LOG_DIR=/workspace/experiments/logs/probe",
        "-e", "NVDS_ENABLE_LATENCY_MEASUREMENT=1",
        "-e", "NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1",
        "-e", f"MV3DT_TEST_DECODER_LOW_LATENCY={os.getenv('MV3DT_TEST_DECODER_LOW_LATENCY', '0')}",
        "-v", f"{binary}:/workspace/proto-app:ro",
        "-v", f"{Path('/home/apsidal/nvidia/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view/models')}:/workspace/models:ro",
        "-v", f"{PROFILE / 'camInfo'}:/workspace/inputs/camInfo:ro",
        "-v", f"{stage}:/workspace/experiments:rw",
        "-w", "/workspace/experiments",
    ]
    queue_override = os.getenv(
        "MV3DT_TEST_CAM04_ANALYTICS_QUEUE_BUFFERS", ""
    ).strip()
    if queue_override:
        if not re.fullmatch(r"\d{1,2}", queue_override) or not 4 <= int(queue_override) <= 64:
            raise ValueError("MV3DT_TEST_CAM04_ANALYTICS_QUEUE_BUFFERS must be 4..64")
        command += [
            "-e",
            f"MV3DT_TEST_CAM04_ANALYTICS_QUEUE_BUFFERS={queue_override}",
        ]
    extra_surfaces_override = os.getenv(
        "MV3DT_TEST_CAM04_DECODER_EXTRA_SURFACES", ""
    ).strip()
    if extra_surfaces_override:
        if not re.fullmatch(r"\d{1,2}", extra_surfaces_override) or int(extra_surfaces_override) > 55:
            raise ValueError("MV3DT_TEST_CAM04_DECODER_EXTRA_SURFACES must be 0..55")
        command += [
            "-e",
            f"MV3DT_TEST_CAM04_DECODER_EXTRA_SURFACES={extra_surfaces_override}",
        ]
    if os.getenv("MV3DT_DUMP_DECODER_AU_SAMPLE", "0") == "1":
        command += [
            "-e",
            "MV3DT_DECODER_AU_DUMP_DIR=/workspace/experiments/logs/probe/bitstream-sample",
        ]
    command += [
        "-e", "MV3DT_CROP_SOCKET=/workspace/experiments/logs/crops.sock",
        "-e", "MV3DT_CROP_LOG_DIR=/workspace/experiments/logs/probe",
        "-e", f"MV3DT_CROP_CAMERA_MAP={audit_camera_mapping(stage)}",
    ]
    if os.getenv("MV3DT_FRAME_AUDIT_LOG"):
        command += [
            "-e", "MV3DT_FRAME_AUDIT_LOG=/workspace/experiments/logs/probe/frame_path_audit.jsonl",
        ]
    if os.getenv("MV3DT_FRAME_AUDIT_LOG") or os.getenv("MV3DT_SOURCE_HEALTH_DIR") or os.getenv("MV3DT_UI_PREVIEW_DIR", "/dev/shm"):
        command += [
            "-e", f"MV3DT_AUDIT_CAMERA_MAP={audit_camera_mapping(stage)}",
        ]
    if os.getenv("MV3DT_CAPTURE_NATIVE_LATENCY") == "1":
        command += [
            "-e", "MV3DT_NATIVE_LATENCY_LOG=/workspace/experiments/logs/probe/native_decoder_latency.jsonl",
        ]
    if os.getenv("MV3DT_SOURCE_HEALTH_DIR"):
        command += [
            "-e", "MV3DT_SOURCE_HEALTH_DIR=/workspace/experiments/logs/probe",
        ]
    ui_preview_dir = os.getenv("MV3DT_UI_PREVIEW_DIR", "/dev/shm")
    if ui_preview_dir:
        command += [
            "--ipc=host",
            "-e", "MV3DT_UI_PREVIEW_DIR=/dev/shm",
            "-e", "MV3DT_UI_PREVIEW_DIAGNOSTICS=/workspace/experiments/logs/probe/preview_decoder_timing.jsonl",
        ]
    if not live:
        video_dir = replay_video_dir(load_profile())
        command += ["-v", f"{video_dir}:/workspace/inputs/videos:ro"]
    command += [image, "/workspace/proto-app", "-c", "/workspace/experiments/config_deepstream.txt"]
    return command


def stop_deepstream_container(container_name: str) -> None:
    subprocess.run(
        ["docker", "stop", "--time", "10", container_name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=20, check=False,
    )
    subprocess.run(
        ["docker", "kill", container_name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=10, check=False,
    )



def make_crop_socket_alias(stage: Path, process_id: int | None = None) -> tuple[Path, Path]:
    """Return a host-visible crop socket path safely below AF_UNIX limits."""
    alias = Path("/tmp") / f"mv3dt-crops-{process_id or os.getpid()}"
    alias.unlink(missing_ok=True)
    alias.symlink_to((stage / "logs").resolve(), target_is_directory=True)
    socket_path = alias / "crops.sock"
    if len(os.fsencode(str(socket_path))) >= 108:
        alias.unlink(missing_ok=True)
        raise RuntimeError("crop socket alias exceeds AF_UNIX path limit")
    return alias, socket_path

def run(
    mode: str,
    duration: float | None,
    skip_render: bool,
    experiment: str | None = None,
    tracker_lifecycle_diagnostics: bool = False,
) -> Path:
    mode = "replay" if mode == "offline" else mode
    if mode not in {"replay", "live"}:
        raise ValueError(f"unsupported source mode: {mode}")
    profile = load_profile()
    accepted_binary = Path(profile["runtime"]["binary"]).resolve()
    binary = Path(os.getenv("MV3DT_BINARY", str(accepted_binary))).resolve()
    audit_mode = bool(os.getenv("MV3DT_FRAME_AUDIT_LOG"))
    health_mode = bool(os.getenv("MV3DT_SOURCE_HEALTH_DIR"))
    diagnostic_mode = audit_mode or health_mode
    # Production assets remain frozen. Diagnostic runs may use exactly one
    # source override, but only when BOTH the modified source and diagnostic
    # binary are independently SHA256-pinned by the runner.
    diagnostic_binary = binary != accepted_binary
    validated_asset_overrides = None
    if diagnostic_binary:
        expected_diagnostic_hash = os.getenv("MV3DT_DIAGNOSTIC_BINARY_SHA256")
        expected_diagnostic_source_hash = os.getenv("MV3DT_DIAGNOSTIC_SOURCE_SHA256")
        diagnostic_source = ROOT / "services/mv3dt_room/native/deepstream_test5_app_main.c"
        if not diagnostic_mode or not binary.is_relative_to(ROOT / ".runtime"):
            raise SystemExit("custom binaries are restricted to .runtime diagnostics")
        if not binary.is_file() or not expected_diagnostic_hash or sha256(binary) != expected_diagnostic_hash:
            raise SystemExit("diagnostic binary is missing or does not match MV3DT_DIAGNOSTIC_BINARY_SHA256")
        if (
            not expected_diagnostic_source_hash
            or sha256(diagnostic_source) != expected_diagnostic_source_hash
        ):
            raise SystemExit(
                "diagnostic source is missing or does not match MV3DT_DIAGNOSTIC_SOURCE_SHA256"
            )
        validated_asset_overrides = {
            "native/deepstream_test5_app_main.c": expected_diagnostic_source_hash
        }

    check = verify(
        ROOT,
        accepted_binary,
        validated_asset_overrides=validated_asset_overrides,
    )
    if not check["ok"]:
        raise SystemExit("validated room-pair asset check failed")
    run_root = ROOT / profile["runtime"]["runtime_root"] / f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    stage = run_root / "run"
    identity_dir = run_root / "identity-live"
    logs = run_root / "logs"
    copy_profile(stage)
    apply_experiment_overrides(stage, run_root, experiment, tracker_lifecycle_diagnostics)
    session_id = run_root.name
    (run_root / "source_mode.json").write_text(json.dumps({
        "source_mode": mode,
        "session_id": session_id,
        "replay_video_dir": str(replay_video_dir(profile)) if mode == "replay" else None,
        "started_at_epoch": time.time(),
        "binary_path": str(binary),
        "binary_sha256": sha256(binary),
        "accepted_binary_path": str(accepted_binary),
        "accepted_binary_sha256": sha256(accepted_binary),
        "diagnostic_binary": diagnostic_binary,
        "diagnostic_source_sha256": (
            os.getenv("MV3DT_DIAGNOSTIC_SOURCE_SHA256") if diagnostic_binary else None
        ),
        "decoder_low_latency_mode_test": os.getenv("MV3DT_TEST_DECODER_LOW_LATENCY", "0") == "1",
    }, indent=2))
    (run_root / "running").touch()
    container_name = f"ai-surveillance-dev-room-{os.getpid()}"
    if mode == "live":
        (run_root / "live").touch()
        uris = camera_uris()
        if set(uris) != {"CAM-01", "CAM-04"}:
            raise RuntimeError("live scope did not resolve exactly CAM-01 and CAM-04")
        live_config(stage, uris, apply_test_camera_latency_overrides(camera_latency_values()))
        timeout_override = apply_test_streammux_timeout(stage)
        record_test_streammux_timeout_override(
            run_root, timeout_override, experiment, tracker_lifecycle_diagnostics
        )
    else:
        (run_root / "replay").touch()
        replay_config(stage)
    monitor_processes = start_monitors(run_root)
    kafka_log = logs / "kafka_current.jsonl"
    capture_python = Path(os.getenv("MV3DT_KAFKA_PYTHON", str(KAFKA_PYTHON)))
    capture_env = dict(os.environ, PYTHONPATH=str(ROOT / "services/mv3dt_room"))
    capture = subprocess.Popen(
        [str(capture_python), str(ROOT / "services/mv3dt_room/capture_kafka.py")],
        stdout=kafka_log.open("w"), stderr=(logs / "kafka_capture.err").open("w"),
        env=capture_env, start_new_session=True,
    )
    sidecar = None
    deepstream = None
    done_file = logs / "deepstream.done"
    # Linux AF_UNIX paths are limited to roughly 108 bytes. Runtime artifact
    # roots are intentionally descriptive and can exceed that limit, while the
    # container binds the same logs directory at a short path. Give the host
    # identity worker an equally short alias to the exact same directory inode.
    crop_socket_alias, crop_socket_host_path = make_crop_socket_alias(stage)
    requested_duration = float(duration or profile.get(mode, {}).get("expected_duration_seconds", 120.0) or 120.0)
    sidecar_duration = requested_duration + 60.0
    try:
        sidecar_env = dict(
            os.environ,
            PYTHONPATH=str(ROOT),
            MV3DT_DEV_ROOM_SOURCE_MODE=mode,
            MV3DT_DEV_ROOM_SESSION_ID=session_id,
        )
        sidecar = subprocess.Popen(
            [str(OSNET_PYTHON), str(ROOT / "services/mv3dt_room/live_identity_worker.py"),
             "--kafka", str(kafka_log), "--output-dir", str(identity_dir),
             "--osnet-model", str(profile["runtime"]["osnet_model"]),
             "--duration", str(sidecar_duration), "--interval", "10",
             "--done-file", str(done_file), "--source-mode", mode,
             "--session-id", session_id,
             "--crop-socket", str(crop_socket_host_path)],
            stdout=(logs / "identity_sidecar.stdout").open("w"),
            stderr=(logs / "identity_sidecar.stderr").open("w"), env=sidecar_env,
            start_new_session=True,
        )
        deepstream = subprocess.Popen(
            deepstream_command(stage, binary, profile["runtime"]["deepstream_image"], mode, container_name),
            stdout=(logs / "deepstream.log").open("w"), stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if mode in {"live", "replay"}:
            run_started = time.monotonic()
            deadline = run_started + requested_duration
            while time.monotonic() < deadline and deepstream.poll() is None and sidecar.poll() is None:
                time.sleep(0.25)
            if sidecar.poll() is not None and deepstream.poll() is None:
                raise RuntimeError(f"identity worker exited early with {sidecar.returncode}")
            if deepstream.poll() is None:
                stop_deepstream_container(container_name)
                deepstream.wait(timeout=20)
            else:
                return_code = deepstream.wait()
                elapsed = time.monotonic() - run_started
                if return_code != 0:
                    raise RuntimeError(f"DeepStream exited with {return_code}; see {logs / 'deepstream.log'}")
                if elapsed < requested_duration:
                    raise RuntimeError(
                        f"DeepStream ended before the requested {mode} duration "
                        f"({elapsed:.1f}s < {requested_duration:.1f}s); inspect {logs / 'deepstream.log'}"
                    )
        else:
            return_code = deepstream.wait()
            if return_code != 0:
                raise RuntimeError(f"DeepStream exited with {return_code}; see {logs / 'deepstream.log'}")
        # Stop Kafka capture only after DeepStream closes, then signal the one
        # shared identity worker to drain the flushed file and finish.
        stop_processes([capture])
        done_file.touch()
        if sidecar is not None:
            sidecar.wait(timeout=90)
            if sidecar.returncode != 0:
                raise RuntimeError(f"identity worker exited with {sidecar.returncode}; see {logs / 'identity_sidecar.stderr'}")
    finally:
        if deepstream is not None and deepstream.poll() is None:
            stop_deepstream_container(container_name)
            try:
                deepstream.wait(timeout=20)
            except subprocess.TimeoutExpired:
                deepstream.kill()
        stop_processes([capture])
        if sidecar is not None and sidecar.poll() is None:
            sidecar.terminate()
            sidecar.wait(timeout=15)
        stop_processes(monitor_processes)
        (run_root / "done").touch()
        (run_root / "running").unlink(missing_ok=True)
        resource_summary(run_root)
        crop_socket_alias.unlink(missing_ok=True)

    if mode == "replay" and not skip_render:
        video_dir = replay_video_dir(profile)
        dataset = ROOT / ".runtime/mv3dt/calibration/dev-room-cam01-cam04-vggt-v1"
        output = run_root / "visual/dev-room-cam01-cam04-production.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        render_env = dict(os.environ, PYTHONPATH=str(ROOT))
        run_checked([str(OSNET_PYTHON), str(ROOT / "scripts/dev_room_mv3dt/render_visual_proof.py"),
                     "--identity-jsonl", str(identity_dir / "global_identity.jsonl"),
                     "--video-dir", str(video_dir), "--dataset-dir", str(dataset),
                     "--output", str(output)], env=render_env)
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("replay", "live", "offline"),
        default=os.getenv("MV3DT_DEV_ROOM_SOURCE_MODE", "live"),
        help="Dev Room source mode; 'offline' remains a compatibility alias for replay",
    )
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument(
        "--experiment", choices=tuple(RECALL_EXPERIMENTS), default=None,
        help="Apply an experiment-only overlay to the staged runtime config; production files stay unchanged",
    )
    parser.add_argument(
        "--tracker-lifecycle-diagnostics", action="store_true",
        help="Export terminated NvDCF tracks for diagnosis without changing tracker decision thresholds",
    )
    args = parser.parse_args()
    print(run(args.mode, args.duration, args.skip_render, args.experiment,
              args.tracker_lifecycle_diagnostics))


if __name__ == "__main__":
    main()
