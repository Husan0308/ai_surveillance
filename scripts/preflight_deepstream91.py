#!/usr/bin/env python3
"""Read-only platform gate. Never starts cameras or reports a camera pass.

The optional container probe uses only the locally downloaded, pinned image.
No host CUDA/TRT libraries, project venvs, camera secrets, or source are mounted.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def version(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"\d+(?:\.\d+)+", value):
        raise ValueError("invalid version")
    return tuple(int(part) for part in value.split("."))


def gpu_check(output: str, platform: dict, gpu_id: int) -> dict:
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != 5 or row[0].strip() != str(gpu_id):
            continue
        _, name, driver, memory, uuid = (s.strip() for s in row)
        return {
            "index": gpu_id, "name": name, "driver": driver,
            "memory_mib": int(memory), "uuid": uuid,
            "gpu_ok": name == platform["gpu_name"] and int(memory) >= platform["minimum_vram_mib"],
            "driver_ok": version(driver) >= version(platform["minimum_driver"]),
        }
    raise ValueError("configured GPU not found")


def run(argv: list[str], timeout: int = 30) -> dict:
    try:
        p = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)
        return {"code": p.returncode, "output": (p.stdout + p.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"code": -1, "output": type(exc).__name__}


def stack_matches(output: str, platform: dict) -> bool:
    # deepstream-app prints TRT's major.minor, so verify the package patch separately.
    ds = re.search(r"DeepStreamSDK\s+(\d+\.\d+\.\d+)", output)
    cuda = re.search(r"CUDA Runtime Version:\s*(\d+\.\d+)", output)
    trt = re.search(r"TRT_PACKAGE=(\S+)", output)
    return bool(ds and cuda and trt
                and ds[1] == platform["deepstream"]
                and cuda[1] == platform["cuda_runtime"]
                and trt[1].split("-", 1)[0] == platform["tensorrt"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", action="store_true", help="also verify the GPU-enabled container stack/plugins")
    parser.add_argument("--report", type=Path, help="save a credential-free JSON report")
    args = parser.parse_args()
    platform = json.loads((ROOT / "config/deepstream-platform.json").read_text())
    result = {"target": platform, "blockers": [], "camera_pass": "NOT_RUN"}
    blockers = result["blockers"]
    gpu_id = 0
    try:
        from services.shared.camera_config import load_settings

        import os
        config = Path(os.environ.get("CAMERA_CONFIG", "config/cameras.yaml"))
        settings = load_settings(config if config.is_absolute() else ROOT / config)
        gpu_id = settings.gpu_id
        ids = [camera.camera_id for camera in settings.cameras]
        result["cameras"] = {cid: "CONFIGURED_NOT_TESTED" for cid in ids}
        if len(ids) != 6 or set(ids) != set(platform["camera_ids"]):
            blockers.append("Exactly CAM-01 through CAM-06 must be enabled")
        if len({camera.uri for camera in settings.cameras}) != 6:
            blockers.append("Six distinct existing RTSP URLs are required")
    except Exception as exc:
        # Config parser exceptions can include credentials; do not print their text.
        blockers.append(f"Camera config unavailable/invalid ({type(exc).__name__})")

    smi = run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,uuid", "--format=csv,noheader,nounits"])
    try:
        if smi["code"]:
            raise ValueError("nvidia-smi failed")
        result["gpu"] = gpu_check(smi["output"], platform, gpu_id)
        if not result["gpu"]["gpu_ok"]:
            blockers.append("Configured GPU must be RTX 3060 with 12 GB VRAM")
        if not result["gpu"]["driver_ok"]:
            blockers.append(f"Active driver must be >= {platform['minimum_driver']}; upgrade then reboot")
    except (ValueError, TypeError):
        blockers.append("Cannot verify configured NVIDIA GPU/driver")

    result["host_deepstream"] = run(["deepstream-app", "--version-all"])
    result["host_nvcc"] = run(["nvcc", "--version"])
    result["docker"] = run(["docker", "info", "--format", "{{range $k, $v := .Runtimes}}{{$k}} {{end}}"])
    if result["docker"]["code"] or "nvidia" not in result["docker"]["output"].split():
        blockers.append("Docker with NVIDIA Container Toolkit must be available")
    result["image"] = run(["docker", "image", "inspect", platform["image"], "--format", "{{.Os}}/{{.Architecture}}"])
    if result["image"]["code"] or result["image"]["output"] != platform["platform"]:
        blockers.append("Pinned DeepStream 9.1 amd64 image must be downloaded")

    if args.container and not blockers:
        script = """set -eu
deepstream-app --version-all
printf 'TRT_PACKAGE='
dpkg-query -W -f='${Version}\\n' libnvinfer10
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
for plugin in rtspsrc rtph264depay rtph265depay h264parse h265parse nvurisrcbin nvv4l2decoder nvstreammux nvmultistreamtiler nvvideoconvert nvdsosd nveglglessink queue; do
    gst-inspect-1.0 "$plugin" >/dev/null
    printf 'PLUGIN_OK=%s\\n' "$plugin"
done
"""
        result["container"] = run([
            "docker", "run", "--rm", "--pull=never", "--network=none",
            "--gpus", f"device={gpu_id}",
            "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics",
            "--entrypoint", "bash", platform["image"], "-c", script,
        ], timeout=90)
        if result["container"]["code"] or not stack_matches(result["container"]["output"], platform):
            blockers.append("Container runtime/plugins or exact CUDA/TRT/DeepStream version check failed")
    elif args.container:
        result["container"] = {"status": "SKIPPED_HOST_BLOCKED"}

    result["status"] = ("BLOCKED" if blockers else
                        "READY_FOR_CAMERA_IMPLEMENTATION" if args.container else "CONTAINER_CHECK_REQUIRED")
    report = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
    print(report, end="")
    return 1 if blockers else 0 if args.container else 2


if __name__ == "__main__":
    raise SystemExit(main())
