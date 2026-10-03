#!/usr/bin/env python3
"""Run the existing camera-only worker with the pinned DeepStream 9.1 plugins.

Host GI/NVIDIA plugin discovery must not select the legacy 7.1 install. No
packages are installed; F1 Python dependencies are mounted read-only and Ubuntu
GI/GStreamer come from the pinned container. No detector/tracker is loaded.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from services.shared.runtime_python import preflight_python

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = "ai-surveillance-preview-foundation"


def command(root: Path, python: Path, output: Path, cameras: str, duration: float,
            reconnect_camera: str | None = None, reconnect_at: float = 20.0) -> list[str]:
    platform = json.loads((root / "config/deepstream-platform.json").read_text())
    locks = root / ".runtime/camera-owner-locks"
    env_file = Path(os.environ.get("V11_ENV_FILE", str(root / ".env"))).resolve()
    if not env_file.is_file():
        raise ValueError("authorized camera environment file is missing")
    cmd = ["docker", "run", "--rm", "--pull=never", "--init", "--name", CONTAINER,
           "--network=host", "--gpus", "all", "--user", f"{os.getuid()}:{os.getgid()}",
           "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
           "-e", "PYTHONNOUSERSITE=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
           "-e", "XDG_CACHE_HOME=/tmp/preview-cache", "-e", f"CAMERA_OWNER_LOCK_DIR={locks}",
           "-e", "V11_ENV_FILE=/run/camera.env",
           "--mount", f"type=bind,src={env_file},dst=/run/camera.env,readonly",
           "--mount", f"type=bind,src={root},dst={root},readonly",
           "--mount", f"type=bind,src={output},dst={output}",
           "--mount", f"type=bind,src={locks},dst={locks}",
           "--mount", "type=bind,src=/dev/shm,dst=/dev/shm", "-w", str(root),
           platform["image"], str(python), "-B", "-m",
           "services.camera_v11.preview_only_runtime", "--cameras", cameras,
           "--stats", str(output / "camera-health.json"), "--duration", str(duration)]
    if reconnect_camera:
        cmd.extend(["--test-reconnect-camera", reconnect_camera,
                    "--test-reconnect-at", str(reconnect_at)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cameras", default=",".join(f"CAM-{i:02d}" for i in range(1, 7)))
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--test-reconnect-camera")
    parser.add_argument("--test-reconnect-at", type=float, default=20.0)
    args = parser.parse_args()
    python = preflight_python(ROOT)
    from services.ml_service.app.config import load_settings
    from services.camera_v11.source_ownership import SourceOwnership
    selected = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    SourceOwnership(selected, ROOT / ".runtime/camera-owner-locks")  # Validate names, no locks yet.
    configured = {c.camera_id: c for c in load_settings().cameras}
    if any(c not in configured or not configured[c].username or not configured[c].password for c in selected):
        raise ValueError("selected cameras require configured authorized credentials")
    output = args.output.resolve()
    # Never overwrite evidence from another run.
    output.mkdir(parents=True, exist_ok=False)
    (ROOT / ".runtime/camera-owner-locks").mkdir(exist_ok=True)
    cmd = command(ROOT, python, output, args.cameras, args.duration,
                  args.test_reconnect_camera, args.test_reconnect_at)
    (output / "launch.json").write_text(json.dumps({"command": cmd, "inference": False,
                                                  "tracking": False}, indent=2) + "\n")
    # Replace the host wrapper: TERM/INT reaches docker, then the worker; no
    # detached container or independently surviving camera owner is introduced.
    os.execvp(cmd[0], cmd)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
