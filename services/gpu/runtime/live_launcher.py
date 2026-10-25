#!/usr/bin/env python3
"""Bounded official DeepStream 9.1 live preview; no connection by default.

--check validates local assets and prints no camera URIs or credentials.
--run requires explicit NVR authorization and independently reviewed network
egress isolation; these operator attestations are NOT firewall verification.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tarfile
import threading
import time

from services.gpu.runtime.live_config import Camera, build_app_config, validate_model_files
from services.shared.camera_config import load_settings
from services.shared.source_ownership import SourceOwnership
from services.shared.credential_redaction import (
    CredentialRedactor, child_environment_without_camera_secrets,
)

ROOT = Path(__file__).resolve().parents[3]
PLATFORM = ROOT / "config/deepstream-platform.json"


def secret_tar(app: str, infer: str) -> bytes:
    """Carry sensitive application config over stdin into container tmpfs."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        for name, value in (("app.txt", app), ("infer.txt", infer)):
            data = value.encode("utf-8")
            item = tarfile.TarInfo(name)
            item.size = len(data)
            item.mode = 0o600
            item.uid = item.gid = 0
            tar.addfile(item, io.BytesIO(data))
    return out.getvalue()


def checked_network(name: str, hostname: str, port: int) -> None:
    if name in {"host", "bridge", "default", "none"} or not name:
        raise ValueError("a separately reviewed, restricted Docker network is required")
    result = subprocess.run(
        ["docker", "network", "inspect", name], capture_output=True,
        text=True, timeout=8, check=True,
        env=child_environment_without_camera_secrets(),
    )
    info = json.loads(result.stdout)[0]
    if info.get("Name") != name or info.get("Driver") != "bridge":
        raise ValueError("expected an existing dedicated bridge network")
    if info.get("EnableIPv6"):
        raise ValueError("IPv6 egress needs separate review; refusing this network")
    if info.get("Internal"):
        raise ValueError("internal network cannot reach the external NVR")
    # A Docker bridge alone never proves egress is restricted to this NVR.
    # --network-egress-verified is an explicit independent ADMIN attestation,
    # not a conclusion made from docker network inspect.
    if not hostname or not 1 <= port <= 65535:
        raise ValueError("invalid reviewed NVR destination")


def display_preflight() -> tuple[str, Path]:
    display = os.environ.get("DISPLAY", "")
    if not display.startswith(":") or any(x in display for x in ("\n", "\r")):
        raise ValueError("a local X11 DISPLAY is required (Wayland-only unverified)")
    number = display[1:].split(".", 1)[0]
    if not number.isdecimal():
        raise ValueError("invalid local X11 display number")
    if not Path(f"/tmp/.X11-unix/X{number}").exists():
        raise ValueError("the X11 socket is unavailable")
    auth = Path(os.environ.get("XAUTHORITY") or Path.home() / ".Xauthority")
    if not auth.is_file():
        raise ValueError("XAUTHORITY file not found; do not use xhost +")
    return display, auth.resolve()


def nvr_sessions_preflight(hostname: str, port: int) -> None:
    """Fail closed if another host process already owns an NVR TCP session."""
    r = subprocess.run(["ss", "-H", "-tn", "state", "established"],
                       capture_output=True, text=True, timeout=5,
                       env=child_environment_without_camera_secrets())
    if r.returncode:
        raise RuntimeError("cannot inspect existing TCP sessions")
    for line in r.stdout.splitlines():
        if f"{hostname}:{port}" in line or f"[{hostname}]:{port}" in line:
            raise RuntimeError("NVR RTSP session already active; do not duplicate camera ownership")


def _docker_args(image: str, models_root: Path, network: str,
                 display: str, xauth: Path, name: str) -> list[str]:
    return [
        "docker", "run", "--rm", "-i", "--pull=never",
        "--name", name, "--log-driver=none", "--network", network,
        "--gpus", "device=0",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--read-only", "--pids-limit=512",
        "--tmpfs", "/tmp:rw,nosuid,nodev,mode=1777",
        "--tmpfs", "/run/live:rw,nosuid,nodev,noexec,mode=1777",
        "--mount", f"type=bind,src={models_root},dst=/models,readonly",
        "--mount", "type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly",
        "--mount", f"type=bind,src={xauth},dst=/run/xauth,readonly",
        "-e", f"DISPLAY={display}", "-e", "XAUTHORITY=/run/xauth",
        "-e", "HOME=/tmp", "-e", "XDG_CACHE_HOME=/tmp",
        "-e", "GST_REGISTRY=/tmp/live-gst-registry.bin",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics,display",
        "--entrypoint", "/bin/sh", image, "-ec",
        "umask 077; tar -xf - -C /run/live; "
        "exec deepstream-app -c /run/live/app.txt --tiledtext",
    ]


def _stop_owned_container(name: str) -> None:
    subprocess.run(["docker", "stop", "--time", "10", name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=20, env=child_environment_without_camera_secrets())


def _pump_lines(proc: subprocess.Popen, redactor: CredentialRedactor) -> None:
    if proc.stdout:
        for line in proc.stdout:
            safe = redactor.sanitize_child_output(line)
            print(safe, end="" if safe.endswith("\n") else "\n", flush=True)


def _ask_operator(proc: subprocess.Popen) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("interactive terminal required to confirm all six live tiles")
    print("Check all SIX physical camera tiles on the desktop.", flush=True)
    print("Enter START only after all six have moving live frames; 120s maximum.", flush=True)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("live viewer exited before operator confirmation")
        ready, _, _ = select.select([sys.stdin], [], [], 1)
        if ready:
            if sys.stdin.readline().strip() != "START":
                raise RuntimeError("operator did not confirm six live feeds")
            return
    raise RuntimeError("live-viewer startup was not confirmed within 120 seconds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path, required=True,
                        help="previously validated PeopleNet FP16 batch-6 nvinfer config")
    parser.add_argument("--models-root", type=Path, default=ROOT / ".runtime/models")
    parser.add_argument("--network", help="administrator-approved six-camera-only network")
    parser.add_argument("--expected-engine-sha256", help="known-good batch-6 engine SHA256 from the validated manifest")
    parser.add_argument("--check", action="store_true", help="no Docker/RTSP connection")
    parser.add_argument("--run", action="store_true", help="bounded physical RTSP run")
    parser.add_argument("--authorize-physical-rtsp", action="store_true")
    parser.add_argument("--nvr-readonly-authorized", action="store_true",
                        help="NVR admin confirmed read-only access to exactly the six channels")
    parser.add_argument("--network-egress-verified", action="store_true",
                        help="host admin independently inspected NVR-only firewall egress")
    args = parser.parse_args(argv)
    if args.check == args.run:
        parser.error("select exactly one of --check and --run")

    redactor = CredentialRedactor()
    for key, value in os.environ.items():
        if "PASSWORD" in key.upper() or "RTSP" in key.upper():
            redactor.add_secret(value)

    proc = None
    owned_name = None
    lock = None
    camera_ownership = None
    try:
        settings = load_settings(ROOT / "config/cameras.yaml")
        cameras = [
            Camera(c.camera_id, c.uri, c.username, c.password)
            for c in settings.cameras
        ]
        for cam in cameras:
            redactor.add_credentials(username=cam.username, password=cam.password, uri=cam.uri)
        app_config, manifest = build_app_config(cameras)
        infer_config = args.infer_config.read_text(encoding="utf-8")
        model_files = validate_model_files(infer_config, args.models_root)
        platform = json.loads(PLATFORM.read_text(encoding="utf-8"))
        models = {key: path.name for key, path in model_files.items()}
        public = {
            "mode": "CHECK" if args.check else "RUN",
            "cameras": manifest["cameras"],
            "channels": manifest["channels"],
            "layout": manifest["layout"],
            "model_file_names_only": models,
            "image": platform["image_tag"],
            "authorization": "not inferred from prior live tests",
        }
        if args.check:
            print(json.dumps(public, indent=2))
            print("CONFIG PREPARATION: PASS; LIVE NETWORK/ACL/EGL: NOT VERIFIED")
            return 0

        if not (args.authorize_physical_rtsp and args.nvr_readonly_authorized
                and args.network_egress_verified):
            raise RuntimeError("missing explicit live authorization, NVR ACL, or egress attestation")
        expected = (args.expected_engine_sha256 or "").strip().lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise RuntimeError("a verified 64-digit batch-6 engine SHA256 is required for live execution")
        digest = hashlib.sha256(model_files["model-engine-file"].read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError("batch-6 engine hash differs from validated evidence")
        display, xauth = display_preflight()
        nvr_sessions_preflight(manifest["nvr_host"], manifest["nvr_port"])
        checked_network(args.network or "", manifest["nvr_host"], manifest["nvr_port"])

        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=7,
            env=child_environment_without_camera_secrets(), check=True,
        )
        if int(gpu.stdout.splitlines()[0].strip()) < 3500:
            raise RuntimeError("insufficient GPU memory; existing workloads take priority")

        lock_file = ROOT / ".runtime" / "six-live-preview.lock"
        lock_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = lock_file.open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Use the same per-camera locks as the previously validated owners.
        # A separate GPU lock alone cannot prevent duplicate NVR sessions.
        camera_ownership = SourceOwnership(manifest["cameras"],
                                           ROOT / ".runtime/camera-owner-locks")
        camera_ownership.__enter__()
        owned_name = f"ai-surveillance-six-live-{os.getpid()}"
        image = platform["image"]
        cmd = _docker_args(image, args.models_root.resolve(), args.network,
                           display, xauth, owned_name)
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            env=child_environment_without_camera_secrets(),
        )
        pump = threading.Thread(target=_pump_lines, args=(proc, redactor), daemon=True)
        pump.start()
        assert proc.stdin is not None
        # stdin is the only credential delivery channel. No URI in Docker argv.
        proc.stdin.buffer.write(secret_tar(app_config, infer_config))
        proc.stdin.close()
        _ask_operator(proc)
        print("Six live feeds confirmed manually. Observing for 600 seconds.", flush=True)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("live viewer exited before 600-second observation")
            time.sleep(min(1.0, deadline - time.monotonic()))
        print("OBSERVATION WINDOW COMPLETE; stopping owned container.", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(redactor.sanitize_exception(exc), file=sys.stderr)
        return 2
    finally:
        if owned_name is not None:
            try:
                _stop_owned_container(owned_name)
            except (OSError, subprocess.SubprocessError):
                pass
        if proc is not None:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if camera_ownership is not None:
            camera_ownership.__exit__(None, None, None)
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
