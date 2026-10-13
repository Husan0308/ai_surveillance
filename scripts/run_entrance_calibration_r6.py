#!/usr/bin/env python3
"""Safely select the fixed r6 recorder owner, then run one Entrance session.

This is an operator command. It may gracefully replace only the known r5
full-stack supervisor because r5's recorder is left FAILED after interruption.
It never kills a process forcibly, never opens an RTSP connection itself, and
never changes AutoMagicCalib on port 8000.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".runtime/full-stack-venv/bin/python"
CANDIDATE = ROOT / ".runtime/sixcam-calibration-recording-20261009-r6-eos-fix"
R6_BINARY = CANDIDATE / "bin/cam-six-yolo-local6-posttracker"
R6_SHA256 = "6d3bf5613bebb0322c7c924b354f5f3c6097aa7b9750467c2d5c83ae58f817bd"
R5_CANDIDATE = ROOT / ".runtime/sixcam-calibration-recording-20261009-r5"
R5_SHA256 = "54edebdb4c2764b819d57014bafab17549ed94f080e0fa43c7bb3cd5cf080604"
ENGINE_SHA256 = "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"
TRACKER_SHA256 = "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355"
V13_SHA256 = "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834"
CONTAINER_IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify_candidate() -> dict:
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        raise RuntimeError(f"project runtime is unavailable: {PYTHON}")
    manifest_path = CANDIDATE / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if digest(R6_BINARY) != R6_SHA256 or manifest.get("binary_sha256") != R6_SHA256:
        raise RuntimeError("r6 executable hash does not match the validated recorder-EOS candidate")
    if manifest.get("model_sha256", {}).get("yolo26m_raw_otm_b6_fp16.engine") != ENGINE_SHA256:
        raise RuntimeError("r6 YOLO engine hash mismatch")
    if manifest.get("tracker_config_sha256") != TRACKER_SHA256:
        raise RuntimeError("r6 local tracker hash mismatch")
    if manifest.get("frozen_v13_reference_sha256") != V13_SHA256:
        raise RuntimeError("r6 frozen V13 reference hash mismatch")
    if manifest.get("deepstream_image") != CONTAINER_IMAGE:
        raise RuntimeError("r6 DeepStream image provenance mismatch")
    profile = manifest.get("detector_profile", {})
    if (profile.get("name") != "yolo26m-fp16-640-primary-768-aux-interval-19-context-verified"
            or profile.get("batch_size") != 6 or profile.get("aux_interval") != 19):
        raise RuntimeError("r6 detector profile is not the prepared six-camera interval-19 candidate")
    return manifest


def _proc(pid: int) -> dict | None:
    base = Path(f"/proc/{pid}")
    try:
        cmd = base.joinpath("cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
        cwd = Path(os.readlink(base / "cwd"))
        env_rows = base.joinpath("environ").read_bytes().split(b"\0")
    except (OSError, PermissionError):
        return None
    env = {}
    for row in env_rows:
        if b"=" in row:
            key, value = row.split(b"=", 1)
            if key in {b"FULL_STACK_OUT", b"FULL_STACK_ANALYTICS_CANDIDATE",
                       b"FULL_STACK_YOLO_CONTEXT_VERIFIER"}:
                env[key.decode()] = value.decode(errors="replace")
    return {"pid": pid, "cmd": cmd, "cwd": cwd, "env": env}


def supervisors() -> list[dict]:
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            process = _proc(int(entry.name))
            if process and "scripts.full_stack_runtime" in process["cmd"] and process["cwd"] == ROOT:
                found.append(process)
    return found


def proc_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def established_rtsp() -> list[str]:
    result = subprocess.run(["ss", "-Htnp", "state", "established", "( dport = :554 )"],
                            text=True, capture_output=True, check=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def containers() -> list[str]:
    result = subprocess.run(["docker", "ps", "--filter", "name=surv-sixcam-analytics",
                             "--format", "{{.Names}}"], text=True, capture_output=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def port_listener_signature(port: int) -> tuple[str, ...]:
    result = subprocess.run(["ss", "-Hltn", "(", "sport", "=", f":{port}", ")"],
                            text=True, capture_output=True, check=True)
    return tuple(sorted(line.strip() for line in result.stdout.splitlines() if line.strip()))


def port_is_free(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.25)
    try:
        code = probe.connect_ex(("127.0.0.1", port))
    finally:
        probe.close()
    return code == 111  # ECONNREFUSED


def stack_launch(process: dict) -> tuple[Path, dict]:
    raw_out = process["env"].get("FULL_STACK_OUT")
    raw_candidate = process["env"].get("FULL_STACK_ANALYTICS_CANDIDATE")
    if not raw_out or not raw_candidate:
        raise RuntimeError("active full-stack owner lacks explicit candidate/output provenance")
    output = Path(raw_out)
    if not output.is_absolute():
        output = (process["cwd"] / output).resolve()
    candidate = Path(raw_candidate)
    if not candidate.is_absolute():
        candidate = (process["cwd"] / candidate).resolve()
    launch_path = output / "launch.json"
    if not launch_path.is_file():
        raise RuntimeError(f"active supervisor has no launch manifest: {launch_path}")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if Path(launch.get("analytics_candidate", "")).resolve() != candidate:
        raise RuntimeError("active launch manifest and supervisor candidate disagree")
    return output, launch


def verify_old_owner(process: dict) -> Path:
    output, launch = stack_launch(process)
    candidate = Path(process["env"]["FULL_STACK_ANALYTICS_CANDIDATE"])
    if not candidate.is_absolute():
        candidate = (process["cwd"] / candidate).resolve()
    if (candidate != R5_CANDIDATE.resolve()
            or output != (ROOT / ".runtime/full-stack-calibration-20261009-r5").resolve()
            or launch.get("analytics_binary_sha256") != R5_SHA256
            or launch.get("yolo_engine_sha256") != ENGINE_SHA256
            or launch.get("local_tracker_sha256") != TRACKER_SHA256
            or launch.get("yolo_context_verifier_enabled") is not True):
        raise RuntimeError("active stack is not the known r5 owner; refusing to stop it")
    child_path = output / "processes.json"
    if not child_path.is_file():
        raise RuntimeError("known r5 supervisor has no child ownership manifest")
    children = json.loads(child_path.read_text(encoding="utf-8"))
    expected = {"analytics", "ml", "api", "frontend"}
    if set(children) != expected:
        raise RuntimeError("known r5 child manifest is incomplete; refusing restart")
    expected_cmds = {
        "analytics": "scripts.run_six_camera_v13_wall_candidate",
        "ml": "services.ml_service.app.main",
        "api": "services.api_service.app.main",
        "frontend": "services.frontend.app.main",
    }
    for name, child_pid in children.items():
        child = _proc(int(child_pid))
        if not child or child["cwd"] != ROOT or expected_cmds[name] not in child["cmd"]:
            raise RuntimeError(f"r5 {name} process identity does not match its owner manifest")
    return output


def verify_r6_owner(process: dict) -> tuple[Path, dict]:
    output, launch = stack_launch(process)
    candidate = Path(process["env"].get("FULL_STACK_ANALYTICS_CANDIDATE", ""))
    if not candidate.is_absolute():
        candidate = (process["cwd"] / candidate).resolve()
    if (candidate != CANDIDATE.resolve()
            or launch.get("analytics_binary_sha256") != R6_SHA256
            or launch.get("yolo_engine_sha256") != ENGINE_SHA256
            or launch.get("local_tracker_sha256") != TRACKER_SHA256
            or launch.get("yolo_context_verifier_enabled") is not True
            or process["env"].get("FULL_STACK_YOLO_CONTEXT_VERIFIER") != "1"):
        raise RuntimeError("active r6 owner provenance is inconsistent; refusing to operate on it")
    children_path = output / "processes.json"
    if not children_path.is_file():
        raise RuntimeError("active r6 owner lacks its child process manifest")
    children = json.loads(children_path.read_text(encoding="utf-8"))
    if set(children) != {"analytics", "ml", "api", "frontend"}:
        raise RuntimeError("active r6 child process manifest is incomplete")
    expected_cmds = {
        "analytics": "scripts.run_six_camera_v13_wall_candidate",
        "ml": "services.ml_service.app.main",
        "api": "services.api_service.app.main",
        "frontend": "services.frontend.app.main",
    }
    for name, child_pid in children.items():
        child = _proc(int(child_pid))
        if not child or child["cwd"] != ROOT or expected_cmds[name] not in child["cmd"]:
            raise RuntimeError(f"active r6 {name} process does not match its owner manifest")
    return output, launch


def wait_old_owner_stopped(process: dict, output: Path, timeout_sec: float = 70) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline and proc_alive(process["pid"]):
        time.sleep(0.2)
    if proc_alive(process["pid"]):
        raise TimeoutError("r5 supervisor did not stop gracefully; no forced kill was attempted")
    shutdown_path = output / "shutdown.json"
    if not shutdown_path.is_file():
        raise RuntimeError("r5 exited without writing shutdown evidence")
    shutdown = json.loads(shutdown_path.read_text(encoding="utf-8"))
    if shutdown.get("status") != "PASS" or shutdown.get("forced_kill") is not False:
        raise RuntimeError("r5 shutdown was not clean; inspect its shutdown.json before retrying")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if not containers() and not established_rtsp() and port_is_free(8100) and port_is_free(8101):
            return
        time.sleep(0.25)
    raise TimeoutError("old container, RTSP sockets, or surveillance ports remain after graceful stop")


def api_ready() -> bool:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:8101/health", timeout=1.5) as response:
            ml = json.loads(response.read())
        with urllib.request.urlopen("http://127.0.0.1:8100/health", timeout=1.5) as response:
            api = json.loads(response.read())
        with urllib.request.urlopen("http://127.0.0.1:8100/api/v1/cameras", timeout=1.5) as response:
            cameras = json.loads(response.read()).get("cameras", [])
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return (ml.get("status") == "ok" and ml.get("online_camera_count") == 6
            and api.get("status") == "ok" and api.get("ml_status") == "ok"
            and len(cameras) == 6 and all(row.get("online") for row in cameras)
            and all(not row.get("pipeline_errors", 0) and not row.get("reconnects", 0)
                    for row in cameras))


def wait_pair_preview_progress(timeout_sec: float = 45.0) -> None:
    from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader
    cameras = ("CAM-02", "CAM-05")
    readers = {camera: PreviewFrameReader(
        f"/dev/shm/v11_ui_preview_{camera.lower().replace('-', '')}_v1.bin")
        for camera in cameras}
    previous: dict[str, tuple[int, int, int]] = {}
    deadline = time.monotonic() + timeout_sec
    try:
        while time.monotonic() < deadline:
            for camera, reader in readers.items():
                prior = previous.get(camera)
                frame = reader.read_latest(max_age_sec=1.0,
                                           after_sequence=prior[0] if prior else None,
                                           metadata_only=True)
                if frame is None or not frame.tracker_snapshot:
                    continue
                snapshot = frame.tracker_snapshot
                if (int(snapshot.get("pts_ns", -1)) != frame.pts_ns
                        or int(snapshot.get("source_frame_num", -1)) != frame.source_frame_num
                        or int(snapshot.get("frame_num", -1)) != frame.tracker_frame_num):
                    continue
                if prior is None:
                    previous[camera] = (frame.sequence, frame.pts_ns, frame.source_frame_num)
                elif (frame.sequence > prior[0] and frame.pts_ns > prior[1]
                      and frame.source_frame_num > prior[2]):
                    previous[camera] = (frame.sequence, frame.pts_ns, frame.source_frame_num)
            if all(camera in previous for camera in cameras):
                # The first valid sample is not enough; require each source to
                # advance again with its own embedded tracker metadata.
                first = dict(previous)
                second_deadline = time.monotonic() + min(4.0, timeout_sec)
                while time.monotonic() < second_deadline:
                    advanced = set()
                    for camera, reader in readers.items():
                        prior = previous[camera]
                        frame = reader.read_latest(max_age_sec=1.0,
                                                   after_sequence=prior[0],
                                                   metadata_only=True)
                        if frame is None or not frame.tracker_snapshot:
                            continue
                        snapshot = frame.tracker_snapshot
                        if (frame.sequence > prior[0] and frame.pts_ns > prior[1]
                                and frame.source_frame_num > prior[2]
                                and int(snapshot.get("pts_ns", -1)) == frame.pts_ns
                                and int(snapshot.get("source_frame_num", -1)) == frame.source_frame_num
                                and int(snapshot.get("frame_num", -1)) == frame.tracker_frame_num):
                            previous[camera] = (frame.sequence, frame.pts_ns)
                            advanced.add(camera)
                    if len(advanced) == len(cameras):
                        return
                    time.sleep(0.03)
                raise TimeoutError("Entrance pair preview did not advance with exact embedded PTS")
            time.sleep(0.03)
    finally:
        for reader in readers.values():
            reader.close()
    raise TimeoutError("fresh CAM-02/CAM-05 shared previews did not become ready")


def verify_live_stack(run_dir: Path, expected_supervisor: int | None = None) -> None:
    launch_path = run_dir / "launch.json"
    status_path = run_dir / "analytics/calibration-recording-status.json"
    if not launch_path.is_file() or not status_path.is_file():
        raise RuntimeError("r6 launch or recorder status manifest is missing")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (launch.get("analytics_binary_sha256") != R6_SHA256
            or launch.get("analytics_candidate") != str(CANDIDATE.resolve())
            or launch.get("yolo_engine_sha256") != ENGINE_SHA256
            or launch.get("local_tracker_sha256") != TRACKER_SHA256
            or launch.get("yolo_context_verifier_enabled") is not True):
        raise RuntimeError("active runtime is not the exact r6 recording candidate/profile")
    if status.get("state") != "IDLE":
        raise RuntimeError(f"native recorder is not IDLE: {status.get('state')}")
    if len(containers()) != 1 or len(established_rtsp()) != 6:
        raise RuntimeError("expected exactly one DeepStream owner and six RTSP sockets")
    if not api_ready():
        raise RuntimeError("ML/API are not healthy with all six cameras online")
    if expected_supervisor is not None and not proc_alive(expected_supervisor):
        raise RuntimeError("r6 supervisor exited during readiness verification")
    wait_pair_preview_progress()


def new_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / f".runtime/full-stack-entrance-recording-r6-{stamp}"
    if path.exists():
        raise FileExistsError(f"refusing to reuse runtime output directory: {path}")
    return path


def start_r6(run_dir: Path, log_path: Path, environment: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            ["bash", "scripts/start_full_live_stack.sh"], cwd=ROOT, env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            close_fds=True, start_new_session=True,
        )
    print(f"Starting the exact r6 six-camera owner; startup log: {log_path.relative_to(ROOT)}", flush=True)
    deadline = time.monotonic() + 130
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"r6 startup exited with code {process.returncode}; inspect {log_path}")
        if (run_dir / "ready.json").is_file():
            try:
                verify_live_stack(run_dir, process.pid)
                print("r6 owner ready: six RTSP sources, ML/API, previews and recorder IDLE verified.", flush=True)
                return process.pid
            except (OSError, ValueError, RuntimeError, TimeoutError):
                # Full-stack writes ready.json when services are healthy; the
                # native preview and recorder status can lag by a few seconds.
                pass
        time.sleep(0.25)
    if proc_alive(process.pid):
        print("r6 did not reach readiness; requesting graceful shutdown (no forced kill).", file=sys.stderr, flush=True)
        os.kill(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 70
        while time.monotonic() < deadline and proc_alive(process.pid):
            time.sleep(0.25)
    raise TimeoutError(f"r6 did not become ready within 130 seconds; inspect {log_path}")


def run(preflight_only: bool = False) -> int:
    verify_candidate()
    amc_before = port_listener_signature(8000)
    if not amc_before:
        raise RuntimeError("AutoMagicCalib listener on port 8000 is not visible; refusing any restart")
    running = supervisors()
    if len(running) > 1:
        raise RuntimeError("multiple full-stack supervisors detected; refusing ambiguous ownership")

    reuse_run: Path | None = None
    old_owner: tuple[dict, Path] | None = None
    if running:
        process = running[0]
        output, launch = stack_launch(process)
        if (launch.get("analytics_binary_sha256") == R6_SHA256
                and launch.get("analytics_candidate") == str(CANDIDATE.resolve())):
            output, launch = verify_r6_owner(process)
            status_path = output / "analytics/calibration-recording-status.json"
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                status = {}
            if status.get("state") == "IDLE":
                reuse_run = output
            else:
                # A failed native recorder is recoverable only by replacing
                # the exact, fully provenance-checked r6 owner.
                old_owner = (process, output)
        else:
            old_owner = (process, verify_old_owner(process))

    if preflight_only:
        if reuse_run:
            print(f"PREFLIGHT PASS: exact r6 owner is already live and recorder is IDLE: {reuse_run}")
        elif old_owner:
            print(f"PREFLIGHT PASS: exact r6 candidate verified; one guarded graceful replacement is needed for {old_owner[1]}")
        elif not running:
            if containers() or established_rtsp() or not port_is_free(8100) or not port_is_free(8101):
                raise RuntimeError("no supervisor owns resources but a container/socket/service port remains")
            print("PREFLIGHT PASS: exact r6 candidate verified; no stack is active, one owner can be started")
        return 0

    if reuse_run is not None:
        run_dir = reuse_run
        print("Exact r6 candidate is already active; reusing its single owner.", flush=True)
    else:
        if old_owner:
            process, output = old_owner
            # Reconfirm the identity immediately before signaling.
            current = _proc(process["pid"])
            if (not current or current["cwd"] != ROOT
                    or "scripts.full_stack_runtime" not in current["cmd"]
                    or current["env"] != process["env"]):
                raise RuntimeError("supervisor identity changed before stop; no process was signaled")
            amc_before_stop = port_listener_signature(8000)
            print(f"Stopping only the verified {output.name} supervisor cleanly before the single-owner r6 restart.", flush=True)
            os.kill(process["pid"], signal.SIGTERM)
            wait_old_owner_stopped(process, output)
            if port_listener_signature(8000) != amc_before_stop:
                raise RuntimeError("port 8000 listener changed during surveillance restart; inspect system state")
            print("Old RTSP sockets released; ports 8100/8101 free; AutoMagicCalib listener unchanged.", flush=True)
        else:
            if containers() or established_rtsp() or not port_is_free(8100) or not port_is_free(8101):
                raise RuntimeError("resources are occupied without a recognized supervisor; refusing a second owner")
        run_dir = new_run_dir()
        environment = dict(os.environ)
        environment["FULL_STACK_OUT"] = str(run_dir)
        environment["FULL_STACK_ANALYTICS_CANDIDATE"] = str(CANDIDATE.resolve())
        environment["FULL_STACK_YOLO_CONTEXT_VERIFIER"] = "1"
        environment["FULL_STACK_ACTIVE_TRACK_VERIFIER"] = "0"
        environment["FULL_STACK_LOWCONF_VERIFIER"] = "0"
        environment["V11_CALIBRATION_RECORDING_ENABLED"] = "1"
        # Retain the currently validated runtime value; this task does not
        # change recovery behavior.
        environment.setdefault("V11_SOURCE_STALL_RECOVERY", "0")
        log_path = ROOT / ".runtime/logs" / f"entrance-r6-start-{run_dir.name.rsplit('-', 1)[-1]}.log"
        start_r6(run_dir, log_path, environment)

    if port_listener_signature(8000) != amc_before:
        raise RuntimeError("AutoMagicCalib port-8000 listener does not match its pre-start state")
    verify_live_stack(run_dir)
    print("Preflight passed. Running this command authorizes the 180-second Entrance recording.", flush=True)
    print("🔴 REC STARTED will appear only after CAM-02 and CAM-05 files are both receiving frames.", flush=True)
    recorder_env = dict(os.environ)
    recorder_env["FULL_STACK_OUT"] = str(run_dir)
    result = subprocess.run(["bash", "scripts/record_entrance_calibration.sh"],
                            cwd=ROOT, env=recorder_env, check=False)
    if result.returncode == 0:
        print("✅ ENTRANCE VIDEO FINISHED. Six-camera live stack remains running.", flush=True)
    elif result.returncode == 130:
        print("Recording was interrupted; native EOS finalization was requested. The six-camera stack remains running.",
              file=sys.stderr, flush=True)
    else:
        print(f"Recording command failed with code {result.returncode}; the verified live stack remains running.",
              file=sys.stderr, flush=True)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true",
                        help="verify exact candidate and current owner without stopping or starting anything")
    args = parser.parse_args()
    try:
        return run(args.preflight_only)
    except Exception as exc:
        print(f"ENTRANCE_R6_COMMAND_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
