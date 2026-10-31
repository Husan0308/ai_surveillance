#!/usr/bin/env python3
"""Safely execute the separately authorized Fix 5F synthetic plugin test.

Default invocation is read-only preflight. GPU access requires an explicit
command-line acknowledgement in addition to operator authorization in-session.
The test itself uses six generated appsrc streams, no network, and the exact
Fix 5E ownership callbacks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / ".runtime/deepstream-fix5-generation-20261031"
CANDIDATE = BASE / "sources/apps/sample_apps/deepstream-app/deepstream-app"
TEST_BINARY = BASE / "review/fix5_plugin_pipeline_test"
MODELS = ROOT / ".runtime/six-source-recorded-validation-20261024-r1"
INFER = ROOT / ".runtime/cam-all-live-fontfix-20261026-r1/infer_config.txt"
ENGINE = MODELS / "engine-cache/resnet34_peoplenet.onnx_b6_gpu0_fp16.engine"
ONNX = MODELS / "engine-cache/resnet34_peoplenet.onnx"
LABELS = MODELS / "model/detector_labels.txt"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b"
TRACKER_IN_IMAGE = "/opt/nvidia/deepstream/deepstream-9.1/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
CANDIDATE_SHA = "87a22a8ddd380676c37229df5791c53b62557356fd28996598561ddf3879ee35"
CANDIDATE_BUILD_ID = "8fb26e4f5f5c791e0a4259a887f8ba36eb7af3c6"
TEST_BINARY_SHA = "3b06b5eda20fb3bb21bd31c7bae3b199b3b56aa312fd4d5a835c3361798dacf5"
TEST_BINARY_BUILD_ID = "b0d942aa444a2bfaa7626ae6ab793b789dd8fac8"
SOURCE_HASHES = {
    BASE / "sources/apps/sample_apps/deepstream-app/six_live_monitor.c":
        "70dcd04db25bfe4f56e5755a77bfe23d82719d7b2ffff36bd891e700575d4fe4",
    BASE / "sources/apps/sample_apps/deepstream-app/deepstream_app.c":
        "3b3aa1e1e7f05635c314018cc16bc0fd9ed3b36a595844f85b16accf6574da43",
    BASE / "sources/apps/sample_apps/deepstream-app/Makefile":
        "d8372b79b91fc0bc45e855ce5c6a7a55403673ed783c6e33b14c4c8605cd2a6e",
    ROOT / "tests/native/fix5_plugin_pipeline_test.mk":
        "3c5fe7777303c465ac01692871fc9a47892866cf7ca8cac12d42edd20f60fd10",
    ROOT / "tests/native/fix5_plugin_pipeline_test.c":
        "23aba399fafb5293ec84a22d86b1323a95d457496a0efb4fe4ba943c3dcb5380",
}
ENGINE_SHA = "3b75d51182b9f4679c328ac339e5d063dde681874f83e388792a933f34e9e3d2"
ONNX_SHA = "a6aa31618f8476261807b5ba26b9841c684708c66a5f0be460a25a972f8ee7ad"
LABELS_SHA = "d58fbd3accfafe6676ec9243f1c8612bd4220c5df9ecb28fa7f2ac1da6bc7411"
INFER_SHA = "b23b1ab55fa2f9380d0514687d07be164930e4d74e6c7b578137530b5c4b6dcf"
TRACKER_SHA = "383002ab8628203a6921616adf46c371e261b9a76509943e4b61657f9102e86c"
IMAGE_ID = "sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b"
OWNER_LABEL = "ai_surveillance.fix5f.run"
TOTAL_LIMIT_SECONDS = 120
PREFLIGHT_LIMIT_SECONDS = 15
STARTUP_LIMIT_SECONDS = 20
INNER_LIMIT_SECONDS = 65
INNER_KILL_AFTER_SECONDS = 5
HOST_GRACEFUL_STOP_SECONDS = 92
HOST_FORCE_SECONDS = 100
GPU_MIN_FREE_MIB = 8192


class Blocked(RuntimeError):
    pass


def remaining_timeout(start_mono: float, maximum: float, *, reserve: float = 2) -> float:
    remaining = TOTAL_LIMIT_SECONDS - (time.monotonic() - start_mono) - reserve
    if remaining <= 0:
        raise Blocked("overall Fix 5F wall-clock deadline reached")
    return min(maximum, remaining)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_readonly(argv: list[str], *, timeout: float = 5) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)


def verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise Blocked(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise Blocked(f"{label} SHA256 mismatch: {actual}")


def inspect_candidate() -> dict[str, str]:
    verify_file(CANDIDATE, CANDIDATE_SHA, "candidate viewer")
    readelf = run_readonly(["readelf", "-n", str(CANDIDATE)])
    match = re.search(r"Build ID:\s*([0-9a-f]+)", readelf.stdout)
    if readelf.returncode or not match or match.group(1) != CANDIDATE_BUILD_ID:
        raise Blocked("candidate Build ID does not match the approved value")
    verify_file(TEST_BINARY, TEST_BINARY_SHA, "Fix 5F test binary")
    test_readelf = run_readonly(["readelf", "-n", str(TEST_BINARY)])
    test_match = re.search(r"Build ID:\s*([0-9a-f]+)", test_readelf.stdout)
    if test_readelf.returncode or not test_match or test_match.group(1) != TEST_BINARY_BUILD_ID:
        raise Blocked("Fix 5F test binary Build ID does not match the reviewed artifact")
    source_hashes = {}
    for source_path, expected_hash in SOURCE_HASHES.items():
        verify_file(source_path, expected_hash, "Fix 5F source input")
        source_hashes[str(source_path.relative_to(ROOT))] = expected_hash
    verify_file(ENGINE, ENGINE_SHA, "PeopleNet batch-6 engine")
    verify_file(ONNX, ONNX_SHA, "PeopleNet ONNX model")
    verify_file(LABELS, LABELS_SHA, "PeopleNet labels")
    verify_file(INFER, INFER_SHA, "PeopleNet inference config")
    return {
        "candidate_sha256": CANDIDATE_SHA,
        "candidate_build_id": CANDIDATE_BUILD_ID,
        "test_binary_sha256": TEST_BINARY_SHA,
        "test_binary_build_id": TEST_BINARY_BUILD_ID,
        "test_source_hashes": source_hashes,
        "engine_sha256": ENGINE_SHA,
        "onnx_sha256": ONNX_SHA,
        "labels_sha256": LABELS_SHA,
        "infer_config_sha256": INFER_SHA,
    }


def image_identity() -> dict:
    result = run_readonly(["docker", "image", "inspect", IMAGE])
    if result.returncode:
        raise Blocked("pinned DeepStream image is unavailable locally")
    try:
        info = json.loads(result.stdout)[0]
    except (ValueError, IndexError, TypeError) as exc:
        raise Blocked("could not parse local image identity") from exc
    if info.get("Id") != IMAGE_ID:
        raise Blocked("local DeepStream image ID does not match the pinned digest")
    return {"image": IMAGE, "image_id": IMAGE_ID}


def gpu_state() -> dict:
    memory = run_readonly([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"
    ])
    processes = run_readonly([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader"
    ])
    if memory.returncode or processes.returncode:
        raise Blocked("GPU state could not be inspected")
    free_mib: list[int] = []
    for line in memory.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1].isdigit():
            raise Blocked("malformed nvidia-smi memory output")
        if fields[0] == "0":
            free_mib.append(int(fields[1]))
    if len(free_mib) != 1 or free_mib[0] < GPU_MIN_FREE_MIB:
        raise Blocked(f"GPU 0 must have at least {GPU_MIN_FREE_MIB} MiB free")
    active = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
    if active:
        raise Blocked("GPU has unrelated compute users; leave them running and retry preflight later")
    return {"gpu": 0, "free_mib": free_mib[0], "active_compute_processes": []}


def owned_containers() -> list[str]:
    result = run_readonly([
        "docker", "ps", "-a", "--filter", f"label={OWNER_LABEL}",
        "--format", "{{.Names}} {{.Status}}"
    ])
    if result.returncode:
        raise Blocked("Docker container ownership state could not be inspected")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def preflight() -> dict:
    files = inspect_candidate()
    image = image_identity()
    gpu = gpu_state()
    old = owned_containers()
    if old:
        raise Blocked("an owned Fix 5F container exists and needs exact-owner review: " + "; ".join(old))
    tracker = run_readonly([
        "docker", "run", "--rm", "--pull=never", "--network", "none",
        "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--entrypoint", "/bin/bash", IMAGE,
        "-lc", f"sha256sum {shlex.quote(TRACKER_IN_IMAGE)}"
    ], timeout=8)
    tracker_hash = tracker.stdout.split()[0] if tracker.returncode == 0 and tracker.stdout.split() else ""
    if tracker_hash != TRACKER_SHA:
        raise Blocked("pinned SDK NvDCF config hash could not be verified")
    return {**files, **image, **gpu, "nvdcf_config_sha256": tracker_hash,
            "test_binary_build_time": True}


def inspect_owned_container(name: str, run_id: str, *, timeout: float = 5) -> dict | None:
    result = run_readonly(["docker", "inspect", name], timeout=timeout)
    if result.returncode:
        # Only an explicit Docker "not found" response proves the exact
        # container is absent. Daemon errors, timeouts, and permission failures
        # must fail closed instead of being misreported as successful cleanup.
        if ("No such object:" in result.stderr or
                "No such container:" in result.stderr):
            return None
        raise Blocked("could not verify exact owned container state")
    try:
        info = json.loads(result.stdout)[0]
    except (ValueError, IndexError, TypeError) as exc:
        raise Blocked("Docker returned malformed exact-container inspection data") from exc
    labels = ((info.get("Config") or {}).get("Labels") or {})
    if labels.get(OWNER_LABEL) != run_id:
        raise Blocked("container name resolves to an object without this run's ownership label")
    container_id = info.get("Id")
    if not isinstance(container_id, str) or not container_id:
        raise Blocked("Docker inspection omitted the exact container ID")
    state = info.get("State") or {}
    return {
        "id": container_id, "name": info.get("Name", "").lstrip("/"),
        "status": state.get("Status"), "exit_code": state.get("ExitCode"),
        "oom_killed": state.get("OOMKilled"), "error": state.get("Error"),
        "started_at": state.get("StartedAt"), "finished_at": state.get("FinishedAt"),
    }


def write_record(path: Path, event: str, **data: object) -> None:
    record = {"event": event, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **data}
    with path.open("a", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def docker_owner_action(action: str, name: str, run_id: str, timeout: float,
                        *, grace_seconds: int = 4) -> bool:
    action_deadline = time.monotonic() + timeout
    state = inspect_owned_container(
        name, run_id, timeout=max(0.1, min(2, action_deadline-time.monotonic())))
    if state is None:
        return True
    # Address the immutable container ID after validating its unique run label;
    # do not re-resolve a mutable name between ownership check and action.
    container_id = state["id"]
    if action == "stop":
        command = ["docker", "stop", "--time", str(grace_seconds), container_id]
    elif action == "kill":
        command = ["docker", "kill", container_id]
    else:
        raise ValueError(action)
    try:
        remaining = action_deadline - time.monotonic()
        if remaining <= 0:
            return False
        result = run_readonly(command, timeout=remaining)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def execute() -> int:
    start_mono = time.monotonic()
    utc_start = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = uuid.uuid4().hex[:12]
    name = f"fix5f-{run_id}"
    run_dir = BASE / "review" / "runs" / f"{utc_start}-{run_id}"
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(run_dir, 0o700)
    report_path = run_dir / "fix5f.jsonl"
    log_path = run_dir / "container.log"
    write_record(report_path, "run_start", run_id=run_id, candidate_sha256=CANDIDATE_SHA,
                 candidate_build_id=CANDIDATE_BUILD_ID, image=IMAGE,
                 test_binary_sha256=sha256(TEST_BINARY), total_limit_seconds=TOTAL_LIMIT_SECONDS)
    container_started = False
    docker_exit: int | None = None
    container_state: dict | None = None
    cleanup_ok = False
    p: subprocess.Popen | None = None
    try:
        pre = preflight()
        if time.monotonic() - start_mono > PREFLIGHT_LIMIT_SECONDS:
            raise Blocked("preflight took over 15 seconds; no GPU container started")
        write_record(report_path, "preflight_pass", **pre)
        command = [
            "docker", "run", "--pull=never", "--name", name,
            "--label", f"{OWNER_LABEL}={run_id}", "--network", "none",
            "--gpus", "device=0", "--memory=8g", "--memory-swap=8g",
            "--pids-limit=256", "--stop-timeout=8", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--mount", f"type=bind,src={TEST_BINARY},dst=/test/fix5_plugin_pipeline_test,readonly",
            "--mount", f"type=bind,src={INFER},dst=/test/infer_config.txt,readonly",
            # Mount only the engine/ONNX files and detector labels. The model
            # root also contains prior validation videos, screenshots, and logs.
            "--mount", f"type=bind,src={ENGINE.parent},dst=/models/engine-cache,readonly",
            "--mount", f"type=bind,src={LABELS.parent},dst=/models/model,readonly",
            "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
            "-e", "GST_REGISTRY=/tmp/fix5f-gst-registry.bin",
            "-e", "HOME=/tmp",
            "-e", "XDG_CACHE_HOME=/tmp",
            "-e", "CUDA_CACHE_PATH=/tmp",
            "-e", "GST_DEBUG=1",
            "--entrypoint", "/usr/bin/timeout", IMAGE,
            "--foreground", "--signal=TERM", f"--kill-after={INNER_KILL_AFTER_SECONDS}s",
            f"{INNER_LIMIT_SECONDS}s", "/test/fix5_plugin_pipeline_test",
            "/test/infer_config.txt",
        ]
        write_record(report_path, "container_start", name=name,
                     command=" ".join(shlex.quote(part) for part in command),
                     network="none", gpu_device=0, inner_timeout_seconds=INNER_LIMIT_SECONDS)
        with log_path.open("ab", buffering=0) as log:
            os.chmod(log_path, 0o600)
            p = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True)
            container_started = True
            # Do not let a late Docker startup push the inner timeout beyond
            # the total test budget. The in-container timeout is independent
            # of this Python supervisor once the container starts.
            startup_deadline = start_mono + STARTUP_LIMIT_SECONDS
            container_seen_running = False
            while p.poll() is None and time.monotonic() < startup_deadline:
                state = inspect_owned_container(name, run_id, timeout=1)
                if state and state.get("status") == "running":
                    container_seen_running = True
                    break
                time.sleep(0.25)
            if p.poll() is None and not container_seen_running:
                raise Blocked("owned test container did not reach running state before startup deadline")
            soft_stop_at = start_mono + HOST_GRACEFUL_STOP_SECONDS
            hard_stop_at = start_mono + HOST_FORCE_SECONDS
            while p.poll() is None:
                now = time.monotonic()
                if now >= soft_stop_at:
                    write_record(report_path, "host_deadline_graceful_stop", elapsed_seconds=round(now-start_mono, 3))
                    docker_owner_action("stop", name, run_id, timeout=5, grace_seconds=4)
                    try:
                        p.wait(timeout=max(0.1, min(4, hard_stop_at - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
                    if p.poll() is not None:
                        break
                    now = time.monotonic()
                    if now < hard_stop_at:
                        time.sleep(min(0.1, hard_stop_at - now))
                        continue
                    break
                time.sleep(0.1)
            if p.poll() is None and time.monotonic() >= hard_stop_at:
                write_record(report_path, "host_deadline_forced_cleanup", elapsed_seconds=round(time.monotonic()-start_mono, 3))
                docker_owner_action("kill", name, run_id, timeout=2)
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # This is only the private docker-run client process group
                    # created above. Container actions remain label-verified.
                    os.killpg(p.pid, signal.SIGTERM)
                    try:
                        p.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait(timeout=1)
            docker_exit = p.returncode

        container_state = inspect_owned_container(
            name, run_id, timeout=remaining_timeout(start_mono, 2, reserve=8))
        write_record(report_path, "container_state_before_remove", state=container_state,
                     docker_run_exit_code=docker_exit)
        if container_state is not None and container_state.get("status") == "running":
            docker_owner_action("stop", name, run_id,
                                timeout=remaining_timeout(start_mono, 4, reserve=8),
                                grace_seconds=2)
            container_state = inspect_owned_container(
                name, run_id, timeout=remaining_timeout(start_mono, 2, reserve=6))
        if container_state is not None and container_state.get("status") == "running":
            docker_owner_action("kill", name, run_id,
                                timeout=remaining_timeout(start_mono, 4, reserve=6))
            container_state = inspect_owned_container(
                name, run_id, timeout=remaining_timeout(start_mono, 2, reserve=4))
        if container_state is not None:
            if inspect_owned_container(name, run_id,
                    timeout=remaining_timeout(start_mono, 2, reserve=4)) is None:
                cleanup_ok = True
            else:
                rm = run_readonly(["docker", "rm", container_state["id"]],
                                  timeout=remaining_timeout(start_mono, 2, reserve=2))
                cleanup_ok = rm.returncode == 0 and inspect_owned_container(
                    container_state["id"], run_id,
                    timeout=remaining_timeout(start_mono, 1, reserve=1)) is None
        else:
            cleanup_ok = True
        write_record(report_path, "cleanup", cleanup_ok=cleanup_ok,
                     docker_run_exit_code=docker_exit, container_state=container_state)
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        test_pass = "FIX5F_PASS " in log_text
        success = test_pass and docker_exit == 0 and container_state is not None and \
            container_state.get("exit_code") == 0 and not container_state.get("oom_killed") and cleanup_ok
        write_record(report_path, "final", status="PASS" if success else "FAIL",
                     fix5f_pass_marker=test_pass, docker_run_exit_code=docker_exit,
                     container_state=container_state, cleanup_ok=cleanup_ok,
                     elapsed_seconds=round(time.monotonic()-start_mono, 3))
        return 0 if success else 1
    except (Blocked, OSError, subprocess.SubprocessError) as exc:
        failure_state: dict | None = None
        failure_cleanup_ok = not container_started
        cleanup_error: str | None = None
        if container_started:
            try:
                failure_state = inspect_owned_container(name, run_id,
                    timeout=remaining_timeout(start_mono, 2, reserve=5))
                write_record(report_path, "container_state_before_failure_cleanup",
                             state=failure_state,
                             docker_run_exit_code=p.returncode if p else None)
                docker_owner_action("stop", name, run_id,
                    timeout=remaining_timeout(start_mono, 4, reserve=5), grace_seconds=2)
                state = inspect_owned_container(name, run_id,
                    timeout=remaining_timeout(start_mono, 2, reserve=4))
                if state and state.get("status") == "running":
                    docker_owner_action("kill", name, run_id,
                        timeout=remaining_timeout(start_mono, 4, reserve=4))
                state = inspect_owned_container(name, run_id,
                    timeout=remaining_timeout(start_mono, 2, reserve=3))
                if state is not None:
                    run_readonly(["docker", "rm", state["id"]],
                        timeout=remaining_timeout(start_mono, 2, reserve=2))
                failure_cleanup_ok = inspect_owned_container(name, run_id,
                    timeout=remaining_timeout(start_mono, 1, reserve=1)) is None
            except (Blocked, OSError, subprocess.SubprocessError) as cleanup_exc:
                cleanup_error = type(cleanup_exc).__name__
                failure_cleanup_ok = False
        write_record(report_path, "failure", category=type(exc).__name__,
                     message=str(exc), docker_run_exit_code=p.returncode if p else None,
                     container_state=failure_state, cleanup_ok=failure_cleanup_ok,
                     cleanup_error_category=cleanup_error,
                     elapsed_seconds=round(time.monotonic()-start_mono, 3))
        print(f"FIX5F_BLOCKED_OR_FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"evidence={run_dir}", file=sys.stderr)
        return 2 if isinstance(exc, Blocked) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true", help="read-only identity/resource preflight")
    parser.add_argument("--operator-authorized-gpu-test", action="store_true",
                        help="required execution latch; separate operator authorization is still required")
    args = parser.parse_args()
    if args.operator_authorized_gpu_test:
        return execute()
    try:
        info = preflight()
    except (Blocked, OSError, subprocess.SubprocessError) as exc:
        print(f"FIX5F_PREFLIGHT_BLOCKED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(info, sort_keys=True))
    if args.preflight_only or not args.operator_authorized_gpu_test:
        if not args.preflight_only:
            print("GPU test not started; pass --operator-authorized-gpu-test only after separate operator authorization")
        return 0
    return execute()


if __name__ == "__main__":
    raise SystemExit(main())
