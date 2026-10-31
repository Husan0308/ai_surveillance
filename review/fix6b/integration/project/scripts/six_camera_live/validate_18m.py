#!/usr/bin/env python3
"""Isolated 18-minute DeepStream 9.1 validation supervisor.

This helper is opt-in and does not alter production monitor.sh. Its transient
user unit and private report are created only after all start preflight checks
succeed. The worker repeats preflight before it starts the camera-capable child.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
# Executing this file by path sets sys.path[0] to scripts/six_camera_live,
# not the repository root. Preflight and cleanup import project modules, so
# bootstrap the root explicitly before any of those imports run.
ROOT_TEXT = str(ROOT)
if ROOT_TEXT not in sys.path:
    sys.path.insert(0, ROOT_TEXT)
RUNTIME = ROOT / ".runtime/deepstream-18m-alarm-fix"
CANDIDATE_ROOT = ROOT / ".runtime/deepstream-fix6-upstream-20261031"
CANDIDATE = CANDIDATE_ROOT / "sources/apps/sample_apps/deepstream-app/deepstream-app"
CANDIDATE_LAUNCHER = CANDIDATE_ROOT / "host/services/gpu/runtime/live_launcher.py"
EXPECTED = "3cea6475a2474f903400825dba8f6f5f705db2fc93d9f49d7b6a5202f582f3b4"
CANDIDATE_LAUNCHER_EXPECTED = "611130f49ba5e0e0e54d4dc654376777638aa0fba278a9e0f0118ab1d010a581"
ENGINE = "3b75d51182b9f4679c328ac339e5d063dde681874f83e388792a933f34e9e3d2"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
UNIT = "ai-surveillance-18m-alarm-fix.service"
TOTAL = 1500
PRE = 1380
STOP = 90
STOP_COMMAND_TIMEOUT = 25
STOP_CLEANUP_MARGIN = 5
PREFLIGHT_BUDGET = 40
REQUIRED_NATIVE_OBSERVATION_SECONDS = 1080
GPU_MIN_FREE_MIB = 5120
CONTAINER_RE = re.compile(r"^ai-surveillance-six-live-(\d+)-[0-9a-f]{12}$")
signal_reason: str | None = None
_cpu_previous: list[int] | None = None


@dataclass(frozen=True)
class HealthResult:
    """One explicit health response shape for preflight and periodic samples."""

    status: int | None
    body: dict | None
    error: str | None = None

    @property
    def healthy(self) -> bool:
        return self.status == 200 and isinstance(self.body, dict) and self.error is None


def call(args: Sequence[str], timeout: float = 8) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def _remaining_timeout(deadline: float, cap: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("preflight_budget_exceeded")
    return min(cap, remaining)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def put(path: Path, row: dict) -> None:
    """Append one private JSONL record and fsync it before returning."""
    path = Path(path)
    fd = os.open(path, os.O_RDWR | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
                 | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise PermissionError("report file is not a user-owned regular file")
        os.fchmod(fd, 0o600)
        if info.st_size:
            previous = os.pread(fd, 1, info.st_size - 1)
            if previous != b"\n":
                # Keep the interrupted bytes intact and make the next valid
                # JSONL record independently readable after a crash.
                os.write(fd, b"\n")
        with os.fdopen(fd, "a", encoding="utf-8", closefd=False) as stream:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


def _decode_health(status: int, payload: bytes) -> HealthResult:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return HealthResult(status, None, "malformed_json")
    if not isinstance(value, dict):
        return HealthResult(status, None, "malformed_health_schema")
    return HealthResult(status, value)


def health(port: int, timeout: float = 3.0) -> HealthResult:
    """GET localhost /health, returning HealthResult for every outcome."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{int(port)}/health",
        headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _decode_health(response.status, response.read(64 * 1024 + 1))
    except urllib.error.HTTPError as exc:
        try:
            return _decode_health(exc.code, exc.read(64 * 1024 + 1))
        except OSError:
            return HealthResult(exc.code, None, "health_body_unreadable")
    except (TimeoutError, socket.timeout):
        return HealthResult(None, None, "network_timeout")
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            return HealthResult(None, None, "network_timeout")
        return HealthResult(None, None, "network_error")
    except (OSError, ValueError):
        return HealthResult(None, None, "network_error")


def gpu_snapshot(timeout: float = 5) -> dict | None:
    result = call([
        "nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,memory.free",
        "--format=csv,noheader,nounits"], timeout)
    if result.returncode != 0 or result.stderr.strip():
        return None
    try:
        values = [float(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
        if len(values) != 5 or any(value < 0 for value in values):
            return None
        return {"utilization_percent": values[0], "temperature_c": values[1],
                "vram_used_mib": values[2], "vram_total_mib": values[3],
                "vram_free_mib": values[4]}
    except (IndexError, ValueError):
        return None


def host_snapshot() -> dict | None:
    global _cpu_previous
    try:
        memory = {
            key: int(value.strip().split()[0])
            for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line
            for key, value in [line.split(":", 1)]
        }
        load = [float(value) for value in Path("/proc/loadavg").read_text().split()[:3]]
        current = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        cpu_percent = None
        if _cpu_previous:
            delta = [new - old for new, old in zip(current, _cpu_previous)]
            total = sum(delta)
            if total > 0:
                cpu_percent = 100 * (total - delta[3]) / total
        _cpu_previous = current
        return {"cpu_percent": cpu_percent, "ram_total_mib": memory["MemTotal"] / 1024,
                "ram_available_mib": memory["MemAvailable"] / 1024, "loadavg": load}
    except (OSError, KeyError, ValueError, IndexError):
        return None


def session(timeout: float = 5) -> tuple[bool, str, str]:
    display = os.environ.get("DISPLAY") or ":0"
    xauthority = os.environ.get("XAUTHORITY") or f"/run/user/{os.getuid()}/gdm/Xauthority"
    if not Path(xauthority).is_file():
        return False, display, xauthority
    try:
        result = call(["xdpyinfo", "-display", display], timeout)
        return result.returncode == 0, display, xauthority
    except (OSError, subprocess.TimeoutExpired):
        return False, display, xauthority


def _live_launcher_module():
    # These existing functions are read-only predicates; import has no start side effects.
    from services.gpu.runtime import live_launcher
    return live_launcher


def _camera_locks_free() -> bool:
    return _live_launcher_module()._camera_locks_free()


def _ipc_clear() -> dict[str, bool]:
    launcher = _live_launcher_module()
    return {**launcher._preview_ipc_checks(enabled=True),
            **launcher._telemetry_ipc_checks(enabled=True)}


def _rtsp_rows(timeout: float = 4) -> tuple[str, ...]:
    from scripts.six_camera_live.rtsp_session_preflight import inspect_established_rtsp_sessions
    def bounded_run(args, **kwargs):
        return subprocess.run(args, timeout=timeout, **kwargs)
    try:
        return inspect_established_rtsp_sessions(run=bounded_run)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ss_command_timeout") from exc


def _container_names(timeout: float = 7) -> tuple[str, ...]:
    result = call(["docker", "ps", "-a", "--format", "{{.Names}}"], timeout)
    if result.returncode != 0 or result.stderr.strip():
        raise RuntimeError("docker_inspection_failed")
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def _owned_container_names(pid: int, names: Sequence[str]) -> tuple[str, ...]:
    return tuple(name for name in names
                 if (match := CONTAINER_RE.fullmatch(name)) and int(match.group(1)) == pid)


def _monitor_lock_clear() -> bool:
    path = ROOT / ".runtime/six-live-preview.lock"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        raw = os.read(fd, 4096)
        if not raw.strip():
            return True
        try:
            owner = json.loads(raw)
        except json.JSONDecodeError:
            return False
        if owner.get("cleanup_ok") is True:
            return True
        return False
    finally:
        os.close(fd)


def _active_unit(timeout: float = 4) -> bool:
    result = call(["systemctl", "--user", "is-active", UNIT], timeout)
    return result.returncode == 0 and result.stdout.strip() in {"active", "activating", "reloading"}


def _verify_candidate_file(path: Path, expected: str, *, unavailable: str,
                           mismatch: str) -> None:
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(unavailable) from exc
    if actual != expected:
        raise RuntimeError(mismatch)


def _verify_runtime_image_pin() -> None:
    """Keep this diagnostic on the existing digest-pinned DS 9.1 runtime."""
    try:
        platform = json.loads((ROOT / "config/deepstream-platform.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("runtime_image_configuration_unavailable") from exc
    if not isinstance(platform, dict) or platform.get("image") != IMAGE:
        raise RuntimeError("runtime_image_digest_mismatch")


def preflight(*, worker_mode: bool = False, deadline: float | None = None) -> dict[str, str]:
    """Complete every mandatory check before a camera-capable child is created."""
    deadline = deadline if deadline is not None else time.monotonic() + PREFLIGHT_BUDGET
    # 1. Candidate identities and runtime pin, before any host resource checks.
    _verify_candidate_file(CANDIDATE, EXPECTED, unavailable="candidate_unavailable",
                          mismatch="candidate_hash_mismatch")
    _verify_candidate_file(
        CANDIDATE_LAUNCHER, CANDIDATE_LAUNCHER_EXPECTED,
        unavailable="candidate_launcher_unavailable",
        mismatch="candidate_launcher_hash_mismatch")
    _verify_runtime_image_pin()

    # 2. X11, Docker, GPU and current resource ownership.
    session_ok, display, xauthority = session(_remaining_timeout(deadline, 5))
    if not session_ok:
        raise RuntimeError("authenticated_x11_unavailable")
    names = _container_names(_remaining_timeout(deadline, 7))
    gpu = gpu_snapshot(_remaining_timeout(deadline, 5))
    if gpu is None or gpu.get("vram_free_mib", 0) < GPU_MIN_FREE_MIB:
        raise RuntimeError("gpu_unavailable_or_insufficient_free_vram")
    if not _camera_locks_free():
        raise RuntimeError("camera_source_lock_owned")
    ipc = _ipc_clear()
    if not all(ipc.values()):
        raise RuntimeError("preview_or_telemetry_ipc_owned_or_stale")
    try:
        rtsp = _rtsp_rows(_remaining_timeout(deadline, 4))
    except RuntimeError as exc:
        raise RuntimeError("rtsp_session_inspection_failed") from exc
    if rtsp:
        raise RuntimeError("existing_rtsp_sessions")

    # 3. Both services must answer live, structured HTTP health checks.
    api = health(8100, _remaining_timeout(deadline, 3))
    ml = health(8001, _remaining_timeout(deadline, 3))
    if not api.healthy or not ml.healthy:
        raise RuntimeError("api_or_ml_health_unavailable")
    if api.body.get("ml_accessible") is not True:
        raise RuntimeError("api_cannot_reach_ml_service")

    # 4. Reject duplicates/orphans only after the non-camera health checks.
    if not _monitor_lock_clear():
        raise RuntimeError("monitor_lock_owned_or_cleanup_unverified")
    if any(CONTAINER_RE.fullmatch(name) for name in names):
        raise RuntimeError("existing_orphan_six_camera_container")
    if not worker_mode and _active_unit(_remaining_timeout(deadline, 4)):
        raise RuntimeError("diagnostic_unit_already_active")
    if not worker_mode:
        manager = call(["systemctl", "--user", "show-environment"], _remaining_timeout(deadline, 5))
        if manager.returncode != 0:
            raise RuntimeError("systemd_user_manager_unavailable")
    if time.monotonic() >= deadline:
        raise RuntimeError("preflight_budget_exceeded")
    return {"display": display, "xauthority": xauthority}


def launcher_cmd() -> list[str]:
    return [
        str(ROOT / ".runtime/full-stack-venv/bin/python"), "-B",
        str(CANDIDATE_LAUNCHER), "--run", "--monitor",
        "--authorize-physical-rtsp", "--network", "bridge",
        "--infer-config", str(ROOT / ".runtime/cam-all-live-fontfix-20261026-r1/infer_config.txt"),
        "--models-root", str(ROOT / ".runtime/six-source-recorded-validation-20261024-r1"),
        "--expected-engine-sha256", ENGINE, "--mosaic-preview", "--mosaic-viewer-binary",
        str(CANDIDATE), "--publish-ml-telemetry", "--source-stall-diagnostics",
        "--upstream-progress-diagnostics",
    ]


def verify_project_imports() -> None:
    """Import-only diagnostic used by CPU tests; it never starts a pipeline."""
    from services.gpu.runtime import live_launcher  # noqa: F401
    from scripts.six_camera_live.rtsp_session_preflight import (  # noqa: F401
        inspect_established_rtsp_sessions,
    )


def _proc_identity(pid: int) -> tuple[str | None, tuple[str, ...] | None]:
    try:
        proc = Path(f"/proc/{pid}")
        if proc.stat().st_uid != os.getuid():
            return None, None
        stat_fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if stat_fields[0] == "Z":
            return None, None
        start_ticks = stat_fields[19]
        argv = tuple((proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0"))
        return start_ticks, argv
    except (OSError, IndexError, UnicodeDecodeError):
        return None, None


def _popen_identity(child: subprocess.Popen, expected_argv: Sequence[str]) -> dict | None:
    if child.poll() is not None or child.args != list(expected_argv):
        return None
    start_ticks, actual_argv = _proc_identity(child.pid)
    if start_ticks is None or actual_argv is None or tuple(expected_argv) != actual_argv:
        return None
    return {"pid": child.pid, "start_ticks": start_ticks, "argv": list(expected_argv)}


def _owned_lock_matches(pid: int, start_ticks: str) -> bool:
    path = ROOT / ".runtime/six-live-preview.lock"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            return False
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            return False
        owner = json.loads(os.read(fd, 4096))
        return (owner.get("mode") == "daily" and owner.get("pid") == pid
                and owner.get("start_ticks") == start_ticks
                and _proc_identity(pid)[0] == start_ticks)
    except (OSError, ValueError, TypeError):
        return False
    finally:
        os.close(fd)


def _send_verified_term(identity: dict) -> bool:
    pid = identity.get("pid")
    start_ticks = identity.get("start_ticks")
    argv = identity.get("argv")
    if (not isinstance(pid, int) or pid <= 1 or not isinstance(start_ticks, str)
            or not isinstance(argv, list)):
        return False
    actual_ticks, actual_argv = _proc_identity(pid)
    if actual_ticks != start_ticks or actual_argv != tuple(argv):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return True
    except OSError:
        return False


@dataclass(frozen=True)
class StopResult:
    exit_code: int
    method: str


def stop_owned(report: Path, reason: str, child: subprocess.Popen | None = None,
               identity: dict | None = None) -> StopResult:
    """Stop only the Popen child we created or the exact persisted child identity."""
    report_ok = True
    try:
        put(report, {"type": "stop_requested", "utc": stamp(), "reason": reason})
    except OSError:
        # A report I/O failure must not prevent bounded cleanup.
        report_ok = False
    if child is not None:
        if child.poll() is not None:
            return StopResult(0, "child_already_exited")
        if identity is None or child.pid != identity.get("pid"):
            return StopResult(2, "owned_child_identity_missing")
        if (child.args != identity.get("argv")
                or _proc_identity(child.pid) != (identity.get("start_ticks"), tuple(identity.get("argv", [])))):
            return StopResult(2, "owned_child_identity_mismatch")
        if _owned_lock_matches(child.pid, identity["start_ticks"]):
            command = [str(ROOT / ".runtime/full-stack-venv/bin/python"), "-B", "-m",
                       "services.gpu.runtime.live_launcher", "--stop-monitoring"]
            try:
                result = call(command, STOP_COMMAND_TIMEOUT)
                stop = StopResult(result.returncode, "verified_monitor_lock")
            except (OSError, subprocess.TimeoutExpired):
                stop = StopResult(124, "verified_monitor_lock_stop_error")
            if child.poll() is None and stop.exit_code != 0:
                try:
                    child.send_signal(signal.SIGTERM)
                    stop = StopResult(stop.exit_code, "verified_child_sigterm_fallback")
                except OSError:
                    pass
        else:
            try:
                child.send_signal(signal.SIGTERM)
                stop = StopResult(0, "exact_popen_child_sigterm")
            except OSError:
                stop = StopResult(2, "exact_popen_child_signal_failed")
    elif identity is not None:
        pid = identity.get("pid")
        start_ticks = identity.get("start_ticks")
        if isinstance(pid, int) and isinstance(start_ticks, str) and _owned_lock_matches(pid, start_ticks):
            command = [str(ROOT / ".runtime/full-stack-venv/bin/python"), "-B", "-m",
                       "services.gpu.runtime.live_launcher", "--stop-monitoring"]
            try:
                result = call(command, STOP_COMMAND_TIMEOUT)
                stop = StopResult(result.returncode, "verified_monitor_lock_execstoppost")
            except (OSError, subprocess.TimeoutExpired):
                stop = StopResult(124, "execstoppost_stop_error")
        elif _send_verified_term(identity):
            stop = StopResult(0, "persisted_exact_child_sigterm")
        else:
            stop = StopResult(2, "persisted_child_identity_unverified")
    else:
        stop = StopResult(2, "no_owned_process_identity")
    try:
        put(report, {"type": "launcher_stop", "utc": stamp(), "exit_code": stop.exit_code,
                     "method": stop.method})
    except OSError:
        report_ok = False
    if not report_ok:
        return StopResult(stop.exit_code or 2, stop.method + "_report_write_failed")
    return stop


def _cleanup_checks(child_pid: int | None, child_rc: int | None) -> dict[str, bool]:
    if child_pid is None:
        # This run created nothing; avoid asserting global locks/containers are
        # clear when preflight may have failed because another owner exists.
        return {"launcher_not_started": True, "owned_resources_not_created": True}
    checks = {"launcher_reaped": child_rc is not None}
    try:
        names = _container_names()
        owned = _owned_container_names(child_pid, names)
        checks["owned_container_absent"] = not owned
        checks["owned_rtsp_sessions_closed"] = not owned
    except Exception:
        checks["owned_container_absent"] = False
        checks["owned_rtsp_sessions_closed"] = False
    try:
        checks["camera_source_locks_free"] = _camera_locks_free()
        ipc = _ipc_clear()
        checks["preview_ipc_clear"] = all(
            value for key, value in ipc.items() if "preview" in key)
        checks["telemetry_ipc_clear"] = all(
            value for key, value in ipc.items() if "telemetry" in key or "tracked_object" in key)
    except Exception:
        checks.update({"camera_source_locks_free": False, "preview_ipc_clear": False,
                       "telemetry_ipc_clear": False})
    return checks


def _last(log_path: Path, pattern: str) -> str | None:
    try:
        matches = re.findall(pattern, log_path.read_text(errors="replace"))
    except OSError:
        return None
    return matches[-1] if matches else None


class NativeTimeline:
    """Incrementally interpret credential-free lifecycle markers from launcher.log.

    Native monotonic values are emitted by the viewer using CLOCK_MONOTONIC;
    Docker uses the host time namespace for this run. Supervisor elapsed values
    use the worker's CLOCK_MONOTONIC origin. Native-viewer elapsed values stay
    unset until the native pipeline start marker has actually been observed.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.offset = 0
        self.pending = b""
        self.native_start_ns: int | None = None
        self.playing_ns: int | None = None
        self.pipeline_stopped_ns: int | None = None
        self.native_exit_ns: int | None = None
        self.native_signal_received_ns: int | None = None
        self.null_state_native_ns: int | None = None
        self.native_exit_code: int | None = None
        self.null_state_completed = False
        self.null_state_evidence: dict | None = None
        self.docker_run_exit_code: int | None = None
        self.container_exit_code: int | None = None
        self.container_state: dict | None = None
        self.cleanup_ok: bool | None = None
        self.failure_stage: str | None = None
        self.failure: str | None = None
        self.failure_diagnostic_path: str | None = None
        self.source_progress_at_stop: list[dict] | None = None
        self.upstream_progress_at_stop: list[dict] | None = None
        self.upstream_progress_enabled = False
        self.upstream_progress_samples_observed: int | None = None
        self.upstream_progress_samples_dropped: int | None = None
        self.upstream_samples_with_native_envelope: int | None = None
        self.upstream_source_events: list[dict] | None = None
        self.upstream_source_events_observed: int | None = None
        self.upstream_source_events_dropped: int | None = None
        self.upstream_diagnostics_report_path: str | None = None
        self.upstream_diagnostics_collected_monotonic_ns: int | None = None
        self.camera_wall_closed = False
        self.live_window_complete = False
        self.watchdog_marker: str | None = None
        self.shutdown_requested_ns: int | None = None
        self._crossed_800 = False
        self._events: list[dict] = []

    def _event(self, name: str, *, native_ns: int | None = None,
               details: dict | None = None, observed_ns: int | None = None) -> None:
        observed_ns = time.monotonic_ns() if observed_ns is None else observed_ns
        elapsed = (None if self.native_start_ns is None or native_ns is None else
                   max(0.0, (native_ns - self.native_start_ns) / 1_000_000_000))
        self._events.append({
            "type": "milestone", "milestone": name, "utc": stamp(),
            "observed_monotonic_ns": observed_ns,
            "native_monotonic_ns": native_ns,
            "elapsed_seconds": elapsed,
            "details": details or {},
        })

    @staticmethod
    def _json_after_prefix(line: str, prefix: str) -> dict | None:
        if not line.startswith(prefix):
            return None
        try:
            value = json.loads(line[len(prefix):].strip())
        except (json.JSONDecodeError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _fields(line: str) -> dict[str, str]:
        return dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line))

    def _failure_details(self, raw_path: str) -> None:
        data = self._read_private_runtime_json(raw_path)
        if data is None:
            return
        stage = data.get("failure_stage")
        failure = data.get("failure")
        if isinstance(stage, str) and isinstance(failure, str):
            self.failure_stage = stage[:120]
            self.failure = failure[:500]
            self.failure_diagnostic_path = str(Path(raw_path.strip()).resolve())
        self._capture_progress_diagnostics(data, raw_path)

    @staticmethod
    def _read_private_runtime_json(raw_path: str) -> dict | None:
        try:
            candidate = Path(raw_path.strip()).resolve(strict=True)
            runtime_root = (ROOT / ".runtime").resolve(strict=True)
            info = candidate.stat()
            if (not candidate.is_relative_to(runtime_root) or not stat.S_ISREG(info.st_mode)
                    or info.st_size > 16 * 1024 * 1024 or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) & 0o077):
                return None
            data = json.loads(candidate.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _capture_progress_diagnostics(self, data: dict, raw_path: str) -> None:
        """Copy only bounded, validated counters from the launcher's private report."""
        raw_progress = data.get("source_progress_messages")
        nested_diagnostics = data.get("diagnostics")
        if not isinstance(raw_progress, list) and isinstance(nested_diagnostics, dict):
            raw_progress = nested_diagnostics.get("source_progress_messages")
        if isinstance(raw_progress, list):
            self.source_progress_at_stop = self._summarize_source_progress(raw_progress)
        source_diagnostics = data.get("source_stall_diagnostics")
        if not isinstance(source_diagnostics, dict):
            nested = data.get("diagnostics")
            source_diagnostics = (nested.get("source_stall_diagnostics")
                                  if isinstance(nested, dict) else None)
        if not isinstance(source_diagnostics, dict):
            return
        self.upstream_progress_enabled = (
            source_diagnostics.get("upstream_progress_enabled") is True)
        observed = source_diagnostics.get("progress_samples_observed")
        dropped = source_diagnostics.get("progress_samples_dropped")
        self.upstream_progress_samples_observed = (
            observed if isinstance(observed, int) and not isinstance(observed, bool)
            and observed >= 0 else None)
        self.upstream_progress_samples_dropped = (
            dropped if isinstance(dropped, int) and not isinstance(dropped, bool)
            and dropped >= 0 else None)
        events = source_diagnostics.get("events")
        events_observed = source_diagnostics.get("events_observed")
        events_dropped = source_diagnostics.get("events_dropped")
        self.upstream_source_events_observed = (
            events_observed if isinstance(events_observed, int)
            and not isinstance(events_observed, bool) and events_observed >= 0 else None)
        self.upstream_source_events_dropped = (
            events_dropped if isinstance(events_dropped, int)
            and not isinstance(events_dropped, bool) and events_dropped >= 0 else None)
        self.upstream_source_events = self._summarize_source_events(events)
        self.upstream_diagnostics_collected_monotonic_ns = time.monotonic_ns()
        self.upstream_diagnostics_report_path = str(Path(raw_path.strip()).resolve())
        raw_samples = source_diagnostics.get("progress_samples")
        self.upstream_samples_with_native_envelope = (
            sum(1 for row in raw_samples if isinstance(row, dict)
                and isinstance(row.get("upstream_diagnostics"), dict)
                and row["upstream_diagnostics"].get("enabled") is True)
            if isinstance(raw_samples, list) else None)
        self.upstream_progress_at_stop = self._summarize_upstream_progress(
            raw_samples,
            sample_capacity_drops=self.upstream_progress_samples_dropped,
            cutoff_native_ns=self.shutdown_requested_ns,
            collected_monotonic_ns=self.upstream_diagnostics_collected_monotonic_ns)

    @staticmethod
    def _summarize_source_events(events: object) -> list[dict] | None:
        if not isinstance(events, list):
            return None
        allowed_events = {
            "source_online", "source_reconnecting", "source_offline",
            "source_warning", "source_error", "source_eos", "source_timeout",
            "source_reconnect_attempt_exceeded", "gstreamer_warning",
            "gstreamer_error", "gstreamer_critical", "gstreamer_fatal",
        }
        result = []
        for event in events[-128:]:
            if not isinstance(event, dict) or event.get("event") not in allowed_events:
                continue
            sid = event.get("source_id")
            generation = event.get("source_generation")
            native_ns = event.get("native_monotonic_ns")
            receipt_ns = event.get("host_receipt_monotonic_ns")
            if not isinstance(sid, int) or isinstance(sid, bool) or not 0 <= sid < 6:
                continue
            if (generation is not None and
                    (not isinstance(generation, int) or isinstance(generation, bool)
                     or generation < 0)):
                continue
            if (native_ns is not None and
                    (not isinstance(native_ns, int) or isinstance(native_ns, bool)
                     or native_ns < 0)):
                continue
            if not isinstance(receipt_ns, int) or isinstance(receipt_ns, bool) or receipt_ns <= 0:
                continue
            severity = event.get("severity")
            if severity not in (None, "warning", "error", "critical", "fatal"):
                severity = None
            condition = event.get("condition")
            if condition not in (None, "attempts_exceeded", "gstreamer_timeout", "timeout",
                                 "end_of_stream"):
                condition = None
            result.append({
                "event": event["event"], "source_id": sid,
                "camera_id": f"CAM-{sid + 1:02d}",
                "source_generation": generation, "severity": severity,
                "condition": condition, "native_monotonic_ns": native_ns,
                "host_receipt_monotonic_ns": receipt_ns,
            })
        return result

    def _rtsp_report_details(self, raw_path: str) -> None:
        data = self._read_private_runtime_json(raw_path)
        if data is None:
            return
        self._capture_progress_diagnostics(data, raw_path)

    @staticmethod
    def _summarize_upstream_progress(samples: object, *,
                                     sample_capacity_drops: int | None,
                                     cutoff_native_ns: int | None,
                                     collected_monotonic_ns: int) -> list[dict] | None:
        if not isinstance(samples, list):
            return None
        by_source: dict[int, list[dict]] = {sid: [] for sid in range(6)}
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            sid = sample.get("source_id")
            generation = sample.get("source_generation")
            receipt_ns = sample.get("host_receipt_monotonic_ns")
            native_ns = sample.get("native_monotonic_ns")
            camera_id = f"CAM-{sid + 1:02d}" if isinstance(sid, int) and 0 <= sid < 6 else None
            if (not isinstance(sid, int) or isinstance(sid, bool) or not 0 <= sid < 6
                    or sample.get("camera_id") != camera_id
                    or not isinstance(generation, int) or isinstance(generation, bool)
                    or generation < 0
                    or not isinstance(receipt_ns, int) or isinstance(receipt_ns, bool)
                    or receipt_ns <= 0
                    or not isinstance(native_ns, int) or isinstance(native_ns, bool)
                    or native_ns < 0
                    or (cutoff_native_ns is not None and native_ns > cutoff_native_ns)):
                continue
            by_source[sid].append(sample)

        summary: list[dict] = []
        for sid in range(6):
            rows = sorted(by_source[sid],
                          key=lambda row: row["host_receipt_monotonic_ns"])
            if not rows:
                summary.append({"source_id": sid, "camera_id": f"CAM-{sid + 1:02d}",
                                "sample_count": 0, "latest_sample": None,
                                "status": "missing"})
                continue
            current = rows[-1]
            previous = rows[-2] if len(rows) > 1 else None
            receipt_ns = current["host_receipt_monotonic_ns"]
            age_ns = collected_monotonic_ns - receipt_ns
            age_seconds = age_ns / 1e9 if age_ns >= 0 else None
            sample_fresh = age_seconds is not None and age_seconds <= 15.0
            generation = current["source_generation"]
            upstream = current.get("upstream_diagnostics")
            native_envelope_enabled = (isinstance(upstream, dict)
                                       and upstream.get("enabled") is True)
            stage_map = upstream.get("stages") if isinstance(upstream, dict) else None
            native_sample_ns = (upstream.get("sample_monotonic_ns")
                                if isinstance(upstream, dict) else None)
            native_sample_valid = (isinstance(native_sample_ns, int)
                                   and not isinstance(native_sample_ns, bool)
                                   and native_sample_ns >= 0)
            stages: dict[str, dict] = {}
            prior_upstream = previous.get("upstream_diagnostics") if previous else None
            prior_stages = (prior_upstream.get("stages")
                            if isinstance(prior_upstream, dict) else None)
            for name in ("rtp_pad", "compressed_pre_decode"):
                raw_stage = stage_map.get(name) if isinstance(stage_map, dict) else None
                attached = isinstance(raw_stage, dict) and raw_stage.get("probe_attached") is True
                valid = (attached and raw_stage.get("valid") is True
                         and raw_stage.get("source_generation") == generation
                         and native_sample_valid)
                buffers = raw_stage.get("buffers") if isinstance(raw_stage, dict) else None
                last_ns = (raw_stage.get("last_progress_monotonic_ns")
                           if isinstance(raw_stage, dict) else None)
                buffers_valid = (isinstance(buffers, int) and not isinstance(buffers, bool)
                                 and buffers >= 0)
                if buffers_valid and buffers == 0:
                    last_valid = last_ns is None
                else:
                    last_valid = (isinstance(last_ns, int) and not isinstance(last_ns, bool)
                                  and last_ns > 0 and native_sample_valid
                                  and last_ns <= native_sample_ns)
                valid = bool(valid and buffers_valid and last_valid)
                prior_stage = (prior_stages.get(name)
                               if isinstance(prior_stages, dict) else None)
                prior_count = (prior_stage.get("buffers")
                               if isinstance(prior_stage, dict) else None)
                delta = None
                if (valid and isinstance(prior_stage, dict)
                        and prior_stage.get("valid") is True
                        and prior_stage.get("probe_attached") is True
                        and prior_stage.get("source_generation") == generation
                        and isinstance(prior_count, int) and not isinstance(prior_count, bool)
                        and prior_count >= 0 and buffers >= prior_count):
                    delta = buffers - prior_count
                progress_age = ((native_sample_ns - last_ns) / 1e9
                                if valid and buffers > 0 else None)
                if not isinstance(raw_stage, dict) or not attached:
                    status = ("upstream_envelope_missing_or_disabled"
                              if not native_envelope_enabled else "missing_probe")
                elif not valid:
                    status = "invalid_or_generation_mismatch"
                elif not sample_fresh:
                    status = "stale_sample"
                elif buffers == 0:
                    status = "zero_progress_observed"
                elif progress_age is not None and progress_age > 15.0:
                    status = "progress_stale"
                else:
                    status = "progress_observed"
                stages[name] = {
                    "probe_attached": attached, "valid": valid,
                    "source_generation": (raw_stage.get("source_generation")
                                          if valid else None),
                    "buffers": buffers if valid else None,
                    "last_progress_monotonic_ns": last_ns if valid else None,
                    "delta_same_generation": delta,
                    "last_progress_age_seconds": progress_age,
                    "status": status,
                }
            sample_count = len(rows)
            summary.append({
                "source_id": sid, "camera_id": f"CAM-{sid + 1:02d}",
                "sample_count": sample_count,
                "sample_capacity_drops": sample_capacity_drops,
                "source_generation": generation,
                "native_monotonic_ns": current["native_monotonic_ns"],
                "host_receipt_monotonic_ns": receipt_ns,
                "latest_sample_age_seconds": age_seconds,
                "sample_fresh_within_15_seconds": sample_fresh,
                "native_upstream_envelope_enabled": native_envelope_enabled,
                "stages": stages,
                "status": "stale_sample" if not sample_fresh else "observed",
            })
        return summary

    def _summarize_source_progress(self, messages: object) -> list[dict] | None:
        if not isinstance(messages, list):
            return None
        by_source: dict[int, list[dict]] = {}
        cutoff = self.shutdown_requested_ns
        for message in messages:
            if not isinstance(message, str) or not message.startswith("SIX_PROGRESS "):
                continue
            try:
                sample = json.loads(message.partition(" ")[2])
            except json.JSONDecodeError:
                continue
            sid = sample.get("source_id")
            event_ns = sample.get("monotonic_ns")
            stages = sample.get("stages")
            if (not isinstance(sid, int) or isinstance(sid, bool) or not 0 <= sid < 6
                    or not isinstance(event_ns, int) or not isinstance(stages, list)
                    or len(stages) < 4 or (cutoff is not None and event_ns > cutoff)):
                continue
            by_source.setdefault(sid, []).append(sample)
        summary = []
        for sid in range(6):
            samples = sorted(by_source.get(sid, []), key=lambda row: row["monotonic_ns"])
            if not samples:
                continue
            current = samples[-1]
            previous = samples[-2] if len(samples) > 1 else None
            decode, tracker = current["stages"][0], current["stages"][3]
            old_decode = previous["stages"][0] if previous else {}
            old_tracker = previous["stages"][3] if previous else {}
            decode_delta = (decode.get("count", 0) - old_decode.get("count", 0)
                            if previous else None)
            tracker_delta = (tracker.get("count", 0) - old_tracker.get("count", 0)
                             if previous else None)
            event_ns = current["monotonic_ns"]
            summary.append({
                "source_id": sid, "camera_id": current.get("camera_id"),
                "event_monotonic_ns": event_ns,
                "decode_count": decode.get("count"),
                "tracker_count": tracker.get("count"),
                "decode_delta_since_previous_sample": decode_delta,
                "tracker_delta_since_previous_sample": tracker_delta,
                "decode_stage_age_seconds": ((event_ns - decode.get("last_ns")) / 1e9
                                              if isinstance(decode.get("last_ns"), int) else None),
                "tracker_stage_age_seconds": ((event_ns - tracker.get("last_ns")) / 1e9
                                               if isinstance(tracker.get("last_ns"), int) else None),
            })
        return summary

    def _consume(self, line: str, observed_ns: int) -> None:
        if not line:
            return
        state = self._json_after_prefix(line, "SIX_PIPELINE ")
        if state is not None:
            native_ns = state.get("monotonic_ns")
            if not isinstance(native_ns, int) or isinstance(native_ns, bool):
                native_ns = None
            status = state.get("status")
            if status == "starting" and self.native_start_ns is None and native_ns is not None:
                self.native_start_ns = native_ns
                self._event("native_viewer_started", native_ns=native_ns,
                            observed_ns=observed_ns)
            elif status == "playing" and native_ns is not None:
                if self.playing_ns is None:
                    self.playing_ns = native_ns
                    self._event("viewer_reached_playing", native_ns=native_ns,
                                observed_ns=observed_ns)
                if (not self._crossed_800 and self.native_start_ns is not None
                        and native_ns - self.native_start_ns >= 800_000_000_000):
                    self._crossed_800 = True
                    self._event("native_800_second_boundary_crossed", native_ns=native_ns,
                                details={"threshold_seconds": 800}, observed_ns=observed_ns)
            elif status == "stopped" and native_ns is not None:
                self.pipeline_stopped_ns = native_ns
                self._event("native_pipeline_stopped", native_ns=native_ns,
                            observed_ns=observed_ns)
            return

        if "DOCKER_SIGNAL event=stop_request_started" in line:
            fields = self._fields(line)
            self.shutdown_requested_ns = self._int_field(fields, "monotonic_ns")
            self._event("shutdown_initiated", native_ns=self.shutdown_requested_ns,
                        details={"signal": fields.get("signal"),
                                 "initiator": "live_launcher"}, observed_ns=observed_ns)
        elif "SIX_SHUTDOWN event=sigint_received" in line:
            fields = self._fields(line)
            self._event("native_sigint_received", details={"signal": fields.get("signal")},
                        observed_ns=observed_ns)
        elif line.startswith("SIX_RTSP_SHUTDOWN event=counter_snapshot "):
            fields = self._fields(line)
            micros = self._int_field(fields, "monotonic_us")
            native_ns = micros * 1000 if micros is not None else None
            if fields.get("phase") == "shutdown_intent" and native_ns is not None:
                self.native_signal_received_ns = native_ns
                self._event("native_shutdown_intent_timestamped", native_ns=native_ns,
                            observed_ns=observed_ns)
            elif fields.get("phase") == "gst_state_null_returned" and native_ns is not None:
                self.null_state_native_ns = native_ns
                self._event("gst_state_null_return_timestamped", native_ns=native_ns,
                            details={"null_state_completed": self.null_state_completed},
                            observed_ns=observed_ns)
        elif "SIX_SHUTDOWN event=pipeline_null_observed" in line:
            fields = self._fields(line)
            evidence = {key: fields.get(key) for key in
                        ("set_result", "query_result", "current_state", "pending_state")}
            self.null_state_completed = (
                fields.get("set_result") == "1" and fields.get("query_result") == "1"
                and fields.get("current_state") == "NULL"
                and fields.get("pending_state") == "VOID_PENDING")
            self.null_state_evidence = evidence
            self._event("graceful_null_state_verified" if self.null_state_completed else
                        "null_state_marker_incomplete", details=evidence,
                        observed_ns=observed_ns)
        elif "SIX_SHUTDOWN event=process_exit_initiated" in line:
            fields = self._fields(line)
            self.native_exit_code = self._int_field(fields, "code")
            self.native_exit_ns = self._int_field(fields, "monotonic_ns")
            if (not self._crossed_800 and self.native_start_ns is not None
                    and self.native_exit_ns is not None
                    and self.native_exit_ns - self.native_start_ns >= 800_000_000_000):
                self._crossed_800 = True
                self._event("native_800_second_boundary_crossed", native_ns=self.native_exit_ns,
                            details={"threshold_seconds": 800,
                                     "observed_at": "native_process_exit_marker"},
                            observed_ns=observed_ns)
            self._event("native_process_exit_initiated", native_ns=self.native_exit_ns,
                        details={"exit_code": self.native_exit_code}, observed_ns=observed_ns)
        elif line.startswith("DOCKER_RUN_EXIT ") or line.startswith("DOCKER_RUN_EXIT_AFTER_CLEANUP "):
            fields = self._fields(line)
            self.docker_run_exit_code = self._int_field(fields, "code")
            self._event("docker_run_exited", details={"exit_code": self.docker_run_exit_code},
                        observed_ns=observed_ns)
        elif line.startswith("DOCKER_STATE "):
            state = self._json_after_prefix(line, "DOCKER_STATE ")
            if state is not None:
                self.container_state = {key: state.get(key) for key in
                                        ("Status", "Running", "OOMKilled", "ExitCode", "Dead",
                                         "StartedAt", "FinishedAt")}
                if state.get("Status") == "exited" and state.get("Running") is False:
                    value = state.get("ExitCode")
                    if isinstance(value, int) and not isinstance(value, bool):
                        self.container_exit_code = value
                self._event("container_exited", details=self.container_state,
                            observed_ns=observed_ns)
        elif line.startswith("CLEANUP_PREDICATES "):
            record = self._json_after_prefix(line, "CLEANUP_PREDICATES ")
            if record is not None and isinstance(record.get("cleanup_ok"), bool):
                self.cleanup_ok = record["cleanup_ok"]
                self._event("exact_owned_cleanup_verified" if self.cleanup_ok else
                            "exact_owned_cleanup_failed", details={"cleanup_ok": self.cleanup_ok},
                            observed_ns=observed_ns)
        elif line.startswith("LIVE FAILURE DIAGNOSTICS: "):
            self._failure_details(line.split(":", 1)[1])
        elif line.startswith("SIX LIVE RTSP SHUTDOWN REPORT: "):
            self._rtsp_report_details(line.partition(":")[2])
        elif "Camera Wall closed; stopping monitoring." in line:
            self.camera_wall_closed = True
        elif "LIVE WINDOW COMPLETE; using reserved shutdown budget." in line:
            self.live_window_complete = True
        elif line.startswith("CONTAINER_WATCHDOG "):
            self.watchdog_marker = line[:500]

    @staticmethod
    def _int_field(fields: dict[str, str], key: str) -> int | None:
        try:
            return int(fields[key])
        except (KeyError, ValueError):
            return None

    def poll(self) -> list[dict]:
        try:
            with self.log_path.open("rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                if size < self.offset:
                    self.offset = 0
                    self.pending = b""
                stream.seek(self.offset)
                chunk = stream.read()
                self.offset = stream.tell()
        except FileNotFoundError:
            return []
        data = self.pending + chunk
        lines = data.split(b"\n")
        self.pending = lines.pop()[-1_000_000:]
        observed_ns = time.monotonic_ns()
        for raw in lines:
            self._consume(raw[:1_000_000].decode("utf-8", errors="replace"), observed_ns)
        events, self._events = self._events, []
        return events

    def final_reason(self, return_code: int | None, signal_stop_reason: str | None) -> tuple[str, str]:
        if signal_stop_reason:
            return signal_stop_reason, "worker_signal"
        if return_code not in (None, 0):
            if self.failure and "source FPS streams stalled" in self.failure:
                return "source_progress_failure", "launcher_nonzero_exit"
            return "launcher_failure", "launcher_nonzero_exit"
        if self.live_window_complete:
            return "deadline", "planned_live_window_completion"
        if self.camera_wall_closed:
            return "camera_wall_closed", "camera_wall_exit"
        return "monitor_exit", "zero_exit_before_planned_completion"

    def native_pipeline_uptime_seconds(self) -> float | None:
        end_ns = self.pipeline_stopped_ns or self.native_exit_ns
        if self.native_start_ns is None or end_ns is None:
            return None
        if end_ns < self.native_start_ns:
            return None
        return (end_ns - self.native_start_ns) / 1_000_000_000

    def planned_observation_complete(self, now_ns: int | None = None) -> bool:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        return (
            self.native_start_ns is not None
            and self.playing_ns is not None
            and self.native_start_ns <= self.playing_ns <= now_ns
            and now_ns - self.playing_ns >=
                REQUIRED_NATIVE_OBSERVATION_SECONDS * 1_000_000_000
        )

    def native_observation_seconds(self) -> float | None:
        """Conservative PLAYING-to-shutdown-request wall-clock span."""
        start = self.playing_ns
        end = self.shutdown_requested_ns
        if start is None or end is None or end < start:
            return None
        return (end - start) / 1_000_000_000

    def observation_met(self, required_seconds: int) -> bool:
        elapsed = self.native_observation_seconds()
        return elapsed is not None and elapsed >= required_seconds

    def crossed_800_seconds(self) -> bool | None:
        end_ns = self.pipeline_stopped_ns or self.native_exit_ns
        if self.native_start_ns is None:
            return None
        return self._crossed_800 or (end_ns is not None and
                                     end_ns - self.native_start_ns >= 800_000_000_000)


def worker(runid: str, *, preflight_fn: Callable | None = None,
           child_factory: Callable | None = None, command_factory: Callable | None = None,
           health_fn: Callable | None = None, session_fn: Callable | None = None,
           cleanup_fn: Callable | None = None, sample_interval: float = 60.0,
           poll_interval: float = 0.5, stop_event=None,
           install_signal_handlers: bool = True) -> int:
    global signal_reason
    signal_reason = None
    run_dir = RUNTIME / runid
    data = json.loads((run_dir / "state.json").read_text())
    worker_started_mono = time.monotonic()
    report = Path(data["report"])
    log_path = run_dir / "launcher.log"
    child = None
    child_identity = None
    log_stream = None
    rc = None
    stop_result = StopResult(0, "not_required")
    exception_category = None
    phase = "worker_preflight"
    reason = "worker_startup_failure"
    termination_source = "worker_startup"
    timeline = NativeTimeline(log_path)
    preflight_fn = preflight_fn or (lambda: preflight(
        worker_mode=True, deadline=time.monotonic() + PREFLIGHT_BUDGET))
    child_factory = child_factory or subprocess.Popen
    command_factory = command_factory or launcher_cmd
    health_fn = health_fn or health
    session_fn = session_fn or session
    cleanup_fn = cleanup_fn or _cleanup_checks

    def onterm(sig, frame):
        global signal_reason
        del frame
        signal_reason = "deadline" if (sig == signal.SIGTERM and
                          time.monotonic() - float(data["start_mono"]) >= PRE - 2) else "manual_or_systemd_stop"

    previous_term = previous_int = None
    if install_signal_handlers:
        previous_term = signal.signal(signal.SIGTERM, onterm)
        previous_int = signal.signal(signal.SIGINT, onterm)
    try:
        # Repeat all checks immediately before creating the camera-capable child.
        preflight_data = preflight_fn()
        if preflight_data:
            data.update({"display": preflight_data["display"], "xauthority": preflight_data["xauthority"]})
        if signal_reason:
            reason = "cancelled_before_launcher"
            raise RuntimeError("stop_received_before_launcher")
        cmd = list(command_factory())
        env = os.environ.copy()
        env["DISPLAY"] = data["display"]
        env["XAUTHORITY"] = data["xauthority"]
        env["SIX_LIVE_SUPPRESS_SHUTDOWN_PAUSE"] = "1"
        env["SIX_LIVE_PROJECT_ROOT"] = str(ROOT)
        env_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ROOT_TEXT + (os.pathsep + env_pythonpath
                                         if env_pythonpath and ROOT_TEXT not in env_pythonpath.split(os.pathsep)
                                         else "")
        phase = "launcher_spawn"
        log_stream = open(log_path, "ab", buffering=0)
        child = child_factory(cmd, cwd=ROOT, env=env, stdout=log_stream, stderr=subprocess.STDOUT,
                              start_new_session=False)
        child_identity = _popen_identity(child, cmd)
        if child_identity is None:
            # Popen owns this exact child. Persist identity failure and do not mistake it for a valid launcher.
            put(report, {"type": "launcher_identity_error", "utc": stamp(), "pid": child.pid})
            raise RuntimeError("spawned_launcher_identity_unverified")
        data["child_identity"] = child_identity
        state_tmp = run_dir / ".state.json.tmp"
        state_tmp.write_text(json.dumps(data, separators=(",", ":")))
        os.chmod(state_tmp, 0o600)
        os.replace(state_tmp, run_dir / "state.json")
        put(report, {"type": "pipeline_started", "utc": stamp(), "pid": child.pid,
                     "candidate_sha256": EXPECTED,
                     "candidate_launcher_sha256": CANDIDATE_LAUNCHER_EXPECTED,
                     "source_stall_diagnostics": True,
                     "upstream_progress_diagnostics": True,
                     "native_upstream_opt_in": "SIX_LIVE_UPSTREAM_DIAGNOSTICS=1",
                     "supervisor_elapsed_seconds": time.monotonic() - float(data["start_mono"])})
        phase = "monitoring"
        reason = "monitor_exit"
        next_sample = time.monotonic()
        while True:
            rc = child.poll()
            for event in timeline.poll():
                put(report, event)
            if rc is not None:
                reason, termination_source = timeline.final_reason(rc, signal_reason)
                break
            if signal_reason:
                reason = signal_reason
                break
            if stop_event is not None and stop_event.is_set():
                reason = "manual_or_systemd_stop"
                break
            if timeline.planned_observation_complete():
                reason = "deadline"
                termination_source = "planned_native_observation_completed"
                break
            session_result = session_fn()
            if not session_result[0]:
                reason = "x11_lost"
                break
            if time.monotonic() >= next_sample:
                phase = "health_sample"
                for event in timeline.poll():
                    put(report, event)
                sample_mono_ns = time.monotonic_ns()
                api = health_fn(8100)
                ml = health_fn(8001)
                for event in timeline.poll():
                    put(report, event)
                api_body = api.body if isinstance(api, HealthResult) else None
                ml_body = ml.body if isinstance(ml, HealthResult) else None
                native_elapsed = (None if timeline.native_start_ns is None else
                                  max(0.0, (sample_mono_ns - timeline.native_start_ns) / 1e9))
                put(report, {"type": "health", "utc": stamp(),
                             "elapsed_seconds": max(0.0, time.monotonic() - worker_started_mono),
                             "elapsed_clock": "CLOCK_MONOTONIC",
                             "native_viewer_elapsed_seconds": native_elapsed,
                             "supervisor_elapsed_seconds": max(
                                 0.0, time.monotonic() - float(data["start_mono"])),
                             "api_http": api.status if isinstance(api, HealthResult) else None,
                             "api_error": api.error if isinstance(api, HealthResult) else "invalid_health_result",
                             "api_ml_accessible": api_body.get("ml_accessible") if api_body else None,
                             "ml_http": ml.status if isinstance(ml, HealthResult) else None,
                             "ml_error": ml.error if isinstance(ml, HealthResult) else "invalid_health_result",
                             "monitoring_status": ml_body.get("monitoring_status") if ml_body else None,
                             "analytics_status": ml_body.get("analytics_status") if ml_body else None,
                             "six_camera_readiness": ml_body.get("six_camera_readiness") if ml_body else None,
                             "shared_pipeline_status": ml_body.get("shared_pipeline_status") if ml_body else None,
                             "cameras": ml_body.get("cameras", []) if ml_body else [],
                             "gpu": gpu_snapshot(), "host": host_snapshot()})
                phase = "monitoring"
                next_sample = time.monotonic() + sample_interval
            time.sleep(poll_interval)
    except Exception as exc:
        exception_category = type(exc).__name__
        reason = "worker_exception"
        try:
            put(report, {"type": "worker_exception", "utc": stamp(), "phase": phase,
                         "category": exception_category})
        except Exception:
            # Keep attempting bounded shutdown even when report storage fails.
            pass
    finally:
        if child is not None:
            if child.poll() is None:
                try:
                    stop_result = stop_owned(report, reason, child, child_identity)
                except Exception as exc:
                    stop_result = StopResult(2, "stop_exception_" + type(exc).__name__)
                    exception_category = exception_category or type(exc).__name__
                try:
                    rc = child.wait(timeout=max(1, STOP - STOP_COMMAND_TIMEOUT - STOP_CLEANUP_MARGIN))
                except subprocess.TimeoutExpired:
                    rc = None
                    stop_result = StopResult(stop_result.exit_code or 124, stop_result.method + "_wait_timeout")
            elif rc is None:
                rc = child.wait()
        # Drain lifecycle markers emitted during normal teardown before
        # classifying the result. The native, Docker and launcher exit codes
        # are intentionally reported as separate values.
        try:
            for event in timeline.poll():
                put(report, event)
        except Exception:
            pass
        try:
            checks = cleanup_fn(child.pid if child is not None else None, rc)
        except Exception:
            checks = {"cleanup_verification_exception": False}
        cleanup_ok = bool(checks) and all(checks.values())
        # A child that exits on its own is not a planned completion. Surface
        # the launcher failure's real cause (when present) instead of folding
        # every return into the generic monitor_exit label.
        if reason == "monitor_exit" and rc is not None:
            reason, termination_source = timeline.final_reason(rc, signal_reason)
        if reason == "monitor_exit" and rc not in (None, 0):
            reason = "launcher_failure"
            termination_source = "launcher_nonzero_exit"
        if reason == "monitor_exit" and rc == 0:
            termination_source = "zero_exit_before_planned_completion"
        if reason == "source_progress_failure":
            termination_source = "launcher_nonzero_exit"
        successful_reason = (
            reason == "deadline"
            and termination_source == "planned_native_observation_completed"
        )
        observation_seconds = timeline.native_observation_seconds()
        observation_met = timeline.observation_met(REQUIRED_NATIVE_OBSERVATION_SECONDS)
        success = (successful_reason and observation_met
                   and exception_category is None and stop_result.exit_code == 0
                   and rc == 0 and cleanup_ok)
        if termination_source != "worker_startup":
            try:
                elapsed_at_exit = (None if timeline.native_start_ns is None else
                                   max(0.0, (time.monotonic_ns() -
                                             timeline.native_start_ns) / 1e9))
                put(report, {"type": "milestone", "milestone": "shutdown_reason_classified",
                             "utc": stamp(), "observed_monotonic_ns": time.monotonic_ns(),
                             "elapsed_seconds": elapsed_at_exit,
                             "details": {"reason": reason,
                                         "termination_source": termination_source,
                                         "launcher_failure_stage": timeline.failure_stage}})
                put(report, {"type": "milestone", "milestone": "launcher_exited",
                             "utc": stamp(), "observed_monotonic_ns": time.monotonic_ns(),
                             "elapsed_seconds": elapsed_at_exit,
                             "details": {"exit_code": rc}})
            except OSError:
                pass
        final = {"type": "final", "utc": stamp(), "reason": reason, "phase": phase,
                 "termination_source": termination_source,
                 "launcher_exit_code": rc,
                 "docker_run_exit_code": timeline.docker_run_exit_code,
                 "container_exit_code": timeline.container_exit_code,
                 "native_exit_code": timeline.native_exit_code,
                 "native_pipeline_uptime_seconds": timeline.native_pipeline_uptime_seconds(),
                  "native_observation_seconds": observation_seconds,
                  "required_native_observation_seconds": REQUIRED_NATIVE_OBSERVATION_SECONDS,
                  "native_observation_met": observation_met,
                 "native_viewer_started_monotonic_ns": timeline.native_start_ns,
                 "native_playing_monotonic_ns": timeline.playing_ns,
                 "native_pipeline_stopped_monotonic_ns": timeline.pipeline_stopped_ns,
                 "native_shutdown_intent_monotonic_ns": timeline.native_signal_received_ns,
                 "null_state_return_monotonic_ns": timeline.null_state_native_ns,
                 "crossed_800_seconds": timeline.crossed_800_seconds(),
                 "null_state_completed": timeline.null_state_completed,
                 "null_state_evidence": timeline.null_state_evidence,
                 "launcher_failure_stage": timeline.failure_stage,
                 "launcher_failure": timeline.failure,
                 "launcher_failure_diagnostic_path": timeline.failure_diagnostic_path,
                 "source_progress_at_stop": timeline.source_progress_at_stop,
                 "upstream_progress_enabled": timeline.upstream_progress_enabled,
                 "upstream_progress_samples_observed": timeline.upstream_progress_samples_observed,
                 "upstream_progress_samples_dropped": timeline.upstream_progress_samples_dropped,
                 "upstream_samples_with_native_envelope":
                     timeline.upstream_samples_with_native_envelope,
                 "upstream_progress_at_stop": timeline.upstream_progress_at_stop,
                 "upstream_source_events": timeline.upstream_source_events,
                 "upstream_source_events_observed": timeline.upstream_source_events_observed,
                 "upstream_source_events_dropped": timeline.upstream_source_events_dropped,
                 "upstream_diagnostics_report_path": timeline.upstream_diagnostics_report_path,
                 "upstream_diagnostics_collected_monotonic_ns":
                     timeline.upstream_diagnostics_collected_monotonic_ns,
                 "container_state": timeline.container_state,
                 "watchdog_marker": timeline.watchdog_marker,
                 "live_window_complete_marker": timeline.live_window_complete,
                 "camera_wall_closed_marker": timeline.camera_wall_closed,
                 "stop_command_exit_code": stop_result.exit_code,
                 "stop_method": stop_result.method, "cleanup_checks": checks,
                 "cleanup_ok": cleanup_ok, "success": success}
        if exception_category is not None:
            final["exception_category"] = exception_category
        try:
            put(report, final)
        except Exception:
            # No synthetic success record is written if persistence is unavailable.
            pass
        if log_stream is not None:
            log_stream.close()
        if install_signal_handlers:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)
    return 0 if success else 1


def poststop(runid: str) -> int:
    run_dir = RUNTIME / runid
    report = run_dir / "monitor_18m.jsonl"
    try:
        rows = []
        if report.exists():
            for line in report.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if rows and rows[-1].get("type") == "final":
            return 0 if rows[-1].get("cleanup_ok") is True else 1
        state = json.loads((run_dir / "state.json").read_text())
        identity = state.get("child_identity")
        try:
            put(report, {"type": "worker_exit_fallback", "utc": stamp(),
                        "child_identity_available": isinstance(identity, dict)})
        except OSError:
            pass
        if identity is not None:
            deadline = time.monotonic() + STOP - STOP_CLEANUP_MARGIN
            stop_result = stop_owned(report, "systemd_execstoppost_fallback", identity=identity)
            while time.monotonic() < deadline:
                ticks, _ = _proc_identity(identity["pid"])
                if ticks != identity.get("start_ticks"):
                    break
                time.sleep(0.1)
        else:
            stop_result = StopResult(2, "child_identity_not_persisted")
        if identity is None:
            # Without a persisted PID/start-time pair we cannot prove that a
            # child is absent, so never turn this fallback into a cleanup PASS.
            checks = {"child_identity_available": False, "owned_resources_verified": False}
        else:
            ticks, _ = _proc_identity(identity["pid"])
            child_gone = ticks != identity.get("start_ticks")
            checks = _cleanup_checks(identity["pid"], 0 if child_gone else None)
            checks["persisted_child_gone"] = child_gone
        cleanup_ok = all(checks.values())
        put(report, {"type": "final", "utc": stamp(), "reason": "supervisor_exit_fallback",
                     "stop_command_exit_code": stop_result.exit_code, "stop_method": stop_result.method,
                     "cleanup_checks": checks, "cleanup_ok": cleanup_ok, "success": False})
        return 0 if cleanup_ok else 1
    except Exception as exc:
        try:
            put(report, {"type": "poststop_failure", "utc": stamp(),
                         "exception_category": type(exc).__name__})
        except Exception:
            pass
        return 1


def start_run(preflight_fn: Callable | None = None, call_fn: Callable | None = None) -> Path:
    """Preflight first; no report, service, or child exists before it passes."""
    requested_mono = time.monotonic()
    preflight_fn = preflight_fn or (lambda: preflight(deadline=requested_mono + PREFLIGHT_BUDGET))
    call_fn = call_fn or call
    preflight_data = preflight_fn()
    elapsed_preflight = time.monotonic() - requested_mono
    if elapsed_preflight >= PRE:
        raise RuntimeError("startup_preflight_exceeded_runtime_budget")
    RUNTIME.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(RUNTIME, 0o700)
    runid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(5)
    run_dir = RUNTIME / runid
    run_dir.mkdir(mode=0o700)
    report = run_dir / "monitor_18m.jsonl"
    report.touch(mode=0o600)
    state = {"run_id": runid, "report": str(report), "display": preflight_data["display"],
             "xauthority": preflight_data["xauthority"], "start_mono": requested_mono,
             "unit": UNIT, "candidate_sha256": EXPECTED,
             "candidate_path": str(CANDIDATE),
             "candidate_launcher_sha256": CANDIDATE_LAUNCHER_EXPECTED,
             "candidate_launcher_path": str(CANDIDATE_LAUNCHER),
             "runtime_image": IMAGE,
             "source_stall_diagnostics": True,
             "upstream_progress_diagnostics": True}
    state_path = run_dir / "state.json"
    state_path.write_text(json.dumps(state, separators=(",", ":")))
    os.chmod(state_path, 0o600)
    exe = sys.executable
    helper = str(Path(__file__).resolve())
    # The transient user unit's RuntimeMaxSec is the independent deadline.
    # It begins only after all preflight, with >= 90 s reserved for cleanup.
    remaining = int(PRE - elapsed_preflight)
    if remaining <= 0:
        raise RuntimeError("startup_runtime_reserve_exhausted")
    command = [
        "systemd-run", "--user", "--collect", f"--unit={UNIT}", "--property=Type=exec",
        "--property=Restart=no", "--property=KillMode=mixed",
        f"--property=RuntimeMaxSec={remaining}s", f"--property=TimeoutStopSec={STOP}s",
        "--property=WorkingDirectory=" + str(ROOT),
        "--property=ExecStopPost=" + f"{exe} -B {helper} poststop {runid}",
        "--setenv=PYTHONPATH=" + ROOT_TEXT,
        "--setenv=DISPLAY=" + preflight_data["display"],
        "--setenv=XAUTHORITY=" + preflight_data["xauthority"],
        exe, "-B", helper, "worker", runid,
    ]
    try:
        result = call_fn(command, 15)
    except subprocess.TimeoutExpired:
        call_fn(["systemctl", "--user", "stop", UNIT], STOP + 15)
        put(report, {"type": "start_failure", "utc": stamp(),
                     "exception_category": "SystemdStartAcknowledgementTimeout"})
        put(report, {"type": "final", "utc": stamp(), "reason": "systemd_start_timeout",
                     "cleanup_ok": False, "success": False})
        raise RuntimeError("systemd_run_acknowledgement_timeout")
    if result.returncode != 0:
        put(report, {"type": "start_failure", "utc": stamp(), "exception_category": "SystemdStartFailed"})
        put(report, {"type": "final", "utc": stamp(), "reason": "systemd_start_failed",
                     "systemd_run_exit_code": result.returncode, "cleanup_ok": True, "success": False})
        raise RuntimeError("transient_service_start_failed")
    active = call_fn(["systemctl", "--user", "show", UNIT, "--property=ActiveState", "--value"], 5)
    if active.returncode != 0 or active.stdout.strip() not in ("active", "activating"):
        call_fn(["systemctl", "--user", "stop", UNIT], STOP + 15)
        put(report, {"type": "start_failure", "utc": stamp(), "exception_category": "TransientUnitInactive"})
        put(report, {"type": "final", "utc": stamp(), "reason": "transient_unit_inactive",
                     "cleanup_ok": False, "success": False})
        raise RuntimeError("transient_unit_failed_post_start_check")
    put(report, {"type": "start", "utc": stamp(), "run_id": runid,
                 "total_limit_seconds": TOTAL, "runtime_max_seconds": remaining,
                 "stop_reserve_seconds": STOP,
                 "systemd_activation_margin_seconds": TOTAL - PRE - STOP,
                 "viewer_sha256": EXPECTED,
                 "candidate_launcher_sha256": CANDIDATE_LAUNCHER_EXPECTED,
                 "runtime_image": IMAGE,
                 "source_stall_diagnostics": True,
                 "upstream_progress_diagnostics": True,
                 "unit": UNIT})
    return report


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "start":
        RUNTIME.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(RUNTIME, 0o700)
        try:
            report = start_run()
        except Exception as exc:
            raise SystemExit(f"START FAILED: {type(exc).__name__}: {exc}") from exc
        print(report)
    elif mode == "stop":
        result = call(["systemctl", "--user", "stop", UNIT], STOP + 15)
        if result.returncode:
            raise SystemExit("stop failed: " + result.stderr[-400:])
        print("stopped " + UNIT)
    elif mode == "status":
        result = call(["systemctl", "--user", "show", UNIT,
                       "--property=ActiveState", "--property=Result", "--property=MainPID"], 5)
        print(result.stdout.strip() if result.returncode == 0 else "unit not loaded")
    elif mode == "import-check":
        verify_project_imports()
        print("PROJECT_IMPORTS_OK")
    elif mode == "worker" and len(sys.argv) == 3:
        return worker(sys.argv[2])
    elif mode == "poststop" and len(sys.argv) == 3:
        return poststop(sys.argv[2])
    else:
        raise SystemExit("usage: validate_18m.py {start|stop|status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
