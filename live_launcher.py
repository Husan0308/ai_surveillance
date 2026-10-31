#!/usr/bin/env python3
"""Official DeepStream 9.1 live viewer; no connection by default.

--run uses a bounded test window; --run --monitor runs until stopped.
Both reuse the same configuration, secure credentials and camera ownership.
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import select
import shlex
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid

from services.gpu.runtime.live_config import (
    CAMERAS, CHANNELS, Camera, build_app_config, camera_uri, validate_model_files,
)
from services.gpu.runtime.live_health import SourceFpsMonitor
from services.shared.camera_config import load_settings
from services.shared.source_ownership import SourceOwnership
from services.shared.runtime_python import runtime_python
from services.shared.credential_redaction import (
    CredentialRedactor, child_environment_without_camera_secrets,
)
from services.gpu.runtime.telemetry_staging import StagedTelemetry
from services.gpu.runtime.telemetry_receiver import NativeTelemetryReceiver, TelemetryPublisherLoop
from services.gpu.runtime.latency_diagnostics import LatencyRunCollector
from services.camera_v11.mosaic_preview_ipc_v1 import (
    CONTAINER_MOSAIC_PREVIEW_DIR, CONTAINER_MOSAIC_PREVIEW_PATH,
    CONTAINER_MOSAIC_FULLSCREEN_PREVIEW_PATH,
    CONTAINER_MOSAIC_FULLSCREEN_REQUEST_PATH,
    DEFAULT_PUBLISH_RATE_FPS, MOSAIC_CAMERA_IDS, MOSAIC_FULLSCREEN_PREVIEW_PATH,
    MOSAIC_FULLSCREEN_REQUEST_PATH, MOSAIC_PREVIEW_PATH, MOSAIC_STALE_AFTER_SEC,
    FullscreenPreviewSlot, MosaicPreviewSlot,
)

ROOT = Path(__file__).resolve().parents[3]
PLATFORM = ROOT / "config/deepstream-platform.json"
BOUNDED_SHUTDOWN_RESERVE_SECONDS = 30.0
DAILY_STOP_GRACE_SECONDS = 15.0
# The container watchdog is a hard-kill fallback after the host's single
# graceful SIGINT and existing 15-second grace. The extra seconds below are
# for timeout/Docker process-start and exit propagation, measured in the last
# run; the host still force-cleans before the absolute run deadline.
BOUNDED_WATCHDOG_OBSERVE_SECONDS = 3.0
DAILY_CLEANUP_DEADLINE_SECONDS = 20.0
DAILY_FORCE_AFTER_SECONDS = 18.0
FULLSCREEN_VERIFIER_PREFLIGHT_TIMEOUT_SECONDS = 20.0


def _process_start_ticks(pid: int) -> str | None:
    """Identify a same-user process without trusting a stale/reused PID."""
    try:
        proc = Path(f"/proc/{pid}")
        if pid <= 1 or proc.stat().st_uid != os.getuid():
            return None
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except (OSError, IndexError):
        return None


def _write_monitor_owner(lock, *, cleanup_ok: bool | None = None,
                         container_name: str | None = None,
                         run_deadline: float | None = None,
                         cleanup_checks: dict | None = None,
                         container_state_before_cleanup: dict | None = None,
                         mosaic_preview_enabled: bool | None = None,
                         telemetry_enabled: bool | None = None) -> None:
    lock.seek(0)
    lock.truncate()
    json.dump({"mode": "daily", "pid": os.getpid(),
               "start_ticks": _process_start_ticks(os.getpid()),
               "cleanup_ok": cleanup_ok, "container_name": container_name,
               "run_deadline_monotonic": run_deadline,
               "mosaic_preview_enabled": mosaic_preview_enabled,
               "telemetry_enabled": telemetry_enabled,
               "container_state_before_cleanup": container_state_before_cleanup,
               "cleanup_checks": cleanup_checks}, lock)
    lock.flush()


def _camera_locks_free() -> bool:
    """Read-only lock check; never delete or replace ownership lockfiles."""
    descriptors = []
    try:
        for camera in CAMERAS:
            path = ROOT / ".runtime/camera-owner-locks" / f"{camera}.lock"
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
            except FileNotFoundError:
                continue
            descriptors.append(fd)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return False
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False
    finally:
        for fd in descriptors:
            os.close(fd)


def _path_absent(path: str | Path) -> bool:
    """Use lstat semantics so dangling links are not mistaken for cleanup."""
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _fullscreen_request_cleared() -> bool:
    path = Path(MOSAIC_FULLSCREEN_REQUEST_PATH)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600):
        return False
    try:
        return path.read_bytes().strip() == b""
    except OSError:
        return False


def _preview_ipc_checks(*, enabled: bool = True) -> dict[str, bool]:
    if not enabled:
        return {
            "mosaic_preview_slot_absent": True,
            "mosaic_preview_producer_lock_absent": True,
            "fullscreen_preview_slot_absent": True,
            "fullscreen_preview_producer_lock_absent": True,
            "fullscreen_request_cleared": True,
        }
    preview = Path(MOSAIC_PREVIEW_PATH)
    fullscreen = Path(MOSAIC_FULLSCREEN_PREVIEW_PATH)
    return {
        "mosaic_preview_slot_absent": _path_absent(preview),
        "mosaic_preview_producer_lock_absent": _path_absent(
            preview.with_name(f"{preview.stem}.producer-lock")),
        "fullscreen_preview_slot_absent": _path_absent(fullscreen),
        "fullscreen_preview_producer_lock_absent": _path_absent(
            fullscreen.with_name(f"{fullscreen.stem}.producer-lock")),
        "fullscreen_request_cleared": _fullscreen_request_cleared(),
    }


def _preview_ipc_clear() -> bool:
    return all(_preview_ipc_checks().values())


def _telemetry_ipc_checks(*, enabled: bool = True) -> dict[str, bool]:
    if not enabled:
        return {"monitoring_telemetry_absent": True,
                "tracked_object_telemetry_absent": True,
                "telemetry_owner_lock_absent": True}
    paths = [Path("/dev/shm/v11_monitoring_telemetry_v1.json")] + [
        Path(f"/dev/shm/v11_tracked_objects_cam{i:02d}_v1.json") for i in range(1, 7)
    ]
    return {
        "monitoring_telemetry_absent": _path_absent(paths[0]),
        "tracked_object_telemetry_absent": all(_path_absent(path) for path in paths[1:]),
        "telemetry_owner_lock_absent": _path_absent(
            Path("/dev/shm") / StagedTelemetry.LOCK_NAME),
    }


def _cleanup_check_snapshot(*, container_owned: bool,
                            container_removed: bool | None,
                            proc: subprocess.Popen | None,
                            camera_wall: subprocess.Popen | None,
                            camera_locks_owned: bool,
                            preview_enabled: bool,
                            run_deadline: float | None,
                            telemetry_enabled: bool = False,
                            telemetry_publisher_stopped: bool = True) -> dict:
    """Persist each cleanup predicate so a false aggregate is diagnosable."""
    container_absent = (not container_owned or proc is None or container_removed is True)
    predicates = {
        "container_absent": container_absent,
        # Every RTSP socket belongs to the one owned DeepStream container.
        "owned_rtsp_sessions_closed": container_absent,
        "docker_run_client_reaped": proc is None or proc.poll() is not None,
        "camera_wall_reaped": camera_wall is None or camera_wall.poll() is not None,
        "camera_source_locks_free": not camera_locks_owned or _camera_locks_free(),
        "telemetry_publisher_stopped": telemetry_publisher_stopped,
    }
    predicates.update(_preview_ipc_checks(enabled=preview_enabled))
    predicates.update(_telemetry_ipc_checks(enabled=telemetry_enabled))
    now = time.monotonic()
    remaining = None if run_deadline is None else run_deadline - now
    predicates["within_total_deadline"] = run_deadline is None or remaining >= 0
    return {
        "predicates": predicates,
        "deadline_remaining_seconds": (None if remaining is None else round(remaining, 3)),
        "cleanup_ok": all(predicates.values()),
        "monitor_lock_release": "owner file descriptor closes as launcher exits",
    }


def _recheck_monitor_cleanup(owner: dict, *, deadline: float | None = None) -> bool:
    """Reconcile an old unknown result only from current, positive evidence."""
    pid = owner.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        return False
    prefix = f"ai-surveillance-six-live-{pid}-"
    name = owner.get("container_name")
    if name is not None and (not isinstance(name, str) or not name.startswith(prefix)):
        return False
    preview_clear = (not owner.get("mosaic_preview_enabled", True)
                     or _preview_ipc_clear())
    telemetry_clear = all(_telemetry_ipc_checks(
        enabled=owner.get("telemetry_enabled", False)).values())
    return (preview_clear and telemetry_clear and _camera_locks_free()
            and _container_absent(name or prefix, prefix=name is None,
                                  deadline=deadline) is True)


def stop_monitoring() -> int:
    """Signal only the verified owner of the existing camera/viewer lock."""
    stop_wait_started = time.monotonic()
    path = ROOT / ".runtime/six-live-preview.lock"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        print("Monitoring is already stopped.")
        return 0
    except OSError:
        print("STOP FAILED: monitoring lock is inaccessible.", file=sys.stderr)
        return 2
    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("monitoring lock belongs to another user")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            text = os.read(fd, 4096)
            owner = json.loads(text) if text.strip() else {}
            if owner.get("mode") == "daily" and owner.get("cleanup_ok") is not True:
                if not _recheck_monitor_cleanup(owner):
                    raise RuntimeError("previous monitoring cleanup was not verified")
                print("Monitoring is stopped; container absence and camera locks reverified.")
                return 0
            print("Monitoring is already stopped.")
            return 0
        os.lseek(fd, 0, os.SEEK_SET)
        owner = json.loads(os.read(fd, 4096))
        pid = owner.get("pid")
        if (owner.get("mode") != "daily" or not isinstance(pid, int)
                or not owner.get("start_ticks")
                or _process_start_ticks(pid) != owner["start_ticks"]):
            raise RuntimeError("active lock has no verified daily monitoring owner")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # The owner may have just finished; still verify its cleanup.
        stop_deadline = time.monotonic() + 25
        bounded_deadline = owner.get("run_deadline_monotonic")
        if isinstance(bounded_deadline, (int, float)) and bounded_deadline > 0:
            stop_deadline = min(stop_deadline, float(bounded_deadline))
        while time.monotonic() < stop_deadline:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(max(0.0, min(0.1, stop_deadline - time.monotonic())))
                continue
            os.lseek(fd, 0, os.SEEK_SET)
            finished = json.loads(os.read(fd, 4096))
            if (finished.get("cleanup_ok") is not True
                    and not _recheck_monitor_cleanup(finished, deadline=stop_deadline)):
                raise RuntimeError("monitoring stopped but cleanup was not verified")
            print("Monitoring stopped; camera and preview resources released; "
                  f"owner_wait_seconds={time.monotonic() - stop_wait_started:.3f}.")
            return 0
        raise RuntimeError("monitoring stop did not complete within its remaining deadline")
    except (OSError, ValueError, RuntimeError):
        print("STOP FAILED: monitoring owner or cleanup could not be verified; "
              f"elapsed_seconds={time.monotonic() - stop_wait_started:.3f}.",
              file=sys.stderr)
        return 2
    finally:
        os.close(fd)


def _camera_wall_environment(*, fullscreen_sequence_output: Path | None = None,
                             pixel_provenance: bool = False,
                             preview_latency_log_path: Path | None = None) -> dict[str, str]:
    """Build the UI environment without tokenless non-loopback API settings."""
    env = child_environment_without_camera_secrets()
    env["V11_UI_MOSAIC_WALL_ENABLED"] = "1"
    env["V11_UI_MOSAIC_PREVIEW_PATH"] = str(MOSAIC_PREVIEW_PATH)
    env["V11_UI_FULLSCREEN_PREVIEW_PATH"] = str(MOSAIC_FULLSCREEN_PREVIEW_PATH)
    env["V11_UI_FULLSCREEN_REQUEST_PATH"] = str(MOSAIC_FULLSCREEN_REQUEST_PATH)
    # Verifier selection is explicit. Ambient settings must not replace the
    # normal daily Camera Wall when the operator did not request verification.
    env.pop("V11_UI_FULLSCREEN_SEQUENCE_OUTPUT", None)
    env.pop("V11_UI_FULLSCREEN_CAPTURE_MODE", None)
    env.pop("V11_UI_MOSAIC_PIXEL_PROVENANCE", None)
    env.pop("MV3DT_PREVIEW_LATENCY_LOG", None)
    if fullscreen_sequence_output is not None:
        env["V11_UI_FULLSCREEN_SEQUENCE_OUTPUT"] = str(fullscreen_sequence_output)
        env["V11_UI_FULLSCREEN_CAPTURE_MODE"] = "qt-widget"
    if pixel_provenance:
        env["V11_UI_MOSAIC_PIXEL_PROVENANCE"] = "1"
    if preview_latency_log_path is not None:
        env["MV3DT_PREVIEW_LATENCY_LOG"] = str(preview_latency_log_path)
    if not env.get("SURVEILLANCE_API_TOKEN", "").strip():
        # The mosaic video is local IPC and does not need API/ML access. If no
        # backend token is configured, keep this UI process local-only instead
        # of inheriting wildcard bind hosts or external frontend URLs.
        for key in ("API_HOST", "ML_HOST", "FRONTEND_API_BASE_URL",
                    "FRONTEND_ML_VIDEO_BASE_URL", "ML_SERVICE_URL"):
            env.pop(key, None)
        env["PYTHON_DOTENV_DISABLED"] = "1"
    return env


def _validated_fullscreen_sequence_output(value: str | None) -> Path:
    """Require a fresh output file inside a private directory owned by this user."""
    raw = (value or "").strip()
    if not raw:
        raise RuntimeError("fullscreen verification output is required")
    output = Path(raw)
    if not output.is_absolute() or not output.parent.is_dir() or os.path.lexists(output):
        raise RuntimeError("fullscreen verification output path is not fresh and usable")
    output_parent = output.parent.stat()
    if output_parent.st_uid != os.getuid() or output_parent.st_mode & 0o077:
        raise RuntimeError("fullscreen verification output directory must be private and owned by this user")
    return output


def _new_latency_evidence_directory() -> tuple[Path, Path]:
    """Create a private per-run directory for native and Qt latency evidence."""
    runtime_dir = ROOT / ".runtime"
    runtime_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = runtime_dir / f"six-live-latency-{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(mode=0o700)
    os.chmod(directory, 0o700)
    info = directory.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise PermissionError("latency evidence directory is not private and user-owned")
    return directory, directory / "qt-paint.jsonl"


def _fullscreen_verifier_preflight(
        output: Path, fullscreen_camera_ids: list[str] | None = None,
        *, pixel_provenance: bool = False) -> None:
    """Start and close the actual verifier UI before any NVR/RTSP operation."""
    env = _camera_wall_environment(fullscreen_sequence_output=output,
                                   pixel_provenance=pixel_provenance)
    command = [str(runtime_python(ROOT)), "-m", "scripts.validate_production_wall",
               "--preflight-only"]
    for camera_id in fullscreen_camera_ids or []:
        command.extend(["--fullscreen-camera", camera_id])
    try:
        result = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True,
            timeout=FULLSCREEN_VERIFIER_PREFLIGHT_TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("fullscreen verifier preflight timed out before camera startup") from exc
    expected = "FULLSCREEN_VERIFIER_PREFLIGHT_PASS camera_source_ids=" + ",".join(
        f"{camera_id}:{source_id}" for source_id, camera_id in enumerate(MOSAIC_CAMERA_IDS)
    )
    if result.returncode != 0 or expected not in result.stdout.splitlines():
        raise RuntimeError(
            f"fullscreen verifier preflight failed before camera startup; exit_code={result.returncode}"
        )
    print(expected, flush=True)


def _start_camera_wall(*, fullscreen_sequence_output: Path | None = None,
                       fullscreen_camera_ids: list[str] | None = None,
                       pixel_provenance: bool = False,
                       grid_only: bool = False,
                       preview_latency_log_path: Path | None = None) -> subprocess.Popen:
    if grid_only and fullscreen_sequence_output is None:
        raise ValueError("grid-only evidence mode requires a private output path")
    env = _camera_wall_environment(fullscreen_sequence_output=fullscreen_sequence_output,
                                   pixel_provenance=pixel_provenance,
                                   preview_latency_log_path=preview_latency_log_path)
    sequence_output = env.get("V11_UI_FULLSCREEN_SEQUENCE_OUTPUT", "").strip()
    command = [str(runtime_python(ROOT))]
    if sequence_output:
        output = _validated_fullscreen_sequence_output(sequence_output)
        capture_mode = env.get("V11_UI_FULLSCREEN_CAPTURE_MODE", "qt-widget").strip()
        if capture_mode not in {"qt-widget", "x11-window"}:
            raise RuntimeError("unsupported fullscreen screenshot capture mode")
        command.extend([
            "-m", "scripts.validate_production_wall", "--output", str(output),
            "--grid-only" if grid_only else "--fullscreen-sequence",
            "--capture-mode", capture_mode,
        ])
        for camera_id in fullscreen_camera_ids or []:
            command.extend(["--fullscreen-camera", camera_id])
    else:
        command.extend(["-m", "services.frontend.app.main"])
    return subprocess.Popen(
        command, cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        start_new_session=True,
    )


class ChildOutputDiagnostics:
    """Bounded, credential-redacted output retained for a failed live launch."""

    _EVENT_MARKERS = (
        "warning", "error", "critical", "fatal", "eos", "signal",
        "reconnect", "timeout", "not-negotiated", "failed", "unknown key",
        "pipeline ready", "pipeline running", "pipeline stopped", "mosaic_", "fullscreen_",
        "six_rtsp_shutdown",
    )
    _ERROR_MARKERS = (
        "not-negotiated", "internal data stream error",
    )
    _SEVERITY = re.compile(r"\b(?:ERROR|CRITICAL|FATAL)\b")
    _GST_SOURCE_SEVERITY = re.compile(
        r"\b(?P<severity>ERROR|WARNING|CRITICAL|FATAL) from "
        r"(?P<element>[A-Za-z0-9_.-]+)\s*:"
    )
    _CANONICAL_SOURCE_ELEMENT = re.compile(r"\bsrc_elem_?(?P<source_id>[0-5])\b")
    _SOURCE_ID_FIELD = re.compile(r"\bsource_id\s*[:=]\s*(?P<source_id>[0-5])\b")

    def __init__(self, *, max_tail: int = 64, max_events: int = 48,
                 max_errors: int = 16, max_rtsp_events: int = 64,
                 max_pixel_provenance_events: int = 12,
                 max_pixel_provenance_line_chars: int = 32768,
                 max_line_chars: int = 1200,
                 source_stall_diagnostics: bool = False,
                 max_source_progress_samples: int = 720,
                 max_source_events: int = 128) -> None:
        self.null_started = threading.Event()
        self.null_completed = threading.Event()
        self._lock = threading.Lock()
        self._tail = deque(maxlen=max_tail)
        self._events = deque(maxlen=max_events)
        self._errors = deque(maxlen=max_errors)
        self._progress = deque(maxlen=max_events)
        self._rtsp_shutdown = deque(maxlen=max_rtsp_events)
        self._rtsp_shutdown_total = 0
        self._pixel_provenance = deque()
        self._pixel_provenance_capacity = max(0, int(max_pixel_provenance_events))
        self._pixel_provenance_line_limit = max(0, int(max_pixel_provenance_line_chars))
        self._pixel_provenance_total = 0
        self._pixel_provenance_dropped = 0
        self._pixel_provenance_summary: str | None = None
        self._mosaic_label_metadata: str | None = None
        self._max_line_chars = max_line_chars
        self._pending_error: dict | None = None
        self._pending_context = 0
        self._source_stall_diagnostics_enabled = bool(source_stall_diagnostics)
        self._source_progress_capacity = max(0, int(max_source_progress_samples))
        self._source_event_capacity = max(0, int(max_source_events))
        self._source_progress_samples = deque()
        self._source_events = deque()
        self._source_progress_total = 0
        self._source_progress_dropped = 0
        self._source_events_total = 0
        self._source_events_dropped = 0
        self._source_guard_failure: dict | None = None

    @staticmethod
    def _valid_source_id(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 6

    def _append_source_event(self, event: dict) -> None:
        if self._source_event_capacity == 0:
            self._source_events_dropped += 1
            return
        if len(self._source_events) >= self._source_event_capacity:
            self._source_events.popleft()
            self._source_events_dropped += 1
        self._source_events.append(event)

    def _observe_source_diagnostic(self, line: str) -> None:
        """Retain bounded numeric progress/event classes, never log text or URIs."""
        host_ns = time.monotonic_ns()
        if line.startswith("SIX_PROGRESS "):
            try:
                sample = json.loads(line.partition(" ")[2])
            except (TypeError, json.JSONDecodeError):
                return
            if not isinstance(sample, dict):
                return
            source_id = sample.get("source_id")
            expected_camera = (f"CAM-{source_id + 1:02d}"
                               if self._valid_source_id(source_id) else None)
            stages = sample.get("stages")
            if (expected_camera is None or sample.get("camera_id") != expected_camera
                    or not isinstance(stages, list) or len(stages) < 4
                    or not all(isinstance(stages[index], dict) for index in (0, 3))):
                return
            decode, tracker = stages[0], stages[3]
            fields = (
                sample.get("monotonic_ns"), sample.get("source_generation"),
                decode.get("count"), decode.get("last_ns"), decode.get("pts"),
                tracker.get("count"), tracker.get("last_ns"), tracker.get("pts"),
            )
            if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                       for value in fields):
                return
            row = {
                "host_receipt_monotonic_ns": host_ns,
                "native_monotonic_ns": fields[0],
                "source_id": source_id,
                "camera_id": expected_camera,
                "source_generation": fields[1],
                "decode_count": fields[2],
                "decode_last_ns": fields[3],
                "decode_pts": fields[4],
                "tracker_count": fields[5],
                "tracker_last_ns": fields[6],
                "tracker_pts": fields[7],
            }
            with self._lock:
                self._source_progress_total += 1
                if self._source_progress_capacity == 0:
                    self._source_progress_dropped += 1
                else:
                    if len(self._source_progress_samples) >= self._source_progress_capacity:
                        self._source_progress_samples.popleft()
                        self._source_progress_dropped += 1
                    self._source_progress_samples.append(row)
            return

        event = None
        source_id = None
        source_generation = None
        native_ns = None
        severity = None
        condition = None
        if line.startswith("SIX_SOURCE_STATE ") or line.startswith("SIX_SOURCE_EVENT "):
            prefix, _, payload = line.partition(" ")
            try:
                record = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                return
            if not isinstance(record, dict) or not self._valid_source_id(record.get("source_id")):
                return
            source_id = record["source_id"]
            generation = record.get("source_generation")
            if isinstance(generation, int) and not isinstance(generation, bool) and generation >= 0:
                source_generation = generation
            native_ns = record.get("monotonic_ns")
            if (not isinstance(native_ns, int) or isinstance(native_ns, bool)
                    or native_ns < 0):
                native_ns = None
            if prefix == "SIX_SOURCE_STATE":
                state = record.get("state")
                if state not in {"online", "reconnecting", "offline"}:
                    return
                event = f"source_{state}"
            else:
                kind = record.get("event")
                if kind not in {"warning", "error", "eos", "timeout",
                                "reconnecting", "reconnect_attempt_exceeded"}:
                    return
                event = f"source_{kind}"
                severity_value = record.get("severity")
                if severity_value in {"warning", "error"}:
                    severity = severity_value
        else:
            lowered = line.lower()
            severity_match = self._GST_SOURCE_SEVERITY.search(line)
            if severity_match:
                severity = severity_match.group("severity").lower()
            source_match = self._SOURCE_ID_FIELD.search(line)
            if source_match:
                source_id = int(source_match.group("source_id"))
            else:
                if severity_match:
                    element_match = self._CANONICAL_SOURCE_ELEMENT.search(
                        severity_match.group("element"))
                    if element_match:
                        source_id = int(element_match.group("source_id"))
            if ("reconnect attempt exceeded" in lowered
                    or "reconnection attempts exceeded" in lowered):
                event = "reconnect_attempt_exceeded"
                condition = "attempts_exceeded"
            elif "timed out" in lowered or "timeout" in lowered:
                event = "timeout"
                condition = "gstreamer_timeout" if severity else "timeout"
            elif "received eos" in lowered or "gst_message_eos" in lowered:
                event = "eos"
                condition = "end_of_stream"
            else:
                if severity_match:
                    event = f"gstreamer_{severity_match.group('severity').lower()}"
        if event is not None:
            with self._lock:
                self._source_events_total += 1
                self._append_source_event({
                    "event": event,
                    "source_id": source_id,
                    "source_generation": source_generation,
                    "severity": severity,
                    "condition": condition,
                    "native_monotonic_ns": native_ns,
                    # Native bus text has no timestamp in this build; this marks
                    # host receipt, not the exact time the GStreamer event arose.
                    "host_receipt_monotonic_ns": host_ns,
                })

    def record_source_guard_failure(self, snapshot: dict) -> None:
        """Capture the unchanged guard decision and numeric source state."""
        if not self._source_stall_diagnostics_enabled:
            return
        safe_fields = (
            "native_monotonic_ns", "received_age_seconds", "decode_count",
            "tracker_count", "decode_fps", "tracker_fps", "stage_age_seconds",
            "consecutive_good",
        )
        cameras = snapshot.get("camera_progress", {})
        safe_cameras = {
            camera_id: {key: values.get(key) for key in safe_fields}
            for camera_id, values in cameras.items()
            if isinstance(camera_id, str) and camera_id.startswith("CAM-")
            and isinstance(values, dict)
        }
        with self._lock:
            self._source_guard_failure = {
                "host_monotonic_ns": time.monotonic_ns(),
                "ready": bool(snapshot.get("ready")),
                "fresh": bool(snapshot.get("fresh")),
                "guard_thresholds": snapshot.get("guard_thresholds", {}),
                "camera_progress": safe_cameras,
            }

    def ingest(self, sanitized_line: str) -> None:
        raw_line = sanitized_line.rstrip("\r\n")
        is_pixel_event = raw_line.startswith("MOSAIC_PIXEL_PROVENANCE ")
        is_pixel_summary = raw_line.startswith("MOSAIC_PIXEL_PROVENANCE_SUMMARY ")
        pixel_line_too_long = (is_pixel_event and
                               len(raw_line) > self._pixel_provenance_line_limit)
        line = raw_line[:self._max_line_chars]
        if is_pixel_event:
            with self._lock:
                self._pixel_provenance_total += 1
                if (pixel_line_too_long or self._pixel_provenance_capacity == 0
                        or len(self._pixel_provenance) >= self._pixel_provenance_capacity):
                    self._pixel_provenance_dropped += 1
                else:
                    # Keep the complete, already-redacted JSON record. The
                    # generic rolling tail remains short and bounded.
                    self._pixel_provenance.append(raw_line)
        elif is_pixel_summary:
            with self._lock:
                self._pixel_provenance_summary = line
        if line.startswith("SIX_TRACKER_FRAME "):
            # Object-level metadata is delivered through the host receiver;
            # retain neither boxes nor object identifiers in diagnostic logs.
            return
        if "SIX_SHUTDOWN event=pipeline_null_started" in line:
            self.null_started.set()
        if "SIX_SHUTDOWN event=pipeline_null_observed" in line:
            self.null_completed.set()
        if self._source_stall_diagnostics_enabled:
            self._observe_source_diagnostic(line)
        lowered = line.lower()
        with self._lock:
            self._tail.append(line)
            # Preserve the one native post-tiler source/pad layout record
            # separately from the rolling event tail. Periodic preview stats
            # otherwise evict this startup-only mapping evidence.
            if line.startswith("MOSAIC_LABEL_METADATA ") and self._mosaic_label_metadata is None:
                self._mosaic_label_metadata = line
            if line.startswith("SIX_RTSP_SHUTDOWN "):
                self._rtsp_shutdown_total += 1
                self._rtsp_shutdown.append(line)
            # SIX_PROGRESS is structured telemetry. Its JSON fields include
            # fatal_metadata and source_errors, not GStreamer log severities.
            if line.startswith("SIX_PROGRESS "):
                self._progress.append(line)
                return
            if any(marker in lowered for marker in self._EVENT_MARKERS):
                self._events.append(line)
            if (self._SEVERITY.search(line)
                    or any(marker in lowered for marker in self._ERROR_MARKERS)):
                self._pending_error = {"message": line, "following_lines": []}
                self._errors.append(self._pending_error)
                self._pending_context = 4
            elif self._pending_context:
                if self._pending_error is not None:
                    self._pending_error["following_lines"].append(line)
                self._pending_context -= 1
                if self._pending_context == 0:
                    self._pending_error = None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "recent_child_output": list(self._tail),
                "diagnostic_events": list(self._events),
                "source_progress_messages": list(self._progress),
                "mosaic_label_metadata": self._mosaic_label_metadata,
                "rtsp_shutdown_events": list(self._rtsp_shutdown),
                "rtsp_shutdown_events_observed": self._rtsp_shutdown_total,
                "rtsp_shutdown_events_dropped": max(
                    0, self._rtsp_shutdown_total - len(self._rtsp_shutdown)),
                "mosaic_pixel_provenance_events": list(self._pixel_provenance),
                "mosaic_pixel_provenance_events_observed": self._pixel_provenance_total,
                "mosaic_pixel_provenance_events_dropped": self._pixel_provenance_dropped,
                "mosaic_pixel_provenance_summary": self._pixel_provenance_summary,
                "source_stall_diagnostics": {
                    "enabled": self._source_stall_diagnostics_enabled,
                    "progress_samples": list(self._source_progress_samples),
                    "progress_samples_observed": self._source_progress_total,
                    "progress_samples_dropped": self._source_progress_dropped,
                    "events": list(self._source_events),
                    "events_observed": self._source_events_total,
                    "events_dropped": self._source_events_dropped,
                    "guard_failure": self._source_guard_failure,
                },
                "gstreamer_errors": [
                    {"message": item["message"],
                     "following_lines": list(item["following_lines"])}
                    for item in self._errors
                ],
            }


class CameraWallOutputDiagnostics:
    """Bounded credential-redacted tail of PySide6 output for failure reports."""

    def __init__(self, *, max_lines: int = 64, max_line_chars: int = 1200) -> None:
        self._lines = deque(maxlen=max_lines)
        self._max_line_chars = max_line_chars
        self._lock = threading.Lock()

    def ingest(self, sanitized_line: str) -> None:
        with self._lock:
            self._lines.append(sanitized_line.rstrip("\r\n")[:self._max_line_chars])

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self._lines)


def _write_failure_report(*, stage: str, exc: BaseException,
                          redactor: CredentialRedactor,
                          diagnostics: ChildOutputDiagnostics,
                          proc: subprocess.Popen | None,
                          exit_code_at_detection: int | None,
                          docker_run_exit_code: int | None = None,
                          docker_run_client_killed: bool = False,
                          cleanup_stop_requested: bool,
                          container_state_before_cleanup: dict | None,
                          container_removed: bool | None,
                          cleanup_checks: dict | None = None,
                          camera_wall_output: CameraWallOutputDiagnostics | None = None) -> Path:
    """Persist an exclusive, private, redacted snapshot after child output drains."""
    report_dir = ROOT / ".runtime"
    report_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = report_dir / f"six-live-failure-{time.time_ns()}-{os.getpid()}.json"
    current_code = proc.returncode if proc is not None else None
    payload = {
        "schema_version": 1,
        "failure_stage": stage,
        "failure": redactor.sanitize_exception(exc),
        "exit_status": {
            "docker_run_returncode_at_failure_detection": exit_code_at_detection,
            "docker_run_exit_code_observed": docker_run_exit_code,
            "docker_run_returncode_after_cleanup": current_code,
            "docker_run_client_killed_by_launcher": docker_run_client_killed,
            "cleanup_stop_requested": cleanup_stop_requested,
            "container_state_before_cleanup": container_state_before_cleanup,
            "container_removed_after_capture": container_removed,
            "note": ("docker run reports the container command status after it starts; "
                     "Docker CLI/setup failures can instead return Docker-specific codes"),
        },
        "cleanup_checks": cleanup_checks,
        "camera_wall_output": (camera_wall_output.snapshot()
                               if camera_wall_output is not None else []),
        **diagnostics.snapshot(),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _redact_report_value(value, redactor: CredentialRedactor):
    if isinstance(value, str):
        return redactor.sanitize(value)
    if isinstance(value, dict):
        return {key: _redact_report_value(item, redactor)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_report_value(item, redactor) for item in value]
    return value


def _atomic_private_json_report(report_dir: Path, prefix: str, payload: dict) -> Path:
    """Write one private JSON artifact atomically; refuse an unsafe directory."""
    report_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory_stat = report_dir.stat()
    if (directory_stat.st_uid != os.getuid()
            or stat.S_IMODE(directory_stat.st_mode) & 0o077):
        raise PermissionError("diagnostic report directory is not private and user-owned")
    path = report_dir / f"{prefix}-{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex[:8]}.json"
    fd, temporary_name = tempfile.mkstemp(prefix=".pending-", dir=report_dir)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(report_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def _private_camera_wall_screenshot(path_value: object, sequence_path: Path) -> bool:
    if not isinstance(path_value, str) or not path_value:
        return False
    path = Path(path_value)
    try:
        if (not path.is_absolute() or path.parent.resolve(strict=True) !=
                sequence_path.parent.resolve(strict=True)):
            return False
        info = path.lstat()
    except OSError:
        return False
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size < 8):
        return False
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            return stream.read(8) == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def _read_camera_wall_visual_evidence(sequence_output: Path | None) -> dict:
    """Summarize verified Qt captures from the bounded UI sequence log.

    A mosaic IPC frame alone never establishes Qt presentation. A fullscreen
    entry counts only when its private Qt screenshot, exact source identity,
    metadata correlation, and two distinct presented frame identities exist.
    Grid capture is reported separately because v4 IPC has no per-tile source
    identities; pixel samples are observations, not a per-camera health claim.
    """
    evidence = {
        "sequence_log_available": False,
        "qt_grid_widget_capture_verified": False,
        "grid_screenshot": None,
        "grid_cards_fresh_at_capture": False,
        "grid_tile_pixel_measurements": {},
        "grid_tile_pixel_measurements_complete": False,
        "cam05_pixel_provenance_observations": [],
        "cam05_pixel_provenance_observations_dropped": 0,
        "direct_qt_painted_frame_verified": False,
        "verified_fullscreen_camera_ids": [],
        "verified_fullscreen_frames": [],
        "all_six_grid_source_content_correlated": False,
        "limitation": "mosaic IPC v4 carries composite freshness, not per-tile source frame identities",
    }
    if sequence_output is None:
        return evidence
    try:
        info = sequence_output.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > 2 * 1024 * 1024):
            return evidence
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(sequence_output, flags)
        with os.fdopen(fd, "rb") as stream:
            raw = stream.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            return evidence
        decoded = [json.loads(line) for line in raw.splitlines() if line.strip()]
        rows = [row for row in decoded if isinstance(row, dict)]
    except (OSError, ValueError, json.JSONDecodeError):
        return evidence
    evidence["sequence_log_available"] = True

    pixel_rows = [row for row in rows
                  if row.get("event") == "cam05_pixel_provenance_observation"]
    evidence["cam05_pixel_provenance_observations"] = pixel_rows[:12]
    evidence["cam05_pixel_provenance_observations_dropped"] = max(0, len(pixel_rows) - 12)

    grid = next((row for row in rows if row.get("event") == "grid_captured"), None)
    if grid is not None:
        cameras = grid.get("cameras")
        screenshot_ok = (
            grid.get("capture_kind") == "qt_widget"
            and grid.get("view") == "main_3x2_grid"
            and _private_camera_wall_screenshot(grid.get("screenshot"), sequence_output)
        )
        fresh_cards = isinstance(cameras, dict) and set(cameras) == set(MOSAIC_CAMERA_IDS)
        if fresh_cards:
            for camera_id in MOSAIC_CAMERA_IDS:
                card = cameras[camera_id]
                age = card.get("frame_age_seconds")
                if (type(card.get("sequence")) is not int or card["sequence"] <= 0
                        or card.get("repainted_sequence") != card.get("sequence")
                        or not isinstance(age, (int, float)) or not math.isfinite(age)
                        or age < 0 or age > MOSAIC_STALE_AFTER_SEC):
                    fresh_cards = False
                    break
        evidence["qt_grid_widget_capture_verified"] = bool(screenshot_ok and fresh_cards)
        evidence["grid_cards_fresh_at_capture"] = bool(fresh_cards)
        if screenshot_ok:
            evidence["grid_screenshot"] = grid.get("screenshot")
        measurements = {
            camera_id: cameras[camera_id].get("qt_video_pixel_evidence")
            for camera_id in MOSAIC_CAMERA_IDS
        } if isinstance(cameras, dict) else {}
        evidence["grid_tile_pixel_measurements"] = measurements
        evidence["grid_tile_pixel_measurements_complete"] = bool(
            set(measurements) == set(MOSAIC_CAMERA_IDS)
            and all(isinstance(item, dict) and item.get("status") == "sampled"
                    for item in measurements.values())
        )

    verified = []
    for row in rows:
        if row.get("event") != "fullscreen_camera_result":
            continue
        camera_id = row.get("camera_id")
        if camera_id not in MOSAIC_CAMERA_IDS:
            continue
        expected_source_id = MOSAIC_CAMERA_IDS.index(camera_id)
        motion = row.get("motion_frames")
        motion_proven = (isinstance(motion, list) and len(motion) >= 2
                         and isinstance(motion[0], dict) and isinstance(motion[1], dict)
                         and motion[0].get("visual_fingerprint")
                         and motion[1].get("visual_fingerprint")
                         and motion[0]["visual_fingerprint"] != motion[1]["visual_fingerprint"]
                         and isinstance(motion[0].get("source_frame_sequence"), int)
                         and isinstance(motion[1].get("source_frame_sequence"), int)
                         and motion[1]["source_frame_sequence"] > motion[0]["source_frame_sequence"])
        screenshot_ok = (
            row.get("capture_kind") == "qt_widget"
            and _private_camera_wall_screenshot(row.get("screenshot"), sequence_output)
        )
        if (row.get("status") == "PASS" and row.get("source_id") == expected_source_id
                and row.get("resolution") == [1920, 1080]
                and type(row.get("presented_frames")) is int
                and row["presented_frames"] >= 2
                and row.get("post_tiler_metadata_correlated") is True
                and row.get("source_selection_confirmed_by_pre_tiler_identity_and_pts") is True
                and row.get("identity_correlation") == "gst_buffer_pts_exact"
                and motion_proven and screenshot_ok):
            verified.append({
                "camera_id": camera_id,
                "source_id": expected_source_id,
                "presented_frames": row["presented_frames"],
                "resolution": row["resolution"],
                "screenshot": row["screenshot"],
            })
    evidence["verified_fullscreen_frames"] = verified
    evidence["verified_fullscreen_camera_ids"] = [item["camera_id"] for item in verified]
    evidence["direct_qt_painted_frame_verified"] = bool(verified)
    return evidence


def _write_rtsp_pause_run_report(*, report_dir: Path, run_status: str,
                                 diagnostics: ChildOutputDiagnostics,
                                 redactor: CredentialRedactor,
                                 viewer_binary: Path | None,
                                 viewer_sha256: str | None = None,
                                 invocation_started: float,
                                 docker_run_exit_code: int | None,
                                 container_exit_code: int | None,
                                 container_state: dict | None,
                                 cleanup_checks: dict | None,
                                 child_output_drained: bool,
                                 failure_stage: str | None,
                                 failure: BaseException | None,
                                 camera_wall_output: CameraWallOutputDiagnostics | None = None,
                                 camera_wall_evidence_path: Path | None = None) -> Path:
    """Persist opt-in shutdown counters and process status after output drains."""
    snapshot = diagnostics.snapshot()
    viewer = None
    if viewer_binary is not None:
        viewer = {"name": viewer_binary.name}
        if viewer_sha256 is None:
            try:
                with viewer_binary.open("rb") as stream:
                    viewer_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            except OSError:
                viewer_sha256 = None
        viewer["sha256"] = viewer_sha256
    payload = {
        "schema_version": 1,
        "report_type": "six_live_rtsp_pause_suppression",
        "run_status": run_status,
        "failure_stage": failure_stage,
        "failure": redactor.sanitize_exception(failure) if failure else None,
        "elapsed_seconds": round(time.monotonic() - invocation_started, 3),
        "viewer": viewer,
        "exit_status": {
            "docker_run_exit_code": docker_run_exit_code,
            "container_exit_code": container_exit_code,
            "container_state_before_removal": container_state,
        },
        "cleanup_checks": cleanup_checks,
        "child_output_drained": child_output_drained,
        "native_null_transition": {
            "started_marker_seen": diagnostics.null_started.is_set(),
            "completed_marker_seen": diagnostics.null_completed.is_set(),
        },
        "rtsp_shutdown_event_count": snapshot["rtsp_shutdown_events_observed"],
        "rtsp_shutdown_events_dropped_from_bounded_buffer":
            snapshot["rtsp_shutdown_events_dropped"],
        "diagnostics": snapshot,
        "camera_wall_output": (camera_wall_output.snapshot()
                               if camera_wall_output is not None else []),
        "visual_evidence": _read_camera_wall_visual_evidence(camera_wall_evidence_path),
    }
    payload = _redact_report_value(payload, redactor)
    return _atomic_private_json_report(report_dir, "six-live-rtsp-pause", payload)


def _write_latency_run_report(*, report_dir: Path,
                              collector: LatencyRunCollector,
                              qt_log_path: Path,
                              host_time_namespace_inode: int,
                              run_status: str,
                              invocation_started: float,
                              viewer_binary: Path | None,
                              viewer_sha256: str | None,
                              failure_stage: str | None,
                              failure: BaseException | None,
                              docker_run_exit_code: int | None,
                              container_exit_code: int | None,
                              cleanup_checks: dict | None,
                              child_output_drained: bool,
                              diagnostics: ChildOutputDiagnostics,
                              redactor: CredentialRedactor) -> Path:
    """Persist bounded latency samples and Qt records after process cleanup."""
    viewer = None
    if viewer_binary is not None:
        viewer = {"name": viewer_binary.name, "sha256": viewer_sha256}
    payload = {
        "schema_version": 1,
        "report_type": "six_camera_latency_measurement",
        "run_status": run_status,
        "elapsed_seconds": round(time.monotonic() - invocation_started, 3),
        "viewer": viewer,
        "failure_stage": failure_stage,
        "failure": redactor.sanitize_exception(failure) if failure else None,
        "exit_status": {
            "docker_run_exit_code": docker_run_exit_code,
            "container_exit_code": container_exit_code,
            "native_null_started": diagnostics.null_started.is_set(),
            "native_null_completed": diagnostics.null_completed.is_set(),
            "child_output_drained": child_output_drained,
        },
        "cleanup_checks": cleanup_checks,
        "qt_log_path": str(qt_log_path),
        "measurements": collector.snapshot(
            qt_log_path, host_time_namespace_inode=host_time_namespace_inode),
    }
    payload = _redact_report_value(payload, redactor)
    return _atomic_private_json_report(report_dir, "latency-report", payload)


def secret_tar(app: str, infer: str, auth: str | None = None) -> bytes:
    """Carry sensitive application config over stdin into container tmpfs."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        members = [("app.txt", app), ("infer.txt", infer)]
        if auth is not None:
            members.append(("auth.ini", auth))
        for name, value in members:
            data = value.encode("utf-8")
            item = tarfile.TarInfo(name)
            item.size = len(data)
            item.mode = 0o600
            item.uid = item.gid = 0
            tar.addfile(item, io.BytesIO(data))
    return out.getvalue()


def native_auth_config(cameras: list[Camera]) -> str:
    """Make the existing native monitor's tmpfs-only credential map."""
    if tuple(camera.camera_id for camera in cameras) != CAMERAS:
        raise ValueError("native auth mapping must match CAM-01..CAM-06 source order")
    lines: list[str] = []
    for index, (camera, channel) in enumerate(zip(cameras, CHANNELS, strict=True)):
        uri, _, _ = camera_uri(camera, channel)
        lines.extend((
            f"[source{index}]",
            f"username64={base64.b64encode(camera.username.encode('utf-8')).decode('ascii')}",
            f"password64={base64.b64encode(camera.password.encode('utf-8')).decode('ascii')}",
            f"uri_sha256={hashlib.sha256(uri.encode('utf-8')).hexdigest()}",
            "",
        ))
    return "\n".join(lines)


def checked_network(name: str, hostname: str, port: int) -> None:
    # Reuse Docker's normal bridge. The host firewall remains in force; this
    # launcher neither edits firewall rules nor publishes container ports.
    if name != "bridge":
        raise ValueError("the existing Docker bridge network is required")
    result = subprocess.run(
        ["docker", "network", "inspect", name], capture_output=True,
        text=True, timeout=8, check=True,
        env=child_environment_without_camera_secrets(),
    )
    info = json.loads(result.stdout)[0]
    if info.get("Name") != name or info.get("Driver") != "bridge":
        raise ValueError("expected Docker's existing bridge network")
    if info.get("EnableIPv6"):
        raise ValueError("IPv6 egress needs separate review; refusing this network")
    if info.get("Internal"):
        raise ValueError("internal network cannot reach the external NVR")
    if not hostname or not 1 <= port <= 65535:
        raise ValueError("invalid configured NVR destination")


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
                 display: str | None, xauth: Path | None, name: str,
                 mosaic_viewer_binary: Path | None = None,
                 mosaic_slot_path: Path | None = None,
                 runtime_limit_seconds: float | None = None,
                 telemetry_enabled: bool = False,
                 shutdown_gdb_dir: Path | None = None,
                 suppress_shutdown_pause: bool = False,
                 mosaic_pixel_provenance: bool = False,
                 latency_measurement: bool = False) -> list[str]:
    mosaic = mosaic_viewer_binary is not None or mosaic_slot_path is not None
    if mosaic and (mosaic_viewer_binary is None or mosaic_slot_path is None):
        raise ValueError("mosaic mode requires both the instrumented binary and private IPC slot")
    if suppress_shutdown_pause and not mosaic:
        raise ValueError("shutdown PAUSE suppression is available only for the isolated mosaic viewer")
    if mosaic_pixel_provenance and not mosaic:
        raise ValueError("mosaic pixel provenance requires the isolated mosaic viewer")
    if latency_measurement and not mosaic:
        raise ValueError("latency measurement requires the isolated mosaic viewer")
    if latency_measurement and not telemetry_enabled:
        raise ValueError("latency measurement requires the existing monitoring telemetry path")
    args = [
        # Keep this uniquely named test container until its state is captured;
        # main() explicitly removes only this owned container after diagnostics.
        "docker", "run", "-i", "--pull=never",
        "--name", name, "--log-driver=none", "--network", network,
        "--gpus", "device=0",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--read-only", "--pids-limit=512",
        "--tmpfs", "/tmp:rw,nosuid,nodev,mode=1777",
        "--tmpfs", "/run/live:rw,nosuid,nodev,noexec,mode=1777",
        "--mount", f"type=bind,src={models_root},dst=/models,readonly",
    ]
    if mosaic:
        assert mosaic_viewer_binary is not None and mosaic_slot_path is not None
        args.extend([
            "--mount", f"type=bind,src={mosaic_viewer_binary},dst=/opt/live/deepstream-app-mosaic,readonly",
            "--mount", f"type=bind,src={mosaic_slot_path.parent},dst={CONTAINER_MOSAIC_PREVIEW_DIR}",
            "-e", "HOME=/tmp", "-e", "XDG_CACHE_HOME=/tmp",
            "-e", "GST_REGISTRY=/tmp/live-gst-registry.bin",
            "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics,display",
            "-e", "V11_UI_MOSAIC_PREVIEW_ENABLE=1",
            "-e", "V11_UI_MOSAIC_PREVIEW_ENABLED=1",
            "-e", f"V11_UI_MOSAIC_PREVIEW_PATH={CONTAINER_MOSAIC_PREVIEW_PATH}",
            "-e", f"V11_UI_MOSAIC_PREVIEW_RATE_FPS={DEFAULT_PUBLISH_RATE_FPS}",
            "-e", "V11_UI_FULLSCREEN_PREVIEW_ENABLE=1",
            "-e", f"V11_UI_FULLSCREEN_PREVIEW_PATH={CONTAINER_MOSAIC_FULLSCREEN_PREVIEW_PATH}",
            "-e", f"V11_UI_FULLSCREEN_REQUEST_PATH={CONTAINER_MOSAIC_FULLSCREEN_REQUEST_PATH}",
            "-e", "SIX_LIVE_AUTH_FILE=/run/live/auth.ini",
        ])
        if telemetry_enabled:
            args.extend(["-e", "SIX_ML_TELEMETRY_ENABLED=1"])
        if mosaic_pixel_provenance:
            args.extend(["-e", "SIX_LIVE_PIXEL_PROVENANCE=1"])
        if latency_measurement:
            args.extend([
                "-e", "SIX_LIVE_LATENCY_SAMPLER=1",
                "-e", "NVDS_ENABLE_LATENCY_MEASUREMENT=1",
                "-e", "NVDS_LATENCY_MEASUREMENT_SILENT=1",
            ])
        if suppress_shutdown_pause:
            args.extend(["-e", "SIX_LIVE_SUPPRESS_SHUTDOWN_PAUSE=1"])
        command = "umask 077; tar -xf - -C /run/live; exec /opt/live/deepstream-app-mosaic -c /run/live/app.txt --tiledtext"
    else:
        if display is None or xauth is None:
            raise ValueError("the normal EGL viewer requires the checked X11 display and Xauthority")
        args.extend([
            "--mount", "type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly",
            "--mount", f"type=bind,src={xauth},dst=/run/xauth,readonly",
            "-e", f"DISPLAY={display}", "-e", "XAUTHORITY=/run/xauth",
            "-e", "HOME=/tmp", "-e", "XDG_CACHE_HOME=/tmp",
            "-e", "GST_REGISTRY=/tmp/live-gst-registry.bin",
            "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics,display",
        ])
        command = "umask 077; tar -xf - -C /run/live; exec deepstream-app -c /run/live/app.txt --tiledtext"
    if shutdown_gdb_dir is not None:
        command = command.replace("exec /opt/live/deepstream-app-mosaic",
                                  "exec python3 /diag/target.py /opt/live/deepstream-app-mosaic")
    if runtime_limit_seconds is not None:
        if runtime_limit_seconds <= 0:
            raise RuntimeError("authorized startup budget exhausted; no camera started")
        # The host owns the one graceful SIGINT. GNU timeout is only the
        # independent hard-kill fallback. --foreground prevents GNU timeout's
        # process-group signal from killing the timeout monitor itself; this
        # preserves the child's abnormal status for Docker to report.
        command = ("exec timeout --foreground --verbose --signal=KILL "
                   f"{runtime_limit_seconds:.3f}s /bin/sh -ec {shlex.quote(command)}")
    args.extend(["--entrypoint", "/bin/sh", image, "-ec", command])
    if shutdown_gdb_dir is not None:
        from scripts.native_shutdown_diagnostics import six_live_debugger_command
        args = six_live_debugger_command(args, image, shutdown_gdb_dir)
    return args


def _budget_timeout(deadline: float | None, cap: float) -> float:
    remaining = cap if deadline is None else min(cap, deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError("authorized wall-clock budget exhausted")
    return remaining


def _timed_shutdown(operation: str, action):
    """Run one teardown action and emit its wall time, including failures."""
    started = time.monotonic()
    try:
        return action()
    finally:
        print(f"SHUTDOWN_TIMING operation={operation} "
              f"elapsed_seconds={time.monotonic() - started:.3f}", flush=True)


def _stop_owned_container(name: str, *, deadline: float | None = None) -> None:
    print(f"DOCKER_SIGNAL event=stop_request_started signal=SIGINT "
          f"monotonic_ns={time.monotonic_ns()}", flush=True)
    result = subprocess.run(["docker", "kill", "--signal", "SIGINT", name],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            timeout=_budget_timeout(deadline, 1),
                            env=child_environment_without_camera_secrets())
    print(f"DOCKER_SIGNAL event=stop_request_completed signal=SIGINT "
          f"returncode={result.returncode} monotonic_ns={time.monotonic_ns()}",
          flush=True)


def _inspect_owned_container(name: str, redactor: CredentialRedactor,
                             *, deadline: float | None = None) -> dict:
    """Read only the container State object before test-owned cleanup."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{json .State}}", name],
            capture_output=True, text=True,
            timeout=_budget_timeout(deadline, 1), check=False,
            env=child_environment_without_camera_secrets(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False,
                "inspection_error": redactor.sanitize_exception(exc)}
    if result.returncode:
        return {"available": False, "inspect_returncode": result.returncode,
                "inspection_error": redactor.sanitize(result.stderr)}
    try:
        state = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"available": False, "inspection_error": "invalid Docker State JSON"}
    fields = ("Status", "Running", "Paused", "Restarting", "OOMKilled", "Dead",
              "ExitCode", "Error", "StartedAt", "FinishedAt", "Pid")
    return {"available": True,
            **{key: redactor.sanitize(state.get(key)) if isinstance(state.get(key), str)
               else state.get(key) for key in fields}}


def _container_absent(name: str, *, prefix: bool = False,
                      deadline: float | None = None) -> bool | None:
    """True/False require a successful Docker query; None means unavailable."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--all", "--filter", f"name={name}", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=_budget_timeout(deadline, 1),
            env=child_environment_without_camera_secrets(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    names = result.stdout.splitlines()
    return not any(item.startswith(name) for item in names) if prefix else name not in names


def _remove_owned_container(name: str, *, deadline: float | None = None) -> bool:
    """Request removal, then verify actual absence, including async completion."""
    _budget_timeout(deadline, 1)  # Do not start another operation after a hard deadline.
    verify_deadline = time.monotonic() + 3
    if deadline is not None:
        verify_deadline = min(verify_deadline, deadline)
    try:
        subprocess.run(
            ["docker", "rm", "--force", name], capture_output=True,
            text=True, timeout=_budget_timeout(verify_deadline, 1),
            env=child_environment_without_camera_secrets(),
        )
    except (OSError, subprocess.SubprocessError):
        pass  # A timed-out CLI does not establish the daemon's final state.
    while time.monotonic() < verify_deadline:
        absent = _container_absent(name, deadline=verify_deadline)
        if absent is True:
            return True
        if absent is None:
            return False  # Daemon unavailability is never evidence of removal.
        time.sleep(max(0.0, min(0.1, verify_deadline - time.monotonic())))
    return False


def _deadline_remove(name: str, deadline: float) -> None:
    """Last-resort host watchdog, independent of input and stdout handling."""
    try:
        _remove_owned_container(name, deadline=deadline)
    except (OSError, subprocess.SubprocessError):
        pass


def _pump_lines(proc: subprocess.Popen, redactor: CredentialRedactor,
                monitor: SourceFpsMonitor,
                diagnostics: ChildOutputDiagnostics,
                telemetry_receiver: NativeTelemetryReceiver | None = None,
                latency_collector: LatencyRunCollector | None = None) -> None:
    if proc.stdout:
        for line in proc.stdout:
            safe = redactor.sanitize_child_output(line)
            if latency_collector is not None:
                latency_collector.ingest_child_line(safe)
            if safe.startswith("SIX_TRACKER_FRAME "):
                if telemetry_receiver is not None:
                    telemetry_receiver.ingest(safe)
                continue
            diagnostics.ingest(safe)
            if telemetry_receiver is not None:
                telemetry_receiver.ingest(safe)
            if latency_collector is not None and safe.startswith("SIX_LATENCY_SAMPLE "):
                # Bounded samples are retained for the private JSON report,
                # rather than emitted as a per-sample terminal stream.
                continue
            accepted_progress = monitor.ingest(safe)
            # The native per-source heartbeat is emitted six times per second;
            # do not print a second status line for every camera sample. Keep
            # the legacy PERF summary for terminals using older app binaries.
            if accepted_progress and safe.startswith("**PERF:"):
                snap = monitor.snapshot()
                if snap["camera_fps"]:
                    status = " ".join(f"{cam}={fps:.1f}fps"
                                      for cam, fps in snap["camera_fps"].items())
                    print(f"LIVE SOURCE FPS: {status}", flush=True)
            print(safe, end="" if safe.endswith("\n") else "\n", flush=True)


def _pump_camera_wall_lines(proc: subprocess.Popen, redactor: CredentialRedactor,
                            diagnostics: CameraWallOutputDiagnostics) -> None:
    """Drain UI output to prevent pipe blockage and retain safe failure context."""
    if proc.stdout:
        for line in proc.stdout:
            diagnostics.ingest(redactor.sanitize_child_output(line))


def _ask_operator(proc: subprocess.Popen | None, monitor: SourceFpsMonitor | None,
                  *, mosaic_preview: bool = False,
                  deadline: float | None = None) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("interactive terminal required to confirm all six live tiles")
    print("Enter START to begin the authorized bounded live run. "
          "The wall-clock budget is already running.", flush=True)
    print("Camera startup follows confirmation; visual approval remains separate.", flush=True)
    deadline = time.monotonic() + 120 if deadline is None else deadline
    response = ""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        exit_code = proc.poll() if proc is not None else None
        if exit_code is not None:
            raise RuntimeError(
                "live viewer exited before operator confirmation; "
                f"docker_run_exit_code={exit_code}"
            )
        ready, _, _ = select.select([sys.stdin], [], [], min(1.0, remaining))
        if ready:
            # select() can report a partial line. readline() would then block
            # past the deadline, so read only bytes already available.
            chunk = os.read(sys.stdin.fileno(), 64)
            if not chunk:
                raise RuntimeError("START terminal closed")
            response += chunk.decode("utf-8", errors="replace")
            if time.monotonic() >= deadline:
                break
            if len(response) > 64:
                raise RuntimeError("invalid START confirmation")
            if "\n" not in response:
                continue
            if response.strip() != "START":
                raise RuntimeError("operator did not confirm six live feeds")
            if monitor is not None and not monitor.snapshot()["ready"]:
                raise RuntimeError("six-source FPS readiness not established; refusing timer start")
            return
    raise RuntimeError("START confirmation deadline expired; no live observation completed")


def configured_cameras(settings) -> list[Camera]:
    """Resolve credentials and the existing low-latency source profile."""
    if settings.rtsp_transport != "tcp":
        raise ValueError("the reviewed camera-only network requires RTSP-over-TCP")
    if not settings.source_drop_on_latency:
        raise ValueError("the live RTSP profile requires drop-on-latency")
    return [
        Camera(
            c.camera_id, c.uri, c.username, c.password,
            latency_ms=(getattr(c, "latency_ms", None)
                        if getattr(c, "latency_ms", None) is not None
                        else settings.source_latency_ms),
            low_latency_mode=bool(getattr(c, "decoder_low_latency_mode", False)),
            extra_surfaces=(getattr(c, "decoder_extra_surfaces", None)
                            if getattr(c, "decoder_extra_surfaces", None) is not None
                            else settings.source_extra_surfaces),
        )
        for c in settings.cameras
    ]


def main(argv: list[str] | None = None) -> int:
    invocation_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path,
                        help="previously validated PeopleNet FP16 batch-6 nvinfer config")
    parser.add_argument("--models-root", type=Path, default=ROOT / ".runtime/models")
    parser.add_argument("--network", default="bridge",
                        help="existing Docker bridge network (default: bridge)")
    parser.add_argument("--expected-engine-sha256", help="known-good batch-6 engine SHA256 from the validated manifest")
    parser.add_argument("--check", action="store_true", help="no Docker/RTSP connection")
    parser.add_argument("--run", action="store_true", help="bounded physical RTSP run")
    parser.add_argument("--monitor", action="store_true",
                        help="daily operation: show the Camera Wall and run until stopped")
    parser.add_argument("--stop-monitoring", action="store_true",
                        help="gracefully stop the verified daily monitoring owner")
    parser.add_argument("--authorize-physical-rtsp", action="store_true")
    parser.add_argument("--duration", type=int,
                        help="total wall-clock budget including startup and cleanup; "
                             "with --monitor, supplying this option enables bounded monitoring")
    parser.add_argument("--mosaic-preview", action="store_true",
                        help="opt into the isolated post-OSD PySide shared-memory mosaic output")
    parser.add_argument("--mosaic-pixel-provenance", action="store_true",
                        help="opt into bounded CAM-05 pixel diagnostics in the isolated native and Qt preview paths")
    parser.add_argument("--mosaic-viewer-binary", type=Path,
                        help="built DeepStream 9.1 instrumented app under .runtime (required with --mosaic-preview)")
    parser.add_argument("--publish-ml-telemetry", action="store_true",
                        help="opt in to host-owned ML IPC publication from a telemetry-capable mosaic viewer")
    parser.add_argument("--measure-latency", action="store_true",
                        help="opt in to bounded DeepStream, mosaic IPC and Qt paint latency evidence")
    parser.add_argument("--source-stall-diagnostics", action="store_true",
                        help="retain bounded source counters and timestamped GStreamer events in failure diagnostics")
    parser.add_argument("--verify-fullscreen", action="store_true",
                        help="preflight and select the existing sequential fullscreen verifier UI")
    parser.add_argument("--fullscreen-camera", dest="fullscreen_cameras", action="append",
                        choices=MOSAIC_CAMERA_IDS,
                        help="camera to include in the sequential fullscreen verifier; repeat to select a subset")
    parser.add_argument("--shutdown-gdb-dir", type=Path,
                        help="isolated bounded shutdown diagnostic only; matching private tooling/symbol bundle")
    args = parser.parse_args(argv)
    pause_suppression_env = os.environ.get("SIX_LIVE_SUPPRESS_SHUTDOWN_PAUSE", "")
    if pause_suppression_env not in {"", "0", "1"}:
        parser.error("SIX_LIVE_SUPPRESS_SHUTDOWN_PAUSE must be 0 or 1")
    suppress_shutdown_pause = pause_suppression_env == "1"
    if suppress_shutdown_pause and not (args.run and args.mosaic_preview):
        parser.error("shutdown PAUSE suppression requires --run with the isolated mosaic viewer")
    if args.shutdown_gdb_dir is not None:
        if args.duration is None or not 31 < args.duration <= 120 or not args.mosaic_viewer_binary:
            parser.error("shutdown GDB requires an isolated mosaic binary and explicit duration 32..120")
        from scripts.native_shutdown_diagnostics import validate_six_live_bundle
        validate_six_live_bundle(args.mosaic_viewer_binary, args.shutdown_gdb_dir)
    bounded_monitor = args.monitor and args.duration is not None
    if args.duration is None:
        args.duration = 600
    if args.stop_monitoring:
        if args.run or args.check or args.monitor:
            parser.error("--stop-monitoring cannot be combined with a start/check mode")
        return stop_monitoring()
    if args.infer_config is None:
        parser.error("--infer-config is required for start/check")
    if args.monitor and (not args.run or not args.mosaic_preview):
        parser.error("--monitor requires --run and the existing mosaic preview")
    if args.verify_fullscreen and not (args.monitor and args.run and args.mosaic_preview):
        parser.error("--verify-fullscreen requires --monitor, --run and --mosaic-preview")
    if args.mosaic_pixel_provenance and not (
            args.monitor and args.run and args.mosaic_preview):
        parser.error("--mosaic-pixel-provenance requires --monitor, --run and --mosaic-preview")
    if (args.verify_fullscreen or args.mosaic_pixel_provenance) and not os.environ.get(
            "V11_UI_FULLSCREEN_SEQUENCE_OUTPUT", "").strip():
        parser.error("verification requires a private V11_UI_FULLSCREEN_SEQUENCE_OUTPUT path")
    if args.fullscreen_cameras and not args.verify_fullscreen:
        parser.error("--fullscreen-camera requires --verify-fullscreen")
    if args.fullscreen_cameras and len(set(args.fullscreen_cameras)) != len(args.fullscreen_cameras):
        parser.error("--fullscreen-camera values must be unique")
    if args.check == args.run:
        parser.error("select exactly one of --check and --run")
    if args.mosaic_preview != (args.mosaic_viewer_binary is not None):
        parser.error("--mosaic-preview and --mosaic-viewer-binary must be supplied together")
    if args.publish_ml_telemetry and not (args.run and args.mosaic_preview):
        parser.error("--publish-ml-telemetry requires --run and --mosaic-preview")
    if args.measure_latency and not (
            args.run and args.monitor and args.mosaic_preview and args.publish_ml_telemetry):
        parser.error("--measure-latency requires --run --monitor --mosaic-preview and --publish-ml-telemetry")
    if args.measure_latency and (args.duration is None or not 32 <= args.duration <= 120):
        parser.error("--measure-latency requires an explicit total duration from 32 to 120 seconds")
    if args.source_stall_diagnostics and not args.run:
        parser.error("--source-stall-diagnostics requires --run")
    if not 1 <= args.duration <= 600:
        parser.error("--duration must be between 1 and 600 seconds")
    bounded_run = not args.monitor or bounded_monitor
    if bounded_run and args.duration <= BOUNDED_SHUTDOWN_RESERVE_SECONDS + 1:
        parser.error(
            "bounded duration must exceed the "
            f"{BOUNDED_SHUTDOWN_RESERVE_SECONDS:g}-second shutdown reserve plus startup margin"
        )

    redactor = CredentialRedactor()
    for key, value in os.environ.items():
        if "PASSWORD" in key.upper() or "RTSP" in key.upper():
            redactor.add_secret(value)

    proc = None
    pump = None
    owned_name = None
    lock = None
    camera_ownership = None
    mosaic_slot = None
    fullscreen_slot = None
    diagnostics = ChildOutputDiagnostics(
        source_stall_diagnostics=args.source_stall_diagnostics)
    latency_collector = LatencyRunCollector() if args.measure_latency else None
    latency_output_dir: Path | None = None
    latency_qt_log_path: Path | None = None
    host_time_namespace_inode: int | None = None
    stage = "load_camera_configuration"
    failure = None
    failure_stage = None
    exit_code_at_detection = None
    container_exit_code = None
    docker_run_exit_code = None
    docker_run_client_killed = False
    cleanup_stop_requested = False
    container_state_before_cleanup = None
    container_removed = None
    run_deadline = None
    container_kill_deadline = None
    watchdog = None
    old_handlers = {}
    shutdown_capture = None
    shutdown_capture_result = {}
    camera_wall = None
    camera_wall_pump = None
    camera_wall_diagnostics = CameraWallOutputDiagnostics()
    telemetry_ipc = None
    telemetry_receiver = None
    telemetry_loop = None
    monitor_owner_written = False
    cleanup_checks = None
    latency_report_failed = False
    child_output_drained = pump is None
    mosaic_binary_sha256 = None
    try:
        settings = load_settings(ROOT / "config/cameras.yaml")
        cameras = configured_cameras(settings)
        for cam in cameras:
            redactor.add_credentials(username=cam.username, password=cam.password, uri=cam.uri)
        app_config, manifest = build_app_config(cameras, mosaic_preview=args.mosaic_preview)
        infer_config = args.infer_config.read_text(encoding="utf-8")
        model_files = validate_model_files(infer_config, args.models_root)
        mosaic_binary = None
        if args.mosaic_preview:
            mosaic_binary = args.mosaic_viewer_binary.resolve(strict=True)
            runtime_root = (ROOT / ".runtime").resolve(strict=True)
            if (not mosaic_binary.is_file() or not os.access(mosaic_binary, os.X_OK)
                    or not mosaic_binary.is_relative_to(runtime_root)):
                raise ValueError("mosaic viewer binary must be executable and staged below .runtime")
                with mosaic_binary.open("rb") as stream:
                    mosaic_binary_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        platform = json.loads(PLATFORM.read_text(encoding="utf-8"))
        models = {key: path.name for key, path in model_files.items()}
        public = {
            "mode": "CHECK" if args.check else "RUN",
            "cameras": manifest["cameras"],
            "channels": manifest["channels"],
            "layout": manifest["layout"],
            "model_file_names_only": models,
            "image": platform["image_tag"],
            "authorization": "per-run physical RTSP authorization required",
            "mosaic_preview": bool(args.mosaic_preview),
            "mosaic_preview_rate_fps": DEFAULT_PUBLISH_RATE_FPS if args.mosaic_preview else None,
        }
        if args.check:
            print(json.dumps(public, indent=2))
            print("CONFIG PREPARATION: PASS; LIVE RTSP NOT STARTED")
            return 0

        stage = "physical_authorization"
        if not args.authorize_physical_rtsp:
            raise RuntimeError("explicit physical RTSP authorization is required")
        expected = (args.expected_engine_sha256 or "").strip().lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise RuntimeError("a verified 64-digit batch-6 engine SHA256 is required for live execution")
        digest = hashlib.sha256(model_files["model-engine-file"].read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError("batch-6 engine hash differs from validated evidence")
        if args.mosaic_preview:
            display, xauth = None, None
            if args.monitor:
                display_preflight()  # The existing host PySide6 window needs X11.
        else:
            display, xauth = display_preflight()
        fullscreen_sequence_output = None
        if args.verify_fullscreen or args.mosaic_pixel_provenance:
            stage = "camera_wall_verifier_preflight"
            fullscreen_sequence_output = _validated_fullscreen_sequence_output(
                os.environ.get("V11_UI_FULLSCREEN_SEQUENCE_OUTPUT")
            )
            _fullscreen_verifier_preflight(
                fullscreen_sequence_output, args.fullscreen_cameras,
                pixel_provenance=args.mosaic_pixel_provenance,
            )
        stage = "live_preflight"
        nvr_sessions_preflight(manifest["nvr_host"], manifest["nvr_port"])
        checked_network(args.network or "", manifest["nvr_host"], manifest["nvr_port"])

        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=7,
            env=child_environment_without_camera_secrets(), check=True,
        )
        if int(gpu.stdout.splitlines()[0].strip()) < 3500:
            raise RuntimeError("insufficient GPU memory; existing workloads take priority")

        if args.measure_latency:
            latency_output_dir, latency_qt_log_path = _new_latency_evidence_directory()
            try:
                host_time_namespace_inode = os.stat("/proc/self/ns/time").st_ino
            except OSError:
                host_time_namespace_inode = None
            if host_time_namespace_inode is None:
                raise RuntimeError("cannot verify host CLOCK_MONOTONIC time namespace")
            print(f"LATENCY_EVIDENCE_DIRECTORY {latency_output_dir}", flush=True)

        # One budget covers confirmation, startup, playback AND shutdown.
        # Stop video early enough for graceful SIGINT, forced removal and reaping.
        startup_deadline = invocation_started + 120
        if bounded_run:
            # Anchor the hard limit before startup/confirmation. Reserve enough
            # time for the 15-second graceful wait plus state/removal checks.
            run_deadline = invocation_started + args.duration
            stop_deadline = run_deadline - BOUNDED_SHUTDOWN_RESERVE_SECONDS
            kill_deadline = run_deadline - min(2.0, args.duration / 4)
            # Host SIGINT begins at stop_deadline. Give native teardown the
            # existing grace interval, then let GNU timeout kill only the
            # child if it is still stuck. This avoids two concurrent SIGINTs
            # and leaves the host's final forced-cleanup margin intact.
            container_kill_deadline = stop_deadline + DAILY_STOP_GRACE_SECONDS
        else:
            stop_deadline = float("inf")
        if threading.current_thread() is threading.main_thread():
            def request_stop(_signum, _frame):
                raise KeyboardInterrupt("operator cancellation")
            stop_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP) if args.monitor else (signal.SIGINT, signal.SIGTERM)
            for sig in stop_signals:
                old_handlers[sig] = signal.signal(sig, request_stop)
        if not args.monitor:
            stage = "operator_confirmation"
            _ask_operator(None, None, mosaic_preview=args.mosaic_preview,
                          deadline=stop_deadline)
            if stop_deadline - time.monotonic() < 1.5:
                raise RuntimeError("insufficient startup budget remaining; no camera started")

        stage = "lock_acquisition"
        lock_file = ROOT / ".runtime" / "six-live-preview.lock"
        lock_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = lock_file.open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Use the same per-camera locks as the previously validated owners.
        # A separate GPU lock alone cannot prevent duplicate NVR sessions.
        camera_ownership = SourceOwnership(manifest["cameras"],
                                           ROOT / ".runtime/camera-owner-locks")
        camera_ownership.__enter__()
        if args.publish_ml_telemetry:
            stage = "telemetry_ipc_claim"
            telemetry_ipc = StagedTelemetry.host_contracts()
            telemetry_receiver = NativeTelemetryReceiver(telemetry_ipc)
            telemetry_ipc.observe_pipeline("starting", timestamp_ns=time.monotonic_ns())
            telemetry_loop = TelemetryPublisherLoop(telemetry_ipc)
            telemetry_loop.start()
        if args.monitor:
            os.fchmod(lock.fileno(), 0o600)
            _write_monitor_owner(lock, run_deadline=run_deadline,
                                 mosaic_preview_enabled=args.mosaic_preview,
                                 telemetry_enabled=args.publish_ml_telemetry)
            monitor_owner_written = True
            stage = "camera_wall_start"
            if fullscreen_sequence_output is None:
                if latency_qt_log_path is None:
                    camera_wall = _start_camera_wall()
                else:
                    camera_wall = _start_camera_wall(
                        preview_latency_log_path=latency_qt_log_path)
            else:
                camera_wall_kwargs = {
                    "fullscreen_sequence_output": fullscreen_sequence_output,
                    "fullscreen_camera_ids": args.fullscreen_cameras,
                    "pixel_provenance": args.mosaic_pixel_provenance,
                    "grid_only": (args.mosaic_pixel_provenance and not args.verify_fullscreen),
                }
                if latency_qt_log_path is not None:
                    camera_wall_kwargs["preview_latency_log_path"] = latency_qt_log_path
                camera_wall = _start_camera_wall(**camera_wall_kwargs)
            if getattr(camera_wall, "stdout", None) is not None:
                camera_wall_pump = threading.Thread(
                    target=_pump_camera_wall_lines,
                    args=(camera_wall, redactor, camera_wall_diagnostics),
                    daemon=True,
                )
                camera_wall_pump.start()
        owned_name = f"ai-surveillance-six-live-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        if args.monitor:
            _write_monitor_owner(lock, container_name=owned_name,
                                 run_deadline=run_deadline,
                                 mosaic_preview_enabled=args.mosaic_preview,
                                 telemetry_enabled=args.publish_ml_telemetry)
        image = platform["image"]
        if args.mosaic_preview:
            mosaic_slot = MosaicPreviewSlot(MOSAIC_PREVIEW_PATH).create()
            fullscreen_slot = FullscreenPreviewSlot(MOSAIC_FULLSCREEN_PREVIEW_PATH).create()
        stage = "container_start"
        runtime_limit = (None if not bounded_run else
                         container_kill_deadline - time.monotonic())
        if runtime_limit is not None and runtime_limit < 1.5:
            raise RuntimeError("insufficient startup budget remaining; no camera started")
        if runtime_limit is not None:
            print("CONTAINER_WATCHDOG pid1=timeout mode=SIGKILL_FALLBACK "
                  f"internal_timeout_configured_seconds={runtime_limit:.3f} "
                  f"internal_kill_deadline_monotonic={container_kill_deadline:.6f} "
                  f"host_stop_deadline_monotonic={stop_deadline:.6f}", flush=True)
        cmd = _docker_args(image, args.models_root.resolve(), args.network,
                           display, xauth, owned_name, mosaic_binary,
                           mosaic_slot.path if mosaic_slot is not None else None,
                           runtime_limit_seconds=runtime_limit,
                           telemetry_enabled=args.publish_ml_telemetry,
                           shutdown_gdb_dir=args.shutdown_gdb_dir,
                           suppress_shutdown_pause=suppress_shutdown_pause,
                           mosaic_pixel_provenance=args.mosaic_pixel_provenance,
                           latency_measurement=args.measure_latency)
        if bounded_run:
            watchdog = threading.Timer(
                max(0.0, kill_deadline - time.monotonic()),
                _deadline_remove, args=(owned_name, run_deadline))
            watchdog.daemon = True
            watchdog.start()
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            env=child_environment_without_camera_secrets(),
            # Keep terminal Ctrl+C from also reaching Docker's signal proxy:
            # the launcher must send exactly one graceful SIGINT to NVIDIA.
            start_new_session=args.monitor,
        )
        monitor = SourceFpsMonitor(min_fps=5.0, max_age_sec=15.0, min_consecutive=2)
        pump = threading.Thread(target=_pump_lines,
                                args=(proc, redactor, monitor, diagnostics,
                                      telemetry_receiver, latency_collector), daemon=True)
        pump.start()
        assert proc.stdin is not None
        # stdin is the only credential delivery channel. No URI in Docker argv.
        auth_config = native_auth_config(cameras) if args.mosaic_preview else None
        proc.stdin.buffer.write(secret_tar(app_config, infer_config, auth_config))
        proc.stdin.close()
        stage = "source_readiness"
        ready_once = False
        while time.monotonic() < stop_deadline:
            if telemetry_loop is not None and telemetry_loop.failure_code:
                raise RuntimeError("ML telemetry IPC publication failed: "
                                   f"{telemetry_loop.failure_code}")
            if camera_wall is not None and camera_wall.poll() is not None:
                if camera_wall.returncode != 0:
                    raise RuntimeError(f"Camera Wall exited; code={camera_wall.returncode}")
                print("Camera Wall closed; stopping monitoring.", flush=True)
                return 0
            exit_code = proc.poll()
            if exit_code is not None:
                raise RuntimeError(
                    "live viewer exited before bounded observation ended; "
                    f"docker_run_exit_code={exit_code}"
                )
            monitor_snapshot = monitor.snapshot()
            if not monitor_snapshot["ready"]:
                if ready_once:
                    diagnostics.record_source_guard_failure(monitor_snapshot)
                    raise RuntimeError("one or more DeepStream source FPS streams stalled or became unmeasurable")
            elif not ready_once:
                ready_once = True
                stage = "observation_window"
                if args.monitor:
                    print("SIX-SOURCE FPS READY; daily monitoring active. "
                          "Close the Camera Wall, press Ctrl+C, or use the stop command.", flush=True)
                else:
                    print("SIX-SOURCE FPS READY; visual approval remains separate. "
                          "The original wall-clock deadline is unchanged.", flush=True)
            if args.monitor and not ready_once and time.monotonic() >= startup_deadline:
                raise RuntimeError("six-source startup did not complete within 120 seconds")
            time.sleep(max(0.0, min(0.1, stop_deadline - time.monotonic())))
        if not ready_once:
            raise RuntimeError("six-source FPS readiness not established within startup budget")
        print("LIVE WINDOW COMPLETE; using reserved shutdown budget.", flush=True)
        return 0
    except (KeyboardInterrupt, OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        if args.monitor and isinstance(exc, KeyboardInterrupt):
            print("Monitoring stop requested; allowing graceful DeepStream shutdown.", flush=True)
            return 0
        failure = exc
        failure_stage = stage
        exit_code_at_detection = proc.poll() if proc is not None else None
        if owned_name is not None:
            container_state_before_cleanup = _inspect_owned_container(
                owned_name, redactor, deadline=run_deadline)
        print(redactor.sanitize_exception(exc), file=sys.stderr)
        return 2
    finally:
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        if args.monitor and owned_name is not None and not bounded_monitor:
            # Daily shutdown has no test-duration pressure: the reference
            # teardown itself sleeps one second before transitioning to NULL.
            run_deadline = time.monotonic() + DAILY_CLEANUP_DEADLINE_SECONDS
            watchdog = threading.Timer(
                DAILY_FORCE_AFTER_SECONDS, _deadline_remove, args=(owned_name, run_deadline)
            )
            watchdog.daemon = True
            watchdog.start()
        # Let Qt close in parallel with native draining, leaving the remaining
        # cleanup budget for actual Docker state verification.
        if camera_wall is not None and camera_wall.poll() is None:
            try:
                _timed_shutdown("camera_wall_terminate", camera_wall.terminate)
            except OSError:
                pass
        if owned_name is not None and (proc is None or proc.poll() is None):
            cleanup_stop_requested = True
            if args.shutdown_gdb_dir is not None:
                from scripts.native_shutdown_diagnostics import start_shutdown_capture
                evidence = args.shutdown_gdb_dir.resolve().parent / ("shutdown-evidence-" + uuid.uuid4().hex)
                evidence.mkdir(mode=0o700)
                shutdown_capture = start_shutdown_capture(
                    owned_name, evidence / "threads.log", diagnostics, proc,
                    run_deadline, shutdown_capture_result)
            try:
                _timed_shutdown(
                    "container_sigint",
                    lambda: _stop_owned_container(owned_name, deadline=run_deadline),
                )
            except subprocess.TimeoutExpired:
                print("DOCKER_SIGNAL event=stop_request result=timeout", file=sys.stderr,
                      flush=True)
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"DOCKER_SIGNAL event=stop_request result=error "
                      f"type={type(exc).__name__}", file=sys.stderr, flush=True)
        if proc is not None:
            try:
                docker_run_exit_code = _timed_shutdown(
                    "docker_run_graceful_wait",
                    lambda: proc.wait(timeout=_budget_timeout(
                        run_deadline, DAILY_STOP_GRACE_SECONDS
                    )),
                )
                print(f"DOCKER_RUN_EXIT code={docker_run_exit_code}", flush=True)
            except subprocess.TimeoutExpired:
                print("DOCKER_RUN_EXIT result=wait_timeout", file=sys.stderr, flush=True)
                # If the bounded run ended at its planned stop point, wait a
                # short, deadline-bounded interval for GNU timeout's KILL-only
                # fallback to terminate the child and expose its real status.
                # For an early failure, keep the existing 15-second grace and
                # proceed to verified host cleanup instead of waiting for a
                # distant watchdog deadline.
                if (bounded_run and container_kill_deadline is not None
                        and proc.poll() is None):
                    watchdog_observe_deadline = min(
                        run_deadline,
                        container_kill_deadline + BOUNDED_WATCHDOG_OBSERVE_SECONDS,
                    )
                    remaining = watchdog_observe_deadline - time.monotonic()
                    if 0 < remaining <= BOUNDED_WATCHDOG_OBSERVE_SECONDS + 0.25:
                        print("DOCKER_RUN_EXIT waiting_for_kill_fallback "
                              f"seconds={remaining:.3f}", flush=True)
                        try:
                            docker_run_exit_code = _timed_shutdown(
                                "docker_run_watchdog_wait",
                                lambda: proc.wait(timeout=_budget_timeout(
                                    run_deadline, remaining
                                )),
                            )
                            print(f"DOCKER_RUN_EXIT code={docker_run_exit_code}", flush=True)
                        except subprocess.TimeoutExpired:
                            print("DOCKER_RUN_EXIT result=watchdog_wait_timeout",
                                  file=sys.stderr, flush=True)
                        except (OSError, subprocess.SubprocessError) as exc:
                            print("DOCKER_RUN_EXIT result=watchdog_wait_error "
                                  f"type={type(exc).__name__}",
                                  file=sys.stderr, flush=True)
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"DOCKER_RUN_EXIT result=wait_error type={type(exc).__name__}",
                      file=sys.stderr, flush=True)
        if shutdown_capture is not None:
            shutdown_capture.join(timeout=_budget_timeout(run_deadline, 0.25))
            print("SHUTDOWN_GDB " + json.dumps(shutdown_capture_result, sort_keys=True), flush=True)
            if shutdown_capture.is_alive() or not shutdown_capture_result.get("ok"):
                failure = RuntimeError("bounded shutdown stack capture failed; see private diagnostic evidence")
                failure_stage = "shutdown_gdb"
        if owned_name is not None:
            if proc is not None:
                native_state = _timed_shutdown(
                    "container_state_inspect",
                    lambda: _inspect_owned_container(owned_name, redactor,
                                                     deadline=run_deadline),
                )
                container_state_before_cleanup = native_state
                if native_state.get("available") and native_state.get("Running") is False:
                    native_exit_code = native_state.get("ExitCode")
                    if isinstance(native_exit_code, int):
                        container_exit_code = native_exit_code
                    print("Graceful DeepStream exit confirmed." if native_state.get("ExitCode") == 0
                          else f"DeepStream container exit code: {native_state.get('ExitCode')}", flush=True)
                    print("DOCKER_STATE " + json.dumps({key: native_state.get(key) for key in
                          ("Status", "Running", "ExitCode", "OOMKilled", "Dead", "Error")},
                          sort_keys=True), flush=True)
                elif native_state.get("available") and native_state.get("Running") is True:
                    print("DeepStream container is still running; forcing owned container after grace period.", flush=True)
                else:
                    print("Native container state unavailable; verifying final cleanup.", flush=True)
            try:
                container_removed = _timed_shutdown(
                    "container_remove_and_verify",
                    lambda: _remove_owned_container(owned_name, deadline=run_deadline),
                )
            except (OSError, subprocess.SubprocessError):
                container_removed = False
        if proc is not None and proc.poll() is None:
            try:
                docker_run_exit_code = _timed_shutdown(
                    "docker_run_reap_after_cleanup",
                    lambda: proc.wait(timeout=_budget_timeout(run_deadline, 1)),
                )
                print(f"DOCKER_RUN_EXIT_AFTER_CLEANUP code={docker_run_exit_code}", flush=True)
            except subprocess.TimeoutExpired:
                # Kill only our Docker CLI, never an unrelated process. Its
                # signal status is not substituted for the container status.
                docker_run_client_killed = True
                proc.kill()
                try:
                    proc.wait(timeout=_budget_timeout(run_deadline, 1))
                except (OSError, subprocess.SubprocessError):
                    pass
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"DOCKER_RUN_EXIT_AFTER_CLEANUP result=wait_error "
                      f"type={type(exc).__name__}", file=sys.stderr, flush=True)
        if pump is not None:
            try:
                pump.join(timeout=_budget_timeout(run_deadline, 0.25))
            except TimeoutError:
                pass
            child_output_drained = not getattr(pump, "is_alive", lambda: False)()
        if camera_wall is not None:
            try:
                camera_wall.wait(timeout=_budget_timeout(run_deadline, 2))
            except (OSError, subprocess.SubprocessError):
                if camera_wall.poll() is None:
                    camera_wall.kill()
                    try:
                        camera_wall.wait(timeout=_budget_timeout(run_deadline, 1))
                    except (OSError, subprocess.SubprocessError):
                        pass
        if camera_wall_pump is not None:
            try:
                camera_wall_pump.join(timeout=_budget_timeout(run_deadline, 0.5))
            except TimeoutError:
                pass
        telemetry_cleanup_ok = True
        if telemetry_loop is not None:
            telemetry_cleanup_ok = _timed_shutdown("ml_telemetry_publish_stop", telemetry_loop.stop)
            print("ML_TELEMETRY_CLEANUP " +
                  ("PASS" if telemetry_cleanup_ok else
                   f"FAIL code={telemetry_loop.failure_code or 'CLEANUP_UNVERIFIED'}"),
                  file=sys.stdout if telemetry_cleanup_ok else sys.stderr, flush=True)
        if owned_name is not None:
            if not container_removed:
                container_removed = _container_absent(owned_name, deadline=run_deadline) is True
            print("OWNED DEEPSTREAM CONTAINER REMOVED" if container_removed else
                  "WARNING: owned DeepStream container removal could not be verified",
                  file=sys.stderr if not container_removed else sys.stdout, flush=True)
        if watchdog is not None and container_removed:
            watchdog.cancel()
        if fullscreen_slot is not None:
            fullscreen_slot.close()
        if mosaic_slot is not None:
            mosaic_slot.close()
        if camera_ownership is not None:
            camera_ownership.__exit__(None, None, None)
        cleanup_checks = _cleanup_check_snapshot(
            container_owned=owned_name is not None,
            container_removed=container_removed,
            proc=proc,
            camera_wall=camera_wall,
            camera_locks_owned=camera_ownership is not None,
            preview_enabled=mosaic_slot is not None,
            run_deadline=run_deadline,
            telemetry_enabled=telemetry_loop is not None,
            telemetry_publisher_stopped=telemetry_cleanup_ok,
        )
        print("CLEANUP_PREDICATES " + json.dumps(cleanup_checks, sort_keys=True),
              file=sys.stdout if cleanup_checks["cleanup_ok"] else sys.stderr,
              flush=True)
        if lock is not None:
            if monitor_owner_written:
                _write_monitor_owner(
                    lock,
                    cleanup_ok=cleanup_checks["cleanup_ok"],
                    container_name=owned_name,
                    run_deadline=run_deadline,
                    cleanup_checks=cleanup_checks,
                    container_state_before_cleanup=container_state_before_cleanup,
                    mosaic_preview_enabled=args.mosaic_preview,
                    telemetry_enabled=args.publish_ml_telemetry,
                )
            lock.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        observed_abnormal = (container_exit_code not in (None, 0) or
                             docker_run_exit_code not in (None, 0) or
                             docker_run_client_killed)
        if observed_abnormal:
            shutdown_detail = (
                "shutdown status: "
                f"container_exit_code={container_exit_code}, "
                f"docker_run_exit_code={docker_run_exit_code}, "
                f"docker_run_client_killed={docker_run_client_killed}"
            )
            failure = (RuntimeError(shutdown_detail) if failure is None else
                       RuntimeError(f"{failure}; {shutdown_detail}"))
            failure_stage = "container_shutdown"
            exit_code_at_detection = (container_exit_code if container_exit_code is not None
                                      else docker_run_exit_code)
            print("RUN_STATUS FAILED stage=container_shutdown " + shutdown_detail,
                  file=sys.stderr, flush=True)
        rtsp_report_failed = False
        if (latency_collector is not None and latency_output_dir is not None
                and latency_qt_log_path is not None and host_time_namespace_inode is not None):
            if (failure is None and not observed_abnormal
                    and cleanup_checks is not None
                    and cleanup_checks.get("cleanup_ok") is True
                    and child_output_drained
                    and diagnostics.null_completed.is_set()
                    and container_exit_code == 0 and docker_run_exit_code == 0):
                latency_status = "completed"
            elif proc is None or not child_output_drained or cleanup_checks is None:
                latency_status = "incomplete"
            else:
                latency_status = "failed"
            try:
                report = _write_latency_run_report(
                    report_dir=latency_output_dir,
                    collector=latency_collector,
                    qt_log_path=latency_qt_log_path,
                    host_time_namespace_inode=host_time_namespace_inode,
                    run_status=latency_status,
                    invocation_started=invocation_started,
                    viewer_binary=mosaic_binary,
                    viewer_sha256=mosaic_binary_sha256,
                    failure_stage=failure_stage,
                    failure=failure,
                    docker_run_exit_code=docker_run_exit_code,
                    container_exit_code=container_exit_code,
                    cleanup_checks=cleanup_checks,
                    child_output_drained=child_output_drained,
                    diagnostics=diagnostics,
                    redactor=redactor,
                )
                print(f"SIX CAMERA LATENCY REPORT: {report}", flush=True)
            except (OSError, ValueError, TypeError, RuntimeError) as report_error:
                latency_report_failed = True
                failure = (RuntimeError("unable to persist private latency report: "
                                        f"{type(report_error).__name__}") if failure is None
                           else RuntimeError(f"{failure}; latency report persistence failed"))
                failure_stage = "latency_report_persistence"
                print("SIX CAMERA LATENCY REPORT: persistence failed "
                      f"({type(report_error).__name__})", file=sys.stderr, flush=True)
        elif latency_collector is not None:
            latency_report_failed = True
            print("SIX CAMERA LATENCY REPORT: output path or host clock domain unavailable",
                  file=sys.stderr, flush=True)
        if suppress_shutdown_pause and proc is not None:
            if (failure is None and not observed_abnormal
                    and cleanup_checks is not None
                    and cleanup_checks.get("cleanup_ok") is True
                    and child_output_drained
                    and container_exit_code == 0
                    and docker_run_exit_code == 0):
                report_status = "completed"
            elif (not child_output_drained or cleanup_checks is None
                  or container_exit_code is None or docker_run_exit_code is None):
                report_status = "incomplete"
            else:
                report_status = "failed"
            try:
                report = _write_rtsp_pause_run_report(
                    report_dir=ROOT / ".runtime" / "six-live-rtsp-pause-reports",
                    run_status=report_status,
                    diagnostics=diagnostics,
                    redactor=redactor,
                    viewer_binary=mosaic_binary,
                    viewer_sha256=mosaic_binary_sha256,
                    invocation_started=invocation_started,
                    docker_run_exit_code=docker_run_exit_code,
                    container_exit_code=container_exit_code,
                    container_state=container_state_before_cleanup,
                    cleanup_checks=cleanup_checks,
                    child_output_drained=child_output_drained,
                    failure_stage=failure_stage,
                    failure=failure,
                    camera_wall_output=camera_wall_diagnostics,
                    camera_wall_evidence_path=fullscreen_sequence_output,
                )
                print(f"SIX LIVE RTSP SHUTDOWN REPORT: {report}", flush=True)
            except (OSError, ValueError, TypeError) as report_error:
                rtsp_report_failed = True
                failure = (RuntimeError("unable to persist private RTSP shutdown report: "
                                        f"{type(report_error).__name__}") if failure is None
                           else RuntimeError(f"{failure}; RTSP shutdown report persistence failed"))
                failure_stage = "rtsp_shutdown_report_persistence"
                print("SIX LIVE RTSP SHUTDOWN REPORT: persistence failed "
                      f"({type(report_error).__name__})", file=sys.stderr, flush=True)
        if failure is not None and proc is not None:
            try:
                report = _write_failure_report(
                    stage=failure_stage or "unknown",
                    exc=failure,
                    redactor=redactor,
                    diagnostics=diagnostics,
                    proc=proc,
                    exit_code_at_detection=exit_code_at_detection,
                    docker_run_exit_code=docker_run_exit_code,
                    docker_run_client_killed=docker_run_client_killed,
                    cleanup_stop_requested=cleanup_stop_requested,
                    container_state_before_cleanup=container_state_before_cleanup,
                    container_removed=container_removed,
                    cleanup_checks=cleanup_checks,
                    camera_wall_output=camera_wall_diagnostics,
                )
                print(f"LIVE FAILURE DIAGNOSTICS: {report}", file=sys.stderr, flush=True)
            except OSError as report_error:
                print("LIVE FAILURE DIAGNOSTICS: unable to persist redacted failure report "
                      f"({type(report_error).__name__})", file=sys.stderr, flush=True)
        if cleanup_checks is not None and not cleanup_checks["cleanup_ok"]:
            return 2
        if (observed_abnormal or rtsp_report_failed or latency_report_failed or
                (shutdown_capture is not None and not shutdown_capture_result.get("ok"))):
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
