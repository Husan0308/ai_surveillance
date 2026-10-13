#!/usr/bin/env python3
"""Interactive host controller for synchronized pair recording and CAM-02 evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader

ROOT = Path(__file__).resolve().parents[1]
CAMERA_PAIRS = {
    "entrance": ("CAM-02", "CAM-05"),
    "main_room": ("CAM-03", "CAM-06"),
}
GREEN_SECONDS = 15
SESSION_SECONDS = 180
SUPPORTED_SESSION_SECONDS = (120, 180)
POST_EXIT_MARGIN_SECONDS = 20
LIFECYCLE_CAMERA = "CAM-02"


class RecordingInterrupted(RuntimeError):
    """Operator requested orderly native EOS/finalization."""


def recorder_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Keep the evidence child on the caller's declared runtime, not stale overrides."""
    env = dict(os.environ if base is None else base)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def declared_python_path(path: Path) -> Path:
    """Make a Python path absolute without dereferencing a venv's python symlink."""
    return Path(os.path.abspath(os.fspath(path)))


def validate_evidence_runtime(python: Path) -> None:
    """Fail before starting capture if the chosen runtime cannot render evidence."""
    result = subprocess.run(
        [str(python), "-B", "-c", "from PySide6.QtCore import QRectF, Qt; from PySide6.QtGui import QImage, QPainter"],
        cwd=ROOT,
        env=recorder_environment(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit={result.returncode}"
        raise RuntimeError(f"evidence recorder runtime cannot import PySide6 ({python}): {detail}")


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


class HumanEvidence:
    def __init__(self, python: Path, output: Path, cameras: tuple[str, str]):
        self.output = output
        self.proc = subprocess.Popen(
            [str(python), "-B", "-u", "-m", "scripts.capture_six_camera_human_validation",
             "--output", str(output), "--cameras", ",".join(cameras)],
            cwd=ROOT, env=recorder_environment(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        self.ready = threading.Event()
        self.failure: list[str] = []
        self._write_lock = threading.Lock()
        self._reader = threading.Thread(target=self._forward, name="human-evidence-status", daemon=True)
        self._reader.start()

    def _forward(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            print(line.rstrip(), flush=True)
            if "HUMAN_VALIDATION_RECORDER_READY" in line:
                self.ready.set()
            if "HUMAN_VALIDATION_CAPTURE_FAIL" in line:
                self.failure.append(line.rstrip())

    def wait_ready(self, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ready.wait(0.1):
                return
            if self.proc.poll() is not None:
                break
        raise RuntimeError("shared-memory tracking evidence recorder did not become ready")

    def mark(self, camera: str, phase: str) -> None:
        if self.proc.poll() is not None or self.proc.stdin is None:
            raise RuntimeError("shared-memory evidence recorder exited before marker")
        with self._write_lock:
            self.proc.stdin.write(f"mark {camera} {phase}\n")
            self.proc.stdin.flush()

    def close(self, timeout: float = 35.0) -> int:
        if self.proc.poll() is None and self.proc.stdin is not None:
            with self._write_lock:
                self.proc.stdin.write("stop\n")
                self.proc.stdin.flush()
                self.proc.stdin.close()
        try:
            result = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            result = self.proc.wait(timeout=5)
        self._reader.join(timeout=2)
        return result


def current_frame(reader: PreviewFrameReader, after_sequence: int | None):
    return reader.read_latest(max_age_sec=1.0, after_sequence=after_sequence, metadata_only=False)


def row_ids(frame) -> set[int]:
    snapshot = frame.tracker_snapshot or {}
    result: set[int] = set()
    for row in snapshot.get("objects", []):
        try:
            result.add(int(row["native_track_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def center_by_id(frame) -> dict[int, tuple[float, float]]:
    snapshot = frame.tracker_snapshot or {}
    result = {}
    for row in snapshot.get("objects", []):
        try:
            native_id = int(row["native_track_id"])
            x1, y1, x2, y2 = (float(value) for value in row["bbox"])
            result[native_id] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        except (KeyError, TypeError, ValueError):
            continue
    return result


def read_status(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def wait_recording_active(status_path: Path, expected_cameras: tuple[str, str],
                          timeout_sec: float = 20.0, interrupted=None) -> dict:
    deadline = time.monotonic() + timeout_sec
    previous_bytes: dict[str, int] = {}
    grew: set[str] = set()
    while time.monotonic() < deadline:
        if interrupted is not None and interrupted():
            raise RecordingInterrupted("operator interrupted during recorder startup")
        status = read_status(status_path)
        if status.get("state") == "FAILED":
            raise RuntimeError("native recorder reported FAILED before active confirmation")
        if status.get("state") == "ACTIVE":
            rows = status.get("cameras", [])
            ready_ids = set()
            for row in rows:
                camera = row.get("camera_id")
                size = int(row.get("video_bytes", 0))
                if (int(row.get("input_frames", 0)) >= 3 and
                        int(row.get("encoded_frames", 0)) >= 3 and size > 0):
                    ready_ids.add(camera)
                if size > previous_bytes.get(camera, 0):
                    grew.add(camera)
                previous_bytes[camera] = size
            expected = set(expected_cameras)
            if ready_ids == expected and grew == expected:
                return status
        time.sleep(0.25)
    raise TimeoutError("both recording branches did not prove active frame receipt and file growth")


def write_operator_summary(path: Path, session: dict, markers: dict, gaps: dict,
                           recovery_ids: dict, evidence_output: Path, errors: list[str]) -> None:
    summary = {
        "schema_version": 1,
        "room": session["room"],
        "session_id": session["session_id"],
        "requested_duration_sec": session.get("duration_sec", SESSION_SECONDS),
        "status": "REVIEW_REQUIRED",
        "calibration_video_validation": "see recording_validation.json",
        "human_review": "REVIEW_REQUIRED; visual classification is not inferred from counts",
        "global_identity": "not emitted by the current local-NvDCF candidate; no cross-camera identity claim",
        "tracking_evidence_directory": str(evidence_output.resolve()),
        "markers": markers,
        "post_exit": {
            camera: {"zero_object_source_frames": count, "required_source_frames": 100,
                    "met": count >= 100}
            for camera, count in gaps.items()
        },
        "positive_native_track_ids": {camera: sorted(ids) for camera, ids in recovery_ids.items()},
        "errors": errors,
    }
    atomic_text(path, json.dumps(summary, indent=2) + "\n")


def run(args) -> int:
    room = args.room
    cameras = CAMERA_PAIRS[room]
    lifecycle_camera = LIFECYCLE_CAMERA if room == "entrance" else cameras[0]
    duration_sec = args.duration_sec
    exit_at_seconds = duration_sec - POST_EXIT_MARGIN_SECONDS
    run_dir = args.run_dir.resolve()
    evidence_output = args.evidence_dir.resolve()
    if not run_dir.is_relative_to(ROOT / ".runtime") or not evidence_output.is_relative_to(ROOT / ".runtime"):
        raise ValueError("controller and tracking evidence paths must remain under .runtime")
    control_path = run_dir / "analytics/calibration-recording-command.tsv"
    status_path = run_dir / "analytics/calibration-recording-status.json"
    calibration_root = ROOT / "calibration_recordings"
    if not control_path.parent.is_dir() or not status_path.is_file():
        raise RuntimeError("the exact live analytics candidate has no active recorder control/status channel")
    before = read_status(status_path)
    if before.get("state") != "IDLE":
        raise RuntimeError(f"recorder must be IDLE before READY; current state={before.get('state')!r}")
    session_id = args.session_id or f"{room}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    directory = calibration_root / room / session_id
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite session directory: {directory}")
    calibration_root.mkdir(parents=True, exist_ok=True)
    (calibration_root / room).mkdir(parents=True, exist_ok=True)
    evidence: HumanEvidence | None = None
    readers = {}
    last_sequence: dict[str, int | None] = {camera: None for camera in cameras}
    last_source_frame: dict[str, int | None] = {camera: None for camera in cameras}
    latest_frames: dict[str, object] = {}
    sequence_gaps = {camera: 0 for camera in cameras}
    zero_frame_runs = {camera: 0 for camera in cameras}
    post_exit_zero_frames = {camera: 0 for camera in cameras}
    positive_ids = {camera: set() for camera in cameras}
    positive_first_ns: dict[str, int] = {}
    first_centers: dict[str, dict[int, tuple[float, float]]] = {camera: {} for camera in cameras}
    movement_marked: set[str] = set()
    markers: dict[str, list[str]] = {camera: [] for camera in cameras}
    errors: list[str] = []
    session_started = False
    exit_prompted = False
    baseline_marked = False
    all_post_exit_marked: set[str] = set()
    start_host_ns = 0
    next_progress = 0
    interrupt_requested = False
    previous_sigint = signal.getsignal(signal.SIGINT)

    def capture_sigint(_signum, _frame) -> None:
        nonlocal interrupt_requested
        interrupt_requested = True

    signal.signal(signal.SIGINT, capture_sigint)

    def mark(camera: str, phase: str) -> None:
        if evidence is None:
            raise RuntimeError("tracking evidence recorder was not initialized")
        evidence.mark(camera, phase)
        markers[camera].append(phase)

    try:
        python = declared_python_path(args.python)
        validate_evidence_runtime(python)
        evidence = HumanEvidence(python, evidence_output, cameras)
        readers = {camera: PreviewFrameReader(
            f"/dev/shm/v11_ui_preview_{camera.lower().replace('-', '')}_v1.bin") for camera in cameras}
        evidence.wait_ready()
        if evidence.proc.poll() is not None:
            raise RuntimeError("tracking evidence recorder exited during startup")
        if interrupt_requested:
            raise RecordingInterrupted("Ctrl+C received before the native recording started")
        command = f"START\t{room}\t{session_id}\t/calibration_recordings/{room}/{session_id}\t{duration_sec}\n"
        atomic_text(control_path, command)
        session_started = True
        active = wait_recording_active(status_path, cameras, interrupted=lambda: interrupt_requested)
        start_host_ns = time.monotonic_ns()
        for camera in cameras:
            mark(camera, "empty-baseline-start")
        print("🔴 REC STARTED — " + " and ".join(cameras) +
              " recording. Hozircha kameraga kirmang.", flush=True)
        print("Both MP4 files are growing; input and encoded frame counters are advancing.", flush=True)
        next_progress = 10

        while True:
            if interrupt_requested:
                raise RecordingInterrupted("Ctrl+C received; requesting orderly native EOS/finalization")
            elapsed = (time.monotonic_ns() - start_host_ns) / 1e9
            if elapsed >= duration_sec:
                break
            for camera, reader in readers.items():
                frame = current_frame(reader, last_sequence[camera])
                if frame is None:
                    continue
                prior = last_sequence[camera]
                if prior is not None and frame.sequence > prior + 1:
                    sequence_gaps[camera] += frame.sequence - prior - 1
                last_sequence[camera] = frame.sequence
                prior_source_frame = last_source_frame[camera]
                last_source_frame[camera] = frame.source_frame_num
                latest_frames[camera] = frame
                snapshot = frame.tracker_snapshot or {}
                if (int(snapshot.get("source_frame_num", -1)) != frame.source_frame_num or
                        int(snapshot.get("pts_ns", -1)) != frame.pts_ns):
                    errors.append(f"{camera} image/metadata frame or PTS mismatch")
                ids = row_ids(frame)
                if elapsed < GREEN_SECONDS and ids:
                    # Capture, but keep the person out until the scheduled movement cue.
                    pass
                elif elapsed >= GREEN_SECONDS and ids:
                    if camera not in positive_ids or not positive_ids[camera]:
                        mark(camera, "person-positive-start")
                        positive_first_ns[camera] = time.monotonic_ns()
                        first_centers[camera] = center_by_id(frame)
                    positive_ids[camera].update(ids)
                    if camera in positive_first_ns and time.monotonic_ns() - positive_first_ns[camera] >= 5_000_000_000:
                        if "person-positive-end" not in markers[camera]:
                            mark(camera, "person-positive-end")
                    centers = center_by_id(frame)
                    snapshot_width = max(1, int(snapshot.get("bbox_coordinate_width") or 1))
                    snapshot_height = max(1, int(snapshot.get("bbox_coordinate_height") or 1))
                    for native_id in ids:
                        first = first_centers[camera].get(native_id)
                        current = centers.get(native_id)
                        if (first and current and
                                ((current[0] - first[0]) / snapshot_width) ** 2 +
                                ((current[1] - first[1]) / snapshot_height) ** 2 >= 0.01 ** 2):
                            if camera not in movement_marked:
                                mark(camera, "movement")
                                movement_marked.add(camera)
                            break
                elif elapsed >= GREEN_SECONDS and camera in positive_first_ns:
                    # The shared recorder remains frame-by-frame evidence; a disappearance is not
                    # marked as final exit until the operator has received the explicit exit cue.
                    pass

                if exit_prompted and camera == lifecycle_camera:
                    if ids:
                        zero_frame_runs[camera] = 0
                        post_exit_zero_frames[camera] = 0
                    else:
                        zero_frame_runs[camera] += 1
                        if (prior_source_frame is not None and
                                frame.source_frame_num == prior_source_frame + 1):
                            post_exit_zero_frames[camera] += 1
                        else:
                            post_exit_zero_frames[camera] = 1
                    if (camera not in all_post_exit_marked and post_exit_zero_frames[camera] >= 100
                            and camera in positive_first_ns):
                        mark(camera, "post-exit")
                        all_post_exit_marked.add(camera)
            elapsed = (time.monotonic_ns() - start_host_ns) / 1e9
            if elapsed >= GREEN_SECONDS and not baseline_marked:
                baseline_marked = True
                for camera in cameras:
                    mark(camera, "empty-baseline-end")
                print("🟢 HARAKATNI BOSHLANG! Entrance ichiga kiring. Kameralar oldida yuring."
                      if room == "entrance" else
                      "🟢 HARAKATNI BOSHLANG! Main Room ichiga kiring. Kameralar oldida yuring.", flush=True)
                print("Walk through both cameras' overlapping view, approach the entrance/door, change direction and distance; re-enter if time remains.", flush=True)
                print("The shared-memory evidence recorder marks frames and PTS; global identity is not available in this local-NvDCF candidate.", flush=True)
            if elapsed >= exit_at_seconds and not exit_prompted:
                exit_prompted = True
                mark(lifecycle_camera, "person-exit")
                print("🟡 CAM-02 HUDUDIDAN CHIQING! Kamera oldidan butunlay chiqing."
                      if room == "entrance" else
                      "🟡 MAIN ROOM HUDUDIDAN CHIQING! Kamera oldidan butunlay chiqing.", flush=True)
                print(f"Stay outside until recording finishes. We are checking for 100 consecutive fresh empty {lifecycle_camera} source frames.", flush=True)
            while elapsed >= next_progress and next_progress < duration_sec:
                state = read_status(status_path)
                per_camera = {row.get("camera_id"): row for row in state.get("cameras", [])}
                summary = " ".join(
                    f"{camera}:{per_camera.get(camera, {}).get('encoded_frames', 0)} frames"
                    for camera in cameras)
                print(f"REC PROGRESS {int(elapsed)}/{duration_sec}s {summary}", flush=True)
                next_progress += 10
            time.sleep(0.01)

        print(f"{duration_sec}-second session timer reached; waiting for MP4 finalization.", flush=True)
        deadline = time.monotonic() + 20
        final_status = {}
        while time.monotonic() < deadline:
            final_status = read_status(status_path)
            if final_status.get("state") in {"COMPLETE", "REVIEW_REQUIRED", "FAILED"}:
                break
            time.sleep(0.2)
        if final_status.get("state") not in {"COMPLETE", "REVIEW_REQUIRED"}:
            raise RuntimeError(f"native recording did not finalize cleanly: {final_status.get('state')!r}")
        print("✅ " + room.replace("_", " ").upper() + " VIDEO FINISHED.", flush=True)
        validation = subprocess.run(
            [str(declared_python_path(args.python)), "-B", str(ROOT / "scripts/validate_calibration_recording.py"),
             str(directory)], cwd=ROOT, capture_output=True, text=True)
        if validation.stdout:
            print(validation.stdout.rstrip(), flush=True)
        if validation.returncode:
            errors.append(validation.stderr.strip() or "calibration video validation failed")
        if lifecycle_camera not in all_post_exit_marked:
            errors.append(f"{lifecycle_camera} did not produce 100 consecutive empty source frames after exit")
        result = evidence.close()
        if result != 0:
            errors.append(f"human tracking evidence recorder exit code {result}")
        summary_path = (directory if directory.exists() else evidence_output)
        summary_path.mkdir(parents=True, exist_ok=True)
        write_operator_summary(summary_path / "operator-validation.json",
                               {"room": room, "session_id": session_id,
                                "duration_sec": duration_sec}, markers,
                               post_exit_zero_frames, positive_ids, evidence_output, errors)
        print(f"Tracking evidence: {evidence_output}", flush=True)
        print(f"Calibration recording: {directory}", flush=True)
        return 0 if not errors else 1
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        cancelled = isinstance(exc, RecordingInterrupted)
        print(f"CALIBRATION_SESSION_{'CANCELLED' if cancelled else 'FAIL'}: {errors[-1]}",
              file=sys.stderr, flush=True)
        if session_started:
            native = read_status(status_path)
            if native.get("state") in {"ARMING", "ACTIVE"}:
                atomic_text(control_path, f"STOP\t{session_id}\n")
                print("STOP requested; waiting for native EOS and MP4 finalization.", flush=True)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                native = read_status(status_path)
                if (native.get("state") in {"COMPLETE", "REVIEW_REQUIRED", "FAILED"} or
                        (native.get("state") == "IDLE" and native.get("last_session_id") != session_id)):
                    break
                time.sleep(0.2)
            errors.append(f"native recorder final state={native.get('state', 'UNKNOWN')}")
        if evidence is not None:
            result = evidence.close()
            if result != 0:
                errors.append(f"human tracking evidence recorder exit code {result}")
        summary_path = directory if session_started or directory.exists() else evidence_output
        summary_path.mkdir(parents=True, exist_ok=True)
        write_operator_summary(summary_path / "operator-validation.json",
                               {"room": room, "session_id": session_id,
                                "duration_sec": duration_sec}, markers,
                               post_exit_zero_frames, positive_ids, evidence_output, errors)
        return 130 if cancelled else 1
    finally:
        for reader in readers.values():
            reader.close()
        signal.signal(signal.SIGINT, previous_sigint)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", choices=tuple(CAMERA_PAIRS), required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="FULL_STACK_OUT for the current single-owner session")
    parser.add_argument("--evidence-dir", type=Path, required=True, help="separate .runtime tracking-evidence directory")
    parser.add_argument("--python", type=Path, default=ROOT / ".runtime/full-stack-venv/bin/python")
    parser.add_argument("--session-id")
    parser.add_argument("--duration-sec", type=int, choices=SUPPORTED_SESSION_SECONDS,
                        default=SESSION_SECONDS)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"CALIBRATION_CONTROLLER_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
