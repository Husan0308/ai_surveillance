from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
DEFAULT_PATH = "/dev/shm/v11_monitoring_telemetry_v1.json"
DEFAULT_CAMERA_IDS = tuple(f"CAM-{index:02d}" for index in range(1, 7))
MAX_SNAPSHOT_BYTES = 256 * 1024
FRESH_MAX_AGE_SEC = 2.0
STALE_MAX_AGE_SEC = 5.0


class MonitoringTelemetryError(ValueError):
    pass


def telemetry_path() -> str:
    return os.environ.get("V11_MONITORING_TELEMETRY_PATH", DEFAULT_PATH)


def _nullable_camera(camera_id: str) -> dict[str, Any]:
    return {
        "camera_id": camera_id,
        "online": False,
        "source_fps": None,
        "render_fps": None,
        "infer_hz": None,
        "queue_depth": None,
        "detector_drops": None,
        "current_box_count": None,
        "positive_inferences": None,
        "detections_total": None,
        "result_age_ms": None,
        "infer_p95_ms": None,
        "render_gap_p95_ms": None,
        "reconnects": None,
        "copy_errors": None,
        "infer_errors": None,
        "meta_errors": None,
        "pipeline_errors": None,
        "warnings": None,
        "preview_exported": None,
        "preview_errors": None,
        "preview_fps": None,
        "preview_age_ms": None,
    }


def offline_snapshot(
    camera_ids: Iterable[str] = DEFAULT_CAMERA_IDS,
    *,
    status: str = "offline",
    reason: str = "telemetry_missing",
    sequence: int = 0,
    age_ms: float | None = None,
) -> dict[str, Any]:
    ids = tuple(camera_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "generated_monotonic_ns": None,
        "generated_epoch_ms": None,
        "snapshot_age_ms": age_ms,
        "telemetry_status": status,
        "reason": reason,
        "runtime": {
            "status": status,
            "camera_count": len(ids),
            "rtsp_source_count": None,
            "ui_rtsp_extra": 0,
            "shared_trt_workers": None,
            "detector_enabled": None,
            "uptime_sec": None,
            "warnings_total": None,
            "errors_total": None,
        },
        "cameras": [_nullable_camera(camera_id) for camera_id in ids],
    }


def validate_snapshot(
    payload: Any,
    *,
    expected_camera_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise MonitoringTelemetryError("snapshot must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise MonitoringTelemetryError("unsupported schema_version")
    sequence = payload.get("sequence")
    if not isinstance(sequence, int) or sequence < 0:
        raise MonitoringTelemetryError("sequence must be a non-negative integer")
    for key in ("generated_monotonic_ns", "generated_epoch_ms"):
        value = payload.get(key)
        if not isinstance(value, int) or value <= 0:
            raise MonitoringTelemetryError(f"{key} must be a positive integer")
    runtime = payload.get("runtime")
    cameras = payload.get("cameras")
    if not isinstance(runtime, dict) or not isinstance(cameras, list):
        raise MonitoringTelemetryError("runtime and cameras are required")
    ids: list[str] = []
    for row in cameras:
        if not isinstance(row, dict):
            raise MonitoringTelemetryError("camera entries must be objects")
        camera_id = row.get("camera_id")
        if not isinstance(camera_id, str) or not camera_id:
            raise MonitoringTelemetryError("camera_id is required")
        ids.append(camera_id)
    if len(ids) != len(set(ids)):
        raise MonitoringTelemetryError("duplicate camera_id")
    if expected_camera_ids is not None and tuple(ids) != tuple(expected_camera_ids):
        raise MonitoringTelemetryError("unexpected camera IDs")
    if runtime.get("camera_count") != len(cameras):
        raise MonitoringTelemetryError("runtime camera_count mismatch")
    return payload


class MonitoringTelemetryWriter:
    """Bounded atomic latest-only JSON publisher."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path or telemetry_path())
        self.sequence = 0
        try:
            previous = validate_snapshot(json.loads(self.path.read_text(encoding="utf-8")))
            self.sequence = previous["sequence"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, MonitoringTelemetryError):
            pass

    def publish(self, payload: dict[str, Any]) -> int:
        next_sequence = self.sequence + 1
        document = dict(payload)
        document["schema_version"] = SCHEMA_VERSION
        document["sequence"] = next_sequence
        document["generated_monotonic_ns"] = time.monotonic_ns()
        document["generated_epoch_ms"] = time.time_ns() // 1_000_000
        validate_snapshot(document)
        encoded = json.dumps(
            document, ensure_ascii=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if len(encoded) > MAX_SNAPSHOT_BYTES:
            raise MonitoringTelemetryError(
                f"snapshot exceeds {MAX_SNAPSHOT_BYTES} byte limit"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            with open(temporary, "wb") as stream:
                stream.write(encoded)
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        self.sequence = next_sequence
        return next_sequence


class MonitoringTelemetryReader:
    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        camera_ids: Iterable[str] = DEFAULT_CAMERA_IDS,
        fresh_max_age_sec: float = FRESH_MAX_AGE_SEC,
        stale_max_age_sec: float = STALE_MAX_AGE_SEC,
    ) -> None:
        self.path = Path(path or telemetry_path())
        self.camera_ids = tuple(camera_ids)
        self.fresh_max_age_sec = fresh_max_age_sec
        self.stale_max_age_sec = stale_max_age_sec

    def read(self, *, now_monotonic_ns: int | None = None) -> dict[str, Any]:
        try:
            raw = self.path.read_bytes()
            if not raw or len(raw) > MAX_SNAPSHOT_BYTES:
                raise MonitoringTelemetryError("invalid snapshot size")
            payload = validate_snapshot(
                json.loads(raw.decode("utf-8")), expected_camera_ids=self.camera_ids
            )
        except FileNotFoundError:
            return offline_snapshot(self.camera_ids)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, MonitoringTelemetryError) as exc:
            return offline_snapshot(
                self.camera_ids, status="offline", reason=f"telemetry_invalid:{type(exc).__name__}"
            )

        now_ns = now_monotonic_ns if now_monotonic_ns is not None else time.monotonic_ns()
        if payload["generated_monotonic_ns"] > now_ns + 1_000_000_000:
            return offline_snapshot(
                self.camera_ids, status="offline", reason="telemetry_timestamp_in_future"
            )
        age_sec = max(0.0, (now_ns - payload["generated_monotonic_ns"]) / 1_000_000_000.0)
        explicit_status = payload["runtime"].get("status")
        if explicit_status in {"offline", "stopped"}:
            status = "offline"
        elif age_sec <= self.fresh_max_age_sec:
            status = "fresh"
        elif age_sec <= self.stale_max_age_sec:
            status = "stale"
        else:
            status = "offline"

        result = dict(payload)
        result["snapshot_age_ms"] = round(age_sec * 1000.0, 3)
        result["telemetry_status"] = status
        runtime = dict(result["runtime"])
        runtime["status"] = "live" if status == "fresh" else status
        result["runtime"] = runtime
        if status != "fresh":
            result["reason"] = "telemetry_stale" if status == "stale" else "telemetry_expired"
            result["cameras"] = [dict(row, online=False) for row in result["cameras"]]
        return result



def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


class RuntimeMonitoringTelemetryPublisher:
    """Publishes a lightweight view of the additive V11 UI runtime at 2 Hz."""

    def __init__(self, runtime: Any, path: str | os.PathLike[str] | None = None) -> None:
        self.runtime = runtime
        self.writer = MonitoringTelemetryWriter(path)
        self.period_sec = max(
            0.25, min(2.0, float(os.environ.get("V11_MONITORING_TELEMETRY_PERIOD_SEC", "0.5")))
        )
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.started_mono = time.monotonic()
        self.last_sample_mono = self.started_mono
        self.last_counts: dict[str, tuple[int, int, int, int]] = {}
        self.publish_errors = 0

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(
            target=self._loop, name="v11-monitoring-telemetry", daemon=False
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        try:
            self.publish_once(runtime_status="stopped")
        except Exception:
            self.publish_errors += 1

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            before = time.monotonic()
            try:
                self.publish_once()
            except Exception as exc:
                self.publish_errors += 1
                count = self.publish_errors
                if count <= 3 or count % 100 == 0:
                    print(
                        "CAMERA_V11_MONITORING_TELEMETRY "
                        f"warning={type(exc).__name__}:{exc} errors={count}",
                        flush=True,
                    )
            remaining = self.period_sec - (time.monotonic() - before)
            self.stop_event.wait(max(0.01, remaining))

    def publish_once(self, *, runtime_status: str = "live") -> int:
        runtime = self.runtime
        now = time.monotonic()
        dt = max(1e-6, now - self.last_sample_mono)
        with runtime.lock:
            pipeline_errors = int(runtime.errors)
            detector_enabled = bool(runtime.detector_enabled)
            worker_alive = bool(
                runtime.detector is not None
                and runtime.detector.process is not None
                and runtime.detector.process.poll() is None
            )
            copied: list[dict[str, Any]] = []
            current_counts: dict[str, tuple[int, int, int, int]] = {}
            for camera_id in runtime.camera_ids:
                state = runtime.states[camera_id]
                snapshot = state.latest_snapshot
                preview_exported = int(runtime.ui_preview_exported.get(camera_id, 0))
                counts = (
                    int(state.decoded),
                    int(state.rendered),
                    int(state.infer_completed),
                    preview_exported,
                )
                current_counts[camera_id] = counts
                copied.append(
                    {
                        "camera_id": camera_id,
                        "counts": counts,
                        "last_render_mono": state.last_render_mono,
                        "snapshot_completed_mono": float(snapshot.completed_mono),
                        "snapshot_box_count": len(snapshot.boxes),
                        "queue_depth": 1 if state.infer_pending else 0,
                        "detector_drops": int(state.infer_gate_drops),
                        "positive_inferences": int(state.positive_buffers),
                        "detections_total": int(state.detections_total),
                        "infer_roundtrip_ms": list(state.infer_roundtrip_ms),
                        "render_gap_ms": list(state.render_gap_ms),
                        "copy_errors": int(state.copy_errors),
                        "infer_errors": int(state.infer_errors),
                        "meta_errors": int(state.meta_errors),
                        "warnings": int(state.warnings),
                        "preview_errors": int(runtime.ui_preview_errors.get(camera_id, 0)),
                        "preview_last_mono": float(runtime.ui_preview_last_mono.get(camera_id, 0.0)),
                    }
                )

        rows: list[dict[str, Any]] = []
        for item in copied:
            camera_id = item["camera_id"]
            counts = item.pop("counts")
            previous = self.last_counts.get(camera_id)
            source_fps = render_fps = infer_hz = preview_fps = None
            if previous is not None:
                source_fps = max(0.0, (counts[0] - previous[0]) / dt)
                render_fps = max(0.0, (counts[1] - previous[1]) / dt)
                infer_hz = max(0.0, (counts[2] - previous[2]) / dt)
                preview_fps = max(0.0, (counts[3] - previous[3]) / dt)
            last_render = item["last_render_mono"]
            render_age = now - last_render if last_render is not None else None
            completed = item["snapshot_completed_mono"]
            result_age_ms = (now - completed) * 1000.0 if completed > 0.0 else None
            if completed <= 0.0:
                current_boxes = None
            elif (now - completed) <= runtime.box_stale_sec:
                current_boxes = item["snapshot_box_count"]
            else:
                current_boxes = 0
            preview_last = item["preview_last_mono"]
            preview_age_ms = (now - preview_last) * 1000.0 if preview_last > 0.0 else None
            online = bool(
                runtime_status == "live"
                and render_age is not None
                and render_age <= FRESH_MAX_AGE_SEC
                and source_fps is not None
                and source_fps > 0.0
            )
            rows.append(
                {
                    "camera_id": camera_id,
                    "online": online,
                    "source_fps": source_fps,
                    "render_fps": render_fps,
                    "infer_hz": infer_hz,
                    "queue_depth": item["queue_depth"],
                    "detector_drops": item["detector_drops"],
                    "current_box_count": current_boxes,
                    "positive_inferences": item["positive_inferences"],
                    "detections_total": item["detections_total"],
                    "result_age_ms": result_age_ms,
                    "infer_p95_ms": _percentile(item["infer_roundtrip_ms"], 0.95),
                    "render_gap_p95_ms": _percentile(item["render_gap_ms"], 0.95),
                    "reconnects": None,
                    "copy_errors": item["copy_errors"],
                    "infer_errors": item["infer_errors"],
                    "meta_errors": item["meta_errors"],
                    "pipeline_errors": None,
                    "warnings": item["warnings"],
                    "preview_exported": counts[3],
                    "preview_errors": item["preview_errors"],
                    "preview_fps": preview_fps,
                    "preview_age_ms": preview_age_ms,
                }
            )

        self.last_counts = current_counts
        self.last_sample_mono = now
        return self.writer.publish(
            {
                "runtime": {
                    "status": runtime_status,
                    "camera_count": len(rows),
                    "rtsp_source_count": len(rows),
                    "ui_rtsp_extra": 0,
                    "shared_trt_workers": int(worker_alive),
                    "detector_enabled": detector_enabled,
                    "uptime_sec": max(0.0, now - self.started_mono),
                    "warnings_total": sum(row["warnings"] for row in rows),
                    "errors_total": pipeline_errors
                    + sum(
                        row["copy_errors"] + row["infer_errors"] + row["meta_errors"]
                        + row["preview_errors"]
                        for row in rows
                    )
                    + self.publish_errors,
                },
                "cameras": rows,
            }
        )
