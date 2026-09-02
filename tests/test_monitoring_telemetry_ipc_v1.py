from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from services.camera_v11.monitoring_telemetry_ipc_v1 import (
    DEFAULT_CAMERA_IDS,
    MAX_SNAPSHOT_BYTES,
    MonitoringTelemetryReader,
    MonitoringTelemetryWriter,
    RuntimeMonitoringTelemetryPublisher,
)


def payload(camera_ids=DEFAULT_CAMERA_IDS):
    cameras = []
    for camera_id in camera_ids:
        cameras.append(
            {
                "camera_id": camera_id,
                "online": True,
                "source_fps": 20.0,
                "render_fps": 19.0,
                "infer_hz": 2.0,
                "queue_depth": 0,
                "detector_drops": 1,
                "current_box_count": 2,
                "positive_inferences": 3,
                "detections_total": 7,
                "result_age_ms": 10.0,
                "infer_p95_ms": 8.0,
                "render_gap_p95_ms": 55.0,
                "reconnects": None,
                "copy_errors": 0,
                "infer_errors": 0,
                "meta_errors": 0,
                "pipeline_errors": None,
                "warnings": 0,
                "preview_exported": 9,
                "preview_errors": 0,
                "preview_fps": 15.0,
                "preview_age_ms": 15.0,
            }
        )
    return {
        "runtime": {
            "status": "live",
            "camera_count": len(cameras),
            "rtsp_source_count": len(cameras),
            "ui_rtsp_extra": 0,
            "shared_trt_workers": 1,
            "detector_enabled": True,
            "uptime_sec": 5.0,
            "warnings_total": 0,
            "errors_total": 0,
        },
        "cameras": cameras,
    }


def test_atomic_latest_snapshot_and_six_camera_schema(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.json"
    writer = MonitoringTelemetryWriter(path)
    reader = MonitoringTelemetryReader(path)
    assert writer.publish(payload()) == 1
    first = reader.read()
    assert first["telemetry_status"] == "fresh"
    assert [row["camera_id"] for row in first["cameras"]] == list(DEFAULT_CAMERA_IDS)
    assert first["runtime"]["rtsp_source_count"] == 6
    assert first["runtime"]["shared_trt_workers"] == 1
    assert path.stat().st_size < MAX_SNAPSHOT_BYTES
    assert writer.publish(payload()) == 2
    assert reader.read()["sequence"] == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_reader_fresh_stale_offline_missing_and_explicit_stop(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.json"
    writer = MonitoringTelemetryWriter(path)
    writer.publish(payload())
    generated = json.loads(path.read_text())["generated_monotonic_ns"]
    reader = MonitoringTelemetryReader(path)
    assert reader.read(now_monotonic_ns=generated + 2_000_000_000)["telemetry_status"] == "fresh"
    stale = reader.read(now_monotonic_ns=generated + 3_000_000_000)
    assert stale["telemetry_status"] == "stale"
    assert not any(row["online"] for row in stale["cameras"])
    expired = reader.read(now_monotonic_ns=generated + 6_000_000_000)
    assert expired["telemetry_status"] == "offline"
    path.unlink()
    missing = reader.read()
    assert missing["runtime"]["status"] == "offline"
    assert all(row["source_fps"] is None for row in missing["cameras"])

    stopped_payload = payload()
    stopped_payload["runtime"]["status"] = "stopped"
    writer.publish(stopped_payload)
    assert reader.read()["telemetry_status"] == "offline"


def test_atomic_replace_never_exposes_partial_json(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.json"
    writer = MonitoringTelemetryWriter(path)
    reader = MonitoringTelemetryReader(path)
    writer.publish(payload())
    failures: list[str] = []

    def publish_many() -> None:
        for _ in range(100):
            writer.publish(payload())

    thread = threading.Thread(target=publish_many)
    thread.start()
    while thread.is_alive():
        current = reader.read()
        if current["telemetry_status"] != "fresh" or len(current["cameras"]) != 6:
            failures.append(str(current.get("reason")))
    thread.join()
    assert not failures
    assert reader.read()["sequence"] == 101


def _fake_runtime():
    now = time.monotonic()
    states = {}
    for index, camera_id in enumerate(DEFAULT_CAMERA_IDS):
        states[camera_id] = SimpleNamespace(
            decoded=100 + index,
            rendered=90 + index,
            infer_completed=10,
            infer_pending=False,
            infer_gate_drops=2,
            positive_buffers=4,
            detections_total=8,
            latest_snapshot=SimpleNamespace(completed_mono=now, boxes=((1, 2, 3, 4, .9),)),
            infer_roundtrip_ms=deque([6.0, 8.0]),
            render_gap_ms=deque([40.0, 55.0]),
            last_render_mono=now,
            copy_errors=0,
            infer_errors=0,
            meta_errors=0,
            warnings=0,
        )
    return SimpleNamespace(
        lock=threading.RLock(),
        errors=0,
        detector_enabled=True,
        detector=SimpleNamespace(process=SimpleNamespace(poll=lambda: None)),
        camera_ids=DEFAULT_CAMERA_IDS,
        states=states,
        ui_preview_exported={camera_id: 10 for camera_id in DEFAULT_CAMERA_IDS},
        ui_preview_errors={camera_id: 0 for camera_id in DEFAULT_CAMERA_IDS},
        ui_preview_last_mono={camera_id: now for camera_id in DEFAULT_CAMERA_IDS},
        box_stale_sec=0.8,
    )


def test_runtime_publisher_reports_measured_deltas_and_null_unavailable(tmp_path: Path) -> None:
    runtime = _fake_runtime()
    publisher = RuntimeMonitoringTelemetryPublisher(runtime, tmp_path / "telemetry.json")
    publisher.publish_once()
    time.sleep(0.01)
    for state in runtime.states.values():
        state.decoded += 1
        state.rendered += 1
        state.infer_completed += 1
    for camera_id in DEFAULT_CAMERA_IDS:
        runtime.ui_preview_exported[camera_id] += 1
    publisher.publish_once()
    result = MonitoringTelemetryReader(tmp_path / "telemetry.json").read()
    assert result["runtime"]["camera_count"] == 6
    assert result["runtime"]["shared_trt_workers"] == 1
    assert all(row["source_fps"] > 0 for row in result["cameras"])
    assert all(row["render_fps"] > 0 for row in result["cameras"])
    assert all(row["current_box_count"] == 1 for row in result["cameras"])
    assert all(row["reconnects"] is None for row in result["cameras"])
    assert all(row["pipeline_errors"] is None for row in result["cameras"])
