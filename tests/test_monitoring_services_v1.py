from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from services.api_service.app.ml_client import MLServiceUnavailable
from services.camera_v11.monitoring_telemetry_ipc_v1 import (
    DEFAULT_CAMERA_IDS,
    MonitoringTelemetryReader,
    MonitoringTelemetryWriter,
)
import pytest


@pytest.fixture(autouse=True)
def _isolated_service_bind_environment(monkeypatch):
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("ML_HOST", "127.0.0.1")
    monkeypatch.delenv("SURVEILLANCE_API_TOKEN", raising=False)


def live_payload() -> dict:
    rows = [
        {
            "camera_id": camera_id,
            "online": True,
            "current_state": "ONLINE",
            "age_since_last_decoded_frame_ms": 10.0,
            "age_since_last_pgie_frame_ms": 12.0,
            "age_since_last_tracker_output_ms": 14.0,
            "age_since_last_preview_ms": 16.0,
            "source_fps": 20.0,
            "render_fps": 19.0,
            "infer_hz": 2.0,
            "queue_depth": 0,
            "detector_drops": 0,
            "current_box_count": index,
            "positive_inferences": 1,
            "detections_total": 1,
            "result_age_ms": 20.0,
            "infer_p95_ms": 5.0,
            "render_gap_p95_ms": 55.0,
            "reconnects": None,
            "copy_errors": 0,
            "infer_errors": 0,
            "meta_errors": 0,
            "pipeline_errors": None,
            "warnings": 0,
            "preview_exported": 1,
            "preview_errors": 0,
            "preview_fps": 15.0,
            "preview_age_ms": 10.0,
        }
        for index, camera_id in enumerate(DEFAULT_CAMERA_IDS)
    ]
    return {
        "runtime": {
            "status": "live",
            "camera_count": 6,
            "rtsp_source_count": 6,
            "ui_rtsp_extra": 0,
            "shared_trt_workers": 1,
            "detector_enabled": True,
            "shared_pipeline_status": "healthy",
            "six_camera_ready": True,
            "uptime_sec": 10.0,
            "warnings_total": 0,
            "errors_total": 0,
        },
        "cameras": rows,
    }


def written_snapshot(tmp_path: Path) -> dict:
    path = tmp_path / "telemetry.json"
    MonitoringTelemetryWriter(path).publish(live_payload())
    return MonitoringTelemetryReader(path).read()


def test_ml_health_and_monitoring_endpoint_are_telemetry_only(tmp_path: Path, monkeypatch) -> None:
    import services.ml_service.app.main as ml_main

    path = tmp_path / "telemetry.json"
    MonitoringTelemetryWriter(path).publish(live_payload())
    monkeypatch.setattr(ml_main, "telemetry", MonitoringTelemetryReader(path))
    monkeypatch.setattr(
        ml_main.room_pair_state, "readiness",
        lambda: {"status": "ready", "ready": True},
    )
    with TestClient(ml_main.app) as client:
        health = client.get("/health").json()
        assert {key: health[key] for key in (
            "service", "status", "monitoring_status", "camera_count", "online_camera_count"
        )} == {
            "service": "ml_service",
            "status": "ok",
            "monitoring_status": "fresh",
            "camera_count": 6,
            "online_camera_count": 6,
        }
        assert "room_pair_readiness" in health
        assert health["process_status"] == "running"
        assert health["analytics_status"] == "ready"
        assert health["six_camera_readiness"] == "ready"
        assert len(health["cameras"]) == 6
        response = client.get("/api/v1/monitoring/snapshot")
        assert response.status_code == 200
        assert response.json()["runtime"]["rtsp_source_count"] == 6
        assert client.get("/video/CAM-01").status_code == 503


def test_ml_process_stays_healthy_when_one_camera_stage_is_offline(tmp_path: Path, monkeypatch) -> None:
    import services.ml_service.app.main as ml_main

    payload = live_payload()
    payload["cameras"][4]["online"] = False
    payload["cameras"][4]["current_state"] = "BACKOFF"
    payload["cameras"][4]["age_since_last_decoded_frame_ms"] = 16_000.0
    path = tmp_path / "telemetry.json"
    MonitoringTelemetryWriter(path).publish(payload)
    monkeypatch.setattr(ml_main, "telemetry", MonitoringTelemetryReader(path))
    with TestClient(ml_main.app) as client:
        health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["process_status"] == "running"
    assert health["analytics_status"] == "degraded"
    assert health["six_camera_readiness"] == "not_ready"
    assert health["online_camera_count"] == 5
    assert health["cameras"][4]["current_state"] == "BACKOFF"


def test_ml_readiness_fails_closed_when_stage_age_missing(tmp_path: Path, monkeypatch) -> None:
    import services.ml_service.app.main as ml_main

    payload = live_payload()
    payload["cameras"][0].pop("age_since_last_tracker_output_ms")
    path = tmp_path / "telemetry.json"
    MonitoringTelemetryWriter(path).publish(payload)
    monkeypatch.setattr(ml_main, "telemetry", MonitoringTelemetryReader(path))
    with TestClient(ml_main.app) as client:
        health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["analytics_status"] == "degraded"
    assert health["six_camera_readiness"] == "not_ready"


def test_ml_stage_freshness_rejects_nonfinite_and_boolean_ages():
    from services.ml_service.app.main import _camera_stages_fresh

    row = live_payload()["cameras"][0]
    assert _camera_stages_fresh(row)
    row["age_since_last_decoded_frame_ms"] = float("nan")
    assert not _camera_stages_fresh(row)
    row["age_since_last_decoded_frame_ms"] = True
    assert not _camera_stages_fresh(row)


class FakeMLClient:
    def __init__(self, payload: dict | None = None, error: str | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls = 0

    async def monitoring_snapshot(self) -> dict:
        self.calls += 1
        if self.error:
            raise MLServiceUnavailable(self.error)
        result = json.loads(json.dumps(self.payload))
        result["sequence"] += self.calls
        result["generated_monotonic_ns"] = time.monotonic_ns()
        result["generated_epoch_ms"] = time.time_ns() // 1_000_000
        return result

    async def tracked_objects(self) -> dict:
        if self.error:
            raise MLServiceUnavailable(self.error)
        return {"status": "fresh", "supported": True, "tracked_objects": []}

    async def close(self) -> None:
        return None


def test_api_proxy_websocket_latest_payload_and_clean_disconnect(tmp_path: Path) -> None:
    import services.api_service.app.main as api_main

    snapshot = written_snapshot(tmp_path)
    with TestClient(api_main.app) as client:
        fake = FakeMLClient(snapshot)
        client.app.state.ml_client = fake
        response = client.get("/api/v1/monitoring/snapshot")
        assert response.status_code == 200
        assert len(response.json()["cameras"]) == 6
        tracked = client.get("/api/v1/tracked-objects")
        assert tracked.status_code == 200 and tracked.json()["supported"] is True
        sequences = []
        with client.websocket_connect("/ws/v1/monitoring") as websocket:
            for _ in range(5):
                message = websocket.receive_json()
                assert [row["camera_id"] for row in message["cameras"]] == list(DEFAULT_CAMERA_IDS)
                assert "image" not in json.dumps(message).lower()
                sequences.append(message["sequence"])
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == 5


def test_api_returns_degraded_contract_when_ml_temporarily_unavailable() -> None:
    import services.api_service.app.main as api_main

    with TestClient(api_main.app) as client:
        client.app.state.ml_client = FakeMLClient(error="connection refused")
        response = client.get("/api/v1/monitoring/snapshot")
        assert response.status_code == 200
        assert response.json()["runtime"]["status"] == "degraded"
        with client.websocket_connect("/ws/v1/monitoring") as websocket:
            message = websocket.receive_json()
            assert message["runtime"]["status"] == "degraded"
            assert all(not row["online"] for row in message["cameras"])
