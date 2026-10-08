from __future__ import annotations

from pathlib import Path
import socket
import time

import pytest
from fastapi.testclient import TestClient

from services.shared.deployment import load_deployment
from services.camera_v11.preview_monitoring_bridge import PreviewMonitoringBridge
from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameWriter


def test_deployment_defaults_and_derived_overrides(monkeypatch):
    for key in ("API_HOST", "API_PORT", "ML_HOST", "ML_PORT", "ML_SERVICE_URL",
                "FRONTEND_API_BASE_URL", "FRONTEND_ML_VIDEO_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    config = load_deployment()
    assert (config.api_port, config.ml_port) == (8100, 8101)
    assert config.api_host == config.ml_host == "0.0.0.0"
    assert config.monitoring_ws_url == "ws://127.0.0.1:8100/ws/v1/monitoring"
    monkeypatch.setenv("API_PORT", "18100")
    monkeypatch.setenv("ML_PORT", "18101")
    config = load_deployment()
    assert config.ml_service_url == "http://127.0.0.1:18101"
    assert config.frontend_api_base_url == "http://127.0.0.1:18100"
    monkeypatch.setenv("FRONTEND_API_BASE_URL", "https://example.org/surveillance/")
    assert load_deployment().monitoring_ws_url == "wss://example.org/surveillance/ws/v1/monitoring"


@pytest.mark.parametrize("value", ["0", "65536", "bad"])
def test_invalid_port_fails_closed(monkeypatch, value):
    monkeypatch.setenv("API_PORT", value)
    with pytest.raises(ValueError):
        load_deployment()


def test_port_conflict_fails_even_if_no_health_endpoint(monkeypatch):
    from scripts.full_stack_runtime import check_ports
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        monkeypatch.setenv("API_HOST", "127.0.0.1")
        monkeypatch.setenv("API_PORT", str(occupied.getsockname()[1]))
        with pytest.raises(RuntimeError, match="occupied"):
            check_ports(load_deployment())


def test_port_check_accepts_recently_closed_service_port(monkeypatch):
    from scripts.full_stack_runtime import check_ports
    with socket.socket() as closed_api, socket.socket() as closed_ml:
        closed_api.bind(("127.0.0.1", 0))
        closed_ml.bind(("127.0.0.1", 0))
        api_port, ml_port = closed_api.getsockname()[1], closed_ml.getsockname()[1]
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", str(api_port))
    monkeypatch.setenv("ML_HOST", "127.0.0.1")
    monkeypatch.setenv("ML_PORT", str(ml_port))
    check_ports(load_deployment())


def test_full_stack_preserves_frame_poll_configuration_without_forcing_a_hot_timer(monkeypatch, tmp_path):
    from scripts.full_stack_runtime import (
        active_track_verifier_enabled, application_environment, lowconf_verifier_enabled,
        verifier_modes, yolo_context_verifier_enabled,
    )
    deployment = type("Deployment", (), {"environment": lambda self: {}})()
    monkeypatch.delenv("FRONTEND_FRAME_REFRESH_INTERVAL_MS", raising=False)
    assert "FRONTEND_FRAME_REFRESH_INTERVAL_MS" not in application_environment({}, deployment, tmp_path)
    assert application_environment({"FRONTEND_FRAME_REFRESH_INTERVAL_MS": "4"}, deployment,
                                   tmp_path)["FRONTEND_FRAME_REFRESH_INTERVAL_MS"] == "4"
    assert lowconf_verifier_enabled({}) is True
    assert lowconf_verifier_enabled({"FULL_STACK_LOWCONF_VERIFIER": "0"}) is False
    with pytest.raises(ValueError, match="must be a boolean"):
        lowconf_verifier_enabled({"FULL_STACK_LOWCONF_VERIFIER": "sometimes"})
    assert yolo_context_verifier_enabled({}) is False
    assert active_track_verifier_enabled({}) is False
    assert verifier_modes({}) == (True, False, False)
    assert verifier_modes({"FULL_STACK_YOLO_CONTEXT_VERIFIER": "1"}) == (False, True, False)
    assert verifier_modes({"FULL_STACK_ACTIVE_TRACK_VERIFIER": "1"}) == (False, False, True)
    with pytest.raises(ValueError, match="choose only one"):
        verifier_modes({"FULL_STACK_YOLO_CONTEXT_VERIFIER": "1",
                       "FULL_STACK_ACTIVE_TRACK_VERIFIER": "1"})
    with pytest.raises(ValueError, match="FULL_STACK_YOLO_CONTEXT_VERIFIER"):
        verifier_modes({"FULL_STACK_YOLO_CONTEXT_VERIFIER": "sometimes"})


def test_full_stack_latency_ui_runner_can_hold_grid_without_fullscreen(tmp_path):
    from scripts.full_stack_runtime import frontend_evidence_args

    output = tmp_path / "run"
    assert frontend_evidence_args(output, grid_only=False) == [
        "-m", "scripts.validate_production_wall", "--output", str(output / "ui.jsonl")]
    assert frontend_evidence_args(output, grid_only=True) == [
        "-m", "scripts.validate_production_wall", "--output", str(output / "ui.jsonl"),
        "--grid-only"]


def test_bridge_rejects_old_session_requires_progress_and_closes_stale(tmp_path):
    path = tmp_path / "preview.bin"
    writer = PreviewFrameWriter(str(path), width=4, height=4)
    writer.publish(bytes(64))
    start = time.monotonic_ns()
    bridge = PreviewMonitoringBridge(tmp_path / "stats.json", session_start_ns=start,
                                    session_id="test", preview_paths={"CAM-01": str(path)})
    closed = False
    try:
        assert not bridge.snapshot()["cameras"][0]["online"]
        writer.publish(bytes(64))
        assert not bridge.snapshot()["cameras"][0]["online"]
        writer.publish(bytes(64))
        row = bridge.snapshot()["cameras"][0]
        assert row["online"] and row["preview_sequence"] == 3
        assert row["source_fps"] is None
        sequence, stamp, _ = bridge.last["CAM-01"]
        bridge.last["CAM-01"] = (sequence, stamp, start - 2_000_000_000)
        assert not bridge.snapshot()["cameras"][0]["online"]
        writer.close()
        closed = True
        assert not bridge.snapshot()["cameras"][0]["online"]
    finally:
        bridge.close()
        if not closed:
            writer.close()


def test_ml_camera_only_health_and_no_historical_identity(tmp_path, monkeypatch):
    from services.camera_v11.monitoring_telemetry_ipc_v1 import MonitoringTelemetryWriter, MonitoringTelemetryReader, offline_snapshot
    import services.ml_service.app.main as ml
    monkeypatch.setenv("SURVEILLANCE_ANALYTICS_ENABLED", "0")
    path = tmp_path / "monitor.json"
    payload = offline_snapshot()
    payload["runtime"]["status"] = "live"
    for row in payload["cameras"]:
        row["online"] = True
    MonitoringTelemetryWriter(path).publish(payload)
    monkeypatch.setattr(ml, "telemetry", MonitoringTelemetryReader(path))
    monkeypatch.setattr(ml, "read_tracked_objects", lambda camera_ids: {
        "status": "fresh", "supported": True, "camera_ids": list(camera_ids),
        "tracked_objects": [{"camera_id": "CAM-06", "native_track_id": 3}],
    })
    with TestClient(ml.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/api/v1/room-pair/identity").json()["people"] == []
        rows = client.get("/cameras").json()["cameras"]
        assert all("camera_id" in r and "id" not in r for r in rows)
        tracked = client.get("/api/v1/tracked-objects").json()
        assert tracked["tracked_objects"][0]["camera_id"] == "CAM-06"
        payload["cameras"][0]["online"] = False
        MonitoringTelemetryWriter(path).publish(payload)
        assert client.get("/health").json()["status"] == "degraded"


def test_api_health_does_not_claim_ok_when_ml_disappears():
    import services.api_service.app.main as api
    from services.api_service.app.ml_client import MLServiceUnavailable
    class Client:
        async def health(self):
            raise MLServiceUnavailable("offline")
    with TestClient(api.app) as client:
        client.app.state.ml_client = Client()
        assert client.get("/health").json() == {
            "service": "api_service", "status": "degraded", "ml_status": "unavailable"}


def test_camera_wall_has_no_video_network_fallback_and_overlay_uses_frame(monkeypatch, tmp_path):
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QImage
    from services.frontend.app.camera_wall import CameraTile
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("FRONTEND_USE_V11_SHARED_MEMORY", "0")
    monkeypatch.setenv("V11_UI_PREVIEW_PATH_CAM02", str(tmp_path / "missing.bin"))
    tile = CameraTile("CAM-02", "http://invalid")
    try:
        assert tile.preview_reader is not None and tile.reader is None
        tile.set_identity_rows([{"bbox": [1, 2, 5, 6], "application_id": "Person_01"}])
        tile._draw_frame(QImage(16, 16, QImage.Format.Format_RGB32))
        assert not tile.is_connected()
    finally:
        tile.close_reader()
        tile.close()


def test_launcher_preserves_f1_resolution_and_full_stack_uses_one_analytics_owner():
    text = Path("scripts/start_full_live_stack.sh").read_text()
    runtime = Path("scripts/full_stack_runtime.py").read_text()
    assert 'PY="$(resolve_python)"' in text
    assert text.index('if [[ "${1:-}" == "--preflight" ]]') < text.index("mkdir -p")
    assert "run_room_pair.py" not in text
    assert "sudo" not in text and "pip install" not in text
    assert "scripts.full_stack_runtime" in text
    assert "run_six_camera_v13_wall_candidate" in runtime
    assert 'spawn("preview"' not in runtime
    assert 'spawn("analytics"' in runtime
    assert "FULL_STACK_ANALYTICS_CANDIDATE" in runtime
    assert "validate_lowconf_verifier_artifacts" in runtime
    assert "--lowconf-verifier" in runtime
    assert 'forced_killed_children.append(name)' in runtime
    assert '"forced_kill": bool(forced_killed_children)' in runtime


def test_full_stack_ml_reads_telemetry_from_analytics_owner(tmp_path):
    from scripts.full_stack_runtime import application_environment
    from services.shared.deployment import load_deployment

    env = application_environment({}, load_deployment(), tmp_path)
    assert env["SURVEILLANCE_ANALYTICS_ENABLED"] == "0"  # local NvDCF only; no MV3DT readiness gate
    assert env["V11_MONITORING_TELEMETRY_PATH"] == str(tmp_path / "analytics" / "monitoring.json")
    assert env["V11_TRACKED_OBJECT_FEED_DIR"] == "/dev/shm"


def test_supervisor_classifies_only_requested_signal_exit_as_clean_shutdown():
    from scripts.full_stack_runtime import child_exit_class

    assert child_exit_class(0, 15) == "success"
    assert child_exit_class(-15, 15) == "expected_signal"
    assert child_exit_class(-11, 15) == "failure"
    assert child_exit_class(1, 15) == "failure"
    assert child_exit_class(-15, None) == "failure"
