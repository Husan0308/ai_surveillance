from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "services/camera_v11/deepstream_trt86_multi_v1.py"
CLIENT = ROOT / "services/frontend/sentinel_v1/monitoring_client_v1.py"
UI_PARTS = ROOT / "services/frontend/sentinel_v1/ui_parts"
FROZEN_BASE_SHA256 = "5372d7e64b7bed43aabf7947f404973e310138629ecbb8f176b67b5967922cdc"


def test_frozen_runtime_and_telemetry_client_architecture() -> None:
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == FROZEN_BASE_SHA256
    source = CLIENT.read_text().lower()
    assert "qwebsocket" in source
    assert "snapshotchanged" in source
    assert "statuschanged" in source
    assert "500, 1000, 2000, 5000" in source
    for forbidden in ("rtsp", "videocapture", "ffmpeg", "base64", "jpeg"):
        assert forbidden not in source


def test_monitoring_page_no_longer_consumes_demo_people_or_events() -> None:
    source = "".join(path.read_text() for path in sorted(UI_PARTS.glob("part_*.pyfrag")))
    monitoring = source[source.index("class MonitoringPage"):source.index("class PersonProfileDialog")]
    assert "PEOPLE" not in monitoring
    assert "EVENTS" not in monitoring
    assert "VISIBLE DETECTIONS" in monitoring
    assert 'self.known_people_value.setText("—")' in monitoring
    assert 'self.unknown_people_value.setText("—")' in monitoring
    for name in (
        "MonitoringPage", "PeoplePage", "EventsPage", "RoomsPage", "SettingsPage",
        "FullscreenCameraGrid", "EnrollmentPage", "EnrollmentDialog", "CameraDialog",
        "PersonProfileDialog", "EventSnapshotDialog",
    ):
        assert f"class {name}" in source


def run_qt(code: str, **extra_env: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "PYTHONPATH": str(ROOT),
            "SENTINEL_CAMERA_IDS": ",".join(f"CAM-{i:02d}" for i in range(1, 7)),
            "SENTINEL_LIVE_PREVIEW_CAMERAS": ",".join(f"CAM-{i:02d}" for i in range(1, 7)),
            **extra_env,
        }
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT, env=env,
        text=True, capture_output=True, check=False,
    )


def test_live_snapshot_updates_camera_model_and_stale_clears_runtime_state() -> None:
    result = run_qt(
        """
        import time
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        import services.frontend.sentinel_v1.ui as ui
        from services.frontend.sentinel_v1.data import CAMERAS
        rows=[]
        for i,camera in enumerate(CAMERAS):
            rows.append({
                'camera_id':camera.id, 'online':True, 'source_fps':20.0+i,
                'render_fps':18.0+i, 'infer_hz':2.0, 'queue_depth':i%2,
                'detector_drops':10+i, 'current_box_count':i,
                'result_age_ms':25.0+i, 'pipeline_errors':None,
                'infer_errors':0, 'meta_errors':0, 'copy_errors':0,
                'preview_errors':0,
            })
        payload={'schema_version':1,'sequence':1,'generated_monotonic_ns':time.monotonic_ns(),
                 'generated_epoch_ms':time.time_ns()//1_000_000,'telemetry_status':'fresh',
                 'runtime':{'status':'live','detector_enabled':True},'cameras':rows}
        window=ui.MainWindow()
        original=(CAMERAS[0].id,CAMERAS[0].name,CAMERAS[0].rtsp,CAMERAS[0].username,CAMERAS[0].password)
        window._apply_monitoring_snapshot(payload)
        assert CAMERAS[0].online and CAMERAS[0].source_fps==20.0 and CAMERAS[0].fps==18.0
        assert CAMERAS[1].queue==1 and CAMERAS[2].current_box_count==2
        page=window.pages[0]
        assert page.total_people_value.text()=='15'
        assert page.known_people_value.text()==page.unknown_people_value.text()=='—'
        assert original==(CAMERAS[0].id,CAMERAS[0].name,CAMERAS[0].rtsp,CAMERAS[0].username,CAMERAS[0].password)
        window._monitoring_status_changed('stale')
        assert not any(camera.online for camera in CAMERAS)
        assert all(camera.fps==0.0 and camera.queue is None for camera in CAMERAS)
        assert page.total_people_value.text()=='—'
        window.close()
        print('UI_TELEMETRY_STATE_PASS')
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "UI_TELEMETRY_STATE_PASS" in result.stdout


def test_qt_json_parser_signals_and_local_staleness() -> None:
    result = run_qt(
        """
        import json,time
        from PySide6.QtCore import QCoreApplication
        app=QCoreApplication([])
        from services.frontend.sentinel_v1.monitoring_client_v1 import MonitoringTelemetryClient
        ids=[f'CAM-{i:02d}' for i in range(1,7)]
        payload={'schema_version':1,'sequence':7,'generated_monotonic_ns':time.monotonic_ns(),
                 'generated_epoch_ms':time.time_ns()//1_000_000,'telemetry_status':'fresh',
                 'runtime':{'status':'live'},'cameras':[{'camera_id':x} for x in ids]}
        client=MonitoringTelemetryClient('ws://127.0.0.1:9/ws',ids)
        snapshots=[]; statuses=[]
        client.snapshotChanged.connect(snapshots.append); client.statusChanged.connect(statuses.append)
        client._message(json.dumps(payload))
        assert snapshots and snapshots[0]['sequence']==7
        client._last_message_mono=time.monotonic()-3
        client._check_freshness(); assert statuses[-1]=='stale'
        client._last_message_mono=time.monotonic()-6
        client._check_freshness(); assert statuses[-1]=='offline'
        assert client._backoff_ms==(500,1000,2000,5000)
        print('QT_CLIENT_PARSE_STALE_PASS')
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "QT_CLIENT_PARSE_STALE_PASS" in result.stdout


def test_realtime_video_poll_does_not_override_websocket_camera_state(tmp_path: Path) -> None:
    preview = tmp_path / "preview.bin"
    result = run_qt(
        """
        import os
        from PySide6.QtWidgets import QApplication
        from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameWriter
        app=QApplication([])
        import services.frontend.sentinel_v1.ui as ui
        from services.frontend.sentinel_v1.data import CAMERAS
        writer=PreviewFrameWriter(os.environ['V11_UI_PREVIEW_PATH_CAM01'],2,1,8)
        writer.publish(bytes([0,0,255,0,0,255,0,0]),object_count=99)
        camera=CAMERAS[0]; camera.online=False; camera.fps=12.5; camera.current_box_count=3
        view=ui.CameraView(camera); view._poll_live_preview()
        assert not view.live_image.isNull()
        assert camera.online is False and camera.fps==12.5
        assert view.live_object_count==3
        view.close(); writer.close(unlink=True)
        print('VIDEO_TELEMETRY_AUTHORITY_PASS')
        """,
        SENTINEL_MONITORING_REALTIME="1",
        V11_UI_PREVIEW_PATH_CAM01=str(preview),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VIDEO_TELEMETRY_AUTHORITY_PASS" in result.stdout



def test_sentinel_process_handles_sigterm_without_traceback() -> None:
    env = os.environ.copy()
    env.update(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "PYTHONPATH": str(ROOT),
            "SENTINEL_CAMERA_IDS": ",".join(f"CAM-{i:02d}" for i in range(1, 7)),
            "SENTINEL_LIVE_PREVIEW_CAMERAS": ",".join(f"CAM-{i:02d}" for i in range(1, 7)),
            "SENTINEL_MONITORING_REALTIME": "1",
            "SENTINEL_MONITORING_WS_URL": "ws://127.0.0.1:9/ws/v1/monitoring",
        }
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "services.frontend.sentinel_v1.main"],
        cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        import time
        time.sleep(0.5)
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    assert process.returncode == 0, stdout + stderr
    assert "Traceback" not in stdout + stderr
    assert "KeyboardInterrupt" not in stdout + stderr
