from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader, PreviewFrameWriter
from services.frontend.sentinel_v1.data import CAMERAS

RUNTIME = ROOT / "services/camera_v11/deepstream_trt86_multi_ui_cam01_cam02_v1.py"
GENERIC_RUNTIME = ROOT / "services/camera_v11/deepstream_trt86_multi_ui_v1.py"
GENERIC_PIPELINE_LAUNCHER = ROOT / "scripts/run_camera_v11_ui_pipeline_v1.sh"
GENERIC_UI_LAUNCHER = ROOT / "scripts/run_sentinel_ui_v1.sh"
BASE_RUNTIME = ROOT / "services/camera_v11/deepstream_trt86_multi_v1.py"
UI = ROOT / "services/frontend/sentinel_v1/ui.py"
UI_PARTS = ROOT / "services/frontend/sentinel_v1/ui_parts"
DATA = ROOT / "services/frontend/sentinel_v1/data.py"
PIPELINE_LAUNCHER = ROOT / "scripts/run_camera_v11_ui_cam01_cam02_pipeline_v1.sh"
UI_LAUNCHER = ROOT / "scripts/run_sentinel_ui_cam01_cam02_v1.sh"


def test_two_independent_preview_files_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = [os.path.join(tmp, "cam01.bin"), os.path.join(tmp, "cam02.bin")]
        writers = [PreviewFrameWriter(path, width=4, height=2, stride=16) for path in paths]
        readers = [PreviewFrameReader(path) for path in paths]
        try:
            payloads = [bytes(range(32)), bytes(reversed(range(32)))]
            writers[0].publish(payloads[0], object_count=1)
            writers[1].publish(payloads[1], object_count=2)
            frames = [reader.read_latest(max_age_sec=10.0) for reader in readers]
            assert frames[0] is not None and frames[1] is not None
            assert frames[0].payload == payloads[0]
            assert frames[1].payload == payloads[1]
            assert frames[0].object_count == 1
            assert frames[1].object_count == 2
        finally:
            for reader in readers:
                reader.close()
            for writer in writers:
                writer.close()


def test_base_six_camera_runtime_remains_subclassed_and_single_detector_worker() -> None:
    compatibility = RUNTIME.read_text()
    runtime = GENERIC_RUNTIME.read_text()
    base = BASE_RUNTIME.read_text()
    assert "class V11DeepStreamTRT86MultiCameraUICam01Cam02V1(V11DeepStreamTRT86MultiCameraUIV1)" in compatibility
    assert "class V11DeepStreamTRT86MultiCameraUIV1(V11DeepStreamTRT86MultiCameraV1)" in runtime
    assert "super()._build_camera(state)" in runtime
    assert "rtsp_extra=0" in runtime
    assert 'DEFAULT_UI_CAMERAS = ("CAM-01", "CAM-02")' in runtime
    assert "Step2TRT86Client()" in base


def test_camera_demo_rows_removed_and_two_runtime_cards_are_staged() -> None:
    data = DATA.read_text()
    assert [camera.id for camera in CAMERAS] == ["CAM-01", "CAM-02"]
    assert 'RUNTIME_CAMERA_IDS = ("CAM-01", "CAM-02")' in data
    assert "CAMERA_ROWS" not in data
    assert '"cam-1"' not in data
    assert '"cam-6"' not in data


def test_full_sentinel_ui_is_preserved_and_two_live_views_are_bound() -> None:
    source = "".join(path.read_text() for path in sorted(UI_PARTS.glob("part_*.pyfrag")))
    wrapper = UI.read_text()
    for name in (
        "MonitoringPage", "PeoplePage", "EventsPage", "RoomsPage", "SettingsPage",
        "FullscreenCameraGrid", "EnrollmentPage", "EnrollmentDialog", "CameraDialog",
        "PersonProfileDialog", "EventSnapshotDialog",
    ):
        assert f"class {name}" in source
    assert 'SENTINEL_LIVE_PREVIEW_CAMERAS' in wrapper
    assert '"CAM-01,CAM-02"' in wrapper
    assert "CameraView.__init__ = _camera_view_init_staged_live" in wrapper
    assert "_preview_path_for_camera" in wrapper
    assert "if self.camera.online and not self.realtime_camera_id" in source


def test_ui_and_preview_taps_do_not_open_extra_rtsp_or_run_extra_inference() -> None:
    source = "".join(path.read_text() for path in sorted(UI_PARTS.glob("part_*.pyfrag"))).lower()
    wrapper = UI.read_text().lower()
    runtime = GENERIC_RUNTIME.read_text().lower()
    for forbidden in ("cv2.videocapture", "ffmpeg"):
        assert forbidden not in source
        assert forbidden not in wrapper
        assert forbidden not in runtime
    assert "nvurisrcbin" not in wrapper
    assert "tensorrt" not in wrapper
    assert "step2trt86client" not in runtime
    assert "rtsp_extra=0" in runtime


def test_launchers_pin_two_ui_cameras_and_latest_only_paths() -> None:
    pipeline = PIPELINE_LAUNCHER.read_text()
    ui_launcher = UI_LAUNCHER.read_text()
    generic_pipeline = GENERIC_PIPELINE_LAUNCHER.read_text()
    generic_ui = GENERIC_UI_LAUNCHER.read_text()
    assert 'V11_UI_STAGE_CAMERAS="CAM-01,CAM-02"' in pipeline
    assert 'V11_UI_STAGE_CAMERAS="CAM-01,CAM-02"' in ui_launcher
    assert 'export V11_UI_PREVIEW_CAMERAS="$UI_CAMERAS"' in generic_pipeline
    assert 'export SENTINEL_LIVE_PREVIEW_CAMERAS="$UI_CAMERAS"' in generic_ui
    assert "/dev/shm/v11_ui_preview_${slug}_v1.bin" in generic_pipeline
    assert "rtsp_extra=0" in generic_pipeline
    assert "detector_workers=1" in generic_pipeline
