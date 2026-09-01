from __future__ import annotations

import os
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader, PreviewFrameWriter
from services.frontend.sentinel_v1.data import CAMERAS

RUNTIME = ROOT / "services/camera_v11/deepstream_trt86_multi_ui_cam01_v1.py"
BASE_RUNTIME = ROOT / "services/camera_v11/deepstream_trt86_multi_v1.py"
UI = ROOT / "services/frontend/sentinel_v1/ui.py"
UI_PARTS = ROOT / "services/frontend/sentinel_v1/ui_parts"
DATA = ROOT / "services/frontend/sentinel_v1/data.py"


def test_ipc_latest_frame_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preview.bin")
        writer = PreviewFrameWriter(path, width=4, height=2, stride=16)
        reader = PreviewFrameReader(path)
        payload = bytes(range(32))
        writer.publish(payload, object_count=2)
        frame = reader.read_latest(max_age_sec=10.0)
        assert frame is not None
        assert (frame.width, frame.height, frame.stride, frame.object_count) == (4, 2, 16, 2)
        assert frame.payload == payload
        reader.close(); writer.close()


def test_base_six_camera_runtime_is_only_subclassed() -> None:
    base = BASE_RUNTIME.read_text(); runtime = RUNTIME.read_text()
    assert "class V11DeepStreamTRT86MultiCameraUICam01V1(V11DeepStreamTRT86MultiCameraV1)" in runtime
    assert "super()._build_camera(state)" in runtime
    assert "rtsp_extra=0" in runtime
    assert "Step2TRT86Client()" in base


def test_camera_demo_rows_stay_removed_and_cam01_remains_first_runtime_card() -> None:
    data = DATA.read_text()
    ids = [camera.id for camera in CAMERAS]
    assert ids and ids[0] == "CAM-01"
    assert all(camera_id.startswith("CAM-") for camera_id in ids)
    assert "RUNTIME_CAMERA_IDS" in data
    assert "CAMERA_ROWS" not in data
    assert '"cam-1"' not in data
    assert '"cam-6"' not in data
    assert "Legacy demo ids such as cam-1..cam-6 are deliberately ignored" in data


def test_full_sentinel_ui_is_preserved_and_cam01_stays_live_at_view_boundary() -> None:
    source = "".join(path.read_text() for path in sorted(UI_PARTS.glob("part_*.pyfrag")))
    wrapper = UI.read_text()
    for name in (
        "MonitoringPage", "PeoplePage", "EventsPage", "RoomsPage", "SettingsPage",
        "FullscreenCameraGrid", "EnrollmentPage", "EnrollmentDialog", "CameraDialog",
        "PersonProfileDialog", "EventSnapshotDialog",
    ):
        assert f"class {name}" in source
    assert "LIVE_PREVIEW_CAMERAS" in wrapper
    assert "CAM-01" in wrapper
    assert "CameraView.__init__ = _camera_view_init_staged_live" in wrapper
    assert "if self.camera.online and not self.realtime_camera_id" in source


def test_ui_never_opens_rtsp_or_runs_inference() -> None:
    source = "".join(path.read_text() for path in sorted(UI_PARTS.glob("part_*.pyfrag"))).lower()
    wrapper = UI.read_text().lower()
    for forbidden in ("cv2.videocapture", "rtspsrc", "nvurisrcbin", "tensorrt"):
        assert forbidden not in source
        assert forbidden not in wrapper
