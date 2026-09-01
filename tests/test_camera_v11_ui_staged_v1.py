from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader, PreviewFrameWriter

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "services/camera_v11/deepstream_trt86_multi_v1.py"
RUNTIME = ROOT / "services/camera_v11/deepstream_trt86_multi_ui_v1.py"
UI = ROOT / "services/frontend/sentinel_v1/ui.py"
UI_PARTS = ROOT / "services/frontend/sentinel_v1/ui_parts"
DATA = ROOT / "services/frontend/sentinel_v1/data.py"
CHECKER = ROOT / "scripts/check_camera_v11_ui_preview_v1_log.py"
FROZEN_BASE_SHA256 = "5372d7e64b7bed43aabf7947f404973e310138629ecbb8f176b67b5967922cdc"
ALL_CAMERAS = tuple(f"CAM-{index:02d}" for index in range(1, 7))
STAGES = (
    ALL_CAMERAS[:3],
    ALL_CAMERAS[:4],
    ALL_CAMERAS[:5],
)


def stage_tag(cameras: tuple[str, ...]) -> str:
    return f"cam01_cam{len(cameras):02d}"


def test_frozen_six_camera_runtime_is_byte_for_byte_unchanged() -> None:
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == FROZEN_BASE_SHA256


def test_generic_runtime_is_additive_latest_only_and_never_opens_capture() -> None:
    source = RUNTIME.read_text()
    lower = source.lower()
    assert "class V11DeepStreamTRT86MultiCameraUIV1(V11DeepStreamTRT86MultiCameraV1)" in source
    assert "super()._build_camera(state)" in source
    assert "source=post-osd-same-pipeline" in source
    assert "rtsp_extra=0" in source
    assert "self._latest_queue(ui_q)" in source
    assert '("max-buffers", 1)' in source
    assert '("drop", True)' in source
    assert '("sync", False)' in source
    assert '("async", False)' in source
    assert '("enable-last-sample", False)' in source
    assert '("wait-on-eos", False)' in source
    assert 'ui_sink.emit("try-pull-sample"' in source
    assert 'ui_sink.connect("new-sample"' not in source
    assert "step2trt86client" not in lower
    for forbidden in ("nvurisrcbin", "rtspsrc", "cv2.videocapture", "ffmpeg"):
        assert forbidden not in lower


def test_six_unique_preview_files_roundtrip_and_reader_recovers_restart() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = [os.path.join(tmp, f"cam{index:02d}.bin") for index in range(1, 7)]
        assert len(paths) == len(set(paths))
        writers = [PreviewFrameWriter(path, width=4, height=2, stride=16) for path in paths]
        readers = [PreviewFrameReader(path) for path in paths]
        try:
            for index, writer in enumerate(writers):
                writer.publish(bytes([index] * 32), object_count=index)
            for index, reader in enumerate(readers):
                frame = reader.read_latest(max_age_sec=10.0)
                assert frame is not None
                assert frame.payload == bytes([index] * 32)
                assert frame.object_count == index

            writers[0].close(unlink=True)
            replacement = PreviewFrameWriter(paths[0], width=4, height=2, stride=16)
            writers[0] = replacement
            replacement.publish(bytes([99] * 32), object_count=9)
            restarted = readers[0].read_latest(max_age_sec=10.0)
            assert restarted is not None
            assert restarted.payload == bytes([99] * 32)
            assert restarted.object_count == 9

            replacement.publish(
                bytes([88] * 32),
                timestamp_ns=time.monotonic_ns() - 2_000_000_000,
            )
            assert readers[0].read_latest(max_age_sec=0.05) is None
        finally:
            for reader in readers:
                reader.close()
            for writer in writers:
                try:
                    writer.close()
                except (OSError, ValueError):
                    pass


@pytest.mark.parametrize("cameras", STAGES)
def test_stage_has_exact_authoritative_cards_and_live_readers(cameras: tuple[str, ...]) -> None:
    joined = ",".join(cameras)
    code = textwrap.dedent(
        """
        import os
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        import services.frontend.sentinel_v1.ui as ui
        from services.frontend.sentinel_v1.data import CAMERAS
        expected = os.environ["SENTINEL_CAMERA_IDS"].split(",")
        assert [camera.id for camera in CAMERAS] == expected
        assert list(ui.LIVE_PREVIEW_CAMERAS) == expected
        views = [ui.CameraView(camera) for camera in CAMERAS]
        assert [view.realtime_camera_id for view in views] == expected
        paths = [str(view.live_reader.path) for view in views]
        assert len(paths) == len(set(paths)) == len(expected)
        fullscreen = ui.FullscreenCameraGrid([], False)
        full_views = fullscreen.findChildren(ui.CameraView)
        assert sorted(view.realtime_camera_id for view in full_views) == sorted(expected)
        assert all(view.live_reader is not None for view in full_views)
        fullscreen.close()
        for view in views:
            view.close()
        print("STAGE_UI_BINDING_PASS", ",".join(expected), ",".join(paths))
        """
    )
    env = os.environ.copy()
    env.update(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "PYTHONPATH": str(ROOT),
            "SENTINEL_CAMERA_IDS": joined,
            "SENTINEL_LIVE_PREVIEW_CAMERAS": joined,
        }
    )
    for index, camera in enumerate(cameras, 1):
        env[f"V11_UI_PREVIEW_PATH_{camera.replace('-', '')}"] = f"/tmp/test-stage-{len(cameras)}-{index}.bin"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STAGE_UI_BINDING_PASS" in result.stdout


def test_bgrx_color_qsize_draw_and_stale_frame_clear() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preview.bin")
        code = textwrap.dedent(
            """
            import os, time
            from PySide6.QtCore import QRect
            from PySide6.QtGui import QImage, QPainter
            from PySide6.QtWidgets import QApplication
            from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameWriter
            app = QApplication.instance() or QApplication([])
            import services.frontend.sentinel_v1.ui as ui
            from services.frontend.sentinel_v1.data import CAMERAS
            path = os.environ["V11_UI_PREVIEW_PATH_CAM01"]
            writer = PreviewFrameWriter(path, width=2, height=1, stride=8)
            writer.publish(bytes([0, 0, 255, 0, 255, 0, 0, 0]))
            view = ui.CameraView(CAMERAS[0])
            view._poll_live_preview()
            assert view.camera.online and not view.live_image.isNull()
            red = view.live_image.pixelColor(0, 0)
            blue = view.live_image.pixelColor(1, 0)
            assert red.red() > 240 and red.green() < 10 and red.blue() < 10
            assert blue.blue() > 240 and blue.green() < 10 and blue.red() < 10
            canvas = QImage(120, 80, QImage.Format_RGB32)
            painter = QPainter(canvas)
            assert view._draw_live_frame(painter, QRect(0, 0, 120, 80))
            painter.end()
            writer.close(unlink=True)
            view.live_last_frame_mono = time.monotonic() - 2.0
            view._poll_live_preview()
            assert not view.camera.online and view.live_image.isNull()
            view.close()
            """
        )
        env = os.environ.copy()
        env.update(
            {
                "QT_QPA_PLATFORM": "offscreen",
                "PYTHONPATH": str(ROOT),
                "SENTINEL_CAMERA_IDS": "CAM-01",
                "SENTINEL_LIVE_PREVIEW_CAMERAS": "CAM-01",
                "V11_UI_PREVIEW_PATH_CAM01": path,
            }
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_full_sentinel_surface_and_qpainter_balance_are_preserved() -> None:
    source = "".join(path.read_text() for path in sorted(UI_PARTS.glob("part_*.pyfrag")))
    wrapper = UI.read_text()
    for name in (
        "MonitoringPage", "PeoplePage", "EventsPage", "RoomsPage", "SettingsPage",
        "FullscreenCameraGrid", "EnrollmentPage", "EnrollmentDialog", "CameraDialog",
        "PersonProfileDialog", "EventSnapshotDialog",
    ):
        assert f"class {name}" in source
    camera_view = source[source.index("class CameraView"):source.index("class EventSnapshot")]
    assert camera_view.count("p.save()") == camera_view.count("p.restore()") == 1
    assert "target_size = rect.size()" in wrapper
    assert 'CameraView._poll_live_preview = _camera_view_poll_live_preview_fresh' in wrapper
    assert "self.live_image = QImage()" in wrapper
    for forbidden in ("cv2.VideoCapture", "nvurisrcbin", "rtspsrc", "tensorrt"):
        assert forbidden.lower() not in (source + wrapper).lower()
    assert "lay.addWidget(CameraView(cam, PEOPLE" in source  # expanded view uses same boundary


@pytest.mark.parametrize("cameras", STAGES)
def test_stage_launchers_pin_exact_prefix(cameras: tuple[str, ...]) -> None:
    tag = stage_tag(cameras)
    joined = ",".join(cameras)
    pipeline = (ROOT / f"scripts/run_camera_v11_ui_{tag}_pipeline_v1.sh").read_text()
    ui = (ROOT / f"scripts/run_sentinel_ui_{tag}_v1.sh").read_text()
    assert f'V11_UI_STAGE_CAMERAS="{joined}"' in pipeline
    assert f'V11_UI_STAGE_CAMERAS="{joined}"' in ui
    assert "run_camera_v11_ui_pipeline_v1.sh" in pipeline
    assert "run_sentinel_ui_v1.sh" in ui


def synthetic_log(ui_cameras: tuple[str, ...], *, duplicate_path=False, preview_queue=0) -> str:
    lines = [
        "CAMERA_V11_DS_YOLO_MULTI_ARCH cameras=6 camera_ids=CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06 "
        "rtsp_sources=6 rtsp_sessions=6 rtsp_per_camera=1 decode=deepstream-nvdec "
        "detector_workers=1 detector_rtsp=0 detector_queue=latest1-per-camera "
        "detector_thread=dedicated scheduler=round-robin gst_nvinfer=0 second_rtsp=0 opencv=0 ffmpeg=0",
        "CAMERA_V11_DS_YOLO_MULTI_DETECTOR_THREAD state=START workers=1 cameras=6 scheduler=round-robin",
        "CAMERA_V11_DS_YOLO_MULTI_START cameras=CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06 state=async",
    ]
    for index, cid in enumerate(ui_cameras):
        path_index = 0 if duplicate_path else index
        lines.append(
            f"CAMERA_V11_UI_PREVIEW_ARCH camera={cid} source=post-osd-same-pipeline "
            f"rtsp_extra=0 queue=latest1 fps=15.0 transport=raw-bgrx-shm "
            f"path=/dev/shm/v11_ui_preview_cam{path_index + 1:02d}_v1.bin"
        )
    for window in range(5):
        for cid in ALL_CAMERAS:
            lines.append(
                f"CAMERA_V11_DS_YOLO_MULTI camera={cid} source_fps=20.00 render_fps=18.00 "
                "infer_hz=1.60 queue=0 infer_count=100 infer_admitted=100 detector_drops=1000 "
                "positive_inferences=0 detections_total=0 max_objects=0 latest_boxes=0 "
                "result_clears=0 stale_expirations=0 metadata_added=0 result_age_ms=100.0 "
                "infer_p95_ms=120.0 render_gap_p95_ms=110.0 detector_thread_alive=1 "
                "worker_alive=1 copy_errors=0 infer_errors=0 meta_errors=0 warnings=0 pipeline_errors=0"
            )
        for index, cid in enumerate(ui_cameras):
            lines.append(
                f"CAMERA_V11_UI_PREVIEW_STATS camera={cid} exported={(window + 1) * 50} "
                f"sequence={(window + 1) * 50} errors=0 queue={preview_queue} age_ms=20.0 "
                "thread_alive=1 file_exists=1 width=640 height=360 stride=2560 "
                f"path=/dev/shm/v11_ui_preview_cam{index + 1:02d}_v1.bin"
            )
    return "\n".join(lines) + "\n"


def run_checker(tmp_path: Path, body: str, cameras: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    log = tmp_path / "ui.log"
    log.write_text(body)
    return subprocess.run(
        [sys.executable, str(CHECKER), "--log", str(log), "--required-ui-cameras", ",".join(cameras)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )


@pytest.mark.parametrize("cameras", STAGES)
def test_checker_accepts_every_staged_milestone(tmp_path: Path, cameras: tuple[str, ...]) -> None:
    result = run_checker(tmp_path, synthetic_log(cameras), cameras)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT=PASS" in result.stdout


@pytest.mark.parametrize(
    ("body_factory", "reason"),
    [
        (lambda cameras: synthetic_log(cameras, duplicate_path=True), "preview_paths_not_unique"),
        (lambda cameras: synthetic_log(cameras, preview_queue=2), "preview_queue=2>1"),
    ],
)
def test_checker_rejects_shared_paths_and_queue_growth(tmp_path: Path, body_factory, reason: str) -> None:
    cameras = ALL_CAMERAS[:3]
    result = run_checker(tmp_path, body_factory(cameras), cameras)
    assert result.returncode == 1
    assert reason in result.stdout
