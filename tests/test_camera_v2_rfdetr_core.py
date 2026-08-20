from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_rfdetr_small_is_the_active_person_detector_contract() -> None:
    backend = source("services/camera_v2/rfdetr_backend.py")
    pascal = source("services/camera_v2/pascal_safe_pipeline.py")
    controller = source("services/camera_v2/camera_wall_runtime.py")
    secure = source("services/camera_v2/secure.py")
    launcher = source("scripts/run_sentinel_vms.sh")

    assert "from rfdetr import RFDETRSmall" in backend
    assert 'RFDETRSmall(device="cuda:0")' in backend
    assert "frame[..., ::-1]" in backend
    assert "include_source_image=False" in backend
    assert "detection._yolo_worker = rfdetr_worker" in backend

    assert "class CameraPascalSafeRuntime(CameraDetectionV2)" in pascal
    assert "SecureCameraWallV2._add_camera(self, index, camera)" in pascal
    assert "def _install_postmux_inference(self)" in pascal
    assert "pascal_postmux_tee" in pascal
    assert "nvstreamdemux" in pascal
    assert 'self._request_src_pad(demux, f"src_{index}")' in pascal
    assert "CAMERA_DETECT_PATH mode=postmux-demux" in pascal
    assert "source_path=direct-to-nvstreammux" in pascal
    assert "mapped_size" in pascal
    assert "row_stride" in pascal
    assert "safe_mux_batches" in pascal
    assert "safe_wall_frames" in pascal
    assert "safe_sink_buffers" in pascal
    assert "CAMERA_STARTUP_STALL" in pascal
    assert "nvtracker=disabled" in pascal

    for forbidden in (
        "CameraPersonTrackingV2",
        "CameraPersonTrackingFinal",
        "libnvds_nvmultiobjecttracker",
        "config_tracker_NvDCF",
    ):
        assert forbidden not in pascal

    assert 'CAMERA_V2_RTSP_TRANSPORT' in secure
    assert 'self._set_if(source, "select-rtp-protocol", 4 if transport == "tcp" else 0)' in secure
    assert 'self._set_if(element, "protocols", 4)' in secure
    assert "def _source_pad_added" in secure
    assert "caps.is_any()" in secure

    assert "def set_focus(source_id: int)" in controller
    assert 'runtime.tiler.set_property("show-source", sid)' in controller
    assert "runtime.set_wall_output_geometry(FOCUS_WIDTH, FOCUS_HEIGHT)" in controller
    assert "def focus(self, source_id: int)" in controller
    assert 'self.command_q.put_nowait(("focus", sid))' in controller
    assert "if target == bound_xid" in controller
    assert "GstVideo.VideoOverlay.set_window_handle" in controller

    assert "export CAMERA_V2_RTSP_TRANSPORT=tcp" in launcher
    assert "export CAMERA_V2_RTSP_LATENCY_MS=250" in launcher
    assert "rtsp=tcp latency=250ms" in launcher
    assert "export CAMERA_V2_PASCAL_SAFE=1" in launcher
    assert "detector_path=postmux-demux" in launcher
    assert "nvtracker=disabled" in launcher
    assert "detector=RF-DETR-S@672x384" in launcher
    assert "ui=camera-only-2x3-click-fullscreen" in launcher


def test_ui_clicks_camera_into_fullscreen_and_escape_restores_grid() -> None:
    native = source("services/camera_v2/sentinel_ui_monitoring_native.py")
    shell = source("services/camera_v2/sentinel_ui.py")

    assert "class NativeVideoHost(QWidget)" in native
    assert "WA_NativeWindow, True" in native
    assert "WA_PaintOnScreen, True" in native
    assert "def paintEngine(self)" in native
    assert "cameraClicked = Signal(int)" in native
    assert "escapeRequested = Signal()" in native
    assert "def _grid_source_at" in native
    assert "self.cameraClicked.emit(int(source_id))" in native
    assert "self.surface.cameraClicked.connect(self._camera_clicked)" in native
    assert "self.surface.escapeRequested.connect(self.exit_fullscreen)" in native
    assert "self.controller.focus(sid)" in native
    assert "window.showFullScreen()" in native
    assert "self.controller.focus(-1)" in native
    assert "window.showMaximized()" in native

    for forbidden in (
        "QWindow",
        "createWindowContainer",
        "ProPipelineController",
        "ProLiveVideoWall",
        "People in Building",
    ):
        assert forbidden not in native

    assert 'BUILD_TAG = "2026.08.20-r18-rtsp-tcp"' in shell
    assert "self.monitoring_page = MonitoringPage()" in shell
    assert "window.showMaximized()" in shell
