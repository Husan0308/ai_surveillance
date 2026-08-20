from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_active_native_wall_is_fixed_two_by_three_with_focus_mode() -> None:
    runtime = source("services/camera_v2/camera_wall_runtime.py")
    assert "CAMERA_COUNT = 6" in runtime
    assert "GRID_COLUMNS = 2" in runtime
    assert "GRID_ROWS = 3" in runtime
    assert "WALL_WIDTH = 1600" in runtime
    assert "WALL_HEIGHT = 1350" in runtime
    assert "FOCUS_WIDTH = 1920" in runtime
    assert "FOCUS_HEIGHT = 1080" in runtime
    assert 'runtime.tiler.set_property("show-source", sid)' in runtime
    assert "runtime.set_wall_output_geometry(FOCUS_WIDTH, FOCUS_HEIGHT)" in runtime
    assert "runtime.set_wall_output_geometry(WALL_WIDTH, WALL_HEIGHT)" in runtime


def test_active_native_video_uses_one_persistent_child_surface() -> None:
    app = source("services/camera_v2/sentinel_ui_monitoring_native.py")
    runtime = source("services/camera_v2/camera_wall_runtime.py")

    assert "class NativeVideoHost(QWidget)" in app
    assert "WA_NativeWindow, True" in app
    assert "WA_DontCreateNativeAncestors, True" in app
    assert "self.surface.nativeReady.connect(self._start_or_bind)" in app
    assert "from .camera_wall_runtime import CameraWallController" in app
    assert "ProPipelineController" not in app
    assert "QWindow" not in app
    assert "createWindowContainer" not in app

    assert "xid == self._last_emitted_xid" in app
    assert "xid == self._last_bound_xid" in app
    assert "bound_xid = 0" in runtime
    assert "if target == bound_xid" in runtime
    assert "GstVideo.VideoOverlay.set_window_handle" in runtime
    assert "runtime.bus.set_sync_handler" in runtime


def test_camera_click_fullscreens_and_escape_restores_grid() -> None:
    app = source("services/camera_v2/sentinel_ui_monitoring_native.py")
    assert "cameraClicked = Signal(int)" in app
    assert "escapeRequested = Signal()" in app
    assert "def _grid_source_at" in app
    assert "wall_aspect = (16.0 * GRID_COLUMNS) / (9.0 * GRID_ROWS)" in app
    assert "self.cameraClicked.emit(int(source_id))" in app
    assert "window.showFullScreen()" in app
    assert "window.showMaximized()" in app
    assert "self.surface.escapeRequested.connect(self.exit_fullscreen)" in app


def test_pascal_runtime_source_path_is_not_split_before_mux() -> None:
    runtime = source("services/camera_v2/pascal_safe_pipeline.py")
    assert "SecureCameraWallV2._add_camera(self, index, camera)" in runtime
    assert "def _install_postmux_inference(self)" in runtime
    assert "pascal_postmux_tee" in runtime
    assert "nvstreamdemux" in runtime
    assert "CAMERA_DETECT_PATH mode=postmux-demux" in runtime
    assert "source_path=direct-to-nvstreammux" in runtime


def test_pascal_runtime_has_no_nvdcf_hot_path() -> None:
    runtime = source("services/camera_v2/pascal_safe_pipeline.py")
    assert "class CameraPascalSafeRuntime(CameraDetectionV2)" in runtime
    assert "tracker=motion-predictor" in runtime
    assert "nvtracker=disabled" in runtime
    for forbidden in (
        "CameraPersonTrackingV2",
        "CameraPersonTrackingFinal",
        "libnvds_nvmultiobjecttracker",
        "config_tracker_NvDCF",
    ):
        assert forbidden not in runtime


def test_pascal_runtime_tracks_every_display_stage() -> None:
    runtime = source("services/camera_v2/pascal_safe_pipeline.py")
    for token in (
        "safe_mux_batches",
        "safe_wall_frames",
        "safe_sink_buffers",
        "source_frames=",
        "CAMERA_STARTUP_STALL",
        "rendered=",
        "dropped=",
    ):
        assert token in runtime


def test_gpu_tiler_uses_high_quality_scaling() -> None:
    wall = source("services/camera_v2/dynamic_wall.py")
    assert 'self._set_if(self.tiler, "compute-hw", 1)' in wall
    assert 'self._set_if(self.tiler, "interpolation-method", 4)' in wall
