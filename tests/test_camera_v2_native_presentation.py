from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_native_wall_keeps_one_k_tile_and_1080p_source_canvas() -> None:
    runtime = source("services/camera_v2/qt_runtime.py")
    live = source("services/camera_v2/sentinel_live_runtime.py")
    assert "TILE_WIDTH = 1024" in runtime
    assert "TILE_HEIGHT = 576" in runtime
    assert "SOURCE_WIDTH = 1920" in runtime
    assert "SOURCE_HEIGHT = 1080" in runtime
    assert 'os.environ["CAMERA_V2_FRAME_WIDTH"] = str(SOURCE_WIDTH)' in runtime
    assert 'os.environ["CAMERA_V2_TILER_COLUMNS"] = str(GRID_COLUMNS)' in runtime
    assert "runtime.set_wall_output_geometry(WALL_WIDTH, WALL_HEIGHT)" not in runtime
    assert 'CAMERA_V2_WALL_WIDTH", "1024"' not in live
    assert "self.wall_width = 1024" not in live
    assert 'self._set_if(self.sink, "force-aspect-ratio", True)' in live


def test_native_video_uses_one_persistent_child_surface() -> None:
    app = source("services/camera_v2/sentinel_app.py")
    runtime = source("services/camera_v2/qt_runtime.py")

    assert "class NativeVideoSurface" in app
    assert "class StableLiveWall" in app
    assert "WA_DontCreateNativeAncestors" in app
    assert "self.video.xidChanged.connect" in app
    assert "self.controller.start(xid)" in app
    assert "self.controller.bind_window(xid)" in app
    assert "ExactCameraTile" not in app
    assert "WA_PaintOnScreen" not in app
    assert "def paintEngine" not in app
    assert "showFullScreen()" not in app
    assert "window.showMaximized()" in app

    # A continuously PLAYING EGL sink owns redraw. Do not force expose/rebind on
    # ordinary resize/focus or re-send the same native window id.
    assert "GstVideo.VideoOverlay.expose" not in runtime
    assert "request_expose" not in runtime
    assert "win_id == self._window_handle" in runtime
    assert "bound_xid[0] == target" in runtime


def test_grid_surface_preserves_exact_two_by_three_aspect() -> None:
    app = source("services/camera_v2/sentinel_app.py")
    assert "GRID_ASPECT = (16.0 * 2.0) / (9.0 * 3.0)" in app
    assert "FOCUS_ASPECT = 16.0 / 9.0" in app
    assert "self.video.setGeometry" in app


def test_gpu_tiler_uses_high_quality_scaling() -> None:
    wall = source("services/camera_v2/dynamic_wall.py")
    assert 'self._set_if(self.tiler, "compute-hw", 1)' in wall
    assert 'self._set_if(self.tiler, "interpolation-method", 4)' in wall


def test_native_bbox_smoother_is_restored_and_wired() -> None:
    smoother = source("services/camera_v2/native_display_smoother.c")
    bridge = source("services/camera_v2/native_bridge.py")

    assert "const float center_alpha_1f = 0.86f" in smoother
    assert "const float velocity_alpha = 0.45f" in smoother
    assert "const float lead_frames = 0.12f" in smoother
    assert "Intentionally a no-op" not in smoother
    assert "SMOOTHER_SOURCE" in bridge
    assert "camera_v2_smooth_display_boxes" in bridge
    assert "self.lib.camera_v2_smooth_display_boxes(buffer_ptr)" in bridge


def test_nvdcf_continuity_does_not_hide_short_pose_misses() -> None:
    tracking = source("services/camera_v2/person_tracking_final.py")
    profile = source("services/camera_v2/tracker_profile.py")
    live = source("services/camera_v2/sentinel_live_runtime.py")

    assert 'CAMERA_V2_MIN_DISPLAY_TRACK_CONF", "0.12"' in tracking
    assert '"minTrackerConfidence": "0.15"' in tracking
    assert '"maxShadowTrackingAge": "40"' in tracking
    assert '"minTrackingConfidenceDuringInactive": "0.12"' in tracking
    assert "self.latency_compensator.max_projection_s = 0.16" in tracking
    assert "self.latency_compensator.projection_gain = 0.62" in tracking
    assert '"maxShadowTrackingAge": "40"' in profile
    assert '"minTrackingConfidenceDuringInactive": "0.12"' in profile
    assert '"outputShadowTracks", "0"' in profile
    assert "UI_TRACK_HOLD_SEC = 2.0" in live
