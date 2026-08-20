from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_native_wall_keeps_one_k_tile_and_1080p_source_canvas() -> None:
    runtime = source("services/camera_v2/qt_runtime.py")
    assert "TILE_WIDTH = 1024" in runtime
    assert "TILE_HEIGHT = 576" in runtime
    assert "SOURCE_WIDTH = 1920" in runtime
    assert "SOURCE_HEIGHT = 1080" in runtime
    assert 'os.environ["CAMERA_V2_FRAME_WIDTH"] = str(SOURCE_WIDTH)' in runtime
    assert "runtime.set_wall_output_geometry(WALL_WIDTH, WALL_HEIGHT)" in runtime


def test_native_video_surface_has_one_painter_owner_without_resize_expose() -> None:
    app = source("services/camera_v2/sentinel_app.py")
    assert "GstVideoOverlay/nveglglessink is the single owner" in app
    assert "event.accept()" in app
    assert "super().paintEvent(event)" not in app
    assert "self.controller.expose" not in app


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

    assert 'CAMERA_V2_MIN_DISPLAY_TRACK_CONF", "0.12"' in tracking
    assert '"minTrackerConfidence": "0.15"' in tracking
    assert '"maxShadowTrackingAge": "40"' in tracking
    assert '"minTrackingConfidenceDuringInactive": "0.12"' in tracking
    assert "self.latency_compensator.max_projection_s = 0.16" in tracking
    assert "self.latency_compensator.projection_gain = 0.62" in tracking
    assert '"maxShadowTrackingAge": "40"' in profile
    assert '"minTrackingConfidenceDuringInactive": "0.12"' in profile
    assert '"outputShadowTracks", "0"' in profile
