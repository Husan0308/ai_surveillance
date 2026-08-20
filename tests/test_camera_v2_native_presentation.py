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


def test_native_video_surface_has_one_painter_owner() -> None:
    app = source("services/camera_v2/sentinel_app.py")
    assert "Native GstVideoOverlay owns every video pixel" in app
    assert "event.accept()" in app
    assert "self.controller.expose" in app
    assert "super().paintEvent(event)" not in app


def test_gpu_tiler_uses_high_quality_scaling() -> None:
    wall = source("services/camera_v2/dynamic_wall.py")
    assert 'self._set_if(self.tiler, "compute-hw", 1)' in wall
    assert 'self._set_if(self.tiler, "interpolation-method", 4)' in wall
