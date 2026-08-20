from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.camera_wall_runtime import (  # noqa: E402
    CAMERA_COUNT,
    GRID_COLUMNS,
    GRID_ROWS,
)
from services.camera_v2.sentinel_ui import BUILD_TAG, MainWindow, main  # noqa: E402
from services.ml_service.app.config import load_settings  # noqa: E402

EXPECTED_BUILD = "2026.08.20-r17-postmux-fullscreen"


def fail(message: str) -> None:
    raise RuntimeError(f"Sentinel UI contract failed: {message}")


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main_preflight() -> int:
    if not callable(main):
        fail("main entry point is missing")
    if BUILD_TAG != EXPECTED_BUILD:
        fail(f"build tag={BUILD_TAG!r}, expected={EXPECTED_BUILD!r}")
    if not issubclass(MainWindow, object):
        fail("MainWindow is missing")
    if CAMERA_COUNT != 6 or (GRID_COLUMNS, GRID_ROWS) != (2, 3):
        fail("camera wall must stay fixed at six cameras / 2x3")

    settings = load_settings()
    if len(settings.cameras) != CAMERA_COUNT:
        fail(f"enabled camera count must be {CAMERA_COUNT}, got {len(settings.cameras)}")

    shell = source("services/camera_v2/sentinel_ui.py")
    for token in (
        'BUILD_TAG = "2026.08.20-r17-postmux-fullscreen"',
        "self.monitoring_page = MonitoringPage()",
        "self.setCentralWidget(self.monitoring_page)",
        "self.monitoring_page.shutdown()",
        "window.showMaximized()",
    ):
        if token not in shell:
            fail(f"camera-only shell guard missing: {token}")

    monitoring = source("services/camera_v2/sentinel_ui_monitoring_native.py")
    for token in (
        "class NativeVideoHost(QWidget)",
        "WA_NativeWindow, True",
        "WA_DontCreateNativeAncestors, True",
        "WA_PaintOnScreen, True",
        "def paintEngine(self)",
        "cameraClicked = Signal(int)",
        "escapeRequested = Signal()",
        "def _grid_source_at",
        "self.cameraClicked.emit(int(source_id))",
        "self.surface.cameraClicked.connect(self._camera_clicked)",
        "window.showFullScreen()",
        "window.showMaximized()",
        "self.controller.focus(sid)",
        "self.controller.focus(-1)",
    ):
        if token not in monitoring:
            fail(f"native monitoring/fullscreen guard missing: {token}")

    runtime = source("services/camera_v2/camera_wall_runtime.py")
    for token in (
        "def set_focus(source_id: int)",
        'runtime.tiler.set_property("show-source", sid)',
        "runtime.set_wall_output_geometry(FOCUS_WIDTH, FOCUS_HEIGHT)",
        "def focus(self, source_id: int)",
        'self.command_q.put_nowait(("focus", sid))',
        "bound_xid = 0",
        "if target == bound_xid",
        "GstVideo.VideoOverlay.set_window_handle",
    ):
        if token not in runtime:
            fail(f"camera wall runtime guard missing: {token}")

    pascal = source("services/camera_v2/pascal_safe_pipeline.py")
    for token in (
        "SecureCameraWallV2._add_camera(self, index, camera)",
        "pascal_postmux_tee",
        "nvstreamdemux",
        "CAMERA_DETECT_PATH mode=postmux-demux",
        "source_path=direct-to-nvstreammux",
    ):
        if token not in pascal:
            fail(f"display-first detector guard missing: {token}")

    launcher = source("scripts/run_sentinel_vms.sh")
    for token in (
        "python scripts/preflight_rfdetr_core.py",
        "python scripts/preflight_pascal_safe.py",
        "python scripts/preflight_sentinel_ui.py",
        "python scripts/preflight_camera_v2_core.py",
        "exec python -m services.camera_v2.monitor_ui",
        "expected_ui=2026.08.20-r17-postmux-fullscreen",
        "detector_path=postmux-demux",
        "ui=camera-only-2x3-click-fullscreen",
    ):
        if token not in launcher:
            fail(f"launcher guard missing: {token}")

    print(f"SENTINEL_PREFLIGHT build={BUILD_TAG} ui=PASS")
    print("SENTINEL_PREFLIGHT wall=6-camera fixed-2x3 click-fullscreen PASS")
    print("SENTINEL_PREFLIGHT native_video=direct-QWidget-xid idempotent-bind PASS")
    print("SENTINEL_PREFLIGHT detector=postmux-demux source-ingest-isolated PASS")
    print("SENTINEL_PREFLIGHT runtime=pascal-safe RF-DETR motion-predictor no-nvtracker PASS")
    print("SENTINEL_UI_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_preflight())
