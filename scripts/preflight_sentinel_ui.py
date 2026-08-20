from __future__ import annotations

import ast
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

EXPECTED_BUILD = "2026.08.20-r16-pascal-safe"


def fail(message: str) -> None:
    raise RuntimeError(f"Sentinel UI contract failed: {message}")


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def called_attributes(text: str) -> set[str]:
    tree = ast.parse(text)
    return {
        str(node.func.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


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
        'BUILD_TAG = "2026.08.20-r16-pascal-safe"',
        "self.monitoring_page = MonitoringPage()",
        "self.setCentralWidget(self.monitoring_page)",
        "self.monitoring_page.shutdown()",
        "window.showMaximized()",
        "mode=camera-only",
    ):
        if token not in shell:
            fail(f"camera-only shell guard missing: {token}")

    for forbidden in (
        "QStackedWidget",
        "PeoplePage",
        "EventsPage",
        "RoomsPage",
        "EnrollmentPage",
        "SettingsPage",
        "sidebar",
        "identity_panel",
    ):
        if forbidden in shell:
            fail(f"camera-only shell contains dashboard UI: {forbidden}")

    monitoring = source("services/camera_v2/sentinel_ui_monitoring_native.py")
    for token in (
        "class NativeVideoHost(QWidget)",
        "WA_NativeWindow, True",
        "WA_DontCreateNativeAncestors, True",
        "WA_NoSystemBackground, True",
        "WA_OpaquePaintEvent, True",
        "WA_PaintOnScreen, True",
        "def paintEngine(self)",
        "xid = int(self.winId())",
        "from .camera_wall_runtime import CameraWallController",
        "self.controller = CameraWallController()",
        "self.surface.nativeReady.connect(self._start_or_bind)",
    ):
        if token not in monitoring:
            fail(f"native monitoring guard missing: {token}")

    for forbidden in (
        "QWindow",
        "createWindowContainer",
        "QFrame",
        "QLabel",
        "ProPipelineController",
        "ProLiveVideoWall",
        "People in Building",
        "open_fullscreen_grid",
    ):
        if forbidden in monitoring:
            fail(f"legacy UI/runtime leaked into monitoring: {forbidden}")

    runtime = source("services/camera_v2/camera_wall_runtime.py")
    for token in (
        "from .pascal_safe_pipeline import CameraPascalSafeRuntime",
        "runtime = CameraPascalSafeRuntime()",
        "bound_xid = 0",
        "if target == bound_xid",
        "GstVideo.VideoOverlay.set_window_handle",
        "runtime.bus.set_sync_handler",
        "GRID_COLUMNS = 2",
        "GRID_ROWS = 3",
    ):
        if token not in runtime:
            fail(f"camera wall runtime guard missing: {token}")

    shell_calls = called_attributes(shell)
    if "showFullScreen" in shell_calls or "showNormal" in shell_calls:
        fail("top-level window mode churn returned")
    if "showMaximized" not in shell_calls:
        fail("camera-only window must start maximized")

    launcher = source("scripts/run_sentinel_vms.sh")
    for token in (
        "python scripts/preflight_rfdetr_core.py",
        "python scripts/preflight_pascal_safe.py",
        "python scripts/preflight_sentinel_ui.py",
        "python scripts/preflight_camera_v2_core.py",
        "exec python -m services.camera_v2.monitor_ui",
        "export QT_QPA_PLATFORM=xcb",
        "expected_ui=2026.08.20-r16-pascal-safe",
    ):
        if token not in launcher:
            fail(f"launcher guard missing: {token}")

    print(f"SENTINEL_PREFLIGHT build={BUILD_TAG} ui=PASS")
    print("SENTINEL_PREFLIGHT shell=camera-only no-dashboard PASS")
    print("SENTINEL_PREFLIGHT native_video=direct-QWidget-xid idempotent-bind PASS")
    print("SENTINEL_PREFLIGHT wall=6-camera fixed-2x3 PASS")
    print("SENTINEL_PREFLIGHT runtime=pascal-safe RF-DETR motion-predictor no-nvtracker PASS")
    print("SENTINEL_UI_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_preflight())
