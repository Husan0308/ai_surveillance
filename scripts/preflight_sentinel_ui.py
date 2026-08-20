from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.sentinel_ui import BUILD_TAG, MainWindow, main
from services.camera_v2.sentinel_ui_monitoring_native import MonitoringPage
from services.camera_v2.sentinel_video import CAMERA_COUNT, GRID_COLUMNS, GRID_ROWS
from services.ml_service.app.config import load_settings

EXPECTED_BUILD = "2026.08.20-r15-camera-only"


def _fail(message: str) -> None:
    raise RuntimeError(f"Sentinel UI contract failed: {message}")


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _require_all(source: str, guards: tuple[str, ...], label: str) -> None:
    for guard in guards:
        if guard not in source:
            _fail(f"{label} guard missing: {guard}")


def _called_attribute_names(source: str) -> set[str]:
    tree = ast.parse(source)
    output: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            output.add(str(node.func.attr))
    return output


def main_preflight() -> int:
    if not callable(main):
        _fail("main entry point is missing")
    if BUILD_TAG != EXPECTED_BUILD:
        _fail(f"build tag={BUILD_TAG!r}, expected={EXPECTED_BUILD!r}")
    if not issubclass(MainWindow, object):
        _fail("MainWindow is missing")

    if CAMERA_COUNT != 6 or (GRID_COLUMNS, GRID_ROWS) != (2, 3):
        _fail("camera wall must stay fixed at six cameras / 2x3")

    settings = load_settings()
    if not 1 <= len(settings.cameras) <= CAMERA_COUNT:
        _fail(f"enabled camera count must be 1..{CAMERA_COUNT}")

    shell = _source("services/camera_v2/sentinel_ui.py")
    _require_all(
        shell,
        (
            'BUILD_TAG = "2026.08.20-r15-camera-only"',
            "self.monitoring_page = MonitoringPage()",
            "self.setCentralWidget(self.monitoring_page)",
            "self.monitoring_page.shutdown()",
            "window.showMaximized()",
            "mode=camera-only",
        ),
        "camera-only shell",
    )

    for forbidden in (
        "QStackedWidget",
        "QButtonGroup",
        "QToolButton",
        "PeoplePage",
        "EventsPage",
        "RoomsPage",
        "EnrollmentPage",
        "SettingsPage",
        "sidebar",
        "identity_panel",
    ):
        if forbidden in shell:
            _fail(f"camera-only shell still contains dashboard UI: {forbidden}")

    shell_calls = _called_attribute_names(shell)
    forbidden_window_calls = {"showFullScreen", "showNormal"}.intersection(shell_calls)
    if forbidden_window_calls:
        _fail("top-level window mode churn returned")

    monitoring = _source("services/camera_v2/sentinel_ui_monitoring_native.py")
    _require_all(
        monitoring,
        (
            "class NativeVideoHost(QWidget)",
            "WA_NativeWindow, True",
            "WA_DontCreateNativeAncestors, True",
            "WA_NoSystemBackground, True",
            "WA_OpaquePaintEvent, True",
            "WA_PaintOnScreen, True",
            "def paintEngine(self)",
            "xid = int(self.winId())",
            "mode=direct-native-qwidget",
            "class MonitoringPage(QWidget)",
            "self.surface = NativeVideoHost(self)",
            "self.surface.nativeReady.connect(self._start_or_bind)",
            "ProPipelineController()",
            "self.controller.poll()",
            "def shutdown",
        ),
        "direct-native camera Monitoring",
    )

    for forbidden in (
        "QWindow",
        "createWindowContainer",
        "QFrame",
        "QLabel",
        "CameraStatusRow",
        "People in Building",
        "known_people",
        "unknown_people",
        "open_fullscreen_grid",
        "set_fullscreen_mode",
        "controller.focus(",
        "ProLiveVideoWall(",
    ):
        if forbidden in monitoring:
            _fail(f"Monitoring still contains unwanted UI/native layer: {forbidden}")

    video = _source("services/camera_v2/sentinel_video_pro.py")
    _require_all(
        video,
        (
            'runtime.tiler.set_property("rows", GRID_ROWS)',
            'runtime.tiler.set_property("columns", GRID_COLUMNS)',
            "runtime.set_wall_output_geometry(WALL_WIDTH, WALL_HEIGHT)",
            "GstVideo.VideoOverlay.set_window_handle",
        ),
        "fixed-grid runtime",
    )

    launcher = _source("scripts/run_sentinel_vms.sh")
    _require_all(
        launcher,
        (
            "python scripts/preflight_rfdetr_core.py",
            "python scripts/preflight_sentinel_ui.py",
            "python scripts/preflight_camera_v2_core.py",
            "exec python -m services.camera_v2.monitor_ui",
            "export QT_QPA_PLATFORM=xcb",
        ),
        "launcher",
    )

    print(f"SENTINEL_PREFLIGHT build={BUILD_TAG} ui=PASS")
    print("SENTINEL_PREFLIGHT shell=camera-only no-dashboard PASS")
    print("SENTINEL_PREFLIGHT native_video=direct-QWidget-xid paint-on-screen PASS")
    print("SENTINEL_PREFLIGHT wall=6-camera fixed-2x3 PASS")
    print("SENTINEL_PREFLIGHT architecture=detector+tracker+pipeline untouched PASS")
    print("SENTINEL_UI_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_preflight())
