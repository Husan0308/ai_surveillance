from __future__ import annotations

import os
import sys

# The deployed desktop is X11/AnyDesk. Let callers override this explicitly, but
# prefer the xcb backend when DISPLAY is available so GstVideoOverlay receives an
# X11 WId rather than a toolkit-owned offscreen/Wayland surface.
if os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QSizePolicy, QWidget

from . import sentinel_exact as ui


class StableLiveWall(QWidget):
    """Native X11 video host for the existing single DeepStream tiled wall.

    Important: Qt must not paint an opaque background into the same native window
    used by nveglglessink. The previous implementation styled the video QWidget
    black, so Qt backing-store repaints could erase/cover EGL output. This host is
    paint-on-screen/no-background and only child overlay widgets are painted by Qt.

    The camera architecture remains unchanged:
      6x RTSP/NVDEC -> nvstreammux -> YOLO/NvDCF -> nvmultistreamtiler -> OSD -> EGL
    """

    fullscreenRequested = ui.Signal(int)

    def __init__(self, controller: ui.SentinelController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.focus_source: int | None = None
        self._boot_requested = False
        self._window_handle = 0

        self.setMinimumSize(760, 560)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_NativeWindow, True)
        self.setAttribute(Qt.WA_PaintOnScreen, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setAttribute(Qt.WA_DontCreateNativeAncestors, True)
        self._window_handle = int(self.winId())

        self.tiles: list[ui.CameraTile] = []
        for source_id in range(6):
            tile = ui.CameraTile(source_id, self)
            tile.setAttribute(Qt.WA_NativeWindow, True)
            tile.setAttribute(Qt.WA_TranslucentBackground, True)
            tile.fullscreenRequested.connect(self.fullscreenRequested)
            tile.heatmapToggled.connect(controller.set_heatmap_enabled)
            self.tiles.append(tile)

    def paintEngine(self):
        # GstVideoOverlay owns this native surface. Returning None prevents Qt's
        # backing store from repainting black over EGL frames.
        return None

    def paintEvent(self, event):
        event.accept()

    def showEvent(self, event):
        super().showEvent(event)
        # winId can change when the widget becomes native/shown. Cache it on the Qt
        # GUI thread; the GStreamer sync handler only reads this integer.
        self._window_handle = int(self.winId())
        self._relayout()
        if not self._boot_requested:
            self._boot_requested = True
            QTimer.singleShot(150, self._start_pipeline)

    def _start_pipeline(self):
        self._window_handle = int(self.winId())
        if self._window_handle <= 0:
            self.controller._set_status("ERROR", "Qt video surface has no native WId")
            return
        print(
            f"CAMERA_QT video_surface_ready wid={self._window_handle} "
            f"platform={QApplication.platformName()}",
            flush=True,
        )
        self.controller.start(self._window_handle)
        for tile in self.tiles:
            tile.raise_()

    def set_focus(self, source_id: int | None):
        self.focus_source = source_id
        self.controller.set_focus_source(source_id)
        self._relayout()

    def resizeEvent(self, event):
        self._relayout()
        for tile in self.tiles:
            tile.raise_()
        super().resizeEvent(event)

    def _relayout(self):
        if self.focus_source is not None:
            for tile in self.tiles:
                if tile.source_id == self.focus_source:
                    tile.focused = True
                    tile.setGeometry(self.rect())
                    tile.show()
                    tile.raise_()
                else:
                    tile.hide()
            return

        tile_w = self.width() / 2.0
        tile_h = self.height() / 3.0
        for source_id, tile in enumerate(self.tiles):
            row, col = divmod(source_id, 2)
            tile.focused = False
            tile.setGeometry(
                int(col * tile_w), int(row * tile_h), int(tile_w), int(tile_h)
            )
            tile.show()
            tile.raise_()

    def refresh(self, snapshot: dict):
        cameras = {
            int(camera.get("source_id", -1)): camera
            for camera in snapshot.get("cameras", [])
        }
        rooms = {
            int(room.get("room_id", -1)): int(room.get("count", 0))
            for room in snapshot.get("rooms", [])
        }
        for tile in self.tiles:
            camera = cameras.get(tile.source_id, {})
            occupancy = rooms.get(tile.source_id // 2 + 1, 0)
            tile.set_live(
                camera,
                occupancy,
                self.controller.heat_points(tile.source_id),
            )


def _refresh_monitoring(self) -> None:
    """Exact UI refresh with truthful camera/backend state and visible errors."""
    try:
        self._snapshot = self.controller.snapshot()
        self.monitoring.refresh(self._snapshot)
        status = self.controller.status
        online = sum(
            1 for camera in self._snapshot.get("cameras", []) if camera.get("online")
        )

        if status == "LIVE":
            self.pipeline_state.setText(f"LIVE · {online}/6")
            self.pipeline_state.setStyleSheet(f"color:{ui.C['known']};")
            self.pipeline_state.setToolTip("")
        elif status.startswith("DEGRADED"):
            self.pipeline_state.setText(status)
            self.pipeline_state.setStyleSheet(f"color:{ui.C['unknown']};")
            self.pipeline_state.setToolTip(self.controller.error)
        elif status in {"ERROR", "VIDEO ERROR", "NO VIDEO"}:
            self.pipeline_state.setText(status)
            self.pipeline_state.setStyleSheet(f"color:{ui.C['offline']};")
            self.pipeline_state.setToolTip(self.controller.error)
        else:
            self.pipeline_state.setText(status)
            self.pipeline_state.setStyleSheet(f"color:{ui.C['muted']};")
            self.pipeline_state.setToolTip(self.controller.error)
    except Exception as exc:
        self.pipeline_state.setText("REFRESH ERROR")
        self.pipeline_state.setStyleSheet(f"color:{ui.C['offline']};")
        self.pipeline_state.setToolTip(str(exc))
        print(f"CAMERA_QT refresh_error {type(exc).__name__}: {exc}", flush=True)


def main() -> int:
    # MainWindow and MonitoringPage resolve these names from sentinel_exact's module
    # globals at construction/runtime. Patch only integration points; every page,
    # spacing, control and original Sentinel visual component remains untouched.
    ui.LiveWall = StableLiveWall
    ui.MainWindow.refresh_monitoring = _refresh_monitoring

    app = QApplication(sys.argv)
    app.setApplicationName("Sentinel VMS")
    app.setOrganizationName("Sentinel")
    app.setStyle("Fusion")
    app.setStyleSheet(ui.APP_QSS)

    print(
        f"CAMERA_QT shell_start platform={QApplication.platformName()} "
        f"display={os.environ.get('DISPLAY', '')}",
        flush=True,
    )
    window = ui.MainWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
