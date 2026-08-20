from __future__ import annotations

import multiprocessing as mp
import os
import sys

# Keep the historical Sentinel shell, but replace its old six transparent video
# overlays with one stable native child window owned by GstVideoOverlay.
from . import sentinel_exact as ui


def _install_reference_contract() -> None:
    from PySide6.QtCore import QEvent, QTimer, Qt, Signal
    from PySide6.QtWidgets import QSizePolicy, QWidget

    ui.ROOMS = (
        {"id": 1, "name": "Lobbi", "capacity": 40, "sources": (0, 1)},
        {"id": 2, "name": "Ofis", "capacity": 25, "sources": (2, 3)},
        {"id": 3, "name": "Ombor", "capacity": 15, "sources": (4, 5)},
    )

    if "QPlainTextEdit" not in ui.APP_QSS:
        ui.APP_QSS = ui.APP_QSS.replace(
            "QLineEdit, QComboBox, QTextEdit",
            "QLineEdit, QComboBox, QPlainTextEdit, QTextEdit",
        ).replace(
            "QLineEdit:focus, QComboBox:focus, QTextEdit:focus",
            "QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus",
        )
    if "QTableWidget" not in ui.APP_QSS:
        ui.APP_QSS = ui.APP_QSS.replace(
            "QDialog { background:",
            "QTableWidget { background: #0e151d; alternate-background-color: #101923; border: 1px solid #22303e; gridline-color: #22303e; selection-background-color: #17313a; }\n"
            "QHeaderView::section { background: #111b25; color: #7e8c99; border: 0; border-bottom: 1px solid #22303e; padding: 9px; font-family: 'DejaVu Sans Mono'; font-size: 10px; }\n"
            "QDialog { background:",
        )

    BaseMonitoringPage = ui.MonitoringPage
    BaseMainWindow = ui.MainWindow

    class NativeVideoSurface(QWidget):
        """One persistent X11 child window for nveglglessink.

        Qt never paints camera cards over this window. The child WId is created once
        after the top-level window reaches its final maximized geometry and is only
        rebound when X11 actually gives the child a different WId.
        """

        xidChanged = Signal(int)
        doubleClicked = Signal(float, float)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._last_xid = 0
            self.setMouseTracking(True)
            self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
            self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)
            self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
            self.setAutoFillBackground(False)
            self.setStyleSheet("background:#000;")

        def _publish_xid(self) -> None:
            try:
                xid = int(self.winId())
            except Exception:
                return
            if xid > 0 and xid != self._last_xid:
                self._last_xid = xid
                self.xidChanged.emit(xid)

        def showEvent(self, event):
            super().showEvent(event)
            QTimer.singleShot(0, self._publish_xid)

        def event(self, event):
            result = super().event(event)
            if event.type() == QEvent.Type.WinIdChange:
                QTimer.singleShot(0, self._publish_xid)
            return result

        def paintEvent(self, event):
            # nveglglessink is the only painter for this native drawable.
            event.accept()

        def mouseDoubleClickEvent(self, event):
            if self.width() > 0 and self.height() > 0:
                pos = event.position()
                nx = max(0.0, min(0.999999, float(pos.x()) / float(self.width())))
                ny = max(0.0, min(0.999999, float(pos.y()) / float(self.height())))
                self.doubleClicked.emit(nx, ny)
            event.accept()

    class StableLiveWall(QWidget):
        """Stable native video host with no transparent child overlays."""

        fullscreenRequested = Signal(int)
        GRID_ASPECT = (16.0 * 2.0) / (9.0 * 3.0)  # exact 2x3 wall = 32:27
        FOCUS_ASPECT = 16.0 / 9.0

        def __init__(self, controller, parent=None):
            super().__init__(parent)
            self.controller = controller
            self.focus_source: int | None = None
            self._boot_requested = False
            self._bound_xid = 0
            self.setMinimumSize(720, 600)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.setStyleSheet("background:#090d12;")

            self.video = NativeVideoSurface(self)
            self.video.xidChanged.connect(self._on_xid_changed)
            self.video.doubleClicked.connect(self._on_double_click)
            self.video.show()

        def _target_aspect(self) -> float:
            return self.FOCUS_ASPECT if self.focus_source is not None else self.GRID_ASPECT

        def _layout_video(self) -> None:
            width = max(1, self.width())
            height = max(1, self.height())
            target = self._target_aspect()
            current = float(width) / float(height)
            if current > target:
                video_h = height
                video_w = max(1, int(round(video_h * target)))
                x = (width - video_w) // 2
                y = 0
            else:
                video_w = width
                video_h = max(1, int(round(video_w / target)))
                x = 0
                y = (height - video_h) // 2
            self.video.setGeometry(x, y, video_w, video_h)

        def _on_xid_changed(self, xid: int) -> None:
            xid = int(xid)
            if xid <= 0 or xid == self._bound_xid:
                return
            self._bound_xid = xid
            if not self._boot_requested:
                self._boot_requested = True
                self.controller.start(xid)
            else:
                self.controller.bind_window(xid)

        def _start_if_needed(self) -> None:
            self._layout_video()
            self.video._publish_xid()
            if not self._boot_requested:
                xid = int(self.video.winId())
                if xid > 0:
                    self._on_xid_changed(xid)

        def showEvent(self, event):
            super().showEvent(event)
            self._layout_video()
            # Start only after the window has settled into its final geometry.
            QTimer.singleShot(120, self._start_if_needed)

        def resizeEvent(self, event):
            self._layout_video()
            QWidget.resizeEvent(self, event)

        def _on_double_click(self, nx: float, ny: float) -> None:
            if self.focus_source is not None:
                self.fullscreenRequested.emit(int(self.focus_source))
                return
            col = min(1, max(0, int(nx * 2.0)))
            row = min(2, max(0, int(ny * 3.0)))
            self.fullscreenRequested.emit(row * 2 + col)

        def set_focus(self, source_id: int | None):
            self.focus_source = source_id
            self.controller.set_focus_source(source_id)
            self._layout_video()

        def refresh(self, _snapshot: dict):
            # Camera status/person OSD comes from DeepStream itself. Keeping this
            # method intentionally paint-free avoids a second UI compositor owner.
            pass

    class StableMonitoringPage(BaseMonitoringPage):
        def __init__(self, controller):
            super().__init__(controller)
            # More vertical room for the 32:27 video wall and a less cramped rail.
            self.layout.setContentsMargins(14, 8, 14, 10)
            self.layout.setSpacing(12)
            self.recent_panel.setMinimumWidth(280)

    # MonitoringPage resolves LiveWall at construction time.
    ui.LiveWall = StableLiveWall
    ui.MonitoringPage = StableMonitoringPage

    class StableMainWindow(BaseMainWindow):
        """Keep one top-level window mode so the native video WId stays stable."""

        def __init__(self):
            super().__init__()
            self.sidebar.setFixedWidth(200)
            self.header.setFixedHeight(60)

        def enter_camera_fullscreen(self, source_id: int):
            self._camera_fullscreen = source_id
            self.sidebar.hide()
            self.header.hide()
            self.monitoring.set_camera_fullscreen(source_id)
            # Do not call showFullScreen(): changing the top-level X11 mode can
            # recreate native children and detach GstVideoOverlay from its WId.

        def exit_camera_fullscreen(self):
            self._camera_fullscreen = None
            self.monitoring.set_camera_fullscreen(None)
            self.sidebar.show()
            self.header.show()

        def toggle_grid_fullscreen(self):
            if self._camera_fullscreen is not None:
                self.exit_camera_fullscreen()
                return
            self._grid_fullscreen = not self._grid_fullscreen
            self.sidebar.setVisible(not self._grid_fullscreen)
            self.header.setVisible(not self._grid_fullscreen)

    ui.MainWindow = StableMainWindow


def main() -> int:
    if os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    mp.freeze_support()
    _install_reference_contract()

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Sentinel VMS")
    app.setOrganizationName("Sentinel")
    app.setStyle("Fusion")
    app.setStyleSheet(ui.APP_QSS)

    window = ui.MainWindow()
    # Maximize before LiveWall starts its child process so its native WId is not
    # created at one geometry and immediately reparented/resized into another.
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
