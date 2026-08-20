from __future__ import annotations

"""Camera-only Qt host for the fixed six-camera DeepStream wall."""

from PySide6.QtCore import QEvent, QTimer, Qt, Signal
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from .camera_wall_runtime import CameraWallController, GRID_COLUMNS, GRID_ROWS


class NativeVideoHost(QWidget):
    """Persistent native QWidget used directly as the display/X11 target."""

    nativeReady = Signal(int)
    cameraClicked = Signal(int)
    escapeRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("nativeVideoHost")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 480)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._last_emitted_xid = 0

        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)
        self.setAutoFillBackground(False)

    def paintEngine(self):
        return None

    def _publish_xid(self) -> None:
        if not self.isVisible():
            return
        try:
            xid = int(self.winId())
        except Exception as exc:
            print(
                f"SENTINEL_VIDEO_SURFACE xid_error={type(exc).__name__}:{exc}",
                flush=True,
            )
            return
        if xid <= 0 or xid == self._last_emitted_xid:
            return
        self._last_emitted_xid = xid
        print(
            f"SENTINEL_VIDEO_SURFACE mode=direct-native-qwidget xid={xid} "
            f"size={self.width()}x{self.height()}",
            flush=True,
        )
        self.nativeReady.emit(xid)

    def _grid_source_at(self, x: float, y: float) -> int | None:
        """Map a click through the aspect-fitted 2x3 wall, ignoring black bars."""

        width = float(max(1, self.width()))
        height = float(max(1, self.height()))
        wall_aspect = (16.0 * GRID_COLUMNS) / (9.0 * GRID_ROWS)
        widget_aspect = width / height

        if widget_aspect > wall_aspect:
            content_h = height
            content_w = content_h * wall_aspect
            left = (width - content_w) * 0.5
            top = 0.0
        else:
            content_w = width
            content_h = content_w / wall_aspect
            left = 0.0
            top = (height - content_h) * 0.5

        if x < left or x >= left + content_w or y < top or y >= top + content_h:
            return None

        nx = (x - left) / max(1.0, content_w)
        ny = (y - top) / max(1.0, content_h)
        column = min(GRID_COLUMNS - 1, max(0, int(nx * GRID_COLUMNS)))
        row = min(GRID_ROWS - 1, max(0, int(ny * GRID_ROWS)))
        source_id = row * GRID_COLUMNS + column
        return source_id if 0 <= source_id < GRID_COLUMNS * GRID_ROWS else None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            pos = event.position()
            source_id = self._grid_source_at(pos.x(), pos.y())
            if source_id is not None:
                self.cameraClicked.emit(int(source_id))
                event.accept()
                return
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.escapeRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(120, self._publish_xid)
        QTimer.singleShot(400, self._publish_xid)

    def event(self, event) -> bool:
        result = super().event(event)
        if event.type() == QEvent.Type.WinIdChange and self.isVisible():
            QTimer.singleShot(0, self._publish_xid)
        return result


class MonitoringPage(QWidget):
    """Only the fixed 2x3 live camera wall with click-to-fullscreen focus."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("cameraOnlyPage")
        self.controller = CameraWallController()
        self._last_bound_xid = 0
        self._focused_source = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.surface = NativeVideoHost(self)
        self.surface.nativeReady.connect(self._start_or_bind)
        self.surface.cameraClicked.connect(self._camera_clicked)
        self.surface.escapeRequested.connect(self.exit_fullscreen)
        layout.addWidget(self.surface, 1)

        self.poll_timer = self.startTimer(250)
        QTimer.singleShot(250, self._ensure_started)

    def _start_or_bind(self, xid: int) -> None:
        xid = int(xid)
        if xid <= 0:
            return

        process = self.controller.process
        if process is not None and not process.is_alive():
            self.controller.stop()
            process = None
            self._last_bound_xid = 0

        if process is not None and process.is_alive() and xid == self._last_bound_xid:
            return

        action = "start" if process is None else "rebind"
        print(f"SENTINEL_UI_BIND action={action} xid={xid}", flush=True)
        self.controller.start_or_bind(xid)
        self._last_bound_xid = xid

    def _camera_clicked(self, source_id: int) -> None:
        sid = int(source_id)
        if sid == self._focused_source:
            self.exit_fullscreen()
            return
        self._focused_source = sid
        self.controller.focus(sid)
        window = self.window()
        if window is not None:
            window.showFullScreen()
        self.surface.setFocus(Qt.FocusReason.MouseFocusReason)
        print(f"SENTINEL_UI_FOCUS source={sid} mode=fullscreen", flush=True)

    def exit_fullscreen(self) -> None:
        if self._focused_source < 0:
            return
        self._focused_source = -1
        self.controller.focus(-1)
        window = self.window()
        if window is not None:
            window.showMaximized()
        self.surface.setFocus(Qt.FocusReason.OtherFocusReason)
        print("SENTINEL_UI_FOCUS source=-1 mode=grid", flush=True)

    def _ensure_started(self) -> None:
        if self.controller.process is None:
            self.surface._publish_xid()

    def timerEvent(self, event) -> None:
        if event.timerId() != self.poll_timer:
            super().timerEvent(event)
            return
        self._ensure_started()
        self.controller.poll()

    def shutdown(self) -> None:
        try:
            self.killTimer(self.poll_timer)
        except Exception:
            pass
        self.controller.stop()
