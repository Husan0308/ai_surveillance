from __future__ import annotations

"""Camera-only Qt host for the fixed six-camera DeepStream wall.

This module intentionally contains no dashboard widgets, overlays, navigation,
counts, cards, or stacked pages. GstVideoOverlay renders into exactly one QWindow
embedded by Qt. The RF-DETR/NvDCF/DeepStream runtime remains unchanged.
"""

from PySide6.QtCore import QEvent, QTimer, Qt, Signal
from PySide6.QtGui import QWindow
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from .sentinel_video_wall_ui import ProPipelineController


class NativeVideoHost(QWidget):
    """Own one embedded native QWindow and expose its X11 window id once."""

    nativeReady = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("nativeVideoHost")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 480)
        self._last_emitted_xid = 0

        self.video_window = QWindow()
        self.video_window.setObjectName("sentinelVideoWindow")

        self.container = QWidget.createWindowContainer(self.video_window, self)
        self.container.setObjectName("nativeVideoContainer")
        self.container.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.container.setMinimumSize(640, 480)
        self.container.setStyleSheet("background:#000000;border:0;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.container, 1)

    def _publish_xid(self) -> None:
        if not self.isVisible():
            return
        try:
            self.video_window.create()
            xid = int(self.video_window.winId())
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
            f"SENTINEL_VIDEO_SURFACE mode=camera-only-qwindow xid={xid} "
            f"size={self.width()}x{self.height()}",
            flush=True,
        )
        self.nativeReady.emit(xid)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(120, self._publish_xid)
        QTimer.singleShot(500, self._publish_xid)

    def event(self, event) -> bool:
        result = super().event(event)
        if event.type() in (QEvent.Type.Show, QEvent.Type.ParentChange):
            QTimer.singleShot(100, self._publish_xid)
        return result


class MonitoringPage(QWidget):
    """Only the fixed 2x3 live camera wall; no other UI is constructed."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("cameraOnlyPage")
        self.controller = ProPipelineController()
        self._last_bound_xid = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.surface = NativeVideoHost(self)
        self.surface.nativeReady.connect(self._start_or_bind)
        layout.addWidget(self.surface, 1)

        # Keep draining runtime status/metrics so the IPC queue cannot accumulate,
        # but intentionally render none of it in this temporary camera-only UI.
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
        try:
            self.surface.video_window.close()
        except Exception:
            pass
