from __future__ import annotations

"""Stable Monitoring page for the native DeepStream camera wall.

This page intentionally owns exactly one native X11 child window. GstVideoOverlay
renders only into that child. There are no Qt labels, frames, hover widgets or
paintable containers layered on top of the native video drawable.

The DeepStream tiler remains in its startup 2x3 grid for the lifetime of the
pipeline. Monitoring never changes the tiler's runtime grid-selection properties
while PLAYING; that avoids the SetSingleSourceMode/not-negotiated failures seen
during the older focus/fullscreen implementation.
"""

from PySide6.QtCore import QEvent, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from services.ml_service.app.config import load_settings

from .sentinel_ui_base import C
from .sentinel_video_wall_ui import ProPipelineController


class NativeVideoSurface(QWidget):
    """The one and only native GstVideoOverlay target used by Monitoring."""

    nativeReady = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_emitted_xid = 0
        self.setObjectName("nativeVideoSurface")
        self.setMinimumSize(640, 540)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Keep ancestors as normal Qt widgets. Only this child is native.
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

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
        print(f"SENTINEL_VIDEO_SURFACE xid={xid}", flush=True)
        self.nativeReady.emit(xid)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(120, self._publish_xid)

    def event(self, event) -> bool:
        result = super().event(event)
        if event.type() == QEvent.Type.WinIdChange and self.isVisible():
            QTimer.singleShot(0, self._publish_xid)
        return result


class MonitoringPage(QWidget):
    """Minimal production monitoring shell: one native wall + one metrics rail."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("pageRoot")
        self.camera_configs = list(load_settings().cameras)
        self.controller = ProPipelineController()
        self._last_bound_xid = 0
        self._fullscreen_active = False

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 7, 8, 8)
        root.setSpacing(8)
        self._root_layout = root

        # No intermediate native host and no Qt child overlays over this widget.
        self.surface = NativeVideoSurface(self)
        self.surface.nativeReady.connect(self._start_or_bind)
        root.addWidget(self.surface, 1)

        self.identity_panel = QFrame(self)
        self.identity_panel.setObjectName("monitorRail")
        self.identity_panel.setFixedWidth(252)
        self.identity_panel.setStyleSheet(
            "QFrame#monitorRail{background:#0b1219;border:1px solid #1d2b37;"
            "border-radius:7px;}"
        )
        rail = QVBoxLayout(self.identity_panel)
        rail.setContentsMargins(14, 13, 14, 13)
        rail.setSpacing(10)

        title = QLabel("People in Building")
        title.setStyleSheet("color:#e7edf3;font-size:11px;font-weight:750;")
        rail.addWidget(title)

        self.total_value = QLabel("0")
        self.total_value.setStyleSheet(
            f"color:{C['text']};font-size:36px;font-weight:850;"
        )
        rail.addWidget(self.total_value)

        split = QHBoxLayout()
        split.setSpacing(16)
        known_box = QVBoxLayout()
        unknown_box = QVBoxLayout()

        known_title = QLabel("KNOWN")
        known_title.setStyleSheet(
            f"color:{C['muted']};font:700 8px 'DejaVu Sans Mono';"
        )
        self.known_value = QLabel("0")
        self.known_value.setStyleSheet(
            f"color:{C['known']};font-size:20px;font-weight:850;"
        )
        known_box.addWidget(known_title)
        known_box.addWidget(self.known_value)

        unknown_title = QLabel("UNKNOWN")
        unknown_title.setStyleSheet(
            f"color:{C['muted']};font:700 8px 'DejaVu Sans Mono';"
        )
        self.unknown_value = QLabel("0")
        self.unknown_value.setStyleSheet(
            f"color:{C['unknown']};font-size:20px;font-weight:850;"
        )
        unknown_box.addWidget(unknown_title)
        unknown_box.addWidget(self.unknown_value)

        split.addLayout(known_box, 1)
        split.addLayout(unknown_box, 1)
        rail.addLayout(split)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background:#17232d;border:0;")
        rail.addWidget(divider)

        self.camera_state = QLabel("Cameras starting…")
        self.camera_state.setWordWrap(True)
        self.camera_state.setStyleSheet(
            f"color:{C['muted']};font:9px 'DejaVu Sans Mono';"
        )
        rail.addWidget(self.camera_state)

        self.pipeline_state = QLabel("STARTING")
        self.pipeline_state.setStyleSheet(
            "color:#e9b84b;background:#201a0e;border:1px solid #4b3b1d;"
            "border-radius:4px;padding:5px 7px;font:700 9px 'DejaVu Sans Mono';"
        )
        rail.addWidget(self.pipeline_state)
        rail.addStretch()
        root.addWidget(self.identity_panel, 0)

        self.poll_timer = self.startTimer(250)

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
        if self.controller.process is not None:
            return
        self.surface._publish_xid()

    def timerEvent(self, event) -> None:
        if event.timerId() != self.poll_timer:
            super().timerEvent(event)
            return

        self._ensure_started()
        status, metrics = self.controller.poll()
        state = str(getattr(status, "state", "STARTING") or "STARTING").upper()

        camera_rows = [
            row for row in metrics.get("cameras", []) if isinstance(row, dict)
        ]
        online_count = sum(1 for row in camera_rows if bool(row.get("online")))
        configured = max(1, len(self.camera_configs))
        fps_values = [
            float(row.get("fps", 0.0) or 0.0)
            for row in camera_rows
            if bool(row.get("online"))
        ]
        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0

        total = max(0, int(metrics.get("total_people", 0) or 0))
        known = max(0, int(metrics.get("known_people", 0) or 0))
        known = min(known, total)
        unknown = max(0, total - known)

        self.total_value.setText(str(total))
        self.known_value.setText(str(known))
        self.unknown_value.setText(str(unknown))
        self.camera_state.setText(
            f"{online_count}/{configured} cameras online\navg {avg_fps:.1f} fps"
        )

        if online_count > 0:
            text = "LIVE" if online_count == configured else f"LIVE {online_count}/{configured}"
            style = (
                "color:#3ddc97;background:#0a1b17;border:1px solid #174238;"
                "border-radius:4px;padding:5px 7px;font:700 9px 'DejaVu Sans Mono';"
            )
        elif state in {"ERROR", "STOPPED"}:
            text = state
            style = (
                "color:#f06464;background:#211215;border:1px solid #4a2529;"
                "border-radius:4px;padding:5px 7px;font:700 9px 'DejaVu Sans Mono';"
            )
        else:
            text = state
            style = (
                "color:#e9b84b;background:#201a0e;border:1px solid #4b3b1d;"
                "border-radius:4px;padding:5px 7px;font:700 9px 'DejaVu Sans Mono';"
            )
        self.pipeline_state.setText(text)
        self.pipeline_state.setStyleSheet(style)

    def _set_app_fullscreen_shell(self, enabled: bool) -> None:
        top = self.window()
        setter = getattr(top, "set_monitoring_fullscreen", None)
        if callable(setter):
            setter(bool(enabled))

    def open_fullscreen_grid(self) -> None:
        # Safe fullscreen only changes Qt shell geometry. DeepStream stays 2x3.
        if self._fullscreen_active:
            return
        self._fullscreen_active = True
        self.identity_panel.hide()
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(0)
        self._set_app_fullscreen_shell(True)

    def exit_fullscreen(self) -> None:
        if not self._fullscreen_active:
            return
        self._fullscreen_active = False
        self._set_app_fullscreen_shell(False)
        self.identity_panel.show()
        self._root_layout.setContentsMargins(8, 7, 8, 8)
        self._root_layout.setSpacing(8)

    def shutdown(self) -> None:
        if self._fullscreen_active:
            self.exit_fullscreen()
        try:
            self.killTimer(self.poll_timer)
        except Exception:
            pass
        self.controller.stop()
