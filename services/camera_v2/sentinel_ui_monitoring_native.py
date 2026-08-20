from __future__ import annotations

"""Production Monitoring page with one stable native video target.

The camera pixels are rendered by GstVideoOverlay into a dedicated QWindow that
Qt embeds with QWidget.createWindowContainer(). All Qt controls live outside that
native child, so normal UI repaints cannot cover or invalidate the EGL surface.

The DeepStream wall remains a fixed 2x3 grid for the lifetime of the pipeline.
Fullscreen only changes the Qt shell/layout; it never mutates the tiler while the
pipeline is PLAYING.
"""

import time

from PySide6.QtCore import QEvent, QTimer, Qt, Signal
from PySide6.QtGui import QWindow
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


class NativeVideoHost(QWidget):
    """Own exactly one embedded QWindow and publish its platform window id."""

    nativeReady = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("nativeVideoHost")
        self.setMinimumSize(720, 608)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._last_emitted_xid = 0

        self.video_window = QWindow()
        self.video_window.setObjectName("sentinelVideoWindow")

        self.container = QWidget.createWindowContainer(self.video_window, self)
        self.container.setObjectName("nativeVideoContainer")
        self.container.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.container.setMinimumSize(640, 540)
        self.container.setStyleSheet("background:#020507;border:0;")
        self.video_window.installEventFilter(self)
        self.container.installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.container, 1)

    def publish_current_xid(self, *, force: bool = False) -> None:
        """Publish the current embedded X11 handle.

        ``createWindowContainer`` may recreate its platform child while the Qt
        widget object survives. A forced publish is also needed after a camera
        process exits: the replacement process must receive the same still-valid
        XID instead of waiting forever for a different one.
        """
        if not self.isVisible() or not self.container.isVisible():
            return
        try:
            xid = int(self.video_window.winId())
        except Exception as exc:
            print(
                f"SENTINEL_VIDEO_SURFACE xid_error={type(exc).__name__}:{exc}",
                flush=True,
            )
            return
        if xid <= 0 or (not force and xid == self._last_emitted_xid):
            return
        self._last_emitted_xid = xid
        print(
            f"SENTINEL_VIDEO_SURFACE mode=qwindow-container xid={xid} "
            f"size={self.width()}x{self.height()}",
            flush=True,
        )
        self.nativeReady.emit(xid)

    # Compatibility for the existing preflight and older callers.
    def _publish_xid(self) -> None:
        self.publish_current_xid()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # createWindowContainer owns the embedded QWindow's visibility/parenting.
        QTimer.singleShot(180, self.publish_current_xid)
        QTimer.singleShot(700, self.publish_current_xid)

    def event(self, event) -> bool:
        result = super().event(event)
        if event.type() in (QEvent.Type.Show, QEvent.Type.ParentChange):
            QTimer.singleShot(120, self.publish_current_xid)
        return result

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        # QWindow platform surfaces can be destroyed/recreated without replacing
        # the Python object. Re-publish after Qt finishes the native transition so
        # GstVideoOverlay never remains attached to a stale XID.
        if watched in (self.video_window, self.container) and event.type() in (
            QEvent.Type.Show,
            QEvent.Type.ParentChange,
            QEvent.Type.WinIdChange,
            QEvent.Type.PlatformSurface,
        ):
            # Show/parent transitions may require a repaint even if X11 reused the
            # same numeric handle, so deliberately rebind that handle.
            QTimer.singleShot(
                0, lambda: self.publish_current_xid(force=True)
            )
        return super().eventFilter(watched, event)


class CameraStatusRow(QFrame):
    def __init__(self, camera_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("cameraStatusRow")
        self.setStyleSheet(
            "QFrame#cameraStatusRow{background:#091018;border:1px solid #17232d;"
            "border-radius:5px;}"
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(9, 6, 9, 6)
        row.setSpacing(7)

        self.dot = QLabel("●")
        self.dot.setFixedWidth(10)
        self.dot.setStyleSheet("color:#5d6974;font-size:9px;")
        row.addWidget(self.dot)

        self.name = QLabel(camera_id)
        self.name.setStyleSheet(
            "color:#dbe4eb;font:700 9px 'DejaVu Sans Mono';"
        )
        row.addWidget(self.name)
        row.addStretch()

        self.fps = QLabel("--")
        self.fps.setStyleSheet(
            f"color:{C['muted']};font:8px 'DejaVu Sans Mono';"
        )
        row.addWidget(self.fps)

    def update_state(self, *, online: bool, fps: float) -> None:
        if online:
            self.dot.setStyleSheet("color:#39d995;font-size:9px;")
            self.fps.setText(f"{fps:.1f} fps")
            self.fps.setStyleSheet(
                "color:#7fd8b6;font:8px 'DejaVu Sans Mono';"
            )
        else:
            self.dot.setStyleSheet("color:#ef6666;font-size:9px;")
            self.fps.setText("offline")
            self.fps.setStyleSheet(
                "color:#b86a6a;font:8px 'DejaVu Sans Mono';"
            )


class MonitoringPage(QWidget):
    """Fixed 2x3 camera wall with a compact production status rail."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("pageRoot")
        self.camera_configs = list(load_settings().cameras)
        self.controller = ProPipelineController()
        self._last_bound_xid = 0
        self._last_bind_at = 0.0
        self._fullscreen_active = False
        self._restart_not_before = 0.0
        self._restart_delay = 0.5

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(12)
        self._root_layout = root

        self.wall_card = QFrame(self)
        self.wall_card.setObjectName("monitorWallCard")
        self.wall_card.setStyleSheet(
            "QFrame#monitorWallCard{background:#05090d;border:1px solid #1b2a36;"
            "border-radius:8px;}"
        )
        wall_layout = QVBoxLayout(self.wall_card)
        wall_layout.setContentsMargins(6, 6, 6, 6)
        wall_layout.setSpacing(0)

        self.surface = NativeVideoHost(self.wall_card)
        self.surface.nativeReady.connect(self._start_or_bind)
        wall_layout.addWidget(self.surface, 1)
        root.addWidget(self.wall_card, 1)

        self.identity_panel = QFrame(self)
        self.identity_panel.setObjectName("monitorRail")
        self.identity_panel.setFixedWidth(270)
        self.identity_panel.setStyleSheet(
            "QFrame#monitorRail{background:#0b1219;border:1px solid #1d2b37;"
            "border-radius:8px;}"
        )
        rail = QVBoxLayout(self.identity_panel)
        rail.setContentsMargins(14, 14, 14, 14)
        rail.setSpacing(10)

        summary_head = QHBoxLayout()
        summary_title = QLabel("People in Building")
        summary_title.setStyleSheet(
            "color:#e8eef4;font-size:11px;font-weight:800;"
        )
        summary_head.addWidget(summary_title)
        summary_head.addStretch()
        self.pipeline_state = QLabel("STARTING")
        self.pipeline_state.setStyleSheet(self._state_style("starting"))
        summary_head.addWidget(self.pipeline_state)
        rail.addLayout(summary_head)

        self.total_value = QLabel("0")
        self.total_value.setStyleSheet(
            f"color:{C['text']};font-size:38px;font-weight:850;letter-spacing:-1px;"
        )
        rail.addWidget(self.total_value)

        split = QHBoxLayout()
        split.setSpacing(12)
        self.known_value, known_box = self._metric_box("KNOWN", C["known"])
        self.unknown_value, unknown_box = self._metric_box("UNKNOWN", C["unknown"])
        split.addWidget(known_box, 1)
        split.addWidget(unknown_box, 1)
        rail.addLayout(split)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background:#17232d;border:0;")
        rail.addWidget(divider)

        cameras_head = QHBoxLayout()
        cameras_title = QLabel("Cameras")
        cameras_title.setStyleSheet(
            "color:#dfe7ed;font-size:10px;font-weight:750;"
        )
        cameras_head.addWidget(cameras_title)
        cameras_head.addStretch()
        self.camera_summary = QLabel("0/0")
        self.camera_summary.setStyleSheet(
            f"color:{C['muted']};font:8px 'DejaVu Sans Mono';"
        )
        cameras_head.addWidget(self.camera_summary)
        rail.addLayout(cameras_head)

        self.camera_rows: dict[int, CameraStatusRow] = {}
        for source_id, camera in enumerate(self.camera_configs[:6]):
            camera_id = str(
                getattr(camera, "camera_id", getattr(camera, "id", f"CAM-{source_id + 1:02d}"))
            )
            row_widget = CameraStatusRow(camera_id, self.identity_panel)
            self.camera_rows[source_id] = row_widget
            rail.addWidget(row_widget)

        rail.addStretch()

        hint = QLabel("2 × 3 live wall")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(
            f"color:{C['muted']};font:8px 'DejaVu Sans Mono';padding:4px;"
        )
        rail.addWidget(hint)
        root.addWidget(self.identity_panel, 0)

        self.poll_timer = self.startTimer(250)
        QTimer.singleShot(350, self._ensure_started)

    @staticmethod
    def _metric_box(title: str, color: str):
        box = QFrame()
        box.setStyleSheet(
            "QFrame{background:#091018;border:1px solid #17232d;border-radius:6px;}"
        )
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        heading = QLabel(title)
        heading.setStyleSheet(
            f"color:{C['muted']};font:700 8px 'DejaVu Sans Mono';"
        )
        value = QLabel("0")
        value.setStyleSheet(f"color:{color};font-size:21px;font-weight:850;")
        layout.addWidget(heading)
        layout.addWidget(value)
        return value, box

    @staticmethod
    def _state_style(state: str) -> str:
        if state == "live":
            return (
                "color:#39d995;background:#0a1b17;border:1px solid #174238;"
                "border-radius:4px;padding:4px 6px;font:700 8px 'DejaVu Sans Mono';"
            )
        if state == "error":
            return (
                "color:#ef6666;background:#211215;border:1px solid #4a2529;"
                "border-radius:4px;padding:4px 6px;font:700 8px 'DejaVu Sans Mono';"
            )
        return (
            "color:#e9b84b;background:#201a0e;border:1px solid #4b3b1d;"
            "border-radius:4px;padding:4px 6px;font:700 8px 'DejaVu Sans Mono';"
        )

    def _start_or_bind(self, xid: int) -> None:
        xid = int(xid)
        if xid <= 0:
            return

        process = self.controller.process
        if process is not None and not process.is_alive():
            self.controller.stop()
            process = None
            self._last_bound_xid = 0
            self._last_bind_at = 0.0

        now = time.monotonic()
        if (
            process is not None
            and process.is_alive()
            and xid == self._last_bound_xid
            and now - self._last_bind_at < 0.15
        ):
            return

        action = "start" if process is None else "rebind"
        print(f"SENTINEL_UI_BIND action={action} xid={xid}", flush=True)
        self.controller.start_or_bind(xid)
        self._last_bound_xid = xid
        self._last_bind_at = now

    def _ensure_started(self) -> None:
        process = self.controller.process
        if process is not None and process.is_alive():
            return

        now = time.monotonic()
        if now < self._restart_not_before:
            return

        if process is not None:
            # Reap/reset the failed child and its queues before a clean restart.
            self.controller.stop()
            self._last_bound_xid = 0
            self._last_bind_at = 0.0

        self._restart_not_before = now + self._restart_delay
        self._restart_delay = min(15.0, self._restart_delay * 2.0)
        self.surface.publish_current_xid(force=True)

    def timerEvent(self, event) -> None:
        if event.timerId() != self.poll_timer:
            super().timerEvent(event)
            return

        self._ensure_started()
        self.surface.publish_current_xid()
        status, metrics = self.controller.poll()
        state = str(getattr(status, "state", "STARTING") or "STARTING").upper()

        camera_rows = {
            int(row.get("source_id", -1)): row
            for row in metrics.get("cameras", [])
            if isinstance(row, dict)
        }
        online_count = 0
        for source_id, widget in self.camera_rows.items():
            row = camera_rows.get(source_id, {})
            online = bool(row.get("online"))
            fps = float(row.get("fps", 0.0) or 0.0)
            widget.update_state(online=online, fps=fps)
            online_count += int(online)

        configured = max(1, len(self.camera_configs))
        self.camera_summary.setText(f"{online_count}/{configured}")

        total = max(0, int(metrics.get("total_people", 0) or 0))
        known = max(0, int(metrics.get("known_people", 0) or 0))
        known = min(known, total)
        unknown = max(0, total - known)
        self.total_value.setText(str(total))
        self.known_value.setText(str(known))
        self.unknown_value.setText(str(unknown))

        if online_count > 0:
            self._restart_delay = 0.5
            self.pipeline_state.setText(
                "LIVE" if online_count == configured else f"LIVE {online_count}/{configured}"
            )
            self.pipeline_state.setStyleSheet(self._state_style("live"))
        elif state in {"ERROR", "STOPPED"}:
            self.pipeline_state.setText(state)
            self.pipeline_state.setStyleSheet(self._state_style("error"))
        else:
            self.pipeline_state.setText(state)
            self.pipeline_state.setStyleSheet(self._state_style("starting"))

    def _set_app_fullscreen_shell(self, enabled: bool) -> None:
        top = self.window()
        setter = getattr(top, "set_monitoring_fullscreen", None)
        if callable(setter):
            setter(bool(enabled))

    def open_fullscreen_grid(self) -> None:
        if self._fullscreen_active:
            return
        self._fullscreen_active = True
        self.identity_panel.hide()
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(0)
        self.wall_card.setStyleSheet(
            "QFrame#monitorWallCard{background:#020507;border:0;border-radius:0;}"
        )
        self._set_app_fullscreen_shell(True)

    def exit_fullscreen(self) -> None:
        if not self._fullscreen_active:
            return
        self._fullscreen_active = False
        self._set_app_fullscreen_shell(False)
        self.identity_panel.show()
        self._root_layout.setContentsMargins(12, 10, 12, 12)
        self._root_layout.setSpacing(12)
        self.wall_card.setStyleSheet(
            "QFrame#monitorWallCard{background:#05090d;border:1px solid #1b2a36;"
            "border-radius:8px;}"
        )

    def shutdown(self) -> None:
        if self._fullscreen_active:
            self.exit_fullscreen()
        try:
            self.killTimer(self.poll_timer)
        except Exception:
            pass
        self.controller.stop()
        try:
            self.surface.video_window.close()
        except Exception:
            pass
