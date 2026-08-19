from __future__ import annotations

from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from services.ml_service.app.config import load_settings

from .sentinel_ui_base import C, Panel
from .sentinel_video_wall_ui import ProLiveVideoWall, ProPipelineController


class MonitoringPage(QWidget):
    """Compact production monitoring wall + occupancy rail.

    The GStreamer sink renders into a native Qt child window. Qt documents that
    QWidget.winId() may change at runtime, so the binding must be refreshed after
    show/native-id/fullscreen transitions instead of being treated as one-shot.
    """

    def __init__(self):
        super().__init__()
        self.setObjectName("pageRoot")
        self.controller = ProPipelineController()
        self._fullscreen_active = False
        self._focused_source: int | None = None
        self._last_bound_xid = 0
        self._bind_watchdog_ticks = 0
        self.camera_configs = list(load_settings().cameras)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(7, 6, 7, 7)
        self.layout.setSpacing(8)

        self.camera_panel = QWidget(self)
        self.camera_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        camera_column = QVBoxLayout(self.camera_panel)
        camera_column.setContentsMargins(0, 0, 0, 0)
        camera_column.setSpacing(0)

        self.wall = ProLiveVideoWall(self.camera_configs, [], self.camera_panel)
        self.wall.installEventFilter(self)
        self.wall.nativeReady.connect(self._start_or_bind)
        self.wall.cameraDoubleClicked.connect(self.expand)
        self.wall.heatmapToggled.connect(self.controller.set_heatmap)
        self.wall.exitFullscreenRequested.connect(self.exit_fullscreen)
        camera_column.addWidget(self.wall, 1)
        self.layout.addWidget(self.camera_panel, 1)

        self.identity_panel = QWidget(self)
        self.identity_panel.setFixedWidth(262)
        identity_rail = QVBoxLayout(self.identity_panel)
        identity_rail.setContentsMargins(0, 0, 0, 0)
        identity_rail.setSpacing(8)

        summary = Panel()
        summary.setFixedHeight(154)
        summary.setStyleSheet(
            "QFrame#panel{background:#0b1219;border:1px solid #1d2b37;border-radius:7px;}"
        )
        summary_layout = QVBoxLayout(summary)
        summary_layout.setContentsMargins(13, 11, 13, 11)
        summary_layout.setSpacing(7)

        summary_head = QHBoxLayout()
        summary_head.setSpacing(6)
        heading = QLabel("People in Building")
        heading.setStyleSheet("color:#e7edf3;font-size:11px;font-weight:750;")
        summary_head.addWidget(heading)
        summary_head.addStretch()
        self.live_badge = QLabel("STARTING")
        self.live_badge.setStyleSheet(self._status_style("starting"))
        summary_head.addWidget(self.live_badge)
        summary_layout.addLayout(summary_head)

        total_row = QHBoxLayout()
        total_row.setSpacing(6)
        self.total_value = QLabel("0")
        self.total_value.setStyleSheet(
            f"color:{C['text']};font-size:35px;font-weight:850;letter-spacing:-1px;"
        )
        total_row.addWidget(self.total_value)
        total_caption = QLabel("people")
        total_caption.setStyleSheet(
            f"color:{C['muted']};font-size:10px;padding-top:13px;"
        )
        total_row.addWidget(total_caption)
        total_row.addStretch()
        summary_layout.addLayout(total_row)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background:#17232d;border:0;")
        summary_layout.addWidget(divider)

        split = QHBoxLayout()
        split.setSpacing(8)
        self.known_value, known_box = self._summary_metric("KNOWN", C["known"])
        self.unknown_value, unknown_box = self._summary_metric("UNKNOWN", C["unknown"])
        split.addWidget(known_box, 1)
        split.addWidget(unknown_box, 1)
        summary_layout.addLayout(split)
        identity_rail.addWidget(summary)

        recent_panel = Panel()
        recent_panel.setStyleSheet(
            "QFrame#panel{background:#0b1219;border:1px solid #1d2b37;border-radius:7px;}"
        )
        recent_layout = QVBoxLayout(recent_panel)
        recent_layout.setContentsMargins(12, 12, 12, 12)
        recent_layout.setSpacing(0)

        recent_head = QHBoxLayout()
        recent_title = QLabel("Recent Views")
        recent_title.setStyleSheet("font-size:11px;font-weight:750;color:#e7edf3;")
        recent_head.addWidget(recent_title)
        recent_head.addStretch()
        self.active_count = QLabel("0 active")
        self.active_count.setStyleSheet(
            f"color:{C['muted']};font:9px 'DejaVu Sans Mono';"
        )
        recent_head.addWidget(self.active_count)
        recent_layout.addLayout(recent_head)
        recent_layout.addSpacing(10)

        empty = QFrame()
        empty.setStyleSheet(
            "QFrame{background:#091018;border:1px dashed #22313f;border-radius:6px;}"
        )
        empty_layout = QVBoxLayout(empty)
        empty_layout.setContentsMargins(12, 13, 12, 13)
        empty_layout.setSpacing(3)
        empty_title = QLabel("Recent activity")
        empty_title.setStyleSheet("font-size:10px;font-weight:700;color:#dbe5ec;")
        empty_layout.addWidget(empty_title)
        empty_text = QLabel("Live face/event crop source ulanganda shu yerda real viewlar chiqadi.")
        empty_text.setWordWrap(True)
        empty_text.setStyleSheet(f"color:{C['muted']};font-size:9px;")
        empty_layout.addWidget(empty_text)
        recent_layout.addWidget(empty)
        recent_layout.addStretch()

        self.view_events = QToolButton()
        self.view_events.setText("View all events")
        self.view_events.setCursor(Qt.PointingHandCursor)
        self.view_events.setStyleSheet(
            "QToolButton{background:#0a1118;color:#cbd6de;border:1px solid #1d2b37;"
            "border-radius:6px;padding:7px 10px;font-size:10px;font-weight:650;}"
            "QToolButton:hover{background:#111c25;border-color:#304555;color:#ffffff;}"
        )
        self.view_events.clicked.connect(self._open_events_page)
        recent_layout.addWidget(self.view_events)

        identity_rail.addWidget(recent_panel, 1)
        self.layout.addWidget(self.identity_panel, 0)

        self.poll_timer = self.startTimer(250)
        QTimer.singleShot(0, lambda: self._ensure_video_binding(True, "startup-0"))
        QTimer.singleShot(180, lambda: self._ensure_video_binding(True, "startup-180"))
        QTimer.singleShot(500, lambda: self._ensure_video_binding(True, "startup-500"))

    @staticmethod
    def _status_style(state: str) -> str:
        state = str(state or "").lower()
        if state == "live":
            color, bg, border = C["known"], "#0a1b17", "#174238"
        elif state in {"error", "offline"}:
            color, bg, border = C["offline"], "#211215", "#4a2529"
        elif state == "warning":
            color, bg, border = C["unknown"], "#211a0e", "#58461f"
        else:
            color, bg, border = C["unknown"], "#201a0e", "#4b3b1d"
        return (
            f"color:{color};background:{bg};border:1px solid {border};"
            "border-radius:4px;padding:3px 6px;font:700 8px 'DejaVu Sans Mono';"
        )

    @staticmethod
    def _summary_metric(heading: str, color: str):
        box = QFrame()
        box.setStyleSheet("QFrame{background:transparent;border:0;}")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(5, 0, 5, 0)
        lay.setSpacing(1)
        title = QLabel(heading)
        title.setStyleSheet(
            f"color:{C['muted']};font:700 8px 'DejaVu Sans Mono';letter-spacing:1px;"
        )
        metric = QLabel("0")
        metric.setStyleSheet(f"color:{color};font-size:20px;font-weight:850;")
        lay.addWidget(title)
        lay.addWidget(metric)
        return metric, box

    def _open_events_page(self) -> None:
        top = self.window()
        switcher = getattr(top, "switch_page", None)
        if callable(switcher):
            switcher(2)

    def _pipeline_alive(self) -> bool:
        process = self.controller.process
        return process is not None and process.is_alive()

    def _bind_window_id(self, xid: int, *, force: bool, reason: str) -> None:
        xid = int(xid)
        if xid <= 0:
            return

        process = self.controller.process
        if process is not None and not process.is_alive():
            # Clear dead process/queues before a clean restart. Otherwise stale
            # ERROR/STOPPED messages can survive into the new UI session.
            self.controller.stop()
            process = None

        if not force and process is not None and process.is_alive() and xid == self._last_bound_xid:
            return

        action = "start" if process is None else "rebind"
        print(f"SENTINEL_UI_BIND action={action} reason={reason} xid={xid}", flush=True)
        self.controller.start_or_bind(xid)
        self._last_bound_xid = xid

    def _ensure_video_binding(self, force: bool = False, reason: str = "watchdog") -> None:
        if reason.startswith("watchdog") and not self.isVisible():
            return
        try:
            xid = int(self.wall.winId())
        except Exception as exc:
            print(f"SENTINEL_UI_BIND_SKIP reason={reason} error={type(exc).__name__}", flush=True)
            return
        self._bind_window_id(xid, force=bool(force), reason=reason)

    def _schedule_rebind(self, reason: str) -> None:
        # Window-manager transitions are asynchronous. Rebind immediately and
        # once again after the native child has settled on the new surface.
        QTimer.singleShot(0, lambda r=reason: self._ensure_video_binding(True, f"{r}-0"))
        QTimer.singleShot(180, lambda r=reason: self._ensure_video_binding(True, f"{r}-180"))
        QTimer.singleShot(450, lambda r=reason: self._ensure_video_binding(True, f"{r}-450"))

    def _start_or_bind(self, xid: int) -> None:
        self._bind_window_id(int(xid), force=True, reason="native-ready")

    def eventFilter(self, watched, event):
        if watched is self.wall:
            event_type = event.type()
            if event_type == QEvent.Type.WinIdChange:
                QTimer.singleShot(
                    0,
                    lambda: self._ensure_video_binding(True, "wall-winid-change"),
                )
            elif event_type == QEvent.Type.Show:
                QTimer.singleShot(
                    0,
                    lambda: self._ensure_video_binding(True, "wall-show"),
                )
            elif event_type == QEvent.Type.ParentChange:
                QTimer.singleShot(
                    0,
                    lambda: self._ensure_video_binding(True, "wall-parent-change"),
                )
        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._schedule_rebind("page-show")

    def timerEvent(self, event):
        if event.timerId() == self.poll_timer:
            self._bind_watchdog_ticks += 1
            if self.controller.process is None:
                self._ensure_video_binding(True, "poll-start")
            elif self._bind_watchdog_ticks % 20 == 0:
                # nveglglessink can lose the child surface across compositor/window
                # state transitions even when the numerical XID stayed the same.
                # A slow 5 s idempotent rebind keeps the video self-healing.
                self._ensure_video_binding(True, "watchdog-5s")

            status, metrics = self.controller.poll()
            self.wall.update_metrics(metrics)
            self.wall.set_pipeline_status(status)

            # Camera metrics are the visual source of truth. A transient GStreamer
            # warning from one source must not make the whole six-camera wall look
            # dead while other cameras are actively producing frames.
            camera_rows = [row for row in metrics.get("cameras", []) if isinstance(row, dict)]
            online_count = sum(1 for row in camera_rows if bool(row.get("online")))
            configured_count = max(1, len(self.camera_configs))
            state = str(getattr(status, "state", "STARTING") or "STARTING").upper()

            if online_count > 0:
                badge_state = "live"
                badge_text = "● LIVE" if online_count == configured_count else f"● {online_count}/{configured_count}"
            elif state in {"ERROR", "STOPPED"}:
                badge_state, badge_text = "error", state
            elif state == "PIPELINE_WARNING":
                badge_state, badge_text = "warning", "WARN"
            else:
                badge_state, badge_text = "starting", "STARTING"

            self.live_badge.setText(badge_text)
            self.live_badge.setStyleSheet(self._status_style(badge_state))

            known = max(0, int(metrics.get("known_people", 0) or 0))
            total = max(0, int(metrics.get("total_people", 0) or 0))
            if known > total:
                known = total
            unknown = max(0, total - known)

            self.total_value.setText(str(total))
            self.known_value.setText(str(known))
            self.unknown_value.setText(str(unknown))
            self.active_count.setText(f"{total} active")
            return
        super().timerEvent(event)

    def _set_app_fullscreen_shell(self, enabled: bool) -> None:
        top = self.window()
        setter = getattr(top, "set_monitoring_fullscreen", None)
        if callable(setter):
            setter(bool(enabled))

    def enter_fullscreen(self, source_id: int | None) -> None:
        if source_id is not None and not (0 <= int(source_id) < len(self.camera_configs)):
            return
        self._fullscreen_active = True
        self._focused_source = None if source_id is None else int(source_id)

        self.controller.focus(self._focused_source)
        self.identity_panel.hide()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.wall.set_fullscreen_mode(True, self._focused_source)
        self._set_app_fullscreen_shell(True)
        self._schedule_rebind("fullscreen-enter")

    def exit_fullscreen(self) -> None:
        if not self._fullscreen_active:
            return
        self.controller.focus(None)
        self._fullscreen_active = False
        self._focused_source = None
        self.wall.set_fullscreen_mode(False, None)
        self.identity_panel.show()
        self.layout.setContentsMargins(7, 6, 7, 7)
        self.layout.setSpacing(8)
        self._set_app_fullscreen_shell(False)
        self._schedule_rebind("fullscreen-exit")

    def open_fullscreen_grid(self) -> None:
        self.enter_fullscreen(None)

    def expand(self, source_id: int) -> None:
        self.enter_fullscreen(int(source_id))

    def shutdown(self) -> None:
        if self._fullscreen_active:
            self.exit_fullscreen()
        try:
            self.killTimer(self.poll_timer)
        except Exception:
            pass
        self.controller.stop()
