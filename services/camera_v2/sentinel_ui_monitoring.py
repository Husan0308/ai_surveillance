from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from services.ml_service.app.config import load_settings

from .sentinel_ui_base import C, Panel, label, panel_layout
from .sentinel_video_wall_ui import ProLiveVideoWall, ProPipelineController


class MonitoringPage(QWidget):
    """Live camera monitoring page."""

    def __init__(self):
        super().__init__()
        self.setObjectName("pageRoot")
        self.controller = ProPipelineController()
        self._fullscreen_active = False
        self._focused_source: int | None = None
        self.camera_configs = list(load_settings().cameras)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(12, 10, 12, 12)
        self.layout.setSpacing(12)

        self.camera_panel = QWidget(self)
        self.camera_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        camera_column = QVBoxLayout(self.camera_panel)
        camera_column.setContentsMargins(0, 0, 0, 0)
        camera_column.setSpacing(0)

        self.wall = ProLiveVideoWall(self.camera_configs, [], self.camera_panel)
        self.wall.nativeReady.connect(self._start_or_bind)
        self.wall.cameraDoubleClicked.connect(self.expand)
        self.wall.heatmapToggled.connect(self.controller.set_heatmap)
        self.wall.exitFullscreenRequested.connect(self.exit_fullscreen)
        camera_column.addWidget(self.wall, 1)
        self.layout.addWidget(self.camera_panel, 1)

        self.identity_panel = QWidget(self)
        self.identity_panel.setMinimumWidth(292)
        self.identity_panel.setMaximumWidth(336)
        identity_rail = QVBoxLayout(self.identity_panel)
        identity_rail.setContentsMargins(0, 0, 0, 0)
        identity_rail.setSpacing(10)

        summary = Panel()
        summary.setMinimumHeight(166)
        summary_layout = QVBoxLayout(summary)
        summary_layout.setContentsMargins(16, 14, 16, 14)
        summary_layout.setSpacing(9)

        summary_head = QHBoxLayout()
        heading = QVBoxLayout()
        heading.setSpacing(2)
        heading.addWidget(label("PEOPLE", "eyebrow"))
        heading.addWidget(label("Hozir binoda", "muted"))
        summary_head.addLayout(heading)
        summary_head.addStretch()
        self.live_badge = QLabel("STARTING")
        self.live_badge.setStyleSheet(self._status_style("starting"))
        summary_head.addWidget(self.live_badge)
        summary_layout.addLayout(summary_head)

        total_row = QHBoxLayout()
        total_row.setSpacing(7)
        self.total_value = QLabel("0")
        self.total_value.setStyleSheet(
            f"color:{C['text']};font-size:40px;font-weight:800;letter-spacing:-1px;"
        )
        total_row.addWidget(self.total_value)
        total_caption = QLabel("people")
        total_caption.setStyleSheet(
            f"color:{C['muted']};font-size:11px;padding-top:18px;"
        )
        total_row.addWidget(total_caption)
        total_row.addStretch()
        summary_layout.addLayout(total_row)

        split = QHBoxLayout()
        split.setSpacing(8)
        self.known_value, known_box = self._summary_metric("KNOWN", C["known"])
        self.unknown_value, unknown_box = self._summary_metric("UNKNOWN", C["unknown"])
        split.addWidget(known_box, 1)
        split.addWidget(unknown_box, 1)
        summary_layout.addLayout(split)
        identity_rail.addWidget(summary)

        recent_panel = Panel()
        recent_layout = panel_layout(recent_panel, (14, 14, 14, 14), 0)
        recent_head = QHBoxLayout()
        recent_head.addWidget(label("Recent Views", "sectionTitle"))
        recent_head.addStretch()
        self.active_count = label("0 active", "mono")
        recent_head.addWidget(self.active_count)
        recent_layout.addLayout(recent_head)
        recent_layout.addSpacing(14)

        empty = QFrame()
        empty.setStyleSheet(
            "QFrame{background:#091018;border:1px dashed #22313f;border-radius:7px;}"
        )
        empty_layout = QVBoxLayout(empty)
        empty_layout.setContentsMargins(14, 18, 14, 18)
        empty_layout.setSpacing(5)
        empty_title = QLabel("Recent activity")
        empty_title.setStyleSheet("font-weight:700;")
        empty_layout.addWidget(empty_title)
        empty_text = QLabel("Hozircha recent-view ma'lumotlari yo'q.")
        empty_text.setWordWrap(True)
        empty_text.setStyleSheet(f"color:{C['muted']};font-size:10px;")
        empty_layout.addWidget(empty_text)
        recent_layout.addWidget(empty)
        recent_layout.addStretch()
        identity_rail.addWidget(recent_panel, 1)
        self.layout.addWidget(self.identity_panel, 0)

        self.poll_timer = self.startTimer(250)

    @staticmethod
    def _status_style(state: str) -> str:
        state = str(state or "").lower()
        if state == "live":
            color = C["known"]
            bg = "#0b1c19"
            border = "#174238"
        elif state in {"error", "offline"}:
            color = C["offline"]
            bg = "#211215"
            border = "#4a2529"
        elif state == "warning":
            color = C["unknown"]
            bg = "#211a0e"
            border = "#58461f"
        else:
            color = C["unknown"]
            bg = "#201a0e"
            border = "#4b3b1d"
        return (
            f"color:{color};background:{bg};border:1px solid {border};"
            "border-radius:5px;padding:4px 7px;font:700 9px 'DejaVu Sans Mono';"
        )

    @staticmethod
    def _summary_metric(heading: str, color: str):
        box = QFrame()
        box.setStyleSheet(
            "QFrame{background:#0a1118;border:1px solid #1b2935;border-radius:6px;}"
        )
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)
        title = QLabel(heading)
        title.setStyleSheet(
            f"color:{C['muted']};font:700 9px 'DejaVu Sans Mono';letter-spacing:1px;"
        )
        metric = QLabel("0")
        metric.setStyleSheet(f"color:{color};font-size:22px;font-weight:800;")
        lay.addWidget(title)
        lay.addWidget(metric)
        return metric, box

    def _start_or_bind(self, xid: int) -> None:
        self.controller.start_or_bind(int(xid))

    def timerEvent(self, event):
        if event.timerId() == self.poll_timer:
            status, metrics = self.controller.poll()
            self.wall.update_metrics(metrics)
            self.wall.set_pipeline_status(status)

            state = str(getattr(status, "state", "STARTING") or "STARTING").upper()
            if state in {"LIVE", "VIDEO_BOUND", "FOCUS", "HEATMAP"}:
                badge_state = "live"
                badge_text = "● LIVE"
            elif state == "PIPELINE_WARNING":
                badge_state = "warning"
                badge_text = "WARN"
            elif state in {"ERROR", "STOPPED"}:
                badge_state = "error"
                badge_text = state
            else:
                badge_state = "starting"
                badge_text = "STARTING"
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

    def exit_fullscreen(self) -> None:
        if not self._fullscreen_active:
            return
        self.controller.focus(None)
        self._fullscreen_active = False
        self._focused_source = None
        self.wall.set_fullscreen_mode(False, None)
        self.identity_panel.show()
        self.layout.setContentsMargins(12, 10, 12, 12)
        self.layout.setSpacing(12)
        self._set_app_fullscreen_shell(False)

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
