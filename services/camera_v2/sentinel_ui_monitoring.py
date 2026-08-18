from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from services.ml_service.app.config import load_settings

from .data import PEOPLE
from .sentinel_ui_base import C, FaceAvatar, Panel, label, panel_layout
from .sentinel_video_pro import ProLiveVideoWall, ProPipelineController


class MonitoringPage(QWidget):
    """Monitoring page that never reparents/rebinds the native DeepStream surface."""

    def __init__(self):
        super().__init__()
        self.setObjectName("pageRoot")
        self.controller = ProPipelineController()
        self._fullscreen_active = False
        self._focused_source: int | None = None
        self.camera_configs = list(load_settings().cameras)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(18, 10, 18, 12)
        self.layout.setSpacing(14)

        inside = [p for p in PEOPLE if p.in_building]
        self.known_count = len([p for p in inside if p.known])
        self.unknown_count = len(inside) - self.known_count

        self.camera_panel = QWidget(self)
        camera_column = QVBoxLayout(self.camera_panel)
        camera_column.setContentsMargins(0, 0, 0, 0)
        camera_column.setSpacing(0)

        self.wall = ProLiveVideoWall(self.camera_configs, PEOPLE, self.camera_panel)
        self.wall.nativeReady.connect(self._start_or_bind)
        self.wall.cameraDoubleClicked.connect(self.expand)
        self.wall.heatmapToggled.connect(self.controller.set_heatmap)
        self.wall.exitFullscreenRequested.connect(self.exit_fullscreen)
        camera_column.addWidget(self.wall, 1)
        self.layout.addWidget(self.camera_panel, 3)

        self.identity_panel = QWidget(self)
        identity_rail = QVBoxLayout(self.identity_panel)
        identity_rail.setContentsMargins(0, 0, 0, 0)
        identity_rail.setSpacing(10)

        summary = Panel()
        summary.setMinimumWidth(315)
        summary.setMinimumHeight(154)
        summary_layout = QVBoxLayout(summary)
        summary_layout.setContentsMargins(16, 14, 16, 14)
        summary_layout.setSpacing(8)

        summary_top = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title_col.addWidget(label("LIVE OCCUPANCY", "eyebrow"))
        title_col.addWidget(label("Current people in building", "muted"))
        summary_top.addLayout(title_col)
        summary_top.addStretch()
        live_badge = QLabel("● LIVE")
        live_badge.setStyleSheet(
            f"color:{C['known']};background:#0b1c19;border:1px solid #174238;"
            "border-radius:5px;padding:4px 7px;font:700 9px 'DejaVu Sans Mono';"
        )
        summary_top.addWidget(live_badge)
        summary_layout.addLayout(summary_top)

        total_row = QHBoxLayout()
        total_row.setSpacing(8)
        self.total_value = QLabel(str(len(inside)))
        self.total_value.setStyleSheet(
            f"color:{C['primary']};font-size:34px;font-weight:800;"
        )
        total_row.addWidget(self.total_value)
        people_text = QLabel("people")
        people_text.setStyleSheet(
            f"color:{C['muted']};font-size:12px;padding-top:12px;"
        )
        total_row.addWidget(people_text)
        total_row.addStretch()
        self.pending_value = QLabel("")
        self.pending_value.setStyleSheet(
            f"color:{C['muted']};font:10px 'DejaVu Sans Mono';"
        )
        total_row.addWidget(self.pending_value)
        summary_layout.addLayout(total_row)

        split = QHBoxLayout()
        split.setSpacing(8)
        self.known_value, known_box = self._summary_metric(
            "KNOWN", self.known_count, C["known"]
        )
        self.unknown_value, unknown_box = self._summary_metric(
            "UNKNOWN", self.unknown_count, C["unknown"]
        )
        split.addWidget(known_box, 1)
        split.addWidget(unknown_box, 1)
        summary_layout.addLayout(split)
        identity_rail.addWidget(summary)

        recent_panel = Panel()
        recent_panel.setMinimumWidth(315)
        recent_layout = panel_layout(recent_panel, (14, 14, 14, 14), 0)
        recent_head = QHBoxLayout()
        recent_head.addWidget(label("Recent Views", "sectionTitle"))
        recent_head.addStretch()
        self.active_count = label(f"{len(inside)} active", "mono")
        recent_head.addWidget(self.active_count)
        recent_layout.addLayout(recent_head)
        recent_layout.addSpacing(8)

        for person in sorted(PEOPLE, key=lambda p: p.last_seen, reverse=True)[:6]:
            recent_layout.addWidget(self.recent_view(person))
        recent_layout.addStretch()
        identity_rail.addWidget(recent_panel, 1)
        self.layout.addWidget(self.identity_panel, 1)

        self.poll_timer = self.startTimer(250)

    @staticmethod
    def _summary_metric(heading: str, value: int, color: str):
        box = QFrame()
        box.setStyleSheet(
            "QFrame{background:#0b1219;border:1px solid #1d2a36;border-radius:6px;}"
        )
        row = QHBoxLayout(box)
        row.setContentsMargins(10, 7, 10, 7)
        row.setSpacing(8)
        heading_label = QLabel(heading)
        heading_label.setStyleSheet(
            f"color:{C['muted']};font:700 9px 'DejaVu Sans Mono';"
        )
        metric = QLabel(str(value))
        metric.setStyleSheet(f"color:{color};font-size:19px;font-weight:800;")
        row.addWidget(heading_label)
        row.addStretch()
        row.addWidget(metric)
        return metric, box

    def _start_or_bind(self, xid: int) -> None:
        self.controller.start_or_bind(int(xid))

    def timerEvent(self, event):
        if event.timerId() == self.poll_timer:
            _status, metrics = self.controller.poll()
            self.wall.update_metrics(metrics)
            total = max(0, int(metrics.get("total_people", 0) or 0))
            self.total_value.setText(str(total))
            self.active_count.setText(f"{total} active")
            pending = max(0, total - self.known_count - self.unknown_count)
            self.pending_value.setText(f"{pending} pending" if pending else "classified")
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

        # Critical: keep the exact same LiveVideoWall native XID. Only ask
        # nvmultistreamtiler to show one source. No EGL window rebind occurs.
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
        self.layout.setContentsMargins(18, 10, 18, 12)
        self.layout.setSpacing(14)
        self._set_app_fullscreen_shell(False)

    def open_fullscreen_grid(self) -> None:
        self.enter_fullscreen(None)

    def recent_view(self, person):
        item = QFrame()
        item.setStyleSheet(
            f"QFrame{{border-bottom:1px solid {C['border']};background:transparent;}}"
        )
        item.setMinimumHeight(58)
        row = QHBoxLayout(item)
        row.setContentsMargins(0, 5, 0, 5)
        info = QVBoxLayout()
        info.setSpacing(3)
        info.addWidget(label(person.label, "sectionTitle"))
        info.addWidget(
            label(person.last_seen.astimezone().strftime("%H:%M:%S"), "mono")
        )
        row.addLayout(info, 1)
        row.addWidget(FaceAvatar(person, 42))
        return item

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
