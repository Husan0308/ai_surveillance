from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .data import CAMERAS, PEOPLE
from .sentinel_ui_base import (
    C,
    FaceAvatar,
    Panel,
    label,
    make_button,
    panel_layout,
)
from .sentinel_video import LiveVideoWall
from .sentinel_video_pro import ProLiveVideoWall, ProPipelineController


class LiveCameraDialog(QDialog):
    """True single-camera fullscreen surface using DeepStream tiler show-source."""

    def __init__(
        self,
        source_id: int,
        controller: ProPipelineController,
        return_xid: int,
        parent=None,
    ):
        super().__init__(parent)
        self.source_id = int(source_id)
        self.controller = controller
        self.return_xid = int(return_xid)

        self.setWindowTitle(f"Sentinel VMS · CAM-{self.source_id + 1:02d}")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Small VMS-style control strip. No desktop title bar and no six-camera
        # shell around the selected source.
        bar = QFrame(self)
        bar.setFixedHeight(46)
        bar.setStyleSheet(
            "QFrame{background:#070c12;border-bottom:1px solid #22303e;}"
        )
        controls = QHBoxLayout(bar)
        controls.setContentsMargins(16, 0, 12, 0)
        controls.setSpacing(8)

        cam = QLabel(f"CAM-{self.source_id + 1:02d}")
        cam.setStyleSheet(
            "color:#f0f4f7;font-weight:700;font-size:13px;letter-spacing:.5px;"
        )
        controls.addWidget(cam)
        live = QLabel("● LIVE")
        live.setStyleSheet(
            f"color:{C['known']};font:700 10px 'DejaVu Sans Mono';"
        )
        controls.addWidget(live)
        controls.addStretch()

        close_button = QToolButton(bar)
        close_button.setText("⛶  Exit Fullscreen")
        close_button.setCursor(Qt.PointingHandCursor)
        close_button.setStyleSheet(
            "QToolButton{background:#101922;border:1px solid #2a3a49;"
            "border-radius:5px;padding:7px 11px;color:#e7edf3;}"
            "QToolButton:hover{background:#182632;border-color:#3d566a;}"
        )
        close_button.clicked.connect(self.close)
        controls.addWidget(close_button)
        layout.addWidget(bar)

        self.wall = LiveVideoWall(CAMERAS, PEOPLE, self)
        for widget in (
            *self.wall.camera_labels,
            *self.wall.status_labels,
            *self.wall.occupancy_labels,
        ):
            widget.hide()
        self.wall.nativeReady.connect(self._bind)
        layout.addWidget(self.wall, 1)

    def _bind(self, xid: int) -> None:
        # One queue command performs EGL rebind + nvmultistreamtiler show-source.
        self.controller.bind_focus(int(xid), self.source_id)

    def closeEvent(self, event):
        if self.return_xid > 0:
            self.controller.bind_focus(self.return_xid, None)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_F11):
            self.close()
            return
        super().keyPressEvent(event)


class FullscreenCameraGrid(QDialog):
    def __init__(
        self,
        controller: ProPipelineController,
        return_xid: int,
        parent=None,
    ):
        super().__init__(parent)
        self.controller = controller
        self.return_xid = int(return_xid)
        self.setWindowTitle("Sentinel VMS · Cameras")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        controls = QHBoxLayout()
        controls.addStretch()
        close_button = make_button("✕  Fullscreen'dan chiqish")
        close_button.clicked.connect(self.close)
        controls.addWidget(close_button)
        layout.addLayout(controls)

        self.wall = ProLiveVideoWall(CAMERAS, PEOPLE, self)
        self.wall.nativeReady.connect(self._bind)
        self.wall.cameraDoubleClicked.connect(self._focus_camera)
        self.wall.heatmapToggled.connect(self.controller.set_heatmap)
        layout.addWidget(self.wall, 1)

    def _bind(self, xid: int) -> None:
        self.controller.bind_focus(int(xid), None)

    def _focus_camera(self, source_id: int) -> None:
        dialog = LiveCameraDialog(
            int(source_id),
            self.controller,
            int(self.wall.winId()),
            self,
        )
        dialog.showFullScreen()
        dialog.exec()

    def closeEvent(self, event):
        if self.return_xid > 0:
            self.controller.bind_focus(self.return_xid, None)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_F11):
            self.close()
            return
        super().keyPressEvent(event)


class MonitoringPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("pageRoot")
        self.controller = ProPipelineController()
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(18, 10, 18, 12)
        self.layout.setSpacing(14)

        inside = [p for p in PEOPLE if p.in_building]
        self.known_count = len([p for p in inside if p.known])
        self.unknown_count = len(inside) - self.known_count

        camera_column = QVBoxLayout()
        camera_column.setSpacing(8)
        self.wall = ProLiveVideoWall(CAMERAS, PEOPLE, self)
        self.wall.nativeReady.connect(self._start_or_bind)
        self.wall.cameraDoubleClicked.connect(self.expand)
        self.wall.heatmapToggled.connect(self.controller.set_heatmap)
        camera_column.addWidget(self.wall, 1)
        self.layout.addLayout(camera_column, 3)

        identity_rail = QVBoxLayout()
        identity_rail.setSpacing(10)

        # One balanced summary card works better in the narrow right rail than
        # three tiny equal-width statistic cards.
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
            "KNOWN",
            self.known_count,
            C["known"],
        )
        self.unknown_value, unknown_box = self._summary_metric(
            "UNKNOWN",
            self.unknown_count,
            C["unknown"],
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
        self.layout.addLayout(identity_rail, 1)

        self.poll_timer = self.startTimer(250)

    @staticmethod
    def _summary_metric(heading: str, value: int, color: str):
        box = QFrame()
        box.setStyleSheet(
            "QFrame{background:#0b1219;border:1px solid #1d2a36;"
            "border-radius:6px;}"
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

            # Known/Unknown are still supplied by the current identity/UI model.
            # If live occupancy contains additional unclassified tracks, display
            # that explicitly instead of showing an apparently broken sum.
            pending = max(0, total - self.known_count - self.unknown_count)
            self.pending_value.setText(
                f"{pending} pending" if pending else "classified"
            )
            return
        super().timerEvent(event)

    def open_fullscreen_grid(self):
        dialog = FullscreenCameraGrid(
            self.controller,
            int(self.wall.winId()),
            self,
        )
        dialog.showFullScreen()
        dialog.exec()

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

    def expand(self, source_id: int):
        dialog = LiveCameraDialog(
            int(source_id),
            self.controller,
            int(self.wall.winId()),
            self,
        )
        dialog.showFullScreen()
        dialog.exec()

    def shutdown(self) -> None:
        try:
            self.killTimer(self.poll_timer)
        except Exception:
            pass
        self.controller.stop()
