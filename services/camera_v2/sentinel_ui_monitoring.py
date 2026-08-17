from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QVBoxLayout, QWidget

from .data import CAMERAS, PEOPLE
from .sentinel_ui_base import C, FaceAvatar, Panel, StatCard, label, make_button, panel_layout
from .sentinel_video import LiveVideoWall, PipelineController


class LiveCameraDialog(QDialog):
    def __init__(self, source_id: int, controller: PipelineController, return_xid: int, parent=None):
        super().__init__(parent)
        self.source_id = int(source_id)
        self.controller = controller
        self.return_xid = int(return_xid)
        self.setWindowTitle(f"Sentinel VMS · CAM-{self.source_id + 1:02d}")
        self.resize(1100, 720)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        controls = QHBoxLayout()
        controls.addWidget(label(f"CAM-{self.source_id + 1:02d}", "sectionTitle"))
        controls.addStretch()
        close_button = make_button("✕  Yopish")
        close_button.clicked.connect(self.close)
        controls.addWidget(close_button)
        layout.addLayout(controls)
        self.wall = LiveVideoWall(CAMERAS, PEOPLE, self)
        for widget in (*self.wall.camera_labels, *self.wall.status_labels, *self.wall.occupancy_labels):
            widget.hide()
        self.wall.nativeReady.connect(self._bind)
        layout.addWidget(self.wall, 1)

    def _bind(self, xid: int) -> None:
        self.controller.start_or_bind(int(xid))
        self.controller.focus(self.source_id)

    def closeEvent(self, event):
        self.controller.focus(None)
        if self.return_xid > 0:
            self.controller.bind(self.return_xid)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_F11):
            self.close()
            return
        super().keyPressEvent(event)


class FullscreenCameraGrid(QDialog):
    def __init__(self, controller: PipelineController, return_xid: int, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.return_xid = int(return_xid)
        self.setWindowTitle("Sentinel VMS · Cameras")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        controls = QHBoxLayout()
        controls.addStretch()
        close_button = make_button("✕  Fullscreen'dan chiqish")
        close_button.clicked.connect(self.close)
        controls.addWidget(close_button)
        layout.addLayout(controls)
        self.wall = LiveVideoWall(CAMERAS, PEOPLE, self)
        self.wall.nativeReady.connect(self._bind)
        self.wall.cameraDoubleClicked.connect(self._focus_camera)
        layout.addWidget(self.wall, 1)

    def _bind(self, xid: int) -> None:
        self.controller.focus(None)
        self.controller.start_or_bind(int(xid))

    def _focus_camera(self, source_id: int) -> None:
        self.controller.focus(int(source_id))

    def closeEvent(self, event):
        self.controller.focus(None)
        if self.return_xid > 0:
            self.controller.bind(self.return_xid)
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
        self.controller = PipelineController()
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(22, 10, 22, 12)
        self.layout.setSpacing(16)
        inside = [p for p in PEOPLE if p.in_building]
        known = len([p for p in inside if p.known])
        unknown = len(inside) - known

        camera_column = QVBoxLayout()
        camera_column.setSpacing(8)
        self.wall = LiveVideoWall(CAMERAS, PEOPLE, self)
        self.wall.nativeReady.connect(self._start_or_bind)
        self.wall.cameraDoubleClicked.connect(self.expand)
        camera_column.addWidget(self.wall, 1)
        self.layout.addLayout(camera_column, 3)

        identity_rail = QVBoxLayout()
        identity_rail.setSpacing(12)
        metrics = QHBoxLayout()
        metrics.setSpacing(10)
        known_card = StatCard("Known", str(known), "known", "Hozir binoda")
        unknown_card = StatCard("Unknown", str(unknown), "unknown", "Hozir binoda")
        known_card.setMinimumWidth(125)
        unknown_card.setMinimumWidth(125)
        metrics.addWidget(known_card)
        metrics.addWidget(unknown_card)
        identity_rail.addLayout(metrics, 1)

        recent_panel = Panel()
        recent_panel.setMinimumWidth(285)
        recent_layout = panel_layout(recent_panel, (14, 14, 14, 14), 0)
        recent_head = QHBoxLayout()
        recent_head.addWidget(label("Recent Views", "sectionTitle"))
        recent_head.addStretch()
        recent_head.addWidget(label(f"{len(inside)} active", "mono"))
        recent_layout.addLayout(recent_head)
        recent_layout.addSpacing(8)
        for person in sorted(PEOPLE, key=lambda p: p.last_seen, reverse=True)[:6]:
            recent_layout.addWidget(self.recent_view(person))
        recent_layout.addStretch()
        identity_rail.addWidget(recent_panel, 3)
        self.layout.addLayout(identity_rail, 1)

        self.poll_timer = self.startTimer(250)

    def _start_or_bind(self, xid: int) -> None:
        self.controller.start_or_bind(int(xid))

    def timerEvent(self, event):
        if event.timerId() == self.poll_timer:
            _status, metrics = self.controller.poll()
            self.wall.update_metrics(metrics)
            return
        super().timerEvent(event)

    def open_fullscreen_grid(self):
        dialog = FullscreenCameraGrid(self.controller, int(self.wall.winId()), self)
        dialog.showFullScreen()
        dialog.exec()

    def recent_view(self, person):
        item = QFrame()
        item.setStyleSheet(f"QFrame{{border-bottom:1px solid {C['border']};background:transparent;}}")
        item.setMinimumHeight(58)
        row = QHBoxLayout(item)
        row.setContentsMargins(0, 5, 0, 5)
        info = QVBoxLayout()
        info.setSpacing(3)
        info.addWidget(label(person.label, "sectionTitle"))
        info.addWidget(label(person.last_seen.astimezone().strftime("%H:%M:%S"), "mono"))
        row.addLayout(info, 1)
        row.addWidget(FaceAvatar(person, 42))
        return item

    def expand(self, source_id: int):
        dialog = LiveCameraDialog(int(source_id), self.controller, int(self.wall.winId()), self)
        dialog.exec()

    def shutdown(self) -> None:
        try:
            self.killTimer(self.poll_timer)
        except Exception:
            pass
        self.controller.stop()
