from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QStackedWidget,
    QVBoxLayout, QWidget, QComboBox,
)

from .qt_runtime import CameraQtController

C = {
    "bg": "#090d12", "sidebar": "#0b1118", "panel": "#0e151d", "panel2": "#111b25",
    "border": "#22303e", "text": "#e7edf3", "muted": "#7e8c99", "primary": "#39d9c5",
    "known": "#3ddc97", "unknown": "#f6b94b", "offline": "#f06464", "blue": "#65a8ff",
}

APP_QSS = f"""
* {{ color:{C['text']}; font-family:'DejaVu Sans'; font-size:12px; }}
QMainWindow,QWidget#root,QWidget#pageRoot,QScrollArea,QScrollArea>QWidget>QWidget {{ background:{C['bg']}; }}
QFrame#sidebar {{ background:{C['sidebar']}; border-right:1px solid {C['border']}; }}
QFrame#header {{ background:{C['bg']}; border-bottom:1px solid {C['border']}; }}
QFrame#panel {{ background:{C['panel']}; border:1px solid {C['border']}; border-radius:7px; }}
QLabel#title {{ font-size:18px; font-weight:700; }}
QLabel#section {{ font-size:14px; font-weight:700; }}
QLabel#muted,QLabel#mono {{ color:{C['muted']}; }}
QLabel#mono {{ font-family:'DejaVu Sans Mono'; font-size:10px; }}
QPushButton {{ background:transparent; border:1px solid {C['border']}; border-radius:6px; padding:7px 11px; }}
QPushButton:hover {{ background:{C['panel2']}; }}
QPushButton:checked {{ background:#163932; color:{C['primary']}; border-color:{C['primary']}; }}
QPushButton#nav {{ text-align:left; border:none; padding:10px 12px; color:{C['muted']}; }}
QPushButton#nav:checked {{ background:{C['panel2']}; color:{C['text']}; border-left:2px solid {C['primary']}; }}
QPushButton#primary {{ background:{C['primary']}; color:#07110f; border-color:{C['primary']}; font-weight:700; }}
QLineEdit,QComboBox {{ background:{C['panel']}; border:1px solid {C['border']}; border-radius:6px; padding:8px; }}
QScrollBar:vertical {{ background:transparent; width:8px; }} QScrollBar::handle:vertical {{ background:{C['border']}; border-radius:4px; }}
"""


def lab(text: str, role: str = "") -> QLabel:
    item = QLabel(text)
    if role:
        item.setObjectName(role)
    return item


class Panel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")


class ScrollPage(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.body = QWidget()
        self.body.setObjectName("pageRoot")
        self.layout = QVBoxLayout(self.body)
        self.layout.setContentsMargins(22, 18, 22, 22)
        self.layout.setSpacing(12)
        self.setWidget(self.body)


class CameraTileOverlay(QWidget):
    fullscreenRequested = Signal(int)
    heatmapToggled = Signal(int, bool)

    def __init__(self, source_id: int, parent=None):
        super().__init__(parent)
        self.source_id = source_id
        self.fps = 0.0
        self.online = False
        self.count = 0
        self.points: list[tuple[float, float, float]] = []
        self.focused = False
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NativeWindow, True)

        self.controls = QWidget(self)
        row = QHBoxLayout(self.controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.heat_btn = QPushButton("Heatmap")
        self.heat_btn.setCheckable(True)
        self.full_btn = QPushButton("⛶")
        self.heat_btn.setCursor(Qt.PointingHandCursor)
        self.full_btn.setCursor(Qt.PointingHandCursor)
        self.heat_btn.toggled.connect(lambda state: self.heatmapToggled.emit(self.source_id, state))
        self.full_btn.clicked.connect(lambda: self.fullscreenRequested.emit(self.source_id))
        row.addWidget(self.heat_btn)
        row.addWidget(self.full_btn)
        self.controls.adjustSize()
        self.controls.hide()

    def enterEvent(self, event):
        self.controls.show()
        self.controls.raise_()
        super().enterEvent(event)

    def leaveEvent(self, event):
        QTimer.singleShot(90, self._hide_if_outside)
        super().leaveEvent(event)

    def _hide_if_outside(self):
        if not self.underMouse() and not self.controls.underMouse():
            self.controls.hide()

    def resizeEvent(self, event):
        self.controls.adjustSize()
        self.controls.move(max(8, self.width() - self.controls.width() - 10), 10)
        self.controls.raise_()
        super().resizeEvent(event)

    def set_live(self, camera: dict, points: list[tuple[float, float, float]]):
        self.fps = float(camera.get("fps", 0.0))
        self.online = bool(camera.get("online", False))
        self.count = int(camera.get("count", 0))
        self.points = points
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(1, 1, -2, -2)
        painter.setPen(QPen(QColor(C["primary"] if self.focused else C["border"]), 2 if self.focused else 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 7, 7)

        if self.heat_btn.isChecked():
            for nx, ny, value in self.points:
                x = nx * self.width()
                y = ny * self.height()
                radius = max(8.0, min(self.width(), self.height()) * (0.026 + 0.018 * value))
                if value < 0.35:
                    color = QColor(25, 155, 255, 45 + int(70 * value))
                elif value < 0.7:
                    color = QColor(255, 214, 30, 55 + int(70 * value))
                else:
                    color = QColor(255, 55, 35, 65 + int(80 * value))
                painter.setPen(Qt.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(QRectF(x - radius, y - radius, radius * 2, radius * 2))

        painter.setPen(QColor(C["text"]))
        painter.setFont(QFont("DejaVu Sans", 9, QFont.Bold))
        painter.drawText(12, 22, f"CAM-{self.source_id + 1:02d}")
        dot = QColor(C["known"] if self.online else C["offline"])
        painter.setPen(Qt.NoPen)
        painter.setBrush(dot)
        painter.drawEllipse(self.width() - 82, self.height() - 25, 7, 7)
        painter.setPen(QColor(C["muted"]))
        painter.setFont(QFont("DejaVu Sans Mono", 8))
        painter.drawText(self.width() - 68, self.height() - 17, f"{self.fps:.1f} fps" if self.online else "OFFLINE")

        badge = QRectF(10, self.height() - 32, 52, 22)
        painter.setPen(QPen(QColor(C["border"]), 1))
        painter.setBrush(QColor(8, 14, 20, 210))
        painter.drawRoundedRect(badge, 5, 5)
        painter.setPen(QColor(C["text"]))
        painter.setFont(QFont("DejaVu Sans Mono", 8, QFont.Bold))
        painter.drawText(badge, Qt.AlignCenter, f"● {self.count}")


class LiveWall(QWidget):
    fullscreenRequested = Signal(int)

    def __init__(self, controller: CameraQtController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.focus_source: int | None = None
        self.setMinimumSize(760, 560)
        self.surface = QWidget(self)
        self.surface.setStyleSheet("background:#000;")
        self.surface.setAttribute(Qt.WA_NativeWindow, True)
        self.surface.setAttribute(Qt.WA_DontCreateNativeAncestors, True)
        _ = self.surface.winId()
        self.overlays = []
        for source_id in range(controller.camera_count):
            overlay = CameraTileOverlay(source_id, self)
            overlay.fullscreenRequested.connect(self.fullscreenRequested)
            overlay.heatmapToggled.connect(controller.set_heatmap_enabled)
            self.overlays.append(overlay)
        QTimer.singleShot(0, self._bind_video)

    def _bind_video(self):
        self.controller.bind_window(int(self.surface.winId()))
        self.controller.start()
        for overlay in self.overlays:
            overlay.raise_()

    def set_focus(self, source_id: int | None):
        self.focus_source = source_id
        self.controller.set_focus_source(source_id)
        for overlay in self.overlays:
            overlay.focused = source_id == overlay.source_id
        self._relayout()

    def resizeEvent(self, event):
        self.surface.setGeometry(self.rect())
        self._relayout()
        super().resizeEvent(event)

    def _relayout(self):
        if not self.overlays:
            return
        if self.focus_source is not None:
            for overlay in self.overlays:
                if overlay.source_id == self.focus_source:
                    overlay.setGeometry(self.rect())
                    overlay.show()
                    overlay.raise_()
                else:
                    overlay.hide()
            return
        tile_w = self.width() / 2.0
        tile_h = self.height() / 3.0
        for source_id, overlay in enumerate(self.overlays):
            row, col = divmod(source_id, 2)
            overlay.setGeometry(int(col * tile_w), int(row * tile_h), int(tile_w), int(tile_h))
            overlay.show()
            overlay.raise_()

    def refresh(self, snapshot: dict):
        cameras = {int(item["source_id"]): item for item in snapshot.get("cameras", [])}
        for overlay in self.overlays:
            overlay.set_live(cameras.get(overlay.source_id, {}), self.controller.heat_points(overlay.source_id))


class MonitoringPage(QWidget):
    fullscreenRequested = Signal(int)

    def __init__(self, controller: CameraQtController):
        super().__init__()
        self.controller = controller
        self.setObjectName("pageRoot")
        root = QHBoxLayout(self)
        root.setContentsMargins(20, 10, 20, 12)
        root.setSpacing(14)
        self.wall = LiveWall(controller)
        self.wall.fullscreenRequested.connect(self.fullscreenRequested)
        root.addWidget(self.wall, 4)

        self.rail = QFrame()
        self.rail.setObjectName("panel")
        self.rail.setMinimumWidth(270)
        rail_layout = QVBoxLayout(self.rail)
        rail_layout.setContentsMargins(14, 14, 14, 14)
        rail_layout.setSpacing(10)
        metrics = QHBoxLayout()
        self.known_value = lab("0", "title")
        self.unknown_value = lab("0", "title")
        for title, value, color in (("Known", self.known_value, C["known"]), ("Unknown", self.unknown_value, C["unknown"])):
            box = Panel()
            layout = QVBoxLayout(box)
            layout.setContentsMargins(10, 9, 10, 9)
            title_label = QLabel(title)
            title_label.setStyleSheet(f"color:{color};font-weight:700;")
            layout.addWidget(title_label)
            layout.addWidget(value)
            metrics.addWidget(box)
        rail_layout.addLayout(metrics)
        rail_layout.addWidget(lab("Recent Views", "section"))
        self.recent = QVBoxLayout()
        self.recent.setSpacing(4)
        rail_layout.addLayout(self.recent)
        rail_layout.addStretch()
        root.addWidget(self.rail, 1)

    def refresh(self, snapshot: dict):
        self.wall.refresh(snapshot)
        tracks = sorted(snapshot.get("tracks", []), key=lambda item: item.get("last_seen", 0), reverse=True)
        self.known_value.setText("0")
        self.unknown_value.setText(str(len(tracks)))
        while self.recent.count():
            item = self.recent.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for track in tracks[:7]:
            row = QFrame()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 6, 0, 6)
            info = QVBoxLayout()
            info.addWidget(lab(track.get("label", "Unknown"), "section"))
            info.addWidget(lab(track.get("camera_id", ""), "mono"))
            row_layout.addLayout(info, 1)
            self.recent.addWidget(row)

    def set_camera_fullscreen(self, source_id: int | None):
        self.rail.setVisible(source_id is None)
        self.wall.set_focus(source_id)


class PeoplePage(ScrollPage):
    def __init__(self):
        super().__init__()
        self.layout.addWidget(lab("People", "title"))
        self.list = QVBoxLayout()
        self.layout.addLayout(self.list)
        self.layout.addStretch()

    def refresh(self, snapshot: dict):
        while self.list.count():
            item = self.list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        tracks = sorted(snapshot.get("tracks", []), key=lambda item: (item.get("source_id", 0), item.get("object_id", 0)))
        if not tracks:
            self.list.addWidget(lab("Hozir aktiv odam yo'q", "muted"))
            return
        for track in tracks:
            panel = Panel()
            row = QHBoxLayout(panel)
            row.addWidget(lab(track["label"], "section"))
            row.addStretch()
            row.addWidget(lab(f"{track['camera_id']}  ·  ID {track['object_id']}", "mono"))
            self.list.addWidget(panel)


class EventsPage(ScrollPage):
    def __init__(self):
        super().__init__()
        self.layout.addWidget(lab("Events", "title"))
        self.list = QVBoxLayout()
        self.layout.addLayout(self.list)
        self.layout.addStretch()

    def refresh(self, snapshot: dict):
        while self.list.count():
            item = self.list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for event in snapshot.get("events", [])[:80]:
            panel = Panel()
            row = QHBoxLayout(panel)
            row.addWidget(lab(time.strftime("%H:%M:%S", time.localtime(event.get("time", 0))), "mono"))
            row.addWidget(lab(event.get("message", "")), 1)
            row.addStretch()
            row.addWidget(lab(event.get("camera_id", ""), "mono"))
            self.list.addWidget(panel)


class RoomsPage(ScrollPage):
    def __init__(self):
        super().__init__()
        self.layout.addWidget(lab("Rooms", "title"))
        self.grid = QGridLayout()
        self.layout.addLayout(self.grid)
        self.cards = []
        for index in range(3):
            panel = Panel()
            layout = QVBoxLayout(panel)
            title = lab(f"Room {index + 1}", "section")
            count = lab("0", "title")
            detail = lab("CAM -- / CAM --", "mono")
            layout.addWidget(title)
            layout.addWidget(count)
            layout.addWidget(detail)
            self.grid.addWidget(panel, 0, index)
            self.cards.append((count, detail))
        self.layout.addStretch()

    def refresh(self, snapshot: dict):
        rooms = snapshot.get("rooms", [])
        for index, (count, detail) in enumerate(self.cards):
            room = rooms[index] if index < len(rooms) else {"count": 0, "camera_counts": [0, 0]}
            camera_counts = room.get("camera_counts", [0, 0])
            count.setText(str(room.get("count", 0)))
            detail.setText(f"CAM-{index * 2 + 1:02d}: {camera_counts[0]}   CAM-{index * 2 + 2:02d}: {camera_counts[1]}")


class EnrollmentPage(ScrollPage):
    def __init__(self):
        super().__init__()
        self.files: list[str] = []
        self.layout.addWidget(lab("Enrollment", "title"))
        self.layout.addWidget(lab("Exactly 10 face images tanlang. Dataset project ichiga saqlanadi.", "muted"))
        panel = Panel()
        form = QVBoxLayout(panel)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Person name")
        self.pick = QPushButton("10 ta rasm tanlash")
        self.pick.clicked.connect(self.pick_files)
        self.file_status = lab("0 / 10", "mono")
        self.profile = QComboBox()
        self.save_btn = QPushButton("Enroll")
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(self.save)
        for widget in (self.name, self.pick, self.file_status, self.profile, self.save_btn):
            form.addWidget(widget)
        self.layout.addWidget(panel)
        self.layout.addStretch()

    def pick_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "10 face images", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if not paths:
            return
        if len(paths) != 10:
            QMessageBox.warning(self, "Enrollment", "Aynan 10 ta rasm tanlang.")
            return
        self.files = paths
        self.file_status.setText("10 / 10 ready")
        self.profile.clear()
        for index, path in enumerate(paths, 1):
            self.profile.addItem(f"Profile {index}: {Path(path).name}", path)

    def save(self):
        name = self.name.text().strip()
        if not name or len(self.files) != 10:
            QMessageBox.warning(self, "Enrollment", "Name va 10 ta rasm kerak.")
            return
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "person"
        root = Path.cwd() / "data" / "enrollments" / safe
        root.mkdir(parents=True, exist_ok=True)
        copied = []
        for index, source in enumerate(self.files, 1):
            destination = root / f"face_{index:02d}{Path(source).suffix.lower()}"
            shutil.copy2(source, destination)
            copied.append(str(destination))
        manifest = {
            "name": name,
            "created_at": time.time(),
            "profile_index": self.profile.currentIndex(),
            "images": copied,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Enrollment", f"Saved: {root}")


class ReportsPage(ScrollPage):
    def __init__(self, controller: CameraQtController):
        super().__init__()
        self.controller = controller
        self.layout.addWidget(lab("Reports", "title"))
        self.layout.addWidget(lab("Realtime event log'ni CSV qilib eksport qiladi.", "muted"))
        button = QPushButton("Export CSV")
        button.setObjectName("primary")
        button.clicked.connect(self.export)
        self.layout.addWidget(button)
        self.layout.addStretch()

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export events", "sentinel_events.csv", "CSV (*.csv)")
        if path:
            self.controller.export_events_csv(path)
            QMessageBox.information(self, "Reports", f"Saved: {path}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = CameraQtController()
        self._camera_fullscreen: int | None = None
        self.setWindowTitle("Sentinel VMS")
        self.resize(1500, 930)
        self.setMinimumSize(1100, 720)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(190)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(12, 18, 12, 14)
        side.addWidget(lab("SENTINEL VMS", "title"))
        side.addSpacing(16)
        shell.addWidget(self.sidebar)

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        shell.addLayout(body, 1)
        self.header = QFrame()
        self.header.setObjectName("header")
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(20, 10, 20, 10)
        self.page_title = lab("Monitoring", "title")
        header_layout.addWidget(self.page_title)
        header_layout.addStretch()
        self.grid_full = QPushButton("⛶ Camera grid")
        self.grid_full.clicked.connect(self.toggle_grid_fullscreen)
        header_layout.addWidget(self.grid_full)
        body.addWidget(self.header)
        self.stack = QStackedWidget()
        body.addWidget(self.stack, 1)

        self.monitoring = MonitoringPage(self.controller)
        self.monitoring.fullscreenRequested.connect(self.enter_camera_fullscreen)
        self.people = PeoplePage()
        self.events = EventsPage()
        self.rooms = RoomsPage()
        self.enrollment = EnrollmentPage()
        self.reports = ReportsPage(self.controller)
        pages = [
            ("Monitoring", self.monitoring), ("People", self.people), ("Events", self.events),
            ("Rooms", self.rooms), ("Enrollment", self.enrollment), ("Reports", self.reports),
        ]
        self.nav = []
        for index, (name, page) in enumerate(pages):
            self.stack.addWidget(page)
            button = QPushButton(name)
            button.setObjectName("nav")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _=False, i=index, n=name: self.switch_page(i, n))
            side.addWidget(button)
            self.nav.append(button)
        side.addStretch()
        self.nav[0].setChecked(True)

        self._snapshot = {"cameras": [], "tracks": [], "events": [], "rooms": []}
        self.timer = QTimer(self)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.slow_timer = QTimer(self)
        self.slow_timer.setInterval(1000)
        self.slow_timer.timeout.connect(self.refresh_lists)
        self.slow_timer.start()

    def switch_page(self, index: int, name: str):
        if self._camera_fullscreen is not None:
            self.exit_camera_fullscreen()
        self.stack.setCurrentIndex(index)
        self.page_title.setText(name)
        for nav_index, button in enumerate(self.nav):
            button.setChecked(nav_index == index)

    def refresh(self):
        try:
            self._snapshot = self.controller.snapshot()
            self.monitoring.refresh(self._snapshot)
        except Exception as exc:
            print(f"CAMERA_QT refresh error: {exc}", file=sys.stderr, flush=True)

    def refresh_lists(self):
        self.people.refresh(self._snapshot)
        self.events.refresh(self._snapshot)
        self.rooms.refresh(self._snapshot)

    def enter_camera_fullscreen(self, source_id: int):
        self._camera_fullscreen = source_id
        self.sidebar.hide()
        self.header.hide()
        self.monitoring.set_camera_fullscreen(source_id)
        self.showFullScreen()

    def exit_camera_fullscreen(self):
        self._camera_fullscreen = None
        self.monitoring.set_camera_fullscreen(None)
        self.sidebar.show()
        self.header.show()
        self.showNormal()
        self.showMaximized()

    def toggle_grid_fullscreen(self):
        if self._camera_fullscreen is not None:
            self.exit_camera_fullscreen()
            return
        if self.isFullScreen():
            self.sidebar.show()
            self.header.show()
            self.showNormal()
            self.showMaximized()
        else:
            self.sidebar.hide()
            self.header.hide()
            self.showFullScreen()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_F11):
            if self._camera_fullscreen is not None:
                self.exit_camera_fullscreen()
                return
            if self.isFullScreen():
                self.sidebar.show()
                self.header.show()
                self.showNormal()
                self.showMaximized()
                return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        self.slow_timer.stop()
        self.controller.stop()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Sentinel VMS")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
    window = MainWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
