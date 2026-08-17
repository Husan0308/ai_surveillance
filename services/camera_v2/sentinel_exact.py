from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QSize, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap, QRadialGradient, QPdfWriter
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QStackedWidget, QTextEdit, QToolButton, QVBoxLayout, QWidget,
)

from .qt_runtime import CameraQtController


C = {
    "bg": "#090d12", "sidebar": "#0b1118", "panel": "#0e151d", "panel2": "#111b25",
    "border": "#22303e", "text": "#e7edf3", "muted": "#7e8c99", "primary": "#39d9c5",
    "known": "#3ddc97", "unknown": "#f6b94b", "offline": "#f06464", "blue": "#65a8ff",
    "violet": "#a78bfa", "field": "#0b1219",
}

APP_QSS = f"""
* {{ color: {C['text']}; font-family: 'DejaVu Sans'; font-size: 12px; }}
QMainWindow, QWidget#root, QWidget#pageRoot, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {C['bg']}; }}
QFrame#sidebar {{ background: {C['sidebar']}; border-right: 1px solid {C['border']}; }}
QFrame#header {{ background: {C['bg']}; border-bottom: 1px solid {C['border']}; }}
QFrame#panel {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 6px; }}
QLabel#title {{ font-size: 18px; font-weight: 700; }}
QLabel#subtitle, QLabel#muted {{ color: {C['muted']}; }}
QLabel#eyebrow {{ color: {C['muted']}; font-family: 'DejaVu Sans Mono'; font-size: 10px; letter-spacing: 1px; }}
QLabel#brand {{ font-size: 14px; font-weight: 700; letter-spacing: 1px; }}
QLabel#metric {{ font-size: 29px; font-weight: 700; }}
QLabel#sectionTitle {{ font-size: 14px; font-weight: 700; }}
QLabel#mono {{ color: {C['muted']}; font-family: 'DejaVu Sans Mono'; font-size: 10px; }}
QPushButton {{ background: transparent; border: 1px solid {C['border']}; border-radius: 5px; padding: 7px 11px; }}
QPushButton:hover {{ background: {C['panel2']}; }}
QPushButton:checked {{ background: {C['panel2']}; color: {C['text']}; }}
QPushButton#primary {{ background: {C['primary']}; border-color: {C['primary']}; color: #07110f; font-weight: 700; }}
QPushButton#primary:hover {{ background: #52e5d3; }}
QPushButton#secondary {{ background: {C['panel2']}; }}
QPushButton#ghost {{ border-color: transparent; color: {C['muted']}; }}
QPushButton#nav {{ text-align: left; border: 0; border-radius: 5px; padding: 9px 12px; color: #9ca9b4; }}
QPushButton#nav:hover {{ background: #111c26; color: {C['text']}; }}
QPushButton#nav:checked {{ background: #14242d; color: {C['primary']}; }}
QToolButton {{ background: transparent; border: 1px solid {C['border']}; border-radius: 5px; padding: 7px; }}
QToolButton:hover {{ background: {C['panel2']}; }}
QLineEdit, QComboBox, QTextEdit {{ background: {C['field']}; border: 1px solid {C['border']}; border-radius: 5px; padding: 7px 9px; selection-background-color: {C['primary']}; selection-color: #07110f; }}
QLineEdit:focus, QComboBox:focus, QTextEdit:focus {{ border-color: {C['primary']}; }}
QComboBox::drop-down {{ border: 0; width: 22px; }}
QComboBox QAbstractItemView {{ background: {C['panel']}; border: 1px solid {C['border']}; selection-background-color: {C['panel2']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #283743; border-radius: 4px; min-height: 35px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QProgressBar {{ background: {C['panel2']}; border: 0; border-radius: 3px; height: 6px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {C['primary']}; border-radius: 3px; }}
QDialog {{ background: {C['bg']}; }}
"""

ROOMS = (
    {"id": 1, "name": "Room 1", "capacity": 40, "sources": (0, 1)},
    {"id": 2, "name": "Room 2", "capacity": 25, "sources": (2, 3)},
    {"id": 3, "name": "Room 3", "capacity": 15, "sources": (4, 5)},
)
TYPE_LABEL = {
    "entry": "Kirish", "exit": "Chiqish", "transition": "Xonalar orasida",
    "unknown": "Unknown paydo bo'ldi", "restricted": "Restricted zone",
    "camera_offline": "Kamera offline", "service": "Servis",
}


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            clear_layout(item.layout())


def label(text: str, role: str | None = None, color: str | None = None) -> QLabel:
    w = QLabel(str(text))
    if role:
        w.setObjectName(role)
    if color:
        w.setStyleSheet(f"color:{color};")
    return w


def make_button(text: str, role: str = "") -> QPushButton:
    b = QPushButton(text)
    if role:
        b.setObjectName(role)
    b.setCursor(Qt.PointingHandCursor)
    return b


def panel_layout(panel, margins=(16, 15, 16, 15), spacing=8):
    lay = QVBoxLayout(panel)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return lay


def camera_name(source_id: int) -> str:
    return f"CAM-{int(source_id) + 1:02d}"


def room_name(room_id: int | None) -> str:
    for room in ROOMS:
        if room["id"] == room_id:
            return room["name"]
    return "—"


def fmt_epoch(value: float | int | None) -> str:
    if not value:
        return "—"
    return time.strftime("%d.%m.%Y, %H:%M:%S", time.localtime(float(value)))


class SentinelController(CameraQtController):
    def __init__(self):
        super().__init__()
        self.aliases: dict[tuple[int, int], str] = {}

    def set_person_name(self, source_id: int, object_id: int, name: str) -> None:
        name = name.strip()
        if name:
            self.aliases[(int(source_id), int(object_id))] = name

    def snapshot(self) -> dict:
        snap = super().snapshot()
        for track in snap.get("tracks", []):
            key = (int(track.get("source_id", -1)), int(track.get("object_id", -1)))
            alias = self.aliases.get(key)
            track["known"] = bool(alias)
            if alias:
                track["label"] = alias
        for event in snap.get("events", []):
            key = (int(event.get("source_id", -1)), int(event.get("object_id", -1)))
            alias = self.aliases.get(key)
            if alias:
                event["label"] = alias
        return snap


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
        self.layout.setContentsMargins(24, 22, 24, 28)
        self.layout.setSpacing(16)
        self.setWidget(self.body)


class StatCard(Panel):
    def __init__(self, heading: str, value: str, tone: str = "text", hint: str = ""):
        super().__init__()
        self.setMinimumHeight(120)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 15, 16, 14)
        lay.setSpacing(5)
        lay.addWidget(label(heading.upper(), "eyebrow"))
        self.value_label = label(str(value), "metric", C[tone])
        lay.addWidget(self.value_label)
        if hint:
            lay.addWidget(label(hint, "muted"))
        lay.addStretch()


class FaceAvatar(QWidget):
    def __init__(self, text: str, known: bool, hue: int, size=64):
        super().__init__()
        self.text, self.known, self.hue = text, known, hue
        self.setFixedSize(size, size)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c1 = QColor.fromHsl(self.hue, 115, 90)
        c2 = QColor.fromHsl((self.hue + 60) % 360, 100, 35)
        g = QRadialGradient(self.width() * .3, self.height() * .2, self.width())
        g.setColorAt(0, c1); g.setColorAt(1, c2)
        p.setBrush(g); p.setPen(QPen(QColor(C['border']), 1))
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)
        initials = "".join(x[0] for x in self.text.replace("_", " ").split()[:2]).upper() or "?"
        p.setPen(QColor(235, 240, 245, 220)); p.setFont(QFont("DejaVu Sans", max(10, self.width() // 4), QFont.Bold))
        p.drawText(self.rect(), Qt.AlignCenter, initials)
        p.setPen(Qt.NoPen); p.setBrush(QColor(C['known'] if self.known else C['unknown']))
        p.drawRect(0, self.height() - 4, self.width(), 4)


class CameraTile(QWidget):
    fullscreenRequested = Signal(int)
    heatmapToggled = Signal(int, bool)

    def __init__(self, source_id: int, parent=None):
        super().__init__(parent)
        self.source_id = source_id
        self.online = False
        self.fps = 0.0
        self.occupancy = 0
        self.points: list[tuple[float, float, float]] = []
        self.focused = False
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.controls = QWidget(self)
        self.controls.setAttribute(Qt.WA_TranslucentBackground, True)
        row = QHBoxLayout(self.controls)
        row.setContentsMargins(0, 0, 0, 0); row.setSpacing(5)
        self.heat_btn = make_button("Heatmap")
        self.heat_btn.setCheckable(True)
        self.full_btn = make_button("⛶")
        self.heat_btn.setToolTip("Movement heatmap")
        self.full_btn.setToolTip("Kamerani fullscreen ko'rish")
        self.heat_btn.toggled.connect(lambda state: self.heatmapToggled.emit(self.source_id, state))
        self.full_btn.clicked.connect(lambda: self.fullscreenRequested.emit(self.source_id))
        row.addWidget(self.heat_btn); row.addWidget(self.full_btn)
        self.controls.adjustSize(); self.controls.hide()

    def enterEvent(self, event):
        self.controls.show(); self.controls.raise_(); super().enterEvent(event)

    def leaveEvent(self, event):
        QTimer.singleShot(90, self._hide_if_outside); super().leaveEvent(event)

    def _hide_if_outside(self):
        if not self.underMouse() and not self.controls.underMouse():
            self.controls.hide()

    def mouseDoubleClickEvent(self, event):
        self.fullscreenRequested.emit(self.source_id); super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        self.controls.adjustSize()
        self.controls.move(max(8, self.width() - self.controls.width() - 10), 29)
        self.controls.raise_(); super().resizeEvent(event)

    def set_live(self, camera: dict, occupancy: int, points: list[tuple[float, float, float]]):
        self.online = bool(camera.get("online", False))
        self.fps = float(camera.get("fps", 0.0))
        self.occupancy = int(occupancy)
        self.points = points
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        p.setPen(QPen(QColor(C['primary'] if self.focused else C['border']), 2 if self.focused else 1))
        p.setBrush(Qt.NoBrush); p.drawRoundedRect(rect, 6, 6)
        if self.heat_btn.isChecked():
            for nx, ny, value in self.points:
                x, y = nx * self.width(), ny * self.height()
                radius = max(5.0, min(self.width(), self.height()) * (0.015 + 0.009 * min(1.0, value)))
                if value < 0.12:
                    color = QColor(40, 165, 255, 24 + int(value * 100))
                elif value < 0.35:
                    color = QColor(255, 218, 45, 28 + int(value * 100))
                else:
                    color = QColor(255, 65, 40, 34 + int(min(1.0, value) * 100))
                p.setPen(Qt.NoPen); p.setBrush(color); p.drawEllipse(QRectF(x-radius, y-radius, radius*2, radius*2))
        if not self.online:
            p.fillRect(rect, QColor(9, 13, 18, 190))
            p.setPen(QColor(C['offline'])); p.setFont(QFont("DejaVu Sans Mono", 10, QFont.Bold))
            p.drawText(QRectF(0, rect.height()/2-20, rect.width(), 20), Qt.AlignCenter, "▱  OFFLINE")
            p.setPen(QColor(C['muted'])); p.setFont(QFont("DejaVu Sans", 8))
            p.drawText(QRectF(20, rect.height()/2+5, rect.width()-40, 25), Qt.AlignHCenter, "RTSP qayta ulanmoqda")
        p.setFont(QFont("DejaVu Sans", 9, QFont.Bold)); p.setPen(QColor(C['text']))
        p.drawText(12, 22, camera_name(self.source_id))
        dot = QColor(C['known'] if self.online else C['offline'])
        p.setPen(Qt.NoPen); p.setBrush(dot); p.drawEllipse(self.width()-77, 13, 7, 7)
        p.setPen(QColor(C['muted'])); p.setFont(QFont("DejaVu Sans Mono", 7))
        p.drawText(self.width()-65, 21, f"{self.fps:.1f} fps" if self.online else "0.0 fps")
        badge = QRectF(10, self.height()-31, 54, 21)
        p.setPen(QPen(QColor(C['border']), 1)); p.setBrush(QColor(8, 14, 20, 220)); p.drawRoundedRect(badge, 5, 5)
        p.setPen(Qt.NoPen); p.setBrush(QColor(C['primary']))
        p.drawEllipse(QRectF(17, self.height()-25, 6, 6)); p.drawRoundedRect(QRectF(15, self.height()-18, 10, 5), 2, 2)
        p.setPen(QColor(C['text'])); p.setFont(QFont("DejaVu Sans Mono", 8, QFont.Bold))
        p.drawText(QRectF(29, self.height()-31, 28, 21), Qt.AlignVCenter, str(self.occupancy))


class LiveWall(QWidget):
    fullscreenRequested = Signal(int)

    def __init__(self, controller: SentinelController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.focus_source: int | None = None
        self._boot_requested = False
        self.setMinimumSize(760, 560)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#000;")
        self.setAttribute(Qt.WA_NativeWindow, True)
        _ = self.winId()
        self.tiles: list[CameraTile] = []
        for source_id in range(6):
            tile = CameraTile(source_id, self)
            tile.fullscreenRequested.connect(self.fullscreenRequested)
            tile.heatmapToggled.connect(controller.set_heatmap_enabled)
            self.tiles.append(tile)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._boot_requested:
            self._boot_requested = True
            QTimer.singleShot(100, self._start_pipeline)

    def _start_pipeline(self):
        self.controller.start(int(self.winId()))
        for tile in self.tiles:
            tile.raise_()

    def set_focus(self, source_id: int | None):
        self.focus_source = source_id
        self.controller.set_focus_source(source_id)
        self._relayout()

    def resizeEvent(self, event):
        self._relayout(); super().resizeEvent(event)

    def _relayout(self):
        if self.focus_source is not None:
            for tile in self.tiles:
                if tile.source_id == self.focus_source:
                    tile.focused = True; tile.setGeometry(self.rect()); tile.show(); tile.raise_()
                else:
                    tile.hide()
            return
        tile_w, tile_h = self.width()/2.0, self.height()/3.0
        for source_id, tile in enumerate(self.tiles):
            row, col = divmod(source_id, 2)
            tile.focused = False
            tile.setGeometry(int(col*tile_w), int(row*tile_h), int(tile_w), int(tile_h))
            tile.show(); tile.raise_()

    def refresh(self, snapshot: dict):
        cameras = {int(c.get("source_id", -1)): c for c in snapshot.get("cameras", [])}
        rooms = {int(r.get("room_id", -1)): int(r.get("count", 0)) for r in snapshot.get("rooms", [])}
        for tile in self.tiles:
            cam = cameras.get(tile.source_id, {})
            occupancy = rooms.get(tile.source_id // 2 + 1, 0)
            tile.set_live(cam, occupancy, self.controller.heat_points(tile.source_id))


class BarChart(QWidget):
    def __init__(self, series=None, labels=None, parent=None):
        super().__init__(parent)
        self.series = series or []
        self.labels = labels or []
        self.setMinimumHeight(220)

    def set_data(self, series, labels):
        self.series, self.labels = series, labels
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        area = self.rect().adjusted(44, 16, -16, -28)
        maxv = max([1] + [v for _, vals, _ in self.series for v in vals])
        p.setFont(QFont("DejaVu Sans Mono", 7))
        for i in range(5):
            y = area.bottom() - i*area.height()/4
            p.setPen(QColor(C['border'])); p.drawLine(area.left(), int(y), area.right(), int(y))
            p.setPen(QColor(C['muted'])); p.drawText(2, int(y-8), 36, 16, Qt.AlignRight|Qt.AlignVCenter, str(round(maxv*i/4)))
        count = max(1, len(self.labels)); group = area.width()/count; bw = min(16, group/(len(self.series)+1))
        for j, text in enumerate(self.labels):
            p.setPen(QColor(C['muted'])); p.drawText(QRectF(area.left()+j*group, area.bottom()+5, group, 18), Qt.AlignCenter, text)
            for k, (_, vals, color) in enumerate(self.series):
                if j >= len(vals):
                    continue
                h = vals[j]/maxv*area.height(); x = area.left()+j*group+(group-len(self.series)*bw)/2+k*bw
                p.fillRect(QRectF(x, area.bottom()-h, bw-3, h), QColor(color))


class MonitoringPage(QWidget):
    fullscreenRequested = Signal(int)

    def __init__(self, controller: SentinelController):
        super().__init__()
        self.controller = controller
        self.setObjectName("pageRoot")
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(22, 10, 22, 12)
        self.layout.setSpacing(16)
        camera_column = QVBoxLayout(); camera_column.setSpacing(8)
        self.wall = LiveWall(controller)
        self.wall.fullscreenRequested.connect(self.fullscreenRequested)
        camera_column.addWidget(self.wall, 1)
        self.layout.addLayout(camera_column, 3)
        self.identity_rail = QVBoxLayout(); self.identity_rail.setSpacing(12)
        metrics = QHBoxLayout(); metrics.setSpacing(10)
        self.known_card = StatCard("Known", "0", "known", "Hozir binoda")
        self.unknown_card = StatCard("Unknown", "0", "unknown", "Hozir binoda")
        self.known_card.setMinimumWidth(125); self.unknown_card.setMinimumWidth(125)
        metrics.addWidget(self.known_card); metrics.addWidget(self.unknown_card)
        self.identity_rail.addLayout(metrics, 1)
        self.recent_panel = Panel(); self.recent_panel.setMinimumWidth(285)
        recent_layout = panel_layout(self.recent_panel, (14, 14, 14, 14), 0)
        recent_head = QHBoxLayout(); recent_head.addWidget(label("Recent Views", "sectionTitle")); recent_head.addStretch()
        self.active_label = label("0 active", "mono"); recent_head.addWidget(self.active_label)
        recent_layout.addLayout(recent_head); recent_layout.addSpacing(8)
        self.recent = QVBoxLayout(); recent_layout.addLayout(self.recent); recent_layout.addStretch()
        self.identity_rail.addWidget(self.recent_panel, 3)
        self.layout.addLayout(self.identity_rail, 1)

    def recent_view(self, track: dict):
        item = QFrame(); item.setStyleSheet(f"QFrame{{border-bottom:1px solid {C['border']};background:transparent;}}")
        item.setMinimumHeight(58)
        row = QHBoxLayout(item); row.setContentsMargins(0, 5, 0, 5)
        info = QVBoxLayout(); info.setSpacing(3)
        text = str(track.get("label", "Unknown")); known = bool(track.get("known", False))
        info.addWidget(label(text, "sectionTitle"))
        ts = track.get("last_seen_epoch")
        info.addWidget(label(time.strftime("%H:%M:%S", time.localtime(ts)) if ts else camera_name(track.get("source_id", 0)), "mono"))
        row.addLayout(info, 1)
        row.addWidget(FaceAvatar(text, known, (int(track.get("object_id", 0))*47)%360, 42))
        return item

    def refresh(self, snapshot: dict):
        self.wall.refresh(snapshot)
        tracks = sorted(snapshot.get("tracks", []), key=lambda p: p.get("last_seen", 0), reverse=True)
        known = sum(1 for t in tracks if t.get("known")); unknown = len(tracks)-known
        self.known_card.value_label.setText(str(known)); self.unknown_card.value_label.setText(str(unknown)); self.active_label.setText(f"{len(tracks)} active")
        clear_layout(self.recent)
        for track in tracks[:6]:
            self.recent.addWidget(self.recent_view(track))

    def set_camera_fullscreen(self, source_id: int | None):
        visible = source_id is None
        self.known_card.setVisible(visible); self.unknown_card.setVisible(visible); self.recent_panel.setVisible(visible)
        self.wall.set_focus(source_id)


class PeoplePage(ScrollPage):
    def __init__(self, controller: SentinelController):
        super().__init__()
        self.controller = controller
        self.snapshot = {"tracks": []}
        self.filter = "all"
        top = QHBoxLayout(); self.buttons = QButtonGroup(self); self.buttons.setExclusive(True)
        for key, text in (("all", "Barchasi"), ("known", "Known"), ("unknown", "Unknown")):
            b = make_button(text); b.setCheckable(True); b.setChecked(key == "all"); b.clicked.connect(lambda _, k=key: self.set_filter(k)); self.buttons.addButton(b); top.addWidget(b)
        self.search = QLineEdit(); self.search.setPlaceholderText("Ism yoki Unknown_XX qidirish"); self.search.setMaximumWidth(320); self.search.textChanged.connect(self.rebuild)
        top.addWidget(self.search); top.addStretch(); self.layout.addLayout(top)
        self.grid = QGridLayout(); self.grid.setSpacing(12); self.layout.addLayout(self.grid); self.layout.addStretch()

    def set_filter(self, value):
        self.filter = value; self.rebuild()

    def refresh(self, snapshot: dict):
        self.snapshot = snapshot; self.rebuild()

    def rebuild(self):
        clear_layout(self.grid)
        q = self.search.text().lower() if hasattr(self, "search") else ""
        items = list(self.snapshot.get("tracks", []))
        if self.filter == "known":
            items = [p for p in items if p.get("known")]
        elif self.filter == "unknown":
            items = [p for p in items if not p.get("known")]
        if q:
            items = [p for p in items if q in str(p.get("label", "")).lower()]
        for i, p in enumerate(items):
            self.grid.addWidget(self.person_card(p), i//3, i%3)
        for col in range(3):
            self.grid.setColumnStretch(col, 1)

    def person_card(self, p: dict):
        card = Panel(); card.setMinimumWidth(295); lay = panel_layout(card, (12, 12, 12, 12), 8)
        text = str(p.get("label", "Unknown")); known = bool(p.get("known")); oid = int(p.get("object_id", 0)); sid = int(p.get("source_id", 0))
        top = QHBoxLayout(); top.addWidget(FaceAvatar(text, known, (oid*47)%360))
        info = QVBoxLayout(); name = make_button(text, "ghost"); name.setStyleSheet("text-align:left;font-weight:700;padding:0;border:0;"); info.addWidget(name)
        info.addWidget(label(f"{sid}:{oid}", "mono")); badge = label("KNOWN" if known else "UNKNOWN")
        badge.setStyleSheet(f"background:{C['known'] if known else C['unknown']};color:{C['bg']};padding:3px 6px;border-radius:3px;font:8px 'DejaVu Sans Mono';")
        info.addWidget(badge, 0, Qt.AlignLeft); info.addStretch(); top.addLayout(info, 1); lay.addLayout(top)
        details = [
            ("Birinchi", fmt_epoch(p.get("first_seen_epoch"))),
            ("Oxirgi", fmt_epoch(p.get("last_seen_epoch"))),
            ("Xona", room_name(p.get("room_id"))),
            ("Kameralar", ", ".join(p.get("cameras") or [camera_name(sid)])),
        ]
        for k, v in details:
            row = QHBoxLayout(); row.addWidget(label(k, "muted")); val = label(v, "mono"); val.setWordWrap(True); row.addWidget(val, 1, Qt.AlignRight); lay.addLayout(row)
        actions = QHBoxLayout()
        if not known:
            b = make_button("⌑  Ism berish", "secondary"); b.clicked.connect(lambda _, p=p: self.rename_person(p)); actions.addWidget(b)
        merge = make_button("⇉  Birlashtirish"); merge.clicked.connect(lambda _, p=p: self.merge_person(p)); actions.addWidget(merge)
        if len(p.get("cameras") or []) > 1:
            split = make_button("⑂  Ajratish", "ghost"); split.clicked.connect(lambda _, p=p: self.split_person(p)); actions.addWidget(split)
        actions.addStretch(); lay.addLayout(actions)
        return card

    def rename_person(self, p: dict):
        dlg = QDialog(self); dlg.setWindowTitle("Unknown odamga ism berish")
        l = QVBoxLayout(dlg); l.addWidget(label("Unknown odamga ism berish", "title")); inp = QLineEdit(); inp.setPlaceholderText("To'liq ism"); l.addWidget(inp)
        bb = QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel); bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); l.addWidget(bb)
        if dlg.exec() and inp.text().strip():
            self.controller.set_person_name(p.get("source_id", 0), p.get("object_id", 0), inp.text().strip())

    def merge_person(self, p: dict):
        QMessageBox.information(self, "Noto'g'ri ajralgan ID'ni birlashtirish", f"ID {p.get('source_id')}:{p.get('object_id')} tanlandi. Global Re-ID backend ulanganda merge qarori shu action orqali yuboriladi.")

    def split_person(self, p: dict):
        QMessageBox.information(self, "ID'ni ajratish", f"ID {p.get('source_id')}:{p.get('object_id')} uchun split action tayyor.")


class EventsPage(ScrollPage):
    TYPE_COLORS = {"entry": C['known'], "exit": C['muted'], "transition": C['blue'], "unknown": C['unknown'], "restricted": C['offline'], "camera_offline": C['offline'], "service": C['violet']}

    def __init__(self):
        super().__init__()
        self.snapshot = {"events": [], "tracks": []}
        filters = QGridLayout(); filters.setSpacing(8)
        self.kind = QComboBox(); self.kind.addItem("Barcha turlar", "all")
        for k, v in TYPE_LABEL.items(): self.kind.addItem(v, k)
        self.room = QComboBox(); self.room.addItem("Barcha xonalar", "all")
        for room in ROOMS: self.room.addItem(room["name"], room["id"])
        self.person = QComboBox(); self.person.addItem("Barcha odamlar", "all")
        self.date = QLineEdit(); self.date.setPlaceholderText("dd.mm.yyyy")
        for i, w in enumerate((self.kind, self.room, self.person, self.date)): filters.addWidget(w, 0, i)
        self.kind.currentIndexChanged.connect(self.rebuild); self.room.currentIndexChanged.connect(self.rebuild); self.person.currentIndexChanged.connect(self.rebuild); self.date.textChanged.connect(self.rebuild)
        self.layout.addLayout(filters); self.list = QVBoxLayout(); self.list.setSpacing(8); self.layout.addLayout(self.list); self.layout.addStretch()

    def refresh(self, snapshot: dict):
        self.snapshot = snapshot
        current = self.person.currentData()
        labels = {}
        for t in snapshot.get("tracks", []): labels[f"{t.get('source_id')}:{t.get('object_id')}"] = t.get("label", "Unknown")
        for e in snapshot.get("events", []): labels[str(e.get("person_id") or f"{e.get('source_id')}:{e.get('object_id')}")] = e.get("label", "Unknown")
        self.person.blockSignals(True); self.person.clear(); self.person.addItem("Barcha odamlar", "all")
        for pid, text in sorted(labels.items(), key=lambda x: str(x[1])): self.person.addItem(str(text), pid)
        idx = self.person.findData(current); self.person.setCurrentIndex(idx if idx >= 0 else 0); self.person.blockSignals(False)
        self.rebuild()

    def rebuild(self):
        clear_layout(self.list)
        kind, room, person, date = self.kind.currentData(), self.room.currentData(), self.person.currentData(), self.date.text().strip()
        rows = []
        for e in self.snapshot.get("events", []):
            pid = str(e.get("person_id") or f"{e.get('source_id')}:{e.get('object_id')}")
            if kind != "all" and e.get("type") != kind: continue
            if room != "all" and e.get("room_id") != room: continue
            if person != "all" and pid != str(person): continue
            if date and date not in fmt_epoch(e.get("time")): continue
            rows.append(e)
        for e in rows:
            kind_key = e.get("type", "service"); color = self.TYPE_COLORS.get(kind_key, C['violet'])
            card = Panel(); card.setMinimumHeight(80); lay = QHBoxLayout(card); lay.setContentsMargins(12, 10, 12, 10); lay.setSpacing(12)
            thumb = QLabel(); thumb.setFixedSize(64, 48); thumb.setStyleSheet(f"background:{color}18;border:1px solid {C['border']};border-radius:4px;"); lay.addWidget(thumb)
            info = QVBoxLayout(); top = QHBoxLayout(); tag = label(TYPE_LABEL.get(kind_key, kind_key)); tag.setStyleSheet(f"background:{color};color:{C['bg']};padding:3px 6px;border-radius:3px;font:8px 'DejaVu Sans Mono';")
            top.addWidget(tag); top.addWidget(label(fmt_epoch(e.get("time")), "mono")); top.addStretch(); info.addLayout(top); info.addWidget(label(e.get("message", ""))); info.addWidget(label(f"{e.get('camera_id','—')} · {room_name(e.get('room_id'))}", "mono")); lay.addLayout(info, 1)
            if e.get("object_id") is not None: lay.addWidget(make_button("Profil", "ghost"))
            self.list.addWidget(card)
        if not rows:
            self.list.addWidget(label("Hali hodisa yo'q", "muted"))


class RoomsPage(ScrollPage):
    def __init__(self):
        super().__init__()
        self.grid = QGridLayout(); self.grid.setSpacing(16); self.layout.addLayout(self.grid); self.layout.addStretch()

    def refresh(self, snapshot: dict):
        clear_layout(self.grid)
        tracks = snapshot.get("tracks", []); cameras = {int(c.get("source_id", -1)): c for c in snapshot.get("cameras", [])}; room_rows = {int(r.get("room_id", -1)): r for r in snapshot.get("rooms", [])}
        for i, room in enumerate(ROOMS):
            room_row = room_rows.get(room["id"], {}); occupants = [p for p in tracks if p.get("room_id") == room["id"]]; count = int(room_row.get("count", 0)); load = round(count/room["capacity"]*100) if room["capacity"] else 0
            card = Panel(); lay = panel_layout(card); top = QHBoxLayout(); top.addWidget(label(room["name"], "sectionTitle")); top.addStretch(); top.addWidget(label(str(count), "metric", C['primary'])); lay.addLayout(top)
            bar = QProgressBar(); bar.setMaximum(100); bar.setValue(min(100, load)); lay.addWidget(bar); lay.addWidget(label(f"sig'im {room['capacity']} · {load}% band", "mono")); lay.addSpacing(8); lay.addWidget(label("KAMERALAR", "eyebrow"))
            for sid in room["sources"]:
                cam = cameras.get(sid, {}); row = QHBoxLayout(); row.addWidget(label(camera_name(sid))); row.addStretch(); online = bool(cam.get("online")); row.addWidget(label(f"{float(cam.get('fps',0)):.1f} fps" if online else "offline", "mono", C['known'] if online else C['offline'])); lay.addLayout(row)
            lay.addSpacing(8); lay.addWidget(label("HOZIR XONADA", "eyebrow"))
            if occupants:
                for p in occupants[:8]:
                    row = QHBoxLayout(); text = str(p.get("label", "Unknown")); row.addWidget(FaceAvatar(text, bool(p.get("known")), (int(p.get("object_id", 0))*47)%360, 32)); row.addWidget(label(text)); row.addStretch(); lay.addLayout(row)
            else:
                lay.addWidget(label("Xona bo'sh", "muted"))
            lay.addStretch(); self.grid.addWidget(card, 0, i); self.grid.setColumnStretch(i, 1)


class EnrollmentPage(ScrollPage):
    def __init__(self):
        super().__init__()
        self.image_paths: list[str] = []; self.profile_index: int | None = None
        body = QHBoxLayout(); body.setSpacing(16)
        form = Panel(); form.setMaximumWidth(370); form_layout = panel_layout(form, (18, 18, 18, 18), 10)
        form_layout.addWidget(label("Shaxs ma'lumotlari", "sectionTitle")); form_layout.addWidget(label("Ism", "muted"))
        self.name = QLineEdit(); self.name.setPlaceholderText("To'liq ism"); form_layout.addWidget(self.name)
        form_layout.addWidget(label("Qo'shimcha ma'lumot", "muted")); self.note = QTextEdit(); self.note.setPlaceholderText("Lavozim, bo'lim, ruxsatlar"); self.note.setMaximumHeight(82); form_layout.addWidget(self.note)
        profile_box = Panel(); profile_layout = QVBoxLayout(profile_box); profile_layout.setContentsMargins(12, 12, 12, 12); profile_layout.addWidget(label("PROFILE PHOTO", "eyebrow"), 0, Qt.AlignCenter)
        self.profile_preview = QLabel("Tanlanmagan"); self.profile_preview.setAlignment(Qt.AlignCenter); self.profile_preview.setFixedSize(170, 170); self.profile_preview.setStyleSheet(f"background:{C['field']};border:1px dashed {C['border']};border-radius:7px;color:{C['muted']};"); profile_layout.addWidget(self.profile_preview, 0, Qt.AlignCenter)
        self.profile_name = label("Rasmlardan birini tanlang", "muted"); self.profile_name.setAlignment(Qt.AlignCenter); profile_layout.addWidget(self.profile_name); form_layout.addWidget(profile_box)
        self.count = label("Rasmlar: 0/10 · Profile photo: tanlanmagan", "muted"); self.count.setStyleSheet(f"border:1px solid {C['border']};border-radius:5px;padding:10px;color:{C['muted']};"); form_layout.addWidget(self.count)
        self.finish_button = make_button("✓  Enroll", "primary"); self.finish_button.clicked.connect(self.finish); form_layout.addWidget(self.finish_button); form_layout.addStretch(); body.addWidget(form, 1)
        photos = Panel(); photos_layout = panel_layout(photos, (18, 18, 18, 18), 10); photos_header = QHBoxLayout(); header_text = QVBoxLayout(); header_text.addWidget(label("10 ta yuz rasmi", "sectionTitle")); header_text.addWidget(label("Bitta odamning aniq, turli burchakdan olingan rasmlarini tanlang.", "muted")); photos_header.addLayout(header_text); photos_header.addStretch()
        choose = make_button("＋  10 ta rasm tanlash", "primary"); choose.clicked.connect(self.select_images); photos_header.addWidget(choose); photos_layout.addLayout(photos_header)
        self.photo_group = QButtonGroup(self); self.photo_group.setExclusive(True); self.photo_buttons = []; self.photo_labels = []; grid = QGridLayout(); grid.setSpacing(12)
        for index in range(10):
            cell = QVBoxLayout(); tile = make_button("＋"); tile.setCheckable(True); tile.setEnabled(False); tile.setMinimumSize(125, 112); tile.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); tile.clicked.connect(lambda _, i=index: self.select_profile(i)); self.photo_group.addButton(tile, index); self.photo_buttons.append(tile); cell.addWidget(tile)
            caption = label(f"Rasm {index+1} · bo'sh", "mono"); caption.setAlignment(Qt.AlignCenter); self.photo_labels.append(caption); cell.addWidget(caption); grid.addLayout(cell, index//5, index%5); grid.setColumnStretch(index%5, 1)
        photos_layout.addLayout(grid); hint = label("10 ta rasm yuklang, keyin eng yaxshi tushgan rasm ustiga bosib profile photo sifatida tanlang.", "muted"); hint.setWordWrap(True); photos_layout.addWidget(hint); photos_layout.addStretch(); body.addWidget(photos, 3)
        self.layout.addLayout(body); self.layout.addStretch()

    def select_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "10 ta yuz rasmini tanlang", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not paths: return
        if len(paths) != 10: QMessageBox.warning(self, "Rasmlar soni", "Aynan 10 ta rasm tanlash kerak."); return
        valid = [p for p in paths if not QPixmap(p).isNull()]
        if len(valid) != 10: QMessageBox.warning(self, "Noto'g'ri fayl", "Tanlangan fayllarning barchasi ochiladigan rasm bo'lishi kerak."); return
        self.image_paths = valid; self.profile_index = None; self.profile_preview.clear(); self.profile_preview.setText("Tanlanmagan"); self.profile_name.setText("Rasmlardan birini tanlang")
        for index, path in enumerate(valid):
            b = self.photo_buttons[index]; b.setEnabled(True); b.setChecked(False); b.setText(""); b.setIcon(QIcon(path)); b.setIconSize(QSize(160, 104)); b.setStyleSheet(f"border:1px solid {C['border']};border-radius:6px;padding:3px;"); self.photo_labels[index].setText(f"Rasm {index+1}")
        self.update_status()

    def select_profile(self, index):
        if index >= len(self.image_paths): return
        self.profile_index = index
        for current, b in enumerate(self.photo_buttons):
            selected = current == index; b.setChecked(selected); border = C['primary'] if selected else C['border']; width = 3 if selected else 1; b.setStyleSheet(f"border:{width}px solid {border};border-radius:6px;padding:3px;")
            self.photo_labels[current].setText(f"Rasm {current+1} · PROFILE" if selected else f"Rasm {current+1}"); self.photo_labels[current].setStyleSheet(f"color:{C['primary'] if selected else C['muted']};font:10px 'DejaVu Sans Mono';")
        pix = QPixmap(self.image_paths[index]); self.profile_preview.setPixmap(pix.scaled(self.profile_preview.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)); self.profile_name.setText(f"Rasm {index+1} tanlandi"); self.update_status()

    def update_status(self):
        profile = f"Rasm {self.profile_index+1}" if self.profile_index is not None else "tanlanmagan"; self.count.setText(f"Rasmlar: {len(self.image_paths)}/10 · Profile photo: {profile}")

    def finish(self):
        name = self.name.text().strip()
        if not name: QMessageBox.warning(self, "Enrollment", "Shaxsning to'liq ismini kiriting."); return
        if len(self.image_paths) != 10: QMessageBox.warning(self, "Enrollment", "Enrollment uchun aynan 10 ta yuz rasmi kerak."); return
        if self.profile_index is None: QMessageBox.warning(self, "Enrollment", "Eng yaxshi rasmni profile photo sifatida tanlang."); return
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "person"; root = Path.cwd()/"data"/"enrollments"/safe; root.mkdir(parents=True, exist_ok=True); copied=[]
        for index, source in enumerate(self.image_paths, 1):
            dest = root/f"face_{index:02d}{Path(source).suffix.lower()}"; shutil.copy2(source, dest); copied.append(str(dest))
        manifest = {"name": name, "note": self.note.toPlainText().strip(), "created_at": time.time(), "profile_index": self.profile_index, "profile_path": copied[self.profile_index], "images": copied}
        (root/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Enrollment", f"{name} bazaga saqlandi.\nProfile photo: Rasm {self.profile_index+1}")


class ReportsPage(ScrollPage):
    def __init__(self, controller: SentinelController):
        super().__init__(); self.controller = controller; self.snapshot = {"events": [], "rooms": [], "tracks": [], "cameras": []}
        top = QHBoxLayout(); top.addStretch(); csv_btn = make_button("⇩  CSV"); pdf_btn = make_button("⇩  PDF", "primary"); csv_btn.clicked.connect(self.export_csv); pdf_btn.clicked.connect(self.export_pdf); top.addWidget(csv_btn); top.addWidget(pdf_btn); self.layout.addLayout(top)
        first = QHBoxLayout(); p1 = Panel(); p1l = panel_layout(p1); p1l.addWidget(label("Soatlik kirish va chiqish", "sectionTitle")); self.flow_chart = BarChart(); p1l.addWidget(self.flow_chart); first.addWidget(p1, 2)
        p2 = Panel(); p2l = panel_layout(p2); p2l.addWidget(label("Known / Unknown", "sectionTitle")); self.identity_chart = BarChart(); p2l.addWidget(self.identity_chart); first.addWidget(p2, 1); self.layout.addLayout(first)
        second = QHBoxLayout(); p3 = Panel(); p3l = panel_layout(p3); p3l.addWidget(label("Xonalar bo'yicha bandlik", "sectionTitle")); self.room_chart = BarChart(); p3l.addWidget(self.room_chart); second.addWidget(p3, 2)
        self.summary = Panel(); self.summary_layout = panel_layout(self.summary); self.summary_layout.addWidget(label("Bugungi xulosa", "sectionTitle")); second.addWidget(self.summary, 1); self.layout.addLayout(second); self.layout.addStretch()

    def refresh(self, snapshot: dict):
        self.snapshot = snapshot
        now = time.localtime(); base_hour = now.tm_hour
        labels = [f"{(base_hour-6+i)%24:02d}" for i in range(7)]; entries = [0]*7; exits = [0]*7
        for e in snapshot.get("events", []):
            ts = e.get("time")
            if not ts: continue
            h = time.localtime(ts).tm_hour
            for i, lab in enumerate(labels):
                if h == int(lab):
                    if e.get("type") == "entry": entries[i] += 1
                    elif e.get("type") == "exit": exits[i] += 1
        self.flow_chart.set_data([("Kirish", entries, C['known']), ("Chiqish", exits, C['unknown'])], labels)
        tracks = snapshot.get("tracks", []); known = sum(1 for t in tracks if t.get("known")); unknown = len(tracks)-known
        self.identity_chart.set_data([("Odam", [known, unknown], C['primary'])], ["Known", "Unknown"])
        rooms = snapshot.get("rooms", []); self.room_chart.set_data([("Bandlik", [int(r.get("count",0)) for r in rooms[:3]], C['blue'])], ["Room 1", "Room 2", "Room 3"])
        while self.summary_layout.count() > 1:
            item = self.summary_layout.takeAt(1)
            if item.widget(): item.widget().deleteLater()
            elif item.layout(): clear_layout(item.layout())
        summary = [("Jami hodisa", str(len(snapshot.get("events", [])))), ("Known", str(known)), ("Unknown", str(unknown)), ("Online kamera", f"{sum(1 for c in snapshot.get('cameras', []) if c.get('online'))}/6"), ("Aktiv odam", str(len(tracks)))]
        for k, v in summary:
            row = QHBoxLayout(); row.addWidget(label(k)); row.addStretch(); row.addWidget(label(v, "mono")); self.summary_layout.addLayout(row)
        self.summary_layout.addStretch()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "CSV export", "sentinel_events.csv", "CSV (*.csv)")
        if path: self.controller.export_events_csv(path); QMessageBox.information(self, "Reports", f"Saved: {path}")

    def export_pdf(self):
        path, _ = QFileDialog.getSaveFileName(self, "PDF export", "sentinel_report.pdf", "PDF (*.pdf)")
        if not path: return
        writer = QPdfWriter(path); writer.setResolution(96); p = QPainter(writer); p.setPen(QColor("#111111")); p.setFont(QFont("DejaVu Sans", 18, QFont.Bold)); p.drawText(70, 90, "Sentinel VMS report"); p.setFont(QFont("DejaVu Sans", 10)); y=135
        lines = [f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", f"Active people: {len(self.snapshot.get('tracks', []))}", f"Events: {len(self.snapshot.get('events', []))}", f"Online cameras: {sum(1 for c in self.snapshot.get('cameras', []) if c.get('online'))}/6"]
        for line_text in lines: p.drawText(70, y, line_text); y += 26
        p.end(); QMessageBox.information(self, "Reports", f"Saved: {path}")


class MainWindow(QMainWindow):
    NAV = [
        ("▣", "Monitoring", "6 ta jonli kamera · bir ekranda · so'nggi kuzatuvlar"),
        ("♙", "People", "0 ta global ID"),
        ("⌁", "Events", "0 ta hodisa"),
        ("▥", "Rooms", "Kameralar orasidagi bir xil odam bir marta hisoblanadi (global ID bo'yicha)"),
        ("♙+", "Enrollment", "10 ta yuz rasmi va profile photo bilan ro'yxatga olish"),
        ("▤", "Reports", "Kunlik va haftalik hisobotlar"),
    ]

    def __init__(self):
        super().__init__()
        self.controller = SentinelController(); self._snapshot = {"cameras": [], "tracks": [], "events": [], "rooms": []}; self._camera_fullscreen: int | None = None; self._grid_fullscreen = False
        self.setWindowTitle("SENTINEL VMS"); self.resize(1440, 900); self.setMinimumSize(1180, 720)
        root = QWidget(); root.setObjectName("root"); self.setCentralWidget(root); main = QHBoxLayout(root); main.setContentsMargins(0,0,0,0); main.setSpacing(0)
        self.sidebar = QFrame(); self.sidebar.setObjectName("sidebar"); self.sidebar.setFixedWidth(224); side = QVBoxLayout(self.sidebar); side.setContentsMargins(0,0,0,0); side.setSpacing(0)
        brand = QFrame(); brand.setFixedHeight(70); brand.setStyleSheet(f"border-bottom:1px solid {C['border']};"); bl = QHBoxLayout(brand); bl.setContentsMargins(16,0,12,0); shield = label("◇", color=C['primary']); shield.setStyleSheet(f"color:{C['primary']};font-size:24px;"); bl.addWidget(shield)
        bt = QVBoxLayout(); bt.setSpacing(1); bt.addWidget(label("SENTINEL VMS", "brand")); bt.addWidget(label("face re-id · 6 cam", "mono")); bl.addLayout(bt); bl.addStretch(); side.addWidget(brand)
        self.nav_group = QButtonGroup(self); self.nav_group.setExclusive(True); self.nav_buttons=[]; navwrap=QWidget(); nl=QVBoxLayout(navwrap); nl.setContentsMargins(8,8,8,8); nl.setSpacing(2)
        for i,(icon,title,_) in enumerate(self.NAV):
            b=make_button(f"{icon:>2}   {title}"); b.setObjectName("nav"); b.setCheckable(True); b.setFixedHeight(38); b.clicked.connect(lambda _,i=i:self.switch_page(i)); self.nav_group.addButton(b); self.nav_buttons.append(b); nl.addWidget(b)
        nl.addStretch(); side.addWidget(navwrap,1); build=label("build 2026.08 · edge worker","mono"); build.setStyleSheet(f"border-top:1px solid {C['border']};padding:14px;color:{C['muted']};"); side.addWidget(build); main.addWidget(self.sidebar)
        content=QWidget(); content_l=QVBoxLayout(content); content_l.setContentsMargins(0,0,0,0); content_l.setSpacing(0); self.header=QFrame(); self.header.setObjectName("header"); self.header.setFixedHeight(70); hl=QHBoxLayout(self.header); hl.setContentsMargins(24,0,24,0)
        titles=QVBoxLayout(); titles.setSpacing(2); self.title=label("Monitoring","title"); self.subtitle=label(self.NAV[0][2],"subtitle"); titles.addWidget(self.title); titles.addWidget(self.subtitle); hl.addLayout(titles); hl.addStretch()
        self.camera_fullscreen=QToolButton(); self.camera_fullscreen.setText("⛶  Fullscreen"); self.camera_fullscreen.setToolTip("Barcha kameralarni fullscreen ko'rish"); self.camera_fullscreen.clicked.connect(self.toggle_grid_fullscreen); hl.addWidget(self.camera_fullscreen); content_l.addWidget(self.header)
        self.stack=QStackedWidget(); self.monitoring=MonitoringPage(self.controller); self.monitoring.fullscreenRequested.connect(self.enter_camera_fullscreen); self.people=PeoplePage(self.controller); self.events=EventsPage(); self.rooms=RoomsPage(); self.enrollment=EnrollmentPage(); self.reports=ReportsPage(self.controller); self.pages=[self.monitoring,self.people,self.events,self.rooms,self.enrollment,self.reports]
        for page in self.pages: self.stack.addWidget(page)
        content_l.addWidget(self.stack,1); main.addWidget(content,1); self.nav_buttons[0].setChecked(True)
        self.fast_timer=QTimer(self); self.fast_timer.setInterval(250); self.fast_timer.timeout.connect(self.refresh_monitoring); self.fast_timer.start(); self.slow_timer=QTimer(self); self.slow_timer.setInterval(1000); self.slow_timer.timeout.connect(self.refresh_pages); self.slow_timer.start()

    def switch_page(self,index):
        if self._camera_fullscreen is not None: self.exit_camera_fullscreen()
        self.stack.setCurrentIndex(index); icon,title,subtitle=self.NAV[index]
        if index == 1: subtitle=f"{len(self._snapshot.get('tracks', []))} ta global ID"
        elif index == 2: subtitle=f"{len(self._snapshot.get('events', []))} ta hodisa"
        self.title.setText(title); self.subtitle.setText(subtitle); self.camera_fullscreen.setVisible(index==0)
        for i,b in enumerate(self.nav_buttons): b.setChecked(i==index)

    def refresh_monitoring(self):
        try:
            self._snapshot=self.controller.snapshot(); self.monitoring.refresh(self._snapshot)
        except Exception as exc:
            print(f"SENTINEL refresh error: {exc}", file=sys.stderr, flush=True)

    def refresh_pages(self):
        self.people.refresh(self._snapshot); self.events.refresh(self._snapshot); self.rooms.refresh(self._snapshot); self.reports.refresh(self._snapshot)
        idx=self.stack.currentIndex()
        if idx in (1,2): self.switch_page(idx)

    def enter_camera_fullscreen(self, source_id:int):
        self._camera_fullscreen=source_id; self.sidebar.hide(); self.header.hide(); self.monitoring.set_camera_fullscreen(source_id); self.showFullScreen()

    def exit_camera_fullscreen(self):
        self._camera_fullscreen=None; self.monitoring.set_camera_fullscreen(None); self.sidebar.show(); self.header.show(); self.showNormal(); self.showMaximized()

    def toggle_grid_fullscreen(self):
        if self._camera_fullscreen is not None: self.exit_camera_fullscreen(); return
        if self._grid_fullscreen:
            self._grid_fullscreen=False; self.sidebar.show(); self.header.show(); self.showNormal(); self.showMaximized()
        else:
            self._grid_fullscreen=True; self.sidebar.hide(); self.header.hide(); self.showFullScreen()

    def keyPressEvent(self,event):
        if event.key() in (Qt.Key_Escape,Qt.Key_F11):
            if self._camera_fullscreen is not None: self.exit_camera_fullscreen(); return
            if self._grid_fullscreen: self.toggle_grid_fullscreen(); return
        super().keyPressEvent(event)

    def closeEvent(self,event):
        self.fast_timer.stop(); self.slow_timer.stop(); self.controller.stop(); event.accept()


def main() -> int:
    app=QApplication(sys.argv); app.setApplicationName("Sentinel VMS"); app.setOrganizationName("Sentinel"); app.setStyle("Fusion"); app.setStyleSheet(APP_QSS)
    window=MainWindow(); window.showMaximized(); return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
