from __future__ import annotations

from collections import deque
import http.client
import json
import threading
import time
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, QSize, QRectF, QPointF
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

CAMERAS = [f"CAM-{i:02d}" for i in range(1, 7)]
ML_HOST = "127.0.0.1"
ML_PORT = 8001

BG = "#00091a"
SIDEBAR_BG = "#000f25"
PANEL = "#001329"
CARD = "#00172f"
CARD_2 = "#001c39"
BORDER = "#07365c"
TEXT = "#f4f7fb"
MUTED = "#94a6bc"
BLUE = "#1168ff"
BLUE_DARK = "#084bd7"
GREEN = "#00e57b"
ORANGE = "#ff8a00"
RED = "#ff3b4d"
CYAN = "#15b8ff"
PURPLE = "#9f4cff"

CAMERA_TITLES = {
    "CAM-01": "Office 1 (A)",
    "CAM-02": "Office 2 (A)",
    "CAM-03": "Office 3 (A)",
    "CAM-04": "Office 1 (B)",
    "CAM-05": "Office 2 (B)",
    "CAM-06": "Office 3 (B)",
}


def font(px: int, weight=QFont.Weight.Normal) -> QFont:
    f = QFont("Inter")
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


def icon_pixmap(kind: str, color: str = TEXT, size: int = 24) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(color), max(1.5, size / 14))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    s = float(size)

    if kind == "menu":
        for y in (.28, .50, .72):
            p.drawLine(QPointF(s*.18, s*y), QPointF(s*.82, s*y))
    elif kind == "monitor":
        p.drawRoundedRect(QRectF(s*.12, s*.16, s*.76, s*.56), s*.06, s*.06)
        p.drawLine(QPointF(s*.40, s*.80), QPointF(s*.60, s*.80))
        p.drawLine(QPointF(s*.50, s*.72), QPointF(s*.50, s*.80))
    elif kind == "person":
        p.drawEllipse(QRectF(s*.38, s*.13, s*.24, s*.24))
        path = QPainterPath()
        path.moveTo(s*.22, s*.82)
        path.cubicTo(s*.27, s*.55, s*.73, s*.55, s*.78, s*.82)
        p.drawPath(path)
    elif kind == "users":
        p.drawEllipse(QRectF(s*.25, s*.14, s*.23, s*.23))
        p.drawEllipse(QRectF(s*.55, s*.21, s*.18, s*.18))
        p.drawArc(QRectF(s*.14, s*.44, s*.48, s*.38), 15*16, 150*16)
        p.drawArc(QRectF(s*.47, s*.50, s*.38, s*.30), 15*16, 145*16)
    elif kind == "bell":
        path = QPainterPath()
        path.moveTo(s*.28, s*.67)
        path.lineTo(s*.34, s*.58)
        path.lineTo(s*.34, s*.40)
        path.cubicTo(s*.34, s*.18, s*.66, s*.18, s*.66, s*.40)
        path.lineTo(s*.66, s*.58)
        path.lineTo(s*.72, s*.67)
        p.drawPath(path)
        p.drawLine(QPointF(s*.27, s*.67), QPointF(s*.73, s*.67))
    elif kind == "report":
        p.drawRoundedRect(QRectF(s*.17, s*.13, s*.66, s*.72), s*.04, s*.04)
        p.drawLine(QPointF(s*.31, s*.67), QPointF(s*.31, s*.50))
        p.drawLine(QPointF(s*.50, s*.67), QPointF(s*.50, s*.35))
        p.drawLine(QPointF(s*.69, s*.67), QPointF(s*.69, s*.44))
    elif kind == "settings":
        p.drawEllipse(QRectF(s*.38, s*.38, s*.24, s*.24))
        p.drawEllipse(QRectF(s*.26, s*.26, s*.48, s*.48))
    elif kind == "camera":
        p.drawRoundedRect(QRectF(s*.14, s*.29, s*.50, s*.39), s*.04, s*.04)
        path = QPainterPath()
        path.moveTo(s*.64, s*.39)
        path.lineTo(s*.85, s*.29)
        path.lineTo(s*.85, s*.69)
        path.lineTo(s*.64, s*.58)
        path.closeSubpath()
        p.drawPath(path)
    elif kind == "activity":
        path = QPainterPath()
        path.moveTo(s*.07, s*.55)
        path.lineTo(s*.27, s*.55)
        path.lineTo(s*.36, s*.22)
        path.lineTo(s*.47, s*.79)
        path.lineTo(s*.57, s*.42)
        path.lineTo(s*.66, s*.55)
        path.lineTo(s*.93, s*.55)
        p.drawPath(path)
    elif kind == "fullscreen":
        for a, b, c in [
            ((.18,.38),(.18,.18),(.38,.18)),
            ((.62,.18),(.82,.18),(.82,.38)),
            ((.18,.62),(.18,.82),(.38,.82)),
            ((.62,.82),(.82,.82),(.82,.62)),
        ]:
            p.drawLine(QPointF(s*a[0],s*a[1]), QPointF(s*b[0],s*b[1]))
            p.drawLine(QPointF(s*b[0],s*b[1]), QPointF(s*c[0],s*c[1]))
    else:
        p.drawRoundedRect(QRectF(s*.20, s*.20, s*.60, s*.60), s*.05, s*.05)
    p.end()
    return pm


def icon(kind: str, color: str = TEXT, size: int = 24) -> QIcon:
    return QIcon(icon_pixmap(kind, color, size))


class FrameReader:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.image = None
        self.version = -1
        self.frames = 0
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name=f"ui-{self.camera_id}", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def latest(self):
        with self.lock:
            return self.image, self.version

    def _run(self):
        conn = None
        version = -1
        while not self.stop_event.is_set():
            try:
                if conn is None:
                    conn = http.client.HTTPConnection(ML_HOST, ML_PORT, timeout=2.0)
                conn.request(
                    "GET",
                    f"/frame/{self.camera_id}?after={version}&wait_ms=180",
                    headers={"Connection": "keep-alive", "Cache-Control": "no-cache"},
                )
                response = conn.getresponse()
                payload = response.read()
                if response.status != 200:
                    raise RuntimeError(response.status)
                next_version = int(response.getheader("X-Frame-Version") or version + 1)
                if next_version <= version:
                    continue
                image = QImage.fromData(payload, "JPG")
                if image.isNull():
                    continue
                version = next_version
                with self.lock:
                    self.image = image
                    self.version = version
                    self.frames += 1
            except Exception:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                conn = None
                self.stop_event.wait(0.08)


class RealtimeState:
    def __init__(self):
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.state = {"connected": False, "health": {}, "detections": {}, "reid": {}}
        self.recent = deque(maxlen=24)
        self.events = deque(maxlen=100)
        self.seen = {}

    def start(self):
        threading.Thread(target=self._run, name="ui-realtime-state", daemon=True).start()

    def stop(self):
        self.stop_event.set()

    def snapshot(self):
        with self.lock:
            return dict(self.state), list(self.recent), list(self.events)

    @staticmethod
    def _json(conn, path):
        conn.request("GET", path, headers={"Connection":"keep-alive","Cache-Control":"no-cache"})
        response = conn.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(response.status)
        return json.loads(payload.decode("utf-8"))

    def _observe_reid(self, payload):
        now = datetime.now().strftime("%H:%M:%S")
        cameras = (((payload or {}).get("state") or {}).get("cameras") or {})
        for camera_id, tracks in cameras.items():
            for track in tracks or []:
                gid = str(track.get("global_id") or "").strip()
                local_id = int(track.get("local_id") or 0)
                if not gid:
                    continue
                key = (camera_id, local_id)
                previous = self.seen.get(key)
                self.seen[key] = gid
                if previous == gid:
                    continue
                event = {
                    "time": now,
                    "camera": camera_id,
                    "gid": gid,
                    "similarity": track.get("similarity"),
                    "reason": str(track.get("reason") or "detected"),
                }
                self.recent.appendleft(event)
                self.events.appendleft(event)

    def _run(self):
        conn = None
        while not self.stop_event.is_set():
            try:
                if conn is None:
                    conn = http.client.HTTPConnection(ML_HOST, ML_PORT, timeout=1.5)
                health = self._json(conn, "/health")
                detections = self._json(conn, "/detections")
                reid = self._json(conn, "/reid")
                self._observe_reid(reid)
                with self.lock:
                    self.state = {
                        "connected": True,
                        "health": health,
                        "detections": detections,
                        "reid": reid,
                    }
                self.stop_event.wait(0.35)
            except Exception:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                conn = None
                with self.lock:
                    self.state = {**self.state, "connected": False}
                self.stop_event.wait(0.7)


class CameraImage(QLabel):
    def __init__(self):
        super().__init__("Connecting…")
        self._image = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, 140)
        self.setStyleSheet(f"background:#020913;color:{MUTED};border:0;")

    def set_frame(self, image: QImage):
        self._image = image
        self._apply()

    def _apply(self):
        if self._image is None or self.width() < 2 or self.height() < 2:
            return
        pix = QPixmap.fromImage(self._image)
        self.setPixmap(
            pix.scaled(
                self.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply()


class CameraTile(QFrame):
    def __init__(self, camera_id: str, number: int):
        super().__init__()
        self.setObjectName("cameraTile")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = QWidget()
        self.header.setFixedHeight(42)
        h = QHBoxLayout(self.header)
        h.setContentsMargins(9, 0, 10, 0)
        chip = QLabel(f"{number:02d}")
        chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chip.setFixedSize(32, 28)
        chip.setFont(font(14, QFont.Weight.DemiBold))
        chip.setStyleSheet(f"background:{BLUE_DARK};border-radius:6px;")
        title = QLabel(CAMERA_TITLES[camera_id])
        title.setFont(font(14, QFont.Weight.Medium))
        live = QLabel("●  LIVE")
        live.setFont(font(12, QFont.Weight.DemiBold))
        live.setStyleSheet(f"color:{GREEN};")
        h.addWidget(chip)
        h.addSpacing(7)
        h.addWidget(title)
        h.addStretch()
        h.addWidget(live)
        layout.addWidget(self.header)

        self.image = CameraImage()
        layout.addWidget(self.image, 1)

        self.footer = QWidget()
        self.footer.setFixedHeight(34)
        f = QHBoxLayout(self.footer)
        f.setContentsMargins(10, 0, 10, 0)
        person_icon = QLabel()
        person_icon.setPixmap(icon_pixmap("users", TEXT, 17))
        self.people = QLabel("0 People")
        self.people.setFont(font(12, QFont.Weight.Medium))
        self.fps = QLabel("-- FPS")
        self.fps.setFont(font(12, QFont.Weight.Medium))
        signal = QLabel("▂▄▆█")
        signal.setFont(font(13, QFont.Weight.Bold))
        signal.setStyleSheet(f"color:{GREEN};")
        f.addWidget(person_icon)
        f.addSpacing(4)
        f.addWidget(self.people)
        f.addStretch()
        f.addWidget(self.fps)
        f.addSpacing(7)
        f.addWidget(signal)
        layout.addWidget(self.footer)

    def set_metrics(self, people: int, fps: float):
        self.people.setText(f"{people} {'Person' if people == 1 else 'People'}")
        self.fps.setText(f"{fps:.0f} FPS" if fps > 0 else "-- FPS")

    def cameras_only(self, enabled: bool):
        self.header.setVisible(not enabled)
        self.footer.setVisible(not enabled)
        self.setStyleSheet("border:0;background:#000;" if enabled else "")


class LivePage(QWidget):
    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        self.title_row = QWidget()
        title_layout = QHBoxLayout(self.title_row)
        title_layout.setContentsMargins(2, 0, 0, 0)
        title = QLabel("Live View")
        title.setFont(font(26, QFont.Weight.DemiBold))
        self.fullscreen = QPushButton()
        self.fullscreen.setObjectName("squareButton")
        self.fullscreen.setIcon(icon("fullscreen", TEXT, 21))
        self.fullscreen.setIconSize(QSize(21,21))
        self.fullscreen.setFixedSize(40, 40)
        title_layout.addWidget(title)
        title_layout.addStretch()
        title_layout.addWidget(self.fullscreen)
        outer.addWidget(self.title_row)

        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(10)
        self.tiles = {}
        for i, camera_id in enumerate(CAMERAS):
            tile = CameraTile(camera_id, i + 1)
            self.tiles[camera_id] = tile
            self.grid.addWidget(tile, i // 2, i % 2)
        for row in range(3):
            self.grid.setRowStretch(row, 1)
        for col in range(2):
            self.grid.setColumnStretch(col, 1)
        outer.addLayout(self.grid, 1)

    def cameras_only(self, enabled: bool):
        self.title_row.setVisible(not enabled)
        self.grid.setHorizontalSpacing(2 if enabled else 10)
        self.grid.setVerticalSpacing(2 if enabled else 10)
        for tile in self.tiles.values():
            tile.cameras_only(enabled)


class NavButton(QPushButton):
    def __init__(self, text: str, kind: str):
        super().__init__(text)
        self.setCheckable(True)
        self.setIcon(icon(kind, TEXT, 23))
        self.setIconSize(QSize(23,23))
        self.setFixedHeight(54)
        self.setFont(font(14, QFont.Weight.Medium))


class Sidebar(QFrame):
    def __init__(self, on_page):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(220)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 14)
        layout.setSpacing(8)

        brand = QHBoxLayout()
        mark = QLabel("◢")
        mark.setFont(font(28, QFont.Weight.Bold))
        mark.setStyleSheet(f"color:{BLUE};")
        name = QLabel("Apsidal")
        name.setFont(font(23, QFont.Weight.DemiBold))
        brand.addWidget(mark)
        brand.addWidget(name)
        brand.addStretch()
        layout.addLayout(brand)
        layout.addSpacing(12)

        self.buttons = {}
        pages = [
            ("Live View", "monitor"),
            ("People", "person"),
            ("Events", "bell"),
            ("Reports", "report"),
            ("Settings", "settings"),
        ]
        for index, (name, kind) in enumerate(pages):
            button = NavButton(name, kind)
            button.clicked.connect(lambda checked=False, i=index: on_page(i))
            self.buttons[index] = button
            layout.addWidget(button)

        layout.addStretch()

        self.status = QFrame()
        self.status.setObjectName("statusCard")
        status_layout = QVBoxLayout(self.status)
        status_layout.setContentsMargins(14, 14, 14, 14)
        top = QHBoxLayout()
        pulse = QLabel()
        pulse.setPixmap(icon_pixmap("activity", GREEN, 25))
        title = QLabel("System Status")
        title.setFont(font(13, QFont.Weight.Medium))
        top.addWidget(pulse)
        top.addWidget(title)
        top.addStretch()
        status_layout.addLayout(top)
        self.status_text = QLabel("Waiting for realtime data")
        self.status_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_text.setWordWrap(True)
        self.status_text.setFont(font(12))
        self.status_text.setStyleSheet(f"color:{MUTED};")
        status_layout.addWidget(self.status_text)
        layout.addWidget(self.status)

    def set_active(self, index: int):
        for i, button in self.buttons.items():
            button.setChecked(i == index)

    def update_realtime(self, state):
        if not state.get("connected"):
            self.status_text.setText("ML service offline")
            self.status_text.setStyleSheet(f"color:{RED};")
            return
        health = state.get("health") or {}
        resources = health.get("service_resources") or {}
        gpu = resources.get("gpu_utilization_percent")
        gpu_text = f" · GPU {gpu}%" if gpu is not None else ""
        self.status_text.setText(
            f"{health.get('online', 0)}/{health.get('total', 6)} cameras online{gpu_text}"
        )
        self.status_text.setStyleSheet(f"color:{GREEN};")


class StatCard(QFrame):
    def __init__(self, title: str, color: str, kind: str):
        super().__init__()
        self.setObjectName("statCard")
        self.setMinimumHeight(105)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 13, 12, 12)
        top = QHBoxLayout()
        ico = QLabel()
        ico.setPixmap(icon_pixmap(kind, color, 30))
        self.value = QLabel("0")
        self.value.setFont(font(25, QFont.Weight.DemiBold))
        top.addWidget(ico)
        top.addSpacing(9)
        top.addWidget(self.value)
        top.addStretch()
        layout.addLayout(top)
        label = QLabel(title)
        label.setFont(font(12))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)


class RecentPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 12)
        header = QHBoxLayout()
        title = QLabel("Recent Views")
        title.setFont(font(16, QFont.Weight.DemiBold))
        view_all = QLabel("View All")
        view_all.setFont(font(12))
        view_all.setStyleSheet(f"color:{CYAN};")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(view_all)
        layout.addLayout(header)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background:{BORDER};")
        layout.addWidget(divider)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 6, 0, 6)
        self.body_layout.setSpacing(8)
        self.scroll.setWidget(self.body)
        layout.addWidget(self.scroll, 1)

    def set_items(self, items):
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if not items:
            empty = QLabel("No recent detections yet")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet(f"color:{MUTED};padding:26px;")
            self.body_layout.addWidget(empty)
            self.body_layout.addStretch()
            return

        for event in items[:8]:
            row = QFrame()
            row.setObjectName("recentRow")
            r = QHBoxLayout(row)
            r.setContentsMargins(8, 8, 8, 8)
            avatar = QLabel()
            avatar.setFixedSize(42,42)
            avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
            avatar.setPixmap(icon_pixmap("person", TEXT, 24))
            avatar.setStyleSheet("background:#213149;border-radius:6px;")
            r.addWidget(avatar)
            text = QVBoxLayout()
            gid = QLabel(str(event.get("gid") or "Unknown"))
            gid.setFont(font(12, QFont.Weight.Medium))
            camera = QLabel(str(event.get("camera") or ""))
            camera.setFont(font(11))
            camera.setStyleSheet(f"color:{MUTED};")
            text.addWidget(gid)
            text.addWidget(camera)
            r.addLayout(text, 1)
            when = QLabel(str(event.get("time") or ""))
            when.setFont(font(11))
            when.setStyleSheet(f"color:{CYAN};")
            r.addWidget(when)
            self.body_layout.addWidget(row)
        self.body_layout.addStretch()


class RightRail(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(330)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        stats = QFrame()
        stats.setObjectName("panel")
        grid = QGridLayout(stats)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setSpacing(10)
        self.total = StatCard("Total People", BLUE, "users")
        self.known = StatCard("Known People", GREEN, "person")
        self.unknown = StatCard("Unknown People", ORANGE, "person")
        self.cameras = StatCard("Active Cameras", CYAN, "camera")
        grid.addWidget(self.total, 0, 0)
        grid.addWidget(self.known, 0, 1)
        grid.addWidget(self.unknown, 1, 0)
        grid.addWidget(self.cameras, 1, 1)
        layout.addWidget(stats)

        self.recent = RecentPanel()
        layout.addWidget(self.recent, 1)

    def update_realtime(self, state, recent):
        reid_state = (((state.get("reid") or {}).get("state") or {}))
        globals_ = reid_state.get("global") or {}
        active = [data for data in globals_.values() if data.get("active_tracks")]
        known = [data for data in active if data.get("name") or data.get("known_name")]
        total = len(active)
        self.total.value.setText(str(total))
        self.known.value.setText(str(len(known)))
        self.unknown.value.setText(str(max(0, total - len(known))))
        health = state.get("health") or {}
        self.cameras.value.setText(f"{health.get('online',0)}/{health.get('total',6)}")
        self.recent.set_items(recent)


def table_item(value):
    item = QTableWidgetItem(str(value))
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


class PeoplePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("People")
        title.setFont(font(26, QFont.Weight.DemiBold))
        subtitle = QLabel("Live identities currently known by the system.")
        subtitle.setStyleSheet(f"color:{MUTED};")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["GLOBAL ID", "CAMERAS", "OBSERVATIONS", "ROOM", "STATUS"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setObjectName("dataTable")
        layout.addWidget(self.table, 1)

    def update_realtime(self, state):
        globals_ = ((((state.get("reid") or {}).get("state") or {}).get("global")) or {})
        rows = []
        for gid, data in globals_.items():
            active_tracks = data.get("active_tracks") or {}
            if not active_tracks:
                continue
            rows.append([
                gid,
                ", ".join(sorted(active_tracks)),
                data.get("observations", 0),
                ", ".join(data.get("active_rooms") or []),
                "Active",
            ])
        self.table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            self.table.setRowHeight(row, 48)
            for column, value in enumerate(values):
                self.table.setItem(row, column, table_item(value))


class EventsPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("Events")
        title.setFont(font(26, QFont.Weight.DemiBold))
        subtitle = QLabel("Realtime detection and identity events.")
        subtitle.setStyleSheet(f"color:{MUTED};")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["TIME", "EVENT", "CAMERA", "IDENTITY", "DETAILS"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setObjectName("dataTable")
        layout.addWidget(self.table, 1)

    def update_realtime(self, events):
        self.table.setRowCount(len(events))
        for row, event in enumerate(events):
            sim = event.get("similarity")
            details = str(event.get("reason") or "detected")
            if isinstance(sim, (int, float)):
                details += f" · {sim:.3f}"
            values = [
                event.get("time", ""),
                "Person detected",
                event.get("camera", ""),
                event.get("gid", ""),
                details,
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, table_item(value))


class EmptyPage(QWidget):
    def __init__(self, title: str):
        super().__init__()
        layout = QVBoxLayout(self)
        heading = QLabel(title)
        heading.setFont(font(26, QFont.Weight.DemiBold))
        layout.addWidget(heading)
        message = QLabel("No realtime module is connected to this page yet.")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setStyleSheet(f"color:{MUTED};")
        layout.addWidget(message, 1)


class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Apsidal")
        self.resize(1672, 941)
        self.setMinimumSize(1180, 720)
        self.camera_only_mode = False

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = Sidebar(self.set_page)
        root_layout.addWidget(self.sidebar)

        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(0,0,0,0)
        body_layout.setSpacing(0)
        root_layout.addWidget(self.body, 1)

        self.topbar = QWidget()
        self.topbar.setObjectName("topbar")
        self.topbar.setFixedHeight(62)
        top = QHBoxLayout(self.topbar)
        top.setContentsMargins(18, 0, 20, 0)
        menu = QPushButton()
        menu.setObjectName("topButton")
        menu.setIcon(icon("menu", TEXT, 22))
        menu.setIconSize(QSize(22,22))
        menu.setFixedSize(38,38)
        menu.clicked.connect(lambda: self.sidebar.setVisible(not self.sidebar.isVisible()))
        top.addWidget(menu)
        top.addStretch()
        self.clock = QLabel()
        self.clock.setFont(font(14, QFont.Weight.Medium))
        self.date = QLabel()
        self.date.setFont(font(13))
        top.addWidget(self.clock)
        top.addSpacing(18)
        divider = QFrame()
        divider.setFixedSize(1,18)
        divider.setStyleSheet(f"background:{BORDER};")
        top.addWidget(divider)
        top.addSpacing(18)
        top.addWidget(self.date)
        body_layout.addWidget(self.topbar)

        self.content = QWidget()
        self.content_layout = QHBoxLayout(self.content)
        self.content_layout.setContentsMargins(16, 10, 16, 14)
        self.content_layout.setSpacing(14)
        body_layout.addWidget(self.content, 1)

        self.stack = QStackedWidget()
        self.live = LivePage()
        self.people = PeoplePage()
        self.events = EventsPage()
        self.reports = EmptyPage("Reports")
        self.settings = EmptyPage("Settings")
        for page in (self.live, self.people, self.events, self.reports, self.settings):
            self.stack.addWidget(page)
        self.content_layout.addWidget(self.stack, 1)

        self.right = RightRail()
        self.content_layout.addWidget(self.right)

        self.live.fullscreen.clicked.connect(self.toggle_camera_fullscreen)

        self.readers = {}
        self.seen_versions = {}
        for camera_id in CAMERAS:
            reader = FrameReader(camera_id)
            reader.start()
            self.readers[camera_id] = reader
            self.seen_versions[camera_id] = -1

        self.realtime = RealtimeState()
        self.realtime.start()

        self.frame_counts = {cid: 0 for cid in CAMERAS}
        self.last_fps_tick = time.monotonic()

        self.render_timer = QTimer(self)
        self.render_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.render_timer.timeout.connect(self.render_frames)
        self.render_timer.start(20)

        self.data_timer = QTimer(self)
        self.data_timer.timeout.connect(self.refresh_data)
        self.data_timer.start(700)

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        self.update_clock()
        self.set_page(0)
        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow,QWidget{{background:{BG};color:{TEXT};}}
            #sidebar{{background:{SIDEBAR_BG};border-right:1px solid #062744;}}
            #topbar{{background:{BG};border-bottom:1px solid #041c34;}}
            QPushButton{{border:0;color:{TEXT};outline:none;}}
            #sidebar QPushButton{{text-align:left;padding-left:16px;border-radius:7px;background:transparent;}}
            #sidebar QPushButton:hover{{background:#06213f;}}
            #sidebar QPushButton:checked{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0b64f5,stop:1 #0646cf);border:1px solid #1474ff;}}
            #statusCard,#panel,#cameraTile{{background:{PANEL};border:1px solid {BORDER};border-radius:8px;}}
            #statCard{{background:{CARD};border:1px solid #052b4c;border-radius:7px;}}
            #recentRow{{background:{CARD};border:1px solid #052a4a;border-radius:6px;}}
            #squareButton,#topButton{{background:{CARD};border:1px solid {BORDER};border-radius:7px;}}
            #squareButton:hover,#topButton:hover{{background:{CARD_2};}}
            QTableWidget#dataTable{{background:{PANEL};border:1px solid {BORDER};border-radius:7px;gridline-color:{BORDER};color:{TEXT};}}
            QTableWidget#dataTable::item{{border-bottom:1px solid #072b49;padding:8px;}}
            QTableWidget#dataTable QHeaderView::section{{background:#00162e;color:#c8d3e1;border:0;border-bottom:1px solid {BORDER};padding:10px;font-size:12px;font-weight:600;}}
            QScrollArea{{background:transparent;border:0;}}
            QScrollArea>QWidget>QWidget{{background:transparent;}}
            QScrollBar:vertical{{background:#001126;width:7px;margin:2px;}}
            QScrollBar::handle:vertical{{background:#174b77;min-height:28px;border-radius:3px;}}
            QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0;}}
        """)

    def set_page(self, index: int):
        self.stack.setCurrentIndex(index)
        self.sidebar.set_active(index)

    def update_clock(self):
        now = datetime.now()
        self.clock.setText(now.strftime("%H:%M:%S"))
        self.date.setText(now.strftime("%-d %b %Y"))

    def render_frames(self):
        for camera_id, reader in self.readers.items():
            image, version = reader.latest()
            if image is not None and version > self.seen_versions[camera_id]:
                self.seen_versions[camera_id] = version
                self.live.tiles[camera_id].image.set_frame(image)

    def refresh_data(self):
        state, recent, events = self.realtime.snapshot()
        detections = ((state.get("detections") or {}).get("cameras") or {})
        now = time.monotonic()
        dt = max(0.1, now - self.last_fps_tick)
        self.last_fps_tick = now

        for camera_id, reader in self.readers.items():
            previous = self.frame_counts[camera_id]
            current = reader.frames
            self.frame_counts[camera_id] = current
            fps = max(0.0, (current - previous) / dt)
            count = len(((detections.get(camera_id) or {}).get("boxes") or []))
            self.live.tiles[camera_id].set_metrics(count, fps)

        self.sidebar.update_realtime(state)
        self.right.update_realtime(state, recent)
        self.people.update_realtime(state)
        self.events.update_realtime(events)

    def toggle_camera_fullscreen(self):
        self.camera_only_mode = not self.camera_only_mode
        self.sidebar.setVisible(not self.camera_only_mode)
        self.topbar.setVisible(not self.camera_only_mode)
        self.right.setVisible(not self.camera_only_mode)
        self.live.cameras_only(self.camera_only_mode)
        self.content_layout.setContentsMargins(0 if self.camera_only_mode else 16, 0 if self.camera_only_mode else 10, 0 if self.camera_only_mode else 16, 0 if self.camera_only_mode else 14)
        self.content_layout.setSpacing(0 if self.camera_only_mode else 14)
        if self.camera_only_mode:
            self.showFullScreen()
        else:
            self.showMaximized()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.camera_only_mode:
            self.toggle_camera_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.render_timer.stop()
        self.data_timer.stop()
        self.clock_timer.stop()
        self.realtime.stop()
        for reader in self.readers.values():
            reader.stop()
        event.accept()


def run():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    window = DashboardWindow()
    window.showMaximized()
    return app.exec()
