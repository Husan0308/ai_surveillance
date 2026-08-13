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
    QApplication, QAbstractItemView, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
)

CAMERAS = [f"CAM-{i:02d}" for i in range(1, 7)]
ML_HOST = "127.0.0.1"
ML_PORT = 8001

BG = "#000a1c"
SIDEBAR = "#000f27"
PANEL = "#001126"
CARD = "#00162f"
BORDER = "#073154"
TEXT = "#f6f8fc"
MUTED = "#a8b5c8"
BLUE = "#0d63ff"
BLUE_2 = "#084bd8"
GREEN = "#00e676"
ORANGE = "#ff8a00"
RED = "#ff334e"
CYAN = "#16b9ff"

CAMERA_TITLES = {
    "CAM-01": "Office 1 (A)",
    "CAM-02": "Office 2 (A)",
    "CAM-03": "Office 3 (A)",
    "CAM-04": "Office 1 (B)",
    "CAM-05": "Office 2 (B)",
    "CAM-06": "Office 3 (B)",
}


def app_font(px: int, weight=QFont.Weight.Normal) -> QFont:
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
        p.drawRoundedRect(QRectF(s*.14, s*.16, s*.72, s*.55), s*.06, s*.06)
        p.drawLine(QPointF(s*.40, s*.78), QPointF(s*.60, s*.78))
        p.drawLine(QPointF(s*.50, s*.71), QPointF(s*.50, s*.78))
    elif kind == "person":
        p.drawEllipse(QRectF(s*.38, s*.14, s*.24, s*.24))
        path = QPainterPath()
        path.moveTo(s*.23, s*.82)
        path.cubicTo(s*.27, s*.56, s*.73, s*.56, s*.77, s*.82)
        p.drawPath(path)
    elif kind == "users":
        p.drawEllipse(QRectF(s*.28, s*.14, s*.22, s*.22))
        p.drawEllipse(QRectF(s*.57, s*.20, s*.18, s*.18))
        p.drawArc(QRectF(s*.17, s*.43, s*.48, s*.40), 15*16, 150*16)
        p.drawArc(QRectF(s*.48, s*.48, s*.38, s*.32), 15*16, 145*16)
    elif kind == "bell":
        path = QPainterPath()
        path.moveTo(s*.28, s*.66)
        path.lineTo(s*.34, s*.58)
        path.lineTo(s*.34, s*.40)
        path.cubicTo(s*.34, s*.16, s*.66, s*.16, s*.66, s*.40)
        path.lineTo(s*.66, s*.58)
        path.lineTo(s*.72, s*.66)
        p.drawPath(path)
        p.drawLine(QPointF(s*.28, s*.66), QPointF(s*.72, s*.66))
    elif kind == "report":
        p.drawRoundedRect(QRectF(s*.18, s*.14, s*.64, s*.70), s*.04, s*.04)
        p.drawLine(QPointF(s*.32, s*.68), QPointF(s*.32, s*.50))
        p.drawLine(QPointF(s*.50, s*.68), QPointF(s*.50, s*.35))
        p.drawLine(QPointF(s*.68, s*.68), QPointF(s*.68, s*.44))
    elif kind == "settings":
        p.drawEllipse(QRectF(s*.38, s*.38, s*.24, s*.24))
        p.drawEllipse(QRectF(s*.27, s*.27, s*.46, s*.46))
        for a, b in [((.5,.12),(.5,.27)),((.5,.73),(.5,.88)),((.12,.5),(.27,.5)),((.73,.5),(.88,.5))]:
            p.drawLine(QPointF(s*a[0], s*a[1]), QPointF(s*b[0], s*b[1]))
    elif kind == "camera":
        p.drawRoundedRect(QRectF(s*.16, s*.30, s*.48, s*.38), s*.04, s*.04)
        path = QPainterPath()
        path.moveTo(s*.64, s*.39)
        path.lineTo(s*.84, s*.30)
        path.lineTo(s*.84, s*.68)
        path.lineTo(s*.64, s*.59)
        path.closeSubpath()
        p.drawPath(path)
    elif kind == "activity":
        path = QPainterPath()
        path.moveTo(s*.08, s*.55)
        path.lineTo(s*.28, s*.55)
        path.lineTo(s*.36, s*.22)
        path.lineTo(s*.47, s*.78)
        path.lineTo(s*.57, s*.42)
        path.lineTo(s*.66, s*.55)
        path.lineTo(s*.92, s*.55)
        p.drawPath(path)
    elif kind == "fullscreen":
        corners = [
            ((.18,.38),(.18,.18),(.38,.18)), ((.62,.18),(.82,.18),(.82,.38)),
            ((.18,.62),(.18,.82),(.38,.82)), ((.62,.82),(.82,.82),(.82,.62)),
        ]
        for a,b,c in corners:
            p.drawLine(QPointF(s*a[0],s*a[1]), QPointF(s*b[0],s*b[1]))
            p.drawLine(QPointF(s*b[0],s*b[1]), QPointF(s*c[0],s*c[1]))
    elif kind == "search":
        p.drawEllipse(QRectF(s*.18,s*.16,s*.46,s*.46))
        p.drawLine(QPointF(s*.57,s*.57), QPointF(s*.82,s*.82))
    else:
        p.drawRoundedRect(QRectF(s*.20,s*.20,s*.60,s*.60), s*.05, s*.05)
    p.end()
    return pm


def make_icon(kind: str, color: str = TEXT, size: int = 24) -> QIcon:
    return QIcon(icon_pixmap(kind, color, size))


class FrameReader:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._image: QImage | None = None
        self._version = -1
        self.frames = 0
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name=f"ui-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def latest(self):
        with self._lock:
            return self._image, self._version

    def _run(self):
        connection = None
        version = -1
        while not self._stop.is_set():
            try:
                if connection is None:
                    connection = http.client.HTTPConnection(ML_HOST, ML_PORT, timeout=2.0)
                connection.request(
                    "GET",
                    f"/frame/{self.camera_id}?after={version}&wait_ms=180",
                    headers={"Connection":"keep-alive","Cache-Control":"no-cache"},
                )
                response = connection.getresponse()
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
                with self._lock:
                    self._image = image
                    self._version = version
                    self.frames += 1
            except Exception:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
                connection = None
                self._stop.wait(0.08)


class RealtimeState:
    def __init__(self):
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self.state = {"connected": False, "health": {}, "detections": {}, "reid": {}}
        self.recent = deque(maxlen=30)
        self.events = deque(maxlen=100)
        self._seen = {}

    def start(self):
        self._thread = threading.Thread(target=self._run, name="ui-state", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def snapshot(self):
        with self._lock:
            return dict(self.state), list(self.recent), list(self.events)

    @staticmethod
    def _json(connection, path):
        connection.request("GET", path, headers={"Connection":"keep-alive","Cache-Control":"no-cache"})
        response = connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(response.status)
        return json.loads(payload.decode("utf-8"))

    def _observe(self, reid_payload):
        cameras = ((reid_payload.get("state") or {}).get("cameras") or {})
        now = datetime.now()
        for camera_id, tracks in cameras.items():
            for track in tracks or []:
                gid = str(track.get("global_id") or "")
                if not gid:
                    continue
                local_id = int(track.get("local_id") or 0)
                key = (camera_id, local_id)
                old = self._seen.get(key)
                self._seen[key] = gid
                if old == gid:
                    continue
                entry = {
                    "time": now.strftime("%H:%M:%S"),
                    "camera": camera_id,
                    "global_id": gid,
                    "similarity": track.get("similarity"),
                    "reason": str(track.get("reason") or "detected"),
                }
                self.recent.appendleft(entry)
                self.events.appendleft(entry)

    def _run(self):
        connection = None
        while not self._stop.is_set():
            try:
                if connection is None:
                    connection = http.client.HTTPConnection(ML_HOST, ML_PORT, timeout=1.5)
                health = self._json(connection, "/health")
                detections = self._json(connection, "/detections")
                reid = self._json(connection, "/reid")
                self._observe(reid)
                with self._lock:
                    self.state = {"connected": True, "health": health, "detections": detections, "reid": reid}
                self._stop.wait(0.35)
            except Exception:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
                connection = None
                with self._lock:
                    self.state = {**self.state, "connected": False}
                self._stop.wait(0.6)


class CameraViewport(QWidget):
    """Fill the tile without stretching the camera image.

    The source aspect ratio is preserved. Any mismatch is handled by a minimal
    center crop, so there are no black side bars and no geometric distortion.
    """
    def __init__(self):
        super().__init__()
        self._image: QImage | None = None
        self.setMinimumSize(180, 100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def set_frame(self, image: QImage):
        self._image = image
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#020913"))
        if self._image is None or self._image.isNull():
            painter.setPen(QColor(MUTED))
            painter.setFont(app_font(13))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Connecting...")
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        iw, ih = self._image.width(), self._image.height()
        tw, th = max(1, self.width()), max(1, self.height())
        scale = max(tw / iw, th / ih)
        dw, dh = iw * scale, ih * scale
        x = (tw - dw) * 0.5
        y = (th - dh) * 0.5
        painter.drawImage(QRectF(x, y, dw, dh), self._image)


class CameraTile(QFrame):
    def __init__(self, camera_id: str, number: int):
        super().__init__()
        self.camera_id = camera_id
        self.setObjectName("cameraTile")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root = QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        self.header = QWidget()
        self.header.setObjectName("cameraHeader")
        self.header.setFixedHeight(46)
        h = QHBoxLayout(self.header)
        h.setContentsMargins(9,0,10,0)
        h.setSpacing(9)
        chip = QLabel(f"{number:02d}")
        chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chip.setFixedSize(35,32)
        chip.setFont(app_font(15, QFont.Weight.DemiBold))
        chip.setStyleSheet(f"background:{BLUE_2};color:white;border-radius:6px;")
        title = QLabel(CAMERA_TITLES[camera_id])
        title.setFont(app_font(16, QFont.Weight.Medium))
        dot = QLabel("●")
        dot.setStyleSheet(f"color:{GREEN};")
        live = QLabel("LIVE")
        live.setFont(app_font(14, QFont.Weight.Medium))
        h.addWidget(chip)
        h.addWidget(title)
        h.addStretch()
        h.addWidget(dot)
        h.addWidget(live)
        root.addWidget(self.header)

        self.viewport = CameraViewport()
        root.addWidget(self.viewport, 1)

        self.footer = QWidget()
        self.footer.setObjectName("cameraFooter")
        self.footer.setFixedHeight(36)
        f = QHBoxLayout(self.footer)
        f.setContentsMargins(10,0,10,0)
        self.people = QLabel("0 People")
        self.people.setFont(app_font(13))
        self.fps = QLabel("-- FPS")
        self.fps.setFont(app_font(13))
        bars = QLabel("▂▄▆█")
        bars.setFont(app_font(13, QFont.Weight.Bold))
        bars.setStyleSheet(f"color:{GREEN};")
        f.addWidget(self.people)
        f.addStretch()
        f.addWidget(self.fps)
        f.addWidget(bars)
        root.addWidget(self.footer)

    def update_metrics(self, count: int, fps: float):
        self.people.setText(f"{count} {'Person' if count == 1 else 'People'}")
        self.fps.setText(f"{fps:.0f} FPS" if fps > 0 else "-- FPS")

    def cameras_only(self, enabled: bool):
        self.header.setVisible(not enabled)
        self.footer.setVisible(not enabled)


class NavButton(QPushButton):
    def __init__(self, text: str, icon_name: str):
        super().__init__(text)
        self.setCheckable(True)
        self.setFixedHeight(60)
        self.setIcon(make_icon(icon_name, TEXT, 26))
        self.setIconSize(QSize(26,26))
        self.setFont(app_font(16, QFont.Weight.Medium))
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class Sidebar(QFrame):
    def __init__(self, change_page):
        super().__init__()
        self.setObjectName("sidebar")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14,16,14,16)
        self._layout.setSpacing(8)

        brand = QWidget()
        b = QHBoxLayout(brand)
        b.setContentsMargins(12,0,0,8)
        mark = QLabel("◢")
        mark.setFont(app_font(32, QFont.Weight.Bold))
        mark.setStyleSheet(f"color:{BLUE};")
        name = QLabel("Apsidal")
        name.setFont(app_font(28, QFont.Weight.DemiBold))
        b.addWidget(mark)
        b.addWidget(name)
        b.addStretch()
        self._layout.addWidget(brand)

        self.buttons = {}
        for index, (label, icon_name) in enumerate([
            ("Live View","monitor"), ("People","person"), ("Events","bell"),
            ("Reports","report"), ("Settings","settings")
        ]):
            button = NavButton(label, icon_name)
            button.clicked.connect(lambda checked=False, i=index: change_page(i))
            self._layout.addWidget(button)
            self.buttons[index] = button

        self._layout.addStretch()
        self.status_card = QFrame()
        self.status_card.setObjectName("statusCard")
        s = QVBoxLayout(self.status_card)
        s.setContentsMargins(16,15,16,15)
        top = QHBoxLayout()
        pulse = QLabel()
        pulse.setPixmap(icon_pixmap("activity", GREEN, 27))
        title = QLabel("System Status")
        title.setFont(app_font(13, QFont.Weight.Medium))
        top.addWidget(pulse)
        top.addWidget(title)
        top.addStretch()
        s.addLayout(top)
        self.system_text = QLabel("Waiting for realtime data")
        self.system_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.system_text.setFont(app_font(12))
        self.system_text.setStyleSheet(f"color:{MUTED};")
        s.addWidget(self.system_text)
        self.cpu = self._resource_row(s, "CPU")
        self.gpu = self._resource_row(s, "GPU")
        self.mem = self._resource_row(s, "Memory")
        self._layout.addWidget(self.status_card)

    def _resource_row(self, layout, label):
        row = QHBoxLayout()
        name = QLabel(label)
        value = QLabel("—")
        name.setFont(app_font(12))
        value.setFont(app_font(12))
        row.addWidget(name)
        row.addStretch()
        row.addWidget(value)
        layout.addLayout(row)
        bar = QProgressBar()
        bar.setRange(0,100)
        bar.setTextVisible(False)
        bar.setFixedHeight(8)
        layout.addWidget(bar)
        return value, bar

    def set_active(self, index: int):
        for key, button in self.buttons.items():
            button.setChecked(key == index)

    def update_live(self, state):
        health = state.get("health") or {}
        res = health.get("service_resources") or {}
        online = int(health.get("online") or 0)
        total = int(health.get("total") or 6)
        connected = bool(state.get("connected"))
        if connected:
            self.system_text.setText(f"{online}/{total} cameras online")
            self.system_text.setStyleSheet(f"color:{GREEN};")
        else:
            self.system_text.setText("ML service offline")
            self.system_text.setStyleSheet(f"color:{RED};")
        cpu = float(res.get("cpu_percent") or 0)
        gpu = float(res.get("gpu_utilization_percent") or 0)
        rss = float(res.get("rss_mb") or 0)
        self.cpu[0].setText(f"{cpu:.0f}%")
        self.cpu[1].setValue(max(0,min(100,int(cpu))))
        self.gpu[0].setText(f"{gpu:.0f}%")
        self.gpu[1].setValue(max(0,min(100,int(gpu))))
        self.mem[0].setText(f"{rss/1024:.1f} GB" if rss >= 1024 else f"{rss:.0f} MB")
        self.mem[1].setValue(max(0,min(100,int(rss / 8192 * 100))))


class StatCard(QFrame):
    def __init__(self, icon_name: str, icon_color: str, label: str):
        super().__init__()
        self.setObjectName("statCard")
        l = QVBoxLayout(self)
        l.setContentsMargins(17,17,13,13)
        top = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(icon_pixmap(icon_name, icon_color, 36))
        self.value = QLabel("0")
        self.value.setFont(app_font(28, QFont.Weight.DemiBold))
        top.addWidget(icon)
        top.addSpacing(8)
        top.addWidget(self.value)
        top.addStretch()
        l.addLayout(top)
        text = QLabel(label)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text.setFont(app_font(13))
        l.addWidget(text)


class RightRail(QWidget):
    def __init__(self):
        super().__init__()
        l = QVBoxLayout(self)
        l.setContentsMargins(0,0,0,0)
        l.setSpacing(14)
        panel = QFrame()
        panel.setObjectName("rightPanel")
        grid = QGridLayout(panel)
        grid.setContentsMargins(12,12,12,12)
        grid.setSpacing(12)
        self.total = StatCard("users", BLUE, "Total People")
        self.known = StatCard("person", GREEN, "Known People")
        self.unknown = StatCard("person", ORANGE, "Unknown People")
        self.cameras = StatCard("camera", CYAN, "Active Cameras")
        grid.addWidget(self.total,0,0)
        grid.addWidget(self.known,0,1)
        grid.addWidget(self.unknown,1,0)
        grid.addWidget(self.cameras,1,1)
        l.addWidget(panel)

        recent = QFrame()
        recent.setObjectName("rightPanel")
        recent_layout = QVBoxLayout(recent)
        recent_layout.setContentsMargins(16,15,16,12)
        head = QHBoxLayout()
        title = QLabel("Recent Views")
        title.setFont(app_font(17, QFont.Weight.DemiBold))
        view = QLabel("View All")
        view.setFont(app_font(13))
        view.setStyleSheet(f"color:{CYAN};")
        head.addWidget(title)
        head.addStretch()
        head.addWidget(view)
        recent_layout.addLayout(head)
        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.recent_body = QWidget()
        self.recent_layout = QVBoxLayout(self.recent_body)
        self.recent_layout.setContentsMargins(0,7,0,7)
        self.recent_layout.setSpacing(8)
        self.scroll.setWidget(self.recent_body)
        recent_layout.addWidget(self.scroll,1)
        l.addWidget(recent,1)

    def update_live(self, state, recent):
        global_state = (((state.get("reid") or {}).get("state") or {}).get("global") or {})
        active = [(gid, value) for gid, value in global_state.items() if value.get("active_tracks")]
        known = sum(1 for _, value in active if value.get("name") or value.get("known_name") or value.get("person_id"))
        total = len(active)
        self.total.value.setText(str(total))
        self.known.value.setText(str(known))
        self.unknown.value.setText(str(max(0,total-known)))
        health = state.get("health") or {}
        self.cameras.value.setText(f"{health.get('online',0)}/{health.get('total',6)}")

        while self.recent_layout.count():
            item = self.recent_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        if not recent:
            empty = QLabel("No recent detections yet")
            empty.setStyleSheet(f"color:{MUTED};")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setMinimumHeight(70)
            self.recent_layout.addWidget(empty)
        else:
            for entry in recent[:10]:
                row = QWidget()
                r = QHBoxLayout(row)
                r.setContentsMargins(0,2,0,2)
                avatar = QLabel()
                avatar.setFixedSize(48,48)
                avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
                avatar.setPixmap(icon_pixmap("person", MUTED, 25))
                avatar.setStyleSheet("background:#243345;border-radius:6px;")
                texts = QVBoxLayout()
                gid = QLabel(entry["global_id"])
                gid.setFont(app_font(13, QFont.Weight.Medium))
                cam = QLabel(f"{entry['camera']} · {entry['time']}")
                cam.setFont(app_font(12))
                cam.setStyleSheet(f"color:{MUTED};")
                texts.addWidget(gid)
                texts.addWidget(cam)
                r.addWidget(avatar)
                r.addLayout(texts,1)
                self.recent_layout.addWidget(row)
        self.recent_layout.addStretch()


class LivePage(QWidget):
    def __init__(self):
        super().__init__()
        l = QVBoxLayout(self)
        l.setContentsMargins(0,0,0,0)
        l.setSpacing(12)
        self.title_row = QWidget()
        title_layout = QHBoxLayout(self.title_row)
        title_layout.setContentsMargins(4,0,0,0)
        title = QLabel("Live View")
        title.setFont(app_font(27, QFont.Weight.DemiBold))
        self.full = QPushButton()
        self.full.setObjectName("squareButton")
        self.full.setIcon(make_icon("fullscreen", TEXT, 23))
        self.full.setIconSize(QSize(23,23))
        self.full.setFixedSize(42,42)
        title_layout.addWidget(title)
        title_layout.addStretch()
        title_layout.addWidget(self.full)
        l.addWidget(self.title_row)
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0,0,0,0)
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(10)
        self.tiles = {}
        for index, camera_id in enumerate(CAMERAS):
            tile = CameraTile(camera_id, index+1)
            self.tiles[camera_id] = tile
            self.grid.addWidget(tile, index//2, index%2)
        for row in range(3):
            self.grid.setRowStretch(row,1)
        for col in range(2):
            self.grid.setColumnStretch(col,1)
        l.addLayout(self.grid,1)

    def cameras_only(self, enabled: bool):
        self.title_row.setVisible(not enabled)
        self.grid.setHorizontalSpacing(2 if enabled else 12)
        self.grid.setVerticalSpacing(2 if enabled else 10)
        for tile in self.tiles.values():
            tile.cameras_only(enabled)


def table_item(text):
    item = QTableWidgetItem(str(text))
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


class PeoplePage(QWidget):
    def __init__(self):
        super().__init__()
        l = QVBoxLayout(self)
        l.setContentsMargins(0,0,0,0)
        header = QLabel("People")
        header.setFont(app_font(27, QFont.Weight.DemiBold))
        subtitle = QLabel("Manage and monitor detected people in real time.")
        subtitle.setStyleSheet(f"color:{MUTED};")
        l.addWidget(header)
        l.addWidget(subtitle)
        self.table = QTableWidget(0,5)
        self.table.setHorizontalHeaderLabels(["GLOBAL ID","CAMERAS","OBSERVATIONS","ROOM","STATUS"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setObjectName("dataTable")
        l.addWidget(self.table,1)

    def update_live(self, state):
        global_state = (((state.get("reid") or {}).get("state") or {}).get("global") or {})
        rows = []
        for gid, value in global_state.items():
            active_tracks = value.get("active_tracks") or {}
            if not active_tracks:
                continue
            rows.append((gid, ", ".join(sorted(active_tracks.keys())), value.get("observations",0), ", ".join(value.get("active_rooms") or []), "Active"))
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self.table.setRowHeight(r,48)
            for c, value in enumerate(row):
                self.table.setItem(r,c,table_item(value))


class EventsPage(QWidget):
    def __init__(self):
        super().__init__()
        l = QVBoxLayout(self)
        l.setContentsMargins(0,0,0,0)
        header = QLabel("Events")
        header.setFont(app_font(27, QFont.Weight.DemiBold))
        subtitle = QLabel("Realtime identity events generated by the current pipeline.")
        subtitle.setStyleSheet(f"color:{MUTED};")
        l.addWidget(header)
        l.addWidget(subtitle)
        self.table = QTableWidget(0,5)
        self.table.setHorizontalHeaderLabels(["TIME","EVENT","CAMERA","IDENTITY","DETAILS"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setObjectName("dataTable")
        l.addWidget(self.table,1)

    def update_live(self, events):
        self.table.setRowCount(len(events))
        for r, entry in enumerate(events):
            sim = entry.get("similarity")
            detail = entry.get("reason","detected")
            if isinstance(sim,(float,int)):
                detail += f" · {sim:.3f}"
            row = [entry["time"],"Person detected",entry["camera"],entry["global_id"],detail]
            self.table.setRowHeight(r,48)
            for c, value in enumerate(row):
                self.table.setItem(r,c,table_item(value))


class PlaceholderPage(QWidget):
    def __init__(self, title):
        super().__init__()
        l = QVBoxLayout(self)
        l.setContentsMargins(0,0,0,0)
        h = QLabel(title)
        h.setFont(app_font(27,QFont.Weight.DemiBold))
        l.addWidget(h)
        card = QFrame()
        card.setObjectName("placeholder")
        c = QVBoxLayout(card)
        text = QLabel("This page has no realtime backend source yet.")
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text.setStyleSheet(f"color:{MUTED};")
        c.addWidget(text)
        l.addWidget(card,1)


class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Apsidal")
        self.resize(1648,960)
        self.setMinimumSize(1180,720)
        self._camera_fullscreen = False
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0,0,0,0)
        outer.setSpacing(0)
        self.sidebar = Sidebar(self.set_page)
        outer.addWidget(self.sidebar)
        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(0,0,0,0)
        body_layout.setSpacing(0)
        outer.addWidget(self.body,1)

        self.topbar = QWidget()
        self.topbar.setObjectName("topbar")
        self.topbar.setFixedHeight(70)
        top = QHBoxLayout(self.topbar)
        top.setContentsMargins(20,0,24,0)
        menu = QPushButton()
        menu.setObjectName("topButton")
        menu.setIcon(make_icon("menu",TEXT,27))
        menu.setIconSize(QSize(27,27))
        menu.setFixedSize(42,42)
        menu.clicked.connect(lambda: self.sidebar.setVisible(not self.sidebar.isVisible()))
        self.clock = QLabel()
        self.clock.setFont(app_font(15,QFont.Weight.Medium))
        self.date = QLabel()
        self.date.setFont(app_font(14))
        top.addWidget(menu)
        top.addStretch()
        top.addWidget(self.clock)
        sep = QFrame()
        sep.setFixedSize(1,20)
        sep.setStyleSheet(f"background:{BORDER};")
        top.addSpacing(12)
        top.addWidget(sep)
        top.addSpacing(12)
        top.addWidget(self.date)
        body_layout.addWidget(self.topbar)

        self.content = QWidget()
        self.content_layout = QHBoxLayout(self.content)
        self.content_layout.setContentsMargins(14,10,14,16)
        self.content_layout.setSpacing(14)
        body_layout.addWidget(self.content,1)
        self.stack = QStackedWidget()
        self.live = LivePage()
        self.people_page = PeoplePage()
        self.events_page = EventsPage()
        self.reports_page = PlaceholderPage("Reports")
        self.settings_page = PlaceholderPage("Settings")
        for page in [self.live,self.people_page,self.events_page,self.reports_page,self.settings_page]:
            self.stack.addWidget(page)
        self.content_layout.addWidget(self.stack,1)
        self.right = RightRail()
        self.content_layout.addWidget(self.right)

        self.readers = {}
        self.seen_versions = {}
        for camera_id in CAMERAS:
            reader = FrameReader(camera_id)
            reader.start()
            self.readers[camera_id] = reader
            self.seen_versions[camera_id] = -1
        self.state_reader = RealtimeState()
        self.state_reader.start()
        self._last_counts = {cid:0 for cid in CAMERAS}
        self._last_fps_time = time.monotonic()
        self.live.full.clicked.connect(self.toggle_cameras_fullscreen)

        self.render_timer = QTimer(self)
        self.render_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.render_timer.timeout.connect(self.render_frames)
        self.render_timer.start(20)
        self.state_timer = QTimer(self)
        self.state_timer.timeout.connect(self.refresh_state)
        self.state_timer.start(500)
        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self.refresh_fps)
        self.fps_timer.start(1000)
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.refresh_clock)
        self.clock_timer.start(1000)
        self.refresh_clock()
        self.apply_style()
        self.set_page(0)
        self._apply_responsive_widths()

    def apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{BG}; color:{TEXT}; }}
            #sidebar {{ background:{SIDEBAR}; border-right:1px solid #07243f; }}
            #topbar {{ background:{BG}; }}
            QPushButton {{ border:0; color:{TEXT}; outline:none; background:transparent; }}
            #sidebar QPushButton {{ text-align:left; padding-left:18px; border-radius:7px; }}
            #sidebar QPushButton:hover {{ background:#061d3a; }}
            #sidebar QPushButton:checked {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #075ff0,stop:1 #073bc8);
                border:1px solid #1470ff;
            }}
            #statusCard, #rightPanel, #cameraTile, #placeholder {{
                background:{PANEL}; border:1px solid {BORDER}; border-radius:9px;
            }}
            #cameraHeader, #cameraFooter {{ background:{CARD}; }}
            #statCard {{ background:{CARD}; border-radius:8px; min-height:104px; }}
            #squareButton, #topButton {{ background:{CARD}; border:1px solid {BORDER}; border-radius:7px; }}
            #squareButton:hover, #topButton:hover {{ background:#08294c; }}
            QProgressBar {{ background:#06325b; border:0; border-radius:4px; }}
            QProgressBar::chunk {{ background:{BLUE}; border-radius:4px; }}
            QTableWidget#dataTable {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:8px; color:{TEXT}; }}
            QTableWidget#dataTable::item {{ border-bottom:1px solid #082b49; padding:8px; }}
            QTableWidget#dataTable QHeaderView::section {{
                background:#00142d; color:#c7d2e2; border:0; border-bottom:1px solid {BORDER};
                padding:10px; font-size:12px; font-weight:600;
            }}
            QScrollArea {{ background:transparent; border:0; }}
            QScrollArea > QWidget > QWidget {{ background:transparent; }}
            QScrollBar:vertical {{ background:#001126; width:7px; margin:2px; }}
            QScrollBar::handle:vertical {{ background:#164770; min-height:28px; border-radius:3px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
        """)

    def _apply_responsive_widths(self):
        width = max(1,self.width())
        if width >= 1550:
            self.sidebar.setFixedWidth(235)
            self.right.setFixedWidth(350)
        elif width >= 1300:
            self.sidebar.setFixedWidth(205)
            self.right.setFixedWidth(300)
        else:
            self.sidebar.setFixedWidth(185)
            self.right.setFixedWidth(260)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._camera_fullscreen:
            self._apply_responsive_widths()

    def set_page(self, index: int):
        self.stack.setCurrentIndex(index)
        self.sidebar.set_active(index)

    def render_frames(self):
        for camera_id, reader in self.readers.items():
            image, version = reader.latest()
            if image is not None and version > self.seen_versions[camera_id]:
                self.seen_versions[camera_id] = version
                self.live.tiles[camera_id].viewport.set_frame(image)

    def refresh_fps(self):
        now = time.monotonic()
        dt = max(0.1, now - self._last_fps_time)
        self._last_fps_time = now
        state, _, _ = self.state_reader.snapshot()
        det_cams = ((state.get("detections") or {}).get("cameras") or {})
        for camera_id, reader in self.readers.items():
            current = reader.frames
            fps = (current - self._last_counts[camera_id]) / dt
            self._last_counts[camera_id] = current
            count = len((det_cams.get(camera_id) or {}).get("boxes") or [])
            self.live.tiles[camera_id].update_metrics(count, fps)

    def refresh_state(self):
        state, recent, events = self.state_reader.snapshot()
        self.sidebar.update_live(state)
        self.right.update_live(state,recent)
        self.people_page.update_live(state)
        self.events_page.update_live(events)

    def refresh_clock(self):
        now = datetime.now()
        self.clock.setText(now.strftime("%H:%M:%S"))
        self.date.setText(now.strftime("%-d %b %Y"))

    def toggle_cameras_fullscreen(self):
        self._camera_fullscreen = not self._camera_fullscreen
        enabled = self._camera_fullscreen
        self.live.cameras_only(enabled)
        self.sidebar.setVisible(not enabled)
        self.topbar.setVisible(not enabled)
        self.right.setVisible(not enabled)
        self.content_layout.setContentsMargins(0 if enabled else 14, 0 if enabled else 10, 0 if enabled else 14, 0 if enabled else 16)
        self.content_layout.setSpacing(0 if enabled else 14)
        if enabled:
            self.showFullScreen()
        else:
            self.showMaximized()
            self._apply_responsive_widths()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self._camera_fullscreen:
            self.toggle_cameras_fullscreen()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.render_timer.stop()
        self.state_timer.stop()
        self.fps_timer.stop()
        self.clock_timer.stop()
        self.state_reader.stop()
        for reader in self.readers.values():
            reader.stop()
        event.accept()


def run():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    window = DashboardWindow()
    window.showMaximized()
    return app.exec()
