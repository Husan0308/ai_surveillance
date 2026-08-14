from __future__ import annotations

from collections import deque
from datetime import datetime
import http.client
import json
import os
from pathlib import Path
import threading

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QKeySequence, QPainter, QPainterPath, QPalette, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .dashboard import ML_HOST, ML_PORT
from .smooth_frame_reader import SmoothFrameReader


class TH:
    BG = "#0f1317"
    PANEL = "#151b22"
    CARD = "#1a212a"
    CARD2 = "#212a35"
    HOVER = "#26303c"
    BORDER = "#2b3542"
    ACCENT = "#2f7df6"
    ACC2 = "#5b9bff"
    OK = "#2ecc71"
    WARN = "#f5c542"
    ERR = "#ef5350"
    TXT = "#e9eef5"
    DIM = "#94a1b3"
    FAINT = "#5d6b7e"


CAMERA_SPECS = [
    ("CAM-01", "Camera 1", "Channel 101"),
    ("CAM-02", "Camera 2", "Channel 201"),
    ("CAM-03", "Camera 3", "Channel 301"),
    ("CAM-04", "Camera 4", "Channel 401"),
    ("CAM-05", "Camera 5", "Channel 501"),
    ("CAM-06", "Camera 6", "Channel 601"),
]


class BackendState:
    """Small read-only adapter between Qt and the current Core v1 API.

    This UI never owns detection/tracking/ReID. It only renders `/health` and
    `/tracks`, so replacing the frontend cannot destabilize the ML hot path.
    """

    def __init__(self):
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._online: dict[str, bool] = {}
        self.events = deque(maxlen=120)
        self.state = {
            "connected": False,
            "health": {},
            "tracks": {},
        }

    @staticmethod
    def _get_json(connection: http.client.HTTPConnection, path: str):
        connection.request(
            "GET",
            path,
            headers={"Connection": "keep-alive", "Cache-Control": "no-cache"},
        )
        response = connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(f"GET {path}: HTTP {response.status}")
        return json.loads(payload.decode("utf-8") or "{}")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="ui-mukammal-state",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)

    def snapshot(self):
        with self._lock:
            return dict(self.state), list(self.events)

    def _camera_edges(self, health: dict):
        now = datetime.now().strftime("%H:%M:%S")
        for camera_id, metrics in (health.get("cameras") or {}).items():
            online = bool(metrics.get("online"))
            previous = self._online.get(camera_id)
            self._online[camera_id] = online
            if previous is None or previous == online:
                continue
            self.events.appendleft(
                {
                    "time": now,
                    "camera": camera_id,
                    "text": "Camera back online" if online else "Camera offline",
                    "level": "ok" if online else "err",
                }
            )

    def _run(self):
        connection = None
        while not self._stop.is_set():
            try:
                if connection is None:
                    connection = http.client.HTTPConnection(ML_HOST, ML_PORT, timeout=2.5)
                health = self._get_json(connection, "/health")
                tracks = self._get_json(connection, "/tracks")
                with self._lock:
                    self._camera_edges(health)
                    self.state = {
                        "connected": True,
                        "health": health,
                        "tracks": tracks,
                    }
            except Exception:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
                connection = None
                with self._lock:
                    self.state = {**self.state, "connected": False}
                self._stop.wait(0.20)
                continue
            self._stop.wait(0.30)


class CameraFeed:
    """One persistent MJPEG connection shared by every view of one camera."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.reader = SmoothFrameReader(camera_id)
        self.surfaces: set[QWidget] = set()

    def start(self):
        self.reader.start()

    def stop(self):
        self.reader.stop()

    def latest(self):
        return self.reader.latest()

    def update_surfaces(self):
        for surface in list(self.surfaces):
            try:
                surface.update()
            except RuntimeError:
                self.surfaces.discard(surface)


class Chip(QFrame):
    def __init__(self, label: str):
        super().__init__()
        self.setObjectName("chip")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(9, 4, 9, 4)
        layout.setSpacing(6)
        title = QLabel(label)
        title.setStyleSheet(f"color:{TH.DIM};font-size:9px;font-weight:700;")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setFixedSize(36, 4)
        self.bar.setTextVisible(False)
        self.value = QLabel("--")
        self.value.setStyleSheet("font-size:10px;font-weight:700;")
        layout.addWidget(title)
        layout.addWidget(self.bar)
        layout.addWidget(self.value)

    def set_value(self, value: float):
        value = max(0.0, min(100.0, float(value or 0.0)))
        self.value.setText(f"{int(round(value))}%")
        self.bar.setValue(int(round(value)))
        color = TH.ERR if value > 85 else (TH.WARN if value > 65 else TH.OK)
        self.bar.setStyleSheet(
            f"QProgressBar{{background:#232c37;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:{color};border-radius:2px;}}"
        )


class Header(QFrame):
    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        self.setObjectName("header")
        self.setFixedHeight(64)
        h = QHBoxLayout(self)
        h.setContentsMargins(14, 0, 14, 0)
        h.setSpacing(10)

        logo = QLabel("◉")
        logo.setFixedSize(34, 34)
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(
            f"background:{TH.ACCENT};border-radius:9px;color:white;"
            "font-size:16px;font-weight:800;"
        )
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("AI Surveillance System")
        title.setStyleSheet("font-size:13.5px;font-weight:800;color:white;")
        sub = QLabel("Operator Console • MUKAMMAL UI")
        sub.setStyleSheet(f"font-size:9px;color:{TH.DIM};")
        title_box.addWidget(title)
        title_box.addWidget(sub)
        h.addWidget(logo)
        h.addLayout(title_box)
        h.addStretch(1)

        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍  Search cameras, events…   ( / )")
        self.search.setMaximumWidth(320)
        self.search.setMinimumWidth(170)
        self.search.returnPressed.connect(lambda: hub.navigate("events"))
        h.addWidget(self.search)
        h.addStretch(1)

        self.gpu = Chip("GPU")
        self.cpu = Chip("CPU")
        self.ram = Chip("RAM")
        for chip in (self.gpu, self.cpu, self.ram):
            h.addWidget(chip)

        self.ai = QLabel("● AI OFFLINE")
        self.ai.setStyleSheet(
            f"color:{TH.WARN};font-size:10px;font-weight:800;"
            "background:#24210f;border:1px solid #443d1a;"
            "border-radius:10px;padding:4px 10px;"
        )
        h.addWidget(self.ai)

        self.cams = QLabel("🎥 0/6")
        self.cams.setStyleSheet(
            f"color:{TH.TXT};font-size:10px;font-weight:700;"
            f"background:{TH.CARD2};border:1px solid {TH.BORDER};"
            "border-radius:10px;padding:4px 10px;"
        )
        h.addWidget(self.cams)

        self.clock = QLabel()
        self.clock.setStyleSheet(
            f"color:{TH.TXT};font-size:10px;font-family:Consolas,monospace;"
        )
        h.addWidget(self.clock)

        avatar = QLabel("OP")
        avatar.setFixedSize(30, 30)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet(
            f"background:{TH.ACCENT};color:white;border-radius:15px;"
            "font-size:10px;font-weight:800;"
        )
        h.addWidget(avatar)
        operator = QLabel("Operator")
        operator.setStyleSheet(f"font-size:10.5px;color:{TH.DIM};font-weight:600;")
        h.addWidget(operator)
        self.tick_clock()

    def tick_clock(self):
        self.clock.setText(datetime.now().strftime("%d %b %Y   %H:%M:%S"))

    def refresh(self, state: dict):
        health = state.get("health") or {}
        resources = health.get("service_resources") or {}
        self.gpu.set_value(float(resources.get("gpu_utilization_percent") or 0.0))
        self.cpu.set_value(float(resources.get("cpu_percent") or 0.0))
        ram = 0.0
        try:
            import psutil

            ram = float(psutil.virtual_memory().percent)
        except Exception:
            rss = float(resources.get("rss_mb") or 0.0)
            ram = min(100.0, rss / 8192.0 * 100.0)
        self.ram.set_value(ram)

        detector = health.get("detector") or {}
        ready = bool(detector.get("ready"))
        connected = bool(state.get("connected"))
        if ready and connected:
            self.ai.setText("● AI ACTIVE")
            self.ai.setStyleSheet(
                f"color:{TH.OK};font-size:10px;font-weight:800;"
                "background:#16241c;border:1px solid #234433;"
                "border-radius:10px;padding:4px 10px;"
            )
        else:
            self.ai.setText("● AI OFFLINE")
            self.ai.setStyleSheet(
                f"color:{TH.WARN};font-size:10px;font-weight:800;"
                "background:#24210f;border:1px solid #443d1a;"
                "border-radius:10px;padding:4px 10px;"
            )
        online = int(health.get("online") or 0)
        total = int(health.get("total") or 6)
        self.cams.setText(f"🎥 {online}/{total}")
        self.cams.setStyleSheet(
            f"color:{TH.TXT if online == total and total else TH.WARN};"
            "font-size:10px;font-weight:700;"
            f"background:{TH.CARD2};border:1px solid {TH.BORDER};"
            "border-radius:10px;padding:4px 10px;"
        )


class SideBar(QFrame):
    changed = Signal(str)
    ITEMS = [
        ("dashboard", "📊", "Dashboard"),
        ("live", "🎥", "Live Cameras"),
        ("people", "👥", "Person Management"),
        ("enroll", "🪪", "Enrollment"),
        ("analytics", "📈", "Analytics"),
        ("events", "⚡", "Events"),
        ("settings", "⚙️", "Settings"),
    ]

    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        self.setObjectName("sidebar")
        self.setMinimumWidth(210)
        self.setMaximumWidth(210)
        self.collapsed = False
        self.buttons = {}
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 14, 10, 10)
        v.setSpacing(4)
        label = QLabel("CONTROL PANEL")
        label.setStyleSheet(
            f"color:{TH.FAINT};font-size:8.5px;font-weight:800;"
            "letter-spacing:2px;padding:0 0 6px 10px;"
        )
        v.addWidget(label)
        for key, icon, text in self.ITEMS:
            button = QPushButton(f"  {icon}   {text}")
            button.setObjectName("sideBtn")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedHeight(42)
            button.clicked.connect(lambda _checked=False, k=key: self.changed.emit(k))
            self.buttons[key] = (button, icon, text)
            v.addWidget(button)
        v.addStretch(1)
        self.collapse = QPushButton("⟨  Collapse")
        self.collapse.setObjectName("sideBtn")
        self.collapse.setFixedHeight(38)
        self.collapse.clicked.connect(hub.toggle_sidebar)
        v.addWidget(self.collapse)

    def set_active(self, key: str):
        for page, (button, _icon, _text) in self.buttons.items():
            button.setChecked(page == key)

    def set_collapsed(self, collapsed: bool):
        self.collapsed = collapsed
        for _key, (button, icon, text) in self.buttons.items():
            button.setText(icon if collapsed else f"  {icon}   {text}")
            button.setProperty("collapsed", collapsed)
            button.style().unpolish(button)
            button.style().polish(button)
        self.collapse.setText("⟩" if collapsed else "⟨  Collapse")


class CameraSurface(QWidget):
    doubleClicked = Signal()

    def __init__(self, feed: CameraFeed):
        super().__init__()
        self.feed = feed
        self.feed.surfaces.add(self)
        self.setMinimumSize(260, 146)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.destroyed.connect(lambda *_: self.feed.surfaces.discard(self))

    def set_feed(self, feed: CameraFeed):
        if feed is self.feed:
            return
        self.feed.surfaces.discard(self)
        self.feed = feed
        self.feed.surfaces.add(self)
        self.update()

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()
        event.accept()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#05070a"))
        image, _version = self.feed.latest()
        if image is None or image.isNull():
            painter.setPen(QColor(TH.FAINT))
            painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, "WAITING FOR CAMERA…")
            painter.end()
            return
        iw, ih = image.width(), image.height()
        if iw <= 0 or ih <= 0:
            painter.end()
            return
        w, h = self.width(), self.height()
        scale = min(w / iw, h / ih)
        tw, th = iw * scale, ih * scale
        target = QRectF((w - tw) / 2.0, (h - th) / 2.0, tw, th)
        painter.drawImage(target, image)
        painter.end()


class PulsingDot(QWidget):
    def __init__(self, color=TH.OK):
        super().__init__()
        self.color = QColor(color)
        self.setFixedSize(14, 14)

    def set_color(self, color: str):
        self.color = QColor(color)
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        c = QColor(self.color)
        c.setAlpha(55)
        p.setBrush(c)
        p.drawEllipse(QPointF(7, 7), 6, 6)
        p.setBrush(self.color)
        p.drawEllipse(QPointF(7, 7), 3.5, 3.5)
        p.end()


class GradientBar(QWidget):
    def __init__(self, height: int):
        super().__init__()
        self.setFixedHeight(height)
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(10, 2, 10, 2)
        self.lay.setSpacing(7)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(6, 9, 13, 205))
        p.end()


class QuickInfo(QFrame):
    def __init__(self, camera_id: str, name: str, location: str):
        super().__init__()
        self.camera_id = camera_id
        self.name = name
        self.location = location
        self.setObjectName("quickInfo")
        self.setFixedWidth(220)
        grid = QGridLayout(self)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(5)
        title = QLabel("QUICK INFO")
        title.setStyleSheet(
            f"color:{TH.ACC2};font-size:8px;font-weight:800;letter-spacing:1.5px;"
        )
        grid.addWidget(title, 0, 0, 1, 2)
        self.values = {}
        rows = [
            ("camera", "Camera"),
            ("status", "Status"),
            ("fps", "Publish FPS"),
            ("source", "Source FPS"),
            ("people", "Tracked People"),
            ("backend", "Backend"),
            ("age", "Frame Age"),
            ("reconnect", "Reconnects"),
        ]
        for row, (key, label) in enumerate(rows, 1):
            left = QLabel(label)
            left.setStyleSheet(f"color:{TH.DIM};font-size:9px;")
            right = QLabel("--")
            right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            right.setStyleSheet(f"color:{TH.TXT};font-size:9px;font-weight:700;")
            grid.addWidget(left, row, 0)
            grid.addWidget(right, row, 1)
            self.values[key] = right

    def refresh(self, state: dict):
        health = state.get("health") or {}
        camera = (health.get("cameras") or {}).get(self.camera_id) or {}
        publisher = (health.get("publishers") or {}).get(self.camera_id) or {}
        tracks = ((state.get("tracks") or {}).get("cameras") or {}).get(self.camera_id) or {}
        online = bool(camera.get("online"))
        self.values["camera"].setText(f"{self.camera_id} · {self.location}")
        self.values["status"].setText("Online" if online else "Offline")
        self.values["status"].setStyleSheet(
            f"color:{TH.OK if online else TH.ERR};font-size:9px;font-weight:700;"
        )
        self.values["fps"].setText(f"{float(publisher.get('publish_rate') or 0):.1f}")
        self.values["source"].setText(f"{float(camera.get('source_fps') or 0):.1f}")
        self.values["people"].setText(str(int(tracks.get("count") or 0)))
        self.values["backend"].setText(str(camera.get("capture_backend") or "--"))
        age = publisher.get("last_publish_source_age_ms")
        self.values["age"].setText(f"{float(age):.0f} ms" if age is not None else "--")
        self.values["reconnect"].setText(str(int(camera.get("reconnects") or 0)))


class CameraCard(QFrame):
    def __init__(self, hub, camera_id: str, name: str, location: str, feed: CameraFeed):
        super().__init__()
        self.hub = hub
        self.camera_id = camera_id
        self.name = name
        self.location = location
        self.feed = feed
        self.setObjectName("camCard")
        self.setMinimumSize(300, 190)
        self.glow = QGraphicsDropShadowEffect(self)
        self.glow.setBlurRadius(0)
        self.glow.setColor(QColor(TH.ACCENT))
        self.glow.setOffset(0, 0)
        self.setGraphicsEffect(self.glow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.surface = CameraSurface(feed)
        self.surface.doubleClicked.connect(lambda: hub.open_fullscreen(camera_id))
        layout.addWidget(self.surface)

        self.top = GradientBar(32)
        self.top.setParent(self)
        self.lbl_id = QLabel(camera_id)
        self.lbl_id.setStyleSheet("color:white;font-size:10.5px;font-weight:800;")
        self.lbl_loc = QLabel(f"· {location}")
        self.lbl_loc.setStyleSheet(f"color:{TH.DIM};font-size:9.5px;")
        self.dot = PulsingDot()
        self.lbl_status = QLabel("Offline")
        self.top.lay.addWidget(self.lbl_id)
        self.top.lay.addWidget(self.lbl_loc)
        self.top.lay.addStretch(1)
        self.top.lay.addWidget(self.dot)
        self.top.lay.addWidget(self.lbl_status)

        self.bottom = GradientBar(30)
        self.bottom.setParent(self)
        self.lbl_fps = QLabel("-- FPS")
        self.lbl_people = QLabel("👥 0")
        self.lbl_ai = QLabel("🤖 AI")
        self.lbl_conn = QLabel("░░░░")
        for label in (self.lbl_fps, self.lbl_people, self.lbl_ai, self.lbl_conn):
            label.setStyleSheet(f"color:{TH.DIM};font-size:9px;font-weight:600;")
            self.bottom.lay.addWidget(label)
        self.bottom.lay.addStretch(1)
        self.live = QLabel("● LIVE")
        self.live.setStyleSheet(f"color:{TH.ERR};font-size:8.5px;font-weight:800;")
        self.bottom.lay.addWidget(self.live)

        self.toolbar = QFrame(self)
        self.toolbar.setObjectName("camToolbar")
        tb = QHBoxLayout(self.toolbar)
        tb.setContentsMargins(4, 4, 4, 4)
        tb.setSpacing(4)
        self.snap = self._tool(tb, "📸", "Snapshot")
        self.full = self._tool(tb, "⛶", "Fullscreen")
        self.more = self._tool(tb, "⋮", "Camera info")
        self.snap.clicked.connect(lambda: hub.snapshot(camera_id))
        self.full.clicked.connect(lambda: hub.open_fullscreen(camera_id))
        self.more.clicked.connect(lambda: self.info.setVisible(not self.info.isVisible()))
        self.toolbar.hide()

        self.info = QuickInfo(camera_id, name, location)
        self.info.setParent(self)
        self.info.hide()

    @staticmethod
    def _tool(layout, text: str, tip: str):
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tip)
        button.setObjectName("camTool")
        button.setFixedSize(30, 30)
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)
        return button

    def resizeEvent(self, event):
        self.top.setGeometry(0, 0, self.width(), 32)
        self.bottom.setGeometry(0, self.height() - 30, self.width(), 30)
        self.toolbar.adjustSize()
        self.toolbar.move(self.width() - self.toolbar.width() - 8, 38)
        self.info.move(self.width() - self.info.width() - 8, 74)
        super().resizeEvent(event)

    def enterEvent(self, event):
        self.toolbar.show()
        self.glow.setBlurRadius(18)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.toolbar.hide()
        self.info.hide()
        self.glow.setBlurRadius(0)
        super().leaveEvent(event)

    def refresh(self, state: dict):
        health = state.get("health") or {}
        camera = (health.get("cameras") or {}).get(self.camera_id) or {}
        publisher = (health.get("publishers") or {}).get(self.camera_id) or {}
        tracks = ((state.get("tracks") or {}).get("cameras") or {}).get(self.camera_id) or {}
        online = bool(camera.get("online"))
        fps = float(publisher.get("publish_rate") or 0.0)
        source_fps = float(camera.get("source_fps") or 0.0)
        people = int(tracks.get("count") or 0)
        ready = bool((health.get("detector") or {}).get("ready"))
        self.lbl_status.setText("Online" if online else "Offline")
        self.lbl_status.setStyleSheet(
            f"color:{TH.OK if online else TH.ERR};font-size:9.5px;font-weight:700;"
        )
        self.dot.set_color(TH.OK if online else TH.ERR)
        self.lbl_fps.setText(f"{fps:.0f} FPS" if online else "-- FPS")
        self.lbl_people.setText(f"👥 {people}" if online else "👥 —")
        self.lbl_ai.setText("🤖 AI ON" if ready else "🤖 AI OFF")
        self.lbl_ai.setStyleSheet(
            f"color:{TH.OK if ready else TH.FAINT};font-size:9px;font-weight:700;"
        )
        quality = 4 if source_fps >= 18 else (3 if source_fps >= 14 else (2 if source_fps >= 8 else 1))
        self.lbl_conn.setText("▂▄▆█"[:quality] + "░" * (4 - quality) if online else "░░░░")
        self.lbl_conn.setStyleSheet(
            f"color:{TH.OK if quality >= 3 else (TH.WARN if quality == 2 else TH.ERR)};"
            "font-size:9px;"
        )
        self.setProperty("offline", not online)
        self.style().unpolish(self)
        self.style().polish(self)
        if self.info.isVisible():
            self.info.refresh(state)


class Page(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("page")
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(14, 12, 14, 12)
        self.v.setSpacing(10)

    def title_row(self, title: str, subtitle: str = ""):
        row = QHBoxLayout()
        row.setSpacing(10)
        label = QLabel(title)
        label.setStyleSheet("font-size:16px;font-weight:800;color:white;")
        row.addWidget(label)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setStyleSheet(f"color:{TH.FAINT};font-size:10px;")
            row.addWidget(sub)
        row.addStretch(1)
        self.v.addLayout(row)
        return row


class DashboardPage(Page):
    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        strip = QHBoxLayout()
        title = QLabel("LIVE WALL")
        title.setStyleSheet("font-size:11px;font-weight:800;color:white;letter-spacing:1px;")
        self.info = QLabel("6 cameras")
        self.info.setStyleSheet(f"color:{TH.DIM};font-size:10px;")
        strip.addWidget(title)
        strip.addWidget(self.info)
        strip.addStretch(1)
        snap = QPushButton("📸 Wall Snapshot")
        snap.setObjectName("btnGhost")
        snap.setFixedHeight(28)
        snap.clicked.connect(hub.wall_snapshot)
        strip.addWidget(snap)
        self.v.addLayout(strip)

        self.grid = QGridLayout()
        self.grid.setSpacing(8)
        self.cards = []
        for index, (camera_id, name, location) in enumerate(CAMERA_SPECS):
            card = CameraCard(hub, camera_id, name, location, hub.feeds[camera_id])
            self.cards.append(card)
            self.grid.addWidget(card, index // 3, index % 3)
        for col in range(3):
            self.grid.setColumnStretch(col, 1)
        for row in range(2):
            self.grid.setRowStretch(row, 1)
        self.v.addLayout(self.grid, 1)

    def refresh(self, state: dict):
        health = state.get("health") or {}
        self.info.setText(
            f"{int(health.get('online') or 0)}/{int(health.get('total') or 6)} cameras online · "
            f"{datetime.now().strftime('%d %b %Y')}"
        )
        for card in self.cards:
            card.refresh(state)


class LivePage(Page):
    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        row = self.title_row("Live Cameras", "Double-click → fullscreen")
        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍 Filter cameras…")
        self.search.setMaximumWidth(190)
        self.search.textChanged.connect(self.relayout)
        row.addWidget(self.search)
        self.layout_cb = QComboBox()
        self.layout_cb.addItems(["3 × 2", "2 × 3", "2 × 2", "1 × 1"])
        self.layout_cb.currentTextChanged.connect(self.relayout)
        row.addWidget(self.layout_cb)
        self.grid = QGridLayout()
        self.grid.setSpacing(8)
        self.cards = [
            CameraCard(hub, camera_id, name, location, hub.feeds[camera_id])
            for camera_id, name, location in CAMERA_SPECS
        ]
        self.v.addLayout(self.grid, 1)
        self.relayout()

    def relayout(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        query = self.search.text().strip().lower()
        visible = [
            card
            for card in self.cards
            if query in f"{card.camera_id} {card.name} {card.location}".lower()
        ]
        cols = int(self.layout_cb.currentText().split("×")[0].strip())
        for index, card in enumerate(visible):
            card.setParent(self)
            card.show()
            self.grid.addWidget(card, index // cols, index % cols)
        for card in self.cards:
            if card not in visible:
                card.hide()
        for index in range(6):
            self.grid.setColumnStretch(index, 1 if index < cols else 0)

    def refresh(self, state: dict):
        for card in self.cards:
            card.refresh(state)


class PersonManagementPage(Page):
    def __init__(self, hub):
        super().__init__()
        row = self.title_row("Person Management", "Face Recognition will be connected next")
        search = QLineEdit()
        search.setPlaceholderText("🔍 Search people…")
        search.setMaximumWidth(220)
        row.addWidget(search)
        enroll = QPushButton("＋ Enroll New")
        enroll.setObjectName("btnPrimary")
        enroll.clicked.connect(lambda: hub.navigate("enroll"))
        row.addWidget(enroll)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Photo", "Name", "Department", "Status", "Last Seen", "Recognitions"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.v.addWidget(self.table, 1)
        note = QLabel("No fake people are loaded. This table will be populated by the Face DB in the next step.")
        note.setAlignment(Qt.AlignCenter)
        note.setStyleSheet(f"color:{TH.FAINT};padding:10px;")
        self.v.addWidget(note)


class EnrollmentPage(Page):
    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        self.title_row("Person Enrollment", "UI ready · face backend is the next step")
        body = QHBoxLayout()
        body.setSpacing(12)
        left = QFrame()
        left.setObjectName("camCard")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        self.camera = QComboBox()
        for camera_id, _name, location in CAMERA_SPECS:
            self.camera.addItem(f"{camera_id} · {location}", camera_id)
        self.camera.currentIndexChanged.connect(self.change_camera)
        lv.addWidget(self.camera)
        self.surface = CameraSurface(hub.feeds[CAMERA_SPECS[0][0]])
        lv.addWidget(self.surface, 1)
        status = QLabel("Face engine is not enabled yet — this is only the final UI shell")
        status.setStyleSheet(f"color:{TH.WARN};padding:9px;font-size:10px;")
        lv.addWidget(status)
        body.addWidget(left, 1)

        right = QFrame()
        right.setObjectName("chartCard")
        right.setFixedWidth(360)
        form = QVBoxLayout(right)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(10)
        title = QLabel("REGISTER NEW PERSON")
        title.setStyleSheet(
            f"color:{TH.ACC2};font-size:9px;font-weight:800;letter-spacing:1.5px;"
        )
        form.addWidget(title)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Full Name *")
        self.dept = QComboBox()
        self.dept.addItems(["Security", "IT", "Finance", "HR", "Operations", "Management"])
        self.emp = QLineEdit()
        self.emp.setPlaceholderText("Employee ID")
        form.addWidget(self.name)
        form.addWidget(self.dept)
        form.addWidget(self.emp)
        progress_label = QLabel("Captured 0 / 10")
        progress_label.setStyleSheet(f"color:{TH.DIM};font-size:10px;font-weight:700;")
        progress = QProgressBar()
        progress.setRange(0, 10)
        progress.setValue(0)
        progress.setTextVisible(False)
        form.addWidget(progress_label)
        form.addWidget(progress)
        thumbs = QGridLayout()
        for index in range(10):
            box = QLabel(str(index + 1))
            box.setFixedSize(52, 52)
            box.setAlignment(Qt.AlignCenter)
            box.setStyleSheet(
                f"color:{TH.FAINT};background:#11161c;border:1px dashed {TH.BORDER};border-radius:6px;"
            )
            thumbs.addWidget(box, index // 5, index % 5)
        form.addLayout(thumbs)
        form.addStretch(1)
        capture = QPushButton("📸 Capture 10 samples")
        capture.setObjectName("btnGhost")
        capture.setEnabled(False)
        register = QPushButton("💾 Register")
        register.setObjectName("btnPrimary")
        register.setEnabled(False)
        form.addWidget(capture)
        form.addWidget(register)
        body.addWidget(right)
        self.v.addLayout(body, 1)

    def change_camera(self):
        camera_id = self.camera.currentData()
        if camera_id in self.hub.feeds:
            self.surface.set_feed(self.hub.feeds[camera_id])


class AnalyticsPage(Page):
    def __init__(self):
        super().__init__()
        self.title_row("Analytics", "live system summary")
        grid = QGridLayout()
        grid.setSpacing(10)
        self.labels = {}
        for index, (key, title) in enumerate(
            [
                ("occupancy", "CURRENT OCCUPANCY"),
                ("gpu", "GPU UTILIZATION"),
                ("fps", "AVERAGE CAMERA FPS"),
                ("global", "GLOBAL TRACK IDS"),
            ]
        ):
            card = QFrame()
            card.setObjectName("chartCard")
            v = QVBoxLayout(card)
            caption = QLabel(title)
            caption.setStyleSheet(f"color:{TH.DIM};font-size:9px;font-weight:800;")
            value = QLabel("--")
            value.setStyleSheet("font-size:30px;font-weight:800;color:white;")
            hint = QLabel("real backend")
            hint.setStyleSheet(f"color:{TH.FAINT};font-size:9px;")
            v.addWidget(caption)
            v.addWidget(value)
            v.addStretch(1)
            v.addWidget(hint)
            self.labels[key] = value
            grid.addWidget(card, index // 2, index % 2)
        self.v.addLayout(grid, 1)

    def refresh(self, state: dict):
        health = state.get("health") or {}
        tracks = state.get("tracks") or {}
        resources = health.get("service_resources") or {}
        publishers = health.get("publishers") or {}
        rates = [float(p.get("publish_rate") or 0.0) for p in publishers.values()]
        global_ids = set()
        for camera in (tracks.get("cameras") or {}).values():
            for track in camera.get("tracks") or []:
                gid = track.get("global_id")
                if gid:
                    global_ids.add(str(gid))
        self.labels["occupancy"].setText(str(int(tracks.get("total") or 0)))
        self.labels["gpu"].setText(f"{float(resources.get('gpu_utilization_percent') or 0):.0f}%")
        self.labels["fps"].setText(f"{(sum(rates) / len(rates) if rates else 0):.1f}")
        self.labels["global"].setText(str(len(global_ids)))


class EventsPage(Page):
    def __init__(self):
        super().__init__()
        row = self.title_row("Events", "camera connectivity events")
        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍 Search…")
        self.search.setMaximumWidth(220)
        row.addWidget(self.search)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Time", "Camera", "Event", "Level"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.v.addWidget(self.table, 1)

    def refresh(self, events: list[dict]):
        query = self.search.text().strip().lower()
        rows = [
            event
            for event in events
            if not query
            or query in f"{event.get('camera')} {event.get('text')} {event.get('level')}".lower()
        ]
        self.table.setRowCount(len(rows))
        for row, event in enumerate(rows):
            values = [event.get("time", ""), event.get("camera", ""), event.get("text", ""), event.get("level", "")]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col == 3:
                    item.setForeground(QColor(TH.OK if value == "ok" else TH.ERR))
                self.table.setItem(row, col, item)


class SettingsPage(Page):
    def __init__(self):
        super().__init__()
        self.title_row("Settings", "UI-only view; backend settings remain unchanged")
        tabs = QTabWidget()
        self.v.addWidget(tabs, 1)

        cameras = QWidget()
        cv = QVBoxLayout(cameras)
        for camera_id, name, location in CAMERA_SPECS:
            row = QFrame()
            row.setObjectName("chartCard")
            h = QHBoxLayout(row)
            h.addWidget(QLabel(f"🎥 {camera_id}  —  {name}"))
            h.addStretch(1)
            loc = QLabel(location)
            loc.setStyleSheet(f"color:{TH.DIM};")
            h.addWidget(loc)
            cv.addWidget(row)
        cv.addStretch(1)
        tabs.addTab(cameras, "🎥 Cameras")

        ai = QWidget()
        form = QFormLayout(ai)
        form.addRow("Detector", QLabel("YOLO26m · CUDA · person-only"))
        form.addRow("Tracking", QLabel("Camera-local ownership tracker"))
        form.addRow("Cross-camera ReID", QLabel("Current backend state preserved"))
        form.addRow("Pose / Heatmap", QLabel("Not enabled"))
        tabs.addTab(ai, "🤖 AI")

        recognition = QWidget()
        rform = QFormLayout(recognition)
        rform.addRow("Face Recognition", QLabel("Next implementation step"))
        rform.addRow("Enrollment", QLabel("UI shell ready"))
        tabs.addTab(recognition, "🆔 Recognition")


class RightPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("rightPanel")
        self.setMinimumWidth(245)
        self.setMaximumWidth(300)
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 14, 12, 10)
        v.setSpacing(10)
        title = QLabel("LIVE STATUS")
        title.setStyleSheet(
            f"color:{TH.FAINT};font-size:8.5px;font-weight:800;letter-spacing:2px;"
        )
        v.addWidget(title)

        stat_row = QHBoxLayout()
        self.known = self._stat("👤 KNOWN", TH.OK)
        self.unknown = self._stat("❓ UNKNOWN", "#f59e42")
        stat_row.addWidget(self.known[0])
        stat_row.addWidget(self.unknown[0])
        v.addLayout(stat_row)
        v.addWidget(self._sep("SYSTEM"))
        self.gpu = self._meter("GPU")
        self.cpu = self._meter("CPU")
        self.fps = self._meter("FPS")
        for widget in (self.gpu[0], self.cpu[0], self.fps[0]):
            v.addWidget(widget)
        v.addWidget(self._sep("ALERTS"))
        self.alerts = QVBoxLayout()
        self.alerts.setSpacing(5)
        v.addLayout(self.alerts)
        v.addWidget(self._sep("RECENT EVENTS"))
        self.recent_box = QVBoxLayout()
        self.recent_box.setSpacing(2)
        self.recent_box.addStretch(1)
        wrap = QWidget()
        wrap.setLayout(self.recent_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(wrap)
        scroll.setFrameShape(QFrame.NoFrame)
        v.addWidget(scroll, 1)

    @staticmethod
    def _stat(label: str, color: str):
        frame = QFrame()
        frame.setObjectName("statCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        number = QLabel("0")
        number.setStyleSheet(f"color:{color};font-size:20px;font-weight:800;")
        caption = QLabel(label)
        caption.setStyleSheet(f"color:{TH.DIM};font-size:8.5px;font-weight:700;")
        layout.addWidget(number)
        layout.addWidget(caption)
        return frame, number

    @staticmethod
    def _sep(text: str):
        label = QLabel(text)
        label.setStyleSheet(
            f"color:{TH.FAINT};font-size:8px;font-weight:800;letter-spacing:1.5px;padding-top:6px;"
        )
        return label

    @staticmethod
    def _meter(label: str):
        widget = QWidget()
        h = QHBoxLayout(widget)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        name = QLabel(label)
        name.setFixedWidth(28)
        name.setStyleSheet(f"color:{TH.DIM};font-size:9px;font-weight:700;")
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setTextVisible(False)
        bar.setFixedHeight(6)
        value = QLabel("--")
        value.setFixedWidth(48)
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value.setStyleSheet(f"color:{TH.TXT};font-size:9px;font-family:Consolas,monospace;")
        h.addWidget(name)
        h.addWidget(bar, 1)
        h.addWidget(value)
        return widget, bar, value

    @staticmethod
    def _clear(layout):
        while layout.count() > 1:
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def refresh(self, state: dict, events: list[dict]):
        health = state.get("health") or {}
        tracks = state.get("tracks") or {}
        resources = health.get("service_resources") or {}
        global_ids = set()
        local_without_global = 0
        for camera in (tracks.get("cameras") or {}).values():
            for track in camera.get("tracks") or []:
                gid = track.get("global_id")
                if gid:
                    global_ids.add(str(gid))
                else:
                    local_without_global += 1
        # Face Recognition is intentionally not connected yet: every tracked
        # identity remains Unknown. Paired-camera G-IDs are counted once.
        unknown = len(global_ids) + local_without_global
        self.known[1].setText("0")
        self.unknown[1].setText(str(unknown))

        gpu = float(resources.get("gpu_utilization_percent") or 0.0)
        cpu = float(resources.get("cpu_percent") or 0.0)
        rates = [float(v.get("publish_rate") or 0.0) for v in (health.get("publishers") or {}).values()]
        fps = sum(rates) / len(rates) if rates else 0.0
        for meter, value, suffix in ((self.gpu, gpu, "%"), (self.cpu, cpu, "%")):
            meter[1].setValue(int(max(0, min(100, round(value)))))
            meter[2].setText(f"{value:.0f}{suffix}")
        self.fps[1].setValue(int(max(0, min(100, fps / 30.0 * 100.0))))
        self.fps[2].setText(f"{fps:.1f}")

        self._clear(self.alerts)
        offline = [
            camera_id
            for camera_id, metrics in (health.get("cameras") or {}).items()
            if not metrics.get("online")
        ]
        if offline:
            for camera_id in offline[:3]:
                label = QLabel(f"│ {camera_id} — Camera offline")
                label.setStyleSheet(
                    f"color:{TH.ERR};background:#1c222b;padding:8px;border-radius:5px;"
                )
                self.alerts.addWidget(label)
        else:
            label = QLabel("● All cameras healthy")
            label.setStyleSheet(f"color:{TH.OK};padding:6px;")
            self.alerts.addWidget(label)

        self._clear(self.recent_box)
        for event in events[:10]:
            color = TH.OK if event.get("level") == "ok" else TH.ERR
            label = QLabel(
                f"{event.get('time','')}  ●  {event.get('camera','')} · {event.get('text','')}"
            )
            label.setStyleSheet(f"color:{color};font-size:9px;padding:2px;")
            self.recent_box.insertWidget(self.recent_box.count() - 1, label)


class FullscreenCamera(QDialog):
    def __init__(self, hub, camera_id: str):
        super().__init__(hub)
        self.setWindowTitle(camera_id)
        self.resize(1280, 720)
        self.setStyleSheet("background:#000;")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        top = QFrame()
        top.setFixedHeight(54)
        top.setStyleSheet(f"background:{TH.PANEL};border-bottom:1px solid {TH.BORDER};")
        h = QHBoxLayout(top)
        title = QLabel(camera_id)
        title.setStyleSheet("font-size:15px;font-weight:800;color:white;")
        h.addWidget(title)
        h.addStretch(1)
        close = QPushButton("← Back")
        close.setObjectName("btnGhost")
        close.clicked.connect(self.accept)
        h.addWidget(close)
        v.addWidget(top)
        self.surface = CameraSurface(hub.feeds[camera_id])
        v.addWidget(self.surface, 1)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Back):
            self.accept()
        else:
            super().keyPressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Surveillance System — MUKAMMAL UI")
        self.resize(1680, 980)
        self.setMinimumSize(1180, 700)
        self.state_reader = BackendState()
        self.feeds = {camera_id: CameraFeed(camera_id) for camera_id, _n, _l in CAMERA_SPECS}
        for feed in self.feeds.values():
            feed.start()
        self.state_reader.start()

        self.header = Header(self)
        self.sidebar = SideBar(self)
        self.sidebar.changed.connect(self.navigate)
        self.right = RightPanel()
        self.stack = QStackedWidget()
        self.dashboard = DashboardPage(self)
        self.live = LivePage(self)
        self.people = PersonManagementPage(self)
        self.enroll = EnrollmentPage(self)
        self.analytics = AnalyticsPage()
        self.events = EventsPage()
        self.settings = SettingsPage()
        self.page_index = {}
        for key, page in [
            ("dashboard", self.dashboard),
            ("live", self.live),
            ("people", self.people),
            ("enroll", self.enroll),
            ("analytics", self.analytics),
            ("events", self.events),
            ("settings", self.settings),
        ]:
            self.page_index[key] = self.stack.addWidget(page)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.addWidget(self.stack)
        splitter.addWidget(self.right)
        splitter.setStretchFactor(0, 80)
        splitter.setStretchFactor(1, 20)

        body = QWidget()
        hb = QHBoxLayout(body)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(0)
        hb.addWidget(self.sidebar)
        hb.addWidget(splitter, 1)

        root = QWidget()
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self.header)
        v.addWidget(body, 1)
        self.setCentralWidget(root)

        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.render_tick)
        self.render_timer.start(33)
        self.state_timer = QTimer(self)
        self.state_timer.timeout.connect(self.refresh_state)
        self.state_timer.start(350)
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.header.tick_clock)
        self.clock_timer.start(1000)

        self.shortcuts = []
        pages = ["dashboard", "live", "people", "enroll", "analytics", "events", "settings"]
        for index, key in enumerate("1234567"):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda page=pages[index]: self.navigate(page))
            self.shortcuts.append(shortcut)
        search_shortcut = QShortcut(QKeySequence("/"), self)
        search_shortcut.activated.connect(self.header.search.setFocus)
        self.shortcuts.append(search_shortcut)
        self.navigate("dashboard")
        self.refresh_state()

    def navigate(self, page: str):
        if page not in self.page_index:
            return
        self.stack.setCurrentIndex(self.page_index[page])
        self.sidebar.set_active(page)

    def toggle_sidebar(self):
        collapsed = not self.sidebar.collapsed
        self.sidebar.set_collapsed(collapsed)
        width = 64 if collapsed else 210
        self.sidebar.setMinimumWidth(width)
        self.sidebar.setMaximumWidth(width)

    def render_tick(self):
        for feed in self.feeds.values():
            feed.update_surfaces()

    def refresh_state(self):
        state, events = self.state_reader.snapshot()
        self.header.refresh(state)
        self.right.refresh(state, events)
        self.dashboard.refresh(state)
        self.live.refresh(state)
        self.analytics.refresh(state)
        self.events.refresh(events)

    def snapshot(self, camera_id: str):
        image, _version = self.feeds[camera_id].latest()
        if image is None or image.isNull():
            self.toast("⚠ Camera frame is not ready")
            return
        Path("snapshots").mkdir(parents=True, exist_ok=True)
        path = Path("snapshots") / f"{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image.save(str(path), "JPG", 92)
        self.toast(f"📸 Snapshot saved · {path}")

    def wall_snapshot(self):
        frames = []
        for camera_id, _name, _location in CAMERA_SPECS:
            image, _version = self.feeds[camera_id].latest()
            if image is not None and not image.isNull():
                frames.append((camera_id, image.copy()))
        if not frames:
            self.toast("⚠ No camera frames are ready")
            return
        width, height = 736, 416
        wall = QPixmap(width * 3, height * 2)
        wall.fill(QColor("#000"))
        painter = QPainter(wall)
        for index, (_camera_id, image) in enumerate(frames):
            x, y = (index % 3) * width, (index // 3) * height
            painter.drawImage(QRectF(x, y, width, height), image)
        painter.end()
        Path("snapshots").mkdir(parents=True, exist_ok=True)
        path = Path("snapshots") / f"wall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        wall.save(str(path), "JPG", 90)
        self.toast(f"📸 Wall snapshot saved · {path}")

    def open_fullscreen(self, camera_id: str):
        dialog = FullscreenCamera(self, camera_id)
        dialog.showMaximized()
        dialog.exec()

    def toast(self, text: str):
        label = QLabel(text, self)
        label.setStyleSheet(
            f"background:{TH.CARD2};color:{TH.TXT};border:1px solid {TH.BORDER};"
            "border-radius:9px;padding:10px 16px;font-size:11px;font-weight:600;"
        )
        label.adjustSize()
        label.move(self.width() - label.width() - 24, self.height() - label.height() - 24)
        label.show()
        label.raise_()
        QTimer.singleShot(2400, label.deleteLater)

    def closeEvent(self, event):
        self.render_timer.stop()
        self.state_timer.stop()
        self.state_reader.stop()
        for feed in self.feeds.values():
            feed.stop()
        event.accept()


STYLE = """
* { font-family: "Segoe UI", "Roboto", "Ubuntu", sans-serif; }
QMainWindow, QDialog { background: #101418; }
QWidget { color: #e9eef5; font-size: 12px; }
QWidget#page { background: #0f1317; }
QFrame#header { background: #151b22; border-bottom: 1px solid #2b3542; }
QFrame#sidebar { background: #151b22; border-right: 1px solid #2b3542; }
QFrame#rightPanel { background: #151b22; }
QFrame#chip { background: #1a212a; border: 1px solid #2b3542; border-radius: 12px; }
QPushButton#sideBtn { text-align: left; padding: 10px 14px; border-radius: 8px;
    color: #94a1b3; font-size: 12.5px; border: none; background: transparent; }
QPushButton#sideBtn[collapsed="true"] { text-align: center; padding: 10px 0; }
QPushButton#sideBtn:hover { background: #202a35; color: #e9eef5; }
QPushButton#sideBtn:checked { background: #2f7df6; color: #fff; font-weight: 700; }
QFrame#camCard { background: #0a0d11; border: 1px solid #2b3542; border-radius: 10px; }
QFrame#camCard[offline="true"] { border: 1px solid #ef5350; }
QFrame#camToolbar { background: rgba(13,17,22,215); border: 1px solid #2b3542; border-radius: 9px; }
QToolButton#camTool, QPushButton#camTool { background: transparent; border: none;
    border-radius: 6px; font-size: 13px; color: #e9eef5; }
QToolButton#camTool:hover, QPushButton#camTool:hover { background: #2f7df6; }
QFrame#quickInfo { background: rgba(13,17,22,235); border: 1px solid #2b3542; border-radius: 9px; }
QFrame#statCard, QFrame#chartCard { background: #1a212a; border: 1px solid #2b3542; border-radius: 10px; }
QPushButton#btnPrimary { background: #2f7df6; color: white; border: none;
    border-radius: 7px; padding: 8px 16px; font-weight: 700; }
QPushButton#btnPrimary:hover { background: #4a8ff7; }
QPushButton#btnPrimary:disabled { background: #233043; color: #5d6b7e; }
QPushButton#btnGhost { background: #232c37; border: 1px solid #2b3542;
    border-radius: 7px; padding: 7px 14px; color: #e9eef5; }
QPushButton#btnGhost:hover { background: #2a3441; }
QLineEdit, QComboBox { background: #232c37; border: 1px solid #2b3542;
    border-radius: 6px; padding: 6px 9px; selection-background-color: #2f7df6; }
QLineEdit:focus, QComboBox:focus { border: 1px solid #2f7df6; }
QComboBox QAbstractItemView { background: #202a35; border: 1px solid #2b3542;
    selection-background-color: #2f7df6; outline: none; }
QProgressBar { background: #232c37; border-radius: 3px; border: none; }
QProgressBar::chunk { background: #2f7df6; border-radius: 3px; }
QTableWidget { background: #1a212a; alternate-background-color: #1d2530;
    gridline-color: #2b3542; border: 1px solid #2b3542; border-radius: 8px;
    selection-background-color: #2f7df6; }
QTableWidget::item { padding: 4px 8px; }
QHeaderView::section { background: #202a35; color: #94a1b3; padding: 7px;
    border: none; border-bottom: 1px solid #2b3542; font-weight: 700; font-size: 10.5px; }
QTabWidget::pane { border: 1px solid #2b3542; border-radius: 8px; background: #1a212a; top: -1px; }
QTabBar::tab { padding: 9px 18px; background: transparent; color: #94a1b3;
    border: 1px solid transparent; border-bottom: none; margin-right: 2px; }
QTabBar::tab:selected { background: #1a212a; color: #fff; border-color: #2b3542; font-weight: 700; }
QMenu { background: #202a35; border: 1px solid #2b3542; border-radius: 8px; padding: 4px; }
QMenu::item { padding: 7px 26px; border-radius: 5px; }
QMenu::item:selected { background: #2f7df6; color: white; }
QScrollArea { background: transparent; }
QScrollBar:vertical { background: transparent; width: 9px; margin: 0; }
QScrollBar::handle:vertical { background: #2b3542; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #3a4757; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QSplitter::handle { background: #2b3542; }
QToolTip { background: #202a35; color: #e9eef5; border: 1px solid #2b3542;
    border-radius: 5px; padding: 5px 8px; }
"""


def run():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(TH.PANEL))
    palette.setColor(QPalette.WindowText, QColor(TH.TXT))
    palette.setColor(QPalette.Base, QColor(TH.CARD))
    palette.setColor(QPalette.Text, QColor(TH.TXT))
    palette.setColor(QPalette.Button, QColor(TH.CARD2))
    palette.setColor(QPalette.ButtonText, QColor(TH.TXT))
    palette.setColor(QPalette.Highlight, QColor(TH.ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("white"))
    app.setPalette(palette)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec()
