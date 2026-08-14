from __future__ import annotations

import http.client
import json
import threading
import time

from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class HeatmapReader:
    """Fetch only the selected room heatmap at a low UI cadence."""

    def __init__(self, host: str, port: int, room_id: str = "ROOM-1"):
        self.host = host
        self.port = int(port)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._room_id = str(room_id)
        self._image: QImage | None = None
        self._summary = {"enabled": False, "rooms": {}}
        self._version = 0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ui-floor-heatmap", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=2.0):
        if self._thread:
            self._thread.join(timeout)

    def set_room(self, room_id: str):
        with self._lock:
            room_id = str(room_id)
            if room_id != self._room_id:
                self._room_id = room_id
                self._image = None
                self._version += 1

    def latest(self):
        with self._lock:
            return self._image, self._version, dict(self._summary), self._room_id

    @staticmethod
    def _read_json(connection, path):
        connection.request("GET", path, headers={"Connection": "keep-alive", "Cache-Control": "no-cache"})
        response = connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(response.status)
        return json.loads(payload.decode("utf-8"))

    def _run(self):
        connection = None
        while not self._stop.is_set():
            try:
                if connection is None:
                    connection = http.client.HTTPConnection(self.host, self.port, timeout=2.0)
                summary = self._read_json(connection, "/heatmap")
                with self._lock:
                    room_id = self._room_id
                connection.request(
                    "GET",
                    f"/heatmap/{room_id}.png",
                    headers={"Connection": "keep-alive", "Cache-Control": "no-cache"},
                )
                response = connection.getresponse()
                payload = response.read()
                if response.status != 200:
                    raise RuntimeError(response.status)
                image = QImage.fromData(payload, "PNG")
                if image.isNull():
                    raise RuntimeError("invalid heatmap PNG")
                with self._lock:
                    self._summary = summary
                    self._image = image
                    self._version += 1
                self._stop.wait(0.75)
            except Exception:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
                connection = None
                with self._lock:
                    self._summary = {**self._summary, "connected": False}
                self._stop.wait(0.8)


class FloorHeatmapCanvas(QWidget):
    def __init__(self, dashboard_module):
        super().__init__()
        self.d = dashboard_module
        self.room_id = "ROOM-1"
        self.image: QImage | None = None
        self.summary = {"enabled": False, "rooms": {}}
        self.setMinimumHeight(420)

    def update_state(self, room_id: str, image: QImage | None, summary: dict):
        self.room_id = str(room_id)
        self.image = image
        self.summary = dict(summary or {})
        self.update()

    def paintEvent(self, event):
        d = self.d
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor(d.PANEL))

        outer = QRectF(22, 18, max(40, self.width() - 44), max(40, self.height() - 36))
        painter.setPen(QPen(QColor(d.BORDER), 2))
        painter.setBrush(QColor("#00152b"))
        painter.drawRoundedRect(outer, 10, 10)

        painter.setPen(QColor(d.TEXT))
        painter.setFont(d.app_font(18, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(outer.left() + 18, outer.top() + 12, outer.width() - 36, 28),
            f"{self.room_id.replace('-', ' ')} · ankle floor activity",
        )

        floor = QRectF(outer.left() + 24, outer.top() + 54, outer.width() - 48, outer.height() - 94)
        painter.setPen(QPen(QColor("#114267"), 1))
        painter.setBrush(QColor("#00192f"))
        painter.drawRoundedRect(floor, 7, 7)

        for index in range(1, 6):
            x = floor.left() + floor.width() * index / 6.0
            y = floor.top() + floor.height() * index / 6.0
            painter.drawLine(QPointF(x, floor.top()), QPointF(x, floor.bottom()))
            painter.drawLine(QPointF(floor.left(), y), QPointF(floor.right(), y))

        room = ((self.summary.get("rooms") or {}).get(self.room_id) or {})
        calibrated = bool(room.get("calibrated"))
        samples = int(room.get("samples") or 0)

        if calibrated and self.image is not None and not self.image.isNull() and samples > 0:
            painter.save()
            painter.setOpacity(0.90)
            painter.drawImage(floor, self.image)
            painter.restore()
        else:
            painter.setPen(QColor(d.ORANGE if not calibrated else d.MUTED))
            painter.setFont(d.app_font(14, QFont.Weight.Medium))
            message = (
                "Floor calibration required\nUse Room Map → Assisted floor calibration first"
                if not calibrated
                else "Waiting for confident ankle samples"
            )
            painter.drawText(floor, Qt.AlignmentFlag.AlignCenter, message)

        legend = QRectF(floor.left(), floor.bottom() + 14, min(260.0, floor.width() * 0.42), 10)
        gradient = QLinearGradient(legend.left(), 0, legend.right(), 0)
        gradient.setColorAt(0.0, QColor("#1a42aa"))
        gradient.setColorAt(0.35, QColor("#13c8d3"))
        gradient.setColorAt(0.68, QColor("#f5db4c"))
        gradient.setColorAt(1.0, QColor("#ef3b24"))
        painter.fillRect(legend, gradient)
        painter.setPen(QColor(d.MUTED))
        painter.setFont(d.app_font(11))
        painter.drawText(QPointF(legend.left(), legend.bottom() + 14), "cool")
        painter.drawText(QPointF(legend.right() - 18, legend.bottom() + 14), "hot")


class HeatmapPage(QWidget):
    def __init__(self, dashboard_module, reader: HeatmapReader):
        super().__init__()
        self.d = dashboard_module
        self.reader = reader
        self.room_id = "ROOM-1"
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        head = QHBoxLayout()
        title = QLabel("Floor Heatmap")
        title.setFont(dashboard_module.app_font(27, QFont.Weight.DemiBold))
        head.addWidget(title)
        head.addStretch()
        self.room_buttons = {}
        for room_id in ("ROOM-1", "ROOM-2", "ROOM-3"):
            button = QPushButton(room_id.replace("-", " "))
            button.setCheckable(True)
            button.setObjectName("roomButton")
            button.clicked.connect(lambda checked=False, rid=room_id: self.set_room(rid))
            self.room_buttons[room_id] = button
            head.addWidget(button)
        root.addLayout(head)

        self.info = QLabel("Pose ankles → calibrated room floor · full heat for 1 hour, then gradual cooling")
        self.info.setFont(dashboard_module.app_font(13))
        self.info.setStyleSheet(f"color:{dashboard_module.MUTED};")
        root.addWidget(self.info)

        self.canvas = FloorHeatmapCanvas(dashboard_module)
        self.canvas.setStyleSheet(
            f"background:{dashboard_module.PANEL};border:1px solid {dashboard_module.BORDER};border-radius:9px;"
        )
        root.addWidget(self.canvas, 1)
        self.set_room("ROOM-1")

    def set_room(self, room_id: str):
        self.room_id = str(room_id)
        for key, button in self.room_buttons.items():
            button.setChecked(key == self.room_id)
        self.reader.set_room(self.room_id)

    def update_live(self, image: QImage | None, summary: dict, room_id: str):
        self.room_id = str(room_id)
        rooms = summary.get("rooms") or {}
        room = rooms.get(self.room_id) or {}
        samples = int(room.get("samples") or 0)
        weighted = float(room.get("weighted_samples") or 0.0)
        calibrated = bool(room.get("calibrated"))
        status = "calibrated" if calibrated else "needs calibration"
        self.info.setText(
            f"{status} · {samples} ankle samples · active heat {weighted:.1f} · "
            "1h full heat → 1h half-life cooling"
        )
        self.canvas.update_state(self.room_id, image, summary)


def install(dashboard_module):
    """Add a floor-heatmap page without modifying the realtime camera hot path."""

    original_window_init = dashboard_module.DashboardWindow.__init__
    original_close_event = dashboard_module.DashboardWindow.closeEvent

    def refresh_heatmap(self):
        image, version, summary, room_id = self.heatmap_reader.latest()
        if version != getattr(self, "_heatmap_seen_version", -1):
            self._heatmap_seen_version = version
            self.heatmap_page.update_live(image, summary, room_id)

    def window_init(self):
        original_window_init(self)
        self.heatmap_reader = HeatmapReader(dashboard_module.ML_HOST, dashboard_module.ML_PORT)
        self.heatmap_page = HeatmapPage(dashboard_module, self.heatmap_reader)
        self.stack.insertWidget(2, self.heatmap_page)

        old_buttons = dict(self.sidebar.buttons)
        heat_button = dashboard_module.NavButton("Heatmap", "activity")
        self.sidebar._layout.insertWidget(3, heat_button)
        for button in old_buttons.values():
            try:
                button.clicked.disconnect()
            except Exception:
                pass
        new_buttons = {
            0: old_buttons[0],
            1: old_buttons[1],
            2: heat_button,
            3: old_buttons[2],
            4: old_buttons[3],
        }
        for index, button in new_buttons.items():
            button.clicked.connect(lambda checked=False, i=index: self.set_page(i))
        self.sidebar.buttons = new_buttons
        self.sidebar.set_active(self.stack.currentIndex())

        self._heatmap_seen_version = -1
        self.heatmap_reader.start()
        self.heatmap_timer = QTimer(self)
        self.heatmap_timer.timeout.connect(lambda: refresh_heatmap(self))
        self.heatmap_timer.start(750)
        refresh_heatmap(self)

    def close_event(self, event):
        if hasattr(self, "heatmap_timer"):
            self.heatmap_timer.stop()
        if hasattr(self, "heatmap_reader"):
            self.heatmap_reader.stop()
            self.heatmap_reader.join(1.5)
        original_close_event(self, event)

    dashboard_module.DashboardWindow.__init__ = window_init
    dashboard_module.DashboardWindow.closeEvent = close_event
