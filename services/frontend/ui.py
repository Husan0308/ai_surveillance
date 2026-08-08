# =====================================================================
#  AI SURVEILLANCE SYSTEM — MUKAMMAL (PERFECT) EDITION UI
#  Python + PySide6  |  ui.py — LOCKED
# =====================================================================

import sys
import os
import math
import csv
from datetime import datetime

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from services.frontend.api_client import ApiClient
from services.frontend.async_api import AsyncApi
from services.frontend.websocket_client import WebSocketClient


# ============================ THEME ==================================
class TH:
    BG     = "#0f1317"
    PANEL  = "#151b22"
    CARD   = "#1a212a"
    CARD2  = "#212a35"
    HOVER  = "#26303c"
    BORDER = "#2b3542"
    ACCENT = "#2f7df6"
    ACC2   = "#5b9bff"
    OK     = "#2ecc71"
    WARN   = "#f5c542"
    ERR    = "#ef5350"
    TXT    = "#e9eef5"
    DIM    = "#94a1b3"
    FAINT  = "#5d6b7e"


FRAME_W, FRAME_H = 640, 360
GW, GH = 64, 36


def heat_color(v):
    stops = [
        (0.0, (40, 110, 255)),
        (0.3, (0, 200, 200)),
        (0.55, (60, 215, 80)),
        (0.8, (250, 200, 40)),
        (1.0, (255, 70, 40)),
    ]

    v = max(0.0, min(1.0, v))

    for i in range(1, len(stops)):
        if v <= stops[i][0]:
            t0, c0 = stops[i - 1]
            t1, c1 = stops[i]
            f = (v - t0) / (t1 - t0) if t1 > t0 else 0
            return tuple(int(c0[k] + (c1[k] - c0[k]) * f) for k in range(3))

    return stops[-1][1]


def clamp(v, a, b):
    return max(a, min(b, v))




# ========================= REALTIME CAMERA STATE =====================
class RealtimeTrack:
    def __init__(self, payload):
        self.track_id = payload.get("local_track_id")
        self.global_id = payload.get("global_id")
        self.person_id = payload.get("person_id")
        self.name = payload.get("display_name") or self.global_id or self.track_id or "Unknown"
        self.conf = float(payload.get("confidence") or 0.0)
        self.known = bool(self.person_id or payload.get("display_name"))
        self._bbox = tuple(float(v) for v in payload.get("bbox", (0, 0, 0, 0)))

    def bbox(self, _width, _height):
        x1, y1, x2, y2 = self._bbox
        return QRectF(x1, y1, max(0.0, x2-x1), max(0.0, y2-y1))


class CameraState(QObject):
    def __init__(self, cid, name, location, enabled=True, max_people=0):
        super().__init__()
        self.id, self.name, self.location = cid, name, location
        self.enabled, self.online = bool(enabled), False
        self.ai_on, self.heat_on, self.recording = True, False, False
        self.fps, self.res = 0.0, "—"
        self.latency, self.packet_loss, self.infer_ms = 0.0, 0.0, 0.0
        self.frame = None
        self.frame_id = None
        self.frame_timestamp = None
        self.tracks = []
        self.surfaces = []
        self.heat = [[0.0] * GW for _ in range(GH)]
        self.hist = [[0.0] * GW for _ in range(GH)]

    @property
    def people(self):
        return self.tracks

    @property
    def conn_quality(self):
        return 4 if self.online and self.frame is not None else 0

    def set_metadata(self, message):
        if self.frame_id is None or int(message.get("frame_id", -1)) != int(self.frame_id):
            return False
        self.tracks = [RealtimeTrack(item) for item in message.get("tracks", ())]
        return True

    def clear_frame(self):
        self.online = False
        self.frame = None
        self.frame_id = None
        self.frame_timestamp = None
        self.tracks = []

    def heat_image(self):
        import numpy as np
        values=np.asarray(self.heat,dtype=np.float32)
        img=QImage(GW,GH,QImage.Format_RGBA8888);img.fill(Qt.transparent)
        maximum=float(values.max()) if values.size else 0.0
        if maximum <= 0:return img
        for y in range(GH):
            for x in range(GW):
                value=float(values[y,x])/maximum
                if value > .05:
                    r,g,b=heat_color(value);img.setPixelColor(x,y,QColor(r,g,b,min(235,int(60+value*160))))
        return img

    def apply_heatmap(self,snapshot):
        import cv2,numpy as np
        values=np.asarray(snapshot.get("values",[]),dtype=np.float32)
        if values.ndim!=2:return
        self.heat=cv2.resize(values,(GW,GH),interpolation=cv2.INTER_LINEAR).tolist()

    @property
    def known_count(self):return sum(1 for track in self.tracks if track.known)
    @property
    def unknown_count(self):return len(self.tracks)-self.known_count


# ============================ SYSTEM =================================
class PersonRecord:
    def __init__(self, name="", dept="", emp_id="", avatar=None):
        self.name, self.dept, self.emp_id = name, dept, emp_id

        self.status="active";self.last_seen=None;self.rec_count=0;self.avatar=avatar
        self.timeline=[0]*24;self.visited=[];self.stay_total=0;self.history=[]


class System(QObject):
    new_event = Signal(dict)
    heatmap_updated=Signal(str)

    def __init__(self):
        super().__init__()
        from services.frontend.video_renderer import MetadataBuffer
        self.api=ApiClient();self.async_api=AsyncApi(self.api);self.websocket=WebSocketClient();self.metadata_buffer=MetadataBuffer()
        self.websocket.message.connect(self._on_remote_message);self.websocket.connect()

        self.settings = {
            "password": "admin",
            "unlocked": False,
            "det_conf": 0.45,
            "face_th": 0.60,
            "model": "YOLOv11m-pose",
            "retention": 30,
            "sound": True,
        }

        specs = [
            ("CAM-01", "Main Lobby", "HQ — Floor 1", True),
            ("CAM-02", "Office Room A", "HQ — Floor 1", True),
            ("CAM-03", "Corridor East", "HQ — Floor 1", True),
            ("CAM-04", "Server Room", "HQ — Basement", True),
            ("CAM-05", "Meeting Room B", "HQ — Floor 2", True),
            ("CAM-06", "Entrance Gate", "Outdoor", True),
            ("CAM-07", "Parking P1", "Outdoor", False),
            ("CAM-08", "Warehouse", "HQ — Basement", True),
        ]

        self.sims = []
        self.video_clients = []
        self.visitors = {}
        self.usage = {}

        for cid, name, loc, on in specs:
            s = CameraState(cid, name, loc, on)

            self.sims.append(s)
            self.usage[cid] = 0
            if on:
                from services.frontend.video_transport import MJPEGClient
                client=MJPEGClient(cid,f"http://127.0.0.1:8001/video/{cid}",self)
                client.frame.connect(self._on_video_frame);client.online.connect(self._on_video_status);client.start();self.video_clients.append(client)

        self.enroll_sim = self.sims[0]

        self.events = []

        self.gpu, self.cpu, self.ram = 0.0, 0.0, 0.0

        self.people = []


    def _on_remote_message(self,message):
        kind=message.get("type","")
        if kind=="frame.metadata":
            self.metadata_buffer.put(message)
            camera=self.sim_by_id(message.get("camera_id"))
            if camera and camera.set_metadata(message):
                for surface in camera.surfaces:surface.update()
        elif kind=="person.identified":
            self.new_event.emit(message)
        elif kind.startswith("camera."):
            camera_id=message.get("camera_id")
            for camera in self.sims:
                if camera.id==camera_id and kind=="camera.offline":camera.clear_frame()
        elif kind.startswith("enrollment.") or kind.startswith("person."):
            self.new_event.emit(message)
        elif kind=="heatmap.updated":
            self.heatmap_updated.emit(message.get("camera_id",""))

    def _on_video_frame(self,camera_id,frame_id,timestamp,image):
        for camera in self.sims:
            if camera.id==camera_id:
                camera.frame=image;camera.online=True;camera.frame_id=frame_id;camera.frame_timestamp=timestamp
                metadata=self.metadata_buffer.match(camera_id,frame_id,timestamp)
                camera.tracks=[]
                if metadata:camera.set_metadata(metadata)
                for surface in camera.surfaces:surface.update()
                break

    def _on_video_status(self,camera_id,online):
        for camera in self.sims:
            if camera.id==camera_id:
                if not online:camera.clear_frame()
                for surface in camera.surfaces:surface.update()
                break

    @staticmethod
    def person_record(data):
        record=PersonRecord(data.get("name") or data.get("display_name") or "Unknown",data.get("department") or "",data.get("employee_id") or "")
        record.db_id=data.get("id") or data.get("person_id");record.status=data.get("status","active");record.metadata=data.get("metadata",{})
        return record

    def push_event(self, e, silent=False):
        # Faqat bugungi eventlarni ko'rsatish
        from datetime import datetime as _dt
        today_str = _dt.now().strftime("%Y-%m-%d")
        event_time = e.get("time")
        if event_time:
            if hasattr(event_time, 'strftime'):
                event_date = event_time.strftime("%Y-%m-%d")
            else:
                event_date = str(event_time)[:10]
            if event_date != today_str:
                return  # Bugun emas — o'tkazib yuborish
        e["time"] = datetime.now()

        self.events.insert(0, e)

        if len(self.events) > 500:
            self.events.pop()

        if not silent:
            self.new_event.emit(e)

    def sim_by_id(self, cid):
        for s in self.sims:
            if s.id == cid:
                return s

        return None

    @property
    def cams_online(self):
        return sum(1 for s in self.sims if s.online)

    def shutdown(self):
        self.websocket.close()
        for client in self.video_clients:client.stop()
        self.async_api.shutdown()


# ========================= BASE WIDGETS ==============================
class GradientBar(QWidget):
    def __init__(self, h, top=True, radius=10):
        super().__init__()

        self.top, self.radius = top, radius

        self.setFixedHeight(h)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 4 if top else 2, 12, 2 if top else 6)
        lay.setSpacing(8)

        self.lay = lay

    def paintEvent(self, e):
        p = QPainter(self)

        w, h, r = self.width(), self.height(), self.radius

        path = QPainterPath()

        if self.top:
            path.addRoundedRect(QRectF(0, -r, w, h + r), r, r)
        else:
            path.addRoundedRect(QRectF(0, 0, w, h + r), r, r)

        g = QLinearGradient(0, 0, 0, h)

        if self.top:
            g.setColorAt(0, QColor(6, 9, 13, 220))
            g.setColorAt(1, QColor(6, 9, 13, 0))
        else:
            g.setColorAt(0, QColor(6, 9, 13, 0))
            g.setColorAt(1, QColor(6, 9, 13, 225))

        p.fillPath(path, QBrush(g))

        p.end()


class PulsingDot(QWidget):
    def __init__(self, color="#2ecc71", size=9):
        super().__init__()

        self.base = QColor(color)
        self.sz = size

        self.setFixedSize(size + 6, size + 6)

        self.t = QTimer(self)
        self.t.setInterval(80)
        self.t.timeout.connect(self.update)
        self.t.start()

    def set_color(self, c):
        self.base = QColor(c)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        ms = QDateTime.currentMSecsSinceEpoch()
        pulse = (math.sin(ms / 300.0) + 1) / 2

        cx, cy = self.width() / 2, self.height() / 2

        c2 = QColor(self.base)
        c2.setAlpha(int(40 * pulse))

        p.setPen(Qt.NoPen)
        p.setBrush(c2)
        p.drawEllipse(QPointF(cx, cy), self.sz / 2 + 3, self.sz / 2 + 3)

        c = QColor(self.base)
        c.setAlpha(int(120 + 135 * pulse))

        p.setBrush(c)
        p.drawEllipse(QPointF(cx, cy), self.sz / 2 + pulse * 1.2, self.sz / 2 + pulse * 1.2)

        p.end()


class VideoSurface(QWidget):
    doubleClicked = Signal()

    def __init__(self, sim, radius=10, face_box=False, zoomable=False):
        super().__init__()

        self.sim, self.radius, self.face_box = sim, radius, face_box

        self.zoomable = zoomable

        self.zoom = 1.0
        self.offset = QPointF(0, 0)

        self.dragging = False
        self.drag_start = QPointF()

        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.setMouseTracking(zoomable)

        sim.surfaces.append(self)

        self.destroyed.connect(
            lambda _=None, s=sim, w=self: s.surfaces.remove(w) if w in s.surfaces else None
        )

    def wheelEvent(self, e):
        if not self.zoomable:
            return

        f = 1.12 if e.angleDelta().y() > 0 else 1 / 1.12

        self.zoom = clamp(self.zoom * f, 1.0, 4.0)

        if self.zoom <= 1.01:
            self.zoom = 1.0
            self.offset = QPointF(0, 0)

        self.setCursor(Qt.OpenHandCursor if self.zoom > 1 else Qt.ArrowCursor)

        self.update()

    def mousePressEvent(self, e):
        if self.zoomable and self.zoom > 1:
            self.dragging = True
            self.drag_start = e.position() - self.offset
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self.dragging:
            self.offset = e.position() - self.drag_start
            self.update()

    def mouseReleaseEvent(self, e):
        self.dragging = False
        self.setCursor(Qt.OpenHandCursor if self.zoom > 1 else Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, e):
        if self.zoomable:
            self.zoom = 1.0
            self.offset = QPointF(0, 0)
            self.update()
        else:
            self.doubleClicked.emit()

    def video_rect(self):
        w, h = self.width(), self.height()

        ar = FRAME_W / FRAME_H

        if w / h > ar:
            nw, nh = h * ar, h
        else:
            nw, nh = w, w / ar

        return QRectF((w - nw) / 2, (h - nh) / 2, nw, nh)

    def paintEvent(self, e):
        p = QPainter(self)

        p.setRenderHint(QPainter.SmoothPixmapTransform)

        p.fillRect(self.rect(), QColor("#05070a"))

        f = self.sim.frame

        if f is None:
            p.setPen(QColor(TH.ERR));p.setFont(QFont("Segoe UI",15,QFont.Bold))
            p.drawText(self.rect().adjusted(0,-16,0,0),Qt.AlignCenter,"⚠ NO SIGNAL")
            p.setPen(QColor(TH.DIM));p.setFont(QFont("Segoe UI",9))
            p.drawText(self.rect().adjusted(0,22,0,0),Qt.AlignCenter,"OFFLINE")
            p.end()
            return

        if self.radius:
            path = QPainterPath()
            path.addRoundedRect(QRectF(self.rect()), self.radius, self.radius)
            p.setClipPath(path)

        p.save()

        if self.zoomable and self.zoom > 1.0:
            p.translate(self.width() / 2, self.height() / 2)
            p.scale(self.zoom, self.zoom)
            p.translate(
                -self.width() / 2 + self.offset.x() / self.zoom,
                -self.height() / 2 + self.offset.y() / self.zoom,
            )

        vr = self.video_rect()

        p.drawImage(vr, f)

        s = self.sim

        if s.online:
            if s.heat_on:
                pm = QPixmap.fromImage(s.heat_image()).scaled(
                    int(vr.width()),
                    int(vr.height()),
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation,
                )

                p.setOpacity(0.6)
                p.drawPixmap(vr.toRect(), pm)
                p.setOpacity(1.0)

            if s.ai_on:
                self._draw_ai(p, vr, f.width(), f.height())

        p.restore()

        if self.zoomable and self.zoom > 1.01:
            p.setPen(QColor(TH.TXT))
            p.setFont(QFont("Segoe UI", 9, QFont.Bold))

            p.drawText(
                self.rect().adjusted(0, 8, -12, 0),
                Qt.AlignRight,
                f"🔍 {self.zoom:.1f}×  (scroll · drag · dbl-click reset)",
            )

        p.end()

    def _map(self, vr, fw, fh, bb):
        return QRectF(
            vr.x() + bb.x() / fw * vr.width(),
            vr.y() + bb.y() / fh * vr.height(),
            bb.width() / fw * vr.width(),
            bb.height() / fh * vr.height(),
        )

    def _draw_ai(self, p, vr, fw, fh):
        p.save()
        p.setRenderHint(QPainter.Antialiasing)

        font = QFont("Segoe UI", 8.5, QFont.Bold)
        p.setFont(font)

        # zy = vr.y() + self.sim.zone_y * vr.height()

        # p.setPen(QPen(QColor(255, 80, 60, 140), 1.5, Qt.DashLine))
        # p.drawLine(QPointF(vr.left(), zy), QPointF(vr.right(), zy))

        for ps in self.sim.people:
            r = self._map(vr, fw, fh, ps.bbox(fw, fh))

            col = QColor(TH.OK) if ps.known else QColor("#f59e42")

            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(col, 2))

            L = min(r.width(), r.height()) * 0.22

            for x1, y1, x2, y2, x3, y3 in [
                (r.left(), r.top() + L, r.left(), r.top(), r.left() + L, r.top()),
                (r.right() - L, r.top(), r.right(), r.top(), r.right(), r.top() + L),
                (r.right(), r.bottom() - L, r.right(), r.bottom(), r.right() - L, r.bottom()),
                (r.left() + L, r.bottom(), r.left(), r.bottom(), r.left(), r.bottom() - L),
            ]:
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
                p.drawLine(QPointF(x2, y2), QPointF(x3, y3))

            p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 70), 1))
            p.drawRect(r)

            if ps.known:
                display_name = str(ps.name)
            else:
                raw_id = str(ps.name or ps.track_id)
                if ":" in raw_id:
                    raw_id = raw_id.rsplit(":", 1)[-1]
                if raw_id.upper().startswith("UNK-"):
                    raw_id = raw_id[4:]
                elif raw_id.lower().startswith("unknown-"):
                    raw_id = raw_id[8:]
                display_name = f"UNK: {raw_id}"
            lbl = f"{display_name} · {int(ps.conf * 100)}%"

            fm = QFontMetrics(font)

            tw = fm.horizontalAdvance(lbl) + 10
            lh = fm.height() + 6

            ly = r.top() - lh - 2 if r.top() - lh - 2 > vr.top() + 2 else r.top() + 2

            p.setPen(Qt.NoPen)
            p.setBrush(QColor(col.red(), col.green(), col.blue(), 215))
            p.drawRoundedRect(QRectF(r.left(), ly, tw, lh), 4, 4)

            p.setPen(QColor("#0c1116"))
            p.drawText(QRectF(r.left() + 5, ly, tw, lh), Qt.AlignVCenter, lbl)

        p.restore()



# ============================ HEADER =================================
class Chip(QFrame):
    def __init__(self, label, icon=""):
        super().__init__()

        self.setObjectName("chip")

        h = QHBoxLayout(self)
        h.setContentsMargins(9, 4, 9, 4)
        h.setSpacing(6)

        t = QLabel(f"{icon} {label}".strip())
        t.setStyleSheet(f"color:{TH.DIM}; font-size:9px; font-weight:700;")

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setFixedSize(36, 4)
        self.bar.setTextVisible(False)

        self.val = QLabel("--")
        self.val.setStyleSheet("font-size:10px; font-weight:700;")

        h.addWidget(t)
        h.addWidget(self.bar)
        h.addWidget(self.val)

    def set_value(self, v):
        self.val.setText(f"{int(v)}%")
        self.bar.setValue(int(v))

        c = TH.ERR if v > 85 else (TH.WARN if v > 65 else TH.OK)

        self.bar.setStyleSheet(
            f"QProgressBar{{background:#232c37;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:{c};border-radius:2px;}}"
        )


class NotificationDropdown(QFrame):
    def __init__(self, mw):
        super().__init__(mw)

        self.mw = mw

        self.setFixedWidth(340)
        self.setFixedHeight(310)

        self.setObjectName("notifDrop")

        self.hide()

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)

        row = QHBoxLayout()

        t = QLabel("🔔 Notifications")
        t.setStyleSheet("font-size:12px; font-weight:800; color:white;")

        clr = QPushButton("Mark all read")
        clr.setObjectName("btnGhost")
        clr.setFixedHeight(24)
        clr.setCursor(Qt.PointingHandCursor)
        clr.clicked.connect(self.clear_all)

        row.addWidget(t)
        row.addStretch(1)
        row.addWidget(clr)

        v.addLayout(row)

        self.box = QVBoxLayout()
        self.box.setSpacing(4)
        self.box.addStretch(1)

        v.addLayout(self.box, 1)

        self.items = []

    def add_alert(self, e):
        if e.get("level") not in ("warn", "err"):
            return

        w = QFrame()
        w.setObjectName("notifItem")

        h = QHBoxLayout(w)
        h.setContentsMargins(8, 6, 8, 6)
        h.setSpacing(8)

        ic = QLabel("⚠️" if e["level"] == "warn" else "🚨")

        tx = QLabel(f"{e['cam']} — {e['person']}")
        tx.setStyleSheet(f"font-size:10px; color:{TH.TXT};")

        tm = QLabel(e["time"].strftime("%H:%M:%S"))
        tm.setStyleSheet(f"color:{TH.FAINT}; font-size:8.5px; font-family:Consolas,monospace;")

        h.addWidget(ic)
        h.addWidget(tx, 1)
        h.addWidget(tm)

        w.setCursor(Qt.PointingHandCursor)

        w.mousePressEvent = lambda ev, _=w: (self.mw.navigate("events"), self.hide())

        self.box.insertWidget(0, w)
        self.items.insert(0, w)

        while len(self.items) > 8:
            old = self.items.pop()
            old.deleteLater()

    def clear_all(self):
        for w in self.items:
            w.deleteLater()

        self.items = []

        self.mw.header._unseen = 0
        self.mw.header.badge.hide()

        self.hide()

    def toggle(self):
        if self.isVisible():
            self.hide()
            return

        gp = self.mw.header.bell.mapToGlobal(QPoint(0, self.mw.header.bell.height()))
        lp = self.mw.mapFromGlobal(gp)

        self.move(lp.x() - self.width() + self.mw.header.bell.width(), lp.y() + 4)

        self.show()
        self.raise_()


class Header(QFrame):
    def __init__(self, sys_, mw):
        super().__init__()

        self.setObjectName("header")

        self.sys, self.mw = sys_, mw

        self.setFixedHeight(64)

        h = QHBoxLayout(self)
        h.setContentsMargins(14, 0, 14, 0)
        h.setSpacing(10)

        logo = QLabel("◉")
        logo.setFixedSize(34, 34)
        logo.setStyleSheet(
            f"background:{TH.ACCENT}; border-radius:9px; color:white;"
            "font-size:16px; font-weight:800;"
        )
        logo.setAlignment(Qt.AlignCenter)

        tt = QVBoxLayout()
        tt.setSpacing(0)

        t1 = QLabel("AI Surveillance System")
        t1.setStyleSheet("font-size:13.5px; font-weight:800; color:white;")

        t2 = QLabel("Operator Console • v3.0 MUKAMMAL")
        t2.setStyleSheet(f"font-size:9px; color:{TH.DIM};")

        tt.addWidget(t1)
        tt.addWidget(t2)

        h.addWidget(logo)
        h.addLayout(tt)
        h.addSpacing(8)
        h.addStretch(1)

        # self.search = QLineEdit()  # ✅ Search removed
        # self.search.setPlaceholderText("🔍  Search events, people, cameras…   ( / )")  # ✅ Search removed
        # self.search.setMaximumWidth(340)  # ✅ Search removed
        # self.search.setMinimumWidth(160)  # ✅ Search removed
        # self.search.returnPressed.connect(lambda: mw.navigate("events"))  # ✅ Search removed

        # h.addWidget(self.search)  # ✅ Search removed
        h.addStretch(1)

        self.chip_gpu = Chip("GPU")
        self.chip_cpu = Chip("CPU")
        self.chip_ram = Chip("RAM")

        for c in (self.chip_gpu, self.chip_cpu, self.chip_ram):
            h.addWidget(c)

        self.ai_chip = QLabel("● AI LOADING…")
        self.ai_chip.setStyleSheet(
            f"color:{TH.WARN}; font-size:10px; font-weight:800;"
            f"background:#24210f; border:1px solid #443d1a;"
            "border-radius:10px; padding:4px 10px;"
        )

        h.addWidget(self.ai_chip)

        self.cam_chip = QLabel("🎥 7/8")
        self.cam_chip.setStyleSheet(
            f"color:{TH.TXT}; font-size:10px; font-weight:700;"
            f"background:{TH.CARD2}; border:1px solid {TH.BORDER};"
            "border-radius:10px; padding:4px 10px;"
        )

        h.addWidget(self.cam_chip)

        self.bell = QPushButton("🔔")
        self.bell.setFixedSize(34, 34)
        self.bell.setObjectName("bellBtn")
        self.bell.setCursor(Qt.PointingHandCursor)

        self.badge = QLabel("0", self.bell)
        self.badge.setFixedSize(16, 16)
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.setStyleSheet(
            f"background:{TH.ERR}; color:white; border-radius:8px;"
            "font-size:8.5px; font-weight:800;"
        )
        self.badge.move(17, 1)
        self.badge.hide()

        self._unseen = 0

        self.bell.clicked.connect(lambda: mw.notif.toggle())

        h.addWidget(self.bell)

        self.clock = QLabel()
        self.clock.setStyleSheet(
            f"color:{TH.TXT}; font-size:11px; font-family:Consolas,monospace;"
        )

        h.addWidget(self.clock)

        av = QLabel("OP")
        av.setFixedSize(30, 30)
        av.setAlignment(Qt.AlignCenter)
        av.setStyleSheet(
            f"background:{TH.ACCENT}; color:white; border-radius:15px;"
            "font-size:10px; font-weight:800;"
        )

        h.addWidget(av)

        un = QLabel("Operator")
        un.setStyleSheet(f"font-size:10.5px; color:{TH.DIM}; font-weight:600;")

        h.addWidget(un)

        self.tick_clock()

    def set_ai_ready(self):
        self.ai_chip.setText("● AI ACTIVE")
        self.ai_chip.setStyleSheet(
            f"color:{TH.OK}; font-size:10px; font-weight:800;"
            f"background:#16241c; border:1px solid #234433;"
            "border-radius:10px; padding:4px 10px;"
        )

    def bump(self):
        self._unseen += 1

        self.badge.setText(str(min(99, self._unseen)))
        self.badge.show()

    def tick_clock(self):
        self.clock.setText(datetime.now().strftime("%a %d %b %Y   %H:%M:%S"))

    def update_stats(self):
        self.chip_gpu.set_value(self.sys.gpu)
        self.chip_cpu.set_value(self.sys.cpu)
        self.chip_ram.set_value(self.sys.ram)

        n, tot = self.sys.cams_online, len(self.sys.sims)

        self.cam_chip.setText(f"🎥 {n}/{tot}")

        self.cam_chip.setStyleSheet(
            f"color:{TH.TXT if n == tot else TH.WARN}; font-size:10px; font-weight:700;"
            f"background:{TH.CARD2}; border:1px solid {TH.BORDER};"
            "border-radius:10px; padding:4px 10px;"
        )


# ============================ SIDEBAR ================================
class SideBar(QFrame):
    changed = Signal(str)

    ITEMS = [
        ("live", "🎥", "Dashboard"),
        ("people", "👥", "Person Management"),
        ("enroll", "🪪", "Enrollment"),
        # ("analytics", "📈", "Analytics"),  # ✅ Analytics nav removed
        ("events", "⚡", "Events"),
        ("settings", "⚙️", "Settings"),
    ]

    def __init__(self, mw):
        super().__init__()

        self.setObjectName("sidebar")

        self.mw = mw

        self.buttons = {}

        self.setMinimumWidth(210)
        self.setMaximumWidth(210)

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 14, 10, 10)
        v.setSpacing(4)

        brand = QLabel("CONTROL PANEL")
        brand.setStyleSheet(
            f"color:{TH.FAINT}; font-size:8.5px; font-weight:800;"
            "letter-spacing:2px; padding:0 0 6px 10px;"
        )

        v.addWidget(brand)

        for key, icon, text in self.ITEMS:
            b = QPushButton(f"  {icon}   {text}")

            b.setObjectName("sideBtn")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(42)
            b.setToolTip(text)

            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))

            self.buttons[key] = (b, icon, text)

            v.addWidget(b)

        v.addStretch(1)

        self.collapse = QPushButton("⟨  Collapse")
        self.collapse.setObjectName("sideBtn")
        self.collapse.setFixedHeight(38)
        self.collapse.setCursor(Qt.PointingHandCursor)
        self.collapse.clicked.connect(mw.toggle_sidebar)

        v.addWidget(self.collapse)

        self.collapsed = False

    def set_active(self, key):
        for k, (b, _, _) in self.buttons.items():
            b.setChecked(k == key)

    def set_collapsed(self, c):
        self.collapsed = c

        for k, (b, icon, text) in self.buttons.items():
            b.setText(icon if c else f"  {icon}   {text}")
            b.setProperty("collapsed", c)
            b.style().unpolish(b)
            b.style().polish(b)

        self.collapse.setText("⟩" if c else "⟨  Collapse")


# ========================= RIGHT PANEL ===============================
class RightPanel(QFrame):
    def __init__(self, sys_):
        super().__init__()

        self.setObjectName("rightPanel")

        self.sys = sys_

        self.setMinimumWidth(250)

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 14, 12, 10)
        v.setSpacing(10)

        t = QLabel("LIVE STATUS")
        t.setStyleSheet(
            f"color:{TH.FAINT}; font-size:8.5px; font-weight:800; letter-spacing:2px;"
        )

        v.addWidget(t)

        row = QHBoxLayout()
        row.setSpacing(8)

        self.known_card = self._stat("👤 KNOWN", TH.OK)
        self.unk_card = self._stat("❓ UNKNOWN", "#f59e42")

        row.addWidget(self.known_card[0])
        row.addWidget(self.unk_card[0])

        v.addLayout(row)

        v.addWidget(self._sep("SYSTEM"))

        self.b_gpu = self._meter("GPU")
        self.b_cpu = self._meter("CPU")
        self.b_fps = self._meter("FPS")

        for w in (self.b_gpu[0], self.b_cpu[0], self.b_fps[0]):
            v.addWidget(w)

        v.addWidget(self._sep("ALERTS"))

        self.alerts = QVBoxLayout()
        self.alerts.setSpacing(5)

        v.addLayout(self.alerts)

        v.addWidget(self._sep("RECENT EVENTS"))

        self.evbox = QVBoxLayout()
        self.evbox.setSpacing(2)
        self.evbox.addStretch(1)

        wrap = QWidget()
        wrap.setLayout(self.evbox)

        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(wrap)
        sc.setFrameShape(QFrame.NoFrame)

        v.addWidget(sc, 1)

        self._alert_items, self._ev_items = [], []

    def _sep(self, txt):
        l = QLabel(txt)

        l.setStyleSheet(
            f"color:{TH.FAINT}; font-size:8px; font-weight:800;"
            "letter-spacing:1.5px; padding-top:6px;"
        )

        return l

    def _stat(self, label, color):
        f = QFrame()
        f.setObjectName("statCard")

        lay = QVBoxLayout(f)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)

        num = QLabel("0")
        num.setStyleSheet(f"color:{color}; font-size:20px; font-weight:800;")

        cap = QLabel(label)
        cap.setStyleSheet(f"color:{TH.DIM}; font-size:8.5px; font-weight:700;")

        lay.addWidget(num)
        lay.addWidget(cap)

        return (f, num)

    def _meter(self, label):
        w = QWidget()

        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        l = QLabel(label)
        l.setFixedWidth(28)
        l.setStyleSheet(f"color:{TH.DIM}; font-size:9px; font-weight:700;")

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setTextVisible(False)
        bar.setFixedHeight(6)

        val = QLabel("--")
        val.setFixedWidth(44)
        val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        val.setStyleSheet(f"color:{TH.TXT}; font-size:9.5px; font-family:Consolas,monospace;")

        h.addWidget(l)
        h.addWidget(bar, 1)
        h.addWidget(val)

        return (w, bar, val)

    def refresh(self):
        sims = [s for s in self.sys.sims if s.online]

        self.known_card[1].setText(str(sum(s.known_count for s in sims)))
        self.unk_card[1].setText(str(sum(s.unknown_count for s in sims)))

        for (bar, val), v in [
            ((self.b_gpu[1], self.b_gpu[2]), self.sys.gpu),
            ((self.b_cpu[1], self.b_cpu[2]), self.sys.cpu),
        ]:
            bar.setValue(int(v))
            val.setText(f"{int(v)}%")

            c = TH.ERR if v > 85 else (TH.WARN if v > 65 else TH.ACCENT)

            bar.setStyleSheet(
                f"QProgressBar{{background:#232c37;border-radius:3px;}}"
                f"QProgressBar::chunk{{background:{c};border-radius:3px;}}"
            )

        avg = sum(s.fps for s in sims) / max(1, len(sims))

        self.b_fps[1].setValue(int(avg / 30 * 100))
        self.b_fps[2].setText(f"{avg:.1f}")
        self.b_fps[2].setStyleSheet(
            f"color:{TH.OK}; font-size:9.5px; font-family:Consolas,monospace;"
        )

    def add_event(self, e):
        # Normalize timestamp - handle both datetime objects and ISO string timestamps
        ts = e.get("time")
        if isinstance(ts, str):
            try:
                from datetime import datetime
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                e["time"] = ts
            except Exception:
                ts = None
        if ts is None or not hasattr(ts, "strftime"):
            ts = datetime.now()
            e["time"] = ts

        lvl = e.get("level", "info")

        if lvl in ("warn", "err"):
            col = TH.WARN if lvl == "warn" else TH.ERR

            f = QFrame()
            f.setObjectName("alertItem")
            f.setStyleSheet(
                f"background:{TH.CARD}; border-left:3px solid {col};"
                "border-radius:6px; padding:2px;"
            )

            h = QHBoxLayout(f)
            h.setContentsMargins(8, 6, 8, 6)
            h.setSpacing(6)

            tx = QLabel(f"{e['cam']} — {e['person']}")
            tx.setStyleSheet(f"font-size:9.5px; color:{TH.TXT};")

            tm = QLabel(ts.strftime("%H:%M:%S"))
            tm.setStyleSheet(
                f"color:{TH.FAINT}; font-size:8.5px; font-family:Consolas,monospace;"
            )

            h.addWidget(tx, 1)
            h.addWidget(tm)

            self.alerts.insertWidget(0, f)
            self._alert_items.insert(0, f)

            while len(self._alert_items) > 3:
                old = self._alert_items.pop()
                old.deleteLater()

        dotc = {
            "ok": TH.OK,
            "warn": TH.WARN,
            "err": TH.ERR,
            "info": TH.ACCENT,
        }.get(lvl, TH.DIM)

        w = QWidget()

        h = QHBoxLayout(w)
        h.setContentsMargins(2, 3, 2, 3)
        h.setSpacing(6)

        tm = QLabel(ts.strftime("%H:%M:%S"))
        tm.setStyleSheet(
            f"color:{TH.FAINT}; font-size:8.5px; font-family:Consolas,monospace;"
        )

        dot = QLabel("●")
        dot.setStyleSheet(f"color:{dotc}; font-size:7px;")

        tx = QLabel(f"{e['cam']} · {e['person']}")
        tx.setStyleSheet(f"color:{TH.DIM}; font-size:9.5px;")

        h.addWidget(tm)
        h.addWidget(dot)
        h.addWidget(tx, 1)

        self.evbox.insertWidget(0, w)
        self._ev_items.insert(0, w)

        while len(self._ev_items) > 25:
            old = self._ev_items.pop()
            old.deleteLater()


# ========================== CAMERA CARD ==============================
class QuickInfo(QFrame):
    def __init__(self, sim):
        super().__init__()

        self.sim = sim

        self.setObjectName("quickInfo")

        self.setFixedWidth(210)

        g = QGridLayout(self)
        g.setContentsMargins(12, 10, 12, 10)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(6)

        t = QLabel("QUICK INFO")
        t.setStyleSheet(
            f"color:{TH.ACC2}; font-size:8px; font-weight:800; letter-spacing:1.5px;"
        )

        g.addWidget(t, 0, 0, 1, 2)

        self.vals = {}

        rows = [
            ("camera", "Camera"),
            ("status", "Status"),
            ("fps", "Current FPS"),
            ("people", "Detected People"),
            ("split", "Known / Unknown"),
            ("rec", "Recording"),
            ("ai", "AI Status"),
            ("lat", "Latency"),
            ("loss", "Packet Loss"),
            ("infer", "AI Inference"),
        ]

        for i, (key, label) in enumerate(rows, start=1):
            l = QLabel(label)
            l.setStyleSheet(f"color:{TH.DIM}; font-size:9.5px;")

            v = QLabel("--")
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            v.setStyleSheet(f"color:{TH.TXT}; font-size:9.5px; font-weight:700;")

            g.addWidget(l, i, 0)
            g.addWidget(v, i, 1)

            self.vals[key] = v

    def refresh(self):
        s = self.sim
        on = s.online

        self.vals["camera"].setText(f"{s.id} — {s.name}")

        # self.vals["status"].setText("🟢 Online" if on else "🔴 Offline")  # ✅ Status label removed
        self.vals["status"].setStyleSheet(
            f"color:{TH.OK if on else TH.ERR}; font-size:9.5px; font-weight:700;"
        )

        self.vals["fps"].setText(f"{s.fps:.1f}" if on else "—")

        self.vals["people"].setText(str(len(s.people)) if on else "—")

        self.vals["split"].setText(f"{s.known_count} / {s.unknown_count}" if on else "—")

        self.vals["rec"].setText("🔴 REC" if s.recording and on else "OFF")

        self.vals["ai"].setText("ON" if s.ai_on else "OFF")
        self.vals["ai"].setStyleSheet(
            f"color:{TH.OK if s.ai_on else TH.FAINT}; font-size:9.5px; font-weight:700;"
        )

        self.vals["lat"].setText(f"{s.latency:.0f} ms" if on else "—")
        self.vals["loss"].setText(f"{s.packet_loss:.1f} %" if on else "—")
        self.vals["infer"].setText(f"{s.infer_ms:.0f} ms" if on else "—")


class CameraCard(QFrame):
    def __init__(self, sim, hub):
        super().__init__()

        self.setObjectName("camCard")

        self.sim, self.hub = sim, hub

        self.setMinimumSize(340, 215)

        self.glow = QGraphicsDropShadowEffect()
        self.glow.setBlurRadius(0)
        self.glow.setColor(QColor(TH.ACCENT))
        self.glow.setOffset(0, 0)

        self.setGraphicsEffect(self.glow)

        self._glow_anim = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.surface = VideoSurface(sim)

        lay.addWidget(self.surface)

        self.top = GradientBar(32, top=True)
        self.top.setParent(self)

        self.lbl_id = QLabel(sim.id)
        self.lbl_id.setStyleSheet("color:white; font-size:11px; font-weight:800;")

        self.lbl_loc = QLabel(f"· {sim.location}")
        self.lbl_loc.setStyleSheet(f"color:{TH.DIM}; font-size:10px;")

        self.dot = PulsingDot(TH.OK)

        # self.lbl_status = QLabel("Online")  # ✅ Status label removed
        # self.lbl_status.setStyleSheet(f"color:{TH.OK}; font-size:10px; font-weight:700;")  # ✅ Status label removed

        self.top.lay.addWidget(self.lbl_id)
        self.top.lay.addWidget(self.lbl_loc)
        self.top.lay.addStretch(1)
        self.top.lay.addWidget(self.dot)
        # self.top.lay.addWidget(self.lbl_status)  # ✅ Status label removed

        self.bot = GradientBar(30, top=False)
        self.bot.setParent(self)

        self.lbl_fps = QLabel()
        self.lbl_res = QLabel(sim.res)
        self.lbl_ppl = QLabel()
        self.lbl_ai = QLabel()
        self.lbl_conn = QLabel()

        for l in (self.lbl_fps, self.lbl_res, self.lbl_ppl, self.lbl_ai, self.lbl_conn):
            l.setStyleSheet(f"color:{TH.DIM}; font-size:9.5px; font-weight:600;")
            self.bot.lay.addWidget(l)

        self.bot.lay.addStretch(1)

        self.lbl_rec = QLabel("● REC")
        self.lbl_rec.setStyleSheet(f"color:{TH.ERR}; font-size:9px; font-weight:800;")

        self.bot.lay.addWidget(self.lbl_rec)

        self.rec_t = QTimer(self)
        self.rec_t.setInterval(600)
        self.rec_t.timeout.connect(
            lambda: self.lbl_rec.setVisible(not self.lbl_rec.isVisible())
        )
        self.rec_t.start()

        self.tb = QFrame(self)
        self.tb.setObjectName("camToolbar")

        th = QHBoxLayout(self.tb)
        th.setContentsMargins(4, 4, 4, 4)
        th.setSpacing(4)

        def tool(txt, tip, check=False):
            b = QToolButton()

            b.setText(txt)
            b.setToolTip(tip)
            b.setObjectName("camTool")
            b.setFixedSize(30, 30)
            b.setCheckable(check)
            b.setCursor(Qt.PointingHandCursor)

            th.addWidget(b)

            return b

        self.btn_ai = tool("🤖", "AI Overlay", True)
        self.btn_heat = tool("🔥", "Heatmap", True)
        self.btn_snap = tool("📸", "Snapshot")
        self.btn_full = tool("⛶", "Fullscreen")
        self.btn_more = tool("⋮", "More")

        self.tb.hide()

        self.btn_ai.toggled.connect(lambda c: self._set_ai(c))
        self.btn_heat.toggled.connect(lambda c: self._set_heat(c))
        self.btn_snap.clicked.connect(lambda: hub.snapshot(sim))
        self.btn_full.clicked.connect(lambda: hub.open_fullscreen(sim))
        self.btn_more.clicked.connect(self._menu)
        self.hub.sys.heatmap_updated.connect(lambda camera_id:self._load_heatmap() if camera_id==self.sim.id and self.sim.heat_on else None)

        self.qi = QuickInfo(sim)
        self.qi.setParent(self)
        self.qi.hide()

        self.surface.doubleClicked.connect(lambda: hub.open_fullscreen(sim))

        self.refresh()

    def _set_ai(self, c):
        self.sim.ai_on = c
        self.refresh()

    def _set_heat(self, c):
        self.sim.heat_on = c
        if c:self._load_heatmap()
        self.refresh()

    def _load_heatmap(self):
        self.hub.sys.async_api.submit(lambda:self.hub.sys.api.get_heatmap(self.sim.id,"live"),lambda snapshot:(self.sim.apply_heatmap(snapshot),self.refresh()),lambda error:self.hub.toast(f"Heatmap API: {error}"))

    def _menu(self):
        m = QMenu(self)

        a1 = m.addAction("📷   Camera details")
        a4 = m.addAction("⎘   Copy Camera ID")

        act = m.exec(QCursor.pos())

        s = self.sim

        if act == a1:
            self.qi.setVisible(not self.qi.isVisible())

        elif act == a4:
            QApplication.clipboard().setText(s.id)

    def resizeEvent(self, e):
        if hasattr(self, "top"):
            self.top.setGeometry(0, 0, self.width(), 32)
            self.bot.setGeometry(0, self.height() - 30, self.width(), 30)

            self.tb.move(self.width() - self.tb.sizeHint().width() - 8, 38)

            self.qi.move(
                self.width() - self.qi.width() - 8,
                38 + self.tb.height() + 6,
            )

        super().resizeEvent(e)

    def enterEvent(self, e):
        self.tb.show()
        self.qi.show()
        self.qi.refresh()

        a = QPropertyAnimation(self.glow, b"blurRadius")
        a.setDuration(150)
        a.setEndValue(18)
        a.start()

        self._glow_anim = a

    def leaveEvent(self, e):
        self.tb.hide()
        self.qi.hide()

        a = QPropertyAnimation(self.glow, b"blurRadius")
        a.setDuration(150)
        a.setEndValue(0)
        a.start()

        self._glow_anim = a

    def refresh(self):
        s = self.sim
        on = s.online

        # self.lbl_status.setText("Online" if on else "Offline")  # ✅ Status label removed
        # self.lbl_status.setStyleSheet(  # ✅ Status label removed
            # f"color:{TH.OK if on else TH.ERR}; font-size:10px; font-weight:700;"  # ✅ Status label removed
        # )  # ✅ Status label removed

        self.dot.set_color(TH.OK if on else TH.ERR)

        self.lbl_fps.setText(f"{s.fps:.1f} FPS" if on else "-- FPS")

        self.lbl_ppl.setText(f"👥 {len(s.people)}" if on else "👥 —")

        self.lbl_ai.setText("🤖 AI ON" if s.ai_on else "🤖 AI OFF")
        self.lbl_ai.setStyleSheet(
            f"color:{TH.OK if s.ai_on else TH.FAINT}; font-size:9.5px; font-weight:700;"
        )

        q = s.conn_quality

        bars = "▂▄▆█"

        self.lbl_conn.setText(bars[:q] + "░" * (4 - q) if on else "░░░░")

        self.lbl_conn.setStyleSheet(
            f"color:{TH.OK if q >= 3 else (TH.WARN if q >= 2 else TH.ERR)}; font-size:9px;"
        )

        self.setProperty("offline", not on)
        self.style().unpolish(self)
        self.style().polish(self)

        for b, v in ((self.btn_ai, s.ai_on), (self.btn_heat, s.heat_on)):
            b.blockSignals(True)
            b.setChecked(v)
            b.blockSignals(False)

        self.qi.refresh()


# =========================== FULLSCREEN ==============================
class FullscreenCam(QDialog):
    def __init__(self, sim, hub):
        super().__init__(None)

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setStyleSheet("background:#000;")

        self.sim, self.hub = sim, hub

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        top = QFrame()
        top.setFixedHeight(56)
        top.setStyleSheet(
            f"background:{TH.PANEL}; border-bottom:1px solid {TH.BORDER};"
        )

        h = QHBoxLayout(top)
        h.setContentsMargins(18, 0, 18, 0)
        h.setSpacing(10)

        self.lbl_id = QLabel(sim.id)
        self.lbl_id.setStyleSheet("font-size:15px; font-weight:800; color:white;")

        self.lbl_loc = QLabel(f"{sim.name} — {sim.location}")
        self.lbl_loc.setStyleSheet(f"color:{TH.DIM}; font-size:11px;")

        self.lbl_st = QLabel()

        h.addWidget(self.lbl_id)
        h.addWidget(self.lbl_loc)
        h.addStretch(1)
        h.addWidget(self.lbl_st)

        back = QPushButton("← Back")
        back.setObjectName("btnGhost")
        back.setCursor(Qt.PointingHandCursor)
        back.clicked.connect(self.accept)

        h.addWidget(back)

        v.addWidget(top)

        self.surface = VideoSurface(sim, radius=0, zoomable=True)

        v.addWidget(self.surface, 1)

        bot = QFrame()
        bot.setFixedHeight(64)
        bot.setStyleSheet(f"background:{TH.PANEL}; border-top:1px solid {TH.BORDER};")

        bh = QHBoxLayout(bot)
        bh.setSpacing(8)

        def fbtn(txt, tip, check=False):
            b = QPushButton(txt)

            b.setObjectName("btnGhost")
            b.setToolTip(tip)
            b.setCheckable(check)
            b.setFixedHeight(38)
            b.setCursor(Qt.PointingHandCursor)

            bh.addWidget(b)

            return b

        bh.addStretch(1)

        self.b_ai = fbtn("🤖  AI", "AI Overlay", True)
        self.b_heat = fbtn("🔥  Heatmap", "Heatmap", True)
        self.b_snap = fbtn("📸  Snapshot", "Snapshot")
        self.b_back = fbtn("←  Back", "Exit fullscreen")

        bh.addStretch(1)

        self.b_ai.toggled.connect(lambda c: setattr(sim, "ai_on", c))
        self.b_heat.toggled.connect(self._set_heat)
        self.b_snap.clicked.connect(lambda: hub.snapshot(sim))
        self.b_back.clicked.connect(self.accept)

        v.addWidget(bot)

        self.refresh()

    def _set_heat(self,enabled):
        self.sim.heat_on=enabled
        if enabled:self.hub.sys.async_api.submit(lambda:self.hub.sys.api.get_heatmap(self.sim.id,"live"),lambda snapshot:(self.sim.apply_heatmap(snapshot),self.refresh()),lambda error:self.hub.toast(f"Heatmap API: {error}"))

    def refresh(self):
        s = self.sim

        self.lbl_st.setText("🟢 ONLINE" if s.online else "🔴 OFFLINE")
        self.lbl_st.setStyleSheet(
            f"color:{TH.OK if s.online else TH.ERR}; font-size:11px; font-weight:800;"
        )

        for b, val in ((self.b_ai, s.ai_on), (self.b_heat, s.heat_on)):
            b.blockSignals(True)
            b.setChecked(val)
            b.blockSignals(False)

    def accept(self):
        self.surface.zoom = 1.0
        self.surface.offset = QPointF(0, 0)

        super().accept()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Escape, Qt.Key_Back):
            self.accept()


# ============================= CHARTS ================================
class ChartCard(QFrame):
    def __init__(self, title):
        super().__init__()

        self.setObjectName("chartCard")

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)

        t = QLabel(title)
        t.setStyleSheet(
            f"color:{TH.DIM}; font-size:10px; font-weight:800; letter-spacing:1px;"
        )

        v.addWidget(t)

        self.body = v


class LineChart(QWidget):
    def __init__(self):
        super().__init__()

        self.series = []

        self.setMinimumHeight(120)

    def set_series(self, series):
        self.series = series
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()

        p.setPen(QPen(QColor(TH.BORDER), 1))

        for i in range(1, 4):
            p.drawLine(0, h * i // 4, w, h * i // 4)

        for data, color, name in self.series:
            if len(data) < 2:
                continue

            mx = max(1, max(data))

            pts = [
                QPointF(i / (len(data) - 1) * w, h - 4 - (v / mx) * (h - 10))
                for i, v in enumerate(data)
            ]

            path = QPainterPath()
            path.moveTo(pts[0])

            for q in pts[1:]:
                path.lineTo(q)

            p.setPen(QPen(QColor(color), 2))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

            fill = QPainterPath(path)
            fill.lineTo(w, h)
            fill.lineTo(0, h)
            fill.closeSubpath()

            c = QColor(color)
            c.setAlpha(40)

            p.fillPath(fill, QBrush(c))

        p.end()


class BarChart(QWidget):
    def __init__(self, horizontal=False):
        super().__init__()
        self.values, self.labels = [], []
        self.color, self.horizontal = TH.ACCENT, horizontal
        self._value_formatter = None
        self._max_value = None
        self._top_values = None
        self._show_minutes = False
        self.setMinimumHeight(120)

    def set_data(self, values, labels, color=None, formatter=None, max_value=None, top_values=None):
        self.values, self.labels = values, labels
        if color: self.color = color
        self._value_formatter = formatter
        self._max_value = max_value
        self._top_values = top_values
        self.update()

    def _label_text(self, v):
        if callable(self._value_formatter): return self._value_formatter(v)
        if self._show_minutes: return f"{int(v)}m"
        return str(int(v))

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        n = len(self.values)
        if n == 0: p.end(); return
        mx = self._max_value if self._max_value else max(1, max(self.values))
        p.setFont(QFont("Segoe UI", 7))
        if self.horizontal:
            bh = max(6, (h - 4 * (n - 1)) // n)
            for i, (v, l) in enumerate(zip(self.values, self.labels)):
                y = i * (bh + 4); bw = int(v / mx * (w - 70))
                p.setPen(QColor(TH.DIM)); p.drawText(QRect(0, y, 52, bh), Qt.AlignVCenter | Qt.AlignRight, l)
                p.setPen(Qt.NoPen); p.setBrush(QColor(TH.CARD2)); p.drawRoundedRect(56, y, w - 56, bh, 3, 3)
                p.setBrush(QColor(self.color))
                if bw > 2: p.drawRoundedRect(56, y, bw, bh, 3, 3)
                if v > 0:
                    text = self._label_text(v)
                    if text:
                        p.setPen(QColor("#ffffff")); fnt = p.font(); fnt.setPixelSize(9); fnt.setBold(True); p.setFont(fnt)
                        fm = p.fontMetrics(); tw = fm.horizontalAdvance(text)
                        tx = 56 + bw - tw - 6 if bw > tw + 10 else 56 + bw + 4
                        p.drawText(tx, y + bh // 2 + fm.ascent() // 2 - 2, text)
        else:
            bw = max(4, (w - 6 * (n - 1)) // n)
            for i, (v, l) in enumerate(zip(self.values, self.labels)):
                x = i * (bw + 6); bh2 = int(v / mx * (h - 16))
                p.setPen(Qt.NoPen); p.setBrush(QColor(TH.CARD2)); p.drawRoundedRect(x, 0, bw, h - 14, 3, 3)
                p.setBrush(QColor(self.color))
                if bh2 > 2: p.drawRoundedRect(x, h - 14 - bh2, bw, bh2, 3, 3)
                if self._top_values and i < len(self._top_values) and self._top_values[i] > 0:
                    tv = self._top_values[i]
                    p.setPen(QColor("#f59e0b")); fnt = p.font(); fnt.setPixelSize(10); fnt.setBold(True); p.setFont(fnt)
                    tt = f"{int(tv)} kishi"; fm = p.fontMetrics(); tw = fm.horizontalAdvance(tt)
                    p.drawText(x + (bw - tw) // 2, max(10, (h - 14 - bh2) - 6), tt)
                if v > 0:
                    text = self._label_text(v)
                    if text:
                        p.setPen(QColor("#ffffff")); fnt = p.font(); fnt.setPixelSize(9); fnt.setBold(True); p.setFont(fnt)
                        fm = p.fontMetrics(); tw = fm.horizontalAdvance(text); tx = x + (bw - tw) // 2
                        ty = (h - 14 - bh2) + bh2 // 2 + fm.ascent() // 2 - 2 if bh2 > 18 else (h - 14 - bh2) - 4
                        p.drawText(tx, ty, text)
                p.setPen(QColor(TH.FAINT)); p.setFont(QFont("Segoe UI", 7))
                p.drawText(QRect(x - 6, h - 13, bw + 12, 12), Qt.AlignHCenter, l)
        p.end()


class Donut(QWidget):
    def __init__(self):
        super().__init__()

        self.known, self.unknown = 1, 0

        self.setMinimumHeight(120)

    def set_values(self, k, u):
        self.known, self.unknown = k, u
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()

        s = min(w * 0.55, h - 30)

        x, y = (w - s) / 2, 6

        total = max(1, self.known + self.unknown)

        a0 = 90 * 16
        a1 = int(self.known / total * 360 * 16)

        p.setPen(Qt.NoPen)

        p.setBrush(QColor(TH.OK))
        p.drawPie(QRectF(x, y, s, s), a0, a1)

        p.setBrush(QColor("#f59e42"))
        p.drawPie(QRectF(x, y, s, s), a0 + a1, 360 * 16 - a1)

        p.setBrush(QColor(TH.CARD))
        p.drawEllipse(QRectF(x + s * 0.22, y + s * 0.22, s * 0.56, s * 0.56))

        p.setPen(QColor(TH.TXT))
        p.setFont(QFont("Segoe UI", 13, QFont.Bold))
        p.drawText(QRectF(x, y, s, s), Qt.AlignCenter, str(self.known + self.unknown))

        p.setFont(QFont("Segoe UI", 8))

        p.setPen(QColor(TH.OK))
        p.drawText(10, y + s + 14, f"● Known {self.known}")

        p.setPen(QColor("#f59e42"))
        p.drawText(w // 2, y + s + 14, f"● Unknown {self.unknown}")

        p.end()


class HeatSummary(QWidget):
    def __init__(self, sims):
        super().__init__()

        self.sims = sims

        self.setMinimumHeight(120)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        w, h = self.width(), self.height()

        cols, rows = 4, 2

        tw, th = (w - 3 * (cols - 1)) // cols, (h - 3 * (rows - 1) - 14) // rows

        p.setFont(QFont("Segoe UI", 7))

        for i, s in enumerate(self.sims[:8]):
            cx, cy = (i % cols) * (tw + 3), (i // cols) * (th + 3 + 11)

            p.fillRect(cx, cy, tw, th, QColor("#0a0d11"))

            if s.online:
                pm = QPixmap.fromImage(s.heat_image()).scaled(
                    tw,
                    th,
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation,
                )

                p.drawPixmap(cx, cy, pm)

            p.setPen(QColor(TH.DIM))
            p.drawText(cx + 3, cy + th + 9, s.id)

        p.end()


# ============================== PAGES ================================
class Page(QWidget):
    def __init__(self):
        super().__init__()

        self.setObjectName("page")

        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(14, 12, 14, 12)
        self.v.setSpacing(10)

    def title_row(self, title, sub=""):
        row = QHBoxLayout()
        row.setSpacing(10)

        t = QLabel(title)
        t.setStyleSheet("font-size:16px; font-weight:800; color:white;")

        row.addWidget(t)

        if sub:
            s = QLabel(sub)
            s.setStyleSheet(f"color:{TH.FAINT}; font-size:10px;")
            row.addWidget(s)

        row.addStretch(1)

        self.v.addLayout(row)

        return row


class DashboardPage(Page):
    def __init__(self, hub):
        super().__init__()

        self.hub = hub

        strip = QHBoxLayout()

        t = QLabel("LIVE WALL")
        t.setStyleSheet(
            "font-size:11px; font-weight:800; color:white; letter-spacing:1px;"
        )

        self.info = QLabel()
        self.info.setStyleSheet(f"color:{TH.DIM}; font-size:10px;")

        strip.addWidget(t)
        strip.addWidget(self.info)
        strip.addStretch(1)

        ws = QPushButton("📸 Wall Snapshot")
        ws.setObjectName("btnGhost")
        ws.setFixedHeight(26)
        ws.setCursor(Qt.PointingHandCursor)
        ws.clicked.connect(self.wall_snapshot)

        strip.addWidget(ws)

        self.v.addLayout(strip)

        g = QGridLayout()
        g.setSpacing(10)

        self.cards = []

        for i, sim in enumerate(hub.sys.sims[:6]):
            c = CameraCard(sim, hub)

            self.cards.append(c)

            g.addWidget(c, i // 2, i % 2)

        for i in range(2):
            g.setColumnStretch(i, 1)

        for i in range(3):
            g.setRowStretch(i, 1)

        self.v.addLayout(g, 1)

    def refresh(self):
        sy = self.hub.sys

        self.info.setText(
            f"  {sy.cams_online}/{len(sy.sims)} cameras online · "
            f"{datetime.now().strftime('%d %b %Y')}"
        )

    def wall_snapshot(self):
        os.makedirs("snapshots", exist_ok=True)

        cols, rows = 2, 3

        tw, th = FRAME_W, FRAME_H

        grid = QPixmap(cols * tw, rows * th)
        grid.fill(QColor("#000"))

        p = QPainter(grid)

        for i, sim in enumerate(self.hub.sys.sims[:6]):
            if sim.frame:
                p.drawImage(i % cols * tw, i // cols * th, sim.frame)

        p.end()

        fn = f"snapshots/wall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        grid.save(fn)

        self.hub.toast(f"📸 Wall snapshot saved — {fn}")


class LivePage(Page):
    def __init__(self, hub):
        super().__init__()

        self.hub = hub

        row = self.title_row(
            "Dashboard",
            "Double-click → fullscreen · scroll → zoom",
        )

        # self.search = QLineEdit()  # ✅ Search removed
        # self.search.setPlaceholderText("🔍 Filter cameras…")  # ✅ Search removed
        # self.search.setMaximumWidth(200)  # ✅ Search removed
        # self.search.textChanged.connect(self.filter_cards)  # ✅ Search removed

        # row.addWidget(self.search)  # ✅ Search removed

        self.layout_cb = QComboBox()
        self.layout_cb.addItems(["3 × 3", "2 × 3", "3 × 2", "2 × 2", "1 × 1"])
        self.layout_cb.setCurrentText("3 × 3")
        self.layout_cb.currentTextChanged.connect(self.change_layout)

        row.addWidget(self.layout_cb)

        self.grid = QGridLayout()
        self.grid.setSpacing(10)

        self.cards = []

        for i, sim in enumerate(hub.sys.sims):
            c = CameraCard(sim, hub)

            self.cards.append(c)

            self.grid.addWidget(c, i // 3, i % 3)

        for i in range(3):
            self.grid.setColumnStretch(i, 1)

        for i in range(3):
            self.grid.setRowStretch(i, 1)

        self.v.addLayout(self.grid, 1)

    def filter_cards(self):
        # q = self.search.text().lower()  # ✅ Search removed

        for card in self.cards:
            s = card.sim

            card.setVisible(q in f"{s.id} {s.name} {s.location}".lower())

    def change_layout(self, txt):
        cols = int(txt.split("×")[0].strip())

        for i in reversed(range(self.grid.count())):
            item = self.grid.itemAt(i)

            if item.widget():
                self.grid.removeWidget(item.widget())

        visible = [c for c in self.cards if not c.isHidden()]

        for i, card in enumerate(visible):
            self.grid.addWidget(card, i // cols, i % cols)

        rows = (len(visible) + cols - 1) // cols

        for i in range(10):
            self.grid.setColumnStretch(i, 1 if i < cols else 0)
            self.grid.setRowStretch(i, 1 if i < rows else 0)


# class AnalyticsPage(Page):
#     def __init__(self, hub):
#         super().__init__()
#         self.hub = hub
#         self.title_row("Analytics", "● real-time")
#         g = QGridLayout(); g.setSpacing(10)
# 
#         self.occ = LineChart()
#         c1 = ChartCard("OCCUPANCY (people, all cameras)"); c1.body.addWidget(self.occ)
# 
#         self.gpu_fps = LineChart()
#         c2 = ChartCard("GPU %  /  FPS"); c2.body.addWidget(self.gpu_fps)
# 
#         self.donut = Donut()
#         c3 = ChartCard("KNOWN vs UNKNOWN"); c3.body.addWidget(self.donut)
# 
#         self.visitors = BarChart()
#         c4 = ChartCard("VISITORS PER CAMERA"); c4.body.addWidget(self.visitors)
# 
#         self.peak = BarChart()
#         c5 = ChartCard("PEAK HOURS · aniq vaqt")
#         peak_scroll = QScrollArea()
#         peak_scroll.setWidgetResizable(True)
#         peak_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
#         peak_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
#         peak_scroll.setMinimumHeight(200)
#         peak_container = QWidget(); peak_container.setMinimumWidth(1400)
#         peak_layout = QHBoxLayout(peak_container); peak_layout.setContentsMargins(4, 4, 4, 4)
#         peak_layout.addWidget(self.peak)
#         peak_scroll.setWidget(peak_container); c5.body.addWidget(peak_scroll)
# 
#         self.heatsum = HeatSummary(hub.sys.sims)
#         c7 = ChartCard("HEATMAP SUMMARY"); c7.body.addWidget(self.heatsum)
# 
#         g.addWidget(c1, 0, 0, 1, 2)
#         g.addWidget(c3, 0, 2)
#         g.addWidget(c2, 0, 3)
#         g.addWidget(c4, 1, 0)
#         g.addWidget(c5, 1, 1, 1, 2)
#         g.addWidget(c7, 1, 3)
#         for i in range(4): g.setColumnStretch(i, 1)
#         g.setRowStretch(0, 1); g.setRowStretch(1, 1)
#         self.v.addLayout(g, 1)
# 
#         self.occ_d, self.gpu_d, self.fps_d = [], [], []
#         self.push()
# 
#     def push(self):
#         sy = self.hub.sys
#         sims = [s for s in sy.sims if s.online]
# 
#         occ = sum(len(s.people) for s in sims)
#         self.occ_d.append(occ)
#         if len(self.occ_d) > 90: self.occ_d.pop(0)
#         self.occ.set_series([(self.occ_d, TH.ACCENT, "occ")])
# 
#         self.gpu_d.append(sy.gpu)
#         self.fps_d.append(sum(s.fps for s in sims) / max(1, len(sims)))
#         if len(self.gpu_d) > 90: self.gpu_d.pop(0); self.fps_d.pop(0)
#         self.gpu_fps.set_series([(self.gpu_d, TH.WARN, "gpu"), (self.fps_d, TH.OK, "fps")])
# 
#         self.donut.set_values(sum(s.known_count for s in sims), sum(s.unknown_count for s in sims))
# 
#         self.visitors.set_data(
#             [sy.visitors.get(s.id, 0) for s in sy.sims[:6]],
#             [s.id[-2:] for s in sy.sims[:6]],
#             formatter=lambda v: str(int(v)), max_value=10,
#         )
#         self.visitors._show_minutes = True
# 
#         peak_minutes, peak_people = [0]*24, [0]*24
#         peak_labels = [f"{i:02d}:00" if i % 3 == 0 else "" for i in range(24)]
#         try:
#             today_str = datetime.now().strftime("%Y-%m-%d")
#             analytics rows are supplied by the system metrics API
#                 h = int(row.get("hour", 0))
#                 occ_sum = int(row.get("occupancy_sum", 0) or 0)
#                 samples = max(1, int(row.get("int_samples", 0) or 1))
#                 maxp = int(row.get("max_occupancy", 0) or 0)
#                 minutes = max(0, min(60, round(occ_sum / samples * 5)))
#                 if 0 <= h < 24:
#                     peak_minutes[h] = minutes
#                     peak_people[h] = maxp if maxp > 0 else round(occ_sum / samples)
#                     if minutes > 0 or maxp > 0:
#                         peak_labels[h] = f"{h:02d}:00→{h:02d}:{max(minutes,1):02d}"
#         except Exception as e:
#             print(f"[Analytics] ⚠ peak hours error: {e}", flush=True)
#         self.peak.set_data(
#             peak_minutes, peak_labels, QColor("#3b82f6"),
#             formatter=lambda v: f'{int(v)}min' if v > 0 else '',
#             max_value=60, top_values=peak_people,
#         )
#         self.peak._show_minutes = True
#         self.heatsum.update()
# 
# 
# 

class EventDetailDialog(QDialog):
    TYPES = {
        "recognized": ("Recognized", TH.OK),
        "unknown": ("Unknown", "#f59e42"),
        "offline": ("Offline", TH.ERR),
        "online": ("Online", TH.OK),
        "snapshot": ("Snapshot", TH.ACCENT),
        "system": ("System", TH.DIM),
        "enrolled": ("Enrolled", TH.OK),
        "intrusion": ("Zone Intrusion", TH.ERR),
        "overstay": ("Overstay", TH.WARN),
    }

    def __init__(self, e, sys_, parent=None):
        super().__init__(parent)

        self.e = e

        self.setWindowTitle(f"Event — {e['type']}")

        self.setModal(True)
        self.setFixedWidth(480)

        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)

        name, col = self.TYPES.get(e["type"], (e["type"], TH.DIM))

        badge = QLabel(name)
        badge.setStyleSheet(
            f"background:{col}; color:#0c1116; font-weight:800;"
            "border-radius:5px; padding:4px 12px; font-size:11px;"
        )
        badge.setFixedWidth(badge.sizeHint().width() + 24)

        v.addWidget(badge)

        g = QGridLayout()
        g.setSpacing(8)

        for i, (label, val) in enumerate([
            ("Time", e["time"].strftime("%Y-%m-%d %H:%M:%S")),
            ("Camera", e["cam"]),
            ("Person", e["person"]),
            ("Confidence", f'{e["conf"] * 100:.1f}%' if e["conf"] < 1 else "—"),
            ("Level", e["level"].upper()),
        ]):
            l = QLabel(label)
            l.setStyleSheet(f"color:{TH.DIM}; font-size:10px; font-weight:700;")

            vl = QLabel(str(val))
            vl.setStyleSheet(f"color:{TH.TXT}; font-size:11px;")

            g.addWidget(l, i, 0)
            g.addWidget(vl, i, 1)

        v.addLayout(g)

                # Snapshot ko'rsatish
        sim = sys_.sim_by_id(e["cam"])
        
        snapshot_path = e.get("snapshot_path")
        
        # Avval snapshot faylni tekshirish
        if snapshot_path and os.path.exists(snapshot_path):
            pm = QPixmap(snapshot_path).scaled(
                440,
                247,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            snap = QLabel()
            snap.setPixmap(pm)
            snap.setStyleSheet(f"border:1px solid {TH.BORDER}; border-radius:8px;")

            v.addWidget(snap)
            
        # Agar fayl yo'q bo'lsa, lekin camera frame bor bo'lsa
        elif sim and sim.frame:
            pm = QPixmap.fromImage(sim.frame).scaled(
                440,
                247,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            snap = QLabel()
            snap.setPixmap(pm)
            snap.setStyleSheet(f"border:1px solid {TH.BORDER}; border-radius:8px;")

            v.addWidget(snap)

        else:
            na = QLabel("📷 Snapshot not available")
            na.setAlignment(Qt.AlignCenter)
            na.setStyleSheet(
                f"color:{TH.FAINT}; background:{TH.CARD};"
                "border-radius:8px; padding:20px; font-size:10px;"
            )

            v.addWidget(na)

        row = QHBoxLayout()

        ack = QPushButton("✓ Acknowledge")
        ack.setObjectName("btnPrimary")
        ack.setCursor(Qt.PointingHandCursor)
        ack.clicked.connect(lambda: self._ack(sys_))

        close = QPushButton("Close")
        close.setObjectName("btnGhost")
        close.setCursor(Qt.PointingHandCursor)
        close.clicked.connect(self.accept)

        row.addStretch(1)
        row.addWidget(ack)
        row.addWidget(close)

        v.addLayout(row)

    def _ack(self, sys_):
        event_id=self.e.get("id") or self.e.get("event_id")
        if not event_id:return
        def done(_):
            self.e["ack"]=True;sys_.push_event(dict(type="system",level="info",cam=self.e.get("cam","SYS"),person="Event acknowledged",conf=1.0));self.accept()
        sys_.async_api.submit(lambda:sys_.api.acknowledge_event(event_id),done,lambda error:QMessageBox.warning(self,"API",error))


class EventsPage(Page):
    TYPES = EventDetailDialog.TYPES

    def __init__(self, hub):

        # PM_AUTO_TIMER_FIX — OLIB TASHLANDI (showEvent + persons_online signal yetarli)
        super().__init__()

        self.hub = hub

        row = self.title_row("Events", f"{len(hub.sys.events)} records | 📅 Bugun")

        # self.search = QLineEdit()  # ✅ Search removed
        # self.search.setPlaceholderText("🔍 Search…")  # ✅ Search removed
        # self.search.setMaximumWidth(200)  # ✅ Search removed
        # self.search.textChanged.connect(self.rebuild)  # ✅ Search removed

        self.flt = QComboBox()
        self.flt.addItems([
            "All",
            "Recognized",
            "Unknown",
            "Alerts",
            "Offline",
            "Snapshot",
            "Intrusion",
            "Overstay",
        ])
        self.flt.currentTextChanged.connect(self.rebuild)

        exp = QPushButton("📄 Export CSV")
        exp.setObjectName("btnGhost")
        exp.setFixedHeight(28)
        exp.setCursor(Qt.PointingHandCursor)
        exp.clicked.connect(self.export_csv)

        # Date picker
        from PySide6.QtCore import QDate
        self.date_picker = QDateEdit()
        self.date_picker.setCalendarPopup(True)
        self.date_picker.setDate(QDate.currentDate())
        self.date_picker.setDisplayFormat("dd/MM/yyyy")
        self.date_picker.setFixedWidth(130)
        self.date_picker.setFixedHeight(28)
        self.date_picker.dateChanged.connect(self._on_date_changed)
        
        today_btn = QPushButton("📅 Bugun")
        today_btn.setObjectName("btnGhost")
        today_btn.setFixedHeight(28)
        today_btn.setCursor(Qt.PointingHandCursor)
        today_btn.clicked.connect(lambda: self.date_picker.setDate(QDate.currentDate()))
        
        row.addWidget(self.date_picker)
        row.addWidget(today_btn)
        row.addWidget(self.flt)
        # row.addWidget(self.search)  # ✅ Search removed
        row.addWidget(exp)

        self.tbl = QTableWidget(0, 6)

        self.tbl.setHorizontalHeaderLabels([
            "Time",
            "Camera",
            "Person",
            "Type",
            "Confidence",
            "📸",
        ])

        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)

        self.tbl.verticalHeader().setVisible(False)

        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setShowGrid(False)
        self.tbl.setAlternatingRowColors(True)

        self.tbl.cellDoubleClicked.connect(self._detail)

        self.v.addWidget(self.tbl, 1)

        self._events = []
        self._load_events_for_date(QDate.currentDate().toString("yyyy-MM-dd"))
        self.rebuild()

    def _match(self, e):
        q = ""  # ✅ Search removed
        f = self.flt.currentText()
        # q = self.search.text().lower()  # ✅ Search removed

        ok = (
            f == "All"
            or (f == "Recognized" and e["type"] in ("recognized", "person_recognized"))
            or (f == "Unknown" and e["type"] in ("unknown", "unknown_person", "unknown_detected"))
            or (f == "Alerts" and e["level"] in ("warn", "err"))
            or (f == "Offline" and e["type"] == "offline")
            or (f == "Snapshot" and e["type"] == "snapshot")
            or (f == "Intrusion" and e["type"] == "intrusion")
            or (f == "Overstay" and e["type"] == "overstay")
        )

        if ok and q:
            ok = q in f'{e["cam"]} {e["person"]} {e["type"]}'.lower()

        return ok

    def _load_events_for_date(self,date_str):
        def loaded(items):
            self._events=[self._normalize_event(item) for item in items];self.rebuild()
        self.hub.sys.async_api.submit(
            lambda:self.hub.sys.api.get_events(from_ts=f"{date_str}T00:00:00Z",to_ts=f"{date_str}T23:59:59Z"),
            loaded,lambda error:self.hub.toast(f"Events API: {error}"))

    @staticmethod
    def _normalize_event(e):
        n = dict(e)
        n.setdefault("cam", n.get("camera_id", ""))
        n.setdefault("person", n.get("person_name", ""))
        n.setdefault("conf", n.get("confidence", 0.0))
        return n

    @staticmethod
    def _event_date(e):
        t = e.get("time")
        if hasattr(t, "strftime"):
            return t.strftime("%Y-%m-%d")
        if isinstance(t, str):
            return t[:10]
        return ""

    @staticmethod
    def _event_time(e):
        t = e.get("time")
        if hasattr(t, "strftime"):
            return t.strftime("%H:%M:%S")
        s = str(t)
        return s[11:19] if len(s) >= 19 else s

    def _load_events_for_date(self,date_str):
        def loaded(items):
            self._events=[self._normalize_event(item) for item in items];self.rebuild()
        self.hub.sys.async_api.submit(
            lambda:self.hub.sys.api.get_events(from_ts=f"{date_str}T00:00:00Z",to_ts=f"{date_str}T23:59:59Z"),
            loaded,lambda error:self.hub.toast(f"Events API: {error}"))

    @staticmethod
    def _normalize_event(e):
        n = dict(e)
        n.setdefault("cam", n.get("camera_id", ""))
        n.setdefault("person", n.get("person_name", ""))
        n.setdefault("conf", n.get("confidence", 0.0))
        return n

    @staticmethod
    def _event_date(e):
        t = e.get("time")
        if hasattr(t, "strftime"):
            return t.strftime("%Y-%m-%d")
        if isinstance(t, str):
            return t[:10]
        return ""

    @staticmethod
    def _event_time(e):
        t = e.get("time")
        if hasattr(t, "strftime"):
            return t.strftime("%H:%M:%S")
        s = str(t)
        return s[11:19] if len(s) >= 19 else s

    def _load_events_for_date(self,date_str):
        def loaded(items):
            self._events=[self._normalize_event(item) for item in items];self.rebuild()
        self.hub.sys.async_api.submit(
            lambda:self.hub.sys.api.get_events(from_ts=f"{date_str}T00:00:00Z",to_ts=f"{date_str}T23:59:59Z"),
            loaded,lambda error:self.hub.toast(f"Events API: {error}"))

    @staticmethod
    def _normalize_event(e):
        n = dict(e)
        n.setdefault("cam", n.get("camera_id", ""))
        n.setdefault("person", n.get("person_name", ""))
        n.setdefault("conf", n.get("confidence", 0.0))
        return n

    @staticmethod
    def _event_date(e):
        t = e.get("time")
        if hasattr(t, "strftime"):
            return t.strftime("%Y-%m-%d")
        if isinstance(t, str):
            return t[:10]
        return ""

    @staticmethod
    def _event_time(e):
        t = e.get("time")
        if hasattr(t, "strftime"):
            return t.strftime("%H:%M:%S")
        s = str(t)
        return s[11:19] if len(s) >= 19 else s

    def _on_date_changed(self, qdate):
        """Date picker o'zgarganda events ni qayta yuklash"""
        date_str = qdate.toString("yyyy-MM-dd")
        print(f"[Events] 📅 Date changed: {date_str}", flush=True)
        self._load_events_for_date(date_str)
        self.rebuild()

    def rebuild(self):
        self.tbl.setRowCount(0)
        if not hasattr(self, '_events'):
            self._events = []
        selected_date = self.date_picker.date().toString("yyyy-MM-dd") if hasattr(self, 'date_picker') else None
        for e in self._events[:300]:
            if selected_date and self._event_date(e) != selected_date:
                continue
            if self._match(e):
                self._insert(e)

    def add_event(self, e):
        e = self._normalize_event(e)
        selected_date = self.date_picker.date().toString("yyyy-MM-dd") if hasattr(self, 'date_picker') else None
        if selected_date and self._event_date(e) != selected_date:
            return
        if not hasattr(self, '_events'):
            self._events = []
        self._events.insert(0, e)
        if self._match(e):
            self._insert(e, top=True)

    def _insert(self, e, top=False):
        r = 0 if top else self.tbl.rowCount()
        self.tbl.insertRow(r)
        name, col = self.TYPES.get(e["type"], (e["type"], TH.DIM))
        vals = [
            self._event_time(e),
            e["cam"],
            e["person"],
            name,
            f'{e["conf"] * 100:.1f}%' if e["conf"] < 1 else "—",
        ]
        for c, txt in enumerate(vals):
            it = QTableWidgetItem(txt)
            it.setTextAlignment(
                Qt.AlignCenter
                if c in (0, 1, 4)
                else Qt.AlignVCenter | Qt.AlignLeft
            )
            if c == 3:
                it.setForeground(QColor(col))
            if c == 0:
                it.setFont(QFont("Consolas", 9))
            if e.get("ack"):
                it.setForeground(QColor(TH.FAINT))
            self.tbl.setItem(r, c, it)
        b = QPushButton("📸")
        b.setFixedSize(30, 24)
        b.setObjectName("camTool")
        cid = e["cam"]
        b.clicked.connect(lambda _=False, i=cid: self._snap(i))
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(0, 2, 0, 2)
        hl.addWidget(b)
        hl.setAlignment(Qt.AlignCenter)
        self.tbl.setCellWidget(r, 5, w)
        self.tbl.setRowHeight(r, 30)
        if self.tbl.rowCount() > 300:
            self.tbl.removeRow(self.tbl.rowCount() - 1)

    def _snap(self, cid):
        s = self.hub.sys.sim_by_id(cid)

        if s:
            self.hub.snapshot(s)
        else:
            self.hub.toast("⚠ Camera not available")

    def _detail(self, r, c):
        it = self.tbl.item(r, 1)
        if not it:
            return
        cam = it.text()
        time_str = self.tbl.item(r, 0).text() if self.tbl.item(r, 0) else ""
        for e in getattr(self, '_events', []):
            if e["cam"] == cam and self._event_time(e) == time_str:
                EventDetailDialog(e, self.hub.sys, self).exec()
                self.rebuild()
                return

    def export_csv(self):
        os.makedirs("exports", exist_ok=True)
        fn = f"exports/events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(fn, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Time", "Camera", "Person", "Type", "Confidence", "Level"])
            for e in getattr(self, '_events', []):
                if self._match(e):
                    t = e.get("time")
                    tstr = t.isoformat() if hasattr(t, "isoformat") else str(t)
                    w.writerow([tstr, e["cam"], e["person"], e["type"], f'{e["conf"]:.3f}', e["level"]])
        self.hub.toast(f"📄 Exported {self.tbl.rowCount()} events → {fn}")

    def showEvent(self, event):
        try:
            super().showEvent(event)
        except Exception:
            pass

        # PM_AUTO_SYNC_FIX
        try:
            if hasattr(self, "sync_db"):
                self.sync_db()
        except Exception as e:
            print(f"[PM] sync_db error: {e}", flush=True)

        try:
            if hasattr(self, "load_persons"):
                self.load_persons()
        except Exception as e:
            print(f"[PM] load_persons error: {e}", flush=True)

        try:
            if hasattr(self, "rebuild"):
                self.rebuild()
        except Exception as e:
            print(f"[PM] rebuild error: {e}", flush=True)

        try:
            if hasattr(self, "refresh"):
                self.refresh()
        except Exception:
            pass

        print("[PM] ✅ Auto DB sync on enter", flush=True)

    def _pm_auto_refresh(self):
        # PM_AUTO_REFRESH_FIX
        try:
            if not self.isVisible():
                return

            if hasattr(self, "load_persons"):
                self.load_persons()

            if hasattr(self, "rebuild"):
                self.rebuild()

        except Exception:
            pass

class PersonManagementPage(Page):
    def __init__(self, hub):

        # PM_AUTO_TIMER_FIX — OLIB TASHLANDI (showEvent + persons_online signal yetarli)
        super().__init__()

        self.hub = hub

        row = self.title_row(
            "Person Management",
            f"{len(hub.sys.people)} registered",
        )

        # # Refresh tugmasi
        # refresh_btn = QPushButton("🔄 Refresh")
        # refresh_btn.setObjectName("btnGhost")
        # refresh_btn.setFixedHeight(28)
        # refresh_btn.setCursor(Qt.PointingHandCursor)
        # refresh_btn.clicked.connect(self.force_refresh)
        
        # row.addWidget(refresh_btn)

        # ✅ DB SYNC tugmasi - eng ishonchli usul
        sync_btn = QPushButton("🔄 DB Sync")
        sync_btn.setObjectName("btnPrimary")
        sync_btn.setFixedHeight(28)
        sync_btn.setCursor(Qt.PointingHandCursor)
        sync_btn.clicked.connect(self.db_sync)
        
        row.addWidget(sync_btn)
        
        # Timer OLIB TASHLANDI — faqat DB Sync yoki signal orqali yangilanish
        # self._refresh_timer = QTimer(self)
        # self._refresh_timer.setInterval(3000)
        # self._refresh_timer.timeout.connect(self.force_refresh)
        # self._refresh_timer.start()

        # self.search = QLineEdit()  # ✅ Search removed
        # self.search.setPlaceholderText("🔍 Search people…")  # ✅ Search removed
        # self.search.setMaximumWidth(220)  # ✅ Search removed
        # self.search.textChanged.connect(self.rebuild)  # ✅ Search removed

        enr = QPushButton("＋ Enroll New")
        enr.setObjectName("btnPrimary")
        enr.setCursor(Qt.PointingHandCursor)
        enr.clicked.connect(lambda: hub.navigate("enroll"))

        # row.addWidget(self.search)  # ✅ Search removed
        row.addWidget(enr)

        self.tbl = QTableWidget(0, 6)

        self.tbl.setHorizontalHeaderLabels([
            "Photo",
            "Name",
            "Department",
            "Status",
            "Last Seen",
            "Recognitions",
        ])

        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)

        self.tbl.verticalHeader().setVisible(False)

        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setShowGrid(False)
        self.tbl.setAlternatingRowColors(True)

        self.tbl.cellClicked.connect(self._profile)

        self.v.addWidget(self.tbl, 1)

        # Online/offline tracking
        self._online_persons = {}  # person_id -> {camera_id, track_id, name}
        
        self.hub.sys.websocket.message.connect(self._on_realtime_message)
        
        self.rebuild()

    def _on_realtime_message(self,message):
        if message.get("type")!="person.identified":return
        self._on_persons_online(message.get("camera_id",""),[message])

    def _on_persons_online(self, camera_id, persons_list):
        """IdentityManager dan kelgan online persons"""
        for p in persons_list:
            pid = p.get("person_id")
            if pid is not None:
                self._online_persons[pid] = {
                    "camera_id": camera_id,
                    "track_id": p.get("track_id"),
                    "name": p.get("name", "Unknown"),
                    "known": p.get("known", False),
                    "stay_sec": p.get("stay_sec", 0),
                    "total_stay": p.get("total_stay", 0),
                }
        # Offline bo'lganlarni tozalash (bu kamerada ko'rinmayotganlar)
        cam_person_ids = {p.get("person_id") for p in persons_list if p.get("person_id")}
        to_remove = [pid for pid, info in self._online_persons.items() 
                     if info["camera_id"] == camera_id and pid not in cam_person_ids]
        for pid in to_remove:
            del self._online_persons[pid]
        
        # Jadvalni yangilash (faqat status ustunini)
        self._update_online_status()

    def _update_online_status(self):
        """Jadvalda online/offline statusni yangilash"""
        for row in range(self.tbl.rowCount()):
            name_item = self.tbl.item(row, 1)
            status_item = self.tbl.item(row, 3)
            if name_item is None or status_item is None:
                continue
            
            # person_id ni topish (name orqali)
            person_name = name_item.text()
            is_online = any(
                info["name"] == person_name or 
                (info["known"] and str(info.get("person_id","")) == person_name)
                for info in self._online_persons.values()
            )
            
            if is_online:
                # status_item.setText("🟢 Online")  # ✅ Status label removed
                status_item.setForeground(QColor("#2ecc71"))
            else:
                pass  # block emptied
                status_item.setForeground(QColor("#95a5a6"))

    def showEvent(self, e):
        """Person Management ochilganda DB dan avto-sync."""
        super().showEvent(e)
        import time as _st
        _n = _st.time()
        if not hasattr(self, "_last_db_sync") or _n - self._last_db_sync > 3:
            self._last_db_sync = _n
            try:
                self.db_sync()
                print("[PM] ✅ Auto DB sync on enter", flush=True)
            except Exception as _se:
                print(f"[PM] showEvent sync xato: {_se}", flush=True)

    def rebuild(self):
        """Jadvalni to'liq qayta qurish — yuz + jonli status"""
        try:
            from datetime import datetime
            print(f"[PM] rebuild START", flush=True)

            self.tbl.clearContents()
            self.tbl.setRowCount(0)

            people = list(self.hub.sys.people)
            print(f"[PM] rebuild: {len(people)} people to display", flush=True)

            q = ""  # ✅ Search removed (disabled)
            visible_count = 0

            # Backup image cache
            face_img_cache = {}

            for rec in people:
                if q:
                    # search_text = f"{rec.name} {rec.dept} {rec.emp_id}".lower()  # EMP REMOVED
                    if q not in search_text:
                        continue

                r = self.tbl.rowCount()
                self.tbl.insertRow(r)

                # ===== PHOTO =====
                ph = QTableWidgetItem()
                avatar_pm = None
                rec_id = getattr(rec, 'db_id', None)
                rec_name = getattr(rec, 'name', '?')

                # 1) PersonRecordUI dan avatar
                if hasattr(rec, 'avatar') and rec.avatar is not None and not rec.avatar.isNull():
                    avatar_pm = rec.avatar

                # 2) face_img_cache dan (face_embeddings.image)
                if avatar_pm is None and rec_id is not None:
                    img_bytes = face_img_cache.get(rec_id)
                    if img_bytes:
                        pm = QPixmap()
                        if pm.loadFromData(img_bytes):
                            avatar_pm = pm

                # 3) Name orqali cache dan qidirish (fallback)
                if avatar_pm is None:
                    for pid, img in face_img_cache.items():
                        # Barcha mavjud ID larni tekshirish
                        try:
                            pm = QPixmap()
                            if pm.loadFromData(img):
                                avatar_pm = pm
                                break  # birinchi topilgan rasm
                        except Exception:
                            continue

                if avatar_pm is not None:
                    scaled = avatar_pm.scaled(38, 38, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                    ph.setData(Qt.DecorationRole, scaled)

                ph.setData(Qt.UserRole, rec)
                self.tbl.setItem(r, 0, ph)

                # ===== NAME =====
                p_name = (getattr(rec, 'name', None) or getattr(rec, 'person_name', None) or "?")
                p_emp = (getattr(rec, 'emp_id', None) or getattr(rec, 'employee_id', None) or "")
                nm = QTableWidgetItem(f"{p_name}\n{p_emp}")
                nm.setData(Qt.UserRole, rec)
                self.tbl.setItem(r, 1, nm)

                # ===== DEPARTMENT =====
                dept_item = QTableWidgetItem(getattr(rec, 'dept', None) or getattr(rec, 'department', None) or "—")
                dept_item.setData(Qt.UserRole, rec)
                self.tbl.setItem(r, 2, dept_item)

                # ===== STATUS =====
                # Online/Offline status
                pid = getattr(rec, 'id', None) or getattr(rec, 'person_id', None)
                is_online = pid in self._online_persons if hasattr(self, '_online_persons') and pid else False
                if is_online:
                    _cam = self._online_persons.get(pid, {}).get("camera_id", "")
                    status_text, status_color = f"🟢 Online · {_cam}", "#2ecc71"
                else:
                    status_text, status_color = "", "#95a5a6"  # ✅ No offline label
                st = QTableWidgetItem(status_text)
                st.setForeground(QColor(status_color))
                st.setData(Qt.UserRole, rec)
                self.tbl.setItem(r, 3, st)

                # ===== LAST SEEN =====
                ls_val = getattr(rec, 'last_seen', None) or getattr(rec, 'last_seen_dt', None)
                if ls_val and hasattr(ls_val, 'strftime'):
                    ls_text = ls_val.strftime("%d/%m/%Y %H:%M")
                elif ls_val:
                    ls_text = str(ls_val)[:16]
                else:
                    ls_text = "—"
                ls_item = QTableWidgetItem(ls_text)
                ls_item.setData(Qt.UserRole, rec)
                self.tbl.setItem(r, 4, ls_item)

                # ===== RECOGNITIONS + STAY TOTAL =====
                rc_val = getattr(rec, 'rec_count', 0) or 0
                stay_val = getattr(rec, 'stay_total', 0) or 0
                stay_text = f"{int(stay_val//60)}m {int(stay_val%60)}s" if stay_val else "0m"
                rc = QTableWidgetItem(f"{rc_val}\n{stay_text}")
                rc.setTextAlignment(Qt.AlignCenter)
                rc.setData(Qt.UserRole, rec)
                self.tbl.setItem(r, 5, rc)

                self.tbl.setRowHeight(r, 50)
                visible_count += 1

            if hasattr(self, 'lbl_count'):
                self.lbl_count.setText(f"{visible_count} registered")

            print(f"[PM] rebuild DONE: {visible_count} rows displayed", flush=True)

            # Auto-refresh timer
            if not hasattr(self, '_presence_timer'):
                from PySide6.QtCore import QTimer
                self._presence_timer = QTimer(self)
                self._presence_timer.setInterval(20000)
                self._presence_timer.timeout.connect(self._auto_refresh_presence)
                self._presence_timer.start()

        except Exception as e:
            print(f"[PM] rebuild ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def _presence_label(self, rec):
        """last_seen asosida jonli status matni va rangi"""
        from datetime import datetime
        if not rec.last_seen:
            return ("", TH.FAINT)  # ✅ No offline label
        try:
            secs = (datetime.now() - rec.last_seen).total_seconds()
        except Exception:
            return ("⚫ Unknown", TH.FAINT)

        if secs < 60:
            return ("🟢 Online", TH.OK)
        elif secs < 300:
            return (f"🟡 {int(secs // 60)}m ago", TH.WARN)
        elif secs < 86400:
            return (f"⚫ {int(secs // 3600)}h ago", TH.FAINT)
        else:
            return (f"⚫ {int(secs // 86400)}d ago", TH.FAINT)

    def _auto_refresh_presence(self):
        """Har 20s da faqat Status ustunini yangilash (tez, to'liq rebuild emas)"""
        try:
            for r in range(self.tbl.rowCount()):
                nm = self.tbl.item(r, 1)
                if not nm:
                    continue
                rec = nm.data(Qt.UserRole)
                if rec is None:
                    continue
                # Online/Offline status
                pid = getattr(rec, 'id', None) or getattr(rec, 'person_id', None)
                is_online = pid in self._online_persons if hasattr(self, '_online_persons') and pid else False
                if is_online:
                    status_text, status_color = "🟢 Online", "#2ecc71"
                else:
                    status_text, status_color = "", "#95a5a6"
                st = QTableWidgetItem(status_text)
                st.setForeground(QColor(status_color))
                self.tbl.setItem(r, 3, st)
        except Exception:
            pass

    def add_record(self, rec):
        self.hub.sys.people.append(rec)

        self.rebuild()

    def _profile(self, r, c):
        """Istalgan ustunga bossa profil ochiladi"""
        rec = None

        # 1) Bosilgan ustundan UserRole olishga urinamiz
        it = self.tbl.item(r, c)
        if it:
            rec = it.data(Qt.UserRole)

        # 2) Bo'sh bo'lsa, Name ustunidan (1) olamiz
        if rec is None:
            name_it = self.tbl.item(r, 1)
            if name_it:
                rec = name_it.data(Qt.UserRole)

        # 3) Hali bo'sh bo'lsa, Photo ustunidan (0) olamiz
        if rec is None:
            photo_it = self.tbl.item(r, 0)
            if photo_it:
                rec = photo_it.data(Qt.UserRole)

        if rec is not None:
            ProfileDialog(rec,api=self.hub.sys.api,async_api=self.hub.sys.async_api).exec()
        else:
            print(f"[PM] _profile: no rec found at row={r} col={c}", flush=True)

    def force_refresh(self):
        def loaded(rows):
            if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):
                self.hub.toast("Persons API returned invalid data");return
            self.hub.sys.people=[self.hub.sys.person_record(row) for row in rows]
            self.rebuild()
        self.hub.sys.async_api.submit(self.hub.sys.api.get_persons,loaded,lambda error:self.hub.toast(f"Persons API: {error}"))

    def db_sync(self):
        self.force_refresh()

    def _pm_auto_refresh(self):
        # PM_AUTO_REFRESH_FIX
        try:
            if not self.isVisible():
                return

            if hasattr(self, "load_persons"):
                self.load_persons()

            if hasattr(self, "rebuild"):
                self.rebuild()

        except Exception:
            pass

class ProfileDialog(QDialog):
    def __init__(self,rec,api,async_api):
        super().__init__()
        self.rec=rec;self.api=api;self.async_api=async_api
        self.setWindowTitle(f"Profile — {rec.name}")
        self.setModal(True)
        self.resize(780, 720)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(12)

        # ===== HEADER (FIXED) =====
        top = QHBoxLayout(); top.setSpacing(14)
        av = QLabel()
        if hasattr(rec, 'avatar') and rec.avatar is not None and not rec.avatar.isNull():
            av.setPixmap(rec.avatar.scaled(84, 84, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        av.setStyleSheet("border-radius:12px;")

        info = QVBoxLayout()
        nm = QLabel(rec.name)
        nm.setStyleSheet("font-size:17px; font-weight:800; color:white;")
        from datetime import datetime
        st_text, st_color = "—", TH.FAINT  # default
        if rec.last_seen:
            try: secs = (datetime.now() - rec.last_seen).total_seconds()
            except Exception: secs = 999999
            if secs < 60: st_text, st_color = "🟢 Online", TH.OK
            elif secs < 300: st_text, st_color = f"🟡 {int(secs // 60)}m ago", TH.WARN
            else: st_text, st_color = "—", TH.FAINT
        # else: st_text, st_color = "⚫ Never seen", TH.FAINT  # ✅ Offline removed
        st = QLabel(st_text)
        st.setStyleSheet(f"color:{st_color}; font-size:11px; font-weight:700;")
        info.addWidget(nm); info.addWidget(st); info.addStretch(1)
        top.addWidget(av); top.addLayout(info); top.addStretch(1)
        outer.addLayout(top)

        # ===== STATS (FIXED) =====
        stats = QHBoxLayout()
        ls_text = rec.last_seen.strftime("%d %b %H:%M") if rec.last_seen else "—"
        for label, val in [("LAST SEEN", ls_text), ("RECOGNITIONS", str(rec.rec_count)), ("STAY TOTAL", f"{rec.stay_total} min")]:
            f = QFrame(); f.setObjectName("statCard")
            fl = QVBoxLayout(f); fl.setContentsMargins(10, 8, 10, 8); fl.setSpacing(2)
            a = QLabel(val); a.setStyleSheet("color:white; font-size:13px; font-weight:800;")
            b = QLabel(label); b.setStyleSheet(f"color:{TH.FAINT}; font-size:8px; font-weight:800;")
            fl.addWidget(a); fl.addWidget(b); stats.addWidget(f)
        outer.addLayout(stats)

        # ===== VISIT HISTORY (SCROLL ICHIDA) =====
        vc = ChartCard("VISIT HISTORY (last 7 days)")
        visit_scroll = QScrollArea()
        visit_scroll.setWidgetResizable(True)
        visit_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        visit_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        visit_scroll.setMaximumHeight(180)
        visit_scroll.setStyleSheet("QScrollArea{border:none; background:transparent;}")
        visit_content = QWidget()
        vv = QVBoxLayout(visit_content); vv.setContentsMargins(0,0,0,0); vv.setSpacing(4)
        visits_loaded = False
        if not visits_loaded:
            l = QLabel("No visit records yet"); l.setStyleSheet(f"color:{TH.FAINT}; font-size:10.5px; font-style:italic;"); vv.addWidget(l)
        vv.addStretch(1)
        visit_scroll.setWidget(visit_content)
        vc.body.addWidget(visit_scroll)
        outer.addWidget(vc)

        # ===== RECENT EVENTS (SCROLL ICHIDA) =====
        ec = ChartCard("RECENT EVENTS")
        event_scroll = QScrollArea()
        event_scroll.setWidgetResizable(True)
        event_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        event_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        event_scroll.setMaximumHeight(180)
        event_scroll.setStyleSheet("QScrollArea{border:none; background:transparent;}")
        event_content = QWidget()
        ev = QVBoxLayout(event_content); ev.setContentsMargins(0,0,0,0); ev.setSpacing(4)
        events_loaded = False
        if not events_loaded:
            l = QLabel("No events recorded yet"); l.setStyleSheet(f"color:{TH.FAINT}; font-size:10.5px; font-style:italic;"); ev.addWidget(l)
        ev.addStretch(1)
        event_scroll.setWidget(event_content)
        ec.body.addWidget(event_scroll)
        outer.addWidget(ec)

        # ===== TIMELINE =====
        tl = ChartCard("TIMELINE (presence min/hour - Today, 60min=full)")
        ch = BarChart()
        timeline_data = [0] * 24
        ch.set_data(timeline_data, [f"{i:02d}" for i in range(24)], QColor("#3b82f6"),
                   formatter=lambda v: f"{int(v)}m" if v > 0 else "", max_value=60)
        ch._show_minutes = True; ch.setMinimumHeight(90)
        tl.body.addWidget(ch); outer.addWidget(tl)

        # ===== BUTTONS (FIXED) =====
        btn_row = QHBoxLayout(); btn_row.addStretch(1)
        edit_btn = QPushButton("✏ Edit"); edit_btn.setObjectName("btnGhost")
        edit_btn.setCursor(Qt.PointingHandCursor); edit_btn.clicked.connect(self._edit_person)
        del_btn = QPushButton("🗑 Delete"); del_btn.setObjectName("btnGhost")
        del_btn.setCursor(Qt.PointingHandCursor); del_btn.setStyleSheet(f"color:{TH.ERR};")
        del_btn.clicked.connect(self._delete_person)
        close_btn = QPushButton("Close"); close_btn.setObjectName("btnPrimary")
        close_btn.setCursor(Qt.PointingHandCursor); close_btn.clicked.connect(self.accept)
        btn_row.addWidget(edit_btn); btn_row.addWidget(del_btn); btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

    def _edit_person(self):
        new_name,ok1=QInputDialog.getText(self,"Edit Name","Name:",text=self.rec.name)
        if not ok1 or not new_name.strip():return
        new_dept,ok2=QInputDialog.getText(self,"Edit Department","Department:",text=getattr(self.rec,"dept",""))
        if not ok2:return
        self.async_api.submit(lambda:self.api.update_person(self.rec.db_id,{"name":new_name.strip(),"department":new_dept.strip()}),lambda _:(setattr(self.rec,"name",new_name.strip()),setattr(self.rec,"dept",new_dept.strip()),self.accept()),lambda error:QMessageBox.warning(self,"API",error))

    def _delete_person(self):
        if QMessageBox.question(self,"Delete Person",f"Are you sure you want to delete {self.rec.name!r}?",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes:return
        self.async_api.submit(lambda:self.api.delete_person(self.rec.db_id),lambda _:self.accept(),lambda error:QMessageBox.warning(self,"API",error))


class EnrollmentPage(Page):
    def __init__(self, hub):
        super().__init__()

        self.hub = hub
        self.session_id=None
        self.hub.sys.websocket.message.connect(self._on_enrollment_event)
        self.sim = hub.sys.enroll_sim

        self.title_row(
            "Person Enrollment",
            "upload 1–10 clear face images · GPU embedding",
        )

        body = QHBoxLayout()
        body.setSpacing(12)

        # ==================== LEFT: IMAGE UPLOAD ====================
        left = QFrame()
        left.setObjectName("camCard")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(28, 28, 28, 28)
        ll.setSpacing(16)

        upload_icon = QLabel("🖼️")
        upload_icon.setAlignment(Qt.AlignCenter)
        upload_icon.setStyleSheet("font-size:72px; border:none;")
        upload_title = QLabel("Upload face images")
        upload_title.setAlignment(Qt.AlignCenter)
        upload_title.setStyleSheet(f"color:{TH.TXT}; font-size:22px; font-weight:800; border:none;")
        upload_hint = QLabel("Choose 1–10 clear photos. Front-facing and well-lit images give the best recognition.")
        upload_hint.setWordWrap(True)
        upload_hint.setAlignment(Qt.AlignCenter)
        upload_hint.setStyleSheet(f"color:{TH.DIM}; font-size:12px; border:none;")
        self.face_status = QLabel("No images selected")
        self.face_status.setAlignment(Qt.AlignCenter)
        self.face_status.setStyleSheet(f"color:{TH.DIM}; font-size:11px; font-weight:700; border:none;")
        self.btn_upload_main = QPushButton("📁  Select face images")
        self.btn_upload_main.setObjectName("btnPrimary")
        self.btn_upload_main.setCursor(Qt.PointingHandCursor)
        self.btn_upload_main.clicked.connect(self.upload_images)

        ll.addStretch(1)
        ll.addWidget(upload_icon)
        ll.addWidget(upload_title)
        ll.addWidget(upload_hint)
        ll.addWidget(self.btn_upload_main, 0, Qt.AlignHCenter)
        ll.addWidget(self.face_status)
        ll.addStretch(1)
        body.addWidget(left, 1)

        # ==================== RIGHT: FORM ====================
        right = QFrame()
        right.setObjectName("chartCard")
        right.setFixedWidth(350)

        f = QVBoxLayout(right)
        f.setContentsMargins(16, 16, 16, 16)
        f.setSpacing(10)

        t = QLabel("REGISTER NEW PERSON")
        t.setStyleSheet(
            f"color:{TH.ACC2}; font-size:9px; font-weight:800; letter-spacing:1.5px;"
        )
        f.addWidget(t)

        self.name = QLineEdit()
        self.name.setPlaceholderText("Full Name *")

        self.dept = QComboBox()
        self.dept.addItems([
            "Security", "IT", "Finance", "HR",
            "Operations", "Management", "Other",
        ])

        self.dept_custom = QLineEdit()
        self.dept_custom.setPlaceholderText("Other department name...")
        self.dept_custom.hide()
        self.dept.currentTextChanged.connect(self._on_dept_changed)

        # self.emp = QLineEdit()  # EMP REMOVED
        # self.emp.setPlaceholderText("Employee ID (auto if empty)")  # EMP REMOVED

        f.addWidget(self.name)
        f.addWidget(self.dept)
        f.addWidget(self.dept_custom)
        # f.addWidget(self.emp)  # EMP REMOVED

        self.prog = QProgressBar()
        self.prog.setRange(0, 10)
        self.prog.setValue(0)
        self.prog.setFixedHeight(8)
        self.prog.setTextVisible(False)

        self.prog_lbl = QLabel("Uploaded 0 / 10")
        self.prog_lbl.setStyleSheet(
            f"color:{TH.DIM}; font-size:10px; font-weight:700;"
        )

        f.addWidget(self.prog_lbl)
        f.addWidget(self.prog)

        self.thumbs = QGridLayout()
        self.thumbs.setSpacing(6)
        f.addLayout(self.thumbs)

        self.emb = QLabel("Embedding: —")
        self.emb.setStyleSheet(
            f"color:{TH.FAINT}; font-size:9.5px; font-family:Consolas,monospace;"
        )
        f.addWidget(self.emb)
        f.addStretch(1)

        # Image-only enrollment; webcam capture is intentionally unavailable.
        self.btn_upload = QPushButton("📁 Select Images")
        self.btn_upload.setObjectName("btnGhost")
        self.btn_upload.setCursor(Qt.PointingHandCursor)
        self.btn_upload.clicked.connect(self.upload_images)
        self.btn_reg = QPushButton("💾  Register")
        self.btn_reg.setObjectName("btnPrimary")
        self.btn_reg.setEnabled(False)
        self.btn_reg.setCursor(Qt.PointingHandCursor)
        self.btn_reg.clicked.connect(self.register)

        f.addWidget(self.btn_upload)
        f.addWidget(self.btn_reg)

        body.addWidget(right)
        self.v.addLayout(body, 1)

        self.captures = []

    def _on_enrollment_event(self,event):
        if event.get("session_id")!=self.session_id:return
        kind=event.get("type")
        if kind=="enrollment.progress":
            captured=int(event.get("captured",0));required=int(event.get("required",10))
            self.prog.setMaximum(required);self.prog.setValue(captured);self.prog_lbl.setText(f"Captured {captured} / {required}")
            self.face_status.setText(event.get("message") or f"Quality {float(event.get('quality',0)):.0%}")
        elif kind=="enrollment.completed":
            self.hub.toast("✅ Enrollment completed");self._reset()
            if hasattr(self.hub,"pm"):self.hub.pm.force_refresh()
        elif kind in ("enrollment.failed","enrollment.cancelled"):
            self.face_status.setText(f"⚠ {event.get('message',kind)}");self.btn_reg.setEnabled(True)

    def upload_images(self):
        self.hub.toast("Enrollment capture is controlled by the ML camera workflow")
        self.face_status.setText("Select a name and start camera enrollment")
        self.btn_reg.setEnabled(True)

    # ==================== REGISTER ====================
    def register(self):
        name=self.name.text().strip()
        if not name:self.hub.toast("⚠ Please enter the person name");self.name.setFocus();return
        dept=self.dept.currentText()
        if dept=="Other":dept=self.dept_custom.text().strip() or "Other"
        camera_id=getattr(self.hub.sys.enroll_sim,"id","CAM-01")
        self.btn_reg.setEnabled(False);self.face_status.setText("Starting enrollment…")
        def started(session):
            self.session_id=session["id"] if "id" in session else session["session_id"]
            self.face_status.setText("Enrollment started — look at the camera")
            self.hub.toast(f"Enrollment started for {name}")
        self.hub.sys.async_api.submit(lambda:self.hub.sys.api.start_enrollment(name,camera_id,dept),started,lambda error:(self.btn_reg.setEnabled(True),self.face_status.setText(f"⚠ {error}")))

    # ==================== RESET ====================
    def _reset(self):
        for i in reversed(range(self.thumbs.count())):
            w = self.thumbs.itemAt(i).widget()
            if w:
                w.deleteLater()

        self.captures = []
        self.prog.setValue(0)
        self.prog_lbl.setText("Uploaded 0 / 10")
        self.emb.setText("Embedding: —")
        self.emb.setStyleSheet(f"color:{TH.FAINT}; font-size:9.5px; font-family:Consolas,monospace;")
        self.btn_reg.setEnabled(False)
        self.name.clear()
        # self.emp.clear()  # EMP REMOVED
        self.face_status.setText("🟢 Ready for next enrollment")

    def _on_dept_changed(self, text):
        if text == "Other":
            self.dept_custom.show()
        else:
            self.dept_custom.hide()

            
class PasswordDialog(QDialog):
    def __init__(self, pwd, parent=None):
        super().__init__(parent)

        self.pwd = pwd

        self.setModal(True)
        self.setFixedWidth(380)

        v = QVBoxLayout(self)
        v.setContentsMargins(26, 26, 26, 22)
        v.setSpacing(10)

        ic = QLabel("🔒")
        ic.setStyleSheet("font-size:30px;")
        ic.setAlignment(Qt.AlignCenter)

        t = QLabel("Restricted Area")
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet("font-size:15px; font-weight:800; color:white;")

        s = QLabel("Administrator password required")
        s.setAlignment(Qt.AlignCenter)
        s.setStyleSheet(f"color:{TH.DIM}; font-size:10.5px;")

        self.edit = QLineEdit()
        self.edit.setEchoMode(QLineEdit.Password)
        self.edit.setPlaceholderText("Password")
        self.edit.returnPressed.connect(self.check)

        self.err = QLabel("Incorrect password")
        self.err.setAlignment(Qt.AlignCenter)
        self.err.setStyleSheet(f"color:{TH.ERR}; font-size:10px;")
        self.err.hide()

        row = QHBoxLayout()

        cancel = QPushButton("Cancel")
        cancel.setObjectName("btnGhost")
        cancel.clicked.connect(self.reject)

        ok = QPushButton("Unlock")
        ok.setObjectName("btnPrimary")
        ok.clicked.connect(self.check)

        row.addStretch(1)
        row.addWidget(cancel)
        row.addWidget(ok)
        row.addStretch(1)

        for w in (ic, t, s, self.edit, self.err):
            v.addWidget(w)

        v.addLayout(row)

        self.edit.setFocus()

    def check(self):
        if self.edit.text() == self.pwd:
            self.accept()
        else:
            self.err.show()

            self.edit.clear()
            self.edit.setFocus()


class SettingsPage(Page):
    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        self.title_row("Settings", "administrator access granted")
        self.tabs = QTabWidget()
        self.v.addWidget(self.tabs, 1)
        st = self.hub.sys.settings

        # ==== Cameras ====
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)
        for s in hub.sys.sims:
            row = QHBoxLayout()
            cb = QCheckBox(f"{s.id}  —  {s.name}")
            cb.setChecked(s.online)
            def mk(on=s, box=cb):
                def f(state):
                    requested=box.isChecked();box.setEnabled(False)
                    def applied(_):
                        on.enabled=requested
                        if not requested:on.clear_frame()
                        box.setEnabled(True)
                    def failed(error):
                        box.blockSignals(True);box.setChecked(on.online);box.blockSignals(False);box.setEnabled(True);hub.toast(f"Camera API: {error}")
                    hub.sys.async_api.submit(lambda:hub.sys.api.update_camera(on.id,{"enabled":requested}),applied,failed)
                return f
            cb.stateChanged.connect(mk())
            res = QComboBox()
            res.addItems(["1920x1080", "2560x1440", "1280x720"])
            fps = QSpinBox()
            fps.setRange(5, 60)
            fps.setValue(25)
            row.addWidget(cb, 1)
            row.addWidget(res)
            row.addWidget(fps)
            lay.addLayout(row)
        lay.addStretch(1)
        self.tabs.addTab(w, "🎥 Cameras")

        # ==== Database ====
        w = QWidget()
        fm = QFormLayout(w)
        fm.setSpacing(12)
        fm.addRow("Storage", QLabel("SQLite · managed by API service"))
        br = QHBoxLayout()
        bb = QPushButton("Backup Now")
        bb.setObjectName("btnGhost")
        bb.clicked.connect(lambda: hub.toast("💾 Database backup started"))
        bv = QPushButton("Vacuum")
        bv.setObjectName("btnGhost")
        bv.clicked.connect(lambda: hub.toast("✅ Database optimized"))
        br.addWidget(bb)
        br.addWidget(bv)
        br.addStretch(1)
        fm.addRow("", br)
        self.tabs.addTab(w, "🗄 Database")

        # ==== Security ====
        w = QWidget()
        fm = QFormLayout(w)
        fm.setSpacing(12)
        self.pw1 = QLineEdit()
        self.pw1.setEchoMode(QLineEdit.Password)
        self.pw2 = QLineEdit()
        self.pw2.setEchoMode(QLineEdit.Password)
        ch = QPushButton("Change Password")
        ch.setObjectName("btnPrimary")
        ch.clicked.connect(self._change_pwd)
        al = QSpinBox()
        al.setRange(1, 120)
        al.setValue(15)
        al.valueChanged.connect(lambda value:hub.sys.async_api.submit(lambda:hub.sys.api.update_settings({"auto_lock_minutes":value}),None,lambda error:hub.toast(f"Settings API: {error}")))
        https = QCheckBox("HTTPS / TLS only")
        https.setChecked(True)
        self.snd = QCheckBox("🔊 Sound alerts for critical events")
        self.snd.setChecked(st.get("sound", True))
        self.snd.stateChanged.connect(lambda _state:hub.sys.async_api.submit(lambda:hub.sys.api.update_settings({"sound_enabled":self.snd.isChecked()}),lambda _:hub.sys.settings.__setitem__("sound",self.snd.isChecked()),lambda error:hub.toast(f"Settings API: {error}")))
        fm.addRow("New password", self.pw1)
        fm.addRow("Repeat password", self.pw2)
        fm.addRow("", ch)
        fm.addRow("Auto-lock (min)", al)
        fm.addRow("", https)
        fm.addRow("", self.snd)
        self.tabs.addTab(w, "🔐 Security")

    def _change_pwd(self):
        p1 = self.pw1.text().strip()
        p2 = self.pw2.text().strip()
        if not p1:
            self.hub.toast("⚠ Parol bo'sh bo'lmasin")
            return
        if p1 != p2:
            self.hub.toast("⚠ Parollar mos emas")
            return
        def saved(_):
            self.hub.sys.settings["password"]=p1;self.hub.toast("✅ Password updated");self.pw1.clear();self.pw2.clear()
        self.hub.sys.async_api.submit(lambda:self.hub.sys.api.update_settings({"login_password":p1}),saved,lambda error:self.hub.toast(f"Settings API: {error}"))


SPLASH_STEPS = [
    (0, "Tizim ishga tushmoqda..."),
    (20, "Sozlamalar yuklanmoqda..."),
    (40, "Kameralar ulanmoqda..."),
    (60, "AI modellar tayyorlanmoqda..."),
    (80, "Ma'lumotlar bazasi ochilmoqda..."),
    (100, "Tayyor!"),
]


class SplashScreen(QDialog):
    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.Dialog)

        self.setFixedSize(480, 300)
        self.setModal(True)

        scr = QApplication.primaryScreen().availableGeometry()

        self.move(scr.center() - self.rect().center())

        v = QVBoxLayout(self)
        v.setContentsMargins(40, 36, 40, 30)
        v.setSpacing(14)

        logo = QLabel("◉")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(f"color:{TH.ACCENT}; font-size:44px; font-weight:900;")

        t = QLabel("AI SURVEILLANCE SYSTEM")
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet(
            "color:white; font-size:18px; font-weight:900; letter-spacing:2px;"
        )

        s = QLabel("MUKAMMAL Edition v3.0")
        s.setAlignment(Qt.AlignCenter)
        s.setStyleSheet(f"color:{TH.DIM}; font-size:10px;")

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFixedHeight(6)
        self.bar.setTextVisible(False)

        self.lbl = QLabel(SPLASH_STEPS[0][1])
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setStyleSheet(f"color:{TH.FAINT}; font-size:10px;")

        for w in (logo, t, s):
            v.addWidget(w)

        v.addSpacing(10)

        v.addWidget(self.bar)
        v.addWidget(self.lbl)

        self.step = 0

        self.timer = QTimer(self)
        self.timer.setInterval(320)
        self.timer.timeout.connect(self.next_step)
        self.timer.start()

    def next_step(self):
        self.step += 1

        if self.step >= len(SPLASH_STEPS):
            self.timer.stop()

            self.accept()

            return

        pct, txt = SPLASH_STEPS[self.step]

        self.bar.setValue(pct)
        self.lbl.setText(txt)


class LoginScreen(QDialog):
    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.Dialog)

        self.setFixedSize(400, 420)
        self.setModal(True)

        scr = QApplication.primaryScreen().availableGeometry()

        self.move(scr.center() - self.rect().center())

        v = QVBoxLayout(self)
        v.setContentsMargins(40, 36, 40, 30)
        v.setSpacing(12)

        logo = QLabel("◉")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(f"color:{TH.ACCENT}; font-size:40px; font-weight:900;")

        t = QLabel("AI Surveillance System")
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet("color:white; font-size:16px; font-weight:800;")

        s = QLabel("Operator Login")
        s.setAlignment(Qt.AlignCenter)
        s.setStyleSheet(f"color:{TH.DIM}; font-size:10.5px;")

        self.user = QLineEdit()
        self.user.setText("admin")
        self.user.setPlaceholderText("Username")

        self.pwd = QLineEdit()
        self.pwd.setEchoMode(QLineEdit.Password)
        self.pwd.setText("admin")
        self.pwd.setPlaceholderText("Password")
        self.pwd.returnPressed.connect(self.check)

        self.err = QLabel("Invalid credentials")
        self.err.setAlignment(Qt.AlignCenter)
        self.err.setStyleSheet(f"color:{TH.ERR}; font-size:10px;")
        self.err.hide()

        btn = QPushButton("🔓  Login")
        btn.setObjectName("btnPrimary")
        btn.setFixedHeight(40)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(self.check)

        hint = QLabel("Default: admin / admin")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color:{TH.FAINT}; font-size:9px;")

        for w in (logo, t, s):
            v.addWidget(w)

        v.addSpacing(8)

        v.addWidget(self.user)
        v.addWidget(self.pwd)
        v.addWidget(self.err)
        v.addWidget(btn)
        v.addWidget(hint)

        v.addStretch(1)

    def check(self):
        if self.user.text().strip() == "admin" and self.pwd.text() == "admin":
            self.accept()
        else:
            self.err.show()

            self.pwd.clear()
            self.pwd.setFocus()


# =========================== MAIN WINDOW =============================
class MainWindow(QMainWindow):
    def __init__(self):

        # PM_AUTO_TIMER_FIX — OLIB TASHLANDI (showEvent + persons_online signal yetarli)
        super().__init__()

        self.setWindowTitle("AI Surveillance System — MUKAMMAL Edition")

        self.resize(1680, 980)
        self.setMinimumSize(1180, 700)

        self.sys = System()

        self.tick_n = 0
        self.fs = None

        self._sb_anim = []
        self._nav_anim = None
        self._force_close = False

        self.header = Header(self.sys, self)

        self.stack = QStackedWidget()

        # self.dash = DashboardPage(self)  # Dashboard olib tashlandi
        self.live = LivePage(self)
        self.pm = PersonManagementPage(self)
        self.enroll = EnrollmentPage(self)
        # self.analytics = AnalyticsPage(self)  # ✅ Analytics commented out
        self.events_pg = EventsPage(self)
        self.settings_pg = SettingsPage(self)

        self.page_index = {}

        for key, w in [
            # ("dashboard", self.dash),  # Dashboard olib tashlandi
            ("live", self.live),
            ("people", self.pm),
            ("enroll", self.enroll),
            # ("analytics", self.analytics),  # ✅ Analytics removed
            ("events", self.events_pg),
            ("settings", self.settings_pg),
        ]:
            self.page_index[key] = self.stack.addWidget(w)

        self.cards = self.live.cards  # Dashboard olib tashlandi

        self.right = RightPanel(self.sys)

        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(1)

        split.addWidget(self.stack)
        split.addWidget(self.right)

        split.setStretchFactor(0, 70)
        split.setStretchFactor(1, 15)

        self.sidebar = SideBar(self)

        body = QWidget()

        hb = QHBoxLayout(body)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(0)

        hb.addWidget(self.sidebar)
        hb.addWidget(split)

        self.central_widget = QWidget()

        vb = QVBoxLayout(self.central_widget)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(0)

        vb.addWidget(self.header)
        vb.addWidget(body, 1)

        self.setCentralWidget(self.central_widget)

        self.notif = NotificationDropdown(self)

        self.fade = QWidget(self.central_widget)
        self.fade.setStyleSheet(f"background:{TH.BG};")
        self.fade.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.fade.hide()

        self.fade_eff = QGraphicsOpacityEffect(self.fade)
        self.fade.setGraphicsEffect(self.fade_eff)
        self.fade_eff.setOpacity(0)

        self.sidebar.changed.connect(self.navigate)
        self.sys.new_event.connect(self.on_event)

        tf = QTimer(self)
        tf.timeout.connect(self.tick)
        tf.start(40)

        ts = QTimer(self)
        ts.timeout.connect(self.slow_tick)
        ts.start(1000)

        for e in reversed(self.sys.events):
            self.right.add_event(e)
            self.events_pg.add_event(e)

        self.navigate("live")

        self.right.refresh()

        QTimer.singleShot(3000, self.header.set_ai_ready)

        pages = [
            "dashboard",
            "live",
            "people",
            "enroll",
            # "analytics",  # ✅ Analytics nav removed
            "events",
            "settings",
        ]

        for i, key in enumerate("123456"):  # ✅ 6 pages (analytics removed)
            QShortcut(QKeySequence(key), self, lambda k=pages[i]: self.navigate(k))

        QShortcut(QKeySequence("/"), self, self._focus_search)

        QShortcut(QKeySequence("Ctrl+E"), self, lambda: self.navigate("events"))

        self._tray_ok = QSystemTrayIcon.isSystemTrayAvailable()

        if self._tray_ok:
            self.tray = QSystemTrayIcon(self)

            ipm = QPixmap(32, 32)
            ipm.fill(Qt.transparent)

            p = QPainter(ipm)
            p.setRenderHint(QPainter.Antialiasing)

            p.setBrush(QColor(TH.ACCENT))
            p.setPen(Qt.NoPen)
            p.drawEllipse(0, 0, 32, 32)

            p.setPen(QColor("white"))
            p.setFont(QFont("Segoe UI", 10, QFont.Bold))
            p.drawText(ipm.rect(), Qt.AlignCenter, "AI")

            p.end()

            self.tray.setIcon(QIcon(ipm))
            self.tray.setToolTip("AI Surveillance System")

            menu = QMenu()

            menu.addAction("Show").triggered.connect(self.showNormal)
            menu.addAction("Hide").triggered.connect(self.hide)
            menu.addAction("Exit").triggered.connect(self._tray_exit)

            self.tray.setContextMenu(menu)

            self.tray.activated.connect(
                lambda r: self.showNormal()
                if r == QSystemTrayIcon.DoubleClick
                else None
            )

            self.tray.show()

    def navigate(self, page):
        if page == "settings" and not self.sys.settings["unlocked"]:
            dlg = PasswordDialog(self.sys.settings["password"], self)

            if dlg.exec() != QDialog.Accepted:
                return

            self.sys.settings["unlocked"] = True

        if self.stack.currentIndex() == self.page_index[page]:
            self.sidebar.set_active(page)

            return

        pos = self.stack.mapTo(self.central_widget, QPoint(0, 0))

        self.fade.setGeometry(pos.x(), pos.y(), self.stack.width(), self.stack.height())

        self.fade.show()
        self.fade.raise_()

        a = QPropertyAnimation(self.fade_eff, b"opacity")
        a.setDuration(110)
        a.setStartValue(0)
        a.setEndValue(1)
        a.finished.connect(lambda: self._finish_nav(page))
        a.start()

        self._nav_anim = a

    def _finish_nav(self, page):
        self.stack.setCurrentIndex(self.page_index[page])

        self.sidebar.set_active(page)

        if page == "dashboard":
            pass  # Dashboard removed

        a = QPropertyAnimation(self.fade_eff, b"opacity")
        a.setDuration(110)
        a.setStartValue(1)
        a.setEndValue(0)
        a.finished.connect(self.fade.hide)
        a.start()

        self._nav_anim = a

    def _focus_search(self):
        if not isinstance(self.focusWidget(), QLineEdit):
            self.header.search.setFocus()

    def toggle_sidebar(self):
        collapsed = not self.sidebar.collapsed

        w = 80 if collapsed else 210

        self.sidebar.set_collapsed(collapsed)

        self._sb_anim = []

        for prop in (b"minimumWidth", b"maximumWidth"):
            a = QPropertyAnimation(self.sidebar, prop)

            a.setDuration(160)
            a.setEndValue(w)
            a.start()

            self._sb_anim.append(a)

    def on_event(self, e):
        self.right.add_event(e)
        self.events_pg.add_event(e)
        self.notif.add_alert(e)

        if e.get("level") in ("warn", "err"):
            self.header.bump()

            if self.sys.settings.get("sound", True):
                QApplication.beep()

    def tick(self):
        self.tick_n += 1

        if self.tick_n % 5 == 0:
            for c in self.cards:
                c.refresh()

            self.right.refresh()

            if self.fs:
                self.fs.refresh()

    def slow_tick(self):
        sy = self.sys
        def apply_metrics(metrics):
            sy.gpu=float(metrics.get("gpu_utilization_percent") or metrics.get("gpu_percent") or 0)
            sy.cpu=float(metrics.get("cpu_percent") or 0);sy.ram=float(metrics.get("memory_percent") or 0);self.header.update_stats()
        sy.async_api.submit(sy.api.get_system_metrics,apply_metrics,lambda _error:None)

        self.header.tick_clock()

        # self.analytics.push()  # ✅ Analytics commented out

    def open_fullscreen(self, sim):
        dlg = FullscreenCam(sim, self)

        self.fs = dlg

        dlg.showFullScreen()
        dlg.exec()

        self.fs = None

    def snapshot(self, sim):
        if not sim.online or sim.frame is None:
            self.toast("⚠ Camera is offline")

            return

        os.makedirs("snapshots", exist_ok=True)

        fn = f"snapshots/{sim.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        sim.frame.save(fn)

        self.toast(f"📸 Snapshot saved — {fn}")

        self.sys.push_event(
            dict(
                type="snapshot",
                level="info",
                cam=sim.id,
                person="Snapshot captured",
                conf=1.0,
            )
        )

    def toast(self, msg):
        t = QLabel(msg, self)

        t.setStyleSheet(
            f"background:{TH.CARD2}; color:{TH.TXT};"
            f"border:1px solid {TH.BORDER}; border-radius:9px;"
            "padding:10px 16px; font-size:11.5px; font-weight:600;"
        )

        t.adjustSize()

        t.move(self.width() - t.width() - 24, self.height() - t.height() - 24)

        t.show()
        t.raise_()

        eff = QGraphicsOpacityEffect(t)
        t.setGraphicsEffect(eff)

        anim = QPropertyAnimation(eff, b"opacity")
        anim.setDuration(500)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)

        anim2 = QPropertyAnimation(t, b"pos")
        anim2.setDuration(500)
        anim2.setStartValue(t.pos())
        anim2.setEndValue(t.pos() + QPoint(0, 14))

        QTimer.singleShot(2100, anim.start)
        QTimer.singleShot(2100, anim2.start)
        QTimer.singleShot(2700, t.deleteLater)

    def _tray_exit(self):
        self._force_close = True

        self.close()

    def closeEvent(self, e):
        if self._tray_ok and not self._force_close:
            self.hide()

            self.tray.showMessage(
                "AI Surveillance System",
                "Minimized to system tray",
                QSystemTrayIcon.Information,
                2000,
            )

            e.ignore()

        else:
            self.sys.shutdown()
            e.accept()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Slash and not isinstance(self.focusWidget(), QLineEdit):
            self.header.search.setFocus()

            e.accept()

        else:
            super().keyPressEvent(e)


# ============================ STYLESHEET =============================
STYLE = """
* { font-family: "Segoe UI", "Roboto", "Ubuntu", sans-serif; }

QMainWindow, QDialog { background: #101418; }

QWidget { color: #e9eef5; font-size: 12px; }

QWidget#page { background: #0f1317; }

QFrame#header { background: #151b22; border-bottom: 1px solid #2b3542; }

QFrame#sidebar { background: #151b22; border-right: 1px solid #2b3542; }

QFrame#rightPanel { background: #151b22; }

QFrame#chip { background: #1a212a; border: 1px solid #2b3542; border-radius: 12px; }

QPushButton#bellBtn { background: #1a212a; border: 1px solid #2b3542;
    border-radius: 17px; font-size: 14px; }

QPushButton#bellBtn:hover { background: #26303c; }

QPushButton#sideBtn { text-align: left; padding: 10px 14px; border-radius: 8px;
    color: #94a1b3; font-size: 12.5px; border: none; background: transparent; }

QPushButton#sideBtn[collapsed="true"] { text-align: center; padding: 10px 0; }

QPushButton#sideBtn:hover { background: #202a35; color: #e9eef5; }

QPushButton#sideBtn:checked { background: #2f7df6; color: #fff; font-weight: 700; }

QFrame#camCard { background: #0a0d11; border: 1px solid #2b3542; border-radius: 10px; }

QFrame#camCard[offline="true"] { border: 1px solid #ef5350; }

QFrame#camToolbar { background: rgba(13,17,22,215); border: 1px solid #2b3542;
    border-radius: 9px; }

QToolButton#camTool, QPushButton#camTool { background: transparent; border: none;
    border-radius: 6px; font-size: 13px; color: #e9eef5; }

QToolButton#camTool:hover, QPushButton#camTool:hover { background: #2f7df6; }

QToolButton#camTool:checked { background: #2f7df6; }

QFrame#quickInfo { background: rgba(13,17,22,228); border: 1px solid #2b3542;
    border-radius: 9px; }

QFrame#statCard, QFrame#chartCard { background: #1a212a;
    border: 1px solid #2b3542; border-radius: 10px; }

QFrame#alertItem { background: #1a212a; }

QFrame#notifDrop { background: #1a212a; border: 1px solid #2b3542;
    border-radius: 10px; }

QFrame#notifItem { background: #212a35; border-radius: 6px; }

QFrame#notifItem:hover { background: #26303c; }

QPushButton#btnPrimary { background: #2f7df6; color: white; border: none;
    border-radius: 7px; padding: 8px 16px; font-weight: 700; }

QPushButton#btnPrimary:hover { background: #4a8ff7; }

QPushButton#btnPrimary:disabled { background: #233043; color: #5d6b7e; }

QPushButton#btnGhost { background: #232c37; border: 1px solid #2b3542;
    border-radius: 7px; padding: 7px 14px; color: #e9eef5; }

QPushButton#btnGhost:hover { background: #2a3441; }

QPushButton#btnGhost:checked { background: #2f7df6; border-color: #5b9bff;
    color: white; }

QLineEdit, QComboBox, QSpinBox { background: #232c37; border: 1px solid #2b3542;
    border-radius: 6px; padding: 6px 9px;
    selection-background-color: #2f7df6; }

QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border: 1px solid #2f7df6; }

QComboBox QAbstractItemView { background: #202a35; border: 1px solid #2b3542;
    selection-background-color: #2f7df6; outline: none; }

QProgressBar { background: #232c37; border-radius: 3px; border: none; }

QProgressBar::chunk { background: #2f7df6; border-radius: 3px; }

QTableWidget { background: #1a212a; alternate-background-color: #1d2530;
    gridline-color: #2b3542; border: 1px solid #2b3542; border-radius: 8px;
    selection-background-color: #2f7df6; }

QTableWidget::item { padding: 4px 8px; }

QHeaderView::section { background: #202a35; color: #94a1b3; padding: 7px;
    border: none; border-bottom: 1px solid #2b3542;
    font-weight: 700; font-size: 10.5px; }

QTabWidget::pane { border: 1px solid #2b3542; border-radius: 8px;
    background: #1a212a; top: -1px; }

QTabBar::tab { padding: 9px 18px; background: transparent; color: #94a1b3;
    border: 1px solid transparent; border-bottom: none;
    border-top-left-radius: 7px; border-top-right-radius: 7px; margin-right: 2px; }

QTabBar::tab:selected { background: #1a212a; color: #fff;
    border-color: #2b3542; font-weight: 700; }

QTabBar::tab:hover { color: #e9eef5; }

QMenu { background: #202a35; border: 1px solid #2b3542;
    border-radius: 8px; padding: 4px; }

QMenu::item { padding: 7px 26px; border-radius: 5px; }

QMenu::item:selected { background: #2f7df6; color: white; }

QScrollArea { background: transparent; }

QScrollBar:vertical { background: transparent; width: 9px; margin: 0; }

QScrollBar::handle:vertical { background: #2b3542; border-radius: 4px;
    min-height: 24px; }

QScrollBar::handle:vertical:hover { background: #3a4757; }

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal { background: transparent; height: 9px; }

QScrollBar::handle:horizontal { background: #2b3542; border-radius: 4px;
    min-width: 24px; }

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

QSlider::groove:horizontal { background: #232c37; height: 5px;
    border-radius: 2px; }

QSlider::handle:horizontal { background: #2f7df6; width: 14px;
    margin: -5px 0; border-radius: 7px; }

QCheckBox { spacing: 8px; }

QCheckBox::indicator { width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid #2b3542; background: #232c37; }

QCheckBox::indicator:checked { background: #2f7df6; border-color: #2f7df6; }

QSplitter::handle { background: #2b3542; }

QToolTip { background: #202a35; color: #e9eef5; border: 1px solid #2b3542;
    border-radius: 5px; padding: 5px 8px; }
"""


# =============================== RUN =================================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyle("Fusion")

    pal = QPalette()

    pal.setColor(QPalette.Window, QColor(TH.PANEL))
    pal.setColor(QPalette.WindowText, QColor(TH.TXT))
    pal.setColor(QPalette.Base, QColor(TH.CARD))
    pal.setColor(QPalette.Text, QColor(TH.TXT))
    pal.setColor(QPalette.Button, QColor(TH.CARD2))
    pal.setColor(QPalette.ButtonText, QColor(TH.TXT))
    pal.setColor(QPalette.Highlight, QColor(TH.ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("white"))

    app.setPalette(pal)

    app.setStyleSheet(STYLE)

    splash = SplashScreen()

    if splash.exec() != QDialog.Accepted:
        sys.exit(0)

    login = LoginScreen()

    if login.exec() != QDialog.Accepted:
        sys.exit(0)

    win = MainWindow()
    win.show()

    sys.exit(app.exec())

    def showEvent(self, event):
        try:
            super().showEvent(event)
        except Exception:
            pass

        # PM_AUTO_SYNC_FIX
        try:
            if hasattr(self, "sync_db"):
                self.sync_db()
        except Exception as e:
            print(f"[PM] sync_db error: {e}", flush=True)

        try:
            if hasattr(self, "load_persons"):
                self.load_persons()
        except Exception as e:
            print(f"[PM] load_persons error: {e}", flush=True)

        try:
            if hasattr(self, "rebuild"):
                self.rebuild()
        except Exception as e:
            print(f"[PM] rebuild error: {e}", flush=True)

        try:
            if hasattr(self, "refresh"):
                self.refresh()
        except Exception:
            pass

        print("[PM] ✅ Auto DB sync on enter", flush=True)

    def _pm_auto_refresh(self):
        # PM_AUTO_REFRESH_FIX
        try:
            if not self.isVisible():
                return

            if hasattr(self, "load_persons"):
                self.load_persons()

            if hasattr(self, "rebuild"):
                self.rebuild()

        except Exception:
            pass
