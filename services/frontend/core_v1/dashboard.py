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
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
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
PURPLE = "#9d3cff"

CAMERA_TITLES = {
    "CAM-01": "Office 1 (A)",
    "CAM-02": "Office 2 (A)",
    "CAM-03": "Office 3 (A)",
    "CAM-04": "Office 1 (B)",
    "CAM-05": "Office 2 (B)",
    "CAM-06": "Office 3 (B)",
}


def font(size: int, weight=QFont.Weight.Normal) -> QFont:
    f = QFont("Inter")
    f.setPixelSize(size)
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
        p.drawEllipse(QRectF(s*.38, s*.15, s*.24, s*.24))
        path = QPainterPath(); path.moveTo(s*.23, s*.82); path.cubicTo(s*.27, s*.56, s*.73, s*.56, s*.77, s*.82); p.drawPath(path)
    elif kind == "users":
        p.drawEllipse(QRectF(s*.29, s*.13, s*.22, s*.22)); p.drawEllipse(QRectF(s*.56, s*.20, s*.18, s*.18))
        p.drawArc(QRectF(s*.18, s*.43, s*.48, s*.40), 15*16, 150*16); p.drawArc(QRectF(s*.48, s*.48, s*.38, s*.32), 15*16, 145*16)
    elif kind == "bell":
        path = QPainterPath(); path.moveTo(s*.28, s*.66); path.lineTo(s*.34, s*.58); path.lineTo(s*.34, s*.40)
        path.cubicTo(s*.34, s*.16, s*.66, s*.16, s*.66, s*.40); path.lineTo(s*.66, s*.58); path.lineTo(s*.72, s*.66); p.drawPath(path)
        p.drawLine(QPointF(s*.28, s*.66), QPointF(s*.72, s*.66))
    elif kind == "report":
        p.drawRoundedRect(QRectF(s*.18, s*.14, s*.64, s*.70), s*.04, s*.04)
        p.drawLine(QPointF(s*.32, s*.67), QPointF(s*.32, s*.50)); p.drawLine(QPointF(s*.50, s*.67), QPointF(s*.50, s*.36)); p.drawLine(QPointF(s*.68, s*.67), QPointF(s*.68, s*.44))
    elif kind == "settings":
        p.drawEllipse(QRectF(s*.38, s*.38, s*.24, s*.24)); p.drawEllipse(QRectF(s*.27, s*.27, s*.46, s*.46))
    elif kind == "camera":
        p.drawRoundedRect(QRectF(s*.16, s*.30, s*.48, s*.38), s*.04, s*.04)
        path = QPainterPath(); path.moveTo(s*.64, s*.39); path.lineTo(s*.84, s*.30); path.lineTo(s*.84, s*.68); path.lineTo(s*.64, s*.59); path.closeSubpath(); p.drawPath(path)
    elif kind == "activity":
        path = QPainterPath(); path.moveTo(s*.08, s*.55); path.lineTo(s*.28, s*.55); path.lineTo(s*.36, s*.22); path.lineTo(s*.47, s*.78); path.lineTo(s*.57, s*.42); path.lineTo(s*.66, s*.55); path.lineTo(s*.92, s*.55); p.drawPath(path)
    elif kind == "fullscreen":
        for a,b,c in [((.18,.38),(.18,.18),(.38,.18)),((.62,.18),(.82,.18),(.82,.38)),((.18,.62),(.18,.82),(.38,.82)),((.62,.82),(.82,.82),(.82,.62))]:
            p.drawLine(QPointF(s*a[0],s*a[1]), QPointF(s*b[0],s*b[1])); p.drawLine(QPointF(s*b[0],s*b[1]), QPointF(s*c[0],s*c[1]))
    else:
        p.drawRoundedRect(QRectF(s*.20, s*.20, s*.60, s*.60), s*.05, s*.05)
    p.end()
    return pm


def icon(kind: str, color: str = TEXT, size: int = 24) -> QIcon:
    return QIcon(icon_pixmap(kind, color, size))


class FrameReader:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._image = None
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
                connection.request("GET", f"/frame/{self.camera_id}?after={version}&wait_ms=180", headers={"Cache-Control":"no-cache","Connection":"keep-alive"})
                response = connection.getresponse(); payload = response.read()
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
                    self._image = image; self._version = version; self.frames += 1
            except Exception:
                if connection is not None:
                    try: connection.close()
                    except Exception: pass
                connection = None
                self._stop.wait(.08)


class RealtimeState:
    def __init__(self):
        self._stop = threading.Event(); self._lock = threading.Lock(); self._thread = None
        self.state = {"connected":False, "health":{}, "detections":{}, "reid":{}}
        self.recent = deque(maxlen=20); self.events = deque(maxlen=100); self._seen = {}

    def start(self):
        self._thread = threading.Thread(target=self._run, name="ui-state", daemon=True); self._thread.start()

    def stop(self): self._stop.set()

    def snapshot(self):
        with self._lock: return dict(self.state), list(self.recent), list(self.events)

    @staticmethod
    def _get_json(connection, path):
        connection.request("GET", path, headers={"Cache-Control":"no-cache","Connection":"keep-alive"})
        response = connection.getresponse(); payload = response.read()
        if response.status != 200: raise RuntimeError(response.status)
        return json.loads(payload.decode("utf-8"))

    def _observe(self, reid):
        now = datetime.now(); cams = ((reid.get("state") or {}).get("cameras") or {})
        for cid, tracks in cams.items():
            for t in tracks or []:
                gid = str(t.get("global_id") or "")
                if not gid: continue
                key = (cid, int(t.get("local_id") or 0)); previous = self._seen.get(key); self._seen[key] = gid
                if previous == gid: continue
                item = {"time":now.strftime("%H:%M:%S"), "camera":cid, "gid":gid, "reason":str(t.get("reason") or "detected"), "similarity":t.get("similarity")}
                self.recent.appendleft(item); self.events.appendleft(item)

    def _run(self):
        connection = None
        while not self._stop.is_set():
            try:
                if connection is None: connection = http.client.HTTPConnection(ML_HOST, ML_PORT, timeout=1.5)
                health = self._get_json(connection, "/health")
                detections = self._get_json(connection, "/detections")
                reid = self._get_json(connection, "/reid")
                self._observe(reid)
                with self._lock: self.state = {"connected":True, "health":health, "detections":detections, "reid":reid}
                self._stop.wait(.35)
            except Exception:
                if connection is not None:
                    try: connection.close()
                    except Exception: pass
                connection = None
                with self._lock: self.state = {**self.state, "connected":False}
                self._stop.wait(.6)


class CameraImage(QLabel):
    """Fill the whole camera area while preserving the entire source frame."""
    def __init__(self):
        super().__init__("Connecting...")
        self._image = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(220, 120)
        self.setStyleSheet("background:#020913;color:#8192aa;border:0;")

    def set_frame(self, image):
        self._image = image; self._apply()

    def _apply(self):
        if self._image is None or self.width() < 2 or self.height() < 2: return
        # IgnoreAspectRatio intentionally removes black side bars while keeping
        # every source pixel visible (no crop). This is only UI presentation.
        pix = QPixmap.fromImage(self._image).scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event); self._apply()


class CameraTile(QFrame):
    def __init__(self, camera_id: str, ordinal: int):
        super().__init__(); self.camera_id = camera_id; self.setObjectName("cameraTile")
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        self.header = QWidget(); self.header.setFixedHeight(44)
        h = QHBoxLayout(self.header); h.setContentsMargins(9,0,10,0)
        chip = QLabel(f"{ordinal:02d}"); chip.setFixedSize(34,32); chip.setAlignment(Qt.AlignmentFlag.AlignCenter); chip.setFont(font(15,QFont.Weight.DemiBold)); chip.setStyleSheet(f"background:{BLUE_2};color:white;border-radius:6px;")
        title = QLabel(CAMERA_TITLES[camera_id]); title.setFont(font(15,QFont.Weight.Medium))
        live = QLabel("● LIVE"); live.setFont(font(13,QFont.Weight.Medium)); live.setStyleSheet(f"color:{GREEN};")
        h.addWidget(chip); h.addWidget(title); h.addStretch(); h.addWidget(live); outer.addWidget(self.header)
        self.image = CameraImage(); outer.addWidget(self.image, 1)
        self.footer = QWidget(); self.footer.setFixedHeight(34)
        f = QHBoxLayout(self.footer); f.setContentsMargins(10,0,10,0)
        pi = QLabel(); pi.setPixmap(icon_pixmap("users", TEXT, 18)); self.people = QLabel("0 People"); self.people.setFont(font(12,QFont.Weight.Medium)); self.fps = QLabel("-- FPS"); self.fps.setFont(font(12,QFont.Weight.Medium)); bars = QLabel("▂▄▆█"); bars.setStyleSheet(f"color:{GREEN};")
        f.addWidget(pi); f.addWidget(self.people); f.addStretch(); f.addWidget(self.fps); f.addWidget(bars); outer.addWidget(self.footer)

    def set_metrics(self, count: int, fps: float):
        self.people.setText(f"{count} {'Person' if count == 1 else 'People'}")
        self.fps.setText(f"{fps:.0f} FPS" if fps > 0 else "-- FPS")

    def camera_only(self, enabled: bool):
        self.header.setVisible(not enabled); self.footer.setVisible(not enabled)
        self.setStyleSheet("border:0;background:#000;" if enabled else "")


class NavButton(QPushButton):
    def __init__(self, text, kind):
        super().__init__(text); self.setCheckable(True); self.setIcon(icon(kind,TEXT,25)); self.setIconSize(QSize(25,25)); self.setFixedHeight(58); self.setFont(font(16,QFont.Weight.Medium)); self.setCursor(Qt.CursorShape.PointingHandCursor)


class Sidebar(QFrame):
    def __init__(self, change_page):
        super().__init__(); self.setObjectName("sidebar"); self.setFixedWidth(235)
        layout = QVBoxLayout(self); layout.setContentsMargins(14,14,14,18); layout.setSpacing(8)
        brand = QHBoxLayout(); logo = QLabel("◢"); logo.setFont(font(32,QFont.Weight.Bold)); logo.setStyleSheet(f"color:{BLUE};"); name = QLabel("Apsidal"); name.setFont(font(27,QFont.Weight.DemiBold)); brand.addWidget(logo); brand.addWidget(name); brand.addStretch(); layout.addLayout(brand); layout.addSpacing(12)
        self.buttons = {}
        for i,(label,kind) in enumerate([("Live View","monitor"),("People","person"),("Events","bell"),("Reports","report"),("Settings","settings")]):
            button = NavButton(label,kind); button.clicked.connect(lambda checked=False,index=i: change_page(index)); layout.addWidget(button); self.buttons[i] = button
        layout.addStretch()
        self.status = QFrame(); self.status.setObjectName("statusCard"); sl = QVBoxLayout(self.status); sl.setContentsMargins(14,14,14,14)
        top = QHBoxLayout(); pulse = QLabel(); pulse.setPixmap(icon_pixmap("activity",GREEN,26)); title = QLabel("System Status"); title.setFont(font(13,QFont.Weight.Medium)); top.addWidget(pulse); top.addWidget(title); top.addStretch(); sl.addLayout(top)
        self.status_text = QLabel("Waiting for realtime data"); self.status_text.setAlignment(Qt.AlignmentFlag.AlignCenter); self.status_text.setFont(font(12)); self.status_text.setStyleSheet(f"color:{MUTED};"); sl.addWidget(self.status_text); layout.addWidget(self.status)

    def set_active(self, index):
        for i,b in self.buttons.items(): b.setChecked(i == index)

    def update_live(self, state):
        if not state.get("connected"):
            self.status_text.setText("ML service offline"); self.status_text.setStyleSheet(f"color:{RED};"); return
        h = state.get("health") or {}; r = h.get("service_resources") or {}
        self.status_text.setText(f"{h.get('online',0)}/{h.get('total',6)} cameras online\nGPU {r.get('gpu_utilization_percent','—')}%")
        self.status_text.setStyleSheet(f"color:{GREEN};")


class StatCard(QFrame):
    def __init__(self, kind, color, label):
        super().__init__(); self.setObjectName("statCard"); l = QVBoxLayout(self); l.setContentsMargins(15,15,15,12)
        row = QHBoxLayout(); ic = QLabel(); ic.setPixmap(icon_pixmap(kind,color,33)); self.value = QLabel("0"); self.value.setFont(font(27,QFont.Weight.DemiBold)); row.addWidget(ic); row.addSpacing(10); row.addWidget(self.value); row.addStretch(); l.addLayout(row)
        text = QLabel(label); text.setAlignment(Qt.AlignmentFlag.AlignCenter); text.setFont(font(12)); l.addWidget(text)


class RightRail(QWidget):
    def __init__(self):
        super().__init__(); self.setFixedWidth(350); layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(14)
        cards = QFrame(); cards.setObjectName("statsPanel"); grid = QGridLayout(cards); grid.setContentsMargins(10,10,10,10); grid.setSpacing(10)
        self.total = StatCard("users",BLUE,"Total People"); self.known = StatCard("person",GREEN,"Known People"); self.unknown = StatCard("person",ORANGE,"Unknown People"); self.cams = StatCard("camera",CYAN,"Active Cameras")
        grid.addWidget(self.total,0,0); grid.addWidget(self.known,0,1); grid.addWidget(self.unknown,1,0); grid.addWidget(self.cams,1,1); layout.addWidget(cards)
        recent = QFrame(); recent.setObjectName("recentCard"); rl = QVBoxLayout(recent); rl.setContentsMargins(15,15,15,12)
        head = QHBoxLayout(); title = QLabel("Recent Views"); title.setFont(font(16,QFont.Weight.DemiBold)); view = QLabel("View All"); view.setStyleSheet(f"color:{CYAN};"); head.addWidget(title); head.addStretch(); head.addWidget(view); rl.addLayout(head)
        self.recent_box = QVBoxLayout(); self.recent_box.setSpacing(8); rl.addLayout(self.recent_box); rl.addStretch(); layout.addWidget(recent,1)

    def update_live(self, state, recent):
        global_state = ((((state.get("reid") or {}).get("state") or {}).get("global")) or {})
        active = [gid for gid,v in global_state.items() if v.get("active_tracks")]
        self.total.value.setText(str(len(active))); self.known.value.setText("0"); self.unknown.value.setText(str(len(active)))
        health = state.get("health") or {}; self.cams.value.setText(f"{health.get('online',0)}/{health.get('total',6)}")
        while self.recent_box.count():
            item = self.recent_box.takeAt(0); w = item.widget();
            if w: w.deleteLater()
        for event in recent[:8]:
            row = QFrame(); row.setObjectName("recentItem"); r = QHBoxLayout(row); r.setContentsMargins(8,7,8,7)
            av = QLabel(); av.setFixedSize(34,34); av.setAlignment(Qt.AlignmentFlag.AlignCenter); av.setPixmap(icon_pixmap("person",MUTED,22)); av.setStyleSheet("background:#223246;border-radius:5px;")
            txt = QLabel(f"{event['gid']}\n{event['camera']}"); txt.setFont(font(11)); tm = QLabel(event['time']); tm.setFont(font(10)); tm.setStyleSheet(f"color:{CYAN};")
            r.addWidget(av); r.addWidget(txt,1); r.addWidget(tm); self.recent_box.addWidget(row)


class LivePage(QWidget):
    def __init__(self):
        super().__init__(); layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(10)
        self.header = QWidget(); h = QHBoxLayout(self.header); h.setContentsMargins(0,0,0,0); title = QLabel("Live View"); title.setFont(font(27,QFont.Weight.DemiBold)); h.addWidget(title); h.addStretch(); self.full = QPushButton(); self.full.setObjectName("squareButton"); self.full.setIcon(icon("fullscreen",TEXT,22)); self.full.setFixedSize(42,42); h.addWidget(self.full); layout.addWidget(self.header)
        self.grid = QGridLayout(); self.grid.setContentsMargins(0,0,0,0); self.grid.setSpacing(10); self.tiles = {}
        for i,cid in enumerate(CAMERAS):
            tile = CameraTile(cid,i+1); self.tiles[cid] = tile; self.grid.addWidget(tile,i//2,i%2)
        for r in range(3): self.grid.setRowStretch(r,1)
        for c in range(2): self.grid.setColumnStretch(c,1)
        layout.addLayout(self.grid,1)

    def camera_only(self, enabled):
        self.header.setVisible(not enabled); self.grid.setSpacing(2 if enabled else 10)
        for tile in self.tiles.values(): tile.camera_only(enabled)


def table_item(value):
    item = QTableWidgetItem(str(value)); item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable); return item


class PeoplePage(QWidget):
    def __init__(self):
        super().__init__(); l = QVBoxLayout(self); title = QLabel("People"); title.setFont(font(27,QFont.Weight.DemiBold)); l.addWidget(title)
        self.table = QTableWidget(0,5); self.table.setHorizontalHeaderLabels(["GLOBAL ID","CAMERAS","OBSERVATIONS","ROOM","STATUS"]); self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); self.table.verticalHeader().hide(); self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection); self.table.setObjectName("dataTable"); l.addWidget(self.table,1)

    def update_live(self,state):
        glob = ((((state.get("reid") or {}).get("state") or {}).get("global")) or {}); rows=[]
        for gid,v in glob.items():
            active=v.get("active_tracks") or {}
            if active: rows.append([gid,", ".join(sorted(active)),v.get("observations",0),", ".join(v.get("active_rooms") or []),"Active"])
        self.table.setRowCount(len(rows))
        for r,row in enumerate(rows):
            for c,value in enumerate(row): self.table.setItem(r,c,table_item(value))


class EventsPage(QWidget):
    def __init__(self):
        super().__init__(); l = QVBoxLayout(self); title = QLabel("Events"); title.setFont(font(27,QFont.Weight.DemiBold)); l.addWidget(title)
        self.table = QTableWidget(0,5); self.table.setHorizontalHeaderLabels(["TIME","EVENT","CAMERA / LOCATION","IDENTITY","DETAILS"]); self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); self.table.verticalHeader().hide(); self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection); self.table.setObjectName("dataTable"); l.addWidget(self.table,1)

    def update_live(self,events):
        self.table.setRowCount(len(events))
        for r,event in enumerate(events):
            sim=event.get("similarity"); details=event.get("reason","detected") + (f" · {sim:.3f}" if isinstance(sim,(int,float)) else "")
            for c,value in enumerate([event["time"],"Person detected",event["camera"],event["gid"],details]): self.table.setItem(r,c,table_item(value))


class EmptyPage(QWidget):
    def __init__(self,title):
        super().__init__(); l=QVBoxLayout(self); t=QLabel(title); t.setFont(font(27,QFont.Weight.DemiBold)); l.addWidget(t); msg=QLabel("Realtime page is not connected yet."); msg.setAlignment(Qt.AlignmentFlag.AlignCenter); msg.setStyleSheet(f"color:{MUTED};"); l.addWidget(msg,1)


class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Apsidal"); self.resize(1672,941); self.setMinimumSize(1280,720); self.camera_mode=False
        root=QWidget(); self.setCentralWidget(root); root_l=QHBoxLayout(root); root_l.setContentsMargins(0,0,0,0); root_l.setSpacing(0)
        self.sidebar=Sidebar(self.set_page); root_l.addWidget(self.sidebar)
        self.body=QWidget(); body_l=QVBoxLayout(self.body); body_l.setContentsMargins(0,0,0,0); body_l.setSpacing(0); root_l.addWidget(self.body,1)
        self.topbar=QWidget(); self.topbar.setObjectName("topbar"); self.topbar.setFixedHeight(68); tb=QHBoxLayout(self.topbar); tb.setContentsMargins(22,0,24,0)
        menu=QPushButton(); menu.setObjectName("topButton"); menu.setIcon(icon("menu",TEXT,25)); menu.setFixedSize(42,42); menu.clicked.connect(lambda:self.sidebar.setVisible(not self.sidebar.isVisible())); tb.addWidget(menu); tb.addStretch(); self.clock=QLabel(); self.clock.setFont(font(15,QFont.Weight.Medium)); self.date=QLabel(); self.date.setFont(font(14)); tb.addWidget(self.clock); tb.addSpacing(18); tb.addWidget(self.date); body_l.addWidget(self.topbar)
        self.content=QWidget(); self.content_layout=QHBoxLayout(self.content); self.content_layout.setContentsMargins(16,10,16,16); self.content_layout.setSpacing(16); body_l.addWidget(self.content,1)
        self.stack=QStackedWidget(); self.live=LivePage(); self.people=PeoplePage(); self.events=EventsPage(); self.reports=EmptyPage("Reports"); self.settings=EmptyPage("Settings")
        for page in (self.live,self.people,self.events,self.reports,self.settings): self.stack.addWidget(page)
        self.right=RightRail(); self.content_layout.addWidget(self.stack,1); self.content_layout.addWidget(self.right)
        self.live.full.clicked.connect(self.toggle_camera_fullscreen)
        self.readers={}; self.versions={}; self.frame_counts={cid:0 for cid in CAMERAS}; self.last_tick=time.monotonic()
        for cid in CAMERAS: r=FrameReader(cid); r.start(); self.readers[cid]=r; self.versions[cid]=-1
        self.state_reader=RealtimeState(); self.state_reader.start()
        self.render_timer=QTimer(self); self.render_timer.setTimerType(Qt.TimerType.PreciseTimer); self.render_timer.timeout.connect(self.render_frames); self.render_timer.start(20)
        self.info_timer=QTimer(self); self.info_timer.timeout.connect(self.update_live_data); self.info_timer.start(500)
        self.clock_timer=QTimer(self); self.clock_timer.timeout.connect(self.update_clock); self.clock_timer.start(1000); self.update_clock(); self.set_page(0); self.apply_theme(); self.showMaximized()

    def apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow,QWidget{{background:{BG};color:{TEXT};}}
            #sidebar{{background:{SIDEBAR};border-right:1px solid #07243f;}}
            #topbar{{background:{BG};}}
            QPushButton{{border:0;color:{TEXT};outline:none;}}
            #sidebar QPushButton{{text-align:left;padding-left:16px;border-radius:7px;background:transparent;}}
            #sidebar QPushButton:hover{{background:#061d3a;}}
            #sidebar QPushButton:checked{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #075ff0,stop:1 #073bc8);border:1px solid #1470ff;}}
            #statusCard,#statsPanel,#recentCard,#cameraTile{{background:{PANEL};border:1px solid {BORDER};border-radius:8px;}}
            #statCard{{background:{CARD};border-radius:7px;}}
            #recentItem{{background:{CARD};border-radius:6px;}}
            #squareButton,#topButton{{background:{CARD};border:1px solid {BORDER};border-radius:7px;}}
            QTableWidget#dataTable{{background:{PANEL};border:1px solid {BORDER};border-radius:7px;color:{TEXT};gridline-color:{BORDER};}}
            QTableWidget#dataTable::item{{border-bottom:1px solid #082b49;padding:8px;}}
            QTableWidget#dataTable QHeaderView::section{{background:#00142d;color:#c7d2e2;border:0;border-bottom:1px solid {BORDER};padding:10px;font-size:12px;font-weight:600;}}
        """)

    def set_page(self,index): self.stack.setCurrentIndex(index); self.sidebar.set_active(index)

    def toggle_camera_fullscreen(self):
        self.camera_mode = not self.camera_mode
        self.sidebar.setVisible(not self.camera_mode); self.topbar.setVisible(not self.camera_mode); self.right.setVisible(not self.camera_mode)
        self.content_layout.setContentsMargins(0,0,0,0) if self.camera_mode else self.content_layout.setContentsMargins(16,10,16,16)
        self.live.camera_only(self.camera_mode)
        if self.camera_mode: self.showFullScreen()
        else: self.showMaximized()

    def keyPressEvent(self,event):
        if event.key() == Qt.Key.Key_Escape and self.camera_mode:
            self.toggle_camera_fullscreen(); return
        super().keyPressEvent(event)

    def update_clock(self):
        now=datetime.now(); self.clock.setText(now.strftime("%H:%M:%S")); self.date.setText(now.strftime("%-d %b %Y"))

    def render_frames(self):
        for cid,reader in self.readers.items():
            image,version=reader.latest()
            if image is not None and version > self.versions[cid]: self.versions[cid]=version; self.live.tiles[cid].image.set_frame(image)

    def update_live_data(self):
        state,recent,events=self.state_reader.snapshot(); self.sidebar.update_live(state); self.right.update_live(state,recent); self.people.update_live(state); self.events.update_live(events)
        detections=((state.get("detections") or {}).get("cameras") or {}); now=time.monotonic(); dt=max(.1,now-self.last_tick); self.last_tick=now
        for cid,reader in self.readers.items():
            current=reader.frames; previous=self.frame_counts[cid]; self.frame_counts[cid]=current; fps=(current-previous)/dt; count=len(((detections.get(cid) or {}).get("boxes") or [])); self.live.tiles[cid].set_metrics(count,fps)

    def closeEvent(self,event):
        self.render_timer.stop(); self.info_timer.stop(); self.clock_timer.stop(); self.state_reader.stop()
        for reader in self.readers.values(): reader.stop()
        event.accept()


def run():
    app=QApplication.instance() or QApplication([]); app.setStyle("Fusion"); window=DashboardWindow(); window.show(); return app.exec()
