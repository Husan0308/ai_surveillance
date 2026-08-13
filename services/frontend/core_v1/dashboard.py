from __future__ import annotations

import http.client
import json
import threading
import time
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, QSize, QRectF, QPointF
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
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
PURPLE = "#9d3cff"
CYAN = "#16b9ff"

CAMERA_TITLES = {
    "CAM-01": "Office 1 (A)",
    "CAM-02": "Office 2 (A)",
    "CAM-03": "Office 3 (A)",
    "CAM-04": "Office 1 (B)",
    "CAM-05": "Office 2 (B)",
    "CAM-06": "Office 3 (B)",
}

PEOPLE_ROWS = [
    ("001", "Husan", "Operator", "Monitoring", "Active", "2 Jul 2025"),
    ("002", "Azizbek", "Supervisor", "Security", "Active", "1 Jul 2025"),
    ("003", "Bobur", "Operator", "Monitoring", "Active", "30 Jun 2025"),
    ("004", "Malika", "Admin", "Administration", "Active", "28 Jun 2025"),
    ("005", "Jahongir", "Operator", "Monitoring", "Active", "27 Jun 2025"),
    ("006", "Sardor", "Operator", "Monitoring", "Inactive", "25 Jun 2025"),
    ("007", "Dilnura", "Supervisor", "Security", "Active", "24 Jun 2025"),
    ("008", "Otabek", "Operator", "Monitoring", "Active", "23 Jun 2025"),
    ("009", "Madina", "Admin", "Administration", "Active", "22 Jun 2025"),
    ("010", "Bekzod", "Supervisor", "Security", "Active", "21 Jun 2025"),
    ("011", "Akmal", "Operator", "Monitoring", "Inactive", "20 Jun 2025"),
    ("012", "Sevinch", "Admin", "Administration", "Active", "19 Jun 2025"),
]

EVENT_ROWS = [
    ("5 Jul 2025\n22:14:31", "Known Person Detected\nID: 001", "CAM 01 - Office 1 (A)", "Husan detected\nConfidence: 92%", "System", "person", BLUE),
    ("5 Jul 2025\n22:13:58", "Unknown Person Detected\nID: Unknown_15", "CAM 03 - Office 3 (A)", "Unknown person detected\nConfidence: 78%", "System", "person", ORANGE),
    ("5 Jul 2025\n22:13:21", "Person Entered\nID: 007", "CAM 02 - Office 2 (A)", "Azizbek entered\nDuration: 00:12:45", "System", "door", GREEN),
    ("5 Jul 2025\n22:12:47", "Person Exited\nID: 008", "CAM 05 - Office 2 (B)", "Bobur exited\nDuration: 00:08:32", "System", "door", RED),
    ("5 Jul 2025\n22:11:03", "Motion Detected", "CAM 04 - Office 1 (B)", "Motion detected\nDuration: 00:00:15", "System", "signal", PURPLE),
    ("5 Jul 2025\n22:10:22", "Camera Offline", "CAM 06 - Office 3 (B)", "Connection lost", "System", "camera", BLUE),
    ("5 Jul 2025\n22:09:18", "Camera Online", "CAM 06 - Office 3 (B)", "Connection restored", "System", "camera", BLUE),
    ("5 Jul 2025\n22:08:44", "System Start", "NVR System", "System started\nVersion: Core v1", "System", "settings", GREEN),
    ("5 Jul 2025\n22:07:30", "Storage Warning", "NVR System", "Storage usage is 85%", "System", "warning", ORANGE),
    ("5 Jul 2025\n22:05:11", "User Login", "Web Dashboard", "Admin logged in", "Husan", "person", BLUE),
]

RECENT_ROWS = [
    ("Husan (ID: 001)", "CAM 01 - Office 1 (A)", "22:14:31"),
    ("Unknown", "CAM 03 - Office 3 (A)", "22:13:58"),
    ("Azizbek (ID: 007)", "CAM 02 - Office 2 (A)", "22:13:21"),
    ("Bobur (ID: 008)", "CAM 05 - Office 2 (B)", "22:12:47"),
    ("Malika (ID: 004)", "CAM 04 - Office 1 (B)", "22:11:33"),
    ("Jahongir (ID: 005)", "CAM 04 - Office 1 (B)", "22:10:19"),
    ("Dilnura (ID: 007)", "CAM 02 - Office 2 (A)", "22:09:05"),
    ("Otabek (ID: 008)", "CAM 05 - Office 2 (B)", "22:08:12"),
]


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
        for y in (.28, .50, .72): p.drawLine(QPointF(s*.18,s*y), QPointF(s*.82,s*y))
    elif kind == "monitor":
        p.drawRoundedRect(QRectF(s*.14,s*.16,s*.72,s*.55),s*.06,s*.06); p.drawLine(QPointF(s*.40,s*.78),QPointF(s*.60,s*.78)); p.drawLine(QPointF(s*.50,s*.71),QPointF(s*.50,s*.78))
    elif kind == "person":
        p.drawEllipse(QRectF(s*.38,s*.15,s*.24,s*.24)); path=QPainterPath(); path.moveTo(s*.23,s*.82); path.cubicTo(s*.27,s*.56,s*.73,s*.56,s*.77,s*.82); p.drawPath(path)
    elif kind == "users":
        p.drawEllipse(QRectF(s*.29,s*.13,s*.22,s*.22)); p.drawEllipse(QRectF(s*.56,s*.20,s*.18,s*.18)); p.drawArc(QRectF(s*.18,s*.43,s*.48,s*.40),15*16,150*16); p.drawArc(QRectF(s*.48,s*.48,s*.38,s*.32),15*16,145*16)
    elif kind == "bell":
        path=QPainterPath(); path.moveTo(s*.28,s*.66); path.lineTo(s*.34,s*.58); path.lineTo(s*.34,s*.40); path.cubicTo(s*.34,s*.16,s*.66,s*.16,s*.66,s*.40); path.lineTo(s*.66,s*.58); path.lineTo(s*.72,s*.66); p.drawPath(path); p.drawLine(QPointF(s*.28,s*.66),QPointF(s*.72,s*.66)); p.drawArc(QRectF(s*.43,s*.67,s*.14,s*.12),0,-180*16)
    elif kind == "report":
        p.drawRoundedRect(QRectF(s*.18,s*.14,s*.64,s*.70),s*.04,s*.04); p.drawLine(QPointF(s*.32,s*.67),QPointF(s*.32,s*.50)); p.drawLine(QPointF(s*.50,s*.67),QPointF(s*.50,s*.36)); p.drawLine(QPointF(s*.68,s*.67),QPointF(s*.68,s*.44))
    elif kind == "settings":
        p.drawEllipse(QRectF(s*.38,s*.38,s*.24,s*.24)); p.drawEllipse(QRectF(s*.27,s*.27,s*.46,s*.46));
        for a,b in [((.50,.12),(.50,.27)),((.50,.73),(.50,.88)),((.12,.50),(.27,.50)),((.73,.50),(.88,.50)),((.23,.23),(.33,.33)),((.67,.67),(.77,.77)),((.67,.33),(.77,.23)),((.23,.77),(.33,.67))]: p.drawLine(QPointF(s*a[0],s*a[1]),QPointF(s*b[0],s*b[1]))
    elif kind == "search":
        p.drawEllipse(QRectF(s*.18,s*.15,s*.46,s*.46)); p.drawLine(QPointF(s*.58,s*.58),QPointF(s*.82,s*.82))
    elif kind == "filter":
        p.drawLine(QPointF(s*.20,s*.30),QPointF(s*.80,s*.30)); p.drawLine(QPointF(s*.30,s*.50),QPointF(s*.70,s*.50)); p.drawLine(QPointF(s*.41,s*.70),QPointF(s*.59,s*.70))
    elif kind == "fullscreen":
        for x1,y1,x2,y2,x3,y3 in [(.18,.38,.18,.18,.38,.18),(.62,.18,.82,.18,.82,.38),(.18,.62,.18,.82,.38,.82),(.62,.82,.82,.82,.82,.62)]: p.drawLine(QPointF(s*x1,s*y1),QPointF(s*x2,s*y2)); p.drawLine(QPointF(s*x2,s*y2),QPointF(s*x3,s*y3))
    elif kind == "camera":
        p.drawRoundedRect(QRectF(s*.16,s*.30,s*.48,s*.38),s*.04,s*.04); path=QPainterPath(); path.moveTo(s*.64,s*.39); path.lineTo(s*.84,s*.30); path.lineTo(s*.84,s*.68); path.lineTo(s*.64,s*.59); path.closeSubpath(); p.drawPath(path)
    elif kind == "activity":
        path=QPainterPath(); path.moveTo(s*.08,s*.55); path.lineTo(s*.28,s*.55); path.lineTo(s*.36,s*.22); path.lineTo(s*.47,s*.78); path.lineTo(s*.57,s*.42); path.lineTo(s*.66,s*.55); path.lineTo(s*.92,s*.55); p.drawPath(path)
    elif kind == "edit":
        p.drawLine(QPointF(s*.28,s*.72),QPointF(s*.68,s*.32)); p.drawLine(QPointF(s*.64,s*.28),QPointF(s*.73,s*.37)); p.drawLine(QPointF(s*.25,s*.75),QPointF(s*.37,s*.72))
    elif kind == "trash":
        p.drawRoundedRect(QRectF(s*.31,s*.34,s*.38,s*.46),s*.02,s*.02); p.drawLine(QPointF(s*.26,s*.28),QPointF(s*.74,s*.28)); p.drawLine(QPointF(s*.42,s*.22),QPointF(s*.58,s*.22)); p.drawLine(QPointF(s*.44,s*.43),QPointF(s*.44,s*.69)); p.drawLine(QPointF(s*.56,s*.43),QPointF(s*.56,s*.69))
    elif kind == "door":
        p.drawRoundedRect(QRectF(s*.28,s*.16,s*.44,s*.68),s*.03,s*.03); p.drawEllipse(QRectF(s*.57,s*.48,s*.05,s*.05))
    elif kind == "signal":
        p.drawEllipse(QRectF(s*.44,s*.44,s*.12,s*.12)); p.drawArc(QRectF(s*.28,s*.28,s*.44,s*.44),45*16,270*16); p.drawArc(QRectF(s*.16,s*.16,s*.68,s*.68),45*16,270*16)
    elif kind == "warning":
        path=QPainterPath(); path.moveTo(s*.50,s*.13); path.lineTo(s*.87,s*.80); path.lineTo(s*.13,s*.80); path.closeSubpath(); p.drawPath(path); p.drawLine(QPointF(s*.50,s*.36),QPointF(s*.50,s*.57)); p.drawPoint(QPointF(s*.50,s*.68))
    elif kind == "plusperson":
        p.drawEllipse(QRectF(s*.22,s*.16,s*.20,s*.20)); p.drawArc(QRectF(s*.12,s*.40,s*.40,s*.34),20*16,140*16); p.drawLine(QPointF(s*.66,s*.36),QPointF(s*.66,s*.68)); p.drawLine(QPointF(s*.50,s*.52),QPointF(s*.82,s*.52))
    p.end()
    return pm


def icon(kind: str, color: str = TEXT, size: int = 24) -> QIcon:
    return QIcon(icon_pixmap(kind, color, size))


def logo_widget() -> QWidget:
    root=QWidget(); l=QHBoxLayout(root); l.setContentsMargins(26,14,10,14); l.setSpacing(12)
    mark=QLabel(); pm=QPixmap(44,44); pm.fill(Qt.GlobalColor.transparent); p=QPainter(pm); p.setRenderHint(QPainter.RenderHint.Antialiasing,True); p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(BLUE)); path=QPainterPath(); path.moveTo(4,35); path.lineTo(19,6); path.lineTo(27,20); path.lineTo(21,30); path.lineTo(16,21); path.lineTo(10,35); path.closeSubpath(); p.drawPath(path); path=QPainterPath(); path.moveTo(25,5); path.lineTo(42,35); path.lineTo(31,35); path.lineTo(19,14); path.closeSubpath(); p.drawPath(path); p.end(); mark.setPixmap(pm)
    name=QLabel("Apsidal"); name.setFont(font(28,QFont.Weight.DemiBold)); l.addWidget(mark); l.addWidget(name); l.addStretch(); return root


class LatestFrameReader:
    def __init__(self,camera_id):
        self.camera_id=camera_id; self._stop=threading.Event(); self._lock=threading.Lock(); self._image=None; self._version=-1; self.frames=0; self.errors=0; self.last_frame_at=0.0; self._thread=None
    def start(self): self._thread=threading.Thread(target=self._run,name=f"frontend-{self.camera_id}",daemon=True); self._thread.start()
    def stop(self): self._stop.set()
    def latest(self):
        with self._lock: return self._image,self._version
    def _run(self):
        version=-1; connection=None
        while not self._stop.is_set():
            try:
                if connection is None: connection=http.client.HTTPConnection(ML_HOST,ML_PORT,timeout=2.0)
                connection.request("GET",f"/frame/{self.camera_id}?after={version}&wait_ms=180",headers={"Cache-Control":"no-cache","Connection":"keep-alive"}); response=connection.getresponse(); jpg=response.read()
                if response.status!=200: raise RuntimeError(response.status)
                next_version=int(response.getheader("X-Frame-Version") or version+1); image=QImage.fromData(jpg,"JPG")
                if image.isNull(): continue
                version=next_version
                with self._lock: self._image=image; self._version=version
                self.frames+=1; self.last_frame_at=time.monotonic()
            except Exception:
                self.errors+=1
                if connection is not None:
                    try: connection.close()
                    except Exception: pass
                connection=None; self._stop.wait(.03)
        if connection is not None:
            try: connection.close()
            except Exception: pass


class DetectionReader:
    def __init__(self):
        self._stop=threading.Event(); self._lock=threading.Lock(); self._counts={cid:0 for cid in CAMERAS}; self._thread=None
    def start(self): self._thread=threading.Thread(target=self._run,name="frontend-detections",daemon=True); self._thread.start()
    def stop(self): self._stop.set()
    def counts(self):
        with self._lock: return dict(self._counts)
    def _run(self):
        connection=None
        while not self._stop.is_set():
            try:
                if connection is None: connection=http.client.HTTPConnection(ML_HOST,ML_PORT,timeout=1.0)
                connection.request("GET","/detections",headers={"Cache-Control":"no-cache","Connection":"keep-alive"}); response=connection.getresponse(); payload=response.read()
                if response.status==200:
                    data=json.loads(payload.decode("utf-8")); cams=data.get("cameras") or {}; counts={cid:len((cams.get(cid) or {}).get("boxes") or []) for cid in CAMERAS}
                    with self._lock: self._counts=counts
                self._stop.wait(.8)
            except Exception:
                if connection is not None:
                    try: connection.close()
                    except Exception: pass
                connection=None; self._stop.wait(1.0)


class CameraImage(QLabel):
    def __init__(self):
        super().__init__("Connecting..."); self._image=None; self.setAlignment(Qt.AlignmentFlag.AlignCenter); self.setMinimumSize(250,120); self.setStyleSheet(f"background:#020913;color:{MUTED};border:0;")
    def set_frame(self,image): self._image=image; self._apply()
    def _apply(self):
        if self._image is None or self.width()<2 or self.height()<2: return
        self.setPixmap(QPixmap.fromImage(self._image).scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatioByExpanding,Qt.TransformationMode.FastTransformation))
    def resizeEvent(self,event): super().resizeEvent(event); self._apply()


class CameraTile(QFrame):
    def __init__(self,camera_id,ordinal):
        super().__init__(); self.setObjectName("cameraTile"); self.camera_id=camera_id; o=QVBoxLayout(self); o.setContentsMargins(0,0,0,0); o.setSpacing(0)
        header=QWidget(); header.setFixedHeight(46); h=QHBoxLayout(header); h.setContentsMargins(9,0,10,0); chip=QLabel(f"{ordinal:02d}"); chip.setFixedSize(34,34); chip.setAlignment(Qt.AlignmentFlag.AlignCenter); chip.setFont(font(16,QFont.Weight.DemiBold)); chip.setStyleSheet(f"background:{BLUE_2};border-radius:7px;"); title=QLabel(CAMERA_TITLES[camera_id]); title.setFont(font(16,QFont.Weight.Medium)); dot=QLabel("●"); dot.setStyleSheet(f"color:{GREEN};"); live=QLabel("LIVE"); live.setFont(font(15,QFont.Weight.Medium)); h.addWidget(chip); h.addWidget(title); h.addStretch(); h.addWidget(dot); h.addWidget(live); o.addWidget(header)
        self.image=CameraImage(); o.addWidget(self.image,1)
        footer=QWidget(); footer.setFixedHeight(35); f=QHBoxLayout(footer); f.setContentsMargins(11,0,10,0); pi=QLabel(); pi.setPixmap(icon_pixmap("users",TEXT,19)); self.people=QLabel("0 People"); self.people.setFont(font(13,QFont.Weight.Medium)); self.fps=QLabel("-- FPS"); self.fps.setFont(font(13,QFont.Weight.Medium)); bars=QLabel("▂▄▆█"); bars.setStyleSheet(f"color:{GREEN};"); bars.setFont(font(13,QFont.Weight.Bold)); f.addWidget(pi); f.addWidget(self.people); f.addStretch(); f.addWidget(self.fps); f.addWidget(bars); o.addWidget(footer)
    def set_metrics(self,count,fps): self.people.setText(f"{count} {'Person' if count==1 else 'People'}"); self.fps.setText(f"{int(round(fps)) if fps else '--'} FPS")


class NavButton(QPushButton):
    def __init__(self,text,kind):
        super().__init__(text); self.setCheckable(True); self.setIcon(icon(kind,TEXT,25)); self.setIconSize(QSize(25,25)); self.setFixedHeight(58); self.setFont(font(16,QFont.Weight.Medium)); self.setCursor(Qt.CursorShape.PointingHandCursor)


class Sidebar(QFrame):
    def __init__(self,change_page):
        super().__init__(); self.setObjectName("sidebar"); self.setFixedWidth(235); l=QVBoxLayout(self); l.setContentsMargins(0,0,0,18); l.setSpacing(0); l.addWidget(logo_widget()); sep=QFrame(); sep.setFixedHeight(1); sep.setStyleSheet(f"background:{BORDER};"); l.addWidget(sep)
        nav=QWidget(); nl=QVBoxLayout(nav); nl.setContentsMargins(14,18,14,0); nl.setSpacing(8); self.buttons={}
        for i,(label,kind) in enumerate([("Live View","monitor"),("People","person"),("Events","bell"),("Reports","report"),("Settings","settings")]):
            b=NavButton(label,kind); b.clicked.connect(lambda checked=False,index=i:change_page(index)); nl.addWidget(b); self.buttons[i]=b
        nl.addStretch(); l.addWidget(nav,1)
        status=QFrame(); status.setObjectName("statusCard"); status.setFixedHeight(320); s=QVBoxLayout(status); s.setContentsMargins(16,16,16,14); top=QHBoxLayout(); pulse=QLabel(); pulse.setPixmap(icon_pixmap("activity",GREEN,28)); st=QLabel("System Status"); st.setFont(font(14,QFont.Weight.Medium)); top.addWidget(pulse); top.addWidget(st); top.addStretch(); s.addLayout(top); normal=QLabel("All systems normal"); normal.setAlignment(Qt.AlignmentFlag.AlignCenter); normal.setFont(font(13)); normal.setStyleSheet(f"color:{GREEN};"); s.addWidget(normal); s.addSpacing(14)
        for label,value,pct,color in [("CPU","28%",28,BLUE),("GPU","41%",41,BLUE),("Memory","6.1 / 15.8 GB",38,CYAN)]:
            row=QHBoxLayout(); a=QLabel(label); b=QLabel(value); a.setFont(font(13)); b.setFont(font(13)); row.addWidget(a); row.addStretch(); row.addWidget(b); s.addLayout(row); bar=QFrame(); bar.setFixedHeight(8); bar.setStyleSheet("background:#06325b;border-radius:4px;"); fill=QFrame(bar); fill.setGeometry(0,0,int(170*pct/100),8); fill.setStyleSheet(f"background:{color};border-radius:4px;"); s.addWidget(bar); s.addSpacing(10)
        l.addWidget(status)
    def set_active(self,index):
        for i,b in self.buttons.items(): b.setChecked(i==index)


class StatCard(QFrame):
    def __init__(self,kind,color,value,label,sub=""):
        super().__init__(); self.setObjectName("statCard"); self.setMinimumHeight(130); l=QVBoxLayout(self); l.setContentsMargins(18,20,16,16); top=QHBoxLayout(); ic=QLabel(); ic.setPixmap(icon_pixmap(kind,color,36)); n=QLabel(value); n.setFont(font(28,QFont.Weight.DemiBold)); top.addWidget(ic); top.addSpacing(11); top.addWidget(n); top.addStretch(); l.addLayout(top); t=QLabel(label); t.setFont(font(14)); t.setAlignment(Qt.AlignmentFlag.AlignCenter); l.addWidget(t)
        if sub: x=QLabel(sub); x.setFont(font(13)); x.setAlignment(Qt.AlignmentFlag.AlignCenter); x.setStyleSheet(f"color:{GREEN};"); l.addWidget(x)


def avatar(size=54):
    x=QLabel(); x.setFixedSize(size,size); x.setAlignment(Qt.AlignmentFlag.AlignCenter); x.setPixmap(icon_pixmap("person","#dce5f1",int(size*.68))); x.setStyleSheet("background:#243345;border:1px solid #526277;border-radius:6px;"); return x


class RecentViews(QFrame):
    def __init__(self):
        super().__init__(); self.setObjectName("recentCard"); l=QVBoxLayout(self); l.setContentsMargins(16,16,16,12); h=QHBoxLayout(); title=QLabel("Recent Views"); title.setFont(font(17,QFont.Weight.DemiBold)); va=QLabel("View All"); va.setFont(font(14)); va.setStyleSheet(f"color:{CYAN};"); h.addWidget(title); h.addStretch(); h.addWidget(va); l.addLayout(h); line=QFrame(); line.setFixedHeight(1); line.setStyleSheet(f"background:{BORDER};"); l.addWidget(line); scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.Shape.NoFrame); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); body=QWidget(); bl=QVBoxLayout(body); bl.setContentsMargins(0,6,0,6); bl.setSpacing(10)
        for name,camera,at in RECENT_ROWS:
            row=QWidget(); r=QHBoxLayout(row); r.setContentsMargins(0,0,0,0); r.setSpacing(12); r.addWidget(avatar()); text=QVBoxLayout(); n=QLabel(name); n.setFont(font(14,QFont.Weight.Medium)); c=QLabel(camera); c.setFont(font(13)); c.setStyleSheet(f"color:{MUTED};"); text.addWidget(n); text.addWidget(c); r.addLayout(text,1); t=QLabel(at); t.setFont(font(13)); t.setStyleSheet(f"color:{CYAN};"); r.addWidget(t); bl.addWidget(row)
        bl.addStretch(); scroll.setWidget(body); l.addWidget(scroll,1)


class RightRail(QWidget):
    def __init__(self):
        super().__init__(); self.setFixedWidth(350); l=QVBoxLayout(self); l.setContentsMargins(0,0,0,0); l.setSpacing(16); cards=QFrame(); cards.setObjectName("statsPanel"); g=QGridLayout(cards); g.setContentsMargins(12,12,12,12); g.setSpacing(12); g.addWidget(StatCard("users",BLUE,"23","Total People"),0,0); g.addWidget(StatCard("person",GREEN,"15","Known People"),0,1); g.addWidget(StatCard("person",ORANGE,"8","Unknown People"),1,0); g.addWidget(StatCard("camera",CYAN,"6/6","Active Cameras","Online"),1,1); l.addWidget(cards); l.addWidget(RecentViews(),1)


class SearchBar(QWidget):
    def __init__(self,placeholder,width=300):
        super().__init__(); self.setFixedWidth(width); self.setObjectName("searchBar"); l=QHBoxLayout(self); l.setContentsMargins(10,0,10,0); i=QLabel(); i.setPixmap(icon_pixmap("search",TEXT,20)); e=QLineEdit(); e.setPlaceholderText(placeholder); e.setFrame(False); e.setFont(font(14)); l.addWidget(i); l.addWidget(e,1)


class PageTitle(QWidget):
    def __init__(self,title,subtitle=""):
        super().__init__(); l=QVBoxLayout(self); l.setContentsMargins(0,0,0,0); l.setSpacing(3); t=QLabel(title); t.setFont(font(27,QFont.Weight.DemiBold)); l.addWidget(t)
        if subtitle: s=QLabel(subtitle); s.setFont(font(14)); s.setStyleSheet(f"color:{MUTED};"); l.addWidget(s)


class LiveViewPage(QWidget):
    def __init__(self):
        super().__init__(); l=QVBoxLayout(self); l.setContentsMargins(0,0,0,0); l.setSpacing(12); h=QHBoxLayout(); h.addWidget(PageTitle("Live View")); h.addStretch(); self.full=QPushButton(); self.full.setObjectName("squareButton"); self.full.setIcon(icon("fullscreen",TEXT,23)); self.full.setIconSize(QSize(23,23)); self.full.setFixedSize(42,42); h.addWidget(self.full); l.addLayout(h); self.tiles={}; g=QGridLayout(); g.setSpacing(12); g.setContentsMargins(0,0,0,0)
        for i,cid in enumerate(CAMERAS): tile=CameraTile(cid,i+1); g.addWidget(tile,i//2,i%2); self.tiles[cid]=tile
        for r in range(3): g.setRowStretch(r,1)
        for c in range(2): g.setColumnStretch(c,1)
        l.addLayout(g,1)


def item(text):
    x=QTableWidgetItem(text); x.setFlags(x.flags() & ~Qt.ItemFlag.ItemIsEditable); x.setTextAlignment(int(Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)); return x


class PeoplePage(QWidget):
    def __init__(self):
        super().__init__(); l=QVBoxLayout(self); l.setContentsMargins(0,0,0,0); l.setSpacing(12); top=QHBoxLayout(); top.addWidget(PageTitle("People","Manage and monitor all personnel in the system."),1); actions=QVBoxLayout(); add=QPushButton(" Add New Person"); add.setObjectName("primaryButton"); add.setIcon(icon("plusperson",TEXT,20)); add.setIconSize(QSize(20,20)); add.setFixedHeight(40); add.setFont(font(14,QFont.Weight.Medium)); actions.addWidget(add,0,Qt.AlignmentFlag.AlignRight); sr=QHBoxLayout(); sr.addWidget(SearchBar("Search people...",280)); filt=QPushButton(); filt.setObjectName("squareButton"); filt.setIcon(icon("filter",TEXT,21)); filt.setFixedSize(44,40); sr.addWidget(filt); actions.addLayout(sr); top.addLayout(actions); l.addLayout(top)
        table=QTableWidget(len(PEOPLE_ROWS),8); table.setHorizontalHeaderLabels(["ID","PHOTO","NAME","ROLE","DEPARTMENT","STATUS","ADDED","ACTIONS"]); table.verticalHeader().setVisible(False); table.setShowGrid(False); table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection); table.setFocusPolicy(Qt.FocusPolicy.NoFocus); table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel); table.setObjectName("dataTable"); table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for row,(pid,name,role,dept,status,added) in enumerate(PEOPLE_ROWS):
            table.setRowHeight(row,58); table.setItem(row,0,item(pid)); ph=QWidget(); pl=QHBoxLayout(ph); pl.setContentsMargins(8,4,8,4); pl.addWidget(avatar(42)); pl.addStretch(); table.setCellWidget(row,1,ph); table.setItem(row,2,item(name)); rc=GREEN if role=="Supervisor" else (PURPLE if role=="Admin" else CYAN); role_lab=QLabel(role); role_lab.setAlignment(Qt.AlignmentFlag.AlignCenter); role_lab.setFont(font(12)); role_lab.setStyleSheet(f"color:{rc};border:1px solid {rc};border-radius:4px;padding:3px 6px;"); rw=QWidget(); rl=QHBoxLayout(rw); rl.setContentsMargins(4,10,4,10); rl.addWidget(role_lab); rl.addStretch(); table.setCellWidget(row,3,rw); table.setItem(row,4,item(dept)); sw=QWidget(); sl=QHBoxLayout(sw); sl.setContentsMargins(3,0,3,0); dot=QLabel("●"); dot.setStyleSheet(f"color:{GREEN if status=='Active' else RED};"); txt=QLabel(status); txt.setStyleSheet(f"color:{GREEN if status=='Active' else TEXT};"); sl.addWidget(dot); sl.addWidget(txt); sl.addStretch(); table.setCellWidget(row,5,sw); table.setItem(row,6,item(added)); aw=QWidget(); al=QHBoxLayout(aw); al.setContentsMargins(0,6,0,6); al.setSpacing(8); edit=QPushButton(); edit.setObjectName("miniButton"); edit.setIcon(icon("edit",TEXT,18)); edit.setFixedSize(36,36); delete=QPushButton(); delete.setObjectName("dangerButton"); delete.setIcon(icon("trash",RED,18)); delete.setFixedSize(36,36); al.addWidget(edit); al.addWidget(delete); al.addStretch(); table.setCellWidget(row,7,aw)
        l.addWidget(table,1)


class EventCell(QWidget):
    def __init__(self,text,kind,color):
        super().__init__(); l=QHBoxLayout(self); l.setContentsMargins(2,3,2,3); l.setSpacing(12); box=QLabel(); box.setFixedSize(42,42); box.setAlignment(Qt.AlignmentFlag.AlignCenter); box.setPixmap(icon_pixmap(kind,color,24)); box.setStyleSheet(f"background:{color}18;border:1px solid {color}88;border-radius:6px;"); t=QLabel(text); t.setFont(font(13)); t.setWordWrap(True); l.addWidget(box); l.addWidget(t,1)


class EventsPage(QWidget):
    def __init__(self):
        super().__init__(); l=QVBoxLayout(self); l.setContentsMargins(0,0,0,0); l.setSpacing(14); h=QHBoxLayout(); h.addWidget(PageTitle("Events","View and manage all system events."),1); h.addWidget(SearchBar("Search events...",280)); filt=QPushButton(); filt.setObjectName("squareButton"); filt.setIcon(icon("filter",TEXT,21)); filt.setFixedSize(44,40); h.addWidget(filt); l.addLayout(h); table=QTableWidget(len(EVENT_ROWS),5); table.setHorizontalHeaderLabels(["TIME","EVENT","CAMERA / LOCATION","DETAILS","BY"]); table.verticalHeader().setVisible(False); table.setShowGrid(False); table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection); table.setFocusPolicy(Qt.FocusPolicy.NoFocus); table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel); table.setObjectName("dataTable"); table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for row,(when,event,camera,details,by,kind,color) in enumerate(EVENT_ROWS): table.setRowHeight(row,68); table.setItem(row,0,item(when)); table.setCellWidget(row,1,EventCell(event,kind,color)); table.setItem(row,2,item(camera)); table.setItem(row,3,item(details)); bw=QWidget(); bl=QHBoxLayout(bw); bl.setContentsMargins(0,0,0,0); ic=QLabel(); ic.setPixmap(icon_pixmap("person",MUTED,20)); lab=QLabel(by); lab.setFont(font(13)); bl.addWidget(ic); bl.addWidget(lab); bl.addStretch(); table.setCellWidget(row,4,bw)
        l.addWidget(table,1)


class Placeholder(QWidget):
    def __init__(self,title):
        super().__init__(); l=QVBoxLayout(self); l.setContentsMargins(0,0,0,0); l.addWidget(PageTitle(title)); card=QFrame(); card.setObjectName("emptyCard"); cl=QVBoxLayout(card); x=QLabel(title); x.setAlignment(Qt.AlignmentFlag.AlignCenter); x.setFont(font(24,QFont.Weight.Medium)); x.setStyleSheet(f"color:{MUTED};"); cl.addWidget(x); l.addWidget(card,1)


class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Apsidal"); self.resize(1672,941); self.setMinimumSize(1280,720); root=QWidget(); self.setCentralWidget(root); rl=QHBoxLayout(root); rl.setContentsMargins(0,0,0,0); rl.setSpacing(0); self.sidebar=Sidebar(self.set_page); rl.addWidget(self.sidebar); body=QWidget(); bl=QVBoxLayout(body); bl.setContentsMargins(0,0,0,0); bl.setSpacing(0); rl.addWidget(body,1)
        top=QWidget(); top.setObjectName("topbar"); top.setFixedHeight(68); tl=QHBoxLayout(top); tl.setContentsMargins(24,0,24,0); menu=QPushButton(); menu.setObjectName("topButton"); menu.setIcon(icon("menu",TEXT,27)); menu.setIconSize(QSize(27,27)); menu.setFixedSize(42,42); menu.clicked.connect(self.toggle_sidebar); tl.addWidget(menu); tl.addStretch(); self.clock=QLabel(); self.clock.setFont(font(16,QFont.Weight.Medium)); self.date=QLabel(); self.date.setFont(font(15)); tl.addWidget(self.clock); div=QFrame(); div.setFixedSize(1,20); div.setStyleSheet(f"background:{BORDER};"); tl.addSpacing(14); tl.addWidget(div); tl.addSpacing(14); tl.addWidget(self.date); bl.addWidget(top)
        content=QWidget(); cl=QHBoxLayout(content); cl.setContentsMargins(16,10,16,16); cl.setSpacing(16); bl.addWidget(content,1); self.stack=QStackedWidget(); self.live=LiveViewPage(); self.people=PeoplePage(); self.events=EventsPage(); self.reports=Placeholder("Reports"); self.settings=Placeholder("Settings");
        for page in (self.live,self.people,self.events,self.reports,self.settings): self.stack.addWidget(page)
        cl.addWidget(self.stack,1); cl.addWidget(RightRail()); self.live.full.clicked.connect(self.toggle_fullscreen)
        self.readers={}; self.seen={};
        for cid in CAMERAS: r=LatestFrameReader(cid); r.start(); self.readers[cid]=r; self.seen[cid]=-1
        self.detections=DetectionReader(); self.detections.start(); self.last_counts={cid:0 for cid in CAMERAS}; self.last_tick=time.monotonic(); self.frame_counts={cid:0 for cid in CAMERAS}
        self.render_timer=QTimer(self); self.render_timer.setTimerType(Qt.TimerType.PreciseTimer); self.render_timer.timeout.connect(self.render); self.render_timer.start(20); self.info_timer=QTimer(self); self.info_timer.timeout.connect(self.update_info); self.info_timer.start(1000); self.clock_timer=QTimer(self); self.clock_timer.timeout.connect(self.update_clock); self.clock_timer.start(1000); self.update_clock(); self.set_page(0); self.apply_theme()
    def apply_theme(self):
        self.setStyleSheet(f"""
        QMainWindow,QWidget{{background:{BG};color:{TEXT};}}
        #sidebar{{background:{SIDEBAR};border-right:1px solid #07243f;}}
        #topbar{{background:{BG};}}
        QPushButton{{border:0;color:{TEXT};outline:none;}}
        #sidebar QPushButton{{text-align:left;padding-left:16px;border-radius:7px;background:transparent;}}
        #sidebar QPushButton:hover{{background:#061d3a;}}
        #sidebar QPushButton:checked{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #075ff0,stop:1 #073bc8);border:1px solid #1470ff;}}
        #statusCard,#statsPanel,#recentCard,#cameraTile,#emptyCard{{background:{PANEL};border:1px solid {BORDER};border-radius:9px;}}
        #statCard{{background:{CARD};border-radius:8px;}}
        #squareButton,#miniButton,#topButton,#dangerButton{{background:{CARD};border:1px solid {BORDER};border-radius:7px;}}
        #squareButton:hover,#miniButton:hover,#topButton:hover{{background:#08294c;}}
        #dangerButton{{border-color:#6e1321;}}
        #primaryButton{{background:#0758e9;border:1px solid #1c71ff;border-radius:6px;padding:0 16px;}}
        #searchBar{{background:#001028;border:1px solid {BORDER};border-radius:7px;}}
        #searchBar QLineEdit{{background:transparent;color:{TEXT};selection-background-color:{BLUE};}}
        QTableWidget#dataTable{{background:{PANEL};border:1px solid {BORDER};border-radius:7px;gridline-color:{BORDER};color:{TEXT};}}
        QTableWidget#dataTable::item{{border-bottom:1px solid #082b49;padding:8px;}}
        QTableWidget#dataTable QHeaderView::section{{background:#00142d;color:#c7d2e2;border:0;border-bottom:1px solid {BORDER};padding:10px;font-size:12px;font-weight:600;}}
        QScrollArea{{background:transparent;border:0;}} QScrollArea>QWidget>QWidget{{background:transparent;}}
        QScrollBar:vertical{{background:#001126;width:7px;margin:2px;}} QScrollBar::handle:vertical{{background:#164770;min-height:28px;border-radius:3px;}} QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0;}}
        """)
    def set_page(self,index): self.stack.setCurrentIndex(index); self.sidebar.set_active(index)
    def toggle_sidebar(self): self.sidebar.setVisible(not self.sidebar.isVisible())
    def toggle_fullscreen(self): self.showNormal() if self.isFullScreen() else self.showFullScreen()
    def update_clock(self): now=datetime.now(); self.clock.setText(now.strftime("%H:%M:%S")); self.date.setText(now.strftime("%-d %b %Y"))
    def render(self):
        for cid,r in self.readers.items():
            image,version=r.latest()
            if image is not None and version>self.seen[cid]: self.seen[cid]=version; self.live.tiles[cid].image.set_frame(image)
    def update_info(self):
        now=time.monotonic(); dt=max(.1,now-self.last_tick); self.last_tick=now; counts=self.detections.counts()
        for cid,r in self.readers.items(): previous=self.frame_counts[cid]; current=r.frames; self.frame_counts[cid]=current; fps=(current-previous)/dt; self.live.tiles[cid].set_metrics(counts.get(cid,0),fps)
    def closeEvent(self,event):
        self.render_timer.stop(); self.info_timer.stop(); self.clock_timer.stop(); self.detections.stop();
        for r in self.readers.values(): r.stop()
        event.accept()


def run():
    app=QApplication.instance() or QApplication([]); app.setStyle("Fusion"); window=DashboardWindow(); window.showMaximized(); return app.exec()
