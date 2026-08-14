from __future__ import annotations

import http.client
import threading
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QProgressBar, QPushButton, QSizePolicy,
    QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .dashboard import CAMERAS, FrameReader, RealtimeState, ML_HOST, ML_PORT

BG = "#0f141a"
TOP = "#151c24"
PANEL = "#151c24"
CARD = "#1b242e"
CARD2 = "#202b37"
BORDER = "#2b3745"
TEXT = "#eef3f8"
MUTED = "#8fa0b5"
FAINT = "#607086"
BLUE = "#347ff0"
GREEN = "#21c978"
ORANGE = "#f0a128"
RED = "#f2545b"
CYAN = "#36b9e9"

CAM_CHANNELS = {
    "CAM-01": "Channel 101", "CAM-02": "Channel 201", "CAM-03": "Channel 301",
    "CAM-04": "Channel 401", "CAM-05": "Channel 501", "CAM-06": "Channel 601",
}


def font(px: int, weight=QFont.Weight.Normal):
    f = QFont("Inter")
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


def _item(value):
    item = QTableWidgetItem(str(value))
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "0%", accent=GREEN):
        super().__init__()
        self.setObjectName("metricCard")
        self.setFixedSize(118, 70)
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        row = QHBoxLayout()
        label = QLabel(title)
        label.setFont(font(11, QFont.Weight.DemiBold))
        label.setStyleSheet(f"color:{MUTED};")
        self.value = QLabel(value)
        self.value.setFont(font(12, QFont.Weight.Bold))
        row.addWidget(label)
        row.addStretch()
        row.addWidget(self.value)
        root.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(5)
        self.bar.setStyleSheet(
            f"QProgressBar{{background:#202a35;border:0;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:{accent};border-radius:2px;}}"
        )
        root.addWidget(self.bar)

    def set_value(self, value: float):
        value = max(0.0, min(100.0, float(value)))
        self.value.setText(f"{value:.0f}%")
        self.bar.setValue(int(value))


class VideoCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.setMinimumSize(220, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def set_image(self, image):
        self.image = image
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#05090d"))
        if self.image is None or self.image.isNull():
            p.setPen(QColor(MUTED))
            p.setFont(font(12))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Connecting...")
            return
        iw, ih = float(self.image.width()), float(self.image.height())
        tw, th = float(max(1, self.width())), float(max(1, self.height()))
        scale = min(tw / iw, th / ih)
        w, h = iw * scale, ih * scale
        rect = QRectF((tw - w) / 2.0, (th - h) / 2.0, w, h)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawImage(rect, self.image)


class CameraCard(QFrame):
    def __init__(self, camera_id: str):
        super().__init__()
        self.camera_id = camera_id
        self.setObjectName("cameraCard")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        head = QFrame()
        head.setFixedHeight(31)
        head.setStyleSheet("background:#151b22;border-bottom:1px solid #29323d;")
        h = QHBoxLayout(head)
        h.setContentsMargins(10, 0, 9, 0)
        self.name = QLabel(f"{camera_id}  ·  {CAM_CHANNELS.get(camera_id, '')}")
        self.name.setFont(font(12, QFont.Weight.Bold))
        self.status = QLabel("● offline")
        self.status.setFont(font(11, QFont.Weight.DemiBold))
        self.status.setStyleSheet(f"color:{RED};")
        h.addWidget(self.name)
        h.addStretch()
        h.addWidget(self.status)
        root.addWidget(head)

        self.video = VideoCanvas()
        root.addWidget(self.video, 1)

        foot = QFrame()
        foot.setFixedHeight(25)
        foot.setStyleSheet("background:#10161c;border-top:1px solid #252e38;")
        f = QHBoxLayout(foot)
        f.setContentsMargins(8, 0, 8, 0)
        self.fps = QLabel("-- FPS")
        self.people = QLabel("👥 0")
        self.ai = QLabel("🤖 AI ON")
        self.ai.setStyleSheet(f"color:{GREEN};")
        for widget in (self.fps, self.people, self.ai):
            widget.setFont(font(10, QFont.Weight.DemiBold))
        f.addWidget(self.fps)
        f.addSpacing(12)
        f.addWidget(self.people)
        f.addStretch()
        f.addWidget(self.ai)
        root.addWidget(foot)

    def set_metrics(self, online: bool, fps: float, count: int):
        self.status.setText("● online" if online else "● offline")
        self.status.setStyleSheet(f"color:{GREEN if online else RED};")
        self.fps.setText(f"{fps:.0f} FPS" if fps else "-- FPS")
        self.people.setText(f"👥 {int(count)}")


class ToggleButton(QPushButton):
    def __init__(self, text: str):
        super().__init__(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(34)
        self.setFont(font(11, QFont.Weight.DemiBold))


class DashboardPage(QWidget):
    def __init__(self, toggle_callback):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        head = QHBoxLayout()
        title = QLabel("Dashboard")
        title.setFont(font(20, QFont.Weight.Bold))
        hint = QLabel("Double-click → fullscreen · scroll → zoom")
        hint.setFont(font(10))
        hint.setStyleSheet(f"color:{FAINT};")
        self.heat = ToggleButton("Heatmap")
        self.pose = ToggleButton("Pose")
        self.heat.clicked.connect(lambda checked: toggle_callback("heatmap", checked))
        self.pose.clicked.connect(lambda checked: toggle_callback("pose", checked))
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("🔍 Filter camera...")
        self.filter.setFixedWidth(160)
        self.grid_select = QComboBox()
        self.grid_select.addItems(["3 × 2", "2 × 2", "1 × 1"])
        head.addWidget(title)
        head.addSpacing(8)
        head.addWidget(hint)
        head.addStretch()
        head.addWidget(self.heat)
        head.addWidget(self.pose)
        head.addSpacing(8)
        head.addWidget(self.filter)
        head.addWidget(self.grid_select)
        root.addLayout(head)

        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(10)
        self.cards = {}
        for i, cid in enumerate(CAMERAS):
            card = CameraCard(cid)
            self.cards[cid] = card
            self.grid.addWidget(card, i // 2, i % 2)
        for r in range(3):
            self.grid.setRowStretch(r, 1)
        for c in range(2):
            self.grid.setColumnStretch(c, 1)
        root.addLayout(self.grid, 1)

    def set_overlay_state(self, heatmap: bool, pose: bool):
        self.heat.blockSignals(True)
        self.pose.blockSignals(True)
        self.heat.setChecked(bool(heatmap))
        self.pose.setChecked(bool(pose))
        self.heat.blockSignals(False)
        self.pose.blockSignals(False)


class PersonManagementPage(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        title = QLabel("Person Management")
        title.setFont(font(20, QFont.Weight.Bold))
        self.count = QLabel("0 active")
        self.count.setStyleSheet(f"color:{MUTED};")
        sync = QPushButton("↻ DB Sync")
        search = QLineEdit(); search.setPlaceholderText("🔍 Search people..."); search.setFixedWidth(160)
        enroll = QPushButton("＋ Enroll New"); enroll.setObjectName("primary")
        head.addWidget(title); head.addWidget(self.count); head.addStretch(); head.addWidget(sync); head.addWidget(search); head.addWidget(enroll)
        root.addLayout(head)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Photo", "Name / Global ID", "Camera", "Status", "Last Seen", "Observations"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide(); self.table.setShowGrid(False)
        root.addWidget(self.table, 1)

    def update_state(self, state):
        global_state = (((state.get("reid") or {}).get("state") or {}).get("global") or {})
        rows = []
        for gid, value in global_state.items():
            tracks = value.get("active_tracks") or {}
            if not tracks:
                continue
            name = value.get("name") or value.get("known_name") or gid
            rows.append(("👤", name, ", ".join(sorted(tracks.keys())), "Active", "now", value.get("observations", 0)))
        self.count.setText(f"{len(rows)} active")
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self.table.setRowHeight(r, 54)
            for c, value in enumerate(row):
                self.table.setItem(r, c, _item(value))


class SimplePage(QWidget):
    def __init__(self, title: str, subtitle: str):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        h = QLabel(title); h.setFont(font(20, QFont.Weight.Bold))
        s = QLabel(subtitle); s.setStyleSheet(f"color:{MUTED};"); s.setFont(font(11))
        root.addWidget(h); root.addWidget(s)
        box = QFrame(); box.setObjectName("panel")
        inside = QVBoxLayout(box); inside.setContentsMargins(22, 22, 22, 22)
        self.body = QLabel("Core v1 realtime service connected.")
        self.body.setFont(font(13)); self.body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inside.addWidget(self.body, 1)
        root.addWidget(box, 1)


class AnalyticsPage(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0)
        h = QLabel("Analytics"); h.setFont(font(20,QFont.Weight.Bold)); root.addWidget(h)
        self.summary = QLabel("Waiting for heatmap / pose metrics..."); self.summary.setStyleSheet(f"color:{MUTED};"); root.addWidget(self.summary)
        self.table = QTableWidget(0,7)
        self.table.setHorizontalHeaderLabels(["Camera","Samples","Pose","BBox fallback","Peak","Ankle skips","Pose frame"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide(); self.table.setShowGrid(False)
        root.addWidget(self.table,1)

    def update_state(self, state):
        heat = (state.get("health") or {}).get("heatmap") or {}
        cams = heat.get("cameras") or {}
        self.summary.setText(f"Heatmap accumulating: {bool(heat.get('accumulating'))} · source: {heat.get('source','—')}")
        self.table.setRowCount(len(cams))
        for r,(cid,v) in enumerate(sorted(cams.items())):
            row=[cid,v.get("samples",0),v.get("pose_samples",0),v.get("bbox_fallback_samples",0),f"{float(v.get('peak') or 0):.2f}",v.get("ankle_skips",0),v.get("last_pose_frame",-1)]
            for c,value in enumerate(row): self.table.setItem(r,c,_item(value))


class EventsPage(QWidget):
    def __init__(self):
        super().__init__()
        root=QVBoxLayout(self); root.setContentsMargins(0,0,0,0)
        head=QHBoxLayout(); title=QLabel("Events"); title.setFont(font(20,QFont.Weight.Bold)); self.records=QLabel("0 records"); self.records.setStyleSheet(f"color:{MUTED};")
        day=QPushButton("📅 Bugun"); search=QLineEdit(); search.setPlaceholderText("🔍 Search..."); export=QPushButton("▣ Export CSV")
        head.addWidget(title); head.addWidget(self.records); head.addStretch(); head.addWidget(day); head.addWidget(search); head.addWidget(export); root.addLayout(head)
        self.table=QTableWidget(0,5); self.table.setHorizontalHeaderLabels(["Time","Camera","Person","Type","Confidence"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); self.table.verticalHeader().hide(); self.table.setShowGrid(False); root.addWidget(self.table,1)

    def update_events(self, events):
        self.records.setText(f"{len(events)} records")
        self.table.setRowCount(len(events))
        for r,e in enumerate(events):
            sim=e.get("similarity"); conf=f"{float(sim):.3f}" if isinstance(sim,(float,int)) else "—"
            row=[e.get("time",""),e.get("camera",""),e.get("global_id","Unknown"),e.get("reason","person_detected"),conf]
            for c,value in enumerate(row): self.table.setItem(r,c,_item(value))


class RightRail(QFrame):
    def __init__(self):
        super().__init__(); self.setObjectName("rightRail"); self.setFixedWidth(275)
        root=QVBoxLayout(self); root.setContentsMargins(14,14,14,14); root.setSpacing(12)
        title=QLabel("LIVE STATUS"); title.setFont(font(11,QFont.Weight.Bold)); title.setStyleSheet(f"color:{MUTED};letter-spacing:1px;"); root.addWidget(title)
        statrow=QHBoxLayout(); self.known=self._stat("0","KNOWN",GREEN); self.unknown=self._stat("0","UNKNOWN",ORANGE); statrow.addWidget(self.known[0]); statrow.addWidget(self.unknown[0]); root.addLayout(statrow)
        sys=QLabel("SYSTEM"); sys.setFont(font(10,QFont.Weight.Bold)); sys.setStyleSheet(f"color:{MUTED};letter-spacing:1px;"); root.addWidget(sys)
        self.gpu=self._bar(root,"GPU"); self.cpu=self._bar(root,"CPU"); self.fps=self._bar(root,"FPS")
        alerts=QLabel("ALERTS"); alerts.setFont(font(10,QFont.Weight.Bold)); alerts.setStyleSheet(f"color:{MUTED};letter-spacing:1px;"); root.addWidget(alerts)
        self.alerts=QVBoxLayout(); root.addLayout(self.alerts)
        recent=QLabel("RECENT EVENTS"); recent.setFont(font(10,QFont.Weight.Bold)); recent.setStyleSheet(f"color:{MUTED};letter-spacing:1px;"); root.addWidget(recent)
        self.recent=QVBoxLayout(); root.addLayout(self.recent); root.addStretch()

    def _stat(self,value,label,color):
        frame=QFrame(); frame.setObjectName("smallStat"); l=QVBoxLayout(frame); l.setContentsMargins(11,8,11,8)
        v=QLabel(value); v.setFont(font(23,QFont.Weight.Bold)); v.setStyleSheet(f"color:{color};"); t=QLabel(label); t.setFont(font(9,QFont.Weight.Bold)); t.setStyleSheet(f"color:{MUTED};"); l.addWidget(v); l.addWidget(t); return frame,v

    def _bar(self,root,label):
        row=QHBoxLayout(); name=QLabel(label); name.setFont(font(10,QFont.Weight.DemiBold)); val=QLabel("0%"); val.setFont(font(10)); row.addWidget(name); row.addStretch(); row.addWidget(val); root.addLayout(row)
        bar=QProgressBar(); bar.setRange(0,100); bar.setTextVisible(False); bar.setFixedHeight(7); root.addWidget(bar); return val,bar

    @staticmethod
    def _clear(layout):
        while layout.count():
            item=layout.takeAt(0); widget=item.widget()
            if widget: widget.deleteLater()

    def update_state(self,state,events):
        global_state=(((state.get("reid") or {}).get("state") or {}).get("global") or {})
        active=[v for v in global_state.values() if v.get("active_tracks")]
        known=sum(1 for v in active if v.get("name") or v.get("known_name") or v.get("person_id"))
        self.known[1].setText(str(known)); self.unknown[1].setText(str(max(0,len(active)-known)))
        health=state.get("health") or {}; res=health.get("service_resources") or {}; gpu=float(res.get("gpu_utilization_percent") or 0); cpu=float(res.get("cpu_percent") or 0)
        pubs=health.get("publishers") or {}; rates=[float(v.get("publish_rate") or 0) for v in pubs.values()]; fps=sum(rates)/len(rates) if rates else 0
        for pair,value,suffix in ((self.gpu,gpu,"%"),(self.cpu,cpu,"%"),(self.fps,min(100,fps*5),"")):
            pair[0].setText(f"{value:.0f}{suffix}"); pair[1].setValue(max(0,min(100,int(value))))
        self._clear(self.alerts)
        offline=[cid for cid,v in (health.get("cameras") or {}).items() if not v.get("online")]
        for cid in offline[:3]:
            lab=QLabel(f"│ {cid} — Camera offline"); lab.setStyleSheet(f"color:{RED};background:#1c222b;padding:8px;border-radius:5px;"); self.alerts.addWidget(lab)
        if not offline:
            lab=QLabel("● All cameras healthy"); lab.setStyleSheet(f"color:{GREEN};padding:6px;"); self.alerts.addWidget(lab)
        self._clear(self.recent)
        for e in events[:8]:
            lab=QLabel(f"{e.get('time','')}  ●  {e.get('camera','')} · {e.get('global_id','Unknown')}"); lab.setFont(font(9)); lab.setStyleSheet(f"color:{MUTED};padding:2px;"); self.recent.addWidget(lab)


class Sidebar(QFrame):
    ITEMS=[("🎥","Dashboard"),("👥","Person Management"),("▣","Enrollment"),("📈","Analytics"),("⚡","Events"),("⚙","Settings")]
    def __init__(self,callback):
        super().__init__(); self.setObjectName("sidebar"); self.setFixedWidth(238)
        root=QVBoxLayout(self); root.setContentsMargins(12,16,12,16); root.setSpacing(8)
        title=QLabel("CONTROL PANEL"); title.setFont(font(10,QFont.Weight.Bold)); title.setStyleSheet(f"color:{MUTED};letter-spacing:1.6px;padding-left:10px;"); root.addWidget(title)
        self.buttons=[]
        for i,(icon,label) in enumerate(self.ITEMS):
            b=QPushButton(f"{icon}   {label}"); b.setCheckable(True); b.setFixedHeight(48); b.setFont(font(12,QFont.Weight.Medium)); b.clicked.connect(lambda checked=False,index=i: callback(index)); root.addWidget(b); self.buttons.append(b)
        root.addStretch(); collapse=QLabel("〈  Collapse"); collapse.setStyleSheet(f"color:{MUTED};padding:8px;"); root.addWidget(collapse); self.set_active(0)
    def set_active(self,index):
        for i,b in enumerate(self.buttons): b.setChecked(i==index)


class TopBar(QFrame):
    def __init__(self):
        super().__init__(); self.setObjectName("topbar"); self.setFixedHeight(72)
        root=QHBoxLayout(self); root.setContentsMargins(16,0,16,0); root.setSpacing(12)
        logo=QLabel("◎"); logo.setAlignment(Qt.AlignmentFlag.AlignCenter); logo.setFixedSize(38,38); logo.setFont(font(24,QFont.Weight.Bold)); logo.setStyleSheet(f"background:{BLUE};color:white;border-radius:10px;")
        texts=QVBoxLayout(); title=QLabel("AI Surveillance System"); title.setFont(font(16,QFont.Weight.Bold)); sub=QLabel("Operator Console • v3.0 MUKAMMAL"); sub.setFont(font(9)); sub.setStyleSheet(f"color:{MUTED};"); texts.addWidget(title); texts.addWidget(sub)
        self.search=QLineEdit(); self.search.setPlaceholderText("🔍  Search events, people..."); self.search.setFixedWidth(185)
        self.gpu=MetricCard("GPU"); self.cpu=MetricCard("CPU"); self.ram=MetricCard("RAM")
        self.ai=QLabel("● AI ACTIVE"); self.ai.setAlignment(Qt.AlignmentFlag.AlignCenter); self.ai.setFixedSize(105,70); self.ai.setFont(font(11,QFont.Weight.Bold)); self.ai.setStyleSheet("background:#12301f;color:#27d17f;border:1px solid #245337;border-radius:10px;")
        self.cams=QLabel("🎥 0/6"); self.cams.setAlignment(Qt.AlignmentFlag.AlignCenter); self.cams.setFixedSize(70,70); self.cams.setStyleSheet("background:#202934;border:1px solid #303b48;border-radius:10px;"); self.cams.setFont(font(11,QFont.Weight.Bold))
        self.clock=QLabel("--:--:--"); self.clock.setFont(font(11,QFont.Weight.Medium)); self.operator=QLabel("●  Operator"); self.operator.setFont(font(11,QFont.Weight.DemiBold))
        root.addWidget(logo); root.addLayout(texts); root.addStretch(1); root.addWidget(self.search); root.addStretch(1); root.addWidget(self.gpu); root.addWidget(self.cpu); root.addWidget(self.ram); root.addWidget(self.ai); root.addWidget(self.cams); root.addWidget(self.clock); root.addWidget(self.operator)

    def update_state(self,state):
        health=state.get("health") or {}; res=health.get("service_resources") or {}; self.gpu.set_value(res.get("gpu_utilization_percent") or 0); self.cpu.set_value(res.get("cpu_percent") or 0)
        rss=float(res.get("rss_mb") or 0); self.ram.set_value(min(100.0,rss/8192.0*100.0)); self.cams.setText(f"🎥 {health.get('online',0)}/{health.get('total',6)}")
        ready=bool((health.get("detector") or {}).get("ready")); self.ai.setText("● AI ACTIVE" if ready else "● AI STARTING"); self.ai.setStyleSheet(("background:#12301f;color:#27d17f;" if ready else "background:#332712;color:#f3b343;")+"border:1px solid #304236;border-radius:10px;")


class OperatorWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("AI Surveillance System — Operator Console"); self.setMinimumSize(1280,760)
        central=QWidget(); self.setCentralWidget(central); outer=QVBoxLayout(central); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        self.top=TopBar(); outer.addWidget(self.top)
        body=QHBoxLayout(); body.setContentsMargins(0,0,0,0); body.setSpacing(0); outer.addLayout(body,1)
        self.sidebar=Sidebar(self.change_page); body.addWidget(self.sidebar)
        center=QWidget(); center.setObjectName("center"); cl=QVBoxLayout(center); cl.setContentsMargins(16,14,16,14); self.stack=QStackedWidget(); cl.addWidget(self.stack); body.addWidget(center,1)
        self.dashboard=DashboardPage(self.set_overlay); self.people=PersonManagementPage(); self.enrollment=SimplePage("Enrollment","Register known people without disturbing the live camera pipeline."); self.analytics=AnalyticsPage(); self.events=EventsPage(); self.settings=SimplePage("Settings","Runtime controls and service configuration.")
        for page in (self.dashboard,self.people,self.enrollment,self.analytics,self.events,self.settings): self.stack.addWidget(page)
        self.rail=RightRail(); body.addWidget(self.rail)

        self.readers={cid:FrameReader(cid) for cid in CAMERAS}; self.state_reader=RealtimeState()
        for reader in self.readers.values(): reader.start()
        self.state_reader.start()
        self._versions={cid:-1 for cid in CAMERAS}
        self.render_timer=QTimer(self); self.render_timer.timeout.connect(self.render_frames); self.render_timer.start(33)
        self.state_timer=QTimer(self); self.state_timer.timeout.connect(self.refresh_state); self.state_timer.start(350)
        self.clock_timer=QTimer(self); self.clock_timer.timeout.connect(self.tick_clock); self.clock_timer.start(1000); self.tick_clock()
        self.apply_style()

    def change_page(self,index): self.stack.setCurrentIndex(index); self.sidebar.set_active(index)

    def tick_clock(self): self.top.clock.setText(datetime.now().strftime("%d %b %Y  %H:%M:%S"))

    def render_frames(self):
        for cid,reader in self.readers.items():
            image,version=reader.latest()
            if image is not None and version>self._versions[cid]: self._versions[cid]=version; self.dashboard.cards[cid].video.set_image(image)

    def refresh_state(self):
        state,recent,events=self.state_reader.snapshot(); self.top.update_state(state); self.rail.update_state(state,events); self.people.update_state(state); self.analytics.update_state(state); self.events.update_events(events)
        health=state.get("health") or {}; cameras=health.get("cameras") or {}; detections=(state.get("detections") or {}).get("cameras") or {}; pubs=health.get("publishers") or {}
        for cid,card in self.dashboard.cards.items():
            cam=cameras.get(cid) or {}; det=detections.get(cid) or {}; count=len(det.get("boxes") or []); fps=float((pubs.get(cid) or {}).get("publish_rate") or cam.get("fps") or 0); card.set_metrics(bool(cam.get("online")),fps,count)
        overlays=health.get("overlays") or {}; self.dashboard.set_overlay_state(bool(overlays.get("heatmap_visible")),bool(overlays.get("pose_visible")))

    def set_overlay(self,kind,enabled):
        def work():
            conn=None
            try:
                conn=http.client.HTTPConnection(ML_HOST,ML_PORT,timeout=2.0); conn.request("POST",f"/overlays/{kind}/{'on' if enabled else 'off'}",headers={"Connection":"close"}); response=conn.getresponse(); response.read()
            except Exception:
                pass
            finally:
                if conn:
                    try: conn.close()
                    except Exception: pass
        threading.Thread(target=work,name=f"ui-toggle-{kind}",daemon=True).start()

    def apply_style(self):
        self.setStyleSheet(f"""
            QWidget {{ background:{BG}; color:{TEXT}; font-family:'Inter','DejaVu Sans'; }}
            #topbar {{ background:{TOP}; border-bottom:1px solid {BORDER}; }}
            #sidebar {{ background:#141b22; border-right:1px solid {BORDER}; }}
            #rightRail {{ background:#11171e; border-left:1px solid {BORDER}; }}
            #center {{ background:#0d1217; }}
            QPushButton {{ background:#202934; border:1px solid #2d3946; border-radius:7px; padding:6px 12px; color:{TEXT}; }}
            QPushButton:hover {{ background:#273341; }}
            QPushButton:checked {{ background:{BLUE}; border-color:{BLUE}; color:white; }}
            QPushButton#primary {{ background:{BLUE}; border-color:{BLUE}; }}
            QLineEdit, QComboBox {{ background:#202934; border:1px solid #2c3947; border-radius:7px; padding:7px 10px; color:{TEXT}; }}
            #metricCard, #smallStat, #panel {{ background:{CARD}; border:1px solid {BORDER}; border-radius:9px; }}
            #cameraCard {{ background:#0a0f14; border:1px solid #25313c; border-radius:3px; }}
            QTableWidget {{ background:{CARD}; alternate-background-color:#202a35; border:1px solid {BORDER}; border-radius:8px; gridline-color:{BORDER}; selection-background-color:#29445f; }}
            QHeaderView::section {{ background:#202b37; color:#aebcd0; padding:9px; border:0; border-bottom:1px solid #314050; font-weight:600; }}
            QTableWidget::item {{ padding:7px; border-bottom:1px solid #24303b; }}
            QProgressBar {{ background:#202a35; border:0; border-radius:3px; }}
            QProgressBar::chunk {{ background:{BLUE}; border-radius:3px; }}
        """)

    def closeEvent(self,event):
        self.render_timer.stop(); self.state_timer.stop(); self.clock_timer.stop(); self.state_reader.stop()
        for reader in self.readers.values(): reader.stop()
        event.accept()


def run():
    app=QApplication.instance() or QApplication([]); app.setStyle("Fusion"); window=OperatorWindow(); window.showMaximized(); return app.exec()
