from __future__ import annotations

import http.client
import json
import math
import threading
import time
from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from shared.config import camera_config


ML_HOST = "127.0.0.1"
ML_PORT = 8001

BG = "#020817"
PANEL = "#071326"
CARD = "#091a31"
BORDER = "#173556"
TEXT = "#f5f8fc"
MUTED = "#91a4bc"
GREEN = "#19d67c"
ORANGE = "#ffad33"
RED = "#ff5266"
CYAN = "#32c5ff"

_CAMERA_CONFIG = [
    item
    for item in camera_config().get("cameras", [])
    if item.get("id") and item.get("online", True)
]
CAMERAS = [str(item["id"]) for item in _CAMERA_CONFIG]
CAMERA_TITLES = {
    str(item["id"]): " · ".join(
        value
        for value in (
            str(item.get("name") or item["id"]),
            str(item.get("location") or ""),
        )
        if value
    )
    for item in _CAMERA_CONFIG
}


def font(size: int, weight=QFont.Weight.Normal) -> QFont:
    value=QFont("Inter");value.setPixelSize(size);value.setWeight(weight);return value


class LatestFrameReader:
    """Long-poll the latest JPEG without creating a presentation queue."""
    def __init__(self,camera_id):
        self.camera_id=camera_id;self._stop=threading.Event();self._lock=threading.Lock();self._image=None;self._version=-1;self.frames=0;self.errors=0;self.last_frame_at=0.0;self.last_status=0;self.last_error="";self._thread=None
    def start(self):self._thread=threading.Thread(target=self._run,name=f"frontend-{self.camera_id}",daemon=True);self._thread.start()
    def stop(self):self._stop.set()
    def latest(self):
        with self._lock:return {"image":self._image,"version":self._version,"frames":self.frames,"errors":self.errors,"last_frame_at":self.last_frame_at,"status":self.last_status,"error":self.last_error}
    def _run(self):
        version=-1;connection=None
        while not self._stop.is_set():
            try:
                if connection is None:connection=http.client.HTTPConnection(ML_HOST,ML_PORT,timeout=2.0)
                connection.request("GET",f"/frame/{self.camera_id}?after={version}&wait_ms=180",headers={"Cache-Control":"no-cache","Connection":"keep-alive"});response=connection.getresponse();jpg=response.read()
                with self._lock:self.last_status=int(response.status)
                if response.status!=200:
                    with self._lock:self.last_error="Kamera kadri hali tayyor emas" if response.status==503 else f"HTTP {response.status}"
                    self._stop.wait(.25 if response.status==503 else .75);continue
                next_version=int(response.getheader("X-Frame-Version") or version+1)
                if next_version<=version:
                    continue
                image=QImage.fromData(jpg,"JPG")
                if image.isNull():raise RuntimeError("JPEG dekodlanmadi")
                version=next_version
                with self._lock:self._image=image;self._version=version;self.frames+=1;self.last_frame_at=time.monotonic();self.last_error=""
            except Exception as exc:
                with self._lock:self.errors+=1;self.last_status=0;self.last_error=str(exc)
                if connection is not None:
                    try:connection.close()
                    except Exception:pass
                connection=None;self._stop.wait(.75)
        if connection is not None:
            try:connection.close()
            except Exception:pass


class LiveStateReader:
    """Poll only real health, detection and ReID state from the ML service."""
    def __init__(self):
        self._stop=threading.Event();self._lock=threading.Lock();self._thread=None;self._state={"connected":False,"error":"ML service bilan aloqa yo'q","updated_at":0.0,"health":{},"detections":{},"reid":{}}
    def start(self):self._thread=threading.Thread(target=self._run,name="frontend-live-state",daemon=True);self._thread.start()
    def stop(self):self._stop.set()
    def snapshot(self):
        with self._lock:return dict(self._state)
    def _set(self,**values):
        with self._lock:self._state={**self._state,**values}
    @staticmethod
    def _json(connection,path):
        connection.request("GET",path,headers={"Cache-Control":"no-cache","Connection":"keep-alive"});response=connection.getresponse();payload=response.read()
        if response.status!=200:raise RuntimeError(f"HTTP {response.status}")
        return json.loads(payload.decode("utf-8"))
    def _run(self):
        connection=None;health={};detections={};reid={};next_health=0.0;next_reid=0.0
        while not self._stop.is_set():
            try:
                if connection is None:connection=http.client.HTTPConnection(ML_HOST,ML_PORT,timeout=1.5)
                now=time.monotonic()
                if now>=next_health:health=self._json(connection,"/health");next_health=now+1.0
                detections=self._json(connection,"/detections")
                if now>=next_reid:reid=self._json(connection,"/reid");next_reid=now+.75
                self._set(connected=True,error="",updated_at=time.monotonic(),health=health,detections=detections,reid=reid);self._stop.wait(.25)
            except Exception as exc:
                if connection is not None:
                    try:connection.close()
                    except Exception:pass
                connection=None;self._set(connected=False,error=str(exc),updated_at=time.monotonic());self._stop.wait(.75)
        if connection is not None:
            try:connection.close()
            except Exception:pass


class CameraImage(QLabel):
    def __init__(self):
        super().__init__("Ulanmoqda...");self._image=None;self.setAlignment(Qt.AlignmentFlag.AlignCenter);self.setMinimumSize(180,90);self.setFont(font(14,QFont.Weight.Medium));self.setStyleSheet(f"background:#010611;color:{MUTED};border:0;")
    def set_frame(self,image):self._image=image;self._apply()
    def set_message(self,message,clear=False):
        if clear:self._image=None;self.clear();self.setText(message)
        elif self._image is None:self.setText(message)
    def _apply(self):
        if self._image is None or self.width()<2 or self.height()<2:return
        # Show the complete source frame; the previous expanding mode cropped it.
        self.setPixmap(QPixmap.fromImage(self._image).scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.FastTransformation))
    def resizeEvent(self,event):super().resizeEvent(event);self._apply()


class CameraTile(QFrame):
    def __init__(self,camera_id,ordinal):
        super().__init__();self.camera_id=camera_id;self.setObjectName("cameraTile");outer=QVBoxLayout(self);outer.setContentsMargins(0,0,0,0);outer.setSpacing(0)
        header=QWidget();header.setFixedHeight(40);row=QHBoxLayout(header);row.setContentsMargins(9,0,9,0)
        chip=QLabel(f"{ordinal:02d}");chip.setFixedSize(29,27);chip.setAlignment(Qt.AlignmentFlag.AlignCenter);chip.setFont(font(12,QFont.Weight.DemiBold));chip.setStyleSheet("background:#164e9b;border-radius:6px;")
        title=QLabel(CAMERA_TITLES.get(camera_id,camera_id));title.setFont(font(13,QFont.Weight.DemiBold));self.status_dot=QLabel("●");self.status_text=QLabel("ULANMOQDA");self.status_text.setFont(font(11,QFont.Weight.DemiBold))
        row.addWidget(chip);row.addWidget(title);row.addStretch();row.addWidget(self.status_dot);row.addWidget(self.status_text);outer.addWidget(header)
        self.image=CameraImage();outer.addWidget(self.image,1)
        footer=QWidget();footer.setFixedHeight(32);row=QHBoxLayout(footer);row.setContentsMargins(9,0,9,0)
        self.people=QLabel("— odam");self.identities=QLabel("ID: —");self.fps=QLabel("— FPS");self.age=QLabel("— ms")
        for value in (self.people,self.identities,self.fps,self.age):value.setFont(font(11,QFont.Weight.Medium))
        self.identities.setStyleSheet(f"color:{MUTED};");self.age.setStyleSheet(f"color:{MUTED};")
        row.addWidget(self.people);row.addSpacing(7);row.addWidget(self.identities);row.addStretch();row.addWidget(self.fps);row.addSpacing(7);row.addWidget(self.age);outer.addWidget(footer)
    def _status(self,text,color):self.status_text.setText(text);self.status_text.setStyleSheet(f"color:{color};");self.status_dot.setStyleSheet(f"color:{color};")
    def update_live(self,connected,camera,publisher,detection_count,identity_count,reader,display_fps):
        frame_age=time.monotonic()-float(reader.get("last_frame_at") or 0.0);online=bool(camera.get("online"));fresh=reader.get("image") is not None and frame_age<=2.5
        if connected and online and fresh:self._status("LIVE",GREEN)
        elif not connected:self._status("ML OFFLINE",RED);self.image.set_message("ML servis ishlamayapti",clear=frame_age>2.5)
        elif camera.get("startup_waiting"):self._status("KUTILMOQDA",ORANGE);self.image.set_message("Kamera ulanmoqda...",clear=frame_age>2.5)
        else:
            self._status("OFFLINE",RED);reason=str(camera.get("last_error") or reader.get("error") or "Kamera oqimi yo'q")
            if "401" in reason or "Unauthorized" in reason:reason="RTSP login yoki parol noto'g'ri"
            if len(reason)>68:reason=reason[:65]+"..."
            self.image.set_message(reason,clear=frame_age>2.5)
        active=(publisher.get("tracker") or {}).get("active_tracks");count=int(active) if isinstance(active,(int,float)) else int(detection_count)
        self.people.setText(f"{count} odam");self.identities.setText(f"ID: {identity_count}")
        source_fps=float(camera.get("source_fps") or 0.0);shown_fps=display_fps if display_fps>0 else source_fps;self.fps.setText(f"{shown_fps:.0f} FPS" if shown_fps>0 else "— FPS")
        age=camera.get("last_frame_age_ms");self.age.setText(f"{float(age):.0f} ms" if isinstance(age,(int,float)) and online else "— ms")


class MetricChip(QFrame):
    def __init__(self,label):
        super().__init__();self.setObjectName("metricChip");row=QHBoxLayout(self);row.setContentsMargins(10,6,10,6);row.setSpacing(6);self.dot=QLabel("●");self.value=QLabel("—");self.value.setFont(font(13,QFont.Weight.DemiBold));name=QLabel(label);name.setFont(font(11));name.setStyleSheet(f"color:{MUTED};");row.addWidget(self.dot);row.addWidget(self.value);row.addWidget(name)
    def set_value(self,value,color):self.value.setText(value);self.dot.setStyleSheet(f"color:{color};")


class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle("Apsidal · Real-time monitoring");self.resize(1600,920);self.setMinimumSize(960,600)
        root=QWidget();self.setCentralWidget(root);layout=QVBoxLayout(root);layout.setContentsMargins(11,9,11,11);layout.setSpacing(8)
        top=QHBoxLayout();brand=QVBoxLayout();title=QLabel("APSIDAL · LIVE");title.setFont(font(19,QFont.Weight.DemiBold));subtitle=QLabel("Real-time camera monitoring");subtitle.setFont(font(10));subtitle.setStyleSheet(f"color:{MUTED};");brand.addWidget(title);brand.addWidget(subtitle);top.addLayout(brand);top.addStretch()
        self.online_chip=MetricChip("kamera");self.people_chip=MetricChip("odam");self.detector_chip=MetricChip("detector");self.reid_chip=MetricChip("ReID");self.gpu_chip=MetricChip("GPU")
        for chip in (self.online_chip,self.people_chip,self.detector_chip,self.reid_chip,self.gpu_chip):top.addWidget(chip)
        self.clock=QLabel();self.clock.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter);self.clock.setFont(font(13,QFont.Weight.Medium));top.addWidget(self.clock)
        full=QPushButton("⛶");full.setObjectName("iconButton");full.setFixedSize(36,36);full.setFont(font(20));full.clicked.connect(self.toggle_fullscreen);top.addWidget(full);layout.addLayout(top)
        self.service_banner=QLabel("ML servisga ulanmoqda...");self.service_banner.setObjectName("serviceBanner");self.service_banner.setFixedHeight(26);self.service_banner.setAlignment(Qt.AlignmentFlag.AlignCenter);self.service_banner.setFont(font(11,QFont.Weight.Medium));layout.addWidget(self.service_banner)
        self.grid_host=QWidget();self.grid=QGridLayout(self.grid_host);self.grid.setContentsMargins(0,0,0,0);self.grid.setSpacing(8);layout.addWidget(self.grid_host,1)
        self.tiles={cid:CameraTile(cid,index+1) for index,cid in enumerate(CAMERAS)};self._grid_columns=0;self._arrange_grid(3)
        self.readers={};self.seen={cid:-1 for cid in CAMERAS};self.frame_counts={cid:0 for cid in CAMERAS};self.display_fps={cid:0.0 for cid in CAMERAS}
        for cid in CAMERAS:reader=LatestFrameReader(cid);reader.start();self.readers[cid]=reader
        self.live_state=LiveStateReader();self.live_state.start();self.last_info_tick=time.monotonic()
        self.render_timer=QTimer(self);self.render_timer.setTimerType(Qt.TimerType.PreciseTimer);self.render_timer.timeout.connect(self.render);self.render_timer.start(25)
        self.info_timer=QTimer(self);self.info_timer.timeout.connect(self.update_live_state);self.info_timer.start(500)
        self.clock_timer=QTimer(self);self.clock_timer.timeout.connect(self.update_clock);self.clock_timer.start(1000);self.update_clock();self.apply_theme()
    def _arrange_grid(self,columns):
        if columns==self._grid_columns:return
        while self.grid.count():self.grid.takeAt(0)
        rows=max(1,math.ceil(len(CAMERAS)/columns))
        for index,cid in enumerate(CAMERAS):self.grid.addWidget(self.tiles[cid],index//columns,index%columns)
        for row in range(rows):self.grid.setRowStretch(row,1)
        for column in range(columns):self.grid.setColumnStretch(column,1)
        self._grid_columns=columns
    def resizeEvent(self,event):super().resizeEvent(event);self._arrange_grid(3 if self.width()>=1180 else 2)
    def apply_theme(self):
        self.setStyleSheet(f"""QMainWindow,QWidget{{background:{BG};color:{TEXT};}}#cameraTile{{background:{PANEL};border:1px solid {BORDER};border-radius:8px;}}#metricChip{{background:{CARD};border:1px solid {BORDER};border-radius:7px;}}#serviceBanner{{background:#301c08;color:{ORANGE};border:1px solid #6f4314;border-radius:6px;}}#iconButton{{background:{CARD};border:1px solid {BORDER};border-radius:7px;color:{TEXT};}}#iconButton:hover{{background:#112b4d;}}""")
    def toggle_fullscreen(self):self.showNormal() if self.isFullScreen() else self.showFullScreen()
    def update_clock(self):now=datetime.now();self.clock.setText(now.strftime("%H:%M:%S\n%d.%m.%Y"))
    def render(self):
        for cid,reader in self.readers.items():
            state=reader.latest();image=state.get("image");version=int(state.get("version",-1))
            if image is not None and version>self.seen[cid]:self.seen[cid]=version;self.tiles[cid].image.set_frame(image)
    def update_live_state(self):
        now=time.monotonic();interval=max(.1,now-self.last_info_tick);self.last_info_tick=now;snapshot=self.live_state.snapshot();connected=bool(snapshot.get("connected")) and now-float(snapshot.get("updated_at") or 0.0)<2.5
        health=snapshot.get("health") or {};cameras=health.get("cameras") or {};publishers=health.get("publishers") or {};detections=(snapshot.get("detections") or {}).get("cameras") or {};reid=(((snapshot.get("reid") or {}).get("state") or {}).get("cameras") or {})
        online=0;total_people=0
        for cid,reader in self.readers.items():
            state=reader.latest();current=int(state.get("frames") or 0);previous=self.frame_counts[cid];self.frame_counts[cid]=current;self.display_fps[cid]=(current-previous)/interval
            camera=cameras.get(cid) or {};publisher=publishers.get(cid) or {};det_count=len((detections.get(cid) or {}).get("boxes") or []);identity_count=len(reid.get(cid) or []);active=(publisher.get("tracker") or {}).get("active_tracks");total_people+=int(active) if isinstance(active,(int,float)) else det_count
            if camera.get("online"):online+=1
            self.tiles[cid].update_live(connected,camera,publisher,det_count,identity_count,state,self.display_fps[cid])
        self.online_chip.set_value(f"{online}/{len(CAMERAS)}",GREEN if online==len(CAMERAS) else RED);self.people_chip.set_value(str(total_people),CYAN)
        detector=health.get("detector") or {};detector_ready=connected and bool(detector) and not detector.get("last_error");self.detector_chip.set_value("ON" if detector_ready else "OFF",GREEN if detector_ready else RED)
        reid_metrics=health.get("reid") or {};reid_ready=connected and bool(reid_metrics.get("ready"));self.reid_chip.set_value("ON" if reid_ready else "OFF",GREEN if reid_ready else ORANGE)
        gpu=(health.get("service_resources") or {}).get("gpu_utilization_percent");self.gpu_chip.set_value(f"{int(gpu)}%" if isinstance(gpu,(int,float)) else "—",GREEN if isinstance(gpu,(int,float)) else MUTED)
        if connected:
            color=GREEN if online==len(CAMERAS) else ORANGE;background="#08291e" if online==len(CAMERAS) else "#301c08";border="#176842" if online==len(CAMERAS) else "#6f4314"
            self.service_banner.setText(f"REAL-TIME · {online}/{len(CAMERAS)} kamera online · backlog yo'q · {datetime.now().strftime('%H:%M:%S')}");self.service_banner.setStyleSheet(f"background:{background};color:{color};border:1px solid {border};border-radius:6px;")
        else:
            detail=str(snapshot.get("error") or "ML service bilan aloqa yo'q");self.service_banner.setText(f"ML SERVICE OFFLINE · {detail}");self.service_banner.setStyleSheet(f"background:#321018;color:{RED};border:1px solid #7a2432;border-radius:6px;")
    def closeEvent(self,event):
        self.render_timer.stop();self.info_timer.stop();self.clock_timer.stop();self.live_state.stop()
        for reader in self.readers.values():reader.stop()
        event.accept()


def run():
    app=QApplication.instance() or QApplication([]);app.setStyle("Fusion");window=DashboardWindow();window.showMaximized();return app.exec()
