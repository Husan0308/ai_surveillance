"""Settings-only editor for normalized hidden recovery polygons."""
from __future__ import annotations
from PySide6.QtCore import QPointF,QRectF,Qt,QTimer,Signal
from PySide6.QtGui import QColor,QPainter,QPainterPath,QPen,QPixmap
from PySide6.QtWidgets import (QCheckBox,QComboBox,QDialog,QHBoxLayout,QInputDialog,
 QLabel,QPushButton,QVBoxLayout,QWidget)

class ROIFrameCanvas(QWidget):
    pointsChanged=Signal(int)
    def __init__(self,camera,parent=None):
        super().__init__(parent);self.camera=camera;self.points=[];self.closed=False;self.hover=None;self.setMinimumSize(640,360);self.setMouseTracking(True);self.setFocusPolicy(Qt.StrongFocus);self.setCursor(Qt.CrossCursor)
    def image_rect(self):
        frame=self.camera.frame
        if frame is None:return QRectF(self.rect())
        scale=min(self.width()/frame.width(),self.height()/frame.height());w,h=frame.width()*scale,frame.height()*scale
        return QRectF((self.width()-w)/2,(self.height()-h)/2,w,h)
    def normalized_at(self,position):
        rect=self.image_rect()
        if rect.isEmpty() or not rect.contains(position):return None
        return QPointF((position.x()-rect.left())/rect.width(),(position.y()-rect.top())/rect.height())
    def screen_at(self,point):
        rect=self.image_rect();return QPointF(rect.left()+point.x()*rect.width(),rect.top()+point.y()*rect.height())
    def set_polygon(self,points):self.points=[QPointF(float(x),float(y)) for x,y in points];self.closed=len(self.points)>=3;self.hover=None;self.pointsChanged.emit(len(self.points));self.update()
    def normalized_polygon(self):return [[round(point.x(),6),round(point.y(),6)] for point in self.points]
    def mousePressEvent(self,event):
        if self.closed:return
        point=self.normalized_at(event.position())
        if point is not None:self.points.append(point);self.pointsChanged.emit(len(self.points));self.update();event.accept()
    def mouseMoveEvent(self,event):self.hover=self.normalized_at(event.position()) if not self.closed else None;self.update()
    def mouseReleaseEvent(self,event):event.accept()
    def paintEvent(self,_event):
        painter=QPainter(self);painter.fillRect(self.rect(),QColor("#10151b"));rect=self.image_rect();frame=self.camera.frame
        if frame is not None:painter.drawPixmap(rect.toRect(),QPixmap.fromImage(frame))
        painter.setRenderHint(QPainter.Antialiasing);painter.setPen(QPen(QColor("#2f7df6"),3));screen=[self.screen_at(p) for p in self.points]
        if len(screen)>=3:
            path=QPainterPath(screen[0]);[path.lineTo(point) for point in screen[1:]]
            if self.closed:path.closeSubpath()
            painter.fillPath(path,QColor(47,125,246,45))
        for first,second in zip(screen,screen[1:]):painter.drawLine(first,second)
        if self.closed and len(screen)>=3:painter.drawLine(screen[-1],screen[0])
        if not self.closed and screen and self.hover is not None:painter.setPen(QPen(QColor("white"),2,Qt.DashLine));painter.drawLine(screen[-1],self.screen_at(self.hover))
        painter.setPen(QPen(QColor("#2f7df6"),2));painter.setBrush(QColor("#5b9bff"))
        for point in screen:painter.drawEllipse(point,6,6)

class ROIEditorDialog(QDialog):
    """ROI boundaries exist only in this dialog; production surfaces never receive them."""
    def __init__(self,hub,camera,parent=None):
        super().__init__(parent);self.hub,self.camera=hub,camera;self.rois=[dict(item) for item in camera.recovery_rois];self.setWindowTitle(f"{camera.id} — Detection Recovery ROI");self.resize(820,560)
        layout=QVBoxLayout(self);top=QHBoxLayout();self.selector=QComboBox();self.enabled=QCheckBox("Enabled");add=QPushButton("Add ROI");top.addWidget(self.selector,1);top.addWidget(self.enabled);top.addWidget(add);layout.addLayout(top);self.canvas=ROIFrameCanvas(camera);layout.addWidget(self.canvas,1);self.point_status=QLabel("Points: 0 — click inside the video");layout.addWidget(self.point_status);self.canvas.pointsChanged.connect(lambda count:self.point_status.setText(f"Points: {count} — click inside the video"))
        buttons=QHBoxLayout();close=QPushButton("Close polygon");reset=QPushButton("Reset");delete=QPushButton("Delete");save=QPushButton("Save");buttons.addWidget(close);buttons.addWidget(reset);buttons.addWidget(delete);buttons.addStretch(1);buttons.addWidget(save);layout.addLayout(buttons)
        self.selector.currentIndexChanged.connect(self._load);add.clicked.connect(self._add);close.clicked.connect(self._close);reset.clicked.connect(lambda:self.canvas.set_polygon([]));delete.clicked.connect(self._delete);save.clicked.connect(self._save)
        self.timer=QTimer(self);self.timer.timeout.connect(self.canvas.update);self.timer.start(250);self._rebuild()
    def _rebuild(self,selected=0):
        self.selector.blockSignals(True);self.selector.clear();self.selector.addItems([item["id"] for item in self.rois]);self.selector.setCurrentIndex(min(selected,len(self.rois)-1));self.selector.blockSignals(False);self._load()
    def _load(self):
        index=self.selector.currentIndex();item=self.rois[index] if 0<=index<len(self.rois) else None;self.enabled.setChecked(bool(item and item.get("enabled",True)));self.canvas.set_polygon(item.get("polygon",[]) if item else [])
    def _commit_current(self):
        index=self.selector.currentIndex()
        if index>=0:self.rois[index]={"id":self.rois[index]["id"],"enabled":self.enabled.isChecked(),"polygon":self.canvas.normalized_polygon()}
    def _add(self):
        name,ok=QInputDialog.getText(self,"Add recovery ROI","Unique ROI ID")
        name=name.strip()
        if not ok or not name:return
        if any(item["id"]==name for item in self.rois):self.hub.toast("ROI ID already exists");return
        self._commit_current();self.rois.append({"id":name,"enabled":True,"polygon":[]});self._rebuild(len(self.rois)-1)
    def _close(self):
        if len(self.canvas.points)<3:self.hub.toast("ROI needs at least three points");return
        self.canvas.closed=True;self.canvas.hover=None;self.canvas.setCursor(Qt.ArrowCursor);self.canvas.update()
    def _delete(self):
        index=self.selector.currentIndex()
        if index>=0:self.rois.pop(index);self._rebuild(max(0,index-1))
    def _save(self):
        self._commit_current()
        if any(len(item.get("polygon",[]))<3 for item in self.rois):self.hub.toast("Every ROI needs at least three points");return
        payload=[{"id":item["id"],"enabled":bool(item.get("enabled",True)),"polygon":item["polygon"]} for item in self.rois]
        def applied(result):self.camera.recovery_rois=[dict(item) for item in result.get("recovery_rois",payload)];self.hub.sys.refresh_cameras();self.accept()
        self.hub.sys.async_api.submit(lambda:self.hub.sys.api.update_camera(self.camera.id,{"recovery_rois":payload}),applied,lambda error:self.hub.toast(f"ROI API: {error}"),owner=self)
