"""Operator entry point for hidden detection-recovery ROI calibration.

Run the API and ML services normally, then execute: ``python roi_setup.py``.
The tool consumes the existing bounded MJPEG previews and persists normalized
polygons through the authoritative camera API.  It never opens RTSP itself.
"""
from __future__ import annotations

import signal
import sys

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout, QHBoxLayout,
    QDialog, QInputDialog, QLabel, QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from services.frontend.api_client import ApiClient
from services.frontend.async_api import AsyncApi
from services.frontend.video_transport import MJPEGClient
from shared.schemas.roi import RecoveryROI
from shared.settings import ServiceSettings


class CameraCanvas(QWidget):
    drawingChanged=Signal(int,bool)
    def __init__(self, camera, selected, parent=None):
        super().__init__(parent); self.camera=dict(camera); self.selected=selected
        self.frame=QImage(); self.online=False; self.rois=[dict(x) for x in camera.get("recovery_rois",())]
        self.current=-1; self.points=[]; self.drawing=False;self.hover_point=None;self.drawing_backup=None;self.mouse_events={"press":0,"move":0,"release":0};self.setMinimumSize(420,236);self.setMouseTracking(True);self.setFocusPolicy(Qt.StrongFocus);self.setCursor(Qt.ArrowCursor)

    @property
    def camera_id(self): return str(self.camera["id"])
    def image_rect(self):
        if self.frame.isNull(): return QRectF(self.rect())
        scale=min(self.width()/self.frame.width(),self.height()/self.frame.height());w=self.frame.width()*scale;h=self.frame.height()*scale
        return QRectF((self.width()-w)/2,(self.height()-h)/2,w,h)
    def normalized_at(self,pos):
        rect=self.image_rect()
        if rect.isEmpty() or not rect.contains(pos):return None
        return QPointF((pos.x()-rect.left())/rect.width(),(pos.y()-rect.top())/rect.height())
    def screen_at(self,point):
        rect=self.image_rect();return QPointF(rect.left()+point.x()*rect.width(),rect.top()+point.y()*rect.height())
    def begin_drawing(self):self.drawing=True;self.hover_point=None;self.setCursor(Qt.CrossCursor);self.setFocus();self.update()
    def select_roi(self,index):
        self.current=index;self.points=[QPointF(*p) for p in self.rois[index]["polygon"]] if 0<=index<len(self.rois) else [];self.drawing=False;self.hover_point=None;self.setCursor(Qt.ArrowCursor);self.drawingChanged.emit(len(self.points),False);self.update()
    def commit(self):
        if 0<=self.current<len(self.rois):self.rois[self.current]["polygon"]=[[round(p.x(),6),round(p.y(),6)] for p in self.points]
    def mousePressEvent(self,event):
        self.selected(self)
        if not self.drawing:return
        self.mouse_events["press"]+=1;point=self.normalized_at(event.position())
        if point is not None:self.points.append(point);self.drawingChanged.emit(len(self.points),True);self.update();event.accept()
    def mouseMoveEvent(self,event):
        self.mouse_events["move"]+=1;self.hover_point=self.normalized_at(event.position()) if self.drawing else None;self.update()
    def mouseReleaseEvent(self,event):self.mouse_events["release"]+=1;event.accept()
    def leaveEvent(self,event):self.hover_point=None;self.update();super().leaveEvent(event)
    def paintEvent(self,_event):
        painter=QPainter(self);painter.fillRect(self.rect(),QColor("#111820"));rect=self.image_rect()
        if not self.frame.isNull():painter.drawPixmap(rect.toRect(),QPixmap.fromImage(self.frame))
        else:
            painter.setPen(QColor("#94a1b3"));painter.drawText(self.rect(),Qt.AlignCenter,"NO SIGNAL\nStart the ML service for preview")
        painter.setRenderHint(QPainter.Antialiasing)
        for index,roi in enumerate(self.rois):
            points=self.points if index==self.current else [QPointF(*p) for p in roi.get("polygon",())]
            screen=[QPointF(rect.left()+p.x()*rect.width(),rect.top()+p.y()*rect.height()) for p in points];selected=index==self.current;color=QColor("#00e5ff") if selected else QColor("#ffb300");painter.setPen(QPen(color,4 if selected else 2,Qt.SolidLine,Qt.RoundCap,Qt.RoundJoin))
            if len(screen)>=3:
                path=QPainterPath(screen[0]);[path.lineTo(point) for point in screen[1:]];path.closeSubpath();painter.fillPath(path,QColor(color.red(),color.green(),color.blue(),58 if selected else 30))
            for a,b in zip(screen,screen[1:]):painter.drawLine(a,b)
            if len(screen)>=3 and (index!=self.current or not self.drawing):painter.drawLine(screen[-1],screen[0])
            painter.setBrush(color)
            for point in screen:painter.drawEllipse(point,6 if selected else 4,6 if selected else 4)
        if self.drawing and self.points and self.hover_point is not None:
            last=self.points[-1];a=QPointF(rect.left()+last.x()*rect.width(),rect.top()+last.y()*rect.height());b=QPointF(rect.left()+self.hover_point.x()*rect.width(),rect.top()+self.hover_point.y()*rect.height());painter.setPen(QPen(QColor("#ffffff"),3,Qt.DashLine,Qt.RoundCap));painter.drawLine(a,b)
        painter.setPen(QColor("white"));painter.fillRect(0,0,self.width(),28,QColor(0,0,0,170));painter.drawText(10,19,f"{self.camera_id}  {'ONLINE' if self.online else 'UNAVAILABLE'}")


class FreezeFrameROIDialog(QDialog):
    """Large static calibration surface; live MJPEG can never repaint it."""
    def __init__(self,camera,frame,parent=None):
        super().__init__(parent);self.setWindowTitle(f"{camera["id"]} — Draw ROI on frozen frame");self.resize(1100,760);self.setModal(True)
        draft={"id":"__draft__","enabled":True,"polygon":[]};frozen={**camera,"recovery_rois":[draft]};self.canvas=CameraCanvas(frozen,lambda _canvas:None,self);self.canvas.frame=frame.copy();self.canvas.online=True;self.canvas.current=0;self.canvas.points=[];self.canvas.begin_drawing();self.canvas.setMinimumSize(900,506)
        root=QVBoxLayout(self);title=QLabel("Frozen frame — click polygon vertices directly on the image");title.setStyleSheet("font-size:18px;font-weight:600");root.addWidget(title);root.addWidget(self.canvas,1)
        controls=QHBoxLayout();self.points_label=QLabel("Points: 0");self.undo=QPushButton("Undo");self.clear=QPushButton("Clear");self.cancel=QPushButton("Cancel");self.done_button=QPushButton("Done");self.done_button.setEnabled(False)
        controls.addWidget(self.points_label);controls.addStretch(1)
        for widget in (self.undo,self.clear,self.cancel,self.done_button):controls.addWidget(widget)
        root.addLayout(controls);self.canvas.drawingChanged.connect(self._changed);self.undo.clicked.connect(self._undo);self.clear.clicked.connect(self._clear);self.cancel.clicked.connect(self.reject);self.done_button.clicked.connect(self.accept)
    def _changed(self,count,_drawing):self.points_label.setText(f"Points: {count}");self.done_button.setEnabled(count>=3)
    def _undo(self):
        if self.canvas.points:self.canvas.points.pop();self.canvas.hover_point=None;self.canvas.drawingChanged.emit(len(self.canvas.points),True);self.canvas.update()
    def _clear(self):self.canvas.points=[];self.canvas.hover_point=None;self.canvas.drawingChanged.emit(0,True);self.canvas.update()
    def polygon(self):return [[round(point.x(),6),round(point.y(),6)] for point in self.canvas.points]
    def accept(self):
        if len(self.canvas.points)<3:return
        self.canvas.drawing=False;self.canvas.setCursor(Qt.ArrowCursor);super().accept()



class ROISetupWindow(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle("Detection Recovery Zone Setup");self.resize(1180,820)
        self.api=ApiClient();self.async_api=AsyncApi(self.api,self);self.clients={};self.cards={};self.active=None
        root=QWidget();self.setCentralWidget(root);outer=QVBoxLayout(root);title=QLabel("Detection Recovery Zones — select a camera, then Add ROI")
        title.setStyleSheet("font-size:20px;font-weight:600");outer.addWidget(title);self.grid=QGridLayout();outer.addLayout(self.grid,1)
        controls=QHBoxLayout();self.selector=QComboBox();self.selector.setMinimumWidth(180);self.add=QPushButton("Add ROI");self.undo=QPushButton("Undo Point");self.clear=QPushButton("Clear Current");self.delete=QPushButton("Delete ROI");self.enabled=QCheckBox("Enabled");self.done=QPushButton("Done")
        controls.addWidget(self.selector)
        for widget in (self.add,self.undo,self.clear,self.delete,self.enabled,self.done):controls.addWidget(widget)
        outer.addLayout(controls);self.status=QLabel("Loading cameras…");outer.addWidget(self.status)
        self.selector.currentIndexChanged.connect(self.select_roi);self.add.clicked.connect(self.add_roi);self.undo.clicked.connect(self.undo_point);self.clear.clicked.connect(self.clear_current);self.delete.clicked.connect(self.delete_roi);self.enabled.toggled.connect(self.toggle_enabled);self.done.clicked.connect(self.save_done);self.done.setEnabled(False)
        self.async_api.submit(self.api.get_cameras,self.load_cameras,self.load_failed,owner=self)

    def load_failed(self,error):self.status.setText(f"API unavailable: {error}. Start the API service and retry.")
    def load_cameras(self,rows):
        base=ServiceSettings.from_env().ml_url.rstrip("/")
        for index,camera in enumerate(sorted(rows,key=lambda x:str(x["id"]))[:6]):
            card=CameraCanvas(camera,self.select_camera,self);self.cards[card.camera_id]=card;self.grid.addWidget(card,index//2,index%2)
            card.drawingChanged.connect(self.drawing_state);client=MJPEGClient(card.camera_id,f"{base}/video/{card.camera_id}",self);client.frame.connect(self.frame);client.online.connect(self.online);client.start();self.clients[card.camera_id]=client
        self.status.setText(f"Loaded {len(self.cards)} cameras. Preview uses the existing ML MJPEG service.")
    def select_camera(self,card):
        if self.active and self.active is not card:self.active.commit();self.active.drawing=False;self.active.update()
        self.active=card;card.setFocus();self.selector.blockSignals(True);self.selector.clear();self.selector.addItems([x["id"] for x in card.rois]);self.selector.setCurrentIndex(0 if card.rois else -1);self.selector.blockSignals(False);card.select_roi(self.selector.currentIndex());self._sync_enabled();card.update()
    def select_roi(self,index):
        if not self.active:return
        self.active.commit();self.active.select_roi(index);self._sync_enabled()
    def _sync_enabled(self):
        self.enabled.blockSignals(True);self.enabled.setChecked(bool(self.active and 0<=self.active.current<len(self.active.rois) and self.active.rois[self.active.current].get("enabled",True)));self.enabled.blockSignals(False)
    def frame(self,camera_id,_frame_id,_timestamp,image):
        card=self.cards.get(camera_id)
        if card:card.frame=image;card.online=True;card.update()
    def online(self,camera_id,value):
        card=self.cards.get(camera_id)
        if card:card.online=bool(value);card.update()
    def add_roi(self):
        if not self.active:return self.status.setText("Select a camera first")
        if self.active.frame.isNull():return self.status.setText("No live frame available to freeze")
        name,ok=QInputDialog.getText(self,"Add ROI","Recovery zone name");name=name.strip()
        if not ok or not name:return
        if any(x["id"]==name for x in self.active.rois):return self.status.setText(f"Duplicate ROI ID: {name}")
        frozen=self.active.frame.copy();dialog=FreezeFrameROIDialog(self.active.camera,frozen,self)
        if dialog.exec()!=QDialog.Accepted:return self.status.setText("ROI drawing cancelled")
        self.active.commit();self.active.rois.append({"id":name,"enabled":True,"polygon":dialog.polygon()});self.selector.addItem(name);self.selector.setCurrentIndex(len(self.active.rois)-1);self.active.select_roi(len(self.active.rois)-1);self.enabled.setChecked(True);self._persist_active()
    def drawing_state(self,count,drawing):
        if self.sender() is not None and self.sender() is not self.active:return
        self.done.setEnabled(bool(drawing and count>=3));self.status.setText(f"Drawing ROI — click points, Done to finish    Points: {count}" if drawing else self.status.text())
    def undo_point(self):
        if self.active and self.active.points:self.active.points.pop();self.active.drawing=True;self.active.hover_point=None;self.active.drawingChanged.emit(len(self.active.points),True);self.active.update()
    def clear_current(self):
        if self.active:self.active.points=[];self.active.drawing=True;self.active.hover_point=None;self.active.drawingChanged.emit(0,True);self.active.update()
    def delete_roi(self):
        if self.active and 0<=self.active.current<len(self.active.rois):self.active.rois.pop(self.active.current);self.selector.removeItem(self.active.current);self.active.select_roi(self.selector.currentIndex());self.status.setText("ROI deleted locally; press Done to persist")
    def toggle_enabled(self,value):
        if self.active and 0<=self.active.current<len(self.active.rois):self.active.rois[self.active.current]["enabled"]=bool(value)
    def _persist_active(self):
        if not self.active:return
        self.active.commit()
        try:payload=[RecoveryROI.model_validate(item).model_dump(mode="json") for item in self.active.rois]
        except Exception as exc:return self.status.setText(f"Cannot save: {exc}")
        camera=self.active;self.done.setEnabled(False);self.status.setText(f"Saving {camera.camera_id}…")
        def saved(result):
            camera.rois=[dict(x) for x in result.get("recovery_rois",payload)];camera.select_roi(max(0,camera.current));self.status.setText(f"Saved: {camera.camera_id} / {", ".join(x["id"] for x in payload) or "no ROIs"}")
        self.async_api.submit(lambda:self.api.update_camera(camera.camera_id,{"recovery_rois":payload}),saved,lambda error:self.status.setText(f"Save failed: {error}"),owner=self)

    def save_done(self):
        if not self.active:return self.status.setText("Select a camera first")
        if not self.active.drawing or len(self.active.points)<3:return self.status.setText("ROI needs at least three valid points")
        self.active.drawing=False;self.active.hover_point=None;self.active.setCursor(Qt.ArrowCursor);self.active.commit();self.active.update()
        try:payload=[RecoveryROI.model_validate(item).model_dump(mode="json") for item in self.active.rois]
        except Exception as exc:return self.status.setText(f"Cannot save: {exc}")
        camera=self.active;self.done.setEnabled(False);self.status.setText(f"Saving {camera.camera_id}…")
        def saved(result):
            camera.rois=[dict(x) for x in result.get("recovery_rois",payload)];camera.select_roi(max(0,camera.current));self.done.setEnabled(False);self.status.setText(f"Saved: {camera.camera_id} / {', '.join(x['id'] for x in payload) or 'no ROIs'}")
        def failed(error):self.done.setEnabled(True);self.status.setText(f"Save failed: {error}")
        self.async_api.submit(lambda:self.api.update_camera(camera.camera_id,{"recovery_rois":payload}),saved,failed,owner=self)
    def keyPressEvent(self,event):
        if event.key()==Qt.Key_Escape and self.active and self.active.drawing:
            index=self.active.current
            if self.active.drawing_backup is None and 0<=index<len(self.active.rois):self.active.rois.pop(index);self.selector.removeItem(index);self.active.select_roi(self.selector.currentIndex())
            else:self.active.points=[QPointF(*p) for p in (self.active.drawing_backup or ())];self.active.drawing=False;self.active.hover_point=None;self.active.update()
            self.done.setEnabled(False);self.status.setText("Drawing cancelled");return
        super().keyPressEvent(event)
    def closeEvent(self,event):
        for client in self.clients.values():client.stop()
        self.clients.clear();self.async_api.shutdown();super().closeEvent(event)


def main():
    app=QApplication(sys.argv);app.setStyle("Fusion");window=ROISetupWindow();window.show();signal.signal(signal.SIGINT,lambda *_:app.quit());signal.signal(signal.SIGTERM,lambda *_:app.quit());timer=QTimer();timer.start(250);timer.timeout.connect(lambda:None);return app.exec()

if __name__=="__main__":raise SystemExit(main())
