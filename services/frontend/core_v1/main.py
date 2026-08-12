from __future__ import annotations
import sys,threading,time,urllib.request
from PySide6.QtCore import Qt,QTimer
from PySide6.QtGui import QImage,QPixmap
from PySide6.QtWidgets import QApplication,QGridLayout,QLabel,QMainWindow,QWidget

CAMERAS=[f'CAM-{i:02d}' for i in range(1,7)]
ML_BASE='http://127.0.0.1:8001'

class LatestFrameReader:
    """Long-poll one JPEG at a time; an HTTP receive backlog cannot form."""
    def __init__(self,camera_id):
        self.camera_id=camera_id;self._stop=threading.Event();self._thread=None;self._lock=threading.Lock();self._image=None;self._version=-1;self.frames=0;self.errors=0
    def start(self):
        self._thread=threading.Thread(target=self._run,name=f'frontend-{self.camera_id}',daemon=True);self._thread.start()
    def stop(self):self._stop.set()
    def latest(self):
        with self._lock:return self._image,self._version
    def _run(self):
        version=-1
        while not self._stop.is_set():
            try:
                url=f'{ML_BASE}/frame/{self.camera_id}?after={version}&wait_ms=250'
                request=urllib.request.Request(url,headers={'Cache-Control':'no-cache','Connection':'keep-alive'})
                with urllib.request.urlopen(request,timeout=2) as response:
                    jpg=response.read()
                    next_version=int(response.headers.get('X-Frame-Version',version+1))
                image=QImage.fromData(jpg,'JPG')
                if image.isNull():continue
                version=next_version
                with self._lock:self._image=image.copy();self._version=version
                self.frames+=1
            except Exception:
                self.errors+=1;self._stop.wait(.05)

class CameraTile(QLabel):
    def __init__(self,camera_id):
        super().__init__(camera_id);self.camera_id=camera_id;self.setAlignment(Qt.AlignCenter);self.setMinimumSize(320,180);self.setStyleSheet('background:#05080d;color:#aab4c0;border:1px solid #253041;')
    def paint_image(self,image):
        pix=QPixmap.fromImage(image);self.setPixmap(pix.scaled(self.size(),Qt.KeepAspectRatio,Qt.FastTransformation))

class Window(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('AI Surveillance — Core v1 Camera Test');self.resize(1500,900)
        root=QWidget();grid=QGridLayout(root);grid.setSpacing(4);grid.setContentsMargins(4,4,4,4);self.setCentralWidget(root)
        self.tiles={};self.readers={};self.seen={}
        for index,camera_id in enumerate(CAMERAS):
            tile=CameraTile(camera_id);grid.addWidget(tile,index//2,index%2);self.tiles[camera_id]=tile
            reader=LatestFrameReader(camera_id);reader.start();self.readers[camera_id]=reader;self.seen[camera_id]=-1
        # 60 Hz UI polling is cheap because readers already hold only one QImage.
        # It minimizes extra presentation delay while source video remains 12 FPS.
        self.timer=QTimer(self);self.timer.timeout.connect(self.render);self.timer.start(16)
    def render(self):
        for cid,reader in self.readers.items():
            image,version=reader.latest()
            if image is not None and version>self.seen[cid]:self.seen[cid]=version;self.tiles[cid].paint_image(image)
    def closeEvent(self,event):
        for reader in self.readers.values():reader.stop()
        event.accept()

if __name__=='__main__':
    app=QApplication(sys.argv);window=Window();window.show();sys.exit(app.exec())
