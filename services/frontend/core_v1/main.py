from __future__ import annotations
import http.client,sys,threading,time
from PySide6.QtCore import Qt,QTimer
from PySide6.QtGui import QImage,QPixmap
from PySide6.QtWidgets import QApplication,QGridLayout,QLabel,QMainWindow,QWidget

CAMERAS=[f'CAM-{i:02d}' for i in range(1,7)]
ML_HOST='127.0.0.1';ML_PORT=8001

class LatestFrameReader:
    """One persistent HTTP connection per camera; only the newest JPEG is kept."""
    def __init__(self,camera_id):
        self.camera_id=camera_id;self._stop=threading.Event();self._thread=None;self._lock=threading.Lock();self._image=None;self._version=-1
        self.frames=0;self.errors=0;self.last_frame_at=0.0
    def start(self):
        self._thread=threading.Thread(target=self._run,name=f'frontend-{self.camera_id}',daemon=True);self._thread.start()
    def stop(self):self._stop.set()
    def latest(self):
        with self._lock:return self._image,self._version
    def _run(self):
        version=-1;connection=None
        while not self._stop.is_set():
            try:
                if connection is None:
                    connection=http.client.HTTPConnection(ML_HOST,ML_PORT,timeout=2.0)
                path=f'/frame/{self.camera_id}?after={version}&wait_ms=180'
                connection.request('GET',path,headers={'Cache-Control':'no-cache','Connection':'keep-alive'})
                response=connection.getresponse();jpg=response.read()
                if response.status!=200:
                    raise RuntimeError(f'HTTP {response.status}')
                next_version=int(response.getheader('X-Frame-Version') or (version+1))
                image=QImage.fromData(jpg,'JPG')
                if image.isNull():
                    continue
                version=next_version
                with self._lock:
                    self._image=image;self._version=version
                self.frames+=1;self.last_frame_at=time.monotonic()
            except Exception:
                self.errors+=1
                if connection is not None:
                    try:connection.close()
                    except Exception:pass
                connection=None
                self._stop.wait(.03)
        if connection is not None:
            try:connection.close()
            except Exception:pass

class CameraTile(QLabel):
    def __init__(self,camera_id):
        super().__init__(camera_id);self.camera_id=camera_id;self.setAlignment(Qt.AlignCenter);self.setMinimumSize(320,180);self.setStyleSheet('background:#05080d;color:#aab4c0;border:1px solid #253041;')
    def paint_image(self,image):
        # Server already downsizes to the presentation size. Keep the UI path
        # intentionally light: one QPixmap conversion/scaling for a genuinely
        # new frame only.
        pix=QPixmap.fromImage(image);self.setPixmap(pix.scaled(self.size(),Qt.KeepAspectRatio,Qt.FastTransformation))

class Window(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('AI Surveillance — Core v1 Camera Test');self.resize(1500,900)
        root=QWidget();grid=QGridLayout(root);grid.setSpacing(4);grid.setContentsMargins(4,4,4,4);self.setCentralWidget(root)
        self.tiles={};self.readers={};self.seen={}
        for index,camera_id in enumerate(CAMERAS):
            tile=CameraTile(camera_id);grid.addWidget(tile,index//2,index%2);self.tiles[camera_id]=tile
            reader=LatestFrameReader(camera_id);reader.start();self.readers[camera_id]=reader;self.seen[camera_id]=-1
        # The source is up to 18 FPS; 20 ms UI polling is frequent enough to
        # present the next newest frame without pointlessly hammering the GUI at
        # 60 Hz. PreciseTimer reduces timer jitter where the OS supports it.
        self.timer=QTimer(self);self.timer.setTimerType(Qt.PreciseTimer);self.timer.timeout.connect(self.render);self.timer.start(20)
    def render(self):
        for cid,reader in self.readers.items():
            image,version=reader.latest()
            if image is not None and version>self.seen[cid]:
                self.seen[cid]=version;self.tiles[cid].paint_image(image)
    def closeEvent(self,event):
        self.timer.stop()
        for reader in self.readers.values():reader.stop()
        event.accept()

if __name__=='__main__':
    app=QApplication(sys.argv);window=Window();window.show();sys.exit(app.exec())
