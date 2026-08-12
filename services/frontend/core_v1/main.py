from __future__ import annotations
import sys,threading,time,urllib.request
from PySide6.QtCore import Qt,QTimer
from PySide6.QtGui import QImage,QPixmap
from PySide6.QtWidgets import QApplication,QGridLayout,QLabel,QMainWindow,QWidget

CAMERAS=[f'CAM-{i:02d}' for i in range(1,7)]
ML_BASE='http://127.0.0.1:8001'

class MjpegReader:
    """Background network reader. Keeps only the newest decoded QImage."""
    def __init__(self,camera_id):
        self.camera_id=camera_id;self.url=f'{ML_BASE}/video/{camera_id}';self._stop=threading.Event();self._thread=None;self._lock=threading.Lock();self._image=None;self._version=0;self.frames=0
    def start(self):
        self._thread=threading.Thread(target=self._run,name=f'frontend-{self.camera_id}',daemon=True);self._thread.start()
    def stop(self):self._stop.set()
    def latest(self):
        with self._lock:return self._image,self._version
    def _run(self):
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(self.url,timeout=5) as response:
                    buffer=bytearray()
                    while not self._stop.is_set():
                        chunk=response.read(16384)
                        if not chunk:break
                        buffer.extend(chunk)
                        while True:
                            start=buffer.find(b'\xff\xd8');end=buffer.find(b'\xff\xd9',start+2)
                            if start<0 or end<0:
                                if len(buffer)>2_000_000:del buffer[:-100_000]
                                break
                            jpg=bytes(buffer[start:end+2]);del buffer[:end+2]
                            image=QImage.fromData(jpg,'JPG')
                            if image.isNull():continue
                            with self._lock:self._image=image.copy();self._version+=1
                            self.frames+=1
            except Exception:
                self._stop.wait(.5)

class CameraTile(QLabel):
    def __init__(self,camera_id):
        super().__init__(camera_id);self.camera_id=camera_id;self.setAlignment(Qt.AlignCenter);self.setMinimumSize(320,180);self.setStyleSheet('background:#05080d;color:#aab4c0;border:1px solid #253041;')
    def paint_image(self,image):
        pix=QPixmap.fromImage(image);self.setPixmap(pix.scaled(self.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))

class Window(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('AI Surveillance — Core v1 Camera Test');self.resize(1500,900)
        root=QWidget();grid=QGridLayout(root);grid.setSpacing(4);grid.setContentsMargins(4,4,4,4);self.setCentralWidget(root)
        self.tiles={};self.readers={};self.seen={}
        for index,camera_id in enumerate(CAMERAS):
            tile=CameraTile(camera_id);grid.addWidget(tile,index//2,index%2);self.tiles[camera_id]=tile
            reader=MjpegReader(camera_id);reader.start();self.readers[camera_id]=reader;self.seen[camera_id]=0
        self.timer=QTimer(self);self.timer.timeout.connect(self.render);self.timer.start(83)
    def render(self):
        for cid,reader in self.readers.items():
            image,version=reader.latest()
            if image is not None and version>self.seen[cid]:self.seen[cid]=version;self.tiles[cid].paint_image(image)
    def closeEvent(self,event):
        for reader in self.readers.values():reader.stop()
        event.accept()

if __name__=='__main__':
    app=QApplication(sys.argv);window=Window();window.show();sys.exit(app.exec())
