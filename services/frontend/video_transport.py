"""Qt-native MJPEG receiver; it never opens RTSP or camera devices."""
import re,time
from PySide6.QtCore import QObject,Signal,QTimer,QUrl
from PySide6.QtGui import QImage
from PySide6.QtNetwork import QNetworkAccessManager,QNetworkRequest
class MJPEGClient(QObject):
 frame=Signal(str,int,float,QImage);online=Signal(str,bool)
 def __init__(self,camera_id,url,parent=None):
  super().__init__(parent);self.camera_id=camera_id;self.url=url;self.network=QNetworkAccessManager(self);self.reply=None;self.buffer=bytearray();self.frame_id=0;self.last_frame=0.0
  self.watchdog=QTimer(self);self.watchdog.setInterval(1000);self.watchdog.timeout.connect(self._check_freshness)
 def start(self):
  self.reply=self.network.get(QNetworkRequest(QUrl(self.url)));self.reply.readyRead.connect(self._read);self.reply.finished.connect(self._finished);self.watchdog.start()
 def stop(self):
  self.watchdog.stop()
  if self.reply:self.reply.abort();self.reply.deleteLater();self.reply=None
 def _finished(self):self.online.emit(self.camera_id,False)
 def _check_freshness(self):
  if self.last_frame and time.monotonic()-self.last_frame>2:self.online.emit(self.camera_id,False)
 def _read(self):
  self.buffer.extend(bytes(self.reply.readAll()));self.online.emit(self.camera_id,True)
  if len(self.buffer)>8_000_000:self.buffer=self.buffer[-4_000_000:]
  while True:
   start=self.buffer.find(b"\xff\xd8");end=self.buffer.find(b"\xff\xd9",max(start,0)+2)
   if start<0 or end<0:return
   header=bytes(self.buffer[max(0,self.buffer.rfind(b"--frame",0,start)):start]);jpeg=bytes(self.buffer[start:end+2]);del self.buffer[:end+2]
   match=re.search(br"X-Frame-Id:\s*(\d+)",header,re.I);self.frame_id=int(match.group(1)) if match else self.frame_id+1
   stamp=re.search(br"X-Timestamp:\s*([0-9.]+)",header,re.I);timestamp=float(stamp.group(1)) if stamp else time.time();self.last_frame=time.monotonic()
   image=QImage.fromData(jpeg,"JPG")
   if not image.isNull():self.frame.emit(self.camera_id,self.frame_id,timestamp,image)
