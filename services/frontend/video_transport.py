"""Qt-native bounded/latest MJPEG receiver; never opens RTSP or camera devices."""
import re,time
from PySide6.QtCore import QObject,Signal,QTimer,QUrl
from PySide6.QtGui import QImage
from PySide6.QtNetwork import QNetworkAccessManager,QNetworkRequest
class MJPEGClient(QObject):
 frame=Signal(str,int,float,QImage);online=Signal(str,bool)
 def __init__(self,camera_id,url,parent=None):
  super().__init__(parent);self.camera_id=camera_id;self.url=url;self.network=QNetworkAccessManager(self);self.reply=None;self.buffer=bytearray();self.retry=QTimer(self);self.retry.setSingleShot(True);self.retry.setInterval(1000);self.retry.timeout.connect(self._reconnect);self.frame_id=0;self.last_frame=0.0;self.receive_fps=0.0;self.transport_latency_ms=0.0;self.dropped_display_frames=0;self._window_at=time.monotonic();self._window_frames=0;self._running=False
  self.watchdog=QTimer(self);self.watchdog.setInterval(1000);self.watchdog.timeout.connect(self._check_freshness)
 def start(self):
  self._running=True
  if self.reply is not None:return
  self.retry.stop();reply=self.network.get(QNetworkRequest(QUrl(self.url)));self.reply=reply;reply.readyRead.connect(lambda r=reply:self._read(r));reply.finished.connect(lambda r=reply:self._finished(r));self.watchdog.start()
 def stop(self):
  self._running=False;self.retry.stop();self.watchdog.stop();reply,self.reply=self.reply,None;self.buffer.clear()
  if reply is None:return
  try:reply.abort()
  except RuntimeError:pass
  try:reply.deleteLater()
  except RuntimeError:pass
 def _finished(self,reply):
  if reply is not self.reply:return
  self.reply=None;self.online.emit(self.camera_id,False)
  try:reply.deleteLater()
  except RuntimeError:pass
  if self._running:self.retry.start()
 def _reconnect(self):
  if self._running and self.reply is None:self.start()
 def _check_freshness(self):
  if self.last_frame and time.monotonic()-self.last_frame>2:self.online.emit(self.camera_id,False)
 def _read(self,reply=None):
  reply=reply or self.reply
  if not self._running or reply is None or reply is not self.reply:return
  self.buffer.extend(bytes(reply.readAll()))
  if len(self.buffer)>8_000_000:self.buffer=self.buffer[-4_000_000:]
  latest=None;complete=0
  while True:
   start=self.buffer.find(b"\xff\xd8");end=self.buffer.find(b"\xff\xd9",max(start,0)+2)
   if start<0 or end<0:break
   header=bytes(self.buffer[max(0,self.buffer.rfind(b"--frame",0,start)):start]);jpeg=bytes(self.buffer[start:end+2]);del self.buffer[:end+2];complete+=1;latest=(header,jpeg)
  if latest is None:return
  self.dropped_display_frames+=max(0,complete-1);header,jpeg=latest
  match=re.search(br"X-Frame-Id:\s*(\d+)",header,re.I);self.frame_id=int(match.group(1)) if match else self.frame_id+1
  stamp=re.search(br"X-Timestamp:\s*([0-9.]+)",header,re.I);timestamp=float(stamp.group(1)) if stamp else time.time();encoded=re.search(br"X-Encoded-At:\s*([0-9.]+)",header,re.I);encoded_at=float(encoded.group(1)) if encoded else timestamp
  now=time.time();self.transport_latency_ms=max(0.0,(now-encoded_at)*1000);self.last_frame=time.monotonic();self._window_frames+=1;elapsed=self.last_frame-self._window_at
  if elapsed>=1:self.receive_fps=self._window_frames/elapsed;self._window_frames=0;self._window_at=self.last_frame
  image=QImage.fromData(jpeg,"JPG")
  if not image.isNull():self.online.emit(self.camera_id,True);self.frame.emit(self.camera_id,self.frame_id,timestamp,image)
