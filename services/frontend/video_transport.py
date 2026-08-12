"""Qt-native bounded/latest MJPEG receiver; never opens RTSP or camera devices."""
import os,re,threading,time
from collections import deque
from PySide6.QtCore import QObject,Signal,QTimer,QUrl,Qt
from PySide6.QtGui import QImage
from PySide6.QtNetwork import QNetworkAccessManager,QNetworkRequest
from shared.config import project_config
DEFAULT_PRESENTATION_FPS=max(1.0,float(project_config().get("display",{}).get("presentation_fps",12)))
class LatestDecodedFrame:
 """Thread-safe single-slot handoff: replacing pending work never builds backlog."""
 def __init__(self):self._item=None;self.replaced=0;self._lock=threading.Lock()
 def put(self,item):
  with self._lock:
   replaced=self._item is not None;self.replaced+=int(replaced);self._item=item;return replaced
 def take(self):
  with self._lock:item,self._item=self._item,None;return item
 def depth(self):
  with self._lock:return int(self._item is not None)
class MJPEGClient(QObject):
 frame=Signal(str,int,float,QImage);online=Signal(str,bool)
 def __init__(self,camera_id,url,parent=None):
  super().__init__(parent);self.camera_id=camera_id;self.url=url;self.network=QNetworkAccessManager(self);self.reply=None;self.buffer=bytearray();self.retry=QTimer(self);self.retry.setSingleShot(True);self.retry.setInterval(1000);self.retry.timeout.connect(self._reconnect);self.frame_id=0;self.last_frame=0.0;self.receive_fps=0.0;self.transport_latency_ms=0.0;self.decode_ms=0.0;self.prepare_ms=0.0;self.gui_schedule_wait_ms=0.0;self.dropped_display_frames=0;self.decoded_frames=0;self.prepared_frames=0;self.replaced_before_render=0;self.rendered_frames=0;self.pending_gui_updates=0;self.max_pending_gui_updates=0;self._raw=LatestDecodedFrame();self._prepared=LatestDecodedFrame();self._decode_wakeup=threading.Event();self._decode_stop=threading.Event();self._decode_thread=None;self._window_at=time.monotonic();self._window_frames=0;self._running=False
  self.presentation_fps=max(1.0,float(os.environ.get("SURVEILLANCE_DISPLAY_FPS",DEFAULT_PRESENTATION_FPS)));self.render_interval_ms=max(1,round(1000/self.presentation_fps));self._intervals={name:deque(maxlen=2000) for name in ("source","send","receive","decode_start","prepared","render","source_to_render","decode")};self._last_stage={}
  self.display_render_fps=0.0;self._render_window_at=time.monotonic();self._render_window_frames=0;self.duplicate_rendered_frame_total=0;self.newer_frame_waiting_while_duplicate_rendered_total=0
  self.raw_display_slot_max=0;self.jpeg_decode_pending_max=0;self.prepared_frame_slot_max=0;self.gui_update_pending_max=0
  self.nonmonotonic_source_frame_total=0;self.nonmonotonic_decoded_frame_total=0;self.nonmonotonic_rendered_frame_total=0;self._last_source_id=None;self._last_decoded_id=None;self._last_rendered_id=None
  self._last_source_timestamp=0.0;self._metrics_lock=threading.Lock()

  self.watchdog=QTimer(self);self.watchdog.setInterval(1000);self.watchdog.timeout.connect(self._check_freshness)
  self.render_timer=QTimer(self);self.render_timer.setTimerType(Qt.PreciseTimer);self.render_timer.setInterval(self.render_interval_ms);self.render_timer.timeout.connect(self._emit_latest)
  # Six render timers starting on the same event-loop turn create a periodic
  # six-camera paint burst. Keep the same cadence, but phase each camera across
  # one render period so video/overlay work is evenly distributed.
  match=re.search(r"(\d+)$",camera_id);ordinal=max(0,(int(match.group(1))-1) if match else 0)
  self.render_phase_ms=(ordinal%6)*self.render_interval_ms//6
  self.render_start_timer=QTimer(self);self.render_start_timer.setSingleShot(True);self.render_start_timer.timeout.connect(self._start_render_timer)
 def _start_render_timer(self):
  if self._running and not self.render_timer.isActive():self.render_timer.start()
 def _record_interval(self,name,value):
  previous=self._last_stage.get(name);self._last_stage[name]=float(value)
  if previous is not None:
   with self._metrics_lock:self._intervals[name].append(max(0.0,(float(value)-previous)*1000.0))
 def _record_sample(self,name,value):
  with self._metrics_lock:self._intervals[name].append(max(0.0,float(value)))
 def _stats(self,name):
  with self._metrics_lock:values=sorted(self._intervals[name])
  def pct(fraction):return values[min(len(values)-1,int((len(values)-1)*fraction))] if values else 0.0
  return {"p50":pct(.50),"p95":pct(.95),"max":values[-1] if values else 0.0}
 def start(self):
  self._running=True
  if self._decode_thread is None or not self._decode_thread.is_alive():
   self._decode_stop.clear();self._decode_thread=threading.Thread(target=self._decode_loop,name=f"jpeg-decode-{self.camera_id}",daemon=False);self._decode_thread.start()
  if self.reply is not None:return
  self.retry.stop();reply=self.network.get(QNetworkRequest(QUrl(self.url)));self.reply=reply;reply.readyRead.connect(lambda r=reply:self._read(r));reply.finished.connect(lambda r=reply:self._finished(r));self.watchdog.start();self.render_start_timer.start(self.render_phase_ms)
 def set_display_mode(self,enabled):
  endpoint=self.url.rsplit("/video/",1)[0]+f"/display-mode/{self.camera_id}?enabled={'true' if enabled else 'false'}";request=QNetworkRequest(QUrl(endpoint));request.setHeader(QNetworkRequest.ContentTypeHeader,"application/octet-stream");reply=self.network.post(request,b"");reply.finished.connect(reply.deleteLater)
 def stop(self):
  self._running=False;self.retry.stop();self.watchdog.stop();self.render_start_timer.stop();self.render_timer.stop();self._decode_stop.set();self._decode_wakeup.set();self._raw.take();self._prepared.take();reply,self.reply=self.reply,None;self.buffer.clear()
  if reply is not None:
   try:reply.abort()
   except RuntimeError:pass
   try:reply.deleteLater()
   except RuntimeError:pass
  thread,self._decode_thread=self._decode_thread,None
  if thread is not None and thread is not threading.current_thread():thread.join(1)
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
  match=re.search(br"X-Frame-Id:\s*(\d+)",header,re.I);next_frame_id=int(match.group(1)) if match else self.frame_id+1
  if self._last_source_id is not None and next_frame_id<=self._last_source_id:self.nonmonotonic_source_frame_total+=1
  self.frame_id=next_frame_id;self._last_source_id=next_frame_id
  stamp=re.search(br"X-Timestamp:\s*([0-9.]+)",header,re.I);timestamp=float(stamp.group(1)) if stamp else time.time();encoded=re.search(br"X-Encoded-At:\s*([0-9.]+)",header,re.I);encoded_at=float(encoded.group(1)) if encoded else timestamp
  now=time.time();received_mono=time.monotonic();self._record_interval("source",timestamp);self._record_interval("send",encoded_at);self._record_interval("receive",received_mono);self._last_source_timestamp=timestamp
  self.transport_latency_ms=max(0.0,(now-encoded_at)*1000);self.last_frame=received_mono;self._window_frames+=1;elapsed=self.last_frame-self._window_at
  if elapsed>=1:self.receive_fps=self._window_frames/elapsed;self._window_frames=0;self._window_at=self.last_frame
  if self._raw.put((self.camera_id,self.frame_id,timestamp,jpeg,received_mono)):self.dropped_display_frames+=1
  self.raw_display_slot_max=max(self.raw_display_slot_max,self._raw.depth());self.jpeg_decode_pending_max=max(self.jpeg_decode_pending_max,self._raw.depth())
  self._decode_wakeup.set()
 def _decode_loop(self):
  while not self._decode_stop.is_set():
   self._decode_wakeup.wait(.5);self._decode_wakeup.clear()
   while not self._decode_stop.is_set():
    item=self._raw.take()
    if item is None:break
    camera_id,frame_id,timestamp,jpeg,received_mono=item;decode_started=time.monotonic();started=time.perf_counter();self._record_interval("decode_start",decode_started);image=QImage.fromData(jpeg,"JPG");completed=time.monotonic();self.decode_ms=(time.perf_counter()-started)*1000;self._record_sample("decode",self.decode_ms);self.prepare_ms=self.decode_ms;self.decoded_frames+=1
    if self._last_decoded_id is not None and frame_id<self._last_decoded_id:self.nonmonotonic_decoded_frame_total+=1
    self._last_decoded_id=frame_id
    if image.isNull():continue
    self.prepared_frames+=1
    self._record_interval("prepared",completed)
    if self._prepared.put((camera_id,frame_id,timestamp,image,completed)):self.replaced_before_render+=1;self.dropped_display_frames+=1
    self.prepared_frame_slot_max=max(self.prepared_frame_slot_max,self._prepared.depth())
 def _emit_latest(self):
  self.pending_gui_updates=self._prepared.depth();self.max_pending_gui_updates=max(self.max_pending_gui_updates,self.pending_gui_updates);self.gui_update_pending_max=max(self.gui_update_pending_max,self.pending_gui_updates)
  item=self._prepared.take();self.pending_gui_updates=self._prepared.depth()
  if item is None:return
  camera_id,frame_id,timestamp,image,prepared_mono=item;now_mono=time.monotonic()
  self._record_interval("render",now_mono);self._record_sample("source_to_render",max(0.0,(time.time()-timestamp)*1000.0))
  self.gui_schedule_wait_ms=max(0.0,(now_mono-prepared_mono)*1000)
  if self._last_rendered_id is not None and frame_id<self._last_rendered_id:self.nonmonotonic_rendered_frame_total+=1
  self._last_rendered_id=frame_id;self.rendered_frames+=1;self._render_window_frames+=1
  elapsed=now_mono-self._render_window_at
  if elapsed>=1.0:self.display_render_fps=self._render_window_frames/elapsed;self._render_window_frames=0;self._render_window_at=now_mono
  # The frame callback marks the camera online. Emitting a second status event
  # for every frame made the UI invalidate and paint each surface twice.
  # Offline transitions remain owned by the watchdog/finished paths.
  self.frame.emit(camera_id,frame_id,timestamp,image)

 def runtime_metrics(self):
  render=self._stats("render")
  with self._metrics_lock:render_values=list(self._intervals["render"])
  jitter=sorted(abs(value-self.render_interval_ms) for value in render_values)
  def pct(values,fraction):return values[min(len(values)-1,int((len(values)-1)*fraction))] if values else 0.0
  return {"presentation_fps_target":self.presentation_fps,"receive_fps":self.receive_fps,"display_render_fps":self.display_render_fps,"rendered_frames":self.rendered_frames,"decoded_frames":self.decoded_frames,"prepared_frames":self.prepared_frames,"replaced_before_render":self.replaced_before_render,"dropped_display_frames":self.dropped_display_frames,"pending_gui_updates":self.pending_gui_updates,"max_pending_gui_updates":self.max_pending_gui_updates,"decode_ms":self.decode_ms,"prepare_ms":self.prepare_ms,"gui_schedule_wait_ms":self.gui_schedule_wait_ms,"source_frame_interval_ms":self._stats("source"),"display_transport_send_interval_ms":self._stats("send"),"display_receive_interval_ms":self._stats("receive"),"jpeg_decode_start_interval_ms":self._stats("decode_start"),"jpeg_decode_ms_stats":self._stats("decode"),"prepared_frame_interval_ms":self._stats("prepared"),"gui_render_interval_ms":render,"display_jitter_ms":{"p50":pct(jitter,.50),"p95":pct(jitter,.95)},"render_gap_over_100ms_total":sum(value>100 for value in render_values),"render_gap_over_150ms_total":sum(value>150 for value in render_values),"render_gap_over_200ms_total":sum(value>200 for value in render_values),"render_gap_over_300ms_total":sum(value>300 for value in render_values),"source_to_render_latency_ms":self._stats("source_to_render"),"duplicate_rendered_frame_total":self.duplicate_rendered_frame_total,"newer_frame_waiting_while_duplicate_rendered_total":self.newer_frame_waiting_while_duplicate_rendered_total,"raw_display_slot_max":self.raw_display_slot_max,"jpeg_decode_pending_max":self.jpeg_decode_pending_max,"prepared_frame_slot_max":self.prepared_frame_slot_max,"gui_update_pending_max":self.gui_update_pending_max,"nonmonotonic_source_frame_total":self.nonmonotonic_source_frame_total,"nonmonotonic_decoded_frame_total":self.nonmonotonic_decoded_frame_total,"nonmonotonic_rendered_frame_total":self.nonmonotonic_rendered_frame_total}
