"""Small internal ML control/status/video API."""
import queue,threading,time
from collections import defaultdict,deque
from fastapi import FastAPI,HTTPException
from fastapi.responses import StreamingResponse
from shared.config import project_config
def display_fps_cap(config):return max(1.0,float(config.get("fps_cap",18)))
_display=project_config().get("display",{});JPEG_QUALITY=max(60,min(95,int(_display.get("jpeg_quality",85))));DISPLAY_FPS=display_fps_cap(_display);DISPLAY_MAX_WIDTH=max(320,int(_display.get("max_width",1280)));DISPLAY_MAX_HEIGHT=max(180,int(_display.get("max_height",720)));DISPLAY_FULLSCREEN_MAX_WIDTH=max(DISPLAY_MAX_WIDTH,int(_display.get("fullscreen_max_width",1280)));DISPLAY_FULLSCREEN_MAX_HEIGHT=max(DISPLAY_MAX_HEIGHT,int(_display.get("fullscreen_max_height",720)))
def display_scale(width,height,high_quality=False):
 max_width=DISPLAY_FULLSCREEN_MAX_WIDTH if high_quality else DISPLAY_MAX_WIDTH;max_height=DISPLAY_FULLSCREEN_MAX_HEIGHT if high_quality else DISPLAY_MAX_HEIGHT
 return min(1.0,max_width/max(1,width),max_height/max(1,height))
class MLRuntimeState:
 def __init__(self):
  self.commands=queue.Queue(64);self.metrics={};self.status={"status":"starting"};self.frames={};self.lock=threading.Lock();self.video_stats={};self._video_intervals=defaultdict(lambda:defaultdict(lambda:deque(maxlen=2000)));self._video_last=defaultdict(dict);self._display_tokens={};self._display_checked={};self._closed=False;self.high_quality_camera=None;self.high_quality_reuses_ai=False
 def _record_video_interval(self,camera_id,stage,now):
  previous=self._video_last[camera_id].get(stage);self._video_last[camera_id][stage]=float(now)
  if previous is not None:self._video_intervals[camera_id][stage].append(max(0.0,(float(now)-previous)*1000.0))
 @staticmethod
 def _interval_stats(values):
  ordered=sorted(values)
  def pct(p):return ordered[min(len(ordered)-1,int((len(ordered)-1)*p))] if ordered else 0.0
  return {"count":len(ordered),"p50":pct(.50),"p95":pct(.95),"max":ordered[-1] if ordered else 0.0,"over_150":sum(v>150 for v in ordered),"over_200":sum(v>200 for v in ordered),"over_300":sum(v>300 for v in ordered)}
 def command(self,item):
  try:self.commands.put_nowait(item)
  except queue.Full:raise HTTPException(429,"ML command queue full")
 def poll(self):
  result=[]
  while True:
   try:result.append(self.commands.get_nowait())
   except queue.Empty:return result
 def frame(self,packet):
  now=time.monotonic();camera_id=packet.camera_id
  with self.lock:
   if self._closed or (self.high_quality_camera==camera_id and not self.high_quality_reuses_ai):return
   stats=self.video_stats.setdefault(camera_id,{"started":now,"reader_frames":0,"display_frames":0,"display_drops":0,"encoded_frames":0,"encoded_bytes":0,"encode_ms_total":0.0,"encode_ms_last":0.0,"encode_ms_max":0.0,"encode_started":None,"last_frame_timestamp":0.0});stats["reader_frames"]+=1;self._record_video_interval(camera_id,"source",packet.capture_monotonic or now)
   previous=self._display_checked.get(camera_id,now);tokens=min(2.0,self._display_tokens.get(camera_id,1.0)+max(0.0,now-previous)*DISPLAY_FPS);self._display_checked[camera_id]=now
   if tokens<1.0:self._display_tokens[camera_id]=tokens;stats["display_drops"]+=1;return
   self._display_tokens[camera_id]=tokens-1.0;stats["display_frames"]+=1;stats["last_frame_timestamp"]=packet.capture_timestamp;self.frames[camera_id]=(packet.frame_id,packet.capture_timestamp,packet.frame)
 def display_frame(self,packet):
  now=time.monotonic();camera_id=packet.camera_id
  with self.lock:
   if self._closed or self.high_quality_camera!=camera_id:return
   stats=self.video_stats.setdefault(camera_id,{"started":now,"reader_frames":0,"display_frames":0,"display_drops":0,"encoded_frames":0,"encoded_bytes":0,"encode_ms_total":0.0,"encode_ms_last":0.0,"encode_ms_max":0.0,"encode_started":None,"last_frame_timestamp":0.0})
   stats["display_frames"]+=1;stats["last_frame_timestamp"]=packet.capture_timestamp;self.frames[camera_id]=(packet.frame_id,packet.capture_timestamp,packet.frame)
 def shutdown(self,timeout=2.0):
  with self.lock:self._closed=True
  return True
 def set_high_quality(self,camera_id,reuses_ai=False):
  with self.lock:self.high_quality_camera=camera_id;self.high_quality_reuses_ai=bool(reuses_ai)
 def clear_high_quality(self,camera_id):
  with self.lock:
   if self.high_quality_camera==camera_id:self.high_quality_camera=None;self.high_quality_reuses_ai=False;self._display_tokens[camera_id]=1.0;self._display_checked[camera_id]=time.monotonic()
 def video_metrics(self):
  now=time.monotonic()
  with self.lock:
   return {cid:{**s,"transport_fps":s["encoded_frames"]/max(1e-6,now-(s["encode_started"] or now)),"display_handoff_fps":s["display_frames"]/max(1e-6,now-s["started"]),"jpeg_quality":JPEG_QUALITY,"display_fps_cap":DISPLAY_FPS,"jpeg_size_avg_bytes":s["encoded_bytes"]/max(1,s["encoded_frames"]),"encode_ms_avg":s["encode_ms_total"]/max(1,s["encoded_frames"]),**{f"{stage}_interval_ms":self._interval_stats(values) for stage,values in self._video_intervals[cid].items()}} for cid,s in self.video_stats.items()}
runtime=MLRuntimeState();app=FastAPI(title="Internal ML Control API")
@app.get("/health")
def health():
 state=dict(runtime.status);cameras=runtime.metrics.get("cameras",{});configured=int(state.get("cameras",len(cameras)));online=sum(1 for item in cameras.values() if item.get("online")) if cameras else 0;detector=bool(state.get("detector_ready",False));workers=bool(state.get("secondary_ready",False));running=state.get("status")=="running";level="healthy" if running and detector and workers and configured and online==configured else "degraded" if running and detector and workers else "unhealthy"
 return {"service":"ml-service","status":level,"components":{"process":state.get("status","unknown"),"cameras":{"configured":configured,"online":online},"detector_ready":detector,"reid_ready":bool(state.get("reid_ready",False)),"face_ready":bool(state.get("face_ready",False)),"secondary_workers":workers},"event_delivery":state.get("event_delivery","unknown")}
@app.get("/ready")
def ready():
 status=runtime.status.get("status","unknown");return {"service":"ml-service","status":status,"ready":status=="running" and bool(runtime.status.get("detector_ready")) and bool(runtime.status.get("secondary_ready")),"cameras_active":runtime.status.get("cameras",0)}
@app.get("/status")
def status():return runtime.status
@app.get("/metrics")
def metrics():return runtime.metrics
@app.post("/commands")
def command(message:dict):
 if not isinstance(message.get("type"),str):raise HTTPException(422,"Command type is required")
 runtime.command(message);return {"accepted":True}
@app.post("/display-mode/{camera_id}")
def display_mode(camera_id:str,enabled:bool):
 runtime.command({"type":"display.source.start" if enabled else "display.source.stop","camera_id":camera_id});return {"accepted":True,"camera_id":camera_id,"enabled":enabled}
@app.get("/video/{camera_id}")
def video(camera_id:str):
 def stream():
  import cv2
  last=-1
  while True:
   with runtime.lock:item=runtime.frames.get(camera_id)
   if item is None or item[0]==last:time.sleep(.01);continue
   last=item[0];started=time.time();frame=item[2];height,width=frame.shape[:2];high_quality=runtime.high_quality_camera==camera_id;scale=display_scale(width,height,high_quality);display_frame=cv2.resize(frame,(max(1,int(width*scale)),max(1,int(height*scale))),interpolation=cv2.INTER_AREA) if scale<1.0 else frame;encoded_height,encoded_width=display_frame.shape[:2];encode_started=time.perf_counter();ok,encoded=cv2.imencode(".jpg",display_frame,[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])
   if ok:
    with runtime.lock:
     stats=runtime.video_stats.get(camera_id)
     if stats is not None:
      encode_ms=(time.perf_counter()-encode_started)*1000;runtime._record_video_interval(camera_id,"mjpeg_encode",time.monotonic());stats["encode_started"]=stats["encode_started"] or time.monotonic();stats["encoded_frames"]+=1;stats["encoded_bytes"]+=int(encoded.nbytes);stats["encode_ms_total"]+=encode_ms;stats["encode_ms_last"]=encode_ms;stats["encode_ms_max"]=max(stats["encode_ms_max"],encode_ms);stats["source_width"]=width;stats["source_height"]=height;stats["encoded_width"]=encoded_width;stats["encoded_height"]=encoded_height;stats["high_quality"]=high_quality
    yield b"--frame\r\nContent-Type: image/jpeg\r\nX-Frame-Id: "+str(item[0]).encode()+b"\r\nX-Timestamp: "+str(item[1]).encode()+b"\r\nX-Encoded-At: "+str(started).encode()+b"\r\n\r\n"+encoded.tobytes()+b"\r\n"
 return StreamingResponse(stream(),media_type="multipart/x-mixed-replace; boundary=frame")
