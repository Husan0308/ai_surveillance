"""Small internal ML control/status/video API."""
import queue,threading,time
from fastapi import FastAPI,HTTPException
from fastapi.responses import StreamingResponse
from shared.config import project_config
_display=project_config().get("display",{});JPEG_QUALITY=max(60,min(95,int(_display.get("jpeg_quality",85))));DISPLAY_FPS=max(1.0,float(_display.get("fps_cap",18)));DISPLAY_MAX_WIDTH=max(320,int(_display.get("max_width",1280)));DISPLAY_MAX_HEIGHT=max(180,int(_display.get("max_height",720)))
class MLRuntimeState:
 def __init__(self):
  self.commands=queue.Queue(64);self.metrics={};self.status={"status":"starting"};self.frames={};self.lock=threading.Lock();self.video_stats={};self._display_tokens={};self._display_checked={}
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
   stats=self.video_stats.setdefault(camera_id,{"started":now,"reader_frames":0,"display_frames":0,"display_drops":0,"encoded_frames":0,"encoded_bytes":0,"encode_ms_total":0.0,"encode_ms_last":0.0,"encode_ms_max":0.0,"encode_started":None,"last_frame_timestamp":0.0});stats["reader_frames"]+=1
   previous=self._display_checked.get(camera_id,now);tokens=min(2.0,self._display_tokens.get(camera_id,1.0)+max(0.0,now-previous)*DISPLAY_FPS);self._display_checked[camera_id]=now
   if tokens<1.0:self._display_tokens[camera_id]=tokens;stats["display_drops"]+=1;return
   self._display_tokens[camera_id]=tokens-1.0;stats["display_frames"]+=1;stats["last_frame_timestamp"]=packet.capture_timestamp;self.frames[camera_id]=(packet.frame_id,packet.capture_timestamp,packet.frame)
 def video_metrics(self):
  now=time.monotonic()
  with self.lock:
   return {cid:{**s,"transport_fps":s["encoded_frames"]/max(1e-6,now-(s["encode_started"] or now)),"display_handoff_fps":s["display_frames"]/max(1e-6,now-s["started"]),"jpeg_quality":JPEG_QUALITY,"display_fps_cap":DISPLAY_FPS,"jpeg_size_avg_bytes":s["encoded_bytes"]/max(1,s["encoded_frames"]),"encode_ms_avg":s["encode_ms_total"]/max(1,s["encoded_frames"])} for cid,s in self.video_stats.items()}
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
@app.get("/video/{camera_id}")
def video(camera_id:str):
 def stream():
  import cv2
  last=-1
  while True:
   with runtime.lock:item=runtime.frames.get(camera_id)
   if item is None or item[0]==last:time.sleep(.01);continue
   last=item[0];started=time.time();frame=item[2];height,width=frame.shape[:2];scale=min(1.0,DISPLAY_MAX_WIDTH/width,DISPLAY_MAX_HEIGHT/height);display_frame=cv2.resize(frame,(max(1,int(width*scale)),max(1,int(height*scale))),interpolation=cv2.INTER_AREA) if scale<1.0 else frame;encode_started=time.perf_counter();ok,encoded=cv2.imencode(".jpg",display_frame,[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])
   if ok:
    with runtime.lock:
     stats=runtime.video_stats.get(camera_id)
     if stats is not None:
      encode_ms=(time.perf_counter()-encode_started)*1000;stats["encode_started"]=stats["encode_started"] or time.monotonic();stats["encoded_frames"]+=1;stats["encoded_bytes"]+=int(encoded.nbytes);stats["encode_ms_total"]+=encode_ms;stats["encode_ms_last"]=encode_ms;stats["encode_ms_max"]=max(stats["encode_ms_max"],encode_ms)
    yield b"--frame\r\nContent-Type: image/jpeg\r\nX-Frame-Id: "+str(item[0]).encode()+b"\r\nX-Timestamp: "+str(item[1]).encode()+b"\r\nX-Encoded-At: "+str(started).encode()+b"\r\n\r\n"+encoded.tobytes()+b"\r\n"
 return StreamingResponse(stream(),media_type="multipart/x-mixed-replace; boundary=frame")
