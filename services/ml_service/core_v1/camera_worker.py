from __future__ import annotations
import threading,time
from dataclasses import dataclass,asdict
from .latest_frame import Frame,LatestFrameStore
from services.ml_service.cameras.gstreamer import GStreamerCapture

@dataclass
class CameraMetrics:
    online: bool=False
    frame_id: int=0
    source_fps: float=0.0
    width: int=0
    height: int=0
    reconnects: int=0
    read_failures: int=0
    last_frame_age_ms: float=0.0

class CameraWorker:
    """Exactly one capture thread and one single-slot frame store per camera."""
    def __init__(self,config:dict,store:LatestFrameStore):
        self.config=dict(config);self.camera_id=str(config["id"]);self.store=store
        self._stop=threading.Event();self._thread=None;self._lock=threading.Lock();self._capture=None
        self._metrics=CameraMetrics();self._fps_started=time.monotonic();self._fps_frames=0
    def start(self):
        if self._thread and self._thread.is_alive():return
        self._stop.clear();self._thread=threading.Thread(target=self._run,name=f"core-v1-{self.camera_id}",daemon=False);self._thread.start()
    def stop(self):
        self._stop.set();cap=self._capture
        if cap is not None:
            try:cap.interrupt()
            except Exception:pass
    def join(self,timeout=5):
        if self._thread:self._thread.join(timeout)
        return not self._thread or not self._thread.is_alive()
    def metrics(self):
        with self._lock:
            result=asdict(self._metrics)
            if self._metrics.frame_id:result["last_frame_age_ms"]=max(0.0,(time.monotonic()-getattr(self,"_last_frame_mono",time.monotonic()))*1000)
            result["replaced_frames"]=self.store.replaced
            return result
    def _open(self):
        cfg={**self.config,"source":self.config.get("display_source") or self.config.get("source"),"codec":self.config.get("display_codec") or self.config.get("codec")}
        return GStreamerCapture(cfg)
    def _run(self):
        delay=max(.25,float(self.config.get("reconnect_interval",2.0)))
        while not self._stop.is_set():
            cap=None
            try:
                cap=self._open();self._capture=cap
                if not cap.isOpened():raise RuntimeError("GStreamer pipeline did not open")
                with self._lock:self._metrics.online=True
                while not self._stop.is_set():
                    ok,image=cap.read()
                    if not ok or image is None:
                        with self._lock:self._metrics.read_failures+=1
                        raise RuntimeError("camera read failed")
                    now=time.time();mono=time.monotonic();h,w=image.shape[:2]
                    with self._lock:
                        self._metrics.frame_id+=1;frame_id=self._metrics.frame_id;self._metrics.width=w;self._metrics.height=h
                        self._fps_frames+=1;elapsed=mono-self._fps_started
                        if elapsed>=1.0:
                            self._metrics.source_fps=self._fps_frames/elapsed;self._fps_frames=0;self._fps_started=mono
                        self._last_frame_mono=mono
                    self.store.put(Frame(self.camera_id,frame_id,now,mono,image,w,h))
            except Exception:
                with self._lock:
                    self._metrics.online=False;self._metrics.reconnects+=1
                if not self._stop.is_set():self._stop.wait(delay)
            finally:
                if cap is not None:
                    try:cap.release()
                    except Exception:pass
                self._capture=None
        with self._lock:self._metrics.online=False
