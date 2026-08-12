from __future__ import annotations
import logging,threading,time
from dataclasses import dataclass,asdict
from .latest_frame import Frame,LatestFrameStore
from services.ml_service.cameras.gstreamer import GStreamerCapture

log=logging.getLogger(__name__)

@dataclass
class CameraMetrics:
    online: bool=False
    frame_id: int=0
    source_fps: float=0.0
    width: int=0
    height: int=0
    reconnects: int=0
    read_failures: int=0
    consecutive_timeouts: int=0
    startup_waiting: bool=False
    last_frame_age_ms: float=0.0
    last_error: str=""

class CameraWorker:
    """Exactly one capture thread and one single-slot frame store per camera.

    Core-v1 deliberately tolerates RTSP/NVDEC startup. GStreamerCapture.read()
    waits up to one second for a sample; treating a single timeout as fatal made
    all six cameras tear down before RTSP SETUP/PLAY and the first keyframe had
    time to complete. After the first frame we also tolerate a short bounded run
    of empty pulls before reconnecting.
    """
    def __init__(self,config:dict,store:LatestFrameStore,core_config:dict|None=None):
        self.config=dict(config);self.core_config=dict(core_config or {});self.camera_id=str(config["id"]);self.store=store
        self._stop=threading.Event();self._thread=None;self._lock=threading.Lock();self._capture=None
        self._metrics=CameraMetrics();self._fps_started=time.monotonic();self._fps_frames=0
        self.startup_grace_sec=max(2.0,float(self.core_config.get("startup_grace_sec",8.0)))
        self.max_read_timeouts=max(1,int(self.core_config.get("max_read_timeouts",3)))
        self.reconnect_delay_sec=max(.25,float(self.core_config.get("reconnect_delay_sec",2.0)))
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
        # Keep the camera-specific RTSP latency/codec, but do not let old
        # reconnect_interval values control this fresh core.
        return GStreamerCapture(cfg)
    def _run(self):
        while not self._stop.is_set():
            cap=None
            try:
                cap=self._open();self._capture=cap
                if not cap.isOpened():raise RuntimeError("GStreamer pipeline did not open")
                opened_at=time.monotonic();had_frame=False;timeouts=0
                with self._lock:
                    self._metrics.online=False;self._metrics.startup_waiting=True;self._metrics.last_error="";self._metrics.consecutive_timeouts=0
                while not self._stop.is_set():
                    ok,image=cap.read()
                    if not ok or image is None:
                        if self._stop.is_set():break
                        with self._lock:
                            self._metrics.read_failures+=1;self._metrics.consecutive_timeouts+=1
                        # A real bus error closes GStreamerCapture immediately.
                        if not cap.isOpened():
                            raise RuntimeError("GStreamer pipeline error before sample")
                        timeouts+=1
                        # Initial RTSP DESCRIBE/SETUP/PLAY + decoder negotiation +
                        # first IDR can legitimately take multiple seconds.
                        if not had_frame and time.monotonic()-opened_at < self.startup_grace_sec:
                            continue
                        # Once live, absorb brief packet/keyframe gaps instead of
                        # destroying a healthy pipeline after one empty pull.
                        if had_frame and timeouts < self.max_read_timeouts:
                            continue
                        reason="startup grace expired without first frame" if not had_frame else f"{timeouts} consecutive read timeouts"
                        raise RuntimeError(reason)
                    had_frame=True;timeouts=0
                    now=time.time();mono=time.monotonic();h,w=image.shape[:2]
                    with self._lock:
                        self._metrics.online=True;self._metrics.startup_waiting=False;self._metrics.consecutive_timeouts=0;self._metrics.last_error=""
                        self._metrics.frame_id+=1;frame_id=self._metrics.frame_id;self._metrics.width=w;self._metrics.height=h
                        self._fps_frames+=1;elapsed=mono-self._fps_started
                        if elapsed>=1.0:
                            self._metrics.source_fps=self._fps_frames/elapsed;self._fps_frames=0;self._fps_started=mono
                        self._last_frame_mono=mono
                    self.store.put(Frame(self.camera_id,frame_id,now,mono,image,w,h))
            except Exception as exc:
                with self._lock:
                    self._metrics.online=False;self._metrics.startup_waiting=False;self._metrics.reconnects+=1;self._metrics.last_error=f"{type(exc).__name__}: {exc}"
                stage={}
                if cap is not None:
                    try:stage=cap.stage_metrics()
                    except Exception:pass
                log.warning("CORE_V1_CAMERA_RECONNECT camera=%s error=%s stage=%s",self.camera_id,exc,stage)
                if not self._stop.is_set():self._stop.wait(self.reconnect_delay_sec)
            finally:
                if cap is not None:
                    try:cap.release()
                    except Exception:pass
                self._capture=None
        with self._lock:self._metrics.online=False;self._metrics.startup_waiting=False
