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
    drift_reconnects: int=0
    read_failures: int=0
    consecutive_timeouts: int=0
    consecutive_lag_samples: int=0
    startup_waiting: bool=False
    last_frame_age_ms: float=0.0
    pipeline_lag_ms: float|None=None
    postdecode_queue_buffers: int|None=None
    source_runtime: dict|None=None
    last_error: str=""

class CameraWorker:
    """Exactly one capture thread and one single-slot frame store per camera.

    If the GStreamer pipeline itself drifts far behind live time for several
    consecutive decoded samples, only that camera pipeline is rebuilt. The
    LatestFrameStore is never turned into a FIFO and no historical frames are
    replayed after reconnect.
    """
    def __init__(self,config:dict,store:LatestFrameStore,core_config:dict|None=None):
        self.config=dict(config);self.core_config=dict(core_config or {});self.camera_id=str(config["id"]);self.store=store
        self._stop=threading.Event();self._thread=None;self._lock=threading.Lock();self._capture=None
        self._metrics=CameraMetrics(source_runtime={});self._fps_started=time.monotonic();self._fps_frames=0
        self.startup_grace_sec=max(2.0,float(self.core_config.get("startup_grace_sec",15.0)))
        self.max_read_timeouts=max(1,int(self.core_config.get("max_read_timeouts",5)))
        self.reconnect_delay_sec=max(.25,float(self.core_config.get("reconnect_delay_sec",2.0)))
        self.max_pipeline_lag_ms=max(0.0,float(self.core_config.get("max_pipeline_lag_ms",0.0)))
        self.max_pipeline_lag_samples=max(1,int(self.core_config.get("max_pipeline_lag_samples",8)))
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
        cfg={
            **self.config,
            "source":self.config.get("display_source") or self.config.get("source"),
            "codec":self.config.get("display_codec") or self.config.get("codec"),
            "latency_ms":int(self.core_config.get("rtsp_latency_ms",self.config.get("latency_ms",100))),
            "drop_on_latency":bool(self.core_config.get("drop_on_latency",False)),
            "decoder_extra_surfaces":int(self.core_config.get("decoder_extra_surfaces",2)),
            "decoder_low_latency_mode":bool(self.core_config.get("decoder_low_latency_mode",False)),
            "capture_timeout_ms":int(self.core_config.get("capture_timeout_ms",1000)),
            "rtsp_transport":str(self.core_config.get("rtsp_transport",self.config.get("rtsp_transport","tcp"))),
            "rtsp_buffer_mode":str(self.core_config.get("rtsp_buffer_mode",self.config.get("rtsp_buffer_mode","auto"))),
            "tcp_timestamp":bool(self.core_config.get("tcp_timestamp",True)),
            "postdecode_queue_buffers":int(self.core_config.get("postdecode_queue_buffers",1)),
        }
        return GStreamerCapture(cfg)
    def _run(self):
        while not self._stop.is_set():
            cap=None
            try:
                cap=self._open();self._capture=cap
                if not cap.isOpened():raise RuntimeError("GStreamer pipeline did not open")
                opened_at=time.monotonic();had_frame=False;timeouts=0;lag_samples=0
                with self._lock:
                    self._metrics.online=False;self._metrics.startup_waiting=True;self._metrics.last_error="";self._metrics.consecutive_timeouts=0;self._metrics.consecutive_lag_samples=0
                    self._metrics.source_runtime=cap.source_runtime() if hasattr(cap,"source_runtime") else {}
                while not self._stop.is_set():
                    ok,image=cap.read()
                    if not ok or image is None:
                        if self._stop.is_set():break
                        with self._lock:
                            self._metrics.read_failures+=1;self._metrics.consecutive_timeouts+=1
                        if not cap.isOpened():
                            detail=cap.last_error() if hasattr(cap,"last_error") else ""
                            raise RuntimeError(f"GStreamer pipeline error before sample: {detail}")
                        timeouts+=1
                        if not had_frame and time.monotonic()-opened_at < self.startup_grace_sec:continue
                        if had_frame and timeouts < self.max_read_timeouts:continue
                        reason="startup grace expired without first frame" if not had_frame else f"{timeouts} consecutive read timeouts"
                        raise RuntimeError(reason)

                    had_frame=True;timeouts=0
                    now=time.time();mono=time.monotonic();h,w=image.shape[:2]
                    pipeline_lag=cap.current_pipeline_lag_ms() if hasattr(cap,"current_pipeline_lag_ms") else None
                    queue_buffers=cap.current_queue_buffers() if hasattr(cap,"current_queue_buffers") else None

                    if self.max_pipeline_lag_ms>0 and pipeline_lag is not None and float(pipeline_lag)>self.max_pipeline_lag_ms:
                        lag_samples+=1
                    else:
                        lag_samples=0

                    with self._lock:
                        self._metrics.online=True;self._metrics.startup_waiting=False;self._metrics.consecutive_timeouts=0;self._metrics.last_error="";self._metrics.pipeline_lag_ms=pipeline_lag;self._metrics.postdecode_queue_buffers=queue_buffers;self._metrics.consecutive_lag_samples=lag_samples
                        self._metrics.frame_id+=1;frame_id=self._metrics.frame_id;self._metrics.width=w;self._metrics.height=h
                        self._fps_frames+=1;elapsed=mono-self._fps_started
                        if elapsed>=1.0:
                            self._metrics.source_fps=self._fps_frames/elapsed;self._fps_frames=0;self._fps_started=mono
                        self._last_frame_mono=mono

                    # Publish the newest sample immediately. If the stream has
                    # drifted, this frame is still the freshest one available;
                    # then rebuild the source so subsequent frames return to live.
                    self.store.put(Frame(self.camera_id,frame_id,now,mono,image,w,h))

                    if self.max_pipeline_lag_ms>0 and lag_samples>=self.max_pipeline_lag_samples:
                        with self._lock:self._metrics.drift_reconnects+=1
                        raise RuntimeError(f"pipeline lag watchdog: {pipeline_lag:.1f}ms for {lag_samples} samples")
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
