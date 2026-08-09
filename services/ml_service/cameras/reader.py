"""Independent reconnecting capture worker for one configured camera."""
from __future__ import annotations
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable
from shared.logging import get_logger
from .buffer import LatestFrameBuffer
from .frame import FramePacket

log = get_logger(__name__)

@dataclass
class ReaderMetrics:
    recv_frame_id: int = 0
    source_fps: float = 0.0
    interarrival_ms: float = 0.0
    max_interarrival_ms: float = 0.0
    last_frame_timestamp: float = 0.0
    reconnect_count: int = 0
    starved_count: int = 0
    online: bool = False
    backend: str = "offline"
    width: int = 0
    height: int = 0
    frame_mean: float = 0.0
    frame_variance: float = 0.0
    last_decode_timestamp: float = 0.0

class CameraReader:
    def __init__(self, config: dict, buffer: LatestFrameBuffer,
                 capture_factory: Callable[[dict], Any] | None = None, on_frame=None) -> None:
        self.config, self.buffer = dict(config), buffer
        self.camera_id = str(config["id"])
        self._factory = capture_factory or self._open_configured
        self._on_frame = on_frame
        self._stop, self._lock = threading.Event(), threading.RLock()
        self._thread, self._capture = None, None
        self._metrics = ReaderMetrics()
        self._last_receive = self._fps_started = 0.0
        self._fps_frames = 0

    @staticmethod
    def _source(config):
        from .gstreamer import authenticated_source
        return authenticated_source(config)

    def _open_configured(self, config):
        from shared.config import project_config
        deepstream = project_config().get("deepstream", {})
        if bool(deepstream.get("enabled", False)):
            from .gstreamer import GStreamerCapture
            configured={**config,"decoder_backend":deepstream.get("decoder_backend","nvv4l2decoder")}
            capture = GStreamerCapture(configured)
            if capture.isOpened() or not bool(deepstream.get("fallback_to_opencv", False)): return capture
            capture.release(); log.warning("%s NVDEC open failed; explicitly falling back to FFmpeg", self.camera_id)
        return self._open_opencv(config)

    def _open_opencv(self, config):
        import cv2
        source = self._source(config)
        cap = cv2.VideoCapture(source, cv2.CAP_ANY if isinstance(source, int) else cv2.CAP_FFMPEG)
        try: cap.backend = "ffmpeg" if not isinstance(source, int) else "opencv"
        except Exception: pass
        timeout = int(float(config.get("connection_timeout", 5)) * 1000)
        for prop, value in ((getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", 53), timeout),
                            (getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", 54), timeout),
                            (cv2.CAP_PROP_BUFFERSIZE, 1)):
            try: cap.set(prop, value)
            except Exception: pass
        return cap

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive(): return
            self._stop.clear()
            self._thread = threading.Thread(target=self.run, name=f"camera-{self.camera_id}", daemon=False)
            self._thread.start()

    def run(self):
        delay = max(.1, float(self.config.get("reconnect_interval", 5)))
        startup_grace=max(2,int(float(self.config.get("connection_timeout",5))*2))
        fail_limit = max(1, int(self.config.get("read_fail_limit", startup_grace)))
        while not self._stop.is_set():
            cap = None
            try:
                cap = self._factory(self.config)
                with self._lock: self._capture = cap
                if cap is None or not cap.isOpened():
                    self._disconnect(True); self._stop.wait(delay); continue
                with self._lock:
                    self._metrics.online = True; self._metrics.backend = getattr(cap, "backend", "ffmpeg")
                failures = 0
                while not self._stop.is_set():
                    ok, frame = cap.read(); received = time.time()
                    if not ok or frame is None:
                        failures += 1
                        if failures >= fail_limit: self._disconnect(True); break
                        self._stop.wait(.05); continue
                    failures = 0; self._accept(frame, received)
            except Exception as exc:
                log.warning("%s capture error: %s", self.camera_id, exc); self._disconnect(True)
            finally:
                if cap is not None:
                    try: cap.release()
                    except Exception: pass
                with self._lock:
                    if self._capture is cap: self._capture = None
            if not self._stop.is_set(): self._stop.wait(delay)
        with self._lock: self._metrics.online = False

    def _accept(self, frame, received):
        height, width = frame.shape[:2]
        with self._lock:
            m = self._metrics
            interarrival = (received - self._last_receive) * 1000 if self._last_receive else 0.0
            m.recv_frame_id += 1; m.interarrival_ms = interarrival
            m.max_interarrival_ms = max(m.max_interarrival_ms, interarrival)
            m.last_frame_timestamp = received; self._last_receive = received
            m.last_decode_timestamp = received; m.width = width; m.height = height
            if hasattr(frame, "__getitem__") and hasattr(frame, "mean"):
                sample = frame[::max(1, height // 90), ::max(1, width // 160)]
                m.frame_mean = float(sample.mean()); m.frame_variance = float(sample.var())
            if not self._fps_started: self._fps_started = received
            self._fps_frames += 1; elapsed = received - self._fps_started
            if elapsed >= 1:
                m.source_fps = self._fps_frames / elapsed; self._fps_frames = 0; self._fps_started = received
            frame_id = m.recv_frame_id
        packet=FramePacket(self.camera_id, frame_id, received, received, frame, width, height)
        self.buffer.put(packet)
        if self._on_frame is not None:self._on_frame(packet)

    def _disconnect(self, reconnect):
        with self._lock:
            was_online = self._metrics.online; self._metrics.online = False; self._metrics.backend = "offline"
            if reconnect: self._metrics.reconnect_count += 1
            if was_online or self._metrics.recv_frame_id: self._metrics.starved_count += 1

    def stop(self):
        self._stop.set()
        with self._lock: cap = self._capture
        if cap is not None:
            try:
                interrupt=getattr(cap,"interrupt",None)
                if interrupt:interrupt()
                else:cap.release()
            except Exception: pass

    def join(self, timeout=None):
        if self._thread: self._thread.join(timeout)
        return not self._thread or not self._thread.is_alive()

    def metrics(self):
        with self._lock: result = asdict(self._metrics)
        result["dropped_old"] = self.buffer.dropped_old
        return result
