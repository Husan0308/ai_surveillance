from __future__ import annotations

from dataclasses import asdict, dataclass
import threading
import time

from services.ml_service.app.deepstream.capture import DeepStreamCapture
from services.ml_service.app.latest_frame import Frame, LatestFrameStore


@dataclass
class CameraMetrics:
    online: bool = False
    frame_id: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    reconnects: int = 0
    read_timeouts: int = 0
    last_frame_age_ms: float | None = None
    last_error: str = ""
    queue_buffers: int | None = None
    transport: str = ""
    backend: str = ""


class CameraWorker:
    def __init__(self, camera, ds_config, store: LatestFrameStore) -> None:
        self.camera = camera
        self.ds = ds_config
        self.store = store
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: DeepStreamCapture | None = None
        self._lock = threading.Lock()
        self._metrics = CameraMetrics()
        self._last_frame_mono = 0.0
        self._fps_started = time.monotonic()
        self._fps_frames = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-{self.camera.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        cap = self._capture
        if cap is not None:
            try:
                cap.close()
            except Exception:
                pass

    def join(self, timeout: float = 3.0) -> None:
        if self._thread:
            self._thread.join(timeout)

    def metrics(self) -> dict:
        with self._lock:
            data = asdict(self._metrics)
            if self._last_frame_mono:
                data["last_frame_age_ms"] = max(
                    0.0, (time.monotonic() - self._last_frame_mono) * 1000.0
                )
            data["replaced_frames"] = self.store.replaced
            return data

    def _run(self) -> None:
        backoff = max(0.5, self.ds.reconnect_delay_sec)
        transport = self.ds.rtsp_transport

        while not self._stop.is_set():
            cap = None
            try:
                print(
                    f"[CAMERA] {self.camera.camera_id} opening {self.camera.uri} "
                    f"codec={self.camera.codec} transport={transport}",
                    flush=True,
                )
                cap = DeepStreamCapture(
                    self.camera.camera_id,
                    self.camera.uri,
                    self.camera.codec,
                    self.ds,
                    transport=transport,
                )
                self._capture = cap
                opened_at = time.monotonic()
                had_frame = False
                consecutive_timeouts = 0

                with self._lock:
                    self._metrics.backend = cap.backend
                    self._metrics.transport = transport

                while not self._stop.is_set():
                    ok, image = cap.read()
                    if not ok or image is None:
                        if self._stop.is_set():
                            break
                        consecutive_timeouts += 1
                        with self._lock:
                            self._metrics.read_timeouts += 1

                        if not had_frame and time.monotonic() - opened_at < self.ds.startup_grace_sec:
                            continue
                        if had_frame and consecutive_timeouts < 3:
                            continue
                        if not cap.is_opened():
                            raise RuntimeError(cap.last_error() or "capture closed")
                        raise RuntimeError(
                            "startup grace expired without first frame"
                            if not had_frame
                            else f"{consecutive_timeouts} consecutive frame timeouts"
                        )

                    had_frame = True
                    consecutive_timeouts = 0
                    backoff = max(0.5, self.ds.reconnect_delay_sec)
                    mono = time.monotonic()
                    height, width = image.shape[:2]

                    with self._lock:
                        self._metrics.online = True
                        self._metrics.last_error = ""
                        self._metrics.frame_id += 1
                        frame_id = self._metrics.frame_id
                        self._metrics.width = width
                        self._metrics.height = height
                        self._metrics.queue_buffers = cap.current_queue_buffers()
                        self._last_frame_mono = mono
                        self._fps_frames += 1
                        elapsed = mono - self._fps_started
                        if elapsed >= 1.0:
                            self._metrics.fps = self._fps_frames / elapsed
                            self._fps_frames = 0
                            self._fps_started = mono

                    self.store.put(Frame(self.camera.camera_id, frame_id, mono, image, width, height))
                    if frame_id == 1:
                        print(
                            f"[CAMERA] {self.camera.camera_id} first frame {width}x{height} "
                            f"backend={cap.backend} transport={transport}",
                            flush=True,
                        )

            except Exception as exc:
                with self._lock:
                    self._metrics.online = False
                    self._metrics.reconnects += 1
                    self._metrics.last_error = f"{type(exc).__name__}: {exc}"

                debug = {}
                if cap is not None:
                    try:
                        debug = cap.debug_info()
                    except Exception:
                        pass

                print(
                    f"[CAMERA] {self.camera.camera_id} reconnect in {backoff:.1f}s: "
                    f"{exc} debug={debug}",
                    flush=True,
                )
                if not self._stop.is_set():
                    self._stop.wait(backoff)
                backoff = min(
                    self.ds.reconnect_delay_max_sec,
                    max(self.ds.reconnect_delay_sec, backoff * 2.0),
                )
            finally:
                if cap is not None:
                    try:
                        cap.close()
                    except Exception:
                        pass
                self._capture = None

        with self._lock:
            self._metrics.online = False
