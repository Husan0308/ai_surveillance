from __future__ import annotations

import threading
import time

import cv2

from services.ml_service.app.latest_frame import LatestFrameStore

try:
    cv2.setNumThreads(1)
    cv2.setUseOptimized(True)
except Exception:
    pass


class LatestJpegPublisher:
    """One event-driven latest JPEG per camera; no presentation backlog."""

    def __init__(self, camera_id: str, store: LatestFrameStore, fps: int, quality: int) -> None:
        self.camera_id = camera_id
        self.store = store
        self.interval = 1.0 / max(1, int(fps))
        self.quality = int(quality)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._jpeg: bytes | None = None
        self._version = 0
        self._encoded = 0
        self._last_encode_ms = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"jpeg-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()

    def join(self, timeout: float = 3.0) -> None:
        if self._thread:
            self._thread.join(timeout)

    def wait_newer(self, last_version: int, timeout: float = 1.0):
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while self._version <= last_version and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return self._jpeg, self._version

    def metrics(self) -> dict:
        with self._lock:
            return {"encoded": self._encoded, "version": self._version, "last_encode_ms": self._last_encode_ms}

    def _run(self) -> None:
        last_store_version = 0
        next_allowed = 0.0
        while not self._stop.is_set():
            frame, store_version = self.store.wait_newer(last_store_version, timeout=0.5)
            if frame is None:
                continue
            now = time.monotonic()
            if now < next_allowed:
                if self._stop.wait(next_allowed - now):
                    break
                latest, latest_version = self.store.get()
                if latest is not None and latest_version > store_version:
                    frame = latest
                    store_version = latest_version
            last_store_version = store_version
            next_allowed = time.monotonic() + self.interval
            started = time.perf_counter()
            ok, encoded = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            encode_ms = (time.perf_counter() - started) * 1000.0
            if not ok:
                continue
            payload = encoded.tobytes()
            with self._condition:
                self._jpeg = payload
                self._version += 1
                self._encoded += 1
                self._last_encode_ms = encode_ms
                self._condition.notify_all()
            if self._version == 1:
                print(f"[MJPEG] {self.camera_id} first JPEG {len(payload)} bytes", flush=True)
