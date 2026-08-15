from __future__ import annotations

import threading
import time

import cv2

try:
    cv2.setNumThreads(1)
    cv2.setUseOptimized(True)
except Exception:
    pass


class CameraOnlyJpegPublisher:
    """Latest-only camera publisher for the DeepStream baseline.

    This class deliberately has no detector, tracker, ReID or face dependency.
    It waits for new camera frames, coalesces bursts to the newest frame, and
    performs only JPEG encoding for the existing MJPEG frontend.
    """

    def __init__(self, camera_id, store, fps=20, quality=70):
        self.camera_id = str(camera_id)
        self.store = store
        self.interval = 1.0 / max(1.0, float(fps))
        self.quality = int(quality)

        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        self._jpeg = None
        self._version = 0
        self._published_monotonic = 0.0
        self._source_frame_id = 0
        self._started_monotonic = time.monotonic()
        self._last_jpeg_ms = 0.0
        self._last_publish_source_age_ms = None

        self.encoded = 0
        self.event_wakeups = 0
        self.coalesced_frames = 0
        self.skipped_same_frame = 0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-only-jpeg-{self.camera_id}",
            daemon=False,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()

    def join(self, timeout=3):
        if self._thread:
            self._thread.join(timeout)
        return not self._thread or not self._thread.is_alive()

    def wait_newer(self, last_version: int, timeout: float = 0.25):
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while self._version <= last_version and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return (
                self._jpeg,
                self._version,
                self._published_monotonic,
                self._source_frame_id,
            )

    def metrics(self):
        now = time.monotonic()
        with self._lock:
            elapsed = max(0.001, now - self._started_monotonic)
            return {
                "publisher_mode": "camera-only-event-driven-latest-only",
                "encoded": self.encoded,
                "publish_rate": self.encoded / elapsed,
                "event_wakeups": self.event_wakeups,
                "coalesced_frames": self.coalesced_frames,
                "skipped_same_frame": self.skipped_same_frame,
                "last_jpeg_ms": self._last_jpeg_ms,
                "frame_budget_ms": self.interval * 1000.0,
                "last_publish_source_age_ms": self._last_publish_source_age_ms,
                "last_published_age_ms": (
                    (now - self._published_monotonic) * 1000.0
                    if self._published_monotonic
                    else None
                ),
                "source_frame_id": self._source_frame_id,
                "ai_enabled": False,
            }

    def track_snapshot(self):
        return []

    def _run(self):
        last_version = 0
        last_frame_id = -1
        next_allowed = 0.0

        while not self._stop.is_set():
            frame, version = self.store.wait_newer(last_version, timeout=0.5)
            if frame is None:
                continue

            event_mono = time.monotonic()
            with self._lock:
                self.event_wakeups += 1

            if next_allowed > event_mono:
                if self._stop.wait(next_allowed - event_mono):
                    break
                latest, latest_version = self.store.get()
                if latest is not None and latest_version > version:
                    with self._lock:
                        self.coalesced_frames += latest_version - version
                    frame, version = latest, latest_version

            last_version = version
            if frame.frame_id == last_frame_id:
                with self._lock:
                    self.skipped_same_frame += 1
                continue

            next_allowed = time.monotonic() + self.interval
            jpeg_started = time.perf_counter()
            ok, encoded = cv2.imencode(
                ".jpg",
                frame.image,
                [cv2.IMWRITE_JPEG_QUALITY, self.quality],
            )
            jpeg_ms = (time.perf_counter() - jpeg_started) * 1000.0
            if not ok:
                continue

            published = time.monotonic()
            payload = encoded.tobytes()

            with self._condition:
                self._jpeg = payload
                self._version += 1
                self._published_monotonic = published
                self._source_frame_id = frame.frame_id
                self._last_jpeg_ms = jpeg_ms
                self._last_publish_source_age_ms = max(
                    0.0,
                    (published - float(frame.captured_monotonic)) * 1000.0,
                )
                self.encoded += 1
                self._condition.notify_all()

            last_frame_id = frame.frame_id
