from __future__ import annotations

import threading
import time

from shared.safe_mmap_frame import SigbusSafeMmapFrameWriter


class CameraOnlyMmapPublisher:
    """Latest-only mmap publisher with no detector/tracker/identity work."""

    def __init__(self, camera_id: str, store, fps: float, max_width: int, max_height: int):
        self.camera_id = str(camera_id)
        self.store = store
        self.interval = 1.0 / max(1.0, float(fps))
        self.max_width = max(1, int(max_width))
        self.max_height = max(1, int(max_height))
        self.writer = SigbusSafeMmapFrameWriter(
            self.camera_id,
            self.max_width,
            self.max_height,
            channels=3,
        )

        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._started = time.monotonic()

        self.published = 0
        self.event_wakeups = 0
        self.coalesced_frames = 0
        self.skipped_same_frame = 0
        self.last_transport_ms = 0.0
        self.last_publish_source_age_ms = None
        self.last_payload_bytes = 0
        self.last_sequence = 0
        self.source_frame_id = 0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-only-mmap-{self.camera_id}",
            daemon=False,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=3):
        if self._thread:
            self._thread.join(timeout)
        return not self._thread or not self._thread.is_alive()

    def metrics(self):
        now = time.monotonic()
        with self._lock:
            elapsed = max(0.001, now - self._started)
            return {
                "publisher_mode": "camera-only-event-driven-mmap-latest-only",
                "transport": "mmap-bgr-double-buffer-sigbus-safe",
                "mmap_path": str(self.writer.path),
                "published": self.published,
                "publish_rate": self.published / elapsed,
                "event_wakeups": self.event_wakeups,
                "coalesced_frames": self.coalesced_frames,
                "skipped_same_frame": self.skipped_same_frame,
                "last_transport_ms": self.last_transport_ms,
                "last_publish_source_age_ms": self.last_publish_source_age_ms,
                "last_payload_bytes": self.last_payload_bytes,
                "last_sequence": self.last_sequence,
                "source_frame_id": self.source_frame_id,
                "frame_budget_ms": self.interval * 1000.0,
                "ai_enabled": False,
            }

    def _run(self):
        last_version = 0
        last_frame_id = -1
        next_allowed = 0.0
        try:
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
                image = frame.image
                if image is None or getattr(image, "ndim", 0) != 3:
                    continue

                height, width = image.shape[:2]
                if width > self.max_width or height > self.max_height:
                    raise RuntimeError(
                        f"camera {self.camera_id} produced {width}x{height}, "
                        f"mmap slot is {self.max_width}x{self.max_height}; "
                        "capture_output size must match the publisher"
                    )

                started = time.perf_counter()
                packet = self.writer.write(
                    image,
                    frame.frame_id,
                    frame.captured_monotonic,
                )
                transport_ms = (time.perf_counter() - started) * 1000.0
                published_mono = packet["published_monotonic_ns"] / 1_000_000_000.0

                with self._lock:
                    self.published += 1
                    self.last_transport_ms = transport_ms
                    self.last_publish_source_age_ms = max(
                        0.0,
                        (published_mono - float(frame.captured_monotonic)) * 1000.0,
                    )
                    self.last_payload_bytes = int(packet["payload_bytes"])
                    self.last_sequence = int(packet["sequence"])
                    self.source_frame_id = int(frame.frame_id)

                last_frame_id = frame.frame_id
        finally:
            self.writer.close(unlink=True)
