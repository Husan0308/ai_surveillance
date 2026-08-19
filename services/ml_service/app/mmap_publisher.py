from __future__ import annotations

import time

import cv2

from shared.safe_mmap_frame import SigbusSafeMmapFrameWriter
from services.ml_service.app.jpeg_publisher import LatestJpegPublisher


class MmapFramePublisher(LatestJpegPublisher):
    """Event-driven latest-only local presentation transport.

    This is the proven camera wall path from the old sigbus-safe-mmap branch:
    no JPEG encode, no HTTP video relay, no presentation queue. The newest
    960x540 BGR frame (with the current local-track overlay) is atomically
    published to a double-buffered mmap file for the Qt frontend.
    """

    def __init__(
        self,
        camera_id,
        store,
        fps,
        max_width,
        max_height,
        detections=None,
        overlay_enabled=True,
        overlay_max_age_ms=900,
    ) -> None:
        super().__init__(
            camera_id,
            store,
            fps=fps,
            quality=88,
            detections=detections,
            overlay_enabled=overlay_enabled,
            overlay_max_age_ms=overlay_max_age_ms,
        )
        self.max_width = max(1, int(max_width))
        self.max_height = max(1, int(max_height))
        self.writer = SigbusSafeMmapFrameWriter(
            self.camera_id,
            self.max_width,
            self.max_height,
            channels=3,
        )
        self.event_wakeups = 0
        self.coalesced_frames = 0
        self._last_transport_ms = 0.0
        self._last_resize_ms = 0.0
        self._last_payload_bytes = 0
        self._last_sequence = 0
        self._last_source_frame_id = 0
        self._last_publish_source_age_ms: float | None = None

    def metrics(self) -> dict:
        with self._lock:
            return {
                "published": self._encoded,
                "version": self._version,
                "publisher_mode": "event-driven-mmap-latest-only",
                "transport": "mmap-bgr-double-buffer-sigbus-safe",
                "mmap_path": str(self.writer.path),
                "event_wakeups": self.event_wakeups,
                "coalesced_frames": self.coalesced_frames,
                "last_transport_ms": self._last_transport_ms,
                "last_resize_ms": self._last_resize_ms,
                "last_payload_bytes": self._last_payload_bytes,
                "last_sequence": self._last_sequence,
                "source_frame_id": self._last_source_frame_id,
                "last_publish_source_age_ms": self._last_publish_source_age_ms,
                "width": self.max_width,
                "height": self.max_height,
            }

    def _run(self) -> None:
        last_store_version = 0
        last_frame_id = -1
        next_allowed = 0.0
        try:
            while not self._stop.is_set():
                frame, store_version = self.store.wait_newer(last_store_version, timeout=0.5)
                if frame is None:
                    continue

                event_mono = time.monotonic()
                with self._lock:
                    self.event_wakeups += 1

                if next_allowed > event_mono:
                    if self._stop.wait(next_allowed - event_mono):
                        break
                    latest, latest_version = self.store.get()
                    if latest is not None and latest_version > store_version:
                        with self._lock:
                            self.coalesced_frames += latest_version - store_version
                        frame, store_version = latest, latest_version

                last_store_version = store_version
                if int(frame.frame_id) == last_frame_id:
                    continue

                next_allowed = time.monotonic() + self.interval
                image = self._image_for_encode(frame)
                source_h, source_w = image.shape[:2]

                resize_started = time.perf_counter()
                scale = min(
                    1.0,
                    self.max_width / max(1, source_w),
                    self.max_height / max(1, source_h),
                )
                if scale < 1.0:
                    image = cv2.resize(
                        image,
                        (max(1, round(source_w * scale)), max(1, round(source_h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                elif image is frame.image:
                    # Writer needs an owned contiguous image because the camera
                    # store may replace its reference immediately after publish.
                    image = image.copy()
                resize_ms = (time.perf_counter() - resize_started) * 1000.0

                transport_started = time.perf_counter()
                packet = self.writer.write(image, frame.frame_id, frame.captured_monotonic)
                transport_ms = (time.perf_counter() - transport_started) * 1000.0
                published = packet["published_monotonic_ns"] / 1_000_000_000.0

                with self._condition:
                    self._version = int(packet["sequence"])
                    self._encoded += 1
                    self._last_transport_ms = transport_ms
                    self._last_resize_ms = resize_ms
                    self._last_payload_bytes = int(packet["payload_bytes"])
                    self._last_sequence = int(packet["sequence"])
                    self._last_source_frame_id = int(frame.frame_id)
                    self._last_publish_source_age_ms = max(
                        0.0,
                        (published - float(frame.captured_monotonic)) * 1000.0,
                    )
                    self._condition.notify_all()

                last_frame_id = int(frame.frame_id)
                if self._encoded == 1:
                    print(
                        f"[MMAP] {self.camera_id} first frame "
                        f"{image.shape[1]}x{image.shape[0]} bytes={int(packet['payload_bytes'])}",
                        flush=True,
                    )
        finally:
            self.writer.close(unlink=True)
