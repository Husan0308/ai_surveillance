from __future__ import annotations

import time

import cv2

from shared.safe_mmap_frame import SigbusSafeMmapFrameWriter

from .tracking_publisher import TrackingJpegPublisher


class TrackingMmapPublisher(TrackingJpegPublisher):
    """Ownership-locked tracker publisher with decode-free local mmap transport.

    It preserves TrackingJpegPublisher's ownership-locked ByteTrack/Hungarian
    tracker, identity provider and track_snapshot contract used by ReID/Face, but
    replaces per-frame JPEG encode + HTTP delivery with a latest-only BGR mmap.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = SigbusSafeMmapFrameWriter(
            self.camera_id,
            self.max_width,
            self.max_height,
            channels=3,
        )
        self.event_wakeups = 0
        self.coalesced_frames = 0
        self._last_transport_ms = 0.0
        self._last_payload_bytes = 0
        self._last_sequence = 0

    def metrics(self):
        payload = super().metrics()
        with self._lock:
            payload.update(
                {
                    "publisher_mode": "ownership-tracking-mmap-latest-only",
                    "transport": "mmap-bgr-double-buffer-sigbus-safe",
                    "mmap_path": str(self.writer.path),
                    "event_wakeups": self.event_wakeups,
                    "coalesced_frames": self.coalesced_frames,
                    "last_transport_ms": self._last_transport_ms,
                    "last_payload_bytes": self._last_payload_bytes,
                    "last_sequence": self._last_sequence,
                }
            )
        return payload

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

                # Presentation is capped but never queued: if a newer frame
                # arrives while waiting for the display interval, replace the old
                # frame before any resize/overlay/copy work is performed.
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

                cycle_gate = time.monotonic()
                next_allowed = cycle_gate + self.interval
                cycle_started = time.perf_counter()
                image = frame.image
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
                else:
                    image = image.copy()
                resize_ms = (time.perf_counter() - resize_started) * 1000.0

                overlay_started = time.perf_counter()
                image = self._draw_detection(
                    image,
                    source_w,
                    source_h,
                    cycle_gate,
                    frame.frame_id,
                    frame.captured_monotonic,
                )
                overlay_ms = (time.perf_counter() - overlay_started) * 1000.0

                transport_started = time.perf_counter()
                packet = self.writer.write(
                    image,
                    frame.frame_id,
                    frame.captured_monotonic,
                )
                transport_ms = (time.perf_counter() - transport_started) * 1000.0
                published = packet["published_monotonic_ns"] / 1_000_000_000.0
                cycle_ms = (time.perf_counter() - cycle_started) * 1000.0

                with self._lock:
                    self._published_monotonic = published
                    self._source_frame_id = frame.frame_id
                    self._last_source_capture_mono = float(frame.captured_monotonic)
                    self._last_resize_ms = resize_ms
                    self._last_overlay_ms = overlay_ms
                    self._last_jpeg_ms = 0.0
                    self._last_encode_ms = cycle_ms
                    self._last_cycle_ms = cycle_ms
                    self._last_publish_source_age_ms = max(
                        0.0,
                        (published - float(frame.captured_monotonic)) * 1000.0,
                    )
                    self._last_transport_ms = transport_ms
                    self._last_payload_bytes = int(packet["payload_bytes"])
                    self._last_sequence = int(packet["sequence"])
                    # Keep historical metric semantics: published presentation
                    # frames, even though there is no JPEG encode in this mode.
                    self.encoded += 1

                last_frame_id = frame.frame_id
        finally:
            self.writer.close(unlink=True)
