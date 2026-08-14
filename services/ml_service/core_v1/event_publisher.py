from __future__ import annotations

import time

import cv2

from .jpeg_publisher import LatestJpegPublisher


class EventDrivenJpegPublisher(LatestJpegPublisher):
    """Publish on camera frame arrival, capped by the configured display FPS.

    The old fixed-rate loop could repeatedly wake between irregular RTSP frame
    arrivals and see the same frame, reducing a ~20 FPS source to 12-17 FPS.
    This publisher sleeps on LatestFrameStore.wait_newer(), then coalesces to the
    newest frame if several arrived during the presentation rate limit.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_wakeups = 0
        self.coalesced_frames = 0
        self._last_event_to_publish_ms = 0.0

    def metrics(self):
        payload = super().metrics()
        with self._lock:
            payload.update(
                {
                    "publisher_mode": "event-driven-latest-only",
                    "event_wakeups": self.event_wakeups,
                    "coalesced_frames": self.coalesced_frames,
                    "last_event_to_publish_ms": self._last_event_to_publish_ms,
                }
            )
        return payload

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

            # Cap presentation FPS without phase-locking to a fixed timer. While
            # waiting for the budget, newer camera frames replace this one.
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

            jpeg_started = time.perf_counter()
            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, self.quality],
            )
            jpeg_ms = (time.perf_counter() - jpeg_started) * 1000.0
            if not ok:
                continue

            published = time.monotonic()
            payload = encoded.tobytes()
            cycle_ms = (time.perf_counter() - cycle_started) * 1000.0
            event_to_publish_ms = max(0.0, (published - event_mono) * 1000.0)

            with self._condition:
                self._jpeg = payload
                self._version += 1
                self._published_monotonic = published
                self._source_frame_id = frame.frame_id
                self._last_source_capture_mono = float(frame.captured_monotonic)
                self._last_resize_ms = resize_ms
                self._last_overlay_ms = overlay_ms
                self._last_jpeg_ms = jpeg_ms
                self._last_encode_ms = cycle_ms
                self._last_cycle_ms = cycle_ms
                self._last_publish_source_age_ms = max(
                    0.0,
                    (published - float(frame.captured_monotonic)) * 1000.0,
                )
                self._last_event_to_publish_ms = event_to_publish_ms
                self.encoded += 1
                self._condition.notify_all()

            last_frame_id = frame.frame_id
