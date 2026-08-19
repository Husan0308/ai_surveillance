from __future__ import annotations

import time

import cv2

from shared.safe_mmap_frame import SigbusSafeMmapFrameWriter
from services.ml_service.app.jpeg_publisher import LatestJpegPublisher
from services.ml_service.app.presentation_smoother import PresentationSmoother


class MmapFramePublisher(LatestJpegPublisher):
    """Event-driven latest-only local presentation transport.

    Video is the proven SIGBUS-safe mmap path. ByteTrack owns identity; an
    independent presentation smoother only predicts/interpolates each existing
    T-ID between sparse detector updates so boxes move at the camera cadence.
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
        # Reuse the publisher lifecycle/condition primitives, but do not use its
        # raw snapshot painter. mmap has its own T-ID-preserving smooth overlay.
        super().__init__(
            camera_id,
            store,
            fps=fps,
            quality=88,
            detections=None,
            overlay_enabled=False,
            overlay_max_age_ms=overlay_max_age_ms,
        )
        self.overlay_store = detections
        self.presentation_overlay_enabled = bool(overlay_enabled)
        self.presentation_overlay_max_age_sec = max(
            0.0, float(overlay_max_age_ms) / 1000.0
        )
        self.smoother = PresentationSmoother(
            hold_ms=850,
            memory_ms=2800,
            prediction_ms=340,
            velocity_damping=0.95,
            size_velocity_damping=0.60,
            max_prediction_shift_boxes=0.55,
            max_prediction_size_ratio=0.06,
            adaptive_error_low=0.08,
            adaptive_error_high=0.25,
            center_response_slow=0.42,
            center_response_fast=0.84,
            size_response=0.30,
            snap_distance_boxes=0.62,
            reversal_damping=0.15,
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
                "overlay": "bytetrack-id-presentation-smoother",
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

    @staticmethod
    def _draw_track(image, track) -> None:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = track.xyxy
        left = max(0, min(width - 1, int(round(x1))))
        top = max(0, min(height - 1, int(round(y1))))
        right = max(0, min(width - 1, int(round(x2))))
        bottom = max(0, min(height - 1, int(round(y2))))
        if right <= left or bottom <= top:
            return

        # Clear old-style operator overlay: anti-aliased amber box and dark
        # readable label. It remains visible over IR and bright colour cameras.
        color = (0, 210, 255)
        thickness = 3 if max(width, height) >= 900 else 2
        cv2.rectangle(image, (left, top), (right, bottom), color, thickness, cv2.LINE_AA)

        label = f"Person T{int(track.track_id)}  {float(track.confidence):.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48 if width >= 900 else 0.42
        text_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        pad_x, pad_y = 6, 4
        label_h = th + baseline + pad_y * 2
        label_top = max(0, top - label_h)
        label_right = min(width - 1, left + tw + pad_x * 2)
        cv2.rectangle(
            image,
            (left, label_top),
            (label_right, top),
            (7, 13, 21),
            -1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            image,
            (left, label_top),
            (label_right, top),
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (left + pad_x, max(th + 1, top - baseline - pad_y)),
            font,
            font_scale,
            (245, 248, 252),
            text_thickness,
            cv2.LINE_AA,
        )

    def _presentation_image(self, frame):
        if not self.presentation_overlay_enabled or self.overlay_store is None:
            return frame.image

        snapshot = self.overlay_store.get(self.camera_id)
        if snapshot is not None:
            observation = float(getattr(snapshot, "captured_monotonic", 0.0))
            delta = float(frame.captured_monotonic) - observation
            # Do not paint a result onto an older video frame. A recent result is
            # accepted once the presentation clock catches it; the smoother then
            # bridges detector gaps using the authoritative T-ID.
            if -0.05 <= delta <= self.presentation_overlay_max_age_sec:
                self.smoother.update(snapshot)

        tracks = self.smoother.visible(float(frame.captured_monotonic))
        if not tracks:
            return frame.image

        image = frame.image.copy()
        for track in tracks:
            self._draw_track(image, track)
        return image

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
                image = self._presentation_image(frame)
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
                        f"{image.shape[1]}x{image.shape[0]} bytes={int(packet['payload_bytes'])} "
                        "overlay=bytetrack-smooth",
                        flush=True,
                    )
        finally:
            self.writer.close(unlink=True)
