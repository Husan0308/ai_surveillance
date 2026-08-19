from __future__ import annotations

import time

import cv2

from services.ml_service.app.jpeg_publisher import LatestJpegPublisher


class OnDemandJpegPublisher(LatestJpegPublisher):
    """Encode JPEG only when a /video client actually asks for a frame.

    The mmap camera wall therefore pays zero JPEG encode cost in the normal hot
    path. This object exists only to preserve the existing diagnostic/fallback
    MJPEG endpoint and smoke tests.
    """

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def join(self, timeout: float = 3.0) -> None:
        return

    def wait_newer(self, last_version: int, timeout: float = 1.0):
        frame, store_version = self.store.wait_newer(last_version, timeout=timeout)
        if frame is None or store_version <= last_version:
            return None, last_version

        started = time.perf_counter()
        image = self._image_for_encode(frame)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        encode_ms = (time.perf_counter() - started) * 1000.0
        if not ok:
            return None, last_version

        payload = encoded.tobytes()
        with self._lock:
            self._encoded += 1
            self._version = int(store_version)
            self._last_encode_ms = encode_ms
        return payload, int(store_version)

    def metrics(self) -> dict:
        payload = super().metrics()
        payload["mode"] = "on-demand-fallback"
        return payload
