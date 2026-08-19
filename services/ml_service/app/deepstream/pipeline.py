from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum

from services.ml_service.app.camera_worker import CameraWorker
from services.ml_service.app.config import Settings
from services.ml_service.app.detector import PersonDetector
from services.ml_service.app.fallback_jpeg import OnDemandJpegPublisher
from services.ml_service.app.latest_frame import LatestFrameStore
from services.ml_service.app.mmap_publisher import MmapFramePublisher
from services.ml_service.app.tracking import PersonTracker


class RuntimeState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    PLAYING = "playing"
    ERROR = "error"


@dataclass(frozen=True)
class RuntimeSnapshot:
    state: RuntimeState
    camera_count: int
    online_camera_count: int
    last_error: str | None


class DeepStreamRuntime:
    """NVDEC -> latest frame -> YOLO -> ByteTrack -> local mmap presentation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._state = RuntimeState.STOPPED
        self._last_error: str | None = None
        self.stores = {camera.camera_id: LatestFrameStore() for camera in settings.cameras}
        self.workers = {
            camera.camera_id: CameraWorker(camera, settings.deepstream, self.stores[camera.camera_id])
            for camera in settings.cameras
        }
        self.detector = PersonDetector(settings.detection, self.stores)
        self.tracker = PersonTracker(
            settings.tracking,
            self.detector.results,
            [camera.camera_id for camera in settings.cameras],
            frame_width=settings.deepstream.display_width,
            frame_height=settings.deepstream.display_height,
            detector_fps=settings.detection.target_fps_per_camera,
        )
        overlay_store = self.tracker.results if settings.tracking.enabled else self.detector.results

        # Canonical local UI path. This is the old proven sigbus-safe mmap design:
        # no JPEG/HTTP encode cost while the desktop wall is running normally.
        self.mmap_publishers = {
            camera.camera_id: MmapFramePublisher(
                camera.camera_id,
                self.stores[camera.camera_id],
                fps=settings.deepstream.display_fps,
                max_width=settings.deepstream.display_width,
                max_height=settings.deepstream.display_height,
                detections=overlay_store,
                overlay_enabled=settings.detection.overlay,
                overlay_max_age_ms=settings.detection.overlay_max_age_ms,
            )
            for camera in settings.cameras
        }

        # /video remains for diagnostics/fallback only. JPEG work happens only
        # when a client is actually connected, never in the normal mmap hot path.
        self.jpeg_fallbacks = {
            camera.camera_id: OnDemandJpegPublisher(
                camera.camera_id,
                self.stores[camera.camera_id],
                fps=settings.deepstream.display_fps,
                quality=settings.deepstream.jpeg_quality,
                detections=overlay_store,
                overlay_enabled=settings.detection.overlay,
                overlay_max_age_ms=settings.detection.overlay_max_age_ms,
            )
            for camera in settings.cameras
        }

    def start(self) -> None:
        with self._lock:
            if self._state in {RuntimeState.STARTING, RuntimeState.PLAYING}:
                return
            self._state = RuntimeState.STARTING
            self._last_error = None

        for publisher in self.mmap_publishers.values():
            publisher.start()
        for index, camera in enumerate(self.settings.cameras):
            self.workers[camera.camera_id].start()
            if index + 1 < len(self.settings.cameras):
                time.sleep(self.settings.deepstream.startup_stagger_sec)
        self.detector.start()
        self.tracker.start()

        with self._lock:
            self._state = RuntimeState.PLAYING

    def stop(self) -> None:
        self.tracker.stop()
        self.tracker.join()
        self.detector.stop()
        self.detector.join()
        for worker in self.workers.values():
            worker.stop()
        for worker in self.workers.values():
            worker.join()
        for publisher in self.mmap_publishers.values():
            publisher.stop()
        for publisher in self.mmap_publishers.values():
            publisher.join()
        with self._lock:
            self._state = RuntimeState.STOPPED

    def snapshot(self) -> RuntimeSnapshot:
        camera_rows = self.camera_metrics()
        online = sum(1 for row in camera_rows if row["online"])
        errors = [f'{row["id"]}: {row["last_error"]}' for row in camera_rows if row["last_error"]]
        detector_metrics = self.detector.metrics()
        if detector_metrics.get("enabled") and detector_metrics.get("state") == "error":
            errors.append(f'detector: {detector_metrics.get("last_error", "unknown error")}')
        tracker_metrics = self.tracker.metrics()
        if tracker_metrics.get("enabled") and tracker_metrics.get("state") == "error":
            errors.append(f'tracker: {tracker_metrics.get("last_error", "unknown error")}')
        with self._lock:
            state = self._state
        return RuntimeSnapshot(
            state=state,
            camera_count=len(camera_rows),
            online_camera_count=online,
            last_error=" | ".join(errors) if errors else None,
        )

    def detector_metrics(self) -> dict:
        return self.detector.metrics()

    def tracker_metrics(self) -> dict:
        return self.tracker.metrics()

    def camera_metrics(self) -> list[dict]:
        rows = []
        for camera in self.settings.cameras:
            camera_id = camera.camera_id
            metrics = self.workers[camera_id].metrics()
            presentation = self.mmap_publishers[camera_id].metrics()
            jpeg = self.jpeg_fallbacks[camera_id].metrics()
            detection = self.detector.camera_metrics(camera_id)
            tracking = self.tracker.camera_metrics(camera_id)
            people = (
                int(tracking.get("active_tracks", 0))
                if self.settings.tracking.enabled
                else int(detection.get("people", 0))
            )
            rows.append(
                {
                    "id": camera_id,
                    **metrics,
                    "people": people,
                    "detection": detection,
                    "tracking": tracking,
                    "presentation": presentation,
                    "jpeg": jpeg,
                }
            )
        return rows

    def has_camera(self, camera_id: str) -> bool:
        return camera_id in self.stores

    def detection_payload(self, camera_id: str) -> dict:
        if not self.has_camera(camera_id):
            raise KeyError(camera_id)
        return self.detector.snapshot_payload(camera_id)

    def tracking_payload(self, camera_id: str) -> dict:
        if not self.has_camera(camera_id):
            raise KeyError(camera_id)
        return self.tracker.snapshot_payload(camera_id)

    def wait_jpeg(self, camera_id: str, last_version: int, timeout: float = 1.0):
        return self.jpeg_fallbacks[camera_id].wait_newer(last_version, timeout)
