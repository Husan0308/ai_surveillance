from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum

from services.ml_service.app.camera_worker import CameraWorker
from services.ml_service.app.config import Settings
from services.ml_service.app.jpeg_publisher import LatestJpegPublisher
from services.ml_service.app.latest_frame import LatestFrameStore


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
    """Six independent DeepStream camera pipelines plus latest-only MJPEG output."""

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
        self.publishers = {
            camera.camera_id: LatestJpegPublisher(
                camera.camera_id,
                self.stores[camera.camera_id],
                fps=settings.deepstream.display_fps,
                quality=settings.deepstream.jpeg_quality,
            )
            for camera in settings.cameras
        }

    def start(self) -> None:
        with self._lock:
            if self._state in {RuntimeState.STARTING, RuntimeState.PLAYING}:
                return
            self._state = RuntimeState.STARTING
            self._last_error = None
        for publisher in self.publishers.values():
            publisher.start()
        for index, camera in enumerate(self.settings.cameras):
            self.workers[camera.camera_id].start()
            if index + 1 < len(self.settings.cameras):
                time.sleep(self.settings.deepstream.startup_stagger_sec)
        with self._lock:
            self._state = RuntimeState.PLAYING

    def stop(self) -> None:
        for worker in self.workers.values():
            worker.stop()
        for worker in self.workers.values():
            worker.join()
        for publisher in self.publishers.values():
            publisher.stop()
        for publisher in self.publishers.values():
            publisher.join()
        with self._lock:
            self._state = RuntimeState.STOPPED

    def snapshot(self) -> RuntimeSnapshot:
        camera_rows = self.camera_metrics()
        online = sum(1 for row in camera_rows if row["online"])
        errors = [f'{row["id"]}: {row["last_error"]}' for row in camera_rows if row["last_error"]]
        with self._lock:
            state = self._state
        return RuntimeSnapshot(
            state=state,
            camera_count=len(camera_rows),
            online_camera_count=online,
            last_error=" | ".join(errors) if errors else None,
        )

    def camera_metrics(self) -> list[dict]:
        rows = []
        for camera in self.settings.cameras:
            metrics = self.workers[camera.camera_id].metrics()
            publisher = self.publishers[camera.camera_id].metrics()
            rows.append({"id": camera.camera_id, **metrics, "jpeg": publisher})
        return rows

    def has_camera(self, camera_id: str) -> bool:
        return camera_id in self.publishers

    def wait_jpeg(self, camera_id: str, last_version: int, timeout: float = 1.0):
        return self.publishers[camera_id].wait_newer(last_version, timeout)
