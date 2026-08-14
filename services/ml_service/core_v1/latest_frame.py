from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any


@dataclass(frozen=True, slots=True)
class Frame:
    camera_id: str
    frame_id: int
    captured_at: float
    captured_monotonic: float
    image: Any
    width: int
    height: int


class LatestFrameStore:
    """Exactly one newest decoded frame.

    A new camera frame replaces the previous frame immediately. There is no
    history, replay queue or analytics cache in the detection-only baseline.
    """

    def __init__(self, history_size: int | None = None) -> None:
        # history_size remains accepted only so old CameraManager construction
        # cannot break while local configs are being pulled across machines.
        del history_size
        self._lock = threading.Lock()
        self._frame = None
        self._version = 0
        self.replaced = 0

    def put(self, frame: Frame) -> None:
        with self._lock:
            if self._frame is not None:
                self.replaced += 1
            self._frame = frame
            self._version += 1

    def get(self):
        with self._lock:
            return self._frame, self._version

    def wait_newer(self, last_version: int, timeout: float = 1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame, version = self.get()
            if frame is not None and version > last_version:
                return frame, version
            time.sleep(0.003)
        return None, last_version
