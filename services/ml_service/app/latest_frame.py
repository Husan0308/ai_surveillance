from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any


@dataclass(frozen=True, slots=True)
class Frame:
    camera_id: str
    frame_id: int
    captured_monotonic: float
    image: Any
    width: int
    height: int


class LatestFrameStore:
    """Single-slot latest-frame store. A newer frame replaces the old one."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._frame: Frame | None = None
        self._version = 0
        self.replaced = 0

    def put(self, frame: Frame) -> None:
        with self._condition:
            if self._frame is not None:
                self.replaced += 1
            self._frame = frame
            self._version += 1
            self._condition.notify_all()

    def get(self) -> tuple[Frame | None, int]:
        with self._lock:
            return self._frame, self._version

    def wait_newer(self, last_version: int, timeout: float = 1.0) -> tuple[Frame | None, int]:
        with self._condition:
            if self._frame is not None and self._version > last_version:
                return self._frame, self._version
            self._condition.wait_for(
                lambda: self._frame is not None and self._version > last_version,
                timeout=max(0.0, float(timeout)),
            )
            if self._frame is not None and self._version > last_version:
                return self._frame, self._version
            return None, last_version
