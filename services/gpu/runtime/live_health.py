"""Read-only per-source liveness guard using official deepstream-app PERF lines.

DeepStream 9.1 prints the recent per-source FPS every configured perf interval,
five seconds in our live config. This is post-pipeline progress evidence, NOT
exact frame-index/PTS continuity or proof of display or exposure timestamps.

No RTSP connection and no native/GPU imports are required by this module.
"""
from __future__ import annotations

import re
import threading
import time

CAMERA_IDS = ("CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06")
# deepstream-app uses **PERF: for one application with multiple sources.
# Each source is printed as recent (cumulative) FPS.
_PERF_LINE = re.compile(r"^\s*\*\*PERF:\s*(.*?)\s*$")
_FPS_PAIR = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)")


class SourceFpsMonitor:
    def __init__(self, *, min_fps: float = 5.0, max_age_sec: float = 15.0,
                 min_consecutive: int = 2) -> None:
        if min_fps <= 0 or max_age_sec <= 0 or min_consecutive < 1:
            raise ValueError("invalid monitoring thresholds")
        self.min_fps = min_fps
        self.max_age_sec = max_age_sec
        self.min_consecutive = min_consecutive
        self._lock = threading.Lock()
        self._seen_at: float | None = None
        self._fps: tuple[float, ...] = ()
        self._streak = 0
        self._samples = 0

    def ingest(self, line: str, *, now: float | None = None) -> bool:
        match = _PERF_LINE.match(line)
        if not match:
            return False
        pairs = _FPS_PAIR.findall(match.group(1))
        # Excludes header lines and half-written lines.
        if len(pairs) != len(CAMERA_IDS):
            return False
        stamp = time.monotonic() if now is None else now
        latest = tuple(float(value) for value, _average in pairs)
        with self._lock:
            if self._seen_at is None or stamp - self._seen_at > self.max_age_sec:
                self._streak = 0
            self._fps = latest
            self._seen_at = stamp
            self._samples += 1
            if all(value >= self.min_fps for value in latest):
                self._streak += 1
            else:
                self._streak = 0
        return True

    def snapshot(self, *, now: float | None = None) -> dict:
        stamp = time.monotonic() if now is None else now
        with self._lock:
            age = None if self._seen_at is None else max(0.0, stamp - self._seen_at)
            fresh = age is not None and age <= self.max_age_sec
            ready = fresh and self._streak >= self.min_consecutive
            return {
                "ready": ready,
                "fresh": fresh,
                "age_seconds": age,
                "samples": self._samples,
                "consecutive_good": self._streak,
                "camera_fps": dict(zip(CAMERA_IDS, self._fps)),
                "limited_to": "DeepStream app recent FPS; not exact frame PTS",
            }
