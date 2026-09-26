"""Low-overhead stage metrics for the room-pair identity runtime."""
from __future__ import annotations

import math
import statistics
import threading
import time
from collections import Counter, defaultdict
from typing import Iterable


def monotonic_ns() -> int:
    return time.monotonic_ns()


def wall_timestamp_ns() -> int:
    return time.time_ns()


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(math.ceil(quantile * len(ordered))) - 1)
    return float(ordered[index])


class IdentityMetrics:
    """Thread-safe counters and bounded samples for a live identity run."""

    def __init__(self, sample_limit: int = 20000) -> None:
        self._lock = threading.Lock()
        self.sample_limit = max(1000, int(sample_limit))
        self.counters: Counter[str] = Counter()
        self.samples: defaultdict[str, list[float]] = defaultdict(list)
        self.maxima: dict[str, float] = {}
        self.batch_sizes: Counter[str] = Counter()
        self.by_camera: defaultdict[str, Counter[str]] = defaultdict(Counter)

    def inc(self, name: str, amount: int = 1, camera_id: str | None = None) -> None:
        with self._lock:
            self.counters[name] += int(amount)
            if camera_id:
                self.by_camera[camera_id][name] += int(amount)

    def observe(self, name: str, value: float, camera_id: str | None = None) -> None:
        value = float(value)
        if not math.isfinite(value):
            return
        with self._lock:
            values = self.samples[name]
            if len(values) < self.sample_limit:
                values.append(value)
            else:
                # Keep a bounded recent tail once the run is unusually long.
                values[len(values) % self.sample_limit] = value
            self.maxima[name] = max(value, self.maxima.get(name, value))
            if camera_id:
                self.by_camera[camera_id][name] += 1

    def observe_batch_size(self, size: int) -> None:
        with self._lock:
            self.batch_sizes[str(int(size))] += 1

    def set_max(self, name: str, value: float) -> None:
        with self._lock:
            self.maxima[name] = max(float(value), self.maxima.get(name, float(value)))

    def snapshot(self) -> dict:
        with self._lock:
            counters = dict(self.counters)
            samples = {name: list(values) for name, values in self.samples.items()}
            maxima = dict(self.maxima)
            batches = dict(self.batch_sizes)
            by_camera = {camera: dict(values) for camera, values in self.by_camera.items()}
        latency = {}
        for name, values in samples.items():
            latency[name] = {
                "count": len(values),
                "mean": statistics.fmean(values) if values else None,
                "p95": _percentile(values, 0.95),
                "max": max(values) if values else None,
            }
        return {
            "counters": counters,
            "latency_ms": latency,
            "max": maxima,
            "osnet_batch_size_distribution": batches,
            "by_camera": by_camera,
        }

    def age_ms(self, received_monotonic_ns: int, now_ns: int | None = None) -> float:
        now_ns = monotonic_ns() if now_ns is None else now_ns
        age = max(0.0, (now_ns - int(received_monotonic_ns)) / 1_000_000.0)
        self.observe("observation_age_at_processing_ms", age)
        return age
