"""Bounded latest-state queues and one-worker OSNet micro-batching."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from services.mv3dt_room.identity_metrics import IdentityMetrics, monotonic_ns


@dataclass
class ObservationEnvelope:
    observation: dict[str, Any]
    generation: int
    crop: Any = None
    critical: bool = False
    submitted_monotonic_ns: int = 0

    def __post_init__(self) -> None:
        if not self.submitted_monotonic_ns:
            self.submitted_monotonic_ns = monotonic_ns()

    @property
    def key(self) -> tuple[str, int, int]:
        obs = self.observation
        return (
            str(obs["camera_id"]),
            int(obs["native_track_id"]),
            int(self.generation),
        )


class LatestObservationQueue:
    """Coalesce routine updates while retaining critical lifecycle events."""

    def __init__(self, max_tracks: int = 256, metrics: IdentityMetrics | None = None):
        self.max_tracks = max(2, int(max_tracks))
        self.metrics = metrics
        self._condition = threading.Condition()
        self._latest: dict[tuple[str, int, int], ObservationEnvelope] = {}
        self._critical: set[tuple[str, int, int]] = set()
        self._max_depth = 0
        self._closed = False

    def submit(self, envelope: ObservationEnvelope) -> None:
        key = envelope.key
        with self._condition:
            if key not in self._latest and len(self._latest) >= self.max_tracks:
                raise RuntimeError("identity latest-state queue capacity exhausted")
            if key in self._latest:
                self.metrics and self.metrics.inc("coalesced_stale_updates")
            self._latest[key] = envelope
            if envelope.critical:
                self._critical.add(key)
            self._max_depth = max(self._max_depth, len(self._latest))
            self.metrics and self.metrics.observe("identity_queue_depth", len(self._latest))
            self.metrics and self.metrics.set_max("identity_latest_queue_depth", len(self._latest))
            self._condition.notify()

    def get_batch(self, limit: int, wait_seconds: float = 0.0) -> list[ObservationEnvelope]:
        deadline = time.monotonic() + max(0.0, float(wait_seconds))
        with self._condition:
            while not self._latest and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._condition.wait(remaining)
            if not self._latest:
                return []
            critical = [key for key in self._critical if key in self._latest]
            keys = critical + [key for key in self._latest if key not in self._critical]
            selected = keys[: max(1, int(limit))]
            result = [self._latest.pop(key) for key in selected]
            for key in selected:
                self._critical.discard(key)
            return result

    def depth(self) -> int:
        with self._condition:
            return len(self._latest)

    def max_depth(self) -> int:
        with self._condition:
            return self._max_depth

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


@dataclass
class ReIdResult:
    envelope: ObservationEnvelope
    vector: Any
    request_start_monotonic_ns: int
    request_end_monotonic_ns: int
    batch_size: int


class AsyncOsnetBatcher:
    """Single bounded CUDA worker with latest-first per-track requests."""

    def __init__(
        self,
        embedder,
        metrics: IdentityMetrics,
        batch_size: int = 8,
        batching_window_ms: float = 12.0,
        max_pending: int = 256,
    ) -> None:
        self.embedder = embedder
        self.metrics = metrics
        self.batch_size = max(1, min(8, int(batch_size)))
        self.batching_window_ms = max(0.0, float(batching_window_ms))
        self.requests = LatestObservationQueue(max_pending, metrics)
        self.results: queue.Queue[ReIdResult] = queue.Queue(maxsize=max_pending)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="osnet-cuda-batcher", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, envelope: ObservationEnvelope) -> None:
        self.requests.submit(envelope)
        self.metrics.observe("osnet_request_queue_depth", self.requests.depth())
        self.metrics.set_max("osnet_request_queue_depth", self.requests.depth())

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self.requests.get_batch(self.batch_size, self.batching_window_ms / 1000.0)
            if not batch:
                continue
            self.metrics.observe("osnet_request_queue_depth_at_dequeue", self.requests.depth())
            self.metrics.observe_batch_size(len(batch))
            started = monotonic_ns()
            try:
                vectors = self.embedder.embed_batch([item.crop for item in batch])
            except Exception as exc:
                self.metrics.inc("osnet_inference_errors")
                self.metrics.inc(f"osnet_error_{type(exc).__name__}")
                continue
            ended = monotonic_ns()
            elapsed_ms = (ended - started) / 1_000_000.0
            self.metrics.observe("osnet_inference_latency_ms", elapsed_ms)
            for envelope, vector in zip(batch, vectors):
                self.results.put(ReIdResult(envelope, vector, started, ended, len(batch)))
                self.metrics.inc("osnet_results_published")

    def drain_results(self, callback: Callable[[ReIdResult], None] | None = None, limit: int = 32):
        count = 0
        drained: list[ReIdResult] = []
        while count < max(1, int(limit)):
            try:
                result = self.results.get_nowait()
            except queue.Empty:
                break
            drained.append(result)
            if callback is not None:
                callback(result)
            count += 1
        self.metrics.set_max("osnet_result_queue_depth", self.results.qsize())
        return count if callback is not None else drained

    def stop(self) -> None:
        self._stop.set()
        self.requests.close()
        self._thread.join(timeout=5.0)

    def runtime_metrics(self) -> dict:
        return {
            "request_queue_depth": self.requests.depth(),
            "request_queue_max_depth": self.requests.max_depth(),
            "result_queue_depth": self.results.qsize(),
            "batch_size_limit": self.batch_size,
            "batching_window_ms": self.batching_window_ms,
            "embedder": self.embedder.metrics(),
        }
