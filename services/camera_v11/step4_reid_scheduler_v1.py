from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .step4_reid_trt86 import V11ReIDTRT86Client


def _pct(values: deque[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return float(ordered[index])


@dataclass(frozen=True)
class ReIDCandidate:
    camera_id: str
    track_id: str
    room_id: str
    captured_ns: int
    bbox_xyxy: tuple[float, float, float, float]
    detector_score: float
    quality: float
    crop_bgr: np.ndarray

    @property
    def key(self) -> tuple[str, str]:
        return self.camera_id, self.track_id


@dataclass(frozen=True)
class ReIDResult:
    candidate: ReIDCandidate
    embedding: np.ndarray
    queue_wait_ms: float
    batch_size: int
    stages: dict[str, float]


class V11ReIDSchedulerV1:
    """One asynchronous TRT86 worker with bounded per-track latest-only requests.

    The detector/tracker thread never waits for ReID inference. Each local track owns
    at most one pending request; a newer crop overwrites its older pending crop. The
    global pending set is bounded as a second safety valve, and stale requests are
    discarded instead of creating latency debt.
    """

    def __init__(
        self,
        on_result: Callable[[ReIDResult], None],
        *,
        max_batch: int = 2,
        max_wait_ms: float = 3.0,
        max_pending: int = 12,
        max_age_ms: float = 300.0,
        client_factory: Callable[[], V11ReIDTRT86Client] = V11ReIDTRT86Client,
    ) -> None:
        self.on_result = on_result
        self.max_batch = max(1, min(4, int(max_batch)))
        self.max_wait_ms = max(0.0, min(10.0, float(max_wait_ms)))
        self.max_pending = max(self.max_batch, min(64, int(max_pending)))
        self.max_age_ms = max(50.0, min(1000.0, float(max_age_ms)))
        self.client_factory = client_factory

        self._cv = threading.Condition()
        self._pending: OrderedDict[
            tuple[str, str], tuple[ReIDCandidate, int]
        ] = OrderedDict()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._fatal: str | None = None

        self.submitted = 0
        self.replaced = 0
        self.overflow_drops = 0
        self.stale_drops = 0
        self.completed = 0
        self.batch_count = 0
        self.callback_errors = 0
        self.worker_errors = 0
        self.batch_hist = {size: 0 for size in range(1, self.max_batch + 1)}
        self.queue_wait_values: deque[float] = deque(maxlen=2048)
        self.wall_values: deque[float] = deque(maxlen=2048)
        self.infer_values: deque[float] = deque(maxlen=2048)

    def start(self, timeout_sec: float = 20.0) -> None:
        with self._cv:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="camera-v11-step4-reid-scheduler",
                daemon=True,
            )
            self._thread.start()
        if not self._ready.wait(timeout=max(1.0, float(timeout_sec))):
            raise TimeoutError("Step4 ReID scheduler worker startup timeout")
        if self._fatal:
            raise RuntimeError(f"Step4 ReID scheduler startup failed: {self._fatal}")

    def submit(self, candidate: ReIDCandidate) -> bool:
        if candidate.crop_bgr.size == 0:
            return False
        enqueued_ns = time.monotonic_ns()
        with self._cv:
            if self._stop or self._fatal:
                return False
            key = candidate.key
            if key in self._pending:
                self._pending[key] = (candidate, enqueued_ns)
                self.replaced += 1
            else:
                if len(self._pending) >= self.max_pending:
                    self._pending.popitem(last=False)
                    self.overflow_drops += 1
                self._pending[key] = (candidate, enqueued_ns)
            self.submitted += 1
            self._cv.notify()
            return True

    def _take_batch(self) -> list[tuple[ReIDCandidate, int]]:
        with self._cv:
            while not self._stop and not self._pending:
                self._cv.wait(timeout=0.25)
            if self._stop:
                return []

            deadline = time.monotonic() + self.max_wait_ms / 1000.0
            while not self._stop and len(self._pending) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._cv.wait(timeout=remaining)
            if self._stop:
                return []

            now_ns = time.monotonic_ns()
            batch: list[tuple[ReIDCandidate, int]] = []
            while self._pending and len(batch) < self.max_batch:
                _key, (candidate, enqueued_ns) = self._pending.popitem(last=False)
                age_ms = max(0.0, (now_ns - candidate.captured_ns) / 1_000_000.0)
                if age_ms > self.max_age_ms:
                    self.stale_drops += 1
                    continue
                batch.append((candidate, enqueued_ns))
            return batch

    def _run(self) -> None:
        client: V11ReIDTRT86Client | None = None
        try:
            client = self.client_factory()
            self._ready.set()
            while True:
                batch_rows = self._take_batch()
                if not batch_rows:
                    with self._cv:
                        if self._stop:
                            break
                    continue

                batch_started_ns = time.monotonic_ns()
                candidates = [row[0] for row in batch_rows]
                started = time.perf_counter()
                embeddings, stages = client.embed_crops([row.crop_bgr for row in candidates])
                wall_ms = (time.perf_counter() - started) * 1000.0
                infer_ms = float(stages.get("inference_ms", 0.0))

                with self._cv:
                    self.batch_count += 1
                    self.batch_hist[len(candidates)] = self.batch_hist.get(len(candidates), 0) + 1
                    self.wall_values.append(wall_ms)
                    if infer_ms > 0.0:
                        self.infer_values.append(infer_ms)

                for (candidate, enqueued_ns), embedding in zip(batch_rows, embeddings, strict=True):
                    queue_wait_ms = max(0.0, (batch_started_ns - enqueued_ns) / 1_000_000.0)
                    result = ReIDResult(
                        candidate=candidate,
                        embedding=np.asarray(embedding, dtype=np.float32).copy(),
                        queue_wait_ms=queue_wait_ms,
                        batch_size=len(candidates),
                        stages=dict(stages),
                    )
                    with self._cv:
                        self.completed += 1
                        self.queue_wait_values.append(queue_wait_ms)
                    try:
                        self.on_result(result)
                    except Exception as exc:  # Result consumers must never kill GPU scheduling.
                        with self._cv:
                            self.callback_errors += 1
                        print(
                            "V11_STEP4_REID_CALLBACK_ERROR "
                            f"camera={candidate.camera_id} track={candidate.track_id} "
                            f"error={type(exc).__name__}:{exc}",
                            flush=True,
                        )
        except Exception as exc:
            with self._cv:
                self.worker_errors += 1
                self._fatal = f"{type(exc).__name__}:{exc}"
                self._stop = True
                self._pending.clear()
                self._cv.notify_all()
            self._ready.set()
            print(f"V11_STEP4_REID_SCHEDULER_FATAL error={self._fatal}", flush=True)
        finally:
            if client is not None:
                client.close()
            self._ready.set()

    def snapshot(self) -> dict[str, object]:
        with self._cv:
            return {
                "pending": len(self._pending),
                "submitted": self.submitted,
                "replaced": self.replaced,
                "overflow_drops": self.overflow_drops,
                "stale_drops": self.stale_drops,
                "completed": self.completed,
                "batches": self.batch_count,
                "batch_hist": dict(self.batch_hist),
                "queue_p50_ms": _pct(self.queue_wait_values, 0.50),
                "queue_p95_ms": _pct(self.queue_wait_values, 0.95),
                "wall_p50_ms": _pct(self.wall_values, 0.50),
                "wall_p95_ms": _pct(self.wall_values, 0.95),
                "infer_p50_ms": _pct(self.infer_values, 0.50),
                "infer_p95_ms": _pct(self.infer_values, 0.95),
                "callback_errors": self.callback_errors,
                "worker_errors": self.worker_errors,
                "fatal": self._fatal,
            }

    def close(self) -> None:
        with self._cv:
            if self._stop and self._thread is None:
                return
            self._stop = True
            self._pending.clear()
            self._cv.notify_all()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        with self._cv:
            self._thread = None
