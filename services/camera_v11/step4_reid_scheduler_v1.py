from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .step4_reid_trt86 import EMBEDDING_DIMENSION, V11ReIDTRT86Client


def _pct(values: deque[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((len(ordered) - 1) * float(quantile)))),
    )
    return float(ordered[index])


@dataclass(frozen=True)
class ReIDCandidateV1:
    camera_id: str
    local_track_id: str
    captured_ns: int
    bbox_xyxy: tuple[float, float, float, float]
    detector_confidence: float
    quality_score: float
    crop_bgr: np.ndarray

    @property
    def key(self) -> tuple[str, str]:
        return self.camera_id, self.local_track_id


@dataclass(frozen=True)
class ReIDResultV1:
    candidate: ReIDCandidateV1
    embedding: np.ndarray
    queue_wait_ms: float
    batch_size: int
    stages: dict[str, float]


class V11ReIDSchedulerV1:
    """Single async TRT worker with bounded latest-only per-track coalescing."""

    def __init__(
        self,
        on_result: Callable[[ReIDResultV1], None],
        *,
        max_batch: int = 2,
        max_wait_ms: float = 4.0,
        max_pending: int = 12,
        max_age_ms: float = 1000.0,
        client_factory: Callable[[], V11ReIDTRT86Client] = V11ReIDTRT86Client,
    ) -> None:
        self.on_result = on_result
        self.max_batch = max(1, min(8, int(max_batch)))
        self.max_wait_ms = max(0.0, min(20.0, float(max_wait_ms)))
        self.max_pending = max(self.max_batch, min(64, int(max_pending)))
        self.max_age_ms = max(100.0, min(5000.0, float(max_age_ms)))
        self.client_factory = client_factory

        self._cv = threading.Condition()
        self._pending: OrderedDict[
            tuple[str, str], tuple[ReIDCandidateV1, int]
        ] = OrderedDict()
        self._accepting = True
        self._close_requested = False
        self._force_stop = False
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._fatal: str | None = None

        self.reid_submitted = 0
        self.reid_completed = 0
        self.reid_replaced_pending = 0
        self.reid_overflow_drop = 0
        self.reid_stale_drop = 0
        self.reid_worker_errors = 0
        self.reid_callback_errors = 0
        self.queue_ms: deque[float] = deque(maxlen=4096)
        self.infer_ms: deque[float] = deque(maxlen=4096)
        self.wall_ms: deque[float] = deque(maxlen=4096)

    def start(self, timeout_sec: float = 25.0) -> None:
        with self._cv:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="camera-v11-step4-reid-gallery-worker",
                daemon=True,
            )
            self._thread.start()
        if not self._ready.wait(timeout=max(1.0, float(timeout_sec))):
            raise TimeoutError("Step4 ReID worker startup timeout")
        if self._fatal is not None:
            raise RuntimeError(f"Step4 ReID worker startup failed: {self._fatal}")

    def submit(self, candidate: ReIDCandidateV1) -> bool:
        crop = candidate.crop_bgr
        if (
            crop is None
            or not isinstance(crop, np.ndarray)
            or crop.dtype != np.uint8
            or crop.ndim != 3
            or crop.shape[2] != 3
            or crop.size == 0
        ):
            return False
        enqueued_ns = time.monotonic_ns()
        with self._cv:
            if not self._accepting or self._fatal is not None:
                return False
            if candidate.key in self._pending:
                self._pending[candidate.key] = (candidate, enqueued_ns)
                self._pending.move_to_end(candidate.key)
                self.reid_replaced_pending += 1
            else:
                if len(self._pending) >= self.max_pending:
                    self._pending.popitem(last=False)
                    self.reid_overflow_drop += 1
                self._pending[candidate.key] = (candidate, enqueued_ns)
            self.reid_submitted += 1
            self._cv.notify()
            return True

    def _take_batch(self) -> list[tuple[ReIDCandidateV1, int]]:
        with self._cv:
            while not self._force_stop and not self._pending:
                if self._close_requested:
                    return []
                self._cv.wait(timeout=0.25)
            if self._force_stop:
                return []

            if not self._close_requested:
                deadline = time.monotonic() + self.max_wait_ms / 1000.0
                while (
                    not self._force_stop
                    and not self._close_requested
                    and len(self._pending) < self.max_batch
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    self._cv.wait(timeout=remaining)
            if self._force_stop:
                return []

            now_ns = time.monotonic_ns()
            batch: list[tuple[ReIDCandidateV1, int]] = []
            while self._pending and len(batch) < self.max_batch:
                _key, (candidate, enqueued_ns) = self._pending.popitem(last=False)
                queue_age_ms = max(0.0, (now_ns - enqueued_ns) / 1_000_000.0)
                if queue_age_ms > self.max_age_ms:
                    self.reid_stale_drop += 1
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
                        if self._force_stop or (
                            self._close_requested and not self._pending
                        ):
                            break
                    continue

                batch_started_ns = time.monotonic_ns()
                candidates = [row[0] for row in batch_rows]
                wall_started = time.perf_counter()
                embeddings, stages = client.embed_crops(
                    [row.crop_bgr for row in candidates]
                )
                wall_ms = (time.perf_counter() - wall_started) * 1000.0
                if embeddings.shape != (len(candidates), EMBEDDING_DIMENSION):
                    raise RuntimeError(
                        f"unexpected scheduler embedding shape={embeddings.shape}"
                    )
                infer_ms = float(stages.get("inference_ms", 0.0))
                with self._cv:
                    self.wall_ms.append(wall_ms)
                    if infer_ms > 0.0:
                        self.infer_ms.append(infer_ms)

                for (candidate, enqueued_ns), embedding in zip(
                    batch_rows, embeddings, strict=True
                ):
                    queue_wait_ms = max(
                        0.0, (batch_started_ns - enqueued_ns) / 1_000_000.0
                    )
                    result = ReIDResultV1(
                        candidate=candidate,
                        embedding=np.ascontiguousarray(embedding, dtype=np.float32),
                        queue_wait_ms=queue_wait_ms,
                        batch_size=len(candidates),
                        stages=dict(stages),
                    )
                    with self._cv:
                        self.reid_completed += 1
                        self.queue_ms.append(queue_wait_ms)
                    try:
                        self.on_result(result)
                    except Exception as exc:
                        with self._cv:
                            self.reid_callback_errors += 1
                        print(
                            "V11_STEP4_REID_GALLERY_CALLBACK_ERROR "
                            f"camera={candidate.camera_id} "
                            f"track={candidate.local_track_id} "
                            f"error={type(exc).__name__}:{exc}",
                            flush=True,
                        )
        except Exception as exc:
            with self._cv:
                self.reid_worker_errors += 1
                self._fatal = f"{type(exc).__name__}:{exc}"
                self._accepting = False
                self._force_stop = True
                self._pending.clear()
                self._cv.notify_all()
            self._ready.set()
            print(
                f"V11_STEP4_REID_GALLERY_WORKER_FATAL error={self._fatal}", flush=True
            )
        finally:
            if client is not None:
                client.close()
            self._ready.set()

    def snapshot(self) -> dict[str, int | float | str | None]:
        with self._cv:
            return {
                "reid_submitted": self.reid_submitted,
                "reid_completed": self.reid_completed,
                "reid_pending": len(self._pending),
                "reid_replaced_pending": self.reid_replaced_pending,
                "reid_overflow_drop": self.reid_overflow_drop,
                "reid_stale_drop": self.reid_stale_drop,
                "reid_worker_errors": self.reid_worker_errors
                + self.reid_callback_errors,
                "reid_queue_p50_ms": _pct(self.queue_ms, 0.50),
                "reid_queue_p95_ms": _pct(self.queue_ms, 0.95),
                "reid_infer_p50_ms": _pct(self.infer_ms, 0.50),
                "reid_infer_p95_ms": _pct(self.infer_ms, 0.95),
                "reid_wall_p50_ms": _pct(self.wall_ms, 0.50),
                "reid_wall_p95_ms": _pct(self.wall_ms, 0.95),
                "fatal": self._fatal,
            }

    def close(self, *, drain: bool = True, timeout_sec: float = 8.0) -> None:
        with self._cv:
            self._accepting = False
            self._close_requested = True
            if not drain:
                self._pending.clear()
                self._force_stop = True
            self._cv.notify_all()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.1, float(timeout_sec)))
        if thread is not None and thread.is_alive():
            with self._cv:
                self._pending.clear()
                self._force_stop = True
                self._cv.notify_all()
            thread.join(timeout=2.0)
        with self._cv:
            self._thread = None
