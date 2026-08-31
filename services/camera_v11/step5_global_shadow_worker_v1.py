from __future__ import annotations

import csv
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

from .step4_reid_same_room_matcher_v1 import SameRoomPairDiagnosticV1
from .step5_global_shadow_v1 import (
    GlobalShadowEventV1,
    GlobalShadowStateMachineV1,
)


TSV_COLUMNS = (
    "timestamp",
    "event",
    "shadow_global_id",
    "room",
    "camera_a",
    "track_a",
    "camera_b",
    "track_b",
    "proposal_count",
    "consecutive_count",
    "state",
    "robust_score",
    "status",
)


@dataclass(frozen=True)
class _CycleStartV1:
    cycle: int
    timestamp_ns: int


@dataclass(frozen=True)
class _CycleEndV1:
    cycle: int
    timestamp_ns: int


@dataclass(frozen=True)
class _ProposalV1:
    cycle: int
    timestamp_ns: int
    row: SameRoomPairDiagnosticV1


class V11GlobalShadowWorkerV1:
    """Bounded asynchronous Step5 state worker fed by Step4 proposals only."""

    def __init__(
        self,
        *,
        tsv_path: str | Path | None = "artifacts/reid/step5_global_shadow_v1.tsv",
        queue_capacity: int = 256,
        confirm_observations: int = 3,
        confirm_consecutive: int = 3,
        expire_provisional_after_missed_cycles: int = 6,
    ) -> None:
        self.tsv_path = Path(tsv_path).expanduser() if tsv_path else None
        self.machine = GlobalShadowStateMachineV1(
            confirm_observations=confirm_observations,
            confirm_consecutive=confirm_consecutive,
            expire_provisional_after_missed_cycles=expire_provisional_after_missed_cycles,
        )
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max(32, int(queue_capacity)))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.worker_errors = 0
        self.queue_dropped = 0
        self.events_written = 0

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            if self.tsv_path is not None:
                self.tsv_path.parent.mkdir(parents=True, exist_ok=True)
                with self.tsv_path.open("w", encoding="utf-8", newline="") as handle:
                    csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(
                        TSV_COLUMNS
                    )
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="camera-v11-step5-global-shadow",
                daemon=True,
            )
            self._thread.start()

    def _put(self, item: object) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self.queue_dropped += 1

    def enqueue_cycle_start(self, cycle: int, timestamp_ns: int) -> None:
        self._put(_CycleStartV1(int(cycle), int(timestamp_ns)))

    def enqueue_proposal(
        self,
        cycle: int,
        timestamp_ns: int,
        row: SameRoomPairDiagnosticV1,
    ) -> None:
        self._put(_ProposalV1(int(cycle), int(timestamp_ns), row))

    def enqueue_cycle_end(self, cycle: int, timestamp_ns: int) -> None:
        self._put(_CycleEndV1(int(cycle), int(timestamp_ns)))

    @staticmethod
    def _number(value: float | None) -> str:
        return "" if value is None else f"{float(value):.6f}"

    def _write_events(self, events: tuple[GlobalShadowEventV1, ...]) -> None:
        if not events:
            return
        if self.tsv_path is not None:
            rows = []
            for event in events:
                rows.append(
                    (
                        str(event.timestamp_ns),
                        event.event,
                        event.shadow_global_id,
                        event.room,
                        event.camera_a,
                        event.track_a,
                        event.camera_b,
                        event.track_b,
                        str(event.proposal_count),
                        str(event.consecutive_count),
                        event.state,
                        self._number(event.robust_score),
                        event.status,
                    )
                )
            with self.tsv_path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)
        with self._lock:
            self.events_written += len(events)

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                events: tuple[GlobalShadowEventV1, ...] = ()
                if isinstance(item, _CycleStartV1):
                    self.machine.begin_cycle(item.cycle)
                elif isinstance(item, _ProposalV1):
                    row = item.row
                    events = self.machine.observe_proposal(
                        cycle=item.cycle,
                        timestamp_ns=item.timestamp_ns,
                        room=row.room,
                        camera_a=row.camera_a,
                        track_a=row.track_a,
                        camera_b=row.camera_b,
                        track_b=row.track_b,
                        robust_score=row.robust_score,
                        reciprocal=bool(row.reciprocal),
                        assigned=bool(row.assigned),
                        status=row.status,
                    )
                elif isinstance(item, _CycleEndV1):
                    events = self.machine.end_cycle(
                        cycle=item.cycle, timestamp_ns=item.timestamp_ns
                    )
                else:
                    raise TypeError(f"unknown Step5 queue item: {type(item).__name__}")
                self._write_events(events)
            except Exception as exc:
                with self._lock:
                    self.worker_errors += 1
                print(
                    "V11_STEP5_GLOBAL_SHADOW_WORKER_ERROR "
                    f"error={type(exc).__name__}:{exc}",
                    flush=True,
                )
            finally:
                self._queue.task_done()

    def snapshot(self) -> dict[str, int | float]:
        row = self.machine.snapshot()
        with self._lock:
            row.update(
                {
                    "queue_pending": self._queue.qsize(),
                    "queue_dropped": self.queue_dropped,
                    "events_written": self.events_written,
                    "worker_errors": self.worker_errors,
                }
            )
        return row

    def close(self, timeout_sec: float = 3.0) -> None:
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._stop.set()
        thread.join(timeout=max(0.1, float(timeout_sec)))
        with self._lock:
            if thread.is_alive():
                self.worker_errors += 1
                print(
                    "V11_STEP5_GLOBAL_SHADOW_WORKER_ERROR error=close_timeout",
                    flush=True,
                )
            self._thread = None
