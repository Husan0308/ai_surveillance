from __future__ import annotations

import csv
import threading
from pathlib import Path

from .step5_global_shadow_v1 import GlobalShadowEventV1
from .step5_global_shadow_worker_v1 import V11GlobalShadowWorkerV1
from .step6_global_shadow_hysteresis_v1 import (
    GlobalShadowHysteresisV1,
    ShadowVerificationEventV1,
)


VERIFY_TSV_COLUMNS = (
    "timestamp",
    "event",
    "shadow_global_id",
    "room",
    "camera_a",
    "track_a",
    "camera_b",
    "track_b",
    "state",
    "clean_observations",
    "total_observations",
    "conflict_events",
    "conflict_streak",
    "robust_score",
    "status",
)


class V11GlobalShadowWorkerStep6TapV1(V11GlobalShadowWorkerV1):
    """Step5 worker with a tiny Step6 verifier tap on the same async thread.

    Camera, tracker, ReID, pair scorer and matcher threads never execute Step6
    state work or Step6 TSV I/O.  The tap runs only after Step5 has already
    generated its shadow events inside the dedicated Step5 worker thread.
    """

    def __init__(
        self,
        *args,
        verify_tsv_path: str | Path | None = "artifacts/reid/step6_global_verify_v1.tsv",
        verify_clean_observations: int = 3,
        recover_clean_observations: int = 3,
        persistent_conflict_observations: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.verify_tsv_path = (
            Path(verify_tsv_path).expanduser() if verify_tsv_path else None
        )
        self.verifier = GlobalShadowHysteresisV1(
            verify_clean_observations=verify_clean_observations,
            recover_clean_observations=recover_clean_observations,
            persistent_conflict_observations=persistent_conflict_observations,
        )
        self._verify_lock = threading.Lock()
        self.verify_events_written = 0
        self.verify_worker_errors = 0

    def start(self) -> None:
        if self.verify_tsv_path is not None:
            self.verify_tsv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.verify_tsv_path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(
                    VERIFY_TSV_COLUMNS
                )
        super().start()

    @staticmethod
    def _number(value: float | None) -> str:
        return "" if value is None else f"{float(value):.6f}"

    def _write_verify_events(
        self, events: tuple[ShadowVerificationEventV1, ...]
    ) -> None:
        if not events:
            return
        if self.verify_tsv_path is not None:
            rows = [
                (
                    str(event.timestamp_ns),
                    event.event,
                    event.shadow_global_id,
                    event.room,
                    event.camera_a,
                    event.track_a,
                    event.camera_b,
                    event.track_b,
                    event.state,
                    str(event.clean_observations),
                    str(event.total_observations),
                    str(event.conflict_events),
                    str(event.conflict_streak),
                    self._number(event.robust_score),
                    event.status,
                )
                for event in events
            ]
            with self.verify_tsv_path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)
        self.verify_events_written += len(events)

    def _write_events(self, events: tuple[GlobalShadowEventV1, ...]) -> None:
        super()._write_events(events)
        if not events:
            return
        verify_events: list[ShadowVerificationEventV1] = []
        try:
            with self._verify_lock:
                for source in events:
                    verify_events.extend(self.verifier.observe_step5_event(source))
                self._write_verify_events(tuple(verify_events))
        except Exception as exc:
            with self._verify_lock:
                self.verify_worker_errors += 1
            print(
                "V11_STEP6_GLOBAL_VERIFY_WORKER_ERROR "
                f"error={type(exc).__name__}:{exc}",
                flush=True,
            )

    def snapshot(self) -> dict[str, int | float]:
        row = super().snapshot()
        with self._verify_lock:
            row.update(self.verifier.snapshot())
            row.update(
                {
                    "verify_events_written": self.verify_events_written,
                    "verify_worker_errors": self.verify_worker_errors,
                }
            )
        return row
