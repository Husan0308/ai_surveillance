from __future__ import annotations

import csv
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Callable

from .step4_reid_gallery_v1 import GalleryViewV1

from .step4_reid_same_room_matcher_v1 import (
    ASSIGNMENT_CONFLICT,
    INSUFFICIENT,
    INVALID,
    LOW_MARGIN,
    LOW_SCORE,
    MATCH_PROPOSED,
    NON_RECIPROCAL,
    STALE,
    SameRoomMatcherConfigV1,
    SameRoomMatrixResultV1,
    SameRoomPairDiagnosticV1,
    match_same_room_camera_pair_v1,
)


TSV_COLUMNS = (
    "timestamp",
    "cycle",
    "room",
    "camera_a",
    "track_a",
    "camera_b",
    "track_b",
    "samples_a",
    "samples_b",
    "robust_score",
    "max_score",
    "top3_mean",
    "median",
    "a_best_mean",
    "a_best_min",
    "b_best_mean",
    "b_best_min",
    "support_ge_050",
    "support_ge_055",
    "support_ge_060",
    "support_ge_065",
    "support_ge_070",
    "support_ge_075",
    "support_ge_080",
    "row_best",
    "row_second",
    "row_margin",
    "column_best",
    "column_second",
    "column_margin",
    "reciprocal",
    "assigned",
    "proposal_seen",
    "proposal_consecutive",
    "proposal_changed",
    "status",
    "reason",
)


def _pct(values: deque[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((len(ordered) - 1) * float(quantile)))),
    )
    return float(ordered[index])


@dataclass
class _ProposalStateV1:
    seen: int = 0
    consecutive: int = 0
    changed: int = 0
    last_cycle: int = 0


class V11SameRoomMatcherShadowWorkerV1:
    """Coalesced same-room matcher; it emits diagnostics and never mutates IDs."""

    def __init__(
        self,
        snapshot_provider: Callable[[], tuple[GalleryViewV1, ...]],
        camera_rooms: dict[str, str],
        *,
        tsv_path: str | Path | None = "artifacts/reid/step4_same_room_matches_v1.tsv",
        config: SameRoomMatcherConfigV1 | None = None,
        max_tracks_per_camera: int = 8,
        min_cycle_interval_sec: float = 2.0,
        phase_delay_sec: float = 0.1,
        affinity_cpu: int | None = None,
    ) -> None:
        self.snapshot_provider = snapshot_provider
        self.camera_rooms = {str(key): str(value) for key, value in camera_rooms.items()}
        self.tsv_path = Path(tsv_path).expanduser() if tsv_path else None
        self.config = config or SameRoomMatcherConfigV1()
        self.max_tracks_per_camera = max(1, min(32, int(max_tracks_per_camera)))
        self.min_cycle_interval_sec = max(0.5, min(10.0, float(min_cycle_interval_sec)))
        self.phase_delay_sec = max(0.0, min(0.5, float(phase_delay_sec)))
        self.affinity_cpu = None if affinity_cpu is None else int(affinity_cpu)
        self._camera_pairs = self._build_camera_pairs()
        self._cv = threading.Condition()
        self._dirty = False
        self._stop = False
        self._thread: threading.Thread | None = None
        self._proposal_states: dict[tuple[str, str, str, str], _ProposalStateV1] = {}
        self._previous_endpoints: dict[tuple[str, str], tuple[str, str]] = {}
        self._matrix_cache: dict[
            tuple[str, str], tuple[tuple[object, ...], SameRoomMatrixResultV1]
        ] = {}

        self.cycles = 0
        self.matrices_built = 0
        self.pairs_considered = 0
        self.pairs_valid = 0
        self.pairs_insufficient = 0
        self.pairs_nonreciprocal = 0
        self.pairs_low_margin = 0
        self.pairs_low_score = 0
        self.assignment_conflicts = 0
        self.proposals = 0
        self.proposal_changes = 0
        self.pairs_stale = 0
        self.pairs_invalid = 0
        self.worker_errors = 0

        self.match_ms: deque[float] = deque(maxlen=4096)

    def _build_camera_pairs(self) -> tuple[tuple[int, str, str, str], ...]:
        by_room: dict[str, list[str]] = {}
        for camera_id, room in self.camera_rooms.items():
            by_room.setdefault(room, []).append(camera_id)
        pairs = []
        for room, cameras in by_room.items():
            for camera_a, camera_b in combinations(sorted(cameras), 2):
                priority = int({camera_a, camera_b} != {"CAM-01", "CAM-04"})
                pairs.append((priority, room, camera_a, camera_b))
        return tuple(sorted(pairs))

    def start(self) -> None:
        with self._cv:
            if self._thread is not None:
                return
            if self.tsv_path is not None:
                self.tsv_path.parent.mkdir(parents=True, exist_ok=True)
                with self.tsv_path.open("w", encoding="utf-8", newline="") as handle:
                    csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(TSV_COLUMNS)
            self._thread = threading.Thread(
                target=self._run,
                name="camera-v11-step4-same-room-matcher-shadow",
                daemon=True,
            )
            self._thread.start()

    def notify(self) -> None:
        with self._cv:
            if not self._stop:
                self._dirty = True
                self._cv.notify()

    @staticmethod
    def _number(value: float | None) -> str:
        return "" if value is None else f"{float(value):.6f}"

    @staticmethod
    def _pair_key(row: SameRoomPairDiagnosticV1) -> tuple[str, str, str, str]:
        return row.camera_a, row.track_a, row.camera_b, row.track_b


    def _stability(
        self,
        row: SameRoomPairDiagnosticV1,
        cycle: int,
        prior_endpoints: dict[tuple[str, str], tuple[str, str]],
        current_endpoints: dict[tuple[str, str], tuple[str, str]],
    ) -> tuple[int, int, int]:
        key = self._pair_key(row)
        state = self._proposal_states.get(key)
        if row.status != MATCH_PROPOSED:
            if state is None:
                return 0, 0, 0
            return state.seen, state.consecutive, state.changed
        if state is None:
            state = _ProposalStateV1()
            self._proposal_states[key] = state
        endpoint_a = (row.camera_a, row.track_a)
        endpoint_b = (row.camera_b, row.track_b)
        partner_a = (row.camera_b, row.track_b)
        partner_b = (row.camera_a, row.track_a)
        changed = int(
            (endpoint_a in prior_endpoints and prior_endpoints[endpoint_a] != partner_a)
            or (endpoint_b in prior_endpoints and prior_endpoints[endpoint_b] != partner_b)
        )
        state.seen += 1
        state.consecutive = state.consecutive + 1 if state.last_cycle == cycle - 1 else 1
        state.changed += changed
        state.last_cycle = cycle
        current_endpoints[endpoint_a] = partner_a
        current_endpoints[endpoint_b] = partner_b
        self.proposal_changes += changed
        return state.seen, state.consecutive, state.changed

    def _matrix_result(
        self,
        camera_a: str,
        camera_b: str,
        left: list[GalleryViewV1],
        right: list[GalleryViewV1],
        now_ns: int,
    ) -> SameRoomMatrixResultV1:
        started = time.perf_counter()
        recent_ns = int(self.config.recent_age_sec * 1e9)
        fingerprint = tuple(
            (
                view.camera_id,
                view.local_track_id,
                tuple(sample.sample_sequence for sample in view.samples),
                now_ns - int(view.last_seen_ns) > recent_ns,
            )
            for view in sorted(
                (*left, *right),
                key=lambda item: (item.camera_id, item.local_track_id),
            )
        )
        key = (camera_a, camera_b)
        cached = self._matrix_cache.get(key)
        if cached is not None and cached[0] == fingerprint:
            return replace(
                cached[1], elapsed_ms=(time.perf_counter() - started) * 1000.0
            )
        result = match_same_room_camera_pair_v1(
            left,
            right,
            self.camera_rooms,
            now_ns=now_ns,
            config=self.config,
        )
        self._matrix_cache[key] = (fingerprint, result)
        return result

    def _tsv_row(
        self,
        timestamp_ns: int,
        cycle: int,
        row: SameRoomPairDiagnosticV1,
        stability: tuple[int, int, int],
    ) -> tuple[str, ...]:
        return (
            str(timestamp_ns),
            str(cycle),
            row.room,
            row.camera_a,
            row.track_a,
            row.camera_b,
            row.track_b,
            str(row.samples_a),
            str(row.samples_b),
            self._number(row.robust_score),
            self._number(row.max_score),
            self._number(row.top3_mean),
            self._number(row.median),
            self._number(row.a_best_mean),
            self._number(row.a_best_min),
            self._number(row.b_best_mean),
            self._number(row.b_best_min),
            str(row.support_ge_050),
            str(row.support_ge_055),
            str(row.support_ge_060),
            str(row.support_ge_065),
            str(row.support_ge_070),
            str(row.support_ge_075),
            str(row.support_ge_080),
            self._number(row.row_best),
            self._number(row.row_second),
            self._number(row.row_margin),
            self._number(row.column_best),
            self._number(row.column_second),
            self._number(row.column_margin),
            str(int(row.reciprocal)),
            str(int(row.assigned)),
            str(stability[0]),
            str(stability[1]),
            str(stability[2]),
            row.status,
            row.reason,
        )

    def _match_cycle(self) -> None:
        timestamp_ns = time.time_ns()
        now_ns = time.monotonic_ns()
        views = self.snapshot_provider()
        by_camera: dict[str, list[GalleryViewV1]] = {}
        for view in views:
            if view.camera_id in self.camera_rooms and view.samples:
                by_camera.setdefault(view.camera_id, []).append(view)
        for camera_views in by_camera.values():
            camera_views.sort(key=lambda view: (-int(view.last_seen_ns), str(view.local_track_id)))
            del camera_views[self.max_tracks_per_camera :]

        with self._cv:
            self.cycles += 1
            cycle = self.cycles
            prior_endpoints = self._previous_endpoints
        current_endpoints: dict[tuple[str, str], tuple[str, str]] = {}

        tsv_rows: list[tuple[str, ...]] = []
        for _priority, _room, camera_a, camera_b in self._camera_pairs:
            left = by_camera.get(camera_a, [])
            right = by_camera.get(camera_b, [])
            if not left or not right:
                continue
            result = self._matrix_result(
                camera_a, camera_b, left, right, now_ns
            )
            with self._cv:
                self.matrices_built += 1
                self.match_ms.append(result.elapsed_ms)
            for row in result.diagnostics:

                with self._cv:
                    self.pairs_considered += 1
                    if row.status not in (INSUFFICIENT, INVALID, STALE):
                        self.pairs_valid += 1
                    self.pairs_insufficient += int(row.status == INSUFFICIENT)
                    self.pairs_nonreciprocal += int(row.status == NON_RECIPROCAL)
                    self.pairs_low_margin += int(row.status == LOW_MARGIN)
                    self.pairs_low_score += int(row.status == LOW_SCORE)
                    self.assignment_conflicts += int(row.status == ASSIGNMENT_CONFLICT)
                    self.proposals += int(row.status == MATCH_PROPOSED)
                    self.pairs_stale += int(row.status == STALE)
                    self.pairs_invalid += int(row.status == INVALID)
                    stability = self._stability(
                        row, cycle, prior_endpoints, current_endpoints
                    )
                tsv_rows.append(self._tsv_row(timestamp_ns, cycle, row, stability))
        with self._cv:
            self._previous_endpoints = current_endpoints

        if self.tsv_path is not None and tsv_rows:
            with self.tsv_path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(tsv_rows)

    def _run(self) -> None:
        if self.affinity_cpu is not None:
            try:
                os.sched_setaffinity(0, {self.affinity_cpu})
            except (AttributeError, OSError, ValueError) as exc:
                with self._cv:
                    self.worker_errors += 1
                print(
                    "V11_STEP4_REID_SAME_ROOM_MATCHER_SHADOW_ERROR "
                    f"error=affinity:{type(exc).__name__}:{exc}",
                    flush=True,
                )
                return
        next_cycle_at = 0.0
        while True:
            with self._cv:
                while not self._dirty and not self._stop:
                    self._cv.wait(timeout=0.5)
                if self._stop and not self._dirty:
                    return
                while not self._stop:
                    remaining = next_cycle_at - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(timeout=remaining)
                self._dirty = False
                phase_until = time.monotonic() + self.phase_delay_sec
                while not self._stop:
                    remaining = phase_until - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(timeout=remaining)
            try:
                self._match_cycle()
                next_cycle_at = time.monotonic() + self.min_cycle_interval_sec
            except Exception as exc:
                with self._cv:
                    self.worker_errors += 1
                print(
                    "V11_STEP4_REID_SAME_ROOM_MATCHER_SHADOW_ERROR "
                    f"error={type(exc).__name__}:{exc}",
                    flush=True,
                )

    def snapshot(self) -> dict[str, int | float]:
        with self._cv:
            return {
                "cycles": self.cycles,
                "matrices_built": self.matrices_built,
                "pairs_considered": self.pairs_considered,
                "pairs_valid": self.pairs_valid,
                "pairs_insufficient": self.pairs_insufficient,
                "pairs_nonreciprocal": self.pairs_nonreciprocal,
                "pairs_low_margin": self.pairs_low_margin,
                "pairs_low_score": self.pairs_low_score,
                "assignment_conflicts": self.assignment_conflicts,
                "proposals": self.proposals,
                "unique_proposals": len(self._proposal_states),
                "proposal_changes": self.proposal_changes,
                "pairs_stale": self.pairs_stale,
                "pairs_invalid": self.pairs_invalid,
                "match_p50_ms": _pct(self.match_ms, 0.50),
                "match_p95_ms": _pct(self.match_ms, 0.95),
                "worker_errors": self.worker_errors,

            }

    def close(self, timeout_sec: float = 3.0) -> None:
        with self._cv:
            if self._thread is None:
                return
            self._dirty = True
            self._stop = True
            self._cv.notify_all()
            thread = self._thread
        thread.join(timeout=max(0.1, float(timeout_sec)))
        with self._cv:
            self._thread = None
