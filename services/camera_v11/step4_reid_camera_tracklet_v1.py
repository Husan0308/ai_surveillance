from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping

from .step4_reid_gallery_v1 import GallerySampleV1, GalleryViewV1
from .step4_reid_pair_scorer_v1 import GalleryPairScoreV1, score_gallery_pair_v1
from .step4_reid_same_room_matcher_v1 import SameRoomPairDiagnosticV1


@dataclass(frozen=True)
class CameraTrackletConfigV1:
    scoped_cameras: frozenset[str] = frozenset({"CAM-01", "CAM-04"})
    recent_gap_sec: float = 8.0
    max_overlap_sec: float = 0.75
    visually_active_sec: float = 2.5
    min_samples: int = 3
    min_robust_score: float = 0.72
    min_margin: float = 0.05
    min_support_ge_065: int = 3
    confirm_cycles: int = 2
    active_overlap_grace_cycles: int = 3

    def __post_init__(self) -> None:
        if not self.scoped_cameras:
            raise ValueError("scoped_cameras must not be empty")
        if self.recent_gap_sec <= 0 or self.visually_active_sec <= 0:
            raise ValueError("time windows must be positive")
        if self.max_overlap_sec < 0:
            raise ValueError("max_overlap_sec must be non-negative")
        if self.min_samples < 3:
            raise ValueError("min_samples must be at least 3")
        if not 0.0 <= self.min_robust_score <= 1.0:
            raise ValueError("min_robust_score must be in [0, 1]")
        if self.min_margin < 0:
            raise ValueError("min_margin must be non-negative")
        if self.min_support_ge_065 < 1:
            raise ValueError("min_support_ge_065 must be positive")
        if self.confirm_cycles < 1:
            raise ValueError("confirm_cycles must be positive")
        if self.active_overlap_grace_cycles < 1:
            raise ValueError("active_overlap_grace_cycles must be positive")


@dataclass
class _StableCameraTrackletV1:
    camera_id: str
    stable_id: str
    raw_members: set[str] = field(default_factory=set)
    created_ns: int = 0
    updated_ns: int = 0


@dataclass
class _ContinuityVoteV1:
    last_cycle: int = 0
    consecutive: int = 0
    best_score: float = -1.0


@dataclass(frozen=True)
class CameraTrackletDiagnosticV1:
    timestamp_ns: int
    cycle: int
    event: str
    camera_id: str
    raw_track_id: str
    stable_id: str
    predecessor_track_id: str = ""
    successor_reid_score: float | None = None
    successor_margin: float | None = None


class CameraTrackletContinuityV1:
    """Shadow-only same-camera local-ID stitcher before Step5 Global Shadow.

    The frozen tracker remains untouched. A new raw local T-ID may inherit a
    stable camera-tracklet ID only when its recent same-camera ReID gallery is a
    reciprocal best match to a recently-lost predecessor, the appearance score
    and margin gates pass, and the evidence repeats across consecutive matcher
    cycles. Simultaneously visible tracks are never stitched.

    A newly-created raw tracker ID is intentionally left unresolved for a small,
    bounded number of matcher cycles when the only plausible predecessor is still
    visually active. This closes the race where the old T-ID is active on the
    first cycle, a fresh stable ID is allocated immediately, and that raw ID can
    never be reconsidered for continuity after the predecessor disappears.
    """

    def __init__(
        self,
        snapshot_provider: Callable[[], tuple[GalleryViewV1, ...]],
        active_provider: Callable[[], Mapping[str, frozenset[str]]],
        *,
        config: CameraTrackletConfigV1 | None = None,
    ) -> None:
        self.snapshot_provider = snapshot_provider
        self.active_provider = active_provider
        self.config = config or CameraTrackletConfigV1()
        self._lock = threading.RLock()
        self._next_id: dict[str, int] = {}
        self._raw_to_stable: dict[tuple[str, str], str] = {}
        self._raw_mapping_cycle: dict[tuple[str, str], int] = {}
        self._stable: dict[str, _StableCameraTrackletV1] = {}
        self._votes: dict[tuple[str, str, str], _ContinuityVoteV1] = {}
        self._deferred_first_cycle: dict[tuple[str, str], int] = {}
        self.refresh_ms: deque[float] = deque(maxlen=2048)
        self.created_total = 0
        self.stitched_total = 0
        self.pending_total = 0
        self.suppressed_total = 0
        self.low_score_total = 0
        self.low_margin_total = 0
        self.nonreciprocal_total = 0
        self.overlap_reject_total = 0
        self.insufficient_total = 0
        self.active_overlap_deferred_total = 0
        self.active_overlap_fallback_allocated_total = 0
        self.same_camera_active_ct_collision = 0
        self.successor_rejected_active_predecessor = 0
        self.successor_reid_score: float | None = None
        self.successor_margin: float | None = None
        self._active_camera_members: frozenset[tuple[str, str]] = frozenset()
        self._diagnostics: deque[CameraTrackletDiagnosticV1] = deque(maxlen=256)

    @staticmethod
    def _pct(values: deque[float], quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(
            len(ordered) - 1,
            max(0, int(round((len(ordered) - 1) * float(quantile)))),
        )
        return float(ordered[index])

    def _allocate(
        self, camera_id: str, raw_track_id: str, now_ns: int, cycle: int
    ) -> str:
        key = (camera_id, raw_track_id)
        existing = self._raw_to_stable.get(key)
        if existing is not None:
            return existing
        number = self._next_id.get(camera_id, 0) + 1
        self._next_id[camera_id] = number
        stable_id = f"{camera_id}-CT{number:05d}"
        record = _StableCameraTrackletV1(
            camera_id=camera_id,
            stable_id=stable_id,
            raw_members={raw_track_id},
            created_ns=int(now_ns),
            updated_ns=int(now_ns),
        )
        self._stable[stable_id] = record
        self._raw_to_stable[key] = stable_id
        self._raw_mapping_cycle[key] = int(cycle)
        self._deferred_first_cycle.pop(key, None)
        self.created_total += 1
        return stable_id

    def _diagnose(
        self,
        *,
        timestamp_ns: int,
        cycle: int,
        event: str,
        camera_id: str,
        raw_track_id: str,
        stable_id: str,
        predecessor_track_id: str = "",
        successor_reid_score: float | None = None,
        successor_margin: float | None = None,
    ) -> None:
        self._diagnostics.append(
            CameraTrackletDiagnosticV1(
                timestamp_ns=int(timestamp_ns),
                cycle=int(cycle),
                event=str(event),
                camera_id=str(camera_id),
                raw_track_id=str(raw_track_id),
                stable_id=str(stable_id),
                predecessor_track_id=str(predecessor_track_id),
                successor_reid_score=successor_reid_score,
                successor_margin=successor_margin,
            )
        )

    @staticmethod
    def _active_predecessor(
        record: _StableCameraTrackletV1,
        raw_track_id: str,
        active: frozenset[str],
    ) -> str | None:
        candidates = sorted(
            member
            for member in record.raw_members
            if member != raw_track_id and member in active
        )
        return candidates[0] if candidates else None

    def _repair_active_collisions(
        self,
        *,
        cycle: int,
        now_ns: int,
        active_snapshot: Mapping[str, frozenset[str]],
    ) -> None:
        grouped: dict[tuple[str, str], list[str]] = {}
        for (camera_id, raw_track_id), stable_id in self._raw_to_stable.items():
            if raw_track_id in active_snapshot.get(camera_id, frozenset()):
                grouped.setdefault((camera_id, stable_id), []).append(raw_track_id)
        for (camera_id, stable_id), raw_members in sorted(grouped.items()):
            if len(raw_members) < 2:
                continue
            ordered = sorted(
                raw_members,
                key=lambda raw: (
                    self._raw_mapping_cycle.get((camera_id, raw), 0),
                    raw,
                ),
            )
            keeper = ordered[0]
            for raw_track_id in ordered[1:]:
                self.same_camera_active_ct_collision += 1
                self._diagnose(
                    timestamp_ns=now_ns,
                    cycle=cycle,
                    event="same_camera_active_ct_collision",
                    camera_id=camera_id,
                    raw_track_id=raw_track_id,
                    stable_id=stable_id,
                    predecessor_track_id=keeper,
                )
                record = self._stable[stable_id]
                record.raw_members.discard(raw_track_id)
                self._raw_to_stable.pop((camera_id, raw_track_id), None)
                self._raw_mapping_cycle.pop((camera_id, raw_track_id), None)
                for vote_key in list(self._votes):
                    if vote_key[:2] == (camera_id, raw_track_id):
                        self._votes.pop(vote_key, None)
                self._allocate(camera_id, raw_track_id, now_ns, cycle)

    @staticmethod
    def _latest_sample_ns(view: GalleryViewV1) -> int:
        return max((int(sample.timestamp_ns) for sample in view.samples), default=0)

    @staticmethod
    def _first_sample_ns(view: GalleryViewV1) -> int:
        return min((int(sample.timestamp_ns) for sample in view.samples), default=0)

    @staticmethod
    def _aggregate_samples(
        record: _StableCameraTrackletV1,
        views_by_key: dict[tuple[str, str], GalleryViewV1],
    ) -> tuple[GallerySampleV1, ...]:
        samples: list[GallerySampleV1] = []
        for raw_track_id in record.raw_members:
            view = views_by_key.get((record.camera_id, raw_track_id))
            if view is not None:
                samples.extend(view.samples)
        samples.sort(key=lambda row: (int(row.timestamp_ns), int(row.sample_sequence)))
        return tuple(samples[-8:])

    def _candidate_records(
        self,
        camera_id: str,
        new_view: GalleryViewV1,
        active: frozenset[str],
        views_by_key: dict[tuple[str, str], GalleryViewV1],
    ) -> tuple[
        list[tuple[_StableCameraTrackletV1, tuple[GallerySampleV1, ...]]], bool
    ]:
        cfg = self.config
        new_first = self._first_sample_ns(new_view)
        candidates: list[tuple[_StableCameraTrackletV1, tuple[GallerySampleV1, ...]]] = []
        blocked_by_active_predecessor = False
        for record in self._stable.values():
            if record.camera_id != camera_id:
                continue
            samples = self._aggregate_samples(record, views_by_key)
            if len(samples) < cfg.min_samples:
                continue
            old_last = max(int(sample.timestamp_ns) for sample in samples)
            gap_ns = int(new_first) - int(old_last)
            if gap_ns > int(cfg.recent_gap_sec * 1e9):
                continue
            if gap_ns < -int(cfg.max_overlap_sec * 1e9):
                self.overlap_reject_total += 1
                continue
            if self._active_predecessor(record, str(new_view.local_track_id), active):
                blocked_by_active_predecessor = True
                continue
            candidates.append((record, samples))
        return candidates, blocked_by_active_predecessor

    @staticmethod
    def _score(
        new_view: GalleryViewV1, predecessor_samples: tuple[GallerySampleV1, ...]
    ) -> GalleryPairScoreV1:
        return score_gallery_pair_v1(
            [sample.embedding for sample in new_view.samples[-8:]],
            [sample.embedding for sample in predecessor_samples[-8:]],
        )

    def _should_defer_active_overlap(
        self, camera_id: str, raw_track_id: str, cycle: int
    ) -> bool:
        key = (camera_id, raw_track_id)
        first = self._deferred_first_cycle.setdefault(key, int(cycle))
        age = int(cycle) - int(first)
        if age < self.config.active_overlap_grace_cycles:
            self.active_overlap_deferred_total += 1
            self.pending_total += 1
            return True
        self._deferred_first_cycle.pop(key, None)
        return False

    def refresh(self, cycle: int, now_ns: int | None = None) -> None:
        started = time.perf_counter_ns()
        current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        views = self.snapshot_provider()
        active_snapshot = {
            str(camera): frozenset(str(track) for track in tracks)
            for camera, tracks in self.active_provider().items()
        }
        views_by_key = {
            (str(view.camera_id), str(view.local_track_id)): view for view in views
        }
        with self._lock:
            cfg = self.config
            self._repair_active_collisions(
                cycle=int(cycle),
                now_ns=current_ns,
                active_snapshot=active_snapshot,
            )
            seen_vote_keys: set[tuple[str, str, str]] = set()
            for camera_id in sorted(cfg.scoped_cameras):
                active = active_snapshot.get(camera_id, frozenset())
                new_views = [
                    view
                    for (camera, raw_track_id), view in views_by_key.items()
                    if camera == camera_id
                    and raw_track_id in active
                    and (camera_id, raw_track_id) not in self._raw_to_stable
                    and len(view.samples) >= cfg.min_samples
                ]
                new_views.sort(key=lambda view: str(view.local_track_id))
                if not new_views:
                    continue
                candidate_map: dict[
                    str, list[tuple[_StableCameraTrackletV1, tuple[GallerySampleV1, ...]]]
                ] = {}
                active_blocked: dict[str, bool] = {}
                for view in new_views:
                    raw_track_id = str(view.local_track_id)
                    candidates, blocked = self._candidate_records(
                        camera_id, view, active, views_by_key
                    )
                    candidate_map[raw_track_id] = candidates
                    active_blocked[raw_track_id] = blocked

                unresolved_for_scoring: list[GalleryViewV1] = []
                for view in new_views:
                    raw_track_id = str(view.local_track_id)
                    if candidate_map[raw_track_id]:
                        unresolved_for_scoring.append(view)
                        continue
                    if active_blocked[raw_track_id]:
                        self.successor_rejected_active_predecessor += 1
                        self._diagnose(
                            timestamp_ns=current_ns,
                            cycle=cycle,
                            event="successor_rejected_active_predecessor",
                            camera_id=camera_id,
                            raw_track_id=raw_track_id,
                            stable_id="",
                        )
                        if self._should_defer_active_overlap(
                            camera_id, raw_track_id, int(cycle)
                        ):
                            continue
                        self.active_overlap_fallback_allocated_total += 1
                        self._allocate(camera_id, raw_track_id, current_ns, cycle)
                        continue
                    self._allocate(camera_id, raw_track_id, current_ns, cycle)
                if not unresolved_for_scoring:
                    continue

                scores: dict[tuple[str, str], GalleryPairScoreV1] = {}
                for view in unresolved_for_scoring:
                    raw_track_id = str(view.local_track_id)
                    for record, predecessor_samples in candidate_map[raw_track_id]:
                        score = self._score(view, predecessor_samples)
                        scores[(raw_track_id, record.stable_id)] = score
                        if score.status == "INSUFFICIENT":
                            self.insufficient_total += 1

                row_rank: dict[str, list[tuple[float, str]]] = {}
                column_rank: dict[str, list[tuple[float, str]]] = {}
                for (raw_track_id, stable_id), score in scores.items():
                    if score.status != "OK" or score.robust_score is None:
                        continue
                    value = float(score.robust_score)
                    if not math.isfinite(value):
                        continue
                    row_rank.setdefault(raw_track_id, []).append((value, stable_id))
                    column_rank.setdefault(stable_id, []).append((value, raw_track_id))
                for values in row_rank.values():
                    values.sort(key=lambda item: (-item[0], item[1]))
                for values in column_rank.values():
                    values.sort(key=lambda item: (-item[0], item[1]))

                for view in unresolved_for_scoring:
                    raw_track_id = str(view.local_track_id)
                    ranked = row_rank.get(raw_track_id, [])
                    if not ranked:
                        self.pending_total += 1
                        continue
                    best_score, stable_id = ranked[0]
                    second_score = ranked[1][0] if len(ranked) > 1 else None
                    reverse = column_rank.get(stable_id, [])
                    reciprocal = bool(reverse and reverse[0][1] == raw_track_id)
                    reverse_second = reverse[1][0] if len(reverse) > 1 else None
                    score = scores[(raw_track_id, stable_id)]
                    margins = [
                        best_score - competitor
                        for competitor in (second_score, reverse_second)
                        if competitor is not None
                    ]
                    minimum_margin = min(margins) if margins else None
                    self.successor_reid_score = float(best_score)
                    self.successor_margin = minimum_margin
                    self._diagnose(
                        timestamp_ns=current_ns,
                        cycle=cycle,
                        event="successor_reid_score",
                        camera_id=camera_id,
                        raw_track_id=raw_track_id,
                        stable_id=stable_id,
                        successor_reid_score=float(best_score),
                        successor_margin=minimum_margin,
                    )
                    self._diagnose(
                        timestamp_ns=current_ns,
                        cycle=cycle,
                        event="successor_margin",
                        camera_id=camera_id,
                        raw_track_id=raw_track_id,
                        stable_id=stable_id,
                        successor_reid_score=float(best_score),
                        successor_margin=minimum_margin,
                    )
                    if not reciprocal:
                        self.nonreciprocal_total += 1
                        self.pending_total += 1
                        continue
                    if (
                        best_score < cfg.min_robust_score
                        or score.support_ge_065 < cfg.min_support_ge_065
                    ):
                        self.low_score_total += 1
                        self.pending_total += 1
                        continue
                    if (
                        second_score is not None
                        and best_score - second_score < cfg.min_margin
                    ) or (
                        reverse_second is not None
                        and best_score - reverse_second < cfg.min_margin
                    ):
                        self.low_margin_total += 1
                        self.pending_total += 1
                        continue
                    vote_key = (camera_id, raw_track_id, stable_id)
                    seen_vote_keys.add(vote_key)
                    vote = self._votes.setdefault(vote_key, _ContinuityVoteV1())
                    vote.consecutive = (
                        vote.consecutive + 1 if vote.last_cycle == int(cycle) - 1 else 1
                    )
                    vote.last_cycle = int(cycle)
                    vote.best_score = max(vote.best_score, float(best_score))
                    if vote.consecutive < cfg.confirm_cycles:
                        self.pending_total += 1
                        continue
                    record = self._stable[stable_id]
                    active_predecessor = self._active_predecessor(
                        record, raw_track_id, active
                    )
                    if active_predecessor is not None:
                        self.successor_rejected_active_predecessor += 1
                        self._diagnose(
                            timestamp_ns=current_ns,
                            cycle=cycle,
                            event="successor_rejected_active_predecessor",
                            camera_id=camera_id,
                            raw_track_id=raw_track_id,
                            stable_id=stable_id,
                            predecessor_track_id=active_predecessor,
                            successor_reid_score=float(best_score),
                            successor_margin=minimum_margin,
                        )
                        self._votes.pop(vote_key, None)
                        self.pending_total += 1
                        continue
                    record.raw_members.add(raw_track_id)
                    record.updated_ns = current_ns
                    self._raw_to_stable[(camera_id, raw_track_id)] = stable_id
                    self._raw_mapping_cycle[(camera_id, raw_track_id)] = int(cycle)
                    self._deferred_first_cycle.pop((camera_id, raw_track_id), None)
                    self.stitched_total += 1

            for key in list(self._votes):
                if key not in seen_vote_keys and self._votes[key].last_cycle < int(cycle):
                    del self._votes[key]
            for key in list(self._deferred_first_cycle):
                camera_id, raw_track_id = key
                if key in self._raw_to_stable or raw_track_id not in active_snapshot.get(
                    camera_id, frozenset()
                ):
                    self._deferred_first_cycle.pop(key, None)
            active_members = set()
            for camera_id, raw_tracks in active_snapshot.items():
                for raw_track_id in raw_tracks:
                    stable_id = self._raw_to_stable.get((camera_id, raw_track_id))
                    if stable_id is not None:
                        active_members.add((camera_id, stable_id))
            self._active_camera_members = frozenset(active_members)
            self.refresh_ms.append(
                (time.perf_counter_ns() - started) / 1_000_000.0
            )

    def canonical_track_id(self, camera_id: str, raw_track_id: str) -> str | None:
        camera = str(camera_id)
        raw = str(raw_track_id)
        if camera not in self.config.scoped_cameras:
            return raw
        with self._lock:
            return self._raw_to_stable.get((camera, raw))

    def canonicalize_proposal(
        self, row: SameRoomPairDiagnosticV1
    ) -> SameRoomPairDiagnosticV1 | None:
        track_a = self.canonical_track_id(row.camera_a, row.track_a)
        track_b = self.canonical_track_id(row.camera_b, row.track_b)
        if track_a is None or track_b is None:
            with self._lock:
                self.suppressed_total += 1
            return None
        if track_a == row.track_a and track_b == row.track_b:
            return row
        return replace(row, track_a=track_a, track_b=track_b)

    def active_camera_members(self) -> frozenset[tuple[str, str]]:
        with self._lock:
            return self._active_camera_members

    def diagnostics(self) -> tuple[CameraTrackletDiagnosticV1, ...]:
        with self._lock:
            return tuple(self._diagnostics)

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "stable_ids": len(self._stable),
                "raw_mapped": len(self._raw_to_stable),
                "stitched_total": self.stitched_total,
                "pending_votes": len(self._votes),
                "deferred_allocations": len(self._deferred_first_cycle),
                "pending_total": self.pending_total,
                "suppressed_total": self.suppressed_total,
                "low_score_total": self.low_score_total,
                "low_margin_total": self.low_margin_total,
                "nonreciprocal_total": self.nonreciprocal_total,
                "overlap_reject_total": self.overlap_reject_total,
                "insufficient_total": self.insufficient_total,
                "active_overlap_deferred_total": self.active_overlap_deferred_total,
                "active_overlap_fallback_allocated_total": self.active_overlap_fallback_allocated_total,
                "same_camera_active_ct_collision": self.same_camera_active_ct_collision,
                "successor_reid_score": self.successor_reid_score or 0.0,
                "successor_margin": self.successor_margin or 0.0,
                "successor_rejected_active_predecessor": self.successor_rejected_active_predecessor,
                "refresh_p50_ms": self._pct(self.refresh_ms, 0.50),
                "refresh_p95_ms": self._pct(self.refresh_ms, 0.95),
            }
