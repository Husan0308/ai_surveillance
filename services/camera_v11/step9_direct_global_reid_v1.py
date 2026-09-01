from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .step4_reid_gallery_v1 import GallerySampleV1, GalleryViewV1
from .step4_reid_pair_scorer_v1 import GalleryPairScoreV1, score_gallery_pair_v1


@dataclass(frozen=True)
class DirectGlobalReIDConfigV1:
    """Conservative direct local-track -> persistent-global ReID association."""

    scoped_cameras: tuple[str, ...] = ("CAM-01", "CAM-04")
    min_track_samples: int = 3
    min_robust_score: float = 0.74
    min_top3_mean: float = 0.78
    min_median_best: float = 0.70
    min_support_ge_070: int = 3
    min_margin: float = 0.06
    plausible_existing_score: float = 0.55
    confirm_evidence: int = 2
    new_identity_evidence: int = 2
    recent_age_sec: float = 12.0
    global_memory_sec: float = 1800.0
    max_global_samples: int = 8

    def __post_init__(self) -> None:
        if self.min_track_samples < 3:
            raise ValueError("min_track_samples must be >=3")
        if self.confirm_evidence < 1 or self.new_identity_evidence < 1:
            raise ValueError("evidence counts must be positive")
        if self.max_global_samples != 8:
            raise ValueError("max_global_samples must remain 8 for Step4 scorer")
        for name in (
            "min_robust_score",
            "min_top3_mean",
            "min_median_best",
            "min_margin",
            "plausible_existing_score",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid {name}")
        if self.plausible_existing_score >= self.min_robust_score:
            raise ValueError("plausible_existing_score must be below min_robust_score")
        if self.min_support_ge_070 < 1:
            raise ValueError("min_support_ge_070 must be positive")
        if self.recent_age_sec <= 0 or self.global_memory_sec <= 0:
            raise ValueError("memory windows must be positive")


@dataclass
class _GlobalIdentityV1:
    global_id: str
    created_ns: int
    last_seen_ns: int
    samples: list[GallerySampleV1] = field(default_factory=list)
    current_members: dict[str, str] = field(default_factory=dict)
    aliases: set[tuple[str, str]] = field(default_factory=set)
    sample_sequences: set[int] = field(default_factory=set)


@dataclass
class _VoteV1:
    target: str
    count: int
    last_signature: tuple[int, int]


@dataclass(frozen=True)
class DirectGlobalDecisionV1:
    camera_id: str
    local_track_id: str
    global_id: str
    decision: str
    robust_score: float | None = None
    margin: float | None = None


def _key(view: GalleryViewV1) -> tuple[str, str]:
    return str(view.camera_id), str(view.local_track_id)


def _signature(view: GalleryViewV1) -> tuple[int, int]:
    if not view.samples:
        return (0, 0)
    return (len(view.samples), max(int(row.sample_sequence) for row in view.samples))


def _score_view_to_samples(
    view: GalleryViewV1, samples: list[GallerySampleV1]
) -> GalleryPairScoreV1:
    return score_gallery_pair_v1(
        [row.embedding for row in view.samples[-8:]],
        [row.embedding for row in samples[-8:]],
    )


class DirectGlobalReIDResolverV1:
    """Persistent appearance-first identity resolver.

    Safety properties:
    - at most one active member per camera per global identity;
    - no cross-global merges;
    - no forced match when evidence is weak or ambiguous;
    - a lost/recreated local track can re-associate to historical global memory;
    - ambiguous evidence remains pending instead of spawning duplicate identities;
    - only accepted tracks update global appearance memory.
    """

    def __init__(self, config: DirectGlobalReIDConfigV1 | None = None) -> None:
        self.config = config or DirectGlobalReIDConfigV1()
        self._lock = threading.RLock()
        self._globals: dict[str, _GlobalIdentityV1] = {}
        self._track_to_global: dict[tuple[str, str], str] = {}
        self._next_global = 1
        self._match_votes: dict[tuple[str, str], _VoteV1] = {}
        self._new_votes: dict[tuple[str, str], _VoteV1] = {}
        self._pair_votes: dict[tuple[tuple[str, str], tuple[str, str]], _VoteV1] = {}
        self._decisions: list[DirectGlobalDecisionV1] = []

        self.created_total = 0
        self.reassociated_total = 0
        self.paired_create_total = 0
        self.no_match_total = 0
        self.low_score_total = 0
        self.low_margin_total = 0
        self.ambiguous_pending_total = 0
        self.same_camera_reject_total = 0
        self.active_collision_repairs = 0

    def _new_id(self) -> str:
        value = f"GID-{self._next_global:06d}"
        self._next_global += 1
        return value

    @staticmethod
    def _representative_samples(samples: list[GallerySampleV1]) -> list[GallerySampleV1]:
        """Keep eight quality+diversity representatives without averaging identities."""
        if len(samples) <= 8:
            return list(samples)
        ordered = sorted(
            samples,
            key=lambda row: (
                -float(row.quality_score),
                -int(row.timestamp_ns),
                int(row.sample_sequence),
            ),
        )
        selected = [ordered[0]]
        remaining = ordered[1:]
        while remaining and len(selected) < 8:
            def utility(row: GallerySampleV1) -> tuple[float, float, int]:
                nearest = max(
                    float(np.dot(row.embedding, pick.embedding)) for pick in selected
                )
                diversity = max(0.0, 1.0 - nearest)
                value = 0.70 * diversity + 0.30 * max(
                    0.0, min(1.0, float(row.quality_score))
                )
                return (value, float(row.quality_score), int(row.timestamp_ns))

            best_index = max(
                range(len(remaining)), key=lambda index: utility(remaining[index])
            )
            selected.append(remaining.pop(best_index))
        return selected

    def _add_samples(self, record: _GlobalIdentityV1, view: GalleryViewV1) -> None:
        changed = False
        for sample in view.samples:
            sequence = int(sample.sample_sequence)
            if sequence in record.sample_sequences:
                continue
            record.sample_sequences.add(sequence)
            record.samples.append(sample)
            changed = True
        if changed:
            record.samples = self._representative_samples(record.samples)
            record.sample_sequences = {
                int(row.sample_sequence) for row in record.samples
            }

    def _bind(
        self,
        key: tuple[str, str],
        global_id: str,
        view: GalleryViewV1,
        now_ns: int,
    ) -> bool:
        camera_id, local_track_id = key
        record = self._globals[global_id]
        current = record.current_members.get(camera_id)
        if current is not None and current != local_track_id:
            return False
        self._track_to_global[key] = global_id
        record.current_members[camera_id] = local_track_id
        record.aliases.add(key)
        record.last_seen_ns = max(
            record.last_seen_ns, int(now_ns), int(view.last_seen_ns)
        )
        self._add_samples(record, view)
        return True

    def _create(
        self, views: tuple[GalleryViewV1, ...], now_ns: int, decision: str
    ) -> str:
        global_id = self._new_id()
        record = _GlobalIdentityV1(global_id, int(now_ns), int(now_ns))
        self._globals[global_id] = record
        for view in views:
            self._bind(_key(view), global_id, view, now_ns)
        self.created_total += 1
        if len(views) > 1:
            self.paired_create_total += 1
        for view in views:
            self._decisions.append(
                DirectGlobalDecisionV1(
                    view.camera_id, view.local_track_id, global_id, decision
                )
            )
        return global_id

    def _quality_ok(self, score: GalleryPairScoreV1) -> bool:
        return bool(
            score.status == "OK"
            and score.robust_score is not None
            and score.top3_mean is not None
            and score.median_of_best_matches is not None
            and float(score.robust_score) >= self.config.min_robust_score
            and float(score.top3_mean) >= self.config.min_top3_mean
            and float(score.median_of_best_matches) >= self.config.min_median_best
            and int(score.support_ge_070) >= self.config.min_support_ge_070
        )

    def _advance_vote(
        self,
        table: dict,
        key,
        target: str,
        signature: tuple[int, int],
    ) -> int:
        vote = table.get(key)
        if vote is None or vote.target != target:
            table[key] = _VoteV1(target=target, count=1, last_signature=signature)
            return 1
        if vote.last_signature == signature:
            return vote.count
        vote.count += 1
        vote.last_signature = signature
        return vote.count

    def _repair_active_collisions(
        self, active: dict[str, frozenset[str]]
    ) -> None:
        for record in self._globals.values():
            for camera_id in tuple(record.current_members):
                member = record.current_members[camera_id]
                if member not in active.get(camera_id, frozenset()):
                    record.current_members.pop(camera_id, None)

        grouped: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for key, global_id in self._track_to_global.items():
            camera_id, local_track_id = key
            if local_track_id in active.get(camera_id, frozenset()):
                grouped.setdefault((global_id, camera_id), []).append(key)
        for (global_id, camera_id), keys in grouped.items():
            if len(keys) <= 1:
                continue
            record = self._globals.get(global_id)
            keep_track = (
                record.current_members.get(camera_id) if record is not None else None
            )
            keep = next(
                (key for key in keys if key[1] == keep_track), sorted(keys)[0]
            )
            for key in keys:
                if key == keep:
                    continue
                self._track_to_global.pop(key, None)
                self.active_collision_repairs += 1
                self._decisions.append(
                    DirectGlobalDecisionV1(
                        key[0], key[1], "", "same_camera_collision_split"
                    )
                )

    def _candidate_globals(
        self,
        view: GalleryViewV1,
        active: dict[str, frozenset[str]],
        now_ns: int,
    ) -> list[tuple[float, str, GalleryPairScoreV1]]:
        rows: list[tuple[float, str, GalleryPairScoreV1]] = []
        camera_id = str(view.camera_id)
        track_id = str(view.local_track_id)
        for global_id, record in self._globals.items():
            if int(now_ns) - int(record.last_seen_ns) > int(
                self.config.global_memory_sec * 1e9
            ):
                continue
            current = record.current_members.get(camera_id)
            if (
                current is not None
                and current != track_id
                and current in active.get(camera_id, frozenset())
            ):
                self.same_camera_reject_total += 1
                continue
            if len(record.samples) < 3:
                continue
            score = _score_view_to_samples(view, record.samples)
            if score.status == "OK" and score.robust_score is not None:
                rows.append((float(score.robust_score), global_id, score))
        rows.sort(key=lambda row: (-row[0], row[1]))
        return rows

    def _try_existing(
        self,
        view: GalleryViewV1,
        active: dict[str, frozenset[str]],
        now_ns: int,
    ) -> str:
        """Return bound, blocked, or none.

        blocked means an existing identity is appearance-plausible but evidence is
        not yet safe enough.  Such a track must remain pending; creating a second
        GID here is exactly how one physical person acquires repeated identities.
        """
        key = _key(view)
        ranked = self._candidate_globals(view, active, now_ns)
        if not ranked:
            return "none"
        best_value, best_id, best_score = ranked[0]
        second_value = ranked[1][0] if len(ranked) > 1 else None
        margin = None if second_value is None else best_value - second_value

        if not self._quality_ok(best_score):
            self.low_score_total += 1
            self.no_match_total += 1
            self._decisions.append(
                DirectGlobalDecisionV1(
                    key[0],
                    key[1],
                    "",
                    "no_match_low_score",
                    best_value,
                    margin,
                )
            )
            if best_value >= self.config.plausible_existing_score:
                self.ambiguous_pending_total += 1
                return "blocked"
            return "none"

        if margin is not None and margin < self.config.min_margin:
            self.low_margin_total += 1
            self.no_match_total += 1
            self.ambiguous_pending_total += 1
            self._decisions.append(
                DirectGlobalDecisionV1(
                    key[0],
                    key[1],
                    "",
                    "no_match_ambiguous",
                    best_value,
                    margin,
                )
            )
            return "blocked"

        count = self._advance_vote(
            self._match_votes, key, best_id, _signature(view)
        )
        if count < self.config.confirm_evidence:
            self.ambiguous_pending_total += 1
            return "blocked"
        if not self._bind(key, best_id, view, now_ns):
            self.same_camera_reject_total += 1
            return "blocked"
        self.reassociated_total += 1
        self._match_votes.pop(key, None)
        self._new_votes.pop(key, None)
        self._decisions.append(
            DirectGlobalDecisionV1(
                key[0], key[1], best_id, "reassociate", best_value, margin
            )
        )
        return "bound"

    def _try_pair_unknowns(
        self,
        views: list[GalleryViewV1],
        now_ns: int,
    ) -> set[tuple[str, str]]:
        used: set[tuple[str, str]] = set()
        cameras = self.config.scoped_cameras
        if len(cameras) != 2:
            return used
        left = [view for view in views if view.camera_id == cameras[0]]
        right = [view for view in views if view.camera_id == cameras[1]]
        if not left or not right:
            return used

        scores: dict[tuple[int, int], GalleryPairScoreV1] = {}
        row_rank: dict[int, list[tuple[float, int]]] = {}
        col_rank: dict[int, list[tuple[float, int]]] = {}
        for i, first in enumerate(left):
            for j, second in enumerate(right):
                score = score_gallery_pair_v1(
                    [row.embedding for row in first.samples[-8:]],
                    [row.embedding for row in second.samples[-8:]],
                )
                scores[(i, j)] = score
                if score.status == "OK" and score.robust_score is not None:
                    value = float(score.robust_score)
                    row_rank.setdefault(i, []).append((value, j))
                    col_rank.setdefault(j, []).append((value, i))
        for values in row_rank.values():
            values.sort(key=lambda row: (-row[0], row[1]))
        for values in col_rank.values():
            values.sort(key=lambda row: (-row[0], row[1]))

        for i, first in enumerate(left):
            ranked = row_rank.get(i, [])
            if not ranked:
                continue
            best_value, j = ranked[0]
            reverse = col_rank.get(j, [])
            if not reverse or reverse[0][1] != i:
                continue
            score = scores[(i, j)]
            if not self._quality_ok(score):
                continue
            row_margin = (
                None if len(ranked) < 2 else best_value - ranked[1][0]
            )
            col_margin = (
                None if len(reverse) < 2 else best_value - reverse[1][0]
            )
            margins = [
                value for value in (row_margin, col_margin) if value is not None
            ]
            margin = min(margins) if margins else None
            if margin is not None and margin < self.config.min_margin:
                continue
            second = right[j]
            first_key, second_key = _key(first), _key(second)
            if first_key in used or second_key in used:
                continue
            pair_key = tuple(sorted((first_key, second_key)))
            signature = (
                max(_signature(first)[1], _signature(second)[1]),
                len(first.samples) + len(second.samples),
            )
            count = self._advance_vote(
                self._pair_votes, pair_key, "pair", signature
            )
            if count < self.config.confirm_evidence:
                continue
            self._create((first, second), now_ns, "paired_create")
            used.add(first_key)
            used.add(second_key)
            self._pair_votes.pop(pair_key, None)
        return used

    def resolve(
        self,
        views: tuple[GalleryViewV1, ...],
        active_by_camera: dict[str, frozenset[str]],
        *,
        now_ns: int | None = None,
    ) -> tuple[DirectGlobalDecisionV1, ...]:
        current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        recent_cutoff = current_ns - int(self.config.recent_age_sec * 1e9)
        active = {
            str(camera): frozenset(str(track) for track in tracks)
            for camera, tracks in active_by_camera.items()
        }
        eligible = [
            view
            for view in views
            if view.camera_id in self.config.scoped_cameras
            and view.local_track_id in active.get(view.camera_id, frozenset())
            and len(view.samples) >= self.config.min_track_samples
            and int(view.last_seen_ns) >= recent_cutoff
        ]
        eligible.sort(key=_key)

        with self._lock:
            start_decisions = len(self._decisions)
            self._repair_active_collisions(active)
            by_key = {_key(view): view for view in eligible}

            for key, view in by_key.items():
                global_id = self._track_to_global.get(key)
                if global_id is None or global_id not in self._globals:
                    continue
                record = self._globals[global_id]
                current = record.current_members.get(key[0])
                if current is None or current == key[1]:
                    record.current_members[key[0]] = key[1]
                    record.last_seen_ns = max(
                        record.last_seen_ns, current_ns, int(view.last_seen_ns)
                    )
                    self._add_samples(record, view)

            unknown = [
                view
                for view in eligible
                if _key(view) not in self._track_to_global
            ]

            safe_for_new: list[GalleryViewV1] = []
            for view in unknown:
                status = self._try_existing(view, active, current_ns)
                if status == "none":
                    safe_for_new.append(view)

            paired = self._try_pair_unknowns(safe_for_new, current_ns)

            for view in safe_for_new:
                key = _key(view)
                if key in paired or key in self._track_to_global:
                    continue
                signature = _signature(view)
                count = self._advance_vote(
                    self._new_votes, key, "new", signature
                )
                if count < self.config.new_identity_evidence:
                    continue
                self._create((view,), current_ns, "single_camera_create")
                self._new_votes.pop(key, None)

            for record in self._globals.values():
                for camera_id in tuple(record.current_members):
                    local_track_id = record.current_members[camera_id]
                    if local_track_id not in active.get(
                        camera_id, frozenset()
                    ):
                        record.current_members.pop(camera_id, None)

            return tuple(self._decisions[start_decisions:])

    def global_for_track(
        self, camera_id: str, local_track_id: str
    ) -> str | None:
        with self._lock:
            return self._track_to_global.get(
                (str(camera_id), str(local_track_id))
            )

    def records(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(
                {
                    "global_id": record.global_id,
                    "current_members": dict(
                        sorted(record.current_members.items())
                    ),
                    "aliases": tuple(sorted(record.aliases)),
                    "samples": len(record.samples),
                    "last_seen_ns": record.last_seen_ns,
                }
                for record in sorted(
                    self._globals.values(), key=lambda row: row.global_id
                )
            )

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            active_members = sum(
                len(row.current_members) for row in self._globals.values()
            )
            return {
                "global_ids": len(self._globals),
                "active_members": active_members,
                "created_total": self.created_total,
                "reassociated_total": self.reassociated_total,
                "paired_create_total": self.paired_create_total,
                "no_match_total": self.no_match_total,
                "low_score_total": self.low_score_total,
                "low_margin_total": self.low_margin_total,
                "ambiguous_pending_total": self.ambiguous_pending_total,
                "same_camera_reject_total": self.same_camera_reject_total,
                "active_collision_repairs": self.active_collision_repairs,
            }
