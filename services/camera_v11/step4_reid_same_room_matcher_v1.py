from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .step4_reid_gallery_v1 import GalleryViewV1
from .step4_reid_pair_scorer_v1 import GalleryPairScoreV1, score_gallery_pair_v1
from .step4_reid_same_room_evidence_padded_v1 import score_gallery_matrix_step3_padded_exact_v1


MATCH_PROPOSED = "MATCH_PROPOSED"
INSUFFICIENT = "INSUFFICIENT"
NON_RECIPROCAL = "NON_RECIPROCAL"
LOW_MARGIN = "LOW_MARGIN"
LOW_SCORE = "LOW_SCORE"
ASSIGNMENT_CONFLICT = "ASSIGNMENT_CONFLICT"
STALE = "STALE"
INVALID = "INVALID"

MATCH_STATUSES = (
    MATCH_PROPOSED,
    INSUFFICIENT,
    NON_RECIPROCAL,
    LOW_MARGIN,
    LOW_SCORE,
    ASSIGNMENT_CONFLICT,
    STALE,
    INVALID,
)


@dataclass(frozen=True)
class SameRoomMatcherConfigV1:
    recent_age_sec: float = 12.0
    min_robust_score: float | None = None
    min_row_margin: float | None = None
    min_column_margin: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.recent_age_sec)) or self.recent_age_sec <= 0:
            raise ValueError("recent_age_sec must be finite and positive")
        for name in ("min_robust_score", "min_row_margin", "min_column_margin"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or None")


@dataclass(frozen=True)
class SameRoomPairDiagnosticV1:
    room: str
    camera_a: str
    track_a: str
    camera_b: str
    track_b: str
    samples_a: int
    samples_b: int
    robust_score: float | None
    max_score: float | None
    top3_mean: float | None
    median: float | None
    a_best_mean: float | None
    a_best_min: float | None
    b_best_mean: float | None
    b_best_min: float | None
    support_ge_050: int
    support_ge_055: int
    support_ge_060: int
    support_ge_065: int
    support_ge_070: int
    support_ge_075: int
    support_ge_080: int
    row_best: float | None
    row_second: float | None
    row_margin: float | None
    column_best: float | None
    column_second: float | None
    column_margin: float | None
    reciprocal: bool
    assigned: bool
    status: str
    reason: str


@dataclass(frozen=True)
class SameRoomMatrixResultV1:
    room: str
    camera_a: str
    camera_b: str
    diagnostics: tuple[SameRoomPairDiagnosticV1, ...]
    elapsed_ms: float

    @property
    def proposals(self) -> tuple[SameRoomPairDiagnosticV1, ...]:
        return tuple(row for row in self.diagnostics if row.status == MATCH_PROPOSED)


PairScorerV1 = Callable[[GalleryViewV1, GalleryViewV1], GalleryPairScoreV1]


def _default_pair_scorer(
    first: GalleryViewV1, second: GalleryViewV1
) -> GalleryPairScoreV1:
    return score_gallery_pair_v1(
        [sample.embedding for sample in first.samples],
        [sample.embedding for sample in second.samples],
    )


def _track_key(view: GalleryViewV1) -> tuple[str, str]:
    return str(view.camera_id), str(view.local_track_id)


def _second_and_margin(values: Sequence[float], best: float) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    ordered = sorted((float(value) for value in values), reverse=True)
    second = ordered[1]
    return second, float(best - second)


def match_same_room_camera_pair_v1(
    views_a: Sequence[GalleryViewV1],
    views_b: Sequence[GalleryViewV1],
    camera_rooms: dict[str, str],
    *,
    now_ns: int | None = None,
    config: SameRoomMatcherConfigV1 | None = None,
    pair_scorer: PairScorerV1 | None = None,
) -> SameRoomMatrixResultV1:
    """Build one same-room cross-camera matrix and return shadow proposals.

    Inputs are copied into canonical order, Step-3 evidence is never modified, and
    assignment is restricted to reciprocal pairs that satisfy enabled diagnostic
    thresholds. ``None`` thresholds are the live no-merge-calibration mode.
    """

    started = time.perf_counter()
    settings = config or SameRoomMatcherConfigV1()
    scorer = pair_scorer or _default_pair_scorer
    left = tuple(sorted(views_a, key=_track_key))
    right = tuple(sorted(views_b, key=_track_key))
    cameras_a = {view.camera_id for view in left}
    cameras_b = {view.camera_id for view in right}
    if len(cameras_a) != 1 or len(cameras_b) != 1:
        raise ValueError("each matrix side must contain exactly one camera")
    camera_a = next(iter(cameras_a))
    camera_b = next(iter(cameras_b))
    if camera_a == camera_b:
        raise ValueError("same-camera matrices are forbidden")
    if camera_a not in camera_rooms or camera_b not in camera_rooms:
        raise ValueError("camera room metadata is required")
    room_a = str(camera_rooms[camera_a])
    room_b = str(camera_rooms[camera_b])
    if not room_a or room_a != room_b:
        raise ValueError("cross-room matrices are forbidden")
    if len({_track_key(view) for view in left}) != len(left):
        raise ValueError("duplicate track on matrix rows")
    if len({_track_key(view) for view in right}) != len(right):
        raise ValueError("duplicate track on matrix columns")

    current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
    recent_ns = int(float(settings.recent_age_sec) * 1e9)
    scores: dict[tuple[int, int], GalleryPairScoreV1] = {}
    structural: dict[tuple[int, int], str] = {}
    batch_scores: dict[tuple[int, int], GalleryPairScoreV1] = {}
    if pair_scorer is None:
        active_rows = [
            index
            for index, view in enumerate(left)
            if current_ns - int(view.last_seen_ns) <= recent_ns
        ]
        active_columns = [
            index
            for index, view in enumerate(right)
            if current_ns - int(view.last_seen_ns) <= recent_ns
        ]
        batched = score_gallery_matrix_step3_padded_exact_v1(
            [left[index] for index in active_rows],
            [right[index] for index in active_columns],
        )
        batch_scores = {
            (active_rows[row_index], active_columns[column_index]): value
            for (row_index, column_index), value in batched.items()
        }
    for row_index, row in enumerate(left):
        for column_index, column in enumerate(right):
            key = (row_index, column_index)
            if (
                current_ns - int(row.last_seen_ns) > recent_ns
                or current_ns - int(column.last_seen_ns) > recent_ns
            ):
                structural[key] = STALE
                scores[key] = GalleryPairScoreV1(
                    status="INVALID", pair_count=0, reason="track exceeds recent-age timeout"
                )
                continue
            score = batch_scores[key] if pair_scorer is None else scorer(row, column)
            scores[key] = score
            if score.status == "OK" and score.robust_score is not None and math.isfinite(score.robust_score):
                structural[key] = "VALID"
            elif score.status == "INSUFFICIENT":
                structural[key] = INSUFFICIENT
            else:
                structural[key] = INVALID

    valid_values_by_row: dict[int, list[tuple[float, int]]] = {}
    valid_values_by_column: dict[int, list[tuple[float, int]]] = {}
    for (row_index, column_index), state in structural.items():
        if state != "VALID":
            continue
        value = float(scores[(row_index, column_index)].robust_score)
        valid_values_by_row.setdefault(row_index, []).append((value, column_index))
        valid_values_by_column.setdefault(column_index, []).append((value, row_index))

    # Canonical track ordering supplies the deterministic tie break.
    row_best_index: dict[int, int] = {}
    row_stats: dict[int, tuple[float, float | None, float | None]] = {}
    for row_index, values in valid_values_by_row.items():
        best_value, best_column = sorted(values, key=lambda item: (-item[0], item[1]))[0]
        second, margin = _second_and_margin([value for value, _ in values], best_value)
        row_best_index[row_index] = best_column
        row_stats[row_index] = (best_value, second, margin)
    column_best_index: dict[int, int] = {}
    column_stats: dict[int, tuple[float, float | None, float | None]] = {}
    for column_index, values in valid_values_by_column.items():
        best_value, best_row = sorted(values, key=lambda item: (-item[0], item[1]))[0]
        second, margin = _second_and_margin([value for value, _ in values], best_value)
        column_best_index[column_index] = best_row
        column_stats[column_index] = (best_value, second, margin)

    eligible: set[tuple[int, int]] = set()
    pre_status: dict[tuple[int, int], tuple[str, str, bool]] = {}
    for key, state in structural.items():
        row_index, column_index = key
        if state != "VALID":
            reason = scores[key].reason or state.lower()
            pre_status[key] = (state, reason, False)
            continue
        reciprocal = (
            row_best_index.get(row_index) == column_index
            and column_best_index.get(column_index) == row_index
        )
        score = float(scores[key].robust_score)
        row_margin = row_stats[row_index][2]
        column_margin = column_stats[column_index][2]
        if not reciprocal:
            pre_status[key] = (NON_RECIPROCAL, "not reciprocal row/column best", False)
        elif settings.min_robust_score is not None and score < settings.min_robust_score:
            pre_status[key] = (LOW_SCORE, "robust score below configured diagnostic threshold", True)
        elif (
            settings.min_row_margin is not None
            and row_margin is not None
            and row_margin < settings.min_row_margin
        ) or (
            settings.min_column_margin is not None
            and column_margin is not None
            and column_margin < settings.min_column_margin
        ):
            pre_status[key] = (LOW_MARGIN, "row or column margin below configured diagnostic threshold", True)
        else:
            eligible.add(key)
            pre_status[key] = (ASSIGNMENT_CONFLICT, "eligible but not selected by assignment", True)

    assigned: set[tuple[int, int]] = set()
    if left and right and eligible:
        sentinel = -1.0e12
        weights = np.full((len(left), len(right)), sentinel, dtype=np.float64)
        for row_index, column_index in eligible:
            weights[row_index, column_index] = float(scores[(row_index, column_index)].robust_score)
        assignment_rows, assignment_columns = linear_sum_assignment(weights, maximize=True)
        for row_index, column_index in zip(assignment_rows.tolist(), assignment_columns.tolist(), strict=True):
            key = (int(row_index), int(column_index))
            if key in eligible:
                assigned.add(key)

    diagnostics: list[SameRoomPairDiagnosticV1] = []
    for row_index, row in enumerate(left):
        for column_index, column in enumerate(right):
            key = (row_index, column_index)
            score = scores[key]
            status, reason, reciprocal = pre_status[key]
            is_assigned = key in assigned
            if is_assigned:
                status = MATCH_PROPOSED
                reason = "reciprocal structurally eligible maximum-weight assignment"
            row_stat = row_stats.get(row_index, (None, None, None))
            column_stat = column_stats.get(column_index, (None, None, None))
            diagnostics.append(
                SameRoomPairDiagnosticV1(
                    room=room_a,
                    camera_a=camera_a,
                    track_a=str(row.local_track_id),
                    camera_b=camera_b,
                    track_b=str(column.local_track_id),
                    samples_a=len(row.samples),
                    samples_b=len(column.samples),
                    robust_score=score.robust_score,
                    max_score=score.max_score,
                    top3_mean=score.top3_mean,
                    median=score.median_score,
                    a_best_mean=score.a_best_mean,
                    a_best_min=score.a_best_min,
                    b_best_mean=score.b_best_mean,
                    b_best_min=score.b_best_min,
                    support_ge_050=score.support_ge_050,
                    support_ge_055=score.support_ge_055,
                    support_ge_060=score.support_ge_060,
                    support_ge_065=score.support_ge_065,
                    support_ge_070=score.support_ge_070,
                    support_ge_075=score.support_ge_075,
                    support_ge_080=score.support_ge_080,
                    row_best=row_stat[0],
                    row_second=row_stat[1],
                    row_margin=row_stat[2],
                    column_best=column_stat[0],
                    column_second=column_stat[1],
                    column_margin=column_stat[2],
                    reciprocal=reciprocal,
                    assigned=is_assigned,
                    status=status,
                    reason=reason,
                )
            )
    return SameRoomMatrixResultV1(
        room=room_a,
        camera_a=camera_a,
        camera_b=camera_b,
        diagnostics=tuple(diagnostics),
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )
