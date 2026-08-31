from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


SUPPORT_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


@dataclass(frozen=True)
class GalleryPairScoreV1:
    status: str
    pair_count: int
    max_score: float | None = None
    top2_mean: float | None = None
    top3_mean: float | None = None
    top5_mean: float | None = None
    median_score: float | None = None
    mean_score: float | None = None
    p75_score: float | None = None
    p90_score: float | None = None
    support_ge_050: int = 0
    support_ge_055: int = 0
    support_ge_060: int = 0
    support_ge_065: int = 0
    support_ge_070: int = 0
    support_ge_075: int = 0
    support_ge_080: int = 0
    a_best_mean: float | None = None
    a_best_min: float | None = None
    b_best_mean: float | None = None
    b_best_min: float | None = None
    median_of_best_matches: float | None = None
    robust_score: float | None = None
    reason: str = ""


def _matrix(
    gallery: Sequence[np.ndarray],
    *,
    expected_dimension: int,
) -> np.ndarray:
    """Build one float64 matrix with bulk validation and no per-row copies.

    The scorer's numerical contract remains float64.  The old implementation
    converted, validated, copied, and finally stacked every row separately.
    Here we allocate the final matrix once, fill it, then validate finiteness
    and L2 norms in vectorized NumPy operations.  This keeps the same accepted
    input semantics while reducing Python/allocation overhead in the shadow
    worker hot path.
    """

    count = len(gallery)
    if count > 8:
        raise ValueError(f"gallery exceeds capacity: {count}>8")
    if count == 0:
        return np.empty((0, expected_dimension), dtype=np.float64)

    rows = np.empty((count, expected_dimension), dtype=np.float64)
    for index, embedding in enumerate(gallery):
        vector = np.asarray(embedding).reshape(-1)
        if vector.size != expected_dimension:
            raise ValueError(
                f"embedding[{index}] dimension={vector.size}, expected={expected_dimension}"
            )
        rows[index] = vector

    finite_rows = np.all(np.isfinite(rows), axis=1)
    if not finite_rows.all():
        index = int(np.flatnonzero(~finite_rows)[0])
        raise ValueError(f"embedding[{index}] is non-finite")

    norms = np.linalg.norm(rows, axis=1)
    invalid_norms = (~np.isfinite(norms)) | (np.abs(norms - 1.0) > 1e-3)
    if invalid_norms.any():
        index = int(np.flatnonzero(invalid_norms)[0])
        raise ValueError(
            f"embedding[{index}] is not L2-normalized norm={float(norms[index])}"
        )
    return rows


def _top_mean(sorted_descending: np.ndarray, count: int) -> float:
    return float(np.mean(sorted_descending[: min(count, sorted_descending.size)]))


def score_gallery_pair_v1(
    gallery_a: Sequence[np.ndarray],
    gallery_b: Sequence[np.ndarray],
    *,
    expected_dimension: int = 256,
) -> GalleryPairScoreV1:
    """Calculate deterministic symmetric diagnostics over the complete cosine matrix.

    The diagnostic robust score is fixed to:
      0.40 * top3_mean
    + 0.25 * median(concat(per-A best, per-B best))
    + 0.20 * p75(all pair scores)
    + 0.15 * max(all pair scores)

    It is deliberately not an identity decision threshold or assignment signal.
    """

    if len(gallery_a) < 3 or len(gallery_b) < 3:
        return GalleryPairScoreV1(
            status="INSUFFICIENT",
            pair_count=len(gallery_a) * len(gallery_b),
            reason="each gallery requires at least 3 embeddings",
        )
    try:
        a = _matrix(gallery_a, expected_dimension=int(expected_dimension))
        b = _matrix(gallery_b, expected_dimension=int(expected_dimension))
        scores = a @ b.T
        if scores.shape != (len(gallery_a), len(gallery_b)):
            raise ValueError(f"unexpected cosine matrix shape={scores.shape}")
        if not np.isfinite(scores).all():
            raise ValueError("cosine matrix contains non-finite scores")
    except (TypeError, ValueError) as exc:
        return GalleryPairScoreV1(
            status="INVALID",
            pair_count=0,
            reason=f"{type(exc).__name__}:{exc}",
        )

    flat = scores.reshape(-1)
    ordered = np.sort(flat)[::-1]
    a_best = np.max(scores, axis=1)
    b_best = np.max(scores, axis=0)
    all_best = np.concatenate((a_best, b_best))
    supports = {
        threshold: int(np.count_nonzero(flat >= threshold))
        for threshold in SUPPORT_THRESHOLDS
    }
    top3_mean = _top_mean(ordered, 3)
    median_best = float(np.median(all_best))
    median_score, p75, p90 = (
        float(value) for value in np.percentile(flat, (50.0, 75.0, 90.0))
    )
    max_score = float(ordered[0])
    robust_score = (
        0.40 * top3_mean
        + 0.25 * median_best
        + 0.20 * p75
        + 0.15 * max_score
    )
    values = (
        max_score,
        top3_mean,
        median_best,
        p75,
        robust_score,
        *a_best.tolist(),
        *b_best.tolist(),
    )
    if not all(math.isfinite(float(value)) for value in values):
        return GalleryPairScoreV1(
            status="INVALID",
            pair_count=int(flat.size),
            reason="derived metric is non-finite",
        )
    return GalleryPairScoreV1(
        status="OK",
        pair_count=int(flat.size),
        max_score=max_score,
        top2_mean=_top_mean(ordered, 2),
        top3_mean=top3_mean,
        top5_mean=_top_mean(ordered, 5),
        median_score=median_score,
        mean_score=float(np.mean(flat)),
        p75_score=p75,
        p90_score=p90,
        support_ge_050=supports[0.50],
        support_ge_055=supports[0.55],
        support_ge_060=supports[0.60],
        support_ge_065=supports[0.65],
        support_ge_070=supports[0.70],
        support_ge_075=supports[0.75],
        support_ge_080=supports[0.80],
        a_best_mean=float(np.mean(a_best)),
        a_best_min=float(np.min(a_best)),
        b_best_mean=float(np.mean(b_best)),
        b_best_min=float(np.min(b_best)),
        median_of_best_matches=median_best,
        robust_score=float(robust_score),
    )
