from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Sequence

import numpy as np

from .step4_reid_gallery_v1 import GalleryViewV1
from .step4_reid_pair_scorer_v1 import GalleryPairScoreV1, SUPPORT_THRESHOLDS

_SAMPLE_CACHE_MAX = 512
_SAMPLE_CACHE_LOCK = threading.Lock()
_SAMPLE_CACHE: OrderedDict[tuple[str, str, int, int], np.ndarray] = OrderedDict()


def _validated_gallery_cached(view: GalleryViewV1, expected_dimension: int) -> np.ndarray:
    if len(view.samples) > 8:
        raise ValueError(f"gallery exceeds capacity: {len(view.samples)}>8")
    rows = []
    for index, sample in enumerate(view.samples):
        key = (
            str(view.camera_id),
            str(view.local_track_id),
            int(sample.sample_sequence),
            id(sample.embedding),
        )
        with _SAMPLE_CACHE_LOCK:
            vector = _SAMPLE_CACHE.get(key)
            if vector is not None:
                _SAMPLE_CACHE.move_to_end(key)
        if vector is None:
            vector = np.asarray(sample.embedding, dtype=np.float64).reshape(-1)
            if vector.size != expected_dimension:
                raise ValueError(
                    f"embedding[{index}] dimension={vector.size}, expected={expected_dimension}"
                )
            if not np.isfinite(vector).all():
                raise ValueError(f"embedding[{index}] is non-finite")
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
                raise ValueError(
                    f"embedding[{index}] is not L2-normalized norm={norm}"
                )
            vector = vector.copy()
            vector.setflags(write=False)
            with _SAMPLE_CACHE_LOCK:
                _SAMPLE_CACHE[key] = vector
                _SAMPLE_CACHE.move_to_end(key)
                while len(_SAMPLE_CACHE) > _SAMPLE_CACHE_MAX:
                    _SAMPLE_CACHE.popitem(last=False)
        rows.append(vector)
    return np.stack(rows, axis=0)


def _quantile_rows(
    sorted_ascending: np.ndarray, counts: np.ndarray, quantile: float
) -> np.ndarray:
    position = (counts.astype(np.float64) - 1.0) * float(quantile)
    lower = np.floor(position).astype(np.int64)
    upper = np.ceil(position).astype(np.int64)
    fraction = position - lower
    row = np.arange(sorted_ascending.shape[0])
    low_value = sorted_ascending[row, lower]
    high_value = sorted_ascending[row, upper]
    return low_value + (high_value - low_value) * fraction


def score_gallery_matrix_step3_padded_exact_v1(
    rows: Sequence[GalleryViewV1],
    columns: Sequence[GalleryViewV1],
    *,
    expected_dimension: int = 256,
) -> dict[tuple[int, int], GalleryPairScoreV1]:
    """Vectorized padded form of the fixed Step-3 scorer, including diagnostics."""

    results: dict[tuple[int, int], GalleryPairScoreV1] = {}
    matrices_a: dict[int, np.ndarray] = {}
    matrices_b: dict[int, np.ndarray] = {}
    errors_a: dict[int, str] = {}
    errors_b: dict[int, str] = {}
    for index, view in enumerate(rows):
        if len(view.samples) >= 3:
            try:
                matrices_a[index] = _validated_gallery_cached(view, int(expected_dimension))
            except (TypeError, ValueError) as exc:
                errors_a[index] = f"{type(exc).__name__}:{exc}"
    for index, view in enumerate(columns):
        if len(view.samples) >= 3:
            try:
                matrices_b[index] = _validated_gallery_cached(view, int(expected_dimension))
            except (TypeError, ValueError) as exc:
                errors_b[index] = f"{type(exc).__name__}:{exc}"

    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            key = (row_index, column_index)
            if len(row.samples) < 3 or len(column.samples) < 3:
                results[key] = GalleryPairScoreV1(
                    status="INSUFFICIENT",
                    pair_count=len(row.samples) * len(column.samples),
                    reason="each gallery requires at least 3 embeddings",
                )
            elif row_index in errors_a or column_index in errors_b:
                results[key] = GalleryPairScoreV1(
                    status="INVALID",
                    pair_count=0,
                    reason=errors_a.get(row_index) or errors_b[column_index],
                )

    valid_rows = sorted(matrices_a)
    valid_columns = sorted(matrices_b)
    if not valid_rows or not valid_columns:
        return results
    padded_a = np.zeros(
        (len(valid_rows), 8, int(expected_dimension)), dtype=np.float64
    )
    padded_b = np.zeros(
        (len(valid_columns), 8, int(expected_dimension)), dtype=np.float64
    )
    lengths_a = np.asarray(
        [matrices_a[index].shape[0] for index in valid_rows], dtype=np.int64
    )
    lengths_b = np.asarray(
        [matrices_b[index].shape[0] for index in valid_columns], dtype=np.int64
    )
    for position, index in enumerate(valid_rows):
        padded_a[position, : lengths_a[position]] = matrices_a[index]
    for position, index in enumerate(valid_columns):
        padded_b[position, : lengths_b[position]] = matrices_b[index]
    cosine = (
        padded_a.reshape(len(valid_rows) * 8, int(expected_dimension))
        @ padded_b.reshape(len(valid_columns) * 8, int(expected_dimension)).T
    ).reshape(len(valid_rows), 8, len(valid_columns), 8).transpose(0, 2, 1, 3)
    mask_a = np.arange(8)[None, :] < lengths_a[:, None]
    mask_b = np.arange(8)[None, :] < lengths_b[:, None]
    mask = mask_a[:, None, :, None] & mask_b[None, :, None, :]
    pair_rows = len(valid_rows) * len(valid_columns)
    flat = cosine.reshape(pair_rows, 64)
    flat_mask = mask.reshape(pair_rows, 64)
    counts = np.count_nonzero(flat_mask, axis=1)
    descending = np.sort(np.where(flat_mask, flat, -np.inf), axis=1)[:, ::-1]
    ascending = np.sort(np.where(flat_mask, flat, np.inf), axis=1)
    maximum = descending[:, 0]
    top2 = np.mean(descending[:, :2], axis=1)
    top3 = np.mean(descending[:, :3], axis=1)
    top5 = np.mean(descending[:, :5], axis=1)
    median = _quantile_rows(ascending, counts, 0.50)
    p75 = _quantile_rows(ascending, counts, 0.75)
    p90 = _quantile_rows(ascending, counts, 0.90)
    mean = np.sum(np.where(flat_mask, flat, 0.0), axis=1) / counts

    a_best = np.max(np.where(mask, cosine, -np.inf), axis=3)
    b_best = np.max(np.where(mask, cosine, -np.inf), axis=2)
    a_mask = np.broadcast_to(mask_a[:, None, :], a_best.shape)
    b_mask = np.broadcast_to(mask_b[None, :, :], b_best.shape)
    a_best_flat = a_best.reshape(pair_rows, 8)
    b_best_flat = b_best.reshape(pair_rows, 8)
    a_mask_flat = a_mask.reshape(pair_rows, 8)
    b_mask_flat = b_mask.reshape(pair_rows, 8)
    lengths_a_pair = np.broadcast_to(
        lengths_a[:, None], (len(valid_rows), len(valid_columns))
    ).reshape(-1)
    lengths_b_pair = np.broadcast_to(
        lengths_b[None, :], (len(valid_rows), len(valid_columns))
    ).reshape(-1)
    a_best_mean = np.sum(np.where(a_mask_flat, a_best_flat, 0.0), axis=1) / lengths_a_pair
    b_best_mean = np.sum(np.where(b_mask_flat, b_best_flat, 0.0), axis=1) / lengths_b_pair
    a_best_min = np.min(np.where(a_mask_flat, a_best_flat, np.inf), axis=1)
    b_best_min = np.min(np.where(b_mask_flat, b_best_flat, np.inf), axis=1)
    all_best = np.concatenate((a_best_flat, b_best_flat), axis=1)
    all_best_mask = np.concatenate((a_mask_flat, b_mask_flat), axis=1)
    best_counts = lengths_a_pair + lengths_b_pair
    sorted_best = np.sort(np.where(all_best_mask, all_best, np.inf), axis=1)
    median_best = _quantile_rows(sorted_best, best_counts, 0.50)
    robust = 0.40 * top3 + 0.25 * median_best + 0.20 * p75 + 0.15 * maximum
    supports = {
        threshold: np.count_nonzero(flat_mask & (flat >= threshold), axis=1)
        for threshold in SUPPORT_THRESHOLDS
    }

    for row_position, row_index in enumerate(valid_rows):
        for column_position, column_index in enumerate(valid_columns):
            flat_index = row_position * len(valid_columns) + column_position
            values = (
                maximum[flat_index],
                top3[flat_index],
                median_best[flat_index],
                p75[flat_index],
                robust[flat_index],
                a_best_mean[flat_index],
                a_best_min[flat_index],
                b_best_mean[flat_index],
                b_best_min[flat_index],
            )
            if not all(math.isfinite(float(value)) for value in values):
                results[(row_index, column_index)] = GalleryPairScoreV1(
                    status="INVALID",
                    pair_count=int(counts[flat_index]),
                    reason="derived metric is non-finite",
                )
                continue
            results[(row_index, column_index)] = GalleryPairScoreV1(
                status="OK",
                pair_count=int(counts[flat_index]),
                max_score=float(maximum[flat_index]),
                top2_mean=float(top2[flat_index]),
                top3_mean=float(top3[flat_index]),
                top5_mean=float(top5[flat_index]),
                median_score=float(median[flat_index]),
                mean_score=float(mean[flat_index]),
                p75_score=float(p75[flat_index]),
                p90_score=float(p90[flat_index]),
                support_ge_050=int(supports[0.50][flat_index]),
                support_ge_055=int(supports[0.55][flat_index]),
                support_ge_060=int(supports[0.60][flat_index]),
                support_ge_065=int(supports[0.65][flat_index]),
                support_ge_070=int(supports[0.70][flat_index]),
                support_ge_075=int(supports[0.75][flat_index]),
                support_ge_080=int(supports[0.80][flat_index]),
                a_best_mean=float(a_best_mean[flat_index]),
                a_best_min=float(a_best_min[flat_index]),
                b_best_mean=float(b_best_mean[flat_index]),
                b_best_min=float(b_best_min[flat_index]),
                median_of_best_matches=float(median_best[flat_index]),
                robust_score=float(robust[flat_index]),
            )
    return results
