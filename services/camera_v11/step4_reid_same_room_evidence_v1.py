from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np

from .step4_reid_gallery_v1 import GalleryViewV1
from .step4_reid_pair_scorer_v1 import GalleryPairScoreV1, SUPPORT_THRESHOLDS


def _validated_gallery(view: GalleryViewV1, expected_dimension: int) -> np.ndarray:
    if len(view.samples) > 8:
        raise ValueError(f"gallery exceeds capacity: {len(view.samples)}>8")
    rows: list[np.ndarray] = []
    for index, sample in enumerate(view.samples):
        vector = np.asarray(sample.embedding, dtype=np.float64).reshape(-1)
        if vector.size != expected_dimension:
            raise ValueError(
                f"embedding[{index}] dimension={vector.size}, expected={expected_dimension}"
            )
        rows.append(vector)
    matrix = np.stack(rows, axis=0)
    finite_rows = np.isfinite(matrix).all(axis=1)
    if not finite_rows.all():
        index = int(np.flatnonzero(~finite_rows)[0])
        raise ValueError(f"embedding[{index}] is non-finite")
    norms = np.linalg.norm(matrix, axis=1)
    valid_norms = np.isfinite(norms) & (np.abs(norms - 1.0) <= 1e-3)
    if not valid_norms.all():
        index = int(np.flatnonzero(~valid_norms)[0])
        raise ValueError(
            f"embedding[{index}] is not L2-normalized norm={float(norms[index])}"
        )
    return matrix


def score_gallery_matrix_step3_exact_v1(
    rows: Sequence[GalleryViewV1],
    columns: Sequence[GalleryViewV1],
    *,
    expected_dimension: int = 256,
) -> dict[tuple[int, int], GalleryPairScoreV1]:
    """Batch the unchanged Step-3 formula across a local-track score matrix.

    Validation, status behavior, diagnostics and robust-score coefficients mirror
    ``score_gallery_pair_v1``. Batching only removes repeated NumPy setup cost.
    """

    results: dict[tuple[int, int], GalleryPairScoreV1] = {}
    matrices_a: dict[int, np.ndarray] = {}
    matrices_b: dict[int, np.ndarray] = {}
    errors_a: dict[int, str] = {}
    errors_b: dict[int, str] = {}
    for index, view in enumerate(rows):
        if len(view.samples) < 3:
            continue
        try:
            matrices_a[index] = _validated_gallery(view, int(expected_dimension))
        except (TypeError, ValueError) as exc:
            errors_a[index] = f"{type(exc).__name__}:{exc}"
    for index, view in enumerate(columns):
        if len(view.samples) < 3:
            continue
        try:
            matrices_b[index] = _validated_gallery(view, int(expected_dimension))
        except (TypeError, ValueError) as exc:
            errors_b[index] = f"{type(exc).__name__}:{exc}"

    groups: defaultdict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
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
                reason = errors_a.get(row_index) or errors_b[column_index]
                results[key] = GalleryPairScoreV1(
                    status="INVALID", pair_count=0, reason=reason
                )
            else:
                groups[(len(row.samples), len(column.samples))].append(key)

    valid_rows = sorted(matrices_a)
    valid_columns = sorted(matrices_b)
    valid_row_position = {value: index for index, value in enumerate(valid_rows)}
    valid_column_position = {value: index for index, value in enumerate(valid_columns)}
    padded_a = np.zeros((len(valid_rows), 8, int(expected_dimension)), dtype=np.float64)
    padded_b = np.zeros((len(valid_columns), 8, int(expected_dimension)), dtype=np.float64)
    for position, index in enumerate(valid_rows):
        padded_a[position, : matrices_a[index].shape[0]] = matrices_a[index]
    for position, index in enumerate(valid_columns):
        padded_b[position, : matrices_b[index].shape[0]] = matrices_b[index]
    all_cosine = np.einsum("aik,bjk->abij", padded_a, padded_b, optimize=True)

    for (samples_a, samples_b), keys in groups.items():
        row_indices = sorted({key[0] for key in keys})
        column_indices = sorted({key[1] for key in keys})
        row_position = {value: index for index, value in enumerate(row_indices)}
        column_position = {value: index for index, value in enumerate(column_indices)}
        cosine = all_cosine[
            np.asarray([valid_row_position[index] for index in row_indices])[:, None],
            np.asarray([valid_column_position[index] for index in column_indices])[None, :],
            :samples_a,
            :samples_b,
        ]
        pair_rows = len(row_indices) * len(column_indices)
        flat = cosine.reshape(pair_rows, samples_a * samples_b)
        if not np.isfinite(flat).all():
            for key in keys:
                results[key] = GalleryPairScoreV1(
                    status="INVALID",
                    pair_count=samples_a * samples_b,
                    reason="cosine matrix contains non-finite scores",
                )
            continue
        ordered = np.sort(flat, axis=1)[:, ::-1]
        a_best = np.max(cosine, axis=3).reshape(pair_rows, samples_a)
        b_best = np.max(cosine, axis=2).reshape(pair_rows, samples_b)
        all_best = np.concatenate((a_best, b_best), axis=1)
        percentile = np.percentile(flat, (75, 90), axis=1)
        maximum = ordered[:, 0]
        top2 = np.mean(ordered[:, : min(2, ordered.shape[1])], axis=1)
        top3 = np.mean(ordered[:, : min(3, ordered.shape[1])], axis=1)
        top5 = np.mean(ordered[:, : min(5, ordered.shape[1])], axis=1)
        median = np.median(flat, axis=1)
        mean = np.mean(flat, axis=1)
        median_best = np.median(all_best, axis=1)
        robust = (
            0.40 * top3
            + 0.25 * median_best
            + 0.20 * percentile[0]
            + 0.15 * maximum
        )
        supports = {
            threshold: np.count_nonzero(flat >= threshold, axis=1)
            for threshold in SUPPORT_THRESHOLDS
        }
        for row_index, column_index in keys:
            flat_index = (
                row_position[row_index] * len(column_indices)
                + column_position[column_index]
            )
            values = (
                maximum[flat_index],
                top3[flat_index],
                median_best[flat_index],
                percentile[0, flat_index],
                robust[flat_index],
                *a_best[flat_index].tolist(),
                *b_best[flat_index].tolist(),
            )
            if not all(math.isfinite(float(value)) for value in values):
                results[(row_index, column_index)] = GalleryPairScoreV1(
                    status="INVALID",
                    pair_count=samples_a * samples_b,
                    reason="derived metric is non-finite",
                )
                continue
            results[(row_index, column_index)] = GalleryPairScoreV1(
                status="OK",
                pair_count=samples_a * samples_b,
                max_score=float(maximum[flat_index]),
                top2_mean=float(top2[flat_index]),
                top3_mean=float(top3[flat_index]),
                top5_mean=float(top5[flat_index]),
                median_score=float(median[flat_index]),
                mean_score=float(mean[flat_index]),
                p75_score=float(percentile[0, flat_index]),
                p90_score=float(percentile[1, flat_index]),
                support_ge_050=int(supports[0.50][flat_index]),
                support_ge_055=int(supports[0.55][flat_index]),
                support_ge_060=int(supports[0.60][flat_index]),
                support_ge_065=int(supports[0.65][flat_index]),
                support_ge_070=int(supports[0.70][flat_index]),
                support_ge_075=int(supports[0.75][flat_index]),
                support_ge_080=int(supports[0.80][flat_index]),
                a_best_mean=float(np.mean(a_best[flat_index])),
                a_best_min=float(np.min(a_best[flat_index])),
                b_best_mean=float(np.mean(b_best[flat_index])),
                b_best_min=float(np.min(b_best[flat_index])),
                median_of_best_matches=float(median_best[flat_index]),
                robust_score=float(robust[flat_index]),
            )
    return results
