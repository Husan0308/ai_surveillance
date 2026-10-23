"""Fail-closed homography diagnostics for operator-entered floor correspondences.

All residuals are in image/layout pixels. This module does not infer physical
scale, lens distortion, camera poses, or correspondence correctness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CAMERA_KEYS = ("cam02", "cam05")


def load_amc_alignment(path: str | Path) -> list[dict[str, Any]]:
    """Read, but never modify, AMC's [camera0, camera1, layout] triplets."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("AMC alignment must be a list of triplets")
    points: list[dict[str, Any]] = []
    for index, triplet in enumerate(payload, start=1):
        if not isinstance(triplet, list) or len(triplet) != 3:
            raise ValueError(f"P{index:02d} must contain CAM-02, CAM-05 and layout coordinates")
        coordinates = [_coordinate_pair(item, f"P{index:02d}") for item in triplet]
        points.append({
            "point_id": f"P{index:02d}",
            "cam02": coordinates[0],
            "cam05": coordinates[1],
            "layout": coordinates[2],
            # These points were already consumed in the prior AMC fit.
            "role": "fit",
            "provenance": "saved_amc_alignment_used_by_prior_fit",
            "cam02_frame_index": None,
            "cam05_frame_index": None,
            "note": "Legacy point; not independent validation ground truth.",
        })
    return points


def _coordinate_pair(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} coordinate must be an [x, y] pair")
    pair = [float(value[0]), float(value[1])]
    if not all(math.isfinite(number) for number in pair):
        raise ValueError(f"{name} coordinate must be finite")
    return pair


def _project(matrix: np.ndarray, xy: list[float] | tuple[float, float]) -> list[float] | None:
    x, y = float(xy[0]), float(xy[1])
    output = matrix @ np.asarray([x, y, 1.0], dtype=np.float64)
    if not np.all(np.isfinite(output)) or abs(float(output[2])) < 1e-12:
        return None
    result = output[:2] / output[2]
    if not np.all(np.isfinite(result)):
        return None
    return [float(result[0]), float(result[1])]


def _distance(first: list[float], second: list[float]) -> float:
    return float(math.hypot(first[0] - second[0], first[1] - second[1]))


def _in_bounds(xy: list[float], shape_hw: tuple[int, int] | None) -> bool | None:
    if shape_hw is None:
        return None
    height, width = shape_hw
    return 0.0 <= xy[0] < width and 0.0 <= xy[1] < height


def _degenerate(points: list[list[float]]) -> bool:
    if len(points) < 4:
        return True
    values = np.asarray(points, dtype=np.float64)
    centered = values - values.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return len(singular) < 2 or singular[0] == 0 or singular[1] / singular[0] < 1e-6


def _fit(src: list[list[float]], dst: list[list[float]]) -> np.ndarray | None:
    if len(src) < 4 or len(src) != len(dst) or _degenerate(src) or _degenerate(dst):
        return None
    matrix, _mask = cv2.findHomography(
        np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64), method=0
    )
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or np.linalg.matrix_rank(matrix) < 3:
        return None
    if abs(matrix[2, 2]) > 1e-12:
        matrix = matrix / matrix[2, 2]
    return matrix


def _rms(errors: list[float]) -> float | None:
    return math.sqrt(sum(value * value for value in errors) / len(errors)) if errors else None


def diagnose_homography(
    points: list[dict[str, Any]],
    camera_key: str,
    *,
    source_shape_hw: tuple[int, int] | None,
    layout_shape_hw: tuple[int, int] | None,
) -> dict[str, Any]:
    """Fit camera->layout H; holdouts are excluded; return residuals and bounds.

    ``role=fit`` participates in the fit. ``role=heldout`` is only evaluated
    against a fit made from fit points. The status never promotes these values
    to physical accuracy: that requires independently verified floor targets.
    """
    if camera_key not in CAMERA_KEYS:
        raise ValueError(f"unsupported camera key: {camera_key}")
    fit_rows: list[dict[str, Any]] = []
    hold_rows: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for point in points:
        pid = str(point.get("point_id", "?"))
        src = point.get(camera_key)
        dst = point.get("layout")
        if not isinstance(src, (list, tuple)) or len(src) != 2 or not isinstance(dst, (list, tuple)) or len(dst) != 2:
            incomplete.append(pid)
            continue
        src_xy = _coordinate_pair(src, pid)
        dst_xy = _coordinate_pair(dst, pid)
        row = {"point_id": pid, "source": src_xy, "layout": dst_xy,
               "source_in_bounds": _in_bounds(src_xy, source_shape_hw),
               "layout_in_bounds": _in_bounds(dst_xy, layout_shape_hw)}
        if point.get("role", "fit") == "heldout":
            hold_rows.append(row)
        elif point.get("role", "fit") == "fit":
            fit_rows.append(row)
        else:
            incomplete.append(pid)

    duplicate_pairs: list[dict[str, str]] = []
    for i, left in enumerate(fit_rows):
        for right in fit_rows[i + 1:]:
            if (_distance(left["source"], right["source"]) <= 1.0
                    or _distance(left["layout"], right["layout"]) <= 1.0):
                duplicate_pairs.append({"first": left["point_id"], "second": right["point_id"]})

    source = [row["source"] for row in fit_rows]
    layout = [row["layout"] for row in fit_rows]
    matrix = _fit(source, layout)
    if matrix is None:
        return {
            "camera_key": camera_key,
            "transform_direction": "camera_image_to_layout_pixel",
            "matrix_row_major": None,
            "fit_point_count": len(fit_rows),
            "heldout_point_count": len(hold_rows),
            "training_rms_layout_px": None,
            "training_errors": [],
            "leave_one_out_rms_layout_px": None,
            "leave_one_out_errors": [],
            "heldout_errors": [],
            "incomplete_point_ids": incomplete,
            "duplicate_point_pairs_within_1px": duplicate_pairs,
            "fit_status": "BLOCKED: need at least four complete, non-collinear fit points",
            "accuracy_status": "UNVERIFIED; pixel fit diagnostics are not physical accuracy",
        }

    training_errors: list[dict[str, Any]] = []
    train_norms: list[float] = []
    for row in fit_rows:
        predicted = _project(matrix, row["source"])
        error = _distance(predicted, row["layout"]) if predicted is not None else None
        if error is not None:
            train_norms.append(error)
        training_errors.append({**row, "predicted_layout": predicted, "error_layout_px": error})

    loo_errors: list[dict[str, Any]] = []
    loo_norms: list[float] = []
    if len(fit_rows) >= 5:
        for omitted_index, omitted in enumerate(fit_rows):
            subset = [row for index, row in enumerate(fit_rows) if index != omitted_index]
            loo_matrix = _fit([row["source"] for row in subset], [row["layout"] for row in subset])
            predicted = _project(loo_matrix, omitted["source"]) if loo_matrix is not None else None
            error = _distance(predicted, omitted["layout"]) if predicted is not None else None
            if error is not None:
                loo_norms.append(error)
            loo_errors.append({"point_id": omitted["point_id"], "predicted_layout": predicted,
                               "error_layout_px": error,
                               "status": "OK" if error is not None else "LOO subset degenerate"})

    heldout_errors: list[dict[str, Any]] = []
    for row in hold_rows:
        predicted = _project(matrix, row["source"])
        error = _distance(predicted, row["layout"]) if predicted is not None else None
        heldout_errors.append({**row, "predicted_layout": predicted, "error_layout_px": error})

    return {
        "camera_key": camera_key,
        "transform_direction": "camera_image_to_layout_pixel",
        "matrix_row_major": matrix.reshape(-1).tolist(),
        "matrix_condition_number": float(np.linalg.cond(matrix)),
        "fit_point_count": len(fit_rows),
        "heldout_point_count": len(hold_rows),
        "training_rms_layout_px": _rms(train_norms),
        "training_errors": training_errors,
        "leave_one_out_rms_layout_px": _rms(loo_norms),
        "leave_one_out_errors": loo_errors,
        "heldout_rms_layout_px": _rms([item["error_layout_px"] for item in heldout_errors
                                       if item["error_layout_px"] is not None]),
        "heldout_errors": heldout_errors,
        "incomplete_point_ids": incomplete,
        "out_of_bounds_point_ids": [row["point_id"] for row in (*fit_rows, *hold_rows)
                                     if row["source_in_bounds"] is False or row["layout_in_bounds"] is False],
        "duplicate_point_pairs_within_1px": duplicate_pairs,
        "fit_status": "FIT ONLY; inspect distribution, correspondences and conditioning",
        "accuracy_status": "UNVERIFIED; holdout is independent only if its floor reference was independently measured",
    }
