from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class CropQuality:
    score: float
    sharpness: float
    resolution: float
    exposure: float
    truncation: float
    detector_confidence: float
    tracker_confidence: float
    overlap: float
    accepted: bool
    reason: str = "ok"


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0.0 else 0.0


def crop_signature(crop: np.ndarray) -> int:
    """64-bit perceptual dHash used only to reject near-duplicate evidence frames."""
    if crop is None or crop.size == 0:
        return 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = tiny[:, 1:] > tiny[:, :-1]
    value = 0
    for bit in diff.reshape(-1):
        value = (value << 1) | int(bool(bit))
    return value


def hamming64(a: int, b: int) -> int:
    return int((int(a) ^ int(b)).bit_count())


def evaluate_crop_quality(
    crop: np.ndarray,
    *,
    source_bbox: tuple[float, float, float, float],
    source_width: int,
    source_height: int,
    detector_confidence: float,
    tracker_confidence: float = 1.0,
    max_other_iou: float = 0.0,
    min_width: int = 22,
    min_height: int = 55,
    min_score: float = 0.34,
) -> CropQuality:
    if crop is None or crop.size == 0:
        return CropQuality(0, 0, 0, 0, 0, 0, 0, 1, False, "empty")
    h, w = crop.shape[:2]
    if w < min_width or h < min_height:
        return CropQuality(0, 0, 0, 0, 0, 0, 0, 1, False, "small")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = _clamp((math.log1p(max(0.0, lap)) - math.log1p(8.0)) /
                       (math.log1p(120.0) - math.log1p(8.0)))
    resolution = _clamp(min(w / 90.0, h / 190.0))
    mean = float(gray.mean())
    exposure = _clamp(1.0 - abs(mean - 128.0) / 118.0)

    x1, y1, x2, y2 = [float(v) for v in source_bbox]
    edge = 3.0
    touches = int(x1 <= edge) + int(y1 <= edge) + int(x2 >= source_width - edge) + int(y2 >= source_height - edge)
    truncation = _clamp(1.0 - 0.22 * touches)
    det = _clamp((float(detector_confidence) - 0.06) / 0.54)
    trk = _clamp((float(tracker_confidence) + 0.1) / 1.1) if tracker_confidence >= 0 else 0.65
    overlap = _clamp(float(max_other_iou) / 0.50)

    score = _clamp(
        0.24 * sharpness
        + 0.21 * resolution
        + 0.14 * exposure
        + 0.14 * truncation
        + 0.14 * det
        + 0.08 * trk
        + 0.05 * (1.0 - overlap)
    )

    if max_other_iou >= 0.58:
        return CropQuality(score, sharpness, resolution, exposure, truncation, det, trk, overlap, False, "overlap")
    if sharpness < 0.08 and resolution < 0.55:
        return CropQuality(score, sharpness, resolution, exposure, truncation, det, trk, overlap, False, "blur-small")
    if truncation < 0.45:
        return CropQuality(score, sharpness, resolution, exposure, truncation, det, trk, overlap, False, "truncated")
    if score < min_score:
        return CropQuality(score, sharpness, resolution, exposure, truncation, det, trk, overlap, False, "low-quality")
    return CropQuality(score, sharpness, resolution, exposure, truncation, det, trk, overlap, True, "ok")
