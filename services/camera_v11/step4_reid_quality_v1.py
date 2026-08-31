from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


DETECTOR_WIDTH = 672
DETECTOR_HEIGHT = 384
DETECTOR_CONTENT_TOP = 3
DETECTOR_CONTENT_HEIGHT = 378


@dataclass(frozen=True)
class ReIDCropQualityDecision:
    accepted: bool
    reason: str
    quality_score: float
    blur_variance: float
    clipped_fraction: float
    aspect_ratio: float
    source_bbox_xyxy: tuple[int, int, int, int] | None
    crop_bgr: np.ndarray | None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _decision(
    reason: str,
    *,
    quality_score: float = 0.0,
    blur_variance: float = 0.0,
    clipped_fraction: float = 0.0,
    aspect_ratio: float = 0.0,
    source_bbox_xyxy: tuple[int, int, int, int] | None = None,
    crop_bgr: np.ndarray | None = None,
) -> ReIDCropQualityDecision:
    return ReIDCropQualityDecision(
        accepted=reason == "accepted",
        reason=reason,
        quality_score=float(quality_score),
        blur_variance=float(blur_variance),
        clipped_fraction=float(clipped_fraction),
        aspect_ratio=float(aspect_ratio),
        source_bbox_xyxy=source_bbox_xyxy,
        crop_bgr=crop_bgr,
    )


def evaluate_reid_crop_quality(
    frame_bgr: np.ndarray | None,
    detector_bbox_xyxy: tuple[float, float, float, float],
    detector_score: float,
    *,
    min_width: int = 24,
    min_height: int = 64,
    min_aspect: float = 0.85,
    max_aspect: float = 6.50,
    max_clipped_fraction: float = 0.20,
    severe_blur_variance: float = 7.0,
    min_quality_score: float = 0.25,
) -> ReIDCropQualityDecision:
    """Evaluate a native-resolution BGR frame and return an owned accepted crop.

    Detector coordinates refer to the frozen 672x384 tensor. Its real 16:9 image
    occupies rows 3:381, so the mapping removes that padding before scaling to the
    native decoded frame. Blur work is bounded to at most 96x192 grayscale pixels.
    """

    if (
        frame_bgr is None
        or not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.dtype != np.uint8
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] < 3
        or frame_bgr.shape[0] <= 0
        or frame_bgr.shape[1] <= 0
    ):
        return _decision("invalid")
    try:
        x1, y1, x2, y2 = (float(value) for value in detector_bbox_xyxy)
        detector_score = float(detector_score)
    except (TypeError, ValueError):
        return _decision("invalid")
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2, detector_score)):
        return _decision("invalid")
    if x2 <= x1 or y2 <= y1:
        return _decision("invalid")

    source_h, source_w = frame_bgr.shape[:2]
    raw_x1 = x1 * source_w / DETECTOR_WIDTH
    raw_x2 = x2 * source_w / DETECTOR_WIDTH
    raw_y1 = (y1 - DETECTOR_CONTENT_TOP) * source_h / DETECTOR_CONTENT_HEIGHT
    raw_y2 = (y2 - DETECTOR_CONTENT_TOP) * source_h / DETECTOR_CONTENT_HEIGHT
    raw_width = raw_x2 - raw_x1
    raw_height = raw_y2 - raw_y1
    if raw_width <= 0.0 or raw_height <= 0.0:
        return _decision("invalid")

    clipped_x1 = max(0.0, min(float(source_w), raw_x1))
    clipped_y1 = max(0.0, min(float(source_h), raw_y1))
    clipped_x2 = max(0.0, min(float(source_w), raw_x2))
    clipped_y2 = max(0.0, min(float(source_h), raw_y2))
    clipped_width = clipped_x2 - clipped_x1
    clipped_height = clipped_y2 - clipped_y1
    if clipped_width <= 1.0 or clipped_height <= 1.0:
        return _decision("invalid")

    raw_area = raw_width * raw_height
    clipped_area = clipped_width * clipped_height
    clipped_fraction = _clamp(1.0 - clipped_area / max(raw_area, 1e-9))
    ix1 = max(0, min(source_w, int(math.floor(clipped_x1))))
    iy1 = max(0, min(source_h, int(math.floor(clipped_y1))))
    ix2 = max(0, min(source_w, int(math.ceil(clipped_x2))))
    iy2 = max(0, min(source_h, int(math.ceil(clipped_y2))))
    width = ix2 - ix1
    height = iy2 - iy1
    source_bbox = (ix1, iy1, ix2, iy2)

    if width < int(min_width) or height < int(min_height):
        return _decision(
            "size",
            clipped_fraction=clipped_fraction,
            source_bbox_xyxy=source_bbox,
        )

    edge_px = max(2, int(round(min(source_w, source_h) * 0.004)))
    touches = sum(
        (
            ix1 <= edge_px,
            iy1 <= edge_px,
            ix2 >= source_w - edge_px,
            iy2 >= source_h - edge_px,
        )
    )
    # One border touch is common for feet at the bottom of CCTV frames. Reject
    # only a real clipped-area loss or two-border corner truncation initially.
    if clipped_fraction > float(max_clipped_fraction) or touches >= 2:
        return _decision(
            "edge",
            clipped_fraction=clipped_fraction,
            source_bbox_xyxy=source_bbox,
        )

    aspect_ratio = height / max(float(width), 1.0)
    if aspect_ratio < float(min_aspect) or aspect_ratio > float(max_aspect):
        return _decision(
            "aspect",
            clipped_fraction=clipped_fraction,
            aspect_ratio=aspect_ratio,
            source_bbox_xyxy=source_bbox,
        )

    crop_view = frame_bgr[iy1:iy2, ix1:ix2, :3]
    if crop_view.size == 0 or crop_view.shape[0] != height or crop_view.shape[1] != width:
        return _decision(
            "invalid",
            clipped_fraction=clipped_fraction,
            aspect_ratio=aspect_ratio,
            source_bbox_xyxy=source_bbox,
        )

    gray = cv2.cvtColor(crop_view, cv2.COLOR_BGR2GRAY)
    analysis_scale = min(1.0, 96.0 / width, 192.0 / height)
    if analysis_scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(8, int(round(width * analysis_scale))), max(16, int(round(height * analysis_scale)))),
            interpolation=cv2.INTER_AREA,
        )
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())
    if not math.isfinite(blur_variance):
        return _decision(
            "invalid",
            clipped_fraction=clipped_fraction,
            aspect_ratio=aspect_ratio,
            source_bbox_xyxy=source_bbox,
        )
    if blur_variance < float(severe_blur_variance):
        return _decision(
            "blur",
            blur_variance=blur_variance,
            clipped_fraction=clipped_fraction,
            aspect_ratio=aspect_ratio,
            source_bbox_xyxy=source_bbox,
        )

    sharpness = _clamp(
        (math.log1p(blur_variance) - math.log1p(severe_blur_variance))
        / (math.log1p(180.0) - math.log1p(severe_blur_variance))
    )
    resolution = _clamp(min(width / 90.0, height / 190.0))
    edge_quality = _clamp(1.0 - clipped_fraction - 0.12 * touches)
    aspect_quality = _clamp(1.0 - abs(math.log(max(aspect_ratio, 1e-6) / 2.4)) / 1.4)
    confidence_quality = _clamp((detector_score - 0.18) / 0.62)
    quality_score = _clamp(
        0.30 * sharpness
        + 0.24 * resolution
        + 0.18 * edge_quality
        + 0.14 * aspect_quality
        + 0.14 * confidence_quality
    )

    # Passing the deliberately permissive hard gates normally clears this floor.
    # If not, attribute the rejection to the weakest measured required component,
    # keeping the requested counter taxonomy complete and mutually exclusive.
    if quality_score < float(min_quality_score):
        weakest = min(
            (
                (sharpness, "blur"),
                (resolution, "size"),
                (edge_quality, "edge"),
                (aspect_quality, "aspect"),
                (confidence_quality, "score"),
            )
        )[1]
        return _decision(
            weakest,
            quality_score=quality_score,
            blur_variance=blur_variance,
            clipped_fraction=clipped_fraction,
            aspect_ratio=aspect_ratio,
            source_bbox_xyxy=source_bbox,
        )

    crop = np.ascontiguousarray(crop_view.copy())
    if crop.size == 0:
        return _decision(
            "invalid",
            quality_score=quality_score,
            blur_variance=blur_variance,
            clipped_fraction=clipped_fraction,
            aspect_ratio=aspect_ratio,
            source_bbox_xyxy=source_bbox,
        )
    return _decision(
        "accepted",
        quality_score=quality_score,
        blur_variance=blur_variance,
        clipped_fraction=clipped_fraction,
        aspect_ratio=aspect_ratio,
        source_bbox_xyxy=source_bbox,
        crop_bgr=crop,
    )
