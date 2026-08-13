from __future__ import annotations

from dataclasses import dataclass
import math

import cv2


@dataclass(frozen=True, slots=True)
class CropDecision:
    accepted: bool
    score: float
    reason: str
    crop: object | None


class PersonCropSelector:
    """Quality gate for real detector crops used by ReID v2.

    Tracklet aggregation needs several useful and different views, not many
    noisy crops. This gate is deliberately cheap and never consumes a predicted
    visual-tracker box: it only receives the exact frame/box from YOLO.
    """

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.min_width = max(16, int(cfg.get("min_width", 42)))
        self.min_height = max(24, int(cfg.get("min_height", 96)))
        self.min_area = max(512, int(cfg.get("min_area", 6500)))
        self.min_conf = max(0.0, float(cfg.get("min_conf", 0.20)))
        self.max_width_height_ratio = max(
            0.30, float(cfg.get("max_width_height_ratio", 0.95))
        )
        self.edge_margin = max(0.0, min(0.25, float(cfg.get("edge_margin", 0.015))))
        self.reject_edge = bool(cfg.get("reject_edge", False))
        self.min_sharpness = max(0.0, float(cfg.get("min_sharpness", 18.0)))
        self.pad_x = max(0.0, min(0.30, float(cfg.get("pad_x", 0.04))))
        self.pad_y = max(0.0, min(0.30, float(cfg.get("pad_y", 0.02))))

    def evaluate(self, frame, box) -> CropDecision:
        image = frame.image
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(float, (box.x1, box.y1, box.x2, box.y2))
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area = bw * bh
        conf = float(box.confidence)

        if conf < self.min_conf:
            return CropDecision(False, 0.0, "low_conf", None)
        if bw < self.min_width or bh < self.min_height or area < self.min_area:
            return CropDecision(False, 0.0, "small", None)

        aspect = bw / max(1.0, bh)
        if aspect > self.max_width_height_ratio:
            return CropDecision(False, 0.0, "wide_fragment", None)

        mx = self.edge_margin * w
        my = self.edge_margin * h
        touches = x1 <= mx or y1 <= my or x2 >= w - mx or y2 >= h - my
        if touches and self.reject_edge:
            return CropDecision(False, 0.0, "edge", None)

        px = bw * self.pad_x
        py = bh * self.pad_y
        ix1 = max(0, min(w - 1, int(math.floor(x1 - px))))
        iy1 = max(0, min(h - 1, int(math.floor(y1 - py))))
        ix2 = max(ix1 + 1, min(w, int(math.ceil(x2 + px))))
        iy2 = max(iy1 + 1, min(h, int(math.ceil(y2 + py))))
        crop = image[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            return CropDecision(False, 0.0, "empty", None)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < self.min_sharpness:
            return CropDecision(False, 0.0, "blur", None)

        frame_ratio = area / max(1.0, float(w * h))
        size_score = min(1.0, math.sqrt(frame_ratio) / 0.45)
        sharp_score = min(1.0, sharpness / max(self.min_sharpness * 5.0, 1.0))
        portrait_score = max(
            0.0,
            min(1.0, (self.max_width_height_ratio - aspect + 0.20) / 0.55),
        )
        score = (
            0.43 * conf
            + 0.25 * size_score
            + 0.18 * sharp_score
            + 0.14 * portrait_score
        )
        if touches:
            score *= 0.84

        return CropDecision(
            True,
            float(score),
            "ok_edge" if touches else "ok",
            crop.copy(),
        )
