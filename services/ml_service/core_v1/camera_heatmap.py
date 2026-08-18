from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np


class FootpointHeatmap:
    """Lightweight recent camera-space heatmap driven only by observed person feet.

    The detector bbox bottom-center is treated as the floor-contact point. Tracker
    prediction/hold boxes never add heat, so temporary visual prediction cannot
    leave ghost trails. The heatmap lives on a small grid and is rendered over the
    camera frame only when needed.
    """

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.grid_width = max(64, int(cfg.get("grid_width", 320)))
        self.grid_height = max(36, int(cfg.get("grid_height", 180)))
        self.sigma = max(1.0, float(cfg.get("sigma", 7.0)))
        self.half_life_sec = max(1.0, float(cfg.get("half_life_sec", 12.0)))
        self.alpha = max(0.0, min(0.75, float(cfg.get("alpha", 0.30))))
        self.saturation = max(0.5, float(cfg.get("saturation", 3.5)))
        self.render_interval_sec = max(0.10, float(cfg.get("render_interval_sec", 0.35)))
        self.min_confidence_weight = max(0.05, min(1.0, float(cfg.get("min_confidence_weight", 0.35))))

        self._grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self._lock = threading.Lock()
        self._last_decay = time.monotonic()
        self._last_render = 0.0
        self._cached_shape: tuple[int, int] | None = None
        self._cached_color: np.ndarray | None = None
        self._cached_mask: np.ndarray | None = None
        self._dirty = True
        self._samples = 0

        radius = max(2, int(math.ceil(self.sigma * 3.0)))
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * self.sigma * self.sigma)).astype(np.float32)
        kernel /= max(1e-6, float(kernel.max()))
        self._kernel = kernel
        self._radius = radius

    def _decay_locked(self, now: float) -> None:
        dt = max(0.0, now - self._last_decay)
        if dt < 0.25:
            return
        factor = 0.5 ** (dt / self.half_life_sec)
        self._grid *= float(factor)
        self._last_decay = now
        self._dirty = True

    def _splat_locked(self, gx: int, gy: int, weight: float) -> None:
        r = self._radius
        x1 = max(0, gx - r)
        y1 = max(0, gy - r)
        x2 = min(self.grid_width, gx + r + 1)
        y2 = min(self.grid_height, gy + r + 1)
        if x2 <= x1 or y2 <= y1:
            return

        kx1 = x1 - (gx - r)
        ky1 = y1 - (gy - r)
        kx2 = kx1 + (x2 - x1)
        ky2 = ky1 + (y2 - y1)
        self._grid[y1:y2, x1:x2] += self._kernel[ky1:ky2, kx1:kx2] * float(weight)

    def observe_boxes(self, boxes, source_width: int, source_height: int, now: float | None = None) -> None:
        if not self.enabled or source_width <= 1 or source_height <= 1:
            return
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            self._decay_locked(now)
            for box in boxes:
                try:
                    x1 = float(box.x1)
                    y1 = float(box.y1)
                    x2 = float(box.x2)
                    y2 = float(box.y2)
                    confidence = float(getattr(box, "confidence", 1.0))
                except (AttributeError, TypeError, ValueError, OverflowError):
                    continue
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2, confidence)):
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue

                # Floor-contact estimate: center of the bottom edge of the person bbox.
                foot_x = (x1 + x2) * 0.5
                foot_y = y2
                if foot_x < 0.0 or foot_x > source_width or foot_y < 0.0 or foot_y > source_height:
                    continue

                gx = int(round((foot_x / max(1.0, float(source_width))) * (self.grid_width - 1)))
                gy = int(round((foot_y / max(1.0, float(source_height))) * (self.grid_height - 1)))
                weight = max(self.min_confidence_weight, min(1.0, confidence))
                self._splat_locked(gx, gy, weight)
                self._samples += 1
            self._dirty = True

    def overlay(self, image: np.ndarray, now: float | None = None) -> np.ndarray:
        if not self.enabled or image is None or image.size == 0 or self.alpha <= 0.0:
            return image
        now = time.monotonic() if now is None else float(now)
        h, w = image.shape[:2]

        with self._lock:
            self._decay_locked(now)
            needs_render = (
                self._dirty
                or self._cached_shape != (h, w)
                or self._cached_color is None
                or now - self._last_render >= self.render_interval_sec
            )
            if needs_render:
                intensity = np.clip(self._grid / self.saturation, 0.0, 1.0)
                gray = np.asarray(np.rint(intensity * 255.0), dtype=np.uint8)
                color_small = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
                color = cv2.resize(color_small, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(intensity, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = np.clip(mask * self.alpha, 0.0, self.alpha).astype(np.float32)
                self._cached_color = color
                self._cached_mask = mask
                self._cached_shape = (h, w)
                self._last_render = now
                self._dirty = False

            color = self._cached_color
            mask = self._cached_mask

        if color is None or mask is None or float(mask.max(initial=0.0)) <= 0.002:
            return image
        mask3 = mask[..., None]
        blended = image.astype(np.float32) * (1.0 - mask3) + color.astype(np.float32) * mask3
        return np.clip(blended, 0, 255).astype(np.uint8)

    def metrics(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "source": "detector_bbox_bottom_center",
                "prediction_boxes_recorded": False,
                "samples": int(self._samples),
                "grid": [self.grid_width, self.grid_height],
                "half_life_sec": self.half_life_sec,
                "alpha": self.alpha,
                "peak": float(self._grid.max(initial=0.0)),
            }
