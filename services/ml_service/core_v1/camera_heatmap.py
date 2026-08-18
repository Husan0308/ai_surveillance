from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np


class FootpointHeatmap:
    """Smooth recent camera-space heatmap from real detector footpoints only.

    A detector bbox bottom-center is treated as the floor-contact point. Heat is
    accumulated by elapsed observation time rather than detector frame count, so
    different detector FPS values do not change the meaning of the map. A quick
    pass remains cool/blue while repeated occupancy or dwell heats toward red.
    Tracker prediction/hold boxes never add heat.
    """

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.grid_width = max(64, int(cfg.get("grid_width", 320)))
        self.grid_height = max(36, int(cfg.get("grid_height", 180)))
        self.sigma = max(1.0, float(cfg.get("sigma", 6.5)))
        self.half_life_sec = max(1.0, float(cfg.get("half_life_sec", 18.0)))
        self.alpha = max(0.0, min(0.75, float(cfg.get("alpha", 0.34))))
        self.saturation = max(0.25, float(cfg.get("saturation", 4.0)))
        self.render_interval_sec = max(0.05, float(cfg.get("render_interval_sec", 0.12)))
        self.min_confidence_weight = max(
            0.05, min(1.0, float(cfg.get("min_confidence_weight", 0.40)))
        )
        self.initial_observation_sec = max(
            0.02, float(cfg.get("initial_observation_sec", 0.25))
        )
        self.max_observation_sec = max(
            self.initial_observation_sec,
            float(cfg.get("max_observation_sec", 0.50)),
        )
        self.spatial_blur_sigma = max(0.0, float(cfg.get("spatial_blur_sigma", 1.0)))
        self.temporal_smoothing_sec = max(
            0.05, float(cfg.get("temporal_smoothing_sec", 0.28))
        )
        self.color_map_name = str(cfg.get("color_map", "jet")).strip().lower()
        self.color_map = {
            "jet": cv2.COLORMAP_JET,
            "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
            "hot": cv2.COLORMAP_HOT,
        }.get(self.color_map_name, cv2.COLORMAP_JET)

        self._grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self._display_grid = np.zeros_like(self._grid)
        self._lock = threading.Lock()
        now = time.monotonic()
        self._last_decay = now
        self._last_render = now
        self._last_observation: float | None = None
        self._cached_shape: tuple[int, int] | None = None
        self._cached_color: np.ndarray | None = None
        self._cached_mask: np.ndarray | None = None
        self._dirty = True
        self._samples = 0
        self._observed_seconds = 0.0

        radius = max(2, int(math.ceil(self.sigma * 3.0)))
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        kernel = np.exp(
            -(xx * xx + yy * yy) / (2.0 * self.sigma * self.sigma)
        ).astype(np.float32)
        kernel /= max(1e-6, float(kernel.max()))
        self._kernel = kernel
        self._radius = radius

    def _decay_locked(self, now: float) -> None:
        dt = max(0.0, now - self._last_decay)
        if dt < 0.10:
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
        self._grid[y1:y2, x1:x2] += (
            self._kernel[ky1:ky2, kx1:kx2] * float(weight)
        )

    def _observation_dt_locked(self, now: float) -> float:
        if self._last_observation is None or now - self._last_observation > 1.0:
            return self.initial_observation_sec
        return max(
            0.02,
            min(self.max_observation_sec, now - self._last_observation),
        )

    def observe_boxes(
        self,
        boxes,
        source_width: int,
        source_height: int,
        now: float | None = None,
    ) -> None:
        if not self.enabled or source_width <= 1 or source_height <= 1:
            return
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            self._decay_locked(now)
            observation_dt = self._observation_dt_locked(now)
            accepted = 0

            for box in boxes:
                try:
                    x1 = float(box.x1)
                    y1 = float(box.y1)
                    x2 = float(box.x2)
                    y2 = float(box.y2)
                    confidence = float(getattr(box, "confidence", 1.0))
                except (AttributeError, TypeError, ValueError, OverflowError):
                    continue
                if not all(
                    math.isfinite(v) for v in (x1, y1, x2, y2, confidence)
                ):
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue

                # Estimated floor-contact point of the person.
                foot_x = (x1 + x2) * 0.5
                foot_y = y2
                if (
                    foot_x < 0.0
                    or foot_x > source_width
                    or foot_y < 0.0
                    or foot_y > source_height
                ):
                    continue

                gx = int(
                    round(
                        (foot_x / max(1.0, float(source_width)))
                        * (self.grid_width - 1)
                    )
                )
                gy = int(
                    round(
                        (foot_y / max(1.0, float(source_height)))
                        * (self.grid_height - 1)
                    )
                )
                confidence_weight = max(
                    self.min_confidence_weight,
                    min(1.0, confidence),
                )
                # Units are approximately observed seconds. This makes a quick
                # walk-through cool, while dwell/repeated visits become hot.
                weight = observation_dt * confidence_weight
                self._splat_locked(gx, gy, weight)
                accepted += 1
                self._samples += 1
                self._observed_seconds += observation_dt

            if accepted:
                self._last_observation = now
                self._dirty = True

    def overlay(self, image: np.ndarray, now: float | None = None) -> np.ndarray:
        if (
            not self.enabled
            or image is None
            or image.size == 0
            or self.alpha <= 0.0
        ):
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
                render_dt = max(0.001, now - self._last_render)
                blend = 1.0 - math.exp(
                    -render_dt / max(0.05, self.temporal_smoothing_sec)
                )
                self._display_grid += (
                    self._grid - self._display_grid
                ) * float(blend)

                intensity = np.clip(
                    self._display_grid / self.saturation,
                    0.0,
                    1.0,
                )
                if self.spatial_blur_sigma > 0.0:
                    intensity = cv2.GaussianBlur(
                        intensity,
                        (0, 0),
                        sigmaX=self.spatial_blur_sigma,
                        sigmaY=self.spatial_blur_sigma,
                        borderType=cv2.BORDER_REPLICATE,
                    )
                    intensity = np.clip(intensity, 0.0, 1.0)

                gray = np.asarray(
                    np.rint(intensity * 255.0),
                    dtype=np.uint8,
                )
                color_small = cv2.applyColorMap(gray, self.color_map)
                color = cv2.resize(
                    color_small,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask = cv2.resize(
                    intensity,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                # sqrt makes low/cool occupancy gently visible without making
                # the empty image blue. High dwell still receives full alpha.
                mask = (
                    np.sqrt(np.clip(mask, 0.0, 1.0)) * self.alpha
                ).astype(np.float32)

                self._cached_color = color
                self._cached_mask = mask
                self._cached_shape = (h, w)
                self._last_render = now
                self._dirty = False

            color = self._cached_color
            mask = self._cached_mask

        if color is None or mask is None:
            return image
        if float(np.max(mask)) <= 0.002:
            return image

        mask3 = mask[..., None]
        blended = (
            image.astype(np.float32) * (1.0 - mask3)
            + color.astype(np.float32) * mask3
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def metrics(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "source": "detector_bbox_bottom_center",
                "prediction_boxes_recorded": False,
                "accumulation": "time_weighted_dwell",
                "samples": int(self._samples),
                "observed_seconds": float(self._observed_seconds),
                "grid": [self.grid_width, self.grid_height],
                "half_life_sec": self.half_life_sec,
                "saturation_seconds": self.saturation,
                "alpha": self.alpha,
                "color_map": self.color_map_name,
                "peak_seconds": float(np.max(self._grid)),
            }
