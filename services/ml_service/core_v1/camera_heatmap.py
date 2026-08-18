from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np


class FootpointHeatmap:
    """Professional camera-space occupancy heatmap from observed track footpoints.

    Heat is accumulated in *seconds of observed occupancy* at each track's
    bbox bottom-center. Consecutive observed positions for the same track are
    interpolated into a continuous path, so walking produces a smooth trail
    instead of isolated circular blobs. Only fresh tracker states produced by a
    real detector observation are accepted; presentation prediction never adds
    heat.
    """

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.grid_width = max(96, int(cfg.get("grid_width", 320)))
        self.grid_height = max(54, int(cfg.get("grid_height", 180)))
        self.sigma = max(1.0, float(cfg.get("sigma", 5.5)))
        self.half_life_sec = max(30.0, float(cfg.get("half_life_sec", 3600.0)))
        self.alpha = max(0.0, min(0.70, float(cfg.get("alpha", 0.42))))

        # Fixed physical meaning: roughly this many observed seconds in the same
        # area are required to reach full red. We intentionally do NOT normalize
        # each frame by the current maximum, because that would make a single
        # person instantly red.
        self.red_seconds = max(1.0, float(cfg.get("red_seconds", 18.0)))
        self.blue_seconds = max(0.05, float(cfg.get("blue_seconds", 0.45)))

        self.render_interval_sec = max(0.05, float(cfg.get("render_interval_sec", 0.10)))
        self.spatial_blur_sigma = max(0.0, float(cfg.get("spatial_blur_sigma", 2.0)))
        self.temporal_smoothing_sec = max(0.05, float(cfg.get("temporal_smoothing_sec", 0.30)))
        self.min_confidence_weight = max(
            0.05, min(1.0, float(cfg.get("min_confidence_weight", 0.35)))
        )
        self.initial_observation_sec = max(
            0.02, float(cfg.get("initial_observation_sec", 0.08))
        )
        self.max_observation_sec = max(
            self.initial_observation_sec,
            float(cfg.get("max_observation_sec", 0.45)),
        )
        self.max_track_gap_sec = max(0.20, float(cfg.get("max_track_gap_sec", 1.0)))
        self.max_jump_ratio = max(0.05, float(cfg.get("max_jump_ratio", 0.25)))
        self.trail_step_sigma = max(0.20, float(cfg.get("trail_step_sigma", 0.55)))

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
        self._cached_shape: tuple[int, int] | None = None
        self._cached_color: np.ndarray | None = None
        self._cached_mask: np.ndarray | None = None
        self._dirty = True
        self._samples = 0
        self._segments = 0
        self._observed_seconds = 0.0

        # track_id -> (grid_x, grid_y, observation_time)
        self._track_points: dict[int, tuple[float, float, float]] = {}

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
        if dt < 0.50:
            return
        factor = 0.5 ** (dt / self.half_life_sec)
        self._grid *= float(factor)
        self._last_decay = now
        self._dirty = True

    def _splat_locked(self, gx: float, gy: float, weight: float) -> None:
        cx = int(round(gx))
        cy = int(round(gy))
        r = self._radius
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(self.grid_width, cx + r + 1)
        y2 = min(self.grid_height, cy + r + 1)
        if x2 <= x1 or y2 <= y1:
            return

        kx1 = x1 - (cx - r)
        ky1 = y1 - (cy - r)
        kx2 = kx1 + (x2 - x1)
        ky2 = ky1 + (y2 - y1)
        self._grid[y1:y2, x1:x2] += (
            self._kernel[ky1:ky2, kx1:kx2] * float(weight)
        )

    def _map_footpoint(self, box, source_width: int, source_height: int):
        try:
            x1 = float(box.x1)
            y1 = float(box.y1)
            x2 = float(box.x2)
            y2 = float(box.y2)
            confidence = float(getattr(box, "confidence", 1.0))
            track_id = int(getattr(box, "track_id", 0) or 0)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if track_id <= 0:
            return None
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, confidence)):
            return None
        if x2 <= x1 or y2 <= y1:
            return None

        foot_x = (x1 + x2) * 0.5
        foot_y = y2
        if (
            foot_x < 0.0
            or foot_x > source_width
            or foot_y < 0.0
            or foot_y > source_height
        ):
            return None

        gx = (
            foot_x / max(1.0, float(source_width))
        ) * (self.grid_width - 1)
        gy = (
            foot_y / max(1.0, float(source_height))
        ) * (self.grid_height - 1)
        confidence_weight = max(
            self.min_confidence_weight,
            min(1.0, confidence),
        )
        return track_id, gx, gy, confidence_weight

    def observe_tracks(
        self,
        boxes,
        source_width: int,
        source_height: int,
        observation_time: float | None = None,
    ) -> None:
        if not self.enabled or source_width <= 1 or source_height <= 1:
            return

        observed_at = (
            time.monotonic() if observation_time is None else float(observation_time)
        )
        diagonal = math.hypot(self.grid_width, self.grid_height)

        with self._lock:
            self._decay_locked(observed_at)
            seen_ids: set[int] = set()

            for box in boxes:
                mapped = self._map_footpoint(box, source_width, source_height)
                if mapped is None:
                    continue
                track_id, gx, gy, confidence_weight = mapped
                seen_ids.add(track_id)

                previous = self._track_points.get(track_id)
                if previous is None:
                    dt = self.initial_observation_sec
                    self._splat_locked(gx, gy, dt * confidence_weight)
                    self._observed_seconds += dt
                else:
                    px, py, previous_time = previous
                    raw_dt = max(0.0, observed_at - previous_time)
                    dt = max(
                        0.02,
                        min(self.max_observation_sec, raw_dt),
                    )
                    distance = math.hypot(gx - px, gy - py)
                    jump_ratio = distance / max(1.0, diagonal)

                    if (
                        raw_dt > self.max_track_gap_sec
                        or jump_ratio > self.max_jump_ratio
                    ):
                        # Do not paint a long line across the room after a lost
                        # track / ID switch. Restart the trail at the new point.
                        self._splat_locked(
                            gx,
                            gy,
                            self.initial_observation_sec * confidence_weight,
                        )
                        self._observed_seconds += self.initial_observation_sec
                    else:
                        step_px = max(1.0, self.sigma * self.trail_step_sigma)
                        steps = max(1, int(math.ceil(distance / step_px)))
                        weight_per_step = (dt * confidence_weight) / steps
                        for index in range(1, steps + 1):
                            t = index / steps
                            ix = px + (gx - px) * t
                            iy = py + (gy - py) * t
                            self._splat_locked(ix, iy, weight_per_step)
                        self._segments += 1
                        self._observed_seconds += dt

                self._track_points[track_id] = (gx, gy, observed_at)
                self._samples += 1

            # Keep the track-history dictionary bounded and prevent an old ID
            # from connecting to a future re-used ID after a long absence.
            stale_before = observed_at - max(2.0, self.max_track_gap_sec * 2.0)
            for track_id, (_x, _y, ts) in list(self._track_points.items()):
                if ts < stale_before:
                    self._track_points.pop(track_id, None)

            if seen_ids:
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

                field = self._display_grid
                if self.spatial_blur_sigma > 0.0:
                    field = cv2.GaussianBlur(
                        field,
                        (0, 0),
                        sigmaX=self.spatial_blur_sigma,
                        sigmaY=self.spatial_blur_sigma,
                        borderType=cv2.BORDER_REPLICATE,
                    )

                # Fixed scale: quick/rare movement stays blue, repeated use and
                # dwell move through cyan/green/yellow and eventually red.
                color_intensity = np.clip(field / self.red_seconds, 0.0, 1.0)
                gray = np.asarray(
                    np.rint(color_intensity * 255.0),
                    dtype=np.uint8,
                )
                color_small = cv2.applyColorMap(gray, self.color_map)

                # Presence mask is separate from color intensity. This keeps the
                # untouched camera image clean instead of tinting the whole frame
                # blue, while making a single pass visible as a soft blue trail.
                presence = np.clip(field / self.blue_seconds, 0.0, 1.0)
                alpha_small = (
                    np.power(presence, 0.70) * self.alpha
                ).astype(np.float32)

                color = cv2.resize(
                    color_small,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask = cv2.resize(
                    alpha_small,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )

                self._cached_color = color
                self._cached_mask = mask
                self._cached_shape = (h, w)
                self._last_render = now
                self._dirty = False

            color = self._cached_color
            mask = self._cached_mask

        if color is None or mask is None or float(np.max(mask)) <= 0.002:
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
                "source": "observed_track_bbox_bottom_center",
                "prediction_boxes_recorded": False,
                "accumulation": "track_interpolated_dwell_seconds",
                "samples": int(self._samples),
                "segments": int(self._segments),
                "observed_seconds": float(self._observed_seconds),
                "grid": [self.grid_width, self.grid_height],
                "half_life_sec": self.half_life_sec,
                "red_seconds": self.red_seconds,
                "blue_seconds": self.blue_seconds,
                "alpha": self.alpha,
                "color_map": self.color_map_name,
                "peak_seconds": float(np.max(self._grid)),
                "active_track_trails": len(self._track_points),
            }
