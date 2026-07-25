import math
from datetime import datetime

import numpy as np

from PySide6.QtGui import QImage, QColor
from PySide6.QtCore import Qt

from backend.core.logger import get_logger

log = get_logger("features.heatmap")


def heat_color(v):
    """
    Blue -> cyan -> green -> yellow -> red
    """

    stops = [
        (0.0, (40, 110, 255)),
        (0.3, (0, 200, 200)),
        (0.55, (60, 215, 80)),
        (0.8, (250, 200, 40)),
        (1.0, (255, 70, 40)),
    ]

    v = max(0.0, min(1.0, float(v)))

    for i in range(1, len(stops)):
        if v <= stops[i][0]:
            t0, c0 = stops[i - 1]
            t1, c1 = stops[i]
            f = (v - t0) / (t1 - t0) if t1 > t0 else 0.0

            return tuple(int(c0[k] + (c1[k] - c0[k]) * f) for k in range(3))

    return stops[-1][1]


def blur_grid(grid):
    """
    Simple 3x3 blur for numpy grid.
    """

    h, w = grid.shape
    out = np.zeros_like(grid)

    for y in range(h):
        for x in range(w):
            y1 = max(0, y - 1)
            y2 = min(h, y + 2)
            x1 = max(0, x - 1)
            x2 = min(w, x + 2)

            out[y, x] = grid[y1:y2, x1:x2].mean()

    return out


class HeatmapEngine:
    """
    Per-camera heatmap engine.

    Modes:
        live    - real-time decay heatmap
        hourly  - current hour accumulated heatmap
        daily   - current day accumulated heatmap

    Important:
        When visualization is OFF, data collection still continues
        if collect_when_off = true.
    """

    def __init__(self, config, camera_id: str):
        self.camera_id = camera_id

        self.gw = int(config.get("heatmap.grid_w", 64))
        self.gh = int(config.get("heatmap.grid_h", 36))

        self.collect_when_off = bool(config.get("heatmap.collect_when_off", True))
        self.blur_enabled = bool(config.get("heatmap.blur", True))
        self.opacity = float(config.get("heatmap.opacity", 0.6))

        decay_name = str(config.get("heatmap.decay", "normal")).lower()

        self.decay_rate = {
            "fast": 0.985,
            "normal": 0.9965,
            "slow": 0.9992,
        }.get(decay_name, 0.9965)

        self.live = np.zeros((self.gh, self.gw), dtype=np.float32)
        self.hist = np.zeros((self.gh, self.gw), dtype=np.float32)
        self.daily = np.zeros((self.gh, self.gw), dtype=np.float32)

        self.hourly = {
            h: np.zeros((self.gh, self.gw), dtype=np.float32)
            for h in range(24)
        }

        self.mode = "live"
        self.on = False

    # ---------------- controls ----------------
    def set_on(self, on: bool):
        self.on = bool(on)

    def set_mode(self, mode: str):
        if mode in ("live", "hourly", "daily"):
            self.mode = mode

    def reset(self, mode: str = None):
        mode = mode or self.mode

        if mode == "live":
            self.live.fill(0.0)
            self.hist.fill(0.0)

        elif mode == "hourly":
            self.hourly[datetime.now().hour].fill(0.0)

        elif mode == "daily":
            self.daily.fill(0.0)

        elif mode == "all":
            self.live.fill(0.0)
            self.hist.fill(0.0)
            self.daily.fill(0.0)
            for h in range(24):
                self.hourly[h].fill(0.0)

    # ---------------- update ----------------
    # def update(self, persons, frame_w: int, frame_h: int, online: bool = True):
    #     if not online and not self.collect_when_off:
    #         return

    #     if frame_w <= 0 or frame_h <= 0:
    #         return

    #     hour = datetime.now().hour

    #     for p in persons:
    #         ankle = getattr(p, "ankle", None)

    #         if ankle is None:
    #             continue

    #         x = float(ankle[0]) / float(frame_w)
    #         y = float(ankle[1]) / float(frame_h)

    #         if x < 0 or x > 1 or y < 0 or y > 1:
    #             continue

    #         self._splat(self.live, x, y, 0.045)
    #         self._splat(self.hist, x, y, 0.0035)
    #         self._splat(self.hourly[hour], x, y, 0.010)
    #         self._splat(self.daily, x, y, 0.005)

    #     self._decay(self.live, self.decay_rate)


    def update(self, persons, frame_w: int, frame_h: int, online: bool = True):
        print("HEATMAP update persons:", len(persons), flush=True)

    def _splat(self, grid, nx, ny, base_weight):
        ix = int(nx * self.gw)
        iy = int(ny * self.gh)

        for dy in (-2, -1, 0, 1, 2):
            yy = iy + dy

            if yy < 0 or yy >= self.gh:
                continue

            for dx in (-2, -1, 0, 1, 2):
                xx = ix + dx

                if xx < 0 or xx >= self.gw:
                    continue

                w = math.exp(-(dx * dx + dy * dy) / 2.2)
                grid[yy, xx] = min(1.6, grid[yy, xx] + w * base_weight)

    def _decay(self, grid, rate):
        grid *= rate

    # ---------------- image ----------------
    def _grid_for_mode(self, mode: str = None):
        mode = mode or self.mode

        if mode == "live":
            return self.live * 0.75 + self.hist * 0.6

        if mode == "hourly":
            hour = datetime.now().hour
            return self.hourly[hour] * 1.0 + self.hist * 0.15

        if mode == "daily":
            return self.daily * 1.0 + self.hist * 0.15

        return self.live

    def get_image(self, mode: str = None):
        grid = self._grid_for_mode(mode)

        if self.blur_enabled:
            grid = blur_grid(grid)

        img = QImage(self.gw, self.gh, QImage.Format_RGBA8888)
        img.fill(Qt.transparent)

        for y in range(self.gh):
            for x in range(self.gw):
                v = float(grid[y, x])

                if v <= 0.05:
                    continue

                r, g, b = heat_color(min(1.0, v))
                a = min(235, int(60 + v * 160))

                img.setPixelColor(x, y, QColor(r, g, b, a))

        return img