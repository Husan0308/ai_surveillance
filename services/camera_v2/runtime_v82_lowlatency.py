from __future__ import annotations

import math
import os
import time

from .runtime_v81_sync import PascalStickySyncRuntime


class PascalLowLatencyOverlayRuntime(PascalStickySyncRuntime):
    """V8.2: restore V8 GPU budget and compensate only the visible bbox.

    V8.1 raised NvDCF from 8 Hz to 12 Hz. On GP107/Pascal that can starve the
    TRT8.6 sidecar because DeepStream and the detector live in separate CUDA
    process contexts. V8.2 goes back to the proven 8 Hz tracker cadence and does
    not alter tracker/detector association geometry.

    The visible 20 FPS wall would otherwise hold an 8 Hz tracker rectangle for
    up to ~125 ms. To remove that visual trailing without spending more GPU, the
    display rectangle receives a short, bounded translation derived from the last
    two real NvDCF centers of the same local track. Size is never predicted,
    association is never predicted, and the compensation horizon is capped.
    """

    def __init__(self) -> None:
        self.v82_comp_max_ms = max(
            20.0,
            min(120.0, float(os.environ.get("CAMERA_V82_DISPLAY_COMP_MAX_MS", "85"))),
        )
        self.v82_comp_gain = max(
            0.0,
            min(1.0, float(os.environ.get("CAMERA_V82_DISPLAY_COMP_GAIN", "0.82"))),
        )
        self.v82_comp_max_diag = max(
            0.05,
            min(0.50, float(os.environ.get("CAMERA_V82_DISPLAY_COMP_MAX_DIAG", "0.30"))),
        )
        self.v82_motion_min_dt = max(
            0.02,
            min(0.12, float(os.environ.get("CAMERA_V82_MOTION_MIN_DT", "0.045"))),
        )
        self.v82_motion_max_dt = max(
            self.v82_motion_min_dt,
            min(0.40, float(os.environ.get("CAMERA_V82_MOTION_MAX_DT", "0.24"))),
        )
        self._v82_motion: dict[
            tuple[int, int],
            tuple[float, tuple[float, float, float, float], float, float],
        ] = {}
        self.v82_comp_draws = 0
        self.v82_comp_shift_sum = 0.0
        self.v82_comp_shift_max = 0.0
        super().__init__()
        print(
            "CAMERA_V82_LOWLATENCY "
            f"tracker={self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            f"display_comp_max={self.v82_comp_max_ms:.0f}ms "
            f"gain={self.v82_comp_gain:.2f} max_shift={self.v82_comp_max_diag:.2f}diag "
            "association_prediction=0 detector_prediction=0",
            flush=True,
        )

    @staticmethod
    def _box_center(box):
        return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])

    def _remember_motion(
        self,
        source_id: int,
        object_id: int,
        box: tuple[float, float, float, float],
        now: float,
    ) -> None:
        key = (int(source_id), int(object_id))
        previous = self._v82_motion.get(key)
        vx = 0.0
        vy = 0.0
        if previous is not None:
            prev_time, prev_box, prev_vx, prev_vy = previous
            dt = now - prev_time
            if self.v82_motion_min_dt <= dt <= self.v82_motion_max_dt:
                pcx, pcy = self._box_center(prev_box)
                cx, cy = self._box_center(box)
                measured_vx = (cx - pcx) / dt
                measured_vy = (cy - pcy) / dt
                # Mild velocity EMA avoids reacting to one noisy DCF center jump.
                vx = 0.60 * measured_vx + 0.40 * prev_vx
                vy = 0.60 * measured_vy + 0.40 * prev_vy
        self._v82_motion[key] = (now, box, vx, vy)

    def _tracker_probe(self, pad, info):
        result = super()._tracker_probe(pad, info)
        now = time.monotonic()
        # The parent has already atomically published real NvDCF rows. Learn motion
        # only from fresh published rows; held cache entries never generate velocity.
        with self.track_cache_lock:
            cache = dict(self.track_cache)
        active: set[tuple[int, int]] = set()
        for source_id, (updated, tracks) in cache.items():
            if abs(updated - now) > 0.060:
                continue
            for track in tracks:
                key = (int(source_id), int(track[0]))
                active.add(key)
                box = tuple(float(v) for v in track[1:5])
                existing = self._v82_motion.get(key)
                if existing is None or updated > existing[0] + 1e-4:
                    self._remember_motion(source_id, int(track[0]), box, updated)
        for key, row in list(self._v82_motion.items()):
            if key not in active and now - row[0] > 1.0:
                self._v82_motion.pop(key, None)
        return result

    def _compensate_track(self, source_id: int, track, age_ms: float):
        object_id = int(track[0])
        x1, y1, x2, y2 = (float(v) for v in track[1:5])
        conf = float(track[5])
        state = self._v82_motion.get((int(source_id), object_id))
        if state is None or self.v82_comp_gain <= 0.0:
            return track, 0.0
        _updated, _box, vx, vy = state
        if abs(vx) + abs(vy) < 1e-3:
            return track, 0.0

        horizon = min(max(0.0, age_ms), self.v82_comp_max_ms) / 1000.0
        dx = vx * horizon * self.v82_comp_gain
        dy = vy * horizon * self.v82_comp_gain

        diag = math.hypot(max(2.0, x2 - x1), max(2.0, y2 - y1))
        max_shift = self.v82_comp_max_diag * max(1.0, diag)
        shift = math.hypot(dx, dy)
        if shift > max_shift and shift > 1e-6:
            scale = max_shift / shift
            dx *= scale
            dy *= scale
            shift = max_shift

        # Translation only. Stable-size logic remains the sole owner of box size.
        width = max(2.0, x2 - x1)
        height = max(2.0, y2 - y1)
        nx1 = max(0.0, min(float(self.display_width - width), x1 + dx))
        ny1 = max(0.0, min(float(self.display_height - height), y1 + dy))
        nx2 = nx1 + width
        ny2 = ny1 + height
        return (object_id, nx1, ny1, nx2, ny2, conf), shift

    def _display_overlay_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None or not self.analytics_enabled:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        with self.track_cache_lock:
            cache = dict(self.track_cache)
        for source_id in self.index_camera:
            row = cache.get(source_id)
            if row is None:
                continue
            updated, tracks = row
            age_ms = max(0.0, (now - updated) * 1000.0)
            if age_ms > self.display_track_max_age_ms:
                continue
            self.v81_overlay_age_samples.append(age_ms)
            self.v81_overlay_draws += 1
            if age_ms > (1000.0 / max(1.0, self.track_fps)) * 1.35:
                self.v81_overlay_held_draws += 1

            drawn = []
            for track in tracks:
                compensated, shift = self._compensate_track(source_id, track, age_ms)
                drawn.append(compensated)
                if shift > 0.0:
                    self.v82_comp_draws += 1
                    self.v82_comp_shift_sum += shift
                    self.v82_comp_shift_max = max(self.v82_comp_shift_max, shift)
            self.bridge.add_tracked_boxes(buffer, source_id, drawn)
        self.bridge.apply_local_track_style(buffer)
        return self.Gst.PadProbeReturn.OK

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        avg_shift = self.v82_comp_shift_sum / max(1, self.v82_comp_draws)
        print(
            "CAMERA_V82_STATS "
            f"comp_draws={self.v82_comp_draws} comp_shift_avg={avg_shift:.1f}px "
            f"comp_shift_max={self.v82_comp_shift_max:.1f}px "
            f"tracker_target={self.track_fps:.1f}Hz "
            "gpu_restore=v8-baseline display_only_comp=1 long_predictor=0",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalLowLatencyOverlayRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
