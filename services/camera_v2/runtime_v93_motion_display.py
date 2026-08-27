from __future__ import annotations

import math
import os
import time
from collections import deque

from .runtime_v92_current_bbox import PascalCurrentFrameBboxRuntime


class PascalMotionDisplayRuntime(PascalCurrentFrameBboxRuntime):
    """V9.3: reduce visible walking lag without mutating NvDCF tracker state.

    V9.2 removed false all-source empty updates and stale detector geometry.  The
    remaining display delay is mostly the gap between real NvDCF outputs (8-10 Hz)
    and the 20 Hz display.  V9.3 keeps the latest two *real* NvDCF display boxes per
    object and applies a very small, bounded center-only extrapolation at draw time.

    This is presentation-only: it never writes predicted geometry back into NvDCF,
    detector association, identity state, or the track cache.  Width/height remain
    the latest real NvDCF size, and extrapolation is clamped by both time and box
    diagonal so a noisy velocity estimate cannot teleport a rectangle.
    """

    def __init__(self) -> None:
        self.v93_horizon_ms = max(
            20.0,
            min(80.0, float(os.environ.get("CAMERA_V93_DISPLAY_COMP_MS", "55"))),
        )
        self.v93_gain = max(
            0.30,
            min(1.00, float(os.environ.get("CAMERA_V93_DISPLAY_COMP_GAIN", "0.85"))),
        )
        self.v93_max_shift_frac = max(
            0.05,
            min(0.35, float(os.environ.get("CAMERA_V93_MAX_SHIFT_FRAC", "0.20"))),
        )
        self.v93_min_sample_dt = max(
            0.025,
            min(0.12, float(os.environ.get("CAMERA_V93_MIN_SAMPLE_DT", "0.045"))),
        )
        self.v93_max_sample_dt = max(
            0.15,
            min(0.50, float(os.environ.get("CAMERA_V93_MAX_SAMPLE_DT", "0.30"))),
        )
        self._v93_motion: dict[
            tuple[int, int], tuple[float, float, float, float, float]
        ] = {}
        # key -> (updated, cx, cy, vx, vy); velocity is display pixels / second.
        self.v93_projected_draws = 0
        self.v93_unprojected_draws = 0
        self.v93_shift_px: deque[float] = deque(maxlen=4096)
        self.v93_age_ms: deque[float] = deque(maxlen=4096)
        super().__init__()
        print(
            "CAMERA_V93_ARCH "
            f"tracker={self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            f"display_comp={self.v93_horizon_ms:.0f}ms gain={self.v93_gain:.2f} "
            f"max_shift={self.v93_max_shift_frac:.2f}diag "
            "center_only=1 size_prediction=0 tracker_state_mutation=0 detector=v91-inprocess",
            flush=True,
        )

    @staticmethod
    def _percentile_v93(values, p: float) -> float:
        rows = sorted(float(v) for v in values)
        if not rows:
            return 0.0
        idx = min(len(rows) - 1, max(0, int(round((len(rows) - 1) * float(p)))))
        return rows[idx]

    def _tracker_probe(self, pad, info):
        ret = super()._tracker_probe(pad, info)
        now = time.monotonic()
        with self.track_cache_lock:
            cache = dict(self.track_cache)

        active: set[tuple[int, int]] = set()
        for source_id, (updated, tracks) in cache.items():
            for track in tracks:
                object_id = int(track[0])
                key = (int(source_id), object_id)
                active.add(key)
                left, top, right, bottom = (float(v) for v in track[1:5])
                cx = 0.5 * (left + right)
                cy = 0.5 * (top + bottom)
                previous = self._v93_motion.get(key)
                if previous is not None and abs(float(updated) - previous[0]) < 1e-6:
                    continue

                vx = vy = 0.0
                if previous is not None:
                    prev_t, prev_cx, prev_cy, prev_vx, prev_vy = previous
                    dt = float(updated) - prev_t
                    if self.v93_min_sample_dt <= dt <= self.v93_max_sample_dt:
                        inst_vx = (cx - prev_cx) / dt
                        inst_vy = (cy - prev_cy) / dt
                        # Mild velocity smoothing only.  The actual draw displacement
                        # is additionally bounded below, so this cannot become a long
                        # CPU tracker/predictor.
                        alpha = 0.70
                        vx = alpha * inst_vx + (1.0 - alpha) * prev_vx
                        vy = alpha * inst_vy + (1.0 - alpha) * prev_vy
                self._v93_motion[key] = (float(updated), cx, cy, vx, vy)

        for key, row in list(self._v93_motion.items()):
            if key not in active and now - row[0] > 0.6:
                self._v93_motion.pop(key, None)
        return ret

    def _project_track_v93(self, source_id: int, track, age_ms: float):
        object_id = int(track[0])
        left, top, right, bottom = (float(v) for v in track[1:5])
        conf = float(track[5])
        width = max(2.0, right - left)
        height = max(2.0, bottom - top)
        motion = self._v93_motion.get((int(source_id), object_id))
        if motion is None or age_ms <= 0.0:
            self.v93_unprojected_draws += 1
            return (object_id, left, top, right, bottom, conf)

        _updated, _cx, _cy, vx, vy = motion
        speed = math.hypot(vx, vy)
        if speed < 1.0:
            self.v93_unprojected_draws += 1
            return (object_id, left, top, right, bottom, conf)

        horizon_s = min(age_ms, self.v93_horizon_ms) / 1000.0
        dx = vx * horizon_s * self.v93_gain
        dy = vy * horizon_s * self.v93_gain
        diag = math.hypot(width, height)
        max_shift = self.v93_max_shift_frac * max(4.0, diag)
        shift = math.hypot(dx, dy)
        if shift > max_shift and shift > 1e-6:
            scale = max_shift / shift
            dx *= scale
            dy *= scale
            shift = max_shift

        # Keep the real NvDCF size; move only its center.  Clip by translating the
        # whole box back inside the display instead of shrinking it at the border.
        new_left = left + dx
        new_right = right + dx
        new_top = top + dy
        new_bottom = bottom + dy
        if new_left < 0.0:
            new_right -= new_left
            new_left = 0.0
        if new_right > self.display_width:
            delta = new_right - self.display_width
            new_left -= delta
            new_right = float(self.display_width)
        if new_top < 0.0:
            new_bottom -= new_top
            new_top = 0.0
        if new_bottom > self.display_height:
            delta = new_bottom - self.display_height
            new_top -= delta
            new_bottom = float(self.display_height)

        self.v93_projected_draws += 1
        self.v93_shift_px.append(float(shift))
        return (object_id, new_left, new_top, new_right, new_bottom, conf)

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
            projected = [
                self._project_track_v93(int(source_id), track, age_ms) for track in tracks
            ]
            self.v93_age_ms.append(age_ms)
            # Keep V9.2 diagnostics populated so its checker/stats remain meaningful.
            self.v92_overlay_age_samples.append(age_ms)
            self.v92_overlay_draws += len(projected)
            self.bridge.add_tracked_boxes(buffer, source_id, projected)

        self.bridge.apply_local_track_style(buffer)
        return self.Gst.PadProbeReturn.OK

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        ages = list(self.v93_age_ms)
        shifts = list(self.v93_shift_px)
        print(
            "CAMERA_V93_STATS "
            f"projected={self.v93_projected_draws} raw={self.v93_unprojected_draws} "
            f"age_p50={self._percentile_v93(ages, 0.50):.0f}ms "
            f"age_p95={self._percentile_v93(ages, 0.95):.0f}ms "
            f"shift_p50={self._percentile_v93(shifts, 0.50):.1f}px "
            f"shift_p95={self._percentile_v93(shifts, 0.95):.1f}px "
            f"gpu_ema={self.v84_gpu_ms_ema:.1f}ms tracker_batches={self.tracker_batches} "
            f"tracked_now={self.tracked_now}",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalMotionDisplayRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
