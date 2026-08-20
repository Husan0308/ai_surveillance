from __future__ import annotations

"""Optical-flow assisted short-term person tracking.

RF-DETR remains the source of truth for person detections. Between detector
corrections this tracker accepts small frame-to-frame motion measurements from a
continuous low-resolution optical-flow branch. Recent optical flow suppresses
open-loop velocity prediction, so display boxes follow measured image motion
instead of running ahead of or lagging behind the person.
"""

import os

from .temporal_tracker import AnchoredPersonTracker, _clamp


class FlowAssistedPersonTracker(AnchoredPersonTracker):
    """AnchoredPersonTracker with measured frame-motion corrections."""

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        # The previous 4.8 s display hold made a bad one-off detection visibly
        # linger in a static room. Keep the hard detector refresh window bounded.
        # Invisible probation candidates still use the base tentative window.
        if "CAMERA_V2_TRACK_HOLD_SEC" not in os.environ:
            self.max_age = 2.8
        self.flow_recent_sec = float(
            os.environ.get("CAMERA_V2_FLOW_RECENT_SEC", "0.18")
        )
        self.flow_min_quality = float(
            os.environ.get("CAMERA_V2_FLOW_MIN_QUALITY", "0.28")
        )
        self.flow_gain = float(os.environ.get("CAMERA_V2_FLOW_GAIN", "0.92"))

    def _predict_state(self, track, when: float):
        last_flow_t = float(getattr(track, "last_flow_t", 0.0) or 0.0)
        if last_flow_t > 0.0 and float(when) - last_flow_t <= self.flow_recent_sec:
            # The current center already came from measured frame motion. Do not
            # add detector-era velocity again or the box will overshoot.
            return track.cx, track.cy, track.w, track.h
        return super()._predict_state(track, when)

    def flow_regions(self, cid: str, now: float):
        """Return source-space boxes that optical flow may follow.

        Only detector-confirmed people are exposed to the continuous flow branch.
        A one-frame RF-DETR false positive therefore cannot acquire a long-lived
        background feature track before it passes birth probation.
        """
        with self.lock:
            current = self.tracks.get(cid, {})
            rows = []
            for tid, track in current.items():
                age = max(0.0, float(now) - track.last_det_t)
                if age > self.max_age or not track.confirmed:
                    continue
                x1, y1, x2, y2 = self._predict_box(track, now)
                if x2 <= x1 or y2 <= y1:
                    continue
                rows.append(
                    {
                        "track_id": int(tid),
                        "box": (float(x1), float(y1), float(x2), float(y2)),
                        "confirmed": True,
                        "age": float(age),
                    }
                )
            return rows

    def apply_flow(
        self,
        cid: str,
        track_id: int,
        dx: float,
        dy: float,
        now: float,
        quality: float,
    ) -> bool:
        """Apply one robust optical-flow displacement in source-frame pixels."""
        quality = float(quality)
        if quality < self.flow_min_quality:
            return False

        with self.lock:
            current = self.tracks.get(cid, {})
            track = current.get(int(track_id))
            if track is None or not track.confirmed:
                return False

            age = max(0.0, float(now) - track.last_det_t)
            if age > self.max_age:
                return False

            # One 20-FPS frame should never teleport a track. These limits are
            # deliberately generous for a fast walking person but reject LK
            # failures that lock onto a monitor/chair/background edge.
            max_dx = self.width * 0.045
            max_dy = self.height * 0.060
            dx = _clamp(dx, -max_dx, max_dx)
            dy = _clamp(dy, -max_dy, max_dy)

            gain = _clamp(self.flow_gain * (0.72 + quality * 0.28), 0.60, 0.97)
            move_x = dx * gain
            move_y = dy * gain
            track.cx = _clamp(track.cx + move_x, 0.0, self.width - 1.0)
            track.cy = _clamp(track.cy + move_y, 0.0, self.height - 1.0)

            previous_flow_t = float(getattr(track, "last_flow_t", 0.0) or 0.0)
            if previous_flow_t > 0.0:
                dt = _clamp(float(now) - previous_flow_t, 0.025, 0.20)
                measured_vx = move_x / dt
                measured_vy = move_y / dt
                # Flow velocity replaces stale detector velocity gradually; it
                # is not used while fresh flow is available, but is useful for a
                # brief dropped motion frame.
                track.vx = track.vx * 0.45 + measured_vx * 0.55
                track.vy = track.vy * 0.45 + measured_vy * 0.55

            track.last_flow_t = float(now)
            track.last_flow_quality = quality
            track.flow_hits = int(getattr(track, "flow_hits", 0)) + 1
            return True
