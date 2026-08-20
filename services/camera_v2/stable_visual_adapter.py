from __future__ import annotations

"""Adapter for the exact old stable adaptive-Kalman/Byte visual tracker.

The core implementation is copied byte-for-byte from
``agent/stable-detection-ui-baseline`` into ``stable_visual_tracker_core.py``.
This adapter only translates Camera V2's lightweight tuple API and adds bounded
20-FPS optical-flow corrections. Detector observations remain the only source of
track births and long-term truth; flow can move an already-confirmed track but
can never create one or refresh its detector age indefinitely.
"""

import math
import os
import threading
from dataclasses import dataclass

import numpy as np

from .stable_visual_tracker_core import VisualBox, VisualTracker


@dataclass(frozen=True)
class _Result:
    frame_id: int
    frame_captured_monotonic: float
    boxes: tuple[VisualBox, ...]


class StableVisualFlowBoxManager:
    """Per-camera old-stable tracker with bounded optical-flow assistance."""

    def __init__(self, width: int, height: int) -> None:
        self.width = float(width)
        self.height = float(height)
        self.lock = threading.RLock()
        self._trackers: dict[str, VisualTracker] = {}
        self._frame_ids: dict[str, int] = {}
        self._last_flow: dict[tuple[str, int], tuple[float, float]] = {}

        # Old stable tracker defaults, with a longer visible hold because the
        # GTX 1050 Ti intentionally runs sparse detector corrections. The actual
        # motion during that hold is measured by optical flow, not open-loop
        # extrapolation. Memory remains bounded so a departed person cannot live
        # forever as a ghost.
        self.hold_ms = int(os.environ.get("CAMERA_V2_STABLE_HOLD_MS", "2800"))
        self.memory_ms = int(os.environ.get("CAMERA_V2_STABLE_MEMORY_MS", "4800"))
        self.prediction_ms = int(os.environ.get("CAMERA_V2_STABLE_PREDICTION_MS", "420"))
        self.flow_recent_sec = float(os.environ.get("CAMERA_V2_FLOW_RECENT_SEC", "0.20"))
        self.flow_min_quality = float(os.environ.get("CAMERA_V2_FLOW_MIN_QUALITY", "0.30"))
        self.flow_gain = float(os.environ.get("CAMERA_V2_FLOW_GAIN", "0.90"))
        self.flow_hard_age_sec = float(os.environ.get("CAMERA_V2_FLOW_HARD_AGE_SEC", "4.20"))

    def _new_tracker(self) -> VisualTracker:
        return VisualTracker(
            hold_ms=self.hold_ms,
            memory_ms=self.memory_ms,
            prediction_ms=self.prediction_ms,
            match_iou=float(os.environ.get("CAMERA_V2_STABLE_MATCH_IOU", "0.12")),
            reacquire_distance=float(os.environ.get("CAMERA_V2_STABLE_REACQUIRE_DISTANCE", "0.85")),
            duplicate_iou=float(os.environ.get("CAMERA_V2_STABLE_DUPLICATE_IOU", "0.68")),
            duplicate_containment=float(os.environ.get("CAMERA_V2_STABLE_DUPLICATE_CONTAINMENT", "0.90")),
            duplicate_center_distance=float(os.environ.get("CAMERA_V2_STABLE_DUPLICATE_CENTER", "0.20")),
            low_conf_confirm=float(os.environ.get("CAMERA_V2_STABLE_LOW_CONFIRM", "0.08")),
            start_conf=float(os.environ.get("CAMERA_V2_STABLE_START_CONF", "0.24")),
            new_track_min_conf=float(os.environ.get("CAMERA_V2_STABLE_NEW_TRACK_CONF", "0.18")),
            strong_confirm_hits=max(1, int(os.environ.get("CAMERA_V2_STABLE_STRONG_HITS", "2"))),
            weak_confirm_hits=max(2, int(os.environ.get("CAMERA_V2_STABLE_WEAK_HITS", "3"))),
            byte_high_conf=float(os.environ.get("CAMERA_V2_STABLE_BYTE_HIGH", "0.24")),
            byte_low_conf=float(os.environ.get("CAMERA_V2_STABLE_BYTE_LOW", "0.08")),
            byte_second_match_iou=float(os.environ.get("CAMERA_V2_STABLE_SECOND_IOU", "0.04")),
            byte_match_center=float(os.environ.get("CAMERA_V2_STABLE_MATCH_CENTER", "0.70")),
            byte_second_match_center=float(os.environ.get("CAMERA_V2_STABLE_SECOND_CENTER", "0.50")),
            low_match_max_age_ms=float(os.environ.get("CAMERA_V2_STABLE_LOW_MATCH_AGE_MS", "900")),
            process_noise=float(os.environ.get("CAMERA_V2_STABLE_PROCESS_NOISE", "0.85")),
            measurement_noise=float(os.environ.get("CAMERA_V2_STABLE_MEASUREMENT_NOISE", "0.90")),
            velocity_damping=float(os.environ.get("CAMERA_V2_STABLE_VELOCITY_DAMPING", "0.96")),
            size_velocity_damping=float(os.environ.get("CAMERA_V2_STABLE_SIZE_DAMPING", "0.60")),
            max_prediction_shift_boxes=float(os.environ.get("CAMERA_V2_STABLE_MAX_SHIFT", "0.68")),
            max_prediction_size_ratio=float(os.environ.get("CAMERA_V2_STABLE_MAX_SIZE_RATIO", "0.08")),
            adaptive_error_low=float(os.environ.get("CAMERA_V2_STABLE_ERROR_LOW", "0.08")),
            adaptive_error_high=float(os.environ.get("CAMERA_V2_STABLE_ERROR_HIGH", "0.25")),
            center_response_slow=float(os.environ.get("CAMERA_V2_STABLE_CENTER_SLOW", "0.42")),
            center_response_fast=float(os.environ.get("CAMERA_V2_STABLE_CENTER_FAST", "0.88")),
            size_response=float(os.environ.get("CAMERA_V2_STABLE_SIZE_RESPONSE", "0.30")),
            snap_distance_boxes=float(os.environ.get("CAMERA_V2_STABLE_SNAP_DISTANCE", "0.65")),
            reversal_damping=float(os.environ.get("CAMERA_V2_STABLE_REVERSAL_DAMPING", "0.15")),
        )

    def _tracker(self, cid: str) -> VisualTracker:
        tracker = self._trackers.get(cid)
        if tracker is None:
            tracker = self._new_tracker()
            self._trackers[cid] = tracker
        return tracker

    def update(self, cid: str, captured_t: float, detections) -> None:
        with self.lock:
            tracker = self._tracker(cid)
            frame_id = self._frame_ids.get(cid, 0) + 1
            self._frame_ids[cid] = frame_id
            boxes = []
            for box, confidence in detections or ():
                try:
                    x1, y1, x2, y2 = [float(v) for v in box]
                    conf = float(confidence)
                except (TypeError, ValueError, OverflowError):
                    continue
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2, conf)):
                    continue
                if conf <= 0.0 or x2 <= x1 or y2 <= y1:
                    continue
                boxes.append(VisualBox(x1, y1, x2, y2, conf))

            result = _Result(
                frame_id=frame_id,
                frame_captured_monotonic=float(captured_t),
                boxes=tuple(boxes),
            )
            tracker.update(
                result,
                now=float(captured_t),
                source_width=self.width,
                source_height=self.height,
            )

    @staticmethod
    def _clip_box(box: VisualBox, width: float, height: float):
        x1 = max(0.0, min(width - 2.0, float(box.x1)))
        y1 = max(0.0, min(height - 2.0, float(box.y1)))
        x2 = max(x1 + 1.0, min(width - 1.0, float(box.x2)))
        y2 = max(y1 + 1.0, min(height - 1.0, float(box.y2)))
        return x1, y1, x2, y2

    def render(self, cid: str, now: float):
        with self.lock:
            tracker = self._trackers.get(cid)
            if tracker is None:
                return []

            visible = tracker.visible(float(now), target_time=float(now))
            rows = [
                (*self._clip_box(box, self.width, self.height), float(box.confidence))
                for box in visible
            ]

            # The exact stable tracker intentionally hides a track after hold_sec.
            # On this sparse-GPU deployment we allow a short extra window only if
            # verified LK motion was seen very recently. This keeps seated/back-
            # facing people visible between detector corrections without allowing
            # an unmeasured ghost to survive indefinitely.
            with tracker._lock:
                for tid, track in tracker._tracks.items():
                    age = max(0.0, float(now) - float(track.last_observation))
                    if age <= tracker.hold_sec or age > self.flow_hard_age_sec:
                        continue
                    if track.hits < tracker.strong_confirm_hits:
                        continue
                    flow_t, flow_quality = self._last_flow.get((cid, int(tid)), (0.0, 0.0))
                    if float(now) - flow_t > self.flow_recent_sec or flow_quality < self.flow_min_quality:
                        continue
                    box = tracker._box_from_track(track, track.mean)
                    conf = max(0.05, float(track.confidence) * math.exp(-0.22 * max(0.0, age - tracker.hold_sec)))
                    rows.append((*self._clip_box(box, self.width, self.height), conf))
            return rows

    def flow_regions(self, cid: str, now: float):
        with self.lock:
            tracker = self._trackers.get(cid)
            if tracker is None:
                return []
            output = []
            with tracker._lock:
                for tid, track in tracker._tracks.items():
                    age = max(0.0, float(now) - float(track.last_observation))
                    if age > self.flow_hard_age_sec or track.hits < tracker.strong_confirm_hits:
                        continue
                    flow_t, _quality = self._last_flow.get((cid, int(tid)), (0.0, 0.0))
                    if flow_t > 0.0 and float(now) - flow_t <= self.flow_recent_sec:
                        box = tracker._box_from_track(track, track.mean)
                    else:
                        predicted, _cov, _horizon = tracker._bounded_prediction(track, float(now))
                        box = tracker._box_from_track(track, predicted)
                    output.append(
                        {
                            "track_id": int(tid),
                            "box": self._clip_box(box, self.width, self.height),
                            "confirmed": True,
                            "age": age,
                        }
                    )
            return output

    def apply_flow(self, cid: str, track_id: int, dx: float, dy: float, now: float, quality: float) -> bool:
        quality = float(quality)
        if not math.isfinite(quality) or quality < self.flow_min_quality:
            return False
        with self.lock:
            tracker = self._trackers.get(cid)
            if tracker is None:
                return False
            with tracker._lock:
                track = tracker._tracks.get(int(track_id))
                if track is None or track.hits < tracker.strong_confirm_hits:
                    return False
                age = max(0.0, float(now) - float(track.last_observation))
                if age > self.flow_hard_age_sec:
                    return False

                max_dx = self.width * 0.045
                max_dy = self.height * 0.060
                dx = float(np.clip(float(dx), -max_dx, max_dx))
                dy = float(np.clip(float(dy), -max_dy, max_dy))
                gain = float(np.clip(self.flow_gain * (0.72 + 0.28 * quality), 0.55, 0.96))
                move_x = dx * gain
                move_y = dy * gain
                track.mean[0] = float(np.clip(track.mean[0] + move_x, 0.0, self.width - 1.0))
                track.mean[1] = float(np.clip(track.mean[1] + move_y, 0.0, self.height - 1.0))

                old_flow_t, _old_quality = self._last_flow.get((cid, int(track_id)), (0.0, 0.0))
                if old_flow_t > 0.0:
                    dt = float(np.clip(float(now) - old_flow_t, 0.025, 0.20))
                    measured = np.asarray([move_x / dt, move_y / dt], dtype=np.float64)
                    track.mean[4:6] = 0.55 * track.mean[4:6] + 0.45 * measured
                self._last_flow[(cid, int(track_id))] = (float(now), quality)
                return True

    def anchors(self, cid: str, now: float):
        with self.lock:
            tracker = self._trackers.get(cid)
            if tracker is None:
                return []
            output = []
            with tracker._lock:
                for tid, track in tracker._tracks.items():
                    age = max(0.0, float(now) - float(track.last_observation))
                    if age > self.flow_hard_age_sec or track.hits < tracker.strong_confirm_hits:
                        continue
                    output.append(
                        {
                            "track_id": int(tid),
                            "cx": float(np.clip(track.mean[0], 0.0, self.width - 1.0)),
                            "cy": float(np.clip(track.mean[1], 0.0, self.height - 1.0)),
                            "age": age,
                            "confirmed": True,
                            "confidence": float(track.confidence),
                        }
                    )
            return output

    def metrics(self):
        with self.lock:
            return {cid: tracker.metrics() for cid, tracker in self._trackers.items()}
