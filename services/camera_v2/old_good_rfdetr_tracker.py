from __future__ import annotations

"""Core-v1 'old good' presentation tracking adapted to RF-DETR-S.

This module intentionally reuses the exact adaptive Kalman/Byte implementation
already restored in :mod:`stable_visual_tracker_core`.  Only the tuple adapter,
RF-DETR confidence calibration, camera-specific birth/exclusion policy and stale
result gate live here.

The canonical logic comes from the old ``rebuild/core-v1-clean`` detection stack:
short visual hold, bounded prediction, 3 second reacquire memory, high/low Byte
association, duplicate/fragment suppression, camera-specific birth thresholds and
CAM-06 false-positive exclusion.  No optical flow or open-ended ghost extension is
used in this layer.
"""

import math
import os
import threading
import time
from types import SimpleNamespace

from .stable_visual_tracker_core import VisualBox, VisualTracker


class _DetectionResult:
    __slots__ = ("frame_id", "frame_captured_monotonic", "boxes")

    def __init__(self, frame_id: int, captured: float, boxes: tuple[VisualBox, ...]):
        self.frame_id = int(frame_id)
        self.frame_captured_monotonic = float(captured)
        self.boxes = boxes


class OldGoodRFDETRBoxManager:
    """Per-camera old-good Kalman/Byte presentation state for RF-DETR boxes."""

    _FRAGMENT_CAMERAS = {"CAM-03", "CAM-05", "CAM-06"}
    _CAMERA_START_CONF = {"CAM-05": 0.24, "CAM-06": 0.28}
    _CAMERA_LOW_CONF = {"CAM-05": 0.12}
    _CAMERA_BIRTH_ZONES = {
        "CAM-05": [(0.27, 0.00, 0.72, 0.54, 0.30)],
        "CAM-06": [(0.36, 0.12, 0.76, 0.49, 0.32)],
    }
    _CAMERA_EXCLUSION_ZONES = {
        "CAM-06": [(0.50, 0.00, 0.78, 0.22)],
    }

    def __init__(self, width: int, height: int) -> None:
        self.width = float(width)
        self.height = float(height)
        self.lock = threading.RLock()
        self._trackers: dict[str, VisualTracker] = {}
        self._frame_ids: dict[str, int] = {}
        self._stale_drops = 0

        # Core-v1 used 800/3000/420.  Keep those semantics.  The launcher drives
        # RF-DETR frequently enough that the short hold remains meaningful.
        self.hold_ms = int(os.environ.get("CAMERA_V2_OLDGOOD_HOLD_MS", "850"))
        self.memory_ms = int(os.environ.get("CAMERA_V2_OLDGOOD_MEMORY_MS", "3000"))
        self.prediction_ms = int(os.environ.get("CAMERA_V2_OLDGOOD_PREDICTION_MS", "420"))
        self.max_result_age = float(
            os.environ.get("CAMERA_V2_OLDGOOD_MAX_RESULT_AGE_SEC", "0.95")
        )
        self.max_age = max(0.05, self.hold_ms / 1000.0)

    def _new_tracker(self, cid: str) -> VisualTracker:
        # RF-DETR confidence is not numerically identical to YOLO confidence.
        # Preserve the old birth/Byte structure while using slightly lower RF-DETR
        # gates; the detector itself still has a separate confidence threshold.
        start_conf = self._CAMERA_START_CONF.get(cid, 0.24)
        low_conf = self._CAMERA_LOW_CONF.get(cid, 0.10)
        return VisualTracker(
            hold_ms=self.hold_ms,
            memory_ms=self.memory_ms,
            prediction_ms=self.prediction_ms,
            match_iou=0.12,
            reacquire_distance=0.85,
            duplicate_iou=0.68,
            duplicate_containment=0.90,
            duplicate_center_distance=0.20,
            fragment_duplicate=cid in self._FRAGMENT_CAMERAS,
            fragment_horizontal_overlap=0.78,
            fragment_x_center=0.18,
            fragment_max_area_ratio=0.55,
            fragment_min_vertical_overlap=0.20,
            fragment_max_vertical_gap=0.06,
            low_conf_confirm=low_conf,
            start_conf=start_conf,
            new_track_min_conf=0.18,
            strong_confirm_hits=2,
            weak_confirm_hits=3,
            byte_high_conf=0.22,
            byte_low_conf=0.10,
            byte_second_match_iou=0.04,
            byte_match_center=0.70,
            byte_second_match_center=0.50,
            low_match_max_age_ms=650,
            process_noise=0.85,
            measurement_noise=0.90,
            velocity_damping=0.96,
            size_velocity_damping=0.60,
            max_prediction_shift_boxes=0.55,
            max_prediction_size_ratio=0.08,
            adaptive_error_low=0.08,
            adaptive_error_high=0.25,
            center_response_slow=0.42,
            center_response_fast=0.88,
            size_response=0.30,
            snap_distance_boxes=0.65,
            reversal_damping=0.15,
            new_track_zones=self._CAMERA_BIRTH_ZONES.get(cid, ()),
            exclusion_zones=self._CAMERA_EXCLUSION_ZONES.get(cid, ()),
            exclusion_max_box_height=0.30,
            exclusion_overlap_threshold=0.15,
        )

    def _tracker(self, cid: str) -> VisualTracker:
        tracker = self._trackers.get(cid)
        if tracker is None:
            tracker = self._new_tracker(cid)
            self._trackers[cid] = tracker
        return tracker

    @staticmethod
    def _valid_box(box, confidence: float) -> VisualBox | None:
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
            conf = float(confidence)
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, conf)):
            return None
        if conf <= 0.0 or x2 <= x1 or y2 <= y1:
            return None
        return VisualBox(x1, y1, x2, y2, conf)

    def update(self, cid: str, captured_t: float, detections) -> None:
        now = time.monotonic()
        age = max(0.0, now - float(captured_t))
        if self.max_result_age > 0.0 and age > self.max_result_age:
            with self.lock:
                self._stale_drops += 1
            return

        with self.lock:
            tracker = self._tracker(cid)
            frame_id = self._frame_ids.get(cid, 0) + 1
            self._frame_ids[cid] = frame_id
            boxes = tuple(
                valid
                for box, confidence in (detections or ())
                if (valid := self._valid_box(box, confidence)) is not None
            )
            tracker.update(
                _DetectionResult(frame_id, float(captured_t), boxes),
                now=now,
                source_width=self.width,
                source_height=self.height,
            )

    @staticmethod
    def _clip(box: VisualBox, width: float, height: float):
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
            return [
                (*self._clip(box, self.width, self.height), float(box.confidence))
                for box in visible
            ]

    @property
    def tracks(self):
        """Compatibility view consumed by the existing Pascal status counter."""
        output: dict[str, dict[int, object]] = {}
        with self.lock:
            for cid, tracker in self._trackers.items():
                rows: dict[int, object] = {}
                with tracker._lock:
                    for tid, track in tracker._tracks.items():
                        rows[int(tid)] = SimpleNamespace(
                            last_det_t=float(track.last_observation),
                            confidence=float(track.confidence),
                        )
                output[cid] = rows
        return output

    def metrics(self) -> dict:
        with self.lock:
            return {
                "stale_drops": self._stale_drops,
                "cameras": {
                    cid: tracker.metrics() for cid, tracker in self._trackers.items()
                },
            }
