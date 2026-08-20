from __future__ import annotations

import time
from types import SimpleNamespace

from .visual_tracker import VisualBox, VisualTracker


class CoreV1VisualAdapter:
    """Adapter around the proven Core-v1 adaptive Kalman/Byte visual tracker.

    RF-DETR detections stay raw.  The adapter only owns presentation continuity:
    low-score continuation, temporal birth confirmation, duplicate suppression,
    capture-time motion compensation and bounded prediction to the current wall
    frame.  A final display-only guard keeps visible head/feet/hands inside the
    rectangle without contaminating future NvDCF/geometry measurements.
    """

    def __init__(self, width: int, height: int, cfg: dict) -> None:
        self.width = float(width)
        self.height = float(height)
        self.cfg = dict(cfg or {})
        self.trackers: dict[str, VisualTracker] = {}
        self.frame_ids: dict[str, int] = {}

    def _tracker(self, camera_id: str) -> VisualTracker:
        tracker = self.trackers.get(camera_id)
        if tracker is not None:
            return tracker

        cfg = self.cfg
        fragment_cameras = {str(v) for v in cfg.get("fragment_duplicate_cameras", [])}
        camera_start = dict(cfg.get("camera_start_conf") or {})
        camera_low = dict(cfg.get("camera_low_conf_confirm") or {})
        camera_birth_zones = dict(cfg.get("camera_new_track_zones") or {})
        camera_exclusion = dict(cfg.get("camera_exclusion_zones") or {})

        tracker = VisualTracker(
            hold_ms=int(cfg.get("hold_ms", 2200)),
            memory_ms=int(cfg.get("memory_ms", 5000)),
            prediction_ms=int(cfg.get("prediction_ms", 1000)),
            match_iou=float(cfg.get("match_iou", 0.12)),
            reacquire_distance=float(cfg.get("reacquire_distance", 1.05)),
            duplicate_iou=float(cfg.get("duplicate_iou", 0.68)),
            duplicate_containment=float(cfg.get("duplicate_containment", 0.90)),
            duplicate_center_distance=float(cfg.get("duplicate_center_distance", 0.20)),
            fragment_duplicate=camera_id in fragment_cameras,
            fragment_horizontal_overlap=float(cfg.get("fragment_horizontal_overlap", 0.78)),
            fragment_x_center=float(cfg.get("fragment_x_center", 0.18)),
            fragment_max_area_ratio=float(cfg.get("fragment_max_area_ratio", 0.55)),
            fragment_min_vertical_overlap=float(cfg.get("fragment_min_vertical_overlap", 0.20)),
            fragment_max_vertical_gap=float(cfg.get("fragment_max_vertical_gap", 0.06)),
            low_conf_confirm=float(camera_low.get(camera_id, cfg.get("low_conf_confirm", 0.06))),
            start_conf=float(camera_start.get(camera_id, cfg.get("start_conf", 0.24))),
            new_track_min_conf=float(cfg.get("new_track_min_conf", 0.14)),
            strong_confirm_hits=int(cfg.get("strong_confirm_hits", 2)),
            weak_confirm_hits=int(cfg.get("weak_confirm_hits", 3)),
            byte_high_conf=float(cfg.get("byte_high_conf", 0.16)),
            byte_low_conf=float(cfg.get("byte_low_conf", 0.06)),
            byte_second_match_iou=float(cfg.get("byte_second_match_iou", 0.04)),
            byte_match_center=float(cfg.get("byte_match_center", 0.70)),
            byte_second_match_center=float(cfg.get("byte_second_match_center", 0.50)),
            low_match_max_age_ms=int(cfg.get("low_match_max_age_ms", 1800)),
            process_noise=float(cfg.get("process_noise", 0.85)),
            measurement_noise=float(cfg.get("measurement_noise", 0.90)),
            velocity_damping=float(cfg.get("velocity_damping", 0.96)),
            size_velocity_damping=float(cfg.get("size_velocity_damping", 0.60)),
            max_prediction_shift_boxes=float(cfg.get("max_prediction_shift_boxes", 0.70)),
            max_prediction_size_ratio=float(cfg.get("max_prediction_size_ratio", 0.10)),
            adaptive_error_low=float(cfg.get("adaptive_error_low", 0.08)),
            adaptive_error_high=float(cfg.get("adaptive_error_high", 0.25)),
            center_response_slow=float(cfg.get("center_response_slow", 0.42)),
            center_response_fast=float(cfg.get("center_response_fast", 0.88)),
            size_response=float(cfg.get("size_response", 0.30)),
            snap_distance_boxes=float(cfg.get("snap_distance_boxes", 0.65)),
            reversal_damping=float(cfg.get("reversal_damping", 0.15)),
            new_track_zones=camera_birth_zones.get(camera_id, []),
            exclusion_zones=camera_exclusion.get(camera_id, []),
            exclusion_max_box_height=float(cfg.get("exclusion_max_box_height", 0.30)),
            exclusion_overlap_threshold=float(cfg.get("exclusion_overlap_threshold", 0.15)),
        )
        self.trackers[camera_id] = tracker
        return tracker

    @staticmethod
    def _as_visual_boxes(detections):
        boxes = []
        for box, confidence in detections:
            x1, y1, x2, y2 = [float(v) for v in box]
            confidence = float(confidence)
            if x2 <= x1 or y2 <= y1 or confidence <= 0.0:
                continue
            boxes.append(VisualBox(x1, y1, x2, y2, confidence))
        return boxes

    def update(self, camera_id: str, captured_t: float, detections) -> None:
        frame_id = self.frame_ids.get(camera_id, 0) + 1
        self.frame_ids[camera_id] = frame_id
        result = SimpleNamespace(
            frame_id=frame_id,
            frame_captured_monotonic=float(captured_t),
            boxes=tuple(self._as_visual_boxes(detections)),
        )
        self._tracker(camera_id).update(
            result,
            now=time.monotonic(),
            source_width=self.width,
            source_height=self.height,
        )

    def _guard(self, box: VisualBox):
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
        width = max(2.0, x2 - x1)
        height = max(2.0, y2 - y1)
        aspect = height / max(1.0, width)

        side = float(self.cfg.get("display_side_margin", 0.08))
        top = float(self.cfg.get("display_top_margin", 0.07))
        bottom = float(self.cfg.get("display_bottom_margin", 0.10))
        if aspect < float(self.cfg.get("sitting_aspect_threshold", 1.55)):
            side += float(self.cfg.get("sitting_extra_side", 0.04))
            bottom += float(self.cfg.get("sitting_extra_bottom", 0.04))

        pad_side = max(5.0, width * side)
        pad_top = max(5.0, height * top)
        pad_bottom = max(7.0, height * bottom)
        return (
            max(0.0, x1 - pad_side),
            max(0.0, y1 - pad_top),
            min(self.width - 1.0, x2 + pad_side),
            min(self.height - 1.0, y2 + pad_bottom),
            float(box.confidence),
        )

    def render(self, camera_id: str, now: float):
        tracker = self.trackers.get(camera_id)
        if tracker is None:
            return []
        boxes = tracker.visible(now=float(now), target_time=float(now))
        return [self._guard(box) for box in boxes]

    def metrics(self, camera_id: str):
        tracker = self.trackers.get(camera_id)
        return tracker.metrics() if tracker is not None else {"algorithm": "adaptive-kalman-byte-visual-v2", "active_tracks": 0}
