from __future__ import annotations

from .event_publisher import EventDrivenJpegPublisher
from .ownership_tracker import OwnershipLockedTracker


def _build_tracker(camera_id: str, config: dict) -> OwnershipLockedTracker:
    cfg = dict(config or {})
    camera_zones = dict(cfg.get("camera_exclusion_zones") or {})
    camera_birth_zones = dict(cfg.get("camera_new_track_zones") or {})
    fragment_cameras = set(str(cid) for cid in (cfg.get("fragment_duplicate_cameras") or []))
    camera_start_conf = dict(cfg.get("camera_start_conf") or {})
    camera_low_conf = dict(cfg.get("camera_low_conf_confirm") or {})

    return OwnershipLockedTracker(
        camera_id=camera_id,
        ownership_lock=cfg.get("ownership_lock", True),
        ownership_min_hits=cfg.get("ownership_min_hits", 3),
        ownership_margin=cfg.get("ownership_margin", 0.09),
        ownership_low_margin_multiplier=cfg.get("ownership_low_margin_multiplier", 1.5),
        ownership_proximity=cfg.get("ownership_proximity", 0.70),
        ownership_overlap_iou=cfg.get("ownership_overlap_iou", 0.08),
        ownership_iou_advantage=cfg.get("ownership_iou_advantage", 0.14),
        ownership_distance_advantage=cfg.get("ownership_distance_advantage", 0.14),
        ownership_max_competitor_age_ms=cfg.get("ownership_max_competitor_age_ms", 700),
        id_namespace_stride=cfg.get("id_namespace_stride", 10000),
        hold_ms=cfg.get("hold_ms", 800),
        memory_ms=cfg.get("memory_ms", 3000),
        prediction_ms=cfg.get("prediction_ms", 420),
        match_iou=cfg.get("match_iou", 0.12),
        reacquire_distance=cfg.get("reacquire_distance", 0.85),
        duplicate_iou=cfg.get("duplicate_iou", 0.68),
        duplicate_containment=cfg.get("duplicate_containment", 0.90),
        duplicate_center_distance=cfg.get("duplicate_center_distance", 0.20),
        fragment_duplicate=(camera_id in fragment_cameras),
        fragment_horizontal_overlap=cfg.get("fragment_horizontal_overlap", 0.78),
        fragment_x_center=cfg.get("fragment_x_center", 0.18),
        fragment_max_area_ratio=cfg.get("fragment_max_area_ratio", 0.55),
        fragment_min_vertical_overlap=cfg.get("fragment_min_vertical_overlap", 0.20),
        fragment_max_vertical_gap=cfg.get("fragment_max_vertical_gap", 0.06),
        low_conf_confirm=camera_low_conf.get(camera_id, cfg.get("low_conf_confirm", 0.10)),
        start_conf=camera_start_conf.get(camera_id, cfg.get("start_conf", 0.25)),
        new_track_min_conf=cfg.get("new_track_min_conf", 0.25),
        strong_confirm_hits=cfg.get("strong_confirm_hits", 2),
        weak_confirm_hits=cfg.get("weak_confirm_hits", 3),
        byte_high_conf=cfg.get("byte_high_conf", 0.25),
        byte_low_conf=cfg.get("byte_low_conf", 0.10),
        byte_second_match_iou=cfg.get("byte_second_match_iou", 0.04),
        byte_match_center=cfg.get("byte_match_center", 0.70),
        byte_second_match_center=cfg.get("byte_second_match_center", 0.50),
        low_match_max_age_ms=cfg.get("low_match_max_age_ms", 650),
        process_noise=cfg.get("process_noise", 0.85),
        measurement_noise=cfg.get("measurement_noise", 0.90),
        velocity_damping=cfg.get("velocity_damping", 0.96),
        size_velocity_damping=cfg.get("size_velocity_damping", 0.60),
        max_prediction_shift_boxes=cfg.get("max_prediction_shift_boxes", 0.68),
        max_prediction_size_ratio=cfg.get("max_prediction_size_ratio", 0.08),
        adaptive_error_low=cfg.get("adaptive_error_low", 0.08),
        adaptive_error_high=cfg.get("adaptive_error_high", 0.25),
        center_response_slow=cfg.get("center_response_slow", 0.42),
        center_response_fast=cfg.get("center_response_fast", 0.88),
        size_response=cfg.get("size_response", 0.30),
        snap_distance_boxes=cfg.get("snap_distance_boxes", 0.65),
        reversal_damping=cfg.get("reversal_damping", 0.15),
        new_track_zones=camera_birth_zones.get(camera_id, []),
        exclusion_zones=camera_zones.get(camera_id, []),
        exclusion_max_box_height=cfg.get("exclusion_max_box_height", 0.24),
        exclusion_overlap_threshold=cfg.get("exclusion_overlap_threshold", 0.35),
        fuse_score=cfg.get("fuse_score", True),
    )


class TrackingJpegPublisher(EventDrivenJpegPublisher):
    """Event-driven latest-frame publisher with ownership-locked camera IDs."""

    def __init__(self, *args, tracker_config=None, **kwargs):
        super().__init__(*args, tracker_config=tracker_config, **kwargs)
        self.visual_tracker = _build_tracker(self.camera_id, dict(tracker_config or {}))

    def _identity_for_box(self, box):
        track_id = int(getattr(box, "track_id", 0) or 0)
        if track_id > 0:
            return {"global_id": self.visual_tracker.display_label(track_id)}
        return super()._identity_for_box(box)

    def track_snapshot(self):
        return self.visual_tracker.snapshot()
