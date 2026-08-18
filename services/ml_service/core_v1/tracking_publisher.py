from __future__ import annotations

import math

import cv2

from .camera_heatmap import FootpointHeatmap
from .event_publisher import EventDrivenJpegPublisher
from .jpeg_publisher import _identity_label
from .ownership_tracker import OwnershipLockedTracker

_BOX_COLOR = (255, 190, 35)
_LABEL_BG = (7, 12, 20)
_LABEL_TEXT = (245, 248, 252)


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


def _draw_corner_box(image, x1: int, y1: int, x2: int, y2: int, label: str) -> None:
    h, w = image.shape[:2]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    corner = max(11, min(30, int(round(min(box_w, box_h) * 0.22))))
    thickness = 2 if max(w, h) < 1200 else 3

    # Clean CCTV-style corner box. No foot ring is drawn here; the heatmap
    # itself represents floor contact, so the overlay stays uncluttered.
    segments = (
        ((x1, y1), (x1 + corner, y1)), ((x1, y1), (x1, y1 + corner)),
        ((x2, y1), (x2 - corner, y1)), ((x2, y1), (x2, y1 + corner)),
        ((x1, y2), (x1 + corner, y2)), ((x1, y2), (x1, y2 - corner)),
        ((x2, y2), (x2 - corner, y2)), ((x2, y2), (x2, y2 - corner)),
    )
    for start, end in segments:
        cv2.line(image, start, end, _BOX_COLOR, thickness, cv2.LINE_AA)

    scale = 0.46 if w < 1200 else 0.52
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, text_thickness)
    pad_x, pad_y = 6, 3
    pill_h = th + baseline + pad_y * 2
    pill_w = min(max(1, w - x1), tw + pad_x * 2)
    top = max(0, y1 - pill_h - 3)
    bottom = min(h - 1, top + pill_h)
    right = min(w - 1, x1 + pill_w)
    cv2.rectangle(image, (x1, top), (right, bottom), _LABEL_BG, -1, cv2.LINE_AA)
    cv2.line(image, (x1, bottom), (right, bottom), _BOX_COLOR, 2, cv2.LINE_AA)
    text_y = max(top + th + pad_y, bottom - baseline - pad_y)
    cv2.putText(
        image,
        label,
        (x1 + pad_x, text_y),
        font,
        scale,
        _LABEL_TEXT,
        text_thickness,
        cv2.LINE_AA,
    )


class TrackingJpegPublisher(EventDrivenJpegPublisher):
    """Latest-frame publisher with time-aligned tracking and footpoint heatmap."""

    def __init__(self, *args, tracker_config=None, heatmap_config=None, **kwargs):
        super().__init__(*args, tracker_config=tracker_config, **kwargs)
        self.visual_tracker = _build_tracker(self.camera_id, dict(tracker_config or {}))
        self.foot_heatmap = FootpointHeatmap(heatmap_config)
        self._heatmap_recorded_key = None

    def _identity_for_box(self, box):
        track_id = int(getattr(box, "track_id", 0) or 0)
        if track_id > 0:
            return {"global_id": self.visual_tracker.display_label(track_id)}
        return super()._identity_for_box(box)

    def _draw_detection(
        self,
        image,
        source_width,
        source_height,
        now,
        display_frame_id,
        display_frame_time,
    ):
        max_age_sec = (
            self.overlay_max_age_ms / 1000.0
            if self.overlay_max_age_ms > 0
            else None
        )

        if self.detections is not None:
            result = self.detections.get(self.camera_id)
            if result is not None:
                key = self._detection_key(result)
                with self._lock:
                    if key != self._current_detection_key:
                        self._current_detection_key = key
                        self._current_detection_accepted = False
                        self._current_future_counted = False
                        self._current_stale_counted = False

                result_time = float(result.frame_captured_monotonic)
                is_future = (
                    int(result.frame_id) > int(display_frame_id)
                    or result_time > float(display_frame_time)
                )

                if is_future:
                    with self._lock:
                        if not self._current_future_counted:
                            self.future_detection_deferrals += 1
                            self._current_future_counted = True
                else:
                    source_age = max(
                        0.0,
                        float(display_frame_time) - result_time,
                    )
                    if max_age_sec is None or source_age <= max_age_sec:
                        self.visual_tracker.update(
                            result,
                            now,
                            source_width,
                            source_height,
                        )
                        with self._lock:
                            self._current_detection_accepted = True

                        if key != self._heatmap_recorded_key:
                            # Ask the tracker for boxes exactly at this detector
                            # observation time and accept only tracks whose last
                            # real observation is this frame (age <= 2 ms). This
                            # gives heatmap paths stable track IDs without ever
                            # painting presentation prediction/hold boxes.
                            observed_tracks = self.visual_tracker.visible(
                                now,
                                target_time=result_time,
                                max_observation_age_sec=0.002,
                            )
                            self.foot_heatmap.observe_tracks(
                                observed_tracks,
                                int(source_width),
                                int(source_height),
                                observation_time=result_time,
                            )
                            self._heatmap_recorded_key = key
                    else:
                        with self._lock:
                            if (
                                not self._current_detection_accepted
                                and not self._current_stale_counted
                            ):
                                self.stale_detection_rejects += 1
                                self._current_stale_counted = True

        # Professional overlay order: video -> heatmap -> tracking -> labels.
        image = self.foot_heatmap.overlay(image, now)

        boxes = self.visual_tracker.visible(
            now,
            target_time=display_frame_time,
            max_observation_age_sec=max_age_sec,
        )
        if not boxes:
            return image

        h, w = image.shape[:2]
        try:
            source_w = float(source_width)
            source_h = float(source_height)
        except (TypeError, ValueError, OverflowError):
            return image
        if (
            not math.isfinite(source_w)
            or not math.isfinite(source_h)
            or source_w <= 0.0
            or source_h <= 0.0
        ):
            return image

        sx = w / source_w
        sy = h / source_h
        for box in boxes:
            try:
                values = [
                    float(box.x1),
                    float(box.y1),
                    float(box.x2),
                    float(box.y2),
                    float(box.confidence),
                ]
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(v) for v in values):
                continue

            bx1, by1, bx2, by2, confidence = values
            bx1 = max(0.0, min(source_w, bx1))
            bx2 = max(0.0, min(source_w, bx2))
            by1 = max(0.0, min(source_h, by1))
            by2 = max(0.0, min(source_h, by2))
            if bx2 <= bx1 or by2 <= by1:
                continue

            x1 = max(0, min(w - 1, int(round(bx1 * sx))))
            y1 = max(0, min(h - 1, int(round(by1 * sy))))
            x2 = max(0, min(w - 1, int(round(bx2 * sx))))
            y2 = max(0, min(h - 1, int(round(by2 * sy))))
            if x2 <= x1 or y2 <= y1:
                continue

            identity = self._identity_for_box(box)
            label, _known = _identity_label(identity, confidence)
            _draw_corner_box(image, x1, y1, x2, y2, label)

        return image

    def track_snapshot(self):
        return self.visual_tracker.snapshot()

    def metrics(self):
        payload = super().metrics()
        payload["heatmap"] = self.foot_heatmap.metrics()
        return payload
