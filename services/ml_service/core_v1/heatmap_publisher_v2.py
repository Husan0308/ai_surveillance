from __future__ import annotations

import math
import threading

import cv2

from .heatmap_publisher import HeatmapJpegPublisher as _BaseHeatmapPublisher
from .jpeg_publisher import LatestJpegPublisher


_COCO_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
)


def _finite_bbox(box):
    try:
        values = [float(box.x1), float(box.y1), float(box.x2), float(box.y2)]
    except (AttributeError, TypeError, ValueError):
        try:
            values = [float(value) for value in box[:4]]
        except Exception:
            return None
    if not all(math.isfinite(value) for value in values):
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_distance(a, b):
    left = _finite_bbox(a)
    right = _finite_bbox(b)
    if left is None or right is None:
        return float("inf")
    ax1, ay1, ax2, ay2 = left
    bx1, by1, bx2, by2 = right
    acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
    bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
    scale = max(20.0, ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1)
    return math.hypot(acx - bcx, acy - bcy) / scale


class HeatmapJpegPublisher(_BaseHeatmapPublisher):
    """Presentation v2: persistent pose + camera heat + detection labels.

    Pose inference stays sparse/isolated. Each fresh pose is cached in normalized
    person-bbox coordinates. Between pose results the keypoints are reprojected
    into the current visual-tracker bbox, so the skeleton follows the person and
    does not blink out merely because the CPU pose worker has not produced a new
    frame yet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pose_cache_lock = threading.RLock()
        self._pose_cache_frame = -1
        self._pose_cache_capture = 0.0
        self._pose_cache = []
        self.pose_cache_max_age_sec = max(4.0, self.pose_overlay_max_age_sec, 12.0)
        self.pose_direct_hold_sec = min(2.0, self.pose_cache_max_age_sec)
        self._pose_cache_updates = 0
        self._pose_reprojected_frames = 0
        self._pose_direct_frames = 0
        self._pose_cache_expired = 0

    def metrics(self):
        payload = super().metrics()
        with self._pose_cache_lock:
            payload["pose_overlay"] = {
                "cache_frame": int(self._pose_cache_frame),
                "cached_people": len(self._pose_cache),
                "cache_updates": int(self._pose_cache_updates),
                "reprojected_frames": int(self._pose_reprojected_frames),
                "direct_frames": int(self._pose_direct_frames),
                "cache_expired": int(self._pose_cache_expired),
                "max_cache_age_ms": self.pose_cache_max_age_sec * 1000.0,
            }
        return payload

    def _refresh_pose_cache(self):
        provider = self.pose_provider
        if provider is None:
            return
        try:
            result = provider.snapshot().get(self.camera_id)
        except Exception:
            return
        if result is None:
            return
        try:
            frame_id = int(result.frame_id)
        except Exception:
            return
        with self._pose_cache_lock:
            if frame_id <= self._pose_cache_frame:
                return

        people = []
        for person in tuple(result.people or ()):
            bbox = _finite_bbox(getattr(person, "bbox", None))
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            relative = []
            for point in tuple(getattr(person, "keypoints", ()) or ()):
                try:
                    x, y, confidence = (
                        float(point.x), float(point.y), float(point.confidence)
                    )
                except Exception:
                    relative.append(None)
                    continue
                if not all(math.isfinite(v) for v in (x, y, confidence)):
                    relative.append(None)
                    continue
                relative.append(((x - x1) / bw, (y - y1) / bh, confidence))
            people.append({"bbox": bbox, "points": relative})

        with self._pose_cache_lock:
            self._pose_cache_frame = frame_id
            self._pose_cache_capture = float(
                getattr(result, "frame_captured_monotonic", 0.0) or 0.0
            )
            self._pose_cache = people
            self._pose_cache_updates += 1

    def _draw_people(self, image, source_width, source_height, display_frame_time, now):
        self._refresh_pose_cache()
        with self._pose_cache_lock:
            capture = float(self._pose_cache_capture)
            cached = [
                {"bbox": tuple(item["bbox"]), "points": list(item["points"])}
                for item in self._pose_cache
            ]
        if not cached or capture <= 0.0:
            return image

        age = max(0.0, float(display_frame_time) - capture)
        if age > self.pose_cache_max_age_sec:
            with self._pose_cache_lock:
                self._pose_cache_expired += 1
            return image

        try:
            visible_boxes = self.visual_tracker.visible(
                now,
                target_time=display_frame_time,
                max_observation_age_sec=(self.overlay_max_age_ms / 1000.0)
                if self.overlay_max_age_ms > 0
                else None,
            )
        except Exception:
            visible_boxes = []

        unused = set(range(len(visible_boxes)))
        assignments = []
        updated_refs = []
        for person in cached:
            best_index = None
            best_distance = float("inf")
            for index in unused:
                distance = _bbox_distance(person["bbox"], visible_boxes[index])
                if distance < best_distance:
                    best_index, best_distance = index, distance
            if best_index is not None and best_distance <= 1.8:
                unused.discard(best_index)
                target = _finite_bbox(visible_boxes[best_index])
                assignments.append((person, target, True))
                updated_refs.append(target)
            elif age <= self.pose_direct_hold_sec:
                assignments.append((person, person["bbox"], False))
                updated_refs.append(person["bbox"])
            else:
                updated_refs.append(person["bbox"])

        h, w = image.shape[:2]
        try:
            sx = w / max(1.0, float(source_width))
            sy = h / max(1.0, float(source_height))
        except Exception:
            return image

        for person, target_bbox, reprojected in assignments:
            if target_bbox is None:
                continue
            x1, y1, x2, y2 = target_bbox
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            points = []
            for relative in person["points"]:
                if relative is None:
                    points.append(None)
                    continue
                rx, ry, confidence = relative
                if confidence < self.pose_overlay_conf:
                    points.append(None)
                    continue
                px = int(round((x1 + rx * bw) * sx))
                py = int(round((y1 + ry * bh) * sy))
                points.append((px, py, confidence))

            for left, right in _COCO_EDGES:
                if (
                    left >= len(points)
                    or right >= len(points)
                    or points[left] is None
                    or points[right] is None
                ):
                    continue
                cv2.line(
                    image,
                    points[left][:2],
                    points[right][:2],
                    (0, 225, 255),
                    2,
                    cv2.LINE_AA,
                )
            for index, point in enumerate(points):
                if point is None:
                    continue
                radius = 4 if index in (15, 16) else 3
                color = (0, 70, 255) if index in (15, 16) else (55, 245, 255)
                cv2.circle(image, point[:2], radius, color, -1, cv2.LINE_AA)

            with self._pose_cache_lock:
                if reprojected:
                    self._pose_reprojected_frames += 1
                else:
                    self._pose_direct_frames += 1

        # Move the cache reference box with the visual track. This makes the next
        # sparse-pose-to-current-box match much more stable for walking people.
        if assignments:
            with self._pose_cache_lock:
                for index, reference in enumerate(updated_refs):
                    if index < len(self._pose_cache) and reference is not None:
                        self._pose_cache[index]["bbox"] = tuple(reference)
        return image

    def _draw_detection(
        self,
        image,
        source_width,
        source_height,
        now,
        display_frame_id,
        display_frame_time,
    ):
        with self._overlay_lock:
            heatmap_visible = self._heatmap_visible
            pose_visible = self._pose_visible

        if heatmap_visible and self.heatmap_provider is not None:
            try:
                image = self.heatmap_provider.overlay(self.camera_id, image)
            except Exception:
                pass

        # Let the authoritative detector + visual tracker update first. This is
        # unchanged from the stable publisher and keeps detection boxes reliable.
        image = LatestJpegPublisher._draw_detection(
            self,
            image,
            source_width,
            source_height,
            now,
            display_frame_id,
            display_frame_time,
        )

        if pose_visible:
            try:
                image = self._draw_people(
                    image, source_width, source_height, display_frame_time, now
                )
            except Exception:
                pass
        return image
