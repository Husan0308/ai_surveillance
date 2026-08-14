from __future__ import annotations

import math
import threading

import cv2

from .jpeg_publisher import LatestJpegPublisher


_COCO_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
)


class HeatmapJpegPublisher(LatestJpegPublisher):
    """JPEG publisher with cheap presentation-only Heatmap/Pose toggles.

    Toggling visibility never stops pose inference or heat accumulation.
    """

    def __init__(
        self,
        *args,
        heatmap_provider=None,
        pose_provider=None,
        heatmap_visible=True,
        pose_visible=False,
        pose_overlay_conf=0.30,
        pose_overlay_max_age_ms=1600,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.heatmap_provider = heatmap_provider
        self.pose_provider = pose_provider
        self._overlay_lock = threading.Lock()
        self._heatmap_visible = bool(heatmap_visible)
        self._pose_visible = bool(pose_visible)
        self.pose_overlay_conf = max(0.0, min(1.0, float(pose_overlay_conf)))
        self.pose_overlay_max_age_sec = max(0.05, float(pose_overlay_max_age_ms) / 1000.0)

    def set_overlay_state(self, *, heatmap=None, pose=None):
        with self._overlay_lock:
            if heatmap is not None:
                self._heatmap_visible = bool(heatmap)
            if pose is not None:
                self._pose_visible = bool(pose)
            return {
                "heatmap_visible": self._heatmap_visible,
                "pose_visible": self._pose_visible,
            }

    def overlay_state(self):
        with self._overlay_lock:
            return {
                "heatmap_visible": self._heatmap_visible,
                "pose_visible": self._pose_visible,
            }

    def metrics(self):
        payload = super().metrics()
        payload["overlays"] = self.overlay_state()
        return payload

    def _draw_pose(self, image, source_width, source_height, display_frame_time):
        provider = self.pose_provider
        if provider is None:
            return image
        try:
            result = provider.snapshot().get(self.camera_id)
        except Exception:
            return image
        if result is None:
            return image
        try:
            age = max(0.0, float(display_frame_time) - float(result.frame_captured_monotonic))
        except Exception:
            return image
        if age > self.pose_overlay_max_age_sec:
            return image

        h, w = image.shape[:2]
        try:
            sx = w / max(1.0, float(source_width))
            sy = h / max(1.0, float(source_height))
        except Exception:
            return image

        for person in tuple(result.people or ()):
            points = []
            for point in tuple(person.keypoints or ()):
                try:
                    x, y, conf = float(point.x), float(point.y), float(point.confidence)
                except Exception:
                    points.append(None)
                    continue
                if not all(math.isfinite(v) for v in (x, y, conf)) or conf < self.pose_overlay_conf:
                    points.append(None)
                    continue
                points.append((int(round(x * sx)), int(round(y * sy)), conf))

            for a, b in _COCO_EDGES:
                if a >= len(points) or b >= len(points) or points[a] is None or points[b] is None:
                    continue
                cv2.line(image, points[a][:2], points[b][:2], (0, 230, 255), 1, cv2.LINE_AA)
            for index, point in enumerate(points):
                if point is None:
                    continue
                radius = 4 if index in (15, 16) else 2
                color = (0, 80, 255) if index in (15, 16) else (70, 240, 255)
                cv2.circle(image, point[:2], radius, color, -1, cv2.LINE_AA)
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

        if pose_visible:
            try:
                image = self._draw_pose(image, source_width, source_height, display_frame_time)
            except Exception:
                pass

        # Detection boxes/identity labels stay on top of optional presentation.
        return super()._draw_detection(
            image,
            source_width,
            source_height,
            now,
            display_frame_id,
            display_frame_time,
        )
