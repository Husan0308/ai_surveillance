from __future__ import annotations

from collections import deque
import math
import threading
import time

import cv2
import numpy as np


class CameraAnkleHeatmapCoordinator:
    """Lightweight per-camera occupancy heatmap.

    Priority: both ankles -> one ankle -> detector bbox bottom-center fallback.
    Accumulation always runs while the feature is enabled; display visibility is
    controlled independently by the JPEG publisher/UI.
    """

    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def __init__(self, pose, frame_stores, config: dict | None = None, detections=None):
        self.pose = pose
        self.detections = detections
        self.frame_stores = dict(frame_stores)
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.grid_width = max(32, int(self.config.get("camera_grid_width", 160)))
        self.grid_height = max(18, int(self.config.get("camera_grid_height", 90)))
        self.poll_sec = max(0.03, float(self.config.get("poll_interval_ms", 120)) / 1000.0)
        self.ankle_conf = max(0.0, min(1.0, float(self.config.get("ankle_conf", 0.30))))
        self.sigma_cells = max(0.5, float(self.config.get("sigma_cells", 3.0)))
        self.pose_weight = max(0.01, float(self.config.get("pose_weight", 1.0)))
        self.bbox_weight = max(0.01, float(self.config.get("bbox_fallback_weight", 0.38)))
        self.fallback_every_n = max(1, int(self.config.get("bbox_fallback_every_n", 4)))
        self.max_fallback_people = max(1, int(self.config.get("max_fallback_people", 6)))
        self.half_life_sec = max(1.0, float(self.config.get("camera_half_life_sec", 3600.0)))
        self.overlay_alpha = max(0.0, min(1.0, float(self.config.get("overlay_alpha", 0.30))))
        self.overlay_threshold = max(0.0, min(1.0, float(self.config.get("overlay_threshold", 0.025))))
        self.dedupe_sec = max(0.0, float(self.config.get("dedupe_window_ms", 450)) / 1000.0)
        self.dedupe_distance = max(0.0, float(self.config.get("dedupe_distance_norm", 0.025)))
        self.smoothing_alpha = max(0.05, min(1.0, float(self.config.get("smoothing_alpha", 0.55))))

        self._grids = {cid: np.zeros((self.grid_height, self.grid_width), dtype=np.float32) for cid in self.frame_stores}
        now = time.monotonic()
        self._last_decay = {cid: now for cid in self.frame_stores}
        self._last_pose_frame = {cid: -1 for cid in self.frame_stores}
        self._last_detection_frame = {cid: -1 for cid in self.frame_stores}
        self._fallback_seen = {cid: 0 for cid in self.frame_stores}
        self._recent = {cid: deque(maxlen=32) for cid in self.frame_stores}
        self._anchors = {cid: [] for cid in self.frame_stores}
        self._samples = {cid: 0 for cid in self.frame_stores}
        self._pose_samples = {cid: 0 for cid in self.frame_stores}
        self._bbox_samples = {cid: 0 for cid in self.frame_stores}
        self._ankle_skips = {cid: 0 for cid in self.frame_stores}
        self._dedupe_skips = {cid: 0 for cid in self.frame_stores}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._errors = 0
        self._last_error = ""

    def start(self):
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera-ankle-heatmap", daemon=False)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=4):
        if self._thread:
            self._thread.join(timeout)

    def _decay_locked(self, camera_id: str, now: float):
        last = float(self._last_decay.get(camera_id, now))
        dt = max(0.0, now - last)
        self._last_decay[camera_id] = now
        if dt > 0.0:
            self._grids[camera_id] *= float(0.5 ** (dt / self.half_life_sec))

    @staticmethod
    def _valid_keypoint(person, index: int, minimum: float):
        points = tuple(getattr(person, "keypoints", ()) or ())
        if index >= len(points):
            return None
        point = points[index]
        try:
            x, y, confidence = float(point.x), float(point.y), float(point.confidence)
        except (AttributeError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x, y, confidence)) or confidence < minimum:
            return None
        return x, y

    def _contact_from_pose(self, person):
        valid = [self._valid_keypoint(person, i, self.ankle_conf) for i in (self.LEFT_ANKLE, self.RIGHT_ANKLE)]
        valid = [point for point in valid if point is not None]
        if len(valid) == 2:
            return ((valid[0][0] + valid[1][0]) * 0.5, (valid[0][1] + valid[1][1]) * 0.5, self.pose_weight, "ankles")
        if len(valid) == 1:
            return (valid[0][0], valid[0][1], self.pose_weight * 0.85, "ankle")
        try:
            x1, _y1, x2, y2 = [float(v) for v in person.bbox]
        except (AttributeError, TypeError, ValueError):
            return None
        if all(math.isfinite(v) for v in (x1, x2, y2)):
            return ((x1 + x2) * 0.5, y2, self.bbox_weight, "pose_bbox")
        return None

    def _source_size(self, camera_id: str):
        store = self.frame_stores.get(camera_id)
        try:
            frame, _ = store.get() if store is not None else (None, None)
        except Exception:
            return None
        image = getattr(frame, "image", None) if frame is not None else None
        if image is None or getattr(image, "ndim", 0) < 2:
            return None
        h, w = image.shape[:2]
        return float(w), float(h)

    def _smooth_locked(self, camera_id, x, y, source_w, source_h, now):
        nx, ny = x / max(1.0, source_w), y / max(1.0, source_h)
        anchors = [a for a in self._anchors[camera_id] if now - a[2] <= 2.0]
        self._anchors[camera_id] = anchors
        best = None
        best_d = 1e9
        for index, (ax, ay, at) in enumerate(anchors):
            d = math.hypot(nx - ax, ny - ay)
            if d < best_d:
                best, best_d = index, d
        if best is not None and best_d <= 0.12:
            ax, ay, _ = anchors[best]
            alpha = self.smoothing_alpha
            nx, ny = alpha * nx + (1.0 - alpha) * ax, alpha * ny + (1.0 - alpha) * ay
            anchors[best] = (nx, ny, now)
        else:
            anchors.append((nx, ny, now))
            if len(anchors) > 12:
                del anchors[:-12]
        return nx * source_w, ny * source_h

    def _duplicate_locked(self, camera_id, x, y, source_w, source_h, now):
        recent = self._recent[camera_id]
        while recent and now - recent[0][2] > max(self.dedupe_sec, 0.01):
            recent.popleft()
        nx, ny = x / max(1.0, source_w), y / max(1.0, source_h)
        if self.dedupe_sec > 0 and any(math.hypot(nx-rx, ny-ry) <= self.dedupe_distance for rx, ry, _ in recent):
            self._dedupe_skips[camera_id] += 1
            return True
        recent.append((nx, ny, now))
        return False

    def _add_sample_locked(self, camera_id, x, y, source_w, source_h, weight, source):
        now = time.monotonic()
        x, y = self._smooth_locked(camera_id, x, y, source_w, source_h, now)
        if self._duplicate_locked(camera_id, x, y, source_w, source_h, now):
            return False
        gx = np.clip(x / max(1.0, source_w) * (self.grid_width - 1), 0.0, self.grid_width - 1.0)
        gy = np.clip(y / max(1.0, source_h) * (self.grid_height - 1), 0.0, self.grid_height - 1.0)
        radius = max(2, int(math.ceil(self.sigma_cells * 3.0)))
        x0, x1 = max(0, int(gx) - radius), min(self.grid_width, int(gx) + radius + 1)
        y0, y1 = max(0, int(gy) - radius), min(self.grid_height, int(gy) + radius + 1)
        xs = np.arange(x0, x1, dtype=np.float32) - np.float32(gx)
        ys = np.arange(y0, y1, dtype=np.float32) - np.float32(gy)
        kernel = np.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) / (2.0 * self.sigma_cells**2)).astype(np.float32)
        self._grids[camera_id][y0:y1, x0:x1] += kernel * np.float32(weight)
        self._samples[camera_id] += 1
        if source.startswith("ankle"):
            self._pose_samples[camera_id] += 1
        else:
            self._bbox_samples[camera_id] += 1
        return True

    @staticmethod
    def _bbox_contact(box):
        try:
            return ((float(box.x1) + float(box.x2)) * 0.5, float(box.y2))
        except (AttributeError, TypeError, ValueError):
            return None

    def _run(self):
        while not self._stop.is_set():
            did_work = False
            try:
                pose_snapshot = self.pose.snapshot() if self.pose is not None else {}
                det_snapshot = self.detections.snapshot() if self.detections is not None else {}
                camera_ids = tuple(self._grids)
                for camera_id in camera_ids:
                    size = self._source_size(camera_id)
                    if not size:
                        continue
                    source_w, source_h = size
                    now = time.monotonic()
                    with self._lock:
                        self._decay_locked(camera_id, now)

                    pose_result = pose_snapshot.get(camera_id)
                    used_pose = False
                    if pose_result is not None and int(pose_result.frame_id) > self._last_pose_frame[camera_id]:
                        self._last_pose_frame[camera_id] = int(pose_result.frame_id)
                        with self._lock:
                            for person in pose_result.people:
                                contact = self._contact_from_pose(person)
                                if contact is None:
                                    self._ankle_skips[camera_id] += 1
                                    continue
                                x, y, weight, source = contact
                                used_pose |= self._add_sample_locked(camera_id, x, y, source_w, source_h, weight, source)
                        did_work |= used_pose

                    detection = det_snapshot.get(camera_id)
                    if detection is not None and int(detection.frame_id) > self._last_detection_frame[camera_id]:
                        self._last_detection_frame[camera_id] = int(detection.frame_id)
                        self._fallback_seen[camera_id] += 1
                        if not used_pose and self._fallback_seen[camera_id] % self.fallback_every_n == 0:
                            with self._lock:
                                for box in sorted(detection.boxes, key=lambda item: float(item.confidence), reverse=True)[:self.max_fallback_people]:
                                    point = self._bbox_contact(box)
                                    if point is None:
                                        continue
                                    did_work |= self._add_sample_locked(camera_id, point[0], point[1], source_w, source_h, self.bbox_weight, "detector_bbox")
                    with self._lock:
                        self._last_error = ""
            except Exception as exc:
                with self._lock:
                    self._errors += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
            if not did_work:
                self._stop.wait(self.poll_sec)

    def overlay(self, camera_id: str, image):
        if not self.enabled or image is None or camera_id not in self._grids:
            return image
        with self._lock:
            self._decay_locked(camera_id, time.monotonic())
            grid = self._grids[camera_id].copy()
        peak = float(grid.max()) if grid.size else 0.0
        if peak <= 1e-8:
            return image
        normalized = np.clip(grid / peak, 0.0, 1.0)
        mask = normalized >= self.overlay_threshold
        if not np.any(mask):
            return image
        heat_u8 = np.uint8(np.round(normalized * 255.0))
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
        heat_color = cv2.resize(heat_color, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        intensity = cv2.resize(np.sqrt(normalized).astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        visible = cv2.resize(mask.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(visible * intensity * self.overlay_alpha, 0.0, self.overlay_alpha)[..., None]
        blended = image.astype(np.float32) * (1.0 - alpha) + heat_color.astype(np.float32) * alpha
        return np.uint8(np.clip(blended, 0, 255))

    def reset(self, camera_id: str | None = None):
        with self._lock:
            targets = [str(camera_id)] if camera_id is not None else list(self._grids)
            for cid in targets:
                if cid not in self._grids:
                    raise ValueError(f"camera not found: {cid}")
                self._grids[cid].fill(0.0)
                self._samples[cid] = self._pose_samples[cid] = self._bbox_samples[cid] = 0
                self._ankle_skips[cid] = self._dedupe_skips[cid] = 0
                self._recent[cid].clear()
                self._anchors[cid] = []
                self._last_decay[cid] = time.monotonic()

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            cameras = {}
            for cid, grid in self._grids.items():
                self._decay_locked(cid, now)
                cameras[cid] = {
                    "samples": int(self._samples[cid]),
                    "pose_samples": int(self._pose_samples[cid]),
                    "bbox_fallback_samples": int(self._bbox_samples[cid]),
                    "ankle_skips": int(self._ankle_skips[cid]),
                    "dedupe_skips": int(self._dedupe_skips[cid]),
                    "peak": float(grid.max()) if grid.size else 0.0,
                    "last_pose_frame": int(self._last_pose_frame[cid]),
                    "last_detection_frame": int(self._last_detection_frame[cid]),
                }
            return {
                "enabled": self.enabled,
                "accumulating": self.enabled,
                "coordinate_system": "camera_pixels",
                "source": "pose ankles with detector bbox-bottom fallback",
                "cameras": cameras,
                "errors": self._errors,
                "last_error": self._last_error,
            }
