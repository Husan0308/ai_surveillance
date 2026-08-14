from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time

import cv2
import numpy as np


@dataclass(slots=True)
class _Bucket:
    started_at: float
    grid: np.ndarray
    samples: int = 0


class FloorHeatmapCoordinator:
    """Room-floor heatmap built from pose ankle contacts.

    Pose keypoints stay in camera pixels. A validated camera->floor homography
    projects the ankle midpoint into normalized room coordinates. Contributions
    remain fully hot for ``hot_hold_sec`` and only then begin exponential cooling.
    Same-room duplicate observations from overlapping cameras are suppressed by a
    short spatio-temporal gate so one real foot contact is not counted twice.
    """

    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def __init__(self, pose, frame_stores, spatial_mapper, config: dict | None = None):
        self.pose = pose
        self.frame_stores = dict(frame_stores)
        self.spatial_mapper = spatial_mapper
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))

        self.grid_width = max(24, int(self.config.get("grid_width", 96)))
        self.grid_height = max(16, int(self.config.get("grid_height", 64)))
        self.poll_sec = max(0.03, float(self.config.get("poll_interval_ms", 120)) / 1000.0)
        self.ankle_conf = max(0.0, min(1.0, float(self.config.get("ankle_conf", 0.35))))
        self.sample_weight = max(0.01, float(self.config.get("sample_weight", 1.0)))
        self.sigma_cells = max(0.5, float(self.config.get("sigma_cells", 2.2)))

        self.hot_hold_sec = max(1.0, float(self.config.get("hot_hold_sec", 3600)))
        self.cool_half_life_sec = max(1.0, float(self.config.get("cool_half_life_sec", 3600)))
        self.bucket_sec = max(30.0, float(self.config.get("bucket_sec", 300)))
        self.max_history_sec = max(
            self.hot_hold_sec + self.cool_half_life_sec,
            float(self.config.get("max_history_sec", 21600)),
        )
        self.min_bucket_weight = max(
            0.0, min(1.0, float(self.config.get("min_bucket_weight", 0.02)))
        )

        self.dedupe_window_sec = max(
            0.0, float(self.config.get("dedupe_window_ms", 800)) / 1000.0
        )
        self.dedupe_distance = max(0.0, float(self.config.get("dedupe_distance", 0.06)))
        self.fallback_bbox_bottom = bool(self.config.get("fallback_bbox_bottom", False))

        mapping = self.spatial_mapper.snapshot() if self.spatial_mapper is not None else {}
        self.room_ids = tuple(sorted(str(room_id) for room_id in (mapping.get("rooms") or {})))
        # Dict-by-bucket avoids ordering assumptions when two camera results around
        # a bucket boundary arrive slightly out of timestamp order.
        self._buckets: dict[str, dict[float, _Bucket]] = {
            room_id: {} for room_id in self.room_ids
        }
        self._recent_points = {room_id: deque() for room_id in self.room_ids}
        self._last_pose_frame: dict[str, int] = {}
        self._last_update_mono = {room_id: 0.0 for room_id in self.room_ids}

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._processed_pose_frames = 0
        self._samples = 0
        self._duplicate_skips = 0
        self._uncalibrated_skips = 0
        self._ankle_skips = 0
        self._projection_skips = 0
        self._errors = 0
        self._last_error = ""

    def start(self):
        if not self.enabled or self.pose is None or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="floor-heatmap",
            daemon=False,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=6):
        if self._thread:
            self._thread.join(timeout)

    def _bucket_weight(self, age_sec: float) -> float:
        age = max(0.0, float(age_sec))
        if age <= self.hot_hold_sec:
            return 1.0
        return 0.5 ** ((age - self.hot_hold_sec) / self.cool_half_life_sec)

    def _bucket_age(self, bucket: _Bucket, now: float) -> float:
        # Weight from the END of the bucket. With 5-minute aggregation this
        # guarantees no sample starts cooling before it has been hot for 1 hour;
        # the maximum extra hold is one bucket (5 minutes by default).
        return max(0.0, float(now) - (float(bucket.started_at) + self.bucket_sec))

    @staticmethod
    def _finite_point(point) -> tuple[float, float] | None:
        try:
            x = float(point.x)
            y = float(point.y)
            confidence = float(point.confidence)
        except (AttributeError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x, y, confidence)):
            return None
        return x, y

    def _ankle_point(self, person) -> tuple[float, float] | None:
        keypoints = tuple(getattr(person, "keypoints", ()) or ())
        valid = []
        for index in (self.LEFT_ANKLE, self.RIGHT_ANKLE):
            if index >= len(keypoints):
                continue
            point = keypoints[index]
            value = self._finite_point(point)
            if value is None or float(point.confidence) < self.ankle_conf:
                continue
            valid.append(value)
        if valid:
            return (
                sum(item[0] for item in valid) / len(valid),
                sum(item[1] for item in valid) / len(valid),
            )
        if not self.fallback_bbox_bottom:
            return None
        try:
            x1, _y1, x2, y2 = [float(v) for v in person.bbox]
        except (AttributeError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x1, x2, y2)):
            return None
        return (x1 + x2) * 0.5, y2

    def _source_size(self, camera_id: str):
        store = self.frame_stores.get(camera_id)
        if store is None:
            return None
        try:
            frame, _version = store.get()
        except Exception:
            return None
        if frame is None:
            return None
        try:
            return float(frame.width), float(frame.height)
        except (AttributeError, TypeError, ValueError):
            return None

    def _is_duplicate(self, room_id: str, observed_at: float, x: float, y: float) -> bool:
        recent = self._recent_points.setdefault(room_id, deque())
        while recent and observed_at - recent[0][0] > self.dedupe_window_sec:
            recent.popleft()
        for _timestamp, px, py in recent:
            if math.hypot(x - px, y - py) <= self.dedupe_distance:
                return True
        recent.append((observed_at, x, y))
        return False

    def _bucket_for(self, room_id: str, observed_at: float) -> _Bucket:
        bucket_start = math.floor(observed_at / self.bucket_sec) * self.bucket_sec
        buckets = self._buckets.setdefault(room_id, {})
        bucket = buckets.get(bucket_start)
        if bucket is None:
            bucket = _Bucket(
                bucket_start,
                np.zeros((self.grid_height, self.grid_width), dtype=np.float32),
            )
            buckets[bucket_start] = bucket
        return bucket

    def _add_gaussian(self, grid: np.ndarray, x: float, y: float, weight: float) -> None:
        cx = int(round(max(0.0, min(1.0, x)) * (self.grid_width - 1)))
        cy = int(round(max(0.0, min(1.0, y)) * (self.grid_height - 1)))
        radius = max(1, int(math.ceil(self.sigma_cells * 3.0)))
        x1 = max(0, cx - radius)
        x2 = min(self.grid_width, cx + radius + 1)
        y1 = max(0, cy - radius)
        y2 = min(self.grid_height, cy + radius + 1)
        yy, xx = np.ogrid[y1:y2, x1:x2]
        distance2 = (xx - cx) ** 2 + (yy - cy) ** 2
        kernel = np.exp(-distance2 / (2.0 * self.sigma_cells * self.sigma_cells))
        grid[y1:y2, x1:x2] += (float(weight) * kernel).astype(np.float32)

    def _prune_locked(self, now: float) -> None:
        for room_id, buckets in self._buckets.items():
            remove = []
            for bucket_start, bucket in buckets.items():
                age = self._bucket_age(bucket, now)
                if age > self.max_history_sec or self._bucket_weight(age) < self.min_bucket_weight:
                    remove.append(bucket_start)
            for bucket_start in remove:
                buckets.pop(bucket_start, None)
            recent = self._recent_points.setdefault(room_id, deque())
            while recent and now - recent[0][0] > self.dedupe_window_sec:
                recent.popleft()

    def _consume_pose_result(self, camera_id: str, result) -> None:
        observed_at = float(getattr(result, "frame_captured_monotonic", time.monotonic()))
        source_size = self._source_size(camera_id)
        people = tuple(getattr(result, "people", ()) or ())
        for person in people:
            point = self._ankle_point(person)
            if point is None:
                self._ankle_skips += 1
                continue
            projection = self.spatial_mapper.project_point(
                camera_id,
                point,
                source_size=source_size,
            ) if self.spatial_mapper is not None else None
            if projection is None:
                room_id = self.spatial_mapper.room_for_camera(camera_id) if self.spatial_mapper is not None else None
                if room_id:
                    self._uncalibrated_skips += 1
                else:
                    self._projection_skips += 1
                continue
            room_id = str(projection.get("room_id") or "")
            if room_id not in self._buckets:
                self._projection_skips += 1
                continue
            x = float(projection["x"])
            y = float(projection["y"])
            if self._is_duplicate(room_id, observed_at, x, y):
                self._duplicate_skips += 1
                continue
            bucket = self._bucket_for(room_id, observed_at)
            self._add_gaussian(bucket.grid, x, y, self.sample_weight)
            bucket.samples += 1
            self._samples += 1
            self._last_update_mono[room_id] = observed_at

    def _run(self):
        while not self._stop.is_set():
            did_work = False
            try:
                snapshot = self.pose.snapshot() if self.pose is not None else {}
                with self._lock:
                    for camera_id, result in snapshot.items():
                        frame_id = int(getattr(result, "frame_id", -1))
                        if frame_id <= int(self._last_pose_frame.get(camera_id, -1)):
                            continue
                        self._last_pose_frame[camera_id] = frame_id
                        self._consume_pose_result(str(camera_id), result)
                        self._processed_pose_frames += 1
                        did_work = True
                    self._prune_locked(time.monotonic())
                    self._last_error = ""
            except Exception as exc:
                with self._lock:
                    self._errors += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
            if not did_work:
                self._stop.wait(self.poll_sec)

    def room_grid(self, room_id: str, now: float | None = None) -> np.ndarray:
        room_id = str(room_id)
        current = float(time.monotonic() if now is None else now)
        # Copy under the lock so PNG/JSON readers never race a Gaussian write.
        with self._lock:
            buckets = [
                (bucket.started_at, bucket.grid.copy())
                for bucket in self._buckets.get(room_id, {}).values()
            ]
        total = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        for started_at, grid in buckets:
            effective_age = max(0.0, current - (started_at + self.bucket_sec))
            weight = self._bucket_weight(effective_age)
            if weight < self.min_bucket_weight:
                continue
            total += grid * np.float32(weight)
        return total

    def normalized_grid(self, room_id: str, now: float | None = None) -> np.ndarray:
        grid = self.room_grid(room_id, now=now)
        peak = float(grid.max()) if grid.size else 0.0
        if peak <= 1e-12:
            return np.zeros_like(grid)
        # Log compression keeps both paths and long-stay hotspots visible.
        return np.log1p(grid) / np.float32(math.log1p(peak))

    def render_png(self, room_id: str, now: float | None = None) -> bytes | None:
        room_id = str(room_id)
        with self._lock:
            exists = room_id in self._buckets
        if not exists:
            return None
        normalized = self.normalized_grid(room_id, now=now)
        level = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
        bgr = cv2.applyColorMap(level, cv2.COLORMAP_TURBO)
        alpha = np.clip(np.power(normalized, 0.72) * 225.0, 0.0, 225.0).astype(np.uint8)
        alpha[level < 5] = 0
        bgra = np.dstack((bgr, alpha))
        ok, encoded = cv2.imencode(".png", bgra)
        return encoded.tobytes() if ok else None

    def reset(self, room_id: str | None = None) -> None:
        with self._lock:
            targets = self.room_ids if room_id is None else (str(room_id),)
            for target in targets:
                if target not in self._buckets:
                    raise ValueError(f"unknown room: {target}")
                self._buckets[target].clear()
                self._recent_points[target].clear()
                self._last_update_mono[target] = 0.0

    def snapshot(self) -> dict:
        now = time.monotonic()
        mapping = self.spatial_mapper.snapshot() if self.spatial_mapper is not None else {}
        rooms_cfg = mapping.get("rooms") or {}
        calibrations = mapping.get("calibrations") or {}
        usable_cameras = {
            str(camera_id)
            for camera_id, item in calibrations.items()
            if item.get("status") in {"good", "calibrated", "automatic"}
            and item.get("homography")
        }
        with self._lock:
            self._prune_locked(now)
            rooms = {}
            for room_id in self.room_ids:
                buckets = tuple(self._buckets.get(room_id, {}).values())
                raw_samples = sum(bucket.samples for bucket in buckets)
                weighted_samples = sum(
                    bucket.samples * self._bucket_weight(self._bucket_age(bucket, now))
                    for bucket in buckets
                )
                last = self._last_update_mono.get(room_id, 0.0)
                cameras = [str(item) for item in ((rooms_cfg.get(room_id) or {}).get("cameras") or [])]
                calibrated_count = sum(camera_id in usable_cameras for camera_id in cameras)
                rooms[room_id] = {
                    "calibrated": calibrated_count > 0,
                    "fused": bool(cameras) and calibrated_count == len(cameras),
                    "calibrated_cameras": calibrated_count,
                    "total_cameras": len(cameras),
                    "samples": raw_samples,
                    "weighted_samples": round(float(weighted_samples), 3),
                    "active_buckets": len(buckets),
                    "last_update_age_sec": round(max(0.0, now - last), 3) if last else None,
                }
            return {
                "enabled": self.enabled,
                "source": "pose_ankles",
                "coordinate_system": "normalized_room_floor",
                "grid_size": [self.grid_width, self.grid_height],
                "cooling": {
                    "hot_hold_sec": self.hot_hold_sec,
                    "cool_half_life_sec": self.cool_half_life_sec,
                    "bucket_sec": self.bucket_sec,
                    "max_history_sec": self.max_history_sec,
                },
                "rooms": rooms,
                "metrics": {
                    "processed_pose_frames": self._processed_pose_frames,
                    "samples": self._samples,
                    "duplicate_skips": self._duplicate_skips,
                    "uncalibrated_skips": self._uncalibrated_skips,
                    "ankle_skips": self._ankle_skips,
                    "projection_skips": self._projection_skips,
                    "errors": self._errors,
                    "last_error": self._last_error,
                },
            }
