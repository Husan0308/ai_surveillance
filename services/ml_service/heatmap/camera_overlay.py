from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np


class CameraAnkleHeatmapCoordinator:
    """Per-camera heatmap accumulated directly in camera image coordinates.

    Consumes pose results, extracts COCO left/right ankle keypoints (15/16),
    accumulates them on a compact normalized grid, and exposes a cheap overlay()
    method for the JPEG publisher. No room homography is required.
    """

    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def __init__(self, pose, frame_stores, config: dict | None = None):
        self.pose = pose
        self.frame_stores = dict(frame_stores)
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.grid_width = max(32, int(self.config.get("camera_grid_width", 160)))
        self.grid_height = max(18, int(self.config.get("camera_grid_height", 90)))
        self.poll_sec = max(0.03, float(self.config.get("poll_interval_ms", 120)) / 1000.0)
        self.ankle_conf = max(0.0, min(1.0, float(self.config.get("ankle_conf", 0.35))))
        self.sigma_cells = max(0.5, float(self.config.get("sigma_cells", 2.4)))
        self.sample_weight = max(0.01, float(self.config.get("sample_weight", 1.0)))
        self.half_life_sec = max(1.0, float(self.config.get("camera_half_life_sec", 3600.0)))
        self.overlay_alpha = max(0.0, min(1.0, float(self.config.get("overlay_alpha", 0.42))))
        self.overlay_threshold = max(0.0, min(1.0, float(self.config.get("overlay_threshold", 0.08))))
        self._grids = {
            cid: np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
            for cid in self.frame_stores
        }
        self._last_decay = {cid: time.monotonic() for cid in self.frame_stores}
        self._last_pose_frame = {cid: -1 for cid in self.frame_stores}
        self._samples = {cid: 0 for cid in self.frame_stores}
        self._ankle_skips = {cid: 0 for cid in self.frame_stores}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._errors = 0
        self._last_error = ""

    def start(self):
        if not self.enabled or self.pose is None or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="camera-ankle-heatmap",
            daemon=False,
        )
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
        if dt <= 0.0:
            return
        self._grids[camera_id] *= float(0.5 ** (dt / self.half_life_sec))

    @staticmethod
    def _valid_keypoint(person, index: int, minimum: float):
        points = tuple(getattr(person, "keypoints", ()) or ())
        if index >= len(points):
            return None
        point = points[index]
        try:
            x = float(point.x)
            y = float(point.y)
            confidence = float(point.confidence)
        except (AttributeError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x, y, confidence)):
            return None
        if confidence < minimum:
            return None
        return x, y

    def _ankle(self, person):
        points = []
        for index in (self.LEFT_ANKLE, self.RIGHT_ANKLE):
            point = self._valid_keypoint(person, index, self.ankle_conf)
            if point is not None:
                points.append(point)
        if not points:
            return None
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    def _source_size(self, camera_id: str):
        store = self.frame_stores.get(camera_id)
        if store is None:
            return None
        try:
            frame, _ = store.get()
        except Exception:
            return None
        image = getattr(frame, "image", None) if frame is not None else None
        if image is None or getattr(image, "ndim", 0) < 2:
            return None
        height, width = image.shape[:2]
        return float(width), float(height)

    def _add_sample_locked(
        self,
        camera_id: str,
        x: float,
        y: float,
        source_w: float,
        source_h: float,
    ):
        gx = max(
            0.0,
            min(
                self.grid_width - 1.0,
                x / max(1.0, source_w) * (self.grid_width - 1),
            ),
        )
        gy = max(
            0.0,
            min(
                self.grid_height - 1.0,
                y / max(1.0, source_h) * (self.grid_height - 1),
            ),
        )
        radius = max(2, int(math.ceil(self.sigma_cells * 3.0)))
        x0 = max(0, int(gx) - radius)
        x1 = min(self.grid_width, int(gx) + radius + 1)
        y0 = max(0, int(gy) - radius)
        y1 = min(self.grid_height, int(gy) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return
        xs = np.arange(x0, x1, dtype=np.float32) - np.float32(gx)
        ys = np.arange(y0, y1, dtype=np.float32) - np.float32(gy)
        kernel = np.exp(
            -(ys[:, None] ** 2 + xs[None, :] ** 2)
            / (2.0 * self.sigma_cells**2)
        ).astype(np.float32)
        self._grids[camera_id][y0:y1, x0:x1] += (
            kernel * np.float32(self.sample_weight)
        )
        self._samples[camera_id] += 1

    def _run(self):
        while not self._stop.is_set():
            did_work = False
            try:
                snapshot = self.pose.snapshot() if self.pose is not None else {}
                for camera_id, result in snapshot.items():
                    camera_id = str(camera_id)
                    if camera_id not in self._grids:
                        continue
                    if int(result.frame_id) <= self._last_pose_frame.get(camera_id, -1):
                        continue
                    self._last_pose_frame[camera_id] = int(result.frame_id)
                    source_size = self._source_size(camera_id)
                    if not source_size:
                        continue
                    source_w, source_h = source_size
                    now = time.monotonic()
                    with self._lock:
                        self._decay_locked(camera_id, now)
                        for person in result.people:
                            ankle = self._ankle(person)
                            if ankle is None:
                                self._ankle_skips[camera_id] += 1
                                continue
                            self._add_sample_locked(
                                camera_id,
                                ankle[0],
                                ankle[1],
                                source_w,
                                source_h,
                            )
                            did_work = True
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
        now = time.monotonic()
        with self._lock:
            self._decay_locked(camera_id, now)
            grid = self._grids[camera_id].copy()
        peak = float(grid.max()) if grid.size else 0.0
        if peak <= 1e-8:
            return image
        normalized = np.clip(grid / peak, 0.0, 1.0)
        mask = normalized >= self.overlay_threshold
        if not np.any(mask):
            return image
        heat_u8 = np.uint8(np.round(normalized * 255.0))
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_color = cv2.resize(
            heat_color,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        alpha_mask = cv2.resize(
            mask.astype(np.float32),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        alpha_mask = np.clip(
            alpha_mask * self.overlay_alpha,
            0.0,
            self.overlay_alpha,
        )[..., None]
        blended = (
            image.astype(np.float32) * (1.0 - alpha_mask)
            + heat_color.astype(np.float32) * alpha_mask
        )
        return np.uint8(np.clip(blended, 0, 255))

    def reset(self, camera_id: str | None = None):
        with self._lock:
            targets = [str(camera_id)] if camera_id is not None else list(self._grids)
            for cid in targets:
                if cid not in self._grids:
                    raise ValueError(f"camera not found: {cid}")
                self._grids[cid].fill(0.0)
                self._samples[cid] = 0
                self._ankle_skips[cid] = 0
                self._last_decay[cid] = time.monotonic()

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            cameras = {}
            for cid, grid in self._grids.items():
                self._decay_locked(cid, now)
                cameras[cid] = {
                    "samples": int(self._samples[cid]),
                    "ankle_skips": int(self._ankle_skips[cid]),
                    "peak": float(grid.max()) if grid.size else 0.0,
                    "last_pose_frame": int(self._last_pose_frame.get(cid, -1)),
                }
            return {
                "enabled": self.enabled,
                "coordinate_system": "camera_pixels",
                "source": "yolo26m-pose ankles 15/16",
                "overlay": True,
                "cameras": cameras,
                "errors": self._errors,
                "last_error": self._last_error,
            }
