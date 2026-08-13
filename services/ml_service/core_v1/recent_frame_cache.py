from __future__ import annotations

from collections import deque
import threading


class RecentInferenceFrameCache:
    """Small per-camera cache of frames that were actually submitted to YOLO.

    Core-v1 camera ingest remains latest-only. This cache exists only so an
    accepted detector result can later obtain the *exact* source frame that
    produced it for ReID cropping. Keeping a few submitted frames is bounded and
    does not create a video backlog.
    """

    def __init__(self, camera_ids, per_camera: int = 4):
        self.per_camera = max(1, int(per_camera))
        self._lock = threading.Lock()
        self._frames = {
            str(camera_id): deque(maxlen=self.per_camera)
            for camera_id in camera_ids
        }
        self._puts = 0
        self._hits = 0
        self._misses = 0

    def put(self, frame) -> None:
        camera_id = str(frame.camera_id)
        with self._lock:
            bucket = self._frames.setdefault(camera_id, deque(maxlen=self.per_camera))
            # One reference only; camera frames are immutable after publication.
            bucket.append(frame)
            self._puts += 1

    def get(self, camera_id: str, frame_id: int):
        camera_id = str(camera_id)
        target = int(frame_id)
        with self._lock:
            bucket = self._frames.get(camera_id)
            if bucket:
                for frame in reversed(bucket):
                    if int(frame.frame_id) == target:
                        self._hits += 1
                        return frame
            self._misses += 1
            return None

    def metrics(self):
        with self._lock:
            return {
                "per_camera_capacity": self.per_camera,
                "cached": {camera_id: len(bucket) for camera_id, bucket in self._frames.items()},
                "puts": self._puts,
                "hits": self._hits,
                "misses": self._misses,
            }
