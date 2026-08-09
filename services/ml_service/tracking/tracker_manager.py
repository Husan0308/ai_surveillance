"""Owns isolated camera trackers and globally batches optional appearance work."""
import threading
import time
from .appearance import AppearanceExtractor, crop_detection
from .camera_tracker import CameraTracker
from .metrics import TrackingMetrics
from .schemas import TrackingBatchResult
from shared.logging import get_logger
log = get_logger(__name__)

class TrackerManager:
    def __init__(self, config=None, appearance_extractor=None):
        config = config or {}; self.config = config.get("tracking", config.get("ai", {}).get("tracker", {}))
        root=config or {}
        self.config=dict(self.config);self.config.setdefault("effective_ai_fps",root.get("ai",{}).get("ai_fps",10))
        self.appearance_enabled = bool(self.config.get("appearance_enabled", False))
        self.appearance_interval = max(1, int(self.config.get("appearance_interval_frames", 3)))
        self.reconnect_grace_ms = float(self.config.get("reconnect_grace_period_ms", 5000))
        self.appearance = appearance_extractor or AppearanceExtractor(None, self.config.get("appearance_device", "cuda"), self.config.get("appearance_batch_size", 32))
        self._lock = threading.RLock(); self._trackers = {}; self._disconnected = {}; self.metrics = TrackingMetrics()
        self.reid_batch_size=0;self.reid_extract_ms=0.0;self.reid_model_instance_count=1 if self.appearance.available else 0

    def _tracker(self, camera_id):
        with self._lock:
            if camera_id not in self._trackers: self._trackers[camera_id] = CameraTracker(camera_id, self.config)
            return self._trackers[camera_id]

    def update(self, camera_detection_result, embeddings=None):
        tracker = self._tracker(camera_detection_result.camera_id); result = tracker.update(camera_detection_result, embeddings)
        self.metrics.update(camera_detection_result.camera_id, tracker.metrics); return result

    def update_batch(self, detection_batch, frames_by_camera=None):
        started = time.time(); frames_by_camera = frames_by_camera or {}; embedding_map = {}
        crop_ms = preprocess_ms = gpu_ms = 0.0
        if self.appearance_enabled and self.appearance.available:
            crop_started = time.perf_counter(); crops, keys = [], []
            for camera in detection_batch.results:
                tracker=self._tracker(camera.camera_id)
                needs_embedding=not tracker.tracks or any(track.state.value in ("TENTATIVE","CONFIRMED") and track.appearance_embedding is None for track in tracker.tracks)
                if not needs_embedding and camera.frame_id % self.appearance_interval: continue
                frame = frames_by_camera.get(camera.camera_id)
                if frame is None: continue
                for index, detection in enumerate(camera.detections):
                    crop = crop_detection(frame, detection.bbox_xyxy)
                    if crop is not None and crop.size: crops.append(crop); keys.append((camera.camera_id, index))
            crop_ms = (time.perf_counter() - crop_started) * 1000
            embeddings, timing = self.appearance.extract_batch(crops)
            embedding_map = dict(zip(keys, embeddings)); preprocess_ms = float(timing.get("preprocess_ms", 0)); gpu_ms = float(timing.get("gpu_ms", 0))
            self.reid_batch_size=len(crops);self.reid_extract_ms=float(timing.get("total_ms",gpu_ms))
        else:self.reid_batch_size=0;self.reid_extract_ms=0.0
        results = []
        for camera in detection_batch.results:
            embeddings = [embedding_map.get((camera.camera_id, i)) for i in range(len(camera.detections))]
            result = self.update(camera, embeddings); tracker = self._tracker(camera.camera_id)
            tracker.metrics.appearance_crop_ms = crop_ms; tracker.metrics.appearance_preprocess_ms = preprocess_ms; tracker.metrics.appearance_gpu_ms = gpu_ms
            results.append(result)
        return TrackingBatchResult(detection_batch.batch_id, started, time.time(), tuple(results))

    def remove_camera(self, camera_id):
        with self._lock: self._trackers.pop(camera_id, None); self._disconnected.pop(camera_id, None)
    def reset_camera(self, camera_id):
        with self._lock: tracker = self._trackers.get(camera_id)
        if tracker: tracker.reset()
    def camera_disconnected(self, camera_id, timestamp=None):
        with self._lock: self._disconnected[camera_id] = timestamp or time.time()
    def camera_reconnected(self, camera_id, timestamp=None):
        now = timestamp or time.time()
        with self._lock: disconnected = self._disconnected.pop(camera_id, None)
        if disconnected is None: return True
        if (now - disconnected) * 1000 <= self.reconnect_grace_ms:
            log.info("%s tracker preserved after short reconnect", camera_id); return True
        self.reset_camera(camera_id); log.info("%s tracker reset after long disconnect", camera_id); return False
    def camera_ids(self):
        with self._lock: return tuple(self._trackers)
    def embedding_for(self,camera_id,track_id):
        tracker=self._tracker(camera_id)
        with tracker._lock:
            for track in tracker.tracks:
                if track.track_id==track_id:return track.appearance_embedding
        return None
    def set_embedding(self,camera_id,track_id,embedding):
        import numpy as np
        tracker=self._tracker(camera_id);value=np.asarray(embedding,np.float32);value/=max(float(np.linalg.norm(value)),1e-12)
        with tracker._lock:
            for track in tracker.tracks:
                if track.track_id==track_id:
                    if track.appearance_embedding is not None and track.appearance_embedding.shape==value.shape:
                        value=.8*track.appearance_embedding+.2*value;value/=max(float(np.linalg.norm(value)),1e-12)
                    track.appearance_embedding=value;track.appearance_version+=1;return True
        return False
    def close(self):
        with self._lock: self._trackers.clear(); self._disconnected.clear()
