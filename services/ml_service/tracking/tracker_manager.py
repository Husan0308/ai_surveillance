"""Owns isolated camera trackers and globally batches optional appearance work."""
import heapq,json,os
from pathlib import Path
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
        self._prediction_heap=[];self._prediction_sequence=0;self._prediction_dir=Path("data/diagnostics/tracking_prediction_worst")
        self._prediction_write_lock=threading.Lock();self._prediction_dirty={};self._prediction_writer=None
        self._prediction_write_event=threading.Event();self._prediction_stop=threading.Event()
        self.reconnect_grace_ms = float(self.config.get("reconnect_grace_period_ms", 5000))
        self.appearance = appearance_extractor or AppearanceExtractor(None, self.config.get("appearance_device", "cuda"), self.config.get("appearance_batch_size", 32))
        self._lock = threading.RLock(); self._trackers = {}; self._disconnected = {}; self.metrics = TrackingMetrics()
        self.reid_batch_size=0;self.reid_extract_ms=0.0;self.reid_model_instance_count=1 if self.appearance.available else 0

    def _tracker(self, camera_id):
        with self._lock:
            if camera_id not in self._trackers: self._trackers[camera_id] = CameraTracker(camera_id, self.config)
            return self._trackers[camera_id]

    def update(self, camera_detection_result, embeddings=None, now_monotonic=None):
        tracker = self._tracker(camera_detection_result.camera_id); result = tracker.update(camera_detection_result, embeddings, now_monotonic)
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
            tracker=self._tracker(camera.camera_id);before=tracker.metrics.prediction_backtest_count
            result = self.update(camera, embeddings); tracker = self._tracker(camera.camera_id)
            added=tracker.metrics.prediction_backtest_count-before
            if added>0 and frames_by_camera.get(camera.camera_id) is not None and os.environ.get("SURVEILLANCE_DEBUG_PREDICTIONS"):
                self._save_prediction_samples(frames_by_camera[camera.camera_id],tracker.metrics.prediction_backtests[-added:],camera.camera_id)
            tracker.metrics.appearance_crop_ms = crop_ms; tracker.metrics.appearance_preprocess_ms = preprocess_ms; tracker.metrics.appearance_gpu_ms = gpu_ms
            results.append(result)
        return TrackingBatchResult(detection_batch.batch_id, started, time.time(), tuple(results))


    def _ensure_prediction_writer(self):
        with self._prediction_write_lock:
            if self._prediction_writer is not None:return
            self._prediction_writer=threading.Thread(target=self._prediction_writer_loop,name="cam05-prediction-writer",daemon=True)
            self._prediction_writer.start()

    def _prediction_writer_loop(self):
        while not self._prediction_stop.is_set():
            self._prediction_write_event.wait(.5);self._prediction_write_event.clear()
            if not self._prediction_stop.is_set():self._flush_prediction_samples()
        self._flush_prediction_samples()

    def _flush_prediction_samples(self):
        with self._prediction_write_lock:
            dirty=self._prediction_dirty;self._prediction_dirty={}
            heap_snapshot=tuple(self._prediction_heap)
        if not dirty and not heap_snapshot:return
        self._prediction_dir.mkdir(parents=True,exist_ok=True)
        import cv2
        for slot,(crop,payload) in dirty.items():
            stem=f"sample_{slot:03d}"
            cv2.imwrite(str(self._prediction_dir/f"{stem}.jpg"),crop)
            (self._prediction_dir/f"{stem}.json").write_text(json.dumps(payload,indent=2,default=float))
        if heap_snapshot:
            manifest={"camera_id":"ALL","capacity":100,"saved":len(heap_snapshot),"minimum_retained_error":min(item[0] for item in heap_snapshot),"updated_at":time.time()}
            (self._prediction_dir/"manifest.json").write_text(json.dumps(manifest,indent=2))

    def _save_prediction_samples(self,frame,audits,camera_id):
        import cv2
        height,width=frame.shape[:2]
        for audit in audits:
            score=float(audit.get("after",{}).get("center_error_normalized",0.0));self._prediction_sequence+=1
            with self._prediction_write_lock:
                if len(self._prediction_heap)<100:
                    slot=len(self._prediction_heap);heapq.heappush(self._prediction_heap,(score,self._prediction_sequence,slot))
                elif score>self._prediction_heap[0][0]:
                    slot=self._prediction_heap[0][2];heapq.heapreplace(self._prediction_heap,(score,self._prediction_sequence,slot))
                else:continue
            boxes=[audit.get(name) for name in ("legacy_bbox","corrected_bbox","actual_bbox") if audit.get(name)]
            x1=max(0,int(min(box[0] for box in boxes))-16);y1=max(0,int(min(box[1] for box in boxes))-16);x2=min(width,int(max(box[2] for box in boxes))+16);y2=min(height,int(max(box[3] for box in boxes))+16)
            if x2<=x1 or y2<=y1:continue
            crop=frame[y1:y2,x1:x2].copy();colors=((255,0,0),(0,255,255),(0,255,0))
            for name,color in zip(("legacy_bbox","corrected_bbox","actual_bbox"),colors):
                box=audit.get(name)
                if box:cv2.rectangle(crop,(round(box[0])-x1,round(box[1])-y1),(round(box[2])-x1,round(box[3])-y1),color,1);cv2.putText(crop,name.split("_")[0],(max(0,round(box[0])-x1),max(10,round(box[1])-y1)),cv2.FONT_HERSHEY_SIMPLEX,.3,color,1,cv2.LINE_AA)
            payload={**audit,"camera_id":camera_id,"crop_origin":[x1,y1],"crop_shape":[x2-x1,y2-y1],"rank_score":score,"saved_at":time.time()}
            with self._prediction_write_lock:self._prediction_dirty[slot]=(crop,payload)
            self._ensure_prediction_writer();self._prediction_write_event.set()
    def predict_visual(self,camera_id,frame_id,capture_timestamp,receive_timestamp,now_monotonic=None):
        with self._lock:tracker=self._trackers.get(camera_id)
        if tracker is None:return None
        return tracker.predict_visual(frame_id,capture_timestamp,receive_timestamp,now_monotonic)

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
    def scheduler_risks(self,now_monotonic=None):
        """Small immutable tracker summary for the single scheduler authority."""
        import numpy as np
        current=time.monotonic() if now_monotonic is None else float(now_monotonic)
        with self._lock:trackers=tuple(self._trackers.values())
        output={}
        for tracker in trackers:
            with tracker._lock:
                retained=[track for track in tracker.tracks if track.state.value in ("CONFIRMED","LOST")]
                age=max((current-track.last_real_observation_monotonic)*1000 for track in retained) if retained else 0.0
                uncertainty=max((float(np.trace(track.motion.covariance[:4,:4])) for track in retained),default=0.0)
                output[tracker.camera_id]={"observation_age_ms":max(0.0,age),"active_person_count":sum(1 for track in tracker.tracks if track.state.value in ("CONFIRMED","TENTATIVE")),"lost_tracks":sum(track.state.value=="LOST" for track in retained),"motion_uncertainty":min(4.0,uncertainty/100.0),"association_ambiguity":tracker.metrics.association_last_ambiguity}
        return output
    def active_track_keys(self):
        with self._lock:trackers=tuple(self._trackers.values())
        keys=set()
        from .schemas import TrackState
        for tracker in trackers:
            with tracker._lock:
                keys.update((tracker.camera_id,track.track_id) for track in tracker.tracks if track.state!=TrackState.REMOVED)
        return keys
    def embedding_for(self,camera_id,track_id):
        tracker=self._tracker(camera_id)
        with tracker._lock:
            for track in tracker.tracks:
                if track.track_id==track_id:return track.appearance_embedding
        return None
    def embedding_evidence_for(self,camera_id,track_id):
        tracker=self._tracker(camera_id)
        with tracker._lock:
            for track in tracker.tracks:
                if track.track_id==track_id:return track.appearance_embedding,float(track.appearance_quality),track.appearance_frame_id,track.appearance_timestamp
        return None,0.0,None,None
    def set_embedding(self,camera_id,track_id,embedding,quality=0.0,frame_id=None,capture_timestamp=None):
        import numpy as np
        if embedding is None:return False
        value=np.asarray(embedding,np.float32).reshape(-1);norm=float(np.linalg.norm(value))
        if value.size<2 or not np.all(np.isfinite(value)) or not np.isfinite(norm) or norm<=1e-12:return False
        tracker=self._tracker(camera_id);value/=norm
        with tracker._lock:
            for track in tracker.tracks:
                if track.track_id==track_id:
                    if track.appearance_embedding is not None and track.appearance_embedding.shape==value.shape:
                        value=.8*track.appearance_embedding+.2*value;value/=max(float(np.linalg.norm(value)),1e-12)
                    track.appearance_embedding=value;track.appearance_quality=float(quality);track.appearance_frame_id=frame_id;track.appearance_timestamp=capture_timestamp;track.appearance_version+=1;return True
        return False
    def close(self):
        self._prediction_stop.set();self._prediction_write_event.set()
        writer=self._prediction_writer
        if writer is not None:writer.join(timeout=3)
        with self._lock: self._trackers.clear(); self._disconnected.clear()
