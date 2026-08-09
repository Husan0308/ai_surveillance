"""Independent custom two-stage motion/IoU tracker for one camera."""
import threading
import numpy as np
import time
from .association import iou_matrix, greedy_match, appearance_matrix, motion_proximity_matrix
from .track import Track
from .schemas import TrackState, CameraTrackResult
from .metrics import CameraTrackingMetrics

class CameraTracker:
    def __init__(self, camera_id, config=None):
        cfg = config or {}; self.camera_id = camera_id; self._lock = threading.RLock()
        self.high = float(cfg.get("track_high_thresh", .22)); self.low = float(cfg.get("track_low_thresh", .05))
        self.new_threshold = float(cfg.get("new_track_thresh", .28)); self.match = float(cfg.get("match_thresh", .35))
        self.relaxed_match = float(cfg.get("relaxed_match_thresh", max(.1, self.match * .6)))
        self.min_hits = int(cfg.get("min_confirmed_hits", 3)); self.nominal_fps = max(1.0, float(cfg.get("effective_ai_fps", 10)))
        self.max_lost_ms = float(cfg.get("max_lost_time_ms", float(cfg.get("lost_memory_seconds", 1.5)) * 1000))
        self.max_lost_frames = int(cfg.get("max_lost_frames", max(1, int(round(self.max_lost_ms * self.nominal_fps / 1000)))))
        self.recovery_motion_enabled = bool(cfg.get("recovery_motion_enabled", True)); self.recovery_max_distance = float(cfg.get("recovery_max_normalized_distance", 1.5))
        self.tracks = []; self._next_id = 1; self.metrics = CameraTrackingMetrics()

    def update(self, result, embeddings=None):
        started = time.perf_counter(); now = result.receive_timestamp
        detections = list(result.detections); embeddings = embeddings or [None] * len(detections)
        self.metrics.detections=len(detections)
        with self._lock:
            for track in self.tracks:
                if track.state != TrackState.REMOVED and track.misses and ((now-track.last_seen_at)*1000 > self.max_lost_ms or track.misses > self.max_lost_frames):
                    track.state=TrackState.REMOVED;self.metrics.removed_tracks+=1;self.metrics.deleted_tracks+=1
            active = [t for t in self.tracks if t.state != TrackState.REMOVED]
            motion_started = time.perf_counter(); predicted = [t.predict() for t in active]
            self.metrics.motion_ms = (time.perf_counter() - motion_started) * 1000
            association_started = time.perf_counter(); matched_tracks, matched_detections = set(), set()

            def stage(track_indices, detection_indices, threshold, label, use_appearance=False):
                if not track_indices or not detection_indices: return
                appearance_used = False
                scores = iou_matrix([predicted[i] for i in track_indices], [detections[j].bbox_xyxy for j in detection_indices])
                if label == "recovery" and self.recovery_motion_enabled:
                    proximity=motion_proximity_matrix([predicted[i] for i in track_indices],[detections[j].bbox_xyxy for j in detection_indices],self.recovery_max_distance)
                    scores=np.maximum(scores,proximity)
                if use_appearance:
                    valid = all(active[i].appearance_embedding is not None for i in track_indices) and all(embeddings[j] is not None for j in detection_indices)
                    if valid:
                        sim_started = time.perf_counter()
                        similarity = appearance_matrix([embeddings[j] for j in detection_indices], [active[i].appearance_embedding for i in track_indices])
                        appearance_used = True
                        self.metrics.appearance_similarity_ms += (time.perf_counter() - sim_started) * 1000
                        scores = .65 * scores + .35 * similarity
                for row, col, _score in greedy_match(scores, threshold):
                    ti, di = track_indices[row], detection_indices[col]
                    if ti in matched_tracks or di in matched_detections: continue
                    recovered = active[ti].update(detections[di].bbox_xyxy, detections[di].confidence,
                                                   result.frame_id, now, embeddings[di], self.min_hits)
                    if label == "high": self.metrics.high_confidence_matches += 1
                    else: self.metrics.low_confidence_recovery_matches += 1
                    if recovered and appearance_used: self.metrics.appearance_assisted_recoveries += 1
                    if recovered: self.metrics.recovered_tracks += 1
                    matched_tracks.add(ti); matched_detections.add(di)

            confirmed = [i for i, t in enumerate(active) if t.state == TrackState.CONFIRMED]
            high = [i for i, d in enumerate(detections) if d.confidence >= self.high]
            stage(confirmed, high, self.match, "high")
            remaining_tracks = [i for i in range(len(active)) if i not in matched_tracks and active[i].state not in (TrackState.LOST, TrackState.REMOVED)]
            remaining_dets = [i for i, d in enumerate(detections) if i not in matched_detections and d.confidence >= self.low]
            stage(remaining_tracks, remaining_dets, self.relaxed_match, "low", True)
            lost = [i for i, track in enumerate(active) if track.state == TrackState.LOST and i not in matched_tracks]
            stage(lost, [i for i in remaining_dets if i not in matched_detections], self.relaxed_match * .8, "recovery", True)
            self.metrics.association_ms = (time.perf_counter() - association_started) * 1000

            for i, track in enumerate(active):
                if i not in matched_tracks: track.miss()
            for index, detection in enumerate(detections):
                if index in matched_detections or detection.confidence < self.new_threshold: continue
                track = Track(self.camera_id, self._next_id, detection.bbox_xyxy, detection.confidence,
                              result.frame_id, now, now, result.frame_id, appearance_embedding=embeddings[index])
                self._next_id += 1; self.tracks.append(track); self.metrics.new_tracks += 1
            self.metrics.unmatched_detections += sum(index not in matched_detections and detection.confidence < self.new_threshold for index,detection in enumerate(detections))
            for track in self.tracks:
                if track.state == TrackState.REMOVED: continue
                if track.misses > self.max_lost_frames or (now - track.last_seen_at) * 1000 > self.max_lost_ms:
                    track.state = TrackState.REMOVED; self.metrics.removed_tracks += 1; self.metrics.deleted_tracks += 1
            visible_tracks = [t for t in self.tracks if t.state in (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST)]; visible = [t.output(now) for t in visible_tracks]
            self.metrics.tracks_active = sum(t.state != TrackState.REMOVED for t in self.tracks)
            self.metrics.tracks_confirmed = sum(t.state == TrackState.CONFIRMED for t in self.tracks)
            self.metrics.average_track_age_seconds = sum(now-t.created_at for t in visible_tracks)/len(visible_tracks) if visible_tracks else 0.0
            lost_tracks = [t for t in visible_tracks if t.state == TrackState.LOST]
            self.metrics.average_lost_duration_seconds = sum(now-t.last_seen_at for t in lost_tracks)/len(lost_tracks) if lost_tracks else 0.0
            self.metrics.tracks_lost = sum(t.state == TrackState.LOST for t in self.tracks)
            self.metrics.tracker_update_ms = (time.perf_counter() - started) * 1000
            return CameraTrackResult(result.camera_id, result.frame_id, result.capture_timestamp, result.receive_timestamp, tuple(visible))

    def reset(self):
        with self._lock: self.tracks.clear(); self._next_id = 1
