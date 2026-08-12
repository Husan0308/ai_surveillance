from dataclasses import asdict, dataclass, field
import threading

@dataclass
class CameraTrackingMetrics:
    tracker_update_ms: float = 0.0
    association_ms: float = 0.0
    motion_ms: float = 0.0
    appearance_crop_ms: float = 0.0
    appearance_preprocess_ms: float = 0.0
    appearance_gpu_ms: float = 0.0
    appearance_similarity_ms: float = 0.0
    tracks_active: int = 0
    tracks_confirmed: int = 0
    tracks_lost: int = 0
    detections: int = 0
    new_tracks: int = 0
    recovered_tracks: int = 0
    tombstone_recoveries: int = 0
    removed_tracks: int = 0
    id_switch_suspected: int = 0
    deferred_new_track_admissions: int = 0
    duplicate_new_track_suppressed: int = 0
    high_confidence_matches: int = 0
    low_confidence_recovery_matches: int = 0
    appearance_assisted_recoveries: int = 0
    unmatched_detections: int = 0
    association_candidates_total: int = 0
    association_geometry_rejections: int = 0
    association_ambiguity_abstentions: int = 0
    association_wrong_match_suspected: int = 0
    association_last_ambiguity: float = 0.0
    association_candidates: list = field(default_factory=list)
    average_track_age_seconds: float = 0.0
    average_lost_duration_seconds: float = 0.0
    local_track_fragments: int = 0
    deleted_tracks: int = 0
    visual_prediction_frames: int = 0
    visual_prediction_boxes: int = 0
    visual_prediction_rate: float = 0.0
    visual_track_created: int = 0
    visual_track_removed: int = 0
    visual_gap_started: int = 0
    visual_gap_ended: int = 0
    visual_gap_count: int = 0
    visual_gap_ms: float = 0.0
    last_real_detection_age_ms: float = 0.0
    prediction_age_ms: float = 0.0
    removal_reason: str = ""
    fragment_events: list = field(default_factory=list)
    new_track_audits: list = field(default_factory=list)
    visual_lifecycle_events: list = field(default_factory=list)
    prediction_backtest_count: int = 0
    prediction_horizon_ms_p50: float = 0.0
    prediction_horizon_ms_p95: float = 0.0
    prediction_center_error_px_before_p50: float = 0.0
    prediction_center_error_px_before_p95: float = 0.0
    prediction_center_error_norm_p50: float = 0.0
    prediction_center_error_norm_p95: float = 0.0
    prediction_center_error_px_after_p50: float = 0.0
    prediction_center_error_px_after_p95: float = 0.0
    prediction_iou_before_p50: float = 0.0
    prediction_iou_before_p05: float = 0.0
    prediction_iou_after_p50: float = 0.0
    prediction_iou_after_p05: float = 0.0
    predicted_vs_observed_velocity_ratio_p50: float = 0.0
    predicted_vs_observed_velocity_ratio_p95: float = 0.0
    prediction_horizon_buckets: dict = field(default_factory=dict)
    prediction_backtests: list = field(default_factory=list)
    temporary_miss_predictions_total: int = 0
    boundary_exit_candidates_total: int = 0
    boundary_exit_visual_hides_total: int = 0
    boundary_exit_removal_delays_ms: list = field(default_factory=list)
    visual_expirations_total: int = 0
    visual_reacquisitions_total: int = 0


class TrackingMetrics:
    def __init__(self): self._lock = threading.Lock(); self.cameras = {}; self.total_tracking_ms = 0.0
    def update(self, camera_id, metrics):
        with self._lock: self.cameras[camera_id] = metrics; self.total_tracking_ms = sum(m.tracker_update_ms for m in self.cameras.values())
    def snapshot(self):
        with self._lock:
            cameras = {cid: asdict(item) for cid, item in self.cameras.items()}
            for item in cameras.values():
                delays=sorted(float(value) for value in item.get("boundary_exit_removal_delays_ms",()))
                def percentile(fraction):return delays[min(len(delays)-1,int((len(delays)-1)*fraction))] if delays else 0.0
                item["boundary_exit_removal_delay_ms_p50"]=percentile(.50)
                item["boundary_exit_removal_delay_ms_p95"]=percentile(.95)
                item["boundary_exit_removal_delay_ms_max"]=delays[-1] if delays else 0.0
            return {"cameras": cameras, "total_tracking_ms": self.total_tracking_ms,
                    "total_people_tracked": sum(item["tracks_active"] for item in cameras.values())}
    def format_compact(self):
        data = self.snapshot(); lines = ["TRACKING"]
        for cid, m in data["cameras"].items():
            lines.append(f"{cid} det:{m['detections']} active:{m['tracks_active']} lost:{m['tracks_lost']} new:{m['new_tracks']} recovered:{m['recovered_tracks']} deleted:{m['deleted_tracks']} fragments:{m['local_track_fragments']} switches:{m['id_switch_suspected']} high:{m['high_confidence_matches']} low:{m['low_confidence_recovery_matches']} appearance:{m['appearance_assisted_recoveries']} unmatched:{m['unmatched_detections']} age:{m['average_track_age_seconds']:.1f}s lost_for:{m['average_lost_duration_seconds']:.1f}s predict:{m['visual_prediction_rate']:.1f}/s visual_created:{m['visual_track_created']} visual_removed:{m['visual_track_removed']} visual_gaps:{m['visual_gap_count']} real_age:{m['last_real_detection_age_ms']:.0f}ms")
        lines.append(f"tracking_total:{data['total_tracking_ms']:.2f}ms")
        return "\n".join(lines)
