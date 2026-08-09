from dataclasses import asdict, dataclass
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
    removed_tracks: int = 0
    id_switch_suspected: int = 0
    high_confidence_matches: int = 0
    low_confidence_recovery_matches: int = 0
    appearance_assisted_recoveries: int = 0
    unmatched_detections: int = 0
    average_track_age_seconds: float = 0.0
    average_lost_duration_seconds: float = 0.0
    local_track_fragments: int = 0
    deleted_tracks: int = 0

class TrackingMetrics:
    def __init__(self): self._lock = threading.Lock(); self.cameras = {}; self.total_tracking_ms = 0.0
    def update(self, camera_id, metrics):
        with self._lock: self.cameras[camera_id] = metrics; self.total_tracking_ms = sum(m.tracker_update_ms for m in self.cameras.values())
    def snapshot(self):
        with self._lock:
            cameras = {cid: asdict(item) for cid, item in self.cameras.items()}
            return {"cameras": cameras, "total_tracking_ms": self.total_tracking_ms,
                    "total_people_tracked": sum(item["tracks_active"] for item in cameras.values())}
    def format_compact(self):
        data = self.snapshot(); lines = ["TRACKING"]
        for cid, m in data["cameras"].items():
            lines.append(f"{cid} det:{m['detections']} active:{m['tracks_active']} lost:{m['tracks_lost']} new:{m['new_tracks']} recovered:{m['recovered_tracks']} deleted:{m['deleted_tracks']} high:{m['high_confidence_matches']} low:{m['low_confidence_recovery_matches']} appearance:{m['appearance_assisted_recoveries']} unmatched:{m['unmatched_detections']} age:{m['average_track_age_seconds']:.1f}s lost_for:{m['average_lost_duration_seconds']:.1f}s")
        lines.append(f"tracking_total:{data['total_tracking_ms']:.2f}ms")
        return "\n".join(lines)
