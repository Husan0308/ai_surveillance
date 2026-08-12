from dataclasses import dataclass
from enum import Enum

class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    REMOVED = "REMOVED"

@dataclass(frozen=True, slots=True)
class TrackedPerson:
    track_id: str
    state: TrackState
    bbox: tuple[float, float, float, float]
    confidence: float
    age_frames: int
    hits: int
    misses: int
    velocity: tuple[float, float]
    camera_id: str = ""
    local_track_id: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    age_seconds: float = 0.0
    lost_duration: float = 0.0
    predicted_bbox: tuple[float, float, float, float] | None = None
    appearance_version: int = 0
    confirmed: bool = False
    observation_type: str = "detected"
    last_detection_timestamp: float = 0.0
    prediction_age_ms: float = 0.0
    detection_source: str = "PREDICTED"
    detection_id: str | None = None
    state_timestamp: float = 0.0
    visual_expires_at: float = 0.0
    track_generation: int = 1
    geometry_monotonic: float = 0.0
    visual_visible: bool = True
    boundary_exit: bool = False

@dataclass(frozen=True, slots=True)
class CameraTrackResult:
    camera_id: str
    frame_id: int
    capture_timestamp: float
    receive_timestamp: float
    tracks: tuple[TrackedPerson, ...]
    capture_monotonic: float = 0.0
    source_width: int = 0
    source_height: int = 0

@dataclass(frozen=True, slots=True)
class TrackingBatchResult:
    batch_id: int
    started_at: float
    completed_at: float
    results: tuple[CameraTrackResult, ...]
