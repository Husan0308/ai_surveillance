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

@dataclass(frozen=True, slots=True)
class CameraTrackResult:
    camera_id: str
    frame_id: int
    capture_timestamp: float
    receive_timestamp: float
    tracks: tuple[TrackedPerson, ...]

@dataclass(frozen=True, slots=True)
class TrackingBatchResult:
    batch_id: int
    started_at: float
    completed_at: float
    results: tuple[CameraTrackResult, ...]
