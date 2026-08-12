from dataclasses import dataclass
from enum import Enum
from typing import Optional

class IdentityStatus(str, Enum):
    ACTIVE="ACTIVE"; RECENTLY_LOST="RECENTLY_LOST"; INACTIVE="INACTIVE"; ARCHIVED="ARCHIVED"; AMBIGUOUS="AMBIGUOUS"
class IdentityType(str, Enum): UNKNOWN="UNKNOWN"; KNOWN="KNOWN"

@dataclass(frozen=True, slots=True)
class IdentityTrackObservation:
    camera_id: str; frame_id: int; local_track_id: str
    bbox: tuple[float,float,float,float]; confidence: float
    timestamp: float; appearance_embedding: object = None; quality_score: float = 1.0
    embedding_frame_id: int | None = None
    embedding_timestamp: float | None = None
    source_width: int = 0
    source_height: int = 0
    detection_source: str = "PREDICTED"
    detection_id: str | None = None

@dataclass(frozen=True, slots=True)
class GlobalTrack:
    local_track_id: str; global_id: Optional[str]
    bbox: tuple[float,float,float,float]; confidence: float
    identity_confidence: float; identity_status: IdentityStatus
    decision_reason: str
    person_id: Optional[str] = None
    display_name: Optional[str] = None
    observation_type: str = "detected"
    last_detection_timestamp: float = 0.0
    prediction_age_ms: float = 0.0
    tracker_state: str = "DETECTED"
    identity_version: int = 0
    detection_source: str = "PREDICTED"
    detection_id: str | None = None
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    state_timestamp: float = 0.0
    visual_expires_at: float = 0.0
    track_generation: int = 1
    geometry_monotonic: float = 0.0
    visual_visible: bool = True
    boundary_exit: bool = False

@dataclass(frozen=True, slots=True)
class GlobalTrackResult:
    camera_id: str; frame_id: int; tracks: tuple[GlobalTrack,...]
    identity_version: int = 0
