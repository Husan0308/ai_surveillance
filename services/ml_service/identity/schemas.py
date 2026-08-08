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

@dataclass(frozen=True, slots=True)
class GlobalTrack:
    local_track_id: str; global_id: Optional[str]
    bbox: tuple[float,float,float,float]; confidence: float
    identity_confidence: float; identity_status: IdentityStatus
    decision_reason: str

@dataclass(frozen=True, slots=True)
class GlobalTrackResult:
    camera_id: str; frame_id: int; tracks: tuple[GlobalTrack,...]
