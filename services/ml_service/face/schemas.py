from dataclasses import dataclass,field
from enum import Enum
from typing import Optional

class FaceDecision(str,Enum): UNKNOWN="UNKNOWN";AMBIGUOUS="AMBIGUOUS";PENDING="PENDING";CONFIRMED="CONFIRMED";REJECTED="REJECTED"
@dataclass(frozen=True,slots=True)
class FaceCandidate: camera_id:str;frame_id:int;local_track_id:str;global_id:str;person_bbox:tuple;confidence:float;timestamp:float
@dataclass(frozen=True,slots=True)
class FaceDetection: bbox:tuple;confidence:float;landmarks:object=None;pose:object=None;embedding:object=None
@dataclass(frozen=True,slots=True)
class FaceQuality: score:float;size_score:float;sharpness_score:float;brightness_score:float;pose_score:float;complete_score:float;accepted:bool
@dataclass(frozen=True,slots=True)
class FaceEmbeddingResult: candidate:FaceCandidate;detection:FaceDetection;quality:FaceQuality;embedding:object
@dataclass(frozen=True,slots=True)
class FaceMatch: person_id:Optional[str];name:Optional[str];similarity:float;confidence:float;second_best_similarity:float;margin:float;decision:FaceDecision
@dataclass
class KnownPersonIdentity: person_id:str;name:str;embeddings:list;created_at:float;updated_at:float;enabled:bool=True
@dataclass
class EnrollmentSample: embedding:object;quality:float;timestamp:float
@dataclass
class EnrollmentSession: session_id:str;person_id:str;name:str;started_at:float;target_samples:int;samples:list=field(default_factory=list);state:str="ACTIVE"
@dataclass(frozen=True,slots=True)
class IdentityResolutionResult: global_id:str;person_id:Optional[str];display_name:Optional[str];face_similarity:float;face_quality:float;identity_confidence:float;decision:FaceDecision;identity_conflict:bool=False
