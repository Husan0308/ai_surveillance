from dataclasses import dataclass, field
import numpy as np
from .schemas import IdentityStatus, IdentityType

@dataclass
class GlobalIdentity:
    global_id: str; created_at: float; last_seen_at: float
    last_camera_id: str; last_local_track_id: str
    status: IdentityStatus = IdentityStatus.ACTIVE
    identity_type: IdentityType = IdentityType.UNKNOWN
    appearance_embedding: np.ndarray | None = None
    appearance_history: list = field(default_factory=list)
    camera_history: list = field(default_factory=list)
    track_history: list = field(default_factory=list)
    confidence: float = 0.0
    active_tracks: dict = field(default_factory=dict)
    active_track_seen: dict = field(default_factory=dict)
    person_id: str | None = None
    display_name: str | None = None
    last_bbox: tuple | None = None
    last_source_size: tuple = (0,0)
    gallery_trusted: bool = True
    gallery_min_similarity: float = 1.0
    gallery_mean_similarity: float = 1.0

    def add_embedding(self, embedding, quality, max_history):
        if embedding is None: return False
        value=np.asarray(embedding,np.float32); norm=np.linalg.norm(value)
        if not norm: return False
        value=value/norm; self.appearance_history.append((value,float(quality),self.last_seen_at))
        self.appearance_history.sort(key=lambda item:(item[1],item[2]),reverse=True)
        del self.appearance_history[max_history:]
        weighted=sum(item[0]*max(item[1],.01) for item in self.appearance_history)
        self.appearance_embedding=weighted/max(np.linalg.norm(weighted),1e-12); return True

    def audit_gallery(self,minimum_internal_similarity=.65):
        values=[np.asarray(item[0],np.float32) for item in self.appearance_history if item[0] is not None]
        similarities=[]
        for index,left in enumerate(values):
            left=left/max(float(np.linalg.norm(left)),1e-12)
            for right in values[index+1:]:
                right=right/max(float(np.linalg.norm(right)),1e-12);similarities.append(float(left@right))
        self.gallery_min_similarity=min(similarities) if similarities else 1.0
        self.gallery_mean_similarity=float(np.mean(similarities)) if similarities else 1.0
        self.gallery_trusted=self.gallery_min_similarity>=float(minimum_internal_similarity)
        return self.gallery_trusted
