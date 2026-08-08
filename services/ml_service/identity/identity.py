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

    def add_embedding(self, embedding, quality, max_history):
        if embedding is None: return False
        value=np.asarray(embedding,np.float32); norm=np.linalg.norm(value)
        if not norm: return False
        value=value/norm; self.appearance_history.append((value,float(quality),self.last_seen_at))
        self.appearance_history.sort(key=lambda item:(item[1],item[2]),reverse=True)
        del self.appearance_history[max_history:]
        weighted=sum(item[0]*max(item[1],.01) for item in self.appearance_history)
        self.appearance_embedding=weighted/max(np.linalg.norm(weighted),1e-12); return True
