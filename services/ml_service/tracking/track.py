from dataclasses import dataclass, field
import time
import numpy as np
from .motion import BoxMotionModel
from .schemas import TrackState, TrackedPerson

@dataclass
class Track:
    camera_id: str
    local_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    first_frame_id: int
    created_at: float
    last_seen_at: float
    last_frame_id: int
    state: TrackState = TrackState.TENTATIVE
    age_frames: int = 1
    hits: int = 1
    misses: int = 0
    appearance_embedding: np.ndarray | None = None
    motion: BoxMotionModel = field(init=False, repr=False)

    def __post_init__(self): self.motion = BoxMotionModel(self.bbox)
    @property
    def track_id(self): return f"{self.camera_id}:TRACK-{self.local_id:05d}"
    @property
    def velocity(self): return tuple(float(v) for v in self.motion.velocity)
    def predict(self): return tuple(float(v) for v in self.motion.predict())

    def update(self, bbox, confidence, frame_id, timestamp, embedding=None, min_hits=3):
        recovered = self.state == TrackState.LOST
        self.motion.update(bbox); self.bbox = tuple(float(v) for v in bbox)
        self.confidence = float(confidence); self.last_frame_id = frame_id; self.last_seen_at = timestamp
        self.age_frames += 1; self.hits += 1; self.misses = 0
        if self.hits >= min_hits: self.state = TrackState.CONFIRMED
        if embedding is not None:
            emb = np.asarray(embedding, np.float32); norm = np.linalg.norm(emb)
            if norm: emb /= norm
            if self.appearance_embedding is not None and self.appearance_embedding.shape == emb.shape:
                emb = .8 * self.appearance_embedding + .2 * emb; emb /= max(np.linalg.norm(emb), 1e-12)
            self.appearance_embedding = emb
        return recovered

    def miss(self):
        self.motion.miss(); self.bbox = tuple(float(v) for v in self.motion.bbox)
        self.age_frames += 1; self.misses += 1; self.state = TrackState.LOST

    def output(self):
        return TrackedPerson(self.track_id, self.state, self.bbox, self.confidence,
                             self.age_frames, self.hits, self.misses, self.velocity)
