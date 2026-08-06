import numpy as np
from scipy.spatial.distance import cosine
import threading

class GlobalIdentityManager:
    """Thread‑safe global gallery that stores a single normalized embedding per person ID.
    Provides fast cosine‑similarity lookup for cross‑camera ReID.
    """
    _lock = threading.Lock()
    _gallery: dict[int, np.ndarray] = {}

    @classmethod
    def register(cls, pid: int, emb: np.ndarray) -> None:
        """Add a new global person ID with its embedding (normalized)."""
        with cls._lock:
            cls._gallery[pid] = emb / np.linalg.norm(emb)

    @classmethod
    def update_embedding(cls, pid: int, emb: np.ndarray, alpha: float = 0.2) -> None:
        """EMA update of the stored embedding – adapts to appearance changes."""
        with cls._lock:
            old = cls._gallery.get(pid)
            if old is None:
                cls._gallery[pid] = emb / np.linalg.norm(emb)
                return
            new = old * (1 - alpha) + emb * alpha
            cls._gallery[pid] = new / np.linalg.norm(new)

    @classmethod
    def find_best_match(cls, emb: np.ndarray):
        """Return (best_pid, similarity) or (None, 0.0) if gallery empty.
        Similarity is cosine similarity in [0, 1].
        """
        with cls._lock:
            if not cls._gallery:
                return None, 0.0
            e = emb / np.linalg.norm(emb)
            best_pid, best_score = None, -1.0
            for pid, stored in cls._gallery.items():
                score = 1 - cosine(e, stored)
                if score > best_score:
                    best_score, best_pid = score, pid
            return best_pid, best_score
