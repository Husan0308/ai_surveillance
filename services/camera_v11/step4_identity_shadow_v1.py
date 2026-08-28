from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    row = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(row))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("invalid zero ReID embedding")
    return row / norm


@dataclass
class _TrackGallery:
    camera_id: str
    track_id: str
    room_id: str
    last_seen: float
    embeddings: list[np.ndarray] = field(default_factory=list)
    qualities: list[float] = field(default_factory=list)


class V11CrossCameraIdentityShadowV1:
    """Calibration-first cross-camera matcher.

    This layer intentionally does not merge or rename Step3 local IDs. It builds a
    tiny per-track ReID gallery and reports the best different-camera candidate,
    score, runner-up margin and confidence band. Live data can then calibrate safe
    merge thresholds without contaminating local tracking with false identities.
    """

    def __init__(
        self,
        *,
        gallery_size: int = 4,
        ttl_sec: float = 30.0,
        weak_similarity: float = 0.66,
        candidate_similarity: float = 0.76,
        strong_similarity: float = 0.84,
        min_margin: float = 0.04,
        strong_margin: float = 0.025,
    ) -> None:
        self.gallery_size = max(2, min(8, int(gallery_size)))
        self.ttl_sec = max(3.0, float(ttl_sec))
        self.weak_similarity = float(weak_similarity)
        self.candidate_similarity = float(candidate_similarity)
        self.strong_similarity = float(strong_similarity)
        self.min_margin = max(0.0, float(min_margin))
        self.strong_margin = max(0.0, float(strong_margin))
        self._lock = threading.RLock()
        self._tracks: dict[tuple[str, str], _TrackGallery] = {}
        self.observations = 0
        self.weak = 0
        self.candidates = 0
        self.strong = 0
        self.ambiguous = 0

    @staticmethod
    def _prototype(gallery: _TrackGallery) -> np.ndarray:
        matrix = np.stack(gallery.embeddings, axis=0)
        weights = np.asarray(gallery.qualities, dtype=np.float32)
        weights = np.maximum(weights, 0.05)
        return _normalize(np.average(matrix, axis=0, weights=weights))

    def _expire(self, now: float) -> None:
        stale = [key for key, row in self._tracks.items() if now - row.last_seen > self.ttl_sec]
        for key in stale:
            self._tracks.pop(key, None)

    def observe(
        self,
        *,
        camera_id: str,
        track_id: str,
        room_id: str,
        embedding: np.ndarray,
        quality: float,
        captured_at: float | None = None,
    ) -> dict[str, object]:
        now = time.monotonic() if captured_at is None else float(captured_at)
        key = (str(camera_id), str(track_id))
        vector = _normalize(embedding)
        quality = max(0.05, min(1.0, float(quality)))

        with self._lock:
            self._expire(now)
            gallery = self._tracks.get(key)
            if gallery is None:
                gallery = _TrackGallery(str(camera_id), str(track_id), str(room_id), now)
                self._tracks[key] = gallery
            gallery.last_seen = max(gallery.last_seen, now)
            gallery.room_id = str(room_id)
            gallery.embeddings.append(vector)
            gallery.qualities.append(quality)
            if len(gallery.embeddings) > self.gallery_size:
                gallery.embeddings = gallery.embeddings[-self.gallery_size :]
                gallery.qualities = gallery.qualities[-self.gallery_size :]

            self.observations += 1
            query = self._prototype(gallery)
            ranked: list[tuple[float, _TrackGallery]] = []
            for other_key, other in self._tracks.items():
                if other_key == key or other.camera_id == gallery.camera_id or not other.embeddings:
                    continue
                score = float(np.dot(query, self._prototype(other)))
                ranked.append((score, other))
            ranked.sort(key=lambda row: row[0], reverse=True)

            if not ranked:
                return {
                    "state": "NONE",
                    "score": 0.0,
                    "margin": 0.0,
                    "candidate_camera": "",
                    "candidate_track": "",
                    "same_room": False,
                    "gallery_samples": len(gallery.embeddings),
                }

            best_score, best = ranked[0]
            runner_score = ranked[1][0] if len(ranked) > 1 else -1.0
            margin = best_score - runner_score
            state = "NONE"
            if best_score >= self.strong_similarity and margin >= self.strong_margin:
                state = "STRONG"
                self.strong += 1
            elif best_score >= self.candidate_similarity and margin >= self.min_margin:
                state = "CANDIDATE"
                self.candidates += 1
            elif best_score >= self.weak_similarity:
                state = "WEAK"
                self.weak += 1
                if best_score >= self.candidate_similarity:
                    self.ambiguous += 1

            return {
                "state": state,
                "score": best_score,
                "margin": margin,
                "candidate_camera": best.camera_id,
                "candidate_track": best.track_id,
                "same_room": bool(gallery.room_id and gallery.room_id == best.room_id),
                "gallery_samples": len(gallery.embeddings),
            }

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "tracks": len(self._tracks),
                "observations": self.observations,
                "weak": self.weak,
                "candidates": self.candidates,
                "strong": self.strong,
                "ambiguous": self.ambiguous,
            }
