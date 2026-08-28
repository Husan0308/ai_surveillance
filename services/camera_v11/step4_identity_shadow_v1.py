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
    best_peer_key: tuple[str, str] | None = None
    best_peer_streak: int = 0
    active: bool = False
    active_since: float | None = None
    inactive_since: float | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.camera_id, self.track_id


class V11CrossCameraIdentityShadowV1:
    """Calibration-first cross-camera matcher with conservative safety gates.

    Similarity follows the NVIDIA tracker gallery idea: the primary ReID score is
    the strongest cosine match between samples in the two short feature galleries.
    For room-handoff mode, cross-room association can additionally require exactly
    one recently-active identity and one recently-lost identity. This prevents two
    unrelated people who are simultaneously active in neighboring rooms from being
    linked by appearance alone.
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
        min_gallery_samples: int = 3,
        candidate_votes: int = 2,
        strong_votes: int = 3,
        min_cross_room_gap_sec: float = 1.5,
        allowed_room_transitions: dict[str, set[str]] | None = None,
        require_handoff_lifecycle: bool = False,
        recently_active_sec: float = 8.0,
        recently_lost_sec: float = 30.0,
    ) -> None:
        self.gallery_size = max(2, min(8, int(gallery_size)))
        self.ttl_sec = max(3.0, float(ttl_sec))
        self.weak_similarity = float(weak_similarity)
        self.candidate_similarity = float(candidate_similarity)
        self.strong_similarity = float(strong_similarity)
        self.min_margin = max(0.0, float(min_margin))
        self.strong_margin = max(0.0, float(strong_margin))
        self.min_gallery_samples = max(2, min(self.gallery_size, int(min_gallery_samples)))
        self.candidate_votes = max(1, int(candidate_votes))
        self.strong_votes = max(self.candidate_votes, int(strong_votes))
        self.min_cross_room_gap_sec = max(0.0, float(min_cross_room_gap_sec))
        self.require_handoff_lifecycle = bool(require_handoff_lifecycle)
        self.recently_active_sec = max(1.0, float(recently_active_sec))
        self.recently_lost_sec = max(self.recently_active_sec, float(recently_lost_sec))
        self.allowed_room_transitions = (
            None
            if allowed_room_transitions is None
            else {
                str(room): {str(peer) for peer in peers}
                for room, peers in allowed_room_transitions.items()
            }
        )
        self._lock = threading.RLock()
        self._tracks: dict[tuple[str, str], _TrackGallery] = {}
        self._activity: dict[tuple[str, str], tuple[bool, float | None, float | None]] = {}
        self.observations = 0
        self.weak = 0
        self.candidates = 0
        self.strong = 0
        self.ambiguous = 0
        self.insufficient_samples = 0
        self.topology_rejects = 0
        self.route_rejects = 0
        self.time_rejects = 0
        self.lifecycle_rejects = 0
        self.nonreciprocal = 0
        self.vote_waits = 0

    def update_track_activity(
        self,
        *,
        camera_id: str,
        track_id: str,
        active: bool,
        observed_at: float | None = None,
    ) -> None:
        now = time.monotonic() if observed_at is None else float(observed_at)
        key = (str(camera_id), str(track_id))
        active = bool(active)
        with self._lock:
            previous = self._activity.get(key)
            if previous is None:
                state = (active, now if active else None, None if active else now)
            else:
                was_active, active_since, inactive_since = previous
                if active and not was_active:
                    state = (True, now, None)
                elif not active and was_active:
                    state = (False, active_since, now)
                else:
                    state = (active, active_since, inactive_since)
            self._activity[key] = state
            gallery = self._tracks.get(key)
            if gallery is not None:
                gallery.active, gallery.active_since, gallery.inactive_since = state

    @staticmethod
    def _prototype(gallery: _TrackGallery) -> np.ndarray:
        matrix = np.stack(gallery.embeddings, axis=0)
        weights = np.asarray(gallery.qualities, dtype=np.float32)
        weights = np.maximum(weights, 0.05)
        return _normalize(np.average(matrix, axis=0, weights=weights))

    @classmethod
    def _gallery_similarity(cls, left: _TrackGallery, right: _TrackGallery) -> tuple[float, float]:
        left_matrix = np.stack(left.embeddings, axis=0)
        right_matrix = np.stack(right.embeddings, axis=0)
        gallery_score = float(np.max(left_matrix @ right_matrix.T))
        prototype_score = float(np.dot(cls._prototype(left), cls._prototype(right)))
        return gallery_score, prototype_score

    def _expire(self, now: float) -> None:
        stale = [key for key, row in self._tracks.items() if now - row.last_seen > self.ttl_sec]
        for key in stale:
            self._tracks.pop(key, None)
            self._activity.pop(key, None)

    def _pair_allowed(
        self,
        left: _TrackGallery,
        right: _TrackGallery,
    ) -> tuple[bool, float, str]:
        if left.key == right.key or left.camera_id == right.camera_id:
            return False, 0.0, "same_camera"

        gap_sec = abs(float(left.last_seen) - float(right.last_seen))
        same_room = bool(left.room_id and left.room_id == right.room_id)
        if same_room:
            return True, gap_sec, "same_room"

        if left.room_id and right.room_id and self.allowed_room_transitions is not None:
            allowed = self.allowed_room_transitions.get(left.room_id, set())
            if right.room_id not in allowed:
                return False, gap_sec, "route"

        if self.require_handoff_lifecycle:
            if left.active == right.active:
                return False, gap_sec, "lifecycle"
            active_row = left if left.active else right
            inactive_row = right if left.active else left
            if active_row.active_since is None or inactive_row.inactive_since is None:
                return False, gap_sec, "lifecycle"
            now = max(float(left.last_seen), float(right.last_seen))
            active_age = max(0.0, now - float(active_row.active_since))
            inactive_age = max(0.0, now - float(inactive_row.inactive_since))
            if active_age > self.recently_active_sec or inactive_age > self.recently_lost_sec:
                return False, gap_sec, "lifecycle"

        if (
            left.room_id
            and right.room_id
            and left.room_id != right.room_id
            and gap_sec < self.min_cross_room_gap_sec
        ):
            return False, gap_sec, "time"
        return True, gap_sec, "cross_room"

    def _rank_for(
        self,
        gallery: _TrackGallery,
    ) -> tuple[list[tuple[float, float, _TrackGallery, float]], int, int, int]:
        ranked: list[tuple[float, float, _TrackGallery, float]] = []
        route_rejects = 0
        time_rejects = 0
        lifecycle_rejects = 0
        for other in self._tracks.values():
            if other.key == gallery.key or other.camera_id == gallery.camera_id:
                continue
            if len(other.embeddings) < self.min_gallery_samples:
                continue
            allowed, gap_sec, reason = self._pair_allowed(gallery, other)
            if not allowed:
                if reason == "route":
                    route_rejects += 1
                elif reason == "time":
                    time_rejects += 1
                elif reason == "lifecycle":
                    lifecycle_rejects += 1
                continue
            gallery_score, prototype_score = self._gallery_similarity(gallery, other)
            ranked.append((gallery_score, prototype_score, other, gap_sec))
        ranked.sort(key=lambda row: row[0], reverse=True)
        return ranked, route_rejects, time_rejects, lifecycle_rejects

    @staticmethod
    def _none_decision(gallery_samples: int, *, reason: str = "none") -> dict[str, object]:
        return {
            "state": "NONE",
            "reason": reason,
            "score": 0.0,
            "prototype_score": 0.0,
            "margin": 0.0,
            "candidate_camera": "",
            "candidate_track": "",
            "same_room": False,
            "gallery_samples": int(gallery_samples),
            "peer_samples": 0,
            "reciprocal": False,
            "consistency_votes": 0,
            "cross_room_gap_sec": -1.0,
        }

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
                active, active_since, inactive_since = self._activity.get(
                    key, (False, None, now)
                )
                gallery = _TrackGallery(
                    str(camera_id),
                    str(track_id),
                    str(room_id),
                    now,
                    active=active,
                    active_since=active_since,
                    inactive_since=inactive_since,
                )
                self._tracks[key] = gallery
            gallery.last_seen = max(gallery.last_seen, now)
            gallery.room_id = str(room_id)
            gallery.embeddings.append(vector)
            gallery.qualities.append(quality)
            if len(gallery.embeddings) > self.gallery_size:
                gallery.embeddings = gallery.embeddings[-self.gallery_size :]
                gallery.qualities = gallery.qualities[-self.gallery_size :]

            self.observations += 1
            if len(gallery.embeddings) < self.min_gallery_samples:
                gallery.best_peer_key = None
                gallery.best_peer_streak = 0
                self.insufficient_samples += 1
                return self._none_decision(len(gallery.embeddings), reason="query_samples")

            ranked, route_rejected, time_rejected, lifecycle_rejected = self._rank_for(gallery)
            self.route_rejects += route_rejected
            self.time_rejects += time_rejected
            self.lifecycle_rejects += lifecycle_rejected
            self.topology_rejects += route_rejected + time_rejected + lifecycle_rejected
            if not ranked:
                gallery.best_peer_key = None
                gallery.best_peer_streak = 0
                return self._none_decision(len(gallery.embeddings), reason="no_feasible_peer")

            best_score, best_prototype_score, best, gap_sec = ranked[0]
            runner_score = ranked[1][0] if len(ranked) > 1 else -1.0
            margin = best_score - runner_score
            best_key = best.key

            if gallery.best_peer_key == best_key:
                gallery.best_peer_streak += 1
            else:
                gallery.best_peer_key = best_key
                gallery.best_peer_streak = 1

            reverse_ranked, reverse_route_rejected, reverse_time_rejected, reverse_lifecycle_rejected = self._rank_for(best)
            self.route_rejects += reverse_route_rejected
            self.time_rejects += reverse_time_rejected
            self.lifecycle_rejects += reverse_lifecycle_rejected
            self.topology_rejects += (
                reverse_route_rejected + reverse_time_rejected + reverse_lifecycle_rejected
            )
            reciprocal = bool(reverse_ranked and reverse_ranked[0][2].key == gallery.key)

            same_room = bool(gallery.room_id and gallery.room_id == best.room_id)
            state = "NONE"
            reason = "below_weak"

            if best_score >= self.weak_similarity:
                state = "WEAK"
                reason = "weak"
                self.weak += 1

                margin_needed = (
                    self.strong_margin if best_score >= self.strong_similarity else self.min_margin
                )
                if margin < margin_needed:
                    reason = "margin"
                    if best_score >= self.candidate_similarity:
                        self.ambiguous += 1
                elif not reciprocal:
                    reason = "nonreciprocal"
                    self.nonreciprocal += 1
                    if best_score >= self.candidate_similarity:
                        self.ambiguous += 1
                elif best_score >= self.strong_similarity:
                    if gallery.best_peer_streak >= self.strong_votes:
                        state = "STRONG"
                        reason = "strong"
                        self.strong += 1
                    else:
                        reason = "vote_wait"
                        self.vote_waits += 1
                        self.ambiguous += 1
                elif best_score >= self.candidate_similarity:
                    if gallery.best_peer_streak >= self.candidate_votes:
                        state = "CANDIDATE"
                        reason = "candidate"
                        self.candidates += 1
                    else:
                        reason = "vote_wait"
                        self.vote_waits += 1
                        self.ambiguous += 1

            return {
                "state": state,
                "reason": reason,
                "score": best_score,
                "prototype_score": best_prototype_score,
                "margin": margin,
                "candidate_camera": best.camera_id,
                "candidate_track": best.track_id,
                "same_room": same_room,
                "gallery_samples": len(gallery.embeddings),
                "peer_samples": len(best.embeddings),
                "reciprocal": reciprocal,
                "consistency_votes": gallery.best_peer_streak,
                "cross_room_gap_sec": -1.0 if same_room else gap_sec,
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
                "insufficient_samples": self.insufficient_samples,
                "topology_rejects": self.topology_rejects,
                "route_rejects": self.route_rejects,
                "time_rejects": self.time_rejects,
                "lifecycle_rejects": self.lifecycle_rejects,
                "nonreciprocal": self.nonreciprocal,
                "vote_waits": self.vote_waits,
            }
