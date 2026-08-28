from __future__ import annotations

import math
import re
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


def _room_slug(room_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(room_id)).strip("_")
    return slug or "Room"


@dataclass
class _PendingTrack:
    camera_id: str
    track_id: str
    room_id: str
    first_seen: float
    last_seen: float
    embeddings: list[np.ndarray] = field(default_factory=list)
    qualities: list[float] = field(default_factory=list)
    best_identity_id: str | None = None
    best_identity_streak: int = 0

    @property
    def key(self) -> tuple[str, str]:
        return self.camera_id, self.track_id


@dataclass
class _RoomIdentity:
    room_id: str
    identity_id: str
    created_at: float
    last_seen: float
    embeddings: list[np.ndarray] = field(default_factory=list)
    qualities: list[float] = field(default_factory=list)
    members: dict[tuple[str, str], float] = field(default_factory=dict)
    active: bool = False
    active_since: float | None = None
    inactive_since: float | None = None


class V11RoomIdentityShadowV1:
    """Fuse stable camera identities across different cameras in one room.

    The camera-tracklet layer owns single-camera stitching. This layer is only for
    same-room peer-camera fusion. Association requires simultaneous activity,
    reciprocal best matching and repeated evidence. It remains shadow-only.
    """

    def __init__(
        self,
        *,
        min_track_samples: int = 3,
        decision_samples: int = 6,
        track_gallery_size: int = 6,
        identity_gallery_size: int = 12,
        join_similarity: float = 0.76,
        weak_hold_similarity: float = 0.55,
        min_margin: float = 0.04,
        ttl_sec: float = 45.0,
        cross_camera_only: bool = True,
        require_active_overlap: bool = True,
        candidate_votes: int = 2,
        reciprocal_join_similarity: float = 0.68,
        max_pair_wait_sec: float = 8.0,
    ) -> None:
        self.min_track_samples = max(2, min(6, int(min_track_samples)))
        self.decision_samples = max(self.min_track_samples, min(10, int(decision_samples)))
        self.track_gallery_size = max(self.decision_samples, min(12, int(track_gallery_size)))
        self.identity_gallery_size = max(4, min(24, int(identity_gallery_size)))
        self.join_similarity = float(join_similarity)
        self.reciprocal_join_similarity = min(
            self.join_similarity, float(reciprocal_join_similarity)
        )
        self.weak_hold_similarity = min(
            self.reciprocal_join_similarity, float(weak_hold_similarity)
        )
        self.min_margin = max(0.0, float(min_margin))
        self.ttl_sec = max(10.0, float(ttl_sec))
        self.cross_camera_only = bool(cross_camera_only)
        self.require_active_overlap = bool(require_active_overlap)
        self.candidate_votes = max(1, int(candidate_votes))
        self.max_pair_wait_sec = max(2.0, float(max_pair_wait_sec))

        self._lock = threading.RLock()
        self._next_by_room: dict[str, int] = {}
        self._pending: dict[tuple[str, str], _PendingTrack] = {}
        self._assigned: dict[tuple[str, str], str] = {}
        self._identities: dict[str, _RoomIdentity] = {}
        self._active_by_camera: dict[str, set[str]] = {}

        self.observations = 0
        self.created = 0
        self.joined = 0
        self.assigned_updates = 0
        self.pending_match_waits = 0
        self.ambiguous_new = 0
        self.same_camera_collision_rejects = 0
        self.inactive_overlap_rejects = 0
        self.reciprocal_rejects = 0
        self.vote_waits = 0
        self.expired = 0
        self.activations = 0
        self.deactivations = 0

    @staticmethod
    def _gallery_score(left: list[np.ndarray], right: list[np.ndarray]) -> float:
        if not left or not right:
            return -1.0
        left_matrix = np.stack(left, axis=0)
        right_matrix = np.stack(right, axis=0)
        return float(np.max(left_matrix @ right_matrix.T))

    def _identity_is_active(self, identity: _RoomIdentity) -> bool:
        return any(
            track_id in self._active_by_camera.get(camera_id, set())
            for camera_id, track_id in identity.members
        )

    def _refresh_activity(self, now: float) -> None:
        for identity in self._identities.values():
            active_now = self._identity_is_active(identity)
            if active_now and not identity.active:
                identity.active = True
                identity.active_since = now
                identity.inactive_since = None
                self.activations += 1
            elif not active_now and identity.active:
                identity.active = False
                identity.inactive_since = now
                self.deactivations += 1

    def update_active_tracks(
        self,
        *,
        camera_id: str,
        track_ids: set[str] | tuple[str, ...] | list[str],
        captured_at: float | None = None,
    ) -> None:
        now = time.monotonic() if captured_at is None else float(captured_at)
        with self._lock:
            self._active_by_camera[str(camera_id)] = {str(v) for v in track_ids}
            self._refresh_activity(now)

    def activity_snapshot(self, captured_at: float | None = None) -> list[dict[str, object]]:
        now = time.monotonic() if captured_at is None else float(captured_at)
        with self._lock:
            self._refresh_activity(now)
            return [
                {
                    "room_id": row.room_id,
                    "room_identity": row.identity_id,
                    "active": row.active,
                    "active_since": row.active_since,
                    "inactive_since": row.inactive_since,
                }
                for row in self._identities.values()
            ]

    @staticmethod
    def _has_camera_member(identity: _RoomIdentity, camera_id: str) -> bool:
        return any(cam == camera_id for cam, _track in identity.members)

    def _incoming_is_active(self, pending: _PendingTrack) -> bool:
        return pending.track_id in self._active_by_camera.get(pending.camera_id, set())

    def _has_active_peer_camera(self, pending: _PendingTrack, identity: _RoomIdentity) -> bool:
        for camera_id, track_id in identity.members:
            if camera_id != pending.camera_id and track_id in self._active_by_camera.get(camera_id, set()):
                return True
        return False

    def _is_reciprocal_best(
        self,
        pending: _PendingTrack,
        identity: _RoomIdentity,
        score: float,
    ) -> bool:
        best_score = -1.0
        best_key: tuple[str, str] | None = None
        for other in self._pending.values():
            if other.room_id != pending.room_id or other.camera_id != pending.camera_id:
                continue
            if len(other.embeddings) < self.min_track_samples or not self._incoming_is_active(other):
                continue
            other_score = self._gallery_score(other.embeddings, identity.embeddings)
            if other_score > best_score:
                best_score = other_score
                best_key = other.key
        return best_key == pending.key and score >= best_score - 1e-6

    def _new_identity(self, pending: _PendingTrack, now: float) -> _RoomIdentity:
        number = self._next_by_room.get(pending.room_id, 1)
        self._next_by_room[pending.room_id] = number + 1
        identity = _RoomIdentity(
            room_id=pending.room_id,
            identity_id=f"{_room_slug(pending.room_id)}-R{number:04d}",
            created_at=now,
            last_seen=now,
            embeddings=list(pending.embeddings),
            qualities=list(pending.qualities),
            members={pending.key: now},
            active=self._incoming_is_active(pending),
        )
        identity.active_since = now if identity.active else None
        identity.inactive_since = None if identity.active else now
        self._identities[identity.identity_id] = identity
        self._assigned[pending.key] = identity.identity_id
        self._pending.pop(pending.key, None)
        self.created += 1
        if identity.active:
            self.activations += 1
        return identity

    def _append_identity(
        self,
        identity: _RoomIdentity,
        *,
        key: tuple[str, str],
        embedding: np.ndarray,
        quality: float,
        now: float,
    ) -> None:
        identity.last_seen = max(identity.last_seen, now)
        identity.members[key] = now
        identity.embeddings.append(embedding)
        identity.qualities.append(quality)
        if len(identity.embeddings) > self.identity_gallery_size:
            rows = list(zip(identity.embeddings, identity.qualities))
            rows = rows[-self.identity_gallery_size :]
            identity.embeddings = [row[0] for row in rows]
            identity.qualities = [row[1] for row in rows]

    def _expire(self, now: float) -> None:
        self._refresh_activity(now)
        for identity_id in [
            key
            for key, row in self._identities.items()
            if not row.active and now - row.last_seen > self.ttl_sec
        ]:
            identity = self._identities.pop(identity_id)
            for member in identity.members:
                if self._assigned.get(member) == identity_id:
                    self._assigned.pop(member, None)
            self.expired += 1
        for key in [
            key for key, row in self._pending.items() if now - row.last_seen > self.ttl_sec
        ]:
            self._pending.pop(key, None)

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
        camera_id, track_id, room_id = str(camera_id), str(track_id), str(room_id)
        key = (camera_id, track_id)
        vector = _normalize(embedding)
        quality = max(0.05, min(1.0, float(quality)))

        def result(
            state: str,
            room_identity: str = "",
            score: float = 0.0,
            margin: float = 0.0,
            samples: int = 0,
            members: int = 0,
            collision_rejects: int = 0,
            inactive_overlap_rejects: int = 0,
            reciprocal: bool = False,
            votes: int = 0,
        ) -> dict[str, object]:
            return {
                "state": state,
                "room_identity": room_identity,
                "score": score,
                "margin": margin,
                "samples": samples,
                "members": members,
                "collision_rejects": collision_rejects,
                "inactive_overlap_rejects": inactive_overlap_rejects,
                "reciprocal": reciprocal,
                "votes": votes,
            }

        with self._lock:
            self._expire(now)
            self.observations += 1

            assigned_id = self._assigned.get(key)
            if assigned_id and assigned_id in self._identities:
                identity = self._identities[assigned_id]
                self._append_identity(identity, key=key, embedding=vector, quality=quality, now=now)
                self._refresh_activity(now)
                self.assigned_updates += 1
                return result(
                    "EXISTING", identity.identity_id, 1.0, 1.0,
                    len(identity.embeddings), len(identity.members), reciprocal=True,
                    votes=self.candidate_votes,
                )

            pending = self._pending.get(key)
            if pending is None:
                pending = _PendingTrack(camera_id, track_id, room_id, now, now)
                self._pending[key] = pending
            pending.last_seen = max(pending.last_seen, now)
            pending.embeddings.append(vector)
            pending.qualities.append(quality)
            if len(pending.embeddings) > self.track_gallery_size:
                pending.embeddings = pending.embeddings[-self.track_gallery_size :]
                pending.qualities = pending.qualities[-self.track_gallery_size :]

            if len(pending.embeddings) < self.min_track_samples:
                return result("PENDING", samples=len(pending.embeddings))

            incoming_active = self._incoming_is_active(pending)
            ranked: list[tuple[float, _RoomIdentity]] = []
            collision_rejects = 0
            inactive_overlap_rejects = 0
            for identity in self._identities.values():
                if identity.room_id != room_id:
                    continue
                # single-camera stitching belongs to the camera tracklet layer.
                if self.cross_camera_only and self._has_camera_member(identity, camera_id):
                    collision_rejects += 1
                    continue
                if self.require_active_overlap and (
                    not incoming_active or not self._has_active_peer_camera(pending, identity)
                ):
                    inactive_overlap_rejects += 1
                    continue
                ranked.append((self._gallery_score(pending.embeddings, identity.embeddings), identity))
            ranked.sort(key=lambda row: row[0], reverse=True)
            self.same_camera_collision_rejects += collision_rejects
            self.inactive_overlap_rejects += inactive_overlap_rejects

            best_score = ranked[0][0] if ranked else -1.0
            runner_score = ranked[1][0] if len(ranked) > 1 else -1.0
            margin = best_score - runner_score if ranked else 0.0
            reciprocal = False

            if ranked:
                best_identity = ranked[0][1]
                reciprocal = self._is_reciprocal_best(pending, best_identity, best_score)
                qualifies = best_score >= self.reciprocal_join_similarity and reciprocal
                if pending.best_identity_id == best_identity.identity_id:
                    if qualifies:
                        pending.best_identity_streak += 1
                else:
                    pending.best_identity_id = best_identity.identity_id
                    pending.best_identity_streak = 1 if qualifies else 0

                if best_score >= self.reciprocal_join_similarity and not reciprocal:
                    self.reciprocal_rejects += 1

                if qualifies and pending.best_identity_streak >= self.candidate_votes:
                    self._assigned[key] = best_identity.identity_id
                    for sample, sample_quality in zip(pending.embeddings, pending.qualities):
                        self._append_identity(
                            best_identity,
                            key=key,
                            embedding=sample,
                            quality=sample_quality,
                            now=now,
                        )
                    self._pending.pop(key, None)
                    self._refresh_activity(now)
                    self.joined += 1
                    return result(
                        "JOIN", best_identity.identity_id, best_score, margin,
                        len(best_identity.embeddings), len(best_identity.members),
                        collision_rejects, inactive_overlap_rejects, True,
                        pending.best_identity_streak,
                    )

                if qualifies:
                    self.vote_waits += 1
                    self.pending_match_waits += 1
                    return result(
                        "PENDING_MATCH", score=best_score, margin=margin,
                        samples=len(pending.embeddings), collision_rejects=collision_rejects,
                        inactive_overlap_rejects=inactive_overlap_rejects,
                        reciprocal=True, votes=pending.best_identity_streak,
                    )

            pair_age = max(0.0, now - pending.first_seen)
            if ranked and pair_age < self.max_pair_wait_sec and (
                len(pending.embeddings) < self.decision_samples
                or best_score >= self.weak_hold_similarity
            ):
                self.pending_match_waits += 1
                return result(
                    "PENDING_MATCH", score=best_score, margin=margin,
                    samples=len(pending.embeddings), collision_rejects=collision_rejects,
                    inactive_overlap_rejects=inactive_overlap_rejects,
                    reciprocal=reciprocal, votes=pending.best_identity_streak,
                )

            ambiguous = bool(ranked and best_score >= self.reciprocal_join_similarity)
            identity = self._new_identity(pending, now)
            if ambiguous:
                self.ambiguous_new += 1
            return result(
                "AMBIGUOUS_NEW" if ambiguous else "NEW",
                identity.identity_id,
                best_score,
                margin,
                len(identity.embeddings),
                len(identity.members),
                collision_rejects,
                inactive_overlap_rejects,
                reciprocal,
                pending.best_identity_streak,
            )

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            self._refresh_activity(time.monotonic())
            return {
                "room_identities": len(self._identities),
                "assigned_tracks": len(self._assigned),
                "pending_tracks": len(self._pending),
                "observations": self.observations,
                "created": self.created,
                "joined": self.joined,
                "assigned_updates": self.assigned_updates,
                "pending_match_waits": self.pending_match_waits,
                "ambiguous_new": self.ambiguous_new,
                "collision_rejects": self.same_camera_collision_rejects,
                "inactive_overlap_rejects": self.inactive_overlap_rejects,
                "reciprocal_rejects": self.reciprocal_rejects,
                "vote_waits": self.vote_waits,
                "expired": self.expired,
                "activations": self.activations,
                "deactivations": self.deactivations,
            }
