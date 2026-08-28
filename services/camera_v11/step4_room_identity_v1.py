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
    last_seen: float
    embeddings: list[np.ndarray] = field(default_factory=list)
    qualities: list[float] = field(default_factory=list)

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
    """Group fragmented local tracks into conservative per-room shadow identities.

    This layer never rewrites Step3 local IDs. A local track first builds a ReID
    gallery. Borderline cross-camera evidence is allowed to mature for several
    samples before opening another room identity, while simultaneous same-camera
    tracks are always kept separate. Activity state is exposed for handoff lifecycle
    gating but does not change the frozen tracker.
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
    ) -> None:
        self.min_track_samples = max(2, min(6, int(min_track_samples)))
        self.decision_samples = max(
            self.min_track_samples, min(10, int(decision_samples))
        )
        self.track_gallery_size = max(
            self.decision_samples, min(12, int(track_gallery_size))
        )
        self.identity_gallery_size = max(4, min(24, int(identity_gallery_size)))
        self.join_similarity = float(join_similarity)
        self.weak_hold_similarity = min(self.join_similarity, float(weak_hold_similarity))
        self.min_margin = max(0.0, float(min_margin))
        self.ttl_sec = max(10.0, float(ttl_sec))
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
        self.expired = 0
        self.activations = 0
        self.deactivations = 0

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
            self._active_by_camera[str(camera_id)] = {str(track_id) for track_id in track_ids}
            self._refresh_activity(now)

    def activity_snapshot(self, captured_at: float | None = None) -> list[dict[str, object]]:
        now = time.monotonic() if captured_at is None else float(captured_at)
        with self._lock:
            self._refresh_activity(now)
            return [
                {
                    "room_id": identity.room_id,
                    "room_identity": identity.identity_id,
                    "active": identity.active,
                    "active_since": identity.active_since,
                    "inactive_since": identity.inactive_since,
                }
                for identity in self._identities.values()
            ]

    @staticmethod
    def _gallery_score(left: list[np.ndarray], right: list[np.ndarray]) -> float:
        if not left or not right:
            return -1.0
        left_matrix = np.stack(left, axis=0)
        right_matrix = np.stack(right, axis=0)
        return float(np.max(left_matrix @ right_matrix.T))

    def _same_camera_collision(self, pending: _PendingTrack, identity: _RoomIdentity) -> bool:
        active = self._active_by_camera.get(pending.camera_id, set())
        for camera_id, track_id in identity.members:
            if camera_id != pending.camera_id or track_id == pending.track_id:
                continue
            if track_id in active:
                return True
        return False

    def _new_identity(self, pending: _PendingTrack, now: float) -> _RoomIdentity:
        room = pending.room_id
        number = self._next_by_room.get(room, 1)
        self._next_by_room[room] = number + 1
        identity_id = f"{_room_slug(room)}-R{number:04d}"
        active = pending.track_id in self._active_by_camera.get(pending.camera_id, set())
        identity = _RoomIdentity(
            room_id=room,
            identity_id=identity_id,
            created_at=now,
            last_seen=now,
            embeddings=list(pending.embeddings),
            qualities=list(pending.qualities),
            members={pending.key: now},
            active=active,
            active_since=now if active else None,
            inactive_since=None if active else now,
        )
        self._identities[identity_id] = identity
        self._assigned[pending.key] = identity_id
        self._pending.pop(pending.key, None)
        self.created += 1
        if active:
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
            recent = rows[-max(2, self.identity_gallery_size // 2) :]
            older = rows[: -len(recent)] if len(rows) > len(recent) else []
            older.sort(key=lambda row: row[1], reverse=True)
            kept = older[: self.identity_gallery_size - len(recent)] + recent
            identity.embeddings = [row[0] for row in kept]
            identity.qualities = [row[1] for row in kept]

    def _expire(self, now: float) -> None:
        self._refresh_activity(now)
        stale_ids: list[str] = []
        for identity_id, identity in self._identities.items():
            if now - identity.last_seen <= self.ttl_sec or identity.active:
                continue
            stale_ids.append(identity_id)
        for identity_id in stale_ids:
            identity = self._identities.pop(identity_id)
            for key in list(identity.members):
                if self._assigned.get(key) == identity_id:
                    self._assigned.pop(key, None)
            self.expired += 1

        stale_pending = [
            key for key, pending in self._pending.items() if now - pending.last_seen > self.ttl_sec
        ]
        for key in stale_pending:
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
        camera_id = str(camera_id)
        track_id = str(track_id)
        room_id = str(room_id)
        key = (camera_id, track_id)
        vector = _normalize(embedding)
        quality = max(0.05, min(1.0, float(quality)))

        with self._lock:
            self._expire(now)
            self.observations += 1

            assigned_id = self._assigned.get(key)
            if assigned_id is not None:
                identity = self._identities.get(assigned_id)
                if identity is not None:
                    self._append_identity(
                        identity,
                        key=key,
                        embedding=vector,
                        quality=quality,
                        now=now,
                    )
                    self._refresh_activity(now)
                    self.assigned_updates += 1
                    return {
                        "state": "EXISTING",
                        "room_identity": identity.identity_id,
                        "score": 1.0,
                        "margin": 1.0,
                        "samples": len(identity.embeddings),
                        "members": len(identity.members),
                        "collision_rejects": 0,
                    }
                self._assigned.pop(key, None)

            pending = self._pending.get(key)
            if pending is None:
                pending = _PendingTrack(camera_id, track_id, room_id, now)
                self._pending[key] = pending
            pending.last_seen = max(pending.last_seen, now)
            pending.room_id = room_id
            pending.embeddings.append(vector)
            pending.qualities.append(quality)
            if len(pending.embeddings) > self.track_gallery_size:
                pending.embeddings = pending.embeddings[-self.track_gallery_size :]
                pending.qualities = pending.qualities[-self.track_gallery_size :]

            if len(pending.embeddings) < self.min_track_samples:
                return {
                    "state": "PENDING",
                    "room_identity": "",
                    "score": 0.0,
                    "margin": 0.0,
                    "samples": len(pending.embeddings),
                    "members": 0,
                    "collision_rejects": 0,
                }

            ranked: list[tuple[float, _RoomIdentity]] = []
            collision_rejects = 0
            for identity in self._identities.values():
                if identity.room_id != room_id:
                    continue
                if self._same_camera_collision(pending, identity):
                    collision_rejects += 1
                    continue
                score = self._gallery_score(pending.embeddings, identity.embeddings)
                ranked.append((score, identity))
            ranked.sort(key=lambda row: row[0], reverse=True)
            self.same_camera_collision_rejects += collision_rejects

            best_score = ranked[0][0] if ranked else -1.0
            runner_score = ranked[1][0] if len(ranked) > 1 else -1.0
            margin = best_score - runner_score if ranked else 0.0

            if ranked and best_score >= self.join_similarity and margin >= self.min_margin:
                identity = ranked[0][1]
                self._assigned[key] = identity.identity_id
                for sample, sample_quality in zip(pending.embeddings, pending.qualities):
                    self._append_identity(
                        identity,
                        key=key,
                        embedding=sample,
                        quality=sample_quality,
                        now=now,
                    )
                self._pending.pop(key, None)
                self._refresh_activity(now)
                self.joined += 1
                return {
                    "state": "JOIN",
                    "room_identity": identity.identity_id,
                    "score": best_score,
                    "margin": margin,
                    "samples": len(identity.embeddings),
                    "members": len(identity.members),
                    "collision_rejects": collision_rejects,
                }

            # Do not permanently split a cross-camera track on only three samples.
            # Borderline evidence often improves as viewpoint/pose changes. Same-camera
            # collisions are intentionally not held because simultaneous tracks from
            # one camera represent different physical people.
            should_hold = bool(
                ranked
                and collision_rejects == 0
                and best_score >= self.weak_hold_similarity
                and len(pending.embeddings) < self.decision_samples
            )
            if should_hold:
                self.pending_match_waits += 1
                return {
                    "state": "PENDING_MATCH",
                    "room_identity": "",
                    "score": best_score,
                    "margin": margin,
                    "samples": len(pending.embeddings),
                    "members": 0,
                    "collision_rejects": collision_rejects,
                }

            ambiguous = bool(ranked and best_score >= self.join_similarity)
            identity = self._new_identity(pending, now)
            if ambiguous:
                self.ambiguous_new += 1
            return {
                "state": "AMBIGUOUS_NEW" if ambiguous else "NEW",
                "room_identity": identity.identity_id,
                "score": best_score,
                "margin": margin,
                "samples": len(identity.embeddings),
                "members": len(identity.members),
                "collision_rejects": collision_rejects,
            }

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
                "expired": self.expired,
                "activations": self.activations,
                "deactivations": self.deactivations,
            }
