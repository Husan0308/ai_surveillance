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


def _bbox_center_distance(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    lcx = 0.5 * (lx1 + lx2)
    lcy = 0.5 * (ly1 + ly2)
    rcx = 0.5 * (rx1 + rx2)
    rcy = 0.5 * (ry1 + ry2)
    scale = max(20.0, 0.5 * (max(1.0, ly2 - ly1) + max(1.0, ry2 - ry1)))
    return float(math.hypot(lcx - rcx, lcy - rcy) / scale)


@dataclass
class _PendingLocalTrack:
    camera_id: str
    track_id: str
    room_id: str
    first_seen: float
    last_seen: float
    embeddings: list[np.ndarray] = field(default_factory=list)
    qualities: list[float] = field(default_factory=list)
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return self.camera_id, self.track_id


@dataclass
class _CameraTrackletIdentity:
    camera_id: str
    identity_id: str
    created_at: float
    last_seen: float
    embeddings: list[np.ndarray] = field(default_factory=list)
    qualities: list[float] = field(default_factory=list)
    members: dict[str, float] = field(default_factory=dict)
    last_bbox: tuple[float, float, float, float] | None = None
    active: bool = False
    active_since: float | None = None
    inactive_since: float | None = None


class V11CameraTrackletShadowV1:
    """Stitch Step3 local-ID fragments inside one physical camera.

    NVIDIA target re-association separates single-view re-association from peer-camera
    re-association. This shadow layer does the same: a newly-created Step3 local track
    may adopt a recently-lost camera-tracklet identity only when appearance and short
    temporal/spatial continuity agree. It never mutates the frozen Step3 tracker.
    """

    def __init__(
        self,
        *,
        min_track_samples: int = 3,
        decision_samples: int = 5,
        track_gallery_size: int = 6,
        identity_gallery_size: int = 12,
        join_similarity: float = 0.80,
        high_similarity: float = 0.90,
        hold_similarity: float = 0.65,
        min_margin: float = 0.04,
        max_reassoc_gap_sec: float = 6.0,
        max_center_distance: float = 4.0,
        ttl_sec: float = 45.0,
    ) -> None:
        self.min_track_samples = max(2, min(6, int(min_track_samples)))
        self.decision_samples = max(self.min_track_samples, min(10, int(decision_samples)))
        self.track_gallery_size = max(self.decision_samples, min(12, int(track_gallery_size)))
        self.identity_gallery_size = max(4, min(24, int(identity_gallery_size)))
        self.join_similarity = float(join_similarity)
        self.high_similarity = max(self.join_similarity, float(high_similarity))
        self.hold_similarity = min(self.join_similarity, float(hold_similarity))
        self.min_margin = max(0.0, float(min_margin))
        self.max_reassoc_gap_sec = max(0.5, float(max_reassoc_gap_sec))
        self.max_center_distance = max(0.5, float(max_center_distance))
        self.ttl_sec = max(self.max_reassoc_gap_sec, float(ttl_sec))

        self._lock = threading.RLock()
        self._next_by_camera: dict[str, int] = {}
        self._pending: dict[tuple[str, str], _PendingLocalTrack] = {}
        self._assigned: dict[tuple[str, str], str] = {}
        self._identities: dict[str, _CameraTrackletIdentity] = {}
        self._active_local_by_camera: dict[str, set[str]] = {}

        self.observations = 0
        self.created = 0
        self.joined = 0
        self.assigned_updates = 0
        self.pending_match_waits = 0
        self.active_rejects = 0
        self.gap_rejects = 0
        self.spatial_rejects = 0
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

    def _identity_is_active(self, identity: _CameraTrackletIdentity) -> bool:
        active = self._active_local_by_camera.get(identity.camera_id, set())
        return any(track_id in active for track_id in identity.members)

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
            self._active_local_by_camera[str(camera_id)] = {str(track_id) for track_id in track_ids}
            self._refresh_activity(now)

    def activity_snapshot(self, captured_at: float | None = None) -> list[dict[str, object]]:
        now = time.monotonic() if captured_at is None else float(captured_at)
        with self._lock:
            self._refresh_activity(now)
            return [
                {
                    "camera_id": identity.camera_id,
                    "camera_identity": identity.identity_id,
                    "active": identity.active,
                    "active_since": identity.active_since,
                    "inactive_since": identity.inactive_since,
                }
                for identity in self._identities.values()
            ]

    def _new_identity(self, pending: _PendingLocalTrack, now: float) -> _CameraTrackletIdentity:
        number = self._next_by_camera.get(pending.camera_id, 1)
        self._next_by_camera[pending.camera_id] = number + 1
        identity_id = f"{pending.camera_id}-C{number:04d}"
        active = pending.track_id in self._active_local_by_camera.get(pending.camera_id, set())
        identity = _CameraTrackletIdentity(
            camera_id=pending.camera_id,
            identity_id=identity_id,
            created_at=now,
            last_seen=now,
            embeddings=list(pending.embeddings),
            qualities=list(pending.qualities),
            members={pending.track_id: now},
            last_bbox=pending.bboxes[-1] if pending.bboxes else None,
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
        identity: _CameraTrackletIdentity,
        *,
        local_track_id: str,
        embedding: np.ndarray,
        quality: float,
        bbox: tuple[float, float, float, float],
        now: float,
    ) -> None:
        identity.last_seen = max(identity.last_seen, now)
        identity.members[local_track_id] = now
        identity.last_bbox = bbox
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
        stale_ids = [
            identity_id
            for identity_id, identity in self._identities.items()
            if not identity.active and now - identity.last_seen > self.ttl_sec
        ]
        for identity_id in stale_ids:
            identity = self._identities.pop(identity_id)
            for local_track_id in list(identity.members):
                key = (identity.camera_id, local_track_id)
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
        bbox_xyxy: tuple[float, float, float, float],
        captured_at: float | None = None,
    ) -> dict[str, object]:
        now = time.monotonic() if captured_at is None else float(captured_at)
        camera_id = str(camera_id)
        track_id = str(track_id)
        room_id = str(room_id)
        key = (camera_id, track_id)
        vector = _normalize(embedding)
        quality = max(0.05, min(1.0, float(quality)))
        bbox = tuple(float(v) for v in bbox_xyxy)

        with self._lock:
            self._expire(now)
            self.observations += 1

            assigned_id = self._assigned.get(key)
            if assigned_id is not None:
                identity = self._identities.get(assigned_id)
                if identity is not None:
                    self._append_identity(
                        identity,
                        local_track_id=track_id,
                        embedding=vector,
                        quality=quality,
                        bbox=bbox,
                        now=now,
                    )
                    self._refresh_activity(now)
                    self.assigned_updates += 1
                    return {
                        "state": "EXISTING",
                        "camera_identity": identity.identity_id,
                        "score": 1.0,
                        "margin": 1.0,
                        "gap_sec": 0.0,
                        "center_distance": 0.0,
                        "samples": len(identity.embeddings),
                        "members": len(identity.members),
                    }
                self._assigned.pop(key, None)

            pending = self._pending.get(key)
            if pending is None:
                pending = _PendingLocalTrack(camera_id, track_id, room_id, now, now)
                self._pending[key] = pending
            pending.last_seen = max(pending.last_seen, now)
            pending.room_id = room_id
            pending.embeddings.append(vector)
            pending.qualities.append(quality)
            pending.bboxes.append(bbox)
            if len(pending.embeddings) > self.track_gallery_size:
                pending.embeddings = pending.embeddings[-self.track_gallery_size :]
                pending.qualities = pending.qualities[-self.track_gallery_size :]
                pending.bboxes = pending.bboxes[-self.track_gallery_size :]

            if len(pending.embeddings) < self.min_track_samples:
                return {
                    "state": "PENDING",
                    "camera_identity": "",
                    "score": 0.0,
                    "margin": 0.0,
                    "gap_sec": 0.0,
                    "center_distance": 0.0,
                    "samples": len(pending.embeddings),
                    "members": 0,
                }

            ranked: list[tuple[float, float, float, _CameraTrackletIdentity]] = []
            for identity in self._identities.values():
                if identity.camera_id != camera_id:
                    continue
                if identity.active:
                    self.active_rejects += 1
                    continue
                gap_sec = max(0.0, now - identity.last_seen)
                if gap_sec > self.max_reassoc_gap_sec:
                    self.gap_rejects += 1
                    continue
                center_distance = (
                    _bbox_center_distance(bbox, identity.last_bbox)
                    if identity.last_bbox is not None
                    else 0.0
                )
                score = self._gallery_score(pending.embeddings, identity.embeddings)
                ranked.append((score, gap_sec, center_distance, identity))
            ranked.sort(key=lambda row: row[0], reverse=True)

            best_score = ranked[0][0] if ranked else -1.0
            runner_score = ranked[1][0] if len(ranked) > 1 else -1.0
            margin = best_score - runner_score if ranked else 0.0
            gap_sec = ranked[0][1] if ranked else 0.0
            center_distance = ranked[0][2] if ranked else 0.0

            if ranked:
                strong_appearance = best_score >= self.high_similarity
                spatial_ok = center_distance <= self.max_center_distance
                join_ok = (
                    margin >= self.min_margin
                    and (
                        strong_appearance
                        or (best_score >= self.join_similarity and spatial_ok)
                    )
                )
                if join_ok:
                    identity = ranked[0][3]
                    self._assigned[key] = identity.identity_id
                    for sample, sample_quality, sample_bbox in zip(
                        pending.embeddings, pending.qualities, pending.bboxes
                    ):
                        self._append_identity(
                            identity,
                            local_track_id=track_id,
                            embedding=sample,
                            quality=sample_quality,
                            bbox=sample_bbox,
                            now=now,
                        )
                    self._pending.pop(key, None)
                    self._refresh_activity(now)
                    self.joined += 1
                    return {
                        "state": "JOIN",
                        "camera_identity": identity.identity_id,
                        "score": best_score,
                        "margin": margin,
                        "gap_sec": gap_sec,
                        "center_distance": center_distance,
                        "samples": len(identity.embeddings),
                        "members": len(identity.members),
                    }
                if best_score >= self.join_similarity and center_distance > self.max_center_distance:
                    self.spatial_rejects += 1

            should_hold = bool(
                ranked
                and best_score >= self.hold_similarity
                and len(pending.embeddings) < self.decision_samples
            )
            if should_hold:
                self.pending_match_waits += 1
                return {
                    "state": "PENDING_MATCH",
                    "camera_identity": "",
                    "score": best_score,
                    "margin": margin,
                    "gap_sec": gap_sec,
                    "center_distance": center_distance,
                    "samples": len(pending.embeddings),
                    "members": 0,
                }

            identity = self._new_identity(pending, now)
            return {
                "state": "NEW",
                "camera_identity": identity.identity_id,
                "score": best_score,
                "margin": margin,
                "gap_sec": gap_sec,
                "center_distance": center_distance,
                "samples": len(identity.embeddings),
                "members": len(identity.members),
            }

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            self._refresh_activity(time.monotonic())
            return {
                "camera_identities": len(self._identities),
                "assigned_tracks": len(self._assigned),
                "pending_tracks": len(self._pending),
                "observations": self.observations,
                "created": self.created,
                "joined": self.joined,
                "assigned_updates": self.assigned_updates,
                "pending_match_waits": self.pending_match_waits,
                "active_rejects": self.active_rejects,
                "gap_rejects": self.gap_rejects,
                "spatial_rejects": self.spatial_rejects,
                "expired": self.expired,
                "activations": self.activations,
                "deactivations": self.deactivations,
            }
