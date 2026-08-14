from __future__ import annotations

from dataclasses import dataclass, field
import re
import threading
import time

import numpy as np


def _norm(vector):
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(arr))
    return arr / max(n, 1e-12)


def _cosine(a, b) -> float:
    return float(np.dot(_norm(a), _norm(b)))


def _gid_number(value: str) -> int:
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else 10**9


@dataclass
class GlobalIdentity:
    global_id: str
    prototype: np.ndarray
    observations: int = 1
    last_seen: float = 0.0
    last_camera: str = ""
    active_tracks: dict[str, str] = field(default_factory=dict)
    gallery: list[np.ndarray] = field(default_factory=list)


class GlobalIdentityManager:
    """Mergeable Global-ID graph driven by stable local tracklets.

    A local track receives a provisional Global ID once its descriptor is mature.
    Same-room pair matching may then merge two provisional identities. The oldest
    ID survives so UI labels do not oscillate. Same-camera duplicates and active
    tracks in different rooms are hard merge conflicts.
    """

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.prototype_alpha = max(0.01, min(1.0, float(cfg.get("prototype_alpha", 0.18))))
        self.gallery_size = max(2, int(cfg.get("gallery_size", 12)))
        self.active_timeout_sec = max(0.5, float(cfg.get("active_timeout_sec", 6.0)))
        self.camera_rooms = {
            str(camera_id): str(room_id)
            for camera_id, room_id in (cfg.get("camera_rooms") or {}).items()
        }
        self.block_concurrent_cross_room = bool(cfg.get("block_concurrent_cross_room", True))

        self._lock = threading.RLock()
        self._identities: dict[str, GlobalIdentity] = {}
        self._track_to_global: dict[str, str] = {}
        self._track_activity: dict[str, float] = {}
        self._counter = 1
        self._created = 0
        self._merges = 0
        self._merge_conflicts = 0
        self._updates = 0
        self._released = 0
        self._expired = 0
        self._merge_scores: list[float] = []

    @staticmethod
    def local_key(camera_id: str, local_track_id: int) -> str:
        return f"{str(camera_id)}:{int(local_track_id)}"

    @staticmethod
    def _camera_from_key(local_key: str) -> str:
        return str(local_key).split(":", 1)[0]

    def _new_identity_locked(self, camera_id: str, local_key: str, descriptor, observed_at: float):
        gid = f"Unknown_{self._counter:03d}"
        self._counter += 1
        vector = _norm(descriptor)
        identity = GlobalIdentity(
            global_id=gid,
            prototype=vector.copy(),
            observations=1,
            last_seen=float(observed_at),
            last_camera=str(camera_id),
            active_tracks={str(camera_id): str(local_key)},
            gallery=[vector.copy()],
        )
        self._identities[gid] = identity
        self._track_to_global[local_key] = gid
        self._track_activity[local_key] = float(observed_at)
        self._created += 1
        return gid

    def _unlink_locked(self, local_key: str) -> None:
        gid = self._track_to_global.pop(local_key, None)
        self._track_activity.pop(local_key, None)
        if gid is None:
            return
        identity = self._identities.get(gid)
        if identity is None:
            return
        for camera_id, key in list(identity.active_tracks.items()):
            if key == local_key:
                identity.active_tracks.pop(camera_id, None)

    def _expire_locked(self, now: float) -> None:
        for local_key, last_active in list(self._track_activity.items()):
            if float(now) - float(last_active) > self.active_timeout_sec:
                self._unlink_locked(local_key)
                self._expired += 1

    def _update_identity_locked(self, identity: GlobalIdentity, camera_id: str, local_key: str, descriptor, observed_at: float):
        vector = _norm(descriptor)
        alpha = self.prototype_alpha
        identity.prototype = _norm((1.0 - alpha) * identity.prototype + alpha * vector)
        identity.gallery.append(vector.copy())
        if len(identity.gallery) > self.gallery_size:
            del identity.gallery[:-self.gallery_size]
        identity.observations += 1
        identity.last_seen = max(float(identity.last_seen), float(observed_at))
        identity.last_camera = str(camera_id)
        identity.active_tracks[str(camera_id)] = str(local_key)
        self._track_activity[str(local_key)] = max(
            float(self._track_activity.get(str(local_key), 0.0)), float(observed_at)
        )
        self._updates += 1

    def ensure_track(self, camera_id: str, local_track_id: int, descriptor, observed_at: float | None = None):
        camera_id = str(camera_id)
        local_key = self.local_key(camera_id, local_track_id)
        observed_at = float(observed_at if observed_at is not None else time.monotonic())
        with self._lock:
            self._expire_locked(observed_at)
            gid = self._track_to_global.get(local_key)
            if gid is None or gid not in self._identities:
                gid = self._new_identity_locked(camera_id, local_key, descriptor, observed_at)
                return gid, "new"
            self._update_identity_locked(self._identities[gid], camera_id, local_key, descriptor, observed_at)
            return gid, "existing"

    def touch_track(self, camera_id: str, local_track_id: int, observed_at: float | None = None):
        camera_id = str(camera_id)
        local_key = self.local_key(camera_id, local_track_id)
        observed_at = float(observed_at if observed_at is not None else time.monotonic())
        with self._lock:
            self._expire_locked(observed_at)
            gid = self._track_to_global.get(local_key)
            identity = self._identities.get(gid) if gid else None
            if identity is None:
                return None
            self._track_activity[local_key] = max(
                float(self._track_activity.get(local_key, 0.0)), observed_at
            )
            identity.active_tracks[camera_id] = local_key
            identity.last_seen = max(float(identity.last_seen), observed_at)
            identity.last_camera = camera_id
            return gid

    def release_track(self, camera_id: str, local_track_id: int) -> None:
        local_key = self.local_key(camera_id, local_track_id)
        with self._lock:
            existed = local_key in self._track_to_global
            self._unlink_locked(local_key)
            if existed:
                self._released += 1

    def lookup_track(self, camera_id: str, local_track_id: int):
        key = self.local_key(camera_id, local_track_id)
        with self._lock:
            return self._track_to_global.get(key)

    def _merge_conflict_locked(self, left: GlobalIdentity, right: GlobalIdentity) -> bool:
        left_keys = set(left.active_tracks.values())
        right_keys = set(right.active_tracks.values())
        if left_keys & right_keys:
            return False

        # One identity cannot represent two different active local tracks from
        # the same camera.
        for camera_id in set(left.active_tracks) & set(right.active_tracks):
            if left.active_tracks[camera_id] != right.active_tracks[camera_id]:
                return True

        if self.block_concurrent_cross_room and self.camera_rooms:
            rooms = {
                self.camera_rooms.get(camera_id)
                for camera_id in list(left.active_tracks) + list(right.active_tracks)
            }
            rooms.discard(None)
            if len(rooms) > 1:
                return True
        return False

    def merge_tracks(self, left_camera: str, left_track: int, right_camera: str, right_track: int, score: float, observed_at: float | None = None):
        observed_at = float(observed_at if observed_at is not None else time.monotonic())
        left_key = self.local_key(left_camera, left_track)
        right_key = self.local_key(right_camera, right_track)
        with self._lock:
            self._expire_locked(observed_at)
            left_gid = self._track_to_global.get(left_key)
            right_gid = self._track_to_global.get(right_key)
            if not left_gid or not right_gid:
                return None, "missing_binding"
            if left_gid == right_gid:
                return left_gid, "already_merged"

            left = self._identities.get(left_gid)
            right = self._identities.get(right_gid)
            if left is None or right is None:
                return None, "missing_identity"
            if self._merge_conflict_locked(left, right):
                self._merge_conflicts += 1
                return None, "conflict"

            # Preserve the oldest numeric Unknown ID for stable labels.
            if _gid_number(right_gid) < _gid_number(left_gid):
                left_gid, right_gid = right_gid, left_gid
                left, right = right, left

            total = max(1, left.observations + right.observations)
            left_weight = left.observations / total
            right_weight = right.observations / total
            left.prototype = _norm(left_weight * left.prototype + right_weight * right.prototype)
            combined_gallery = left.gallery + right.gallery
            combined_gallery = combined_gallery[-self.gallery_size :]
            left.gallery = [item.copy() for item in combined_gallery]
            left.observations += right.observations
            left.last_seen = max(left.last_seen, right.last_seen, observed_at)
            if right.last_seen >= left.last_seen:
                left.last_camera = right.last_camera

            for camera_id, local_key in right.active_tracks.items():
                left.active_tracks[camera_id] = local_key
            for local_key, gid in list(self._track_to_global.items()):
                if gid == right_gid:
                    self._track_to_global[local_key] = left_gid

            self._identities.pop(right_gid, None)
            self._merges += 1
            self._merge_scores.append(float(score))
            if len(self._merge_scores) > 256:
                del self._merge_scores[:-256]
            return left_gid, "merged"

    def identity_similarity(self, gid: str, descriptor) -> float | None:
        vector = _norm(descriptor)
        with self._lock:
            identity = self._identities.get(str(gid))
            if identity is None:
                return None
            prototype_score = _cosine(vector, identity.prototype)
            gallery_scores = sorted(
                (_cosine(vector, item) for item in identity.gallery), reverse=True
            )
            top = gallery_scores[: min(3, len(gallery_scores))]
            consensus = sum(top) / len(top) if top else prototype_score
            return float(0.45 * prototype_score + 0.55 * consensus)

    def snapshot(self):
        with self._lock:
            return {
                gid: {
                    "observations": identity.observations,
                    "last_seen": identity.last_seen,
                    "last_camera": identity.last_camera,
                    "active_tracks": dict(identity.active_tracks),
                    "active_rooms": sorted({
                        self.camera_rooms.get(camera_id, "unknown")
                        for camera_id in identity.active_tracks
                    }),
                    "gallery_size": len(identity.gallery),
                }
                for gid, identity in sorted(self._identities.items())
            }

    def metrics(self):
        with self._lock:
            scores = list(self._merge_scores)
            return {
                "algorithm": "tracklet-identity-graph-v2",
                "global_identities": len(self._identities),
                "active_local_tracks": len(self._track_to_global),
                "created": self._created,
                "merges": self._merges,
                "merge_conflicts": self._merge_conflicts,
                "updates": self._updates,
                "released": self._released,
                "expired": self._expired,
                "camera_rooms": dict(self.camera_rooms),
                "merge_score_last": scores[-1] if scores else None,
                "merge_score_min": min(scores) if scores else None,
                "merge_score_max": max(scores) if scores else None,
            }
