from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time

import numpy as np


def _norm(vector):
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(arr))
    return arr / max(n, 1e-12)


def _cosine(a, b) -> float:
    return float(np.dot(_norm(a), _norm(b)))


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
    """Conservative cross-camera identity association.

    Appearance is combined with physical camera-room constraints. Two local
    tracks may share one Global ID concurrently only when they are in cameras
    mapped to the same room (typically overlapping views). Different-room active
    tracks are a hard conflict even if OSNet appearance happens to be similar.

    ReID embeddings update appearance; local track activity independently keeps
    an established binding alive between sparse embedding refreshes.
    """

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.match_threshold = float(cfg.get("match_threshold", 0.78))
        self.strong_threshold = float(cfg.get("strong_threshold", 0.84))
        self.ambiguity_margin = max(0.0, float(cfg.get("ambiguity_margin", 0.045)))
        self.prototype_alpha = max(0.01, min(1.0, float(cfg.get("prototype_alpha", 0.22))))
        self.gallery_size = max(1, int(cfg.get("gallery_size", 8)))
        self.active_timeout_sec = max(0.2, float(cfg.get("active_timeout_sec", 6.0)))
        self.max_transition_sec = max(0.1, float(cfg.get("max_transition_sec", 45.0)))
        self.camera_rooms = {
            str(camera_id): str(room_id)
            for camera_id, room_id in (cfg.get("camera_rooms") or {}).items()
        }
        self.block_concurrent_cross_room = bool(cfg.get("block_concurrent_cross_room", True))
        self.topology = {
            str(src): {
                str(dst): tuple(map(float, bounds))
                for dst, bounds in (targets or {}).items()
            }
            for src, targets in (cfg.get("topology") or {}).items()
        }

        self._lock = threading.Lock()
        self._identities: dict[str, GlobalIdentity] = {}
        self._track_to_global: dict[str, str] = {}
        self._track_activity: dict[str, float] = {}
        self._counter = 1
        self._merges = 0
        self._new = 0
        self._existing_updates = 0
        self._ambiguous = 0
        self._conflicts = 0
        self._room_rejects = 0
        self._released = 0
        self._expired = 0
        self._merge_scores: list[float] = []

    @staticmethod
    def _local_key(camera_id: str, local_track_id: int) -> str:
        return f"{str(camera_id)}:{int(local_track_id)}"

    def _new_identity(self, camera_id: str, local_key: str, embedding, observed_at: float):
        gid = f"Unknown_{self._counter:03d}"
        self._counter += 1
        vector = _norm(embedding)
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
        self._track_to_global[str(local_key)] = gid
        self._track_activity[str(local_key)] = float(observed_at)
        self._new += 1
        return gid, 1.0, "new"

    def _unlink_track_locked(self, local_key: str) -> None:
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

    def _expire_active(self, now: float):
        for local_key, last_active in list(self._track_activity.items()):
            if float(now) - float(last_active) <= self.active_timeout_sec:
                continue
            self._unlink_track_locked(local_key)
            self._expired += 1

    def _transition_allowed(self, identity: GlobalIdentity, camera_id: str, observed_at: float) -> bool:
        if identity.last_camera == camera_id:
            return True
        gap = max(0.0, float(observed_at) - float(identity.last_seen))
        if gap > self.max_transition_sec:
            return True
        if not self.topology:
            return True
        targets = self.topology.get(identity.last_camera)
        if not targets:
            return False
        bounds = targets.get(camera_id)
        if bounds is None:
            return False
        min_sec, max_sec = bounds if len(bounds) >= 2 else (0.0, bounds[0])
        return float(min_sec) <= gap <= float(max_sec)

    def _same_camera_conflict(self, identity: GlobalIdentity, camera_id: str, local_key: str) -> bool:
        active = identity.active_tracks.get(camera_id)
        return active is not None and active != local_key

    def _concurrent_room_conflict(self, identity: GlobalIdentity, camera_id: str) -> bool:
        if not self.block_concurrent_cross_room or not self.camera_rooms:
            return False
        candidate_room = self.camera_rooms.get(str(camera_id))
        if candidate_room is None:
            return False
        for active_camera in identity.active_tracks:
            active_room = self.camera_rooms.get(str(active_camera))
            if active_room is not None and active_room != candidate_room:
                return True
        return False

    @staticmethod
    def _similarity_to_identity(vector, identity: GlobalIdentity) -> float:
        prototype_score = _cosine(vector, identity.prototype)
        gallery_score = max(
            (_cosine(vector, item) for item in identity.gallery),
            default=prototype_score,
        )
        return float(max(prototype_score, gallery_score))

    def touch_track(self, camera_id: str, local_track_id: int, observed_at: float | None = None):
        """Refresh activity for an already-bound local track without ReID work."""
        camera_id = str(camera_id)
        local_key = self._local_key(camera_id, local_track_id)
        observed_at = float(observed_at if observed_at is not None else time.monotonic())
        with self._lock:
            self._expire_active(observed_at)
            gid = self._track_to_global.get(local_key)
            identity = self._identities.get(gid) if gid else None
            if identity is None:
                return None
            self._track_activity[local_key] = observed_at
            identity.active_tracks[camera_id] = local_key
            identity.last_seen = max(float(identity.last_seen), observed_at)
            identity.last_camera = camera_id
            return gid

    def release_track(self, camera_id: str, local_track_id: int) -> None:
        local_key = self._local_key(camera_id, local_track_id)
        with self._lock:
            existed = local_key in self._track_to_global
            self._unlink_track_locked(local_key)
            if existed:
                self._released += 1

    def assign(self, *, camera_id: str, local_track_id: int, embedding, observed_at: float | None = None):
        camera_id = str(camera_id)
        local_key = self._local_key(camera_id, local_track_id)
        observed_at = float(observed_at if observed_at is not None else time.monotonic())
        vector = _norm(embedding)

        with self._lock:
            self._expire_active(observed_at)
            existing_gid = self._track_to_global.get(local_key)
            if existing_gid and existing_gid in self._identities:
                identity = self._identities[existing_gid]
                similarity = self._similarity_to_identity(vector, identity)
                self._update_identity(identity, camera_id, local_key, vector, observed_at)
                self._track_activity[local_key] = observed_at
                self._existing_updates += 1
                return existing_gid, similarity, "existing"

            candidates = []
            for gid, identity in self._identities.items():
                if self._same_camera_conflict(identity, camera_id, local_key):
                    continue
                if self._concurrent_room_conflict(identity, camera_id):
                    self._room_rejects += 1
                    continue
                if not self._transition_allowed(identity, camera_id, observed_at):
                    continue
                score = self._similarity_to_identity(vector, identity)
                candidates.append((score, gid, identity))

            candidates.sort(key=lambda item: item[0], reverse=True)
            if not candidates or candidates[0][0] < self.match_threshold:
                return self._new_identity(camera_id, local_key, vector, observed_at)

            best_score, best_gid, best_identity = candidates[0]
            second_score = candidates[1][0] if len(candidates) > 1 else -1.0
            if best_score < self.strong_threshold and best_score - second_score < self.ambiguity_margin:
                self._ambiguous += 1
                return self._new_identity(camera_id, local_key, vector, observed_at)

            if self._same_camera_conflict(best_identity, camera_id, local_key):
                self._conflicts += 1
                return self._new_identity(camera_id, local_key, vector, observed_at)
            if self._concurrent_room_conflict(best_identity, camera_id):
                self._room_rejects += 1
                return self._new_identity(camera_id, local_key, vector, observed_at)

            self._track_to_global[local_key] = best_gid
            self._track_activity[local_key] = observed_at
            self._update_identity(best_identity, camera_id, local_key, vector, observed_at)
            self._merges += 1
            self._merge_scores.append(float(best_score))
            if len(self._merge_scores) > 128:
                del self._merge_scores[:-128]
            return best_gid, float(best_score), "merged"

    def _update_identity(self, identity: GlobalIdentity, camera_id: str, local_key: str, vector, observed_at: float):
        alpha = self.prototype_alpha
        identity.prototype = _norm((1.0 - alpha) * identity.prototype + alpha * vector)
        identity.gallery.append(_norm(vector).copy())
        if len(identity.gallery) > self.gallery_size:
            del identity.gallery[:-self.gallery_size]
        identity.observations += 1
        identity.last_seen = float(observed_at)
        identity.last_camera = str(camera_id)
        identity.active_tracks[str(camera_id)] = str(local_key)

    def lookup_track(self, camera_id: str, local_track_id: int):
        key = self._local_key(camera_id, local_track_id)
        with self._lock:
            return self._track_to_global.get(key)

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
                for gid, identity in self._identities.items()
            }

    def metrics(self):
        with self._lock:
            scores = list(self._merge_scores)
            return {
                "global_identities": len(self._identities),
                "active_local_tracks": len(self._track_to_global),
                "new": self._new,
                "merges": self._merges,
                "existing_updates": self._existing_updates,
                "ambiguous": self._ambiguous,
                "conflicts": self._conflicts,
                "room_rejects": self._room_rejects,
                "released": self._released,
                "expired": self._expired,
                "match_threshold": self.match_threshold,
                "strong_threshold": self.strong_threshold,
                "camera_rooms": dict(self.camera_rooms),
                "merge_score_last": scores[-1] if scores else None,
                "merge_score_min": min(scores) if scores else None,
                "merge_score_max": max(scores) if scores else None,
            }
