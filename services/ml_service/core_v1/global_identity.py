from __future__ import annotations

from dataclasses import dataclass, field
import math
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

    Appearance is necessary but never sufficient: same-camera concurrency is a
    hard conflict, optional camera topology/time windows gate impossible moves,
    and ambiguous nearest neighbours are not merged.
    """

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.match_threshold = float(cfg.get("match_threshold", 0.78))
        self.strong_threshold = float(cfg.get("strong_threshold", 0.84))
        self.ambiguity_margin = max(0.0, float(cfg.get("ambiguity_margin", 0.045)))
        self.prototype_alpha = max(0.01, min(1.0, float(cfg.get("prototype_alpha", 0.22))))
        self.gallery_size = max(1, int(cfg.get("gallery_size", 8)))
        self.active_timeout_sec = max(0.2, float(cfg.get("active_timeout_sec", 2.5)))
        self.max_transition_sec = max(0.1, float(cfg.get("max_transition_sec", 45.0)))
        self.topology = {
            str(src): {str(dst): tuple(map(float, bounds)) for dst, bounds in (targets or {}).items()}
            for src, targets in (cfg.get("topology") or {}).items()
        }
        self._lock = threading.Lock()
        self._identities: dict[str, GlobalIdentity] = {}
        self._track_to_global: dict[str, str] = {}
        self._counter = 1
        self._merges = 0
        self._new = 0
        self._ambiguous = 0
        self._conflicts = 0

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
        self._new += 1
        return gid, 1.0, "new"

    def _expire_active(self, now: float):
        active_local = set(self._track_to_global)
        for identity in self._identities.values():
            if now - identity.last_seen > self.active_timeout_sec:
                identity.active_tracks.clear()
        for key in list(active_local):
            gid = self._track_to_global.get(key)
            identity = self._identities.get(gid) if gid else None
            if identity is None or key not in identity.active_tracks.values():
                self._track_to_global.pop(key, None)

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

    def assign(self, *, camera_id: str, local_track_id: int, embedding, observed_at: float | None = None):
        camera_id = str(camera_id)
        local_key = f"{camera_id}:{int(local_track_id)}"
        observed_at = float(observed_at if observed_at is not None else time.monotonic())
        vector = _norm(embedding)

        with self._lock:
            self._expire_active(observed_at)
            existing_gid = self._track_to_global.get(local_key)
            if existing_gid and existing_gid in self._identities:
                identity = self._identities[existing_gid]
                self._update_identity(identity, camera_id, local_key, vector, observed_at)
                return existing_gid, 1.0, "existing"

            candidates = []
            for gid, identity in self._identities.items():
                if self._same_camera_conflict(identity, camera_id, local_key):
                    continue
                if not self._transition_allowed(identity, camera_id, observed_at):
                    continue
                prototype_score = _cosine(vector, identity.prototype)
                gallery_score = max((_cosine(vector, item) for item in identity.gallery), default=prototype_score)
                score = max(prototype_score, gallery_score)
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

            self._track_to_global[local_key] = best_gid
            self._update_identity(best_identity, camera_id, local_key, vector, observed_at)
            self._merges += 1
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
        key = f"{camera_id}:{int(local_track_id)}"
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
                    "gallery_size": len(identity.gallery),
                }
                for gid, identity in self._identities.items()
            }

    def metrics(self):
        with self._lock:
            return {
                "global_identities": len(self._identities),
                "active_local_tracks": len(self._track_to_global),
                "new": self._new,
                "merges": self._merges,
                "ambiguous": self._ambiguous,
                "conflicts": self._conflicts,
                "match_threshold": self.match_threshold,
                "strong_threshold": self.strong_threshold,
            }
