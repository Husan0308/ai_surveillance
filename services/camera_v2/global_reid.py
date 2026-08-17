from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass, field


Vector = tuple[float, ...]
LocalKey = tuple[int, int]


def _normalize(values) -> Vector | None:
    values = tuple(float(v) for v in values)
    norm2 = sum(v * v for v in values)
    if not values or norm2 <= 1e-12:
        return None
    inv = 1.0 / math.sqrt(norm2)
    return tuple(v * inv for v in values)


def _dot(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        return -1.0
    return sum(x * y for x, y in zip(a, b))


def _parse_room_map() -> dict[int, int]:
    """Map nvstreammux source indexes to physical rooms.

    cameras.yaml order is CAM-01..CAM-06, while the real room pairs are:
      CAM-01 + CAM-02
      CAM-03 + CAM-06
      CAM-05 + CAM-04

    The previous source_id//2 rule incorrectly paired CAM-03/CAM-04 and
    CAM-05/CAM-06. That made the conflict guard reject legitimate same-person
    matches and was the main reason conflicts grew into the thousands while
    strong matches stayed near zero.
    """
    default = "0:0,1:0,2:1,5:1,4:2,3:2"
    raw = os.environ.get("CAMERA_V2_REID_ROOM_MAP", default)
    mapping: dict[int, int] = {}
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        left, right = token.split(":", 1)
        try:
            mapping[int(left.strip())] = int(right.strip())
        except ValueError:
            continue
    return mapping or {0: 0, 1: 0, 2: 1, 5: 1, 4: 2, 3: 2}


@dataclass
class GalleryItem:
    vector: Vector
    source_id: int
    seen_at: float
    quality: float


@dataclass
class GlobalProfile:
    global_id: int
    created_at: float
    last_seen: float
    last_source: int
    last_room: int
    gallery: deque[GalleryItem] = field(default_factory=deque)
    centroid: Vector | None = None
    sample_count: int = 0
    known_name: str = ""


@dataclass
class LocalBinding:
    global_id: int
    first_seen: float
    last_seen: float
    last_source: int
    merge_candidate: int | None = None
    merge_votes: int = 0


class GlobalReIDManager:
    """Session-level multi-camera identity manager.

    NvDCF object IDs remain local to a camera. Appearance ReID maps those local
    tracks to a stable global Unknown_N identity. The matcher uses physical room
    topology, active-track conflict guards, a gallery/centroid score, ambiguity
    margin and delayed fragment reconciliation.
    """

    def __init__(self) -> None:
        self.match_threshold = float(os.environ.get("CAMERA_V2_REID_MATCH", "0.72"))
        self.strong_threshold = float(os.environ.get("CAMERA_V2_REID_STRONG", "0.80"))
        self.same_room_threshold = float(os.environ.get("CAMERA_V2_REID_SAME_ROOM", "0.68"))
        self.same_camera_threshold = float(os.environ.get("CAMERA_V2_REID_SAME_CAMERA", "0.70"))
        self.min_margin = float(os.environ.get("CAMERA_V2_REID_MARGIN", "0.025"))
        self.merge_votes_required = max(1, int(os.environ.get("CAMERA_V2_REID_MERGE_VOTES", "2")))
        self.gallery_limit = max(6, int(os.environ.get("CAMERA_V2_REID_GALLERY", "24")))
        self.profile_ttl = float(os.environ.get("CAMERA_V2_REID_PROFILE_TTL", "1800"))
        self.binding_ttl = float(os.environ.get("CAMERA_V2_REID_BINDING_TTL", "15"))
        self.teleport_guard = float(os.environ.get("CAMERA_V2_REID_TELEPORT_GUARD", "1.0"))
        self.max_transition_gap = float(os.environ.get("CAMERA_V2_REID_MAX_TRANSITION", "180"))
        self.room_map = _parse_room_map()
        self.next_global_id = 1
        self.profiles: dict[int, GlobalProfile] = {}
        self.bindings: dict[LocalKey, LocalBinding] = {}
        self.aliases: dict[int, int] = {}
        self.stats = {
            "observations": 0,
            "new_global": 0,
            "direct_match": 0,
            "strong_match": 0,
            "merged": 0,
            "ambiguous_new": 0,
            "rejected_conflict": 0,
        }

    def room_of(self, source_id: int) -> int:
        source_id = int(source_id)
        return self.room_map.get(source_id, source_id)

    def _resolve(self, global_id: int) -> int:
        root = int(global_id)
        trail = []
        while root in self.aliases:
            trail.append(root)
            root = self.aliases[root]
        for item in trail:
            self.aliases[item] = root
        return root

    def _new_profile(self, source_id: int, now: float) -> GlobalProfile:
        gid = self.next_global_id
        self.next_global_id += 1
        profile = GlobalProfile(
            global_id=gid,
            created_at=now,
            last_seen=now,
            last_source=int(source_id),
            last_room=self.room_of(source_id),
            gallery=deque(maxlen=self.gallery_limit),
        )
        self.profiles[gid] = profile
        self.stats["new_global"] += 1
        return profile

    def _active_conflict(self, profile: GlobalProfile, source_id: int, object_id: int, now: float) -> bool:
        source_id = int(source_id)
        room = self.room_of(source_id)
        for (sid, oid), binding in self.bindings.items():
            if sid == source_id and oid == int(object_id):
                continue
            if self._resolve(binding.global_id) != profile.global_id:
                continue
            if now - binding.last_seen > self.teleport_guard:
                continue
            other_room = self.room_of(sid)
            # The same person may be visible simultaneously in the two cameras
            # covering one physical room. Different rooms are mutually exclusive.
            if other_room != room:
                return True
            # Two simultaneous local tracks in one exact camera cannot be one body.
            if sid == source_id:
                return True
        return False

    def _score_profile(self, profile: GlobalProfile, vector: Vector) -> float:
        if not profile.gallery:
            return -1.0
        sims = sorted((_dot(vector, item.vector) for item in profile.gallery), reverse=True)
        top = sims[: min(3, len(sims))]
        best = top[0]
        top_mean = sum(top) / len(top)
        # Keep a strong viewpoint match useful without trusting one neighbour only.
        gallery_score = 0.60 * best + 0.40 * top_mean
        centroid_score = _dot(vector, profile.centroid) if profile.centroid else gallery_score
        return 0.72 * gallery_score + 0.28 * centroid_score

    def _threshold_for(self, profile: GlobalProfile, source_id: int, now: float) -> float:
        source_id = int(source_id)
        if source_id == profile.last_source:
            return self.same_camera_threshold
        room = self.room_of(source_id)
        if room == profile.last_room:
            return self.same_room_threshold
        if now - profile.last_seen > self.max_transition_gap:
            return max(self.strong_threshold, self.match_threshold + 0.05)
        return self.match_threshold

    def _best_candidate(
        self,
        vector: Vector,
        source_id: int,
        object_id: int,
        now: float,
        *,
        exclude: int | None = None,
    ) -> tuple[int | None, float, float, float]:
        scored: list[tuple[float, float, int]] = []
        conflict_skips = 0
        for raw_gid, profile in self.profiles.items():
            gid = self._resolve(raw_gid)
            if gid != raw_gid:
                continue
            if exclude is not None and gid == self._resolve(exclude):
                continue
            if now - profile.last_seen > self.profile_ttl:
                continue
            if self._active_conflict(profile, source_id, object_id, now):
                conflict_skips += 1
                continue
            score = self._score_profile(profile, vector)
            threshold = self._threshold_for(profile, source_id, now)
            scored.append((score, threshold, gid))

        self.stats["rejected_conflict"] += conflict_skips
        if not scored:
            return None, -1.0, -1.0, self.match_threshold
        scored.sort(reverse=True)
        best_score, threshold, best_gid = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -1.0
        return best_gid, best_score, second_score, threshold

    def _update_profile(self, profile: GlobalProfile, vector: Vector, source_id: int, now: float, quality: float) -> None:
        profile.last_seen = now
        profile.last_source = int(source_id)
        profile.last_room = self.room_of(source_id)
        profile.sample_count += 1

        add = True
        if profile.gallery:
            newest = profile.gallery[-1]
            if newest.source_id == int(source_id) and now - newest.seen_at < 0.45:
                if _dot(vector, newest.vector) > 0.992:
                    add = False
        if add:
            profile.gallery.append(GalleryItem(vector, int(source_id), now, float(quality)))

        if profile.centroid is None:
            profile.centroid = vector
        else:
            alpha = 0.18 if quality >= 0.45 else 0.10
            mixed = tuple((1.0 - alpha) * a + alpha * b for a, b in zip(profile.centroid, vector))
            profile.centroid = _normalize(mixed) or profile.centroid

    def _merge_profiles(self, child_gid: int, parent_gid: int, now: float) -> int:
        child_gid = self._resolve(child_gid)
        parent_gid = self._resolve(parent_gid)
        if child_gid == parent_gid:
            return parent_gid
        child = self.profiles.get(child_gid)
        parent = self.profiles.get(parent_gid)
        if child is None or parent is None:
            return parent_gid

        if child.created_at < parent.created_at:
            child_gid, parent_gid = parent_gid, child_gid
            child, parent = parent, child

        combined = list(parent.gallery) + list(child.gallery)
        combined.sort(key=lambda item: (item.quality, item.seen_at), reverse=True)
        parent.gallery.clear()
        for item in sorted(combined[: self.gallery_limit], key=lambda item: item.seen_at):
            parent.gallery.append(item)

        vectors = [item.vector for item in parent.gallery]
        if vectors:
            mean = tuple(sum(values) / len(values) for values in zip(*vectors))
            parent.centroid = _normalize(mean)
        parent.sample_count += child.sample_count
        if child.last_seen > parent.last_seen:
            parent.last_seen = child.last_seen
            parent.last_source = child.last_source
            parent.last_room = child.last_room
        else:
            parent.last_seen = max(parent.last_seen, now)
        if not parent.known_name and child.known_name:
            parent.known_name = child.known_name

        self.aliases[child_gid] = parent_gid
        self.profiles.pop(child_gid, None)
        for binding in self.bindings.values():
            if binding.global_id == child_gid or self._resolve(binding.global_id) == child_gid:
                binding.global_id = parent_gid
                binding.merge_candidate = None
                binding.merge_votes = 0
        self.stats["merged"] += 1
        return parent_gid

    def _consider_reconciliation(
        self,
        key: LocalKey,
        binding: LocalBinding,
        vector: Vector,
        source_id: int,
        object_id: int,
        now: float,
    ) -> None:
        current_gid = self._resolve(binding.global_id)
        current = self.profiles.get(current_gid)
        if current is None:
            return
        if current.sample_count > 18 and now - current.created_at > 12.0:
            return

        candidate_gid, score, second, threshold = self._best_candidate(
            vector,
            source_id,
            object_id,
            now,
            exclude=current_gid,
        )
        if candidate_gid is None or score < threshold or score - second < self.min_margin:
            binding.merge_candidate = None
            binding.merge_votes = 0
            return

        if binding.merge_candidate == candidate_gid:
            binding.merge_votes += 1
        else:
            binding.merge_candidate = candidate_gid
            binding.merge_votes = 1

        if binding.merge_votes >= self.merge_votes_required:
            binding.global_id = self._merge_profiles(current_gid, candidate_gid, now)
            binding.merge_candidate = None
            binding.merge_votes = 0

    def observe(self, rows: list[dict], now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        self._expire(now)

        for row in rows:
            source_id = int(row.get("source_id", -1))
            object_id = int(row.get("object_id", -1))
            if source_id < 0 or object_id < 0:
                continue
            vector = _normalize(row.get("feature", ()))
            if vector is None:
                continue
            self.stats["observations"] += 1
            key = (source_id, object_id)
            detector_conf = max(0.0, float(row.get("confidence", 0.0) or 0.0))
            tracker_conf = max(0.0, float(row.get("tracker_confidence", 0.0) or 0.0))
            quality = max(detector_conf, tracker_conf)

            binding = self.bindings.get(key)
            if binding is not None and now - binding.last_seen <= self.binding_ttl:
                gid = self._resolve(binding.global_id)
                binding.global_id = gid
                binding.last_seen = now
                binding.last_source = source_id
                profile = self.profiles.get(gid)
                if profile is None:
                    profile = self._new_profile(source_id, now)
                    binding.global_id = profile.global_id
                self._update_profile(profile, vector, source_id, now, quality)
                self._consider_reconciliation(key, binding, vector, source_id, object_id, now)
                continue

            best_gid, best_score, second_score, threshold = self._best_candidate(
                vector,
                source_id,
                object_id,
                now,
            )
            accepted = (
                best_gid is not None
                and best_score >= threshold
                and best_score - second_score >= self.min_margin
            )
            if accepted:
                profile = self.profiles[best_gid]
                self.stats["direct_match"] += 1
                if best_score >= self.strong_threshold:
                    self.stats["strong_match"] += 1
            else:
                profile = self._new_profile(source_id, now)
                if best_gid is not None:
                    self.stats["ambiguous_new"] += 1

            binding = LocalBinding(
                global_id=profile.global_id,
                first_seen=now,
                last_seen=now,
                last_source=source_id,
            )
            self.bindings[key] = binding
            self._update_profile(profile, vector, source_id, now, quality)
            if not accepted:
                self._consider_reconciliation(key, binding, vector, source_id, object_id, now)

    def _expire(self, now: float) -> None:
        stale_bindings = [
            key for key, binding in self.bindings.items()
            if now - binding.last_seen > self.binding_ttl
        ]
        for key in stale_bindings:
            self.bindings.pop(key, None)

        stale_profiles = [
            gid for gid, profile in self.profiles.items()
            if now - profile.last_seen > self.profile_ttl
        ]
        for gid in stale_profiles:
            self.profiles.pop(gid, None)

    def set_known_name(self, global_id: int, name: str) -> None:
        gid = self._resolve(int(global_id))
        profile = self.profiles.get(gid)
        if profile is not None:
            profile.known_name = str(name).strip()

    def label_for(self, global_id: int) -> str:
        gid = self._resolve(int(global_id))
        profile = self.profiles.get(gid)
        if profile and profile.known_name:
            return profile.known_name
        return f"Unknown_{gid:02d}"

    def label_assignments(self) -> list[tuple[int, int, str]]:
        output: list[tuple[int, int, str]] = []
        for (source_id, object_id), binding in self.bindings.items():
            gid = self._resolve(binding.global_id)
            output.append((source_id, object_id, self.label_for(gid)))
        return output

    def snapshot(self) -> dict:
        return {
            "global_count": len(self.profiles),
            "local_bindings": len(self.bindings),
            "room_map": dict(self.room_map),
            "stats": dict(self.stats),
            "profiles": [
                {
                    "global_id": gid,
                    "label": self.label_for(gid),
                    "samples": profile.sample_count,
                    "gallery": len(profile.gallery),
                    "last_source": profile.last_source,
                    "last_room": profile.last_room,
                    "known": bool(profile.known_name),
                }
                for gid, profile in sorted(self.profiles.items())
            ],
        }
