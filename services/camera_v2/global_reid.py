from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass, field


Vector = tuple[float, ...]
LocalKey = tuple[int, int]
BBox = tuple[float, float, float, float]


def _normalize(values) -> Vector | None:
    values = tuple(float(v) for v in values)
    norm2 = sum(v * v for v in values)
    if not values or norm2 <= 1e-12:
        return None
    inv = 1.0 / math.sqrt(norm2)
    return tuple(v * inv for v in values)


def _dot(a: Vector | None, b: Vector | None) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    return sum(x * y for x, y in zip(a, b))


def _mean_normalized(vectors: list[Vector]) -> Vector | None:
    if not vectors:
        return None
    mean = tuple(sum(values) / len(values) for values in zip(*vectors))
    return _normalize(mean)


def _parse_room_map() -> dict[int, int]:
    # cameras.yaml source order is CAM-01..CAM-06. The currently configured RTSP
    # channel pairs are 101+201, 301+401 and 501+601, therefore the physical-room
    # source pairs are CAM-01+02, CAM-03+06 and CAM-05+04. Override with
    # CAMERA_V2_REID_ROOM_MAP if the installation is rewired.
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
    color: Vector | None
    source_id: int
    room_id: int
    seen_at: float
    quality: float


@dataclass
class RoomMemory:
    room_id: int
    gallery: deque[GalleryItem]
    centroid: Vector | None = None
    color_centroid: Vector | None = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    samples: int = 0


@dataclass
class GlobalProfile:
    global_id: int
    created_at: float
    last_seen: float
    last_source: int
    last_room: int
    gallery: deque[GalleryItem]
    centroid: Vector | None = None
    color_centroid: Vector | None = None
    rooms: dict[int, RoomMemory] = field(default_factory=dict)
    sample_count: int = 0
    known_name: str = ""


@dataclass
class LocalBinding:
    global_id: int
    first_seen: float
    last_seen: float
    last_source: int
    last_room: int
    last_bbox: BBox | None = None
    merge_candidate: int | None = None
    merge_votes: int = 0


class GlobalReIDManager:
    """Room-tracklet based identity manager for the fixed six-camera office.

    A single frame embedding is deliberately not treated as an identity. Each
    NvDCF local track first builds a short multi-frame prototype. Once associated,
    every Global ID maintains both a global gallery and a separate appearance
    memory for each physical room. Two peer cameras in one room therefore reinforce
    one room-level person signature instead of competing as independent identities.
    When the person appears in another room, the new track is compared against the
    remembered per-room signatures plus the global gallery and temporal continuity.

    ReID only maps local NvDCF IDs to Global IDs; it never changes bbox geometry.
    """

    def __init__(self) -> None:
        # Thresholds are context-specific. Same-room peer-camera association may be
        # looser because room topology + simultaneous one-to-one activity are strong
        # cues. New-room transitions remain stricter.
        self.transition_threshold = float(os.environ.get("CAMERA_V2_REID_MATCH", "0.64"))
        self.strong_threshold = float(os.environ.get("CAMERA_V2_REID_STRONG", "0.77"))
        self.same_room_threshold = float(os.environ.get("CAMERA_V2_REID_SAME_ROOM", "0.56"))
        self.covisible_threshold = float(os.environ.get("CAMERA_V2_REID_COVISIBLE", "0.52"))
        self.same_camera_threshold = float(os.environ.get("CAMERA_V2_REID_SAME_CAMERA", "0.60"))
        self.revisit_threshold = float(os.environ.get("CAMERA_V2_REID_REVISIT", "0.58"))
        self.continuation_threshold = float(os.environ.get("CAMERA_V2_REID_CONTINUATION", "0.57"))
        self.min_margin = float(os.environ.get("CAMERA_V2_REID_MARGIN", "0.010"))
        self.min_new_track_samples = max(1, int(os.environ.get("CAMERA_V2_REID_MIN_TRACKLET_SAMPLES", "2")))
        self.merge_votes_required = max(1, int(os.environ.get("CAMERA_V2_REID_MERGE_VOTES", "2")))
        self.gallery_limit = max(12, int(os.environ.get("CAMERA_V2_REID_GALLERY", "36")))
        self.room_gallery_limit = max(6, int(os.environ.get("CAMERA_V2_REID_ROOM_GALLERY", "16")))
        self.profile_ttl = float(os.environ.get("CAMERA_V2_REID_PROFILE_TTL", "1800"))
        self.binding_ttl = float(os.environ.get("CAMERA_V2_REID_BINDING_TTL", "20"))
        self.active_ttl = float(os.environ.get("CAMERA_V2_REID_ACTIVE_TTL", "0.45"))
        self.continuation_gap = float(os.environ.get("CAMERA_V2_REID_CONTINUATION_GAP", "3.0"))
        self.transition_bonus_window = float(os.environ.get("CAMERA_V2_REID_TRANSITION_WINDOW", "45"))
        self.max_transition_gap = float(os.environ.get("CAMERA_V2_REID_MAX_TRANSITION", "300"))
        self.room_map = _parse_room_map()

        self.next_global_id = 1
        self.profiles: dict[int, GlobalProfile] = {}
        self.bindings: dict[LocalKey, LocalBinding] = {}
        self.aliases: dict[int, int] = {}
        self.active_seen: dict[LocalKey, float] = {}
        self.local_evidence: dict[LocalKey, deque[tuple[Vector, Vector | None, float]]] = {}

        self.stats = {
            "observations": 0,
            "new_global": 0,
            "direct_match": 0,
            "covisible_match": 0,
            "continuation_match": 0,
            "revisit_match": 0,
            "transition_match": 0,
            "room_memory_match": 0,
            "strong_match": 0,
            "merged": 0,
            "ambiguous_new": 0,
            "pending_tracklet": 0,
            "rejected_conflict": 0,
            "last_best_milli": -1000,
            "last_second_milli": -1000,
            "last_threshold_milli": 0,
            "last_reid_milli": -1000,
            "last_color_milli": -1000,
            "last_room_milli": -1000,
            "last_context": "none",
        }

    def room_of(self, source_id: int) -> int:
        source_id = int(source_id)
        return self.room_map.get(source_id, source_id)

    def _resolve(self, global_id: int) -> int:
        root = int(global_id)
        trail: list[int] = []
        while root in self.aliases:
            trail.append(root)
            root = self.aliases[root]
        for item in trail:
            self.aliases[item] = root
        return root

    def _is_active(self, key: LocalKey, now: float) -> bool:
        seen = self.active_seen.get(key)
        return seen is not None and now - seen <= self.active_ttl

    def update_active_tracks(self, rows: list[dict], now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        for row in rows:
            sid = int(row.get("source_id", -1))
            oid = int(row.get("object_id", -1))
            if sid < 0 or oid < 0:
                continue
            key = (sid, oid)
            self.active_seen[key] = now
            binding = self.bindings.get(key)
            if binding is not None:
                bbox = self._bbox_from_row(row)
                if bbox is not None:
                    binding.last_bbox = bbox
        stale = [
            key for key, seen in self.active_seen.items()
            if now - seen > max(2.0, self.active_ttl * 4.0)
        ]
        for key in stale:
            self.active_seen.pop(key, None)

    @staticmethod
    def _bbox_from_row(row: dict) -> BBox | None:
        raw = row.get("bbox")
        if raw is not None and len(raw) == 4:
            box = tuple(float(v) for v in raw)
            if box[2] > box[0] and box[3] > box[1]:
                return box  # type: ignore[return-value]
        if all(name in row for name in ("left", "top", "width", "height")):
            left = float(row["left"])
            top = float(row["top"])
            width = float(row["width"])
            height = float(row["height"])
            if width > 1.0 and height > 1.0:
                return (left, top, left + width, top + height)
        return None

    @staticmethod
    def _bbox_distance(a: BBox | None, b: BBox | None) -> float:
        if a is None or b is None:
            return 99.0
        acx, acy = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
        bcx, bcy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
        bw, bh = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
        return math.hypot(acx - bcx, acy - bcy) / max(24.0, math.hypot(bw, bh))

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
        gid = self._resolve(profile.global_id)
        for key, binding in self.bindings.items():
            sid, oid = key
            if sid == source_id and oid == int(object_id):
                continue
            if self._resolve(binding.global_id) != gid or not self._is_active(key, now):
                continue
            other_room = self.room_of(sid)
            if other_room != room:
                return True
            if sid == source_id:
                return True
        return False

    def _profile_covisible_same_room(self, profile: GlobalProfile, source_id: int, now: float) -> bool:
        room = self.room_of(source_id)
        gid = self._resolve(profile.global_id)
        for key, binding in self.bindings.items():
            sid, _ = key
            if sid == int(source_id) or not self._is_active(key, now):
                continue
            if self._resolve(binding.global_id) == gid and self.room_of(sid) == room:
                return True
        return False

    @staticmethod
    def _score_gallery(
        items: list[GalleryItem],
        centroid: Vector | None,
        color_centroid: Vector | None,
        vector: Vector,
        color: Vector | None,
    ) -> tuple[float, float, float]:
        if not items:
            return -1.0, -1.0, -1.0
        reid_sims = sorted((_dot(vector, item.vector) for item in items), reverse=True)
        top = reid_sims[: min(4, len(reid_sims))]
        best = top[0]
        top_mean = sum(top) / len(top)
        centroid_score = _dot(vector, centroid) if centroid else top_mean
        reid_score = 0.58 * best + 0.27 * top_mean + 0.15 * centroid_score

        color_score = -1.0
        if color and color_centroid:
            color_sims = sorted((_dot(color, item.color) for item in items if item.color), reverse=True)
            if color_sims:
                color_score = 0.65 * color_sims[0] + 0.35 * _dot(color, color_centroid)

        combined = reid_score if color_score < 0.0 else 0.90 * reid_score + 0.10 * color_score
        return combined, reid_score, color_score

    def _score_profile(
        self,
        profile: GlobalProfile,
        vector: Vector,
        color: Vector | None,
        source_id: int,
        now: float,
    ) -> tuple[float, float, float, float, float, bool, str]:
        target_room = self.room_of(source_id)
        global_score, global_reid, global_color = self._score_gallery(
            list(profile.gallery), profile.centroid, profile.color_centroid, vector, color
        )
        if global_score < -0.5:
            return -1.0, -1.0, -1.0, -1.0, self.transition_threshold, False, "empty"

        room_score = -1.0
        room_reid = -1.0
        room_color = -1.0
        room_memory = profile.rooms.get(target_room)
        if room_memory is not None and room_memory.gallery:
            room_score, room_reid, room_color = self._score_gallery(
                list(room_memory.gallery),
                room_memory.centroid,
                room_memory.color_centroid,
                vector,
                color,
            )

        covisible = False
        context = "transition"
        threshold = self.transition_threshold
        score = global_score

        if int(source_id) == profile.last_source:
            context = "same_camera"
            threshold = self.same_camera_threshold
            if room_score >= 0.0:
                score = 0.70 * room_score + 0.30 * global_score
        elif target_room == profile.last_room:
            covisible = self._profile_covisible_same_room(profile, source_id, now)
            context = "covisible" if covisible else "same_room"
            threshold = self.covisible_threshold if covisible else self.same_room_threshold
            if room_score >= 0.0:
                # This is the key room-tracklet rule: when peer cameras cover the
                # same room, compare mostly against that person's room signature.
                score = 0.78 * room_score + 0.22 * global_score
        elif room_score >= 0.0:
            # Person has visited this destination room before. Same-room lighting
            # and viewpoint history is more discriminative than a network-wide mean.
            context = "revisit"
            threshold = self.revisit_threshold
            score = 0.68 * room_score + 0.32 * global_score
        else:
            # First observed transition into this room. Preserve the old room memory
            # and use a small recency prior rather than blindly lowering appearance.
            context = "transition"
            age = max(0.0, now - profile.last_seen)
            if age <= self.transition_bonus_window:
                bonus = 0.028 * (1.0 - age / max(1.0, self.transition_bonus_window))
                score += max(0.0, bonus)
            if age > self.max_transition_gap:
                threshold = max(self.strong_threshold, self.transition_threshold + 0.06)

        reid_score = room_reid if room_reid >= 0.0 and context in {"same_camera", "same_room", "covisible", "revisit"} else global_reid
        color_score = room_color if room_color >= 0.0 and context in {"same_camera", "same_room", "covisible", "revisit"} else global_color
        return score, reid_score, color_score, room_score, threshold, covisible, context

    def _best_candidate(
        self,
        vector: Vector,
        color: Vector | None,
        source_id: int,
        object_id: int,
        now: float,
        *,
        exclude: int | None = None,
    ) -> tuple[int | None, float, float, float, float, float, float, bool, str]:
        scored: list[tuple[float, float, int, float, float, float, bool, str]] = []
        conflict_seen = False
        for raw_gid, profile in self.profiles.items():
            gid = self._resolve(raw_gid)
            if gid != raw_gid:
                continue
            if exclude is not None and gid == self._resolve(exclude):
                continue
            if now - profile.last_seen > self.profile_ttl:
                continue
            if self._active_conflict(profile, source_id, object_id, now):
                conflict_seen = True
                continue
            score, reid_score, color_score, room_score, threshold, covisible, context = self._score_profile(
                profile, vector, color, source_id, now
            )
            scored.append((score, threshold, gid, reid_score, color_score, room_score, covisible, context))

        if conflict_seen:
            self.stats["rejected_conflict"] += 1
        if not scored:
            return None, -1.0, -1.0, self.transition_threshold, -1.0, -1.0, -1.0, False, "none"
        scored.sort(reverse=True)
        best_score, threshold, best_gid, reid_score, color_score, room_score, covisible, context = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -1.0
        self.stats["last_best_milli"] = int(round(best_score * 1000))
        self.stats["last_second_milli"] = int(round(second_score * 1000))
        self.stats["last_threshold_milli"] = int(round(threshold * 1000))
        self.stats["last_reid_milli"] = int(round(reid_score * 1000))
        self.stats["last_color_milli"] = int(round(color_score * 1000))
        self.stats["last_room_milli"] = int(round(room_score * 1000))
        self.stats["last_context"] = context
        return best_gid, best_score, second_score, threshold, reid_score, color_score, room_score, covisible, context

    def _aggregate_local(
        self, key: LocalKey, vector: Vector, color: Vector | None, now: float
    ) -> tuple[Vector, Vector | None, int]:
        evidence = self.local_evidence.get(key)
        if evidence is None:
            evidence = deque(maxlen=6)
            self.local_evidence[key] = evidence
        evidence.append((vector, color, now))
        recent = list(evidence)[-5:]
        vectors = [item[0] for item in recent]
        colors = [item[1] for item in recent if item[1]]
        return _mean_normalized(vectors) or vector, _mean_normalized(colors) if colors else color, len(recent)

    def _update_room_memory(
        self,
        profile: GlobalProfile,
        item: GalleryItem,
    ) -> None:
        room = profile.rooms.get(item.room_id)
        if room is None:
            room = RoomMemory(
                room_id=item.room_id,
                gallery=deque(maxlen=self.room_gallery_limit),
                first_seen=item.seen_at,
                last_seen=item.seen_at,
            )
            profile.rooms[item.room_id] = room
        room.last_seen = item.seen_at
        room.samples += 1

        add = True
        if room.gallery:
            newest = room.gallery[-1]
            if newest.source_id == item.source_id and item.seen_at - newest.seen_at < 0.45:
                if _dot(item.vector, newest.vector) > 0.993:
                    add = False
        if add:
            room.gallery.append(item)
            room.centroid = _mean_normalized([entry.vector for entry in room.gallery]) or room.centroid
            colors = [entry.color for entry in room.gallery if entry.color]
            if colors:
                room.color_centroid = _mean_normalized(colors) or room.color_centroid

    def _update_profile(
        self,
        profile: GlobalProfile,
        vector: Vector,
        color: Vector | None,
        source_id: int,
        now: float,
        quality: float,
    ) -> None:
        room_id = self.room_of(source_id)
        profile.last_seen = now
        profile.last_source = int(source_id)
        profile.last_room = room_id
        profile.sample_count += 1
        item = GalleryItem(vector, color, int(source_id), room_id, now, float(quality))

        add = True
        if profile.gallery:
            newest = profile.gallery[-1]
            if newest.source_id == int(source_id) and now - newest.seen_at < 0.45:
                if _dot(vector, newest.vector) > 0.993:
                    add = False
        if add:
            profile.gallery.append(item)
            profile.centroid = _mean_normalized([entry.vector for entry in profile.gallery]) or profile.centroid
            colors = [entry.color for entry in profile.gallery if entry.color]
            if colors:
                profile.color_centroid = _mean_normalized(colors) or profile.color_centroid
        self._update_room_memory(profile, item)

    def _find_same_camera_continuation(
        self,
        source_id: int,
        object_id: int,
        vector: Vector,
        color: Vector | None,
        bbox: BBox | None,
        now: float,
    ) -> tuple[int | None, float]:
        best_gid: int | None = None
        best_total = -1.0
        for key, binding in self.bindings.items():
            sid, oid = key
            if sid != int(source_id) or oid == int(object_id) or self._is_active(key, now):
                continue
            age = now - binding.last_seen
            if age < 0.0 or age > self.continuation_gap:
                continue
            gid = self._resolve(binding.global_id)
            profile = self.profiles.get(gid)
            if profile is None or self._active_conflict(profile, source_id, object_id, now):
                continue
            score, _r, _c, _room, _thr, _co, _ctx = self._score_profile(profile, vector, color, source_id, now)
            distance = self._bbox_distance(binding.last_bbox, bbox)
            if distance > 1.6:
                continue
            spatial = max(0.0, 1.0 - distance / 1.6)
            total = 0.82 * score + 0.18 * spatial
            if total > best_total:
                best_total = total
                best_gid = gid
        if best_gid is not None and best_total >= self.continuation_threshold:
            return best_gid, best_total
        return None, best_total

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
        parent.centroid = _mean_normalized([item.vector for item in parent.gallery]) or parent.centroid
        colors = [item.color for item in parent.gallery if item.color]
        if colors:
            parent.color_centroid = _mean_normalized(colors) or parent.color_centroid

        # Rebuild per-room memories from both profiles so a remembered appearance in
        # any room survives fragment reconciliation.
        merged_items = list(parent.gallery)
        parent.rooms = {}
        for item in merged_items:
            self._update_room_memory(parent, item)

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
            if self._resolve(binding.global_id) == child_gid or binding.global_id == child_gid:
                binding.global_id = parent_gid
                binding.merge_candidate = None
                binding.merge_votes = 0
        self.stats["merged"] += 1
        return parent_gid

    def _consider_reconciliation(
        self,
        binding: LocalBinding,
        vector: Vector,
        color: Vector | None,
        source_id: int,
        object_id: int,
        now: float,
    ) -> None:
        current_gid = self._resolve(binding.global_id)
        current = self.profiles.get(current_gid)
        if current is None:
            return
        if current.sample_count > 28 and now - current.created_at > 25.0:
            return
        candidate_gid, score, second, threshold, _r, _c, _room, _co, _ctx = self._best_candidate(
            vector, color, source_id, object_id, now, exclude=current_gid
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
            color = _normalize(row.get("color_feature", ()))
            bbox = self._bbox_from_row(row)
            key = (source_id, object_id)
            agg_vector, agg_color, evidence_count = self._aggregate_local(key, vector, color, now)
            self.stats["observations"] += 1
            detector_conf = max(0.0, float(row.get("confidence", 0.0) or 0.0))
            tracker_conf = max(0.0, float(row.get("tracker_confidence", 0.0) or 0.0))
            quality = max(detector_conf, tracker_conf)

            binding = self.bindings.get(key)
            if binding is not None and now - binding.last_seen <= self.binding_ttl:
                gid = self._resolve(binding.global_id)
                binding.global_id = gid
                binding.last_seen = now
                binding.last_source = source_id
                binding.last_room = self.room_of(source_id)
                if bbox is not None:
                    binding.last_bbox = bbox
                profile = self.profiles.get(gid)
                if profile is None:
                    profile = self._new_profile(source_id, now)
                    binding.global_id = profile.global_id
                self._update_profile(profile, vector, color, source_id, now, quality)
                self._consider_reconciliation(binding, agg_vector, agg_color, source_id, object_id, now)
                continue

            continuation_gid, _cont_score = self._find_same_camera_continuation(
                source_id, object_id, agg_vector, agg_color, bbox, now
            )
            accepted = False
            context = "none"
            if continuation_gid is not None:
                profile = self.profiles[continuation_gid]
                accepted = True
                context = "continuation"
                self.stats["continuation_match"] += 1
            else:
                (
                    best_gid,
                    best_score,
                    second_score,
                    threshold,
                    reid_score,
                    color_score,
                    room_score,
                    covisible,
                    context,
                ) = self._best_candidate(agg_vector, agg_color, source_id, object_id, now)
                accepted = (
                    best_gid is not None
                    and best_score >= threshold
                    and best_score - second_score >= self.min_margin
                )
                # For a brand-new room transition, require a short tracklet unless
                # appearance is already very strong. This prevents one bad frame from
                # creating/merging identities, while peer-camera same-room matches can
                # be immediate because room topology is a strong extra cue.
                if (
                    accepted
                    and context == "transition"
                    and evidence_count < self.min_new_track_samples
                    and best_score < self.strong_threshold
                ):
                    accepted = False

                # Same-room peer matching may use the lower threshold only when the
                # ReID component itself is plausible; colour alone cannot force it.
                if accepted and context == "covisible" and reid_score < 0.49:
                    accepted = False

                if accepted:
                    profile = self.profiles[best_gid]
                    self.stats["direct_match"] += 1
                    if context == "covisible":
                        self.stats["covisible_match"] += 1
                    elif context == "revisit":
                        self.stats["revisit_match"] += 1
                    elif context == "transition":
                        self.stats["transition_match"] += 1
                    if room_score >= 0.0 and context in {"same_room", "covisible", "revisit"}:
                        self.stats["room_memory_match"] += 1
                    if best_score >= self.strong_threshold:
                        self.stats["strong_match"] += 1
                else:
                    # Delay the creation of a brand-new Global ID until the local
                    # track has at least two independent embeddings. The UI may show
                    # the NvDCF local Unknown_ID briefly, but the identity database is
                    # not polluted by single-frame fragments.
                    if evidence_count < self.min_new_track_samples:
                        self.stats["pending_tracklet"] += 1
                        continue
                    profile = self._new_profile(source_id, now)
                    if best_gid is not None:
                        self.stats["ambiguous_new"] += 1

            binding = LocalBinding(
                global_id=profile.global_id,
                first_seen=now,
                last_seen=now,
                last_source=source_id,
                last_room=self.room_of(source_id),
                last_bbox=bbox,
            )
            self.bindings[key] = binding
            self._update_profile(profile, vector, color, source_id, now, quality)
            if not accepted:
                self._consider_reconciliation(binding, agg_vector, agg_color, source_id, object_id, now)

    def _expire(self, now: float) -> None:
        stale_bindings = [
            key for key, binding in self.bindings.items()
            if now - binding.last_seen > self.binding_ttl
        ]
        for key in stale_bindings:
            self.bindings.pop(key, None)
            self.local_evidence.pop(key, None)
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
            "active_tracks": len(self.active_seen),
            "room_memories": sum(len(profile.rooms) for profile in self.profiles.values()),
            "stats": dict(self.stats),
            "profiles": [
                {
                    "global_id": gid,
                    "label": self.label_for(gid),
                    "samples": profile.sample_count,
                    "gallery": len(profile.gallery),
                    "last_source": profile.last_source,
                    "last_room": profile.last_room,
                    "rooms": {
                        room_id: {
                            "samples": memory.samples,
                            "gallery": len(memory.gallery),
                            "last_seen": memory.last_seen,
                        }
                        for room_id, memory in sorted(profile.rooms.items())
                    },
                    "known": bool(profile.known_name),
                }
                for gid, profile in sorted(self.profiles.items())
            ],
        }
