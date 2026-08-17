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
    try:
        values = tuple(float(v) for v in values)
    except Exception:
        return None
    if not values:
        return None
    norm2 = sum(v * v for v in values)
    if norm2 <= 1e-12 or not math.isfinite(norm2):
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
    # cameras.yaml source order is CAM-01..CAM-06.
    # Physical room pairs: CAM-01+02, CAM-03+06, CAM-05+04.
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
class EvidenceItem:
    vector: Vector
    color: Vector | None
    seen_at: float
    quality: float


@dataclass
class GalleryItem:
    vector: Vector
    color: Vector | None
    source_id: int
    room_id: int
    seen_at: float
    quality: float
    owner: LocalKey | None = None


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
    state: str = "provisional"  # provisional | confirmed | anchor
    confirm_votes: int = 0
    bad_votes: int = 0
    switch_candidate: int | None = None
    switch_votes: int = 0
    last_score: float = -1.0
    last_threshold: float = 1.0
    last_committed_at: float = 0.0


class GlobalReIDManager:
    """Reversible room-tracklet identity manager.

    Local NvDCF tracks are the geometry truth. Global IDs are soft identity labels.
    Existing identities are never irreversibly contaminated by a single track:
    candidate matches begin provisional, embeddings stay in per-track quarantine,
    and only confirmed evidence is committed to global/room galleries.

    Confirmed bindings are continuously rechecked using leave-one-track-out scoring.
    If later evidence contradicts the assignment, contributions from that local track
    are removed from the old profile and the track is re-associated to the best
    compatible previous identity or receives a new Global ID.

    Same-camera duplicates and simultaneous cross-room use of one Global ID are hard
    cannot-link constraints. Peer cameras in the same room may share a Global ID only
    while their independent local tracklet prototypes remain appearance-compatible.
    """

    def __init__(self) -> None:
        self.transition_threshold = float(os.environ.get("CAMERA_V2_REID_MATCH", "0.66"))
        self.strong_threshold = float(os.environ.get("CAMERA_V2_REID_STRONG", "0.78"))
        self.same_room_threshold = float(os.environ.get("CAMERA_V2_REID_SAME_ROOM", "0.59"))
        self.covisible_threshold = float(os.environ.get("CAMERA_V2_REID_COVISIBLE", "0.57"))
        self.same_camera_threshold = float(os.environ.get("CAMERA_V2_REID_SAME_CAMERA", "0.61"))
        self.revisit_threshold = float(os.environ.get("CAMERA_V2_REID_REVISIT", "0.60"))
        self.continuation_threshold = float(os.environ.get("CAMERA_V2_REID_CONTINUATION", "0.59"))
        self.min_margin = float(os.environ.get("CAMERA_V2_REID_MARGIN", "0.018"))

        self.min_new_track_samples = max(
            2, int(os.environ.get("CAMERA_V2_REID_MIN_TRACKLET_SAMPLES", "3"))
        )
        self.confirm_votes_required = max(
            2, int(os.environ.get("CAMERA_V2_REID_CONFIRM_VOTES", "3"))
        )
        self.correction_votes_required = max(
            2, int(os.environ.get("CAMERA_V2_REID_CORRECTION_VOTES", "2"))
        )
        self.switch_margin = float(os.environ.get("CAMERA_V2_REID_SWITCH_MARGIN", "0.045"))
        self.release_delta = float(os.environ.get("CAMERA_V2_REID_RELEASE_DELTA", "0.055"))
        self.peer_min_reid = float(os.environ.get("CAMERA_V2_REID_PEER_MIN_REID", "0.50"))
        self.peer_confirm_reid = float(os.environ.get("CAMERA_V2_REID_PEER_CONFIRM_REID", "0.56"))

        self.gallery_limit = max(12, int(os.environ.get("CAMERA_V2_REID_GALLERY", "40")))
        self.room_gallery_limit = max(8, int(os.environ.get("CAMERA_V2_REID_ROOM_GALLERY", "18")))
        self.profile_ttl = float(os.environ.get("CAMERA_V2_REID_PROFILE_TTL", "1800"))
        self.binding_ttl = float(os.environ.get("CAMERA_V2_REID_BINDING_TTL", "20"))
        self.active_ttl = float(os.environ.get("CAMERA_V2_REID_ACTIVE_TTL", "0.55"))
        self.continuation_gap = float(os.environ.get("CAMERA_V2_REID_CONTINUATION_GAP", "3.0"))
        self.transition_bonus_window = float(os.environ.get("CAMERA_V2_REID_TRANSITION_WINDOW", "45"))
        self.max_transition_gap = float(os.environ.get("CAMERA_V2_REID_MAX_TRANSITION", "300"))
        self.commit_interval = float(os.environ.get("CAMERA_V2_REID_COMMIT_INTERVAL", "0.65"))
        self.room_map = _parse_room_map()

        self.next_global_id = 1
        self.profiles: dict[int, GlobalProfile] = {}
        self.bindings: dict[LocalKey, LocalBinding] = {}
        self.aliases: dict[int, int] = {}
        self.active_seen: dict[LocalKey, float] = {}
        self.local_evidence: dict[LocalKey, deque[EvidenceItem]] = {}
        self.cannot_link: dict[frozenset[LocalKey], float] = {}

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
            "provisional": 0,
            "confirmed": 0,
            "reassigned": 0,
            "corrections": 0,
            "rollbacks": 0,
            "peer_reject": 0,
            "quarantined": 0,
            "committed": 0,
            "orphan_profiles_removed": 0,
            "last_best_milli": -1000,
            "last_second_milli": -1000,
            "last_threshold_milli": 0,
            "last_reid_milli": -1000,
            "last_color_milli": -1000,
            "last_room_milli": -1000,
            "last_current_milli": -1000,
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
            key
            for key, seen in self.active_seen.items()
            if now - seen > max(2.5, self.active_ttl * 5.0)
        ]
        for key in stale:
            self.active_seen.pop(key, None)

        stale_links = [pair for pair, until in self.cannot_link.items() if until < now]
        for pair in stale_links:
            self.cannot_link.pop(pair, None)

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

    def _evidence(self, key: LocalKey) -> list[EvidenceItem]:
        return list(self.local_evidence.get(key, ()))

    def _track_prototype(self, key: LocalKey) -> tuple[Vector | None, Vector | None, int]:
        evidence = self._evidence(key)[-7:]
        if not evidence:
            return None, None, 0
        vectors = [item.vector for item in evidence]
        if len(vectors) == 1:
            colors = [item.color for item in evidence if item.color]
            return vectors[0], _mean_normalized(colors) if colors else evidence[0].color, 1

        averages: list[float] = []
        for i, vector in enumerate(vectors):
            sims = [_dot(vector, other) for j, other in enumerate(vectors) if j != i]
            averages.append(sum(sims) / max(1, len(sims)))
        medoid = vectors[max(range(len(vectors)), key=lambda i: averages[i])]
        inliers = [v for v in vectors if _dot(medoid, v) >= 0.42]
        if len(inliers) < max(2, len(vectors) // 2):
            inliers = vectors
        vector = _mean_normalized(inliers) or medoid

        color_values = [item.color for item in evidence if item.color]
        color = _mean_normalized(color_values) if color_values else None
        return vector, color, len(evidence)

    def _add_evidence(
        self,
        key: LocalKey,
        vector: Vector,
        color: Vector | None,
        now: float,
        quality: float,
    ) -> tuple[Vector, Vector | None, int]:
        evidence = self.local_evidence.get(key)
        if evidence is None:
            evidence = deque(maxlen=8)
            self.local_evidence[key] = evidence
        evidence.append(EvidenceItem(vector, color, now, float(quality)))
        agg_vector, agg_color, count = self._track_prototype(key)
        return agg_vector or vector, agg_color if agg_color is not None else color, count

    @staticmethod
    def _score_gallery(
        items: list[GalleryItem],
        vector: Vector,
        color: Vector | None,
    ) -> tuple[float, float, float]:
        if not items:
            return -1.0, -1.0, -1.0
        reid_sims = sorted((_dot(vector, item.vector) for item in items), reverse=True)
        top = reid_sims[: min(5, len(reid_sims))]
        best = top[0]
        top_mean = sum(top) / len(top)
        centroid = _mean_normalized([item.vector for item in items])
        centroid_score = _dot(vector, centroid) if centroid else top_mean
        reid_score = 0.50 * best + 0.32 * top_mean + 0.18 * centroid_score

        color_score = -1.0
        color_items = [item.color for item in items if item.color]
        if color and color_items:
            color_sims = sorted((_dot(color, item) for item in color_items), reverse=True)
            color_centroid = _mean_normalized(color_items)
            color_score = 0.65 * color_sims[0] + 0.35 * _dot(color, color_centroid)

        combined = reid_score if color_score < 0.0 else 0.92 * reid_score + 0.08 * color_score
        return combined, reid_score, color_score

    def _profile_items(
        self,
        profile: GlobalProfile,
        room_id: int | None = None,
        *,
        exclude_owner: LocalKey | None = None,
    ) -> list[GalleryItem]:
        if room_id is None:
            items = list(profile.gallery)
        else:
            memory = profile.rooms.get(room_id)
            items = list(memory.gallery) if memory is not None else []
        if exclude_owner is not None:
            items = [item for item in items if item.owner != exclude_owner]
        return items

    def _active_bindings_for_gid(self, gid: int, now: float) -> list[tuple[LocalKey, LocalBinding]]:
        gid = self._resolve(gid)
        return [
            (key, binding)
            for key, binding in self.bindings.items()
            if self._resolve(binding.global_id) == gid and self._is_active(key, now)
        ]

    def _peer_compatibility(
        self,
        gid: int,
        key: LocalKey,
        vector: Vector,
        color: Vector | None,
        now: float,
    ) -> tuple[str, float]:
        source_id, _ = key
        room_id = self.room_of(source_id)
        peer_scores: list[float] = []

        for peer_key, _binding in self._active_bindings_for_gid(gid, now):
            if peer_key == key:
                continue
            peer_source, _ = peer_key
            if peer_source == source_id:
                return "incompatible", -1.0
            if self.room_of(peer_source) != room_id:
                return "incompatible", -1.0

            pair = frozenset((key, peer_key))
            if self.cannot_link.get(pair, 0.0) >= now:
                return "incompatible", -1.0

            peer_vector, peer_color, peer_count = self._track_prototype(peer_key)
            if peer_vector is None or peer_count < 2:
                continue
            reid = _dot(vector, peer_vector)
            score = reid
            if color and peer_color:
                score = 0.92 * reid + 0.08 * _dot(color, peer_color)
            peer_scores.append(score)
            if reid < self.peer_min_reid:
                self.cannot_link[pair] = now + 4.0
                self.stats["peer_reject"] += 1
                return "incompatible", reid

        if not peer_scores:
            return "none", -1.0
        return "compatible", min(peer_scores)

    def _hard_conflict(
        self,
        profile: GlobalProfile,
        key: LocalKey,
        vector: Vector,
        color: Vector | None,
        now: float,
    ) -> tuple[bool, str, float]:
        source_id, _ = key
        target_room = self.room_of(source_id)
        gid = self._resolve(profile.global_id)

        for other_key, _binding in self._active_bindings_for_gid(gid, now):
            if other_key == key:
                continue
            other_source, _ = other_key
            if other_source == source_id:
                return True, "same_camera", -1.0
            if self.room_of(other_source) != target_room:
                return True, "cross_room", -1.0

        peer_state, peer_score = self._peer_compatibility(gid, key, vector, color, now)
        if peer_state == "incompatible":
            return True, "peer_cannot_link", peer_score
        return False, peer_state, peer_score

    def _score_profile(
        self,
        profile: GlobalProfile,
        vector: Vector,
        color: Vector | None,
        source_id: int,
        now: float,
        *,
        exclude_owner: LocalKey | None = None,
    ) -> tuple[float, float, float, float, float, bool, str]:
        target_room = self.room_of(source_id)
        global_items = self._profile_items(profile, exclude_owner=exclude_owner)
        global_score, global_reid, global_color = self._score_gallery(global_items, vector, color)
        if global_score < -0.5:
            return -1.0, -1.0, -1.0, -1.0, self.transition_threshold, False, "empty"

        room_items = self._profile_items(profile, target_room, exclude_owner=exclude_owner)
        room_score, room_reid, room_color = self._score_gallery(room_items, vector, color)

        context = "transition"
        threshold = self.transition_threshold
        score = global_score
        covisible = False

        if int(source_id) == profile.last_source:
            context = "same_camera"
            threshold = self.same_camera_threshold
            if room_score >= 0.0:
                score = 0.72 * room_score + 0.28 * global_score
        elif target_room == profile.last_room:
            covisible = any(
                peer_key[0] != int(source_id)
                and self.room_of(peer_key[0]) == target_room
                for peer_key, _ in self._active_bindings_for_gid(profile.global_id, now)
            )
            context = "covisible" if covisible else "same_room"
            threshold = self.covisible_threshold if covisible else self.same_room_threshold
            if room_score >= 0.0:
                score = 0.80 * room_score + 0.20 * global_score
        elif room_score >= 0.0:
            context = "revisit"
            threshold = self.revisit_threshold
            score = 0.70 * room_score + 0.30 * global_score
        else:
            age = max(0.0, now - profile.last_seen)
            if age <= self.transition_bonus_window:
                score += 0.020 * (1.0 - age / max(1.0, self.transition_bonus_window))
            if age > self.max_transition_gap:
                threshold = max(self.strong_threshold, self.transition_threshold + 0.05)

        reid_score = (
            room_reid
            if room_reid >= 0.0 and context in {"same_camera", "same_room", "covisible", "revisit"}
            else global_reid
        )
        color_score = (
            room_color
            if room_color >= 0.0 and context in {"same_camera", "same_room", "covisible", "revisit"}
            else global_color
        )
        return score, reid_score, color_score, room_score, threshold, covisible, context

    def _rank_candidates(
        self,
        vector: Vector,
        color: Vector | None,
        key: LocalKey,
        now: float,
        *,
        exclude_gid: int | None = None,
    ) -> list[tuple[float, int, float, float, float, float, bool, str]]:
        source_id, _ = key
        rows: list[tuple[float, int, float, float, float, float, bool, str]] = []
        for raw_gid, profile in self.profiles.items():
            gid = self._resolve(raw_gid)
            if gid != raw_gid:
                continue
            if exclude_gid is not None and gid == self._resolve(exclude_gid):
                continue
            if now - profile.last_seen > self.profile_ttl:
                continue
            conflict, _reason, _peer = self._hard_conflict(profile, key, vector, color, now)
            if conflict:
                self.stats["rejected_conflict"] += 1
                continue
            score, reid, color_score, room_score, threshold, covisible, context = self._score_profile(
                profile, vector, color, source_id, now, exclude_owner=key
            )
            rows.append((score, gid, threshold, reid, color_score, room_score, covisible, context))
        rows.sort(reverse=True, key=lambda row: row[0])
        return rows

    def _candidate_decision(
        self,
        vector: Vector,
        color: Vector | None,
        key: LocalKey,
        now: float,
        *,
        exclude_gid: int | None = None,
    ) -> tuple[int | None, float, float, float, float, float, float, bool, str, bool]:
        rows = self._rank_candidates(vector, color, key, now, exclude_gid=exclude_gid)
        if not rows:
            return None, -1.0, -1.0, self.transition_threshold, -1.0, -1.0, -1.0, False, "none", False
        best = rows[0]
        second = rows[1][0] if len(rows) > 1 else -1.0
        score, gid, threshold, reid, color_score, room_score, covisible, context = best
        accepted = score >= threshold and (second < 0.0 or score - second >= self.min_margin)

        self.stats["last_best_milli"] = int(round(score * 1000))
        self.stats["last_second_milli"] = int(round(second * 1000))
        self.stats["last_threshold_milli"] = int(round(threshold * 1000))
        self.stats["last_reid_milli"] = int(round(reid * 1000))
        self.stats["last_color_milli"] = int(round(color_score * 1000))
        self.stats["last_room_milli"] = int(round(room_score * 1000))
        self.stats["last_context"] = context
        return gid, score, second, threshold, reid, color_score, room_score, covisible, context, accepted

    def _rebuild_profile(self, profile: GlobalProfile) -> None:
        items = list(profile.gallery)
        profile.centroid = _mean_normalized([item.vector for item in items])
        colors = [item.color for item in items if item.color]
        profile.color_centroid = _mean_normalized(colors) if colors else None
        profile.rooms = {}
        profile.sample_count = len(items)
        if items:
            latest = max(items, key=lambda item: item.seen_at)
            profile.last_seen = latest.seen_at
            profile.last_source = latest.source_id
            profile.last_room = latest.room_id
        for item in items:
            room = profile.rooms.get(item.room_id)
            if room is None:
                room = RoomMemory(
                    room_id=item.room_id,
                    gallery=deque(maxlen=self.room_gallery_limit),
                    first_seen=item.seen_at,
                    last_seen=item.seen_at,
                )
                profile.rooms[item.room_id] = room
            room.gallery.append(item)
            room.samples += 1
            room.last_seen = max(room.last_seen, item.seen_at)
        for room in profile.rooms.values():
            room.centroid = _mean_normalized([item.vector for item in room.gallery])
            room_colors = [item.color for item in room.gallery if item.color]
            room.color_centroid = _mean_normalized(room_colors) if room_colors else None

    def _remove_owner_contributions(self, gid: int, key: LocalKey) -> None:
        gid = self._resolve(gid)
        profile = self.profiles.get(gid)
        if profile is None:
            return
        before = len(profile.gallery)
        kept = [item for item in profile.gallery if item.owner != key]
        if len(kept) == before:
            return
        profile.gallery = deque(kept[-self.gallery_limit :], maxlen=self.gallery_limit)
        self._rebuild_profile(profile)
        self.stats["rollbacks"] += before - len(kept)
        self._remove_orphan_profile_if_safe(gid)

    def _remove_orphan_profile_if_safe(self, gid: int) -> None:
        gid = self._resolve(gid)
        profile = self.profiles.get(gid)
        if profile is None or profile.gallery or profile.known_name:
            return
        if any(self._resolve(binding.global_id) == gid for binding in self.bindings.values()):
            return
        self.profiles.pop(gid, None)
        self.stats["orphan_profiles_removed"] += 1

    def _commit_to_profile(
        self,
        binding: LocalBinding,
        key: LocalKey,
        vector: Vector,
        color: Vector | None,
        source_id: int,
        now: float,
        quality: float,
        *,
        force: bool = False,
    ) -> None:
        if binding.state == "provisional":
            self.stats["quarantined"] += 1
            return
        if not force and binding.last_committed_at and now - binding.last_committed_at < self.commit_interval:
            return
        gid = self._resolve(binding.global_id)
        profile = self.profiles.get(gid)
        if profile is None:
            return
        room_id = self.room_of(source_id)
        item = GalleryItem(
            vector=vector,
            color=color,
            source_id=int(source_id),
            room_id=room_id,
            seen_at=now,
            quality=float(quality),
            owner=key,
        )

        same_owner = [entry for entry in profile.gallery if entry.owner == key]
        if same_owner and now - same_owner[-1].seen_at < 0.45:
            if _dot(vector, same_owner[-1].vector) > 0.993:
                return

        profile.gallery.append(item)
        profile.last_seen = now
        profile.last_source = int(source_id)
        profile.last_room = room_id
        binding.last_committed_at = now
        self._rebuild_profile(profile)
        self.stats["committed"] += 1

    def _create_anchor(
        self,
        key: LocalKey,
        vector: Vector,
        color: Vector | None,
        bbox: BBox | None,
        now: float,
        quality: float,
    ) -> LocalBinding:
        source_id, _ = key
        profile = self._new_profile(source_id, now)
        binding = LocalBinding(
            global_id=profile.global_id,
            first_seen=now,
            last_seen=now,
            last_source=source_id,
            last_room=self.room_of(source_id),
            last_bbox=bbox,
            state="anchor",
            confirm_votes=self.confirm_votes_required,
        )
        self.bindings[key] = binding
        self._commit_to_profile(binding, key, vector, color, source_id, now, quality, force=True)
        self.stats["confirmed"] += 1
        return binding

    def _find_same_camera_continuation(
        self,
        key: LocalKey,
        vector: Vector,
        color: Vector | None,
        bbox: BBox | None,
        now: float,
    ) -> tuple[int | None, float]:
        source_id, object_id = key
        best_gid: int | None = None
        best_total = -1.0
        for old_key, old_binding in self.bindings.items():
            sid, oid = old_key
            if sid != source_id or oid == object_id or self._is_active(old_key, now):
                continue
            age = now - old_binding.last_seen
            if age < 0.0 or age > self.continuation_gap:
                continue
            gid = self._resolve(old_binding.global_id)
            profile = self.profiles.get(gid)
            if profile is None:
                continue
            conflict, _reason, _peer = self._hard_conflict(profile, key, vector, color, now)
            if conflict:
                continue
            score, _r, _c, _room, _thr, _co, _ctx = self._score_profile(
                profile, vector, color, source_id, now, exclude_owner=key
            )
            if score < 0.0:
                continue
            distance = self._bbox_distance(old_binding.last_bbox, bbox)
            if distance > 1.6:
                continue
            spatial = max(0.0, 1.0 - distance / 1.6)
            total = 0.84 * score + 0.16 * spatial
            if total > best_total:
                best_total = total
                best_gid = gid
        if best_gid is not None and best_total >= self.continuation_threshold:
            return best_gid, best_total
        return None, best_total

    def _current_score(
        self,
        binding: LocalBinding,
        key: LocalKey,
        vector: Vector,
        color: Vector | None,
        now: float,
    ) -> tuple[float, float, bool, str, float]:
        gid = self._resolve(binding.global_id)
        profile = self.profiles.get(gid)
        if profile is None:
            return -1.0, 1.0, True, "missing", -1.0
        conflict, reason, peer_score = self._hard_conflict(profile, key, vector, color, now)
        if conflict:
            return -1.0, 1.0, True, reason, peer_score
        score, reid, _color, _room, threshold, _co, context = self._score_profile(
            profile, vector, color, key[0], now, exclude_owner=key
        )
        self.stats["last_current_milli"] = int(round(score * 1000))
        return score, threshold, False, context, reid

    def _switch_binding(
        self,
        binding: LocalBinding,
        key: LocalKey,
        new_gid: int,
        now: float,
        *,
        provisional: bool = True,
    ) -> None:
        old_gid = self._resolve(binding.global_id)
        new_gid = self._resolve(new_gid)
        if old_gid == new_gid:
            return
        self._remove_owner_contributions(old_gid, key)
        binding.global_id = new_gid
        binding.state = "provisional" if provisional else "confirmed"
        binding.confirm_votes = 1 if provisional else self.confirm_votes_required
        binding.bad_votes = 0
        binding.switch_candidate = None
        binding.switch_votes = 0
        binding.last_committed_at = 0.0
        self.stats["reassigned"] += 1
        self.stats["corrections"] += 1

    def _correct_to_new_anchor(
        self,
        binding: LocalBinding,
        key: LocalKey,
        vector: Vector,
        color: Vector | None,
        bbox: BBox | None,
        now: float,
        quality: float,
    ) -> None:
        old_gid = self._resolve(binding.global_id)
        self._remove_owner_contributions(old_gid, key)
        source_id, _ = key
        profile = self._new_profile(source_id, now)
        binding.global_id = profile.global_id
        binding.state = "anchor"
        binding.confirm_votes = self.confirm_votes_required
        binding.bad_votes = 0
        binding.switch_candidate = None
        binding.switch_votes = 0
        binding.last_committed_at = 0.0
        binding.last_bbox = bbox
        self._commit_to_profile(binding, key, vector, color, source_id, now, quality, force=True)
        self.stats["corrections"] += 1
        self.stats["confirmed"] += 1

    def _reassess_binding(
        self,
        binding: LocalBinding,
        key: LocalKey,
        vector: Vector,
        color: Vector | None,
        bbox: BBox | None,
        now: float,
        quality: float,
    ) -> None:
        current_gid = self._resolve(binding.global_id)
        binding.global_id = current_gid
        current_score, current_thr, hard_bad, _current_ctx, current_reid = self._current_score(
            binding, key, vector, color, now
        )
        (
            alt_gid,
            alt_score,
            alt_second,
            alt_thr,
            alt_reid,
            _alt_color,
            _alt_room,
            _alt_covis,
            _alt_ctx,
            alt_accepted,
        ) = self._candidate_decision(vector, color, key, now, exclude_gid=current_gid)

        peer_state, peer_score = self._peer_compatibility(current_gid, key, vector, color, now)
        peer_confirm_ok = peer_state != "compatible" or peer_score >= self.peer_confirm_reid

        current_valid = (
            not hard_bad
            and current_score >= current_thr
            and current_reid >= max(0.46, current_thr - 0.10)
            and peer_confirm_ok
        )

        alt_clearly_better = (
            alt_accepted
            and alt_gid is not None
            and (
                hard_bad
                or current_score < 0.0
                or alt_score >= current_score + self.switch_margin
            )
        )

        if binding.state == "provisional":
            self.stats["quarantined"] += 1
            if alt_clearly_better:
                if binding.switch_candidate == alt_gid:
                    binding.switch_votes += 1
                else:
                    binding.switch_candidate = alt_gid
                    binding.switch_votes = 1
                if binding.switch_votes >= self.correction_votes_required:
                    self._switch_binding(binding, key, int(alt_gid), now, provisional=True)
                return

            if current_valid:
                binding.confirm_votes += 1
                binding.bad_votes = 0
                binding.switch_candidate = None
                binding.switch_votes = 0
                if binding.confirm_votes >= self.confirm_votes_required:
                    binding.state = "confirmed"
                    self.stats["confirmed"] += 1
                    self._commit_to_profile(
                        binding, key, vector, color, key[0], now, quality, force=True
                    )
                return

            binding.bad_votes += 1
            if binding.bad_votes >= self.correction_votes_required:
                if alt_accepted and alt_gid is not None:
                    self._switch_binding(binding, key, int(alt_gid), now, provisional=True)
                else:
                    self._correct_to_new_anchor(binding, key, vector, color, bbox, now, quality)
            return

        if alt_clearly_better:
            if binding.switch_candidate == alt_gid:
                binding.switch_votes += 1
            else:
                binding.switch_candidate = alt_gid
                binding.switch_votes = 1
        else:
            binding.switch_candidate = None
            binding.switch_votes = 0

        weak_current = hard_bad or (
            binding.state != "anchor"
            and current_score >= 0.0
            and current_score < current_thr - self.release_delta
        )
        binding.bad_votes = binding.bad_votes + 1 if weak_current else 0

        if binding.switch_votes >= self.correction_votes_required and alt_gid is not None:
            self._switch_binding(binding, key, int(alt_gid), now, provisional=True)
            return

        if binding.bad_votes >= self.correction_votes_required:
            if alt_accepted and alt_gid is not None:
                self._switch_binding(binding, key, int(alt_gid), now, provisional=True)
            elif binding.state != "anchor":
                self._correct_to_new_anchor(binding, key, vector, color, bbox, now, quality)
            return

        if binding.state in {"confirmed", "anchor"} and not hard_bad:
            self._commit_to_profile(binding, key, vector, color, key[0], now, quality)

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
            detector_conf = max(0.0, float(row.get("confidence", 0.0) or 0.0))
            tracker_conf = max(0.0, float(row.get("tracker_confidence", 0.0) or 0.0))
            quality = max(detector_conf, tracker_conf)

            agg_vector, agg_color, evidence_count = self._add_evidence(
                key, vector, color, now, quality
            )
            self.stats["observations"] += 1

            binding = self.bindings.get(key)
            if binding is not None and now - binding.last_seen <= self.binding_ttl:
                binding.last_seen = now
                binding.last_source = source_id
                binding.last_room = self.room_of(source_id)
                if bbox is not None:
                    binding.last_bbox = bbox
                self._reassess_binding(
                    binding, key, agg_vector, agg_color, bbox, now, quality
                )
                continue

            if evidence_count < self.min_new_track_samples:
                self.stats["pending_tracklet"] += 1
                continue

            continuation_gid, _cont_score = self._find_same_camera_continuation(
                key, agg_vector, agg_color, bbox, now
            )
            if continuation_gid is not None:
                binding = LocalBinding(
                    global_id=continuation_gid,
                    first_seen=now,
                    last_seen=now,
                    last_source=source_id,
                    last_room=self.room_of(source_id),
                    last_bbox=bbox,
                    state="provisional",
                    confirm_votes=1,
                )
                self.bindings[key] = binding
                self.stats["continuation_match"] += 1
                self.stats["provisional"] += 1
                continue

            (
                best_gid,
                best_score,
                second_score,
                threshold,
                reid_score,
                _color_score,
                room_score,
                covisible,
                context,
                accepted,
            ) = self._candidate_decision(agg_vector, agg_color, key, now)

            if accepted and best_gid is not None:
                if context == "covisible" and reid_score < self.peer_min_reid:
                    accepted = False

            if accepted and best_gid is not None:
                binding = LocalBinding(
                    global_id=best_gid,
                    first_seen=now,
                    last_seen=now,
                    last_source=source_id,
                    last_room=self.room_of(source_id),
                    last_bbox=bbox,
                    state="provisional",
                    confirm_votes=1,
                    last_score=best_score,
                    last_threshold=threshold,
                )
                self.bindings[key] = binding
                self.stats["direct_match"] += 1
                self.stats["provisional"] += 1
                if covisible:
                    self.stats["covisible_match"] += 1
                if context == "revisit":
                    self.stats["revisit_match"] += 1
                elif context == "transition":
                    self.stats["transition_match"] += 1
                if room_score >= 0.0 and context in {"same_room", "covisible", "revisit"}:
                    self.stats["room_memory_match"] += 1
                if best_score >= self.strong_threshold:
                    self.stats["strong_match"] += 1
                continue

            if best_gid is not None:
                self.stats["ambiguous_new"] += 1
            self._create_anchor(key, agg_vector, agg_color, bbox, now, quality)

    def _expire(self, now: float) -> None:
        stale_bindings = [
            key
            for key, binding in self.bindings.items()
            if now - binding.last_seen > self.binding_ttl
        ]
        for key in stale_bindings:
            self.bindings.pop(key, None)
            self.local_evidence.pop(key, None)

        stale_profiles = [
            gid
            for gid, profile in self.profiles.items()
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
        provisional = sum(1 for binding in self.bindings.values() if binding.state == "provisional")
        confirmed = len(self.bindings) - provisional
        return {
            "global_count": len(self.profiles),
            "local_bindings": len(self.bindings),
            "room_map": dict(self.room_map),
            "active_tracks": len(self.active_seen),
            "room_memories": sum(len(profile.rooms) for profile in self.profiles.values()),
            "provisional_bindings": provisional,
            "confirmed_bindings": confirmed,
            "cannot_links": len(self.cannot_link),
            "stats": dict(self.stats),
            "profiles": [
                {
                    "global_id": gid,
                    "label": self.label_for(gid),
                    "samples": profile.sample_count,
                    "gallery": len(profile.gallery),
                    "last_source": profile.last_source,
                    "last_room": profile.last_room,
                    "owners": len({item.owner for item in profile.gallery if item.owner is not None}),
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
