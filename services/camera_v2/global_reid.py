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
    # cameras.yaml source order: CAM-01,02,03,04,05,06.
    # Physical room pairs: 01+02, 03+06, 05+04.
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
    color_centroid: Vector | None = None
    sample_count: int = 0
    known_name: str = ""


@dataclass
class LocalBinding:
    global_id: int
    first_seen: float
    last_seen: float
    last_source: int
    last_bbox: BBox | None = None
    merge_candidate: int | None = None
    merge_votes: int = 0


class GlobalReIDManager:
    """Cross-camera identity manager for fixed, partially-overlapping CCTV views.

    ReIdentificationNet is only one cue. Decisions also use physical room topology,
    *currently active* NvDCF tracks, short local-track consensus, same-camera spatial
    continuity after a tracker-ID reset, and a weak clothing-colour descriptor.
    Appearance never changes tracker geometry; it only maps local IDs to Global IDs.
    """

    def __init__(self) -> None:
        self.match_threshold = float(os.environ.get("CAMERA_V2_REID_MATCH", "0.68"))
        self.strong_threshold = float(os.environ.get("CAMERA_V2_REID_STRONG", "0.78"))
        self.same_room_threshold = float(os.environ.get("CAMERA_V2_REID_SAME_ROOM", "0.60"))
        self.covisible_threshold = float(os.environ.get("CAMERA_V2_REID_COVISIBLE", "0.56"))
        self.same_camera_threshold = float(os.environ.get("CAMERA_V2_REID_SAME_CAMERA", "0.62"))
        self.continuation_threshold = float(os.environ.get("CAMERA_V2_REID_CONTINUATION", "0.58"))
        self.min_margin = float(os.environ.get("CAMERA_V2_REID_MARGIN", "0.012"))
        self.merge_votes_required = max(1, int(os.environ.get("CAMERA_V2_REID_MERGE_VOTES", "2")))
        self.gallery_limit = max(8, int(os.environ.get("CAMERA_V2_REID_GALLERY", "32")))
        self.profile_ttl = float(os.environ.get("CAMERA_V2_REID_PROFILE_TTL", "1800"))
        self.binding_ttl = float(os.environ.get("CAMERA_V2_REID_BINDING_TTL", "15"))
        self.active_ttl = float(os.environ.get("CAMERA_V2_REID_ACTIVE_TTL", "0.45"))
        self.continuation_gap = float(os.environ.get("CAMERA_V2_REID_CONTINUATION_GAP", "2.5"))
        self.max_transition_gap = float(os.environ.get("CAMERA_V2_REID_MAX_TRANSITION", "180"))
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
            "strong_match": 0,
            "merged": 0,
            "ambiguous_new": 0,
            "rejected_conflict": 0,
            "last_best_milli": -1000,
            "last_second_milli": -1000,
            "last_threshold_milli": 0,
            "last_reid_milli": -1000,
            "last_color_milli": -1000,
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
        """Refresh actual NvDCF activity from each (possibly partial) mux batch."""
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
        stale = [key for key, seen in self.active_seen.items() if now - seen > max(2.0, self.active_ttl * 4.0)]
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

    def _score_profile(self, profile: GlobalProfile, vector: Vector, color: Vector | None) -> tuple[float, float, float]:
        if not profile.gallery:
            return -1.0, -1.0, -1.0
        reid_sims = sorted((_dot(vector, item.vector) for item in profile.gallery), reverse=True)
        top = reid_sims[: min(4, len(reid_sims))]
        best = top[0]
        top_mean = sum(top) / len(top)
        centroid_score = _dot(vector, profile.centroid) if profile.centroid else top_mean
        reid_score = 0.55 * best + 0.30 * top_mean + 0.15 * centroid_score

        color_score = -1.0
        if color and profile.color_centroid:
            color_sims = sorted(
                (_dot(color, item.color) for item in profile.gallery if item.color),
                reverse=True,
            )
            if color_sims:
                color_score = 0.65 * color_sims[0] + 0.35 * _dot(color, profile.color_centroid)

        # Clothing colour is a low-weight stabilizer only; ReID remains dominant.
        combined = reid_score if color_score < 0.0 else 0.88 * reid_score + 0.12 * color_score
        return combined, reid_score, color_score

    def _threshold_for(self, profile: GlobalProfile, source_id: int, now: float) -> tuple[float, bool]:
        source_id = int(source_id)
        if source_id == profile.last_source:
            return self.same_camera_threshold, False
        room = self.room_of(source_id)
        if room == profile.last_room:
            covisible = self._profile_covisible_same_room(profile, source_id, now)
            return (self.covisible_threshold if covisible else self.same_room_threshold), covisible
        if now - profile.last_seen > self.max_transition_gap:
            return max(self.strong_threshold, self.match_threshold + 0.05), False
        return self.match_threshold, False

    def _best_candidate(
        self,
        vector: Vector,
        color: Vector | None,
        source_id: int,
        object_id: int,
        now: float,
        *,
        exclude: int | None = None,
    ) -> tuple[int | None, float, float, float, float, float, bool]:
        scored: list[tuple[float, float, int, float, float, bool]] = []
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
            score, reid_score, color_score = self._score_profile(profile, vector, color)
            threshold, covisible = self._threshold_for(profile, source_id, now)
            scored.append((score, threshold, gid, reid_score, color_score, covisible))

        if conflict_seen:
            self.stats["rejected_conflict"] += 1
        if not scored:
            return None, -1.0, -1.0, self.match_threshold, -1.0, -1.0, False
        scored.sort(reverse=True)
        best_score, threshold, best_gid, reid_score, color_score, covisible = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -1.0
        self.stats["last_best_milli"] = int(round(best_score * 1000))
        self.stats["last_second_milli"] = int(round(second_score * 1000))
        self.stats["last_threshold_milli"] = int(round(threshold * 1000))
        self.stats["last_reid_milli"] = int(round(reid_score * 1000))
        self.stats["last_color_milli"] = int(round(color_score * 1000))
        return best_gid, best_score, second_score, threshold, reid_score, color_score, covisible

    def _aggregate_local(self, key: LocalKey, vector: Vector, color: Vector | None, now: float) -> tuple[Vector, Vector | None]:
        evidence = self.local_evidence.get(key)
        if evidence is None:
            evidence = deque(maxlen=5)
            self.local_evidence[key] = evidence
        evidence.append((vector, color, now))
        vectors = [item[0] for item in list(evidence)[-4:]]
        colors = [item[1] for item in list(evidence)[-4:] if item[1]]
        return _mean_normalized(vectors) or vector, _mean_normalized(colors) if colors else color

    def _update_profile(
        self,
        profile: GlobalProfile,
        vector: Vector,
        color: Vector | None,
        source_id: int,
        now: float,
        quality: float,
    ) -> None:
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
            profile.gallery.append(GalleryItem(vector, color, int(source_id), now, float(quality)))

        if profile.centroid is None:
            profile.centroid = vector
        else:
            alpha = 0.16 if quality >= 0.45 else 0.09
            mixed = tuple((1.0 - alpha) * a + alpha * b for a, b in zip(profile.centroid, vector))
            profile.centroid = _normalize(mixed) or profile.centroid
        if color:
            if profile.color_centroid is None:
                profile.color_centroid = color
            else:
                mixed_color = tuple(0.88 * a + 0.12 * b for a, b in zip(profile.color_centroid, color))
                profile.color_centroid = _normalize(mixed_color) or profile.color_centroid

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
            score, _reid, _color = self._score_profile(profile, vector, color)
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
        parent.color_centroid = _mean_normalized([item.color for item in parent.gallery if item.color]) or parent.color_centroid
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
            if binding.global_id == child_gid:
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
        if current.sample_count > 24 and now - current.created_at > 20.0:
            return
        candidate_gid, score, second, threshold, _r, _c, _co = self._best_candidate(
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
            agg_vector, agg_color = self._aggregate_local(key, vector, color, now)
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
            covisible = False
            if continuation_gid is not None:
                profile = self.profiles[continuation_gid]
                accepted = True
                self.stats["continuation_match"] += 1
            else:
                best_gid, best_score, second_score, threshold, _r, _c, covisible = self._best_candidate(
                    agg_vector, agg_color, source_id, object_id, now
                )
                accepted = (
                    best_gid is not None
                    and best_score >= threshold
                    and best_score - second_score >= self.min_margin
                )
                if accepted:
                    profile = self.profiles[best_gid]
                    self.stats["direct_match"] += 1
                    if covisible:
                        self.stats["covisible_match"] += 1
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
                last_bbox=bbox,
            )
            self.bindings[key] = binding
            self._update_profile(profile, vector, color, source_id, now, quality)
            if not accepted:
                self._consider_reconciliation(binding, agg_vector, agg_color, source_id, object_id, now)

    def _expire(self, now: float) -> None:
        stale_bindings = [key for key, binding in self.bindings.items() if now - binding.last_seen > self.binding_ttl]
        for key in stale_bindings:
            self.bindings.pop(key, None)
            self.local_evidence.pop(key, None)
        stale_profiles = [gid for gid, profile in self.profiles.items() if now - profile.last_seen > self.profile_ttl]
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
