from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


STATE_COLLECTING = "COLLECTING"
STATE_TENTATIVE = "TENTATIVE"
STATE_CONFIRMED = "CONFIRMED"
STATE_LOST = "LOST"
STATE_SUSPECT = "SUSPECT"


def normalize(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("invalid zero ReID embedding")
    return value / norm


def cosine(a, b) -> float:
    return float(np.dot(normalize(a), normalize(b)))


def bbox_center(box: tuple[float, float, float, float] | None) -> tuple[float, float] | None:
    if not box:
        return None
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass
class EvidenceEmbedding:
    embedding: np.ndarray
    quality: float
    captured_at: float
    camera_id: str
    local_id: int
    room_id: str
    bbox: tuple[float, float, float, float]
    jpeg: bytes | None = None

    def __post_init__(self) -> None:
        self.embedding = normalize(self.embedding)
        self.quality = float(max(0.0, min(1.0, self.quality)))
        self.local_id = int(self.local_id)

    @property
    def origin_key(self) -> tuple[str, int]:
        return (self.camera_id, self.local_id)


@dataclass
class CandidateScore:
    global_id: int
    score: float
    raw_similarity: float
    prototype_similarity: float
    best_similarity: float
    margin: float = 0.0
    reason: str = "gallery"


@dataclass
class TrackletState:
    camera_id: str
    local_id: int
    room_id: str
    created_at: float
    last_seen: float
    last_bbox: tuple[float, float, float, float] | None = None
    samples: list[EvidenceEmbedding] = field(default_factory=list)
    state: str = STATE_COLLECTING
    global_id: int | None = None
    assigned_score: float = 0.0
    assigned_margin: float = 0.0
    assigned_at: float = 0.0
    positive_votes: int = 0
    negative_votes: int = 0
    evidence_round: int = 0
    qwen_verdict: str = "UNCERTAIN"
    qwen_confidence: float = 0.0
    qwen_version: int = -1
    last_candidate_id: int | None = None
    rejected_global_until: dict[int, float] = field(default_factory=dict)
    contradiction_votes: int = 0

    @property
    def key(self) -> tuple[str, int]:
        return (self.camera_id, self.local_id)


@dataclass
class GlobalIdentity:
    global_id: int
    created_at: float
    last_seen: float
    last_camera: str
    last_room: str
    confirmed: bool = False
    gallery: list[EvidenceEmbedding] = field(default_factory=list)
    quarantine: list[EvidenceEmbedding] = field(default_factory=list)
    prototype: np.ndarray | None = None
    active_tracks: set[tuple[str, int]] = field(default_factory=set)
    suspect: bool = False


class GlobalIdentityCore:
    """Conservative multi-camera identity state machine.

    Local NvDCF IDs remain authoritative inside a camera. This class only owns the
    cross-camera identity layer. It deliberately prefers a temporary split over a
    false merge: topology conflicts, runner-up margin, multi-shot confirmation,
    quarantine galleries, VLM disagreement and reversible bindings are all handled
    before a tracklet may modify a confirmed Global-ID gallery.
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = dict(config or {})
        self.max_track_samples = max(3, int(cfg.get("max_track_samples", 10)))
        self.top_k = max(2, min(5, int(cfg.get("top_k", 3))))
        self.gallery_size = max(self.top_k, int(cfg.get("gallery_size", 12)))
        self.min_samples = max(2, int(cfg.get("min_samples", 3)))
        self.new_identity_confirm_samples = max(
            self.min_samples, int(cfg.get("new_identity_confirm_samples", 4))
        )
        self.provisional_similarity = float(cfg.get("provisional_similarity", 0.66))
        self.confirm_similarity = float(cfg.get("confirm_similarity", 0.76))
        self.strong_similarity = float(cfg.get("strong_similarity", 0.84))
        self.reject_similarity = float(cfg.get("reject_similarity", 0.56))
        self.qwen_rescue_similarity = float(cfg.get("qwen_rescue_similarity", 0.58))
        self.min_margin = max(0.0, float(cfg.get("min_margin", 0.04)))
        self.strong_margin = max(0.0, float(cfg.get("strong_margin", 0.025)))
        self.confirm_votes = max(1, int(cfg.get("confirm_votes", 2)))
        self.active_timeout = max(0.4, float(cfg.get("active_timeout_sec", 1.8)))
        self.missing_active_grace = max(
            0.05, float(cfg.get("missing_active_grace_sec", 0.35))
        )
        self.lost_timeout = max(self.active_timeout, float(cfg.get("lost_timeout_sec", 8.0)))
        self.gallery_ttl = max(30.0, float(cfg.get("gallery_ttl_sec", 21600.0)))
        self.same_camera_reconnect_sec = max(
            0.5, float(cfg.get("same_camera_reconnect_sec", 8.0))
        )
        self.same_camera_reconnect_similarity = float(
            cfg.get("same_camera_reconnect_similarity", 0.62)
        )
        self.same_camera_reconnect_confirm_similarity = float(
            cfg.get("same_camera_reconnect_confirm_similarity", 0.72)
        )
        self.min_room_transition_sec = max(
            0.0, float(cfg.get("min_room_transition_sec", 0.8))
        )
        self.qwen_same_confidence = float(cfg.get("qwen_same_confidence", 0.70))
        self.qwen_different_confidence = float(
            cfg.get("qwen_different_confidence", 0.75)
        )
        self.qwen_negative_ttl = max(2.0, float(cfg.get("qwen_negative_ttl_sec", 20.0)))
        self.rollback_contradictions = max(2, int(cfg.get("rollback_contradictions", 2)))

        self.camera_rooms = {
            str(k): str(v) for k, v in dict(cfg.get("camera_rooms") or {}).items()
        }
        self._lock = threading.RLock()
        self._tracks: dict[tuple[str, int], TrackletState] = {}
        self._globals: dict[int, GlobalIdentity] = {}
        self._next_global_id = 1
        self._metrics = {
            "new_globals": 0,
            "candidate_matches": 0,
            "confirmed_matches": 0,
            "hard_conflicts": 0,
            "margin_rejects": 0,
            "qwen_same": 0,
            "qwen_different": 0,
            "qwen_uncertain": 0,
            "qwen_rescues": 0,
            "rollbacks": 0,
            "gallery_commits": 0,
            "self_outlier_rejects": 0,
        }

    def room_for_camera(self, camera_id: str, room_id: str | None = None) -> str:
        if room_id:
            return str(room_id)
        return self.camera_rooms.get(str(camera_id), "")

    @staticmethod
    def _weighted_prototype(samples: Iterable[EvidenceEmbedding]) -> np.ndarray:
        rows = list(samples)
        if not rows:
            raise ValueError("cannot build prototype from empty evidence")
        matrix = np.stack([sample.embedding for sample in rows], axis=0)
        weights = np.asarray([max(0.05, sample.quality) for sample in rows], dtype=np.float32)
        return normalize(np.average(matrix, axis=0, weights=weights))

    def _diverse_top(
        self, samples: list[EvidenceEmbedding], count: int | None = None
    ) -> list[EvidenceEmbedding]:
        if not samples:
            return []
        count = min(len(samples), int(count or self.top_k))
        remaining = list(range(len(samples)))
        first = max(remaining, key=lambda i: samples[i].quality)
        selected = [first]
        remaining.remove(first)
        while remaining and len(selected) < count:
            def utility(index: int) -> float:
                similarity = max(
                    cosine(samples[index].embedding, samples[j].embedding)
                    for j in selected
                )
                diversity = max(0.0, min(1.0, 1.0 - similarity))
                return samples[index].quality * 0.68 + diversity * 0.32

            best = max(remaining, key=utility)
            selected.append(best)
            remaining.remove(best)
        return [samples[i] for i in selected]

    def _candidate_evidence(self, identity: GlobalIdentity) -> list[EvidenceEmbedding]:
        if identity.gallery:
            return identity.gallery
        return identity.quarantine

    def _hard_candidate_allowed(
        self,
        identity: GlobalIdentity,
        track: TrackletState,
        now: float,
    ) -> bool:
        if now - identity.last_seen > self.gallery_ttl:
            return False
        if track.rejected_global_until.get(identity.global_id, 0.0) > now:
            return False

        for active_key in list(identity.active_tracks):
            active = self._tracks.get(active_key)
            if active is None or now - active.last_seen > self.active_timeout:
                identity.active_tracks.discard(active_key)
                continue
            if active.key == track.key:
                continue
            if active.camera_id == track.camera_id:
                self._metrics["hard_conflicts"] += 1
                return False
            if active.room_id and track.room_id and active.room_id != track.room_id:
                self._metrics["hard_conflicts"] += 1
                return False

        if (
            identity.last_room
            and track.room_id
            and identity.last_room != track.room_id
            and now - identity.last_seen < self.min_room_transition_sec
        ):
            self._metrics["hard_conflicts"] += 1
            return False
        return True

    def _candidate_score(
        self,
        track: TrackletState,
        identity: GlobalIdentity,
        now: float,
    ) -> CandidateScore | None:
        candidate_gallery = self._candidate_evidence(identity)
        if not candidate_gallery or not self._hard_candidate_allowed(identity, track, now):
            return None
        new_samples = self._diverse_top(track.samples)
        old_samples = self._diverse_top(candidate_gallery, min(self.gallery_size, 6))
        if not new_samples or not old_samples:
            return None

        new_matrix = np.stack([s.embedding for s in new_samples], axis=0)
        old_matrix = np.stack([s.embedding for s in old_samples], axis=0)
        matrix = new_matrix @ old_matrix.T
        best_per_new = np.max(matrix, axis=1)
        q = np.asarray([max(0.05, s.quality) for s in new_samples], dtype=np.float32)
        raw = float(np.average(best_per_new, weights=q))
        best = float(np.max(matrix))
        new_proto = self._weighted_prototype(new_samples)
        old_proto = (
            identity.prototype
            if identity.prototype is not None
            else self._weighted_prototype(old_samples)
        )
        proto = cosine(new_proto, old_proto)
        score = raw * 0.55 + proto * 0.30 + best * 0.15
        reason = "gallery"

        gap = max(0.0, now - identity.last_seen)
        if identity.last_camera == track.camera_id and gap <= self.same_camera_reconnect_sec:
            score += 0.035
            reason = "same_camera_reconnect"
        elif identity.last_room and identity.last_room == track.room_id and gap <= self.active_timeout:
            score += 0.020
            reason = "same_room_overlap"
        score = max(-1.0, min(1.0, score))
        return CandidateScore(identity.global_id, score, raw, proto, best, reason=reason)

    def rank_candidates(
        self, track: TrackletState, now: float | None = None
    ) -> list[CandidateScore]:
        now = time.monotonic() if now is None else float(now)
        ranked: list[CandidateScore] = []
        for identity in self._globals.values():
            row = self._candidate_score(track, identity, now)
            if row is not None:
                ranked.append(row)
        ranked.sort(key=lambda row: row.score, reverse=True)
        for index, row in enumerate(ranked):
            runner = ranked[index + 1].score if index + 1 < len(ranked) else -1.0
            row.margin = row.score - runner
        return ranked

    def _new_identity(self, track: TrackletState, now: float) -> GlobalIdentity:
        global_id = self._next_global_id
        self._next_global_id += 1
        selected = self._diverse_top(track.samples)
        identity = GlobalIdentity(
            global_id=global_id,
            created_at=now,
            last_seen=now,
            last_camera=track.camera_id,
            last_room=track.room_id,
            confirmed=False,
            quarantine=list(selected),
            prototype=self._weighted_prototype(selected),
        )
        identity.active_tracks.add(track.key)
        self._globals[global_id] = identity
        track.global_id = global_id
        track.state = STATE_TENTATIVE
        track.assigned_score = 1.0
        track.assigned_margin = 1.0
        track.assigned_at = now
        self._metrics["new_globals"] += 1
        return identity

    def _assign_candidate(
        self,
        track: TrackletState,
        candidate: CandidateScore,
        now: float,
    ) -> None:
        identity = self._globals[candidate.global_id]
        track.global_id = identity.global_id
        track.state = STATE_TENTATIVE
        track.assigned_score = candidate.score
        track.assigned_margin = candidate.margin
        track.assigned_at = now
        track.last_candidate_id = identity.global_id
        track.positive_votes = max(1, track.positive_votes)
        identity.active_tracks.add(track.key)
        self._metrics["candidate_matches"] += 1

    def _rebuild_identity_gallery(self, identity: GlobalIdentity) -> None:
        identity.gallery = self._diverse_top(identity.gallery, self.gallery_size)
        identity.prototype = (
            self._weighted_prototype(identity.gallery) if identity.gallery else None
        )

    def _commit_track_to_gallery(self, track: TrackletState, now: float) -> None:
        if track.global_id is None:
            return
        identity = self._globals.get(track.global_id)
        if identity is None:
            return
        for sample in self._diverse_top(track.samples):
            if identity.gallery:
                best = max(cosine(sample.embedding, old.embedding) for old in identity.gallery)
                if best > 0.985:
                    continue
            identity.gallery.append(sample)
        self._rebuild_identity_gallery(identity)
        identity.quarantine = []
        identity.confirmed = True
        identity.suspect = False
        identity.last_seen = now
        identity.last_camera = track.camera_id
        identity.last_room = track.room_id
        identity.active_tracks.add(track.key)
        track.state = STATE_CONFIRMED
        track.contradiction_votes = 0
        self._metrics["gallery_commits"] += 1

    def _remove_track_contribution(self, identity: GlobalIdentity, key: tuple[str, int]) -> None:
        before = len(identity.gallery)
        identity.gallery = [row for row in identity.gallery if row.origin_key != key]
        if len(identity.gallery) != before:
            self._rebuild_identity_gallery(identity)
        identity.quarantine = [row for row in identity.quarantine if row.origin_key != key]

    def _detach_track(
        self,
        track: TrackletState,
        now: float,
        reject_gid: int | None = None,
        *,
        rollback_gallery: bool = False,
    ) -> None:
        old_gid = track.global_id
        if old_gid is not None:
            identity = self._globals.get(old_gid)
            if identity is not None:
                identity.active_tracks.discard(track.key)
                if rollback_gallery:
                    self._remove_track_contribution(identity, track.key)
        if reject_gid is not None:
            track.rejected_global_until[reject_gid] = now + self.qwen_negative_ttl
        track.global_id = None
        track.state = STATE_COLLECTING
        track.assigned_score = 0.0
        track.assigned_margin = 0.0
        track.positive_votes = 0
        track.negative_votes = 0
        track.last_candidate_id = None
        track.qwen_verdict = "UNCERTAIN"
        track.qwen_confidence = 0.0

    def _evaluate(self, track: TrackletState, now: float) -> dict:
        if len(track.samples) < self.min_samples:
            return {"action": "collect", "state": track.state, "global_id": track.global_id}

        track.evidence_round += 1
        if track.global_id is None:
            ranked = self.rank_candidates(track, now)
            best = ranked[0] if ranked else None
            if best is None or best.score < self.reject_similarity:
                identity = self._new_identity(track, now)
                if len(track.samples) >= self.new_identity_confirm_samples:
                    self._commit_track_to_gallery(track, now)
                return {"action": "new", "global_id": identity.global_id, "state": track.state}

            margin_needed = (
                self.strong_margin if best.score >= self.strong_similarity else self.min_margin
            )
            if best.score >= self.provisional_similarity and best.margin >= margin_needed:
                self._assign_candidate(track, best, now)
                return {
                    "action": "candidate_tentative",
                    "global_id": track.global_id,
                    "candidate": best.global_id,
                    "score": best.score,
                    "margin": best.margin,
                    "state": track.state,
                    "needs_qwen": True,
                    "evidence_version": track.evidence_round,
                }

            if best.score >= self.provisional_similarity:
                self._metrics["margin_rejects"] += 1
            if len(track.samples) >= self.max_track_samples:
                identity = self._new_identity(track, now)
                self._commit_track_to_gallery(track, now)
                return {
                    "action": "new_ambiguous",
                    "global_id": identity.global_id,
                    "state": track.state,
                }
            return {
                "action": "ambiguous",
                "candidate": best.global_id,
                "score": best.score,
                "margin": best.margin,
                "state": track.state,
                "needs_qwen": best.score >= self.qwen_rescue_similarity,
                "evidence_version": track.evidence_round,
            }

        identity = self._globals.get(track.global_id)
        if identity is None:
            self._detach_track(track, now)
            return {"action": "reset_missing_global", "state": track.state}

        if (
            not identity.confirmed
            and identity.created_at == track.assigned_at
            and track.assigned_score >= 0.999
        ):
            if len(track.samples) >= self.new_identity_confirm_samples:
                self._commit_track_to_gallery(track, now)
                return {
                    "action": "confirm_new",
                    "global_id": track.global_id,
                    "state": track.state,
                }
            return {
                "action": "new_tentative",
                "global_id": track.global_id,
                "state": track.state,
            }

        ranked = self.rank_candidates(track, now)
        current = next((row for row in ranked if row.global_id == track.global_id), None)
        if current is not None:
            track.assigned_score = current.score
            track.assigned_margin = current.margin
            margin_needed = (
                self.strong_margin if current.score >= self.strong_similarity else self.min_margin
            )
            if current.score >= self.confirm_similarity and current.margin >= margin_needed:
                track.positive_votes += 1
            elif (
                current.reason == "same_camera_reconnect"
                and current.score >= self.same_camera_reconnect_confirm_similarity
            ):
                track.positive_votes += 1
            elif current.score < self.reject_similarity:
                track.negative_votes += 1

        qwen_same = (
            track.qwen_verdict == "SAME"
            and track.qwen_confidence >= self.qwen_same_confidence
        )
        qwen_diff = (
            track.qwen_verdict == "DIFFERENT"
            and track.qwen_confidence >= self.qwen_different_confidence
        )

        if track.state == STATE_TENTATIVE:
            if qwen_diff:
                rejected = track.global_id
                self._detach_track(track, now, reject_gid=rejected)
                return {
                    "action": "qwen_reject",
                    "rejected_global": rejected,
                    "state": track.state,
                }
            if qwen_same and track.assigned_score >= self.provisional_similarity:
                self._commit_track_to_gallery(track, now)
                self._metrics["confirmed_matches"] += 1
                return {
                    "action": "qwen_confirm",
                    "global_id": track.global_id,
                    "state": track.state,
                }
            if track.positive_votes >= self.confirm_votes:
                self._commit_track_to_gallery(track, now)
                self._metrics["confirmed_matches"] += 1
                return {
                    "action": "vote_confirm",
                    "global_id": track.global_id,
                    "state": track.state,
                }

        elif track.state in {STATE_CONFIRMED, STATE_SUSPECT, STATE_LOST}:
            if qwen_diff and track.assigned_score < self.reject_similarity:
                track.contradiction_votes += 1
                track.state = STATE_SUSPECT
                identity.suspect = True
            elif qwen_same or track.assigned_score >= self.provisional_similarity:
                track.contradiction_votes = max(0, track.contradiction_votes - 1)
                if track.contradiction_votes == 0:
                    track.state = STATE_CONFIRMED
                    identity.suspect = False
            if track.contradiction_votes >= self.rollback_contradictions:
                rejected = track.global_id
                self._detach_track(
                    track,
                    now,
                    reject_gid=rejected,
                    rollback_gallery=True,
                )
                self._metrics["rollbacks"] += 1
                return {
                    "action": "rollback",
                    "rejected_global": rejected,
                    "state": track.state,
                }

        return {
            "action": "keep",
            "global_id": track.global_id,
            "state": track.state,
            "score": track.assigned_score,
            "votes": track.positive_votes,
            "needs_qwen": track.state == STATE_TENTATIVE,
            "evidence_version": track.evidence_round,
        }

    def observe_embedding(
        self,
        *,
        camera_id: str,
        local_id: int,
        embedding,
        quality: float,
        captured_at: float | None = None,
        room_id: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        jpeg: bytes | None = None,
    ) -> dict:
        now = time.monotonic() if captured_at is None else float(captured_at)
        camera_id = str(camera_id)
        room = self.room_for_camera(camera_id, room_id)
        box = tuple(float(v) for v in (bbox or (0.0, 0.0, 0.0, 0.0)))
        sample = EvidenceEmbedding(
            embedding=embedding,
            quality=quality,
            captured_at=now,
            camera_id=camera_id,
            local_id=int(local_id),
            room_id=room,
            bbox=box,
            jpeg=jpeg,
        )
        key = (camera_id, int(local_id))
        with self._lock:
            track = self._tracks.get(key)
            if track is None:
                track = TrackletState(camera_id, int(local_id), room, now, now)
                self._tracks[key] = track
            track.last_seen = max(track.last_seen, now)
            track.room_id = room or track.room_id
            track.last_bbox = box
            if track.state == STATE_LOST and track.global_id is not None:
                identity = self._globals.get(track.global_id)
                track.state = STATE_CONFIRMED if identity and identity.confirmed else STATE_TENTATIVE

            if track.samples:
                prototype = self._weighted_prototype(self._diverse_top(track.samples))
                similarity = cosine(sample.embedding, prototype)
                if len(track.samples) >= 2 and similarity < 0.40:
                    self._metrics["self_outlier_rejects"] += 1
                    return {"action": "self_outlier_reject", "similarity": similarity}

            track.samples.append(sample)
            if len(track.samples) > self.max_track_samples:
                track.samples = self._diverse_top(track.samples, self.max_track_samples)
            return self._evaluate(track, now)

    def observe_track_activity(
        self,
        camera_id: str,
        local_id: int,
        *,
        room_id: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        seen_at: float | None = None,
    ) -> None:
        now = time.monotonic() if seen_at is None else float(seen_at)
        key = (str(camera_id), int(local_id))
        with self._lock:
            track = self._tracks.get(key)
            if track is None:
                track = TrackletState(
                    str(camera_id),
                    int(local_id),
                    self.room_for_camera(camera_id, room_id),
                    now,
                    now,
                )
                self._tracks[key] = track
            track.last_seen = max(track.last_seen, now)
            if room_id:
                track.room_id = str(room_id)
            if bbox is not None:
                track.last_bbox = tuple(float(v) for v in bbox)
            if track.global_id is not None:
                identity = self._globals.get(track.global_id)
                if identity is not None:
                    identity.active_tracks.add(track.key)
                    if track.state in {STATE_CONFIRMED, STATE_LOST} and identity.confirmed:
                        track.state = STATE_CONFIRMED
                        identity.last_seen = max(identity.last_seen, now)
                        identity.last_camera = track.camera_id
                        identity.last_room = track.room_id

    def observe_camera_snapshot(
        self,
        camera_id: str,
        local_ids: Iterable[int],
        *,
        seen_at: float | None = None,
    ) -> None:
        now = time.monotonic() if seen_at is None else float(seen_at)
        camera_id = str(camera_id)
        visible = {int(value) for value in local_ids}
        with self._lock:
            for key, track in self._tracks.items():
                if key[0] != camera_id or key[1] in visible:
                    continue
                if now - track.last_seen <= self.missing_active_grace:
                    continue
                if track.global_id is not None:
                    identity = self._globals.get(track.global_id)
                    if identity is not None:
                        identity.active_tracks.discard(track.key)
                if track.state == STATE_CONFIRMED:
                    track.state = STATE_LOST

    def apply_qwen_result(
        self,
        camera_id: str,
        local_id: int,
        global_id: int,
        verdict: str,
        confidence: float,
        *,
        evidence_version: int | None = None,
        now: float | None = None,
    ) -> dict:
        now = time.monotonic() if now is None else float(now)
        key = (str(camera_id), int(local_id))
        verdict = str(verdict).upper().strip()
        if verdict not in {"SAME", "DIFFERENT", "UNCERTAIN"}:
            verdict = "UNCERTAIN"
        confidence = max(0.0, min(1.0, float(confidence)))
        with self._lock:
            track = self._tracks.get(key)
            if track is None:
                return {"action": "stale_qwen_missing_track"}
            if evidence_version is not None and evidence_version < track.qwen_version:
                return {"action": "stale_qwen_version"}

            if verdict == "SAME":
                self._metrics["qwen_same"] += 1
            elif verdict == "DIFFERENT":
                self._metrics["qwen_different"] += 1
            else:
                self._metrics["qwen_uncertain"] += 1

            gid = int(global_id)
            if track.global_id is None:
                ranked = self.rank_candidates(track, now)
                candidate = next((row for row in ranked if row.global_id == gid), None)
                if candidate is None:
                    return {"action": "stale_qwen_candidate_unavailable"}
                if verdict == "DIFFERENT" and confidence >= self.qwen_different_confidence:
                    track.rejected_global_until[gid] = now + self.qwen_negative_ttl
                    return {"action": "qwen_candidate_reject", "rejected_global": gid}
                if (
                    verdict == "SAME"
                    and confidence >= self.qwen_same_confidence
                    and candidate.score >= self.qwen_rescue_similarity
                ):
                    self._assign_candidate(track, candidate, now)
                    track.positive_votes = max(track.positive_votes, 1)
                    self._metrics["qwen_rescues"] += 1
                else:
                    return {"action": "qwen_candidate_uncertain"}

            if track.global_id != gid:
                return {"action": "stale_qwen_wrong_candidate"}
            track.qwen_verdict = verdict
            track.qwen_confidence = confidence
            track.qwen_version = (
                track.evidence_round if evidence_version is None else int(evidence_version)
            )
            return self._evaluate(track, now)

    def maintenance(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            for track in self._tracks.values():
                if now - track.last_seen > self.active_timeout and track.state in {
                    STATE_TENTATIVE,
                    STATE_CONFIRMED,
                    STATE_SUSPECT,
                }:
                    if track.state == STATE_CONFIRMED:
                        track.state = STATE_LOST
                    if track.global_id is not None:
                        identity = self._globals.get(track.global_id)
                        if identity is not None:
                            identity.active_tracks.discard(track.key)
            stale_tracks = [
                key for key, track in self._tracks.items()
                if now - track.last_seen > self.gallery_ttl
            ]
            for key in stale_tracks:
                self._tracks.pop(key, None)
            stale_globals = [
                gid for gid, identity in self._globals.items()
                if now - identity.last_seen > self.gallery_ttl and not identity.active_tracks
            ]
            for gid in stale_globals:
                self._globals.pop(gid, None)

    def binding_for_track(self, camera_id: str, local_id: int) -> dict | None:
        with self._lock:
            track = self._tracks.get((str(camera_id), int(local_id)))
            if track is None or track.global_id is None:
                return None
            return {
                "global_id": int(track.global_id),
                "state": track.state,
                "score": float(track.assigned_score),
                "margin": float(track.assigned_margin),
                "qwen_verdict": track.qwen_verdict,
                "qwen_confidence": float(track.qwen_confidence),
            }

    def bindings(self) -> dict[tuple[str, int], dict]:
        with self._lock:
            output = {}
            for key, track in self._tracks.items():
                if track.global_id is None:
                    continue
                output[key] = {
                    "global_id": int(track.global_id),
                    "state": track.state,
                    "score": float(track.assigned_score),
                }
            return output

    def comparison_payload(
        self,
        camera_id: str,
        local_id: int,
        candidate_global_id: int | None = None,
    ) -> dict | None:
        key = (str(camera_id), int(local_id))
        with self._lock:
            track = self._tracks.get(key)
            if track is None:
                return None
            gid = int(candidate_global_id) if candidate_global_id is not None else track.global_id
            if gid is None:
                return None
            identity = self._globals.get(gid)
            if identity is None:
                return None
            if not self._hard_candidate_allowed(identity, track, time.monotonic()):
                return None
            old = [
                row
                for row in self._diverse_top(self._candidate_evidence(identity), 3)
                if row.jpeg
            ]
            new = [row for row in self._diverse_top(track.samples, 3) if row.jpeg]
            if not old or not new:
                return None
            return {
                "camera_id": track.camera_id,
                "local_id": track.local_id,
                "global_id": gid,
                "evidence_version": track.evidence_round,
                "old_jpegs": [row.jpeg for row in old],
                "new_jpegs": [row.jpeg for row in new],
                "reid_score": float(track.assigned_score),
                "room_id": track.room_id,
            }

    def metrics(self) -> dict:
        with self._lock:
            states: dict[str, int] = {}
            for track in self._tracks.values():
                states[track.state] = states.get(track.state, 0) + 1
            return {
                **self._metrics,
                "tracks": len(self._tracks),
                "globals": len(self._globals),
                "confirmed_globals": sum(1 for row in self._globals.values() if row.confirmed),
                "states": states,
            }
