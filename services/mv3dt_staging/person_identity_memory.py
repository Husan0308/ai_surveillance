"""Experimental, in-memory person identity memory for offline MV3DT review.

This module is intentionally not wired into production surveillance or the
existing MV3DT event importer. It stores no image crops or embeddings on disk.
Automatic re-identification is fail-closed: a caller must provide an explicit
calibrated threshold profile and validated world-coordinate evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite
from typing import Sequence
import uuid

import numpy as np


class IdentityMemoryError(ValueError):
    """Invalid identity input or unsafe alias lineage."""


@dataclass(frozen=True)
class MemoryConfig:
    retention_seconds: float = 30.0
    active_timeout_seconds: float = 1.5
    max_embeddings_per_person: int = 8
    min_embedding_quality: float = 0.75
    min_similarity: float | None = None
    min_best_second_margin: float | None = None
    threshold_profile_id: str | None = None
    max_speed_units_per_second: float | None = None
    allowed_camera_transitions: frozenset[frozenset[str]] = frozenset()
    # Do not treat the Entrance Room's current unverified calibration as motion
    # evidence; set this only for an independently validated coordinate frame.
    require_validated_location: bool = True
    best_prototype_count: int = 3

    def __post_init__(self) -> None:
        if self.retention_seconds <= 0 or self.active_timeout_seconds < 0:
            raise IdentityMemoryError("retention and active timeout must be non-negative")
        if self.max_embeddings_per_person < 1 or self.best_prototype_count < 1:
            raise IdentityMemoryError("gallery sizes must be positive")
        if not 0 <= self.min_embedding_quality <= 1:
            raise IdentityMemoryError("min_embedding_quality must be in [0, 1]")
        if self.min_similarity is not None and not -1 <= self.min_similarity <= 1:
            raise IdentityMemoryError("min_similarity must be in [-1, 1]")
        if self.min_best_second_margin is not None and not 0 <= self.min_best_second_margin <= 2:
            raise IdentityMemoryError("min_best_second_margin must be in [0, 2]")
        if self.max_speed_units_per_second is not None and self.max_speed_units_per_second <= 0:
            raise IdentityMemoryError("max speed must be positive")

    @property
    def auto_match_calibrated(self) -> bool:
        return bool(
            self.threshold_profile_id
            and self.min_similarity is not None
            and self.min_best_second_margin is not None
        )


@dataclass(frozen=True)
class IdentityObservation:
    run_id: str
    camera_id: str
    raw_id: str
    timestamp_seconds: float
    embedding: Sequence[float] | np.ndarray | None = None
    embedding_quality: float = 0.0
    world_xy: tuple[float, float] | None = None
    world_geometry_validated: bool = False
    raw_id_scope: str = "camera_local"  # or "mv3dt_global"
    frame_id: int | None = None


@dataclass(frozen=True)
class AdoptionEvent:
    source_log_sha256: str
    camera_id: str
    source_id: int
    old_raw_id: str
    adopted_raw_id: str
    event_frame: int


@dataclass
class _PersonRecord:
    persistent_person_id: str
    last_seen_seconds: float
    last_camera_id: str
    last_world_xy: tuple[float, float] | None
    location_validated: bool
    embeddings: list[np.ndarray] = field(default_factory=list)
    raw_keys: set[tuple[str, str, str, str]] = field(default_factory=set)


class PersonIdentityMemory:
    """Bounded temporary gallery with conservative, auditable assignments.

    `raw_id_scope="mv3dt_global"` preserves one native MV3DT ID across its
    camera observations. `camera_local` keeps equal numeric IDs on separate
    cameras isolated. Re-ID across new raw keys is never accepted unless the
    caller supplies a calibrated threshold profile and validated geometry.
    """

    def __init__(self, config: MemoryConfig | None = None, *, run_id: str | None = None):
        self.config = config or MemoryConfig()
        self.run_id = run_id or str(uuid.uuid4())
        self._persons: dict[str, _PersonRecord] = {}
        self._raw_to_person: dict[tuple[str, str, str, str], str] = {}
        self._display_aliases: dict[tuple[str, str, str, str], str] = {}
        self._adoption_events: list[AdoptionEvent] = []
        self._next_person_number = 1
        self._last_decisions: list[dict] = []

    def _raw_key(self, observation: IdentityObservation) -> tuple[str, str, str, str]:
        if observation.raw_id_scope not in {"camera_local", "mv3dt_global"}:
            raise IdentityMemoryError("raw_id_scope must be camera_local or mv3dt_global")
        camera_scope = observation.camera_id if observation.raw_id_scope == "camera_local" else "*"
        return (observation.run_id, observation.raw_id_scope, camera_scope, str(observation.raw_id))

    @staticmethod
    def _normalize_embedding(value: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise IdentityMemoryError("embedding must be non-empty and finite")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise IdentityMemoryError("embedding norm must be non-zero")
        return vector / norm

    def _new_person(self, observation: IdentityObservation, key: tuple[str, str, str, str],
                    embedding: np.ndarray | None) -> _PersonRecord:
        pid = f"person-provisional-{self._next_person_number:05d}"
        self._next_person_number += 1
        record = _PersonRecord(
            persistent_person_id=pid,
            last_seen_seconds=observation.timestamp_seconds,
            last_camera_id=observation.camera_id,
            last_world_xy=observation.world_xy,
            location_validated=observation.world_geometry_validated,
            raw_keys={key},
        )
        if embedding is not None and observation.embedding_quality >= self.config.min_embedding_quality:
            record.embeddings.append(embedding)
        self._persons[pid] = record
        self._raw_to_person[key] = pid
        return record

    def _active_conflict(self, record: _PersonRecord, key: tuple[str, str, str, str], now: float) -> bool:
        for other_key in record.raw_keys:
            if other_key == key:
                continue
            # A different raw identity currently active cannot be merged into
            # this person. Cross-camera overlap should already share a native
            # MV3DT global ID; otherwise remain conservative.
            other_pid = self._raw_to_person.get(other_key)
            other = self._persons.get(other_pid or "")
            if other is not None and now - other.last_seen_seconds <= self.config.active_timeout_seconds:
                return True
        return False

    def _camera_transition_ok(self, old: str, new: str) -> bool:
        if old == new:
            return True
        edge = frozenset((old, new))
        return edge in self.config.allowed_camera_transitions

    def _location_evidence(self, record: _PersonRecord, obs: IdentityObservation) -> tuple[bool, str, float | None]:
        if not self._camera_transition_ok(record.last_camera_id, obs.camera_id):
            return False, "camera_transition_not_allowed", None
        if obs.world_xy is None or record.last_world_xy is None:
            return (not self.config.require_validated_location, "location_unavailable", None)
        if self.config.require_validated_location and not (obs.world_geometry_validated and record.location_validated):
            return False, "world_geometry_unvalidated", None
        distance = hypot(obs.world_xy[0] - record.last_world_xy[0], obs.world_xy[1] - record.last_world_xy[1])
        dt = max(obs.timestamp_seconds - record.last_seen_seconds, 1e-6)
        if self.config.max_speed_units_per_second is not None and distance / dt > self.config.max_speed_units_per_second:
            return False, "movement_implausible", distance
        return True, "location_compatible", distance

    def _similarity(self, record: _PersonRecord, embedding: np.ndarray) -> float | None:
        if not record.embeddings:
            return None
        scores = sorted((float(np.dot(embedding, sample)) for sample in record.embeddings), reverse=True)
        top = scores[: self.config.best_prototype_count]
        # Median of the best few is less sensitive to a single contaminated crop
        # than a max-only gallery lookup.
        return float(np.median(top))

    def observe(self, observation: IdentityObservation) -> dict:
        if not isfinite(observation.timestamp_seconds) or observation.timestamp_seconds < 0:
            raise IdentityMemoryError("timestamp_seconds must be finite and non-negative")
        if not observation.run_id or not observation.camera_id or not str(observation.raw_id):
            raise IdentityMemoryError("run, camera and raw ID are required")
        if not isfinite(observation.embedding_quality) or not 0 <= observation.embedding_quality <= 1:
            raise IdentityMemoryError("embedding_quality must be in [0, 1]")
        self.expire(observation.timestamp_seconds)
        key = self._raw_key(observation)
        embedding = self._normalize_embedding(observation.embedding)
        pid = self._raw_to_person.get(key)
        decision = "raw_identity_continued"
        candidate_scores: list[tuple[str, float, str]] = []
        confidence: float | None = None
        if pid is None:
            qualified = embedding is not None and observation.embedding_quality >= self.config.min_embedding_quality
            if qualified:
                for candidate_id, record in self._persons.items():
                    if not record.embeddings or self._active_conflict(record, key, observation.timestamp_seconds):
                        continue
                    gap = observation.timestamp_seconds - record.last_seen_seconds
                    if gap < 0 or gap > self.config.retention_seconds:
                        continue
                    location_ok, location_reason, _ = self._location_evidence(record, observation)
                    score = self._similarity(record, embedding)
                    if score is not None:
                        candidate_scores.append((candidate_id, score, location_reason if location_ok else location_reason))
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            if candidate_scores:
                best_id, best_score, location_reason = candidate_scores[0]
                second_score = candidate_scores[1][1] if len(candidate_scores) > 1 else -1.0
                margin = best_score - second_score
                confidence = best_score
                location_ok, _, _ = self._location_evidence(self._persons[best_id], observation)
                accepted = bool(
                    self.config.auto_match_calibrated
                    and location_ok
                    and best_score >= float(self.config.min_similarity)
                    and margin >= float(self.config.min_best_second_margin)
                )
                if accepted:
                    pid = best_id
                    self._raw_to_person[key] = pid
                    self._persons[pid].raw_keys.add(key)
                    decision = "appearance_reidentified"
                elif not location_ok and not self.config.auto_match_calibrated:
                    decision = f"provisional_{location_reason}_and_uncalibrated_thresholds"
                elif not location_ok:
                    decision = f"provisional_{location_reason}"
                elif not self.config.auto_match_calibrated:
                    decision = "provisional_uncalibrated_thresholds"
                elif best_score < float(self.config.min_similarity):
                    decision = "provisional_below_similarity_threshold"
                else:
                    decision = "provisional_ambiguous_margin"
            else:
                decision = "provisional_no_eligible_gallery_match"
            if pid is None:
                record = self._new_person(observation, key, embedding if qualified else None)
                pid = record.persistent_person_id
                if not qualified:
                    decision = "provisional_embedding_unavailable_or_low_quality"
        else:
            # A detector/tracker observation can arrive before a usable crop
            # embedding is available. On the first later high-quality feature,
            # give the provisional identity one conservative chance to match a
            # lost gallery; otherwise it would be pinned to its provisional ID
            # forever merely because the first frame lacked a feature.
            current_record = self._persons[pid]
            qualified = embedding is not None and observation.embedding_quality >= self.config.min_embedding_quality
            if qualified and not current_record.embeddings:
                for candidate_id, candidate_record in self._persons.items():
                    if candidate_id == pid or not candidate_record.embeddings:
                        continue
                    if self._active_conflict(candidate_record, key, observation.timestamp_seconds):
                        continue
                    gap = observation.timestamp_seconds - candidate_record.last_seen_seconds
                    if gap < 0 or gap > self.config.retention_seconds:
                        continue
                    location_ok, location_reason, _ = self._location_evidence(candidate_record, observation)
                    score = self._similarity(candidate_record, embedding)
                    if score is not None:
                        candidate_scores.append((candidate_id, score, location_reason))
                candidate_scores.sort(key=lambda item: item[1], reverse=True)
                if candidate_scores:
                    best_id, best_score, _ = candidate_scores[0]
                    second_score = candidate_scores[1][1] if len(candidate_scores) > 1 else -1.0
                    margin = best_score - second_score
                    confidence = best_score
                    best_record = self._persons[best_id]
                    location_ok, location_reason, _ = self._location_evidence(best_record, observation)
                    accepted = bool(
                        self.config.auto_match_calibrated
                        and location_ok
                        and best_score >= float(self.config.min_similarity)
                        and margin >= float(self.config.min_best_second_margin)
                    )
                    if accepted:
                        current_record.raw_keys.discard(key)
                        self._raw_to_person[key] = best_id
                        best_record.raw_keys.add(key)
                        if not current_record.raw_keys and not current_record.embeddings:
                            self._persons.pop(pid, None)
                        pid = best_id
                        decision = "appearance_reidentified"
                    elif not location_ok:
                        decision = f"provisional_{location_reason}"
                    elif not self.config.auto_match_calibrated:
                        decision = "provisional_uncalibrated_thresholds"
                    elif best_score < float(self.config.min_similarity):
                        decision = "provisional_below_similarity_threshold"
                    else:
                        decision = "provisional_ambiguous_margin"
                else:
                    decision = "provisional_no_eligible_gallery_match"
        record = self._persons[pid]
        if observation.timestamp_seconds < record.last_seen_seconds:
            raise IdentityMemoryError("observation arrived out of order for an existing identity")
        record.last_seen_seconds = observation.timestamp_seconds
        record.last_camera_id = observation.camera_id
        record.last_world_xy = observation.world_xy
        record.location_validated = observation.world_geometry_validated
        if embedding is not None and observation.embedding_quality >= self.config.min_embedding_quality:
            if not record.embeddings or decision in {"raw_identity_continued", "appearance_reidentified"}:
                record.embeddings.append(embedding)
                del record.embeddings[:-self.config.max_embeddings_per_person]
        display_alias = self._display_alias_for_key(key)
        result = {
            "run_id": observation.run_id,
            "camera_id": observation.camera_id,
            "raw_id": str(observation.raw_id),
            "raw_id_scope": observation.raw_id_scope,
            "persistent_person_id": pid,
            "display_id": display_alias or pid,
            "match_status": decision,
            "match_confidence": confidence,
            "candidate_scores": [
                {"persistent_person_id": candidate, "cosine_similarity": score, "location": reason}
                for candidate, score, reason in candidate_scores
            ],
            "frame_id": observation.frame_id,
            "identity_is_recognized": False,
        }
        self._last_decisions.append(result)
        return result

    def expire(self, now_seconds: float) -> list[str]:
        expired = [pid for pid, person in self._persons.items()
                   if now_seconds - person.last_seen_seconds > self.config.retention_seconds]
        for pid in expired:
            person = self._persons.pop(pid)
            for key in person.raw_keys:
                self._raw_to_person.pop(key, None)
                self._display_aliases.pop(key, None)
        return expired

    def apply_confirmed_adoption(self, event: AdoptionEvent, *, expected_source_log_sha256: str,
                                 retrospective_offline: bool = True) -> dict:
        """Apply a hash-bound *display* alias only; it never merges person IDs."""
        if event.source_log_sha256.lower() != expected_source_log_sha256.lower():
            raise IdentityMemoryError("adoption source-log hash mismatch")
        if not retrospective_offline:
            raise IdentityMemoryError("confirmed adoption aliases are offline replay only")
        old_key = (self.run_id, "camera_local", event.camera_id, str(event.old_raw_id))
        new_key = (self.run_id, "camera_local", event.camera_id, str(event.adopted_raw_id))
        if old_key == new_key:
            raise IdentityMemoryError("self-referential adoption alias")
        existing = self._display_aliases.get(old_key)
        if existing is not None and existing != event.adopted_raw_id:
            raise IdentityMemoryError("conflicting adoption aliases")
        self._display_aliases[old_key] = event.adopted_raw_id
        # Detect cycles after insertion; rollback the untrusted rule if invalid.
        try:
            self._resolve_alias(old_key)
        except IdentityMemoryError:
            self._display_aliases.pop(old_key, None)
            raise
        self._adoption_events.append(event)
        return {"applied": True, "scope": "display_only", "persistent_ids_merged": False,
                "camera_id": event.camera_id, "event_frame": event.event_frame}

    def _resolve_alias(self, key: tuple[str, str, str, str]) -> str | None:
        visited: set[tuple[str, str, str, str]] = set()
        current = key
        while current in self._display_aliases:
            if current in visited:
                raise IdentityMemoryError("adoption alias cycle")
            visited.add(current)
            next_raw = self._display_aliases[current]
            current = (current[0], current[1], current[2], next_raw)
        return current[3] if visited else None

    def _display_alias_for_key(self, key: tuple[str, str, str, str]) -> str | None:
        """Return a hash-bound alias label for either end of an adoption chain."""
        for source in self._display_aliases:
            if source[:3] != key[:3]:
                continue
            resolved = self._resolve_alias(source)
            if resolved == key[3] or source == key:
                return f"adopted:{key[2]}:{resolved}"
        resolved = self._resolve_alias(key)
        if resolved is not None:
            return f"adopted:{key[2]}:{resolved}"
        return None

    def snapshot(self) -> dict:
        return {
            "run_id": self.run_id,
            "memory_size": len(self._persons),
            "embedding_count": sum(len(p.embeddings) for p in self._persons.values()),
            "threshold_profile_id": self.config.threshold_profile_id,
            "auto_reid_enabled": self.config.auto_match_calibrated,
            "adoption_event_count": len(self._adoption_events),
            "raw_observation_decisions": len(self._last_decisions),
        }

    def clear(self) -> None:
        self._persons.clear()
        self._raw_to_person.clear()
        self._display_aliases.clear()
        self._adoption_events.clear()
        self._last_decisions.clear()

