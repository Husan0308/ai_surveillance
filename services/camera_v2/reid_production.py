from __future__ import annotations

import time

import numpy as np

from .global_identity import (
    CandidateScore,
    GlobalIdentity,
    GlobalIdentityCore,
    STATE_CONFIRMED,
    STATE_SUSPECT,
    STATE_TENTATIVE,
    TrackletState,
    cosine,
)
from .reid_runtime import ReIdIdentityEngine


class ProductionGlobalIdentityCore(GlobalIdentityCore):
    """False-merge-first hardening around the base reversible identity state.

    Important differences from a naive gallery matcher:
    - a confirmed cross-camera track is evaluated against identity evidence that
      excludes its own samples, so a bad merge cannot validate itself;
    - low-confidence/Qwen-only confirmations do not update the canonical gallery;
    - repeated fresh ReID contradictions can rollback a confirmed non-canonical
      binding even when no new Qwen response arrives;
    - a very strong Qwen SAME may confirm a gray-zone track without allowing that
      track to poison the canonical appearance gallery.
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = dict(config or {})
        super().__init__(cfg)
        self.prototype_update_similarity = float(
            cfg.get("prototype_update_similarity", 0.82)
        )
        self.reid_rollback_votes = max(3, int(cfg.get("reid_rollback_votes", 4)))
        self.qwen_gray_confirm_confidence = float(
            cfg.get("qwen_gray_confirm_confidence", 0.90)
        )
        self.qwen_suspect_confidence = float(
            cfg.get("qwen_suspect_confidence", 0.90)
        )
        self._metrics.setdefault("gallery_update_skips", 0)
        self._metrics.setdefault("reid_contradiction_rollbacks", 0)
        self._metrics.setdefault("qwen_gray_confirms", 0)
        self._metrics.setdefault("qwen_suspects", 0)

    @staticmethod
    def _canonical_origin(identity: GlobalIdentity) -> tuple[str, int] | None:
        if not identity.gallery:
            return None
        first = min(identity.gallery, key=lambda row: row.captured_at)
        return first.origin_key

    def _candidate_score(
        self,
        track: TrackletState,
        identity: GlobalIdentity,
        now: float,
    ) -> CandidateScore | None:
        candidate_gallery = list(self._candidate_evidence(identity))
        if not candidate_gallery or not self._hard_candidate_allowed(identity, track, now):
            return None

        # Once a non-canonical track has been merged, never let its own gallery
        # contribution become evidence that the merge was correct.
        canonical = self._canonical_origin(identity)
        if (
            track.global_id == identity.global_id
            and canonical is not None
            and track.key != canonical
        ):
            independent = [row for row in candidate_gallery if row.origin_key != track.key]
            if independent:
                candidate_gallery = independent

        new_samples = self._diverse_top(track.samples)
        old_samples = self._diverse_top(candidate_gallery, min(self.gallery_size, 6))
        if not new_samples or not old_samples:
            return None

        new_matrix = np.stack([row.embedding for row in new_samples], axis=0)
        old_matrix = np.stack([row.embedding for row in old_samples], axis=0)
        matrix = new_matrix @ old_matrix.T
        best_per_new = np.max(matrix, axis=1)
        q = np.asarray(
            [max(0.05, row.quality) for row in new_samples], dtype=np.float32
        )
        raw = float(np.average(best_per_new, weights=q))
        best = float(np.max(matrix))
        new_proto = self._weighted_prototype(new_samples)
        old_proto = self._weighted_prototype(old_samples)
        proto = cosine(new_proto, old_proto)
        score = raw * 0.55 + proto * 0.30 + best * 0.15
        reason = "independent_gallery"

        gap = max(0.0, now - identity.last_seen)
        if identity.last_camera == track.camera_id and gap <= self.same_camera_reconnect_sec:
            score += 0.035
            reason = "same_camera_reconnect"
        elif identity.last_room and identity.last_room == track.room_id and gap <= self.active_timeout:
            score += 0.020
            reason = "same_room_overlap"
        score = max(-1.0, min(1.0, score))
        return CandidateScore(
            identity.global_id, score, raw, proto, best, reason=reason
        )

    def _confirm_without_gallery(self, track: TrackletState, now: float) -> None:
        if track.global_id is None:
            return
        identity = self._globals.get(track.global_id)
        if identity is None:
            return
        identity.confirmed = True
        identity.suspect = False
        identity.last_seen = now
        identity.last_camera = track.camera_id
        identity.last_room = track.room_id
        identity.active_tracks.add(track.key)
        track.state = STATE_CONFIRMED
        track.contradiction_votes = 0

    def _commit_track_to_gallery(self, track: TrackletState, now: float) -> None:
        if track.global_id is None:
            return
        identity = self._globals.get(track.global_id)
        if identity is None:
            return

        # The first track creates the canonical identity gallery. Later camera
        # observations may use the same G-ID without automatically entering that
        # canonical gallery. Only strong independent ReID evidence may update it.
        if not identity.confirmed:
            super()._commit_track_to_gallery(track, now)
            return

        allow_update = (
            track.assigned_score >= self.prototype_update_similarity
            and track.qwen_verdict != "DIFFERENT"
        )
        if allow_update:
            for sample in self._diverse_top(track.samples):
                if identity.gallery:
                    best = max(
                        cosine(sample.embedding, old.embedding)
                        for old in identity.gallery
                    )
                    if best > 0.985:
                        continue
                identity.gallery.append(sample)
            self._rebuild_identity_gallery(identity)
            self._metrics["gallery_commits"] += 1
        else:
            self._metrics["gallery_update_skips"] += 1

        identity.quarantine = []
        self._confirm_without_gallery(track, now)

    def _evaluate(self, track: TrackletState, now: float) -> dict:
        result = super()._evaluate(track, now)

        # The base core records a fresh low-similarity observation in negative_votes.
        # Use those independent fresh rounds for continuous post-confirm correction.
        if (
            track.global_id is not None
            and track.state in {STATE_CONFIRMED, STATE_SUSPECT}
        ):
            if track.assigned_score >= self.confirm_similarity:
                track.negative_votes = 0
            elif track.assigned_score < self.reject_similarity:
                track.state = STATE_SUSPECT
                identity = self._globals.get(track.global_id)
                if identity is not None:
                    identity.suspect = True

            if track.negative_votes >= self.reid_rollback_votes:
                rejected = track.global_id
                self._detach_track(
                    track,
                    now,
                    reject_gid=rejected,
                    rollback_gallery=True,
                )
                self._metrics["rollbacks"] += 1
                self._metrics["reid_contradiction_rollbacks"] += 1
                return {
                    "action": "reid_rollback",
                    "rejected_global": rejected,
                    "state": track.state,
                }

        # A Qwen-rescued gray-zone candidate must not occupy a G-ID forever when
        # all available appearance evidence stays below the normal provisional gate.
        if (
            track.global_id is not None
            and track.state == STATE_TENTATIVE
            and len(track.samples) >= self.max_track_samples
            and track.assigned_score < self.qwen_rescue_similarity
        ):
            rejected = track.global_id
            self._detach_track(track, now, reject_gid=rejected)
            identity = self._new_identity(track, now)
            self._commit_track_to_gallery(track, now)
            return {
                "action": "tentative_timeout_new",
                "rejected_global": rejected,
                "global_id": identity.global_id,
                "state": track.state,
            }
        return result

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
        when = time.monotonic() if now is None else float(now)
        verdict_u = str(verdict).upper().strip()
        result = super().apply_qwen_result(
            camera_id,
            local_id,
            global_id,
            verdict_u,
            confidence,
            evidence_version=evidence_version,
            now=when,
        )
        key = (str(camera_id), int(local_id))
        with self._lock:
            track = self._tracks.get(key)
            if track is None:
                return result

            # Strong Qwen SAME may resolve a legitimate front/back gray-zone match,
            # but the visual-only decision is not allowed to update canonical ReID.
            if (
                verdict_u == "SAME"
                and float(confidence) >= self.qwen_gray_confirm_confidence
                and track.global_id == int(global_id)
                and track.state == STATE_TENTATIVE
                and track.assigned_score >= self.qwen_rescue_similarity
            ):
                self._confirm_without_gallery(track, when)
                self._metrics["confirmed_matches"] += 1
                self._metrics["qwen_gray_confirms"] += 1
                result = {
                    "action": "qwen_gray_confirm",
                    "global_id": track.global_id,
                    "state": track.state,
                }

            # A single VLM disagreement cannot tear down a strong ReID match, but it
            # may mark it SUSPECT once. It must be corroborated by fresh ReID/Qwen
            # evidence before rollback.
            elif (
                verdict_u == "DIFFERENT"
                and float(confidence) >= self.qwen_suspect_confidence
                and track.global_id == int(global_id)
                and track.state == STATE_CONFIRMED
            ):
                track.contradiction_votes += 1
                track.state = STATE_SUSPECT
                identity = self._globals.get(track.global_id)
                if identity is not None:
                    identity.suspect = True
                self._metrics["qwen_suspects"] += 1

            # Consume this VLM result once. Reusing the same DIFFERENT verdict on
            # every new crop would fake multiple independent contradiction votes.
            track.qwen_verdict = "UNCERTAIN"
            track.qwen_confidence = 0.0
        return result


class ProductionReIdIdentityEngine(ReIdIdentityEngine):
    """ReIdIdentityEngine using the production-hardened identity state machine."""

    def __init__(
        self,
        camera_rooms: dict[str, str],
        config: dict | None = None,
        *,
        root=None,
        embedder=None,
        qwen=None,
    ) -> None:
        cfg = dict(config or {})
        super().__init__(
            camera_rooms,
            cfg,
            root=root,
            embedder=embedder,
            qwen=qwen,
        )
        cfg["camera_rooms"] = dict(camera_rooms)
        self.core = ProductionGlobalIdentityCore(cfg)
