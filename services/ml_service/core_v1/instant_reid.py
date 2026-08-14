from __future__ import annotations

import time

import numpy as np

from .conservative_reid import ConservativeGlobalReIdCoordinator
from .global_reid import GlobalIdentity, TrackletState, _cosine, _normalize
from .local_tracker import linear_sum_assignment


class InstantGlobalReIdCoordinator(ConservativeGlobalReIdCoordinator):
    """Immediate visible Global IDs plus conservative cross-camera reconciliation.

    A local track gets a visible G-ID immediately, before ReID is ready. Appearance
    evidence then reconciles provisional IDs across overlapping camera pairs. Fast
    one-shot matches are treated as provisional and are validated again once the
    tracklet has multiple embeddings. Local detection/tracking never waits on this.
    """

    def __init__(self, *args, **kwargs):
        config = dict(args[2] if len(args) >= 3 else kwargs.get("config") or {})
        super().__init__(*args, **kwargs)
        self.fast_pair_similarity = float(config.get("fast_pair_similarity", 0.84))
        self.fast_single_similarity = float(config.get("fast_single_similarity", 0.79))
        self.fast_pair_margin = max(0.0, float(config.get("fast_pair_margin", 0.045)))
        self.fast_confirm_similarity = float(config.get("fast_confirm_similarity", 0.73))
        self.confirm_min_samples = max(2, int(config.get("confirm_min_samples", self.min_samples)))
        self.confirmed_pair_merge_similarity = float(
            config.get("confirmed_pair_merge_similarity", 0.88)
        )

        self._provisional_by_track: dict[tuple[str, int], str] = {}
        self._provisional_globals: set[str] = set()
        self._aliases: dict[str, str] = {}
        self._provisional_created = 0
        self._fast_pair_matches = 0
        self._single_person_fast_matches = 0
        self._pair_reconciles = 0
        self._canonical_merges = 0
        self._fast_rollbacks = 0

    def _canonical_gid(self, global_id: str | None) -> str | None:
        if not global_id:
            return None
        gid = str(global_id)
        seen = set()
        while gid in self._aliases and gid not in seen:
            seen.add(gid)
            gid = self._aliases[gid]
        return gid

    def _reserve_visible_id(self, key: tuple[str, int]) -> str:
        gid = self._provisional_by_track.get(key)
        if gid is not None:
            return self._canonical_gid(gid) or gid
        gid = f"{self.global_prefix}{self._next_global:03d}"
        self._next_global += 1
        self._provisional_by_track[key] = gid
        self._provisional_created += 1
        return gid

    def _active_state_count(self, camera_id: str, now: float) -> int:
        return sum(
            1
            for state in self._tracks.values()
            if state.camera_id == camera_id and now - state.last_seen <= self.active_timeout
        )

    def _paired_camera_is_single(self, camera_id: str, other_camera: str, now: float) -> bool:
        return (
            self._active_state_count(camera_id, now) <= 1
            and self._active_state_count(other_camera, now) <= 1
        )

    def _register_own_provisional(
        self, state: TrackletState, vector: np.ndarray, now: float
    ) -> str:
        key = (state.camera_id, state.track_id)
        gid = self._reserve_visible_id(key)
        if gid not in self._globals:
            self._globals[gid] = GlobalIdentity(
                global_id=gid,
                prototype=_normalize(vector),
                created_at=now,
                last_seen=now,
                last_camera=state.camera_id,
            )
            self._new_globals += 1
        self._provisional_globals.add(gid)
        state.global_id = gid
        state.assignment_similarity = 1.0
        state.assignment_reason = "provisional_embedding"
        return gid

    def _fast_pair_candidate(
        self,
        state: TrackletState,
        vector: np.ndarray,
        now: float,
    ):
        ranked = []
        for gid, identity in self._globals.items():
            canonical = self._canonical_gid(gid)
            if canonical != gid or identity.prototype is None:
                continue
            active = self._active_bindings(gid, now)
            paired_active = [
                (camera_id, track_id)
                for camera_id, track_id in active
                if camera_id != state.camera_id
                and self._same_overlap_group(camera_id, state.camera_id)
            ]
            if not paired_active:
                continue
            if not self._candidate_allowed(identity, state.camera_id, state.track_id, now):
                continue
            similarity = _cosine(vector, identity.prototype)
            other_camera = paired_active[0][0]
            single = self._paired_camera_is_single(state.camera_id, other_camera, now)
            threshold = self.fast_single_similarity if single else self.fast_pair_similarity
            ranked.append((similarity, threshold, single, identity))

        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            return None
        best_similarity, threshold, single, best = ranked[0]
        second_similarity = ranked[1][0] if len(ranked) > 1 else -1.0
        if best_similarity < threshold:
            return None
        if len(ranked) > 1 and best_similarity - second_similarity < self.fast_pair_margin:
            return None
        return best, float(best_similarity), bool(single)

    def _merge_global_ids(self, gid_a: str, gid_b: str, now: float, similarity: float) -> str:
        a = self._canonical_gid(gid_a) or gid_a
        b = self._canonical_gid(gid_b) or gid_b
        if a == b:
            return a
        ia = self._globals.get(a)
        ib = self._globals.get(b)
        if ia is None or ib is None:
            return a if ia is not None else b

        # Oldest identity owns the canonical ID. Before merging, guarantee that
        # no camera would end up with two active tracks carrying one Global ID.
        target, source = (ia, ib) if ia.created_at <= ib.created_at else (ib, ia)
        target_gid, source_gid = target.global_id, source.global_id
        target_active = self._active_bindings(target_gid, now)
        source_active = self._active_bindings(source_gid, now)
        target_cameras = {camera_id for camera_id, _ in target_active}
        source_cameras = {camera_id for camera_id, _ in source_active}
        if target_cameras & source_cameras:
            return target_gid

        for state in self._tracks.values():
            if self._canonical_gid(state.global_id) == source_gid:
                state.global_id = target_gid
                state.assignment_similarity = float(similarity)
                state.assignment_reason = "pair_canonical_merge"

        self._aliases[source_gid] = target_gid
        self._provisional_globals.discard(source_gid)
        if source_gid in self._globals:
            del self._globals[source_gid]
        target.last_seen = max(target.last_seen, source.last_seen, now)
        target.matches += source.matches + 1
        self._canonical_merges += 1
        return target_gid

    def _confirm_provisional(self, state: TrackletState, prototype: np.ndarray, quality: float, now: float):
        own_gid = self._canonical_gid(state.global_id)
        if own_gid is None or own_gid not in self._provisional_globals:
            return

        own_identity = self._globals.get(own_gid)
        ranked = []
        for gid, identity in self._globals.items():
            if gid == own_gid or identity.prototype is None:
                continue
            if not self._candidate_allowed(identity, state.camera_id, state.track_id, now):
                continue
            ranked.append((_cosine(prototype, identity.prototype), identity))
        ranked.sort(key=lambda item: item[0], reverse=True)

        chosen = None
        chosen_similarity = None
        if ranked:
            best_similarity, best = ranked[0]
            threshold = self._threshold_for(state.camera_id, best.last_camera)
            second_similarity = ranked[1][0] if len(ranked) > 1 else -1.0
            margin = best_similarity - second_similarity
            second_is_viable = second_similarity >= threshold
            normal_margin = margin >= self.second_best_margin
            strong_margin = (
                best_similarity >= self.strong_similarity
                and margin >= self.strong_second_best_margin
            )
            if best_similarity >= threshold and (
                normal_margin or strong_margin or not second_is_viable
            ):
                chosen = best
                chosen_similarity = float(best_similarity)
            elif best_similarity >= threshold:
                self._ambiguous_rejects += 1

        if chosen is not None:
            merged = self._merge_global_ids(own_gid, chosen.global_id, now, chosen_similarity)
            state.global_id = merged
            state.assignment_similarity = chosen_similarity
            state.assignment_reason = "confirmed_gallery_merge"
            self._global_matches += 1
        else:
            if own_identity is not None:
                own_identity.prototype = _normalize(prototype)
                own_identity.last_seen = now
                own_identity.last_camera = state.camera_id
            self._provisional_globals.discard(own_gid)
            state.assignment_reason = "confirmed_new"
            state.assignment_similarity = 1.0

    def _validate_fast_match(self, state: TrackletState, prototype: np.ndarray, now: float):
        gid = self._canonical_gid(state.global_id)
        identity = self._globals.get(gid) if gid else None
        if identity is None or identity.prototype is None:
            return
        similarity = _cosine(prototype, identity.prototype)
        if similarity >= self.fast_confirm_similarity:
            state.assignment_reason = "fast_overlap_confirmed"
            state.assignment_similarity = float(similarity)
            return

        # A one-shot fast match was wrong: restore the track's own reserved ID.
        key = (state.camera_id, state.track_id)
        own_gid = self._provisional_by_track.get(key) or self._reserve_visible_id(key)
        self._aliases.pop(own_gid, None)
        self._globals[own_gid] = GlobalIdentity(
            global_id=own_gid,
            prototype=_normalize(prototype),
            created_at=now,
            last_seen=now,
            last_camera=state.camera_id,
        )
        self._provisional_globals.discard(own_gid)
        state.global_id = own_gid
        state.assignment_similarity = 1.0
        state.assignment_reason = "fast_match_rollback"
        self._fast_rollbacks += 1

    def _accept_embedding(self, state: TrackletState, embedding, quality: float, now: float):
        vector = _normalize(embedding)
        if len(state.embeddings) >= 2:
            prototype = self._tracklet_prototype(state)
            similarity = _cosine(vector, prototype)
            if similarity < self.tracklet_outlier_similarity:
                state.outlier_rejects += 1
                self._outlier_rejects += 1
                return

        state.embeddings.append(vector)
        state.qualities.append(float(quality))
        if len(state.embeddings) > self.max_samples:
            state.embeddings.pop(0)
            state.qualities.pop(0)

        if state.global_id is None:
            fast = self._fast_pair_candidate(state, vector, now)
            if fast is not None:
                identity, similarity, single = fast
                state.global_id = identity.global_id
                state.assignment_similarity = similarity
                state.assignment_reason = "fast_overlap_match"
                visible = self._reserve_visible_id((state.camera_id, state.track_id))
                if visible != identity.global_id:
                    self._aliases[visible] = identity.global_id
                self._fast_pair_matches += 1
                if single:
                    self._single_person_fast_matches += 1
            else:
                self._register_own_provisional(state, vector, now)

        prototype = self._tracklet_prototype(state)
        gid = self._canonical_gid(state.global_id)
        identity = self._globals.get(gid) if gid else None
        if gid in self._provisional_globals and identity is not None:
            identity.prototype = prototype
            identity.last_seen = now
            identity.last_camera = state.camera_id

        if len(state.embeddings) >= self.confirm_min_samples:
            if state.assignment_reason == "fast_overlap_match":
                self._validate_fast_match(state, prototype, now)
            elif gid in self._provisional_globals:
                self._confirm_provisional(
                    state,
                    prototype,
                    quality=float(np.mean(state.qualities)),
                    now=now,
                )

        self._reconcile_overlap_pairs(now)

    def _reconcile_overlap_pairs(self, now: float):
        for group in self.overlap_groups:
            cameras = sorted(group)
            if len(cameras) != 2:
                continue
            left = [
                state for state in self._tracks.values()
                if state.camera_id == cameras[0]
                and state.embeddings
                and state.global_id
                and now - state.last_seen <= self.active_timeout
            ]
            right = [
                state for state in self._tracks.values()
                if state.camera_id == cameras[1]
                and state.embeddings
                and state.global_id
                and now - state.last_seen <= self.active_timeout
            ]
            if not left or not right:
                continue

            similarities = np.zeros((len(left), len(right)), dtype=np.float32)
            for i, a in enumerate(left):
                pa = self._tracklet_prototype(a)
                for j, b in enumerate(right):
                    pb = self._tracklet_prototype(b)
                    similarities[i, j] = _cosine(pa, pb)

            rows, cols = linear_sum_assignment(1.0 - similarities)
            single = len(left) == 1 and len(right) == 1
            for row, col in zip(rows.tolist(), cols.tolist()):
                sim = float(similarities[row, col])
                a, b = left[row], right[col]
                gid_a = self._canonical_gid(a.global_id)
                gid_b = self._canonical_gid(b.global_id)
                if not gid_a or not gid_b or gid_a == gid_b:
                    continue
                both_confirmed = (
                    gid_a not in self._provisional_globals
                    and gid_b not in self._provisional_globals
                )
                threshold = (
                    self.confirmed_pair_merge_similarity
                    if both_confirmed
                    else (self.fast_single_similarity if single else self.fast_pair_similarity)
                )
                if sim < threshold:
                    continue

                row_scores = np.sort(similarities[row])[::-1]
                col_scores = np.sort(similarities[:, col])[::-1]
                row_margin = sim - float(row_scores[1]) if len(row_scores) > 1 else 1.0
                col_margin = sim - float(col_scores[1]) if len(col_scores) > 1 else 1.0
                if min(row_margin, col_margin) < self.fast_pair_margin:
                    continue
                merged = self._merge_global_ids(gid_a, gid_b, now, sim)
                a.global_id = merged
                b.global_id = merged
                a.assignment_similarity = sim
                b.assignment_similarity = sim
                a.assignment_reason = "pair_global_assignment"
                b.assignment_reason = "pair_global_assignment"
                self._pair_reconciles += 1

    def identity_for_track(self, camera_id: str, track_id: int):
        key = (str(camera_id), int(track_id))
        if key[1] <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            state = self._tracks.get(key)
            if state is None:
                state = TrackletState(key[0], key[1], last_seen=now)
                self._tracks[key] = state
            state.last_seen = max(state.last_seen, now)
            visible_gid = self._canonical_gid(state.global_id)
            if visible_gid is None:
                visible_gid = self._reserve_visible_id(key)
            return {
                "global_id": visible_gid,
                "known": False,
                "reid_similarity": state.assignment_similarity,
                "reid_reason": state.assignment_reason if state.global_id else "instant_provisional",
                "provisional": state.global_id is None
                or visible_gid in self._provisional_globals,
            }

    def snapshot(self):
        payload = super().snapshot()
        now = time.monotonic()
        with self._lock:
            existing = {(row["camera_id"], row["track_id"]) for row in payload["bindings"]}
            for key, gid in sorted(self._provisional_by_track.items()):
                if key in existing:
                    continue
                state = self._tracks.get(key)
                if state is None:
                    continue
                payload["bindings"].append(
                    {
                        "camera_id": key[0],
                        "track_id": key[1],
                        "global_id": self._canonical_gid(gid),
                        "samples": len(state.embeddings),
                        "similarity": state.assignment_similarity,
                        "reason": "instant_provisional",
                        "active": now - state.last_seen <= self.active_timeout,
                    }
                )
            for row in payload["bindings"]:
                row["global_id"] = self._canonical_gid(row.get("global_id"))
                row["provisional"] = row.get("global_id") in self._provisional_globals
            payload["aliases"] = dict(self._aliases)
        return payload

    def metrics(self):
        payload = super().metrics()
        payload.update(
            {
                "mode": "cpu-osnet-instant-provisional+pair-reconcile",
                "global_id_policy": "instant-visible+one-shot-pair+multi-shot-confirmation",
                "fast_pair_similarity": self.fast_pair_similarity,
                "fast_single_similarity": self.fast_single_similarity,
                "fast_pair_margin": self.fast_pair_margin,
                "confirm_min_samples": self.confirm_min_samples,
                "provisional_created": self._provisional_created,
                "provisional_globals": len(self._provisional_globals),
                "fast_pair_matches": self._fast_pair_matches,
                "single_person_fast_matches": self._single_person_fast_matches,
                "pair_reconciles": self._pair_reconciles,
                "canonical_merges": self._canonical_merges,
                "fast_rollbacks": self._fast_rollbacks,
                "aliases": len(self._aliases),
            }
        )
        return payload
