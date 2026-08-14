from __future__ import annotations

import time

from .global_reid import GlobalReIdCoordinator, TrackletState, _cosine, _normalize


class ConservativeGlobalReIdCoordinator(GlobalReIdCoordinator):
    """Global ReID policy that never lets a look-alike bypass gallery margin.

    A very high absolute similarity is useful only when the runner-up is clearly
    worse. If two identities are both very similar, the observation is ambiguous
    and receives a fresh Global ID rather than risking identity contamination.
    """

    def __init__(self, *args, **kwargs):
        config = dict(args[2] if len(args) >= 3 else kwargs.get("config") or {})
        self.strong_second_best_margin = max(
            0.0, float(config.get("strong_second_best_margin", 0.025))
        )
        super().__init__(*args, **kwargs)

    def resolve_tracklet(
        self,
        camera_id: str,
        track_id: int,
        prototype,
        quality: float = 1.0,
        now: float | None = None,
    ) -> str:
        now = time.monotonic() if now is None else float(now)
        key = (str(camera_id), int(track_id))
        vector = _normalize(prototype)
        with self._lock:
            state = self._tracks.get(key)
            if state is None:
                state = TrackletState(key[0], key[1], last_seen=now)
                self._tracks[key] = state
            state.last_seen = max(state.last_seen, now)
            if state.global_id is not None:
                return state.global_id

            ranked = []
            for identity in self._globals.values():
                if not self._candidate_allowed(identity, key[0], key[1], now):
                    continue
                ranked.append((_cosine(vector, identity.prototype), identity))
            ranked.sort(key=lambda item: item[0], reverse=True)

            chosen = None
            chosen_similarity = None
            if ranked:
                best_similarity, best = ranked[0]
                threshold = self._threshold_for(key[0], best.last_camera)
                second_similarity = ranked[1][0] if len(ranked) > 1 else -1.0
                margin = best_similarity - second_similarity
                second_is_viable = second_similarity >= threshold
                normal_margin = margin >= self.second_best_margin
                strong_margin = (
                    best_similarity >= self.strong_similarity
                    and margin >= self.strong_second_best_margin
                )
                unopposed = not second_is_viable

                if best_similarity >= threshold and (
                    normal_margin or strong_margin or unopposed
                ):
                    chosen = best
                    chosen_similarity = best_similarity
                elif best_similarity >= threshold:
                    self._ambiguous_rejects += 1

            if chosen is None:
                chosen = self._new_global(vector, key[0], now)
                chosen_similarity = 1.0
                state.assignment_reason = "new_global"
            else:
                self._global_matches += 1
                state.assignment_reason = "matched_gallery"
                chosen.matches += 1

            state.global_id = chosen.global_id
            state.assignment_similarity = float(chosen_similarity)
            chosen.last_seen = now
            chosen.last_camera = key[0]

            if (
                chosen_similarity >= self.prototype_update_similarity
                and quality >= 0.55
                and state.assignment_reason == "matched_gallery"
            ):
                alpha = self.prototype_update_alpha * min(
                    1.0, max(0.25, float(quality))
                )
                chosen.prototype = _normalize(
                    (1.0 - alpha) * chosen.prototype + alpha * vector
                )
                chosen.prototype_updates += 1
                self._prototype_updates += 1
            return chosen.global_id

    def metrics(self):
        payload = super().metrics()
        payload["strong_second_best_margin"] = self.strong_second_best_margin
        payload["global_id_policy"] = "conservative-absolute+runnerup-margin+active-uniqueness"
        return payload
