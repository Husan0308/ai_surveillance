from __future__ import annotations

import os
import time
from collections import defaultdict

from .adaptive_reid import AdaptiveTrackletReID, PairVote


class StableAdaptiveTrackletReID(AdaptiveTrackletReID):
    """Adaptive tracklet ReID with fresh-evidence voting and identity leases.

    A merge is intentionally hard to obtain and sticky once obtained:
      * every positive vote must contain a newer observation from BOTH cameras;
      * while both peer local tracks stay active, each is reserved to the other;
      * an active reservation blocks a different peer from stealing the identity;
      * a confirmed pair is broken only after repeated fresh, very-low similarity;
      * correction sends the weaker/newer side to a fresh anchor, never to a
        speculative alternative profile.

    This is a hysteresis layer: merge threshold and release threshold are different,
    preventing IDs from bouncing when a score sits close to the decision boundary.
    """

    def __init__(self, manager) -> None:
        super().__init__(manager)
        self.last_observed: dict[tuple[int, int], float] = {}
        self.last_vote_observed: dict[frozenset, tuple[float, float]] = {}
        self.peer_owner: dict[tuple[int, int], tuple[int, int]] = {}
        self.lock_bad_votes: dict[frozenset, int] = defaultdict(int)
        self.last_lock_audit: dict[frozenset, tuple[float, float]] = {}

        self.release_floor = float(os.environ.get("CAMERA_V2_ADAPT_RELEASE_FLOOR", "0.32"))
        self.release_votes = max(5, int(os.environ.get("CAMERA_V2_ADAPT_RELEASE_VOTES", "6")))
        self.sticky_bonus = max(0.0, float(os.environ.get("CAMERA_V2_ADAPT_STICKY_BONUS", "0.025")))
        self.stats.setdefault("fresh_vote_skip", 0)
        self.stats.setdefault("peer_lock_blocks", 0)
        self.stats.setdefault("peer_locks", 0)
        self.stats.setdefault("lock_releases", 0)
        self.stats.setdefault("lock_corrections", 0)

    def observe_rows(self, rows: list[dict], now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        # Record fresh observation time even when the feature is intentionally
        # discarded as a near-duplicate from a stationary person.
        for row in rows:
            sid = int(row.get("source_id", -1))
            oid = int(row.get("object_id", -1))
            if sid >= 0 and oid >= 0 and row.get("feature"):
                self.last_observed[(sid, oid)] = now
        super().observe_rows(rows, now)

    def _active_locked_peer(self, key, now: float):
        peer = self.peer_owner.get(key)
        if peer is None:
            return None
        if not self.manager._is_active(peer, now):
            self.peer_owner.pop(key, None)
            if self.peer_owner.get(peer) == key:
                self.peer_owner.pop(peer, None)
            self.stats["lock_releases"] += 1
            return None
        return peer

    def tracklet_similarity(self, a, b, now: float) -> float:
        lock_a = self._active_locked_peer(a, now)
        lock_b = self._active_locked_peer(b, now)
        if lock_a is not None and lock_a != b:
            self.stats["peer_lock_blocks"] += 1
            return -1.0
        if lock_b is not None and lock_b != a:
            self.stats["peer_lock_blocks"] += 1
            return -1.0

        score = super().tracklet_similarity(a, b, now)
        if score >= -0.5 and lock_a == b and lock_b == a:
            score = min(1.0, score + self.sticky_bonus)
        return score

    def _update_vote(self, pair, good: bool, score: float, margin: float, now: float) -> bool:
        keys = sorted(tuple(pair))
        if len(keys) != 2:
            return False
        a, b = keys
        obs_a = float(self.last_observed.get(a, 0.0))
        obs_b = float(self.last_observed.get(b, 0.0))
        prev_a, prev_b = self.last_vote_observed.get(pair, (0.0, 0.0))

        # One worker result from some unrelated camera must not replay an old pair
        # and increment its vote. Both cameras need genuinely newer evidence.
        if good and (obs_a <= prev_a + 1e-6 or obs_b <= prev_b + 1e-6):
            self.stats["fresh_vote_skip"] += 1
            return False

        if good:
            self.last_vote_observed[pair] = (obs_a, obs_b)
        return super()._update_vote(pair, good, score, margin, now)

    def _merge(self, a, b, now: float) -> bool:
        lock_a = self._active_locked_peer(a, now)
        lock_b = self._active_locked_peer(b, now)
        if lock_a is not None and lock_a != b:
            return False
        if lock_b is not None and lock_b != a:
            return False

        merged = super()._merge(a, b, now)
        if merged:
            self.peer_owner[a] = b
            self.peer_owner[b] = a
            pair = self._pair_key(a, b) if hasattr(self, "_pair_key") else frozenset((a, b))
            self.lock_bad_votes[pair] = 0
            self.last_lock_audit[pair] = (
                float(self.last_observed.get(a, 0.0)),
                float(self.last_observed.get(b, 0.0)),
            )
            self.stats["peer_locks"] += 1
        return merged

    def _fresh_anchor(self, key, now: float) -> None:
        manager = self.manager
        binding = manager.bindings.get(key)
        if binding is None:
            return
        vector, color, count = manager._track_prototype(key)
        if vector is not None and count >= 2:
            quality = max((item.quality for item in manager._evidence(key)), default=0.5)
            manager._correct_to_new_anchor(
                binding,
                key,
                vector,
                color,
                binding.last_bbox,
                now,
                quality,
            )
            return

        old_gid = manager._resolve(binding.global_id)
        manager._remove_owner_contributions(old_gid, key)
        profile = manager._new_profile(key[0], now)
        binding.global_id = profile.global_id
        binding.state = "anchor"
        binding.confirm_votes = manager.confirm_votes_required
        binding.bad_votes = 0
        binding.switch_candidate = None
        binding.switch_votes = 0
        binding.last_committed_at = 0.0

    def _audit_peer_locks(self, now: float) -> None:
        seen = set()
        for a, b in list(self.peer_owner.items()):
            pair = frozenset((a, b))
            if pair in seen:
                continue
            seen.add(pair)

            if self.peer_owner.get(b) != a:
                self.peer_owner.pop(a, None)
                continue
            if not self.manager._is_active(a, now) or not self.manager._is_active(b, now):
                self.peer_owner.pop(a, None)
                self.peer_owner.pop(b, None)
                self.lock_bad_votes.pop(pair, None)
                self.last_lock_audit.pop(pair, None)
                self.stats["lock_releases"] += 1
                continue

            obs_a = float(self.last_observed.get(a, 0.0))
            obs_b = float(self.last_observed.get(b, 0.0))
            prev_a, prev_b = self.last_lock_audit.get(pair, (0.0, 0.0))
            if obs_a <= prev_a + 1e-6 or obs_b <= prev_b + 1e-6:
                continue
            self.last_lock_audit[pair] = (obs_a, obs_b)

            # Audit with the raw base similarity (without sticky bonus).
            score = super().tracklet_similarity(a, b, now)
            if score <= self.release_floor:
                self.lock_bad_votes[pair] += 1
            else:
                self.lock_bad_votes[pair] = 0

            if self.lock_bad_votes[pair] < self.release_votes:
                continue

            # NVIDIA-style correction principle: the more recent/weaker target
            # discards the propagated identity and returns to re-association.
            loser = a if self._binding_strength(a, now) < self._binding_strength(b, now) else b
            self.peer_owner.pop(a, None)
            self.peer_owner.pop(b, None)
            self.lock_bad_votes.pop(pair, None)
            self.last_lock_audit.pop(pair, None)
            self._fresh_anchor(loser, now)
            self.stats["corrections"] += 1
            self.stats["lock_corrections"] += 1

    def reconcile(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        self._audit_peer_locks(now)
        super().reconcile(now)

    def snapshot(self) -> dict:
        row = super().snapshot()
        row.update(
            {
                "peer_locks_active": len(self.peer_owner) // 2,
                "fresh_vote_skip": self.stats["fresh_vote_skip"],
                "peer_lock_blocks": self.stats["peer_lock_blocks"],
                "lock_releases": self.stats["lock_releases"],
                "lock_corrections": self.stats["lock_corrections"],
                "release_floor": self.release_floor,
            }
        )
        return row
