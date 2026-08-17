from __future__ import annotations

import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass

Vector = tuple[float, ...]
LocalKey = tuple[int, int]


def _normalize(values) -> Vector | None:
    try:
        row = tuple(float(v) for v in values)
    except Exception:
        return None
    if not row:
        return None
    norm2 = sum(v * v for v in row)
    if norm2 <= 1e-12 or not math.isfinite(norm2):
        return None
    inv = 1.0 / math.sqrt(norm2)
    return tuple(v * inv for v in row)


def _dot(a: Vector | None, b: Vector | None) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    return float(sum(x * y for x, y in zip(a, b)))


def _weighted_mean(vectors: list[Vector], weights: list[float]) -> Vector | None:
    if not vectors or len(vectors) != len(weights):
        return None
    total = sum(max(1e-6, float(w)) for w in weights)
    mean = tuple(
        sum(v[i] * max(1e-6, float(w)) for v, w in zip(vectors, weights)) / total
        for i in range(len(vectors[0]))
    )
    return _normalize(mean)


def _quantile(values: deque[float] | list[float], q: float) -> float:
    rows = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not rows:
        return -1.0
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(rows) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return rows[lo]
    frac = pos - lo
    return rows[lo] * (1.0 - frac) + rows[hi] * frac


@dataclass
class BankSample:
    vector: Vector
    color: Vector | None
    quality: float
    seen_at: float


@dataclass
class PairVote:
    votes: int = 0
    first_at: float = 0.0
    last_at: float = 0.0
    bad_votes: int = 0


class AdaptiveTrackletReID:
    """Training-free, tracklet-level peer-camera identity reconciler.

    The embedding network stays frozen. Adaptation happens only in memory and in
    the decision layer: each NvDCF local track owns a small diverse feature bank;
    peer cameras are compared with robust multi-frame similarity; camera-pair
    thresholds learn only from conservative online negatives; and a merge requires
    one-to-one mutual-best evidence that persists over time.

    This deliberately never changes detector/tracker geometry and never performs
    gradient updates, so stationary office data cannot overfit the neural network.
    """

    def __init__(self, manager) -> None:
        self.manager = manager
        self.bank_size = max(6, min(24, int(os.environ.get("CAMERA_V2_ADAPT_BANK", "12"))))
        self.bank_ttl = max(8.0, float(os.environ.get("CAMERA_V2_ADAPT_BANK_TTL", "35")))
        self.min_samples = max(2, int(os.environ.get("CAMERA_V2_ADAPT_MIN_SAMPLES", "3")))
        self.duplicate_similarity = float(os.environ.get("CAMERA_V2_ADAPT_DUP_SIM", "0.985"))
        self.redundant_similarity = float(os.environ.get("CAMERA_V2_ADAPT_REDUNDANT_SIM", "0.965"))
        self.min_quality = max(0.0, float(os.environ.get("CAMERA_V2_ADAPT_MIN_QUALITY", "0.08")))

        self.scan_interval = max(0.20, float(os.environ.get("CAMERA_V2_ADAPT_SCAN_SEC", "0.45")))
        self.base_same = float(os.environ.get("CAMERA_V2_ADAPT_BASE_SAME", "0.50"))
        self.max_same = float(os.environ.get("CAMERA_V2_ADAPT_MAX_SAME", "0.68"))
        self.min_margin = float(os.environ.get("CAMERA_V2_ADAPT_MARGIN", "0.045"))
        self.negative_quantile = float(os.environ.get("CAMERA_V2_ADAPT_NEG_Q", "0.90"))
        self.negative_gap = float(os.environ.get("CAMERA_V2_ADAPT_NEG_GAP", "0.055"))
        self.votes_required = max(3, int(os.environ.get("CAMERA_V2_ADAPT_VOTES", "4")))
        self.min_vote_span = max(0.4, float(os.environ.get("CAMERA_V2_ADAPT_VOTE_SPAN", "1.20")))
        self.very_strong = float(os.environ.get("CAMERA_V2_ADAPT_VERY_STRONG", "0.67"))
        self.split_floor = float(os.environ.get("CAMERA_V2_ADAPT_SPLIT_FLOOR", "0.30"))
        self.split_votes = max(4, int(os.environ.get("CAMERA_V2_ADAPT_SPLIT_VOTES", "6")))
        self.alt_margin = float(os.environ.get("CAMERA_V2_ADAPT_ALT_MARGIN", "0.10"))

        self.banks: dict[LocalKey, deque[BankSample]] = {}
        self.pair_negatives: dict[tuple[int, int], deque[float]] = defaultdict(
            lambda: deque(maxlen=160)
        )
        self.pair_votes: dict[frozenset[LocalKey], PairVote] = {}
        self.last_scan = 0.0
        self.last_pair_score = -1.0
        self.last_pair_margin = -1.0
        self.last_threshold = self.base_same
        self.last_camera_pair = "none"

        self.stats = {
            "rows": 0,
            "bank_add": 0,
            "duplicate_skip": 0,
            "bank_replace": 0,
            "scans": 0,
            "comparisons": 0,
            "mutual": 0,
            "adaptive_merges": 0,
            "samecam_collisions": 0,
            "samecam_repairs": 0,
            "contradictions": 0,
            "corrections": 0,
            "negative_updates": 0,
        }

    def _prune_bank(self, key: LocalKey, now: float) -> deque[BankSample]:
        bank = self.banks.get(key)
        if bank is None:
            bank = deque(maxlen=self.bank_size)
            self.banks[key] = bank
            return bank
        kept = [row for row in bank if now - row.seen_at <= self.bank_ttl]
        if len(kept) != len(bank):
            bank = deque(kept[-self.bank_size :], maxlen=self.bank_size)
            self.banks[key] = bank
        return bank

    @staticmethod
    def _sample_quality(row: dict) -> float:
        det = max(0.0, float(row.get("confidence", 0.0) or 0.0))
        trk = max(0.0, float(row.get("tracker_confidence", 0.0) or 0.0))
        return max(det, trk)

    def observe_rows(self, rows: list[dict], now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        for row in rows:
            sid = int(row.get("source_id", -1))
            oid = int(row.get("object_id", -1))
            vector = _normalize(row.get("feature", ()))
            if sid < 0 or oid < 0 or vector is None:
                continue
            color = _normalize(row.get("color_feature", ()))
            quality = self._sample_quality(row)
            if quality < self.min_quality:
                continue
            key = (sid, oid)
            bank = self._prune_bank(key, now)
            sample = BankSample(vector, color, quality, now)
            self.stats["rows"] += 1

            if not bank:
                bank.append(sample)
                self.stats["bank_add"] += 1
                continue

            sims = [_dot(vector, old.vector) for old in bank]
            nearest_index = max(range(len(sims)), key=sims.__getitem__)
            nearest_sim = sims[nearest_index]
            nearest = bank[nearest_index]

            if nearest_sim >= self.duplicate_similarity:
                # A stationary person can generate thousands of near-identical
                # crops. Keep only a materially better representative.
                if quality >= nearest.quality + 0.08:
                    rows2 = list(bank)
                    rows2[nearest_index] = sample
                    self.banks[key] = deque(rows2, maxlen=self.bank_size)
                    self.stats["bank_replace"] += 1
                else:
                    self.stats["duplicate_skip"] += 1
                continue

            if len(bank) < self.bank_size:
                bank.append(sample)
                self.stats["bank_add"] += 1
                continue

            # Full bank: evict the most redundant, lowest-quality old sample only
            # when the new sample adds viewpoint/pose diversity or quality.
            rows2 = list(bank)
            redundancy: list[float] = []
            for i, old in enumerate(rows2):
                others = [_dot(old.vector, x.vector) for j, x in enumerate(rows2) if j != i]
                redundancy.append(max(others) if others else -1.0)
            victim = max(
                range(len(rows2)),
                key=lambda i: 0.72 * redundancy[i] - 0.28 * rows2[i].quality,
            )
            adds_diversity = nearest_sim < self.redundant_similarity
            better_quality = quality >= rows2[victim].quality + 0.08
            if adds_diversity or better_quality:
                rows2[victim] = sample
                self.banks[key] = deque(rows2, maxlen=self.bank_size)
                self.stats["bank_replace"] += 1
            else:
                self.stats["duplicate_skip"] += 1

        # Bound memory when NvDCF IDs disappear.
        stale = []
        for key, bank in self.banks.items():
            if not bank or now - bank[-1].seen_at > self.bank_ttl * 1.5:
                stale.append(key)
        for key in stale:
            self.banks.pop(key, None)
            self.pair_votes = {
                pair: vote for pair, vote in self.pair_votes.items() if key not in pair
            }

    def _prototype(self, key: LocalKey, now: float) -> tuple[Vector | None, Vector | None, int]:
        bank = list(self._prune_bank(key, now))
        if not bank:
            return None, None, 0
        vectors = [row.vector for row in bank]
        if len(vectors) == 1:
            return vectors[0], bank[0].color, 1

        means = []
        for i, vector in enumerate(vectors):
            sims = [_dot(vector, other) for j, other in enumerate(vectors) if j != i]
            means.append(sum(sims) / max(1, len(sims)))
        medoid_index = max(range(len(vectors)), key=lambda i: means[i])
        medoid = vectors[medoid_index]
        inliers = [
            (row.vector, 0.55 + 0.45 * min(1.0, row.quality))
            for row in bank
            if _dot(medoid, row.vector) >= 0.34
        ]
        if len(inliers) < 2:
            inliers = [(row.vector, 1.0) for row in bank]
        vector = _weighted_mean([v for v, _ in inliers], [w for _, w in inliers]) or medoid

        colors = [(row.color, 0.55 + 0.45 * min(1.0, row.quality)) for row in bank if row.color]
        color = _weighted_mean(
            [v for v, _ in colors if v is not None],
            [w for v, w in colors if v is not None],
        ) if colors else None
        return vector, color, len(bank)

    def tracklet_similarity(self, a: LocalKey, b: LocalKey, now: float) -> float:
        ba = list(self._prune_bank(a, now))
        bb = list(self._prune_bank(b, now))
        if len(ba) < self.min_samples or len(bb) < self.min_samples:
            return -1.0
        sims = sorted((_dot(x.vector, y.vector) for x in ba for y in bb), reverse=True)
        if not sims:
            return -1.0
        topn = max(2, min(6, int(math.ceil(len(sims) * 0.20))))
        top = sims[:topn]
        top_mean = sum(top) / len(top)
        stable_top = top[min(len(top) - 1, max(0, len(top) // 2))]
        va, ca, _ = self._prototype(a, now)
        vb, cb, _ = self._prototype(b, now)
        proto = _dot(va, vb)
        score = 0.56 * top_mean + 0.34 * proto + 0.10 * stable_top
        if ca and cb:
            score = 0.97 * score + 0.03 * _dot(ca, cb)
        return max(-1.0, min(1.0, float(score)))

    def _camera_threshold(self, camera_pair: tuple[int, int]) -> float:
        negatives = self.pair_negatives[camera_pair]
        threshold = self.base_same
        if len(negatives) >= 12:
            q = _quantile(negatives, self.negative_quantile)
            threshold = max(threshold, q + self.negative_gap)
        return max(self.base_same, min(self.max_same, threshold))

    def _binding_strength(self, key: LocalKey, now: float) -> tuple[int, int, int, int, float]:
        binding = self.manager.bindings.get(key)
        if binding is None:
            return (-1, -1, -1, -1, 0.0)
        gid = self.manager._resolve(binding.global_id)
        profile = self.manager.profiles.get(gid)
        known = int(bool(profile is not None and profile.known_name))
        state = {"provisional": 0, "anchor": 1, "confirmed": 2}.get(binding.state, 0)
        samples = int(profile.sample_count) if profile is not None else 0
        bank_n = len(self._prune_bank(key, now))
        older = -float(binding.first_seen)
        return (known, state, samples, bank_n, older)

    def _detach(self, key: LocalKey, now: float) -> None:
        manager = self.manager
        binding = manager.bindings.get(key)
        if binding is None:
            return
        old_gid = manager._resolve(binding.global_id)
        vector, color, count = manager._track_prototype(key)
        if vector is not None and count >= 2:
            decision = manager._candidate_decision(vector, color, key, now, exclude_gid=old_gid)
            alt_gid = decision[0]
            accepted = bool(decision[-1])
            if accepted and alt_gid is not None:
                manager._switch_binding(binding, key, int(alt_gid), now, provisional=True)
                return
            quality = max((item.quality for item in manager._evidence(key)), default=0.5)
            manager._correct_to_new_anchor(
                binding, key, vector, color, binding.last_bbox, now, quality
            )
            return

        manager._remove_owner_contributions(old_gid, key)
        profile = manager._new_profile(key[0], now)
        binding.global_id = profile.global_id
        binding.state = "provisional"
        binding.confirm_votes = 0
        binding.bad_votes = 0
        binding.switch_candidate = None
        binding.switch_votes = 0
        binding.last_committed_at = 0.0

    def _repair_same_camera(self, now: float) -> None:
        groups = defaultdict(list)
        for key, binding in self.manager.bindings.items():
            if self.manager._is_active(key, now):
                groups[(key[0], self.manager._resolve(binding.global_id))].append(key)
        for (_source, _gid), keys in groups.items():
            if len(keys) <= 1:
                continue
            self.stats["samecam_collisions"] += len(keys) - 1
            keys.sort(key=lambda k: self._binding_strength(k, now), reverse=True)
            keeper = keys[0]
            for loser in keys[1:]:
                self.manager.cannot_link[frozenset((keeper, loser))] = now + 30.0
                self._detach(loser, now)
                self.stats["samecam_repairs"] += 1

    def _merge(self, a: LocalKey, b: LocalKey, now: float) -> bool:
        manager = self.manager
        ba = manager.bindings.get(a)
        bb = manager.bindings.get(b)
        if ba is None or bb is None or a[0] == b[0]:
            return False
        if not manager._is_active(a, now) or not manager._is_active(b, now):
            return False
        if manager.room_of(a[0]) != manager.room_of(b[0]):
            return False
        ga = manager._resolve(ba.global_id)
        gb = manager._resolve(bb.global_id)
        if ga == gb:
            return True

        keep, move = (a, b)
        if self._binding_strength(b, now) > self._binding_strength(a, now):
            keep, move = b, a
        keep_gid = manager._resolve(manager.bindings[keep].global_id)
        # Only this local peer pair is released from an appearance cannot-link.
        # Same-camera and cross-room impossibilities are checked above and are not cleared.
        manager.cannot_link.pop(frozenset((a, b)), None)
        manager._switch_binding(
            manager.bindings[move], move, keep_gid, now, provisional=False
        )
        moved = manager.bindings[move]
        moved.confirm_votes = manager.confirm_votes_required
        moved.bad_votes = 0
        self.stats["adaptive_merges"] += 1
        return True

    def _update_vote(
        self,
        pair: frozenset[LocalKey],
        good: bool,
        score: float,
        margin: float,
        now: float,
    ) -> bool:
        state = self.pair_votes.get(pair)
        if state is None:
            state = PairVote()
            self.pair_votes[pair] = state
        if not good:
            state.votes = max(0, state.votes - 1)
            if state.votes == 0:
                state.first_at = 0.0
            state.last_at = now
            return False
        if state.votes == 0:
            state.first_at = now
        state.votes += 1
        state.last_at = now
        required = self.votes_required
        span_required = self.min_vote_span
        if score >= self.very_strong and margin >= max(0.08, self.min_margin):
            required = max(3, self.votes_required - 1)
            span_required = max(0.65, self.min_vote_span * 0.60)
        return state.votes >= required and now - state.first_at >= span_required

    def reconcile(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        self._repair_same_camera(now)
        if now - self.last_scan < self.scan_interval:
            return
        self.last_scan = now
        self.stats["scans"] += 1

        active = [
            key
            for key in self.manager.bindings
            if self.manager._is_active(key, now)
            and len(self._prune_bank(key, now)) >= self.min_samples
        ]
        by_room = defaultdict(lambda: defaultdict(list))
        for key in active:
            by_room[self.manager.room_of(key[0])][key[0]].append(key)

        seen_vote_pairs: set[frozenset[LocalKey]] = set()
        for _room, source_rows in by_room.items():
            sources = sorted(source_rows)
            for si, source_a in enumerate(sources):
                for source_b in sources[si + 1 :]:
                    camera_pair = (min(source_a, source_b), max(source_a, source_b))
                    akeys = source_rows[source_a]
                    bkeys = source_rows[source_b]
                    if not akeys or not bkeys:
                        continue

                    matrix: dict[tuple[LocalKey, LocalKey], float] = {}
                    for a in akeys:
                        for b in bkeys:
                            score = self.tracklet_similarity(a, b, now)
                            if score >= -0.5:
                                matrix[(a, b)] = score
                                self.stats["comparisons"] += 1
                    if not matrix:
                        continue

                    rows_a = defaultdict(list)
                    rows_b = defaultdict(list)
                    for (a, b), score in matrix.items():
                        rows_a[a].append((score, b))
                        rows_b[b].append((score, a))
                    for rows in list(rows_a.values()) + list(rows_b.values()):
                        rows.sort(reverse=True, key=lambda x: x[0])

                    best_a = {a: rows[0] for a, rows in rows_a.items() if rows}
                    best_b = {b: rows[0] for b, rows in rows_b.items() if rows}

                    # Conservative online negative calibration: only learn from a
                    # pair when both ends prefer somebody else by a clear margin.
                    for (a, b), score in matrix.items():
                        ba = best_a.get(a)
                        bb = best_b.get(b)
                        if ba is None or bb is None:
                            continue
                        if ba[1] == b or bb[1] == a:
                            continue
                        if ba[0] >= score + 0.05 and bb[0] >= score + 0.05:
                            self.pair_negatives[camera_pair].append(score)
                            self.stats["negative_updates"] += 1

                    threshold = self._camera_threshold(camera_pair)
                    self.last_threshold = threshold
                    self.last_camera_pair = f"{camera_pair[0]}-{camera_pair[1]}"

                    for a, (score, b) in best_a.items():
                        reciprocal = best_b.get(b)
                        if reciprocal is None or reciprocal[1] != a:
                            continue
                        second_a = rows_a[a][1][0] if len(rows_a[a]) > 1 else -1.0
                        second_b = rows_b[b][1][0] if len(rows_b[b]) > 1 else -1.0
                        margin_a = score - second_a if second_a >= -0.5 else 1.0
                        margin_b = score - second_b if second_b >= -0.5 else 1.0
                        margin = min(margin_a, margin_b)
                        self.stats["mutual"] += 1
                        self.last_pair_score = score
                        self.last_pair_margin = margin
                        pair = frozenset((a, b))
                        seen_vote_pairs.add(pair)

                        ga = self.manager._resolve(self.manager.bindings[a].global_id)
                        gb = self.manager._resolve(self.manager.bindings[b].global_id)
                        good = score >= threshold and margin >= self.min_margin
                        if ga != gb:
                            if self._update_vote(pair, good, score, margin, now):
                                if self._merge(a, b, now):
                                    self.pair_votes.pop(pair, None)
                        else:
                            # Shared IDs stay under continuous audit. A split is
                            # allowed only when this pair becomes implausible AND one
                            # side has a clearly better alternative, never from a
                            # low cross-view score alone.
                            state = self.pair_votes.setdefault(pair, PairVote())
                            alt_a = best_a.get(a)
                            alt_b = best_b.get(b)
                            better_alt = False
                            if alt_a and alt_a[1] != b and alt_a[0] >= score + self.alt_margin:
                                better_alt = True
                            if alt_b and alt_b[1] != a and alt_b[0] >= score + self.alt_margin:
                                better_alt = True
                            contradictory = better_alt and score <= max(
                                self.split_floor, threshold - 0.12
                            )
                            state.bad_votes = state.bad_votes + 1 if contradictory else 0
                            if contradictory:
                                self.stats["contradictions"] += 1
                            if state.bad_votes >= self.split_votes:
                                loser = a if self._binding_strength(a, now) < self._binding_strength(b, now) else b
                                self._detach(loser, now)
                                self.stats["corrections"] += 1
                                self.pair_votes.pop(pair, None)

        # Votes must be fresh; disappearing or non-reciprocal pairs decay instead
        # of causing a delayed merge after the scene has changed.
        for pair, state in list(self.pair_votes.items()):
            if pair in seen_vote_pairs:
                continue
            state.votes = max(0, state.votes - 1)
            state.bad_votes = max(0, state.bad_votes - 1)
            if state.votes == 0 and state.bad_votes == 0 and now - state.last_at > 3.0:
                self.pair_votes.pop(pair, None)

    def snapshot(self) -> dict:
        thresholds = {
            f"{a}-{b}": self._camera_threshold((a, b))
            for a, b in sorted(self.pair_negatives)
        }
        return {
            **self.stats,
            "banks": len(self.banks),
            "bank_samples": sum(len(bank) for bank in self.banks.values()),
            "vote_pairs": len(self.pair_votes),
            "last_pair_score": self.last_pair_score,
            "last_pair_margin": self.last_pair_margin,
            "last_threshold": self.last_threshold,
            "last_camera_pair": self.last_camera_pair,
            "thresholds": thresholds,
        }
