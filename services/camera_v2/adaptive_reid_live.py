from __future__ import annotations

import time
from collections import deque

from .adaptive_reid import AdaptiveTrackletReID, BankSample, LocalKey, _dot, _normalize


class LiveAdaptiveTrackletReID(AdaptiveTrackletReID):
    """Stationary-office hardening for AdaptiveTrackletReID.

    A seated worker may yield almost identical embeddings for minutes. We still
    need several independent observations before association, but after bootstrap
    we must not fill the whole bank with duplicate poses. This layer therefore
    keeps the first `min_samples` observations even when nearly identical, then
    switches to diversity-preserving replacement. Temporal votes also advance only
    when both peer tracks have produced fresh ReID observations since the last vote.
    """

    def __init__(self, manager) -> None:
        self.last_observed: dict[LocalKey, float] = {}
        super().__init__(manager)

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
            self.last_observed[key] = now
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
                # Bootstrap still requires several truly separate observations,
                # even if the worker is sitting motionless and all crops look alike.
                if len(bank) < self.min_samples:
                    bank.append(sample)
                    self.stats["bank_add"] += 1
                elif quality >= nearest.quality + 0.08:
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

        stale = []
        for key, bank in self.banks.items():
            last = self.last_observed.get(key, bank[-1].seen_at if bank else 0.0)
            if not bank or now - last > self.bank_ttl * 1.5:
                stale.append(key)
        for key in stale:
            self.banks.pop(key, None)
            self.last_observed.pop(key, None)
            self.pair_votes = {
                pair: vote for pair, vote in self.pair_votes.items() if key not in pair
            }

    def _update_vote(self, pair, good: bool, score: float, margin: float, now: float) -> bool:
        if good:
            keys = tuple(pair)
            if len(keys) == 2:
                evidence_at = min(
                    self.last_observed.get(keys[0], 0.0),
                    self.last_observed.get(keys[1], 0.0),
                )
                state = self.pair_votes.get(pair)
                if evidence_at <= 0.0:
                    return False
                if state is not None and state.last_at > 0.0 and evidence_at <= state.last_at + 1e-6:
                    return False
                # Use evidence time rather than service-loop time. This prevents a
                # fast probe loop from turning one crop into four fake votes.
                now = evidence_at
        return super()._update_vote(pair, good, score, margin, now)
