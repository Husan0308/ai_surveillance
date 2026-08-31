from __future__ import annotations

import csv
import time
from typing import Callable

from .step4_reid_pair_score_cache_v1 import score_gallery_pair_step3_cached_v1
from .step4_reid_pair_shadow_v1 import V11GalleryPairShadowWorkerV1


class V11GalleryPairShadowWorkerCachedV2(V11GalleryPairShadowWorkerV1):
    """Step-3 shadow scorer that publishes exact reusable pair evidence.

    The authoritative Step-3 scorer still runs on every cache miss. Results are
    keyed by immutable gallery sample fingerprints so the later Step-4 matcher can
    consume the exact same GalleryPairScoreV1 instead of recomputing it.

    A lightweight completion callback lets Step4 wake only *after* this Step3
    cycle has finished publishing cache evidence. This removes the former race in
    which Step3 and Step4 were independently woken by the same ReID result and the
    matcher could inspect a newer gallery fingerprint before Step3 had published
    the corresponding score. The callback carries no frame or embedding data and
    never runs camera/tracker/ReID work.
    """

    def __init__(
        self,
        *args,
        on_scores_published: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._on_scores_published = on_scores_published

    def set_scores_published_callback(
        self, callback: Callable[[], None] | None
    ) -> None:
        self._on_scores_published = callback

    def _score_cycle(self) -> None:
        views = self.snapshot_provider()
        tsv_rows: list[tuple[str, ...]] = []
        for _priority, first, second, context, fingerprint in self._candidates(views):
            pair_key = self._pair_key(first, second)
            self._fingerprints[pair_key] = fingerprint
            with self._cv:
                self.pairs_considered += 1
                if context == "same_room":
                    self.same_room_pairs += 1
                else:
                    self.different_room_pairs += 1

            started_ns = time.perf_counter_ns()
            score = score_gallery_pair_step3_cached_v1(first, second)
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0

            with self._cv:
                self.score_ms.append(elapsed_ms)
                if score.status == "OK":
                    self.pairs_scored += 1
                elif score.status == "INSUFFICIENT":
                    self.pairs_insufficient += 1
                else:
                    self.pairs_invalid += 1

            tsv_row = self._tsv_row(first, second, context, score)
            if tsv_row is not None:
                tsv_rows.append(tsv_row)

        # Step4 is a consumer of this exact cache evidence. Wake it only after the
        # complete Step3 scoring cycle has published its results. Call even when
        # there were no changed candidates: an unchanged exact score can still be
        # a valid repeated matcher observation while the underlying local tracks
        # remain active/recent.
        callback = self._on_scores_published
        if callback is not None:
            callback()

        if self.tsv_path is not None and tsv_rows:
            with self.tsv_path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(
                    tsv_rows
                )
