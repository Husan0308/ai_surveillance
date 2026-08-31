from __future__ import annotations

import csv
import threading
import time
from collections import deque
from itertools import combinations
from pathlib import Path
from typing import Callable

from .step4_reid_gallery_v1 import GalleryViewV1
from .step4_reid_pair_scorer_v1 import GalleryPairScoreV1, score_gallery_pair_v1


TSV_COLUMNS = (
    "timestamp",
    "camera_a",
    "track_a",
    "samples_a",
    "camera_b",
    "track_b",
    "samples_b",
    "context",
    "pair_count",
    "max",
    "top2_mean",
    "top3_mean",
    "median",
    "mean",
    "p75",
    "p90",
    "a_best_mean",
    "a_best_min",
    "b_best_mean",
    "b_best_min",
    "support_ge_050",
    "support_ge_055",
    "support_ge_060",
    "support_ge_065",
    "support_ge_070",
    "support_ge_075",
    "support_ge_080",
    "robust_score",
)


def _pct(values: deque[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((len(ordered) - 1) * float(quantile)))),
    )
    return float(ordered[index])


class V11GalleryPairShadowWorkerV1:
    """Coalesced CPU shadow scorer over bounded, changed cross-camera galleries."""

    def __init__(
        self,
        snapshot_provider: Callable[[], tuple[GalleryViewV1, ...]],
        camera_rooms: dict[str, str],
        *,
        tsv_path: str | Path | None = "artifacts/reid/step4_pair_scores_v1.tsv",
        max_candidates: int = 24,
        recent_sec: float = 12.0,
        min_cycle_interval_sec: float = 2.0,
    ) -> None:
        self.snapshot_provider = snapshot_provider
        self.camera_rooms = {str(key): str(value) for key, value in camera_rooms.items()}
        self.tsv_path = Path(tsv_path).expanduser() if tsv_path else None
        self.max_candidates = max(1, min(64, int(max_candidates)))
        self.recent_ns = int(max(3.0, min(60.0, float(recent_sec))) * 1e9)
        self.min_cycle_interval_sec = max(
            0.5, min(10.0, float(min_cycle_interval_sec))
        )
        self._cv = threading.Condition()
        self._dirty = False
        self._stop = False
        self._thread: threading.Thread | None = None
        self._fingerprints: dict[
            tuple[tuple[str, str], tuple[str, str]], tuple[tuple[int, ...], tuple[int, ...]]
        ] = {}
        self.pairs_considered = 0
        self.pairs_scored = 0
        self.pairs_insufficient = 0
        self.pairs_invalid = 0
        self.same_room_pairs = 0
        self.different_room_pairs = 0
        self.worker_errors = 0
        self.score_ms: deque[float] = deque(maxlen=4096)

    def start(self) -> None:
        with self._cv:
            if self._thread is not None:
                return
            if self.tsv_path is not None:
                self.tsv_path.parent.mkdir(parents=True, exist_ok=True)
                with self.tsv_path.open("w", encoding="utf-8", newline="") as handle:
                    csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(
                        TSV_COLUMNS
                    )
            self._thread = threading.Thread(
                target=self._run,
                name="camera-v11-step4-reid-pair-shadow",
                daemon=True,
            )
            self._thread.start()

    def notify(self) -> None:
        with self._cv:
            if self._stop:
                return
            self._dirty = True
            self._cv.notify()

    @staticmethod
    def _view_key(view: GalleryViewV1) -> tuple[str, str]:
        return view.camera_id, view.local_track_id

    def _pair_key(
        self, first: GalleryViewV1, second: GalleryViewV1
    ) -> tuple[tuple[str, str], tuple[str, str]]:
        return self._view_key(first), self._view_key(second)

    def _context(self, camera_a: str, camera_b: str) -> str:
        room_a = self.camera_rooms[camera_a]
        room_b = self.camera_rooms[camera_b]
        return "same_room" if room_a == room_b else "different_room"

    def _candidates(
        self, views: tuple[GalleryViewV1, ...]
    ) -> list[
        tuple[
            tuple[int, int, tuple[str, str], tuple[str, str]],
            GalleryViewV1,
            GalleryViewV1,
            str,
            tuple[tuple[int, ...], tuple[int, ...]],
        ]
    ]:
        cutoff = time.monotonic_ns() - self.recent_ns
        recent = [
            view for view in views if view.samples and int(view.last_seen_ns) >= cutoff
        ]
        rows = []
        active_pair_keys = set()
        for left, right in combinations(recent, 2):
            if left.camera_id == right.camera_id:
                continue
            first, second = sorted((left, right), key=self._view_key)
            pair_key = self._pair_key(first, second)
            active_pair_keys.add(pair_key)
            fingerprint = (
                tuple(sample.sample_sequence for sample in first.samples),
                tuple(sample.sample_sequence for sample in second.samples),
            )
            if self._fingerprints.get(pair_key) == fingerprint:
                continue
            camera_pair = frozenset((first.camera_id, second.camera_id))
            target_priority = int(camera_pair != frozenset(("CAM-01", "CAM-04")))
            newest = max(int(first.last_seen_ns), int(second.last_seen_ns))
            context = self._context(first.camera_id, second.camera_id)
            priority = (
                target_priority,
                -newest,
                self._view_key(first),
                self._view_key(second),
            )
            rows.append((priority, first, second, context, fingerprint))
        self._fingerprints = {
            key: value
            for key, value in self._fingerprints.items()
            if key in active_pair_keys
        }
        rows.sort(key=lambda row: row[0])
        return rows[: self.max_candidates]

    @staticmethod
    def _number(value: float | None) -> str:
        return "" if value is None else f"{float(value):.6f}"

    def _tsv_row(
        self,
        first: GalleryViewV1,
        second: GalleryViewV1,
        context: str,
        score: GalleryPairScoreV1,
    ) -> tuple[str, ...] | None:
        if self.tsv_path is None or score.status != "OK":
            return None
        return (
            str(time.time_ns()),
            first.camera_id,
            first.local_track_id,
            str(len(first.samples)),
            second.camera_id,
            second.local_track_id,
            str(len(second.samples)),
            context,
            str(score.pair_count),
            self._number(score.max_score),
            self._number(score.top2_mean),
            self._number(score.top3_mean),
            self._number(score.median_score),
            self._number(score.mean_score),
            self._number(score.p75_score),
            self._number(score.p90_score),
            self._number(score.a_best_mean),
            self._number(score.a_best_min),
            self._number(score.b_best_mean),
            self._number(score.b_best_min),
            str(score.support_ge_050),
            str(score.support_ge_055),
            str(score.support_ge_060),
            str(score.support_ge_065),
            str(score.support_ge_070),
            str(score.support_ge_075),
            str(score.support_ge_080),
            self._number(score.robust_score),
        )

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
            started = time.perf_counter()
            score = score_gallery_pair_v1(
                [sample.embedding for sample in first.samples],
                [sample.embedding for sample in second.samples],
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
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
        if self.tsv_path is not None and tsv_rows:
            with self.tsv_path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(
                    tsv_rows
                )

    def _run(self) -> None:
        next_cycle_at = 0.0
        while True:
            with self._cv:
                while not self._dirty and not self._stop:
                    self._cv.wait(timeout=0.5)
                if self._stop and not self._dirty:
                    return
                while not self._stop:
                    remaining = next_cycle_at - time.monotonic()
                    if remaining <= 0.0:
                        break
                    self._cv.wait(timeout=remaining)
                self._dirty = False
            try:
                self._score_cycle()
                next_cycle_at = time.monotonic() + self.min_cycle_interval_sec
            except Exception as exc:
                with self._cv:
                    self.worker_errors += 1
                print(
                    "V11_STEP4_REID_PAIR_SHADOW_ERROR "
                    f"error={type(exc).__name__}:{exc}",
                    flush=True,
                )

    def snapshot(self) -> dict[str, int | float]:
        with self._cv:
            return {
                "pairs_considered": self.pairs_considered,
                "pairs_scored": self.pairs_scored,
                "pairs_insufficient": self.pairs_insufficient,
                "pairs_invalid": self.pairs_invalid,
                "same_room_pairs": self.same_room_pairs,
                "different_room_pairs": self.different_room_pairs,
                "score_p50_ms": _pct(self.score_ms, 0.50),
                "score_p95_ms": _pct(self.score_ms, 0.95),
                "worker_errors": self.worker_errors,
            }

    def close(self, timeout_sec: float = 3.0) -> None:
        with self._cv:
            if self._thread is None:
                return
            self._dirty = True
            self._stop = True
            self._cv.notify_all()
            thread = self._thread
        thread.join(timeout=max(0.1, float(timeout_sec)))
        with self._cv:
            self._thread = None
