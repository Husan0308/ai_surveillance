from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


def _pct(values: deque[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((len(ordered) - 1) * float(quantile)))),
    )
    return float(ordered[index])


@dataclass(frozen=True)
class GallerySampleV1:
    camera_id: str
    local_track_id: str
    timestamp_ns: int
    embedding: np.ndarray
    quality_score: float
    detector_confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    sample_sequence: int


@dataclass(frozen=True)
class GalleryUpdateV1:
    accepted: bool
    action: str
    max_cosine: float
    sample_sequence: int | None
    evicted_sequence: int | None = None


@dataclass(frozen=True)
class GalleryViewV1:
    camera_id: str
    local_track_id: str
    last_seen_ns: int
    samples: tuple[GallerySampleV1, ...]


@dataclass
class _TrackGalleryV1:
    samples: list[GallerySampleV1] = field(default_factory=list)
    last_seen_ns: int = 0


class DiverseReIDGalleryV1:
    """Bounded per-local-track gallery retaining quality, diversity and recency.

    The first three valid samples bootstrap unconditionally. Later samples with
    cosine >= 0.975 are duplicates and only replace their nearest neighbour when
    Step-1 crop quality improves by at least 0.08. At capacity, each possible
    eight-of-nine subset is scored as 50% mean quality, 35% nearest-neighbour
    diversity, and 15% recency rank. The best subset wins with stable tie-breaks.
    This makes retention deterministic and explicitly avoids FIFO eviction.
    """

    def __init__(
        self,
        *,
        expected_dimension: int = 256,
        capacity: int = 8,
        bootstrap_samples: int = 3,
        duplicate_cosine: float = 0.975,
        quality_replace_gain: float = 0.08,
        expiry_sec: float = 12.0,
    ) -> None:
        self.expected_dimension = int(expected_dimension)
        self.capacity = int(capacity)
        self.bootstrap_samples = int(bootstrap_samples)
        self.duplicate_cosine = float(duplicate_cosine)
        self.quality_replace_gain = float(quality_replace_gain)
        self.expiry_ns = int(float(expiry_sec) * 1_000_000_000.0)
        if self.expected_dimension <= 0:
            raise ValueError("expected_dimension must be positive")
        if self.capacity != 8:
            raise ValueError("Step4 gallery capacity must remain exactly 8")
        if self.bootstrap_samples != 3:
            raise ValueError("Step4 gallery bootstrap must remain exactly 3")
        if not math.isclose(self.duplicate_cosine, 0.975, abs_tol=1e-12):
            raise ValueError("Step4 duplicate cosine must remain 0.975")
        if not 0.0 < self.quality_replace_gain <= 0.5:
            raise ValueError("invalid quality replacement gain")
        if self.expiry_ns <= 0:
            raise ValueError("expiry_sec must be positive")

        self._lock = threading.RLock()
        self._tracks: dict[tuple[str, str], _TrackGalleryV1] = {}
        self._sequence = 0
        self.gallery_bootstrap_add = 0
        self.gallery_diverse_add = 0
        self.gallery_duplicate_drop = 0
        self.gallery_quality_replace = 0
        self.gallery_full_reject_or_replace = 0
        self.gallery_invalid_reject = 0
        self.gallery_cleanup = 0
        self.update_ms: deque[float] = deque(maxlen=4096)

    def _embedding(self, value: np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.size != self.expected_dimension:
            raise ValueError(
                f"expected {self.expected_dimension}-D embedding, got {vector.size}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("non-finite embedding")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("zero or invalid embedding norm")
        normalized = np.ascontiguousarray(vector / norm, dtype=np.float32)
        normalized_norm = float(np.linalg.norm(normalized))
        if not math.isfinite(normalized_norm) or abs(normalized_norm - 1.0) > 1e-4:
            raise ValueError(f"embedding normalization failed norm={normalized_norm}")
        normalized.setflags(write=False)
        return normalized

    @staticmethod
    def _retention_utility(
        samples: list[GallerySampleV1],
        recency_rank: dict[int, float],
    ) -> float:
        qualities = np.asarray(
            [max(0.0, min(1.0, row.quality_score)) for row in samples],
            dtype=np.float64,
        )
        matrix = np.stack([row.embedding for row in samples], axis=0).astype(
            np.float64, copy=False
        )
        similarities = np.clip(matrix @ matrix.T, -1.0, 1.0)
        np.fill_diagonal(similarities, -1.0)
        nearest = np.max(similarities, axis=1)
        diversity = np.clip((1.0 - nearest) * 0.5, 0.0, 1.0)
        recency = np.asarray(
            [recency_rank[row.sample_sequence] for row in samples], dtype=np.float64
        )
        return float(
            0.50 * qualities.mean()
            + 0.35 * diversity.mean()
            + 0.15 * recency.mean()
        )

    @staticmethod
    def _redundancy(samples: list[GallerySampleV1], index: int) -> float:
        target = samples[index].embedding
        return max(
            float(np.dot(target, row.embedding))
            for other, row in enumerate(samples)
            if other != index
        )

    def _select_full_eviction(self, samples: list[GallerySampleV1]) -> int:
        ordered = sorted(
            samples,
            key=lambda row: (row.timestamp_ns, row.sample_sequence),
        )
        denominator = max(1, len(ordered) - 1)
        recency_rank = {
            row.sample_sequence: rank / denominator for rank, row in enumerate(ordered)
        }
        choices: list[tuple[tuple[float, float, float, int, int], int]] = []
        for index, evicted in enumerate(samples):
            retained = samples[:index] + samples[index + 1 :]
            utility = self._retention_utility(retained, recency_rank)
            tie_break = (
                round(utility, 12),
                -float(evicted.quality_score),
                self._redundancy(samples, index),
                -int(evicted.timestamp_ns),
                -int(evicted.sample_sequence),
            )
            choices.append((tie_break, index))
        return max(choices)[1]

    def update(
        self,
        *,
        camera_id: str,
        local_track_id: str,
        timestamp_ns: int,
        embedding: np.ndarray,
        quality_score: float,
        detector_confidence: float,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> GalleryUpdateV1:
        started = time.perf_counter()
        try:
            vector = self._embedding(embedding)
            camera = str(camera_id)
            track = str(local_track_id)
            if not camera or not track:
                raise ValueError("camera_id and local_track_id are required")
            timestamp = int(timestamp_ns)
            quality = float(quality_score)
            confidence = float(detector_confidence)
            bbox = tuple(float(value) for value in bbox_xyxy)
            if len(bbox) != 4 or not all(
                math.isfinite(value) for value in (*bbox, quality, confidence)
            ):
                raise ValueError("invalid gallery metadata")
        except (TypeError, ValueError):
            with self._lock:
                self.gallery_invalid_reject += 1
                self.update_ms.append((time.perf_counter() - started) * 1000.0)
            return GalleryUpdateV1(False, "invalid", -1.0, None)

        key = (camera, track)
        with self._lock:
            self._sequence += 1
            sample = GallerySampleV1(
                camera_id=camera,
                local_track_id=track,
                timestamp_ns=timestamp,
                embedding=vector,
                quality_score=quality,
                detector_confidence=confidence,
                bbox_xyxy=bbox,
                sample_sequence=self._sequence,
            )
            gallery = self._tracks.setdefault(key, _TrackGalleryV1())
            gallery.last_seen_ns = max(gallery.last_seen_ns, timestamp)
            current = gallery.samples
            max_cosine = -1.0

            if len(current) < self.bootstrap_samples:
                if current:
                    max_cosine = max(
                        float(np.dot(vector, row.embedding)) for row in current
                    )
                current.append(sample)
                self.gallery_bootstrap_add += 1
                decision = GalleryUpdateV1(
                    True, "bootstrap_add", max_cosine, sample.sample_sequence
                )
            else:
                cosine = [float(np.dot(vector, row.embedding)) for row in current]
                nearest_index = int(np.argmax(cosine))
                max_cosine = cosine[nearest_index]
                nearest = current[nearest_index]
                if max_cosine >= self.duplicate_cosine:
                    if quality >= nearest.quality_score + self.quality_replace_gain:
                        current[nearest_index] = sample
                        self.gallery_quality_replace += 1
                        decision = GalleryUpdateV1(
                            True,
                            "quality_replace",
                            max_cosine,
                            sample.sample_sequence,
                            nearest.sample_sequence,
                        )
                    else:
                        self.gallery_duplicate_drop += 1
                        decision = GalleryUpdateV1(
                            False, "duplicate_drop", max_cosine, sample.sample_sequence
                        )
                elif len(current) < self.capacity:
                    current.append(sample)
                    self.gallery_diverse_add += 1
                    decision = GalleryUpdateV1(
                        True, "diverse_add", max_cosine, sample.sample_sequence
                    )
                else:
                    pool = [*current, sample]
                    evict_index = self._select_full_eviction(pool)
                    evicted = pool[evict_index]
                    self.gallery_full_reject_or_replace += 1
                    if evicted.sample_sequence == sample.sample_sequence:
                        decision = GalleryUpdateV1(
                            False,
                            "full_reject",
                            max_cosine,
                            sample.sample_sequence,
                            sample.sample_sequence,
                        )
                    else:
                        gallery.samples = [
                            row for index, row in enumerate(pool) if index != evict_index
                        ]
                        self.gallery_diverse_add += 1
                        decision = GalleryUpdateV1(
                            True,
                            "full_replace",
                            max_cosine,
                            sample.sample_sequence,
                            evicted.sample_sequence,
                        )
            self.update_ms.append((time.perf_counter() - started) * 1000.0)
            return decision

    def touch_active(
        self,
        camera_id: str,
        active_local_track_ids: set[str] | frozenset[str],
        timestamp_ns: int,
    ) -> int:
        camera = str(camera_id)
        active = {str(track_id) for track_id in active_local_track_ids}
        timestamp = int(timestamp_ns)
        with self._lock:
            removed = 0
            for key in [key for key in self._tracks if key[0] == camera and key[1] not in active]:
                del self._tracks[key]
                removed += 1
            for key, gallery in self._tracks.items():
                if key[0] == camera and key[1] in active:
                    gallery.last_seen_ns = max(gallery.last_seen_ns, timestamp)
            self.gallery_cleanup += removed
            return removed

    def remove_track(self, camera_id: str, local_track_id: str) -> bool:
        key = (str(camera_id), str(local_track_id))
        with self._lock:
            removed = self._tracks.pop(key, None) is not None
            self.gallery_cleanup += int(removed)
            return removed

    def cleanup_expired(self, now_ns: int) -> int:
        cutoff = int(now_ns) - self.expiry_ns
        with self._lock:
            expired = [
                key for key, gallery in self._tracks.items() if gallery.last_seen_ns < cutoff
            ]
            for key in expired:
                del self._tracks[key]
            self.gallery_cleanup += len(expired)
            return len(expired)

    def samples_for(self, camera_id: str, local_track_id: str) -> tuple[GallerySampleV1, ...]:
        key = (str(camera_id), str(local_track_id))
        with self._lock:
            gallery = self._tracks.get(key)
            return tuple(gallery.samples) if gallery is not None else ()

    def gallery_views(self) -> tuple[GalleryViewV1, ...]:
        """Return an immutable, stably ordered snapshot for shadow diagnostics."""
        with self._lock:
            return tuple(
                GalleryViewV1(
                    camera_id=key[0],
                    local_track_id=key[1],
                    last_seen_ns=int(gallery.last_seen_ns),
                    samples=tuple(gallery.samples),
                )
                for key, gallery in sorted(self._tracks.items())
            )

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            counts = [len(gallery.samples) for gallery in self._tracks.values()]
            return {
                "gallery_tracks": len(counts),
                "gallery_samples": sum(counts),
                "gallery_tracks_ge3": sum(count >= 3 for count in counts),
                "gallery_max_samples": max(counts, default=0),
                "gallery_bootstrap_add": self.gallery_bootstrap_add,
                "gallery_diverse_add": self.gallery_diverse_add,
                "gallery_duplicate_drop": self.gallery_duplicate_drop,
                "gallery_quality_replace": self.gallery_quality_replace,
                "gallery_full_reject_or_replace": self.gallery_full_reject_or_replace,
                "gallery_invalid_reject": self.gallery_invalid_reject,
                "gallery_cleanup": self.gallery_cleanup,
                "gallery_update_p50_ms": _pct(self.update_ms, 0.50),
                "gallery_update_p95_ms": _pct(self.update_ms, 0.95),
            }
