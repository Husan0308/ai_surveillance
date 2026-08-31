from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    row = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(row))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("invalid zero ReID embedding")
    return row / norm


def _pct(values: deque[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


@dataclass(frozen=True)
class CropGateDecision:
    accepted: bool
    reason: str
    quality: float
    blur: float
    aspect_hw: float
    edge_contacts: int
    step_ms: float


@dataclass
class _DiverseGallery:
    embeddings: list[np.ndarray] = field(default_factory=list)
    qualities: list[float] = field(default_factory=list)


class V11ReIDQualityDiversityGateV1:
    """Cheap pre-inference crop gate plus post-inference embedding diversity gate.

    This layer is intentionally conservative and shadow-safe:
    - it never changes detector/tracker IDs;
    - severe crop defects are rejected before GPU ReID work;
    - near-duplicate embeddings do not repeatedly pollute downstream galleries;
    - a materially higher-quality duplicate replaces the nearest stored sample.
    """

    def __init__(
        self,
        *,
        min_quality: float = 0.34,
        min_blur: float = 18.0,
        min_aspect_hw: float = 0.90,
        max_aspect_hw: float = 6.0,
        reject_edge_contacts: int = 2,
        duplicate_cosine: float = 0.975,
        replace_quality_gain: float = 0.08,
        gallery_size: int = 8,
    ) -> None:
        self.min_quality = max(0.05, min(0.90, float(min_quality)))
        self.min_blur = max(0.0, float(min_blur))
        self.min_aspect_hw = max(0.4, float(min_aspect_hw))
        self.max_aspect_hw = max(self.min_aspect_hw + 0.2, float(max_aspect_hw))
        self.reject_edge_contacts = max(1, min(4, int(reject_edge_contacts)))
        self.duplicate_cosine = max(0.85, min(0.9999, float(duplicate_cosine)))
        self.replace_quality_gain = max(0.0, min(0.50, float(replace_quality_gain)))
        self.gallery_size = max(2, min(16, int(gallery_size)))

        self._lock = threading.RLock()
        self._galleries: dict[tuple[str, str], _DiverseGallery] = {}

        self.crop_checked = 0
        self.crop_accepted = 0
        self.reject_edge = 0
        self.reject_blur = 0
        self.reject_aspect = 0
        self.reject_quality = 0
        self.embedding_checked = 0
        self.embedding_accepted = 0
        self.duplicate_drops = 0
        self.duplicate_replacements = 0
        self.crop_gate_ms: deque[float] = deque(maxlen=2048)
        self.blur_values: deque[float] = deque(maxlen=2048)
        self.quality_values: deque[float] = deque(maxlen=2048)
        self.nearest_cos_values: deque[float] = deque(maxlen=2048)

    @staticmethod
    def _sharpness(crop_bgr: np.ndarray) -> float:
        h, w = crop_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return 0.0
        scale = min(1.0, 128.0 / float(max(h, w)))
        if scale < 0.999:
            small = cv2.resize(
                crop_bgr,
                (max(8, int(round(w * scale))), max(8, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = crop_bgr
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())

    def evaluate_crop(
        self,
        *,
        crop_bgr: np.ndarray,
        detector_score: float,
        frame_width: int,
        frame_height: int,
        bbox_xyxy: tuple[int, int, int, int],
    ) -> CropGateDecision:
        started = time.perf_counter()
        x1, y1, x2, y2 = bbox_xyxy
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect_hw = height / float(width)
        edge_contacts = int(x1 <= 1) + int(y1 <= 1) + int(x2 >= frame_width - 1) + int(
            y2 >= frame_height - 1
        )
        blur = self._sharpness(crop_bgr)

        size_score = min(1.0, height / 180.0) * min(1.0, width / 80.0)
        sharp_score = min(1.0, blur / max(1.0, self.min_blur * 4.0))
        aspect_center = 2.5
        aspect_score = max(0.0, 1.0 - abs(aspect_hw - aspect_center) / 3.5)
        edge_score = max(0.0, 1.0 - 0.35 * edge_contacts)
        quality = max(
            0.0,
            min(
                1.0,
                0.35 * max(0.0, min(1.0, float(detector_score)))
                + 0.25 * size_score
                + 0.20 * sharp_score
                + 0.10 * aspect_score
                + 0.10 * edge_score,
            ),
        )

        reason = "accept"
        accepted = True
        if edge_contacts >= self.reject_edge_contacts:
            accepted = False
            reason = "edge"
        elif aspect_hw < self.min_aspect_hw or aspect_hw > self.max_aspect_hw:
            accepted = False
            reason = "aspect"
        elif blur < self.min_blur:
            accepted = False
            reason = "blur"
        elif quality < self.min_quality:
            accepted = False
            reason = "quality"

        step_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self.crop_checked += 1
            self.crop_gate_ms.append(step_ms)
            self.blur_values.append(blur)
            self.quality_values.append(quality)
            if accepted:
                self.crop_accepted += 1
            elif reason == "edge":
                self.reject_edge += 1
            elif reason == "aspect":
                self.reject_aspect += 1
            elif reason == "blur":
                self.reject_blur += 1
            else:
                self.reject_quality += 1

        return CropGateDecision(
            accepted=accepted,
            reason=reason,
            quality=quality,
            blur=blur,
            aspect_hw=aspect_hw,
            edge_contacts=edge_contacts,
            step_ms=step_ms,
        )

    def accept_embedding(
        self,
        *,
        camera_id: str,
        track_id: str,
        embedding: np.ndarray,
        quality: float,
    ) -> tuple[bool, str, float]:
        vector = _normalize(embedding)
        key = (str(camera_id), str(track_id))
        with self._lock:
            self.embedding_checked += 1
            gallery = self._galleries.setdefault(key, _DiverseGallery())
            if not gallery.embeddings:
                gallery.embeddings.append(vector)
                gallery.qualities.append(float(quality))
                self.embedding_accepted += 1
                return True, "first", -1.0

            scores = [float(np.dot(vector, old)) for old in gallery.embeddings]
            nearest_index = int(np.argmax(scores))
            nearest = scores[nearest_index]
            self.nearest_cos_values.append(nearest)

            if nearest >= self.duplicate_cosine:
                old_quality = float(gallery.qualities[nearest_index])
                if float(quality) >= old_quality + self.replace_quality_gain:
                    gallery.embeddings[nearest_index] = vector
                    gallery.qualities[nearest_index] = float(quality)
                    self.duplicate_replacements += 1
                    self.embedding_accepted += 1
                    return True, "replace", nearest
                self.duplicate_drops += 1
                return False, "duplicate", nearest

            if len(gallery.embeddings) >= self.gallery_size:
                # Prefer diversity: replace the stored sample that is most redundant
                # with the rest, but only if the new sample is not lower quality by a
                # large amount. This keeps gallery size bounded and avoids FIFO churn.
                matrix = np.stack(gallery.embeddings, axis=0)
                similarities = matrix @ matrix.T
                np.fill_diagonal(similarities, -1.0)
                redundancy = np.max(similarities, axis=1)
                replace_index = int(np.argmax(redundancy))
                if float(quality) + 0.10 < float(gallery.qualities[replace_index]):
                    self.duplicate_drops += 1
                    return False, "gallery_quality", nearest
                gallery.embeddings[replace_index] = vector
                gallery.qualities[replace_index] = float(quality)
            else:
                gallery.embeddings.append(vector)
                gallery.qualities.append(float(quality))

            self.embedding_accepted += 1
            return True, "diverse", nearest

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "tracks": len(self._galleries),
                "crop_checked": self.crop_checked,
                "crop_accepted": self.crop_accepted,
                "reject_edge": self.reject_edge,
                "reject_blur": self.reject_blur,
                "reject_aspect": self.reject_aspect,
                "reject_quality": self.reject_quality,
                "embedding_checked": self.embedding_checked,
                "embedding_accepted": self.embedding_accepted,
                "duplicate_drops": self.duplicate_drops,
                "duplicate_replacements": self.duplicate_replacements,
                "gate_p50_ms": _pct(self.crop_gate_ms, 0.50),
                "gate_p95_ms": _pct(self.crop_gate_ms, 0.95),
                "blur_p50": _pct(self.blur_values, 0.50),
                "quality_p50": _pct(self.quality_values, 0.50),
                "nearest_cos_p50": _pct(self.nearest_cos_values, 0.50),
                "nearest_cos_p95": _pct(self.nearest_cos_values, 0.95),
            }
