from __future__ import annotations

import time

import numpy as np

from .face_service import FaceGallery, FaceRecognitionService, GalleryMatch, _normalize


class SafeFaceGallery(FaceGallery):
    """Face gallery that never bypasses ambiguity merely because scores are high."""

    def __init__(self, root, config):
        super().__init__(root, config)
        self.strong_second_best_margin = max(
            0.0, float(config.get("strong_second_best_margin", 0.025))
        )

    def match(self, embedding) -> GalleryMatch | None:
        query = _normalize(embedding)
        if query is None:
            return None
        with self._lock:
            candidates = []
            for person in self._people.values():
                similarities = sorted(
                    (float(np.dot(query, prototype)) for prototype in person["embeddings"]),
                    reverse=True,
                )
                if not similarities:
                    continue
                top = similarities[: min(2, len(similarities))]
                score = float(sum(top) / len(top))
                candidates.append((score, person))

            if not candidates:
                return None
            candidates.sort(key=lambda item: item[0], reverse=True)
            best_score, best = candidates[0]
            second = candidates[1][0] if len(candidates) > 1 else -1.0
            margin = best_score - second
            if best_score < self.match_similarity:
                return None

            required_margin = (
                self.strong_second_best_margin
                if best_score >= self.strong_similarity
                else self.second_best_margin
            )
            if len(candidates) > 1 and margin < required_margin:
                return None

            return GalleryMatch(
                best["person_id"],
                best["name"],
                best["department"],
                best_score,
                second,
                margin,
            )


class SafeFaceRecognitionService(FaceRecognitionService):
    """Production guard layer around the CPU face side-path.

    Besides ambiguity-safe matching, enrollment tokens must all come from the
    same camera-local track and their embeddings must form one coherent cluster.
    This makes a transient tracker ownership change fail enrollment instead of
    silently registering two physical people under one name.
    """

    def __init__(self, stores, publishers, config, root, base_identity=None):
        super().__init__(stores, publishers, config, root, base_identity=base_identity)
        self.gallery = SafeFaceGallery(self.root, self.config)
        self.enrollment_consistency_similarity = float(
            self.config.get("enrollment_consistency_similarity", 0.35)
        )
        self.enrollment_max_outliers = max(
            0, int(self.config.get("enrollment_max_outliers", 1))
        )

    def commit_enrollment(self, name: str, department: str, employee_id: str, tokens: list[str]) -> dict:
        unique_tokens = list(dict.fromkeys(str(token) for token in tokens if token))
        if len(unique_tokens) < self.enrollment_target:
            raise ValueError(f"{self.enrollment_target} accepted samples are required")

        now = time.monotonic()
        with self._lock:
            samples = []
            sources = set()
            for token in unique_tokens[: self.enrollment_target]:
                value = self._enrollment_tokens.get(token)
                if value is None or now - value["created_mono"] > self.enrollment_token_ttl:
                    raise ValueError("an enrollment sample expired; capture again")
                samples.append(dict(value))
                sources.add((str(value.get("camera_id") or ""), int(value.get("track_id") or 0)))

        if len(sources) != 1:
            raise ValueError("enrollment samples changed person track; reset and capture one person again")

        embeddings = [_normalize(sample.get("embedding")) for sample in samples]
        if any(embedding is None for embedding in embeddings):
            raise ValueError("invalid enrollment embedding")
        matrix = np.stack(embeddings, axis=0)
        centroid = _normalize(np.mean(matrix, axis=0))
        if centroid is None:
            raise ValueError("invalid enrollment centroid")
        similarities = matrix @ centroid
        outliers = int(np.sum(similarities < self.enrollment_consistency_similarity))
        if outliers > self.enrollment_max_outliers:
            raise ValueError(
                "enrollment samples are not one consistent face; reset and recapture the same person"
            )

        person = self.gallery.enroll(name, department, employee_id, samples)
        with self._lock:
            for token in unique_tokens:
                self._enrollment_tokens.pop(token, None)
        return person
