from __future__ import annotations

import time

import numpy as np

from .global_identity import GlobalIdentityManager, _cosine as identity_cosine, _norm as identity_norm
from .reid_service import ReIDCoordinator as BaseReIDCoordinator, _FeatureSample, _cosine, _norm


class HardenedGlobalIdentityManager(GlobalIdentityManager):
    """Global identity manager with conservative historical reuse.

    New local tracklets are compared with the existing inactive gallery before a
    new Unknown ID is created. Prototype/gallery updates are gated so one bad
    crop cannot contaminate a long-lived identity.
    """

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        cfg = dict(config or {})
        self.historical_threshold = float(cfg.get("historical_match_threshold", 0.60))
        self.historical_strong = float(cfg.get("historical_strong_threshold", 0.70))
        self.historical_margin = max(0.0, float(cfg.get("historical_margin", 0.035)))
        self.prototype_update_min_similarity = float(cfg.get("prototype_update_min_similarity", 0.66))
        self.gallery_update_min_similarity = float(cfg.get("gallery_update_min_similarity", 0.70))
        self.merge_min_score = float(cfg.get("merge_min_score", 0.58))
        self._historical_attempts = 0
        self._historical_matches = 0
        self._historical_rejects = 0
        self._prototype_freezes = 0
        self._gallery_freezes = 0

    @staticmethod
    def _identity_score(identity, vector) -> float:
        prototype = identity_cosine(vector, identity.prototype)
        gallery = sorted((identity_cosine(vector, item) for item in identity.gallery), reverse=True)
        top = gallery[: min(3, len(gallery))]
        consensus = sum(top) / len(top) if top else prototype
        return float(0.40 * prototype + 0.60 * consensus)

    def _candidate_allowed_locked(self, identity, camera_id: str) -> bool:
        active_key = identity.active_tracks.get(camera_id)
        if active_key:
            return False
        if self.block_concurrent_cross_room and self.camera_rooms:
            room = self.camera_rooms.get(camera_id)
            active_rooms = {
                self.camera_rooms.get(active_camera)
                for active_camera in identity.active_tracks
            }
            active_rooms.discard(None)
            if room and active_rooms and room not in active_rooms:
                return False
        return True

    def _safe_update_locked(self, identity, camera_id, local_key, descriptor, observed_at, similarity):
        vector = identity_norm(descriptor)
        if similarity >= self.prototype_update_min_similarity:
            alpha = self.prototype_alpha
            identity.prototype = identity_norm((1.0 - alpha) * identity.prototype + alpha * vector)
        else:
            self._prototype_freezes += 1
        if similarity >= self.gallery_update_min_similarity:
            identity.gallery.append(vector.copy())
            if len(identity.gallery) > self.gallery_size:
                del identity.gallery[:-self.gallery_size]
        else:
            self._gallery_freezes += 1
        identity.observations += 1
        identity.last_seen = max(float(identity.last_seen), float(observed_at))
        identity.last_camera = str(camera_id)
        identity.active_tracks[str(camera_id)] = str(local_key)
        self._track_activity[str(local_key)] = max(float(self._track_activity.get(str(local_key), 0.0)), float(observed_at))
        self._updates += 1

    def ensure_track(self, camera_id: str, local_track_id: int, descriptor, observed_at: float | None = None):
        camera_id = str(camera_id)
        local_key = self.local_key(camera_id, local_track_id)
        observed_at = float(observed_at if observed_at is not None else time.monotonic())
        vector = identity_norm(descriptor)
        with self._lock:
            self._expire_locked(observed_at)
            gid = self._track_to_global.get(local_key)
            if gid and gid in self._identities:
                identity = self._identities[gid]
                similarity = self._identity_score(identity, vector)
                self._safe_update_locked(identity, camera_id, local_key, vector, observed_at, similarity)
                return gid, "existing"

            self._historical_attempts += 1
            ranked = []
            for candidate_gid, identity in self._identities.items():
                if not self._candidate_allowed_locked(identity, camera_id):
                    continue
                ranked.append((self._identity_score(identity, vector), candidate_gid, identity))
            ranked.sort(key=lambda item: item[0], reverse=True)
            if ranked:
                best_score, best_gid, best_identity = ranked[0]
                second = ranked[1][0] if len(ranked) > 1 else -1.0
                margin = best_score - second
                strong = best_score >= self.historical_strong
                if best_score >= self.historical_threshold and (strong or margin >= self.historical_margin):
                    self._track_to_global[local_key] = best_gid
                    self._track_activity[local_key] = observed_at
                    self._safe_update_locked(best_identity, camera_id, local_key, vector, observed_at, best_score)
                    self._historical_matches += 1
                    return best_gid, "historical"
            self._historical_rejects += 1
            return self._new_identity_locked(camera_id, local_key, vector, observed_at), "new"

    def merge_tracks(self, left_camera, left_track, right_camera, right_track, score, observed_at=None):
        if float(score) < self.merge_min_score:
            return None, "score_below_merge_floor"
        return super().merge_tracks(left_camera, left_track, right_camera, right_track, score, observed_at)

    def metrics(self):
        payload = super().metrics()
        payload.update(
            {
                "historical_attempts": self._historical_attempts,
                "historical_matches": self._historical_matches,
                "historical_rejects": self._historical_rejects,
                "prototype_freezes": self._prototype_freezes,
                "gallery_freezes": self._gallery_freezes,
                "historical_match_threshold": self.historical_threshold,
                "historical_margin": self.historical_margin,
            }
        )
        return payload


class HardenedReIDCoordinator(BaseReIDCoordinator):
    """OSNet tracklet ReID with outlier rejection and conservative pairing."""

    def __init__(self, frame_stores, detections, config: dict | None = None, spatial_mapper=None):
        super().__init__(frame_stores, detections, config, spatial_mapper=spatial_mapper)
        self.identities = HardenedGlobalIdentityManager(self.config.get("identity"))
        tracklet = dict(self.config.get("tracklet") or {})
        pair = dict(self.config.get("pair_matching") or {})
        self.seed_outlier_min_cos = float(tracklet.get("seed_outlier_min_cos", 0.35))
        self.mature_outlier_min_cos = float(tracklet.get("mature_outlier_min_cos", 0.52))
        self.outlier_quality_override = float(tracklet.get("outlier_quality_override", 0.12))
        self.pair_min_samples = max(self.min_descriptor_samples, int(tracklet.get("pair_min_samples", self.min_descriptor_samples)))
        self.pair_min_quality = max(0.0, float(tracklet.get("pair_min_quality", 0.28)))
        self.pair_max_time_gap = max(0.1, float(pair.get("max_time_gap_sec", 3.0)))
        self._outlier_rejects = 0
        self._pair_quality_rejects = 0
        self._pair_time_rejects = 0

    def _update_descriptor(self, track, embedding, quality, observed_at):
        vector = _norm(embedding)
        if track.samples:
            similarities = [_cosine(vector, sample.embedding) for sample in track.samples]
            nearest = max(similarities)
            best_q = max(sample.quality for sample in track.samples)
            if nearest >= self.duplicate_feature_cos and quality < best_q + self.duplicate_quality_gain:
                self._duplicate_features += 1
                return False
            threshold = self.mature_outlier_min_cos if len(track.samples) >= self.min_descriptor_samples else self.seed_outlier_min_cos
            if nearest < threshold and quality < best_q + self.outlier_quality_override:
                self._outlier_rejects += 1
                return False

        track.samples.append(_FeatureSample(vector, float(quality), float(observed_at)))
        ranked = sorted(track.samples, key=lambda sample: sample.quality, reverse=True)[: min(self.topk, len(track.samples))]
        if len(ranked) >= 3:
            center = _norm(sum(sample.embedding for sample in ranked))
            ranked = sorted(
                ranked,
                key=lambda sample: (0.70 * _cosine(sample.embedding, center) + 0.30 * sample.quality),
                reverse=True,
            )[: self.topk]
        base = _norm(sum(sample.embedding for sample in ranked))
        weights = []
        for sample in ranked:
            consensus = max(0.05, (1.0 + _cosine(sample.embedding, base)) * 0.5)
            weights.append(max(0.05, sample.quality) * (consensus**3))
        weights = np.asarray(weights, dtype=np.float32)
        weights = weights / max(float(weights.sum()), 1e-12)
        track.descriptor = _norm(sum(float(weight) * sample.embedding for weight, sample in zip(weights, ranked)))
        track.descriptor_version += 1
        self._descriptor_updates += 1
        return True

    def _mature(self, camera_id):
        now = time.monotonic()
        result = []
        for track in self._tracks.get(camera_id, {}).values():
            if track.descriptor is None or len(track.samples) < self.pair_min_samples:
                continue
            if now - track.last_seen > self.track_timeout:
                continue
            if track.best_quality < self.pair_min_quality:
                self._pair_quality_rejects += 1
                continue
            result.append(track)
        return result

    def _fusion_detail(self, a, b, appearance):
        detail = super()._fusion_detail(a, b, appearance)
        time_gap = abs(float(a.last_seen) - float(b.last_seen))
        detail["track_time_gap_sec"] = float(time_gap)
        if time_gap > self.pair_max_time_gap:
            detail["impossible"] = True
            self._pair_time_rejects += 1
        return detail

    def metrics(self):
        payload = super().metrics()
        payload.update(
            {
                "algorithm": "osnet-tracklet-hardened-v4",
                "outlier_rejects": self._outlier_rejects,
                "pair_quality_rejects": self._pair_quality_rejects,
                "pair_time_rejects": self._pair_time_rejects,
                "pair_min_samples": self.pair_min_samples,
                "pair_min_quality": self.pair_min_quality,
                "pair_max_time_gap_sec": self.pair_max_time_gap,
            }
        )
        payload["identity_hardening"] = self.identities.metrics()
        return payload
