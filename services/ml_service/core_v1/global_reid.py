from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from .reid_embedder import OsnetCpuEmbedder


def _normalize(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("invalid zero ReID embedding")
    return value / norm


def _cosine(a, b) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class TrackletState:
    camera_id: str
    track_id: int
    embeddings: list[np.ndarray] = field(default_factory=list)
    qualities: list[float] = field(default_factory=list)
    last_seen: float = 0.0
    last_sample: float = 0.0
    last_frame_version: int = -1
    bbox: tuple[float, float, float, float] | None = None
    global_id: str | None = None
    assignment_similarity: float | None = None
    assignment_reason: str = "pending"
    outlier_rejects: int = 0


@dataclass
class GlobalIdentity:
    global_id: str
    prototype: np.ndarray
    created_at: float
    last_seen: float
    last_camera: str
    prototype_updates: int = 0
    matches: int = 0


class GlobalReIdCoordinator:
    """Low-rate CPU ReID side-path for cross-camera Global IDs.

    Local tracking remains authoritative inside each camera. This coordinator
    samples only mature visible tracks, builds a multi-shot tracklet descriptor,
    and conservatively links tracklets across cameras. Ambiguous matches create a
    new Global ID instead of contaminating an existing identity prototype.
    """

    def __init__(
        self,
        stores: dict,
        publishers: dict,
        config: dict,
        root: Path,
        *,
        embedder=None,
    ):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.stores = stores
        self.publishers = publishers
        self.poll_interval = max(0.03, float(cfg.get("poll_interval_ms", 120)) / 1000.0)
        self.sample_interval = max(0.15, float(cfg.get("sample_interval_ms", 650)) / 1000.0)
        self.max_batch = max(1, int(cfg.get("max_batch", 6)))
        self.min_samples = max(1, int(cfg.get("min_samples", 3)))
        self.max_samples = max(self.min_samples, int(cfg.get("max_samples", 6)))
        self.min_crop_height = max(20, int(cfg.get("min_crop_height_px", 70)))
        self.min_crop_width = max(10, int(cfg.get("min_crop_width_px", 24)))
        self.min_area_ratio = max(0.0, float(cfg.get("min_area_ratio", 0.012)))
        self.min_track_confidence = max(0.0, float(cfg.get("min_track_confidence", 0.24)))
        self.max_crop_overlap_iou = max(0.0, min(1.0, float(cfg.get("max_crop_overlap_iou", 0.24))))
        self.min_blur_variance = max(0.0, float(cfg.get("min_blur_variance", 20.0)))
        self.tracklet_outlier_similarity = float(cfg.get("tracklet_outlier_similarity", 0.52))

        self.match_similarity = float(cfg.get("match_similarity", 0.74))
        self.same_group_similarity = float(cfg.get("same_group_similarity", 0.69))
        self.same_camera_similarity = float(cfg.get("same_camera_similarity", 0.72))
        self.strong_similarity = float(cfg.get("strong_similarity", 0.84))
        self.second_best_margin = max(0.0, float(cfg.get("second_best_margin", 0.055)))
        self.prototype_update_similarity = float(cfg.get("prototype_update_similarity", 0.84))
        self.prototype_update_alpha = max(0.01, min(0.50, float(cfg.get("prototype_update_alpha", 0.12))))
        self.active_timeout = max(0.3, float(cfg.get("active_timeout_sec", 1.6)))
        self.gallery_ttl = max(10.0, float(cfg.get("gallery_ttl_sec", 7200.0)))
        self.min_cross_group_transition = max(
            0.0, float(cfg.get("min_cross_group_transition_sec", 1.5))
        )
        self.global_prefix = str(cfg.get("global_prefix", "G"))

        self.overlap_groups = [
            frozenset(str(camera) for camera in group)
            for group in (cfg.get("overlap_groups") or [])
            if len(group) >= 2
        ]
        self.pair_thresholds = {
            str(key): float(value)
            for key, value in dict(cfg.get("pair_thresholds") or {}).items()
        }

        self.embedder = embedder or OsnetCpuEmbedder(cfg, root)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._tracks: dict[tuple[str, int], TrackletState] = {}
        self._globals: dict[str, GlobalIdentity] = {}
        self._next_global = 1
        self._started = 0.0
        self._cycles = 0
        self._samples_queued = 0
        self._samples_embedded = 0
        self._quality_rejects = 0
        self._overlap_rejects = 0
        self._outlier_rejects = 0
        self._global_matches = 0
        self._new_globals = 0
        self._ambiguous_rejects = 0
        self._active_conflicts = 0
        self._transition_rejects = 0
        self._prototype_updates = 0
        self._last_error = ""

    def start(self):
        if not self.enabled or self._thread is not None:
            return
        self._started = time.monotonic()
        self._thread = threading.Thread(
            target=self._run,
            name="core-v1-global-reid",
            daemon=False,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=5.0):
        if self._thread is not None:
            self._thread.join(timeout)

    def _same_overlap_group(self, camera_a: str, camera_b: str) -> bool:
        if camera_a == camera_b:
            return True
        return any(camera_a in group and camera_b in group for group in self.overlap_groups)

    @staticmethod
    def _pair_key(camera_a: str, camera_b: str) -> str:
        return "|".join(sorted((str(camera_a), str(camera_b))))

    def _threshold_for(self, camera_a: str, camera_b: str) -> float:
        pair = self._pair_key(camera_a, camera_b)
        if pair in self.pair_thresholds:
            return float(self.pair_thresholds[pair])
        if camera_a == camera_b:
            return self.same_camera_similarity
        if self._same_overlap_group(camera_a, camera_b):
            return self.same_group_similarity
        return self.match_similarity

    def _active_bindings(self, global_id: str, now: float):
        result = []
        for key, state in self._tracks.items():
            if state.global_id != global_id:
                continue
            if now - state.last_seen <= self.active_timeout:
                result.append(key)
        return result

    def _candidate_allowed(
        self,
        identity: GlobalIdentity,
        camera_id: str,
        track_id: int,
        now: float,
    ) -> bool:
        if now - identity.last_seen > self.gallery_ttl:
            return False
        active = self._active_bindings(identity.global_id, now)
        for active_camera, active_track in active:
            if active_camera == camera_id and active_track != track_id:
                self._active_conflicts += 1
                return False
            if active_camera != camera_id and not self._same_overlap_group(active_camera, camera_id):
                self._active_conflicts += 1
                return False

        if (
            identity.last_camera != camera_id
            and not self._same_overlap_group(identity.last_camera, camera_id)
            and now - identity.last_seen < self.min_cross_group_transition
        ):
            self._transition_rejects += 1
            return False
        return True

    def _new_global(self, prototype: np.ndarray, camera_id: str, now: float) -> GlobalIdentity:
        global_id = f"{self.global_prefix}{self._next_global:03d}"
        self._next_global += 1
        identity = GlobalIdentity(
            global_id=global_id,
            prototype=_normalize(prototype),
            created_at=now,
            last_seen=now,
            last_camera=camera_id,
        )
        self._globals[global_id] = identity
        self._new_globals += 1
        return identity

    def resolve_tracklet(
        self,
        camera_id: str,
        track_id: int,
        prototype,
        quality: float = 1.0,
        now: float | None = None,
    ) -> str:
        """Resolve a stable tracklet prototype to a conservative Global ID."""
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

            ranked: list[tuple[float, GlobalIdentity]] = []
            for identity in self._globals.values():
                if not self._candidate_allowed(identity, key[0], key[1], now):
                    continue
                similarity = _cosine(vector, identity.prototype)
                ranked.append((similarity, identity))
            ranked.sort(key=lambda item: item[0], reverse=True)

            chosen = None
            chosen_similarity = None
            if ranked:
                best_similarity, best = ranked[0]
                threshold = self._threshold_for(key[0], best.last_camera)
                second_similarity = ranked[1][0] if len(ranked) > 1 else -1.0
                has_margin = best_similarity - second_similarity >= self.second_best_margin
                if best_similarity >= threshold and (
                    has_margin or best_similarity >= self.strong_similarity
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
                alpha = self.prototype_update_alpha * min(1.0, max(0.25, float(quality)))
                chosen.prototype = _normalize(
                    (1.0 - alpha) * chosen.prototype + alpha * vector
                )
                chosen.prototype_updates += 1
                self._prototype_updates += 1
            return chosen.global_id

    def _tracklet_prototype(self, state: TrackletState) -> np.ndarray:
        weights = np.asarray(state.qualities, dtype=np.float32)
        weights = np.maximum(weights, 0.05)
        matrix = np.stack(state.embeddings, axis=0)
        vector = np.average(matrix, axis=0, weights=weights)
        return _normalize(vector)

    def _accept_embedding(self, state: TrackletState, embedding, quality: float, now: float):
        vector = _normalize(embedding)
        if len(state.embeddings) >= 2:
            prototype = self._tracklet_prototype(state)
            similarity = _cosine(vector, prototype)
            if similarity < self.tracklet_outlier_similarity:
                state.outlier_rejects += 1
                self._outlier_rejects += 1
                return

        state.embeddings.append(vector)
        state.qualities.append(float(quality))
        if len(state.embeddings) > self.max_samples:
            state.embeddings.pop(0)
            state.qualities.pop(0)
        if len(state.embeddings) < self.min_samples:
            return

        prototype = self._tracklet_prototype(state)
        if state.global_id is None:
            self.resolve_tracklet(
                state.camera_id,
                state.track_id,
                prototype,
                quality=float(np.mean(state.qualities)),
                now=now,
            )
        else:
            identity = self._globals.get(state.global_id)
            if identity is not None:
                identity.last_seen = max(identity.last_seen, now)
                identity.last_camera = state.camera_id

    def _crop_for_track(self, image: np.ndarray, track: dict, others: list[dict]):
        h, w = image.shape[:2]
        try:
            x1, y1, x2, y2 = [float(v) for v in track.get("bbox", [])]
            confidence = float(track.get("confidence", 0.0))
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, confidence)):
            return None
        bw = x2 - x1
        bh = y2 - y1
        if (
            confidence < self.min_track_confidence
            or bw < self.min_crop_width
            or bh < self.min_crop_height
            or bw * bh < float(w * h) * self.min_area_ratio
            or bh / max(1.0, bw) < 0.75
        ):
            self._quality_rejects += 1
            return None

        for other in others:
            if other is track:
                continue
            other_box = other.get("bbox") or []
            if len(other_box) == 4 and _iou_xyxy((x1, y1, x2, y2), other_box) > self.max_crop_overlap_iou:
                self._overlap_rejects += 1
                return None

        pad_x = 0.04 * bw
        pad_top = 0.03 * bh
        pad_bottom = 0.02 * bh
        ix1 = max(0, int(math.floor(x1 - pad_x)))
        iy1 = max(0, int(math.floor(y1 - pad_top)))
        ix2 = min(w, int(math.ceil(x2 + pad_x)))
        iy2 = min(h, int(math.ceil(y2 + pad_bottom)))
        if ix2 - ix1 < self.min_crop_width or iy2 - iy1 < self.min_crop_height:
            self._quality_rejects += 1
            return None
        crop = image[iy1:iy2, ix1:ix2].copy()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur < self.min_blur_variance:
            self._quality_rejects += 1
            return None

        height_score = min(1.0, bh / 180.0)
        blur_score = min(1.0, blur / 140.0)
        touches_edge = ix1 <= 1 or iy1 <= 1 or ix2 >= w - 1 or iy2 >= h - 1
        edge_score = 0.55 if touches_edge else 1.0
        quality = (
            0.40 * min(1.0, confidence)
            + 0.25 * height_score
            + 0.20 * blur_score
            + 0.15 * edge_score
        )
        return crop, float(quality)

    def _collect_candidates(self, now: float):
        candidates = []
        for camera_id, store in self.stores.items():
            frame, version = store.get()
            publisher = self.publishers.get(camera_id)
            if frame is None or publisher is None:
                continue
            tracks = publisher.track_snapshot()
            for track in tracks:
                try:
                    track_id = int(track.get("track_id") or 0)
                except (TypeError, ValueError):
                    continue
                if track_id <= 0:
                    continue
                key = (str(camera_id), track_id)
                with self._lock:
                    state = self._tracks.get(key)
                    if state is None:
                        state = TrackletState(key[0], key[1])
                        self._tracks[key] = state
                    state.last_seen = now
                    bbox = track.get("bbox") or []
                    if len(bbox) == 4:
                        state.bbox = tuple(float(v) for v in bbox)
                    if (
                        now - state.last_sample < self.sample_interval
                        or int(version) == state.last_frame_version
                    ):
                        continue

                prepared = self._crop_for_track(frame.image, track, tracks)
                if prepared is None:
                    continue
                crop, quality = prepared
                with self._lock:
                    state.last_sample = now
                    state.last_frame_version = int(version)
                candidates.append((quality, key, crop))

        candidates.sort(key=lambda item: item[0], reverse=True)
        if len(candidates) > self.max_batch:
            candidates = candidates[: self.max_batch]
        self._samples_queued += len(candidates)
        return candidates

    def _run(self):
        while not self._stop.is_set():
            cycle_started = time.monotonic()
            self._cycles += 1
            try:
                candidates = self._collect_candidates(cycle_started)
                if candidates:
                    features = self.embedder.embed_batch([item[2] for item in candidates])
                    now = time.monotonic()
                    for (_, key, _crop), embedding in zip(candidates, features):
                        with self._lock:
                            state = self._tracks.get(key)
                            if state is None:
                                continue
                            quality = next(
                                item[0] for item in candidates if item[1] == key
                            )
                            self._accept_embedding(state, embedding, quality, now)
                            self._samples_embedded += 1
                self._last_error = ""
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - cycle_started
            self._stop.wait(max(0.0, self.poll_interval - elapsed))

    def identity_for_track(self, camera_id: str, track_id: int):
        key = (str(camera_id), int(track_id))
        with self._lock:
            state = self._tracks.get(key)
            if state is None or state.global_id is None:
                return None
            return {
                "global_id": state.global_id,
                "known": False,
                "reid_similarity": state.assignment_similarity,
                "reid_reason": state.assignment_reason,
            }

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            identities = []
            for global_id in sorted(self._globals):
                identity = self._globals[global_id]
                active = self._active_bindings(global_id, now)
                identities.append(
                    {
                        "global_id": global_id,
                        "active": [
                            {"camera_id": camera_id, "track_id": track_id}
                            for camera_id, track_id in active
                        ],
                        "last_camera": identity.last_camera,
                        "age_sec": max(0.0, now - identity.last_seen),
                        "matches": identity.matches,
                        "prototype_updates": identity.prototype_updates,
                    }
                )
            bindings = []
            for (camera_id, track_id), state in sorted(self._tracks.items()):
                if state.global_id is None:
                    continue
                bindings.append(
                    {
                        "camera_id": camera_id,
                        "track_id": track_id,
                        "global_id": state.global_id,
                        "samples": len(state.embeddings),
                        "similarity": state.assignment_similarity,
                        "reason": state.assignment_reason,
                        "active": now - state.last_seen <= self.active_timeout,
                    }
                )
            return {"identities": identities, "bindings": bindings}

    def metrics(self):
        now = time.monotonic()
        with self._lock:
            assigned = sum(state.global_id is not None for state in self._tracks.values())
            active_globals = sum(
                bool(self._active_bindings(global_id, now)) for global_id in self._globals
            )
            payload = {
                "enabled": self.enabled,
                "ready": bool(self.embedder.metrics().get("ready")),
                "mode": "cpu-tracklet-osnet-sidepath",
                "detector_gating": False,
                "gpu_used": False,
                "global_id_policy": "conservative-margin-active-uniqueness",
                "global_count": len(self._globals),
                "active_global_count": active_globals,
                "tracklets": len(self._tracks),
                "assigned_tracklets": assigned,
                "cycles": self._cycles,
                "samples_queued": self._samples_queued,
                "samples_embedded": self._samples_embedded,
                "quality_rejects": self._quality_rejects,
                "overlap_rejects": self._overlap_rejects,
                "outlier_rejects": self._outlier_rejects,
                "global_matches": self._global_matches,
                "new_globals": self._new_globals,
                "ambiguous_rejects": self._ambiguous_rejects,
                "active_conflicts": self._active_conflicts,
                "transition_rejects": self._transition_rejects,
                "prototype_updates": self._prototype_updates,
                "last_error": self._last_error,
                "uptime_sec": max(0.0, now - self._started) if self._started else 0.0,
                "overlap_groups": [sorted(group) for group in self.overlap_groups],
                "embedder": self.embedder.metrics(),
            }
            return payload
