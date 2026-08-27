from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

try:  # Optional: exact global assignment when scipy already exists in the ML env.
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except Exception:  # pragma: no cover - deterministic greedy fallback has no dependency.
    _linear_sum_assignment = None


@dataclass
class Detection:
    bbox: np.ndarray  # xyxy in detector-content pixels
    score: float
    appearance: np.ndarray | None = None


@dataclass
class TrackSnapshot:
    camera_id: str
    track_id: str
    state: str
    confirmed: bool
    predicted: bool
    score: float
    hits: int
    age_sec: float
    since_detection_sec: float
    bbox_xyxy: tuple[float, float, float, float]
    bbox_norm: tuple[float, float, float, float]
    velocity_norm_s: tuple[float, float, float, float]


@dataclass
class TrackerUpdate:
    camera_id: str
    detections: int
    high_detections: int
    low_detections: int
    active: int
    renderable: int
    matched_high: int
    matched_low: int
    created: int
    recovered: int
    newly_lost: int
    removed: int
    snapshots: list[TrackSnapshot]
    step_ms: float


@dataclass
class _Track:
    number: int
    camera_id: str
    state_vec: np.ndarray  # cx,cy,w,h
    velocity: np.ndarray  # vx,vy,vw,vh pixels/sec
    score: float
    appearance: np.ndarray | None
    created_at: float
    state_time: float
    last_detection: float
    last_measurement: np.ndarray
    hits: int = 1
    status: str = "tentative"  # tentative, tracked, lost, removed
    lost_since: float | None = None

    @property
    def track_id(self) -> str:
        return f"{self.camera_id}-T{self.number:05d}"


class LocalPersonTracker:
    """Time-aware, CPU-only, ByteTrack-style per-camera tracker.

    The detector is intentionally sparse (~2 Hz), so frame-count-based SORT/ByteTrack
    assumptions are not ideal. This implementation keeps ByteTrack's useful two-stage
    high/low confidence association but makes motion, lifecycle and prediction depend
    on monotonic seconds instead of detector frame numbers.

    No image model runs here. A tiny color descriptor is extracted from already-copied
    detector frames and is used only as a weak same-camera association hint when people
    are close. It is not a ReID/global-identity feature.
    """

    def __init__(
        self,
        camera_id: str,
        frame_width: int,
        frame_height: int,
        *,
        low_thresh: float = 0.18,
        high_thresh: float = 0.30,
        new_track_thresh: float = 0.30,
        match_thresh: float = 0.22,
        low_match_thresh: float = 0.18,
        confirm_hits: int = 2,
        tentative_ttl_sec: float = 0.9,
        shadow_sec: float = 0.9,
        max_lost_sec: float = 2.5,
        appearance_weight: float = 0.18,
    ) -> None:
        self.camera_id = camera_id
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.frame_diag = math.hypot(self.frame_width, self.frame_height)
        self.low_thresh = float(low_thresh)
        self.high_thresh = max(self.low_thresh, float(high_thresh))
        self.new_track_thresh = max(self.high_thresh, float(new_track_thresh))
        self.match_thresh = float(match_thresh)
        self.low_match_thresh = float(low_match_thresh)
        self.confirm_hits = max(1, int(confirm_hits))
        self.tentative_ttl_sec = max(0.2, float(tentative_ttl_sec))
        self.shadow_sec = max(0.0, float(shadow_sec))
        self.max_lost_sec = max(self.shadow_sec, float(max_lost_sec))
        self.appearance_weight = min(0.35, max(0.0, float(appearance_weight)))
        self._next_id = 1
        self._tracks: list[_Track] = []

    @staticmethod
    def _xyxy_to_state(box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = (float(v) for v in box)
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        return np.array((x1 + 0.5 * w, y1 + 0.5 * h, w, h), dtype=np.float64)

    def _state_to_xyxy(self, state: np.ndarray) -> np.ndarray:
        cx, cy, w, h = (float(v) for v in state)
        w = min(float(self.frame_width), max(4.0, w))
        h = min(float(self.frame_height), max(4.0, h))
        x1 = min(float(self.frame_width - 1), max(0.0, cx - 0.5 * w))
        y1 = min(float(self.frame_height - 1), max(0.0, cy - 0.5 * h))
        x2 = min(float(self.frame_width - 1), max(x1 + 1.0, cx + 0.5 * w))
        y2 = min(float(self.frame_height - 1), max(y1 + 1.0, cy + 0.5 * h))
        return np.array((x1, y1, x2, y2), dtype=np.float64)

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        iw = max(0.0, x2 - x1)
        ih = max(0.0, y2 - y1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        aa = max(1.0, float(a[2] - a[0])) * max(1.0, float(a[3] - a[1]))
        bb = max(1.0, float(b[2] - b[0])) * max(1.0, float(b[3] - b[1]))
        return inter / max(1e-6, aa + bb - inter)

    @staticmethod
    def _appearance_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
        if a is None or b is None or a.size != b.size:
            return None
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-8:
            return None
        return min(1.0, max(0.0, float(np.dot(a, b) / denom)))

    def _predict_to(self, track: _Track, timestamp: float) -> None:
        dt = max(0.0, min(2.0, timestamp - track.state_time))
        if dt <= 0.0:
            return
        track.state_vec = track.state_vec + track.velocity * dt
        track.state_vec[2] = min(float(self.frame_width), max(4.0, track.state_vec[2]))
        track.state_vec[3] = min(float(self.frame_height), max(4.0, track.state_vec[3]))
        track.state_time = timestamp

    def _pair_score(self, track: _Track, det: Detection, timestamp: float, *, low_stage: bool) -> float:
        tbox = self._state_to_xyxy(track.state_vec)
        dbox = det.bbox
        tw = max(1.0, float(tbox[2] - tbox[0]))
        th = max(1.0, float(tbox[3] - tbox[1]))
        dw = max(1.0, float(dbox[2] - dbox[0]))
        dh = max(1.0, float(dbox[3] - dbox[1]))
        area_ratio = (dw * dh) / max(1.0, tw * th)
        if area_ratio < 0.22 or area_ratio > 4.5:
            return -1.0

        tcx = 0.5 * float(tbox[0] + tbox[2])
        tcy = 0.5 * float(tbox[1] + tbox[3])
        dcx = 0.5 * float(dbox[0] + dbox[2])
        dcy = 0.5 * float(dbox[1] + dbox[3])
        dist = math.hypot(dcx - tcx, dcy - tcy)
        box_diag = math.hypot(tw, th)
        since_det = max(0.0, timestamp - track.last_detection)
        uncertainty = min(180.0, 95.0 * since_det)
        max_center = min(self.frame_diag * 0.52, max(55.0, 1.55 * box_diag + uncertainty))
        if dist > max_center:
            return -1.0

        iou = self._iou(tbox, dbox)
        proximity = max(0.0, 1.0 - dist / max(1.0, max_center))
        # Size continuity prevents a nearby large/small person swap in crowded views.
        log_area = abs(math.log(max(1e-6, area_ratio)))
        size_score = math.exp(-0.75 * log_area)
        geom = 0.58 * iou + 0.30 * proximity + 0.12 * size_score

        app = self._appearance_similarity(track.appearance, det.appearance)
        if app is not None and not low_stage:
            score = (1.0 - self.appearance_weight) * geom + self.appearance_weight * app
        else:
            score = geom

        # Require at least some useful geometric support; appearance may break ties,
        # but it must never teleport a track across the frame.
        if iou < 0.005 and proximity < 0.32:
            return -1.0
        return float(score)

    def _assign(
        self,
        tracks: list[_Track],
        detections: list[Detection],
        timestamp: float,
        threshold: float,
        *,
        low_stage: bool,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        scores = np.full((len(tracks), len(detections)), -1.0, dtype=np.float64)
        for ti, track in enumerate(tracks):
            for di, det in enumerate(detections):
                scores[ti, di] = self._pair_score(track, det, timestamp, low_stage=low_stage)

        matches: list[tuple[int, int]] = []
        if _linear_sum_assignment is not None:
            cost = np.where(scores >= threshold, 1.0 - scores, 1e6)
            rows, cols = _linear_sum_assignment(cost)
            for ti, di in zip(rows.tolist(), cols.tolist()):
                if scores[ti, di] >= threshold:
                    matches.append((ti, di))
        else:
            candidates: list[tuple[float, int, int]] = []
            for ti in range(scores.shape[0]):
                for di in range(scores.shape[1]):
                    if scores[ti, di] >= threshold:
                        candidates.append((float(scores[ti, di]), ti, di))
            candidates.sort(reverse=True)
            used_t: set[int] = set()
            used_d: set[int] = set()
            for _score, ti, di in candidates:
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti)
                used_d.add(di)
                matches.append((ti, di))

        matched_t = {ti for ti, _ in matches}
        matched_d = {di for _, di in matches}
        return (
            matches,
            [i for i in range(len(tracks)) if i not in matched_t],
            [i for i in range(len(detections)) if i not in matched_d],
        )

    def _update_track(self, track: _Track, det: Detection, timestamp: float) -> bool:
        recovered = track.status == "lost"
        measurement = self._xyxy_to_state(det.bbox)
        prior = track.state_vec.copy()
        dt_measure = max(1e-3, timestamp - track.last_detection)

        # Sparse detector: favor the measurement for position, but keep enough motion
        # memory to generate smooth 20 FPS UI-side prediction between 500 ms updates.
        alpha = 0.78
        track.state_vec = prior + alpha * (measurement - prior)
        inst_velocity = (measurement - track.last_measurement) / dt_measure
        velocity_gain = min(0.42, max(0.18, 0.22 + 0.12 * dt_measure))
        track.velocity = (1.0 - velocity_gain) * track.velocity + velocity_gain * inst_velocity

        # Bound impossible extrapolation after one bad detection.
        max_vx = 1.35 * self.frame_width
        max_vy = 1.35 * self.frame_height
        track.velocity[0] = float(np.clip(track.velocity[0], -max_vx, max_vx))
        track.velocity[1] = float(np.clip(track.velocity[1], -max_vy, max_vy))
        track.velocity[2] = float(np.clip(track.velocity[2], -0.80 * self.frame_width, 0.80 * self.frame_width))
        track.velocity[3] = float(np.clip(track.velocity[3], -0.80 * self.frame_height, 0.80 * self.frame_height))

        if det.appearance is not None:
            if track.appearance is None:
                track.appearance = det.appearance.copy()
            else:
                track.appearance = 0.82 * track.appearance + 0.18 * det.appearance
                norm = float(np.linalg.norm(track.appearance))
                if norm > 1e-8:
                    track.appearance /= norm

        track.last_measurement = measurement
        track.last_detection = timestamp
        track.state_time = timestamp
        track.score = float(det.score)
        track.hits += 1
        track.lost_since = None
        if track.hits >= self.confirm_hits:
            track.status = "tracked"
        else:
            track.status = "tentative"
        return recovered

    def _new_track(self, det: Detection, timestamp: float) -> _Track:
        measurement = self._xyxy_to_state(det.bbox)
        track = _Track(
            number=self._next_id,
            camera_id=self.camera_id,
            state_vec=measurement.copy(),
            velocity=np.zeros(4, dtype=np.float64),
            score=float(det.score),
            appearance=None if det.appearance is None else det.appearance.copy(),
            created_at=timestamp,
            state_time=timestamp,
            last_detection=timestamp,
            last_measurement=measurement.copy(),
        )
        self._next_id += 1
        if self.confirm_hits <= 1:
            track.status = "tracked"
        self._tracks.append(track)
        return track

    def _snapshot(self, track: _Track, timestamp: float, *, predicted: bool) -> TrackSnapshot:
        box = self._state_to_xyxy(track.state_vec)
        x1, y1, x2, y2 = (float(v) for v in box)
        return TrackSnapshot(
            camera_id=self.camera_id,
            track_id=track.track_id,
            state=track.status,
            confirmed=track.hits >= self.confirm_hits,
            predicted=predicted,
            score=float(track.score),
            hits=track.hits,
            age_sec=max(0.0, timestamp - track.created_at),
            since_detection_sec=max(0.0, timestamp - track.last_detection),
            bbox_xyxy=(x1, y1, x2, y2),
            bbox_norm=(
                x1 / self.frame_width,
                y1 / self.frame_height,
                x2 / self.frame_width,
                y2 / self.frame_height,
            ),
            velocity_norm_s=(
                float(track.velocity[0]) / self.frame_width,
                float(track.velocity[1]) / self.frame_height,
                float(track.velocity[2]) / self.frame_width,
                float(track.velocity[3]) / self.frame_height,
            ),
        )

    def update(self, detections: Iterable[Detection], timestamp: float) -> TrackerUpdate:
        started = time.perf_counter()
        timestamp = float(timestamp)
        rows = [d for d in detections if d.score >= self.low_thresh]
        high = [d for d in rows if d.score >= self.high_thresh]
        low = [d for d in rows if self.low_thresh <= d.score < self.high_thresh]

        live_tracks = [t for t in self._tracks if t.status != "removed"]
        for track in live_tracks:
            self._predict_to(track, timestamp)

        # First stage: high-confidence detections can match active or recently lost tracks.
        stage1_tracks = [
            t
            for t in live_tracks
            if t.status in ("tracked", "tentative", "lost")
            and timestamp - t.last_detection <= self.max_lost_sec
        ]
        matches1, unmatched_t1, unmatched_high = self._assign(
            stage1_tracks,
            high,
            timestamp,
            self.match_thresh,
            low_stage=False,
        )
        matched_tracks: set[int] = set()
        recovered = 0
        for ti, di in matches1:
            track = stage1_tracks[ti]
            if self._update_track(track, high[di], timestamp):
                recovered += 1
            matched_tracks.add(id(track))

        # Second stage: ByteTrack idea—low-confidence boxes may preserve an already
        # confirmed track, but they never create a new identity.
        stage2_tracks = [
            stage1_tracks[i]
            for i in unmatched_t1
            if stage1_tracks[i].status == "tracked"
        ]
        matches2, _unmatched_t2, _unmatched_low = self._assign(
            stage2_tracks,
            low,
            timestamp,
            self.low_match_thresh,
            low_stage=True,
        )
        for ti, di in matches2:
            track = stage2_tracks[ti]
            self._update_track(track, low[di], timestamp)
            matched_tracks.add(id(track))

        created = 0
        for di in unmatched_high:
            det = high[di]
            if det.score < self.new_track_thresh:
                continue
            self._new_track(det, timestamp)
            created += 1

        newly_lost = 0
        removed = 0
        for track in self._tracks:
            if track.status == "removed" or id(track) in matched_tracks:
                continue
            since = timestamp - track.last_detection
            if track.status == "tentative":
                if since > self.tentative_ttl_sec:
                    track.status = "removed"
                    removed += 1
                continue
            if track.status == "tracked" and since > 1e-6:
                track.status = "lost"
                track.lost_since = timestamp
                newly_lost += 1
            if track.status == "lost" and since > self.max_lost_sec:
                track.status = "removed"
                removed += 1

        snapshots: list[TrackSnapshot] = []
        for track in self._tracks:
            if track.status == "removed":
                continue
            since = timestamp - track.last_detection
            if track.status == "tracked":
                snapshots.append(self._snapshot(track, timestamp, predicted=False))
            elif track.status == "tentative" and since <= 1e-6:
                snapshots.append(self._snapshot(track, timestamp, predicted=False))
            elif track.status == "lost" and track.hits >= self.confirm_hits and since <= self.shadow_sec:
                snapshots.append(self._snapshot(track, timestamp, predicted=True))

        active = sum(1 for t in self._tracks if t.status != "removed")
        step_ms = (time.perf_counter() - started) * 1000.0
        return TrackerUpdate(
            camera_id=self.camera_id,
            detections=len(rows),
            high_detections=len(high),
            low_detections=len(low),
            active=active,
            renderable=len(snapshots),
            matched_high=len(matches1),
            matched_low=len(matches2),
            created=created,
            recovered=recovered,
            newly_lost=newly_lost,
            removed=removed,
            snapshots=snapshots,
            step_ms=step_ms,
        )


def appearance_descriptor(frame_bgr: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray | None:
    """Cheap 24-D body-color descriptor from an already available detector frame."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
        return None
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    # Ignore box edges/head/feet where background and pose changes dominate.
    xa = int(max(0, min(w - 1, round(x1 + 0.12 * bw))))
    xb = int(max(xa + 1, min(w, round(x2 - 0.12 * bw))))
    ya = int(max(0, min(h - 1, round(y1 + 0.18 * bh))))
    yb = int(max(ya + 1, min(h, round(y2 - 0.12 * bh))))
    crop = frame_bgr[ya:yb, xa:xb, :3]
    if crop.size < 3 * 64:
        return None
    # Downsample by striding; no OpenCV and no extra GPU work.
    sy = max(1, crop.shape[0] // 32)
    sx = max(1, crop.shape[1] // 24)
    sample = crop[::sy, ::sx]
    parts: list[np.ndarray] = []
    for channel in range(3):
        hist, _ = np.histogram(sample[..., channel], bins=8, range=(0, 256))
        parts.append(hist.astype(np.float32))
    descriptor = np.concatenate(parts)
    norm = float(np.linalg.norm(descriptor))
    if norm <= 1e-8:
        return None
    descriptor /= norm
    return descriptor


class MultiCameraLocalTracker:
    def __init__(self, camera_ids: Iterable[str], width: int, height: int, **kwargs) -> None:
        self.trackers = {
            cid: LocalPersonTracker(cid, width, height, **kwargs) for cid in camera_ids
        }

    def update(
        self,
        camera_id: str,
        boxes: Iterable[Iterable[float]],
        frame_bgr: np.ndarray,
        captured_ns: int,
    ) -> TrackerUpdate:
        detections: list[Detection] = []
        for row in boxes:
            values = list(row)
            if len(values) != 5:
                continue
            x1, y1, x2, y2, score = (float(v) for v in values)
            bbox = np.array((x1, y1, x2, y2), dtype=np.float64)
            detections.append(
                Detection(
                    bbox=bbox,
                    score=score,
                    appearance=appearance_descriptor(frame_bgr, bbox),
                )
            )
        timestamp = captured_ns / 1_000_000_000.0
        return self.trackers[camera_id].update(detections, timestamp)
