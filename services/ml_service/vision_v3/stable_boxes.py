from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class _Track:
    key: int
    mean: np.ndarray
    covariance: np.ndarray
    confidence: float
    state_time: float
    last_observation: float
    hits: int


@dataclass(slots=True)
class _Birth:
    box: tuple[float, float, float, float]
    confidence: float
    last_seen: float
    hits: int


def _state(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(2.0, x2 - x1)
    h = max(2.0, y2 - y1)
    return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5, w, h], dtype=np.float64)


def _box_from_state(state):
    cx, cy, w, h = [float(v) for v in state[:4]]
    w = max(2.0, w)
    h = max(2.0, h)
    return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a, b) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _containment(a, b) -> float:
    inter = _intersection(a, b)
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0.0 else 0.0


def _center_distance(a, b) -> float:
    sa = _state(a)
    sb = _state(b)
    scale = max(20.0, sa[2], sa[3], sb[2], sb[3])
    return math.hypot(sa[0] - sb[0], sa[1] - sb[1]) / scale


class StableFullBodyManager:
    """Kalman + Byte-style visual continuity for RF-DETR display boxes.

    This is intentionally display-only. Raw RF-DETR detections remain the future
    source of truth for NvDCF/geometry. The manager accepts low-confidence person
    observations to recover an existing visual track, but low-confidence noise is
    not allowed to create a visible box immediately.
    """

    def __init__(self, width: int, height: int, cfg: dict) -> None:
        self.width = float(width)
        self.height = float(height)
        self.lock = threading.RLock()
        self.tracks: dict[str, dict[int, _Track]] = {}
        self.births: dict[str, list[_Birth]] = {}
        self.next_key = 1

        # Values are based on the previously stable core-v1 visual tracker.
        self.hold_sec = float(cfg.get("hold_sec", 1.20))
        self.memory_sec = float(cfg.get("memory_sec", 3.0))
        self.predict_sec = float(cfg.get("predict_sec", 0.55))
        self.high_conf = float(cfg.get("byte_high_conf", 0.22))
        self.low_conf = float(cfg.get("byte_low_conf", 0.06))
        self.start_conf = float(cfg.get("start_conf", 0.30))
        self.new_track_min_conf = float(cfg.get("new_track_min_conf", 0.16))
        self.birth_hits = max(2, int(cfg.get("birth_hits", 2)))
        self.match_iou = float(cfg.get("match_iou", 0.10))
        self.second_iou = float(cfg.get("second_match_iou", 0.04))
        self.match_center = float(cfg.get("match_center", 0.75))
        self.second_center = float(cfg.get("second_match_center", 0.52))
        self.low_match_max_age = float(cfg.get("low_match_max_age_sec", 0.85))
        self.process_noise = float(cfg.get("process_noise", 0.85))
        self.measurement_noise = float(cfg.get("measurement_noise", 0.90))
        self.velocity_damping = float(cfg.get("velocity_damping", 0.96))
        self.max_prediction_shift_boxes = float(cfg.get("max_prediction_shift_boxes", 0.55))
        self.max_prediction_size_ratio = float(cfg.get("max_prediction_size_ratio", 0.08))
        self.shrink_measure_alpha = float(cfg.get("shrink_measure_alpha", 0.28))

        self.duplicate_iou = float(cfg.get("duplicate_iou", 0.68))
        self.duplicate_containment = float(cfg.get("duplicate_containment", 0.90))
        self.duplicate_center = float(cfg.get("duplicate_center_distance", 0.20))

        self.side_margin = float(cfg.get("side_margin", 0.08))
        self.top_margin = float(cfg.get("top_margin", 0.07))
        self.bottom_margin = float(cfg.get("bottom_margin", 0.10))
        self.sitting_extra_side = float(cfg.get("sitting_extra_side", 0.04))
        self.sitting_extra_bottom = float(cfg.get("sitting_extra_bottom", 0.04))
        self.sitting_aspect_threshold = float(cfg.get("sitting_aspect_threshold", 1.55))

    def _dedupe(self, detections):
        rows = []
        for box, conf in detections:
            conf = float(conf)
            if conf < self.low_conf:
                continue
            b = tuple(float(v) for v in box)
            if b[2] <= b[0] or b[3] <= b[1]:
                continue
            rows.append((b, conf))
        rows.sort(key=lambda row: row[1], reverse=True)
        kept = []
        for box, conf in rows:
            duplicate = False
            for old, _ in kept:
                if _iou(box, old) >= self.duplicate_iou:
                    duplicate = True
                    break
                if _containment(box, old) >= self.duplicate_containment and _center_distance(box, old) <= self.duplicate_center:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((box, conf))
        return kept

    def _init_track(self, box, confidence: float, when: float) -> _Track:
        z = _state(box)
        scale = max(20.0, z[2], z[3])
        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = z
        covariance = np.diag(
            [
                (0.06 * scale) ** 2,
                (0.06 * scale) ** 2,
                (0.08 * scale) ** 2,
                (0.08 * scale) ** 2,
                (0.70 * scale) ** 2,
                (0.70 * scale) ** 2,
                (0.25 * scale) ** 2,
                (0.25 * scale) ** 2,
            ]
        )
        track = _Track(self.next_key, mean, covariance, confidence, when, when, 1)
        self.next_key += 1
        return track

    def _predict_arrays(self, mean: np.ndarray, covariance: np.ndarray, dt: float):
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return mean.copy(), covariance.copy()
        f = np.eye(8, dtype=np.float64)
        f[0, 4] = dt
        f[1, 5] = dt
        f[2, 6] = dt
        f[3, 7] = dt
        out = f @ mean
        out[4:] *= self.velocity_damping ** max(1.0, dt * 10.0)
        scale = max(20.0, out[2], out[3])
        q_pos = (0.012 * scale * self.process_noise * (1.0 + 3.0 * dt)) ** 2
        q_size = (0.010 * scale * self.process_noise * (1.0 + 2.0 * dt)) ** 2
        q_vel = (0.16 * scale * self.process_noise * max(0.05, dt)) ** 2
        q = np.diag([q_pos, q_pos, q_size, q_size, q_vel, q_vel, q_vel * 0.25, q_vel * 0.25])
        cov = f @ covariance @ f.T + q
        return out, cov

    def _predict_track(self, track: _Track, when: float):
        dt = min(self.predict_sec, max(0.0, when - track.state_time))
        mean, covariance = self._predict_arrays(track.mean, track.covariance, dt)
        if dt > 0.0:
            start_w = max(2.0, track.mean[2])
            start_h = max(2.0, track.mean[3])
            max_dx = max(start_w, start_h) * self.max_prediction_shift_boxes
            mean[0] = max(track.mean[0] - max_dx, min(track.mean[0] + max_dx, mean[0]))
            mean[1] = max(track.mean[1] - max_dx, min(track.mean[1] + max_dx, mean[1]))
            min_w = start_w * (1.0 - self.max_prediction_size_ratio)
            max_w = start_w * (1.0 + self.max_prediction_size_ratio)
            min_h = start_h * (1.0 - self.max_prediction_size_ratio)
            max_h = start_h * (1.0 + self.max_prediction_size_ratio)
            mean[2] = max(min_w, min(max_w, mean[2]))
            mean[3] = max(min_h, min(max_h, mean[3]))
        return mean, covariance

    def _correct(self, track: _Track, box, confidence: float, when: float) -> None:
        predicted, covariance = self._predict_track(track, when)
        z = _state(box)
        # Tight/partial detector boxes are allowed to expand a track immediately,
        # but cannot collapse its width/height in one observation.
        if z[2] < predicted[2]:
            z[2] = predicted[2] + (z[2] - predicted[2]) * self.shrink_measure_alpha
        if z[3] < predicted[3]:
            z[3] = predicted[3] + (z[3] - predicted[3]) * self.shrink_measure_alpha

        h = np.zeros((4, 8), dtype=np.float64)
        h[:4, :4] = np.eye(4)
        scale = max(20.0, predicted[2], predicted[3])
        r_pos = (0.035 * scale * self.measurement_noise) ** 2
        r_size = (0.055 * scale * self.measurement_noise) ** 2
        r = np.diag([r_pos, r_pos, r_size, r_size])
        innovation = z - (h @ predicted)
        s = h @ covariance @ h.T + r
        try:
            k = covariance @ h.T @ np.linalg.inv(s)
        except np.linalg.LinAlgError:
            k = covariance @ h.T @ np.linalg.pinv(s)
        mean = predicted + k @ innovation
        cov = (np.eye(8) - k @ h) @ covariance
        mean[2] = max(4.0, mean[2])
        mean[3] = max(8.0, mean[3])
        track.mean = mean
        track.covariance = cov
        track.confidence = float(confidence)
        track.state_time = when
        track.last_observation = when
        track.hits += 1

    def _association_score(self, predicted_box, det_box, second: bool) -> float | None:
        iou = _iou(predicted_box, det_box)
        dist = _center_distance(predicted_box, det_box)
        iou_gate = self.second_iou if second else self.match_iou
        center_gate = self.second_center if second else self.match_center
        if iou < iou_gate and dist > center_gate:
            return None
        return iou * 0.72 + max(0.0, 1.0 - dist) * 0.28

    def _match(self, current, dets, when: float, second: bool, allowed_tracks=None):
        candidates = []
        for key, track in current.items():
            if allowed_tracks is not None and key not in allowed_tracks:
                continue
            if second and when - track.last_observation > self.low_match_max_age:
                continue
            predicted, _ = self._predict_track(track, when)
            pbox = _box_from_state(predicted)
            for di, (box, _conf) in enumerate(dets):
                score = self._association_score(pbox, box, second)
                if score is not None:
                    candidates.append((score, key, di))
        candidates.sort(reverse=True)
        used_tracks = set()
        used_dets = set()
        matches = []
        for _score, key, di in candidates:
            if key in used_tracks or di in used_dets:
                continue
            used_tracks.add(key)
            used_dets.add(di)
            matches.append((key, di))
        return matches, used_tracks, used_dets

    def _update_births(self, camera_id: str, dets, when: float):
        births = [b for b in self.births.setdefault(camera_id, []) if when - b.last_seen <= 1.8]
        promoted = []
        used = set()
        for box, conf in dets:
            if conf >= self.start_conf:
                promoted.append((box, conf))
                continue
            if conf < self.new_track_min_conf:
                continue
            best = None
            for bi, birth in enumerate(births):
                if bi in used:
                    continue
                score = _iou(box, birth.box) * 0.70 + max(0.0, 1.0 - _center_distance(box, birth.box)) * 0.30
                if best is None or score > best[0]:
                    best = (score, bi)
            if best is not None and best[0] >= 0.34:
                birth = births[best[1]]
                birth.box = box
                birth.confidence = max(birth.confidence, conf)
                birth.last_seen = when
                birth.hits += 1
                used.add(best[1])
                if birth.hits >= self.birth_hits:
                    promoted.append((box, max(conf, birth.confidence)))
                    births.pop(best[1])
                    used = {i - 1 if i > best[1] else i for i in used if i != best[1]}
            else:
                births.append(_Birth(box, conf, when, 1))
        self.births[camera_id] = births
        return promoted

    def update(self, camera_id: str, captured_t: float, detections) -> None:
        detections = self._dedupe(detections)
        high = [(b, c) for b, c in detections if c >= self.high_conf]
        low = [(b, c) for b, c in detections if self.low_conf <= c < self.high_conf]
        with self.lock:
            current = self.tracks.setdefault(camera_id, {})
            high_matches, high_tracks, high_dets = self._match(current, high, captured_t, second=False)
            for key, di in high_matches:
                box, conf = high[di]
                self._correct(current[key], box, conf, captured_t)

            remaining_tracks = set(current) - high_tracks
            low_matches, low_tracks, _ = self._match(current, low, captured_t, second=True, allowed_tracks=remaining_tracks)
            for key, di in low_matches:
                box, conf = low[di]
                self._correct(current[key], box, conf, captured_t)

            unmatched_high = [row for i, row in enumerate(high) if i not in high_dets]
            for box, conf in self._update_births(camera_id, unmatched_high, captured_t):
                track = self._init_track(box, conf, captured_t)
                track.hits = self.birth_hits if conf < self.start_conf else 1
                current[track.key] = track

            stale = [key for key, track in current.items() if captured_t - track.last_observation > self.memory_sec]
            for key in stale:
                current.pop(key, None)

    def _guard(self, box):
        x1, y1, x2, y2 = box
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        aspect = h / max(1.0, w)
        side = self.side_margin + (self.sitting_extra_side if aspect < self.sitting_aspect_threshold else 0.0)
        bottom = self.bottom_margin + (self.sitting_extra_bottom if aspect < self.sitting_aspect_threshold else 0.0)
        pad_x = max(5.0, w * side)
        pad_top = max(5.0, h * self.top_margin)
        pad_bottom = max(7.0, h * bottom)
        x1 -= pad_x
        x2 += pad_x
        y1 -= pad_top
        y2 += pad_bottom
        bw = x2 - x1
        bh = y2 - y1
        sx = -x1 if x1 < 0.0 else ((self.width - 1.0) - x2 if x2 > self.width - 1.0 else 0.0)
        sy = -y1 if y1 < 0.0 else ((self.height - 1.0) - y2 if y2 > self.height - 1.0 else 0.0)
        x1 += sx
        x2 = x1 + bw
        y1 += sy
        y2 = y1 + bh
        return (
            max(0.0, min(self.width - 1.0, x1)),
            max(0.0, min(self.height - 1.0, y1)),
            max(0.0, min(self.width - 1.0, x2)),
            max(0.0, min(self.height - 1.0, y2)),
        )

    def render(self, camera_id: str, now: float):
        with self.lock:
            current = self.tracks.get(camera_id, {})
            rows = []
            for track in current.values():
                age = now - track.last_observation
                if age > self.hold_sec:
                    continue
                predicted, _ = self._predict_track(track, now)
                guarded = self._guard(_box_from_state(predicted))
                x1, y1, x2, y2 = guarded
                if x2 > x1 and y2 > y1:
                    rows.append((x1, y1, x2, y2, float(track.confidence)))
            return rows
