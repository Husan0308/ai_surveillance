from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(slots=True)
class VisualBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass(slots=True)
class _Track:
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    confidence: float
    state_time: float
    last_observation: float
    last_seen_wall: float
    hits: int


@dataclass(slots=True)
class _BirthCandidate:
    box: VisualBox
    last_seen: float
    hits: int


def _width(box: VisualBox) -> float:
    return max(0.0, box.x2 - box.x1)


def _height(box: VisualBox) -> float:
    return max(0.0, box.y2 - box.y1)


def _area(box: VisualBox) -> float:
    return _width(box) * _height(box)


def _center_size(box: VisualBox):
    w = max(2.0, _width(box))
    h = max(2.0, _height(box))
    return (box.x1 + box.x2) * 0.5, (box.y1 + box.y2) * 0.5, w, h


def _from_center_size(cx: float, cy: float, w: float, h: float, confidence: float) -> VisualBox:
    w = max(2.0, float(w))
    h = max(2.0, float(h))
    return VisualBox(cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5, confidence)


def _intersection(a: VisualBox, b: VisualBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a: VisualBox, b: VisualBox) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _containment(a: VisualBox, b: VisualBox) -> float:
    inter = _intersection(a, b)
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0 else 0.0


def _center_distance(a: VisualBox, b: VisualBox) -> float:
    acx, acy, aw, ah = _center_size(a)
    bcx, bcy, bw, bh = _center_size(b)
    scale = max(20.0, max(aw, ah, bw, bh))
    return math.hypot(acx - bcx, acy - bcy) / scale


def _horizontal_overlap_ratio(a: VisualBox, b: VisualBox) -> float:
    overlap = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    return overlap / max(1.0, min(_width(a), _width(b)))


def _vertical_overlap_ratio(a: VisualBox, b: VisualBox) -> float:
    overlap = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return overlap / max(1.0, min(_height(a), _height(b)))


def _vertical_gap_ratio(a: VisualBox, b: VisualBox) -> float:
    if a.y2 < b.y1:
        gap = b.y1 - a.y2
    elif b.y2 < a.y1:
        gap = a.y1 - b.y2
    else:
        gap = 0.0
    return gap / max(1.0, max(_height(a), _height(b)))


def _x_center_ratio(a: VisualBox, b: VisualBox) -> float:
    acx = (a.x1 + a.x2) * 0.5
    bcx = (b.x1 + b.x2) * 0.5
    return abs(acx - bcx) / max(1.0, max(_width(a), _width(b)))


def _area_ratio(a: VisualBox, b: VisualBox) -> float:
    aa = _area(a)
    bb = _area(b)
    return min(aa, bb) / max(1.0, max(aa, bb))


class VisualTracker:
    """Paddle/ByteTrack-inspired visual-only per-camera tracker.

    The important behavior is the same class of design used by production MOT
    trackers: a Kalman motion state predicts current position, high-confidence
    detections associate first, low-confidence detections get a second chance to
    recover an existing track, and only stronger repeated detections may create a
    new track.

    This tracker is intentionally *visual only*. Its track_id or predicted boxes
    must never be used as ReID/global-identity evidence.
    """

    def __init__(
        self,
        *,
        hold_ms=900,
        memory_ms=6000,
        prediction_ms=550,
        match_iou=0.10,
        reacquire_distance=1.05,
        duplicate_iou=0.50,
        duplicate_containment=0.82,
        duplicate_center_distance=0.40,
        fragment_duplicate=False,
        fragment_horizontal_overlap=0.70,
        fragment_x_center=0.30,
        fragment_max_area_ratio=0.70,
        fragment_min_vertical_overlap=0.12,
        fragment_max_vertical_gap=0.10,
        low_conf_confirm=0.08,
        start_conf=0.34,
        new_track_min_conf=0.24,
        strong_confirm_hits=2,
        weak_confirm_hits=3,
        byte_high_conf=0.24,
        byte_low_conf=0.08,
        byte_second_match_iou=0.04,
        byte_match_center=0.82,
        byte_second_match_center=0.55,
        low_match_max_age_ms=700,
        process_noise=1.0,
        measurement_noise=1.0,
        velocity_damping=0.985,
        max_prediction_shift_boxes=0.70,
        max_prediction_size_ratio=0.12,
        new_track_zones=None,
        exclusion_zones=None,
        exclusion_max_box_height=0.24,
        exclusion_overlap_threshold=0.35,
        # Backward-compatible legacy knobs. They are deliberately ignored now;
        # Kalman predict/correct replaces the old hand-written EMA smoother.
        smoothing=None,
        center_smoothing=None,
        size_smoothing=None,
        velocity_smoothing=None,
        adaptive_center_smoothing=None,
        adaptive_error_boxes=None,
        snap_distance_boxes=None,
        reversal_damping=None,
    ):
        self.hold_sec = max(0.05, float(hold_ms) / 1000.0)
        self.memory_sec = max(self.hold_sec, float(memory_ms) / 1000.0)
        self.prediction_sec = max(0.0, float(prediction_ms) / 1000.0)
        self.match_iou = max(0.0, min(1.0, float(match_iou)))
        self.reacquire_distance = max(0.1, float(reacquire_distance))
        self.duplicate_iou = float(duplicate_iou)
        self.duplicate_containment = float(duplicate_containment)
        self.duplicate_center_distance = max(0.0, float(duplicate_center_distance))
        self.fragment_duplicate = bool(fragment_duplicate)
        self.fragment_horizontal_overlap = float(fragment_horizontal_overlap)
        self.fragment_x_center = float(fragment_x_center)
        self.fragment_max_area_ratio = float(fragment_max_area_ratio)
        self.fragment_min_vertical_overlap = float(fragment_min_vertical_overlap)
        self.fragment_max_vertical_gap = float(fragment_max_vertical_gap)

        self.low_conf_confirm = max(0.0, float(low_conf_confirm))
        self.start_conf = max(0.0, float(start_conf))
        self.new_track_min_conf = max(0.0, float(new_track_min_conf))
        self.strong_confirm_hits = max(1, int(strong_confirm_hits))
        self.weak_confirm_hits = max(self.strong_confirm_hits, int(weak_confirm_hits))
        self.byte_high_conf = max(self.low_conf_confirm, float(byte_high_conf))
        self.byte_low_conf = max(0.0, min(self.byte_high_conf, float(byte_low_conf)))
        self.byte_second_match_iou = max(0.0, min(1.0, float(byte_second_match_iou)))
        self.byte_match_center = max(0.1, float(byte_match_center))
        self.byte_second_match_center = max(0.1, float(byte_second_match_center))
        self.low_match_max_age_sec = max(0.05, float(low_match_max_age_ms) / 1000.0)

        self.process_noise = max(0.05, float(process_noise))
        self.measurement_noise = max(0.05, float(measurement_noise))
        self.velocity_damping = max(0.80, min(1.0, float(velocity_damping)))
        self.max_prediction_shift_boxes = max(0.10, float(max_prediction_shift_boxes))
        self.max_prediction_size_ratio = max(0.0, min(0.50, float(max_prediction_size_ratio)))

        self.new_track_zones = [
            tuple(float(v) for v in zone[:5])
            for zone in (new_track_zones or [])
            if len(zone) >= 5
        ]
        self.exclusion_zones = [
            tuple(float(v) for v in zone[:4])
            for zone in (exclusion_zones or [])
            if len(zone) >= 4
        ]
        self.exclusion_max_box_height = max(0.0, float(exclusion_max_box_height))
        self.exclusion_overlap_threshold = max(0.0, min(1.0, float(exclusion_overlap_threshold)))

        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._last_result_frame_id = -1
        self._birth_candidates: list[_BirthCandidate] = []

        self._updates = 0
        self._high_matches = 0
        self._low_matches = 0
        self._births = 0
        self._predicted_visible = 0
        self._pruned = 0

    def _excluded(self, box: VisualBox, source_width, source_height) -> bool:
        if not self.exclusion_zones or not source_width or not source_height:
            return False
        width = max(1.0, float(source_width))
        height = max(1.0, float(source_height))
        cx = ((box.x1 + box.x2) * 0.5) / width
        cy = ((box.y1 + box.y2) * 0.5) / height
        box_h = max(0.0, box.y2 - box.y1) / height
        if box_h > self.exclusion_max_box_height:
            return False
        normalized = VisualBox(
            box.x1 / width, box.y1 / height, box.x2 / width, box.y2 / height, box.confidence
        )
        for x1, y1, x2, y2 in self.exclusion_zones:
            zone = VisualBox(x1, y1, x2, y2, 1.0)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
            if _intersection(normalized, zone) / max(1e-9, _area(normalized)) >= self.exclusion_overlap_threshold:
                return True
        return False

    def _birth_threshold(self, box: VisualBox, source_width, source_height) -> float:
        threshold = self.new_track_min_conf
        if not self.new_track_zones or not source_width or not source_height:
            return threshold
        width = max(1.0, float(source_width))
        height = max(1.0, float(source_height))
        cx = ((box.x1 + box.x2) * 0.5) / width
        cy = ((box.y1 + box.y2) * 0.5) / height
        for x1, y1, x2, y2, zone_conf in self.new_track_zones:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                threshold = max(threshold, float(zone_conf))
        return threshold

    def _fragment_duplicate_match(self, box: VisualBox, other: VisualBox) -> bool:
        if not self.fragment_duplicate:
            return False
        if _area_ratio(box, other) > self.fragment_max_area_ratio:
            return False
        if _horizontal_overlap_ratio(box, other) < self.fragment_horizontal_overlap:
            return False
        if _x_center_ratio(box, other) > self.fragment_x_center:
            return False
        return (
            _vertical_overlap_ratio(box, other) >= self.fragment_min_vertical_overlap
            or _vertical_gap_ratio(box, other) <= self.fragment_max_vertical_gap
        )

    def _is_duplicate(self, box: VisualBox, other: VisualBox) -> bool:
        iou = _iou(box, other)
        containment = _containment(box, other)
        center_distance = _center_distance(box, other)
        return (
            iou >= self.duplicate_iou
            or (
                containment >= self.duplicate_containment
                and center_distance <= self.duplicate_center_distance
            )
            or self._fragment_duplicate_match(box, other)
        )

    def _dedupe(self, boxes, source_width=None, source_height=None) -> list[VisualBox]:
        candidates = [
            VisualBox(float(b.x1), float(b.y1), float(b.x2), float(b.y2), float(b.confidence))
            for b in boxes
        ]
        candidates = [
            b for b in candidates if not self._excluded(b, source_width, source_height)
        ]
        candidates.sort(key=lambda b: b.confidence, reverse=True)
        kept: list[VisualBox] = []
        for box in candidates:
            if any(self._is_duplicate(box, other) for other in kept):
                continue
            kept.append(box)
        return kept

    @staticmethod
    def _measurement(box: VisualBox) -> np.ndarray:
        cx, cy, w, h = _center_size(box)
        return np.asarray([cx, cy, w, h], dtype=np.float64)

    def _init_track(self, box: VisualBox, observation: float, now: float, hits: int) -> _Track:
        z = self._measurement(box)
        scale = max(20.0, z[2], z[3])
        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = z
        pos = (0.06 * scale) ** 2
        size = (0.08 * scale) ** 2
        velocity = (0.70 * scale) ** 2
        covariance = np.diag([pos, pos, size, size, velocity, velocity, velocity, velocity])
        track = _Track(
            track_id=self._next_id,
            mean=mean,
            covariance=covariance,
            confidence=float(box.confidence),
            state_time=float(observation),
            last_observation=float(observation),
            last_seen_wall=float(now),
            hits=max(1, int(hits)),
        )
        self._next_id += 1
        return track

    def _predict_state(self, mean: np.ndarray, covariance: np.ndarray, dt: float):
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return mean.copy(), covariance.copy()

        transition = np.eye(8, dtype=np.float64)
        transition[0, 4] = dt
        transition[1, 5] = dt
        transition[2, 6] = dt
        transition[3, 7] = dt

        predicted_mean = transition @ mean
        scale = max(20.0, predicted_mean[2], predicted_mean[3])
        q_pos = (0.012 * scale * self.process_noise * (1.0 + 3.0 * dt)) ** 2
        q_size = (0.010 * scale * self.process_noise * (1.0 + 2.0 * dt)) ** 2
        q_vel = (0.16 * scale * self.process_noise * max(0.05, dt)) ** 2
        process_cov = np.diag([q_pos, q_pos, q_size, q_size, q_vel, q_vel, q_vel, q_vel])
        predicted_covariance = transition @ covariance @ transition.T + process_cov
        predicted_mean[2] = max(2.0, predicted_mean[2])
        predicted_mean[3] = max(2.0, predicted_mean[3])
        return predicted_mean, predicted_covariance

    def _predict_track_to(self, track: _Track, target_time: float) -> None:
        dt = max(0.0, float(target_time) - track.state_time)
        if dt <= 0.0:
            return
        track.mean, track.covariance = self._predict_state(track.mean, track.covariance, dt)
        track.mean[4:] *= self.velocity_damping
        track.state_time = float(target_time)

    def _correct_track(self, track: _Track, box: VisualBox, observation: float, now: float) -> None:
        z = self._measurement(box)
        scale = max(20.0, z[2], z[3])
        confidence = max(0.01, min(1.0, float(box.confidence)))
        confidence_factor = 1.0 + (1.0 - confidence) * 2.2
        pos_std = 0.022 * scale * self.measurement_noise * confidence_factor
        size_std = 0.035 * scale * self.measurement_noise * confidence_factor
        measurement_cov = np.diag([
            pos_std * pos_std,
            pos_std * pos_std,
            size_std * size_std,
            size_std * size_std,
        ])

        innovation = z - track.mean[:4]
        innovation_cov = track.covariance[:4, :4] + measurement_cov
        try:
            kalman_gain = np.linalg.solve(
                innovation_cov.T, track.covariance[:, :4].T
            ).T
        except np.linalg.LinAlgError:
            kalman_gain = track.covariance[:, :4] @ np.linalg.pinv(innovation_cov)

        track.mean = track.mean + kalman_gain @ innovation
        track.covariance = track.covariance - kalman_gain @ track.covariance[:4, :]
        track.covariance = (track.covariance + track.covariance.T) * 0.5
        track.mean[2] = max(2.0, track.mean[2])
        track.mean[3] = max(2.0, track.mean[3])
        track.confidence = max(float(box.confidence), track.confidence * 0.80)
        track.state_time = float(observation)
        track.last_observation = float(observation)
        track.last_seen_wall = float(now)
        track.hits += 1

    @staticmethod
    def _box_from_track(track: _Track, mean: np.ndarray | None = None) -> VisualBox:
        state = track.mean if mean is None else mean
        return _from_center_size(
            float(state[0]), float(state[1]), float(state[2]), float(state[3]), track.confidence
        )

    def _associate(
        self,
        track_ids: list[int],
        detections: list[VisualBox],
        *,
        min_iou: float,
        max_center: float,
        observation: float,
        low_stage: bool,
    ):
        pairs = []
        for tid in track_ids:
            track = self._tracks[tid]
            ref = self._box_from_track(track)
            actual_max_center = max_center
            if not low_stage and observation - track.last_observation > self.hold_sec:
                actual_max_center = max(actual_max_center, self.reacquire_distance)
            for di, det in enumerate(detections):
                iou = _iou(ref, det)
                distance = _center_distance(ref, det)
                if iou < min_iou and distance > actual_max_center:
                    continue
                # IoU dominates. Center distance resolves fast-motion cases where
                # the predicted and observed boxes barely overlap.
                cost = (1.0 - iou) + 0.20 * min(2.0, distance) + 0.04 * (1.0 - det.confidence)
                pairs.append((cost, tid, di))

        pairs.sort(key=lambda item: item[0])
        used_tracks = set()
        used_dets = set()
        matches = []
        for _cost, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(di)
            matches.append((tid, di))
        return matches, used_tracks, used_dets

    def _confirm_birth(self, det: VisualBox, now: float, required_hits: int) -> bool:
        fresh = [c for c in self._birth_candidates if now - c.last_seen <= 1.25]
        self._birth_candidates = fresh

        best = None
        best_score = float('inf')
        for candidate in fresh:
            iou = _iou(candidate.box, det)
            distance = _center_distance(candidate.box, det)
            if iou < 0.12 and distance > 0.55:
                continue
            score = (1.0 - iou) + 0.25 * distance
            if score < best_score:
                best = candidate
                best_score = score

        if best is None:
            self._birth_candidates.append(_BirthCandidate(det, now, 1))
            return required_hits <= 1

        best.box = det
        best.last_seen = now
        best.hits += 1
        if best.hits >= required_hits:
            self._birth_candidates.remove(best)
            return True
        return False

    def update(self, result, now: float, source_width=None, source_height=None) -> None:
        if result is None or int(result.frame_id) == self._last_result_frame_id:
            return
        self._last_result_frame_id = int(result.frame_id)
        self._updates += 1

        observation = float(getattr(result, 'frame_captured_monotonic', now) or now)
        if not math.isfinite(observation) or observation <= 0:
            observation = now

        detections = self._dedupe(result.boxes, source_width, source_height)

        for tid in list(self._tracks):
            if now - self._tracks[tid].last_seen_wall > self.memory_sec:
                del self._tracks[tid]
                self._pruned += 1

        for track in self._tracks.values():
            self._predict_track_to(track, observation)

        high = [d for d in detections if d.confidence >= self.byte_high_conf]
        low = [d for d in detections if self.byte_low_conf <= d.confidence < self.byte_high_conf]
        track_ids = list(self._tracks)

        matches1, used_tracks1, used_high = self._associate(
            track_ids,
            high,
            min_iou=self.match_iou,
            max_center=self.byte_match_center,
            observation=observation,
            low_stage=False,
        )
        for tid, di in matches1:
            self._correct_track(self._tracks[tid], high[di], observation, now)
            self._high_matches += 1

        remaining_tracks = [tid for tid in track_ids if tid not in used_tracks1]
        recent_tracks = [
            tid
            for tid in remaining_tracks
            if observation - self._tracks[tid].last_observation <= self.low_match_max_age_sec
        ]
        matches2, used_tracks2, used_low = self._associate(
            recent_tracks,
            low,
            min_iou=self.byte_second_match_iou,
            max_center=self.byte_second_match_center,
            observation=observation,
            low_stage=True,
        )
        for tid, di in matches2:
            self._correct_track(self._tracks[tid], low[di], observation, now)
            self._low_matches += 1

        unmatched_high = [d for i, d in enumerate(high) if i not in used_high]
        # Medium detections below byte_high_conf may prove a birth only when they
        # meet the configured birth threshold. Very weak boxes never create one.
        unmatched_medium = [d for i, d in enumerate(low) if i not in used_low]
        for det in unmatched_high + unmatched_medium:
            birth_threshold = self._birth_threshold(det, source_width, source_height)
            strong_threshold = max(self.start_conf, birth_threshold)
            if det.confidence >= strong_threshold:
                required_hits = self.strong_confirm_hits
            elif det.confidence >= birth_threshold:
                required_hits = self.weak_confirm_hits
            else:
                continue
            if not self._confirm_birth(det, now, required_hits):
                continue
            track = self._init_track(det, observation, now, required_hits)
            self._tracks[track.track_id] = track
            self._births += 1

    def _visible_prediction(self, track: _Track, target_time: float):
        dt = max(0.0, float(target_time) - track.state_time)
        dt = min(dt, self.prediction_sec)
        mean, _covariance = self._predict_state(track.mean, track.covariance, dt)

        base = self._box_from_track(track)
        predicted = self._box_from_track(track, mean)
        bcx, bcy, bw, bh = _center_size(base)
        pcx, pcy, pw, ph = _center_size(predicted)
        dx = pcx - bcx
        dy = pcy - bcy
        max_shift = max(12.0, max(bw, bh) * self.max_prediction_shift_boxes)
        magnitude = math.hypot(dx, dy)
        if magnitude > max_shift and magnitude > 1e-6:
            ratio = max_shift / magnitude
            pcx = bcx + dx * ratio
            pcy = bcy + dy * ratio

        max_dw = bw * self.max_prediction_size_ratio
        max_dh = bh * self.max_prediction_size_ratio
        pw = bw + max(-max_dw, min(max_dw, pw - bw))
        ph = bh + max(-max_dh, min(max_dh, ph - bh))
        return _from_center_size(pcx, pcy, pw, ph, track.confidence)

    def visible(
        self,
        now: float,
        target_time: float | None = None,
        max_observation_age_sec: float | None = None,
    ) -> list[VisualBox]:
        target = float(target_time if target_time is not None else now)
        candidates = []
        for track in self._tracks.values():
            wall_age = now - track.last_seen_wall
            if wall_age > self.hold_sec:
                continue
            source_age = max(0.0, target - track.last_observation)
            if max_observation_age_sec is not None and source_age > max_observation_age_sec:
                continue
            if track.hits < self.strong_confirm_hits:
                continue

            box = self._visible_prediction(track, target)
            if target > track.state_time + 0.01:
                self._predicted_visible += 1
            box.confidence = max(
                0.01,
                track.confidence * (1.0 - 0.25 * min(1.0, wall_age / self.hold_sec)),
            )
            candidates.append(box)

        candidates.sort(key=lambda b: b.confidence, reverse=True)
        visible: list[VisualBox] = []
        for box in candidates:
            if any(self._is_duplicate(box, other) for other in visible):
                continue
            visible.append(box)
        return visible

    def metrics(self):
        return {
            'algorithm': 'kalman-byte-visual',
            'tracks_in_memory': len(self._tracks),
            'birth_candidates': len(self._birth_candidates),
            'updates': self._updates,
            'high_matches': self._high_matches,
            'low_matches': self._low_matches,
            'births': self._births,
            'predicted_visible': self._predicted_visible,
            'pruned': self._pruned,
            'byte_high_conf': self.byte_high_conf,
            'byte_low_conf': self.byte_low_conf,
            'prediction_ms': self.prediction_sec * 1000.0,
        }
