from __future__ import annotations

from dataclasses import dataclass
import math
import threading

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
    last_measurement: np.ndarray
    last_motion_observation: float
    motion_anchor_confidence: float
    last_match_frame_id: int
    reacquire_pending: bool


@dataclass(slots=True)
class _BirthCandidate:
    box: VisualBox
    observation_time: float
    last_seen_wall: float
    last_frame_id: int
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
    """Cheap camera-local adaptive Kalman/Byte presentation tracker.

    State is ``[cx, cy, w, h, vx, vy, vw, vh]`` at the last *real matched
    detector observation*. Association and display prediction are non-mutating,
    bounded projections from that state. Consequently an empty detector result
    can neither extend the prediction horizon nor turn a short visual hold into
    indefinite dead reckoning.

    This tracker is presentation-only. Neither its IDs nor predicted boxes are
    exposed to the ReID/Global-ID evidence path.
    """

    def __init__(
        self,
        *,
        hold_ms=800,
        memory_ms=3000,
        prediction_ms=420,
        match_iou=0.12,
        reacquire_distance=0.85,
        duplicate_iou=0.68,
        duplicate_containment=0.90,
        duplicate_center_distance=0.20,
        fragment_duplicate=False,
        fragment_horizontal_overlap=0.78,
        fragment_x_center=0.18,
        fragment_max_area_ratio=0.55,
        fragment_min_vertical_overlap=0.20,
        fragment_max_vertical_gap=0.06,
        low_conf_confirm=0.08,
        start_conf=0.34,
        new_track_min_conf=0.24,
        strong_confirm_hits=2,
        weak_confirm_hits=3,
        byte_high_conf=0.24,
        byte_low_conf=0.08,
        byte_second_match_iou=0.04,
        byte_match_center=0.70,
        byte_second_match_center=0.50,
        low_match_max_age_ms=650,
        process_noise=0.85,
        measurement_noise=0.90,
        velocity_damping=0.96,
        size_velocity_damping=0.60,
        max_prediction_shift_boxes=0.55,
        max_prediction_size_ratio=0.08,
        adaptive_error_low=0.08,
        adaptive_error_high=0.25,
        center_response_slow=0.42,
        center_response_fast=0.88,
        size_response=0.30,
        snap_distance_boxes=0.65,
        reversal_damping=0.15,
        new_track_zones=None,
        exclusion_zones=None,
        exclusion_max_box_height=0.24,
        exclusion_overlap_threshold=0.35,
        # Accepted only so older local config can still be parsed. The adaptive
        # Kalman controls above are the canonical smoothing controls.
        smoothing=None,
        center_smoothing=None,
        size_smoothing=None,
        velocity_smoothing=None,
        adaptive_center_smoothing=None,
        adaptive_error_boxes=None,
    ):
        self.hold_sec = max(0.05, float(hold_ms) / 1000.0)
        self.memory_sec = max(self.hold_sec, float(memory_ms) / 1000.0)
        self.prediction_sec = max(0.0, float(prediction_ms) / 1000.0)
        self.match_iou = max(0.0, min(1.0, float(match_iou)))
        self.reacquire_distance = max(0.1, float(reacquire_distance))
        self.duplicate_iou = max(0.0, min(1.0, float(duplicate_iou)))
        self.duplicate_containment = max(0.0, min(1.0, float(duplicate_containment)))
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
        self.byte_low_conf = max(
            self.low_conf_confirm,
            min(self.byte_high_conf, float(byte_low_conf)),
        )
        self.byte_second_match_iou = max(0.0, min(1.0, float(byte_second_match_iou)))
        self.byte_match_center = max(0.1, float(byte_match_center))
        self.byte_second_match_center = max(0.1, float(byte_second_match_center))
        self.low_match_max_age_sec = max(0.05, float(low_match_max_age_ms) / 1000.0)

        self.process_noise = max(0.05, float(process_noise))
        self.measurement_noise = max(0.05, float(measurement_noise))
        self.velocity_damping = max(0.80, min(1.0, float(velocity_damping)))
        self.size_velocity_damping = max(0.20, min(1.0, float(size_velocity_damping)))
        self.max_prediction_shift_boxes = max(0.10, float(max_prediction_shift_boxes))
        self.max_prediction_size_ratio = max(0.0, min(0.50, float(max_prediction_size_ratio)))
        self.adaptive_error_low = max(0.0, float(adaptive_error_low))
        self.adaptive_error_high = max(
            self.adaptive_error_low + 1e-6, float(adaptive_error_high)
        )
        self.center_response_slow = max(0.05, min(1.0, float(center_response_slow)))
        self.center_response_fast = max(
            self.center_response_slow, min(1.0, float(center_response_fast))
        )
        self.size_response = max(0.05, min(0.90, float(size_response)))
        self.snap_distance_boxes = max(
            self.adaptive_error_high, float(snap_distance_boxes)
        )
        self.reversal_damping = max(0.0, min(0.80, float(reversal_damping)))

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
        self.exclusion_overlap_threshold = max(
            0.0, min(1.0, float(exclusion_overlap_threshold))
        )

        self._lock = threading.RLock()
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._last_result_frame_id = -1
        self._last_result_observation = 0.0
        self._last_target_time = 0.0
        self._last_clock_now = 0.0
        self._birth_candidates: list[_BirthCandidate] = []

        self._updates = 0
        self._high_matches = 0
        self._low_matches = 0
        self._births = 0
        self._prediction_renders = 0
        self._pruned = 0
        self._corrections = 0
        self._snaps = 0
        self._direction_reversals = 0
        self._stale_prediction_rejects = 0
        self._invalid_detections = 0
        self._observation_age_sum_ms = 0.0
        self._observation_age_samples = 0
        self._prediction_horizon_sum_ms = 0.0
        self._prediction_horizon_samples = 0

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
            overlap = _intersection(normalized, zone) / max(1e-9, _area(normalized))
            if overlap >= self.exclusion_overlap_threshold:
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
        source_w = None
        source_h = None
        try:
            width = float(source_width)
            height = float(source_height)
            if math.isfinite(width) and math.isfinite(height) and width > 0.0 and height > 0.0:
                source_w = width
                source_h = height
        except (TypeError, ValueError):
            pass

        candidates: list[VisualBox] = []
        for raw in boxes or ():
            try:
                values = [
                    float(raw.x1),
                    float(raw.y1),
                    float(raw.x2),
                    float(raw.y2),
                    float(raw.confidence),
                ]
            except (AttributeError, TypeError, ValueError, OverflowError):
                self._invalid_detections += 1
                continue
            if not all(math.isfinite(value) for value in values):
                self._invalid_detections += 1
                continue
            x1, y1, x2, y2, confidence = values
            if confidence <= 0.0 or x2 <= x1 or y2 <= y1:
                self._invalid_detections += 1
                continue
            if source_w is not None and source_h is not None:
                x1 = max(0.0, min(source_w, x1))
                x2 = max(0.0, min(source_w, x2))
                y1 = max(0.0, min(source_h, y1))
                y2 = max(0.0, min(source_h, y2))
                if x2 <= x1 or y2 <= y1:
                    self._invalid_detections += 1
                    continue
            box = VisualBox(x1, y1, x2, y2, min(1.0, confidence))
            if not self._excluded(box, source_width, source_height):
                candidates.append(box)

        candidates.sort(key=lambda b: b.confidence, reverse=True)
        kept: list[VisualBox] = []
        for box in candidates:
            if any(self._is_duplicate(box, other) for other in kept):
                continue
            kept.append(box)
        return kept

    @staticmethod
    def _measurement(box: VisualBox) -> np.ndarray:
        return np.asarray(_center_size(box), dtype=np.float64)

    @staticmethod
    def _damped_motion(damping: float, dt: float) -> tuple[float, float]:
        """Return displacement multiplier and remaining velocity for ``dt``.

        Damping is defined per 100 ms, making behavior independent of detector
        cadence rather than applying one arbitrary decay per result.
        """
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return 0.0, 1.0
        decay = float(damping) ** (dt / 0.1)
        if damping >= 0.999999:
            return dt, decay
        rate = -math.log(float(damping)) / 0.1
        return (1.0 - decay) / max(rate, 1e-9), decay

    def _predict_state(self, mean: np.ndarray, covariance: np.ndarray, dt: float):
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return mean.copy(), covariance.copy()

        center_motion, center_decay = self._damped_motion(self.velocity_damping, dt)
        size_motion, size_decay = self._damped_motion(self.size_velocity_damping, dt)
        transition = np.eye(8, dtype=np.float64)
        transition[0, 4] = center_motion
        transition[1, 5] = center_motion
        transition[2, 6] = size_motion
        transition[3, 7] = size_motion
        transition[4, 4] = center_decay
        transition[5, 5] = center_decay
        transition[6, 6] = size_decay
        transition[7, 7] = size_decay

        predicted_mean = transition @ mean
        w = max(20.0, float(mean[2]))
        h = max(20.0, float(mean[3]))
        time_scale = math.sqrt(max(1e-4, dt / 0.1))
        qx = 0.018 * w * self.process_noise * time_scale
        qy = 0.018 * h * self.process_noise * time_scale
        qw = 0.012 * w * self.process_noise * time_scale
        qh = 0.012 * h * self.process_noise * time_scale
        qvx = 0.10 * w * self.process_noise * time_scale
        qvy = 0.10 * h * self.process_noise * time_scale
        qvw = 0.025 * w * self.process_noise * time_scale
        qvh = 0.025 * h * self.process_noise * time_scale
        process_cov = np.diag(
            np.square([qx, qy, qw, qh, qvx, qvy, qvw, qvh])
        )
        predicted_covariance = transition @ covariance @ transition.T + process_cov
        predicted_mean[2] = max(2.0, predicted_mean[2])
        predicted_mean[3] = max(2.0, predicted_mean[3])
        return predicted_mean, predicted_covariance

    def _bounded_prediction(self, track: _Track, target_time: float):
        elapsed = max(0.0, float(target_time) - track.state_time)
        horizon = min(elapsed, self.prediction_sec)
        mean, covariance = self._predict_state(track.mean, track.covariance, horizon)

        # If the observation gap exceeds the allowed presentation horizon, stop
        # translating but keep decaying velocity so dormant tracks cannot fly into
        # a later association.
        remaining = elapsed - horizon
        if remaining > 0.0:
            _unused, center_decay = self._damped_motion(self.velocity_damping, remaining)
            _unused, size_decay = self._damped_motion(self.size_velocity_damping, remaining)
            mean[4:6] *= center_decay
            mean[6:8] *= size_decay

        base_cx, base_cy, base_w, base_h = track.mean[:4]
        dx = float(mean[0] - base_cx)
        dy = float(mean[1] - base_cy)
        max_dx = max(12.0, float(base_w) * self.max_prediction_shift_boxes)
        max_dy = max(12.0, float(base_h) * self.max_prediction_shift_boxes)
        normalized = math.hypot(dx / max_dx, dy / max_dy)
        if normalized > 1.0:
            mean[0] = base_cx + dx / normalized
            mean[1] = base_cy + dy / normalized

        max_dw = float(base_w) * self.max_prediction_size_ratio
        max_dh = float(base_h) * self.max_prediction_size_ratio
        mean[2] = max(2.0, base_w + max(-max_dw, min(max_dw, mean[2] - base_w)))
        mean[3] = max(2.0, base_h + max(-max_dh, min(max_dh, mean[3] - base_h)))
        return mean, covariance, horizon

    def _observation_prediction(self, track: _Track, target_time: float):
        """Project to a real observation time without display-only center caps.

        The display cap intentionally limits speculative pixels shown to users.
        Reusing it for association made a valid fast track appear artificially
        behind after an irregular detector interval, causing fragmentation.
        Active tracks therefore use their full damped motion model for matching;
        dormant tracks remain bounded for safe reacquisition.
        """
        elapsed = max(0.0, float(target_time) - track.state_time)
        if elapsed > self.hold_sec:
            return self._bounded_prediction(track, target_time)

        mean, covariance = self._predict_state(track.mean, track.covariance, elapsed)
        base_w = float(track.mean[2])
        base_h = float(track.mean[3])
        max_dw = base_w * self.max_prediction_size_ratio
        max_dh = base_h * self.max_prediction_size_ratio
        mean[2] = max(2.0, base_w + max(-max_dw, min(max_dw, mean[2] - base_w)))
        mean[3] = max(2.0, base_h + max(-max_dh, min(max_dh, mean[3] - base_h)))
        return mean, covariance, elapsed

    def _init_track(
        self,
        box: VisualBox,
        observation: float,
        now: float,
        frame_id: int,
        hits: int,
        previous_box: VisualBox | None = None,
        previous_observation: float | None = None,
    ) -> _Track:
        z = self._measurement(box)
        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = z
        if previous_box is not None and previous_observation is not None:
            dt = float(observation) - float(previous_observation)
            if dt > 1e-3:
                previous = self._measurement(previous_box)
                measured = (z - previous) / dt
                mean[4] = float(np.clip(measured[0], -5.0 * z[2], 5.0 * z[2]))
                mean[5] = float(np.clip(measured[1], -5.0 * z[3], 5.0 * z[3]))
                # Size velocity begins at zero. A second sequence of real
                # observations must establish a scale trend before extrapolation.

        px = (0.06 * max(20.0, z[2])) ** 2
        py = (0.06 * max(20.0, z[3])) ** 2
        sw = (0.08 * max(20.0, z[2])) ** 2
        sh = (0.08 * max(20.0, z[3])) ** 2
        vx = (0.55 * max(20.0, z[2])) ** 2
        vy = (0.55 * max(20.0, z[3])) ** 2
        vw = (0.15 * max(20.0, z[2])) ** 2
        vh = (0.15 * max(20.0, z[3])) ** 2
        covariance = np.diag([px, py, sw, sh, vx, vy, vw, vh])
        track = _Track(
            track_id=self._next_id,
            mean=mean,
            covariance=covariance,
            confidence=float(box.confidence),
            state_time=float(observation),
            last_observation=float(observation),
            last_seen_wall=float(now),
            hits=max(1, int(hits)),
            last_measurement=z.copy(),
            last_motion_observation=float(observation),
            motion_anchor_confidence=float(box.confidence),
            last_match_frame_id=int(frame_id),
            reacquire_pending=False,
        )
        self._next_id += 1
        return track

    def _adaptive_center_response(self, error: float) -> tuple[float, bool]:
        if error >= self.snap_distance_boxes:
            return 1.0, True
        if error <= self.adaptive_error_low:
            return self.center_response_slow, False
        if error >= self.adaptive_error_high:
            return self.center_response_fast, False
        ratio = (error - self.adaptive_error_low) / (
            self.adaptive_error_high - self.adaptive_error_low
        )
        response = self.center_response_slow + ratio * (
            self.center_response_fast - self.center_response_slow
        )
        return response, False

    def _detection_reliability(self, confidence: float) -> float:
        return min(
            1.0,
            max(
                0.0,
                (float(confidence) - self.byte_low_conf)
                / max(1e-6, self.start_conf - self.byte_low_conf),
            ),
        )

    def _correct_track(
        self,
        track: _Track,
        box: VisualBox,
        observation: float,
        now: float,
        frame_id: int,
        *,
        low_stage: bool = False,
    ) -> None:
        observation_gap = max(0.0, float(observation) - track.last_observation)
        prior, prior_covariance, _horizon = self._observation_prediction(
            track, observation
        )
        confidence = max(0.01, min(1.0, float(box.confidence)))
        reliability = self._detection_reliability(confidence)
        reliability2 = reliability * reliability

        z = self._measurement(box)
        # A centered fragment or full-frame false positive must not make width and
        # height breathe violently. Limit the real-observation size innovation as
        # a function of confidence before it reaches either Kalman correction or
        # size-velocity estimation.
        size_quality = min(
            1.0,
            max(
                0.0,
                (confidence - self.byte_high_conf)
                / max(1e-6, 1.0 - self.byte_high_conf),
            ),
        )
        size_cap_ratio = 0.08 + 0.32 * size_quality
        size_cap = np.maximum(2.0, prior[2:4]) * size_cap_ratio
        z[2:4] = prior[2:4] + np.clip(
            z[2:4] - prior[2:4], -size_cap, size_cap
        )

        innovation = z - prior[:4]
        sx = max(20.0, 0.5 * (prior[2] + z[2]))
        sy = max(20.0, 0.5 * (prior[3] + z[3]))
        normalized_error = math.hypot(innovation[0] / sx, innovation[1] / sy)
        requested_center_response, _far_error = self._adaptive_center_response(
            normalized_error
        )

        # Correction authority is continuous across the Byte high/low boundary.
        # A 0.240 box therefore cannot suddenly gain snap authority that a 0.239
        # box did not have.
        center_response = min(
            requested_center_response,
            0.20 + 0.68 * reliability2,
        )
        size_response = min(
            self.size_response,
            0.10 + 0.20 * reliability2,
        )
        # Ramp reinitialization behavior across an error band instead of at a
        # single distance. This removes the old 0.01px snap-threshold cliff while
        # still making a clearly far, reliable observation correct aggressively.
        far_start = self.adaptive_error_high
        far_ratio = min(
            1.0,
            max(
                0.0,
                (normalized_error - far_start)
                / max(1e-6, self.snap_distance_boxes - far_start),
            ),
        )
        far_curve = far_ratio * far_ratio * (3.0 - 2.0 * far_ratio)
        far_strength = reliability2 * far_curve
        center_response = max(
            center_response,
            0.20 + 0.79 * far_strength,
        )
        # ``snapped`` is telemetry only; correction dynamics use far_strength.
        snapped = far_strength >= 0.50

        confidence_factor = 1.0 + (1.0 - confidence) * 2.0
        pos_x = 0.025 * sx * self.measurement_noise * confidence_factor
        pos_y = 0.025 * sy * self.measurement_noise * confidence_factor
        size_x = 0.055 * max(20.0, z[2]) * self.measurement_noise * confidence_factor
        size_y = 0.055 * max(20.0, z[3]) * self.measurement_noise * confidence_factor
        measurement_covariance = np.diag(
            np.square([pos_x, pos_y, size_x, size_y])
        )
        innovation_covariance = prior_covariance[:4, :4] + measurement_covariance
        try:
            gain = np.linalg.solve(
                innovation_covariance.T, prior_covariance[:, :4].T
            ).T
        except np.linalg.LinAlgError:
            gain = prior_covariance[:, :4] @ np.linalg.pinv(innovation_covariance)
        kalman_posterior = prior + gain @ innovation
        posterior = kalman_posterior.copy()
        covariance = prior_covariance - gain @ prior_covariance[:4, :]
        covariance = (covariance + covariance.T) * 0.5

        adaptive_center = prior[:2] + center_response * innovation[:2]
        adaptive_size = prior[2:4] + size_response * innovation[2:4]
        if far_strength > 0.0:
            posterior[:2] = (
                far_strength * adaptive_center
                + (1.0 - far_strength)
                * (0.15 * kalman_posterior[:2] + 0.85 * adaptive_center)
            )
        else:
            posterior[:2] = 0.15 * kalman_posterior[:2] + 0.85 * adaptive_center
        posterior[2:4] = 0.20 * kalman_posterior[2:4] + 0.80 * adaptive_size
        posterior[2:4] = prior[2:4] + np.clip(
            posterior[2:4] - prior[2:4], -size_cap, size_cap
        )

        motion_dt = max(
            0.0, float(observation) - track.last_motion_observation
        )
        if motion_dt > 1e-3:
            displacement = z - track.last_measurement
            measured_velocity = displacement / motion_dt
            measured_center_velocity = measured_velocity[:2].copy()
            old_velocity = prior[4:6].copy()

            # A far correction reinitializes position; the large innovation is
            # not itself proof of equally large sustained velocity. Preserve the
            # already learned motion and let following real observations establish
            # a new rate instead of making the box fly beyond the corrected person.
            if far_strength > 0.0:
                measured_center_velocity = (
                    (1.0 - far_strength) * measured_center_velocity
                    + far_strength * old_velocity
                )

            base_velocity_response = 0.10 + 0.60 * reliability2
            slowdown_strength = np.zeros(2, dtype=np.float64)
            reversed_axis = False
            for axis, dimension in enumerate((z[2], z[3])):
                old_speed = abs(float(old_velocity[axis]))
                measured_speed = abs(float(measured_center_velocity[axis]))
                stop_threshold = 0.025 * max(20.0, float(dimension))
                strength = 0.0
                is_reversal = (
                    old_velocity[axis] * measured_center_velocity[axis] < 0.0
                    and abs(float(displacement[axis]))
                    >= self.adaptive_error_low * max(20.0, float(dimension))
                )
                if old_speed * motion_dt >= stop_threshold and old_speed > 1e-6:
                    if is_reversal:
                        strength = reliability2
                    else:
                        speed_ratio = measured_speed / old_speed
                        u = max(0.0, min(1.0, (0.85 - speed_ratio) / 0.30))
                        smooth = u * u * (3.0 - 2.0 * u)
                        strength = reliability2 * smooth
                if far_strength > 0.0:
                    strength = max(strength, far_strength)

                slowdown_strength[axis] = strength
                old_velocity[axis] *= 1.0 - strength * (
                    1.0 - self.reversal_damping
                )
                if not is_reversal:
                    measured_center_velocity[axis] *= 1.0 - 0.65 * strength
                elif strength >= 0.50:
                    reversed_axis = True

            velocity_response = base_velocity_response + slowdown_strength * (
                0.78 - base_velocity_response
            )
            center_velocity = (
                (1.0 - velocity_response) * old_velocity
                + velocity_response * measured_center_velocity
            )
            center_limits = np.asarray(
                [5.0 * z[2], 5.0 * z[3]], dtype=np.float64
            )
            kalman_velocity_weight = (
                (0.02 + 0.08 * reliability2) * (1.0 - far_strength)
            )
            center_velocity = (
                kalman_velocity_weight * kalman_posterior[4:6]
                + (1.0 - kalman_velocity_weight) * center_velocity
            )
            posterior[4:6] = np.clip(
                center_velocity, -center_limits, center_limits
            )

            size_velocity_response = 0.02 + 0.04 * reliability2
            size_velocity = (
                (1.0 - size_velocity_response) * prior[6:8]
                + size_velocity_response * measured_velocity[2:4]
            )
            size_denominator = max(self.prediction_sec, 0.10)
            size_limits = np.asarray(
                [
                    z[2] * self.max_prediction_size_ratio / size_denominator,
                    z[3] * self.max_prediction_size_ratio / size_denominator,
                ],
                dtype=np.float64,
            )
            size_velocity = (
                0.10 * reliability2 * kalman_posterior[6:8]
                + (1.0 - 0.10 * reliability2) * size_velocity
            )
            posterior[6:8] = np.clip(size_velocity, -size_limits, size_limits)
            if reversed_axis:
                self._direction_reversals += 1
        else:
            posterior[4:6] = prior[4:6]
            posterior[6:8] = prior[6:8]

        posterior[2] = max(2.0, posterior[2])
        posterior[3] = max(2.0, posterior[3])
        track.mean = posterior
        track.covariance = covariance
        track.confidence = max(confidence, track.confidence * 0.80)
        track.state_time = float(observation)
        track.last_observation = float(observation)
        track.last_seen_wall = float(now)
        # Maintain a confidence-weighted motion anchor on every real match.
        # Weak observations influence the next velocity baseline only in
        # proportion to their authority, avoiding both weak rebound and a hard
        # anchor transition at start_conf.
        anchor_weight = min(
            1.0,
            reliability2 * reliability2 * reliability2 * reliability2
            + 0.25 * far_strength,
        )
        if motion_dt > 1e-3:
            track.last_measurement = (
                (1.0 - anchor_weight) * track.last_measurement
                + anchor_weight * z
            )
            track.last_motion_observation = (
                (1.0 - anchor_weight) * track.last_motion_observation
                + anchor_weight * float(observation)
            )
            track.motion_anchor_confidence = (
                (1.0 - anchor_weight) * track.motion_anchor_confidence
                + anchor_weight * confidence
            )
        track.last_match_frame_id = int(frame_id)
        if observation_gap > self.hold_sec:
            track.reacquire_pending = True
        track.hits += 1
        self._corrections += 1
        if snapped:
            self._snaps += 1

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
        observation: float,
        decisive_high: bool,
    ):
        pairs = []
        for tid in track_ids:
            track = self._tracks[tid]
            predicted, predicted_covariance, _horizon = self._observation_prediction(
                track, observation
            )
            ref = self._box_from_track(track, predicted)
            age = max(0.0, observation - track.last_observation)
            for di, det in enumerate(detections):
                if (
                    det.confidence < self.byte_high_conf
                    and age > self.low_match_max_age_sec
                ):
                    continue
                iou = _iou(ref, det)
                rcx, rcy, rw, rh = _center_size(ref)
                dcx, dcy, dw, dh = _center_size(det)
                sx = max(20.0, 0.5 * (rw + dw))
                sy = max(20.0, 0.5 * (rh + dh))
                distance = math.hypot((dcx - rcx) / sx, (dcy - rcy) / sy)
                isotropic_distance = _center_distance(ref, det)
                area_similarity = _area_ratio(ref, det)
                if area_similarity < 0.18:
                    continue

                reliability = self._detection_reliability(det.confidence)
                reliability2 = reliability * reliability
                center_cap = self.byte_second_match_center + reliability * (
                    self.byte_match_center - self.byte_second_match_center
                )
                center_gate = 0.20 + reliability2 * (center_cap - 0.20)
                iou_gate = self.byte_second_match_iou + reliability * (
                    self.match_iou - self.byte_second_match_iou
                )
                if age <= self.hold_sec:
                    sigma_norm = math.hypot(
                        math.sqrt(max(0.0, float(predicted_covariance[0, 0]))) / sx,
                        math.sqrt(max(0.0, float(predicted_covariance[1, 1]))) / sy,
                    )
                    center_gate += min(0.20, 2.0 * sigma_norm)
                else:
                    center_gate = max(
                        center_gate,
                        0.20
                        + reliability2 * (self.reacquire_distance - 0.20),
                    )

                far_quality = min(
                    1.0,
                    max(
                        0.0,
                        (det.confidence - self.start_conf)
                        / max(1e-6, 1.0 - self.start_conf),
                    ),
                )
                far_gate = reliability * (
                    self.snap_distance_boxes
                    + far_quality
                    * max(
                        0.0,
                        self.reacquire_distance - self.snap_distance_boxes,
                    )
                )
                strong_far_recovery = (
                    area_similarity >= 0.45
                    and isotropic_distance <= far_gate
                )

                if decisive_high:
                    trust = min(
                        1.0,
                        max(
                            0.0,
                            (det.confidence - self.byte_high_conf)
                            / max(1e-6, self.start_conf - self.byte_high_conf),
                        ),
                    )
                    decisive_center = 0.12 + trust * (center_gate - 0.12)
                    decisive_iou = 0.55 + trust * (iou_gate - 0.55)
                    accepted = (
                        iou >= decisive_iou
                        or distance <= decisive_center
                        or strong_far_recovery
                    )
                else:
                    accepted = (
                        iou >= iou_gate
                        or distance <= center_gate
                        or strong_far_recovery
                    )
                    if (
                        det.confidence < self.start_conf
                        and distance > center_gate
                    ):
                        accepted = False
                if not accepted:
                    continue

                size_cost = -math.log(max(1e-6, area_similarity))
                cost = (
                    0.55 * min(3.0, distance)
                    + 0.35 * (1.0 - iou)
                    + 0.08 * min(3.0, size_cost)
                    + 0.02 * (1.0 - det.confidence)
                )
                pairs.append((cost, tid, di))

        pairs.sort(key=lambda item: item[0])
        used_tracks = set()
        used_detections = set()
        matches = []
        for _cost, tid, di in pairs:
            if tid in used_tracks or di in used_detections:
                continue
            used_tracks.add(tid)
            used_detections.add(di)
            matches.append((tid, di))
        return matches, used_tracks, used_detections

    def _confirm_birth(
        self,
        det: VisualBox,
        observation: float,
        now: float,
        frame_id: int,
        required_hits: int,
        used_candidates: set[int],
    ):
        self._birth_candidates = [
            candidate
            for candidate in self._birth_candidates
            if now - candidate.last_seen_wall <= 1.25
        ]
        best = None
        best_score = float('inf')
        for candidate in self._birth_candidates:
            if id(candidate) in used_candidates or candidate.last_frame_id == frame_id:
                continue
            iou = _iou(candidate.box, det)
            distance = _center_distance(candidate.box, det)
            elapsed = max(0.0, observation - candidate.observation_time)
            max_center = min(
                1.35,
                max(0.35, 0.35 + 2.5 * min(elapsed, 0.40)),
            )
            if iou < 0.12 and distance > max_center:
                continue
            score = (1.0 - iou) + 0.25 * distance
            if score < best_score:
                best = candidate
                best_score = score

        if best is None:
            if required_hits <= 1:
                return det, observation, 1
            self._birth_candidates.append(
                _BirthCandidate(det, observation, now, frame_id, 1)
            )
            return None

        used_candidates.add(id(best))
        previous_box = best.box
        previous_observation = best.observation_time
        best.box = det
        best.observation_time = observation
        best.last_seen_wall = now
        best.last_frame_id = frame_id
        best.hits += 1
        if best.hits >= required_hits:
            hits = best.hits
            self._birth_candidates.remove(best)
            return previous_box, previous_observation, hits
        return None

    def _overlaps_existing_track(self, det: VisualBox, observation: float) -> bool:
        for track in self._tracks.values():
            predicted, _covariance, _horizon = self._bounded_prediction(track, observation)
            box = self._box_from_track(track, predicted)
            if self._is_duplicate(det, box):
                return True
        return False

    def update(self, result, now: float, source_width=None, source_height=None) -> None:
        if result is None:
            return
        with self._lock:
            frame_id = int(result.frame_id)
            if frame_id <= self._last_result_frame_id:
                return
            observation = float(getattr(result, 'frame_captured_monotonic', now) or now)
            if not math.isfinite(observation) or observation <= 0.0:
                observation = float(now)
            if (
                self._last_result_observation > 0.0
                and observation + 1e-6 < self._last_result_observation
            ):
                return

            self._last_result_frame_id = frame_id
            self._last_result_observation = observation
            self._last_target_time = max(self._last_target_time, observation)
            self._last_clock_now = float(now)
            self._updates += 1
            detections = self._dedupe(result.boxes, source_width, source_height)

            for tid in list(self._tracks):
                if now - self._tracks[tid].last_seen_wall > self.memory_sec:
                    del self._tracks[tid]
                    self._pruned += 1

            high = [d for d in detections if d.confidence >= self.byte_high_conf]
            low = [
                d
                for d in detections
                if self.byte_low_conf <= d.confidence < self.byte_high_conf
            ]
            track_ids = list(self._tracks)
            matches1, used_tracks1, used_high_stage1 = self._associate(
                track_ids,
                high,
                observation=observation,
                decisive_high=True,
            )
            used_high = set(used_high_stage1)
            for tid, di in matches1:
                self._correct_track(
                    self._tracks[tid], high[di], observation, now, frame_id
                )
                self._high_matches += 1

            # Borderline-high and low boxes compete together for the remaining
            # tracks. This prevents a 0.240 distractor from pre-empting a better
            # 0.239 continuation solely because it crossed a confidence boundary.
            remaining_tracks = [tid for tid in track_ids if tid not in used_tracks1]
            fallback = [
                (high[index], True, index)
                for index in range(len(high))
                if index not in used_high
            ]
            fallback.extend((det, False, index) for index, det in enumerate(low))
            fallback_boxes = [entry[0] for entry in fallback]
            matches2, _used_tracks2, _used_fallback = self._associate(
                remaining_tracks,
                fallback_boxes,
                observation=observation,
                decisive_high=False,
            )
            for tid, fallback_index in matches2:
                det, is_high, original_index = fallback[fallback_index]
                self._correct_track(
                    self._tracks[tid],
                    det,
                    observation,
                    now,
                    frame_id,
                    low_stage=not is_high,
                )
                if is_high:
                    used_high.add(original_index)
                    self._high_matches += 1
                else:
                    self._low_matches += 1

            # Byte stage 2 is continuation-only. Only unmatched high-confidence
            # observations may accumulate evidence for a new visual track.
            unmatched_high = [d for i, d in enumerate(high) if i not in used_high]
            used_candidates: set[int] = set()
            for det in unmatched_high:
                if self._overlaps_existing_track(det, observation):
                    continue
                birth_threshold = self._birth_threshold(
                    det, source_width, source_height
                )
                if det.confidence < birth_threshold:
                    continue
                required_hits = (
                    self.strong_confirm_hits
                    if det.confidence >= max(self.start_conf, birth_threshold)
                    else self.weak_confirm_hits
                )
                confirmation = self._confirm_birth(
                    det,
                    observation,
                    now,
                    frame_id,
                    required_hits,
                    used_candidates,
                )
                if confirmation is None:
                    continue
                previous_box, previous_observation, hits = confirmation
                track = self._init_track(
                    det,
                    observation,
                    now,
                    frame_id,
                    hits,
                    previous_box,
                    previous_observation,
                )
                self._tracks[track.track_id] = track
                self._births += 1
                self._birth_candidates = [
                    candidate
                    for candidate in self._birth_candidates
                    if not self._is_duplicate(candidate.box, det)
                ]

    def _visible_prediction(self, track: _Track, target_time: float):
        mean, _covariance, horizon = self._bounded_prediction(track, target_time)
        return self._box_from_track(track, mean), horizon

    def visible(
        self,
        now: float,
        target_time: float | None = None,
        max_observation_age_sec: float | None = None,
    ) -> list[VisualBox]:
        with self._lock:
            target = float(target_time if target_time is not None else now)
            self._last_clock_now = float(now)
            self._last_target_time = max(self._last_target_time, target)
            candidates = []
            for track in self._tracks.values():
                source_age = target - track.last_observation
                if source_age < -1e-6:
                    self._stale_prediction_rejects += 1
                    continue
                source_age = max(0.0, source_age)
                if source_age > self.hold_sec:
                    self._stale_prediction_rejects += 1
                    continue
                if (
                    max_observation_age_sec is not None
                    and source_age > max_observation_age_sec
                ):
                    self._stale_prediction_rejects += 1
                    continue
                if track.hits < self.strong_confirm_hits:
                    continue

                box, horizon = self._visible_prediction(track, target)
                box.confidence = max(
                    0.01,
                    track.confidence
                    * (1.0 - 0.25 * min(1.0, source_age / self.hold_sec)),
                )
                candidates.append((source_age, track, box, horizon))

            candidates.sort(key=lambda item: (item[0], -item[2].confidence))

            # Geometry alone cannot distinguish overlapping people. Suppression
            # is therefore restricted to a track that was reacquired after being
            # visually stale, paired one-to-one with a more continuously observed
            # overlapping track. Two tracks backed by the latest detector result
            # always remain distinct.
            conflicts = []
            pending_conflicts: set[int] = set()
            for left in range(len(candidates)):
                _left_age, left_track, left_box, _left_horizon = candidates[left]
                for right in range(left + 1, len(candidates)):
                    _right_age, right_track, right_box, _right_horizon = candidates[right]
                    if not (
                        left_track.reacquire_pending
                        or right_track.reacquire_pending
                    ):
                        continue
                    if (
                        left_track.last_match_frame_id == self._last_result_frame_id
                        and right_track.last_match_frame_id == self._last_result_frame_id
                    ):
                        continue
                    if (
                        abs(
                            left_track.last_observation
                            - right_track.last_observation
                        )
                        > self.low_match_max_age_sec
                    ):
                        continue
                    overlap = _iou(left_box, right_box)
                    distance = _center_distance(left_box, right_box)
                    if not self._is_duplicate(left_box, right_box) and not (
                        overlap >= 0.45 and distance <= 0.30
                    ):
                        continue
                    pending_conflicts.add(left_track.track_id)
                    pending_conflicts.add(right_track.track_id)
                    conflicts.append((-overlap, distance, left, right))

            conflicts.sort()
            paired: set[int] = set()
            suppressed: set[int] = set()
            for _negative_overlap, _distance, left, right in conflicts:
                if left in paired or right in paired:
                    continue
                left_track = candidates[left][1]
                right_track = candidates[right][1]
                left_rank = (
                    left_track.last_observation,
                    candidates[left][2].confidence,
                    left_track.hits,
                    -left_track.track_id,
                )
                right_rank = (
                    right_track.last_observation,
                    candidates[right][2].confidence,
                    right_track.hits,
                    -right_track.track_id,
                )
                suppressed.add(right if left_rank >= right_rank else left)
                paired.add(left)
                paired.add(right)

            for _source_age, track, _box, _horizon in candidates:
                if (
                    track.reacquire_pending
                    and track.track_id not in pending_conflicts
                ):
                    track.reacquire_pending = False

            visible = [
                item for index, item in enumerate(candidates) if index not in suppressed
            ]
            for source_age, _track, _box, horizon in visible:
                if horizon > 0.001:
                    self._prediction_renders += 1
                    self._prediction_horizon_sum_ms += horizon * 1000.0
                    self._prediction_horizon_samples += 1
                self._observation_age_sum_ms += source_age * 1000.0
                self._observation_age_samples += 1
            return [box for _source_age, _track, box, _horizon in visible]

    def metrics(self):
        with self._lock:
            target = self._last_target_time
            active_tracks = sum(
                1
                for track in self._tracks.values()
                if target >= track.last_observation
                and target - track.last_observation <= self.hold_sec
                and track.hits >= self.strong_confirm_hits
            )
            average_age = (
                self._observation_age_sum_ms / self._observation_age_samples
                if self._observation_age_samples
                else 0.0
            )
            average_horizon = (
                self._prediction_horizon_sum_ms / self._prediction_horizon_samples
                if self._prediction_horizon_samples
                else 0.0
            )
            return {
                'algorithm': 'adaptive-kalman-byte-visual-v2',
                'active_tracks': active_tracks,
                'tracks_in_memory': len(self._tracks),
                'birth_candidates': len(self._birth_candidates),
                'updates': self._updates,
                'high_matches': self._high_matches,
                'low_matches': self._low_matches,
                'births': self._births,
                'prediction_renders': self._prediction_renders,
                'predicted_visible': self._prediction_renders,
                'average_observation_age_ms': average_age,
                'average_prediction_horizon_ms': average_horizon,
                'corrections': self._corrections,
                'snaps': self._snaps,
                'direction_reversals': self._direction_reversals,
                'stale_prediction_rejects': self._stale_prediction_rejects,
                'invalid_detections': self._invalid_detections,
                'pruned': self._pruned,
                'byte_high_conf': self.byte_high_conf,
                'byte_low_conf': self.byte_low_conf,
                'prediction_ms': self.prediction_sec * 1000.0,
                'max_prediction_shift_boxes': self.max_prediction_shift_boxes,
                'max_prediction_size_ratio': self.max_prediction_size_ratio,
            }
