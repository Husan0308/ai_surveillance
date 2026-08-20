from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

import numpy as np


@dataclass(frozen=True, slots=True)
class PresentedTrack:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(slots=True)
class _State:
    track_id: int
    box: np.ndarray
    velocity: np.ndarray
    confidence: float
    last_observation: float
    last_seen_wall: float
    last_measurement: np.ndarray


class PresentationSmoother:
    """Old proven adaptive motion presentation keyed by ByteTrack T-IDs.

    ByteTrack remains the only identity/association owner. This class never
    creates, merges or reassigns IDs. It ports only the proven visual-motion
    behavior from the old core_v1 VisualTracker:

    * timestamp-aware damped velocity,
    * adaptive center response (slow for tiny jitter, fast for real motion),
    * strong damping on stops/direction reversals,
    * bounded prediction from the last real observation,
    * bounded size correction instead of speculative box breathing.

    Empty/stale results can never extend the prediction horizon because every
    projection is anchored to ``last_observation`` from a real ByteTrack result.
    """

    def __init__(
        self,
        *,
        hold_ms: int = 850,
        memory_ms: int = 2800,
        prediction_ms: int = 340,
        velocity_damping: float = 0.95,
        max_prediction_shift_boxes: float = 0.55,
        max_prediction_size_ratio: float = 0.06,
        adaptive_error_low: float = 0.08,
        adaptive_error_high: float = 0.25,
        center_response_slow: float = 0.42,
        center_response_fast: float = 0.84,
        size_response: float = 0.30,
        snap_distance_boxes: float = 0.62,
        reversal_damping: float = 0.15,
        low_conf: float = 0.08,
        strong_conf: float = 0.34,
    ) -> None:
        self.hold_sec = max(0.05, float(hold_ms) / 1000.0)
        self.memory_sec = max(self.hold_sec, float(memory_ms) / 1000.0)
        self.prediction_sec = max(0.0, float(prediction_ms) / 1000.0)
        self.velocity_damping = max(0.80, min(1.0, float(velocity_damping)))
        self.max_prediction_shift_boxes = max(0.10, float(max_prediction_shift_boxes))
        self.max_prediction_size_ratio = max(
            0.0, min(0.50, float(max_prediction_size_ratio))
        )
        self.adaptive_error_low = max(0.0, float(adaptive_error_low))
        self.adaptive_error_high = max(
            self.adaptive_error_low + 1e-6, float(adaptive_error_high)
        )
        self.center_response_slow = max(
            0.05, min(1.0, float(center_response_slow))
        )
        self.center_response_fast = max(
            self.center_response_slow, min(1.0, float(center_response_fast))
        )
        self.size_response = max(0.05, min(0.90, float(size_response)))
        self.snap_distance_boxes = max(
            self.adaptive_error_high, float(snap_distance_boxes)
        )
        self.reversal_damping = max(0.0, min(0.80, float(reversal_damping)))
        self.low_conf = max(0.0, float(low_conf))
        self.strong_conf = max(self.low_conf + 1e-6, float(strong_conf))

        self._states: dict[int, _State] = {}
        self._last_frame_id = -1
        self._lock = threading.RLock()

    @staticmethod
    def _measurement(xyxy) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        width = max(2.0, x2 - x1)
        height = max(2.0, y2 - y1)
        return np.asarray(
            [(x1 + x2) * 0.5, (y1 + y2) * 0.5, width, height],
            dtype=np.float64,
        )

    @staticmethod
    def _presented(box: np.ndarray, confidence: float, track_id: int) -> PresentedTrack:
        cx, cy, width, height = [float(v) for v in box]
        width = max(2.0, width)
        height = max(2.0, height)
        return PresentedTrack(
            track_id=int(track_id),
            xyxy=(
                cx - width * 0.5,
                cy - height * 0.5,
                cx + width * 0.5,
                cy + height * 0.5,
            ),
            confidence=float(confidence),
        )

    @staticmethod
    def _damped_motion(damping: float, dt: float) -> tuple[float, float]:
        """Displacement multiplier + remaining velocity for ``dt``.

        Damping is defined per 100 ms, matching the old visual tracker so motion
        behavior stays stable when detector cadence varies.
        """
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return 0.0, 1.0
        decay = float(damping) ** (dt / 0.1)
        if damping >= 0.999999:
            return dt, decay
        rate = -math.log(float(damping)) / 0.1
        return (1.0 - decay) / max(rate, 1e-9), decay

    def _reliability(self, confidence: float) -> float:
        return min(
            1.0,
            max(
                0.0,
                (float(confidence) - self.low_conf)
                / max(1e-6, self.strong_conf - self.low_conf),
            ),
        )

    def _adaptive_center_response(self, error_boxes: float) -> tuple[float, bool]:
        if error_boxes >= self.snap_distance_boxes:
            return 1.0, True
        if error_boxes <= self.adaptive_error_low:
            return self.center_response_slow, False
        if error_boxes >= self.adaptive_error_high:
            return self.center_response_fast, False
        ratio = (error_boxes - self.adaptive_error_low) / (
            self.adaptive_error_high - self.adaptive_error_low
        )
        response = self.center_response_slow + ratio * (
            self.center_response_fast - self.center_response_slow
        )
        return response, False

    def _init_state(
        self,
        track_id: int,
        measurement: np.ndarray,
        observation: float,
        now: float,
        confidence: float,
    ) -> _State:
        return _State(
            track_id=int(track_id),
            box=measurement.copy(),
            velocity=np.zeros(2, dtype=np.float64),
            confidence=float(confidence),
            last_observation=float(observation),
            last_seen_wall=float(now),
            last_measurement=measurement.copy(),
        )

    def _prior_center(self, state: _State, observation: float) -> np.ndarray:
        elapsed = max(0.0, float(observation) - state.last_observation)
        motion_dt, _decay = self._damped_motion(
            self.velocity_damping,
            min(elapsed, self.prediction_sec),
        )
        return state.box[:2] + state.velocity * motion_dt

    def _correct(
        self,
        state: _State,
        measurement: np.ndarray,
        observation: float,
        confidence: float,
    ) -> None:
        dt = max(1e-3, float(observation) - state.last_observation)
        previous_box = state.box.copy()
        prior_center = self._prior_center(state, observation)
        reliability = self._reliability(confidence)
        reliability2 = reliability * reliability

        dimensions = np.maximum(
            20.0,
            0.5 * (np.maximum(2.0, previous_box[2:4]) + measurement[2:4]),
        )
        innovation = measurement[:2] - prior_center
        normalized_error = math.hypot(
            float(innovation[0]) / float(dimensions[0]),
            float(innovation[1]) / float(dimensions[1]),
        )
        requested_response, snapped = self._adaptive_center_response(normalized_error)

        # This is the key old behavior: tiny detector jitter is smoothed, but a
        # genuinely moving person quickly gets 80%+ measurement authority.
        center_response = min(
            requested_response,
            0.20 + 0.68 * reliability2,
        )
        far_ratio = min(
            1.0,
            max(
                0.0,
                (normalized_error - self.adaptive_error_high)
                / max(1e-6, self.snap_distance_boxes - self.adaptive_error_high),
            ),
        )
        far_curve = far_ratio * far_ratio * (3.0 - 2.0 * far_ratio)
        far_strength = reliability2 * far_curve
        center_response = max(center_response, 0.20 + 0.79 * far_strength)
        if snapped and reliability >= 0.70:
            center_response = 1.0

        center = prior_center + center_response * innovation

        measured_velocity = (measurement[:2] - state.last_measurement[:2]) / dt
        velocity_limit = np.asarray(
            [5.0 * measurement[2], 5.0 * measurement[3]], dtype=np.float64
        )
        measured_velocity = np.clip(measured_velocity, -velocity_limit, velocity_limit)
        old_velocity = state.velocity.copy()

        # The old tracker explicitly killed stale momentum on stop/reversal.
        slowdown = np.zeros(2, dtype=np.float64)
        for axis in (0, 1):
            dimension = max(20.0, float(measurement[2 + axis]))
            displacement = float(
                measurement[axis] - state.last_measurement[axis]
            )
            old_speed = abs(float(old_velocity[axis]))
            measured_speed = abs(float(measured_velocity[axis]))
            is_reversal = (
                old_velocity[axis] * measured_velocity[axis] < 0.0
                and abs(displacement) >= self.adaptive_error_low * dimension
            )
            strength = 0.0
            if old_speed * dt >= 0.025 * dimension and old_speed > 1e-6:
                if is_reversal:
                    strength = reliability2
                else:
                    speed_ratio = measured_speed / old_speed
                    u = max(0.0, min(1.0, (0.85 - speed_ratio) / 0.30))
                    smooth = u * u * (3.0 - 2.0 * u)
                    strength = reliability2 * smooth
            strength = max(strength, far_strength)
            slowdown[axis] = strength
            old_velocity[axis] *= 1.0 - strength * (1.0 - self.reversal_damping)
            if not is_reversal:
                measured_velocity[axis] *= 1.0 - 0.65 * strength

        base_velocity_response = 0.10 + 0.60 * reliability2
        velocity_response = base_velocity_response + slowdown * (
            0.78 - base_velocity_response
        )
        state.velocity = (
            (1.0 - velocity_response) * old_velocity
            + velocity_response * measured_velocity
        )
        state.velocity = np.clip(state.velocity, -velocity_limit, velocity_limit)

        # Old bbox geometry changed size conservatively so partial detections did
        # not make the rectangle breathe. High-confidence measurements still move
        # size faster than weak fragments.
        old_size = np.maximum(2.0, previous_box[2:4])
        size_quality = min(
            1.0,
            max(
                0.0,
                (float(confidence) - 0.24) / max(1e-6, 1.0 - 0.24),
            ),
        )
        size_cap_ratio = 0.08 + 0.32 * size_quality
        size_cap = old_size * size_cap_ratio
        bounded_size = old_size + np.clip(
            measurement[2:4] - old_size,
            -size_cap,
            size_cap,
        )
        size_authority = min(
            self.size_response,
            0.10 + 0.20 * reliability2,
        )
        size = old_size + size_authority * (bounded_size - old_size)

        state.box = np.concatenate((center, np.maximum(2.0, size))).astype(np.float64)
        state.last_measurement = measurement.copy()
        state.last_observation = float(observation)

    def update(self, snapshot) -> None:
        if snapshot is None:
            return
        frame_id = int(getattr(snapshot, "frame_id", -1))
        with self._lock:
            if frame_id <= self._last_frame_id:
                return
            self._last_frame_id = frame_id

            now = time.monotonic()
            observation = float(getattr(snapshot, "captured_monotonic", now))
            if not math.isfinite(observation) or observation <= 0.0:
                observation = now

            for row in getattr(snapshot, "detections", ()) or ():
                track_id = getattr(row, "track_id", None)
                if track_id is None:
                    continue
                track_id = int(track_id)
                measurement = self._measurement(row.xyxy)
                confidence = float(getattr(row, "confidence", 0.0))
                state = self._states.get(track_id)
                if state is None:
                    self._states[track_id] = self._init_state(
                        track_id,
                        measurement,
                        observation,
                        now,
                        confidence,
                    )
                    continue

                self._correct(state, measurement, observation, confidence)
                state.confidence = confidence
                state.last_seen_wall = float(now)

            stale = [
                track_id
                for track_id, state in self._states.items()
                if now - state.last_seen_wall > self.memory_sec
            ]
            for track_id in stale:
                self._states.pop(track_id, None)

    def visible(self, target_time: float) -> tuple[PresentedTrack, ...]:
        now = time.monotonic()
        output: list[PresentedTrack] = []
        with self._lock:
            for state in self._states.values():
                source_age = max(0.0, float(target_time) - state.last_observation)
                if source_age > self.hold_sec or now - state.last_seen_wall > self.hold_sec:
                    continue

                horizon = min(source_age, self.prediction_sec)
                motion_dt, _decay = self._damped_motion(
                    self.velocity_damping,
                    horizon,
                )
                presented = state.box.copy()
                shift = state.velocity * motion_dt

                max_shift = np.asarray(
                    [
                        max(12.0, state.box[2] * self.max_prediction_shift_boxes),
                        max(12.0, state.box[3] * self.max_prediction_shift_boxes),
                    ],
                    dtype=np.float64,
                )
                normalized = math.hypot(
                    float(shift[0]) / float(max_shift[0]),
                    float(shift[1]) / float(max_shift[1]),
                )
                if normalized > 1.0:
                    shift /= normalized
                presented[:2] += shift

                # Never extrapolate size between detector observations. The old
                # tracker bounded scale prediction tightly; keeping measured size
                # is even safer for presentation-only ByteTrack overlays.
                confidence = max(
                    0.01,
                    state.confidence
                    * (1.0 - 0.25 * min(1.0, source_age / self.hold_sec)),
                )
                output.append(
                    self._presented(presented, confidence, state.track_id)
                )
        return tuple(output)
