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
    """Responsive legacy-style presentation keyed by canonical ByteTrack T-IDs.

    ByteTrack remains the only identity/association owner. This class never
    creates, merges or reassigns IDs. It restores the old proven presentation
    policy that visibly followed people well: real detector/tracker measurements
    dominate immediately, measured velocity only bridges the short detector gap,
    and prediction is hard-capped so a box cannot trail or fly away.

    The important design split is:
      * ByteTrack owns identity.
      * The newest real measurement owns bbox geometry.
      * Velocity is presentation-only and short lived.
    """

    def __init__(
        self,
        *,
        hold_ms: int = 700,
        memory_ms: int = 6000,
        prediction_ms: int = 200,
        process_noise: float = 1.0,
        measurement_noise: float = 0.70,
        velocity_damping: float = 0.96,
        max_prediction_shift_boxes: float = 0.35,
        max_prediction_size_ratio: float = 0.08,
        snap_distance_boxes: float = 0.55,
        reversal_damping: float = 0.15,
        measurement_response: float = 0.96,
        velocity_response: float = 0.65,
        size_response: float = 0.70,
    ) -> None:
        self.hold_sec = max(0.05, float(hold_ms) / 1000.0)
        self.memory_sec = max(self.hold_sec, float(memory_ms) / 1000.0)
        self.prediction_sec = max(0.0, float(prediction_ms) / 1000.0)
        # Kept for backward-compatible constructor calls. The restored responsive
        # path intentionally does not let a covariance filter overrule geometry.
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.velocity_damping = max(0.80, min(1.0, float(velocity_damping)))
        self.max_prediction_shift_boxes = max(0.05, float(max_prediction_shift_boxes))
        self.max_prediction_size_ratio = max(0.0, min(0.50, float(max_prediction_size_ratio)))
        self.snap_distance_boxes = max(0.10, float(snap_distance_boxes))
        self.reversal_damping = max(0.0, min(0.80, float(reversal_damping)))
        self.measurement_response = max(0.50, min(1.0, float(measurement_response)))
        self.velocity_response = max(0.05, min(1.0, float(velocity_response)))
        self.size_response = max(0.05, min(1.0, float(size_response)))
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
    def _damped_displacement(damping: float, dt: float) -> tuple[float, float]:
        """Return displacement multiplier and remaining velocity.

        Damping is defined per 100 ms just like the old responsive visual tracker,
        so behavior does not depend on an arbitrary render cadence.
        """
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return 0.0, 1.0
        decay = float(damping) ** (dt / 0.1)
        if damping >= 0.999999:
            return dt, decay
        rate = -math.log(float(damping)) / 0.1
        return (1.0 - decay) / max(rate, 1e-9), decay

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

    def _correct(self, state: _State, measurement: np.ndarray, observation: float) -> None:
        dt = max(1e-3, float(observation) - state.last_observation)
        previous_box = state.box.copy()
        old_velocity = state.velocity.copy()

        # Project only to the real observation time. The measurement still owns
        # the final geometry; this prior exists only to estimate whether motion
        # changed direction and to avoid jitter on tiny detector noise.
        motion_dt, _decay = self._damped_displacement(
            self.velocity_damping,
            min(dt, self.prediction_sec),
        )
        prior_center = previous_box[:2] + old_velocity * motion_dt

        measured_velocity = (measurement[:2] - state.last_measurement[:2]) / dt
        dimensions = np.maximum(20.0, measurement[2:4])
        velocity_limit = np.asarray(
            [5.0 * dimensions[0], 5.0 * dimensions[1]], dtype=np.float64
        )
        measured_velocity = np.clip(measured_velocity, -velocity_limit, velocity_limit)

        # Direction reversal was the most visible source of boxes being left
        # behind. Kill stale momentum before mixing the new measured velocity.
        for axis in (0, 1):
            displacement = float(measurement[axis] - state.last_measurement[axis])
            reversal_threshold = 0.03 * float(dimensions[axis])
            if (
                old_velocity[axis] * measured_velocity[axis] < 0.0
                and abs(displacement) >= reversal_threshold
            ):
                old_velocity[axis] *= self.reversal_damping

        state.velocity = (
            (1.0 - self.velocity_response) * old_velocity
            + self.velocity_response * measured_velocity
        )
        state.velocity = np.clip(state.velocity, -velocity_limit, velocity_limit)

        error = measurement[:2] - prior_center
        normalized_error = math.hypot(
            float(error[0]) / float(dimensions[0]),
            float(error[1]) / float(dimensions[1]),
        )

        # Old good behavior: the real observation wins. For a clearly displaced
        # person snap completely; otherwise keep only 4% of the predicted center.
        if normalized_error >= self.snap_distance_boxes:
            center = measurement[:2].copy()
            # A far correction is not evidence for equally huge future momentum.
            state.velocity *= 0.65
        else:
            center = (
                (1.0 - self.measurement_response) * prior_center
                + self.measurement_response * measurement[:2]
            )

        # Keep width/height responsive but bounded so partial detections do not
        # make a person's rectangle breathe violently from one result to the next.
        old_size = np.maximum(2.0, previous_box[2:4])
        requested_delta = measurement[2:4] - old_size
        max_delta = old_size * 0.35
        bounded_measurement_size = old_size + np.clip(
            requested_delta, -max_delta, max_delta
        )
        size = (
            (1.0 - self.size_response) * old_size
            + self.size_response * bounded_measurement_size
        )

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
            seen_ids: set[int] = set()
            for row in getattr(snapshot, "detections", ()) or ():
                track_id = getattr(row, "track_id", None)
                if track_id is None:
                    continue
                track_id = int(track_id)
                seen_ids.add(track_id)
                measurement = self._measurement(row.xyxy)
                confidence = float(getattr(row, "confidence", 0.0))
                state = self._states.get(track_id)
                if state is None:
                    self._states[track_id] = self._init_state(
                        track_id, measurement, observation, now, confidence
                    )
                    continue

                self._correct(state, measurement, observation)
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
                if now - state.last_seen_wall > self.hold_sec:
                    continue

                elapsed = max(0.0, float(target_time) - state.last_observation)
                horizon = min(elapsed, self.prediction_sec)
                motion_dt, _decay = self._damped_displacement(
                    self.velocity_damping,
                    horizon,
                )
                presented = state.box.copy()
                shift = state.velocity * motion_dt

                # Prediction only bridges detector cadence. It is never allowed to
                # drag the rectangle a large fraction of the person size.
                max_shift = np.asarray(
                    [
                        max(8.0, state.box[2] * self.max_prediction_shift_boxes),
                        max(8.0, state.box[3] * self.max_prediction_shift_boxes),
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

                # Size is deliberately not extrapolated. The latest measured body
                # envelope is safer than speculative box breathing between frames.
                output.append(
                    self._presented(presented, state.confidence, state.track_id)
                )
        return tuple(output)
