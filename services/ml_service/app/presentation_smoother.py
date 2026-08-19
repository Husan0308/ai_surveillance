from __future__ import annotations

from dataclasses import dataclass
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
    mean: np.ndarray
    covariance: np.ndarray
    confidence: float
    state_time: float
    last_observation: float
    last_seen_wall: float


class PresentationSmoother:
    """Timestamp-aware Kalman presentation keyed by canonical ByteTrack T-IDs.

    ByteTrack remains the only identity/association owner. This class never
    creates, merges or reassigns IDs; it only projects each already-associated
    T-ID from the detector timestamp to the current camera-frame timestamp.

    The implementation is deliberately biased against visible trailing:
    measurements are trusted strongly, direction reversals damp stale velocity,
    large innovations snap back to the measured person, and extrapolation is
    tightly bounded so a box cannot run far ahead of the person.
    """

    def __init__(
        self,
        *,
        hold_ms: int = 700,
        memory_ms: int = 2800,
        prediction_ms: int = 280,
        process_noise: float = 1.0,
        measurement_noise: float = 0.70,
        velocity_damping: float = 0.985,
        max_prediction_shift_boxes: float = 0.48,
        max_prediction_size_ratio: float = 0.08,
        snap_distance_boxes: float = 0.42,
        reversal_damping: float = 0.12,
    ) -> None:
        self.hold_sec = max(0.05, float(hold_ms) / 1000.0)
        self.memory_sec = max(self.hold_sec, float(memory_ms) / 1000.0)
        self.prediction_sec = max(0.0, float(prediction_ms) / 1000.0)
        self.process_noise = max(0.05, float(process_noise))
        self.measurement_noise = max(0.05, float(measurement_noise))
        self.velocity_damping = max(0.80, min(1.0, float(velocity_damping)))
        self.max_prediction_shift_boxes = max(0.10, float(max_prediction_shift_boxes))
        self.max_prediction_size_ratio = max(0.0, min(0.50, float(max_prediction_size_ratio)))
        self.snap_distance_boxes = max(0.10, float(snap_distance_boxes))
        self.reversal_damping = max(0.0, min(0.80, float(reversal_damping)))
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
    def _box_from_mean(mean: np.ndarray, confidence: float, track_id: int) -> PresentedTrack:
        cx, cy, width, height = [float(v) for v in mean[:4]]
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

    def _init_state(self, track_id: int, measurement: np.ndarray, observation: float, now: float, confidence: float) -> _State:
        scale = max(20.0, float(measurement[2]), float(measurement[3]))
        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = measurement
        pos = (0.035 * scale) ** 2
        size = (0.050 * scale) ** 2
        velocity = (0.55 * scale) ** 2
        covariance = np.diag(
            [pos, pos, size, size, velocity, velocity, velocity, velocity]
        ).astype(np.float64)
        return _State(
            track_id=int(track_id),
            mean=mean,
            covariance=covariance,
            confidence=float(confidence),
            state_time=float(observation),
            last_observation=float(observation),
            last_seen_wall=float(now),
        )

    def _predict_arrays(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        dt: float,
        *,
        cap_horizon: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = max(0.0, float(dt))
        if cap_horizon:
            dt = min(dt, self.prediction_sec)
        if dt <= 0.0:
            return mean.copy(), covariance.copy()

        decay = self.velocity_damping ** (dt / 0.1)
        transition = np.eye(8, dtype=np.float64)
        for position, velocity in ((0, 4), (1, 5), (2, 6), (3, 7)):
            transition[position, velocity] = dt
            transition[velocity, velocity] = decay

        predicted = transition @ mean
        scale = max(20.0, float(predicted[2]), float(predicted[3]))
        q_pos = (0.010 * scale * self.process_noise * (1.0 + 3.0 * dt)) ** 2
        q_size = (0.008 * scale * self.process_noise * (1.0 + 2.0 * dt)) ** 2
        q_vel = (0.14 * scale * self.process_noise * max(0.05, dt)) ** 2
        process_cov = np.diag(
            [q_pos, q_pos, q_size, q_size, q_vel, q_vel, q_vel, q_vel]
        )
        predicted_cov = transition @ covariance @ transition.T + process_cov

        # Presentation prediction is allowed to bridge the detector cadence, not
        # to invent large movement. Clamp both center travel and box breathing.
        if cap_horizon:
            base_w = max(2.0, float(mean[2]))
            base_h = max(2.0, float(mean[3]))
            dx = float(predicted[0] - mean[0])
            dy = float(predicted[1] - mean[1])
            max_dx = max(10.0, base_w * self.max_prediction_shift_boxes)
            max_dy = max(10.0, base_h * self.max_prediction_shift_boxes)
            normalized = ((dx / max_dx) ** 2 + (dy / max_dy) ** 2) ** 0.5
            if normalized > 1.0:
                predicted[0] = mean[0] + dx / normalized
                predicted[1] = mean[1] + dy / normalized

            max_dw = base_w * self.max_prediction_size_ratio
            max_dh = base_h * self.max_prediction_size_ratio
            predicted[2] = mean[2] + np.clip(predicted[2] - mean[2], -max_dw, max_dw)
            predicted[3] = mean[3] + np.clip(predicted[3] - mean[3], -max_dh, max_dh)

        predicted[2] = max(2.0, float(predicted[2]))
        predicted[3] = max(2.0, float(predicted[3]))
        return predicted, predicted_cov

    def _correct(self, state: _State, measurement: np.ndarray, observation: float) -> None:
        dt = max(0.0, float(observation) - state.state_time)
        predicted, predicted_cov = self._predict_arrays(
            state.mean, state.covariance, dt, cap_horizon=False
        )

        scale = max(
            20.0,
            float(predicted[2]),
            float(predicted[3]),
            float(measurement[2]),
            float(measurement[3]),
        )
        innovation = measurement - predicted[:4]
        center_error_boxes = (
            float(np.hypot(innovation[0], innovation[1])) / max(scale, 1.0)
        )

        # A direction reversal is where extrapolated boxes most visibly run in
        # front of a person. Kill stale velocity before correcting the state.
        if predicted[4] * innovation[0] < 0.0 and abs(float(innovation[0])) > 0.03 * scale:
            predicted[4] *= self.reversal_damping
        if predicted[5] * innovation[1] < 0.0 and abs(float(innovation[1])) > 0.03 * scale:
            predicted[5] *= self.reversal_damping

        if center_error_boxes >= self.snap_distance_boxes:
            # The detector has clearly disagreed with the motion estimate. Trust
            # the observed person now instead of visibly easing toward them.
            previous_center = state.mean[:2].copy()
            observed_dt = max(1e-3, float(observation) - state.last_observation)
            predicted[:4] = measurement
            measured_velocity = (measurement[:2] - previous_center) / observed_dt
            predicted[4] = 0.65 * float(measured_velocity[0])
            predicted[5] = 0.65 * float(measured_velocity[1])
            predicted[6] *= 0.25
            predicted[7] *= 0.25
            reset_pos = (0.035 * scale) ** 2
            reset_size = (0.050 * scale) ** 2
            predicted_cov[:4, :4] = np.diag(
                [reset_pos, reset_pos, reset_size, reset_size]
            )
            state.mean = predicted
            state.covariance = predicted_cov
            state.state_time = float(observation)
            return

        measurement_matrix = np.zeros((4, 8), dtype=np.float64)
        measurement_matrix[:4, :4] = np.eye(4, dtype=np.float64)
        r_pos = (0.022 * scale * self.measurement_noise) ** 2
        r_size = (0.030 * scale * self.measurement_noise) ** 2
        measurement_cov = np.diag([r_pos, r_pos, r_size, r_size])

        projected_cov = (
            measurement_matrix @ predicted_cov @ measurement_matrix.T + measurement_cov
        )
        kalman_gain = (
            predicted_cov
            @ measurement_matrix.T
            @ np.linalg.inv(projected_cov)
        )
        corrected = predicted + kalman_gain @ innovation
        identity = np.eye(8, dtype=np.float64)
        corrected_cov = (
            identity - kalman_gain @ measurement_matrix
        ) @ predicted_cov

        # Give the current measurement a final small direct vote. This removes
        # the classic smooth-but-behind visual lag while the Kalman velocity still
        # supplies between-detector-frame motion.
        corrected[0] = 0.18 * corrected[0] + 0.82 * measurement[0]
        corrected[1] = 0.18 * corrected[1] + 0.82 * measurement[1]
        corrected[2] = 0.35 * corrected[2] + 0.65 * measurement[2]
        corrected[3] = 0.35 * corrected[3] + 0.65 * measurement[3]
        corrected[2] = max(2.0, float(corrected[2]))
        corrected[3] = max(2.0, float(corrected[3]))

        state.mean = corrected
        state.covariance = corrected_cov
        state.state_time = float(observation)

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
                        track_id, measurement, observation, now, confidence
                    )
                    continue

                self._correct(state, measurement, observation)
                state.confidence = confidence
                state.last_observation = float(observation)
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
                dt = max(0.0, float(target_time) - state.state_time)
                predicted, _ = self._predict_arrays(
                    state.mean, state.covariance, dt, cap_horizon=True
                )
                output.append(
                    self._box_from_mean(predicted, state.confidence, state.track_id)
                )
        return tuple(output)
