from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time


@dataclass(frozen=True, slots=True)
class PresentedTrack:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(slots=True)
class _State:
    track_id: int
    cx: float
    cy: float
    width: float
    height: float
    vx: float
    vy: float
    vw: float
    vh: float
    confidence: float
    last_observation: float
    last_seen_wall: float


class PresentationSmoother:
    """20 FPS display smoother keyed by canonical camera-local ByteTrack IDs.

    This restores the old visual-tracker presentation behavior without creating
    a second identity system. ByteTrack remains authoritative for T-IDs; this
    layer only interpolates/predicts the box between sparse detector updates and
    briefly holds it through short missed detections.
    """

    def __init__(
        self,
        *,
        hold_ms: int = 850,
        memory_ms: int = 2800,
        prediction_ms: int = 340,
        velocity_damping: float = 0.95,
        size_velocity_damping: float = 0.60,
        max_prediction_shift_boxes: float = 0.55,
        max_prediction_size_ratio: float = 0.06,
        adaptive_error_low: float = 0.08,
        adaptive_error_high: float = 0.25,
        center_response_slow: float = 0.42,
        center_response_fast: float = 0.84,
        size_response: float = 0.30,
        snap_distance_boxes: float = 0.62,
        reversal_damping: float = 0.15,
    ) -> None:
        self.hold_sec = max(0.05, float(hold_ms) / 1000.0)
        self.memory_sec = max(self.hold_sec, float(memory_ms) / 1000.0)
        self.prediction_sec = max(0.0, float(prediction_ms) / 1000.0)
        self.velocity_damping = max(0.80, min(1.0, float(velocity_damping)))
        self.size_velocity_damping = max(0.20, min(1.0, float(size_velocity_damping)))
        self.max_prediction_shift_boxes = max(0.10, float(max_prediction_shift_boxes))
        self.max_prediction_size_ratio = max(0.0, min(0.50, float(max_prediction_size_ratio)))
        self.adaptive_error_low = max(0.0, float(adaptive_error_low))
        self.adaptive_error_high = max(self.adaptive_error_low + 1e-6, float(adaptive_error_high))
        self.center_response_slow = max(0.05, min(1.0, float(center_response_slow)))
        self.center_response_fast = max(
            self.center_response_slow, min(1.0, float(center_response_fast))
        )
        self.size_response = max(0.05, min(0.90, float(size_response)))
        self.snap_distance_boxes = max(self.adaptive_error_high, float(snap_distance_boxes))
        self.reversal_damping = max(0.0, min(0.80, float(reversal_damping)))
        self._states: dict[int, _State] = {}
        self._last_frame_id = -1
        self._lock = threading.RLock()

    @staticmethod
    def _center_size(xyxy) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        width = max(2.0, x2 - x1)
        height = max(2.0, y2 - y1)
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5, width, height

    @staticmethod
    def _damped_motion(damping: float, dt: float) -> tuple[float, float]:
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return 0.0, 1.0
        decay = float(damping) ** (dt / 0.1)
        if damping >= 0.999999:
            return dt, decay
        rate = -math.log(float(damping)) / 0.1
        return (1.0 - decay) / max(rate, 1e-9), decay

    def _predict(self, state: _State, target_time: float) -> tuple[float, float, float, float]:
        dt = max(0.0, float(target_time) - state.last_observation)
        horizon = min(dt, self.prediction_sec)
        center_motion, _ = self._damped_motion(self.velocity_damping, horizon)
        size_motion, _ = self._damped_motion(self.size_velocity_damping, horizon)

        cx = state.cx + state.vx * center_motion
        cy = state.cy + state.vy * center_motion
        width = max(2.0, state.width + state.vw * size_motion)
        height = max(2.0, state.height + state.vh * size_motion)

        dx = cx - state.cx
        dy = cy - state.cy
        max_dx = max(12.0, state.width * self.max_prediction_shift_boxes)
        max_dy = max(12.0, state.height * self.max_prediction_shift_boxes)
        normalized = math.hypot(dx / max_dx, dy / max_dy)
        if normalized > 1.0:
            cx = state.cx + dx / normalized
            cy = state.cy + dy / normalized

        max_dw = state.width * self.max_prediction_size_ratio
        max_dh = state.height * self.max_prediction_size_ratio
        width = state.width + max(-max_dw, min(max_dw, width - state.width))
        height = state.height + max(-max_dh, min(max_dh, height - state.height))
        return cx, cy, max(2.0, width), max(2.0, height)

    def _response(self, error_boxes: float) -> float:
        if error_boxes >= self.snap_distance_boxes:
            return 1.0
        if error_boxes <= self.adaptive_error_low:
            return self.center_response_slow
        if error_boxes >= self.adaptive_error_high:
            return self.center_response_fast
        ratio = (error_boxes - self.adaptive_error_low) / (
            self.adaptive_error_high - self.adaptive_error_low
        )
        return self.center_response_slow + ratio * (
            self.center_response_fast - self.center_response_slow
        )

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
            seen: set[int] = set()

            for row in getattr(snapshot, "detections", ()) or ():
                track_id = getattr(row, "track_id", None)
                if track_id is None:
                    continue
                track_id = int(track_id)
                seen.add(track_id)
                mcx, mcy, mw, mh = self._center_size(row.xyxy)
                confidence = float(getattr(row, "confidence", 0.0))
                state = self._states.get(track_id)

                if state is None:
                    self._states[track_id] = _State(
                        track_id=track_id,
                        cx=mcx,
                        cy=mcy,
                        width=mw,
                        height=mh,
                        vx=0.0,
                        vy=0.0,
                        vw=0.0,
                        vh=0.0,
                        confidence=confidence,
                        last_observation=observation,
                        last_seen_wall=now,
                    )
                    continue

                dt = max(1e-3, observation - state.last_observation)
                pcx, pcy, pw, ph = self._predict(state, observation)
                scale = max(20.0, pw, ph, mw, mh)
                error_boxes = math.hypot(mcx - pcx, mcy - pcy) / scale
                response = self._response(error_boxes)

                ncx = pcx + response * (mcx - pcx)
                ncy = pcy + response * (mcy - pcy)
                nwidth = pw + self.size_response * (mw - pw)
                nheight = ph + self.size_response * (mh - ph)

                measured_vx = (ncx - state.cx) / dt
                measured_vy = (ncy - state.cy) / dt
                measured_vw = (nwidth - state.width) / dt
                measured_vh = (nheight - state.height) / dt

                if state.vx * measured_vx < 0.0:
                    state.vx *= self.reversal_damping
                if state.vy * measured_vy < 0.0:
                    state.vy *= self.reversal_damping

                state.vx = 0.55 * state.vx + 0.45 * measured_vx
                state.vy = 0.55 * state.vy + 0.45 * measured_vy
                state.vw = 0.75 * state.vw + 0.25 * measured_vw
                state.vh = 0.75 * state.vh + 0.25 * measured_vh
                state.cx = ncx
                state.cy = ncy
                state.width = max(2.0, nwidth)
                state.height = max(2.0, nheight)
                state.confidence = confidence
                state.last_observation = observation
                state.last_seen_wall = now

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
                cx, cy, width, height = self._predict(state, target_time)
                output.append(
                    PresentedTrack(
                        track_id=state.track_id,
                        xyxy=(
                            cx - width * 0.5,
                            cy - height * 0.5,
                            cx + width * 0.5,
                            cy + height * 0.5,
                        ),
                        confidence=state.confidence,
                    )
                )
        return tuple(output)
