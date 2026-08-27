from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from services.ml_service.app.local_tracker import Detection, TrackSnapshot, TrackerUpdate, appearance_descriptor
from services.ml_service.app.local_tracker_sparse_v3 import ObservationRecoveryPersonTracker


class BoxStableObservationRecoveryTracker(ObservationRecoveryPersonTracker):
    """Step 4 v4: keep V3 association, stabilize the rendered person box.

    Two live-camera failure modes are handled here without adding a pose model or GPU
    tracker:

    * a raised arm / partial-person detection can create a smaller nested person box whose
      IoU with the existing full-body track is deceptively low; use intersection-over-
      smaller-area + appearance + center proximity to veto that duplicate ID;
    * detector box width/height can jump when limbs move. Association still sees the raw
      detector measurement, but the published/rendered box uses a separate bounded state
      anchored by horizontal center + bottom edge, with slower size updates.

    The render state NEVER feeds back into data association, so this cannot make the
    tracker teleport to a UI-smoothed box. No pose, ReID, NvDCF or extra GPU inference.
    """

    def __init__(
        self,
        *args,
        nested_duplicate_ios: float = 0.82,
        nested_duplicate_app_floor: float = 0.58,
        nested_duplicate_center_frac: float = 0.28,
        render_anchor_alpha: float = 0.72,
        render_size_alpha: float = 0.20,
        render_recovery_size_alpha: float = 0.34,
        render_max_size_step: float = 0.28,
        render_velocity_gain: float = 0.30,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.nested_duplicate_ios = min(0.98, max(0.60, float(nested_duplicate_ios)))
        self.nested_duplicate_app_floor = min(
            0.95, max(0.0, float(nested_duplicate_app_floor))
        )
        self.nested_duplicate_center_frac = min(
            0.60, max(0.10, float(nested_duplicate_center_frac))
        )
        self.render_anchor_alpha = min(0.95, max(0.20, float(render_anchor_alpha)))
        self.render_size_alpha = min(0.60, max(0.05, float(render_size_alpha)))
        self.render_recovery_size_alpha = min(
            0.75, max(self.render_size_alpha, float(render_recovery_size_alpha))
        )
        self.render_max_size_step = min(0.70, max(0.08, float(render_max_size_step)))
        self.render_velocity_gain = min(0.70, max(0.05, float(render_velocity_gain)))
        self._render_state: dict[int, np.ndarray] = {}
        self._render_velocity: dict[int, np.ndarray] = {}
        self._render_time: dict[int, float] = {}

    @staticmethod
    def _intersection_over_smaller(a: np.ndarray, b: np.ndarray) -> float:
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
        return inter / max(1.0, min(aa, bb))

    @staticmethod
    def _center_distance_fraction(a: np.ndarray, b: np.ndarray) -> float:
        acx = 0.5 * float(a[0] + a[2])
        acy = 0.5 * float(a[1] + a[3])
        bcx = 0.5 * float(b[0] + b[2])
        bcy = 0.5 * float(b[1] + b[3])
        aw = max(1.0, float(a[2] - a[0]))
        ah = max(1.0, float(a[3] - a[1]))
        return math.hypot(acx - bcx, acy - bcy) / max(1.0, math.hypot(aw, ah))

    def _is_duplicate_of_matched(self, det: Detection, matched_tracks: list[object]) -> bool:
        if super()._is_duplicate_of_matched(det, matched_tracks):
            return True

        # IoU is weak when one detector box is an upper-body/nested version of an
        # existing full-person track. Intersection over the smaller box catches this
        # while appearance + center proximity prevents suppressing a nearby second person.
        for track in self._tracks:
            if track.status == "removed" or track.hits < self.confirm_hits:
                continue
            predicted = self._state_to_xyxy(track.state_vec)
            anchor = self._state_to_xyxy(track.last_measurement)
            best_ios = max(
                self._intersection_over_smaller(predicted, det.bbox),
                self._intersection_over_smaller(anchor, det.bbox),
            )
            if best_ios < self.nested_duplicate_ios:
                continue
            center_frac = min(
                self._center_distance_fraction(predicted, det.bbox),
                self._center_distance_fraction(anchor, det.bbox),
            )
            if center_frac > self.nested_duplicate_center_frac:
                continue
            app = self._appearance_similarity(track.appearance, det.appearance)
            if app is None:
                if best_ios >= 0.94 and center_frac <= 0.18:
                    return True
            elif app >= self.nested_duplicate_app_floor:
                return True
        return False

    def _new_track(self, det: Detection, timestamp: float):
        track = super()._new_track(det, timestamp)
        state = track.last_measurement.copy()
        self._render_state[track.number] = state
        self._render_velocity[track.number] = np.zeros(4, dtype=np.float64)
        self._render_time[track.number] = float(timestamp)
        return track

    @staticmethod
    def _anchored_state(prev: np.ndarray, target: np.ndarray, anchor_alpha: float, size_alpha: float) -> np.ndarray:
        # Smooth cx and bottom edge rather than raw cy. Raising an arm commonly changes
        # detector height/width and bbox center, while the person's lower anchor changes
        # much less. This keeps the visible rectangle attached to the person.
        prev_cx, prev_cy, prev_w, prev_h = (float(v) for v in prev)
        tgt_cx, tgt_cy, tgt_w, tgt_h = (float(v) for v in target)
        prev_bottom = prev_cy + 0.5 * prev_h
        tgt_bottom = tgt_cy + 0.5 * tgt_h

        new_w = prev_w + size_alpha * (tgt_w - prev_w)
        new_h = prev_h + size_alpha * (tgt_h - prev_h)
        new_cx = prev_cx + anchor_alpha * (tgt_cx - prev_cx)
        new_bottom = prev_bottom + anchor_alpha * (tgt_bottom - prev_bottom)
        new_cy = new_bottom - 0.5 * new_h
        return np.array((new_cx, new_cy, new_w, new_h), dtype=np.float64)

    def _update_track(self, track, det: Detection, timestamp: float) -> bool:
        prev_last_detection = float(track.last_detection)
        was_lost = track.status == "lost"
        recovered = super()._update_track(track, det, timestamp)

        target = track.last_measurement.copy()
        prev = self._render_state.get(track.number)
        if prev is None:
            prev = target.copy()

        # One limb/partial-box observation must not suddenly double or halve the visible
        # rectangle. Real approach/retreat is still followed across subsequent 2 Hz hits.
        lower = 1.0 - self.render_max_size_step
        upper = 1.0 + self.render_max_size_step
        target[2] = float(np.clip(target[2], prev[2] * lower, prev[2] * upper))
        target[3] = float(np.clip(target[3], prev[3] * lower, prev[3] * upper))

        size_alpha = self.render_recovery_size_alpha if was_lost else self.render_size_alpha
        new_state = self._anchored_state(
            prev,
            target,
            self.render_anchor_alpha if not was_lost else min(0.90, self.render_anchor_alpha + 0.10),
            size_alpha,
        )
        new_state[2] = min(float(self.frame_width), max(4.0, new_state[2]))
        new_state[3] = min(float(self.frame_height), max(4.0, new_state[3]))

        dt = max(1e-3, float(timestamp) - prev_last_detection)
        inst_velocity = (new_state - prev) / dt
        old_velocity = self._render_velocity.get(
            track.number, np.zeros(4, dtype=np.float64)
        )
        velocity = (
            (1.0 - self.render_velocity_gain) * old_velocity
            + self.render_velocity_gain * inst_velocity
        )
        # Size velocity is deliberately conservative; arm motion should not make the UI
        # box breathe between detector observations.
        velocity[2] = float(np.clip(velocity[2], -0.28 * self.frame_width, 0.28 * self.frame_width))
        velocity[3] = float(np.clip(velocity[3], -0.28 * self.frame_height, 0.28 * self.frame_height))
        velocity[0] = float(np.clip(velocity[0], -1.10 * self.frame_width, 1.10 * self.frame_width))
        velocity[1] = float(np.clip(velocity[1], -1.10 * self.frame_height, 1.10 * self.frame_height))

        self._render_state[track.number] = new_state
        self._render_velocity[track.number] = velocity
        self._render_time[track.number] = float(timestamp)
        return recovered

    def _snapshot(self, track, timestamp: float, *, predicted: bool) -> TrackSnapshot:
        state = self._render_state.get(track.number, track.state_vec).copy()
        velocity = self._render_velocity.get(track.number, track.velocity).copy()

        if predicted:
            dt = max(0.0, min(self.shadow_sec, float(timestamp) - float(track.last_detection)))
            # Predict mainly location; keep size changes heavily damped during a miss.
            state[0] += velocity[0] * dt
            state[1] += velocity[1] * dt
            state[2] += 0.25 * velocity[2] * dt
            state[3] += 0.25 * velocity[3] * dt

        box = self._state_to_xyxy(state)
        x1, y1, x2, y2 = (float(v) for v in box)
        return TrackSnapshot(
            camera_id=self.camera_id,
            track_id=track.track_id,
            state=track.status,
            confirmed=track.hits >= self.confirm_hits,
            predicted=predicted,
            score=float(track.score),
            hits=track.hits,
            age_sec=max(0.0, float(timestamp) - float(track.created_at)),
            since_detection_sec=max(0.0, float(timestamp) - float(track.last_detection)),
            bbox_xyxy=(x1, y1, x2, y2),
            bbox_norm=(
                x1 / self.frame_width,
                y1 / self.frame_height,
                x2 / self.frame_width,
                y2 / self.frame_height,
            ),
            velocity_norm_s=(
                float(velocity[0]) / self.frame_width,
                float(velocity[1]) / self.frame_height,
                float(velocity[2]) / self.frame_width,
                float(velocity[3]) / self.frame_height,
            ),
        )

    def update(self, detections: Iterable[Detection], timestamp: float) -> TrackerUpdate:
        result = super().update(detections, timestamp)
        live = {track.number for track in self._tracks if track.status != "removed"}
        for store in (self._render_state, self._render_velocity, self._render_time):
            for number in list(store):
                if number not in live:
                    store.pop(number, None)
        return result


class MultiCameraBoxStableTracker:
    def __init__(self, camera_ids: Iterable[str], width: int, height: int, **kwargs) -> None:
        self.trackers = {
            cid: BoxStableObservationRecoveryTracker(cid, width, height, **kwargs)
            for cid in camera_ids
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
