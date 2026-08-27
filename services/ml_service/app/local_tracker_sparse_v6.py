from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from services.ml_service.app.local_tracker import (
    Detection,
    TrackSnapshot,
    TrackerUpdate,
    appearance_descriptor,
)
from services.ml_service.app.local_tracker_sparse_v3 import ObservationRecoveryPersonTracker
from services.ml_service.app.local_tracker_sparse_v4 import BoxStableObservationRecoveryTracker


class BodyEnvelopeObservationRecoveryTracker(BoxStableObservationRecoveryTracker):
    """V6: no-teleport sparse association + fast-open/slow-close body envelope.

    Association still uses the raw detector box. Only the published/rendered box gets
    the body-envelope policy, so larger arm/leg-safe rectangles never feed back into
    matching.

    The V4 symmetric size smoother was intentionally conservative, but at ~2 Hz it could
    take several detector hits to grow around a raised arm or a bent body. V6 reacts
    asymmetrically: expansion is fast, contraction is slow, and a small pose-adaptive
    safety margin is added at snapshot time.

    Lost-track recovery also gets a hard geometry gate. The lightweight color hint may
    break close ties, but it cannot resurrect an ID on a distant person.
    """

    def __init__(
        self,
        *args,
        lost_low_jump_diag: float = 1.05,
        lost_high_jump_diag: float = 1.35,
        render_expand_alpha: float = 0.82,
        render_contract_alpha: float = 0.14,
        render_expand_max_step: float = 0.72,
        render_contract_max_step: float = 0.22,
        envelope_pad_x: float = 0.07,
        envelope_pad_top: float = 0.04,
        envelope_pad_bottom: float = 0.03,
        envelope_compact_extra_x: float = 0.06,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lost_low_jump_diag = min(1.50, max(0.55, float(lost_low_jump_diag)))
        self.lost_high_jump_diag = min(2.00, max(self.lost_low_jump_diag, float(lost_high_jump_diag)))
        self.render_expand_alpha = min(0.98, max(0.35, float(render_expand_alpha)))
        self.render_contract_alpha = min(0.45, max(0.03, float(render_contract_alpha)))
        self.render_expand_max_step = min(1.50, max(0.20, float(render_expand_max_step)))
        self.render_contract_max_step = min(0.55, max(0.05, float(render_contract_max_step)))
        self.envelope_pad_x = min(0.20, max(0.0, float(envelope_pad_x)))
        self.envelope_pad_top = min(0.16, max(0.0, float(envelope_pad_top)))
        self.envelope_pad_bottom = min(0.12, max(0.0, float(envelope_pad_bottom)))
        self.envelope_compact_extra_x = min(0.15, max(0.0, float(envelope_compact_extra_x)))

    @staticmethod
    def _box_center_and_diag(box: np.ndarray) -> tuple[float, float, float]:
        x1, y1, x2, y2 = (float(v) for v in box)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2), math.hypot(w, h)

    def _pair_score(self, track, det: Detection, timestamp: float, *, low_stage: bool) -> float:
        score = super()._pair_score(track, det, timestamp, low_stage=low_stage)
        if score < 0.0 or track.status != "lost":
            return score

        # V5 live logs showed a low-score lost track being recovered on a different
        # person after a large center jump. Color similarity is too weak to authorize
        # that. Use the better of the motion prediction and the last real observation,
        # but require one of them to remain spatially plausible.
        predicted = self._state_to_xyxy(track.state_vec)
        anchor = self._state_to_xyxy(track.last_measurement)
        dcx, dcy, _ = self._box_center_and_diag(det.bbox)
        pcx, pcy, pdiag = self._box_center_and_diag(predicted)
        acx, acy, adiag = self._box_center_and_diag(anchor)
        pred_dist = math.hypot(dcx - pcx, dcy - pcy)
        anchor_dist = math.hypot(dcx - acx, dcy - acy)
        dist = min(pred_dist, anchor_dist)
        diag = max(1.0, min(pdiag, adiag))
        overlap = max(self._iou(predicted, det.bbox), self._iou(anchor, det.bbox))

        since_det = max(0.0, float(timestamp) - float(track.last_detection))
        base_limit = self.lost_low_jump_diag if low_stage else self.lost_high_jump_diag
        # Allow a little more travel over a longer detector gap, but cap it tightly.
        limit = min(base_limit * 1.18, base_limit * (1.0 + 0.08 * since_det))
        if overlap < 0.02 and dist > limit * diag:
            return -1.0
        return score

    @staticmethod
    def _asymmetric_size(
        previous: float,
        target: float,
        *,
        expand_alpha: float,
        contract_alpha: float,
        expand_max_step: float,
        contract_max_step: float,
    ) -> float:
        previous = max(4.0, float(previous))
        target = max(4.0, float(target))
        if target >= previous:
            bounded = min(target, previous * (1.0 + expand_max_step))
            alpha = expand_alpha
        else:
            bounded = max(target, previous * (1.0 - contract_max_step))
            alpha = contract_alpha
        return previous + alpha * (bounded - previous)

    def _update_track(self, track, det: Detection, timestamp: float) -> bool:
        prev_last_detection = float(track.last_detection)
        was_lost = track.status == "lost"

        # Intentionally bypass V4's symmetric render-size update. Keep V3 association
        # and track-state update, then apply V6's body-envelope render policy.
        recovered = ObservationRecoveryPersonTracker._update_track(self, track, det, timestamp)

        target = track.last_measurement.copy()
        prev = self._render_state.get(track.number)
        if prev is None:
            prev = target.copy()

        prev_cx, prev_cy, prev_w, prev_h = (float(v) for v in prev)
        tgt_cx, tgt_cy, tgt_w, tgt_h = (float(v) for v in target)

        new_w = self._asymmetric_size(
            prev_w,
            tgt_w,
            expand_alpha=self.render_expand_alpha,
            contract_alpha=self.render_contract_alpha,
            expand_max_step=self.render_expand_max_step,
            contract_max_step=self.render_contract_max_step,
        )
        new_h = self._asymmetric_size(
            prev_h,
            tgt_h,
            expand_alpha=self.render_expand_alpha,
            contract_alpha=self.render_contract_alpha,
            expand_max_step=self.render_expand_max_step,
            contract_max_step=self.render_contract_max_step,
        )

        # Horizontal center + bottom edge remain stable under normal limb motion.
        # On recovery, follow position slightly faster because the strict no-teleport
        # gate above has already rejected implausible matches.
        anchor_alpha = self.render_anchor_alpha
        if was_lost:
            anchor_alpha = min(0.92, anchor_alpha + 0.12)
        prev_bottom = prev_cy + 0.5 * prev_h
        tgt_bottom = tgt_cy + 0.5 * tgt_h
        new_cx = prev_cx + anchor_alpha * (tgt_cx - prev_cx)
        new_bottom = prev_bottom + anchor_alpha * (tgt_bottom - prev_bottom)
        new_cy = new_bottom - 0.5 * new_h

        new_state = np.array((new_cx, new_cy, new_w, new_h), dtype=np.float64)
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
        # The viewer only uses center velocity. Keep size velocity conservative so a
        # single arm-up measurement cannot cause inter-frame box breathing.
        velocity[2] = float(np.clip(velocity[2], -0.22 * self.frame_width, 0.22 * self.frame_width))
        velocity[3] = float(np.clip(velocity[3], -0.22 * self.frame_height, 0.22 * self.frame_height))
        velocity[0] = float(np.clip(velocity[0], -1.10 * self.frame_width, 1.10 * self.frame_width))
        velocity[1] = float(np.clip(velocity[1], -1.10 * self.frame_height, 1.10 * self.frame_height))

        self._render_state[track.number] = new_state
        self._render_velocity[track.number] = velocity
        self._render_time[track.number] = float(timestamp)
        return recovered

    def _body_envelope(self, box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = (float(v) for v in box)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        aspect = w / h

        # Standing people need only a small safety margin. Sitting/bending/arm-out
        # boxes are more compact/wide, so increase horizontal breathing room smoothly.
        compact = min(1.0, max(0.0, (aspect - 0.34) / 0.46))
        pad_x_frac = self.envelope_pad_x + compact * self.envelope_compact_extra_x
        pad_x = max(2.0, pad_x_frac * w)
        pad_top = max(1.0, self.envelope_pad_top * h)
        pad_bottom = max(1.0, self.envelope_pad_bottom * h)

        return np.array(
            (
                max(0.0, x1 - pad_x),
                max(0.0, y1 - pad_top),
                min(float(self.frame_width - 1), x2 + pad_x),
                min(float(self.frame_height - 1), y2 + pad_bottom),
            ),
            dtype=np.float64,
        )

    def _snapshot(self, track, timestamp: float, *, predicted: bool) -> TrackSnapshot:
        state = self._render_state.get(track.number, track.state_vec).copy()
        velocity = self._render_velocity.get(track.number, track.velocity).copy()

        if predicted:
            dt = max(0.0, min(self.shadow_sec, float(timestamp) - float(track.last_detection)))
            # Shadow is internal/optional. Move the envelope but do not predict its size.
            state[0] += velocity[0] * dt
            state[1] += velocity[1] * dt

        box = self._state_to_xyxy(state)
        if not predicted:
            # Hard guarantee for the user's visual contract: every coordinate emitted
            # by the latest person detector must be inside the published box. The
            # smoothed state provides temporal stability, while this union makes a newly
            # raised arm / bent torso / extended leg visible immediately instead of
            # waiting several 2 Hz updates for the size smoother to catch up.
            raw = self._state_to_xyxy(track.last_measurement)
            box = np.array(
                (
                    min(float(box[0]), float(raw[0])),
                    min(float(box[1]), float(raw[1])),
                    max(float(box[2]), float(raw[2])),
                    max(float(box[3]), float(raw[3])),
                ),
                dtype=np.float64,
            )
        box = self._body_envelope(box)
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
                0.0,
                0.0,
            ),
        )


class MultiCameraBodyEnvelopeTracker:
    def __init__(self, camera_ids: Iterable[str], width: int, height: int, **kwargs) -> None:
        self.trackers = {
            cid: BodyEnvelopeObservationRecoveryTracker(cid, width, height, **kwargs)
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
        update = self.trackers[camera_id].update(detections, timestamp)
        # Dataclass instances are intentionally not slotted. Carry capture time to the
        # service so visual metadata can compensate detector/inference latency.
        setattr(update, "captured_ns", int(captured_ns))
        return update
