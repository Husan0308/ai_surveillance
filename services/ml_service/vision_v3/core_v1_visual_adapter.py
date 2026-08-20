from __future__ import annotations

"""Exact presentation behavior from the previously working RF-DETR-S branch.

The failed Vision V3 experiment mixed a very low detector threshold with a new
Byte/Kalman birth policy.  The known-good ``agent/rfdetr-s-core-final`` runtime
used a much simpler bounded motion smoother.  This adapter reproduces that logic
while keeping the Vision V3 ``update/render`` contract.
"""

import math
import threading
from dataclasses import dataclass


def _xyxy_to_state(box) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (
        (x1 + x2) * 0.5,
        (y1 + y2) * 0.5,
        max(2.0, x2 - x1),
        max(2.0, y2 - y1),
    )


def _state_to_xyxy(cx: float, cy: float, w: float, h: float):
    return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class _MotionTrack:
    track_id: int
    cx: float
    cy: float
    w: float
    h: float
    vx: float
    vy: float
    vw: float
    vh: float
    last_det_t: float
    confidence: float


class CoreV1VisualAdapter:
    """Known-good Camera V2 SmoothBoxManager behind the V3 adapter API."""

    def __init__(self, width: int, height: int, cfg: dict) -> None:
        self.width = float(width)
        self.height = float(height)
        self.cfg = dict(cfg or {})
        self.lock = threading.RLock()
        self.tracks: dict[str, dict[int, _MotionTrack]] = {}
        self.next_id = 1

        self.side_margin = float(self.cfg.get("side_margin", 0.08))
        self.top_margin = float(self.cfg.get("top_margin", 0.04))
        self.bottom_margin = float(self.cfg.get("bottom_margin", 0.10))
        self.max_age = float(self.cfg.get("max_age_sec", 1.80))
        self.max_predict = float(self.cfg.get("max_predict_sec", 0.75))

        self.association_score_min = float(
            self.cfg.get("association_score_min", 0.12)
        )
        self.velocity_keep = float(self.cfg.get("velocity_keep", 0.55))
        self.velocity_measure = float(self.cfg.get("velocity_measure", 0.45))
        self.size_velocity_keep = float(
            self.cfg.get("size_velocity_keep", 0.70)
        )
        self.size_velocity_measure = float(
            self.cfg.get("size_velocity_measure", 0.30)
        )
        self.position_alpha = float(self.cfg.get("position_alpha", 0.82))
        self.grow_width_alpha = float(self.cfg.get("grow_width_alpha", 0.75))
        self.shrink_width_alpha = float(
            self.cfg.get("shrink_width_alpha", 0.28)
        )
        self.grow_height_alpha = float(self.cfg.get("grow_height_alpha", 0.78))
        self.shrink_height_alpha = float(
            self.cfg.get("shrink_height_alpha", 0.25)
        )
        self.prediction_damping = float(
            self.cfg.get("prediction_damping", 0.75)
        )
        self.prediction_size_gain = float(
            self.cfg.get("prediction_size_gain", 0.35)
        )

        self._updates = 0
        self._births = 0
        self._matches = 0
        self._pruned = 0

    def _guard_box(self, box):
        x1, y1, x2, y2 = [float(v) for v in box]
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        x1 -= w * self.side_margin
        x2 += w * self.side_margin
        y1 -= h * self.top_margin
        y2 += h * self.bottom_margin
        return (
            max(0.0, x1),
            max(0.0, y1),
            min(self.width - 1.0, x2),
            min(self.height - 1.0, y2),
        )

    def _predict(self, track: _MotionTrack, when: float):
        dt = min(self.max_predict, max(0.0, float(when) - track.last_det_t))
        damping = 1.0 / (1.0 + self.prediction_damping * dt)
        cx = track.cx + track.vx * dt * damping
        cy = track.cy + track.vy * dt * damping
        w = max(8.0, track.w + track.vw * dt * self.prediction_size_gain)
        h = max(16.0, track.h + track.vh * dt * self.prediction_size_gain)
        x1, y1, x2, y2 = _state_to_xyxy(cx, cy, w, h)

        shift_x = 0.0
        shift_y = 0.0
        if x1 < 0.0:
            shift_x = -x1
        elif x2 > self.width - 1.0:
            shift_x = (self.width - 1.0) - x2
        if y1 < 0.0:
            shift_y = -y1
        elif y2 > self.height - 1.0:
            shift_y = (self.height - 1.0) - y2
        return (x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y)

    def update(self, camera_id: str, captured_t: float, detections) -> None:
        guarded = []
        for box, confidence in detections:
            try:
                confidence = float(confidence)
                guarded_box = self._guard_box(box)
            except Exception:
                continue
            if confidence <= 0.0:
                continue
            guarded.append((guarded_box, confidence))

        with self.lock:
            self._updates += 1
            current = self.tracks.setdefault(camera_id, {})
            track_ids = list(current)
            candidates = []

            for track_id in track_ids:
                track = current[track_id]
                predicted = self._predict(track, captured_t)
                pcx, pcy, pw, ph = _xyxy_to_state(predicted)
                for detection_index, (box, _confidence) in enumerate(guarded):
                    dcx, dcy, _dw, _dh = _xyxy_to_state(box)
                    distance = math.hypot(dcx - pcx, dcy - pcy) / max(
                        30.0, math.hypot(pw, ph)
                    )
                    score = _iou(predicted, box) * 0.75 + max(
                        0.0, 1.0 - distance
                    ) * 0.25
                    if score >= self.association_score_min:
                        candidates.append((score, track_id, detection_index))

            candidates.sort(reverse=True)
            used_tracks: set[int] = set()
            used_detections: set[int] = set()
            matches = []
            for _score, track_id, detection_index in candidates:
                if track_id in used_tracks or detection_index in used_detections:
                    continue
                used_tracks.add(track_id)
                used_detections.add(detection_index)
                matches.append((track_id, detection_index))

            for track_id, detection_index in matches:
                self._matches += 1
                track = current[track_id]
                box, confidence = guarded[detection_index]
                mcx, mcy, mw, mh = _xyxy_to_state(box)
                dt = max(0.05, float(captured_t) - track.last_det_t)
                predicted_box = self._predict(track, captured_t)
                pcx, pcy, pw, ph = _xyxy_to_state(predicted_box)

                measured_vx = (mcx - track.cx) / dt
                measured_vy = (mcy - track.cy) / dt
                max_vx = self.width * 0.90
                max_vy = self.height * 0.90
                measured_vx = max(-max_vx, min(max_vx, measured_vx))
                measured_vy = max(-max_vy, min(max_vy, measured_vy))

                track.vx = (
                    track.vx * self.velocity_keep
                    + measured_vx * self.velocity_measure
                )
                track.vy = (
                    track.vy * self.velocity_keep
                    + measured_vy * self.velocity_measure
                )
                track.vw = (
                    track.vw * self.size_velocity_keep
                    + ((mw - track.w) / dt) * self.size_velocity_measure
                )
                track.vh = (
                    track.vh * self.size_velocity_keep
                    + ((mh - track.h) / dt) * self.size_velocity_measure
                )

                width_alpha = (
                    self.grow_width_alpha if mw >= pw else self.shrink_width_alpha
                )
                height_alpha = (
                    self.grow_height_alpha if mh >= ph else self.shrink_height_alpha
                )
                track.cx = pcx + (mcx - pcx) * self.position_alpha
                track.cy = pcy + (mcy - pcy) * self.position_alpha
                track.w = pw + (mw - pw) * width_alpha
                track.h = ph + (mh - ph) * height_alpha
                track.last_det_t = float(captured_t)
                track.confidence = confidence

            for detection_index, (box, confidence) in enumerate(guarded):
                if detection_index in used_detections:
                    continue
                cx, cy, w, h = _xyxy_to_state(box)
                track_id = self.next_id
                self.next_id += 1
                current[track_id] = _MotionTrack(
                    track_id,
                    cx,
                    cy,
                    w,
                    h,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float(captured_t),
                    confidence,
                )
                self._births += 1

            stale = [
                track_id
                for track_id, track in current.items()
                if float(captured_t) - track.last_det_t > self.max_age
            ]
            for track_id in stale:
                current.pop(track_id, None)
                self._pruned += 1

    def render(self, camera_id: str, now: float):
        with self.lock:
            current = self.tracks.get(camera_id, {})
            rows = []
            stale = []
            for track_id, track in current.items():
                age = float(now) - track.last_det_t
                if age > self.max_age:
                    stale.append(track_id)
                    continue
                x1, y1, x2, y2 = self._predict(track, now)
                if x2 > x1 and y2 > y1:
                    rows.append((x1, y1, x2, y2, track.confidence))
            for track_id in stale:
                current.pop(track_id, None)
                self._pruned += 1
            return rows

    def metrics(self, camera_id: str):
        with self.lock:
            return {
                "algorithm": "proven-camera-v2-motion-smoother",
                "active_tracks": len(self.tracks.get(camera_id, {})),
                "updates": self._updates,
                "births": self._births,
                "matches": self._matches,
                "pruned": self._pruned,
                "max_age_ms": self.max_age * 1000.0,
                "max_predict_ms": self.max_predict * 1000.0,
            }
