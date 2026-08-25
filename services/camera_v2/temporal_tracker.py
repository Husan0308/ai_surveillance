from __future__ import annotations

"""Low-cost temporal person tracking for the Pascal RF-DETR path.

RF-DETR is intentionally sparse on the GTX 1050 Ti. This module keeps a stable
per-camera person state between detector corrections without retaining any
GStreamer/NVMM buffers. The state is centered on a persistent body anchor and
uses bounded constant-velocity prediction, posture-tolerant association and
asymmetric box-size smoothing.

This is deliberately not a long-term identity/ReID tracker. It only owns the
short temporal continuity needed for a box to stay attached to the same person
between RF-DETR observations.
"""

import math
import os
import threading
from dataclasses import dataclass


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


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


def _size_similarity(pw: float, ph: float, dw: float, dh: float) -> float:
    # Log-ratio stays well behaved when a standing person sits/bends and the
    # detector rectangle changes height abruptly.
    rw = max(1e-4, dw / max(2.0, pw))
    rh = max(1e-4, dh / max(2.0, ph))
    penalty = abs(math.log(rw)) + abs(math.log(rh))
    return math.exp(-0.55 * penalty)


@dataclass
class AnchorTrack:
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
    hits: int = 1
    misses: int = 0
    confirmed: bool = False


class AnchoredPersonTracker:
    """Persistent short-term person tracks with a stable center anchor.

    Public interface intentionally matches detection.SmoothBoxManager:
      update(camera_id, captured_time, detections)
      render(camera_id, now) -> [(x1, y1, x2, y2, confidence), ...]

    A new candidate is probationary. It is never rendered until a second
    spatially-consistent RF-DETR observation confirms it, except for an explicitly
    high-confidence observation above ``CAMERA_V2_TRACK_INSTANT_CONF``. This
    mirrors the useful probation behavior from the earlier stable tracker and
    prevents one-frame furniture/monitor false positives becoming long ghosts.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = float(width)
        self.height = float(height)
        self.lock = threading.RLock()
        self.tracks: dict[str, dict[int, AnchorTrack]] = {}
        self.next_id = 1

        self.side_margin = float(os.environ.get("CAMERA_V2_BOX_SIDE_MARGIN", "0.08"))
        self.top_margin = float(os.environ.get("CAMERA_V2_BOX_TOP_MARGIN", "0.04"))
        self.bottom_margin = float(os.environ.get("CAMERA_V2_BOX_BOTTOM_MARGIN", "0.10"))

        # Keep invisible probation candidates long enough to see the next sparse
        # detector observation. Confirmed display state is intentionally much
        # shorter than the previous 4.8 s flow experiment so a rejected/lost
        # person cannot leave a long green ghost in the room.
        self.max_age = float(os.environ.get("CAMERA_V2_TRACK_HOLD_SEC", "2.80"))
        self.tentative_age = float(
            os.environ.get("CAMERA_V2_TRACK_TENTATIVE_SEC", "2.60")
        )
        self.predict_horizon = float(
            os.environ.get("CAMERA_V2_TRACK_PREDICT_SEC", "0.70")
        )
        self.velocity_drag = float(
            os.environ.get("CAMERA_V2_TRACK_VELOCITY_DRAG", "1.35")
        )

        # The old value 0.48 allowed a single plausible-looking false positive to
        # become a rendered track immediately. Default to strict birth probation;
        # only exceptionally strong RF-DETR evidence may skip the second hit.
        self.instant_confirm_conf = float(
            os.environ.get("CAMERA_V2_TRACK_INSTANT_CONF", "0.85")
        )
        self.confirm_hits = max(
            2, int(os.environ.get("CAMERA_V2_TRACK_CONFIRM_HITS", "2"))
        )
        self.match_floor = float(
            os.environ.get("CAMERA_V2_TRACK_MATCH_FLOOR", "0.16")
        )
        self.birth_suppress_score = float(
            os.environ.get("CAMERA_V2_TRACK_BIRTH_SUPPRESS_SCORE", "0.62")
        )

    def _guard_box(self, box):
        x1, y1, x2, y2 = [float(v) for v in box]
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        x1 -= w * self.side_margin
        x2 += w * self.side_margin
        y1 -= h * self.top_margin
        y2 += h * self.bottom_margin
        return (
            _clamp(x1, 0.0, self.width - 2.0),
            _clamp(y1, 0.0, self.height - 2.0),
            _clamp(x2, 1.0, self.width - 1.0),
            _clamp(y2, 1.0, self.height - 1.0),
        )

    def _bounded_motion_dt(self, age: float) -> float:
        """Integrate a decaying velocity without unbounded forward drift."""
        age = _clamp(age, 0.0, self.predict_horizon)
        drag = max(0.0, self.velocity_drag)
        if drag <= 1e-6:
            return age
        return (1.0 - math.exp(-drag * age)) / drag

    def _predict_state(self, track: AnchorTrack, when: float):
        age = max(0.0, float(when) - track.last_det_t)
        motion_dt = self._bounded_motion_dt(age)
        cx = track.cx + track.vx * motion_dt
        cy = track.cy + track.vy * motion_dt

        # Width/height velocity is intentionally much weaker than position. A
        # bent or seated person may suddenly get a shorter detector box; letting
        # that size velocity run between detections causes visible breathing.
        size_dt = min(motion_dt, 0.35)
        w = max(12.0, track.w + track.vw * size_dt * 0.22)
        h = max(24.0, track.h + track.vh * size_dt * 0.22)
        return cx, cy, w, h

    def _predict_box(self, track: AnchorTrack, when: float):
        cx, cy, w, h = self._predict_state(track, when)
        x1, y1, x2, y2 = _state_to_xyxy(cx, cy, w, h)

        # Translate the whole rectangle back into frame bounds instead of
        # clipping one edge; this preserves the body anchor near borders.
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

    def _association_score(self, track: AnchorTrack, box, when: float) -> float | None:
        pred = self._predict_box(track, when)
        pcx, pcy, pw, ph = _xyxy_to_state(pred)
        dcx, dcy, dw, dh = _xyxy_to_state(box)

        diag = max(60.0, 0.5 * (math.hypot(pw, ph) + math.hypot(dw, dh)))
        center_dist = math.hypot(dcx - pcx, dcy - pcy) / diag

        # Bottom-center is useful when standing -> bending/sitting changes the
        # upper body rectangle much more than the person's floor-side location.
        p_bottom_y = pcy + ph * 0.5
        d_bottom_y = dcy + dh * 0.5
        bottom_dist = math.hypot(dcx - pcx, d_bottom_y - p_bottom_y) / diag

        overlap = _iou(pred, box)
        size_sim = _size_similarity(pw, ph, dw, dh)

        # Wide gate: low IoU is allowed if the body/foot anchors still agree.
        # This keeps the same track through sitting, bending and partial
        # chair/desk occlusion.
        if overlap < 0.01 and center_dist > 1.55 and bottom_dist > 1.25:
            return None
        if size_sim < 0.12 and overlap < 0.08:
            return None

        center_score = max(0.0, 1.0 - center_dist / 1.55)
        bottom_score = max(0.0, 1.0 - bottom_dist / 1.25)
        return (
            overlap * 0.43
            + center_score * 0.34
            + bottom_score * 0.15
            + size_sim * 0.08
        )

    def _correct_track(
        self, track: AnchorTrack, box, conf: float, captured_t: float
    ) -> None:
        mcx, mcy, mw, mh = _xyxy_to_state(box)
        pcx, pcy, pw, ph = self._predict_state(track, captured_t)
        dt = max(0.08, captured_t - track.last_det_t)

        # Velocity is measured from the previous corrected anchor, then heavily
        # damped. If the detector residual opposes the current velocity, kill the
        # stale momentum quickly so the box does not remain in front/behind.
        raw_vx = (mcx - track.cx) / dt
        raw_vy = (mcy - track.cy) / dt
        max_vx = self.width * 0.62
        max_vy = self.height * 0.75
        raw_vx = _clamp(raw_vx, -max_vx, max_vx)
        raw_vy = _clamp(raw_vy, -max_vy, max_vy)

        residual_x = mcx - pcx
        residual_y = mcy - pcy
        if track.vx * residual_x < 0.0:
            track.vx *= 0.38
        if track.vy * residual_y < 0.0:
            track.vy *= 0.38

        # Small detector jitter should not create motion. Larger real movement
        # receives a faster velocity correction.
        jitter_radius = max(5.0, 0.025 * math.hypot(pw, ph))
        movement = math.hypot(mcx - track.cx, mcy - track.cy)
        velocity_gain = 0.18 if movement < jitter_radius else 0.34
        track.vx = track.vx * (1.0 - velocity_gain) + raw_vx * velocity_gain
        track.vy = track.vy * (1.0 - velocity_gain) + raw_vy * velocity_gain

        if movement < jitter_radius:
            track.vx *= 0.72
            track.vy *= 0.72

        # Prediction supplies smooth intermediate motion; detector measurements
        # remain truth and pull the anchor back when prediction is wrong.
        pos_gain = 0.80 if conf >= 0.35 else 0.72
        track.cx = pcx + residual_x * pos_gain
        track.cy = pcy + residual_y * pos_gain

        raw_vw = (mw - track.w) / dt
        raw_vh = (mh - track.h) / dt
        track.vw = track.vw * 0.78 + raw_vw * 0.22
        track.vh = track.vh * 0.80 + raw_vh * 0.20

        # Expand quickly so arms/legs are not cut off. Shrink slowly so bending,
        # turning sideways or chair occlusion does not make the rectangle pulse
        # or collapse around only the visible torso.
        w_gain = 0.68 if mw >= pw else 0.20
        h_gain = 0.72 if mh >= ph else 0.16
        track.w = pw + (mw - pw) * w_gain
        track.h = ph + (mh - ph) * h_gain

        track.last_det_t = captured_t
        track.confidence = float(conf)
        track.hits += 1
        track.misses = 0
        if conf >= self.instant_confirm_conf or track.hits >= self.confirm_hits:
            track.confirmed = True

    def update(
        self,
        cid: str,
        captured_t: float,
        detections: list[tuple[tuple[float, float, float, float], float]],
    ) -> None:
        guarded = [(self._guard_box(box), float(conf)) for box, conf in detections]
        with self.lock:
            current = self.tracks.setdefault(cid, {})
            # Confirmed tracks own continuity. A new/tentative track must not
            # steal a detection from an already confirmed person merely because
            # its instantaneous geometry score is slightly higher.
            candidates: list[tuple[int, float, int, int]] = []

            for tid, track in current.items():
                for di, (box, _conf) in enumerate(guarded):
                    score = self._association_score(track, box, captured_t)
                    if score is not None and score >= self.match_floor:
                        confirmed_priority = 1 if track.confirmed else 0
                        candidates.append((confirmed_priority, score, tid, di))

            candidates.sort(reverse=True)
            used_tracks: set[int] = set()
            used_dets: set[int] = set()
            matches: list[tuple[int, int]] = []
            for _confirmed_priority, _score, tid, di in candidates:
                if tid in used_tracks or di in used_dets:
                    continue
                used_tracks.add(tid)
                used_dets.add(di)
                matches.append((tid, di))

            for tid, di in matches:
                box, conf = guarded[di]
                self._correct_track(current[tid], box, conf, captured_t)

            for tid, track in current.items():
                if tid not in used_tracks:
                    track.misses += 1

                    # Probation confirmation must be consecutive evidence.
                    # Sporadic duplicate RF-DETR boxes must not accumulate hits
                    # over time until they accidentally become a real track.
                    if not track.confirmed:
                        track.hits = 0

            # Unmatched detections normally enter probation. However, RF-DETR
            # can emit two slightly different boxes for the same person. If an
            # unmatched box still strongly agrees with an existing confirmed
            # track, treat it as a duplicate observation instead of birthing a
            # second identity for the same body.
            for di, (box, conf) in enumerate(guarded):
                if di in used_dets:
                    continue

                duplicate_of_confirmed = False
                for track in current.values():
                    if not track.confirmed:
                        continue
                    score = self._association_score(track, box, captured_t)
                    if (
                        score is not None
                        and score >= self.birth_suppress_score
                    ):
                        duplicate_of_confirmed = True
                        break

                if duplicate_of_confirmed:
                    continue

                cx, cy, w, h = _xyxy_to_state(box)
                tid = self.next_id
                self.next_id += 1
                current[tid] = AnchorTrack(
                    track_id=tid,
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    vx=0.0,
                    vy=0.0,
                    vw=0.0,
                    vh=0.0,
                    last_det_t=float(captured_t),
                    confidence=conf,
                    confirmed=conf >= self.instant_confirm_conf,
                )

            stale: list[int] = []
            for tid, track in current.items():
                age = captured_t - track.last_det_t
                limit = self.max_age if track.confirmed else self.tentative_age
                if age > limit:
                    stale.append(tid)
            for tid in stale:
                current.pop(tid, None)

    def render(
        self, cid: str, now: float
    ) -> list[tuple[float, float, float, float, float]]:
        with self.lock:
            current = self.tracks.get(cid, {})
            rows: list[tuple[float, float, float, float, float]] = []
            stale: list[int] = []
            for tid, track in current.items():
                age = max(0.0, float(now) - track.last_det_t)
                limit = self.max_age if track.confirmed else self.tentative_age
                if age > limit:
                    stale.append(tid)
                    continue

                # Probation candidates are NEVER shown. This is the critical
                # difference from the previous implementation, which displayed a
                # single >=0.30/0.48 false positive and then flow kept it alive.
                if not track.confirmed:
                    continue

                x1, y1, x2, y2 = self._predict_box(track, now)
                if x2 <= x1 or y2 <= y1:
                    continue

                decay_age = max(0.0, age - 0.65)
                shown_conf = max(
                    0.05, track.confidence * math.exp(-0.10 * decay_age)
                )
                rows.append((x1, y1, x2, y2, shown_conf))

            for tid in stale:
                current.pop(tid, None)
            return rows

    def anchors(self, cid: str, now: float):
        """Return confirmed persistent center anchors for flow/OSD-dot stages."""
        with self.lock:
            current = self.tracks.get(cid, {})
            output = []
            for tid, track in current.items():
                age = max(0.0, float(now) - track.last_det_t)
                limit = self.max_age if track.confirmed else self.tentative_age
                if age > limit or not track.confirmed:
                    continue
                cx, cy, _w, _h = self._predict_state(track, now)
                output.append(
                    {
                        "track_id": int(tid),
                        "cx": _clamp(cx, 0.0, self.width - 1.0),
                        "cy": _clamp(cy, 0.0, self.height - 1.0),
                        "age": age,
                        "confirmed": True,
                        "confidence": float(track.confidence),
                    }
                )
            return output
