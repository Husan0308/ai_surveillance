from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PreparedDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class _HistoryDetection:
    box: tuple[float, float, float, float]
    captured_t: float


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _diag(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(20.0, math.hypot(max(1.0, x2 - x1), max(1.0, y2 - y1)))


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0.0 else 0.0


class DetectorLatencyCompensator:
    """Project only stale detector observations to the live injection timestamp.

    External YOLO runs asynchronously from the live DeepStream wall. The bbox was
    measured on the captured frame, while nvtracker receives it roughly 100-220ms
    later. For a walking target, injecting the old coordinates makes the detector
    correction pull NvDCF behind the person. We estimate center velocity from two
    detector observations and project only by the measured result age. NvDCF remains
    the sole temporal tracker.
    """

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = float(frame_width)
        self.frame_height = float(frame_height)
        self.history: dict[str, list[_HistoryDetection]] = {}
        # Logs on the GTX 1050 Ti show detector result age commonly around
        # 160-200ms. Allow the whole observed age to be compensated, with a slight
        # >1 gain because the previous 0.88 gain still visibly trailed walkers.
        self.max_projection_s = 0.28
        self.projection_gain = 1.08
        self.max_speed_x = self.frame_width * 1.25
        self.max_speed_y = self.frame_height * 1.25

    def prepare(
        self,
        cid: str,
        captured_t: float,
        boxes: list[tuple[float, float, float, float, float]],
    ) -> list[PreparedDetection]:
        previous = self.history.get(cid, [])
        current_plain = [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2, _ in boxes]

        candidates: list[tuple[float, int, int]] = []
        for ci, box in enumerate(current_plain):
            ccx, ccy = _center(box)
            for pi, old in enumerate(previous):
                dt = captured_t - old.captured_t
                if dt <= 0.03 or dt > 2.0:
                    continue
                pcx, pcy = _center(old.box)
                dist = math.hypot(ccx - pcx, ccy - pcy) / max(_diag(box), _diag(old.box))
                iou = _iou(box, old.box)
                if iou < 0.02 and dist > 0.72:
                    continue
                score = iou * 0.68 + max(0.0, 1.0 - dist) * 0.32
                candidates.append((score, ci, pi))

        candidates.sort(reverse=True)
        used_current: set[int] = set()
        used_previous: set[int] = set()
        matches: dict[int, int] = {}
        for score, ci, pi in candidates:
            if score < 0.12 or ci in used_current or pi in used_previous:
                continue
            used_current.add(ci)
            used_previous.add(pi)
            matches[ci] = pi

        output: list[PreparedDetection] = []
        for ci, row in enumerate(boxes):
            x1, y1, x2, y2, confidence = [float(v) for v in row]
            vx = 0.0
            vy = 0.0
            pi = matches.get(ci)
            if pi is not None:
                old = previous[pi]
                dt = captured_t - old.captured_t
                if dt > 0.03:
                    ccx, ccy = _center((x1, y1, x2, y2))
                    pcx, pcy = _center(old.box)
                    vx = max(-self.max_speed_x, min(self.max_speed_x, (ccx - pcx) / dt))
                    vy = max(-self.max_speed_y, min(self.max_speed_y, (ccy - pcy) / dt))
            output.append(PreparedDetection(x1, y1, x2, y2, confidence, vx, vy))

        self.history[cid] = [_HistoryDetection(box=b, captured_t=captured_t) for b in current_plain]
        return output

    def project(
        self,
        prepared: list[PreparedDetection],
        captured_t: float,
        now: float,
    ) -> tuple[list[tuple[float, float, float, float, float]], float]:
        age = max(0.0, now - captured_t)
        dt = min(self.max_projection_s, age)
        output: list[tuple[float, float, float, float, float]] = []
        for det in prepared:
            dx = det.vx * dt * self.projection_gain
            dy = det.vy * dt * self.projection_gain
            width = max(2.0, det.x2 - det.x1)
            height = max(2.0, det.y2 - det.y1)
            left = det.x1 + dx
            top = det.y1 + dy
            right = det.x2 + dx
            bottom = det.y2 + dy

            if left < 0.0:
                right -= left
                left = 0.0
            if right > self.frame_width - 1.0:
                shift = right - (self.frame_width - 1.0)
                left -= shift
                right -= shift
            if top < 0.0:
                bottom -= top
                top = 0.0
            if bottom > self.frame_height - 1.0:
                shift = bottom - (self.frame_height - 1.0)
                top -= shift
                bottom -= shift

            left = max(0.0, min(self.frame_width - 2.0, left))
            top = max(0.0, min(self.frame_height - 2.0, top))
            right = min(self.frame_width - 1.0, max(left + min(2.0, width), right))
            bottom = min(self.frame_height - 1.0, max(top + min(2.0, height), bottom))
            output.append((left, top, right, bottom, det.confidence))
        return output, age * 1000.0
