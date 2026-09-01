from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .bbox_overlay_ipc_v1 import predict_bbox_norm


@dataclass
class SingleTargetLockConfigV1:
    enabled: bool = True
    acquire_updates: int = 2
    min_hits: int = 2
    min_confidence: float = 0.22
    min_area_norm: float = 0.008
    max_since_detection_sec: float = 0.35
    hold_sec: float = 1.10
    release_sec: float = 1.60
    handoff_window_sec: float = 1.35
    handoff_updates: int = 2
    handoff_min_iou: float = 0.16
    handoff_max_center_distance: float = 0.14
    handoff_min_area_ratio: float = 0.45
    handoff_max_area_ratio: float = 2.20

    def __post_init__(self) -> None:
        self.acquire_updates = max(1, int(self.acquire_updates))
        self.min_hits = max(1, int(self.min_hits))
        self.min_confidence = min(1.0, max(0.0, float(self.min_confidence)))
        self.min_area_norm = min(1.0, max(0.0, float(self.min_area_norm)))
        self.max_since_detection_sec = max(0.0, float(self.max_since_detection_sec))
        self.hold_sec = max(0.0, float(self.hold_sec))
        self.release_sec = max(self.hold_sec, float(self.release_sec))
        self.handoff_window_sec = max(0.0, float(self.handoff_window_sec))
        self.handoff_updates = max(1, int(self.handoff_updates))
        self.handoff_min_iou = min(1.0, max(0.0, float(self.handoff_min_iou)))
        self.handoff_max_center_distance = max(0.0, float(self.handoff_max_center_distance))
        self.handoff_min_area_ratio = max(0.01, float(self.handoff_min_area_ratio))
        self.handoff_max_area_ratio = max(self.handoff_min_area_ratio, float(self.handoff_max_area_ratio))


@dataclass
class _CameraLockState:
    locked_id: str | None = None
    candidate_id: str | None = None
    candidate_streak: int = 0
    successor_id: str | None = None
    successor_streak: int = 0
    last_real_ns: int = 0
    last_real_track: dict[str, Any] | None = None
    acquisitions: int = 0
    handoffs: int = 0
    releases: int = 0
    hold_outputs: int = 0
    suppressed_total: int = 0
    input_max: int = 0
    output_max: int = 0
    violations: int = 0


def _bbox(track: dict[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        values = tuple(float(v) for v in track["bbox_norm"])
    except (KeyError, TypeError, ValueError):
        return None
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    return inter / max(1e-9, _area(a) + _area(b) - inter)


def _center_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    acx = 0.5 * (a[0] + a[2])
    acy = 0.5 * (a[1] + a[3])
    bcx = 0.5 * (b[0] + b[2])
    bcy = 0.5 * (b[1] + b[3])
    return math.hypot(acx - bcx, acy - bcy)


class SingleTargetBboxLockV1:
    """Single-person display gate for the bbox-only live acceptance stage.

    This class does NOT mutate the detector or tracker. It selects at most one
    already-confirmed local track per camera for display. Once acquired, all
    competing local tracks are suppressed until the lock is explicitly released
    or a spatially consistent successor is confirmed across repeated updates.

    It is intentionally a single-person validation mode, not the final multi-person
    production policy.
    """

    def __init__(self, config: SingleTargetLockConfigV1 | None = None) -> None:
        self.config = config or SingleTargetLockConfigV1()
        self._states: dict[str, _CameraLockState] = {}

    def _state(self, camera_id: str) -> _CameraLockState:
        return self._states.setdefault(str(camera_id), _CameraLockState())

    def _eligible(self, track: dict[str, Any]) -> bool:
        if str(track.get("state", "")) != "tracked":
            return False
        if bool(track.get("predicted", False)):
            return False
        try:
            hits = int(track.get("hits", 0))
            score = float(track.get("confidence", 0.0))
            since_det = float(track.get("since_detection_sec", 999.0))
        except (TypeError, ValueError):
            return False
        box = _bbox(track)
        if box is None:
            return False
        return (
            hits >= self.config.min_hits
            and math.isfinite(score)
            and score >= self.config.min_confidence
            and math.isfinite(since_det)
            and since_det <= self.config.max_since_detection_sec
            and _area(box) >= self.config.min_area_norm
        )

    @staticmethod
    def _rank(track: dict[str, Any]) -> tuple[int, float, float, str]:
        box = _bbox(track)
        return (
            int(track.get("hits", 0)),
            float(track.get("confidence", 0.0)),
            _area(box) if box is not None else 0.0,
            str(track.get("track_id", "")),
        )

    def _best_acquisition_candidate(self, tracks: list[dict[str, Any]]) -> dict[str, Any] | None:
        eligible = [track for track in tracks if self._eligible(track)]
        if not eligible:
            return None
        return max(eligible, key=self._rank)

    def _record_real(self, state: _CameraLockState, track: dict[str, Any], captured_ns: int) -> None:
        state.last_real_ns = int(captured_ns)
        state.last_real_track = dict(track)

    def _acquire(self, state: _CameraLockState, track: dict[str, Any], captured_ns: int) -> list[dict[str, Any]]:
        state.locked_id = str(track["track_id"])
        state.candidate_id = None
        state.candidate_streak = 0
        state.successor_id = None
        state.successor_streak = 0
        state.acquisitions += 1
        self._record_real(state, track, captured_ns)
        return [track]

    def _held_track(self, state: _CameraLockState, captured_ns: int) -> dict[str, Any] | None:
        if state.last_real_track is None or state.last_real_ns <= 0:
            return None
        dt = max(0.0, (int(captured_ns) - state.last_real_ns) / 1_000_000_000.0)
        if dt > self.config.hold_sec:
            return None
        held = dict(state.last_real_track)
        held["bbox_norm"] = list(
            predict_bbox_norm(
                held["bbox_norm"],
                held.get("velocity_norm_s", (0.0, 0.0, 0.0, 0.0)),
                dt,
            )
        )
        held["predicted"] = True
        held["state"] = "lost"
        held["since_detection_sec"] = dt
        state.hold_outputs += 1
        return held

    def _successor_candidate(
        self,
        state: _CameraLockState,
        tracks: list[dict[str, Any]],
        captured_ns: int,
    ) -> dict[str, Any] | None:
        if state.last_real_track is None or state.last_real_ns <= 0:
            return None
        dt = max(0.0, (int(captured_ns) - state.last_real_ns) / 1_000_000_000.0)
        if dt > self.config.handoff_window_sec:
            return None
        old_box = predict_bbox_norm(
            state.last_real_track["bbox_norm"],
            state.last_real_track.get("velocity_norm_s", (0.0, 0.0, 0.0, 0.0)),
            dt,
        )
        old_area = max(1e-9, _area(old_box))
        candidates: list[tuple[float, float, int, float, dict[str, Any]]] = []
        for track in tracks:
            if str(track.get("track_id", "")) == state.locked_id or not self._eligible(track):
                continue
            box = _bbox(track)
            if box is None:
                continue
            area_ratio = _area(box) / old_area
            if area_ratio < self.config.handoff_min_area_ratio or area_ratio > self.config.handoff_max_area_ratio:
                continue
            overlap = _iou(old_box, box)
            center = _center_distance(old_box, box)
            if overlap < self.config.handoff_min_iou and center > self.config.handoff_max_center_distance:
                continue
            candidates.append(
                (
                    overlap,
                    -center,
                    int(track.get("hits", 0)),
                    float(track.get("confidence", 0.0)),
                    track,
                )
            )
        if not candidates:
            return None
        candidates.sort(key=lambda row: row[:-1], reverse=True)
        return candidates[0][-1]

    def _finalize(self, state: _CameraLockState, raw_count: int, output: list[dict[str, Any]]) -> list[dict[str, Any]]:
        state.input_max = max(state.input_max, int(raw_count))
        state.output_max = max(state.output_max, len(output))
        if len(output) > 1:
            state.violations += 1
            output = output[:1]
        state.suppressed_total += max(0, int(raw_count) - min(1, len(output)))
        return output

    def select(self, camera_id: str, tracks: Iterable[dict[str, Any]], captured_ns: int) -> list[dict[str, Any]]:
        rows = [dict(track) for track in tracks]
        state = self._state(camera_id)
        if not self.config.enabled:
            return self._finalize(state, len(rows), rows)

        if state.locked_id is not None:
            locked = next((track for track in rows if str(track.get("track_id", "")) == state.locked_id), None)
            if locked is not None:
                fresh = self._eligible(locked)
                if fresh:
                    self._record_real(state, locked, captured_ns)
                    state.successor_id = None
                    state.successor_streak = 0
                    return self._finalize(state, len(rows), [locked])

                if state.last_real_ns > 0:
                    age = max(0.0, (int(captured_ns) - state.last_real_ns) / 1_000_000_000.0)
                    if age <= self.config.hold_sec:
                        # Prefer the tracker's own prediction while it remains within
                        # the bounded hold window; it is the same locked target.
                        return self._finalize(state, len(rows), [locked])

            successor = self._successor_candidate(state, rows, captured_ns)
            if successor is not None:
                successor_id = str(successor["track_id"])
                if state.successor_id == successor_id:
                    state.successor_streak += 1
                else:
                    state.successor_id = successor_id
                    state.successor_streak = 1
                if state.successor_streak >= self.config.handoff_updates:
                    state.locked_id = successor_id
                    state.handoffs += 1
                    state.successor_id = None
                    state.successor_streak = 0
                    self._record_real(state, successor, captured_ns)
                    return self._finalize(state, len(rows), [successor])
            else:
                state.successor_id = None
                state.successor_streak = 0

            held = self._held_track(state, captured_ns)
            if held is not None:
                return self._finalize(state, len(rows), [held])

            age = (
                max(0.0, (int(captured_ns) - state.last_real_ns) / 1_000_000_000.0)
                if state.last_real_ns > 0
                else self.config.release_sec + 1.0
            )
            if age <= self.config.release_sec:
                # Deliberately show no replacement box while the old target is still
                # in the lock-release grace period. This prevents a nearby false
                # positive from stealing the display immediately.
                return self._finalize(state, len(rows), [])

            state.locked_id = None
            state.last_real_ns = 0
            state.last_real_track = None
            state.successor_id = None
            state.successor_streak = 0
            state.candidate_id = None
            state.candidate_streak = 0
            state.releases += 1

        candidate = self._best_acquisition_candidate(rows)
        if candidate is None:
            state.candidate_id = None
            state.candidate_streak = 0
            return self._finalize(state, len(rows), [])

        candidate_id = str(candidate["track_id"])
        if state.candidate_id == candidate_id:
            state.candidate_streak += 1
        else:
            state.candidate_id = candidate_id
            state.candidate_streak = 1

        if state.candidate_streak < self.config.acquire_updates:
            return self._finalize(state, len(rows), [])
        return self._finalize(state, len(rows), self._acquire(state, candidate, captured_ns))

    def stats(self, camera_id: str) -> dict[str, Any]:
        state = self._state(camera_id)
        return {
            "enabled": int(self.config.enabled),
            "locked": state.locked_id or "-",
            "candidate": state.candidate_id or "-",
            "acquired": state.acquisitions,
            "handoff": state.handoffs,
            "released": state.releases,
            "hold_outputs": state.hold_outputs,
            "suppressed": state.suppressed_total,
            "input_max": state.input_max,
            "output_max": state.output_max,
            "violations": state.violations,
        }
