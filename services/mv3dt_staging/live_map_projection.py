"""Fail-closed live map projection for the Entrance native-coordinate candidate.

The module consumes *tracker* observations only after they have been joined to
their exact cached source frame by source ID, input frame index, and source
PTS. Camera-local tracker IDs remain namespaced by camera. Map-pixel locations
are experimental and are never clipped into the layout.
"""

from __future__ import annotations

from collections import defaultdict, deque
import math
import time
from typing import Any

import numpy as np


CAMERA_TO_SOURCE_ID = {"CAM-02": 0, "CAM-05": 1}
CAMERA_TO_MATRIX_KEY = {"CAM-02": "cam02", "CAM-05": "cam05"}
CAMERA_COLORS_BGR = {"CAM-02": (255, 190, 60), "CAM-05": (60, 205, 255)}


def choose_marker_label_position(
    anchor_xy: tuple[float, float],
    label_wh: tuple[int, int],
    bounds_xywh: tuple[int, int, int, int],
    occupied_xywh: list[tuple[int, int, int, int]],
    *,
    gap: int = 3,
) -> tuple[int, int] | None:
    """Place a map label in-bounds without covering markers or prior labels.

    Returns the label's top-left position in the same display-pixel space as
    ``bounds_xywh``. Geometry/marker coordinates are never changed. ``None``
    means the current map pane is too crowded to show every label safely.
    """
    try:
        anchor_x, anchor_y = (float(value) for value in anchor_xy)
        label_width, label_height = (int(value) for value in label_wh)
        bounds_x, bounds_y, bounds_width, bounds_height = (
            int(value) for value in bounds_xywh
        )
        gap = max(0, int(gap))
    except (TypeError, ValueError):
        return None
    if (not math.isfinite(anchor_x) or not math.isfinite(anchor_y)
            or label_width <= 0 or label_height <= 0
            or bounds_width < label_width or bounds_height < label_height):
        return None

    min_x, min_y = bounds_x, bounds_y
    max_x = bounds_x + bounds_width - label_width
    max_y = bounds_y + bounds_height - label_height
    offsets = (
        (10, -label_height // 2),
        (-label_width - 10, -label_height // 2),
        (10, -label_height - 8),
        (-label_width - 10, -label_height - 8),
        (10, 8),
        (-label_width - 10, 8),
        (-label_width // 2, -label_height - 10),
        (-label_width // 2, 10),
    )
    candidates = {
        (round(anchor_x + dx), round(anchor_y + dy)) for dx, dy in offsets
    }

    # Fallback to a bounded display grid, ordered by distance to the marker.
    # The grid is small (the map pane is a few hundred pixels wide) and keeps
    # dense labels readable without affecting the high-rate projection path.
    step_x = max(8, label_width // 3)
    step_y = max(8, label_height + gap)
    grid = [
        (x, y)
        for y in range(min_y, max_y + 1, step_y)
        for x in range(min_x, max_x + 1, step_x)
    ]
    candidates.update(grid)

    def distance(position: tuple[int, int]) -> float:
        x, y = position
        return math.dist((x + label_width / 2, y + label_height / 2),
                         (anchor_x, anchor_y))

    for x, y in sorted(candidates, key=distance):
        if x < min_x or y < min_y or x > max_x or y > max_y:
            continue
        collides = any(
            x < ox + ow + gap and x + label_width + gap > ox
            and y < oy + oh + gap and y + label_height + gap > oy
            for ox, oy, ow, oh in occupied_xywh
        )
        if not collides:
            return int(x), int(y)
    return None


def project_anchor_unbounded(
    matrix: np.ndarray, box: list[float] | tuple[float, ...]
) -> tuple[float, float] | None:
    """Project a 1920x1080 inference box's bottom-centre without clipping."""
    if len(box) != 4:
        return None
    try:
        left, top, width, height = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if (not all(math.isfinite(value) for value in (left, top, width, height))
            or width <= 0 or height <= 0):
        return None
    point = np.asarray([left + width / 2.0, top + height, 1.0], dtype=np.float64)
    out = np.asarray(matrix, dtype=np.float64) @ point
    if out.shape != (3,) or not np.all(np.isfinite(out)) or abs(float(out[2])) < 1e-12:
        return None
    x, y = (out[:2] / out[2]).tolist()
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return float(x), float(y)


class FrameJoinError(ValueError):
    """Tracker metadata cannot be proven to describe the supplied source frame."""


class LiveNativeMapProjection:
    """Project exact-frame NvDCF observations into an experimental 2D map."""

    def __init__(
        self,
        matrices: dict[str, np.ndarray],
        *,
        layout_width: int,
        layout_height: int,
        person_class_id: int | None = None,
        trail_points: int = 16,
        marker_max_age_seconds: float = 0.75,
        jump_threshold_layout_px: float = 75.0,
        jump_max_interval_seconds: float = 0.25,
    ) -> None:
        if set(matrices) != {"cam02", "cam05"}:
            raise ValueError("both CAM-02 and CAM-05 candidate homographies are required")
        if layout_width <= 0 or layout_height <= 0 or trail_points < 2:
            raise ValueError("invalid layout dimensions or trail bound")
        for key, value in matrices.items():
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{key} homography must be a finite 3x3 matrix")
        if marker_max_age_seconds <= 0:
            raise ValueError("marker expiry must be positive")

        self.matrices = {key: np.asarray(value, dtype=np.float64).copy()
                         for key, value in matrices.items()}
        self.layout_width = int(layout_width)
        self.layout_height = int(layout_height)
        self.person_class_id = person_class_id
        self.trail_points = int(trail_points)
        self.marker_max_age_ns = int(marker_max_age_seconds * 1e9)
        self.jump_threshold_layout_px = float(jump_threshold_layout_px)
        self.jump_max_interval_ns = int(jump_max_interval_seconds * 1e9)
        self._last_frame: dict[str, int] = {}
        self._last_pts: dict[str, int] = {}
        self._last_projected: dict[tuple[str, str], tuple[int, int, tuple[float, float]]] = {}
        self._trails: dict[tuple[str, str], deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.trail_points)
        )
        self._states: dict[str, dict[str, Any]] = {}
        self.counters: dict[str, dict[str, int]] = {
            camera: {"frames": 0, "projected": 0, "out_of_bounds": 0,
                     "invalid": 0, "duplicate_local_ids": 0,
                     "large_jump_candidates": 0, "stale_snapshots": 0}
            for camera in CAMERA_TO_SOURCE_ID
        }

    def update(
        self,
        camera_id: str,
        record: dict[str, Any],
        *,
        cached_source_id: int,
        cached_frame_index: int,
        cached_source_pts_ns: int,
        now_mono_ns: int | None = None,
    ) -> dict[str, Any]:
        """Validate the sidecar join and update this camera's marker snapshot.

        No nearest-frame fallback is allowed. If a join fails, callers should
        stop the preview rather than display a plausible but incorrect map.
        """
        if camera_id not in CAMERA_TO_SOURCE_ID:
            raise FrameJoinError(f"unexpected camera mapping: {camera_id}")
        now = time.monotonic_ns() if now_mono_ns is None else int(now_mono_ns)
        expected_sid = CAMERA_TO_SOURCE_ID[camera_id]
        try:
            source_id = int(record["source_id"])
            frame_index = int(record["input_frame_index"])
            source_pts_ns = int(record["source_pts_ns"])
            cached_source_id = int(cached_source_id)
            cached_frame_index = int(cached_frame_index)
            cached_source_pts_ns = int(cached_source_pts_ns)
        except (KeyError, TypeError, ValueError) as exc:
            raise FrameJoinError(f"{camera_id}: missing/non-integer source frame metadata") from exc
        if source_id != expected_sid or cached_source_id != expected_sid:
            raise FrameJoinError(f"{camera_id}: source ID does not match the fixed camera mapping")
        if frame_index != cached_frame_index:
            raise FrameJoinError(f"{camera_id}: tracker/frame index mismatch")
        if source_pts_ns < 0 or source_pts_ns != cached_source_pts_ns:
            raise FrameJoinError(f"{camera_id}: tracker PTS differs from cached source PTS")
        previous_index = self._last_frame.get(camera_id, -1)
        previous_pts = self._last_pts.get(camera_id, -1)
        if frame_index <= previous_index or source_pts_ns <= previous_pts:
            raise FrameJoinError(f"{camera_id}: stale or non-monotonic tracker frame/PTS")

        raw_objects = record.get("objects")
        if not isinstance(raw_objects, list):
            raise FrameJoinError(f"{camera_id}: tracker record has no object list")
        selected: list[tuple[str, dict[str, Any]]] = []
        occurrences: dict[str, int] = defaultdict(int)
        for obj in raw_objects:
            if not isinstance(obj, dict):
                continue
            if self.person_class_id is not None:
                try:
                    if int(obj.get("class_id", -1)) != self.person_class_id:
                        continue
                except (TypeError, ValueError):
                    continue
            tracker_id = obj.get("tracker_id")
            if tracker_id is None or str(tracker_id) in {"", "-1", "18446744073709551615"}:
                continue
            local_id = str(tracker_id)
            occurrences[local_id] += 1
            selected.append((local_id, obj))

        duplicate_ids = {tracker_id for tracker_id, count in occurrences.items() if count > 1}
        valid_markers: list[dict[str, Any]] = []
        out_of_bounds_ids: list[str] = []
        invalid_ids: list[str] = []
        jump_ids: list[str] = []
        for local_id, obj in selected:
            if local_id in duplicate_ids:
                continue
            try:
                box = [float(obj[field]) for field in ("left", "top", "width", "height")]
            except (KeyError, TypeError, ValueError):
                invalid_ids.append(local_id)
                continue
            if not all(math.isfinite(value) for value in box):
                invalid_ids.append(local_id)
                continue
            xy = project_anchor_unbounded(self.matrices[CAMERA_TO_MATRIX_KEY[camera_id]], box)
            if xy is None:
                invalid_ids.append(local_id)
                continue
            if not (0.0 <= xy[0] < self.layout_width and 0.0 <= xy[1] < self.layout_height):
                # Keep the measured coordinate in diagnostics, but do not draw
                # it at a clamped or substituted location.
                out_of_bounds_ids.append(local_id)
                continue

            previous = self._last_projected.get((camera_id, local_id))
            if previous is not None:
                old_frame, old_pts, old_xy = previous
                delta_ns = source_pts_ns - old_pts
                if (old_frame == frame_index - 1 and 0 < delta_ns <= self.jump_max_interval_ns
                        and math.dist(old_xy, xy) > self.jump_threshold_layout_px):
                    jump_ids.append(local_id)
            self._last_projected[(camera_id, local_id)] = (frame_index, source_pts_ns, xy)
            self._trails[(camera_id, local_id)].append(xy)
            valid_markers.append({
                "camera_id": camera_id,
                "source_id": source_id,
                "frame_index": frame_index,
                "source_pts_ns": source_pts_ns,
                "local_tracker_id": local_id,
                "foot_anchor_inference_xy": [box[0] + box[2] / 2.0, box[1] + box[3]],
                "layout_xy": [float(xy[0]), float(xy[1])],
                "trail": [[float(x), float(y)] for x, y in self._trails[(camera_id, local_id)]],
                "jump_candidate": local_id in jump_ids,
            })

        self._last_frame[camera_id] = frame_index
        self._last_pts[camera_id] = source_pts_ns
        counts = self.counters[camera_id]
        counts["frames"] += 1
        counts["projected"] += len(valid_markers)
        counts["out_of_bounds"] += len(out_of_bounds_ids)
        counts["invalid"] += len(invalid_ids)
        counts["duplicate_local_ids"] += len(duplicate_ids)
        counts["large_jump_candidates"] += len(jump_ids)
        self._states[camera_id] = {
            "camera_id": camera_id,
            "source_id": source_id,
            "frame_index": frame_index,
            "source_pts_ns": source_pts_ns,
            "updated_mono_ns": now,
            "observation_time_domain": "HOST_MONOTONIC_ARRIVAL",
            "markers": valid_markers,
            "out_of_bounds_ids": sorted(out_of_bounds_ids),
            "invalid_ids": sorted(invalid_ids),
            "duplicate_local_ids": sorted(duplicate_ids),
            "jump_candidate_ids": sorted(jump_ids),
            "cross_camera_sync": "UNVERIFIED; independent camera source PTS",
        }
        return self.snapshot(camera_id, now_mono_ns=now)

    def snapshot(self, camera_id: str, *, now_mono_ns: int | None = None) -> dict[str, Any]:
        now = time.monotonic_ns() if now_mono_ns is None else int(now_mono_ns)
        state = self._states.get(camera_id)
        if state is None:
            return {"camera_id": camera_id, "status": "WAITING", "markers": []}
        age_ns = max(0, now - int(state["updated_mono_ns"]))
        if age_ns > self.marker_max_age_ns:
            return {**state, "status": "STALE", "markers": [], "age_ms": age_ns / 1e6}
        return {**state, "status": "LIVE", "age_ms": age_ns / 1e6}
