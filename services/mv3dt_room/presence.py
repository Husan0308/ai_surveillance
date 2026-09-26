"""Strict current-camera presence rules for the scoped room-pair view.

This module is deliberately downstream of identity resolution. It does not
change detector, tracker, calibration, bbox recovery, MVA, or ReID behavior;
it only decides which already-resolved observations are renderable now.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

CAMERAS = ("CAM-01", "CAM-04")


def application_id(row: dict[str, Any]) -> str:
    return str(
        row.get("application_id")
        or row.get("persistent_global_person_id")
        or row.get("global_person_id")
        or "Unknown"
    )


def _frame_number(row: dict[str, Any]) -> int | None:
    try:
        return int(row["frame"])
    except (KeyError, TypeError, ValueError):
        return None


def valid_world(world: Any) -> bool:
    if not isinstance(world, (list, tuple)) or len(world) < 2:
        return False
    try:
        return math.isfinite(float(world[0])) and math.isfinite(float(world[1]))
    except (TypeError, ValueError):
        return False


def current_frame(rows: Iterable[dict[str, Any]]) -> int | None:
    frames = [_frame_number(row) for row in rows]
    valid = [frame for frame in frames if frame is not None]
    return max(valid) if valid else None


def active_observations(
    rows: Iterable[dict[str, Any]],
    frame: int | None = None,
) -> tuple[int | None, list[dict[str, Any]]]:
    """Return only the latest frame and one observation per camera/app ID.

    Historical observations are intentionally excluded. If a camera emits
    multiple native fragments for one application identity in the same frame,
    the highest-confidence observation is the sole renderable observation for
    that camera/application pair.
    """
    scoped = [row for row in rows if row.get("camera_id") in CAMERAS]
    selected_frame = current_frame(scoped) if frame is None else int(frame)
    if selected_frame is None:
        return None, []
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in scoped:
        if _frame_number(row) != selected_frame:
            continue
        key = (str(row.get("camera_id")), application_id(row))
        confidence = float(row.get("confidence") or 0.0)
        previous = latest.get(key)
        if previous is None or confidence > float(previous.get("confidence") or 0.0):
            latest[key] = row
    return selected_frame, list(latest.values())


def fused_world_positions(rows: Iterable[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Fuse current valid camera positions to one world position per app ID."""
    positions: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        world = row.get("world")
        if valid_world(world):
            positions[application_id(row)].append((float(world[0]), float(world[1])))
    return {
        identity: (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
        for identity, points in positions.items()
    }



class CurrentPresencePositionFilter:
    """Smooth only currently observed positions; never retain visible ghosts."""

    def __init__(self, alpha: float = 0.30) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._positions: dict[str, tuple[float, float]] = {}

    def reset(self) -> None:
        self._positions.clear()

    def update(self, rows: Iterable[dict[str, Any]]) -> dict[str, tuple[float, float]]:
        measured = fused_world_positions(rows)
        # Presence remains strict: disappearing from every camera removes both
        # the marker and its visual filter state in this same update.
        for identity in set(self._positions) - set(measured):
            del self._positions[identity]
        for identity, point in measured.items():
            previous = self._positions.get(identity)
            if previous is None:
                self._positions[identity] = point
            else:
                self._positions[identity] = (
                    previous[0] + self.alpha * (point[0] - previous[0]),
                    previous[1] + self.alpha * (point[1] - previous[1]),
                )
        return dict(self._positions)

def presence_sets(rows: Iterable[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """Return active camera-observed IDs and one-per-ID map-renderable IDs."""
    active = {application_id(row) for row in rows}
    renderable = set(fused_world_positions(rows))
    return active, renderable


def assert_presence_contract(
    active_rows: Iterable[dict[str, Any]],
    rendered_ids: Iterable[str],
    *,
    duplicate_ids: Iterable[str] = (),
    native_rendered_ids: Iterable[str] = (),
    stale_rendered_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Assert strict camera/map presence and return an auditable summary."""
    rows = list(active_rows)
    active_ids, renderable_ids = presence_sets(rows)
    rendered = {str(identity) for identity in rendered_ids}
    duplicates = {str(identity) for identity in duplicate_ids}
    native = {str(identity) for identity in native_rendered_ids}
    stale = {str(identity) for identity in stale_rendered_ids}
    assert not duplicates, f"duplicate BEV markers: {sorted(duplicates)}"
    assert not native, f"native IDs rendered as BEV markers: {sorted(native)}"
    assert not stale, f"stale BEV markers: {sorted(stale)}"
    assert rendered == renderable_ids, {
        "active_camera_ids": sorted(active_ids),
        "renderable_ids": sorted(renderable_ids),
        "rendered_ids": sorted(rendered),
    }
    assert rendered <= active_ids, "BEV marker without an active camera observation"
    return {
        "active_camera_ids": sorted(active_ids),
        "renderable_ids": sorted(renderable_ids),
        "rendered_ids": sorted(rendered),
        "no_marker_without_active_camera_observation": rendered <= active_ids,
        "no_duplicate_application_markers": not duplicates,
        "no_orphan_native_ghost_marker": not native,
        "no_stale_last_position_marker": not stale,
    }
