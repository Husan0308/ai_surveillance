from __future__ import annotations

from collections.abc import Iterable
from typing import Any


CAMERA_ORDER = {"CAM-01": 0, "CAM-04": 1, "CAM-02": 2, "CAM-05": 3}


def update_active_observations(
    previous: Iterable[dict[str, Any]],
    new_rows: Iterable[dict[str, Any]],
    *,
    current_frame: int,
    expiry_frames: int = 30,
) -> list[dict[str, Any]]:
    """Keep the latest row per (camera, MV3DT output ID), never fuse positions."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in (*tuple(previous), *tuple(new_rows)):
        camera = str(row.get("camera_id", ""))
        track_id = str(row.get("mv3dt_global_id", ""))
        try:
            frame = int(row.get("frame_id", -1))
        except (TypeError, ValueError):
            continue
        if camera not in CAMERA_ORDER or not track_id or frame > current_frame or current_frame - frame > expiry_frames:
            continue
        key = (camera, track_id)
        old = latest.get(key)
        if old is None or int(old["frame_id"]) < frame:
            latest[key] = dict(row)
    return sorted(latest.values(), key=lambda row: (
        CAMERA_ORDER[str(row["camera_id"])], str(row["mv3dt_global_id"])
    ))


def group_for_display(rows: Iterable[dict[str, Any]], *, use_display_aliases: bool = False,
                      separate_by_camera: bool = False) -> list[dict[str, Any]]:
    """Group by raw MV3DT ID or an explicitly selected presentation ID.

    Camera observations and their coordinates remain separate in either mode.
    """
    grouped: dict[str | tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        camera = str(row.get("camera_id", ""))
        raw_id = str(row.get("raw_global_id", row["mv3dt_global_id"]))
        display_id = str(row.get("display_global_id", raw_id)) if use_display_aliases else raw_id
        copied = dict(row)
        copied["raw_global_id"] = raw_id
        scoped_id = f"{camera} / {display_id}" if separate_by_camera else display_id
        copied["display_global_id"] = scoped_id
        # Keep the source MV3DT ID unchanged; a source-qualified label is only
        # a presentation key and is never a replacement tracker identity.
        copied["mv3dt_global_id"] = raw_id if separate_by_camera else display_id
        group_key: str | tuple[str, str] = (camera, display_id) if separate_by_camera else display_id
        grouped.setdefault(group_key, []).append(copied)
    result = []
    for group_key, observations in grouped.items():
        observations.sort(key=lambda row: CAMERA_ORDER.get(str(row["camera_id"]), 99))
        camera = str(group_key[0]) if isinstance(group_key, tuple) else None
        track_id = observations[0]["display_global_id"]
        result.append({"mv3dt_global_id": observations[0]["mv3dt_global_id"],
                       "display_global_id": track_id, "observations": observations,
                       "cameras": [row["camera_id"] for row in observations],
                       "camera_scope": camera})
    return sorted(result, key=lambda item: (
        CAMERA_ORDER.get(str(item["camera_scope"] or ""), 99), str(item["mv3dt_global_id"])
    ))


def display_state(rows: Iterable[dict[str, Any]], *, current_frame: int,
                  visual_review: dict[str, Any] | None = None,
                  use_display_aliases: bool = False,
                  separate_by_camera: bool = False) -> dict[str, Any]:
    """Count current observations, and group IDs without guessing associations.

    A representative camera position is selected deterministically; positions
    are not averaged. Different IDs remain distinct unless an explicit,
    source-verified review covers this exact frame and all current IDs.
    """
    recent = list(rows)
    current = [row for row in recent if int(row["frame_id"]) == current_frame]
    raw_ids = {str(row.get("raw_global_id", row["mv3dt_global_id"])) for row in current}
    groups = group_for_display(current, use_display_aliases=use_display_aliases,
                               separate_by_camera=separate_by_camera)
    display_ids = {group["display_global_id"] for group in groups}
    aliases: dict[str, str] = {}
    confirmed = None
    review = None
    review_suppressed = None
    if separate_by_camera and visual_review:
        review_suppressed = "Camera-scoped labels are enabled; cross-camera visual grouping is suppressed."
    elif use_display_aliases and visual_review:
        review_suppressed = "Turn off adoption aliases to view raw-ID-bound visual-review counts."
    if (not separate_by_camera and not use_display_aliases and visual_review and visual_review.get("verified_for_source") is True
            and visual_review.get("frame_id") == current_frame):
        reviewed_groups = visual_review.get("people_groups", [])
        members = [str(member) for group in reviewed_groups for member in group.get("member_global_ids", [])]
        if (len(members) == len(set(members)) and set(members) == raw_ids and reviewed_groups
                and all(group.get("primary_global_id") in group.get("member_global_ids", [])
                        for group in reviewed_groups)):
            aliases = {str(member): str(group["primary_global_id"]) for group in reviewed_groups
                       for member in group["member_global_ids"] if member != group["primary_global_id"]}
            confirmed = len(reviewed_groups)
            review = visual_review
    markers = []
    for group in groups:
        output_id = group["display_global_id"]
        if output_id in aliases:
            continue
        representative = dict(group["observations"][0])
        representative["raw_global_ids"] = sorted({str(row["raw_global_id"]) for row in group["observations"]})
        representative["mv3dt_global_id"] = (
            str(representative["raw_global_id"]) if separate_by_camera else output_id
        )
        representative["display_global_id"] = output_id
        representative["observing_cameras"] = list(dict.fromkeys(group["cameras"]))
        representative["reviewed_duplicate_ids"] = sorted(alias for alias, primary in aliases.items() if primary == output_id)
        markers.append(representative)
    return {"camera_observations": len(current), "unique_global_ids": len(raw_ids),
            "display_unique_ids": len(display_ids),
            "camera_scoped_track_labels": len(groups) if separate_by_camera else None,
            "camera_scoped_ids_enabled": separate_by_camera,
            "display_aliases_enabled": use_display_aliases,
            "display_aliases_applied": sum(bool(row.get("display_alias_applied")) for row in current)
            if use_display_aliases else 0,
            "confirmed_visible_people": confirmed, "primary_markers": markers,
            "recent_observations": len(recent) - len(current), "current_observations": current,
            "duplicate_global_ids": aliases, "visual_review": review,
            "visual_review_suppressed_reason": review_suppressed}


def world_to_map_pixel(world_x: float, world_y: float, matrix: Iterable[float]) -> tuple[float, float]:
    values = tuple(float(value) for value in matrix)
    if len(values) != 9:
        raise ValueError("world-to-map transform must be a row-major 3x3 matrix")
    x = values[0] * float(world_x) + values[1] * float(world_y) + values[2]
    y = values[3] * float(world_x) + values[4] * float(world_y) + values[5]
    w = values[6] * float(world_x) + values[7] * float(world_y) + values[8]
    if w == 0.0:
        raise ValueError("world-to-map transform projected to infinity")
    return x / w, y / w

