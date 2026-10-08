"""Latest-only canonical post-NvDCF object feed shared by native, API and UI."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_DIR = Path("/dev/shm")
CAMERA_IDS = tuple(f"CAM-{index:02d}" for index in range(1, 7))
SCHEMA_VERSION = 1
MAX_SNAPSHOT_BYTES = 256 * 1024


def feed_dir() -> Path:
    return Path(os.getenv("V11_TRACKED_OBJECT_FEED_DIR", str(DEFAULT_DIR)))


def camera_feed_path(camera_id: str) -> Path:
    if camera_id not in CAMERA_IDS:
        raise ValueError(f"unknown camera id: {camera_id}")
    return feed_dir() / f"v11_tracked_objects_cam{camera_id[-2:]}_v1.json"


def read_camera_snapshot(camera_id: str, *, max_age_sec: float = 0.25,
                         now_monotonic_ns: int | None = None) -> dict[str, Any] | None:
    """Read one atomically replaced camera snapshot, rejecting stale sessions."""
    try:
        raw = camera_feed_path(camera_id).read_bytes()
        if not raw or len(raw) > MAX_SNAPSHOT_BYTES:
            return None
        snapshot = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != SCHEMA_VERSION:
        return None
    if snapshot.get("camera_id") != camera_id or not isinstance(snapshot.get("generation"), str):
        return None
    try:
        stamp = int(snapshot["timestamp_monotonic_ns"])
        frame = int(snapshot["frame_num"])
        pts = int(snapshot["pts_ns"])
        width = int(snapshot["source_frame_width"])
        height = int(snapshot["source_frame_height"])
        bbox_width = int(snapshot.get("bbox_coordinate_width") or snapshot.get("pipeline_width") or width)
        bbox_height = int(snapshot.get("bbox_coordinate_height") or snapshot.get("pipeline_height") or height)
        pipeline_width = int(snapshot.get("pipeline_width") or 0)
        pipeline_height = int(snapshot.get("pipeline_height") or 0)
    except (KeyError, TypeError, ValueError):
        return None
    now = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
    age = now - stamp
    if stamp <= 0 or age < 0 or age > int(max_age_sec * 1_000_000_000):
        return None
    if frame < 0 or pts < 0 or width <= 0 or height <= 0 or bbox_width <= 0 or bbox_height <= 0:
        return None
    # When a producer explicitly declares pipeline-space coordinates, its
    # bbox extent must be the actual pipeline canvas, not the decoded source
    # dimensions. Fail closed on inconsistent metadata rather than painting a
    # valid rectangle in the wrong place.
    if snapshot.get("bbox_coordinate_space") == "pipeline" and (
            pipeline_width <= 0 or pipeline_height <= 0
            or (bbox_width, bbox_height) != (pipeline_width, pipeline_height)):
        return None
    objects = snapshot.get("objects")
    if not isinstance(objects, list):
        return None
    clean: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in objects:
        if not isinstance(row, dict) or row.get("camera_id") != camera_id:
            return None
        try:
            native = int(row["native_track_id"])
            object_frame = int(row["frame_num"])
            object_pts = int(row["pts_ns"])
            bbox = [float(value) for value in row["bbox"]]
            object_width = int(row.get("bbox_coordinate_width") or bbox_width)
            object_height = int(row.get("bbox_coordinate_height") or bbox_height)
        except (KeyError, TypeError, ValueError):
            return None
        if (native < 0 or native in seen or object_frame != frame or object_pts != pts
                or len(bbox) != 4 or not all(math.isfinite(value) for value in bbox)
                or object_width <= 0 or object_height <= 0
                or bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]
                or bbox[2] > object_width or bbox[3] > object_height):
            return None
        if row.get("bbox_coordinate_space") == "pipeline" and (
                (object_width, object_height) != (bbox_width, bbox_height)):
            return None
        if row.get("bbox_source") not in {
                "TRACKER", "PGIE_SHADOW_FALLBACK", "SHADOW_CONTEXT_VERIFIED"}:
            return None
        seen.add(native)
        clean.append(row | {"bbox_coordinate_width": object_width,
                            "bbox_coordinate_height": object_height})
    proposals = snapshot.get("detector_proposals", [])
    if not isinstance(proposals, list):
        return None
    clean_proposals: list[dict[str, Any]] = []
    for row in proposals:
        if not isinstance(row, dict) or row.get("camera_id") != camera_id:
            continue
        try:
            proposal_frame = int(row["frame_num"])
            proposal_pts = int(row["pts_ns"])
            proposal_index = int(row["proposal_index"])
            confidence = float(row["confidence"])
            bbox = [float(value) for value in row["bbox"]]
            proposal_width = int(row.get("bbox_coordinate_width") or row.get("source_frame_width")
                                 or bbox_width)
            proposal_height = int(row.get("bbox_coordinate_height") or row.get("source_frame_height")
                                  or bbox_height)
        except (KeyError, TypeError, ValueError):
            continue
        if (proposal_frame != frame or proposal_pts != pts or proposal_index < 0
                or not 0.0 <= confidence <= 1.0 or len(bbox) != 4
                or any(not math.isfinite(value) for value in bbox)
                or proposal_width <= 0 or proposal_height <= 0
                or bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]
                or bbox[2] > proposal_width or bbox[3] > proposal_height):
            continue
        if row.get("bbox_coordinate_space") == "pipeline" and (
                (proposal_width, proposal_height) != (bbox_width, bbox_height)):
            continue
        clean_proposals.append(row | {"bbox_coordinate_width": proposal_width,
                                      "bbox_coordinate_height": proposal_height,
                                      "source_frame_width": int(row.get("source_frame_width") or proposal_width),
                                      "source_frame_height": int(row.get("source_frame_height") or proposal_height)})
    return snapshot | {"snapshot_age_ms": round(age / 1_000_000.0, 3), "objects": clean,
                       "detector_proposals": clean_proposals}


def read_snapshot(*, camera_ids: tuple[str, ...] = CAMERA_IDS,
                  max_age_sec: float = 0.5,
                  now_monotonic_ns: int | None = None) -> dict[str, Any]:
    now = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
    cameras: dict[str, dict[str, Any] | None] = {
        camera: read_camera_snapshot(camera, max_age_sec=max_age_sec, now_monotonic_ns=now)
        for camera in camera_ids
    }
    fresh = [camera for camera, row in cameras.items() if row is not None]
    rows = [obj for row in cameras.values() if row is not None for obj in row["objects"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "fresh" if len(fresh) == len(camera_ids) else "partial" if fresh else "offline",
        "supported": True,
        "camera_ids": list(camera_ids),
        "online_cameras": fresh,
        "cameras": cameras,
        "tracked_objects": rows,
    }
