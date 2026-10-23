"""Build a private offline Entrance map review from scale-only local tracks.

The renderer is deliberately fail-closed: it accepts only an immutable
LIVE_NATIVE_ALIGNMENT candidate, exact raw source hashes, and tracker rows
whose source frame/PTS crosswalk matches those same recordings. It never
combines camera-local IDs or changes source tracking records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from services.mv3dt_staging.live_map_projection import LiveNativeMapProjection
from services.mv3dt_staging.live_map_projection import project_anchor_unbounded

from services.mv3dt_staging.calibration_editor import LIVE_NATIVE_SPACE
from services.mv3dt_staging.calibration_geometry import diagnose_homography
from services.mv3dt_staging.live_tracking.native_frame_bridge import (
    DETECTOR_HEIGHT,
    DETECTOR_WIDTH,
    NativeFullFovScale,
    SOURCE_SIZE,
)


CAMERA_COLORS = {"CAM-02": (255, 190, 60), "CAM-05": (60, 205, 255)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_anchor(matrix: np.ndarray, box: list[float] | tuple[float, ...],
                   layout_width: int, layout_height: int) -> tuple[float, float] | None:
    """Project the box bottom-centre in 1920x1080 inference coordinates."""
    point = project_anchor_unbounded(matrix, box)
    if point is None:
        return None
    x, y = point
    if not (0 <= x < layout_width and 0 <= y < layout_height):
        return None
    return point


def _draw_map_labels(map_view: np.ndarray,
                     labels: list[tuple[tuple[int, int], str, tuple[int, int, int]]]) -> int:
    """Place short camera/local-ID labels without overlapping placed labels."""
    reserved: list[tuple[int, int, int, int]] = []
    placed_count = 0
    directions = ((1, 0), (1, -1), (1, 1), (-1, 0), (-1, -1), (-1, 1),
                  (0, -1), (0, 1))
    height, width = map_view.shape[:2]
    for point, text, color in labels:
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        chosen = None
        for radius in (10, 18, 28, 40, 54, 70, 90):
            for dx, dy in directions:
                x = max(3, min(width - text_width - 5, point[0] + dx * radius))
                y = max(text_height + 5, min(height - baseline - 4, point[1] + dy * radius))
                rect = (x - 2, y - text_height - 2, x + text_width + 2, y + baseline + 2)
                if all(rect[2] < other[0] or rect[0] > other[2]
                       or rect[3] < other[1] or rect[1] > other[3]
                       for other in reserved):
                    chosen = (x, y, rect)
                    break
            if chosen is not None:
                break
        if chosen is None:
            continue
        x, y, rect = chosen
        reserved.append(rect)
        if math.dist(point, (x, y)) > 24:
            cv2.line(map_view, point, (x, y - text_height // 2), color, 1, cv2.LINE_AA)
        cv2.rectangle(map_view, (rect[0], rect[1]), (rect[2], rect[3]), (16, 16, 16), -1)
        cv2.putText(map_view, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, color, 1, cv2.LINE_AA)
        placed_count += 1
    return placed_count


def load_validated_candidate(
    alignment_path: Path,
    *,
    video_paths: dict[str, Path],
    layout_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Validate the fresh scale-only fit and two new holdouts per camera."""
    candidate = json.loads(alignment_path.read_text(encoding="utf-8"))
    if candidate.get("coordinate_space") != LIVE_NATIVE_SPACE:
        raise ValueError("alignment is not in the LIVE native scale-only coordinate space")
    if candidate.get("camera_mapping") != {"cam02": "CAM-02", "cam05": "CAM-05"}:
        raise ValueError("camera order is not CAM-02=cam02, CAM-05=cam05")
    expected_dims = {"cam02": [DETECTOR_WIDTH, DETECTOR_HEIGHT],
                     "cam05": [DETECTOR_WIDTH, DETECTOR_HEIGHT]}
    if candidate.get("camera_coordinate_dimensions") != expected_dims:
        raise ValueError("alignment camera coordinates are not both 1920x1080")
    layout_meta = candidate.get("layout") or {}
    if layout_meta.get("sha256") != file_sha256(layout_path):
        raise ValueError("layout hash differs from the layout used for correspondence selection")
    layout = cv2.imread(str(layout_path), cv2.IMREAD_COLOR)
    if layout is None:
        raise ValueError("cannot decode Entrance layout")
    layout_height, layout_width = layout.shape[:2]

    for key, path in video_paths.items():
        meta = candidate.get("source_files", {}).get(key) or {}
        camera_id = "CAM-02" if key == "cam02" else "CAM-05"
        expected_width, expected_height = SOURCE_SIZE[camera_id]
        if meta.get("sha256") != file_sha256(path):
            raise ValueError(f"{camera_id} source hash differs from the saved alignment")
        if (meta.get("width"), meta.get("height")) != (expected_width, expected_height):
            raise ValueError(f"{camera_id} source dimensions are not the verified native geometry")

    matrices: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for key, camera_key in (("cam02", "cam02"), ("cam05", "cam05")):
        source = candidate.get("source_files", {}).get(key, {})
        valid_rows = []
        for point in candidate.get("points", []):
            camera_xy = point.get(camera_key)
            layout_xy = point.get("layout")
            if (camera_xy is None or layout_xy is None
                    or point.get(f"{camera_key}_source_sha256") != source.get("sha256")
                    or point.get(f"{camera_key}_coordinate_space") != LIVE_NATIVE_SPACE
                    or point.get("layout_verified") is not True
                    or point.get("layout_source_sha256") != layout_meta.get("sha256")):
                continue
            valid_rows.append(point)
        fit_count = sum(row.get("role") == "fit" for row in valid_rows)
        independent_holdouts = [row for row in valid_rows
                                if row.get("role") == "heldout"
                                and row.get("independent_of_prior_fit") is True
                                and row.get("prior_alignment_reference") is not True
                                and row.get("provenance") == "operator_selected_new_floor_landmark"]
        if fit_count < 4 or len(independent_holdouts) < 2:
            raise ValueError(f"{camera_key}: need >=4 complete FIT points and >=2 NEW independent HELD-OUT points")
        report = diagnose_homography(
            valid_rows, camera_key,
            source_shape_hw=(DETECTOR_HEIGHT, DETECTOR_WIDTH),
            layout_shape_hw=(layout_height, layout_width),
        )
        if (report["matrix_row_major"] is None or report.get("out_of_bounds_point_ids")
                or report.get("duplicate_point_pairs_within_1px")):
            raise ValueError(f"{camera_key}: fit has invalid matrix, bounds or duplicate-point diagnostics")
        matrices[camera_key] = np.asarray(report["matrix_row_major"], dtype=np.float64).reshape(3, 3)
        diagnostics[camera_key] = report
    return matrices, diagnostics, {"image": layout, "width": layout_width, "height": layout_height}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: row must be an object")
                rows.append(value)
    return rows


def load_tracking_crosswalk(tracking_dir: Path, video_paths: dict[str, Path]) -> tuple[dict, dict, dict]:
    """Check raw hashes, camera/source mapping, frame indexes and source PTS."""
    provenance = json.loads((tracking_dir / "provenance.json").read_text(encoding="utf-8"))
    expected_ids = {"source_id_0": "CAM-02", "source_id_1": "CAM-05"}
    if provenance.get("source_id_mapping") != expected_ids:
        raise ValueError("tracking provenance camera/source ordering is not CAM-02 then CAM-05")
    by_camera = {row.get("camera_id"): row for row in provenance.get("inputs", [])}
    for key, camera_id in (("cam02", "CAM-02"), ("cam05", "CAM-05")):
        record = by_camera.get(camera_id)
        if not record or record.get("sha256") != file_sha256(video_paths[key]):
            raise ValueError(f"saved tracking run does not use the exact current {camera_id} source")

    local_pts: dict[str, dict[int, int]] = {"CAM-02": {}, "CAM-05": {}}
    for row in _read_jsonl(tracking_dir / "source-frames.jsonl"):
        sid = int(row["source_id"])
        camera_id = expected_ids[f"source_id_{sid}"]
        frame = int(row["source_local_frame_num"])
        if frame in local_pts[camera_id]:
            raise ValueError(f"duplicate source-frame crosswalk entry: {camera_id} frame {frame}")
        local_pts[camera_id][frame] = int(row["source_pts_ns"])

    frame_metadata: dict[str, dict[int, dict[str, Any]]] = {
        "CAM-02": {}, "CAM-05": {}
    }
    for row in _read_jsonl(tracking_dir / "frames.jsonl"):
        sid = int(row["source_id"])
        camera_id = expected_ids[f"source_id_{sid}"]
        frame = int(row["frame"])
        if frame in frame_metadata[camera_id]:
            raise ValueError(f"duplicate tracker frame metadata: {camera_id} frame {frame}")
        native_width, native_height = SOURCE_SIZE[camera_id]
        if (int(row.get("source_width", -1)), int(row.get("source_height", -1))) != (native_width, native_height):
            raise ValueError(f"recorded native geometry mismatch in tracker frame metadata: {camera_id}")
        if (int(row.get("pipeline_width", -1)), int(row.get("pipeline_height", -1))) != (DETECTOR_WIDTH, DETECTOR_HEIGHT):
            raise ValueError(f"tracker observations are not in live 1920x1080 coordinates: {camera_id}")
        if frame not in local_pts[camera_id] or int(row["pts_ns"]) != local_pts[camera_id][frame]:
            raise ValueError(f"frame metadata PTS/index is not tied to source frame: {camera_id} frame {frame}")
        frame_metadata[camera_id][frame] = row

    tracks: dict[str, dict[int, list[dict[str, Any]]]] = {
        "CAM-02": defaultdict(list), "CAM-05": defaultdict(list)
    }
    seen = set()
    for row in _read_jsonl(tracking_dir / "tracks.jsonl"):
        sid = int(row["source_id"])
        camera_id = expected_ids[f"source_id_{sid}"]
        frame = int(row["frame"])
        if frame not in local_pts[camera_id]:
            raise ValueError(f"tracker frame lacks source-index crosswalk: {camera_id} frame {frame}")
        if int(row["pts_ns"]) != local_pts[camera_id][frame]:
            raise ValueError(f"tracker PTS differs from source PTS: {camera_id} frame {frame}")
        local_id = str(row["object_id"])
        key = (camera_id, frame, local_id)
        if key in seen:
            raise ValueError(f"duplicate tracker observation for one camera/frame/local ID: {key}")
        seen.add(key)
        box = row.get("box")
        if (not isinstance(box, list) or len(box) != 4
                or not all(math.isfinite(float(v)) for v in box)):
            raise ValueError(f"invalid tracker bbox: {camera_id} frame {frame}")
        tracks[camera_id][frame].append(row)

    for key, camera_id in (("cam02", "CAM-02"), ("cam05", "CAM-05")):
        capture = cv2.VideoCapture(str(video_paths[key]))
        if not capture.isOpened():
            raise ValueError(f"cannot decode {camera_id} source")
        expected_width, expected_height = SOURCE_SIZE[camera_id]
        observed = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        capture.release()
        if observed != (expected_width, expected_height):
            raise ValueError(f"{camera_id} decoder geometry differs from expected native dimensions")
        if not local_pts[camera_id] or min(local_pts[camera_id]) != 0:
            raise ValueError(f"{camera_id} source frame crosswalk is empty or lacks frame zero")
        if set(frame_metadata[camera_id]) != set(local_pts[camera_id]):
            raise ValueError(f"{camera_id} frame metadata does not exactly cover source frame indexes")
    return tracks, local_pts, provenance


def _camera_panel(frame_bgr: np.ndarray, camera_id: str,
                  observations: list[dict[str, Any]],
                  projection_warning_ids: set[str] | None = None,
                  provisional_global_ids: dict[str, dict[str, Any]] | None = None,
                  ) -> tuple[np.ndarray, list[list[float]]]:
    frame = frame_bgr.copy()
    anchors: list[list[float]] = []
    color = CAMERA_COLORS[camera_id]
    projection_warning_ids = projection_warning_ids or set()
    provisional_global_ids = provisional_global_ids or {}
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, 25), (14, 14, 14), -1)
    cv2.putText(frame, "DEMO / RECORDED VIDEO - NOT LIVE", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    for row in observations:
        left, top, width, height = (float(v) for v in row["box"])
        x0, y0 = round(left), round(top)
        x1, y1 = round(left + width), round(top + height)
        anchor = [left + width / 2.0, top + height]
        anchors.append(anchor)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.circle(frame, (round(anchor[0]), round(anchor[1])), 5, color, -1)
        if str(row["object_id"]) in projection_warning_ids:
            cv2.circle(frame, (round(anchor[0]), round(anchor[1])), 10,
                       (0, 0, 255), 2)
        label = f"{camera_id} local {row['object_id']}"
        provisional = provisional_global_ids.get(str(row["object_id"]))
        if provisional:
            label += f" | AUTO {provisional['provisional_global_id']}?"
        cv2.putText(frame, label, (max(0, x0), max(22, y0 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    return cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA), anchors


def diagnose_projected_motion(
    tracks: dict[str, dict[int, list[dict[str, Any]]]],
    local_pts: dict[str, dict[int, int]],
    matrices: dict[str, np.ndarray],
    *,
    layout_width: int,
    layout_height: int,
    jump_threshold_layout_px: float = 75.0,
    max_adjacent_interval_seconds: float = 0.25,
) -> dict[str, Any]:
    """Flag large one-frame projected moves without asserting physical error.

    Movement is evaluated only for the same camera-local tracker ID on
    adjacent source frame indexes with increasing, nearby source PTS.  The
    threshold is deliberately a map-pixel review heuristic: map pixels are
    not metres, so a flag is a candidate for visual review, not a proven ID
    switch or impossible movement.
    """
    result: dict[str, Any] = {
        "status": "diagnostic_heuristic_only",
        "jump_threshold_layout_px": float(jump_threshold_layout_px),
        "max_adjacent_interval_seconds": float(max_adjacent_interval_seconds),
        "units_warning": "layout pixels are not physical distance; flags do not establish an identity error",
        "cameras": {},
    }
    for camera_id in ("CAM-02", "CAM-05"):
        matrix_key = "cam02" if camera_id == "CAM-02" else "cam05"
        observations_by_frame = tracks[camera_id]
        projected_by_frame: dict[int, dict[str, tuple[float, float]]] = {}
        invalid = 0
        out_of_bounds = 0
        invalid_events: list[dict[str, Any]] = []
        out_of_bounds_events: list[dict[str, Any]] = []
        obs_count = 0
        id_frames: dict[str, list[int]] = defaultdict(list)
        previous_by_id: dict[str, tuple[int, int, tuple[float, float] | None]] = {}
        jump_events: list[dict[str, Any]] = []
        distances: list[float] = []
        for frame_index in sorted(observations_by_frame):
            frame_pts = local_pts[camera_id].get(frame_index)
            if frame_pts is None:
                raise ValueError(f"missing source PTS for {camera_id} frame {frame_index}")
            frame_map: dict[str, tuple[float, float]] = {}
            for row in observations_by_frame[frame_index]:
                obs_count += 1
                local_id = str(row["object_id"])
                id_frames[local_id].append(frame_index)
                point = project_anchor_unbounded(matrices[matrix_key], row["box"])
                current = tuple(point) if point is not None else None
                if current is None:
                    invalid += 1
                    invalid_events.append({
                        "camera_id": camera_id,
                        "frame": frame_index,
                        "source_pts_ns": frame_pts,
                        "local_tracker_id": local_id,
                        "status": "nonfinite_or_invalid_projection",
                    })
                elif not (0 <= current[0] < layout_width and 0 <= current[1] < layout_height):
                    out_of_bounds += 1
                    out_of_bounds_events.append({
                        "camera_id": camera_id,
                        "frame": frame_index,
                        "source_pts_ns": frame_pts,
                        "local_tracker_id": local_id,
                        "projected_layout_xy_px": list(current),
                        "status": "outside_layout_bounds",
                    })
                    current = None
                else:
                    frame_map[local_id] = current
                prior = previous_by_id.get(local_id)
                if prior is not None:
                    prior_frame, prior_pts, prior_xy = prior
                    delta_seconds = (frame_pts - prior_pts) / 1e9
                    if (prior_frame == frame_index - 1 and prior_xy is not None
                            and current is not None and 0 < delta_seconds <= max_adjacent_interval_seconds):
                        distance = math.dist(prior_xy, current)
                        distances.append(distance)
                        if distance > jump_threshold_layout_px:
                            jump_events.append({
                                "camera_id": camera_id,
                                "local_tracker_id": local_id,
                                "from_frame": prior_frame,
                                "to_frame": frame_index,
                                "from_pts_ns": prior_pts,
                                "to_pts_ns": frame_pts,
                                "elapsed_seconds": delta_seconds,
                                "projected_displacement_layout_px": distance,
                                "status": "large_projected_jump_candidate_review_visually",
                            })
                previous_by_id[local_id] = (frame_index, frame_pts, current)
            projected_by_frame[frame_index] = frame_map

        id_ranges = []
        for local_id, frame_indexes in sorted(id_frames.items(), key=lambda item: item[0]):
            ordered = sorted(frame_indexes)
            segments = []
            start = previous = ordered[0]
            for value in ordered[1:]:
                if value != previous + 1:
                    segments.append([start, previous])
                    start = value
                previous = value
            segments.append([start, previous])
            id_ranges.append({"local_tracker_id": local_id,
                              "observation_count": len(ordered),
                              "frame_ranges": segments})

        # A local ID disappearing as another starts nearby is worth review,
        # but is not evidence that the observations belong to one person.
        handoff_candidates = []
        frames = sorted(projected_by_frame)
        for prior_frame, frame_index in zip(frames, frames[1:]):
            if frame_index != prior_frame + 1:
                continue
            prior = projected_by_frame[prior_frame]
            current = projected_by_frame[frame_index]
            ended = set(prior) - set(current)
            started = set(current) - set(prior)
            for old_id in sorted(ended):
                for new_id in sorted(started):
                    distance = math.dist(prior[old_id], current[new_id])
                    if distance <= jump_threshold_layout_px:
                        handoff_candidates.append({
                            "camera_id": camera_id,
                            "from_local_tracker_id": old_id,
                            "to_local_tracker_id": new_id,
                            "from_frame": prior_frame,
                            "to_frame": frame_index,
                            "projected_separation_layout_px": distance,
                            "status": "possible_local_id_replacement_review_only_not_merged",
                        })

        result["cameras"][camera_id] = {
            "tracker_observation_count": obs_count,
            "distinct_camera_local_ids": len(id_frames),
            "out_of_bounds_projected_anchors": out_of_bounds,
            "invalid_projected_anchors": invalid,
            "out_of_bounds_events": out_of_bounds_events,
            "invalid_projection_events": invalid_events,
            "consecutive_same_id_displacement_count": len(distances),
            "consecutive_displacement_layout_px": {
                "median": float(np.median(distances)) if distances else None,
                "p95": float(np.percentile(distances, 95)) if distances else None,
                "max": max(distances) if distances else None,
            },
            "large_projected_jump_candidates": jump_events,
            "possible_local_id_replacement_candidates": handoff_candidates,
            "local_id_frame_ranges": id_ranges,
        }
    return result


def build_preview(
    *,
    alignment_path: Path,
    video_paths: dict[str, Path],
    layout_path: Path,
    tracking_dir: Path,
    output_dir: Path,
    frame_limit: int | None = None,
    provisional_global_ids: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    matrices, diagnostics, map_meta = load_validated_candidate(
        alignment_path, video_paths=video_paths, layout_path=layout_path)
    tracks, local_pts, _provenance = load_tracking_crosswalk(tracking_dir, video_paths)
    provisional_global_ids = provisional_global_ids or {"CAM-02": {}, "CAM-05": {}}
    if set(provisional_global_ids) != {"CAM-02", "CAM-05"}:
        raise ValueError("provisional ID map must preserve separate CAM-02 and CAM-05 namespaces")
    for camera_id, entries in provisional_global_ids.items():
        if not isinstance(entries, dict):
            raise ValueError(f"{camera_id} provisional ID map must be an object")
        for local_id, entry in entries.items():
            if not isinstance(entry, dict) or not str(entry.get("provisional_global_id", "")):
                raise ValueError(f"invalid provisional ID mapping for {camera_id} local {local_id}")
    frame_count = min(len(local_pts["CAM-02"]), len(local_pts["CAM-05"]))
    if frame_limit is not None:
        frame_count = min(frame_count, int(frame_limit))
    if frame_count <= 0:
        raise ValueError("no common by-index source frames available for preview")
    motion = diagnose_projected_motion(
        tracks, local_pts, matrices,
        layout_width=map_meta["width"], layout_height=map_meta["height"],
    )
    jump_lookup = {
        (event["camera_id"], event["to_frame"], event["local_tracker_id"])
        for camera in motion["cameras"].values()
        for event in camera["large_projected_jump_candidates"]
    }
    handoff_lookup = {
        (event["camera_id"], event["to_frame"], event["to_local_tracker_id"])
        for camera in motion["cameras"].values()
        for event in camera["possible_local_id_replacement_candidates"]
    }
    oob_lookup: dict[tuple[str, int], set[str]] = defaultdict(set)
    for camera_id, camera_info in motion["cameras"].items():
        for event in camera_info["out_of_bounds_events"]:
            oob_lookup[(camera_id, event["frame"])].add(event["local_tracker_id"])

    output_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output_dir.parent, 0o700)
    output_dir.mkdir(mode=0o700, exist_ok=False)
    os.chmod(output_dir, 0o700)
    video_path = output_dir / "entrance_live_native_map_review.mp4"
    timeline_path = output_dir / "timeline.jsonl"
    width, height, output_fps = 1760, 360, 20.0
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("MP4 review writer failed to initialize")
    captures = {key: cv2.VideoCapture(str(path)) for key, path in video_paths.items()}
    if not all(cap.isOpened() for cap in captures.values()):
        writer.release()
        for cap in captures.values():
            cap.release()
        raise RuntimeError("a source recording could not be opened")
    scalers = {"cam02": NativeFullFovScale("CAM-02"), "cam05": NativeFullFovScale("CAM-05")}
    live_projection = LiveNativeMapProjection(
        matrices, layout_width=map_meta["width"], layout_height=map_meta["height"],
        person_class_id=None, trail_points=16,
    )
    source_frame_counts = {"cam02": 0, "cam05": 0}
    out_of_bounds = {"CAM-02": 0, "CAM-05": 0}
    invalid_projection = {"CAM-02": 0, "CAM-05": 0}
    projected = {"CAM-02": 0, "CAM-05": 0}
    map_labels_drawn_total = 0
    map_labels_suppressed_total = 0
    best_review_frame: np.ndarray | None = None
    best_review_frame_index: int | None = None
    best_observation_count = -1
    best_representative_rank = (-1, -1)
    try:
        with timeline_path.open("x", encoding="utf-8") as timeline:
            for frame_index in range(frame_count):
                panels = {}
                projection_states = {}
                for key, camera_id in (("cam02", "CAM-02"), ("cam05", "CAM-05")):
                    cap = captures[key]
                    ok, raw = cap.read()
                    if not ok:
                        raise RuntimeError(f"source decode stopped at {camera_id} frame {frame_index}")
                    decoded_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    if decoded_index != frame_index:
                        raise RuntimeError(
                            f"source/overlay frame mismatch at {camera_id}: expected {frame_index}, decoded {decoded_index}"
                        )
                    source_frame_counts[key] += 1
                    scaled = scalers[key](raw)
                    source_id = 0 if camera_id == "CAM-02" else 1
                    pts = local_pts[camera_id][frame_index]
                    observations = tracks[camera_id].get(frame_index, [])
                    projection_states[camera_id] = live_projection.update(
                        camera_id,
                        {
                            "source_id": source_id,
                            "input_frame_index": frame_index,
                            "source_pts_ns": pts,
                            "objects": [
                                {"tracker_id": row["object_id"],
                                 "left": row["box"][0], "top": row["box"][1],
                                 "width": row["box"][2], "height": row["box"][3]}
                                for row in observations
                            ],
                        },
                        cached_source_id=source_id,
                        cached_frame_index=frame_index,
                        cached_source_pts_ns=pts,
                        now_mono_ns=(frame_index + 1) * 50_000_000,
                    )
                    panels[camera_id], _anchors = _camera_panel(
                        scaled, camera_id, observations,
                        oob_lookup.get((camera_id, frame_index), set()),
                        provisional_global_ids.get(camera_id, {}))

                map_view = cv2.resize(map_meta["image"], (480, 360), interpolation=cv2.INTER_AREA)
                cv2.rectangle(map_view, (0, 0), (479, 50), (14, 14, 14), -1)
                cv2.putText(map_view, "EXPERIMENTAL / UNCALIBRATED", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(map_view, "AUTO PROVISIONAL IDS - OFFLINE ONLY", (8, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.39, (80, 230, 255), 1, cv2.LINE_AA)
                frame_errors = {"CAM-02": 0, "CAM-05": 0}
                frame_invalid = {"CAM-02": 0, "CAM-05": 0}
                map_labels = []
                for camera_id in ("CAM-02", "CAM-05"):
                    scale_x, scale_y = 480 / map_meta["width"], 360 / map_meta["height"]
                    state = projection_states[camera_id]
                    frame_errors[camera_id] = len(state["out_of_bounds_ids"])
                    frame_invalid[camera_id] = len(state["invalid_ids"])
                    out_of_bounds[camera_id] += frame_errors[camera_id]
                    invalid_projection[camera_id] += frame_invalid[camera_id]
                    for marker in state["markers"]:
                        projected[camera_id] += 1
                        anchor = marker["layout_xy"]
                        point = (round(anchor[0] * scale_x), round(anchor[1] * scale_y))
                        color = CAMERA_COLORS[camera_id]
                        trail = marker["trail"]
                        if len(trail) >= 2:
                            trail_points = np.asarray([
                                (round(x * scale_x), round(y * scale_y)) for x, y in trail
                            ], dtype=np.int32)
                            cv2.polylines(map_view, [trail_points], False, color, 2, cv2.LINE_AA)
                        cv2.circle(map_view, point, 6, color, -1)
                        local_id = marker["local_tracker_id"]
                        if marker["jump_candidate"]:
                            cv2.circle(map_view, point, 11, (0, 0, 255), 2)
                            cv2.putText(map_view, "JUMP?", (point[0] + 7, point[1] + 13),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
                        if (camera_id, frame_index, local_id) in jump_lookup:
                            cv2.circle(map_view, point, 11, (0, 0, 255), 2)
                            cv2.putText(map_view, "JUMP?", (point[0] + 7, point[1] + 13),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
                        if (camera_id, frame_index, local_id) in handoff_lookup:
                            cv2.circle(map_view, point, 10, (0, 165, 255), 2)
                        provisional = provisional_global_ids.get(camera_id, {}).get(local_id)
                        if provisional:
                            marker_label = f"{provisional['provisional_global_id']}? {camera_id[-2:]}:{local_id}"
                        else:
                            marker_label = f"{camera_id[-2:]}:{local_id}"
                        map_labels.append((point, marker_label, color))
                labels_drawn = _draw_map_labels(map_view, map_labels)
                map_labels_drawn_total += labels_drawn
                map_labels_suppressed_total += len(map_labels) - labels_drawn
                cv2.putText(map_view,
                            f"OOB 02:{frame_errors['CAM-02']} 05:{frame_errors['CAM-05']}  INVALID 02:{frame_invalid['CAM-02']} 05:{frame_invalid['CAM-05']}",
                            (8, 346), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (40, 40, 230) if any(frame_errors.values()) or any(frame_invalid.values()) else (245, 245, 245),
                            1, cv2.LINE_AA)
                matched_rows = [entry for camera_entries in provisional_global_ids.values()
                                for entry in camera_entries.values()]
                if matched_rows:
                    unique_matches = {}
                    for entry in matched_rows:
                        unique_matches.setdefault(str(entry["provisional_global_id"]), entry)
                    score_text = " ".join(
                        f"{entry['provisional_global_id']} cos={float(entry['median_cosine_similarity']):.3f}*"
                        for entry in unique_matches.values()
                    )
                    cv2.putText(map_view, score_text[:58], (8, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 230, 255), 1, cv2.LINE_AA)
                    cv2.putText(map_view, "*similarity, not probability; match unverified", (8, 82),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (245, 245, 245), 1, cv2.LINE_AA)
                pair = np.hstack((panels["CAM-02"], panels["CAM-05"], map_view))
                writer.write(pair)
                observation_count = sum(len(tracks[camera_id].get(frame_index, []))
                                         for camera_id in ("CAM-02", "CAM-05"))
                matched_camera_count = sum(
                    any(str(row["object_id"]) in provisional_global_ids.get(camera_id, {})
                        for row in tracks[camera_id].get(frame_index, []))
                    for camera_id in ("CAM-02", "CAM-05")
                )
                representative_rank = (matched_camera_count, observation_count)
                if representative_rank > best_representative_rank:
                    best_representative_rank = representative_rank
                    best_observation_count = observation_count
                    best_review_frame_index = frame_index
                    best_review_frame = pair.copy()
                timeline.write(json.dumps({
                    "review_frame": frame_index,
                    "cam02_source_frame": frame_index,
                    "cam02_source_pts_ns": local_pts["CAM-02"][frame_index],
                    "cam02_local_track_observations": len(tracks["CAM-02"].get(frame_index, [])),
                    "cam05_source_frame": frame_index,
                    "cam05_source_pts_ns": local_pts["CAM-05"][frame_index],
                    "cam05_local_track_observations": len(tracks["CAM-05"].get(frame_index, [])),
                    "out_of_map_projection_count": frame_errors,
                    "invalid_projection_count": frame_invalid,
                    "map_labels_suppressed": len(map_labels) - labels_drawn,
                    "projection_join_status": {
                        camera_id: {
                            "source_id": projection_states[camera_id]["source_id"],
                            "frame_index": projection_states[camera_id]["frame_index"],
                            "source_pts_ns": projection_states[camera_id]["source_pts_ns"],
                        } for camera_id in ("CAM-02", "CAM-05")
                    },
                    "projected_jump_candidates": {
                        camera_id: sum(1 for item in motion["cameras"][camera_id]["large_projected_jump_candidates"]
                                       if item["to_frame"] == frame_index)
                        for camera_id in ("CAM-02", "CAM-05")
                    },
                    "possible_local_id_replacements": {
                        camera_id: sum(1 for item in motion["cameras"][camera_id]["possible_local_id_replacement_candidates"]
                                       if item["to_frame"] == frame_index)
                        for camera_id in ("CAM-02", "CAM-05")
                    },
                    "provisional_global_ids": {
                        camera_id: {
                            str(row["object_id"]): provisional_global_ids.get(camera_id, {}).get(
                                str(row["object_id"]), {}).get("provisional_global_id")
                            for row in tracks[camera_id].get(frame_index, [])
                            if str(row["object_id"]) in provisional_global_ids.get(camera_id, {})
                        }
                        for camera_id in ("CAM-02", "CAM-05")
                    },
                    "cross_camera_sync_claim": False,
                }, separators=(",", ":")) + "\n")
    finally:
        writer.release()
        for cap in captures.values():
            cap.release()

    screenshot_path = output_dir / "representative-review-frame.png"
    if best_review_frame is None or not cv2.imwrite(str(screenshot_path), best_review_frame):
        raise RuntimeError("could not save a representative frame-matched review image")

    manifest = {
        "status": "diagnostic_only",
        "video": str(video_path.resolve()),
        "timeline": str(timeline_path.resolve()),
        "frame_count": frame_count,
        "review_fps": output_fps,
        "review_duration_seconds": frame_count / output_fps,
        "source_frame_counts": source_frame_counts,
        "source_total_frame_counts": {"cam02": len(local_pts["CAM-02"]),
                                       "cam05": len(local_pts["CAM-05"])},
        "omitted_tail_frames_from_longer_source": {
            "cam02": max(0, len(local_pts["CAM-02"]) - frame_count),
            "cam05": max(0, len(local_pts["CAM-05"]) - frame_count),
        },
        "source_hashes": {key: file_sha256(path) for key, path in video_paths.items()},
        "input_coordinate_space": LIVE_NATIVE_SPACE,
        "camera_ids_are_local_and_distinct": True,
        "provisional_global_ids": provisional_global_ids,
        "automatic_cross_camera_id_status": "OFFLINE PROVISIONAL ONLY; not ground-truth verified",
        "cross_camera_identity_probability": "NOT CALIBRATED",
        "cross_camera_sync": "UNVERIFIED; paired by ordinal source frame index only",
        "map_status": "EXPERIMENTAL / UNCALIBRATED",
        "out_of_bounds_projection_count": out_of_bounds,
        "invalid_projection_count": invalid_projection,
        "projected_anchor_count": projected,
        "shared_live_projection_counters": live_projection.counters,
        "map_labels_drawn_total": map_labels_drawn_total,
        "map_labels_suppressed_total": map_labels_suppressed_total,
        "homography_diagnostics_pixels_only": diagnostics,
        "projected_motion_diagnostics": motion,
        "representative_review_frame": str(screenshot_path.resolve()),
        "representative_review_frame_index": best_review_frame_index,
        "representative_review_frame_observation_count": best_observation_count,
    }
    manifest_path = output_dir / "preview_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(video_path, 0o600)
    os.chmod(timeline_path, 0o600)
    os.chmod(manifest_path, 0o600)
    os.chmod(screenshot_path, 0o600)
    motion_path = output_dir / "movement_diagnostics.json"
    with motion_path.open("x", encoding="utf-8") as stream:
        json.dump(motion, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(motion_path, 0o600)
    return manifest


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--cam02", type=Path,
                        default=root / "calibration_recordings/entrance/entrance-20261009T130526Z/camera_1.mp4")
    parser.add_argument("--cam05", type=Path,
                        default=root / "calibration_recordings/entrance/entrance-20261009T130526Z/camera_2.mp4")
    parser.add_argument("--layout", type=Path,
                        default=Path("/home/apsidal/auto-magic-calib/projects/project_20261019_101845_7932/manual_adjustment/layout.png"))
    parser.add_argument("--tracking-dir", type=Path,
                        default=root / ".runtime/reliability-entrance-r9-replay-physical-map-20261010-r1")
    parser.add_argument("--output-dir", type=Path,
                        default=root / ".runtime/entrance-live-native-map-review")
    parser.add_argument("--frames", type=int)
    args = parser.parse_args(argv)
    manifest = build_preview(
        alignment_path=args.alignment,
        video_paths={"cam02": args.cam02, "cam05": args.cam05},
        layout_path=args.layout,
        tracking_dir=args.tracking_dir,
        output_dir=args.output_dir,
        frame_limit=args.frames,
    )
    print(json.dumps({key: manifest[key] for key in (
        "video", "frame_count", "review_duration_seconds", "out_of_bounds_projection_count",
        "projected_anchor_count", "cross_camera_sync", "map_status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
