"""Conservative offline OSNet association for provisional CAM-02/CAM-05 IDs.

This module consumes only the preserved OSNet feature bundle and the exact
local NvDCF/source-frame evidence used to extract those features.  It never
reads review-decision files, changes tracker records, writes production
identity state, or uses unverified map geometry / cross-camera timing as
identity truth.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
from typing import Any

import cv2
import numpy as np

from services.mv3dt_staging.native_map_preview import (
    build_preview,
    load_tracking_crosswalk,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / ".runtime/entrance-offline-global-id-review-20261021-r1"
DEFAULT_TRACKING = ROOT / ".runtime/reliability-entrance-r9-replay-physical-map-20261010-r1"
DEFAULT_CAM02 = ROOT / "calibration_recordings/entrance/entrance-20261009T130526Z/camera_1.mp4"
DEFAULT_CAM05 = ROOT / "calibration_recordings/entrance/entrance-20261009T130526Z/camera_2.mp4"
DEFAULT_ALIGNMENT = ROOT / ".runtime/entrance-calibration-editor/live-native-versions/LIVE_NATIVE_ALIGNMENT-v0007-20261021T123354Z.json"
DEFAULT_LAYOUT = Path("/home/apsidal/auto-magic-calib/projects/project_20261019_101845_7932/manual_adjustment/layout.png")
CAMERAS = ("CAM-02", "CAM-05")
SOURCE_IDS = {0: "CAM-02", 1: "CAM-05"}
FEATURE_DIMENSION = 512
EXPECTED_BUNDLE_HASHES = {
    "manifest.json": "7add5886b7895db51cd20a05b45f82acf4e71a48c55c350d082e2b3c3f6d385c",
    "osnet_embeddings.npy": "cc6cbc1b11c725dffa003d36a9629215f9106fcfc41777afa91158e6b21b04d5",
    "embedding_samples.jsonl": "72bb4325ec2c86f11534ee1b61f02c3183dc9976fb43c99db77a1e796fec5f43",
}
EXPECTED_MODEL_SHA256 = "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f"
EXPECTED_SOURCE_HASHES = {
    "CAM-02": "0216d2f8cf42996ec603a52882c2d12ee62974788057ffc7290c3636009a01b0",
    "CAM-05": "0cd0ae6ddd31272d1ab049f7d81f12d199f0653ab2d0cd953fa6de87453f4370",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected an object")
            result.append(value)
    return result


def _box_iou(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = (float(value) for value in left)
    rx, ry, rw, rh = (float(value) for value in right)
    intersection_w = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
    intersection_h = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    intersection = intersection_w * intersection_h
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def _track_groups(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        camera = str(record.get("camera_id"))
        local_id = str(record.get("local_track_id"))
        if camera not in CAMERAS or not local_id:
            raise ValueError("feature row has an unknown camera or empty local tracker ID")
        grouped[(camera, local_id)].append(index)
    return grouped


def _track_observations(
    tracks: dict[str, dict[int, list[dict[str, Any]]]],
) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    result: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
        camera: defaultdict(dict) for camera in CAMERAS
    }
    for camera in CAMERAS:
        for frame, rows in tracks[camera].items():
            for row in rows:
                local_id = str(row["object_id"])
                result[camera][local_id][int(frame)] = row
    return result


def _validate_feature_lineage(
    vectors: np.ndarray,
    records: list[dict[str, Any]],
    *,
    bundle_dir: Path,
    source_pts: dict[str, dict[int, int]],
    tracks: dict[str, dict[int, list[dict[str, Any]]]],
) -> None:
    if vectors.shape != (len(records), FEATURE_DIMENSION):
        raise ValueError("OSNet feature matrix must be one 512-D row per feature record")
    if not np.isfinite(vectors).all():
        raise ValueError("OSNet feature matrix contains nonfinite values")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("OSNet feature vectors are not L2-normalized")
    if len({str(row.get("sample_id")) for row in records}) != len(records):
        raise ValueError("feature sample IDs are not unique")

    observations = _track_observations(tracks)
    for row in records:
        camera = row["camera_id"]
        expected_source_id = 0 if camera == "CAM-02" else 1
        frame = int(row["frame"])
        local_id = str(row["local_track_id"])
        if int(row.get("source_id", -1)) != expected_source_id:
            raise ValueError(f"{camera} feature sample has the wrong source namespace")
        if frame not in source_pts[camera] or int(row["source_pts_ns"]) != source_pts[camera][frame]:
            raise ValueError(f"{camera} feature sample lacks an exact source-frame/PTS join")
        observation = observations[camera].get(local_id, {}).get(frame)
        if observation is None:
            raise ValueError(f"{camera} feature sample does not map to a tracker observation")
        if not np.allclose(observation["box"], row["bbox_inference_xywh"], atol=1e-3):
            raise ValueError(f"{camera} feature sample bbox differs from its tracker row")
        crop_path = (bundle_dir / str(row.get("crop_path", ""))).resolve()
        if not crop_path.is_relative_to(bundle_dir.resolve()) or not crop_path.is_file():
            raise ValueError("feature crop path is missing or escapes its private bundle")
        crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if crop is None or crop.size == 0:
            raise ValueError(f"{camera} supporting crop is unreadable")
        expected_wh = row.get("crop_dimensions")
        if expected_wh and [int(crop.shape[1]), int(crop.shape[0])] != [int(x) for x in expected_wh]:
            raise ValueError(f"{camera} supporting crop dimensions differ from its metadata")


def _mine_same_camera_controls(
    vectors: np.ndarray,
    grouped: dict[tuple[str, str], list[int]],
    tracks: dict[str, dict[int, list[dict[str, Any]]]],
    *,
    minimum_separated_frames: int = 8,
) -> tuple[list[dict[str, Any]], set[tuple[str, str, str]]]:
    """Mine conservative same-camera appearance-separation controls.

    Two local IDs are a control only when they have at least the specified
    number of shared frames where boxes barely overlap and their centers are
    separated by 0.75 of the taller box. This is only a score-separation
    reference, not a DIFFERENT_PEOPLE label: duplicate tracker observations
    can still occur even when boxes are separated.
    """
    observations = _track_observations(tracks)
    controls: list[dict[str, Any]] = []
    duplicate_pairs: set[tuple[str, str, str]] = set()
    for camera in CAMERAS:
        ids = sorted((local_id for cam, local_id in grouped if cam == camera),
                     key=lambda value: (int(value) if value.isdigit() else 10**12, value))
        for index, left_id in enumerate(ids):
            for right_id in ids[index + 1:]:
                shared_frames = sorted(set(observations[camera].get(left_id, {}))
                                       & set(observations[camera].get(right_id, {})))
                separated_frames = 0
                duplicate_overlap_frames = 0
                for frame in shared_frames:
                    left = observations[camera][left_id][frame]["box"]
                    right = observations[camera][right_id][frame]["box"]
                    overlap = _box_iou(left, right)
                    if overlap >= 0.5:
                        duplicate_overlap_frames += 1
                    lx, ly, lw, lh = (float(value) for value in left)
                    rx, ry, rw, rh = (float(value) for value in right)
                    center_distance = math.dist((lx + lw / 2, ly + lh / 2),
                                                (rx + rw / 2, ry + rh / 2))
                    if overlap <= 0.05 and center_distance >= 0.75 * max(lh, rh):
                        separated_frames += 1
                key = (camera, left_id, right_id)
                if duplicate_overlap_frames >= 2:
                    duplicate_pairs.add(key)
                if separated_frames < minimum_separated_frames:
                    continue
                left_group = grouped[(camera, left_id)]
                right_group = grouped[(camera, right_id)]
                similarity = np.clip(vectors[left_group] @ vectors[right_group].T, -1.0, 1.0)
                controls.append({
                    "camera_id": camera,
                    "local_track_ids": [left_id, right_id],
                    "separated_shared_frame_count": separated_frames,
                    "high_overlap_shared_frame_count": duplicate_overlap_frames,
                    "cosine_similarity_median": float(np.median(similarity)),
                    "cosine_similarity_min": float(np.min(similarity)),
                    "semantics": (
                        "appearance-separation control only; not a DIFFERENT_PEOPLE label; "
                        "duplicate tracking remains possible"
                    ),
                })
    return controls, duplicate_pairs


def score_and_assign_provisional_ids(
    vectors: np.ndarray,
    records: list[dict[str, Any]],
    tracks: dict[str, dict[int, list[dict[str, Any]]]],
    *,
    minimum_crops_per_track: int = 2,
    minimum_separated_control_frames: int = 8,
    control_reference_max: float | None = None,
    control_reference_source: str | None = None,
    duplicate_track_conflicts: set[tuple[str, str, str]] | None = None,
    diagnostic_logging: bool = False,
    diagnostic_now_ns: int | None = None,
) -> dict[str, Any]:
    """Apply mutual-best, gallery-consistency, margin, and control-separation gates.

    ``control_reference_max`` is an optional, already-audited same-camera
    appearance-control envelope for bounded live shadow mode.  Such controls
    are only a conservative score reference: they are not DIFFERENT_PEOPLE
    labels and may themselves contain duplicate-track observations.
    """
    scoring_started = time.perf_counter()
    if control_reference_max is not None and (
        not math.isfinite(float(control_reference_max))
        or not -1.0 <= float(control_reference_max) <= 1.0
    ):
        raise ValueError("control reference must be a finite cosine in [-1, 1]")
    if control_reference_source is not None and not control_reference_source.strip():
        raise ValueError("control reference source cannot be empty")
    if vectors.ndim != 2 or vectors.shape[0] != len(records):
        raise ValueError("embedding rows and feature records do not align")
    if vectors.shape[1] != FEATURE_DIMENSION or not np.isfinite(vectors).all():
        raise ValueError("features must be finite 512-D OSNet vectors")
    if not np.allclose(np.linalg.norm(vectors.astype(np.float64), axis=1), 1.0, atol=1e-4):
        raise ValueError("features must be L2 normalized")
    grouped = _track_groups(records)
    controls, duplicate_pairs = _mine_same_camera_controls(
        vectors, grouped, tracks,
        minimum_separated_frames=minimum_separated_control_frames,
    )
    if duplicate_track_conflicts:
        duplicate_pairs.update(
            (str(camera), str(left), str(right))
            for camera, left, right in duplicate_track_conflicts
        )
    if not controls and control_reference_max is None:
        return {
            "matches": [], "candidate_pairs": [],
            "same_camera_negative_like_controls": [],
            "same_camera_separation_controls": [],
            "same_camera_control_summary": {
                "count": 0,
                "semantics": "no automatic control pair met the screening rule",
            },
            "duplicate_track_pairs": [],
            "potential_local_fragmentation_candidates": [],
            "one_to_many_conflicts_rejected": [],
            "rule_contract": {
                "minimum_crops_per_track": minimum_crops_per_track,
                "production_identity_write": False,
                "identity_probability_calibrated": False,
                "saved_human_review_decisions_read": False,
            },
            "status": "ABSTAIN_NO_AUTOMATIC_CONTROLS",
            "reason": "No same-camera simultaneous distinct-track controls support an appearance separation check.",
            "scoring_wall_ms": (time.perf_counter() - scoring_started) * 1000.0,
            **({"decision_trace": []} if diagnostic_logging else {}),
        }
    observed_control_max = max(
        (row["cosine_similarity_median"] for row in controls), default=None
    )
    control_values = [value for value in (observed_control_max, control_reference_max)
                      if value is not None]
    control_max = max(float(value) for value in control_values)
    cam02_tracks = sorted((key for key in grouped if key[0] == "CAM-02"),
                          key=lambda key: (int(key[1]) if key[1].isdigit() else 10**12, key[1]))
    cam05_tracks = sorted((key for key in grouped if key[0] == "CAM-05"),
                          key=lambda key: (int(key[1]) if key[1].isdigit() else 10**12, key[1]))
    pair_rows: list[dict[str, Any]] = []
    similarity_matrices: dict[tuple[str, str], np.ndarray] = {}
    for left_key in cam02_tracks:
        for right_key in cam05_tracks:
            matrix = np.clip(vectors[grouped[left_key]] @ vectors[grouped[right_key]].T,
                             -1.0, 1.0).astype(np.float64)
            similarity_matrices[(left_key[1], right_key[1])] = matrix
            left_profile = matrix.mean(axis=1)
            right_profile = matrix.mean(axis=0)
            pair_rows.append({
                "camera_local_pair": {"CAM-02": left_key[1], "CAM-05": right_key[1]},
                "cosine_similarity_median": float(np.median(matrix)),
                "cosine_similarity_min": float(np.min(matrix)),
                "cosine_similarity_max": float(np.max(matrix)),
                "gallery_comparisons": int(matrix.size),
                "samples": {"CAM-02": int(matrix.shape[0]), "CAM-05": int(matrix.shape[1])},
                "gallery_mean_score_sd": {
                    "CAM-02": float(np.std(left_profile, ddof=1)) if len(left_profile) > 1 else None,
                    "CAM-05": float(np.std(right_profile, ddof=1)) if len(right_profile) > 1 else None,
                },
                "gallery_sample_provenance": {
                    camera: [
                        {"sample_id": records[sample_index].get("sample_id"),
                         "frame": int(records[sample_index]["frame"]),
                         "source_pts_ns": int(records[sample_index]["source_pts_ns"]),
                         "crop_path": records[sample_index].get("crop_path")}
                        for sample_index in grouped[key]
                    ]
                    for camera, key in (("CAM-02", left_key), ("CAM-05", right_key))
                },
                "confidence_semantics": "cosine similarity is an uncalibrated appearance score, not identity probability",
            })

    for camera in CAMERAS:
        own_field = "CAM-02" if camera == "CAM-02" else "CAM-05"
        other_field = "CAM-05" if camera == "CAM-02" else "CAM-02"
        by_local: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pair in pair_rows:
            by_local[pair["camera_local_pair"][own_field]].append(pair)
        for rows in by_local.values():
            rows.sort(key=lambda row: row["cosine_similarity_median"], reverse=True)
            for rank, row in enumerate(rows, start=1):
                row[f"rank_for_{camera}"] = rank
                row[f"margin_to_next_for_{camera}"] = (
                    row["cosine_similarity_median"] - rows[rank]["cosine_similarity_median"]
                    if rank < len(rows) else None
                )
    for pair in pair_rows:
        left_id = pair["camera_local_pair"]["CAM-02"]
        right_id = pair["camera_local_pair"]["CAM-05"]
        duplicate_conflict = any(
            (camera == "CAM-02" and left_id in (a, b))
            or (camera == "CAM-05" and right_id in (a, b))
            for camera, a, b in duplicate_pairs
        )
        margins_pass = True
        for camera in CAMERAS:
            margin = pair.get(f"margin_to_next_for_{camera}")
            dispersion = pair["gallery_mean_score_sd"][camera]
            if margin is None or dispersion is None or margin <= 2.0 * dispersion:
                margins_pass = False
        control_gate_name = (
            "every_gallery_similarity_exceeds_mined_control_max"
            if observed_control_max is not None
            else "every_gallery_similarity_exceeds_preserved_control_reference"
        )
        gates = {
            "minimum_gallery_size": (pair["samples"]["CAM-02"] >= minimum_crops_per_track
                                     and pair["samples"]["CAM-05"] >= minimum_crops_per_track),
            "reciprocal_best_match": pair.get("rank_for_CAM-02") == 1 and pair.get("rank_for_CAM-05") == 1,
            "positive_margin_exceeds_twice_observed_gallery_dispersion": margins_pass,
            "no_repeated_high_overlap_local_track_conflict": not duplicate_conflict,
        }
        gates[control_gate_name] = pair["cosine_similarity_min"] > control_max
        pair["automatic_candidate_gates"] = gates
        pair["hardest_control_median_similarity"] = control_max
        pair["hardest_control_margin_min"] = pair["cosine_similarity_min"] - control_max
        pair["automatic_candidate"] = all(gates.values())
        pair["status"] = "ELIGIBLE_PROVISIONAL_MATCH" if pair["automatic_candidate"] else "AMBIGUOUS_OR_REJECTED"

    # Deterministic one-to-one assignment. Mutual-best already rules out most
    # collisions, while this explicit guard prevents future policy changes from
    # assigning a local track twice.
    accepted_rows = sorted((row for row in pair_rows if row["automatic_candidate"]),
                           key=lambda row: row["cosine_similarity_median"], reverse=True)
    used02: set[str] = set()
    used05: set[str] = set()
    matches = []
    conflicts = []
    for row in accepted_rows:
        left_id = row["camera_local_pair"]["CAM-02"]
        right_id = row["camera_local_pair"]["CAM-05"]
        if left_id in used02 or right_id in used05:
            conflicts.append({
                "camera_local_pair": row["camera_local_pair"],
                "status": "rejected_one_to_many_assignment_conflict",
            })
            continue
        global_id = f"G{len(matches) + 1:03d}"
        row["provisional_global_id"] = global_id
        row["assignment_status"] = "AUTOMATIC_PROVISIONAL_ONLY"
        matches.append(row)
        used02.add(left_id)
        used05.add(right_id)

    # Possible local fragmentation candidates are surfaced only for diagnosis;
    # they never share an ID automatically.
    observations = _track_observations(tracks)
    fragmentation_candidates = []
    for camera in CAMERAS:
        ids = sorted((local_id for cam, local_id in grouped if cam == camera),
                     key=lambda value: (int(value) if value.isdigit() else 10**12, value))
        for index, left_id in enumerate(ids):
            for right_id in ids[index + 1:]:
                common_frames = set(observations[camera].get(left_id, {})) & set(observations[camera].get(right_id, {}))
                if common_frames:
                    continue
                matrix = vectors[grouped[(camera, left_id)]] @ vectors[grouped[(camera, right_id)]].T
                median = float(np.median(matrix))
                if median > control_max:
                    fragmentation_candidates.append({
                        "camera_id": camera,
                        "local_track_ids": [left_id, right_id],
                        "cosine_similarity_median": median,
                        "hardest_control_median_similarity": control_max,
                        "status": "possible_fragment_or_lookalike_review_only_not_merged",
                    })

    result = {
        "status": "PROVISIONAL_MATCHES_ASSIGNED" if matches else "NO_MATCH_PASSED_CONSERVATIVE_GATES",
        "matches": matches,
        "candidate_pairs": sorted(pair_rows, key=lambda row: row["cosine_similarity_median"], reverse=True),
        "same_camera_negative_like_controls": controls,
        "same_camera_separation_controls": controls,
        "same_camera_control_summary": {
            "count": len(controls),
            "median_similarity_median": (
                float(np.median([x["cosine_similarity_median"] for x in controls]))
                if controls else None
            ),
            "p90_similarity_median": (
                float(np.percentile([x["cosine_similarity_median"] for x in controls], 90))
                if controls else None
            ),
            "p95_similarity_median": (
                float(np.percentile([x["cosine_similarity_median"] for x in controls], 95))
                if controls else None
            ),
            "maximum_similarity_median": observed_control_max,
            "preserved_reference_maximum": control_reference_max,
            "control_maximum_used_for_screening": control_max,
            "reference_source": control_reference_source,
            "semantics": (
                "same-camera appearance-separation controls only; not independently verified "
                "DIFFERENT_PEOPLE labels; duplicate tracking is still possible"
            ),
            "geometry_rule": "shared frames with IoU <= 0.05 and center separation >= 0.75 x taller bbox",
            "minimum_shared_frames": minimum_separated_control_frames,
        },
        "duplicate_track_pairs": [
            {"camera_id": camera, "local_track_ids": [a, b],
             "status": "repeated_high_overlap_box_conflict_do_not_merge"}
            for camera, a, b in sorted(duplicate_pairs)
        ],
        "potential_local_fragmentation_candidates": fragmentation_candidates,
        "one_to_many_conflicts_rejected": conflicts,
        "rule_contract": {
            "minimum_crops_per_track": minimum_crops_per_track,
            "mutual_best_match": True,
            "gallery_minimum_must_exceed_control_reference_maximum": True,
            "same_camera_controls_are_ground_truth_negatives": False,
            "preserved_control_reference_source": control_reference_source,
            "rank_margin_must_exceed_two_times_observed_gallery_score_sd": True,
            "one_to_one_assignment": True,
            "cross_camera_pts_used_as_sync_gate": False,
            "experimental_map_geometry_used_as_identity_gate": False,
            "saved_human_review_decisions_read": False,
            "identity_probability_calibrated": False,
            "false_merge_rate_measured": False,
            "production_identity_write": False,
        },
    }
    if diagnostic_logging:
        trace_rows = []
        for pair in pair_rows:
            local_pair = pair["camera_local_pair"]
            provenance = {}
            for camera, local_id in (("CAM-02", local_pair["CAM-02"]),
                                     ("CAM-05", local_pair["CAM-05"])):
                key = (camera, local_id)
                sample_indexes = grouped.get(key, [])
                samples = [records[index] for index in sample_indexes]
                latest = max(
                    samples,
                    key=lambda row: (int(row.get("observed_monotonic_ns", -1)),
                                     int(row.get("frame", -1))),
                    default={},
                )
                oldest = min(samples, key=lambda row: int(row.get("frame", -1)), default={})
                observed_ns = latest.get("observed_monotonic_ns")
                age_ms = None
                if diagnostic_now_ns is not None and observed_ns is not None:
                    age_ms = max(0.0, (int(diagnostic_now_ns) - int(observed_ns)) / 1e6)
                provenance[camera] = {
                    "camera_id": camera,
                    "source_id": latest.get("source_id"),
                    "local_track_id": str(local_id),
                    "latest_frame": int(latest["frame"]) if latest.get("frame") is not None else None,
                    "latest_source_pts_ns": (
                        int(latest["source_pts_ns"])
                        if latest.get("source_pts_ns") is not None else None
                    ),
                    "oldest_frame": int(oldest["frame"]) if oldest.get("frame") is not None else None,
                    "gallery_sample_count": len(samples),
                    "freshest_sample_age_ms_monotonic": age_ms,
                    "freshness_clock": "process_monotonic; source PTS remains separate",
                }
            competitor_scores = {}
            competitor_pairs = {}
            best_for_camera = {}
            for camera in CAMERAS:
                local_id = local_pair[camera]
                alternatives = [
                    other
                    for other in pair_rows
                    if other is not pair and other["camera_local_pair"][camera] == local_id
                ]
                competitor = max(
                    alternatives,
                    key=lambda other: float(other["cosine_similarity_median"]),
                    default=None,
                )
                competitor_scores[camera] = (
                    float(competitor["cosine_similarity_median"]) if competitor else None
                )
                competitor_pairs[camera] = (
                    dict(competitor["camera_local_pair"]) if competitor else None
                )
                best_for_camera[camera] = pair.get(f"rank_for_{camera}") == 1
            gates = dict(pair.get("automatic_candidate_gates", {}))
            failed_gates = [name for name, passed in gates.items() if not passed]
            trace_rows.append({
                "camera_local_pair": dict(local_pair),
                "source_observations": provenance,
                "candidate_global_id": pair.get("provisional_global_id"),
                "best_similarity_score": float(pair["cosine_similarity_median"]),
                "gallery_minimum_similarity_score": float(pair["cosine_similarity_min"]),
                "competing_similarity_score_by_camera": competitor_scores,
                "competing_track_pair_by_camera": competitor_pairs,
                "best_for_camera": best_for_camera,
                "reciprocal_match": {
                    "CAM-02": pair.get("rank_for_CAM-02") == 1,
                    "CAM-05": pair.get("rank_for_CAM-05") == 1,
                },
                "margin_result": {
                    camera: {
                        "margin": pair.get(f"margin_to_next_for_{camera}"),
                        "observed_dispersion": pair["gallery_mean_score_sd"].get(camera),
                        "required_strictly_greater_than": (
                            2.0 * pair["gallery_mean_score_sd"][camera]
                            if pair["gallery_mean_score_sd"].get(camera) is not None else None
                        ),
                        "passed": (
                            pair.get(f"margin_to_next_for_{camera}") is not None
                            and pair["gallery_mean_score_sd"].get(camera) is not None
                            and pair[f"margin_to_next_for_{camera}"]
                            > 2.0 * pair["gallery_mean_score_sd"][camera]
                        ),
                    }
                    for camera in CAMERAS
                },
                "gates": gates,
                "decision": "ACCEPTED_BY_MATCHER" if pair.get("automatic_candidate") else "REJECTED_OR_AMBIGUOUS",
                "reason": "all existing matcher gates passed" if not failed_gates
                          else "failed existing gate(s): " + ", ".join(failed_gates),
                "failed_gates": failed_gates,
            })
        result["decision_trace"] = trace_rows
    result["scoring_wall_ms"] = (time.perf_counter() - scoring_started) * 1000.0
    return result


def _summarize_track_durations(
    tracks: dict[str, dict[int, list[dict[str, Any]]]],
    source_pts: dict[str, dict[int, int]],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_camera: dict[str, dict[str, list[int]]] = {camera: defaultdict(list) for camera in CAMERAS}
    for camera in CAMERAS:
        for frame, rows in tracks[camera].items():
            for row in rows:
                by_camera[camera][str(row["object_id"])].append(int(frame))
    summary: dict[str, dict[str, dict[str, Any]]] = {camera: {} for camera in CAMERAS}
    for camera in CAMERAS:
        for local_id, frame_rows in by_camera[camera].items():
            indexes = sorted(set(frame_rows))
            start_pts = source_pts[camera][indexes[0]]
            end_pts = source_pts[camera][indexes[-1]]
            summary[camera][local_id] = {
                "observation_count": len(frame_rows),
                "first_frame": indexes[0],
                "last_frame": indexes[-1],
                "source_pts_start_ns": start_pts,
                "source_pts_end_ns": end_pts,
                "source_relative_duration_sec": max(0.0, (end_pts - start_pts) / 1e9),
                "timing_semantics": "per-file relative timing only; cross-camera exposure sync is unverified",
            }
    return summary


def load_verified_inputs(
    *,
    bundle_dir: Path,
    tracking_dir: Path,
    video_paths: dict[str, Path],
    enforce_preserved_hashes: bool = True,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]], dict[str, dict[int, list[dict[str, Any]]]], dict[str, dict[int, int]]]:
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "manifest.json"
    vectors_path = bundle_dir / "osnet_embeddings.npy"
    records_path = bundle_dir / "embedding_samples.jsonl"
    paths = {"manifest.json": manifest_path, "osnet_embeddings.npy": vectors_path,
             "embedding_samples.jsonl": records_path}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"missing preserved OSNet artifact: {name}")
        if enforce_preserved_hashes and sha256(path) != EXPECTED_BUNDLE_HASHES[name]:
            raise ValueError(f"preserved OSNet artifact hash mismatch: {name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("feature_shape", [None, None])[1] != FEATURE_DIMENSION:
        raise ValueError("feature bundle is not 512-dimensional")
    if manifest.get("model", {}).get("checkpoint_sha256") != EXPECTED_MODEL_SHA256:
        raise ValueError("feature manifest does not name the accepted OSNet checkpoint")
    model_path = Path(manifest["model"]["checkpoint_path"])
    if not model_path.is_file() or sha256(model_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("accepted OSNet checkpoint is missing or changed")
    video_paths = {camera: Path(path).resolve() for camera, path in video_paths.items()}
    if set(video_paths) != set(CAMERAS):
        raise ValueError("exactly CAM-02 and CAM-05 source videos are required")
    for camera, key in (("CAM-02", "cam02"), ("CAM-05", "cam05")):
        path = video_paths[camera]
        if not path.is_file() or sha256(path) != EXPECTED_SOURCE_HASHES[camera]:
            raise ValueError(f"{camera} source is missing or does not match the preserved hash")
        if manifest.get("source_hashes", {}).get(camera) != EXPECTED_SOURCE_HASHES[camera]:
            raise ValueError(f"feature bundle source hash disagrees for {camera}")
    tracks, source_pts, _provenance = load_tracking_crosswalk(
        Path(tracking_dir).resolve(), {"cam02": video_paths["CAM-02"], "cam05": video_paths["CAM-05"]})
    vectors = np.load(vectors_path, allow_pickle=False)
    records = _read_jsonl(records_path)
    _validate_feature_lineage(vectors, records, bundle_dir=bundle_dir,
                              source_pts=source_pts, tracks=tracks)
    return manifest, vectors, records, tracks, source_pts


def _write_private_json(path: Path, data: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def resolve_provisional_assignments(output_dir: Path) -> dict[str, dict[str, str] | None]:
    """Replay the append-only revision ledger over the original proposals."""
    output_dir = Path(output_dir).resolve()
    assignments = json.loads((output_dir / "automatic_global_id_assignments.json").read_text(encoding="utf-8"))
    current: dict[str, dict[str, str] | None] = {
        str(row["provisional_global_id"]): {
            camera: str(row["camera_local_pair"][camera]) for camera in CAMERAS
        }
        for row in assignments.get("matches", [])
    }
    history = output_dir / "assignment_revision_history.jsonl"
    if history.exists():
        for event in _read_jsonl(history):
            global_id = str(event["provisional_global_id"])
            if global_id not in current or event.get("previous_pair") != current[global_id]:
                raise ValueError("revision history does not follow the provisional assignment lineage")
            replacement = event.get("replacement_pair")
            if replacement is not None and (
                set(replacement) != set(CAMERAS) or not all(str(value) for value in replacement.values())
            ):
                raise ValueError("revision event contains an invalid replacement track pair")
            current[global_id] = replacement
    used: dict[str, set[str]] = {camera: set() for camera in CAMERAS}
    for pair in current.values():
        if pair is None:
            continue
        for camera in CAMERAS:
            local_id = pair[camera]
            if local_id in used[camera]:
                raise ValueError("revision history creates a one-to-many local-track assignment")
            used[camera].add(local_id)
    return current


def append_provisional_revision(
    output_dir: Path,
    provisional_global_id: str,
    *,
    action: str,
    rationale: str,
    replacement_pair: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Append a private revoke/replace event; raw tracker records stay immutable.

    A replacement must already pass the automatic gates stored in this run's
    immutable diagnostics. This does not turn a revision note into ground truth.
    """
    output_dir = Path(output_dir).resolve()
    if action not in {"revoke", "replace"} or not rationale.strip():
        raise ValueError("revision requires action revoke/replace and a non-empty rationale")
    current = resolve_provisional_assignments(output_dir)
    if provisional_global_id not in current or current[provisional_global_id] is None:
        raise ValueError("provisional Global ID is unknown or already revoked")
    if action == "revoke" and replacement_pair is not None:
        raise ValueError("revoke cannot include a replacement pair")
    diagnostics = json.loads((output_dir / "automatic_global_id_diagnostics.json").read_text(encoding="utf-8"))
    if action == "replace":
        if not replacement_pair or set(replacement_pair) != set(CAMERAS):
            raise ValueError("replace requires one CAM-02 and one CAM-05 local track")
        replacement_pair = {camera: str(replacement_pair[camera]) for camera in CAMERAS}
        eligible = any(
            row.get("automatic_candidate") is True
            and {camera: str(row["camera_local_pair"][camera]) for camera in CAMERAS} == replacement_pair
            for row in diagnostics.get("candidate_pairs", [])
        )
        if not eligible:
            raise ValueError("replacement pair did not pass this run's automatic candidate gates")
        for other_gid, other_pair in current.items():
            if other_gid != provisional_global_id and other_pair is not None and any(
                other_pair[camera] == replacement_pair[camera] for camera in CAMERAS
            ):
                raise ValueError("replacement would create a one-to-many assignment conflict")
    else:
        replacement_pair = None
    history = output_dir / "assignment_revision_history.jsonl"
    if not history.is_file():
        raise ValueError("append-only assignment revision history file is missing")
    with history.open("a+", encoding="utf-8") as stream:
        stream.seek(0)
        revision_number = sum(1 for line in stream if line.strip()) + 1
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps({
            "schema_version": 1,
            "revision_number": revision_number,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "provisional_global_id": provisional_global_id,
            "action": action,
            "previous_pair": current[provisional_global_id],
            "replacement_pair": replacement_pair,
            "rationale": rationale.strip(),
            "evidence_status": "provisional_only; not independently verified person identity",
            "production_identity_write": False,
        }, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    history.chmod(0o600)
    return {"provisional_global_id": provisional_global_id,
            "current_pair": replacement_pair,
            "revision_history": str(history)}


def build_offline_demo(
    *,
    bundle_dir: Path,
    tracking_dir: Path,
    video_paths: dict[str, Path],
    alignment_path: Path,
    layout_path: Path,
    output_dir: Path,
    frame_limit: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    os.umask(0o077)
    output_dir = output_dir.resolve()
    runtime_root = (ROOT / ".runtime").resolve()
    if not output_dir.is_relative_to(runtime_root):
        raise ValueError("automatic provisional Global ID output must stay under this checkout's .runtime")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite offline Global ID output: {output_dir}")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest, vectors, records, tracks, source_pts = load_verified_inputs(
        bundle_dir=bundle_dir, tracking_dir=tracking_dir, video_paths=video_paths)
    match_result = score_and_assign_provisional_ids(vectors, records, tracks)
    summaries = _summarize_track_durations(tracks, source_pts)
    match_result["track_durations"] = summaries
    match_result["feature_track_count"] = {
        camera: len({str(row["local_track_id"]) for row in records if row["camera_id"] == camera})
        for camera in CAMERAS
    }
    match_result["tracked_local_id_count"] = {
        camera: len({str(row["object_id"]) for frame_rows in tracks[camera].values() for row in frame_rows})
        for camera in CAMERAS
    }
    featured_ids = {
        camera: {str(row["local_track_id"]) for row in records if row["camera_id"] == camera}
        for camera in CAMERAS
    }
    match_result["tracks_without_usable_features"] = {
        camera: sorted(
            {str(row["object_id"]) for frame_rows in tracks[camera].values() for row in frame_rows}
            - featured_ids[camera],
            key=lambda value: (int(value) if value.isdigit() else 10**12, value),
        ) for camera in CAMERAS
    }
    for pair in match_result["candidate_pairs"]:
        pair["track_duration_context"] = {
            camera: summaries[camera].get(pair["camera_local_pair"][camera])
            for camera in CAMERAS
        }
    # Map both camera-local tracks to the same provisional identifier only for
    # pairs passing all gates. No other consumer or production state is touched.
    provisional_map: dict[str, dict[str, dict[str, Any]]] = {camera: {} for camera in CAMERAS}
    for match in match_result["matches"]:
        gid = match["provisional_global_id"]
        for camera in CAMERAS:
            local_id = match["camera_local_pair"][camera]
            provisional_map[camera][local_id] = {
                "provisional_global_id": gid,
                "status": "AUTO_PROVISIONAL_ONLY",
                "median_cosine_similarity": match["cosine_similarity_median"],
            }
    assignments = {
        "schema_version": 1,
        "mode": "OFFLINE_EXPERIMENTAL_PROVISIONAL_ONLY",
        "provisional_id_mapping": provisional_map,
        "matches": match_result["matches"],
        "unmatched_feature_tracks": {
            camera: sorted(
                ({str(row["local_track_id"]) for row in records if row["camera_id"] == camera}
                 - set(provisional_map[camera])),
                key=lambda value: (int(value) if value.isdigit() else 10**12, value),
            ) for camera in CAMERAS
        },
        "uncertain_pair_count": sum(not row["automatic_candidate"] for row in match_result["candidate_pairs"]),
        "one_to_many_conflicts_rejected": match_result["one_to_many_conflicts_rejected"],
        "saved_human_review_decisions_read": False,
        "production_identity_write": False,
    }
    _write_private_json(output_dir / "automatic_global_id_assignments.json", assignments)
    _write_private_json(output_dir / "automatic_global_id_diagnostics.json", match_result)
    revision_history = output_dir / "assignment_revision_history.jsonl"
    with revision_history.open("x", encoding="utf-8"):
        pass
    revision_history.chmod(0o600)

    scores_path = output_dir / "pairwise_scores.jsonl"
    with scores_path.open("x", encoding="utf-8") as stream:
        for row in match_result["candidate_pairs"]:
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    scores_path.chmod(0o600)

    preview_dir = output_dir / "preview"
    preview_started = time.perf_counter()
    preview_manifest = build_preview(
        alignment_path=alignment_path,
        video_paths={"cam02": video_paths["CAM-02"], "cam05": video_paths["CAM-05"]},
        layout_path=layout_path,
        tracking_dir=tracking_dir,
        output_dir=preview_dir,
        frame_limit=frame_limit,
        provisional_global_ids=provisional_map,
    )
    preview_elapsed = time.perf_counter() - preview_started
    elapsed = time.perf_counter() - started
    current_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    performance = {
        "lineage_validation_and_output_wall_ms": (elapsed - preview_elapsed) * 1000.0,
        "pairwise_matching_algorithm_wall_ms": match_result["scoring_wall_ms"],
        "preview_render_wall_sec": preview_elapsed,
        "total_wall_sec": elapsed,
        "process_cpu_sec": time.process_time() - cpu_started,
        "process_cpu_equivalent_cores_average": (time.process_time() - cpu_started) / elapsed if elapsed else None,
        "process_peak_rss_mib": current_rss_kib / 1024.0,
        "feature_extraction_rerun": False,
        "osnet_inference_rerun": False,
        "gpu_inference": False,
        "source_replay_nominal_fps": 20.0,
        "live_fps_impact": "NOT_MEASURED; offline post-processing only",
        "osnet_extraction_historical": manifest.get("performance"),
    }
    final_manifest = {
        "status": match_result["status"],
        "project": "Entrance Room automatic OSNet provisional identity DEMO",
        "offline_only": True,
        "no_camera_or_gpu_inference": True,
        "provisional_ids_are_not_verified_identity": True,
        "production_identity_database_written": False,
        "saved_manual_review_decisions_consumed": False,
        "source_hashes": {camera: sha256(path) for camera, path in video_paths.items()},
        "feature_bundle_hashes": EXPECTED_BUNDLE_HASHES,
        "osnet_checkpoint_sha256": EXPECTED_MODEL_SHA256,
        "alignment_path": str(alignment_path.resolve()),
        "alignment_sha256": sha256(alignment_path),
        "alignment_name": "LIVE_NATIVE_ALIGNMENT-v0007",
        "alignment_status": "EXPERIMENTAL / UNCALIBRATED; auxiliary visualization only",
        "cross_camera_sync": "UNVERIFIED; per-file source PTS retained but not used as a cross-camera timing gate",
        "pairwise_score_count": len(match_result["candidate_pairs"]),
        "provisional_match_count": len(match_result["matches"]),
        "same_camera_negative_like_control_count": len(match_result["same_camera_negative_like_controls"]),
        "unmatched_feature_tracks": assignments["unmatched_feature_tracks"],
        "uncertain_pair_count": assignments["uncertain_pair_count"],
        "preview_manifest": str((preview_dir / "preview_manifest.json").resolve()),
        "preview_video": preview_manifest["video"],
        "preview_screenshot": preview_manifest["representative_review_frame"],
        "assignment_revision_history": str(revision_history.resolve()),
        "performance": performance,
        "limitations": [
            "No independently verified SAME_PERSON or DIFFERENT_PEOPLE ground truth was consumed.",
            "Same-camera simultaneous separated tracks are score-separation controls only, not ground-truth negatives; duplicate tracks remain possible.",
            "Cosine similarity and the provisional gate are uncalibrated; false-match rate cannot be measured.",
            "Relative source timestamps are not proof of cross-camera exposure synchronization.",
            "The v0007 map is uncalibrated and does not establish physical position or identity.",
            "Provisional IDs are offline annotations only and may be revised; they are not persisted to production.",
        ],
    }
    _write_private_json(output_dir / "manifest.json", final_manifest)
    return final_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--tracking-dir", type=Path, default=DEFAULT_TRACKING)
    parser.add_argument("--cam02", type=Path, default=DEFAULT_CAM02)
    parser.add_argument("--cam05", type=Path, default=DEFAULT_CAM05)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int)
    args = parser.parse_args(argv)
    result = build_offline_demo(
        bundle_dir=args.bundle,
        tracking_dir=args.tracking_dir,
        video_paths={"CAM-02": args.cam02, "CAM-05": args.cam05},
        alignment_path=args.alignment,
        layout_path=args.layout,
        output_dir=args.output_dir,
        frame_limit=args.frames,
    )
    print(json.dumps({key: result[key] for key in (
        "status", "provisional_match_count", "same_camera_negative_like_control_count",
        "uncertain_pair_count", "preview_video", "preview_screenshot", "performance")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
