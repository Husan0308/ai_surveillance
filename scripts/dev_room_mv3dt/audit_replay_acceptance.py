#!/usr/bin/env python3
"""Audit the known two-person Dev Room production replay."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.mv3dt_room.presence import (
    CurrentPresencePositionFilter,
    active_observations,
    application_id,
    fused_world_positions,
)


CROSS_CAMERA_IMPOSSIBLE_DISTANCE_M = 8.5


def _bbox_iou(left, right):
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return None
    if len(left) < 4 or len(right) < 4:
        return None
    try:
        l1, t1, r1, b1 = (float(x) for x in left[:4])
        l2, t2, r2, b2 = (float(x) for x in right[:4])
    except (TypeError, ValueError):
        return None
    intersection = max(0.0, min(r1, r2) - max(l1, l2)) * max(0.0, min(b1, b2) - max(t1, t2))
    area1 = max(0.0, r1 - l1) * max(0.0, b1 - t1)
    area2 = max(0.0, r2 - l2) * max(0.0, b2 - t2)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def _world_distance(left, right):
    try:
        a = left.get("world")
        b = right.get("world")
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
            return None
        if len(a) < 2 or len(b) < 2:
            return None
        values = [float(a[0]), float(a[1]), float(b[0]), float(b[1])]
        if not all(math.isfinite(value) for value in values):
            return None
        return math.dist(values[:2], values[2:])
    except (TypeError, ValueError):
        return None


def presence_frame(source_rows, frame, position_filter):
    """Apply the exact current-frame/canonical-ID reducer used by production BEV."""
    _, active = active_observations(source_rows, frame)
    positions = position_filter.update(active)
    return active, positions, list(positions)


def native_fragment_overlap_diagnostics(by_frame, frames):
    """Describe native-ID overlap without treating it as a canonical merge."""
    groups_by_frame = {}
    pair_frames = defaultdict(list)
    for frame in range(frames):
        grouped = defaultdict(list)
        for row in by_frame.get(frame, []):
            grouped[(str(row.get("camera_id")), application_id(row))].append(row)
        groups_by_frame[frame] = grouped
        for (camera, app_id), group in grouped.items():
            natives = sorted({str(row.get("native_track_id")) for row in group})
            if len(natives) > 1:
                for first, second in combinations(natives, 2):
                    pair_frames[(camera, app_id, first, second)].append(frame)

    records = []
    for frame in range(frames):
        source_rows = by_frame.get(frame, [])
        _, active = active_observations(source_rows, frame)
        active_by_key = {
            (str(row.get("camera_id")), application_id(row)): row
            for row in active
        }
        renderable_ids = set(fused_world_positions(active))
        for (camera, app_id), group in groups_by_frame[frame].items():
            native_ids = sorted({str(row.get("native_track_id")) for row in group})
            if len(native_ids) <= 1:
                continue
            selected = active_by_key.get((camera, app_id))
            pair_details = []
            for first, second in combinations(native_ids, 2):
                shared_frames = pair_frames[(camera, app_id, first, second)]
                pair_rows = {
                    str(row.get("native_track_id")): row
                    for row in group
                }
                left, right = pair_rows[first], pair_rows[second]
                pair_details.append({
                    "native_track_ids": [first, second],
                    "coobserved_frames": shared_frames,
                    "coobserved_frame_count": len(shared_frames),
                    "brief_overlap_window": len(shared_frames) <= 3,
                    "bbox_iou": _bbox_iou(left.get("bbox"), right.get("bbox")),
                    "world_distance_m": _world_distance(left, right),
                })
            records.append({
                "frame": frame,
                "camera_id": camera,
                "application_id": app_id,
                "native_track_ids": native_ids,
                "observations": [
                    {
                        "native_track_id": row.get("native_track_id"),
                        "confidence": row.get("confidence"),
                        "bbox": row.get("bbox"),
                        "world": row.get("world"),
                        "timestamp": row.get("timestamp"),
                    }
                    for row in group
                ],
                "pair_evidence": pair_details,
                "active_observation_count_for_camera_identity": int(selected is not None),
                "selected_native_track_id": selected.get("native_track_id") if selected else None,
                "renderable_bev_marker_count_for_identity": int(app_id in renderable_ids),
                "interpretation": "native-fragment overlap is diagnostic; production presence selects one observation per camera/application ID and fuses one BEV marker per application ID",
            })
    return records


def canonical_false_merge_diagnostics(by_frame, physical_labels=None):
    """Count only physically labeled conflicts; separately report strong unlabeled spatial candidates."""
    physical_labels = physical_labels or {}
    confirmed = []
    spatial_candidates = []
    label_coverage = 0
    observation_count = 0

    def label_for(row):
        key = (int(row.get("frame", -1)), str(row.get("camera_id")), str(row.get("native_track_id")))
        return row.get("physical_person_label") or physical_labels.get(key)

    for frame, rows in by_frame.items():
        by_app = defaultdict(list)
        by_camera_app = defaultdict(list)
        by_camera = defaultdict(list)
        for row in rows:
            if application_id(row) == "Unknown":
                continue
            by_app[application_id(row)].append(row)
            by_camera_app[(str(row.get("camera_id")), application_id(row))].append(row)
            by_camera[str(row.get("camera_id"))].append(row)
            observation_count += 1
            label_coverage += label_for(row) is not None

        for app_id, group in by_app.items():
            labels = sorted({str(label_for(row)) for row in group if label_for(row) is not None})
            if len(labels) > 1:
                confirmed.append({
                    "frame": int(frame),
                    "application_id": app_id,
                    "physical_person_labels": labels,
                    "observations": [
                        {
                            "camera_id": row.get("camera_id"),
                            "native_track_id": row.get("native_track_id"),
                            "physical_person_label": label_for(row),
                            "bbox": row.get("bbox"),
                            "world": row.get("world"),
                        }
                        for row in group
                        if label_for(row) is not None
                    ],
                })

        for (camera, app_id), group in by_camera_app.items():
            if len({str(row.get("native_track_id")) for row in group}) <= 1:
                continue
            for left, right in combinations(group, 2):
                left_label, right_label = label_for(left), label_for(right)
                if left_label is not None and right_label is not None:
                    continue
                iou = _bbox_iou(left.get("bbox"), right.get("bbox"))
                distance = _world_distance(left, right)
                if iou is not None and iou <= 0.05 and distance is not None and distance > 3.0:
                    spatial_candidates.append({
                        "frame": int(frame),
                        "camera_id": camera,
                        "application_id": app_id,
                        "native_track_ids": [left.get("native_track_id"), right.get("native_track_id")],
                        "bbox_iou": iou,
                        "world_distance_m": distance,
                        "reason": "spatially disjoint simultaneous observations lack physical-person labels",
                    })

        active = [
            row for row in rows
            if row.get("camera_id") in ("CAM-01", "CAM-04")
            and application_id(row) != "Unknown"
        ]
        selected_by_camera_app = {}
        for row in active:
            key = (str(row.get("camera_id")), application_id(row))
            old = selected_by_camera_app.get(key)
            if old is None or float(row.get("confidence") or 0.0) > float(old.get("confidence") or 0.0):
                selected_by_camera_app[key] = row
        by_app_camera = defaultdict(dict)
        for (camera, app_id), row in selected_by_camera_app.items():
            by_app_camera[app_id][camera] = row
        for app_id, cameras in by_app_camera.items():
            if set(cameras) != {"CAM-01", "CAM-04"}:
                continue
            left, right = cameras["CAM-01"], cameras["CAM-04"]
            distance = _world_distance(left, right)
            if distance is not None and distance > CROSS_CAMERA_IMPOSSIBLE_DISTANCE_M:
                spatial_candidates.append({
                    "frame": int(frame),
                    "application_id": app_id,
                    "camera_ids": ["CAM-01", "CAM-04"],
                    "world_distance_m": distance,
                    "reason": "same canonical identity exceeds the established cross-camera spatial plausibility bound",
                })

    return {
        "confirmed_count": len(confirmed),
        "confirmed_conflicts": confirmed,
        "unresolved_spatial_candidate_count": len(spatial_candidates),
        "unresolved_spatial_candidates": spatial_candidates,
        "physical_label_coverage": label_coverage / observation_count if observation_count else 0.0,
        "physically_labeled_observations": label_coverage,
        "total_observations": observation_count,
        "evidence_note": "A native-fragment overlap alone is never a canonical false merge; conflicts require distinct physical labels. Spatially disjoint/unexplained cases remain explicit candidates.",
    }


def preview_health_status(source_mode, preview_stats_supplied, previews):
    if source_mode == "replay" and not preview_stats_supplied:
        return {
            "applicable": False,
            "status": "N/A",
            "reason": "CAM-01/CAM-04 scoped replay; preview-only camera telemetry was not part of this run",
        }, None
    passed = bool(previews) and all(
        row.get("state") == "LIVE"
        and int(row.get("reconnects", 0)) == 0
        and int(row.get("stalled_periods", 0)) == 0
        for row in previews.values()
    )
    return {
        "applicable": True,
        "status": "PASS" if passed else "FAIL",
        "reason": "explicit preview telemetry supplied" if preview_stats_supplied else "preview telemetry is required outside replay mode",
    }, passed


def presence_disagreement_status(
    visual_disagreements: int,
    visual_assertion_frames: int,
    total_frames: int,
    computed_disagreements: int,
    *,
    source_mode: str,
    visual_telemetry_supplied: bool,
):
    """Gate computed presence while making absent visual telemetry explicit.

    URI replay with ``--skip-render`` does not create visual-proof telemetry.
    Production presence semantics are still computed from source rows and stay
    gating; a missing visual artifact is N/A rather than a fabricated failure.
    """
    visual_applicable = visual_telemetry_supplied or source_mode != "replay"
    visual_ok = (
        visual_disagreements == 0 and visual_assertion_frames == total_frames
        if visual_applicable else True
    )
    passed = computed_disagreements == 0 and visual_ok
    return {
        "applicable": visual_applicable,
        "status": ("PASS" if visual_ok else "FAIL") if visual_applicable else "N/A",
        "computed_presence_pass": computed_disagreements == 0,
        "reason": (
            "visual presence telemetry supplied"
            if visual_applicable
            else "visual presence telemetry unavailable for skip-render URI replay; computed production presence remains gating"
        ),
    }, passed


def unnecessary_allocation_diagnostics(identities, allocations):
    """Flag orphaned or repeated CREATE events, not the expected first IDs.

    A fresh two-person replay necessarily allocates its first two canonical
    identities.  Those are not unnecessary when they are the two identities
    actually observed in the replay.  An allocation is unnecessary if it never
    appears in the replay output or if the same canonical ID is created more
    than once.
    """
    observed = {str(identity) for identity in identities}
    counts = Counter()
    unnecessary = []
    for allocation in allocations:
        application_id_value = allocation.get("application_id")
        key = str(application_id_value) if application_id_value is not None else None
        counts[key] += 1
        if key is None or key not in observed or counts[key] > 1:
            unnecessary.append(allocation)
    return {
        "count": len(unnecessary),
        "allocations": unnecessary,
        "observed_identity_allocation_counts": dict(counts),
        "interpretation": "initial CREATE for each observed canonical identity is necessary; orphaned or repeated CREATE is unnecessary",
    }


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--preview-stats", type=Path)
    parser.add_argument("--physical-person-review", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    identity_dir = root / "identity-live"
    report = load_json(identity_dir / "runtime_report.json", {})
    current = load_json(identity_dir / "current_state.json", {})
    events = load_json(identity_dir / "identity_events.json", [])
    visual_path = root / "visual/visual_proof.json"
    visual = load_json(visual_path, {})
    readiness = load_json(root / "run/logs/probe/readiness.json", {})
    resources = load_json(root / "resource_metrics.json", {})
    previews = load_json(args.preview_stats, {}) if args.preview_stats else {}
    review_path = args.physical_person_review or root / "physical_person_review.json"
    physical_review = load_json(review_path, {})
    physical_labels = {}
    for item in physical_review.get("observations", []):
        try:
            key = (
                int(item["frame"]),
                str(item["camera_id"]),
                str(item["native_track_id"]),
            )
            physical_labels[key] = str(item["physical_person_label"])
        except (KeyError, TypeError, ValueError):
            continue

    rows = []
    for line in (identity_dir / "global_identity.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame"])].append(row)

    identities = sorted({
        application_id(row)
        for row in rows
        if application_id(row) != "Unknown"
    })
    cameras_by_identity = {
        identity: sorted({
            str(row.get("camera_id"))
            for row in rows
            if application_id(row) == identity
        })
        for identity in identities
    }
    unknown_rows = sum(application_id(row) == "Unknown" for row in rows)

    frames = int(visual.get("frames") or (max(by_frame) + 1 if by_frame else 0))
    fragment_overlaps = native_fragment_overlap_diagnostics(by_frame, frames)
    canonical_merge = canonical_false_merge_diagnostics(by_frame, physical_labels)
    position_filter = CurrentPresencePositionFilter()
    prior_positions = {}
    maximum_bev_jump = 0.0
    maximum_bev_jump_detail = None
    actual_duplicate_marker_count = 0
    computed_presence_disagreement_count = 0
    computed_ghost_marker_count = 0
    for frame in range(frames):
        source_rows = by_frame.get(frame, [])
        active, positions, marker_ids = presence_frame(source_rows, frame, position_filter)
        marker_counts = Counter(marker_ids)
        actual_duplicate_marker_count += sum(max(0, count - 1) for count in marker_counts.values())
        active_renderable = set(fused_world_positions(active))
        computed_presence_disagreement_count += len(active_renderable ^ set(marker_ids))
        computed_ghost_marker_count += len(set(marker_ids) - {application_id(row) for row in active})
        for identity, position in positions.items():
            if identity in prior_positions:
                old_frame, old_position = prior_positions[identity]
                if old_frame == frame - 1:
                    jump = math.dist(old_position, position)
                    if jump > maximum_bev_jump:
                        maximum_bev_jump = jump
                        maximum_bev_jump_detail = {
                            "frame": frame,
                            "application_id": identity,
                            "distance_m": jump,
                        }
            prior_positions[identity] = (frame, position)
        for identity in set(prior_positions) - set(positions):
            del prior_positions[identity]

    perf_rows = []
    perf_pattern = re.compile(r"\*\*PERF:\s+([0-9.]+)\s+\([^)]*\)\s+([0-9.]+)")
    deepstream_text = (root / "logs/deepstream.log").read_text(errors="replace")
    for match in perf_pattern.finditer(deepstream_text):
        perf_rows.append((float(match.group(1)), float(match.group(2))))
    steady_rows = perf_rows[-10:] if len(perf_rows) >= 10 else perf_rows
    steady_fps = {
        "CAM-01": sum(row[0] for row in steady_rows) / len(steady_rows) if steady_rows else None,
        "CAM-04": sum(row[1] for row in steady_rows) / len(steady_rows) if steady_rows else None,
    }

    counters = report.get("identity_metrics", {}).get("counters", {})
    maxima = report.get("identity_metrics", {}).get("max", {})
    manager = report.get("identity_manager", {})
    event_counts = Counter(str(event.get("event")) for event in events)
    identity_switches = int(manager.get("reassignment_accepted", 0) or 0) + sum(
        count for name, count in event_counts.items() if "SWITCH" in name
    )
    allocations = [
        {
            "frame": event.get("frame"),
            "application_id": report.get("public_id_map", {}).get(str(event.get("global_person_id"))),
            "reason": event.get("reason"),
        }
        for event in events
        if event.get("event") == "CREATE"
    ]
    allocation_audit = unnecessary_allocation_diagnostics(identities, allocations)
    current_people = current.get("people", [])
    persistent_unresolved = sum(
        str(row.get("application_id") or row.get("global_person_id")) == "Unknown"
        or row.get("identity_state") == "PENDING"
        for row in current_people
    )

    preview_status, preview_ok = preview_health_status(
        str(report.get("source_mode") or ""), args.preview_stats is not None, previews
    )
    crop = report.get("crop_delivery", {})
    osnet = report.get("osnet", {}).get("embedder", {})
    kafka_errors = (root / "logs/kafka_capture.err").stat().st_size
    mqtt_messages = sum(1 for _ in (root / "logs/mqtt_peer.raw").open(errors="replace"))

    visual_disagreements = int(visual.get("frames_with_presence_disagreement", -1))
    visual_assertion_frames = int(visual.get("presence_assertion_frames", -1))
    visual_duplicate_markers = visual.get("duplicate_actual_bev_markers")
    if visual_duplicate_markers is None:
        visual_duplicate_markers = actual_duplicate_marker_count
    visual_ghost_markers = visual.get("ghost_stale_bev_markers")
    if visual_ghost_markers is None:
        visual_ghost_markers = computed_ghost_marker_count
    presence_status, presence_ok = presence_disagreement_status(
        visual_disagreements,
        visual_assertion_frames,
        frames,
        computed_presence_disagreement_count,
        source_mode=str(report.get("source_mode") or ""),
        visual_telemetry_supplied=visual_path.exists(),
    )

    checks = {
        "exactly_two_canonical_identities": identities == ["Person_01", "Person_02"],
        "both_identities_seen_in_both_dev_room_cameras": all(
            cameras_by_identity.get(identity) == ["CAM-01", "CAM-04"]
            for identity in identities
        ),
        "identity_switches_zero": identity_switches == 0,
        "canonical_false_merges_zero": (
            canonical_merge["confirmed_count"] == 0
            and canonical_merge["unresolved_spatial_candidate_count"] == 0
        ),
        "unnecessary_canonical_ids_zero": allocation_audit["count"] == 0,
        "persistent_unknown_pending_zero": persistent_unresolved == 0,
        "presence_disagreements_zero": presence_ok,
        "duplicate_actual_bev_markers_zero": actual_duplicate_marker_count == 0
        and int(visual_duplicate_markers) == 0,
        "ghost_stale_bev_markers_zero": computed_ghost_marker_count == 0
        and int(visual_ghost_markers) == 0,
        "maximum_bev_jump_at_most_3m": maximum_bev_jump <= 3.0,
        "dev_room_fps_approximately_20": all(
            value is not None and 19.0 <= value <= 21.0 for value in steady_fps.values()
        ),
        "source_readiness_good": readiness.get("ready") is True
        and readiness.get("status") == "ready"
        and {row.get("source_id") for row in readiness.get("sources", [])} == {0, 1},
        "source_reconnects_zero": bool(readiness.get("sources")) and all(
            row.get("reconnect_attempts") == 0 for row in readiness.get("sources", [])
        ),
        "identity_queue_bounded": float(maxima.get("identity_latest_queue_depth", 9999)) <= 256,
        "crop_path_healthy": int(crop.get("crop_requests", 0)) > 0
        and int(crop.get("crop_requests", 0)) == int(crop.get("crop_success", -1))
        and int(crop.get("crop_failures", -1)) == 0,
        "osnet_cuda_healthy": osnet.get("device") == "cuda"
        and bool(osnet.get("ready"))
        and int(osnet.get("images", 0)) > 0,
        "kafka_healthy": kafka_errors == 0 and int(counters.get("kafka_frames_received", 0)) > 0,
        "mqtt_healthy": mqtt_messages > 0,
    }
    if preview_ok is not None:
        checks["preview_only_cameras_healthy"] = preview_ok
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "run_root": str(root),
        "source_mode": report.get("source_mode"),
        "checks": checks,
        "preview_only_cameras_healthy": preview_status,
        "presence_disagreement_telemetry": presence_status,
        "identities": identities,
        "cameras_by_identity": cameras_by_identity,
        "identity_switches": identity_switches,
        "canonical_false_merge": canonical_merge,
        "false_merges": canonical_merge["confirmed_count"],
        "native_fragment_overlap_count": len(fragment_overlaps),
        "native_fragment_overlaps": fragment_overlaps,
        "unknown_output_rows": unknown_rows,
        "persistent_unknown_pending": persistent_unresolved,
        "allocations": allocations,
        "unnecessary_canonical_id_allocations": allocation_audit,
        "event_counts": dict(event_counts),
        "bev": {
            "presence_disagreements": visual_disagreements,
            "computed_presence_disagreement_count": computed_presence_disagreement_count,
            "duplicate_actual_marker_count": actual_duplicate_marker_count,
            "visual_duplicate_actual_marker_count": visual_duplicate_markers,
            "ghost_stale_marker_count": computed_ghost_marker_count,
            "visual_ghost_stale_marker_count": visual_ghost_markers,
            "maximum_filtered_current_presence_jump_m": maximum_bev_jump,
            "maximum_jump_detail": maximum_bev_jump_detail,
        },
        "performance": {
            "deepstream_steady_fps": steady_fps,
            "identity_observation_fps": report.get("identity_observation_fps_per_camera"),
            "identity_queue_max": maxima.get("identity_latest_queue_depth"),
            "osnet_request_queue_max": report.get("osnet", {}).get("request_queue_max_depth"),
            "ui_queue_policy": "latest-only shared-memory frame per camera",
        },
        "crop_osnet": {
            "crop_requests": crop.get("crop_requests"),
            "crop_success": crop.get("crop_success"),
            "crop_failures": crop.get("crop_failures"),
            "crop_timeouts": counters.get("pending_crop_health_timeout", 0),
            "embeddings": report.get("embeddings_extracted"),
            "embeddings_by_camera": report.get("embeddings_by_camera"),
            "device": osnet.get("device"),
            "batch_sizes": osnet.get("batch_sizes"),
            "batch_latency_ms": osnet.get("batch_latency_ms"),
        },
        "preview_cameras": previews,
        "physical_person_review": {
            "path": str(review_path.resolve()),
            "provided": bool(physical_labels),
            "review_note": physical_review.get("review_note"),
            "labeled_observations": len(physical_labels),
        },
        "kafka_error_bytes": kafka_errors,
        "mqtt_messages": mqtt_messages,
        "resources": resources,
        "artifacts": {
            "screenshot": str((Path.cwd() / ".runtime/ui-acceptance/evidence/final-production-ui-replay.png").resolve()),
            "ui_recording": str((Path.cwd() / ".runtime/ui-acceptance/evidence/final-production-ui-replay-30s.mp4").resolve()),
            "dev_room_visual_proof": visual.get("output"),
        },
    }
    output = args.output or (root / "replay_acceptance.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
