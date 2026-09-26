#!/usr/bin/env python3
"""Audit the known two-person Dev Room production replay."""
from __future__ import annotations

import argparse
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
)


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--preview-stats", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    identity_dir = root / "identity-live"
    report = load_json(identity_dir / "runtime_report.json", {})
    current = load_json(identity_dir / "current_state.json", {})
    events = load_json(identity_dir / "identity_events.json", [])
    visual = load_json(root / "visual/visual_proof.json", {})
    resources = load_json(root / "resource_metrics.json", {})
    previews = load_json(args.preview_stats, {}) if args.preview_stats else {}

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

    duplicate_keys = 0
    position_filter = CurrentPresencePositionFilter()
    prior_positions = {}
    maximum_bev_jump = 0.0
    maximum_bev_jump_detail = None
    frames = int(visual.get("frames") or (max(by_frame) + 1 if by_frame else 0))
    for frame in range(frames):
        source_rows = by_frame.get(frame, [])
        _, active = active_observations(source_rows, frame)
        grouped = defaultdict(set)
        for row in source_rows:
            grouped[(row.get("camera_id"), application_id(row))].add(row.get("native_track_id"))
        duplicate_keys += sum(len(native_ids) > 1 for native_ids in grouped.values())
        positions = position_filter.update(active)
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
    current_people = current.get("people", [])
    persistent_unresolved = sum(
        str(row.get("application_id") or row.get("global_person_id")) == "Unknown"
        or row.get("identity_state") == "PENDING"
        for row in current_people
    )

    preview_ok = bool(previews) and all(
        row.get("state") == "LIVE"
        and int(row.get("reconnects", 0)) == 0
        and int(row.get("stalled_periods", 0)) == 0
        for row in previews.values()
    )
    crop = report.get("crop_delivery", {})
    osnet = report.get("osnet", {}).get("embedder", {})
    kafka_errors = (root / "logs/kafka_capture.err").stat().st_size
    mqtt_messages = sum(1 for _ in (root / "logs/mqtt_peer.raw").open(errors="replace"))

    checks = {
        "exactly_two_canonical_identities": len(identities) == 2,
        "both_identities_seen_in_both_dev_room_cameras": all(
            cameras_by_identity.get(identity) == ["CAM-01", "CAM-04"]
            for identity in identities
        ),
        "identity_switches_zero": identity_switches == 0,
        "false_merges_zero": duplicate_keys == 0,
        "persistent_unknown_pending_zero": persistent_unresolved == 0,
        "presence_disagreements_zero": int(visual.get("frames_with_presence_disagreement", -1)) == 0,
        "duplicate_bev_markers_zero": duplicate_keys == 0,
        "ghost_bev_markers_zero": int(visual.get("frames_with_presence_disagreement", -1)) == 0,
        "maximum_bev_jump_at_most_3m": maximum_bev_jump <= 3.0,
        "dev_room_fps_approximately_20": all(
            value is not None and 19.0 <= value <= 21.0 for value in steady_fps.values()
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
        "preview_only_cameras_healthy": preview_ok,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "run_root": str(root),
        "source_mode": report.get("source_mode"),
        "checks": checks,
        "identities": identities,
        "cameras_by_identity": cameras_by_identity,
        "identity_switches": identity_switches,
        "false_merges": duplicate_keys,
        "unknown_output_rows": unknown_rows,
        "persistent_unknown_pending": persistent_unresolved,
        "allocations": allocations,
        "event_counts": dict(event_counts),
        "bev": {
            "presence_disagreements": visual.get("frames_with_presence_disagreement"),
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
