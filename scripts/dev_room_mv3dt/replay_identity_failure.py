#!/usr/bin/env python3
"""Deterministically audit a captured room-pair identity decision.

This replays metadata only; no detector, tracker, RTSP, crop reader, or OSNet
runtime is started.  If a run has optional embedding_debug.jsonl, its vectors
are used.  Older runs correctly report unavailable cosine values instead of
inventing them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from services.mv3dt_room.global_identity_manager import GlobalIdentityManager


def load_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--camera", default="CAM-01")
    parser.add_argument("--native-track", type=int, required=True)
    args = parser.parse_args()
    identity_dir = args.run_dir / "identity-live"
    rows = list(load_jsonl(identity_dir / "global_identity.jsonl"))
    events = json.loads((identity_dir / "identity_events.json").read_text())
    vectors = {}
    vector_path = identity_dir / "embedding_debug.jsonl"
    if vector_path.exists():
        for item in load_jsonl(vector_path):
            key = (str(item.get("camera_id")), int(item.get("native_track_id", -1)), int(item.get("frame", -1)))
            vectors[key] = np.asarray(item["vector"], dtype=np.float32)

    manager = GlobalIdentityManager(np.empty((0, 512), np.float32))
    # CREATE events are recorded in actual decision order. Source frame
    # numbers can arrive out of order across asynchronous ReID results.
    create_events = [e for e in events if e.get("event") == "CREATE"]
    for event in create_events:
        if int(event["frame"]) >= args.frame:
            continue
        gid = manager.create(int(event["frame"]), "replay_existing_canonical")
        if gid != int(event["global_person_id"]):
            raise RuntimeError(f"canonical sequence mismatch: expected {event['global_person_id']}, got {gid}")

    target = None
    active = []
    ordered = sorted(enumerate(rows), key=lambda pair: (int(pair[1].get("frame", -1)), pair[0]))
    for _, row in ordered:
        frame = int(row.get("frame", -1))
        gid = row.get("canonical_internal_identity")
        if frame > args.frame:
            break
        if frame == args.frame and str(row.get("camera_id")) == args.camera and int(row.get("native_track_id", -1)) == args.native_track:
            target = row
            continue
        if gid is None or int(gid) not in manager.identities:
            continue
        obs = {key: row.get(key) for key in ("camera_id", "native_track_id", "frame", "timestamp", "bbox", "world", "confidence", "visibility", "crop_quality", "crop_quality_usable")}
        vector = vectors.get((str(row.get("camera_id")), int(row.get("native_track_id", -1)), frame))
        manager._finalize_observation(int(gid), obs, vector, {"reason": "replay_state"}, 0)
        if frame == args.frame:
            active.append({**obs, "global_person_id_num": int(gid)})
    if target is None:
        raise SystemExit("target observation not found")

    target_obs = {key: target.get(key) for key in ("camera_id", "native_track_id", "frame", "timestamp", "bbox", "world", "confidence", "visibility", "crop_quality", "crop_quality_usable")}
    vector = vectors.get((args.camera, args.native_track, args.frame))
    trace = manager.candidate_trace(target_obs, vector, active)
    resolved, resolution_evidence = manager.resolve_existing(target_obs, vector, active)
    novelty = manager.novelty_evidence(target_obs, vector, active, 2, 2)
    crop_events = []
    crop_log = args.run_dir / "run" / "logs" / "probe" / "crop_encoder.jsonl"
    if crop_log.exists():
        crop_events = [event for event in load_jsonl(crop_log)
                       if str(event.get("camera_id")) == args.camera
                       and int(event.get("native_track_id", -1)) == args.native_track
                       and event.get("event") in {"crop_request", "crop_delivered", "crop_success", "crop_failure"}]
    allocation = next((event for event in events if event.get("event") == "NEW_CONFIRMED"
                       and int(event.get("frame", -1)) == args.frame
                       and int(event.get("native_track_id", -1)) == args.native_track), None)
    report = {
        "run_dir": str(args.run_dir),
        "target": {"camera_id": args.camera, "native_track_id": args.native_track, "frame": args.frame},
        "embedding_data_available": vector is not None,
        "pending_start_frame": min((int(e["frame_num"]) for e in crop_events if e.get("event") == "crop_request"), default=None),
        "crop_events": crop_events,
        "original_allocation_event": allocation,
        "original_allocation_branch": "LiveIdentityWorker._apply_reid_results -> manager.confirm_new at historical line 490",
        "candidate_trace": trace,
        "replayed_resolution": {
            "global_person_id": resolved,
            "application_id": manager.application_id(resolved) if resolved is not None else "Unknown",
            "evidence": resolution_evidence,
        },
        "novelty_review_with_captured_metadata": novelty,
        "replay_limitation": None if vector is not None else "The failed run did not persist OSNet vectors; cosine values are unavailable, not inferred.",
    }
    output = identity_dir / f"frame-{args.frame}-decision-trace.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
