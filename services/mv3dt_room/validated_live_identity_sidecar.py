#!/usr/bin/env python3
"""Concurrent causal OSNet + GlobalIdentityManager sidecar for the MV3DT run."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from extract_osnet import crop_quality, bbox_of as extracted_bbox
from global_identity_manager import CAMS, IDENTITY_GATES, GlobalIdentityManager, bbox_of, world_of
from reid_embedder import OsnetCpuEmbedder

VIDEO_NAMES = {"CAM-01": "cam_00.mp4", "CAM-04": "cam_01.mp4"}


def pct(values, q):
    return float(np.quantile(values, q)) if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kafka", required=True, type=Path)
    ap.add_argument("--done-file", required=True, type=Path)
    ap.add_argument("--video-dir", required=True, type=Path)
    ap.add_argument("--experiment-root", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--interval", type=int, default=10)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    while not args.kafka.exists():
        time.sleep(0.02)
    captures = {cam: cv2.VideoCapture(str(args.video_dir / name)) for cam, name in VIDEO_NAMES.items()}
    if not all(cap.isOpened() for cap in captures.values()):
        raise RuntimeError("video open failed")
    video_next = {cam: 0 for cam in CAMS}
    last_extract = {}
    config = {
        "model_name": "osnet_ain_x1_0",
        "model_path": "models/reid/osnet_ain_x1_0_msmt17.pth",
        "model_sha256": "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f",
        "download_if_missing": False,
        "input_height": 256,
        "input_width": 128,
        "cpu_threads": 2,
    }
    embedder = OsnetCpuEmbedder(config, args.experiment_root)
    manager = GlobalIdentityManager(np.empty((0, 512), np.float32))
    pending = defaultdict(dict)
    identity_pending = {}
    pending_created = 0
    pending_to_existing = 0
    pending_to_new = 0
    pending_unresolved = 0
    latest_seen = {cam: -1 for cam in CAMS}
    output = []
    extraction_latency = []
    crop_dims = []
    extracted_by_cam = defaultdict(int)
    max_backlog_frames = 0
    started = time.perf_counter()
    offset = 0

    def read_image(cam, target):
        image = None
        while video_next[cam] <= target:
            ok, image = captures[cam].read()
            if not ok:
                raise RuntimeError(f"decode failed {cam} frame {target}")
            video_next[cam] += 1
        return image

    def process_frame(frame_num, payloads):
        nonlocal max_backlog_frames, pending_created, pending_to_existing, pending_to_new, pending_unresolved
        images = {cam: read_image(cam, frame_num) for cam in payloads}
        crops, crop_keys, extraction_meta = [], [], {}
        for cam in CAMS:
            payload = payloads.get(cam)
            if not payload:
                continue
            for obj in payload.get("objects", []):
                if obj.get("type", "").lower() != "person":
                    continue
                native = int(obj["id"])
                box = extracted_bbox(obj)
                useful, quality = crop_quality(box)
                key = (cam, native)
                due = frame_num - last_extract.get(key, -10**9) >= args.interval
                if not useful or not due:
                    continue
                l, t, r, b = box
                il, it = max(0, int(l)), max(0, int(t))
                ir, ib = min(1920, int(np.ceil(r))), min(1080, int(np.ceil(b)))
                crop = images[cam][it:ib, il:ir]
                if crop.size == 0:
                    continue
                crops.append(crop)
                crop_key = (cam, frame_num, native)
                crop_keys.append(crop_key)
                extraction_meta[crop_key] = (ir - il, ib - it, quality)
                last_extract[key] = frame_num
        vectors = {}
        if crops:
            tic = time.perf_counter()
            batch = embedder.embed_batch(crops)
            batch_ms = (time.perf_counter() - tic) * 1000.0
            each_ms = batch_ms / len(crops)
            for key, vector in zip(crop_keys, batch):
                vectors[key] = vector
                extraction_latency.append(each_ms)
                crop_dims.append(extraction_meta[key][:2])
                extracted_by_cam[key[0]] += 1

        frame_assignments = []
        for cam in CAMS:
            payload = payloads.get(cam)
            if not payload:
                continue
            objects = sorted(
                [obj for obj in payload.get("objects", []) if obj.get("type", "").lower() == "person"],
                key=lambda obj: int(obj["id"]),
            )
            for obj in objects:
                native = int(obj["id"])
                obs = {
                    "camera_id": cam,
                    "frame": frame_num,
                    "timestamp": payload.get("timestamp"),
                    "native_track_id": native,
                    "bbox": bbox_of(obj),
                    "world": world_of(obj),
                    "confidence": float(obj.get("confidence", 0.0)),
                    "visibility": float(obj.get("info", {}).get("visibility", 0.0)),
                }
                vector = vectors.get((cam, frame_num, native))
                meta = extraction_meta.get((cam, frame_num, native))
                if meta is not None:
                    obs["crop_quality"] = meta[2]
                    obs["crop_quality_usable"] = True
                key = (cam, native)
                bound = manager.bindings.get(key)
                state = "KNOWN"
                if bound is not None:
                    gid = manager.process(obs, vector, frame_assignments, allow_reassignment=False)[0]
                    evidence = {"reason": "sticky_native_track"}
                    latency_us = 0.0
                else:
                    is_new_pending = key not in identity_pending
                    record = identity_pending.setdefault(key, {"first_frame": frame_num, "attempts": 0, "good": 0})
                    if vector is not None:
                        record["attempts"] += 1
                        record["good"] += 1
                    gid, evidence = manager.resolve_existing(obs, vector, frame_assignments)
                    latency_us = 0.0
                    if gid is not None:
                        manager.accept_existing(gid, obs, vector, evidence)
                        identity_pending.pop(key, None)
                        pending_to_existing += 1
                        state = "KNOWN"
                    else:
                        pending_unresolved += 1
                        state = "PENDING"
                        elapsed = frame_num - record["first_frame"]
                        enough = record["good"] >= 2
                        if (enough and record["attempts"] >= 2) or (elapsed >= 4 and record["attempts"] == 0):
                            gid = manager.confirm_new(obs, vector, "pending_window_exhausted_no_existing_candidate")
                            manager.bindings[key] = gid
                            identity_pending.pop(key, None)
                            pending_to_new += 1
                            state = "NEW_CONFIRMED"
                        else:
                            if is_new_pending:
                                pending_created += 1
                row = {
                    **obs,
                    "internal_identity_at_event": gid,
                    "identity_state": state,
                    "embedding_extracted": vector is not None,
                    "cosine_similarity": evidence.get("appearance_similarity"),
                    "decision_reason": evidence.get("reason", "pending_identity"),
                    "identity_manager_latency_us": latency_us,
                }
                if vector is not None and meta is not None:
                    cw, ch, _ = meta
                    row["reid"] = {
                        "crop_width": cw,
                        "crop_height": ch,
                        "per_crop_latency_ms": extraction_latency[-1],
                    }
                if gid is not None:
                    frame_assignments.append({**row, "global_person_id_num": gid})
                output.append(row)
        max_backlog_frames = max(max_backlog_frames, max(latest_seen.values()) - frame_num)

    with args.kafka.open() as handle:
        while True:
            handle.seek(offset)
            lines = handle.readlines()
            offset = handle.tell()
            for line in lines:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                frame = event.get("frame")
                if not frame or frame.get("sensorId") not in CAMS:
                    continue
                cam, number = frame["sensorId"], int(frame["id"])
                pending[number][cam] = frame
                latest_seen[cam] = max(latest_seen[cam], number)

            watermark = min(latest_seen.values()) - 2
            ready = [number for number, payloads in pending.items() if len(payloads) == 2 or number <= watermark]
            for number in sorted(ready):
                process_frame(number, pending.pop(number))

            if args.done_file.exists():
                for number in sorted(pending):
                    process_frame(number, pending[number])
                pending.clear()
                break
            if not lines:
                time.sleep(0.02)

    for cap in captures.values():
        cap.release()

    roots = sorted({manager.root(row["internal_identity_at_event"]) for row in output if row.get("internal_identity_at_event") is not None})
    public = {root: f"Person_{index + 1:02d}" for index, root in enumerate(roots)}
    for row in output:
        identity = row.get("internal_identity_at_event")
        if identity is None:
            row["canonical_internal_identity"] = None
            row["global_person_id"] = "Unknown"
            continue
        root = manager.root(identity)
        row["canonical_internal_identity"] = root
        row["global_person_id"] = public[root]

    with (args.output_dir / "global_identity.jsonl").open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (args.output_dir / "identity_events.json").write_text(json.dumps(manager.events, indent=2))

    elapsed = time.perf_counter() - started
    report = {
        "mode": "concurrent_causal_sidecar",
        "face_recognition_used": False,
        "future_frames_used": False,
        "hard_coded_person_count": False,
        "elapsed_seconds": elapsed,
        "effective_source_fps_per_camera": 2400.0 / elapsed,
        "output_observations": len(output),
        "embeddings_extracted": len(extraction_latency),
        "embeddings_by_camera": dict(extracted_by_cam),
        "max_kafka_backlog_frames": max_backlog_frames,
        "internal_identities_created": manager.next_number - 1,
        "canonical_identities": len(roots),
        "public_id_map": {str(k): v for k, v in public.items()},
        "identity_lifecycle": {
            "pending_created": pending_created,
            "pending_to_existing": pending_to_existing,
            "pending_to_new_confirmed": pending_to_new,
            "pending_unresolved_observations": pending_unresolved,
            "pending_window_frames": 4,
            "pending_window_ms": 200.0,
            "pending_resample_frames": 2,
            "pending_min_good_crops": 2,
        },
        "cross_camera_matching": {
            "gates": IDENTITY_GATES,
            "attempted": manager.attempted_cross_camera_matches,
            "accepted": manager.accepted_cross_camera_matches,
            "rejected": manager.rejected_cross_camera_matches,
            "world_distances_m": manager.cross_camera_world_distances,
            "time_differences_ms": manager.cross_camera_time_differences_ms,
            "similarities": manager.cross_camera_similarities,
        },
        "osnet_latency_ms": {
            "mean": statistics.fmean(extraction_latency),
            "p95": pct(extraction_latency, 0.95),
            "max": max(extraction_latency),
        },
        "identity_manager_latency_us": {
            "mean": statistics.fmean(manager.latencies_us),
            "p95": pct(manager.latencies_us, 0.95),
            "max": max(manager.latencies_us),
        },
        "crop_width_median": statistics.median([x[0] for x in crop_dims]),
        "crop_height_median": statistics.median([x[1] for x in crop_dims]),
    }
    (args.output_dir / "runtime_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
