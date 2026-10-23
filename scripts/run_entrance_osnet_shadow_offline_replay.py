#!/usr/bin/env python3
"""Exercise the live OSNet shadow sidecar over preserved Entrance footage.

This replays the already-recorded YOLO26m/NvDCF track rows against the exact
source frames. It does not run detector/tracker inference, access RTSP, or write
identity state. Results are private, bounded diagnostics under .runtime.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import threading
import time

import cv2
import numpy as np

from services.mv3dt_room.reid_embedder import OsnetEmbedder
from services.mv3dt_staging.automatic_provisional_global_id import (
    CAMERAS,
    DEFAULT_BUNDLE,
    DEFAULT_CAM02,
    DEFAULT_CAM05,
    DEFAULT_TRACKING,
    EXPECTED_MODEL_SHA256,
    load_verified_inputs,
    score_and_assign_provisional_ids,
    sha256,
)
from services.mv3dt_staging.live_tracking.live_bridge import detector_person_class
from services.mv3dt_staging.live_tracking.native_frame_bridge import NativeFullFovScale
from services.mv3dt_staging.native_map_preview import build_preview
from services.mv3dt_staging.provisional_reid_shadow import (
    DEFAULT_SAMPLE_WINDOW_FRAMES,
    ProvisionalReIdShadow,
    crop_quality_rejection,
    crop_quality_score,
    crop_quality_xywh,
    load_preserved_control_reference,
    measure_crop_quality,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".runtime/entrance-osnet-shadow-offline-20261022-r5"
CHECKPOINT = Path(
    "/home/apsidal/nvidia/DeepStream/src/apps/reference_apps/"
    "deepstream-tracker-3d-multi-view/experiments/deepstream/"
    "dev-room-pn263-global-identity-20260921/models/reid/"
    "osnet_ain_x1_0_msmt17.pth"
)
ALIGNMENT = ROOT / ".runtime/entrance-calibration-editor/live-native-versions/LIVE_NATIVE_ALIGNMENT-v0007-20261021T123354Z.json"


def _gpu_sample() -> dict:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=2, check=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"},
    )
    values = [part.strip() for part in result.stdout.strip().split(",")]
    if len(values) != 5:
        raise RuntimeError("unexpected nvidia-smi telemetry row")
    return {
        "timestamp": values[0], "gpu_utilization_percent": int(values[1]),
        "memory_used_mib": int(values[2]), "memory_total_mib": int(values[3]),
        "temperature_c": int(values[4]),
    }


def _atomic_private_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)


def _candidate_metadata_rank(row: dict) -> tuple[float, float, float, int]:
    x, y, width, height = (float(value) for value in row["box"])
    area_fraction = width * height / (1920.0 * 1080.0)
    return (
        area_fraction,
        float(row.get("tracker_confidence", 0.0)),
        float(row.get("detector_confidence", 0.0)),
        -int(row["frame"]),
    )


def _plan_candidate_rows(tracks: dict, source_pts: dict, sample_every: int) -> tuple[dict, dict]:
    """Pick a small metadata-ranked pool in each of a track's last 3 windows.

    Image sharpness, contrast, and overlap are evaluated after exact frame
    decoding. The preliminary rank only bounds decoding work and uses the same
    area/tracker/detector fields as the existing automatic gallery selector.
    """
    by_track_window: dict[tuple[str, str, int], list[dict]] = {}
    stats = {camera: {"tracker_observations": 0, "geometry_eligible": 0,
                      "geometry_rejected": 0, "candidate_rows_preselected": 0}
             for camera in CAMERAS}
    for camera in CAMERAS:
        for frame, frame_rows in tracks[camera].items():
            for row in frame_rows:
                stats[camera]["tracker_observations"] += 1
                box = [float(value) for value in row["box"]]
                useful, _geometry = crop_quality_xywh(box)
                if not useful:
                    stats[camera]["geometry_rejected"] += 1
                    continue
                stats[camera]["geometry_eligible"] += 1
                local_id = str(row["object_id"])
                window = int(frame) // DEFAULT_SAMPLE_WINDOW_FRAMES
                candidate = {
                    "camera_id": camera,
                    "source_id": 0 if camera == "CAM-02" else 1,
                    "local_track_id": local_id,
                    "frame": int(frame),
                    "source_pts_ns": int(source_pts[camera][int(frame)]),
                    "box": box,
                    "detector_confidence": float(row.get("detector_confidence", 0.0)),
                    "tracker_confidence": float(row.get("tracker_confidence", 0.0)),
                    "window_index": window,
                }
                by_track_window.setdefault((camera, local_id, window), []).append(candidate)

    planned: dict[str, dict[int, list[dict]]] = {camera: {} for camera in CAMERAS}
    for (camera, local_id), all_keys in {
        (key[0], key[1]): None for key in by_track_window
    }.items():
        windows = sorted(key[2] for key in by_track_window if key[:2] == (camera, local_id))[-3:]
        for window in windows:
            candidates = sorted(
                by_track_window[(camera, local_id, window)],
                key=_candidate_metadata_rank, reverse=True,
            )
            chosen = []
            for candidate in candidates:
                if any(abs(candidate["frame"] - prior["frame"]) < sample_every for prior in chosen):
                    continue
                chosen.append(candidate)
                if len(chosen) >= 3:
                    break
            stats[camera]["candidate_rows_preselected"] += len(chosen)
            for candidate in chosen:
                planned[camera].setdefault(candidate["frame"], []).append(candidate)
    return planned, stats


def run(args: argparse.Namespace) -> dict:
    os.umask(0o077)
    out = args.output.resolve()
    if not out.is_relative_to((ROOT / ".runtime").resolve()):
        raise ValueError("offline shadow output must remain under this checkout's .runtime")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing replay evidence: {out}")

    videos = {"CAM-02": args.cam02.resolve(), "CAM-05": args.cam05.resolve()}
    _manifest, _old_vectors, _old_records, tracks, source_pts = load_verified_inputs(
        bundle_dir=args.bundle.resolve(), tracking_dir=args.tracking_dir.resolve(),
        video_paths=videos, enforce_preserved_hashes=True,
    )
    if sha256(CHECKPOINT) != EXPECTED_MODEL_SHA256:
        raise ValueError("accepted OSNet checkpoint hash mismatch")
    if not ALIGNMENT.is_file():
        raise FileNotFoundError("saved LIVE_NATIVE_ALIGNMENT-v0007 is missing")

    gpu_before = [_gpu_sample() for _ in range(3)]
    initial_util = [row["gpu_utilization_percent"] for row in gpu_before]
    if max(initial_util) >= 80:
        raise RuntimeError("offline shadow replay withheld: GPU baseline is at/above 80%")
    if min(row["memory_total_mib"] - row["memory_used_mib"] for row in gpu_before) < 3000:
        raise RuntimeError("offline shadow replay withheld: less than 3 GiB free GPU memory")

    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cfg = {
        "model_name": "osnet_ain_x1_0", "model_path": str(CHECKPOINT).lstrip("/"),
        "model_sha256": EXPECTED_MODEL_SHA256, "download_if_missing": False,
        "input_height": 256, "input_width": 128, "device": args.device,
        "cpu_threads": 2,
    }
    embedder = OsnetEmbedder(cfg, Path("/"))
    embedder._load()
    warmup_started = time.perf_counter()
    warmup = embedder.embed_batch([np.zeros((256, 128, 3), dtype=np.uint8)])
    if warmup.shape != (1, 512) or not np.isfinite(warmup).all():
        raise RuntimeError("OSNet preflight warmup failed")
    warmup_ms = (time.perf_counter() - warmup_started) * 1000.0
    reference_max, reference_source = load_preserved_control_reference()
    candidate_plan, candidate_plan_stats = _plan_candidate_rows(
        tracks, source_pts, args.sample_every,
    )
    candidate_frames = {
        camera: sorted(candidate_plan[camera]) for camera in CAMERAS
    }
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(out, 0o700)
    sample_dir = out / "quality_gallery_crops"
    sample_dir.mkdir(mode=0o700)
    sidecar = ProvisionalReIdShadow(
        embedder, control_reference_max=reference_max,
        control_reference_source=reference_source,
        person_class_id=detector_person_class("yolo26m"),
        sample_every_frames=args.sample_every,
        max_crops_per_track=3, sample_window_frames=DEFAULT_SAMPLE_WINDOW_FRAMES,
        queue_size=32, max_gallery_tracks=128,
        control_history_frames=300, batch_size=8,
        decision_diagnostics=args.decision_diagnostics,
    )

    stop_monitor = threading.Event()
    gpu_samples: list[dict] = []
    monitor_errors: list[str] = []

    def sample_gpu() -> None:
        while not stop_monitor.wait(0.5):
            try:
                gpu_samples.append(_gpu_sample())
            except Exception as exc:  # telemetry failure is reported, not hidden
                monitor_errors.append(type(exc).__name__)

    monitor = threading.Thread(target=sample_gpu, name="offline-shadow-gpu-telemetry", daemon=True)
    monitor.start()
    replay_start = time.perf_counter()
    cpu_start = time.process_time()
    frame_counts: dict[str, int] = {}
    observations_submitted: dict[str, int] = {}
    sample_candidate_frames = {camera: len(candidate_frames[camera]) for camera in CAMERAS}
    per_camera_seconds: dict[str, float] = {}
    capture_errors: list[str] = []
    crop_quality_rejections = {camera: {} for camera in CAMERAS}
    measured_candidate_rows: list[dict] = []
    selected_gallery: dict[tuple[str, str, int], dict] = {}
    try:
        # Decode a bounded metadata-ranked pool in the last three 120-frame
        # windows per local track. The exact source frame is resized with the
        # live scale-only transform. Within each time window, select by image
        # sharpness/contrast, size, detector/tracker score, and an explicit
        # other-box overlap proxy; this avoids first-eligible crop bias without
        # decoding every frame in the 180-second source.
        for camera in CAMERAS:
            cap = cv2.VideoCapture(str(videos[camera]))
            if not cap.isOpened():
                raise RuntimeError(f"could not open preserved offline {camera} video")
            scale = NativeFullFovScale(camera)
            start_camera = time.perf_counter()
            count = 0
            try:
                for frame_index in candidate_frames[camera]:
                    if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                        raise RuntimeError(f"{camera}: decoder refused exact seek to frame {frame_index}")
                    ok, native = cap.read()
                    if not ok:
                        raise RuntimeError(f"{camera}: could not decode sample frame {frame_index}")
                    decoded_after = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                    if decoded_after != frame_index + 1:
                        raise RuntimeError(
                            f"{camera}: decoder frame position mismatch at {frame_index} (reported {decoded_after - 1})"
                        )
                    inference_frame = scale(native)
                    frame_rows = tracks[camera].get(frame_index, [])
                    candidates = candidate_plan[camera][frame_index]
                    boxes_by_id = {str(row["object_id"]): row["box"] for row in frame_rows}
                    for candidate in candidates:
                        box = candidate["box"]
                        x, y, width, height = box
                        crop = inference_frame[max(0, int(y)):min(1080, int(np.ceil(y + height))),
                                               max(0, int(x)):min(1920, int(np.ceil(x + width)))]
                        quality = measure_crop_quality(
                            crop, box,
                            [other_box for local_id, other_box in boxes_by_id.items()
                             if local_id != candidate["local_track_id"]],
                            detector_confidence=candidate["detector_confidence"],
                            tracker_confidence=candidate["tracker_confidence"],
                        )
                        useful, geometry = crop_quality_xywh(box)
                        quality.update(geometry)
                        reason = crop_quality_rejection(quality) if useful else "bbox_geometry"
                        entry = {
                            **candidate,
                            "crop_quality": quality,
                            "rejection_reason": reason,
                        }
                        measured_candidate_rows.append(entry)
                        if reason:
                            counters = crop_quality_rejections[camera]
                            counters[reason] = counters.get(reason, 0) + 1
                            continue
                        key = (camera, candidate["local_track_id"], candidate["window_index"])
                        previous = selected_gallery.get(key)
                        if (previous is None or
                                quality["relative_quality_score"] >
                                previous["crop_quality"]["relative_quality_score"]):
                            selected_gallery[key] = {**entry, "crop": crop.copy()}
                    count += 1
                per_camera_seconds[camera] = time.perf_counter() - start_camera
            finally:
                cap.release()

        selected_by_camera: dict[str, dict[int, list[dict]]] = {camera: {} for camera in CAMERAS}
        for (camera, _local_id, _window), entry in selected_gallery.items():
            selected_by_camera[camera].setdefault(entry["frame"], []).append(entry)
        for camera in CAMERAS:
            rows_submitted = 0
            cap = cv2.VideoCapture(str(videos[camera]))
            if not cap.isOpened():
                raise RuntimeError(f"could not reopen preserved {camera} source for exact sidecar submission")
            try:
                for frame_index in sorted(selected_by_camera[camera]):
                    entries = selected_by_camera[camera][frame_index]
                    # The winner crop is already quality-scored; preserve the
                    # exact-frame sidecar join with one sequential capture.
                    if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                        raise RuntimeError(f"{camera}: sidecar replay seek failed at frame {frame_index}")
                    ok, native = cap.read()
                    if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != frame_index + 1:
                        raise RuntimeError(f"{camera}: sidecar replay frame index mismatch at {frame_index}")
                    inference_frame = NativeFullFovScale(camera)(native)
                    pts = int(source_pts[camera][frame_index])
                    objects = [{
                        "class_id": detector_person_class("yolo26m"),
                        "tracker_id": row["local_track_id"],
                        "left": row["box"][0], "top": row["box"][1],
                        "width": row["box"][2], "height": row["box"][3],
                        "confidence": row["detector_confidence"],
                        "tracker_confidence": row["tracker_confidence"],
                    } for row in entries]
                    sidecar.submit_frame(
                        source_id=0 if camera == "CAM-02" else 1,
                        camera_id=camera, frame_index=frame_index, source_pts_ns=pts,
                        cached_source_id=0 if camera == "CAM-02" else 1,
                        cached_frame_index=frame_index, cached_source_pts_ns=pts,
                        inference_frame=inference_frame, objects=objects,
                    )
                    rows_submitted += len(entries)
            finally:
                cap.release()
            frame_counts[camera] = len(selected_by_camera[camera])
            observations_submitted[camera] = rows_submitted
    finally:
        summary_before_close = sidecar.close(timeout_sec=20.0)
        stop_monitor.set()
        monitor.join(timeout=2.0)
    elapsed = time.perf_counter() - replay_start
    gpu_after = [_gpu_sample() for _ in range(2)]
    gpu_samples.extend(gpu_after)
    snapshot = sidecar.snapshot()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_gpu = max((row["gpu_utilization_percent"] for row in gpu_samples), default=None)
    max_mem = max((row["memory_used_mib"] for row in gpu_samples), default=None)
    before_mem = gpu_before[-1]["memory_used_mib"]
    diagnostics = score_and_assign_provisional_ids(
        np.asarray(sidecar._vectors, dtype=np.float32).reshape((-1, 512)),
        list(sidecar._records), tracks,
        minimum_crops_per_track=2, minimum_separated_control_frames=8,
        control_reference_max=reference_max,
        control_reference_source=reference_source,
        duplicate_track_conflicts=set(sidecar._duplicate_track_conflicts),
        diagnostic_logging=args.decision_diagnostics,
        diagnostic_now_ns=time.monotonic_ns() if args.decision_diagnostics else None,
    )
    tracks_count = {
        camera: len({str(row["object_id"]) for frame_rows in tracks[camera].values() for row in frame_rows})
        for camera in CAMERAS
    }
    tracker_observation_count = {
        camera: sum(len(frame_rows) for frame_rows in tracks[camera].values())
        for camera in CAMERAS
    }
    active_pairs = diagnostics.get("matches", [])
    provisional_map: dict[str, dict[str, dict[str, Any]]] = {camera: {} for camera in CAMERAS}
    for match in active_pairs:
        gid = match["provisional_global_id"]
        for camera in CAMERAS:
            local_id = str(match["camera_local_pair"][camera])
            provisional_map[camera][local_id] = {
                "provisional_global_id": gid,
                "status": "AUTO_PROVISIONAL_ONLY",
                "median_cosine_similarity": match["cosine_similarity_median"],
            }
    quality_gallery_rows = []
    for (camera, local_id, window), entry in sorted(selected_gallery.items()):
        crop_name = f"{camera}_local-{local_id}_window-{window}_f{int(entry['frame']):06d}.jpg"
        crop_path = sample_dir / crop_name
        if not cv2.imwrite(str(crop_path), entry["crop"], [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"could not save selected quality gallery crop: {crop_name}")
        quality_gallery_rows.append({
            "camera_id": camera, "local_track_id": local_id,
            "window_index": int(window), "frame": int(entry["frame"]),
            "source_pts_ns": int(entry["source_pts_ns"]),
            "source_id": int(entry["source_id"]),
            "bbox_inference_xywh": entry["box"],
            "detector_confidence": entry["detector_confidence"],
            "tracker_confidence": entry["tracker_confidence"],
            "crop_quality": entry["crop_quality"],
            "private_crop_path": str(crop_path),
        })
    preview_manifest = build_preview(
        alignment_path=ALIGNMENT,
        video_paths={"cam02": videos["CAM-02"], "cam05": videos["CAM-05"]},
        layout_path=Path("/home/apsidal/auto-magic-calib/projects/project_20261019_101845_7932/manual_adjustment/layout.png"),
        tracking_dir=args.tracking_dir.resolve(), output_dir=out / "preview",
        provisional_global_ids=provisional_map,
    )
    decision_events = summary_before_close.pop("decision_diagnostic_events", [])
    decision_diagnostics_path = None
    if args.decision_diagnostics:
        decision_diagnostics_path = out / "matcher-decision-diagnostics.jsonl"
        with decision_diagnostics_path.open("x", encoding="utf-8") as stream:
            os.chmod(decision_diagnostics_path, 0o600)
            for event in decision_events:
                stream.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    result = {
        "schema_version": 1,
        "status": "PASS" if not capture_errors and snapshot["last_error"] is None
                  and snapshot["last_join_error"] is None and not snapshot["worker_alive"] else "FAIL",
        "mode": "OFFLINE_REPLAY_OF_PRESERVED_YOLO26M_NVDCF_TRACKS_WITH_REAL_OSNET_SHADOW_SIDECAR",
        "physical_rtsp_opened": False,
        "detector_or_tracker_inference_rerun": False,
        "production_identity_write": False,
        "human_identity_labels_consumed": False,
        "source_video_sha256": {camera: sha256(path) for camera, path in videos.items()},
        "tracking_run": str(args.tracking_dir.resolve()),
        "tracking_provenance_sha256": sha256(args.tracking_dir / "provenance.json"),
        "source_frame_crosswalk_sha256": sha256(args.tracking_dir / "source-frames.jsonl"),
        "tracker_rows_sha256": sha256(args.tracking_dir / "tracks.jsonl"),
        "source_order": {"source_id_0": "CAM-02", "source_id_1": "CAM-05"},
        "source_pts": "preserved per-file PTS crosswalk; not used as inter-camera synchronization",
        "inference_geometry": "1920x1080 scale-only full field of view; same NativeFullFovScale used by live bridge",
        "alignment": {
            "name": "LIVE_NATIVE_ALIGNMENT-v0007",
            "sha256": sha256(ALIGNMENT),
            "use": "display-only; not used in identity matching",
        },
        "osnet": {
            "checkpoint_sha256": EXPECTED_MODEL_SHA256,
            "device": args.device,
            "feature_dimension": 512,
            "l2_normalized": True,
            "synthetic_warmup_ms": warmup_ms,
            "sidecar_sample_interval_frames": args.sample_every,
            "quality_window_frames": DEFAULT_SAMPLE_WINDOW_FRAMES,
            "bounded_gallery_samples_per_track": 3,
            "sampler": "quality-ranked sample per 120-frame temporal window; keep up to the last three windows per local track",
            "preprocessing_contract": "same OsnetEmbedder: BGR crop -> resize 128x256 INTER_LINEAR -> RGB -> [0,1] -> ImageNet mean/std -> L2 normalize",
        },
        "offline_replay": {
            "sample_frames_decoded": frame_counts,
            "quality_candidate_frames_decoded": sample_candidate_frames,
            "quality_candidate_plan_by_camera": candidate_plan_stats,
            "quality_candidate_count": len(measured_candidate_rows),
            "quality_rejected_counts_by_camera": crop_quality_rejections,
            "quality_selected_gallery_count": len(quality_gallery_rows),
            "quality_gallery_samples": quality_gallery_rows,
            "quality_gallery_crop_dir": str(sample_dir),
            "quality_policy": {
                "selection": "metadata-ranked crop candidates, then image-quality ranking within 120-frame windows; area, Laplacian sharpness, contrast, detector/tracker confidence, other-person-box overlap proxy",
                "hard_rejects": "existing geometry policy; Laplacian variance <25 AND grayscale stddev <6; max other-person-box intersection / target area >=0.80",
                "overlap_is_occlusion_ground_truth": False,
                "quality_score_is_identity_score": False,
                "cross_camera_sync_used": False,
            },
            "measured_candidate_quality_rows": measured_candidate_rows,
            "all_tracker_metadata_frames_validated_by_hash_pinned_crosswalk": {
                camera: len(source_pts[camera]) for camera in CAMERAS
            },
            "tracker_observations_submitted_at_sample_frames": observations_submitted,
            "all_tracker_observations_in_preserved_crosswalk": tracker_observation_count,
            "local_track_counts": tracks_count,
            "wall_sec": elapsed,
            "per_camera_wall_sec": per_camera_seconds,
            "source_nominal_fps": "CAM-02 1439200/72049; CAM-05 20/1",
            "sample_frame_decode_rate_fps": {
                camera: frame_counts.get(camera, 0) / float(per_camera_seconds.get(camera, 1.0))
                if per_camera_seconds.get(camera) else None for camera in CAMERAS
            },
            "capture_errors": capture_errors,
            "exact_sample_decode_position_check": "PASS; CAP_PROP_POS_FRAMES matched each requested original frame index",
            "source_pts_interpretation": "exact per-camera source PTS retained; no inter-camera exposure synchronization claim",
        },
        "association": {
            "active_provisional_ids": active_pairs,
            "candidate_pairs_last_update": len(diagnostics.get("candidate_pairs", [])),
            "uncertain_pairs_last_update": sum(
                not row.get("automatic_candidate", False) for row in diagnostics.get("candidate_pairs", [])
            ),
            "one_to_many_conflicts": len(diagnostics.get("one_to_many_conflicts_rejected", [])),
            "same_frame_contradictory_assignments": snapshot["same_frame_contradictory_assignments"],
            "duplicate_track_conflict_pairs": len(diagnostics.get("duplicate_track_pairs", [])),
            "possible_local_fragmentation_candidates": len(diagnostics.get("potential_local_fragmentation_candidates", [])),
            "revision_events": snapshot["revision_events"],
            "matcher_decision_diagnostics": {
                "enabled": args.decision_diagnostics,
                "event_count": len(decision_events),
                "dropped_at_bound": summary_before_close.get(
                    "decision_diagnostic_events_dropped_at_bound", 0
                ),
                "path": str(decision_diagnostics_path) if decision_diagnostics_path else None,
                "schema": "bounded JSONL; camera-local IDs, source frame/PTS, matcher gates and withdrawal reason; no embeddings or crops",
            },
            "final_match_diagnostics": diagnostics,
            "cross_camera_sync_used_for_match": False,
            "geometry_used_for_match": False,
            "cosine_is_probability": False,
            "false_match_rate": "UNMEASURED: no independently verified positive/hard-negative labels",
        },
        "sidecar_runtime": summary_before_close,
        "osnet_embedder_metrics": embedder.metrics(),
        "recorded_demo": {
            "preview_video": preview_manifest["video"],
            "representative_screenshot": preview_manifest["representative_review_frame"],
            "mode": "recorded CAM-02/CAM-05 video plus preserved NvDCF local tracks; not live",
            "map": "EXPERIMENTAL / UNCALIBRATED",
            "camera_local_ids_remain_separate": True,
            "provisional_ids": "only current conservative-gate candidates; not verified identity",
        },
        "resource": {
            "gpu_baseline_samples": gpu_before,
            "gpu_samples_during_replay": gpu_samples,
            "gpu_max_utilization_sample_percent": max_gpu,
            "gpu_memory_before_mib": before_mem,
            "gpu_memory_peak_sample_mib": max_mem,
            "gpu_memory_total_mib": gpu_samples[0]["memory_total_mib"] if gpu_samples else None,
            "gpu_sample_note": "nvidia-smi polling can miss short kernels; device-total utilization only",
            "process_cpu_sec": time.process_time() - cpu_start,
            "process_peak_rss_mib": peak / 1024.0,
            "gpu_monitor_errors": monitor_errors,
            "torch_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / (1024**2))
                if args.device == "cuda" else None,
            "torch_peak_reserved_mib": float(torch.cuda.max_memory_reserved() / (1024**2))
                if args.device == "cuda" else None,
        },
        "limitations": [
            "This replays saved YOLO26m/NvDCF outputs; it does not re-run the detector or tracker.",
            "A provisional match is an appearance candidate only, not verified person identity.",
            "No ground-truth identity labels were consumed; false-match rate and identity accuracy are unmeasured.",
            "Same-camera separation controls are not DIFFERENT_PEOPLE labels; duplicate tracking remains possible.",
            "Equal frame numbers and source-relative PTS do not establish cross-camera exposure synchronization.",
            "The experimental v0007 map is not used as identity evidence and remains uncalibrated.",
            "No simultaneous YOLO26m/NvDCF + OSNet + PySide6 GPU load was measured; LIVE shadow performance remains unproven until combined workload validation.",
        ],
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_private_json(out / "offline_shadow_replay.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--tracking-dir", type=Path, default=DEFAULT_TRACKING)
    parser.add_argument("--cam02", type=Path, default=DEFAULT_CAM02)
    parser.add_argument("--cam05", type=Path, default=DEFAULT_CAM05)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument(
        "--decision-diagnostics", action="store_true",
        help="write a bounded private matcher gate/withdrawal trace (does not alter matching rules)",
    )
    args = parser.parse_args()
    if args.sample_every < 1:
        parser.error("--sample-every must be positive")
    result = run(args)
    print(json.dumps({
        "status": result["status"], "output": str(args.output.resolve() / "offline_shadow_replay.json"),
        "offline_replay": result["offline_replay"],
        "association": {key: result["association"][key] for key in (
            "active_provisional_ids", "candidate_pairs_last_update", "uncertain_pairs_last_update",
            "one_to_many_conflicts", "duplicate_track_conflict_pairs", "possible_local_fragmentation_candidates",
        )},
        "resource": {key: result["resource"][key] for key in (
            "gpu_max_utilization_sample_percent", "gpu_memory_before_mib", "gpu_memory_peak_sample_mib",
            "process_cpu_sec", "process_peak_rss_mib", "torch_peak_allocated_mib", "torch_peak_reserved_mib",
        )},
        "sidecar_timing_ms": result["sidecar_runtime"].get("timing_ms"),
    }, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
