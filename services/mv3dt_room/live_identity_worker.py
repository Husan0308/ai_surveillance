#!/usr/bin/env python3
"""Real-time, latest-state GlobalIdentityManager adapter for the Dev Room pair."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from services.mv3dt_room.deepstream_crop_receiver import DeepStreamCropReceiver
from services.mv3dt_room.extract_osnet import H, W, crop_quality
from services.mv3dt_room.global_identity_manager import CAMS, IDENTITY_GATES, GlobalIdentityManager, bbox_iou, bbox_of, distance, world_of
from services.mv3dt_room.identity_gallery_store import IdentityGalleryStore
from services.mv3dt_room.identity_metrics import IdentityMetrics, monotonic_ns, wall_timestamp_ns
from services.mv3dt_room.identity_queue import AsyncOsnetBatcher, LatestObservationQueue, ObservationEnvelope, ReIdResult
from services.mv3dt_room.reid_embedder import OsnetEmbedder


def iso_timestamp(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1_000_000_000.0, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def matching_only_crop_quality(quality: dict) -> bool:
    """Permit informative borderline crops for matching, never for gallery."""
    box = quality.get("quality_bbox") or ()
    bottom_edge_only = (len(box) == 4
        and float(box[0]) > 2.0 and float(box[1]) > 2.0
        and float(box[2]) < W - 2.0
        and H - 2.0 <= float(box[3]) <= H + 2.0)
    return (0.16 <= float(quality.get("aspect", 0.0)) <= 1.05
            and float(quality.get("width", 0.0)) >= 28.0
            and float(quality.get("height", 0.0)) >= 96.0
            and float(quality.get("area_fraction", 0.0)) >= 0.003
            and float(quality.get("inside_fraction", 0.0)) >= 0.97
            and (not quality.get("border_truncated", True) or bottom_edge_only))


@dataclass
class TrackState:
    camera_id: str
    native_track_id: int
    generation: int = 1
    last_seen_frame: int = -1
    last_seen_receive_monotonic_ns: int = 0
    latest: dict | None = None
    canonical_internal_identity: int | None = None
    pending_generation: int | None = None
    pending_frame: int = -1
    last_accepted_frame: int = -1
    terminated: bool = False
    termination_emitted: bool = False
    identity_state: str = "PENDING"
    pending_started_frame: int = -1
    pending_deadline_frame: int = -1
    pending_attempts: int = 0
    pending_good_crops: int = 0
    pending_last_request_frame: int = -10**9
    pending_request_frame: int = -1
    pending_was_reacquisition: bool = False
    crop_health_timeout_emitted: bool = False
    pending_crop_request_frames: list[int] = field(default_factory=list)
    pending_crop_received_frames: list[int] = field(default_factory=list)
    pending_crop_quality_rejected: int = 0
    pending_embedding_completed_frames: list[int] = field(default_factory=list)


class LiveIdentityWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audit_handle = (self.output_dir / "identity_path_trace.jsonl").open(
            "w", buffering=1
        )
        self.stop_requested = False
        configured_gallery_db = getattr(args, "gallery_db", None) or os.getenv("MV3DT_IDENTITY_GALLERY_DB")
        if configured_gallery_db:
            gallery_db = Path(configured_gallery_db)
        else:
            # Session output is .../<mode-session>/identity-live; the parent
            # of the session directory is the durable Dev Room profile root.
            gallery_db = self.output_dir.parent.parent / "identity_gallery.sqlite3"
        self.gallery_store = IdentityGalleryStore(gallery_db)
        self.manager = GlobalIdentityManager(
            np.empty((0, 512), np.float32), gallery_store=self.gallery_store
        )
        self.metrics = IdentityMetrics()
        self.public_ids: dict[int, str] = dict(self.manager.application_ids)
        self.next_public = max(
            [int(value.rsplit("_", 1)[-1]) for value in self.public_ids.values() if value.rsplit("_", 1)[-1].isdigit()]
            or [0]
        ) + 1
        self.track_states: dict[tuple[str, int], TrackState] = {}
        self.latest_camera_frame = {camera: -1 for camera in CAMS}
        self.latest_camera_rows: dict[str, list[dict]] = {camera: [] for camera in CAMS}
        self.last_published_presence: set[tuple[str, int, str]] = set()
        self.output_rows: list[dict] = []
        self.decision_trace_rows: list[dict] = []
        self.persist_embeddings = os.environ.get("MV3DT_IDENTITY_DEBUG_PERSIST_EMBEDDINGS", "0") == "1"
        self.embedding_debug_path = self.output_dir / "embedding_debug.jsonl"
        self.reacquisition_frames: list[float] = []
        self.identity_resolution_frames: list[float] = []
        self.received_frames = defaultdict(int)
        self.last_report = 0.0
        self.last_publish = 0.0
        self.started = time.perf_counter()
        self.kafka_offset = 0
        # Derived from the current 20 FPS source cadence and measured CUDA
        # OSNet mean latency (~51 ms): three inference latencies span 4 frames.
        self.osnet_measured_mean_ms = 51.0
        self.pending_window_frames = max(4, int(math.ceil(
            3.0 * self.osnet_measured_mean_ms / 1000.0 * IDENTITY_GATES["source_fps"])))
        self.pending_resample_frames = max(1, int(math.ceil(
            self.osnet_measured_mean_ms / 1000.0 * IDENTITY_GATES["source_fps"])))
        self.pending_min_good_crops = 2
        self.crop_receiver = DeepStreamCropReceiver(args.crop_socket, self.metrics)
        self.pending_crop_waiting: set[tuple[str, int]] = set()
        self.observation_queue = LatestObservationQueue(max_tracks=256, metrics=self.metrics)

        config = {
            "model_name": "osnet_ain_x1_0",
            "model_path": str(args.osnet_model),
            "model_sha256": "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f",
            "download_if_missing": False,
            "input_height": 256,
            "input_width": 128,
            "device": "cuda",
            "cpu_threads": 2,
        }
        self.embedder = OsnetEmbedder(config, Path("/"))
        self.reid = AsyncOsnetBatcher(self.embedder, self.metrics, batch_size=8, batching_window_ms=12.0, max_pending=256)

    def _audit(self, event: str, **fields) -> None:
        """Flush acceptance diagnostics as append-only JSONL in this run's .runtime."""
        self.audit_handle.write(json.dumps(
            {"event": event, "wall_timestamp": iso_timestamp(wall_timestamp_ns()), **fields},
            separators=(",", ":"), default=str,
        ) + "\n")
        self.audit_handle.flush()

    def public_id(self, internal: int) -> str:
        root = self.manager.root(internal)
        if root not in self.public_ids:
            self.public_ids[root] = self.manager.application_id(root)
            self.next_public = max(self.next_public, int(self.public_ids[root].rsplit("_", 1)[-1]) + 1)
        return self.public_ids[root]

    def _make_observation(self, camera: str, frame: int, payload: dict, obj: dict, receive_mono: int, receive_wall: int) -> dict:
        started = monotonic_ns()
        obs = {
            "camera_id": camera,
            "frame": frame,
            "source_frame_number": frame,
            "timestamp": payload.get("timestamp"),
            "source_timestamp": payload.get("timestamp"),
            "native_track_id": int(obj["id"]),
            "bbox": bbox_of(obj),
            "world": world_of(obj),
            "confidence": float(obj.get("confidence", 0.0)),
            "visibility": float(obj.get("info", {}).get("visibility", 0.0)),
            "receive_timestamp": iso_timestamp(receive_wall),
            "receive_wall_timestamp_ns": receive_wall,
            "receive_monotonic_ns": receive_mono,
        }
        self.metrics.observe("bbox_world_adapter_latency_ms", (monotonic_ns() - started) / 1_000_000.0, camera)
        return obs

    def _begin_pending(self, state: TrackState, frame: int, reacquisition: bool = False) -> None:
        state.identity_state = "PENDING"
        state.pending_started_frame = int(frame)
        state.pending_deadline_frame = int(frame) + self.pending_window_frames
        state.pending_attempts = 0
        state.pending_good_crops = 0
        state.pending_last_request_frame = -10**9
        state.pending_request_frame = -1
        state.pending_was_reacquisition = reacquisition
        state.crop_health_timeout_emitted = False

    def _track(self, obs: dict) -> tuple[TrackState, bool]:
        key = (obs["camera_id"], obs["native_track_id"])
        state = self.track_states.get(key)
        critical = state is None or state.terminated
        if state is None:
            state = TrackState(*key)
            self.track_states[key] = state
            self._begin_pending(state, obs["frame"], reacquisition=False)
            self.metrics.inc("native_track_births", camera_id=obs["camera_id"])
            self.metrics.inc("pending_created", camera_id=obs["camera_id"])
        elif state.terminated:
            state.generation += 1
            state.terminated = False
            state.termination_emitted = False
            state.canonical_internal_identity = None
            state.pending_generation = None
            state.last_accepted_frame = -1
            self.manager.unbind_track(*key)
            self._begin_pending(state, obs["frame"], reacquisition=True)
            self.metrics.inc("native_track_reacquisitions", camera_id=obs["camera_id"])
            self.metrics.inc("pending_created", camera_id=obs["camera_id"])
        state.last_seen_frame = obs["frame"]
        state.last_seen_receive_monotonic_ns = int(obs["receive_monotonic_ns"])
        state.latest = obs
        return state, critical

    def _mark_terminations(self, camera: str, current_keys: set[tuple[str, int]], frame: int) -> None:
        for key, state in self.track_states.items():
            if key[0] != camera or state.terminated or state.last_seen_frame < 0:
                continue
            if key in current_keys or frame - state.last_seen_frame < self.args.termination_gap:
                continue
            state.terminated = True
            state.pending_generation = None
            self.pending_crop_waiting.discard(key)
            self.manager.unbind_track(*key)
            self._audit(
                "native_track_termination", camera_id=camera,
                native_track_id=key[1], last_frame=state.last_seen_frame,
                observed_termination_frame=frame, gap_frames=frame - state.last_seen_frame,
                canonical_identity=self.public_id(state.canonical_internal_identity)
                if state.canonical_internal_identity is not None else None,
                prior_identity_state=state.identity_state,
            )
            if not state.termination_emitted:
                state.termination_emitted = True
                self.metrics.inc("native_track_terminations", camera_id=camera)

    def _ingest_frame(self, camera: str, frame: int, payload: dict, receive_mono: int, receive_wall: int) -> None:
        self.latest_camera_frame[camera] = frame
        self.received_frames[camera] += 1
        objects = [item for item in payload.get("objects", []) if item.get("type", "").lower() == "person"]
        current_keys: set[tuple[str, int]] = set()
        rows: list[dict] = []
        for obj in objects:
            obs = self._make_observation(camera, frame, payload, obj, receive_mono, receive_wall)
            key = (camera, obs["native_track_id"])
            current_keys.add(key)
            rows.append(obs)
            state, critical = self._track(obs)
            self._audit(
                "production_identity_input", observation=obs,
                native_track_generation=state.generation, native_track_birth=critical,
                tracker_confidence=obj.get("trackerConfidence", obj.get("tracker_confidence")),
                tracker_state=obj.get("trackerState", obj.get("tracker_state")),
                tracker_age=obj.get("trackerAge", obj.get("tracker_age")),
                kafka_object_info=obj.get("info", {}),
            )
            self.observation_queue.submit(ObservationEnvelope(obs, state.generation, critical=critical))
        self.latest_camera_rows[camera] = rows
        self._mark_terminations(camera, current_keys, frame)

    def _active_assignments(self, exclude: tuple[str, int] | None = None) -> list[dict]:
        result = []
        for key, state in self.track_states.items():
            if exclude is not None and key == exclude:
                continue
            if state.terminated or state.latest is None or state.canonical_internal_identity is None:
                continue
            # A track retained for termination/reacquisition memory is not a
            # simultaneous observation after its camera advances past it.
            # Published camera presence follows the same current-frame rule.
            if int(state.latest["frame"]) != self.latest_camera_frame[state.camera_id]:
                continue
            row = dict(state.latest)
            row["global_person_id_num"] = self.manager.root(state.canonical_internal_identity)
            result.append(row)
        return result

    def _crop_for_observation(self, obs: dict) -> np.ndarray | None:
        track_state = self.track_states.get((obs["camera_id"], int(obs["native_track_id"])))
        record = self.crop_receiver.get(obs)
        if record is None and track_state is not None and track_state.identity_state == "PENDING":
            # The encoder and Kafka metadata are asynchronous. If the exact
            # observation frame was already superseded before JPEG delivery,
            # consume only a buffered crop from this same track/generation at
            # or before the current state frame.
            record = self.crop_receiver.get_for_track(
                obs["camera_id"], int(obs["native_track_id"]),
                track_state.generation, int(obs["frame"]),
                max(int(track_state.pending_started_frame), 0),
            )
        wait_key = (str(obs["camera_id"]), int(obs["native_track_id"]))
        if record is None:
            self._audit(
                "crop_unavailable_for_observation", camera_id=obs["camera_id"],
                frame=obs["frame"], native_track_id=obs["native_track_id"],
                track_generation=track_state.generation if track_state else None,
                pending_started_frame=track_state.pending_started_frame if track_state else None,
            )
            self.metrics.inc("crop_delivery_waits", camera_id=obs["camera_id"])
            if wait_key not in self.pending_crop_waiting:
                self.pending_crop_waiting.add(wait_key)
                self.metrics.inc("waiting_for_crop", camera_id=obs["camera_id"])
                self.metrics.inc("pending_tracks_waiting_for_crop", camera_id=obs["camera_id"])
                self.metrics.set_max("pending_tracks_waiting_for_crop_max", len(self.pending_crop_waiting))
            return None
        self.pending_crop_waiting.discard(wait_key)
        self.metrics.inc("crop_received", camera_id=obs["camera_id"])
        if track_state is not None:
            track_state.pending_crop_received_frames.append(int(record.header.get("frame_num", obs["frame"])))

        header = record.header
        expected = {
            "camera_id": str(obs["camera_id"]),
            "native_track_id": int(obs["native_track_id"]),
        }
        try:
            actual = {
                "camera_id": str(header["camera_id"]),
                "native_track_id": int(header["native_track_id"]),
                "frame_num": int(header["frame_num"]),
            }
        except (KeyError, TypeError, ValueError):
            self._audit("crop_rejected_provenance", camera_id=obs["camera_id"],
                        frame=obs["frame"], native_track_id=obs["native_track_id"],
                        reason="required crop header fields missing", header=header)
            self.metrics.inc("crop_failures", camera_id=obs["camera_id"])
            return None
        if actual["camera_id"] != expected["camera_id"] or actual["native_track_id"] != expected["native_track_id"]:
            self._audit("crop_rejected_provenance", expected=expected, actual=actual,
                        reason="camera_or_native_track_mismatch")
            self.metrics.inc("crop_provenance_rejected", camera_id=obs["camera_id"])
            return None
        if actual["frame_num"] > int(obs["frame"]):
            self._audit("crop_rejected_provenance", expected=expected, actual=actual,
                        reason="crop_from_future_frame")
            self.metrics.inc("crop_provenance_rejected", camera_id=obs["camera_id"])
            return None
        if track_state is None:
            self._audit("crop_rejected_provenance", expected=expected, actual=actual,
                        reason="track_state_missing")
            self.metrics.inc("crop_provenance_rejected", camera_id=obs["camera_id"])
            return None
        if actual["frame_num"] < max(int(track_state.pending_started_frame), 0):
            self._audit("crop_rejected_provenance", expected=expected, actual=actual,
                        reason="crop_before_pending_generation")
            self.metrics.inc("crop_provenance_rejected", camera_id=obs["camera_id"])
            return None

        provenance_bbox = tuple(float(value) for value in header.get("bbox", ()))
        if len(provenance_bbox) != 4:
            self._audit("crop_rejected_provenance", expected=expected, actual=actual,
                        reason="bbox_missing", header=header)
            self.metrics.inc("crop_provenance_rejected", camera_id=obs["camera_id"])
            return None
        source_generation = int(header.get("track_generation", track_state.generation))
        if source_generation != track_state.generation:
            self._audit("crop_rejected_provenance", expected=expected, actual=actual,
                        reason="track_generation_mismatch", source_generation=source_generation,
                        current_generation=track_state.generation)
            self.metrics.inc("crop_provenance_rejected", camera_id=obs["camera_id"])
            return None
        obs["track_generation"] = track_state.generation
        obs["crop_provenance"] = {
            "camera_id": header.get("camera_id"),
            "source_id": header.get("source_id"),
            "pad_index": header.get("pad_index"),
            "native_track_id": header.get("native_track_id"),
            "frame_num": header.get("frame_num"),
            "observation_frame_num": obs.get("frame"),
            "source_frame_width": header.get("source_frame_width"),
            "source_frame_height": header.get("source_frame_height"),
            "pts": header.get("pts"),
            "ntp_timestamp": header.get("ntp_timestamp"),
            "bbox": list(provenance_bbox),
            "track_generation": header.get("track_generation"),
            "canonical_identity": self.public_id(self.manager.root(track_state.canonical_internal_identity))
            if track_state.canonical_internal_identity is not None else None,
        }
        source_width = float(header.get("source_frame_width", W) or W)
        source_height = float(header.get("source_frame_height", H) or H)
        if source_width > 0.0 and source_height > 0.0:
            quality_bbox = (
                provenance_bbox[0] * W / source_width,
                provenance_bbox[1] * H / source_height,
                provenance_bbox[2] * W / source_width,
                provenance_bbox[3] * H / source_height,
            )
        else:
            quality_bbox = provenance_bbox
        useful, quality = crop_quality(quality_bbox)
        quality["source_bbox"] = list(provenance_bbox)
        quality["quality_bbox"] = list(quality_bbox)
        obs["crop_quality"] = quality
        obs["crop_quality_usable"] = useful
        obs["crop_quality_matching_only"] = not useful and matching_only_crop_quality(quality)
        if not useful and not obs["crop_quality_matching_only"]:
            self._audit(
                "crop_rejected_quality", camera_id=obs["camera_id"],
                frame=obs["frame"], native_track_id=obs["native_track_id"],
                crop_frame=actual["frame_num"], bbox=list(provenance_bbox),
                crop_dimensions=list(record.image.shape[:2]), crop_quality=quality,
            )
            self.metrics.inc("osnet_crop_rejected", camera_id=obs["camera_id"])
            self.metrics.inc("crop_rejected_quality", camera_id=obs["camera_id"])
            if track_state is not None:
                track_state.pending_crop_quality_rejected += 1
            return None
        if obs["crop_quality_matching_only"]:
            self.metrics.inc("crop_matching_only", camera_id=obs["camera_id"])
            if quality.get("border_truncated"):
                self.metrics.inc("crop_matching_only_bottom_border", camera_id=obs["camera_id"])
        image = record.image
        if image.ndim != 3 or image.shape[0] < 16 or image.shape[1] < 8:
            self._audit(
                "crop_rejected_dimensions", camera_id=obs["camera_id"],
                frame=obs["frame"], native_track_id=obs["native_track_id"],
                crop_frame=actual["frame_num"], crop_dimensions=list(image.shape),
            )
            self.metrics.inc("crop_failures", camera_id=obs["camera_id"])
            return None
        # Delivery latency is measured once by DeepStreamCropReceiver from
        # encoder completion to worker receipt; avoid a negative duplicate
        # sample here because the request may predate the cached crop.
        self.metrics.observe("crop_age_at_osnet_ms", (monotonic_ns() - record.received_monotonic_ns) / 1_000_000.0, obs["camera_id"])
        self._audit(
            "crop_accepted_for_matching" if obs["crop_quality_matching_only"] else "crop_accepted",
            camera_id=obs["camera_id"], frame=obs["frame"],
            native_track_id=obs["native_track_id"], track_generation=track_state.generation,
            crop_frame=actual["frame_num"], bbox=list(provenance_bbox),
            crop_dimensions=list(image.shape[:2]), crop_quality=quality,
            crop_age_ms=(monotonic_ns() - record.received_monotonic_ns) / 1_000_000.0,
        )
        return image

    def _schedule_reid(self, state: TrackState) -> None:
        if state.latest is None or state.identity_state == "KNOWN":
            return
        latest_frame = int(state.latest["frame"])
        if state.pending_request_frame >= 0:
            return
        if latest_frame - state.pending_last_request_frame < self.pending_resample_frames:
            return
        crop = self._crop_for_observation(state.latest)
        if crop is None:
            return
        state.pending_generation = state.generation
        state.pending_frame = latest_frame
        state.pending_request_frame = latest_frame
        state.pending_last_request_frame = latest_frame
        state.pending_attempts += 1
        state.pending_good_crops += 1
        state.pending_crop_request_frames.append(latest_frame)
        self._audit(
            "osnet_request_submitted", camera_id=state.camera_id, frame=latest_frame,
            native_track_id=state.native_track_id, track_generation=state.generation,
            bbox=state.latest.get("bbox"), crop_quality=state.latest.get("crop_quality"),
            crop_dimensions=list(crop.shape[:2]), pending_attempts=state.pending_attempts,
        )
        self.reid.submit(ObservationEnvelope(dict(state.latest), state.generation, crop=crop, critical=True))
        self.metrics.inc("osnet_requests_submitted", camera_id=state.camera_id)
        self.metrics.inc("embedding_pending", camera_id=state.camera_id)

    def _check_crop_health(self) -> None:
        """Raise an observable health event instead of hiding a crop stall."""
        for state in self.track_states.values():
            if state.terminated or state.identity_state != "PENDING" or state.latest is None:
                continue
            if state.crop_health_timeout_emitted or state.pending_good_crops > 0:
                continue
            if int(state.latest["frame"]) <= state.pending_deadline_frame + self.pending_resample_frames:
                continue
            state.crop_health_timeout_emitted = True
            self.metrics.inc("pending_crop_health_timeout", camera_id=state.camera_id)
            self.metrics.inc("crop_timeout_reason_no_usable_crop", camera_id=state.camera_id)

    def _row(
        self,
        obs: dict,
        internal: int | None,
        diagnostics: dict,
        start_ns: int,
        end_ns: int,
        start_wall_ns: int,
        end_wall_ns: int,
        vector: np.ndarray | None = None,
        reid_result: ReIdResult | None = None,
    ) -> dict:
        application = self.public_id(internal) if internal is not None else "Unknown"
        receive_mono = int(obs["receive_monotonic_ns"])
        row = {
            **obs,
            "internal_identity_at_event": internal,
            "canonical_internal_identity": self.manager.root(internal) if internal is not None else None,
            "global_person_id": application,
            "application_id": application,
            "embedding_extracted": vector is not None,
            "cosine_similarity": diagnostics.get("appearance_similarity"),
            "decision_reason": diagnostics.get("reason", "unknown"),
            "identity_state": diagnostics.get("identity_state", "KNOWN" if internal is not None else "PENDING"),
            "identity_processing_start_timestamp": iso_timestamp(start_wall_ns),
            "identity_processing_start_monotonic_ns": start_ns,
            "identity_processing_end_timestamp": iso_timestamp(end_wall_ns),
            "identity_processing_end_monotonic_ns": end_ns,
            "age_at_processing_ms": max(0.0, (start_ns - receive_mono) / 1_000_000.0),
            "age_at_publication_ms": None,
        }
        if reid_result is not None:
            row["reid"] = {
                "batch_size": reid_result.batch_size,
                "inference_latency_ms": (reid_result.request_end_monotonic_ns - reid_result.request_start_monotonic_ns) / 1_000_000.0,
            }
        self._audit(
            "identity_decision", observation_frame=obs.get("frame"),
            camera_id=obs.get("camera_id"), native_track_id=obs.get("native_track_id"),
            track_generation=obs.get("track_generation"),
            application_id=application, identity_state=row["identity_state"],
            embedding_extracted=row["embedding_extracted"],
            cosine_similarity=row["cosine_similarity"],
            decision_reason=row["decision_reason"],
            decision_evidence=diagnostics, reid=row.get("reid"),
        )
        self.metrics.observe("age_at_processing_ms", row["age_at_processing_ms"], obs["camera_id"])
        self.metrics.observe("identity_decision_latency_ms", (end_ns - start_ns) / 1_000_000.0, obs["camera_id"])
        return row

    def _publish_state(self) -> None:
        started = monotonic_ns()
        people = []
        for camera in CAMS:
            for obs in self.latest_camera_rows[camera]:
                state = self.track_states.get((camera, int(obs["native_track_id"])))
                if state is None or state.terminated:
                    continue
                internal = state.canonical_internal_identity
                people.append({
                    **obs,
                    "internal_identity_at_event": internal,
                    "canonical_internal_identity": self.manager.root(internal) if internal is not None else None,
                    "global_person_id": self.public_id(internal) if internal is not None else "Unknown",
                    "application_id": self.public_id(internal) if internal is not None else "Unknown",
                    "decision_reason": "current_latest_state" if internal is not None else "identity_pending",
                    "identity_state": state.identity_state,
                    "embedding_extracted": False,
                })
        published_mono = monotonic_ns()
        for row in people:
            age = max(0.0, (published_mono - int(row["receive_monotonic_ns"])) / 1_000_000.0)
            row["age_at_publication_ms"] = age
            self.metrics.observe("age_at_publication_ms", age, row["camera_id"])
        state = {
            "profile": "dev-room-cam01-cam04",
            "status": "running",
            "source_mode": self.args.source_mode,
            "session_id": self.args.session_id,
            "active_frame_by_camera": dict(self.latest_camera_frame),
            "published_timestamp": iso_timestamp(wall_timestamp_ns()),
            "people": people,
        }
        current_presence = {
            (str(row.get("camera_id")), int(row.get("native_track_id", -1)),
             str(row.get("application_id", "Unknown")))
            for row in people
        }
        for camera, native_id, app_id in sorted(current_presence - self.last_published_presence):
            self._audit(
                "production_presence_published", camera_id=camera,
                native_track_id=native_id, application_id=app_id,
                bev_marker_eligible=app_id != "Unknown",
                observation=next((row for row in people
                                  if row.get("camera_id") == camera
                                  and int(row.get("native_track_id", -1)) == native_id), None),
            )
        for camera, native_id, app_id in sorted(self.last_published_presence - current_presence):
            self._audit(
                "production_presence_removed", camera_id=camera,
                native_track_id=native_id, application_id=app_id,
                reason="not in current camera observations",
            )
        self.last_published_presence = current_presence
        temp = self.output_dir / "current_state.json.tmp"
        temp.write_text(json.dumps(state, separators=(",", ":")))
        temp.replace(self.output_dir / "current_state.json")
        self.metrics.observe("ui_state_publication_latency_ms", (monotonic_ns() - started) / 1_000_000.0)

    def _supersede_same_camera_fragment(self, obs: dict, internal: int, evidence: dict) -> None:
        if evidence.get("reason") != "same_camera_fragment_handoff":
            return
        canonical = self.manager.root(internal)
        for key, other_state in self.track_states.items():
            if key[0] != obs["camera_id"] or key[1] == int(obs["native_track_id"]):
                continue
            if other_state.terminated or other_state.latest is None:
                continue
            if other_state.canonical_internal_identity is None:
                continue
            if self.manager.root(other_state.canonical_internal_identity) != canonical:
                continue
            other_obs = other_state.latest
            other_distance = distance(obs.get("world"), other_obs.get("world"))
            if (other_distance is None
                    or other_distance > IDENTITY_GATES["cross_camera_world_tight_m"]
                    or bbox_iou(obs.get("bbox"), other_obs.get("bbox")) < 0.20):
                continue
            other_state.terminated = True
            other_state.pending_generation = None
            self.pending_crop_waiting.discard(key)
            self.manager.unbind_track(*key)
            if not other_state.termination_emitted:
                other_state.termination_emitted = True
                self.metrics.inc("native_track_fragments_superseded", camera_id=obs["camera_id"])

    def _apply_sticky(self, envelope: ObservationEnvelope) -> None:
        obs = envelope.observation
        key = (obs["camera_id"], obs["native_track_id"])
        state = self.track_states.get(key)
        if state is None or state.generation != envelope.generation or state.latest is not obs:
            self.metrics.inc("superseded_observations_discarded", camera_id=obs["camera_id"])
            return
        if state.canonical_internal_identity is None:
            self._schedule_reid(state)
            return
        start = monotonic_ns()
        start_wall = wall_timestamp_ns()
        internal, evidence, _ = self.manager.process(obs, None, self._active_assignments(key), allow_reassignment=False)
        end = monotonic_ns()
        end_wall = wall_timestamp_ns()
        state.canonical_internal_identity = self.manager.root(internal)
        state.identity_state = "KNOWN"
        state.last_accepted_frame = int(obs["frame"])
        evidence = {**evidence, "identity_state": "KNOWN"}
        self.output_rows.append(self._row(obs, state.canonical_internal_identity, evidence, start, end, start_wall, end_wall))
        self.metrics.inc("identity_decisions", camera_id=obs["camera_id"])

    def _apply_reid_results(self, results: list[ReIdResult]) -> None:
        valid: list[ReIdResult] = []
        for result in results:
            obs = result.envelope.observation
            key = (obs["camera_id"], obs["native_track_id"])
            state = self.track_states.get(key)
            stale_reason = None
            if state is None:
                stale_reason = "track_removed"
            elif state.generation != result.envelope.generation:
                stale_reason = "track_generation_changed"
            elif state.canonical_internal_identity is not None:
                stale_reason = "canonical_assignment_superseded"
            elif state.identity_state == "KNOWN":
                stale_reason = "canonical_assignment_superseded"
            if stale_reason:
                self.metrics.inc("stale_reid_results_discarded", camera_id=obs["camera_id"])
                self.metrics.inc("embedding_stale", camera_id=obs["camera_id"])
                self.metrics.inc(f"stale_reid_{stale_reason}")
                continue
            # A result older than the latest pending observation remains valid
            # as evidence for this same generation. It cannot mutate a newer
            # KNOWN assignment; it can only resolve the still-PENDING track.
            if state.latest is not None and int(state.latest["frame"]) > int(obs["frame"]):
                self.metrics.inc("stale_reid_pending_evidence", camera_id=obs["camera_id"])
            if state.pending_request_frame == int(obs["frame"]):
                state.pending_request_frame = -1
            # The persistent recent-anchor path requires more than one usable
            # crop. This is evidence metadata, not a visibility/track state.
            obs["pending_good_crops"] = state.pending_good_crops
            valid.append(result)
        if not valid:
            return
        start = monotonic_ns()
        start_wall = wall_timestamp_ns()
        resolved = self.manager.process_batch(
            [{"obs": result.envelope.observation, "vector": result.vector} for result in valid],
            self._active_assignments(),
        )
        end = monotonic_ns()
        end_wall = wall_timestamp_ns()
        for result, (internal, evidence, _latency_us) in zip(valid, resolved):
            obs = result.envelope.observation
            key = (obs["camera_id"], obs["native_track_id"])
            state = self.track_states.get(key)
            if state is None or state.generation != result.envelope.generation:
                self.metrics.inc("stale_reid_results_discarded", camera_id=obs["camera_id"])
                continue
            if self.persist_embeddings:
                with self.embedding_debug_path.open("a") as debug_handle:
                    debug_handle.write(json.dumps({
                        "camera_id": obs.get("camera_id"),
                        "native_track_id": obs.get("native_track_id"),
                        "frame": obs.get("frame"),
                        "timestamp": obs.get("timestamp"),
                        "track_generation": result.envelope.generation,
                        "bbox": obs.get("bbox"),
                        "world": obs.get("world"),
                        "vector": np.asarray(result.vector, dtype=np.float32).tolist(),
                    }, separators=(",", ":")) + "\n")
            state.pending_generation = None
            if internal is None:
                self.metrics.inc("pending_match_attempts", camera_id=obs["camera_id"])
                self.metrics.inc("embedding_received", camera_id=obs["camera_id"])
                state.pending_embedding_completed_frames.append(int(obs["frame"]))
                latest_frame = int(state.latest["frame"]) if state.latest is not None else int(obs["frame"])
                novelty = (
                    {"positive": False, "reason": "matching_only_crop_not_novelty",
                     "candidates": self.manager.candidate_trace(obs, result.vector, self._active_assignments())}
                    if obs.get("crop_quality_matching_only") else
                    self.manager.novelty_evidence(
                        obs, result.vector, self._active_assignments(),
                        state.pending_good_crops, state.pending_attempts,
                    )
                )
                self.decision_trace_rows.append({
                    "observation": {key: obs.get(key) for key in ("camera_id", "native_track_id", "frame", "timestamp", "world", "bbox")},
                    "pending_started_frame": state.pending_started_frame,
                    "pending_age_frames": max(0, latest_frame - state.pending_started_frame),
                    "pending_attempts": state.pending_attempts,
                    "pending_good_crops": state.pending_good_crops,
                    "crop_request_frames": list(state.pending_crop_request_frames),
                    "crop_received_frames": list(state.pending_crop_received_frames),
                    "crop_quality_rejected": state.pending_crop_quality_rejected,
                    "embedding_completed_frames": list(state.pending_embedding_completed_frames),
                    "candidate_trace": novelty.get("candidates", self.manager.candidate_trace(obs, result.vector, self._active_assignments())),
                    "novelty": {key: value for key, value in novelty.items() if key != "candidates"},
                    "allocation_branch": "pending_new_identity" if novelty.get("positive") else "pending_wait_for_evidence",
                })
                self._audit(
                    "identity_candidate_evaluated",
                    decision_trace=self.decision_trace_rows[-1],
                    best_candidate_similarity=evidence.get("appearance_similarity"),
                    evidence=evidence,
                )
                evidence = {**evidence, "identity_state": "PENDING", "novelty_reason": novelty.get("reason")}
                if novelty.get("positive"):
                    self._audit(
                        "canonical_identity_allocation",
                        reason="positive_novelty_evidence", novelty=novelty,
                        observation={key: obs.get(key) for key in
                                     ("camera_id", "native_track_id", "frame", "bbox", "world")},
                    )
                    internal = self.manager.confirm_new(
                        obs, result.vector,
                        reason="positive_novelty_evidence",
                    )
                    state.canonical_internal_identity = self.manager.root(internal)
                    state.identity_state = "NEW_CONFIRMED"
                    state.last_accepted_frame = int(obs["frame"])
                    self.metrics.inc("pending_to_new_confirmed", camera_id=obs["camera_id"])
                    self.metrics.inc("genuinely_new_application_id_allocations", camera_id=obs["camera_id"])
                    self.metrics.inc("positive_novelty_confirmations", camera_id=obs["camera_id"])
                    evidence = {**evidence, "reason": "new_confirmed", "identity_state": "NEW_CONFIRMED"}
                else:
                    self.metrics.inc("pending_novelty_rejected", camera_id=obs["camera_id"])
                    self.metrics.inc("pending_unresolved", camera_id=obs["camera_id"])
                    self._schedule_reid(state)
                    continue
            else:
                self._audit(
                    "canonical_identity_reuse",
                    application_id=self.public_id(self.manager.root(internal)),
                    reason=evidence.get("reason"),
                    observation={key: obs.get(key) for key in
                                 ("camera_id", "native_track_id", "frame", "bbox", "world")},
                    evidence=evidence,
                )
                self.metrics.inc("embedding_received", camera_id=obs["camera_id"])
                state.pending_embedding_completed_frames.append(int(obs["frame"]))
                state.canonical_internal_identity = self.manager.root(internal)
                state.identity_state = "KNOWN"
                state.last_accepted_frame = int(obs["frame"])
                self._supersede_same_camera_fragment(obs, state.canonical_internal_identity, evidence)
                self.metrics.inc("pending_to_existing", camera_id=obs["camera_id"])
                self.metrics.inc("successful_native_reassociations", camera_id=obs["camera_id"])
                resolution_frames = float(max(0, int(obs["frame"]) - state.pending_started_frame))
                self.identity_resolution_frames.append(resolution_frames)
                if state.pending_was_reacquisition:
                    self.reacquisition_frames.append(resolution_frames)
                    self.metrics.inc("id_reacquisition_resolved", camera_id=obs["camera_id"])
                evidence = {**evidence, "identity_state": "KNOWN"}
            state.pending_was_reacquisition = False
            self.output_rows.append(self._row(
                obs, state.canonical_internal_identity, evidence,
                start, end, start_wall, end_wall, result.vector, result,
            ))
            self.metrics.inc("identity_decisions", camera_id=obs["camera_id"])

    def _process_latest_queue(self) -> None:
        for envelope in self.observation_queue.get_batch(128):
            self.metrics.age_ms(int(envelope.observation["receive_monotonic_ns"]))
            state = self.track_states.get((envelope.observation["camera_id"], envelope.observation["native_track_id"]))
            if state is None:
                continue
            if envelope.critical or state.canonical_internal_identity is None:
                # Resolve geometry/time/native continuity from the newest state
                # even when crop extraction is unavailable. Any in-flight ReID
                # result for the superseded pending generation is discarded by
                # the existing stale-result checks.
                if state.canonical_internal_identity is None and state.latest is not None:
                    latest = state.latest
                    start = monotonic_ns()
                    start_wall = wall_timestamp_ns()
                    internal, evidence = self.manager.resolve_existing(
                        latest, None, self._active_assignments((latest["camera_id"], latest["native_track_id"])),
                    )
                    end = monotonic_ns()
                    end_wall = wall_timestamp_ns()
                    if internal is not None:
                        self.manager.accept_existing(internal, latest, None, evidence)
                        state.pending_generation = None
                        state.pending_request_frame = -1
                        state.canonical_internal_identity = self.manager.root(internal)
                        state.identity_state = "KNOWN"
                        state.last_accepted_frame = int(latest["frame"])
                        resolution_frames = float(max(0, int(latest["frame"]) - state.pending_started_frame))
                        self.identity_resolution_frames.append(resolution_frames)
                        self.metrics.inc("pending_to_existing", camera_id=latest["camera_id"])
                        self.metrics.inc("successful_native_reassociations", camera_id=latest["camera_id"])
                        if state.pending_was_reacquisition:
                            self.reacquisition_frames.append(resolution_frames)
                            self.metrics.inc("id_reacquisition_resolved", camera_id=latest["camera_id"])
                        state.pending_was_reacquisition = False
                        evidence = {**evidence, "identity_state": "KNOWN", "resolution_path": "latest_geometry_or_native"}
                        self.output_rows.append(self._row(
                            latest, state.canonical_internal_identity, evidence,
                            start, end, start_wall, end_wall,
                        ))
                        self.metrics.inc("identity_decisions", camera_id=latest["camera_id"])
                        continue
                self._schedule_reid(state)
            else:
                self._apply_sticky(envelope)
        # A crop can arrive on the metadata socket just after the latest
        # Kafka observation was processed. Retry pending tracks so the bounded
        # cache is consumed without reopening a camera or extending blind
        # identity processing of old frames.
        for state in self.track_states.values():
            if state.terminated or state.identity_state != "PENDING" or state.latest is None:
                continue
            if state.pending_request_frame < 0:
                self._schedule_reid(state)
        self.metrics.observe("global_identity_manager_queue_depth", 0.0)
        self._check_crop_health()

    def _parse_new_kafka(self, path: Path) -> int:
        if not path.exists():
            return 0
        with path.open() as handle:
            handle.seek(self.kafka_offset)
            lines = handle.readlines()
            self.kafka_offset = handle.tell()
        self.metrics.observe("kafka_input_queue_depth", len(lines))
        self.metrics.set_max("kafka_input_queue_depth", len(lines))
        for line in lines:
            parse_started = monotonic_ns()
            receive_wall = wall_timestamp_ns()
            try:
                event = json.loads(line)
            except ValueError:
                self.metrics.inc("kafka_parse_errors")
                continue
            self.metrics.observe("observation_parsing_latency_ms", (monotonic_ns() - parse_started) / 1_000_000.0)
            frame = event.get("frame")
            if not frame or frame.get("sensorId") not in CAMS:
                continue
            camera = str(frame["sensorId"])
            self._ingest_frame(camera, int(frame["id"]), frame, parse_started, receive_wall)
            self.metrics.inc("kafka_frames_received", camera_id=camera)
        return len(lines)

    def report(self) -> dict:
        elapsed = max(time.perf_counter() - self.started, 1e-6)
        metrics = self.metrics.snapshot()
        embedder_metrics = self.embedder.metrics()
        return {
            "mode": "production_latest_state_cuda_osnet",
            "source_mode": self.args.source_mode,
            "session_id": self.args.session_id,
            "face_recognition_used": False,
            "future_frames_used": False,
            "hard_coded_person_count": False,
            "elapsed_seconds": elapsed,
            # These are person-observation rates, not decoded source-frame
            # rates. DeepStream source FPS is measured independently by the
            # per-source health/PERF probe. Keeping the semantics explicit
            # prevents sparse no-object frames from being misreported as an
            # identity-worker backlog.
            "identity_observation_fps_per_camera": {
                camera: self.received_frames[camera] / elapsed for camera in CAMS
            },
            "identity_decision_fps_per_camera": {
                camera: metrics["by_camera"].get(camera, {}).get("identity_decisions", 0) / elapsed
                for camera in CAMS
            },
            "output_observations": len(self.output_rows),
            "identity_gallery": self.gallery_store.stats(),
            "identity_persistence": dict(self.manager.persistence_stats),
            "embeddings_extracted": int(embedder_metrics.get("images", 0)),
            "embeddings_by_camera": {camera: metrics["by_camera"].get(camera, {}).get("osnet_requests_submitted", 0) for camera in CAMS},
            "internal_identities_created": self.manager.next_number - 1,
            "application_global_ids_active": len({self.manager.root(key) for key in self.public_ids}),
            "public_id_map": {str(key): value for key, value in self.public_ids.items()},
            "osnet": self.reid.runtime_metrics(),
            "identity_metrics": metrics,
            "identity_manager": {
                "latency_us": {
                    "mean": float(np.mean(self.manager.latencies_us)) if self.manager.latencies_us else None,
                    "p95": float(np.quantile(self.manager.latencies_us, 0.95)) if self.manager.latencies_us else None,
                    "max": max(self.manager.latencies_us) if self.manager.latencies_us else None,
                },
                "reassignment_attempts": self.manager.reassignment_attempts,
                "reassignment_rejected": self.manager.reassignment_rejected,
                "reassignment_accepted": self.manager.reassignment_accepted,
                "conflict_guard_events": self.manager.conflict_guard_events,
            },
            "id_reacquisition_latency_frames": {
                "count": len(self.reacquisition_frames),
                "median": float(np.median(self.reacquisition_frames)) if self.reacquisition_frames else None,
                "p95": float(np.quantile(self.reacquisition_frames, 0.95)) if self.reacquisition_frames else None,
                "max": max(self.reacquisition_frames) if self.reacquisition_frames else None,
            },
            "identity_lifecycle": {
                "pending_window_frames": self.pending_window_frames,
                "pending_window_ms": self.pending_window_frames * 1000.0 / IDENTITY_GATES["source_fps"],
                "pending_resample_frames": self.pending_resample_frames,
                "pending_min_good_crops": self.pending_min_good_crops,
                "pending_created": metrics["counters"].get("pending_created", 0),
                "pending_to_existing": metrics["counters"].get("pending_to_existing", 0),
                "pending_to_new_confirmed": metrics["counters"].get("pending_to_new_confirmed", 0),
                "pending_unresolved": metrics["counters"].get("pending_unresolved", 0),
                "pending_novelty_rejected": metrics["counters"].get("pending_novelty_rejected", 0),
                "positive_novelty_confirmations": metrics["counters"].get("positive_novelty_confirmations", 0),
                "identity_resolution_frames": {
                    "count": len(self.identity_resolution_frames),
                    "median": float(np.median(self.identity_resolution_frames)) if self.identity_resolution_frames else None,
                    "p95": float(np.quantile(self.identity_resolution_frames, 0.95)) if self.identity_resolution_frames else None,
                    "max": max(self.identity_resolution_frames) if self.identity_resolution_frames else None,
                },
            },
            "cross_camera_matching": {
                "gates": IDENTITY_GATES,
                "attempted": self.manager.attempted_cross_camera_matches,
                "accepted": self.manager.accepted_cross_camera_matches,
                "rejected": self.manager.rejected_cross_camera_matches,
                "world_distances_m": self.manager.cross_camera_world_distances,
                "time_differences_ms": self.manager.cross_camera_time_differences_ms,
                "similarities": self.manager.cross_camera_similarities,
                "persistent_reacquisition_similarities": self.manager.persistent_reacquisition_similarities,
            },
            "crop_delivery": {
                "socket": self.crop_receiver.metrics_snapshot(),
                "pending_tracks_waiting_for_crop_current": len(
                    {(camera, native) for camera, native in self.pending_crop_waiting
                     if not self.track_states.get((camera, native), TrackState(camera, native)).terminated}
                ),
                "crop_requests": metrics["counters"].get("crop_requests", 0),
                "crop_success": metrics["counters"].get("crop_success", 0),
                "crop_failures": metrics["counters"].get("crop_failures", 0),
                "by_camera": {
                    camera: {
                        key: value for key, value in metrics["by_camera"].get(camera, {}).items()
                        if key.startswith("crop_")
                    }
                    for camera in CAMS
                },
                "by_reason": {
                    key.removeprefix("crop_reason_"): value
                    for key, value in metrics["counters"].items()
                    if key.startswith("crop_reason_")
                },
            },
        }

    def write(self) -> None:
        with (self.output_dir / "global_identity.jsonl").open("w") as handle:
            for row in self.output_rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        (self.output_dir / "identity_events.json").write_text(json.dumps(self.manager.events, indent=2))
        with (self.output_dir / "identity_decision_trace.jsonl").open("w") as handle:
            for row in self.decision_trace_rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        (self.output_dir / "reacquisition.json").write_text(json.dumps({
            "count": len(self.reacquisition_frames),
            "frames": self.reacquisition_frames,
            "median": float(np.median(self.reacquisition_frames)) if self.reacquisition_frames else None,
            "p95": float(np.quantile(self.reacquisition_frames, 0.95)) if self.reacquisition_frames else None,
            "max": max(self.reacquisition_frames) if self.reacquisition_frames else None,
        }, indent=2))
        (self.output_dir / "runtime_report.json").write_text(json.dumps(self.report(), indent=2))

    def run(self) -> None:
        previous_term = signal.signal(signal.SIGTERM, self._request_stop)
        previous_int = signal.signal(signal.SIGINT, self._request_stop)
        self.crop_receiver.start()
        self.reid.start()
        started_live = time.monotonic()
        done_idle_since: float | None = None
        try:
            while not self.stop_requested and time.monotonic() - started_live < self.args.duration:
                lines = self._parse_new_kafka(self.args.kafka)
                self._process_latest_queue()
                self._apply_reid_results(self.reid.drain_results(None, limit=64))
                self._process_latest_queue()
                now = time.monotonic()
                if now - self.last_publish >= 0.05:
                    self._publish_state()
                    self.last_publish = now
                self.metrics.set_max("identity_latest_queue_depth", self.observation_queue.depth())
                self.metrics.set_max("osnet_request_queue_depth", self.reid.requests.depth())
                if now - self.last_report >= 2.0:
                    self.write()
                    self.last_report = now
                if self.args.done_file is not None and self.args.done_file.exists():
                    queues_idle = self.observation_queue.depth() == 0 and self.reid.requests.depth() == 0
                    if lines or not queues_idle:
                        done_idle_since = None
                    elif done_idle_since is None:
                        done_idle_since = now
                    elif now - done_idle_since >= 2.0:
                        break
                if not lines:
                    time.sleep(0.005)
        finally:
            self.reid.stop()
            self._apply_reid_results(self.reid.drain_results(None, limit=256))
            self.crop_receiver.stop()
            self._publish_state()
            self.write()
            self.gallery_store.close()
            self._audit("identity_worker_shutdown", requested_stop=self.stop_requested)
            self.audit_handle.close()
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)

    def _request_stop(self, _signum, _frame) -> None:
        # Convert process termination into the ordinary queue-drain/flush path;
        # no identity decisions or timings are changed.
        self.stop_requested = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kafka", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--osnet-model", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=180.0, help="hard safety limit; done-file normally ends the session")
    parser.add_argument("--done-file", type=Path)
    parser.add_argument("--source-mode", choices=("replay", "live"), default=os.getenv("MV3DT_DEV_ROOM_SOURCE_MODE", "live"))
    parser.add_argument("--session-id", default=os.getenv("MV3DT_DEV_ROOM_SESSION_ID", "unspecified"))
    parser.add_argument("--interval", type=int, default=10, help="retained for CLI compatibility; stable tracks are not sampled")
    parser.add_argument("--termination-gap", type=int, default=10)
    parser.add_argument("--crop-socket", type=Path, required=True)
    parser.add_argument("--gallery-db", type=Path, help="durable SQLite canonical gallery")
    args = parser.parse_args()
    LiveIdentityWorker(args).run()


if __name__ == "__main__":
    main()
