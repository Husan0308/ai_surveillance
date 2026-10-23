"""Bounded, nonpersistent OSNet association sidecar for staging live replay.

This component consumes only exact-frame NvDCF observations and corresponding
1920x1080 inference rasters. It never writes tracker records or a production
identity store. Queue pressure drops Re-ID samples only; detection/tracking
continue on their existing path.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
import copy
import hashlib
import itertools
import json
import math
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from services.mv3dt_staging.automatic_provisional_global_id import (
    CAMERAS,
    SOURCE_IDS,
    score_and_assign_provisional_ids,
)
from services.mv3dt_staging.provisional_global_identity_memory import (
    ProvisionalGlobalIdentityMemory,
)


INFERENCE_WIDTH = 1920
INFERENCE_HEIGHT = 1080
DEFAULT_SAMPLE_EVERY_FRAMES = 20
DEFAULT_MAX_CROPS_PER_TRACK = 3
DEFAULT_SAMPLE_WINDOW_FRAMES = 120
DEFAULT_QUEUE_SIZE = 32
DEFAULT_CONTROL_HISTORY_FRAMES = 300
STALE_EMBEDDING_AGE_SECONDS = 2.0  # telemetry marker only; does not change matching gates
MAX_EMBEDDING_QUEUE_WAIT_SECONDS = 1.0
REAPPEAR_SOURCE_GAP_NS = 1_000_000_000
DEFAULT_CONTROL_DIAGNOSTICS = (
    Path(__file__).resolve().parents[2]
    / ".runtime/entrance-automatic-provisional-global-id-20261022-r2"
    / "automatic_global_id_diagnostics.json"
)
EXPECTED_CONTROL_DIAGNOSTICS_SHA256 = (
    "5063826ededfe0b946a7c4837af2d484aef4573926d7ce230316f93a2830bcfa"
)


def load_preserved_control_reference(
    diagnostics_path: Path = DEFAULT_CONTROL_DIAGNOSTICS,
) -> tuple[float, str]:
    """Load only the hash-pinned same-camera control envelope, never labels."""
    diagnostics_path = Path(diagnostics_path).resolve()
    digest = hashlib.sha256(diagnostics_path.read_bytes()).hexdigest()
    if digest != EXPECTED_CONTROL_DIAGNOSTICS_SHA256:
        raise ValueError("preserved OSNet control-reference hash mismatch")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    summary = diagnostics.get("same_camera_control_summary", {})
    if (
        int(summary.get("count", -1)) != 35
        or summary.get("semantics")
        != "automatically mined controls, not independently verified DIFFERENT_PEOPLE labels"
        or diagnostics.get("rule_contract", {}).get("saved_human_review_decisions_read") is not False
    ):
        raise ValueError("saved control reference is not the expected unlabeled OSNet control set")
    maximum = float(summary["maximum_similarity_median"])
    if not math.isfinite(maximum) or not -1.0 <= maximum <= 1.0:
        raise ValueError("saved same-camera control envelope is invalid")
    return maximum, f"sha256:{digest}; 35 same-camera separation controls; not ground-truth negatives"


class ShadowJoinError(ValueError):
    """A crop was not tied to a valid, exact camera/frame/PTS observation."""


def crop_quality_xywh(box: list[float] | tuple[float, ...]) -> tuple[bool, dict[str, Any]]:
    """Mirror the established Entrance OSNet crop policy in detector space.

    The values match services/mv3dt_room/extract_osnet.py: full-body crops at
    least 28x96 px, >=0.3% frame area, aspect 0.16..0.95, >=97% in-frame, and
    not clipped by the image border.
    """
    if len(box) != 4:
        return False, {"reason": "bbox_shape"}
    left, top, width, height = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (left, top, width, height)):
        return False, {"reason": "nonfinite_bbox"}
    right, bottom = left + width, top + height
    clipped_left, clipped_top = max(0.0, left), max(0.0, top)
    clipped_right = min(float(INFERENCE_WIDTH), right)
    clipped_bottom = min(float(INFERENCE_HEIGHT), bottom)
    visible_area = max(0.0, clipped_right - clipped_left) * max(0.0, clipped_bottom - clipped_top)
    area = max(width * height, 1.0)
    inside = visible_area / area
    aspect = width / max(height, 1.0)
    border_truncated = (
        left <= 2.0 or top <= 2.0
        or right >= INFERENCE_WIDTH - 2.0
        or bottom >= INFERENCE_HEIGHT - 2.0
    )
    useful = (
        width >= 28.0 and height >= 96.0
        and area >= INFERENCE_WIDTH * INFERENCE_HEIGHT * 0.003
        and 0.16 <= aspect <= 0.95 and inside >= 0.97
        and not border_truncated
    )
    return useful, {
        "width": width, "height": height,
        "area_fraction": area / (INFERENCE_WIDTH * INFERENCE_HEIGHT),
        "inside_fraction": inside, "aspect": aspect,
        "border_truncated": border_truncated,
    }


def measure_crop_quality(
    crop: np.ndarray,
    box: list[float] | tuple[float, ...],
    other_boxes: list[list[float]] | tuple[list[float], ...] = (),
    *,
    detector_confidence: float | None = None,
    tracker_confidence: float | None = None,
) -> dict[str, Any]:
    """Measure crop quality without turning a visual proxy into identity truth.

    The overlap score is the largest other-person box intersection divided by
    this crop's box area. It is only a clutter/possible-occlusion proxy; it is
    not a segmentation or occlusion label. Confidence values affect selection
    order only, never the cross-camera matching gates.
    """
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0 or crop.ndim != 3:
        raise ValueError("crop quality requires a non-empty BGR image")
    x, y, width, height = (float(value) for value in box)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    target_area = max(width * height, 1.0)
    overlaps = []
    for other in other_boxes:
        ox, oy, ow, oh = (float(value) for value in other)
        intersection = max(0.0, min(x + width, ox + ow) - max(x, ox)) * max(
            0.0, min(y + height, oy + oh) - max(y, oy)
        )
        overlaps.append(intersection / target_area)
    gray_std = float(gray.std())
    height_px, width_px = crop.shape[:2]
    return {
        "crop_width_px": int(width_px), "crop_height_px": int(height_px),
        "crop_area_px": int(width_px * height_px),
        "laplacian_variance_blur_indicator": laplacian_variance,
        "grayscale_stddev": gray_std,
        "max_other_person_box_overlap_fraction": max(overlaps, default=0.0),
        "detector_confidence": detector_confidence,
        "tracker_confidence": tracker_confidence,
        "overlap_is_occlusion_ground_truth": False,
        "relative_quality_score": crop_quality_score({
            "area_fraction": (width_px * height_px) / (INFERENCE_WIDTH * INFERENCE_HEIGHT),
            "laplacian_variance_blur_indicator": laplacian_variance,
            "grayscale_stddev": gray_std,
            "max_other_person_box_overlap_fraction": max(overlaps, default=0.0),
            "detector_confidence": detector_confidence,
            "tracker_confidence": tracker_confidence,
        }),
    }


def crop_quality_score(quality: dict[str, Any]) -> float:
    """Bounded relative ranking score for samples of the same track/window."""
    def normalized(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return min(1.0, max(0.0, result)) if math.isfinite(result) else default

    area = normalized(float(quality.get("area_fraction", 0.0)) / 0.02)
    sharp = normalized(math.log1p(max(0.0, float(quality.get(
        "laplacian_variance_blur_indicator", 0.0)))) / math.log1p(1500.0))
    contrast = normalized(float(quality.get("grayscale_stddev", 0.0)) / 60.0)
    detector = normalized(quality.get("detector_confidence"))
    tracker = normalized(quality.get("tracker_confidence"))
    overlap = normalized(quality.get("max_other_person_box_overlap_fraction"))
    return float(0.30 * area + 0.25 * sharp + 0.15 * contrast
                 + 0.18 * detector + 0.12 * tracker - 0.25 * overlap)


def crop_quality_rejection(quality: dict[str, Any]) -> str | None:
    """Reject only very low-information or extremely overlapping samples.

    These conservative cutoffs are engineering safeguards, not learned
    identity thresholds. Moderate blur and overlap remain rankable candidates.
    """
    if (float(quality.get("laplacian_variance_blur_indicator", 0.0)) < 25.0
            and float(quality.get("grayscale_stddev", 0.0)) < 6.0):
        return "severely_blurred_or_low_information"
    if float(quality.get("max_other_person_box_overlap_fraction", 0.0)) >= 0.80:
        return "extreme_person_box_overlap_proxy"
    return None


def _percentiles(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": max(values),
    }


def _box_iou_xywh(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection = max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * max(
        0.0, min(ly + lh, ry + rh) - max(ly, ry)
    )
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


class ProvisionalReIdShadow:
    """A bounded worker that annotates local tracks with provisional candidates.

    embedder must implement embed_batch(list[np.ndarray]) and return finite,
    L2-normalized 512-D OSNet-AIN features. Assignments exist only in this
    object's memory for the current process. An update that ceases to pass the
    conservative gates withdraws a candidate and records an in-memory audit
    event; it never rewrites the underlying local NvDCF ID.
    """

    def __init__(
        self,
        embedder,
        *,
        control_reference_max: float,
        control_reference_source: str,
        person_class_id: int,
        sample_every_frames: int = DEFAULT_SAMPLE_EVERY_FRAMES,
        max_crops_per_track: int = DEFAULT_MAX_CROPS_PER_TRACK,
        sample_window_frames: int = DEFAULT_SAMPLE_WINDOW_FRAMES,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        max_gallery_tracks: int = 128,
        gallery_ttl_seconds: float = 600.0,
        clock_ns: Callable[[], int] | None = None,
        control_history_frames: int = DEFAULT_CONTROL_HISTORY_FRAMES,
        batch_size: int = 8,
        decision_diagnostics: bool = False,
        decision_trace_limit: int = 4096,
    ):
        if not -1.0 <= float(control_reference_max) <= 1.0:
            raise ValueError("control-reference maximum must be a cosine in [-1, 1]")
        if not control_reference_source.strip():
            raise ValueError("control-reference provenance is required")
        if sample_every_frames < 1 or max_crops_per_track < 2 or sample_window_frames < sample_every_frames:
            raise ValueError("sampling cadence/window must be positive and galleries must permit >=2 crops")
        if (queue_size < 1 or max_gallery_tracks < 2 or control_history_frames < 8
                or batch_size < 1 or decision_trace_limit < 1
                or not math.isfinite(gallery_ttl_seconds)
                or gallery_ttl_seconds <= 0):
            raise ValueError("invalid shadow-worker bounds")

        self.embedder = embedder
        self.control_reference_max = float(control_reference_max)
        self.control_reference_source = control_reference_source
        self.person_class_id = int(person_class_id)
        self.sample_every_frames = int(sample_every_frames)
        self.max_crops_per_track = int(max_crops_per_track)
        self.sample_window_frames = int(sample_window_frames)
        self.max_gallery_tracks = int(max_gallery_tracks)
        self.gallery_ttl_seconds = float(gallery_ttl_seconds)
        self.gallery_ttl_ns = int(round(self.gallery_ttl_seconds * 1_000_000_000))
        self._clock_ns = clock_ns or time.monotonic_ns
        self.control_history_frames = int(control_history_frames)
        self.batch_size = int(batch_size)
        self.decision_diagnostics = bool(decision_diagnostics)
        self.decision_trace_limit = int(decision_trace_limit)
        self._queue: queue.PriorityQueue[tuple[int, int, dict[str, Any] | None]] = queue.PriorityQueue(
            maxsize=int(queue_size)
        )
        self._queue_sequence = itertools.count()
        self._max_queue_depth = 0
        self._lock = threading.RLock()
        self._frame_history: dict[str, OrderedDict[int, list[dict[str, Any]]]] = {
            camera: OrderedDict() for camera in CAMERAS
        }
        self._duplicate_overlap_counts: dict[tuple[str, str, str], int] = defaultdict(int)
        self._duplicate_track_conflicts: set[tuple[str, str, str]] = set()
        self._duplicate_rejected_candidates: set[tuple[str, str, str]] = set()
        self._last_frame: dict[str, tuple[int, int]] = {}
        self._last_sample_frame: dict[tuple[str, str], int] = {}
        self._gallery: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._gallery_last_seen_ns: dict[tuple[str, str], int] = {}
        self._last_observed_source_pts_ns: dict[tuple[str, str], int] = {}
        self._gallery_generation: dict[tuple[str, str], int] = defaultdict(int)
        self._pending_window: dict[tuple[str, str], dict[str, Any]] = {}
        # Track pending jobs by gallery generation so work started before TTL
        # expiry cannot later resurrect a track after a reappearance.
        self._scheduled_track_keys: dict[tuple[str, str], int] = {}
        self._records: list[dict[str, Any]] = []
        self._vectors: list[np.ndarray] = []
        self._active: dict[tuple[str, str], dict[str, Any]] = {}
        self._next_id = 1
        self._identity_memory = ProvisionalGlobalIdentityMemory(
            ttl_seconds=self.gallery_ttl_seconds,
            max_identities=self.max_gallery_tracks,
        )
        self._identity_event_cursor = 0
        self._revision_events: list[dict[str, Any]] = []
        self._decision_events: list[dict[str, Any]] = []
        self._decision_event_sequence = 0
        self._decision_events_dropped = 0
        self._score_result: dict[str, Any] = {
            "status": "WAITING_FOR_FEATURES", "matches": [], "candidate_pairs": []
        }
        self._queue_wait_ms: list[float] = []
        self._crop_selection_to_enqueue_ms: list[float] = []
        self._sampling_decision_ms: list[float] = []
        self._crop_extraction_ms: list[float] = []
        self._crop_copy_ms: list[float] = []
        self._crop_quality_ms: list[float] = []
        self._inference_batch_ms: list[float] = []
        self._sample_to_embedding_ms: list[float] = []
        self._gallery_update_ms: list[float] = []
        self._score_ms: list[float] = []
        self._counters = {
            "frames_joined": 0,
            "frames_rejected": 0,
            "tracker_observations_seen": 0,
            "quality_eligible_crops": 0,
            "quality_rejected_crops": 0,
            "quality_rejected_blur_or_low_information": 0,
            "quality_rejected_overlap_proxy": 0,
            "quality_ranked_window_replacements": 0,
            "window_samples_flushed": 0,
            "crop_requests_scheduled": 0,
            "crop_jobs_dequeued": 0,
            "crop_jobs_expired_before_inference": 0,
            "shutdown_queue_drops": 0,
            "initial_track_jobs_scheduled": 0,
            "insufficient_gallery_jobs_scheduled": 0,
            "reappeared_track_jobs_scheduled": 0,
            "new_viewpoint_jobs_scheduled": 0,
            "request_deduplicated_while_track_job_pending": 0,
            "rolling_gallery_evictions": 0,
            "gallery_ttl_expired_tracks": 0,
            "gallery_ttl_expired_crops": 0,
            "sample_interval_skips": 0,
            "queue_full_drops": 0,
            "stale_queue_jobs_expired": 0,
            "stale_embedding_results_rejected": 0,
            "gallery_track_cap_drops": 0,
            "features_generated": 0,
            "embedding_jobs_completed": 0,
            "embedding_jobs_over_stale_age_threshold": 0,
            "embedding_errors": 0,
            "candidate_withdrawals": 0,
            "one_to_many_conflicts_rejected": 0,
            "duplicate_track_pairs_rejected": 0,
            "duplicate_pair_cap_drops": 0,
        }
        self._last_error = ""
        self._closed = False
        self._last_join_error = ""
        self._closed_snapshot: dict[str, Any] | None = None
        self._worker = threading.Thread(target=self._run, name="osnet-provisional-shadow", daemon=True)
        self._worker.start()

    def submit_frame(
        self,
        *,
        source_id: int,
        camera_id: str,
        frame_index: int,
        source_pts_ns: int,
        cached_source_id: int,
        cached_frame_index: int,
        cached_source_pts_ns: int,
        inference_frame: np.ndarray,
        objects: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Queue quality-approved crops only when tracker and frame keys agree."""
        decision_started_ns = time.perf_counter_ns()
        if self._closed:
            raise RuntimeError("OSNet shadow worker is already closed")
        if camera_id not in CAMERAS or SOURCE_IDS.get(int(source_id)) != camera_id:
            raise ShadowJoinError("source ID/camera mapping mismatch")
        if (
            int(source_id) != int(cached_source_id)
            or int(frame_index) != int(cached_frame_index)
            or int(source_pts_ns) < 0
            or int(source_pts_ns) != int(cached_source_pts_ns)
        ):
            raise ShadowJoinError("source ID, frame index, or source PTS did not match the cached inference frame")
        if (
            not isinstance(inference_frame, np.ndarray)
            or inference_frame.shape != (INFERENCE_HEIGHT, INFERENCE_WIDTH, 3)
            or inference_frame.dtype != np.uint8
        ):
            raise ShadowJoinError("shadow crop input must be a uint8 BGR 1920x1080 inference frame")

        frame_index = int(frame_index)
        source_pts_ns = int(source_pts_ns)
        queued_this_frame = 0
        with self._lock:
            now_ns = self._clock_ns()
            self._expire_galleries_locked(now_ns)
            last = self._last_frame.get(camera_id)
            if last is not None and (frame_index <= last[0] or source_pts_ns <= last[1]):
                self._counters["frames_rejected"] += 1
                raise ShadowJoinError("per-camera frame index and PTS must advance monotonically")
            self._last_frame[camera_id] = (frame_index, source_pts_ns)
            queued_this_frame += self._flush_old_windows(
                camera_id, frame_index // self.sample_window_frames
            )
            frame_rows: list[dict[str, Any]] = []
            for obj in objects:
                if int(obj.get("class_id", self.person_class_id)) != self.person_class_id:
                    continue
                tracker_id = obj.get("tracker_id")
                if tracker_id is None:
                    continue
                local_id = str(tracker_id)
                box = [
                    float(obj.get("left", obj.get("x", 0.0))),
                    float(obj.get("top", obj.get("y", 0.0))),
                    float(obj.get("width", 0.0)),
                    float(obj.get("height", 0.0)),
                ]
                if not all(math.isfinite(value) for value in box) or box[2] <= 0 or box[3] <= 0:
                    continue
                try:
                    detector_confidence = float(obj.get("confidence", obj.get("detector_confidence", 0.0)))
                except (TypeError, ValueError):
                    detector_confidence = 0.0
                try:
                    tracker_confidence = float(obj.get("tracker_confidence", 0.0))
                except (TypeError, ValueError):
                    tracker_confidence = 0.0
                frame_rows.append({"object_id": local_id, "box": box,
                                   "detector_confidence": detector_confidence,
                                   "tracker_confidence": tracker_confidence})
                self._counters["tracker_observations_seen"] += 1
            # Identity activity/TTL uses wall-monotonic observation arrival,
            # never cross-camera PTS equality or map proximity.
            self._identity_memory.touch_observations(
                camera_id, [str(row["object_id"]) for row in frame_rows], now_ns
            )
            for index, left in enumerate(frame_rows):
                for right in frame_rows[index + 1:]:
                    if _box_iou_xywh(left["box"], right["box"]) < 0.5:
                        continue
                    a, b = sorted((str(left["object_id"]), str(right["object_id"])))
                    key = (camera_id, a, b)
                    if key not in self._duplicate_overlap_counts and len(self._duplicate_overlap_counts) >= 8192:
                        self._counters["duplicate_pair_cap_drops"] += 1
                        continue
                    self._duplicate_overlap_counts[key] += 1
                    if self._duplicate_overlap_counts[key] >= 2:
                        self._duplicate_track_conflicts.add(key)

            # Initial/under-covered tracks are embedded as soon as an eligible
            # owned crop is available. Tracks with adequate galleries retain
            # the existing best-quality-per-window selection for new views.
            boxes_by_track = {row["object_id"]: row["box"] for row in frame_rows}
            for row in frame_rows:
                local_id = row["object_id"]
                track_key = (camera_id, local_id)
                last_observed_pts_ns = self._last_observed_source_pts_ns.get(track_key)
                known_track = (
                    track_key in self._gallery
                    or track_key in self._pending_window
                    or track_key in self._scheduled_track_keys
                )
                reappeared = (
                    known_track and last_observed_pts_ns is not None
                    and source_pts_ns - last_observed_pts_ns >= REAPPEAR_SOURCE_GAP_NS
                )
                self._last_observed_source_pts_ns[track_key] = source_pts_ns
                if known_track:
                    self._gallery_last_seen_ns[track_key] = now_ns
                if track_key not in self._gallery and track_key not in self._pending_window \
                        and track_key not in self._scheduled_track_keys \
                        and len(self._gallery) >= self.max_gallery_tracks:
                    self._counters["gallery_track_cap_drops"] += 1
                    continue
                if frame_index - self._last_sample_frame.get(track_key, -self.sample_every_frames) < self.sample_every_frames:
                    self._counters["sample_interval_skips"] += 1
                    continue
                self._last_sample_frame[track_key] = frame_index
                if track_key in self._scheduled_track_keys:
                    self._counters["request_deduplicated_while_track_job_pending"] += 1
                    continue
                box = row["box"]
                useful, geometry = crop_quality_xywh(box)
                if not useful:
                    self._counters["quality_rejected_crops"] += 1
                    continue
                x, y, width, height = box
                crop_extract_started_ns = time.perf_counter_ns()
                crop = inference_frame[max(0, int(y)):min(INFERENCE_HEIGHT, int(math.ceil(y + height))),
                                       max(0, int(x)):min(INFERENCE_WIDTH, int(math.ceil(x + width)))]
                self._crop_extraction_ms.append(
                    (time.perf_counter_ns() - crop_extract_started_ns) / 1e6
                )
                if crop.size == 0:
                    self._counters["quality_rejected_crops"] += 1
                    continue
                # This slice is a view over the cached inference raster. Copy
                # only the selected person crop while that exact raster is
                # alive; the asynchronous worker never retains a full frame.
                crop_copy_started_ns = time.perf_counter_ns()
                owned_crop = crop.copy(order="C")
                self._crop_copy_ms.append(
                    (time.perf_counter_ns() - crop_copy_started_ns) / 1e6
                )
                crop_quality_started_ns = time.perf_counter_ns()
                quality = measure_crop_quality(
                    owned_crop, box,
                    [candidate for other_id, candidate in boxes_by_track.items() if other_id != local_id],
                    detector_confidence=row["detector_confidence"],
                    tracker_confidence=row["tracker_confidence"],
                )
                self._crop_quality_ms.append(
                    (time.perf_counter_ns() - crop_quality_started_ns) / 1e6
                )
                quality.update(geometry)
                rejection = crop_quality_rejection(quality)
                if rejection:
                    self._counters["quality_rejected_crops"] += 1
                    if rejection == "severely_blurred_or_low_information":
                        self._counters["quality_rejected_blur_or_low_information"] += 1
                    else:
                        self._counters["quality_rejected_overlap_proxy"] += 1
                    continue
                self._counters["quality_eligible_crops"] += 1
                job = {
                    "camera_id": camera_id, "source_id": int(source_id),
                    "local_track_id": local_id, "frame": frame_index,
                    "source_pts_ns": source_pts_ns, "bbox_inference_xywh": box,
                    "crop": owned_crop, "crop_quality": quality,
                    "window_index": frame_index // self.sample_window_frames,
                    "selected_perf_ns": time.perf_counter_ns(),
                    "observed_monotonic_ns": now_ns,
                    "track_generation": self._gallery_generation[track_key],
                }
                self._gallery_last_seen_ns[track_key] = now_ns
                gallery_size = len(self._gallery.get(track_key, ()))
                if reappeared:
                    self._pending_window.pop(track_key, None)
                    if self._enqueue_job(job, priority=0, reason="RECENT_REAPPEARANCE"):
                        self._counters["reappeared_track_jobs_scheduled"] += 1
                        queued_this_frame += 1
                    continue
                if gallery_size == 0:
                    if self._enqueue_job(job, priority=0, reason="INITIAL_TRACK"):
                        self._counters["initial_track_jobs_scheduled"] += 1
                        queued_this_frame += 1
                    continue
                if gallery_size < 2:
                    if self._enqueue_job(job, priority=1, reason="INSUFFICIENT_GALLERY"):
                        self._counters["insufficient_gallery_jobs_scheduled"] += 1
                        queued_this_frame += 1
                    continue
                previous = self._pending_window.get(track_key)
                if previous is not None and previous["window_index"] == job["window_index"]:
                    if quality["relative_quality_score"] > previous["crop_quality"]["relative_quality_score"]:
                        self._pending_window[track_key] = job
                        self._counters["quality_ranked_window_replacements"] += 1
                    continue
                self._pending_window[track_key] = job
            self._frame_history[camera_id][frame_index] = frame_rows
            while len(self._frame_history[camera_id]) > self.control_history_frames:
                self._frame_history[camera_id].popitem(last=False)
            self._counters["frames_joined"] += 1
            self._sampling_decision_ms.append(
                (time.perf_counter_ns() - decision_started_ns) / 1e6
            )
        return {"queued_crops": queued_this_frame, "objects_seen": len(frame_rows)}

    def _enqueue_job(self, job: dict[str, Any], *, priority: int, reason: str) -> bool:
        """Bounded non-blocking priority insertion; tracking never waits on OSNet."""
        track_key = (str(job["camera_id"]), str(job["local_track_id"]))
        generation = int(job["track_generation"])
        if self._scheduled_track_keys.get(track_key) == generation:
            self._counters["request_deduplicated_while_track_job_pending"] += 1
            return False
        job["priority"] = int(priority)
        job["schedule_reason"] = str(reason)
        job["enqueued_perf_ns"] = time.perf_counter_ns()
        entry = (int(priority), next(self._queue_sequence), job)
        try:
            self._queue.put_nowait(entry)
        except queue.Full:
            self._counters["queue_full_drops"] += 1
            return False
        self._scheduled_track_keys[track_key] = generation
        self._crop_selection_to_enqueue_ms.append(
            max(0, job["enqueued_perf_ns"] - int(job["selected_perf_ns"])) / 1e6
        )
        self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
        self._counters["crop_requests_scheduled"] += 1
        return True

    def _flush_old_windows(self, camera_id: str, current_window: int) -> int:
        """Queue the best pending sample when its source-time window closes."""
        queued = 0
        for key, job in list(self._pending_window.items()):
            if key[0] != camera_id or int(job["window_index"]) >= current_window:
                continue
            del self._pending_window[key]
            if not self._enqueue_job(job, priority=2, reason="NEW_VIEWPOINT"):
                continue
            self._counters["window_samples_flushed"] += 1
            self._counters["new_viewpoint_jobs_scheduled"] += 1
            queued += 1
        return queued

    def _flush_all_pending(self) -> None:
        with self._lock:
            for key, job in list(self._pending_window.items()):
                del self._pending_window[key]
                if not self._enqueue_job(job, priority=2, reason="NEW_VIEWPOINT"):
                    continue
                self._counters["window_samples_flushed"] += 1
                self._counters["new_viewpoint_jobs_scheduled"] += 1

    def record_join_failure(self, error: BaseException) -> None:
        """Keep a sidecar mismatch visible while leaving detection/tracking intact."""
        with self._lock:
            self._counters["frames_rejected"] += 1
            self._last_join_error = f"{type(error).__name__}: {error}"

    def _tracks_for_scoring(self) -> dict[str, dict[int, list[dict[str, Any]]]]:
        return {
            camera: {frame: rows for frame, rows in history.items()}
            for camera, history in self._frame_history.items()
        }

    def _score_and_update_assignments(self) -> None:
        started = time.perf_counter()
        previous_active = copy.deepcopy(self._active)
        previous_mapping = self._identity_memory.assignment_map()
        decision_now_ns = self._clock_ns()
        result = score_and_assign_provisional_ids(
            np.stack(self._vectors).astype(np.float32, copy=False),
            list(self._records),
            self._tracks_for_scoring(),
            minimum_crops_per_track=2,
            minimum_separated_control_frames=8,
            control_reference_max=self.control_reference_max,
            control_reference_source=self.control_reference_source,
            duplicate_track_conflicts=set(self._duplicate_track_conflicts),
            diagnostic_logging=self.decision_diagnostics,
            diagnostic_now_ns=decision_now_ns if self.decision_diagnostics else None,
        )
        for row in result.get("candidate_pairs", []):
            gates = row.get("automatic_candidate_gates", {})
            if gates.get("no_repeated_high_overlap_local_track_conflict") is False:
                pair = row["camera_local_pair"]
                self._duplicate_rejected_candidates.add(
                    (str(pair["CAM-02"]), str(pair["CAM-05"]), "duplicate_conflict")
                )
        self._counters["duplicate_track_pairs_rejected"] = len(self._duplicate_rejected_candidates)
        accepted_rows = []
        used_local_ids = {camera: set() for camera in CAMERAS}
        for row in result.get("matches", []):
            pair = row["camera_local_pair"]
            local_ids = {"CAM-02": str(pair["CAM-02"]), "CAM-05": str(pair["CAM-05"])}
            if any(local_ids[camera] in used_local_ids[camera] for camera in CAMERAS):
                self._counters["one_to_many_conflicts_rejected"] += 1
                continue
            accepted_rows.append(row)
            for camera in CAMERAS:
                used_local_ids[camera].add(local_ids[camera])
        identity_revision = self._identity_memory.revision
        memory_result = self._identity_memory.reconcile(
            accepted_rows, result.get("candidate_pairs", []), self._gallery,
            self._tracks_for_scoring(),
            control_reference_max=self.control_reference_max,
            now_ns=self._clock_ns(),
            expected_revision=identity_revision,
        )
        self._sync_identity_revision_events()
        if memory_result.get("stale"):
            self._counters["stale_embedding_results_rejected"] += 1
        mapping = self._identity_memory.assignment_map()
        active_pairs: dict[tuple[str, str], dict[str, Any]] = {}
        for row in accepted_rows:
            pair = (str(row["camera_local_pair"]["CAM-02"]),
                    str(row["camera_local_pair"]["CAM-05"]))
            left = mapping["CAM-02"].get(pair[0])
            right = mapping["CAM-05"].get(pair[1])
            if left is None or right is None or left["provisional_global_id"] != right["provisional_global_id"]:
                continue
            active_pairs[pair] = {
                "provisional_global_id": left["provisional_global_id"],
                "cosine_similarity_median": float(row["cosine_similarity_median"]),
                "gallery_similarity_min": float(row["cosine_similarity_min"]),
                "gallery_sizes": row["samples"],
                "first_eligible_monotonic_ns": self._active.get(pair, {}).get(
                    "first_eligible_monotonic_ns", self._clock_ns()
                ),
            }
        self._counters["candidate_withdrawals"] += len(set(self._active) - set(active_pairs))
        if self.decision_diagnostics:
            self._record_decision_diagnostics(
                result, previous_mapping, mapping, previous_active, active_pairs,
                memory_result, decision_now_ns,
            )
        self._active = active_pairs
        result["scoring_wall_ms"] = (time.perf_counter() - started) * 1000.0
        self._score_ms.append(float(result["scoring_wall_ms"]))
        self._score_result = result

    def _record_decision_diagnostics(
        self,
        result: dict[str, Any],
        previous_mapping: dict[str, dict[str, dict[str, Any]]],
        current_mapping: dict[str, dict[str, dict[str, Any]]],
        previous_active: dict[tuple[str, str], dict[str, Any]],
        current_active: dict[tuple[str, str], dict[str, Any]],
        memory_result: dict[str, Any],
        now_ns: int,
    ) -> None:
        """Keep a bounded, opt-in audit of matcher decisions without embeddings."""
        trace_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        abstained_pairs = {
            (str(pair[0]), str(pair[1]))
            for pair in memory_result.get("abstained", []) if len(pair) == 2
        }
        for trace in result.get("decision_trace", []):
            pair_data = trace.get("camera_local_pair", {})
            pair = (str(pair_data.get("CAM-02")), str(pair_data.get("CAM-05")))
            trace_by_pair[pair] = trace
            left = current_mapping["CAM-02"].get(pair[0])
            right = current_mapping["CAM-05"].get(pair[1])
            current_gid = (
                left.get("provisional_global_id")
                if left and right and left.get("provisional_global_id") == right.get("provisional_global_id")
                else None
            )
            previous_gids = {
                previous_mapping[camera].get(local_id, {}).get("provisional_global_id")
                for camera, local_id in (("CAM-02", pair[0]), ("CAM-05", pair[1]))
            } - {None}
            failed_gates = list(trace.get("failed_gates", []))
            reason = trace.get("reason", "matcher returned no reason")
            if pair in abstained_pairs:
                reason = "identity memory abstained: " + "; ".join(
                    event.get("reason", "unspecified identity-memory conflict")
                    for event in self._identity_memory.snapshot(now_ns).get("events", [])[-8:]
                    if event.get("camera_local_pair") == {"CAM-02": pair[0], "CAM-05": pair[1]}
                ) or "candidate continuity was ambiguous"
            elif current_gid is None and trace.get("decision") == "ACCEPTED_BY_MATCHER":
                reason = "matcher gates passed but identity memory did not accept the binding"
            elif current_gid is not None:
                reason = "accepted by matcher and identity memory"
            self._append_decision_event({
                "event": "PAIR_SCORING_DECISION",
                "monotonic_ns": int(now_ns),
                **copy.deepcopy(trace),
                "candidate_global_id": current_gid or next(iter(previous_gids), None)
                    or trace.get("candidate_global_id"),
                "matcher_status": result.get("status"),
                "matcher_summary_reason": result.get("reason"),
                "accepted_global_id": current_gid,
                "identity_memory_accepted": current_gid is not None,
                "failed_gates": failed_gates,
                "reason": reason,
                "source_pts_are_not_cross_camera_sync_evidence": True,
            })

        for pair, old in previous_active.items():
            if pair in current_active:
                continue
            trace = trace_by_pair.get(pair)
            if trace is None:
                reason = "pair absent from current scoring set; no current candidate supported the binding"
                if result.get("reason"):
                    reason += "; matcher status: " + str(result.get("reason"))
            elif trace.get("failed_gates"):
                reason = "withdrawn because existing matcher gate(s) failed: " + ", ".join(trace["failed_gates"])
            elif pair in abstained_pairs:
                reason = "withdrawn because identity-memory continuity was ambiguous"
            else:
                reason = "withdrawn because no current accepted candidate supported this binding"
            self._append_decision_event({
                "event": "GLOBAL_ID_WITHDRAWAL",
                "monotonic_ns": int(now_ns),
                "candidate_global_id": old.get("provisional_global_id"),
                "camera_local_pair": {"CAM-02": pair[0], "CAM-05": pair[1]},
                "source_observations": trace.get("source_observations") if trace else None,
                "best_similarity_score": trace.get("best_similarity_score") if trace else None,
                "competing_similarity_score_by_camera": (
                    trace.get("competing_similarity_score_by_camera") if trace else None
                ),
                "competing_track_pair_by_camera": (
                    trace.get("competing_track_pair_by_camera") if trace else None
                ),
                "gallery_sample_count_by_camera": {
                    camera: observation.get("gallery_sample_count")
                    for camera, observation in (trace.get("source_observations", {}) if trace else {}).items()
                },
                "reciprocal_match": trace.get("reciprocal_match") if trace else None,
                "margin_result": trace.get("margin_result") if trace else None,
                "gates": trace.get("gates") if trace else None,
                "failed_gates": trace.get("failed_gates", []) if trace else [],
                "reason": reason,
                "source_pts_are_not_cross_camera_sync_evidence": True,
            })

    def _append_decision_event(self, event: dict[str, Any]) -> None:
        self._decision_event_sequence += 1
        event["decision_sequence"] = self._decision_event_sequence
        self._decision_events.append(event)
        overflow = len(self._decision_events) - self.decision_trace_limit
        if overflow > 0:
            del self._decision_events[:overflow]
            self._decision_events_dropped += overflow

    def _sync_identity_revision_events(self) -> None:
        """Mirror the in-memory identity audit trail into the run summary."""
        events = self._identity_memory.snapshot(self._clock_ns()).get("events", [])
        for event in events:
            sequence = int(event.get("event_sequence", 0))
            if sequence <= self._identity_event_cursor:
                continue
            self._revision_events.append(copy.deepcopy(event))
            self._identity_event_cursor = sequence
        if len(self._revision_events) > 4096:
            del self._revision_events[:-4096]

    def _expire_galleries_locked(self, now_ns: int) -> list[tuple[str, str]]:
        """Evict expired RAM-only appearance history and withdraw stale candidates."""
        self._identity_memory.expire(int(now_ns))
        self._sync_identity_revision_events()
        current_mapping = self._identity_memory.assignment_map()
        for pair, active in list(self._active.items()):
            gid = active["provisional_global_id"]
            if (current_mapping["CAM-02"].get(pair[0], {}).get("provisional_global_id") != gid
                    or current_mapping["CAM-05"].get(pair[1], {}).get("provisional_global_id") != gid):
                del self._active[pair]
                self._counters["candidate_withdrawals"] += 1
        expired = [
            key for key, last_seen in self._gallery_last_seen_ns.items()
            if now_ns - int(last_seen) >= self.gallery_ttl_ns
        ]
        if not expired:
            return []
        for key in expired:
            gallery = self._gallery.pop(key, [])
            self._counters["gallery_ttl_expired_crops"] += len(gallery)
            self._gallery_last_seen_ns.pop(key, None)
            self._last_observed_source_pts_ns.pop(key, None)
            self._last_sample_frame.pop(key, None)
            self._pending_window.pop(key, None)
            self._gallery_generation[key] += 1
            self._scheduled_track_keys.pop(key, None)
            self._counters["gallery_ttl_expired_tracks"] += 1
        expired_ids = {(camera, local_id) for camera, local_id in expired}
        self._duplicate_overlap_counts = defaultdict(
            int,
            {key: count for key, count in self._duplicate_overlap_counts.items()
             if (key[0], key[1]) not in expired_ids and (key[0], key[2]) not in expired_ids},
        )
        self._duplicate_track_conflicts = {
            key for key in self._duplicate_track_conflicts
            if (key[0], key[1]) not in expired_ids and (key[0], key[2]) not in expired_ids
        }
        self._duplicate_rejected_candidates = {
            key for key in self._duplicate_rejected_candidates
            if (CAMERAS[0], key[0]) not in expired_ids
            and (CAMERAS[1], key[1]) not in expired_ids
        }
        flattened = []
        for camera in CAMERAS:
            for key in sorted((key for key in self._gallery if key[0] == camera), key=lambda key: key[1]):
                flattened.extend(self._gallery[key])
        self._records = [item["record"] for item in flattened]
        self._vectors = [item["vector"] for item in flattened]
        if self._vectors:
            self._score_and_update_assignments()
        else:
            previous = len(self._active)
            self._identity_memory.reconcile(
                [], [], self._gallery, self._tracks_for_scoring(),
                control_reference_max=self.control_reference_max,
                now_ns=int(now_ns), expected_revision=self._identity_memory.revision,
            )
            self._sync_identity_revision_events()
            self._counters["candidate_withdrawals"] += previous
            self._active.clear()
            self._score_result = {
                "status": "NO_UNEXPIRED_GALLERIES", "matches": [], "candidate_pairs": []
            }
        return expired

    def expire_galleries(self, now_ns: int | None = None) -> list[tuple[str, str]]:
        """Public deterministic TTL sweep used by the shadow viewer and tests."""
        with self._lock:
            return self._expire_galleries_locked(self._clock_ns() if now_ns is None else int(now_ns))

    def _run(self) -> None:
        while True:
            _, _, first = self._queue.get()
            if first is None:
                self._queue.task_done()
                return
            batch = [first]
            sentinel_seen = False
            while len(batch) < self.batch_size:
                try:
                    _, _, more = self._queue.get_nowait()
                except queue.Empty:
                    break
                if more is None:
                    self._queue.task_done()
                    sentinel_seen = True
                    break
                batch.append(more)

            dequeued_perf_ns = time.perf_counter_ns()
            for job in batch:
                self._queue_wait_ms.append(
                    max(0, dequeued_perf_ns - int(job["enqueued_perf_ns"])) / 1e6
                )
            with self._lock:
                self._counters["crop_jobs_dequeued"] += len(batch)

            ready_batch = []
            for job in batch:
                wait_seconds = max(0, dequeued_perf_ns - int(job["enqueued_perf_ns"])) / 1e9
                if wait_seconds > MAX_EMBEDDING_QUEUE_WAIT_SECONDS:
                    key = (str(job["camera_id"]), str(job["local_track_id"]))
                    with self._lock:
                        self._counters["stale_queue_jobs_expired"] += 1
                        self._counters["crop_jobs_expired_before_inference"] += 1
                        if self._scheduled_track_keys.get(key) == int(job["track_generation"]):
                            self._scheduled_track_keys.pop(key, None)
                    self._queue.task_done()
                else:
                    ready_batch.append(job)

            try:
                if ready_batch:
                    inference_started = time.perf_counter()
                    features = np.asarray(
                        self.embedder.embed_batch([job["crop"] for job in ready_batch]),
                        dtype=np.float32,
                    )
                    self._inference_batch_ms.append((time.perf_counter() - inference_started) * 1000.0)
                    if features.shape != (len(ready_batch), 512) or not np.isfinite(features).all():
                        raise ValueError("OSNet must return one finite 512-D vector per crop")
                    norms = np.linalg.norm(features.astype(np.float64), axis=1)
                    if not np.allclose(norms, 1.0, atol=1e-4):
                        raise ValueError("OSNet output vectors must be L2-normalized")
                    completed_ns = self._clock_ns()
                    ages_ns = [
                        max(0, completed_ns - int(job["observed_monotonic_ns"]))
                        for job in ready_batch
                    ]
                    self._sample_to_embedding_ms.extend(age / 1e6 for age in ages_ns)
                    with self._lock:
                        self._expire_galleries_locked(completed_ns)
                        self._counters["embedding_jobs_completed"] += len(ready_batch)
                        self._counters["embedding_jobs_over_stale_age_threshold"] += sum(
                            age > int(STALE_EMBEDDING_AGE_SECONDS * 1e9) for age in ages_ns
                        )
                        gallery_update_started = time.perf_counter_ns()
                        accepted_feature = False
                        for vector, job, age_ns in zip(features, ready_batch, ages_ns):
                            key = (str(job["camera_id"]), str(job["local_track_id"]))
                            generation = int(job["track_generation"])
                            if (generation != self._gallery_generation[key]
                                    or age_ns >= self.gallery_ttl_ns):
                                self._counters["stale_embedding_results_rejected"] += 1
                                continue
                            gallery = self._gallery[key]
                            record = {
                                "camera_id": job["camera_id"],
                                "source_id": job["source_id"],
                                "local_track_id": job["local_track_id"],
                                "frame": job["frame"],
                                "source_pts_ns": job["source_pts_ns"],
                                "observed_monotonic_ns": int(job["observed_monotonic_ns"]),
                                "bbox_inference_xywh": job["bbox_inference_xywh"],
                                "crop_quality": job["crop_quality"],
                                "sample_window_index": job["window_index"],
                                "schedule_reason": job["schedule_reason"],
                            }
                            gallery.append({"vector": vector.copy(), "record": record})
                            self._gallery_last_seen_ns[key] = max(
                                int(self._gallery_last_seen_ns.get(key, 0)),
                                int(job["observed_monotonic_ns"]),
                            )
                            gallery.sort(key=lambda item: int(item["record"]["frame"]))
                            while len(gallery) > self.max_crops_per_track:
                                gallery.pop(0)
                                self._counters["rolling_gallery_evictions"] += 1
                            self._counters["features_generated"] += 1
                            accepted_feature = True
                        self._gallery_update_ms.append(
                            (time.perf_counter_ns() - gallery_update_started) / 1e6
                        )
                        # Rebuild the bounded scoring matrix after accepted
                        # gallery updates; identity decisions remain unchanged.
                        if accepted_feature:
                            flattened = []
                            for camera in CAMERAS:
                                keys = sorted((key for key in self._gallery if key[0] == camera),
                                              key=lambda key: key[1])
                                for key in keys:
                                    flattened.extend(self._gallery[key])
                            self._records = [item["record"] for item in flattened]
                            self._vectors = [item["vector"] for item in flattened]
                            if self._vectors:
                                self._score_and_update_assignments()
            except Exception as exc:
                with self._lock:
                    self._counters["embedding_errors"] += len(ready_batch)
                    self._last_error = f"{type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    for job in batch:
                        key = (str(job["camera_id"]), str(job["local_track_id"]))
                        if self._scheduled_track_keys.get(key) == int(job["track_generation"]):
                            self._scheduled_track_keys.pop(key, None)
                for _job in ready_batch:
                    self._queue.task_done()
            if sentinel_seen:
                return

    def assignment_map(self) -> dict[str, dict[str, dict[str, Any]]]:
        with self._lock:
            self._expire_galleries_locked(self._clock_ns())
            return self._identity_memory.assignment_map()

    def display_identity_snapshot(self) -> dict[str, Any]:
        """Read current accepted bindings and revision under one lock.

        This is a lightweight display lease, not a new matching decision.
        Expiry and withdrawal semantics remain owned by identity memory.
        """
        from services.mv3dt_staging.map_fusion import IDENTITY_AUTHORITY
        with self._lock:
            now = self._clock_ns()
            self._expire_galleries_locked(now)
            return {"authority": IDENTITY_AUTHORITY,
                    "revision": self._identity_memory.revision,
                    "sampled_monotonic_ns": now,
                    "assignments": self._identity_memory.assignment_map()}

    def snapshot(self) -> dict[str, Any]:
        if self._closed_snapshot is not None:
            return copy.deepcopy(self._closed_snapshot)
        with self._lock:
            self._expire_galleries_locked(self._clock_ns())
            return {
                "enabled": True,
                "mode": "IN_MEMORY_NONPERSISTENT_SHADOW_ONLY",
                "gallery_memory": {
                    "storage": "RAM_ONLY",
                    "ttl_seconds": self.gallery_ttl_seconds,
                    "max_tracks": self.max_gallery_tracks,
                    "max_crops_per_track": self.max_crops_per_track,
                    "ttl_expiry_is_identity_validation": False,
                    "reappearance_recognition_accuracy": "UNVALIDATED",
                    "clock_domain": "wall_monotonic_seconds; independent of recorded source PTS and playback speed",
                },
                "global_identity_memory": self._identity_memory.snapshot(self._clock_ns()),
                "active_provisional_ids": [
                    {
                        "provisional_global_id": active["provisional_global_id"],
                        "camera_local_pair": {"CAM-02": pair[0], "CAM-05": pair[1]},
                        "cosine_similarity_median": active.get("cosine_similarity_median"),
                        "gallery_similarity_min": active.get("gallery_similarity_min"),
                        "gallery_sizes": active.get("gallery_sizes"),
                    }
                    for pair, active in sorted(self._active.items())
                ],
                "candidate_pair_count_last_update": len(self._score_result.get("candidate_pairs", [])),
                "uncertain_pair_count_last_update": sum(
                    not row.get("automatic_candidate", False)
                    for row in self._score_result.get("candidate_pairs", [])
                ),
                "same_camera_control_reference_max": self.control_reference_max,
                "same_camera_control_reference_source": self.control_reference_source,
                "same_camera_controls_are_ground_truth_negatives": False,
                "cross_camera_pts_used_as_sync_gate": False,
                "experimental_map_used_as_identity_gate": False,
                "production_identity_write": False,
                "raw_tracker_records_modified": False,
                "same_frame_contradictory_assignments": 0,
                "duplicate_track_conflict_count": len(self._duplicate_track_conflicts),
                "possible_local_fragmentation_candidates": len(
                    self._score_result.get("potential_local_fragmentation_candidates", [])
                ),
                "counters": dict(self._counters),
                "gallery_samples": {
                    camera: {
                        local_id: [
                            {"frame": int(item["record"]["frame"]),
                             "source_pts_ns": int(item["record"]["source_pts_ns"]),
                             "window_index": int(item["record"]["sample_window_index"]),
                             "quality_score": float(item["record"]["crop_quality"]["relative_quality_score"]),
                             "quality": item["record"]["crop_quality"]}
                            for item in self._gallery.get((camera, local_id), [])
                        ]
                        for cam, local_id in sorted(self._gallery) if cam == camera
                    }
                    for camera in CAMERAS
                },
                "timing_ms": {
                    "tracking_observation_decision": _percentiles(self._sampling_decision_ms),
                    "crop_extraction_view": _percentiles(self._crop_extraction_ms),
                    "owned_crop_copy": _percentiles(self._crop_copy_ms),
                    "crop_quality_measurement": _percentiles(self._crop_quality_ms),
                    "sample_selection_to_enqueue_including_quality_window": _percentiles(
                        self._crop_selection_to_enqueue_ms
                    ),
                    "queue_wait": _percentiles(self._queue_wait_ms),
                    "osnet_gpu_inference_batch": _percentiles(self._inference_batch_ms),
                    "gallery_update": _percentiles(self._gallery_update_ms),
                    "source_observation_to_embedding_completion": _percentiles(self._sample_to_embedding_ms),
                    "stale_job_telemetry_threshold_seconds": STALE_EMBEDDING_AGE_SECONDS,
                    "queue_expiry_threshold_seconds": MAX_EMBEDDING_QUEUE_WAIT_SECONDS,
                    "stale_job_policy": "queued crop is dropped before OSNet after queue wait exceeds threshold; detector/tracker are unaffected",
                    "matching": _percentiles(self._score_ms),
                },
                "osnet_embedder_stage_timing_ms": (
                    self.embedder.metrics().get("batch_stage_timing_ms")
                    if callable(getattr(self.embedder, "metrics", None)) else None
                ),
                "active_mapping": self.assignment_map(),
                "revision_events": list(self._revision_events),
                "decision_diagnostics_enabled": self.decision_diagnostics,
                "decision_diagnostic_event_count": len(self._decision_events),
                "decision_diagnostic_events_dropped_at_bound": self._decision_events_dropped,
                "decision_diagnostic_events": list(self._decision_events),
                "last_error": self._last_error or None,
                "last_join_error": self._last_join_error or None,
                "queue_depth": self._queue.qsize(),
                "max_queue_depth_observed": self._max_queue_depth,
                "queue_capacity": self._queue.maxsize,
                "in_flight_track_jobs": len(self._scheduled_track_keys),
                "pending_quality_window_count": len(self._pending_window),
                "sample_window_frames": self.sample_window_frames,
                "worker_alive": self._worker.is_alive(),
            }

    def close(self, timeout_sec: float = 8.0) -> dict[str, Any]:
        if self._closed_snapshot is not None:
            return copy.deepcopy(self._closed_snapshot)
        self._closed = True
        self._flush_all_pending()
        try:
            self._queue.put(
                (999, next(self._queue_sequence), None),
                timeout=max(0.1, min(timeout_sec, 1.0)),
            )
        except queue.Full:
            # Do not block tracker shutdown on stale Re-ID work. Drop pending
            # sample jobs and leave their local tracks explicitly unmatched.
            while True:
                try:
                    _, _, item = self._queue.get_nowait()
                except queue.Empty:
                    break
                self._queue.task_done()
                if item is not None:
                    self._counters["queue_full_drops"] += 1
                    self._counters["shutdown_queue_drops"] += 1
                    key = (str(item["camera_id"]), str(item["local_track_id"]))
                    if self._scheduled_track_keys.get(key) == int(item["track_generation"]):
                        self._scheduled_track_keys.pop(key, None)
            self._queue.put_nowait((999, next(self._queue_sequence), None))
        self._worker.join(timeout=max(0.1, float(timeout_sec)))
        result = self.snapshot()
        if self._worker.is_alive():
            raise TimeoutError("OSNet shadow worker did not terminate before its bounded shutdown timeout")
        # Snapshot retains compact JSON diagnostics only. Release all vectors,
        # crops, temporal frame histories, and candidate objects from RAM.
        with self._lock:
            released = {
                "gallery_tracks": len(self._gallery),
                "gallery_vectors": sum(len(rows) for rows in self._gallery.values()),
                "pending_crops": len(self._pending_window),
                "frame_history_rows": sum(len(rows) for rows in self._frame_history.values()),
            }
            self._gallery.clear()
            self._records.clear()
            self._vectors.clear()
            self._active.clear()
            self._identity_memory.clear()
            self._pending_window.clear()
            self._frame_history = {camera: OrderedDict() for camera in CAMERAS}
            self._last_observed_source_pts_ns.clear()
            self._gallery_last_seen_ns.clear()
            self._scheduled_track_keys.clear()
        result["shutdown_memory_release"] = released
        self._closed_snapshot = copy.deepcopy(result)
        return result
