"""Read-only API/state adapter for the scoped CAM-01/CAM-04 MV3DT runtime."""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import threading
from pathlib import Path
from typing import Any

from services.mv3dt_room.presence import (
    CAMERAS,
    active_observations,
    application_id as row_application_id,
    assert_presence_contract,
)


# Verified live RTSP dimensions for the Dev Room acceptance profile. Runtime
# crop provenance, when available, takes precedence over these profile values.
VERIFIED_DEV_ROOM_SOURCE_DIMENSIONS = {
    "CAM-01": (2560, 1440),
    "CAM-04": (3200, 1800),
}


def _json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except (OSError, ValueError):
        return default


class RoomPairState:
    def __init__(self, root: str | Path | None = None) -> None:
        self.base_root = Path(root or os.getenv("MV3DT_ROOM_RUNTIME_ROOT", ".runtime/mv3dt/dev-room-cam01-cam04"))
        self.root = self.base_root
        self.identity_dir = self.root / "identity-live"
        self.logs_dir = self.root / "logs"
        self._active_root: Path | None = None
        self._lock = threading.Lock()
        self._identity_mtime: int | None = None
        self._rows: list[dict[str, Any]] = []
        self._state_mtime: int | None = None
        self._state: dict[str, Any] | None = None
        self._show_native_ids = os.getenv("MV3DT_DEBUG_IDS", "0") == "1"
        self._acceptance_dims_offset = 0
        self._acceptance_dims: dict[str, tuple[int, int]] = {}
        self._acceptance_trace_offset = 0
        self._acceptance_trace: dict[tuple[str, int, int], dict[str, Any]] = {}

    def _refresh_acceptance_source_dimensions(self) -> dict[str, tuple[int, int]]:
        """Incrementally read logged frame dimensions for acceptance UI hit-testing.

        Crop provenance already records the DeepStream source dimensions. This
        read-only tailer avoids guessing a scale and never participates in
        detector, tracker, or identity decisions.
        """
        path = self.identity_dir / "global_identity.jsonl"
        try:
            size = path.stat().st_size
        except OSError:
            return dict(self._acceptance_dims)
        if size < self._acceptance_dims_offset:
            self._acceptance_dims_offset = 0
            self._acceptance_dims = {}
        try:
            with path.open("rb") as handle:
                handle.seek(self._acceptance_dims_offset)
                data = handle.read()
        except OSError:
            return dict(self._acceptance_dims)
        complete = data.rfind(b"\n") + 1
        if complete <= 0:
            return dict(self._acceptance_dims)
        for line in data[:complete].splitlines():
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            camera = row.get("camera_id")
            provenance = row.get("crop_provenance")
            if camera not in CAMERAS or not isinstance(provenance, dict):
                continue
            try:
                width = int(provenance.get("source_frame_width") or 0)
                height = int(provenance.get("source_frame_height") or 0)
            except (TypeError, ValueError):
                continue
            if width > 0 and height > 0:
                self._acceptance_dims[str(camera)] = (width, height)
        self._acceptance_dims_offset += complete
        return dict(self._acceptance_dims)

    @staticmethod
    def _acceptance_candidate_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
        fields = (
            "application_id", "cosine_best_gallery", "gallery_camera_similarity",
            "gallery_size", "world_distance_m", "timestamp_difference_ms",
            "final_candidate_score", "exact_rejection_reason", "active",
        )
        result = {}
        for field in fields:
            value = candidate.get(field)
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            if value is None or isinstance(value, (str, int, float, bool)):
                result[field] = value
        return result

    def _refresh_acceptance_identity_trace(self) -> None:
        """Incrementally read redacted identity evidence for acceptance review only."""
        path = self.identity_dir / "identity_path_trace.jsonl"
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size < self._acceptance_trace_offset:
            self._acceptance_trace_offset = 0
            self._acceptance_trace.clear()
        try:
            with path.open("rb") as handle:
                handle.seek(self._acceptance_trace_offset)
                data = handle.read()
        except OSError:
            return
        complete = data.rfind(b"\n") + 1
        if complete <= 0:
            return
        for line in data[:complete].splitlines():
            try:
                event = json.loads(line)
            except (TypeError, ValueError):
                continue
            kind = event.get("event")
            camera = event.get("camera_id")
            native = event.get("native_track_id")
            frame = event.get("observation_frame")
            if kind == "identity_candidate_evaluated":
                trace = event.get("decision_trace") or {}
                observation = trace.get("observation") or {}
                camera = observation.get("camera_id")
                native = observation.get("native_track_id")
                frame = observation.get("frame")
                candidates = trace.get("candidate_trace") or []
                item = {
                    "event": kind,
                    "best_candidate_similarity": event.get("best_candidate_similarity"),
                    "candidate_reason": (trace.get("novelty") or {}).get("reason"),
                    "candidates": [
                        self._acceptance_candidate_evidence(candidate)
                        for candidate in candidates[:8]
                        if isinstance(candidate, dict)
                    ],
                }
            elif kind == "identity_decision":
                details = event.get("decision_evidence") or {}
                item = {
                    "event": kind,
                    "identity_state": event.get("identity_state"),
                    "decision_reason": event.get("decision_reason"),
                    "cosine_similarity": details.get("appearance_similarity"),
                }
            else:
                continue
            try:
                key = (str(camera), int(native), int(frame))
            except (TypeError, ValueError):
                continue
            previous = self._acceptance_trace.get(key, {})
            previous.update(item)
            self._acceptance_trace[key] = previous
            while len(self._acceptance_trace) > 256:
                self._acceptance_trace.pop(next(iter(self._acceptance_trace)))
        self._acceptance_trace_offset += complete

    def _refresh_active_root(self) -> None:
        if (self.base_root / "identity-live").exists() or (self.base_root / "logs").exists():
            active = self.base_root
        else:
            candidates = [
                path for path in self.base_root.glob("*/")
                if (path / "identity-live").exists()
                or (path / "running").exists()
                or (path / "source_mode.json").exists()
            ]
            # A crashed/old run may leave a stale `running` marker behind.
            # Select the newest run directory; its marker still determines
            # status, but an old marker must not pin the UI to stale state.
            active = max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else self.base_root
        if active != self._active_root:
            self.root = active
            self.identity_dir = active / "identity-live"
            self.logs_dir = active / "logs"
            self._active_root = active
            self._identity_mtime = None
            self._state_mtime = None
            self._rows = []
            self._state = None
            self._acceptance_dims_offset = 0
            self._acceptance_dims = {}
            self._acceptance_trace_offset = 0
            self._acceptance_trace = {}

    def _current_state(self) -> dict[str, Any] | None:
        self._refresh_active_root()
        path = self.identity_dir / "current_state.json"
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            self._state_mtime = None
            self._state = None
            return None
        if mtime == self._state_mtime:
            state = self._state
            return state if isinstance(state, dict) else None
        state = _json(path, None)
        self._state_mtime = mtime
        self._state = state if isinstance(state, dict) else None
        return self._state

    def readiness(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_active_root()
            candidates = (
                self.root / "run/logs/probe/readiness.json",
                self.root / "logs/probe/readiness.json",
            )
            for path in candidates:
                payload = _json(path, None)
                if isinstance(payload, dict):
                    return payload
        return {"status": "not_ready", "ready": False, "reason": "readiness_missing"}

    def _identity_rows(self) -> list[dict[str, Any]]:
        self._refresh_active_root()
        path = self.identity_dir / "global_identity.jsonl"
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            return self._rows
        if mtime == self._identity_mtime:
            return self._rows
        rows: list[dict[str, Any]] = []
        try:
            with path.open() as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("camera_id") in CAMERAS:
                        rows.append(row)
        except OSError:
            return self._rows
        self._identity_mtime, self._rows = mtime, rows
        return rows

    def _metrics(
        self,
        rows: list[dict[str, Any]],
        active_rows: list[dict[str, Any]],
        report: dict[str, Any],
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "bbox_corrections_per_camera": {camera: 0 for camera in CAMERAS},
            "bbox_correction_trigger_reason": {},
            "native_ids_created": 0,
            "application_global_ids_active": 0,
            "osnet_embedding_count": int(report.get("embeddings_extracted", 0) or 0),
            "osnet_matching_latency_ms": report.get("osnet_latency_ms", {}),
            "global_identity_manager_matching_latency_us": report.get("identity_manager_latency_us", {}),
            "id_reacquisition_latency": _json(self.identity_dir / "reacquisition.json", {}),
            "false_conflicting_match_guard_events": 0,
            "per_camera_fps": {camera: report.get("effective_source_fps_per_camera") for camera in CAMERAS},
            "gpu_cpu_ram": _json(self.root / "resource_metrics.json", {}),
        }
        identity_metrics = report.get("identity_metrics", {}) if isinstance(report, dict) else {}
        metrics["identity_queue_depth"] = identity_metrics.get("max", {}).get("identity_latest_queue_depth", 0)
        metrics["observation_age_at_processing_ms"] = identity_metrics.get("latency_ms", {}).get("age_at_processing_ms", {})
        metrics["observation_age_at_publication_ms"] = identity_metrics.get("latency_ms", {}).get("age_at_publication_ms", {})
        metrics["stage_latency_ms"] = identity_metrics.get("latency_ms", {})
        metrics["stale_reid_results_discarded"] = identity_metrics.get("counters", {}).get("stale_reid_results_discarded", 0)
        metrics["coalesced_stale_updates"] = identity_metrics.get("counters", {}).get("coalesced_stale_updates", 0)
        metrics["identity_reassociations"] = identity_metrics.get("counters", {}).get("successful_native_reassociations", 0)
        metrics["new_application_id_allocations"] = identity_metrics.get("counters", {}).get("genuinely_new_application_id_allocations", 0)
        metrics["attempted_reassignments"] = report.get("identity_manager", {}).get("reassignment_attempts", 0)
        metrics["rejected_reassignments"] = report.get("identity_manager", {}).get("reassignment_rejected", 0)
        metrics["osnet_batch_size_distribution"] = report.get("osnet", {}).get("embedder", {}).get("batch_sizes", {})
        native_ids = {(row.get("camera_id"), row.get("native_track_id")) for row in rows}
        metrics["native_ids_created"] = len({key for key in native_ids if key[0] in CAMERAS and key[1] is not None})
        active = {
            row_application_id(row)
            for row in active_rows
        }
        metrics["application_global_ids_active"] = len({item for item in active if item and item != "Unknown"})
        decisions = self.logs_dir / "probe" / "bbox_decisions.csv"
        if not decisions.exists():
            decisions = self.root / "run/logs/probe/bbox_decisions.csv"
        try:
            with decisions.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("applied") != "1":
                        continue
                    camera = row.get("source")
                    if camera in metrics["bbox_corrections_per_camera"]:
                        metrics["bbox_corrections_per_camera"][camera] += 1
                    reason = row.get("trigger_reason") or "unknown"
                    metrics["bbox_correction_trigger_reason"][reason] = metrics["bbox_correction_trigger_reason"].get(reason, 0) + 1
        except OSError:
            pass
        events = _json(self.identity_dir / "identity_events.json", [])
        if isinstance(events, list):
            metrics["false_conflicting_match_guard_events"] = sum(
                1 for event in events
                if any(token in json.dumps(event).lower() for token in ("veto", "conflict", "guard"))
            )
            if not metrics["id_reacquisition_latency"]:
                gaps = [
                    float(event["evidence"]["frame_gap"])
                    for event in events
                    if event.get("event") == "ACQUIRE"
                    and event.get("evidence", {}).get("frame_gap") is not None
                ]
                if gaps:
                    ordered = sorted(gaps)
                    index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
                    metrics["id_reacquisition_latency"] = {
                        "count": len(gaps),
                        "frames": gaps,
                        "median": statistics.median(gaps),
                        "p95": ordered[index],
                        "max": max(gaps),
                    }
        return metrics

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            current_state = self._current_state()
            rows = list(self._identity_rows()) if current_state is None else list(current_state.get("people", []))
            report = _json(self.identity_dir / "runtime_report.json", {})
        if current_state is not None:
            active_rows = rows
            frames = [
                int(value)
                for value in current_state.get("active_frame_by_camera", {}).values()
                if value is not None and int(value) >= 0
            ]
            active_frame = max(frames) if frames else None
        else:
            active_frame, active_rows = active_observations(rows)
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for row in active_rows:
            camera = row.get("camera_id")
            identity = row_application_id(row)
            if camera in CAMERAS:
                # Native/MV3DT fragments are implementation details. Keep only
                # the current observation for each application identity per
                # camera; native ID remains in debug metadata.
                latest[(camera, identity)] = row
        people = []
        for row in latest.values():
            identity = row_application_id(row)
            debug = {
                "internal_identity": row.get("canonical_internal_identity", row.get("internal_identity_at_event")),
                "decision_reason": row.get("decision_reason"),
                "embedding_extracted": row.get("embedding_extracted", False),
            }
            if self._show_native_ids:
                debug["native_mv3dt_id"] = row.get("native_track_id")
            people.append({
                "application_id": identity,
                "camera_id": row.get("camera_id"),
                "bbox": row.get("bbox"),
                "world": row.get("world"),
                "confidence": row.get("confidence"),
                "frame": row.get("frame"),
                "timestamp": row.get("timestamp"),
                "debug": debug,
            })
        people.sort(key=lambda row: (row["camera_id"], row["application_id"], row["debug"].get("native_mv3dt_id") or -1))
        source = _json(self.root / "source_mode.json", {})
        source_mode = str(source.get("source_mode") or ("live" if (self.root / "live").exists() else "replay"))
        if source_mode not in {"replay", "live"}:
            source_mode = "live"
        session_id = str(source.get("session_id") or self.root.name)
        status = "running" if (self.root / "running").exists() else "idle"
        if (self.root / "done").exists():
            status = "complete"
        presence_assertions = assert_presence_contract(
            active_rows,
            {
                person["application_id"]
                for person in people
                if person.get("world") is not None
            },
        )
        presence_assertions["active_frame"] = active_frame
        return {
            "profile": "dev-room-cam01-cam04",
            "status": status,
            "source_mode": source_mode,
            "session_id": session_id,
            "readiness": self.readiness(),
            "cameras": list(CAMERAS),
            "face_recognition": False,
            "people": people,
            "presence": presence_assertions,
            "metrics": self._metrics(rows, active_rows, report),
            "runtime_report": report,
            "native_ids_are_debug_only": True,
        }

    def acceptance_candidates(self) -> dict[str, Any]:
        """Read-only candidate rows for the opt-in operator acceptance UI.

        This endpoint is deliberately separate from the production identity
        snapshot. It exposes native IDs only to an acceptance-mode caller and
        has no write path into the tracker or identity manager.
        """
        with self._lock:
            state = self._current_state()
            source_dimensions = self._refresh_acceptance_source_dimensions()
            self._refresh_acceptance_identity_trace()
            if not isinstance(state, dict):
                return {
                    "acceptance_only": True,
                    "status": "not_ready",
                    "source_mode": "live",
                    "session_id": self.root.name,
                    "candidates": [],
                }
            source_dimensions_source = {camera: "runtime_crop_provenance" for camera in source_dimensions}
            if state.get("source_mode") == "live":
                for camera, defaults in VERIFIED_DEV_ROOM_SOURCE_DIMENSIONS.items():
                    if camera not in source_dimensions:
                        camera_key = camera.replace("-", "")
                        try:
                            width = int(os.getenv(f"MV3DT_OVERLAY_WIDTH_{camera_key}", defaults[0]))
                            height = int(os.getenv(f"MV3DT_OVERLAY_HEIGHT_{camera_key}", defaults[1]))
                        except (TypeError, ValueError):
                            continue
                        if width > 0 and height > 0:
                            source_dimensions[camera] = (width, height)
                            source_dimensions_source[camera] = "verified_dev_room_profile_or_env"
            candidates = []
            for row in state.get("people", []):
                camera = row.get("camera_id")
                bbox = row.get("bbox")
                if camera not in CAMERAS or row.get("native_track_id") is None:
                    continue
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue
                try:
                    cosine_similarity = float(row["cosine_similarity"])
                    if not math.isfinite(cosine_similarity):
                        cosine_similarity = None
                except (KeyError, TypeError, ValueError):
                    cosine_similarity = None
                try:
                    pending_age_frames = max(0, int(row["pending_age_frames"]))
                except (KeyError, TypeError, ValueError):
                    pending_age_frames = None
                frame = int(row.get("source_frame_number", row.get("frame", -1)))
                native_track_id = int(row["native_track_id"])
                evidence = self._acceptance_trace.get((str(camera), native_track_id, frame), {})
                evidence_similarity = evidence.get("cosine_similarity", evidence.get("best_candidate_similarity"))
                try:
                    evidence_similarity = float(evidence_similarity) if evidence_similarity is not None else None
                except (TypeError, ValueError):
                    evidence_similarity = None
                if evidence_similarity is not None and not math.isfinite(evidence_similarity):
                    evidence_similarity = None
                candidate_summaries = evidence.get("candidates", [])
                top_candidate = max(
                    (item for item in candidate_summaries if item.get("application_id")),
                    key=lambda item: float(item.get("final_candidate_score") or -1e30),
                    default={},
                )
                candidates.append({
                    "camera_id": camera,
                    "frame": frame,
                    "timestamp": row.get("source_timestamp") or row.get("timestamp"),
                    "native_track_id": native_track_id,
                    "global_person_id": row.get("global_person_id", row.get("application_id", "Unknown")),
                    "application_id": row.get("application_id", row.get("global_person_id", "Unknown")),
                    "bbox": [float(value) for value in bbox[:4]],
                    "world": row.get("world"),
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "visibility": float(row.get("visibility", 0.0) or 0.0),
                    "identity_state": row.get("identity_state"),
                    "decision_reason": row.get("decision_reason") or evidence.get("decision_reason") or evidence.get("candidate_reason"),
                    "cosine_similarity": cosine_similarity if cosine_similarity is not None else evidence_similarity,
                    "pending_age_frames": pending_age_frames,
                    "gallery_candidate": row.get("gallery_candidate") or top_candidate.get("application_id"),
                    "identity_evidence": evidence,
                    "source_frame_width": int(row.get("source_frame_width") or source_dimensions.get(camera, (0, 0))[0]),
                    "source_frame_height": int(row.get("source_frame_height") or source_dimensions.get(camera, (0, 0))[1]),
                })
            return {
                "acceptance_only": True,
                "status": state.get("status", "running"),
                "source_mode": state.get("source_mode", "live"),
                "session_id": state.get("session_id", self.root.name),
                "active_frame_by_camera": state.get("active_frame_by_camera", {}),
                "source_dimensions_by_camera": {
                    camera: {"width": dims[0], "height": dims[1]}
                    for camera, dims in source_dimensions.items()
                },
                "source_dimensions_source": source_dimensions_source,
                "coordinate_dimensions_ready": all(camera in source_dimensions for camera in CAMERAS),
                "candidates": candidates,
            }
