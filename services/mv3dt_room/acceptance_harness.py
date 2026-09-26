"""Operator-confirmed live acceptance annotations, isolated from production identity.

This module only consumes UI/API snapshots and operator-selected observations.
It has no dependency on GlobalIdentityManager, OSNet, tracker, or MVA and never
publishes annotation data back to those systems.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable


CAMERAS = {"CAM-01", "CAM-04"}
TESTERS = ("Tester_A", "Tester_B")


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "acceptance"


def _canonical(row: dict[str, Any]) -> str:
    return str(row.get("global_person_id") or row.get("application_id") or "Unknown")


class AcceptanceHarness:
    """Small event recorder for independently evaluating human-confirmed tracks."""

    def __init__(
        self,
        artifact_root: Path | str,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], str] = _utc_now,
        exit_grace_seconds: float = 2.0,
        absent_confirmation_seconds: float = 10.0,
        near_pass_start_m: float = 1.5,
        near_pass_end_m: float = 2.0,
        pre_exit_stability_seconds: float = 5.0,
        pre_exit_min_observations: int = 3,
        minimum_subtest_seconds: float = 120.0,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self._monotonic = monotonic
        self._wall_time = wall_time
        self.exit_grace_seconds = max(0.1, float(exit_grace_seconds))
        self.absent_confirmation_seconds = max(self.exit_grace_seconds, float(absent_confirmation_seconds))
        self.near_pass_start_m = float(near_pass_start_m)
        self.near_pass_end_m = max(self.near_pass_start_m, float(near_pass_end_m))
        self.pre_exit_stability_seconds = max(0.0, float(pre_exit_stability_seconds))
        self.pre_exit_min_observations = max(2, int(pre_exit_min_observations))
        self.minimum_subtest_seconds = max(0.0, float(minimum_subtest_seconds))
        self.enabled = False
        self.subtest: str | None = None
        self.session_id = ""
        self.events_path: Path | None = None
        self.testers: dict[str, dict[str, Any]] = {}
        self.latest_observations: dict[str, dict[str, Any]] = {}
        self.near_pass_active = False
        self.near_pass_start_mono: float | None = None
        self.near_pass_end_mono: float | None = None
        self.post_cross_presence_since: float | None = None
        self.failure: dict[str, Any] | None = None
        self._last_snapshot_frame: dict[tuple[str, str], int] = {}
        self.both_present_recorded = False
        self.post_cross_stability_recorded = False
        self.exit_tester = "Tester_A"
        self._identity_change_keys: set[tuple[str, str, int, int]] = set()
        self.expected_runtime_session: str | None = None
        self._runtime_session_warning_logged = False
        self.started_monotonic: float | None = None
        self.completed_monotonic: float | None = None
        self.operator_confirmed_distinct_people = False

    def enable(self, enabled: bool) -> None:
        requested = bool(enabled)
        if self.enabled and not requested and self.subtest is not None and not self.failure:
            completed = all(item["done"] for item in self.checklist().get("items", []))
            if not completed:
                self.failure = {"kind": "acceptance_mode_disabled_mid_subtest"}
                self._emit("ACCEPTANCE_MODE_DISABLED", {"subtest_invalidated": True})
        self.enabled = requested

    def start_subtest(self, subtest: str, session_id: str) -> Path:
        value = str(subtest).upper().replace("SUBTEST_", "")
        if value not in {"A", "B"}:
            raise ValueError("subtest must be A or B")
        if not self.enabled:
            raise RuntimeError("acceptance mode is disabled")
        if value == "A":
            return self.prepare_subtest_a(session_id)
        self.subtest = value
        self.session_id = _safe_name(session_id or "live")
        self.testers = {}
        self.latest_observations = {}
        self.near_pass_active = False
        self.near_pass_start_mono = None
        self.near_pass_end_mono = None
        self.post_cross_presence_since = None
        self.failure = None
        self._last_snapshot_frame = {}
        self.both_present_recorded = False
        self.post_cross_stability_recorded = False
        self.exit_tester = "Tester_A"
        self._identity_change_keys = set()
        self.expected_runtime_session = None
        self._runtime_session_warning_logged = False
        self.started_monotonic = self._monotonic()
        self.operator_confirmed_distinct_people = False
        self.completed_monotonic = None
        self.events_path = self.artifact_root / f"{self.session_id}_subtest_{value.lower()}.jsonl"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if self.events_path.exists():
            raise FileExistsError(f"acceptance session already exists: {self.events_path}")
        self._emit("SUBTEST_STARTED", {"subtest": value, "session_id": self.session_id})
        return self.events_path

    def prepare_subtest_a(self, session_id: str) -> Path:
        """Open Subtest A in WAITING state; its timer starts only after explicit confirmation."""
        if not self.enabled:
            raise RuntimeError("acceptance mode is disabled")
        self.subtest = "A"
        self.session_id = _safe_name(session_id or "live")
        self.testers = {}
        self.latest_observations = {}
        self.near_pass_active = False
        self.near_pass_start_mono = None
        self.near_pass_end_mono = None
        self.post_cross_presence_since = None
        self.failure = None
        self._last_snapshot_frame = {}
        self.both_present_recorded = False
        self.post_cross_stability_recorded = False
        self._identity_change_keys = set()
        self.expected_runtime_session = None
        self._runtime_session_warning_logged = False
        self.started_monotonic = None
        self.completed_monotonic = None
        self.operator_confirmed_distinct_people = False
        self.exit_tester = "Tester_A"
        self.events_path = self.artifact_root / f"{self.session_id}_subtest_a.jsonl"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if self.events_path.exists():
            raise FileExistsError(f"acceptance session already exists: {self.events_path}")
        self._emit("SUBTEST_A_WAITING", {
            "subtest": "A",
            "session_id": self.session_id,
            "waiting_for": ["Tester_A", "Tester_B", "operator_confirmation"],
            "timer_started": False,
        })
        return self.events_path

    @staticmethod
    def _same_selected_observation(a: dict[str, Any], b: dict[str, Any]) -> bool:
        """Reject assigning the exact same observed bbox twice, independent of Person_XX."""
        return (
            a["camera_id"] == b["camera_id"]
            and a["frame"] == b["frame"]
            and a["native_track_id"] == b["native_track_id"]
            and a["bbox"] == b["bbox"]
        )

    def confirm_subtest_a_ready(self, operator_confirmed_distinct_people: bool) -> dict[str, Any]:
        if not self.enabled or self.subtest != "A":
            raise RuntimeError("Subtest A is not waiting for operator assignments")
        a, b = self.testers.get("Tester_A"), self.testers.get("Tester_B")
        if not a or not b:
            raise RuntimeError("WAITING FOR: select visible physical observations for Tester_A and Tester_B")
        if self._same_selected_observation(a["initial_observation"], b["initial_observation"]):
            raise RuntimeError("Tester_A and Tester_B must be selected from distinct bbox observations")
        if not operator_confirmed_distinct_people:
            raise RuntimeError("operator must confirm Tester_A and Tester_B are different physical people")
        self.operator_confirmed_distinct_people = True
        details = {
            "testers": {
                tester: {
                    "Person_XX": record["canonical_person_id"],
                    "camera": record["initial_camera"],
                    "frame": record["initial_observation"]["frame"],
                    "source_timestamp": record["initial_observation"]["timestamp"],
                    "native_track_id": record["initial_native_track_id"],
                    "bbox": record["initial_observation"]["bbox"],
                }
                for tester, record in (("Tester_A", a), ("Tester_B", b))
            },
            "operator_confirmed_different_physical_people": True,
            "ground_truth_only": True,
        }
        if a["canonical_person_id"] == b["canonical_person_id"]:
            self.failure = {
                "kind": "FAIL_FALSE_MERGE_AT_START",
                **details,
            }
            return self._emit("FAIL_FALSE_MERGE_AT_START", self.failure)
        self.started_monotonic = self._monotonic()
        self._emit("SUBTEST_STARTED", {"subtest": "A", "session_id": self.session_id, **details})
        return self._emit("SUBTEST_A_READY", {
            **details,
            "timer_started": True,
            "started_monotonic": self.started_monotonic,
        })

    def _emit(self, event: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        item = {
            "event": event,
            "timestamp": self._wall_time(),
            "subtest": self.subtest,
            "session_id": self.session_id,
            **(payload or {}),
        }
        if self.events_path is not None:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        return item

    @staticmethod
    def validate_observation(row: dict[str, Any]) -> dict[str, Any]:
        camera = str(row.get("camera_id", ""))
        if camera not in CAMERAS:
            raise ValueError("acceptance observations must be from CAM-01 or CAM-04")
        if row.get("native_track_id") is None:
            raise ValueError("operator assignment requires a native track ID")
        bbox = row.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            raise ValueError("operator assignment requires a bbox")
        bbox = [float(x) for x in bbox[:4]]
        if not all(math.isfinite(value) for value in bbox):
            raise ValueError("operator assignment bbox must contain finite coordinates")
        frame = int(row.get("source_frame_number", row.get("frame", -1)))
        if frame < 0:
            raise ValueError("operator assignment requires a valid source frame number")
        world = None
        if isinstance(row.get("world"), (list, tuple)) and len(row["world"]) >= 2:
            candidate_world = [float(x) for x in row["world"][:2]]
            if all(math.isfinite(value) for value in candidate_world):
                world = candidate_world
        similarity = row.get("cosine_similarity")
        try:
            similarity = float(similarity) if similarity is not None else None
        except (TypeError, ValueError):
            similarity = None
        if similarity is not None and not math.isfinite(similarity):
            similarity = None
        try:
            confidence = float(row.get("confidence", 0.0) or 0.0)
            visibility = float(row.get("visibility", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence, visibility = 0.0, 0.0
        if not math.isfinite(confidence):
            confidence = 0.0
        if not math.isfinite(visibility):
            visibility = 0.0
        pending_age = row.get("pending_age_frames")
        try:
            pending_age = max(0, int(pending_age)) if pending_age is not None else None
        except (TypeError, ValueError):
            pending_age = None
        identity_evidence = row.get("identity_evidence")
        try:
            serialized_evidence = json.dumps(identity_evidence, allow_nan=False, separators=(",", ":"))
            identity_evidence = json.loads(serialized_evidence) if len(serialized_evidence) <= 8192 else None
        except (TypeError, ValueError):
            identity_evidence = None
        return {
            "camera_id": camera,
            "frame": frame,
            "timestamp": str(row.get("source_timestamp") or row.get("timestamp") or ""),
            "native_track_id": int(row["native_track_id"]),
            "global_person_id": _canonical(row),
            "bbox": bbox,
            "world": world,
            "confidence": confidence,
            "visibility": visibility,
            # Optional, read-only diagnostics. These never leave the acceptance
            # event recorder and are not consumed by any production component.
            "identity_state": row.get("identity_state"),
            "decision_reason": row.get("decision_reason"),
            "cosine_similarity": similarity,
            "pending_age_frames": pending_age,
            "gallery_candidate": row.get("gallery_candidate") if isinstance(row.get("gallery_candidate"), (str, int)) else None,
            "identity_evidence": identity_evidence,
        }

    def assign(self, tester: str, row: dict[str, Any], crop_path: str | None = None) -> list[dict[str, Any]]:
        if not self.enabled:
            raise RuntimeError("acceptance mode is disabled")
        if self.subtest is None:
            raise RuntimeError("start Subtest A or B before assigning a tester")
        if tester not in TESTERS:
            raise ValueError("tester must be Tester_A or Tester_B")
        observation = self.validate_observation(row)
        existing_record = self.testers.get(tester)
        awaiting_reentry = bool(existing_record and existing_record.get("full_exit") is not None)
        if observation["global_person_id"] == "Unknown" and not awaiting_reentry:
            raise ValueError("select a visible bbox with a canonical Person_XX identity")
        other_tester = "Tester_B" if tester == "Tester_A" else "Tester_A"
        other_record = self.testers.get(other_tester)
        if other_record and self._same_selected_observation(
            observation, other_record["initial_observation"]
        ):
            raise ValueError("Tester_A and Tester_B must be selected from distinct physical bbox observations")
        now = self._monotonic()
        record = existing_record
        emitted: list[dict[str, Any]] = []
        if record is None:
            record = {
                "canonical_person_id": observation["global_person_id"],
                "initial_camera": observation["camera_id"],
                "initial_native_track_id": observation["native_track_id"],
                "initial_timestamp": observation["timestamp"] or self._wall_time(),
                "reference_crop": crop_path,
                "seen_cameras": {observation["camera_id"]},
                "last_seen_monotonic": now,
                "last_seen_timestamp": observation["timestamp"] or self._wall_time(),
                "current_observation": observation,
                "exit_requested": False,
                "full_exit": None,
                "absent_10s": False,
                "completed_absence_10s": False,
                "crossed_camera_at": None,
                "cross_camera_verified": False,
                "identity_changes": [],
                "reentry_done": False,
                "same_person_reacquired": False,
                "reentry_candidate_logged": False,
                "present_since_monotonic": now,
                "currently_visible": True,
                "stable_before_exit": False,
                "initial_observation": observation,
                "stable_observation_count": 1,
                "stable_last_frame_by_camera": {observation["camera_id"]: observation["frame"]},
            }
            self.testers[tester] = record
            emitted.append(self._emit(f"{tester.upper()}_ASSIGNED", {
                "tester": tester,
                "initial_camera": observation["camera_id"],
                "native_track_id": observation["native_track_id"],
                "Person_XX": observation["global_person_id"],
                "frame": observation["frame"],
                "source_timestamp": observation["timestamp"],
                "reference_crop": crop_path,
                "ground_truth_only": True,
            }))
        else:
            expected = str(record["canonical_person_id"])
            actual = str(observation["global_person_id"])
            if actual != expected and not (awaiting_reentry and actual == "Unknown"):
                change = {
                    "tester": tester,
                    "camera": observation["camera_id"],
                    "frame": observation["frame"],
                    "timestamp": observation["timestamp"],
                    "native_track_id": observation["native_track_id"],
                    "assigned_person_id": actual,
                    "expected_person_id": expected,
                    "bbox": observation["bbox"],
                    "world": observation["world"],
                }
                record["identity_changes"].append(change)
                change_key = (
                    tester,
                    observation["camera_id"],
                    observation["native_track_id"],
                    observation["frame"],
                )
                if change_key not in self._identity_change_keys:
                    self._identity_change_keys.add(change_key)
                    self.failure = self.failure or {"kind": "identity_change", **change}
                    emitted.append(self._emit("IDENTITY_CHANGE", change))
            previous_camera = record["current_observation"]["camera_id"]
            previous_native = record["current_observation"]["native_track_id"]
            if observation["camera_id"] != previous_camera and observation["global_person_id"] == expected:
                record["seen_cameras"].add(observation["camera_id"])
                record["crossed_camera_at"] = now
                record["cross_camera_verified"] = True
                emitted.append(self._emit(f"CROSS_CAMERA_{tester[-1]}", {
                    "tester": tester,
                    "Person_XX": expected,
                    "from_camera": previous_camera,
                    "to_camera": observation["camera_id"],
                    "from_native_track_id": previous_native,
                    "to_native_track_id": observation["native_track_id"],
                    "frame": observation["frame"],
                    "source_timestamp": observation["timestamp"],
                    "timestamp": observation["timestamp"],
                    "operator_confirmed_physical_continuity": True,
                }))
            if observation["native_track_id"] != previous_native:
                emitted.append(self._emit("OPERATOR_CONFIRMED_NATIVE_FRAGMENT", {
                    "tester": tester,
                    "camera": observation["camera_id"],
                    "previous_native_track_id": previous_native,
                    "native_track_id": observation["native_track_id"],
                    "Person_XX": expected,
                    "frame": observation["frame"],
                }))
            if record.get("full_exit") is not None:
                expected = str(record["canonical_person_id"])
                actual = str(observation["global_person_id"])
                reentry = {
                    "tester": tester,
                    "camera": observation["camera_id"],
                    "frame": observation["frame"],
                    "timestamp": observation["timestamp"],
                    "native_track_id": observation["native_track_id"],
                    "previous_person_id": expected,
                    "reentry_person_id": actual,
                    "absence_duration_seconds": max(0.0, now - float(record["full_exit"].get("last_visible_monotonic", record["full_exit"]["monotonic"]))),
                    "bbox": observation["bbox"],
                    "world": observation["world"],
                    "identity_state": observation["identity_state"],
                    "decision_reason": observation["decision_reason"],
                    "cosine_similarity": observation["cosine_similarity"],
                    "pending_age_frames": observation["pending_age_frames"],
                    "gallery_candidate": observation["gallery_candidate"],
                    "identity_evidence": observation["identity_evidence"],
                    "pending_duration_seconds": max(
                        0.0, now - float(record.get("reentry_pending_since", now))
                    ) if record.get("reentry_pending_since") is not None else 0.0,
                    "time_to_reacquire_seconds": max(0.0, now - float(record["full_exit"]["monotonic"])),
                }
                if actual == "Unknown":
                    record.setdefault("reentry_pending_since", now)
                    record["reentry_pending_observation"] = observation
                    record["reentry_candidate_logged"] = True
                    emitted.append(self._emit("REENTRY_PENDING", {
                        **reentry,
                        "first_reentry_timestamp": observation["timestamp"],
                        "note": "ground-truth selected; awaiting normal production identity resolution",
                    }))
                else:
                    emitted.append(self._emit("REENTRY", reentry))
                if actual == "Unknown":
                    pass
                elif not record.get("absent_10s"):
                    self.failure = self.failure or {"kind": "reentry_before_10s_absence", **reentry}
                    emitted.append(self._emit("REENTRY_TOO_EARLY", {**reentry, "required_absence_seconds": 10}))
                elif actual == expected:
                    record["reentry_done"] = True
                    record["same_person_reacquired"] = True
                    emitted.append(self._emit("REACQUISITION", {**reentry, "same_person_id": True}))
                else:
                    self.failure = self.failure or {"kind": "reacquisition_identity_change", **reentry}
                    emitted.append(self._emit("REACQUISITION_FAILED", {**reentry, "same_person_id": False}))
                if actual != "Unknown":
                    record["full_exit"] = None
                    record["absent_10s"] = False
                    record["exit_requested"] = False
                    record["reentry_candidate_logged"] = False
                    record.pop("reentry_pending_since", None)
                    record.pop("reentry_pending_observation", None)
            record["last_seen_monotonic"] = now
            record["last_seen_timestamp"] = observation["timestamp"] or self._wall_time()
            record["currently_visible"] = True
            record["present_since_monotonic"] = now
            record["stable_observation_count"] = 1
            record["stable_last_frame_by_camera"] = {observation["camera_id"]: observation["frame"]}
            record["seen_cameras"].add(observation["camera_id"])
            record["current_observation"] = observation

        # Separate label assignments must not collapse onto one canonical identity.
        other = "Tester_B" if tester == "Tester_A" else "Tester_A"
        other_record = self.testers.get(other)
        if other_record and other_record["canonical_person_id"] == observation["global_person_id"]:
            merge = {
                "tester": tester,
                "other_tester": other,
                "Person_XX": observation["global_person_id"],
                "camera": observation["camera_id"],
                "frame": observation["frame"],
                "timestamp": observation["timestamp"],
            }
            if self.subtest == "A" and self.started_monotonic is None:
                emitted.append(self._emit("SAME_CANONICAL_ID_AWAITING_OPERATOR_CONFIRMATION", merge))
            else:
                self.failure = self.failure or {"kind": "false_merge", **merge}
                emitted.append(self._emit("FALSE_MERGE", merge))
        self.latest_observations[tester] = observation
        return emitted

    def record_identity_candidate_review(
        self,
        tester: str,
        row: dict[str, Any],
        operator_confirms_same_person: bool | None,
        candidate_crop_path: str | None,
    ) -> dict[str, Any]:
        """Record human review of a changed canonical ID without affecting production.

        A YES is only evidence for the acceptance audit. The caller may then
        submit the observation to ``assign`` so a confirmed ID change is
        classified by this harness. A NO never replaces the tester anchor.
        """
        if not self.enabled:
            raise RuntimeError("acceptance mode is disabled")
        if tester not in self.testers:
            raise RuntimeError("assign the tester before reviewing a candidate identity")
        observation = self.validate_observation(row)
        record = self.testers[tester]
        return self._emit("IDENTITY_CANDIDATE_REVIEWED", {
            "tester": tester,
            "reference_person_id": record["canonical_person_id"],
            "candidate_person_id": observation["global_person_id"],
            "operator_confirms_same_person": operator_confirms_same_person,
            "camera": observation["camera_id"],
            "frame": observation["frame"],
            "timestamp": observation["timestamp"],
            "native_track_id": observation["native_track_id"],
            "bbox": observation["bbox"],
            "world": observation["world"],
            "reference_crop": record.get("reference_crop"),
            "candidate_crop": candidate_crop_path,
            "ground_truth_only": True,
        })

    def begin_exit_test(self, tester: str) -> dict[str, Any]:
        if not self.enabled or self.subtest != "B":
            raise RuntimeError("BEGIN EXIT TEST is available only in acceptance Subtest B")
        record = self.testers.get(tester)
        if not record:
            raise RuntimeError("assign the tester to a visible bbox before beginning the exit test")
        if tester not in TESTERS:
            raise ValueError("tester must be Tester_A or Tester_B")
        now = self._monotonic()
        stable_since = record.get("present_since_monotonic")
        last_seen = float(record.get("last_seen_monotonic", 0.0))
        if not record.get("currently_visible") or now - last_seen > 1.5:
            raise RuntimeError(f"wait for a fresh live observation of {tester} before beginning the exit test")
        if stable_since is None or now - float(stable_since) < self.pre_exit_stability_seconds:
            raise RuntimeError(f"keep {tester} visibly tracked for {self.pre_exit_stability_seconds:.0f}s before beginning the exit test")
        if int(record.get("stable_observation_count", 0)) < self.pre_exit_min_observations:
            raise RuntimeError(
                f"wait for at least {self.pre_exit_min_observations} fresh observations of {tester} before beginning the exit test"
            )
        self.exit_tester = tester
        record["stable_before_exit"] = True
        record["exit_requested"] = True
        record["last_seen_monotonic"] = now
        record["last_seen_timestamp"] = (
            record["current_observation"].get("timestamp") or self._wall_time()
        )
        record["full_exit"] = None
        record["absent_10s"] = False
        record["completed_absence_10s"] = False
        return self._emit("BEGIN_EXIT_TEST", {
            "tester": tester,
            "Person_XX_before_exit": record["canonical_person_id"],
            "last_visible_timestamp": record["last_seen_timestamp"],
            "camera": record["current_observation"]["camera_id"],
            "native_track_id": record["current_observation"]["native_track_id"],
            "source_timestamp": record["current_observation"]["timestamp"],
            "stable_observation_count": record.get("stable_observation_count", 0),
        })

    def can_begin_exit_test(self, tester: str) -> bool:
        """Acceptance-only UI readiness; it does not affect production state."""
        if not self.enabled or self.subtest != "B" or tester not in TESTERS:
            return False
        record = self.testers.get(tester)
        if not record or not record.get("currently_visible"):
            return False
        now = self._monotonic()
        last_seen = float(record.get("last_seen_monotonic", 0.0))
        stable_since = record.get("present_since_monotonic")
        return bool(
            now - last_seen <= 1.5
            and stable_since is not None
            and now - float(stable_since) >= self.pre_exit_stability_seconds
            and int(record.get("stable_observation_count", 0)) >= self.pre_exit_min_observations
        )

    def observe(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.enabled or self.subtest is None:
            return []
        people = []
        for row in snapshot.get("people", []):
            if not isinstance(row, dict) or row.get("camera_id") not in CAMERAS:
                continue
            try:
                normalized = self.validate_observation(row)
            except (TypeError, ValueError, OverflowError):
                continue
            people.append({**row, **normalized, "frame": normalized["frame"], "timestamp": normalized["timestamp"]})
        now = self._monotonic()
        by_frame: dict[str, int] = {}
        by_person_id: dict[str, list[dict[str, Any]]] = {}
        for row in people:
            camera = str(row["camera_id"])
            frame = int(row.get("source_frame_number", row.get("frame", -1)))
            by_frame[camera] = max(frame, by_frame.get(camera, -1))
            identity = _canonical(row)
            by_person_id.setdefault(identity, []).append(row)

        emitted: list[dict[str, Any]] = []
        for tester, record in self.testers.items():
            identity = str(record["canonical_person_id"])
            current = by_person_id.get(identity, [])
            if current:
                # These are system observations only; operator-confirmed physical
                # continuity is still required to change a tester's native track.
                for row in current:
                    camera = str(row["camera_id"])
                    frame = int(row.get("source_frame_number", row.get("frame", -1)))
                    snapshot_key = (tester, camera)
                    if self._last_snapshot_frame.get(snapshot_key) != frame:
                        emitted.append(self._emit("TESTER_OBSERVED", {
                            "tester": tester,
                            "Person_XX": identity,
                            "camera": camera,
                            "native_track_id": row.get("native_track_id"),
                            "frame": frame,
                            "timestamp": row.get("source_timestamp") or row.get("timestamp"),
                            "world": row.get("world"),
                            "bbox": row.get("bbox"),
                            "source_mode": snapshot.get("source_mode"),
                        }))
                        self._last_snapshot_frame[snapshot_key] = frame
                record["last_observed_person_rows"] = current
                if not record.get("currently_visible"):
                    record["present_since_monotonic"] = now
                    record["stable_observation_count"] = 0
                    record["stable_last_frame_by_camera"] = {}
                for row in current:
                    camera = str(row["camera_id"])
                    frame = int(row.get("source_frame_number", row.get("frame", -1)))
                    last_frame = record.setdefault("stable_last_frame_by_camera", {}).get(camera, -1)
                    if frame > last_frame:
                        record["stable_last_frame_by_camera"][camera] = frame
                        record["stable_observation_count"] = int(record.get("stable_observation_count", 0)) + 1
                record["currently_visible"] = True
                if record.get("present_since_monotonic") is None:
                    record["present_since_monotonic"] = now
                # Keep the last-visible clock current while the person remains
                # present. A same-ID return is only confirmed as REENTRY when
                # the operator selects the returning bbox; observations alone
                # cannot establish physical ground truth.
                record["last_seen_monotonic"] = now
                latest_current = max(current, key=lambda row: int(row.get("frame", -1)))
                record["last_seen_timestamp"] = (
                    latest_current.get("source_timestamp")
                    or latest_current.get("timestamp")
                    or self._wall_time()
                )
                # Refresh the stored operator-selected observation only while
                # its exact native track remains the same. A new native track
                # requires another explicit operator selection.
                selected = record.get("current_observation", {})
                same_native = [
                    row for row in current
                    if row.get("camera_id") == selected.get("camera_id")
                    and int(row.get("native_track_id", -1)) == int(selected.get("native_track_id", -2))
                ]
                if same_native:
                    record["current_observation"] = self.validate_observation(
                        max(same_native, key=lambda row: int(row.get("frame", -1)))
                    )
                if record.get("exit_requested") and record.get("full_exit") is not None and not record.get("reentry_candidate_logged"):
                    record["reentry_candidate_logged"] = True
                    emitted.append(self._emit("REENTRY_CANDIDATE", {
                        "tester": tester,
                        "Person_XX_before_exit": identity,
                        "observed_person_id": identity,
                        "cameras": sorted({str(row["camera_id"]) for row in current}),
                        "operator_confirmation_required": True,
                    }))
            elif record.get("exit_requested"):
                record["currently_visible"] = False
                record["present_since_monotonic"] = None
                last_seen = float(record["last_seen_monotonic"])
                absent = max(0.0, now - last_seen)
                if record.get("full_exit") is None and absent >= self.exit_grace_seconds:
                    full_exit = {
                        "tester": tester,
                        "Person_XX_before_exit": identity,
                        "last_visible_timestamp": record.get("last_seen_timestamp"),
                        "last_visible_monotonic": last_seen,
                        "full_exit_timestamp": self._wall_time(),
                        "absence_seconds_at_detection": absent,
                        "cameras_checked": sorted(CAMERAS),
                        "visible_presence_empty": True,
                        "bev_marker_expected_absent": True,
                    }
                    record["full_exit"] = {"monotonic": now, **full_exit}
                    emitted.append(self._emit("FULL_EXIT", full_exit))
                elif record.get("full_exit") is not None and not record.get("absent_10s"):
                    full_exit_mono = float(record["full_exit"]["monotonic"])
                    if now - full_exit_mono >= self.absent_confirmation_seconds:
                        record["absent_10s"] = True
                        record["completed_absence_10s"] = True
                        emitted.append(self._emit("ABSENT_10S", {
                            "tester": tester,
                            "Person_XX_before_exit": identity,
                            "full_exit_timestamp": record["full_exit"]["full_exit_timestamp"],
                            "absent_seconds": now - full_exit_mono,
                            "visible_presence_empty": True,
                        }))
            else:
                record["currently_visible"] = False
                record["present_since_monotonic"] = None
                record["stable_observation_count"] = 0
                record["stable_last_frame_by_camera"] = {}

        if self.started_monotonic is not None:
            emitted.extend(self._check_subtest_a(people, by_person_id, snapshot, now))
        emitted.extend(self._maybe_mark_event_sequence_complete(now))
        return emitted

    def _maybe_mark_event_sequence_complete(self, now: float) -> list[dict[str, Any]]:
        if self.completed_monotonic is not None or self.failure is not None:
            return []
        if self.started_monotonic is None or now - self.started_monotonic < self.minimum_subtest_seconds:
            return []
        if not all(item["done"] for item in self.checklist().get("items", [])):
            return []
        self.completed_monotonic = now
        return [self._emit("EVENT_SEQUENCE_COMPLETE", {
            "subtest": self.subtest,
            "duration_seconds": now - self.started_monotonic,
            "note": "event sequence only; runtime health and visual acceptance are reviewed separately",
        })]

    def _check_subtest_a(self, people, by_person_id, snapshot, now) -> list[dict[str, Any]]:
        if self.subtest != "A":
            return []
        emitted = []
        a, b = self.testers.get("Tester_A"), self.testers.get("Tester_B")
        if a and b and a["canonical_person_id"] == b["canonical_person_id"]:
            self.failure = self.failure or {"kind": "false_merge", "testers": ["Tester_A", "Tester_B"]}
            return [self._emit("FALSE_MERGE", {"testers": ["Tester_A", "Tester_B"], "Person_XX": a["canonical_person_id"]})]
        if not a or not b:
            return emitted
        a_rows = by_person_id.get(str(a["canonical_person_id"]), [])
        b_rows = by_person_id.get(str(b["canonical_person_id"]), [])
        if a_rows and b_rows:
            if not getattr(self, "both_present_recorded", False):
                self.both_present_recorded = True
                emitted.append(self._emit("BOTH_PRESENT", {
                    "Tester_A": a["canonical_person_id"],
                    "Tester_B": b["canonical_person_id"],
                    "frames": {r["camera_id"]: r.get("frame") for r in a_rows + b_rows},
                }))
        distances = []
        # Use only same-camera simultaneous visibility for this conservative
        # acceptance cue; cross-camera stale positions can otherwise create a
        # misleading near-pass event.
        for camera in CAMERAS:
            camera_a = [r.get("world") for r in a_rows if r.get("camera_id") == camera and isinstance(r.get("world"), (list, tuple)) and len(r["world"]) >= 2]
            camera_b = [r.get("world") for r in b_rows if r.get("camera_id") == camera and isinstance(r.get("world"), (list, tuple)) and len(r["world"]) >= 2]
            distances.extend(
                math.dist((float(x[0]), float(x[1])), (float(y[0]), float(y[1])))
                for x in camera_a for y in camera_b
            )
        if distances:
            distance = min(distances)
            if distance <= self.near_pass_start_m and not self.near_pass_active:
                self.near_pass_active = True
                self.near_pass_start_mono = now
                self.near_pass_end_mono = None
                self.post_cross_presence_since = None
                emitted.append(self._emit("NEAR_PASS_START", {
                    "Tester_A": a["canonical_person_id"],
                    "Tester_B": b["canonical_person_id"],
                    "minimum_world_distance_m": distance,
                    "frames": {r["camera_id"]: r.get("frame") for r in a_rows + b_rows},
                }))
            elif distance >= self.near_pass_end_m and self.near_pass_active:
                self.near_pass_active = False
                self.near_pass_end_mono = now
                self.post_cross_presence_since = now if a_rows and b_rows else None
                emitted.append(self._emit("NEAR_PASS_END", {
                    "Tester_A": a["canonical_person_id"],
                    "Tester_B": b["canonical_person_id"],
                    "world_distance_m": distance,
                    "frames": {r["camera_id"]: r.get("frame") for r in a_rows + b_rows},
                }))
        if self.near_pass_end_mono is not None and not self.near_pass_active:
            if not a_rows or not b_rows:
                self.post_cross_presence_since = None
            elif self.post_cross_presence_since is None:
                self.post_cross_presence_since = now
        stable_since = self.post_cross_presence_since
        if stable_since is not None and now - stable_since >= 30.0 and a_rows and b_rows:
            if not getattr(self, "post_cross_stability_recorded", False):
                self.post_cross_stability_recorded = True
                emitted.append(self._emit("POST_CROSS_STABLE_30S", {
                    "Tester_A": a["canonical_person_id"],
                    "Tester_B": b["canonical_person_id"],
                    "duration_seconds": now - stable_since,
                }))
        return emitted

    def checklist(self) -> dict[str, Any]:
        a, b = self.testers.get("Tester_A"), self.testers.get("Tester_B")
        if self.subtest == "A":
            return {
                "subtest": "A",
                "state": "RUNNING" if self.started_monotonic is not None else "WAITING",
                "items": [
                    {"label": "Tester_A selected", "done": bool(a)},
                    {"label": "Tester_B selected", "done": bool(b)},
                    {"label": "operator confirmed different physical people", "done": bool(self.operator_confirmed_distinct_people)},
                    {"label": "timed Subtest A started", "done": self.started_monotonic is not None},
                    {"label": "both people present", "done": bool(self.both_present_recorded)},
                    {"label": "cross-camera continuity", "done": bool((a and a.get("cross_camera_verified")) or (b and b.get("cross_camera_verified")))},
                    {"label": "near-pass/crossing", "done": bool(getattr(self, "near_pass_end_mono", None) is not None)},
                    {"label": "post-cross continuity stable 30s", "done": bool(getattr(self, "post_cross_stability_recorded", False))},
                    {"label": "live run duration >= 2 minutes", "done": bool(self.started_monotonic is not None and self._monotonic() - self.started_monotonic >= self.minimum_subtest_seconds)},
                ],
            }
        tester = self.testers.get(getattr(self, "exit_tester", "Tester_A"))
        exited = bool(tester and tester.get("full_exit"))
        reentry_observed = bool(tester and tester.get("reentry_pending_since") is not None)
        reentered = bool(tester and tester.get("reentry_done"))
        return {
            "subtest": "B",
            "items": [
                {"label": f"{getattr(self, 'exit_tester', 'Tester_A')} stable before exit", "done": bool(tester and tester.get("stable_before_exit"))},
                {"label": "full exit detected", "done": exited or reentered or reentry_observed},
                {"label": "10+ seconds absent", "done": bool(tester and tester.get("completed_absence_10s"))},
                {"label": "reentry detected", "done": reentered or reentry_observed},
                {"label": "same Person_XX reacquired", "done": bool(tester and tester.get("same_person_reacquired"))},
                {"label": "live run duration >= 2 minutes", "done": bool(self.started_monotonic is not None and self._monotonic() - self.started_monotonic >= self.minimum_subtest_seconds)},
            ],
        }

    def next_action(self) -> str:
        if not self.enabled or self.subtest is None:
            return "Enable acceptance mode and start Subtest A or B"
        if self.subtest == "A":
            a, b = self.testers.get("Tester_A"), self.testers.get("Tester_B")
            if self.failure:
                return f"ACCEPTANCE FAILURE: {self.failure.get('kind', 'unknown')}"
            if not a:
                return "WAITING FOR: select a visible physical bbox as Tester_A"
            if not b:
                return "WAITING FOR: select a DIFFERENT visible physical bbox as Tester_B"
            if self.started_monotonic is None:
                return "Confirm Tester_A and Tester_B are different physical people, then start the timed Subtest A"
            if not self.both_present_recorded:
                return "NEXT ACTION: keep both testers visible together briefly"
            if not a.get("cross_camera_verified") and not b.get("cross_camera_verified"):
                return "NEXT ACTION: operator-confirm one tester in the other camera"
            if not getattr(self, "near_pass_end_mono", None):
                return "NEXT ACTION: Tester_A and Tester_B pass close to / cross each other"
            if not getattr(self, "post_cross_stability_recorded", False):
                return "Continue both people in view for 30 seconds after crossing"
            if self.started_monotonic is not None and self._monotonic() - self.started_monotonic < self.minimum_subtest_seconds:
                return "Continue the live subtest until at least 2 minutes total"
            return "Subtest A evidence complete"
        tester_name = getattr(self, "exit_tester", "Tester_A")
        record = self.testers.get(tester_name)
        if not record:
            return f"Select a visible bbox and assign {tester_name}"
        if record.get("reentry_done"):
            if self.started_monotonic is not None and self._monotonic() - self.started_monotonic < self.minimum_subtest_seconds:
                return "Continue the live subtest until at least 2 minutes total"
            return "Subtest B event evidence complete"
        if not record.get("exit_requested"):
            return f"Press BEGIN EXIT TEST, then have {tester_name} leave both camera views"
        if not record.get("full_exit"):
            return f"NEXT ACTION: {tester_name} leave both camera views"
        if record.get("reentry_pending_since") is not None:
            return f"Waiting for normal production identity resolution for {tester_name}'s returning track"
        if record.get("reentry_candidate_logged"):
            return f"NEXT ACTION: select {tester_name}'s returning bbox to confirm the previous Person_XX"
        if not record.get("absent_10s"):
            return f"Keep {tester_name} absent for at least 10 seconds"
        if record.get("same_person_reacquired") and self.started_monotonic is not None and self._monotonic() - self.started_monotonic < self.minimum_subtest_seconds:
            return "Continue the live subtest until at least 2 minutes total"
        return f"NEXT ACTION: {tester_name} re-enter; select their bbox to confirm reacquisition"
