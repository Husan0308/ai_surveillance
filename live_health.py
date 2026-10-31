"""Read-only per-source liveness guard using official deepstream-app PERF lines.

DeepStream 9.1 prints the recent per-source FPS every configured perf interval,
five seconds in our live config. This is post-pipeline progress evidence, NOT
exact frame-index/PTS continuity or proof of display or exposure timestamps.

No RTSP connection and no native/GPU imports are required by this module.
"""
from __future__ import annotations

import re
import json
import threading
import time

CAMERA_IDS = ("CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06")
# deepstream-app uses **PERF: for one application with multiple sources.
# Each source is printed as recent (cumulative) FPS.
_PERF_LINE = re.compile(r"^\s*\*\*PERF:\s*(.*?)\s*$")
_FPS_PAIR = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)")


class SourceFpsMonitor:
    def __init__(self, *, min_fps: float = 5.0, max_age_sec: float = 15.0,
                 min_consecutive: int = 2) -> None:
        if min_fps <= 0 or max_age_sec <= 0 or min_consecutive < 1:
            raise ValueError("invalid monitoring thresholds")
        self.min_fps = min_fps
        self.max_age_sec = max_age_sec
        self.min_consecutive = min_consecutive
        self._lock = threading.Lock()
        self._seen_at: float | None = None
        self._fps: tuple[float, ...] = ()
        self._streak = 0
        self._samples = 0
        self._native_progress_seen = False
        self._native = {
            camera_id: {
                "received_at": None,
                "event_ns": None,
                "decode_count": None,
                "tracker_count": None,
                "decode_fps": 0.0,
                "tracker_fps": 0.0,
                "stage_age_seconds": None,
                "streak": 0,
            }
            for camera_id in CAMERA_IDS
        }

    @staticmethod
    def _nonnegative_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def _ingest_native_progress(self, line: str, stamp: float) -> bool:
        prefix, separator, payload = line.partition(" ")
        if prefix != "SIX_PROGRESS" or not separator:
            return False
        try:
            sample = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return False
        if not isinstance(sample, dict):
            return False

        source_id = sample.get("source_id")
        if (not self._nonnegative_int(source_id) or source_id >= len(CAMERA_IDS)
                or sample.get("camera_id") != CAMERA_IDS[source_id]
                or sample.get("fatal_metadata") is not False):
            return False
        event_ns = sample.get("monotonic_ns")
        stages = sample.get("stages")
        if (not self._nonnegative_int(event_ns) or event_ns == 0
                or not isinstance(stages, list) or len(stages) < 4):
            return False
        decode, tracker = stages[0], stages[3]
        if not isinstance(decode, dict) or not isinstance(tracker, dict):
            return False
        decode_count, decode_last_ns = decode.get("count"), decode.get("last_ns")
        tracker_count, tracker_last_ns = tracker.get("count"), tracker.get("last_ns")
        if not all(self._nonnegative_int(v) for v in (
                decode_count, decode_last_ns, tracker_count, tracker_last_ns)):
            return False

        # Native monotonic time comes from the same host kernel clock inside
        # the default container time namespace. Rates use deltas of that clock;
        # receipt time is used separately for the supervisor's freshness check.
        stage_age_ns = event_ns - max(decode_last_ns, tracker_last_ns)
        if stage_age_ns < 0:
            return False

        camera_id = CAMERA_IDS[source_id]
        with self._lock:
            self._native_progress_seen = True
            state = self._native[camera_id]
            previous_ns = state["event_ns"]
            previous_decode = state["decode_count"]
            previous_tracker = state["tracker_count"]
            decode_fps = tracker_fps = 0.0
            progressing = False
            if (previous_ns is not None and event_ns > previous_ns
                    and previous_decode is not None and previous_tracker is not None):
                elapsed = (event_ns - previous_ns) / 1_000_000_000.0
                decode_delta = decode_count - previous_decode
                tracker_delta = tracker_count - previous_tracker
                if decode_delta >= 0 and tracker_delta >= 0 and elapsed > 0:
                    decode_fps = decode_delta / elapsed
                    tracker_fps = tracker_delta / elapsed
                    progressing = decode_delta > 0 and tracker_delta > 0

            stage_age_seconds = stage_age_ns / 1_000_000_000.0
            good = (progressing and decode_fps >= self.min_fps
                    and tracker_fps >= self.min_fps
                    and stage_age_seconds <= self.max_age_sec)
            state.update({
                "received_at": stamp,
                "event_ns": event_ns,
                "decode_count": decode_count,
                "tracker_count": tracker_count,
                "decode_fps": decode_fps,
                "tracker_fps": tracker_fps,
                "stage_age_seconds": stage_age_seconds,
                "streak": state["streak"] + 1 if good else 0,
            })
            self._samples += 1
        return True

    def ingest(self, line: str, *, now: float | None = None) -> bool:
        stamp = time.monotonic() if now is None else now
        if line.startswith("SIX_PROGRESS "):
            return self._ingest_native_progress(line, stamp)
        match = _PERF_LINE.match(line)
        if not match:
            return False
        pairs = _FPS_PAIR.findall(match.group(1))
        # Excludes header lines and half-written lines.
        if len(pairs) != len(CAMERA_IDS):
            return False
        latest = tuple(float(value) for value, _average in pairs)
        with self._lock:
            if self._seen_at is None or stamp - self._seen_at > self.max_age_sec:
                self._streak = 0
            self._fps = latest
            self._seen_at = stamp
            self._samples += 1
            if all(value >= self.min_fps for value in latest):
                self._streak += 1
            else:
                self._streak = 0
        return True

    def snapshot(self, *, now: float | None = None) -> dict:
        stamp = time.monotonic() if now is None else now
        with self._lock:
            if self._native_progress_seen:
                ages = {
                    camera_id: (None if state["received_at"] is None else
                                max(0.0, stamp - state["received_at"]))
                    for camera_id, state in self._native.items()
                }
                fresh = all(
                    age is not None and age <= self.max_age_sec
                    and self._native[camera_id]["stage_age_seconds"] is not None
                    and self._native[camera_id]["stage_age_seconds"] <= self.max_age_sec
                    for camera_id, age in ages.items()
                )
                ready = fresh and all(
                    state["streak"] >= self.min_consecutive
                    for state in self._native.values()
                )
                return {
                    "ready": ready,
                    "fresh": fresh,
                    "age_seconds": max((age for age in ages.values() if age is not None),
                                        default=None),
                    "samples": self._samples,
                    "consecutive_good": min(
                        (state["streak"] for state in self._native.values()), default=0),
                    "camera_fps": {
                        camera_id: state["tracker_fps"]
                        for camera_id, state in self._native.items()
                    },
                    "camera_stage_fps": {
                        camera_id: {"decode": state["decode_fps"],
                                    "tracker": state["tracker_fps"]}
                        for camera_id, state in self._native.items()
                    },
                    "camera_stage_age_seconds": {
                        camera_id: state["stage_age_seconds"]
                        for camera_id, state in self._native.items()
                    },
                    # Keep the exact native counter state available to an
                    # opt-in failure diagnostic. These are observations from
                    # the same monotonic progress records used by the guard;
                    # they do not change readiness or failure thresholds.
                    "camera_progress": {
                        camera_id: {
                            "native_monotonic_ns": state["event_ns"],
                            "received_age_seconds": ages[camera_id],
                            "decode_count": state["decode_count"],
                            "tracker_count": state["tracker_count"],
                            "decode_fps": state["decode_fps"],
                            "tracker_fps": state["tracker_fps"],
                            "stage_age_seconds": state["stage_age_seconds"],
                            "consecutive_good": state["streak"],
                        }
                        for camera_id, state in self._native.items()
                    },
                    "source_progress_mode": "native_decode_and_tracker_counters",
                    "guard_thresholds": {
                        "minimum_fps": self.min_fps,
                        "maximum_age_seconds": self.max_age_sec,
                        "required_consecutive_good_samples": self.min_consecutive,
                    },
                    "limited_to": "in-process stage frame counts; not visual display or exact exposure time",
                }
            age = None if self._seen_at is None else max(0.0, stamp - self._seen_at)
            fresh = age is not None and age <= self.max_age_sec
            ready = fresh and self._streak >= self.min_consecutive
            return {
                "ready": ready,
                "fresh": fresh,
                "age_seconds": age,
                "samples": self._samples,
                "consecutive_good": self._streak,
                "camera_fps": dict(zip(CAMERA_IDS, self._fps)),
                "camera_stage_fps": {},
                "source_progress_mode": "deepstream_perf_log",
                "guard_thresholds": {
                    "minimum_fps": self.min_fps,
                    "maximum_age_seconds": self.max_age_sec,
                    "required_consecutive_good_samples": self.min_consecutive,
                },
                "limited_to": "DeepStream app recent FPS; not exact frame PTS",
            }
