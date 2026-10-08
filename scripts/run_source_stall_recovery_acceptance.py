"""Controlled, source-local stall recovery acceptance for the six-camera stack."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from services.camera_v11.tracked_object_ipc_v1 import read_camera_snapshot
from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader
from scripts.run_six_camera_v13_wall_candidate import (
    CAMERAS, has_auxiliary_yolo_profile, parse_source_stall_test_plan,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = "CAM-02@30,CAM-02@110,CAM-02@190,CAM-03@270"
EVENTS = {
    "TEST_SOURCE_STALL_BEGIN",
    "SOURCE_STALL_RECOVERY_BEGIN",
    "SOURCE_STALL_OLD_OWNER_REMOVED",
    "SOURCE_STALL_RECOVERY_RECONNECTED",
    "SOURCE_STALL_RECOVERY_PASS",
}


def recovery_verifier_environment(candidate_manifest: dict) -> dict[str, str]:
    profile = candidate_manifest.get("detector_profile") or {}
    if has_auxiliary_yolo_profile(profile):
        return {
            "FULL_STACK_LOWCONF_VERIFIER": "0",
            "FULL_STACK_ACTIVE_TRACK_VERIFIER": "0",
            "FULL_STACK_YOLO_CONTEXT_VERIFIER": "1",
        }
    if profile.get("name") == "yolo26m-fp16-896-context-verified":
        return {
            "FULL_STACK_LOWCONF_VERIFIER": "0",
            "FULL_STACK_ACTIVE_TRACK_VERIFIER": "1",
            "FULL_STACK_YOLO_CONTEXT_VERIFIER": "0",
        }
    return {
        "FULL_STACK_LOWCONF_VERIFIER": "1",
        "FULL_STACK_ACTIVE_TRACK_VERIFIER": "0",
        "FULL_STACK_YOLO_CONTEXT_VERIFIER": "0",
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def protected_hashes(candidate: Path) -> dict[str, str]:
    manifest = json.loads((candidate / "build-manifest.json").read_text())
    profile = manifest.get("detector_profile") or {}
    paths = {
        "production_yolo_config": ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt",
        "local_tracker_config": ROOT / "config/deepstream/config_tracker_NvDCF_v13_local6.yml",
        "frozen_v13_config": ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml",
        "cam00_calibration": ROOT / "config/mv3dt_dev_room/camInfo/cam_00.yml",
        "cam01_calibration": ROOT / "config/mv3dt_dev_room/camInfo/cam_01.yml",
        "candidate_yolo_engine": candidate / "models/yolo26m_raw_otm_b6_fp16.engine",
        "candidate_yolo_parser": candidate / "models/libyolo26rawperson.so",
        "candidate_binary": candidate / "bin/cam-six-yolo-local6-posttracker",
        "candidate_primary_config": candidate / "config/config_infer_primary_yolo26m_raw_otm.txt",
    }
    if profile.get("name") == "yolo26m-fp16-896-context-verified":
        paths.update({
            "candidate_yolo_engine": candidate / "models/yolo26m_raw_otm_896_b6_fp16.engine",
            "candidate_yolo_parser": candidate / "models/libyolo26rawperson_896.so",
        })
    if has_auxiliary_yolo_profile(profile):
        side = int(profile["aux_input"][0])
        aux_config = profile["aux_config"]
        aux_metamux = profile.get("metamux_config") or f"config_metamux_yolo26m_aux{side}.txt"
        aux_engine = profile.get("aux_engine") or f"yolo26m_raw_otm_{side}_b6_fp16.engine"
        aux_parser = profile.get("aux_parser") or f"libyolo26rawperson_{side}.so"
        paths.update({
            "candidate_aux_yolo_engine": candidate / "models" / aux_engine,
            "candidate_aux_yolo_parser": candidate / "models" / aux_parser,
            "candidate_aux_yolo_config": candidate / "config" / aux_config,
            "candidate_metamux_config": candidate / "config" / aux_metamux,
        })
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items() if path.is_file()}


def established_rtsp_sockets() -> list[str]:
    result = subprocess.run(
        ["ss", "-Htnp", "state", "established", "( dport = :554 )"],
        check=True, capture_output=True, text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def surveillance_listeners() -> dict[str, int]:
    result: dict[str, int] = {}
    for name, port in (("api_8100", 8100), ("ml_8101", 8101), ("amc_8000", 8000)):
        output = subprocess.run(
            ["ss", "-Hlnpt", f"( sport = :{port} )"], check=True,
            capture_output=True, text=True,
        ).stdout
        result[name] = sum(bool(line.strip()) for line in output.splitlines())
    return result


def docker_analytics_containers() -> list[tuple[str, str]]:
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=surv-sixcam-analytics", "--format", "{{.ID}}\t{{.Names}}"],
        check=True, capture_output=True, text=True,
    )
    return [tuple(line.split("\t", 1)) for line in result.stdout.splitlines() if line.strip()]


def parse_native_events(path: Path, start_offset: int) -> tuple[int, list[dict]]:
    if not path.exists():
        return start_offset, []
    with path.open("r", errors="replace") as stream:
        stream.seek(start_offset)
        lines = stream.readlines()
        offset = stream.tell()
    parsed: list[dict] = []
    for line in lines:
        fields = line.strip().split(maxsplit=2)
        if len(fields) < 2 or fields[1] not in EVENTS:
            continue
        parsed.append({"camera_id": fields[0], "event": fields[1],
                       "detail": fields[2] if len(fields) > 2 else ""})
    return offset, parsed


def read_latest_fresh(reader: PreviewFrameReader, *, metadata_only: bool,
                      attempts: int = 4, retry_delay_sec: float = 0.002):
    """Retry an atomic shared-memory read briefly if publication is in flight.

    This is a diagnostic-reader retry only: every accepted sample still comes
    from one committed slot, and V5's embedded source-frame/PTS checks remain
    authoritative.  A persistently invalid or stale slot fails closed.
    """
    if attempts < 1 or retry_delay_sec < 0:
        raise ValueError("preview read attempts must be positive and retry delay non-negative")
    for attempt in range(1, attempts + 1):
        frame = reader.read_latest(max_age_sec=1.0, metadata_only=metadata_only)
        if frame is not None:
            return frame, attempt
        if attempt < attempts and retry_delay_sec:
            time.sleep(retry_delay_sec)
    return None, attempts


def event_detail_int(row: dict, key: str) -> int | None:
    for token in str(row.get("detail", "")).split():
        name, separator, value = token.partition("=")
        if separator and name == key:
            try:
                return int(value)
            except ValueError:
                return None
    return None


def camera_snapshot(camera_id: str, monitor_row: dict, reader: PreviewFrameReader,
                    include_pixels: bool = False) -> dict:
    frame, preview_read_attempts = read_latest_fresh(
        reader, metadata_only=not include_pixels)
    tracker = read_camera_snapshot(camera_id, max_age_sec=1.0)
    preview_info = None
    if frame is not None:
        embedded = frame.tracker_snapshot or {}
        preview_info = {
            "sequence": frame.sequence,
            "source_frame_num": frame.source_frame_num,
            "tracker_frame_num": frame.tracker_frame_num,
            "pts_ns": frame.pts_ns,
            "width": frame.width,
            "height": frame.height,
            "stride": frame.stride,
            "age_ms": max(0.0, (time.monotonic_ns() - frame.timestamp_ns) / 1e6),
            "embedded_metadata_matches": preview_pair_matches(frame),
            "read_attempts": preview_read_attempts,
        }
        if include_pixels:
            payload = frame.payload
            preview_info["payload_sha256"] = hashlib.sha256(payload).hexdigest()
            preview_info["payload_min"] = min(payload) if payload else None
            preview_info["payload_max"] = max(payload) if payload else None
            preview_info["payload_mean"] = (sum(payload) / len(payload)) if payload else None
            preview_info["payload_all_zero"] = not payload or not any(payload)
    # The UI consumes the metadata embedded atomically in this preview slot.
    # A separately-read latest tracker snapshot can legitimately advance just
    # before/after the preview reader, so it is diagnostic rather than the
    # image/bbox synchronization contract.
    independent_snapshot_matches = bool(
        frame is not None and tracker is not None and preview_info and
        frame.source_frame_num == int(tracker.get("source_frame_num", -1)) and
        frame.tracker_frame_num == int(tracker.get("frame_num", -1)) and
        frame.pts_ns == int(tracker.get("pts_ns", -1))
    )
    return {
        "camera_id": camera_id,
        "online": bool(monitor_row.get("online", False)),
        "decoded_frames_total": int(monitor_row.get("decoded_frames", 0)),
        "analytics_input_frames_total": int(monitor_row.get("analytics_input_frames_total", 0)),
        "pgie_frames_total": int(monitor_row.get("pgie_frames_total", 0)),
        "canonical_tracker_updates_total": int(monitor_row.get("canonical_tracker_updates_total", 0)),
        "canonical_frame_num": int(monitor_row.get("canonical_frame_num", -1)),
        "preview_frames_total": int(monitor_row.get("preview_frames_total", 0)),
        "source_fps": float(monitor_row.get("source_fps", 0.0)),
        "infer_hz": float(monitor_row.get("infer_hz", 0.0)),
        "preview_fps": float(monitor_row.get("preview_fps", 0.0)),
        "queue_depth": int(monitor_row.get("queue_depth", -1)),
        "queue_high_water": int(monitor_row.get("queue_high_water", -1)),
        "reconnects": int(monitor_row.get("reconnects", 0)),
        "pipeline_errors": int(monitor_row.get("pipeline_errors", 0)),
        "preview": preview_info,
        "preview_read_attempts": preview_read_attempts,
        "canonical_snapshot": ({
            "frame_num": int(tracker["frame_num"]),
            "source_frame_num": int(tracker["source_frame_num"]),
            "pts_ns": int(tracker["pts_ns"]),
            "timestamp_monotonic_ns": int(tracker["timestamp_monotonic_ns"]),
            "object_count": len(tracker["objects"]),
        } if tracker else None),
        "image_bbox_sync": bool(frame is not None and preview_info and
                                 preview_info["embedded_metadata_matches"]),
        "independent_latest_snapshot_matches_preview": independent_snapshot_matches,
    }


def preview_pair_matches(frame) -> bool:
    """Whether the frame's atomically embedded canonical rows name that frame."""
    embedded = frame.tracker_snapshot or {}
    return (
        int(embedded.get("source_frame_num", -1)) == frame.source_frame_num
        and int(embedded.get("frame_num", -1)) == frame.tracker_frame_num
        and int(embedded.get("pts_ns", -1)) == frame.pts_ns
    )


def all_camera_feeds_ready(camera_rows: dict[str, dict]) -> bool:
    """Require valid live telemetry and a fresh same-frame preview from all six."""
    if set(camera_rows) != set(CAMERAS):
        return False
    for row in camera_rows.values():
        preview = row.get("preview")
        if not (
            row.get("online") is True
            and row.get("source_fps", 0.0) >= 15.0
            and row.get("decoded_frames_total", 0) > 0
            and row.get("pgie_frames_total", 0) > 0
            and row.get("canonical_tracker_updates_total", 0) > 0
            and row.get("queue_depth", -1) >= 0
            and row.get("queue_high_water", -1) >= 0
            and preview
            and preview.get("age_ms", float("inf")) <= 1000.0
            and preview.get("embedded_metadata_matches") is True
        ):
            return False
    return True


def capture_recovered_preview(camera_id: str, reader: PreviewFrameReader) -> dict | None:
    """Capture the actual published pixels immediately after recovery passes."""
    frame, read_attempts = read_latest_fresh(reader, metadata_only=False)
    if frame is None:
        return None
    payload = frame.payload
    return {
        "camera_id": camera_id,
        "read_attempts": read_attempts,
        "sequence": frame.sequence,
        "source_frame_num": frame.source_frame_num,
        "tracker_frame_num": frame.tracker_frame_num,
        "pts_ns": frame.pts_ns,
        "width": frame.width,
        "height": frame.height,
        "age_ms": max(0.0, (time.monotonic_ns() - frame.timestamp_ns) / 1e6),
        "image_bbox_sync": preview_pair_matches(frame),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_all_zero": not payload or not any(payload),
        "payload_min": min(payload) if payload else None,
        "payload_max": max(payload) if payload else None,
        "payload_mean": sum(payload) / len(payload) if payload else None,
    }


def recovery_cycles(events: list[dict], expected_plan: list[dict]) -> list[dict]:
    """Bind each planned test to its own source-local recovery event sequence.

    Recovery events from an unrelated source must never satisfy a planned
    cycle.  Segmenting on TEST_SOURCE_STALL_BEGIN also makes an absent or
    interleaved recovery explicit instead of consuming later events by name.
    """
    cycles: list[dict] = []
    starts = [index for index, event in enumerate(events)
              if event["event"] == "TEST_SOURCE_STALL_BEGIN"]
    cursor = 0
    expected_events = [
        "TEST_SOURCE_STALL_BEGIN", "SOURCE_STALL_RECOVERY_BEGIN",
        "SOURCE_STALL_OLD_OWNER_REMOVED", "SOURCE_STALL_RECOVERY_RECONNECTED",
        "SOURCE_STALL_RECOVERY_PASS",
    ]
    for expected in expected_plan:
        if cursor >= len(starts):
            cycles.append({"camera_id": expected["camera_id"],
                           "expected_at_sec": expected["at_sec"], "events": [],
                           "event_order_pass": False, "passed": False,
                           "reason": "planned source interruption was not observed"})
            continue
        start_index = starts[cursor]
        cursor += 1
        stop_index = starts[cursor] if cursor < len(starts) else len(events)
        segment = events[start_index:stop_index]
        target_rows = [row for row in segment
                       if row["camera_id"] == expected["camera_id"]
                       and row["event"] in expected_events]
        names = [row["event"] for row in target_rows]
        ordered = names == expected_events
        same_camera = all(row["camera_id"] == expected["camera_id"] for row in target_rows)
        interleaved = [row for row in segment[1:]
                       if row["event"] in expected_events[1:]
                       and row not in target_rows]
        cycles.append({"camera_id": expected["camera_id"], "expected_at_sec": expected["at_sec"],
                       "events": target_rows, "interleaved_unplanned_events": interleaved,
                       "event_order_pass": ordered and same_camera,
                       "passed": ordered and same_camera})
    return cycles


def unexpected_recovery_events(events: list[dict], cycles: list[dict]) -> list[dict]:
    """Return recovery lifecycle rows not consumed by a planned cycle."""
    expected_ids = {id(row) for cycle in cycles for row in cycle.get("events", [])}
    lifecycle_names = {
        "SOURCE_STALL_RECOVERY_BEGIN", "SOURCE_STALL_OLD_OWNER_REMOVED",
        "SOURCE_STALL_RECOVERY_RECONNECTED", "SOURCE_STALL_RECOVERY_PASS",
    }
    return [row for row in events
            if row["event"] in lifecycle_names and id(row) not in expected_ids]


def planned_recovery_complete(events: list[dict], expected_plan: list[dict]) -> bool:
    """Only end the live run after every planned interruption has passed."""
    observed_tests = sum(row["event"] == "TEST_SOURCE_STALL_BEGIN" for row in events)
    cycles = recovery_cycles(events, expected_plan)
    return (observed_tests == len(expected_plan) and len(cycles) == len(expected_plan)
            and all(cycle.get("event_order_pass") for cycle in cycles))


def run(output: Path, candidate: Path, duration_limit: int, plan_text: str) -> int:
    output = output.resolve()
    candidate = candidate.resolve()
    if not output.is_relative_to(ROOT / ".runtime") or not candidate.is_relative_to(ROOT / ".runtime"):
        raise ValueError("recovery evidence and candidate must remain under .runtime")
    if output.exists():
        raise FileExistsError(output)
    plan = parse_source_stall_test_plan(plan_text, "1")
    if len(plan) != 4 or [row["camera_id"] for row in plan] != ["CAM-02", "CAM-02", "CAM-02", "CAM-03"]:
        raise ValueError("acceptance schedule must contain three CAM-02 cycles followed by one CAM-03 cycle")
    if duration_limit < plan[-1]["at_sec"] + 30:
        raise ValueError("duration limit must include final recovery and evidence margin")
    candidate_manifest = json.loads((candidate / "build-manifest.json").read_text())
    verifier_environment = recovery_verifier_environment(candidate_manifest)
    output.mkdir(parents=True)
    stack_output = output / "stack"
    env = os.environ.copy()
    env.update({
        "FULL_STACK_OUT": str(stack_output),
        "FULL_STACK_ANALYTICS_CANDIDATE": str(candidate),
        **verifier_environment,
        "V11_SOURCE_STALL_RECOVERY": "1",
        "V11_TEST_SOURCE_STALL_PLAN": plan_text,
        # The local dotenv currently contains legacy 8000/8001 values. The
        # authorized deployment contract is explicit and must preserve AMC.
        "API_HOST": "0.0.0.0",
        "API_PORT": "8100",
        "ML_HOST": "0.0.0.0",
        "ML_PORT": "8101",
        "ML_SERVICE_URL": "http://127.0.0.1:8101",
        "FRONTEND_API_BASE_URL": "http://127.0.0.1:8100",
        "FRONTEND_ML_VIDEO_BASE_URL": "http://127.0.0.1:8101",
    })
    env.pop("FULL_STACK_UI_EVIDENCE", None)
    readers = {camera: PreviewFrameReader(
        f"/dev/shm/v11_ui_preview_{camera.lower().replace('-', '')}_v1.bin") for camera in CAMERAS}
    before = {
        "rtsp_socket_count": len(established_rtsp_sockets()),
        "service_listeners": surveillance_listeners(),
        "candidate_containers": docker_analytics_containers(),
        "authorized_deployment_ports": {"api": 8100, "ml": 8101, "amc_preserved": 8000},
        "protected_hashes": protected_hashes(candidate),
        "candidate_binary_sha256": hashlib.sha256(
            (candidate / "bin/cam-six-yolo-local6-posttracker").read_bytes()).hexdigest(),
        "runtime_flags": {
            **verifier_environment,
            "V11_SOURCE_STALL_RECOVERY": "1",
            "V11_TEST_SOURCE_STALL_PLAN": plan_text,
        },
    }
    if before["rtsp_socket_count"] != 0:
        raise RuntimeError("refusing to start recovery test while any RTSP owner is active")
    if before["service_listeners"]["api_8100"] or before["service_listeners"]["ml_8101"]:
        raise RuntimeError("surveillance API/ML port is occupied before the test")
    write_json(output / "preflight.json", before)
    write_json(output / "protected-before.json", before["protected_hashes"])
    event_path = output / "events.jsonl"
    sample_path = output / "observations.jsonl"
    stack_log = (output / "supervisor.log").open("x")
    proc = subprocess.Popen([sys.executable, "-B", "-u", "-m", "scripts.full_stack_runtime"],
                            cwd=ROOT, env=env, stdout=stack_log, stderr=subprocess.STDOUT)
    native_offset = 0
    events: list[dict] = []
    observations: list[dict] = []
    event_indices: list[dict] = []
    container_ids: set[str] = set()
    max_sockets = 0
    run_start = time.monotonic()
    ready = False
    ready_streak = 0
    last_preview_sequences: dict[str, int] = {}
    stop_sent = False
    all_passed_since: float | None = None
    result = 1
    stop_reason = "not all four recovery cycles completed"
    try:
        with event_path.open("x") as event_out, sample_path.open("x") as sample_out:
            while time.monotonic() - run_start < duration_limit:
                if proc.poll() is not None:
                    stop_reason = f"full-stack supervisor exited early with {proc.returncode}"
                    break
                supervisor_ready = (stack_output / "ready.json").exists()
                # The wrapper log only contains supervisor/runner diagnostics;
                # DeepStream lifecycle events are in the per-run native log.
                native_log = stack_output / "analytics" / "native.log"
                native_offset, new_events = parse_native_events(native_log, native_offset)
                for row in new_events:
                    row["sample_index"] = len(observations)
                    row["observed_monotonic_ns"] = time.monotonic_ns()
                    if row["event"] == "SOURCE_STALL_RECOVERY_PASS":
                        row["post_recovery_pixel_capture"] = capture_recovered_preview(
                            row["camera_id"], readers[row["camera_id"]])
                    events.append(row)
                    event_out.write(json.dumps(row, sort_keys=True) + "\n")
                    event_out.flush()
                    if row["event"] == "SOURCE_STALL_RECOVERY_PASS":
                        stop_reason = "four source-local recoveries completed"
                monitor = {}
                try:
                    monitor = json.loads((stack_output / "analytics" / "monitoring.json").read_text())
                except (OSError, json.JSONDecodeError):
                    pass
                by_camera = {row.get("camera_id"): row for row in monitor.get("cameras", [])}
                camera_rows = {
                    camera: camera_snapshot(camera, by_camera.get(camera, {}), readers[camera])
                    for camera in CAMERAS
                }
                feeds_ready = supervisor_ready and all_camera_feeds_ready(camera_rows)
                feeds_advanced = feeds_ready
                for camera, row in camera_rows.items():
                    preview = row.get("preview")
                    if preview is None:
                        feeds_advanced = False
                        continue
                    previous_sequence = last_preview_sequences.get(camera)
                    if previous_sequence is None or int(preview["sequence"]) <= previous_sequence:
                        feeds_advanced = False
                    last_preview_sequences[camera] = int(preview["sequence"])
                ready_streak = ready_streak + 1 if feeds_advanced else 0
                # Exclude service-ready but not-yet-published startup samples.
                # The feed must be fresh and advance for every camera on two
                # consecutive polls before recovery/queue gates begin.
                ready = ready or ready_streak >= 2
                sockets = established_rtsp_sockets()
                max_sockets = max(max_sockets, len(sockets))
                containers = docker_analytics_containers()
                container_ids.update(container_id for container_id, _ in containers)
                sample = {
                    "sample_index": len(observations),
                    "monotonic_ns": time.monotonic_ns(),
                    "ready": ready,
                    "all_six_fresh_preview_pairs": feeds_ready,
                    "all_six_previews_advanced_this_poll": feeds_advanced,
                    "socket_count": len(sockets),
                    "rtsp_socket_owners": sockets,
                    "containers": containers,
                    "cameras": camera_rows,
                }
                observations.append(sample)
                sample_out.write(json.dumps(sample, sort_keys=True) + "\n")
                sample_out.flush()
                if planned_recovery_complete(events, plan):
                    all_passed_since = all_passed_since or time.monotonic()
                    if time.monotonic() - all_passed_since >= 1.5:
                        stop_sent = True
                        break
                else:
                    all_passed_since = None
                time.sleep(0.25)
        cycles = recovery_cycles(events, plan)
        unexpected_events = unexpected_recovery_events(events, cycles)
        expected_recovery_counts: dict[str, int] = {}
        for cycle_index, cycle in enumerate(cycles):
            rows = cycle["events"]
            expected_recovery_counts[cycle["camera_id"]] = expected_recovery_counts.get(cycle["camera_id"], 0) + 1
            expected_recovery_count = expected_recovery_counts[cycle["camera_id"]]
            if not cycle["event_order_pass"]:
                continue
            begin_index = int(rows[0]["sample_index"])
            pass_index = int(rows[-1]["sample_index"])
            if begin_index >= len(observations) or pass_index >= len(observations):
                cycle["passed"] = False
                cycle["reason"] = "event has no surrounding monitoring samples"
                continue
            before_sample = observations[begin_index]
            after_sample = observations[pass_index]
            target = cycle["camera_id"]
            target_before = before_sample["cameras"][target]
            target_after = after_sample["cameras"][target]
            required_counters = ("decoded_frames_total", "analytics_input_frames_total", "pgie_frames_total",
                                 "canonical_tracker_updates_total", "preview_frames_total")
            target_progress = all(target_after[key] > target_before[key] for key in required_counters)
            peer_progress = {}
            for camera in CAMERAS:
                if camera == target:
                    continue
                left = before_sample["cameras"][camera]
                right = after_sample["cameras"][camera]
                peer_progress[camera] = all(right[key] > left[key] for key in required_counters)
            cycle["target_progress_all_stages"] = target_progress
            cycle["unaffected_camera_progress"] = peer_progress
            cycle["max_socket_count_during_test"] = max(
                (sample["socket_count"] for sample in observations[begin_index:pass_index + 1]), default=0)
            cycle["socket_count_at_pass"] = after_sample["socket_count"]
            cycle["exact_preview_tracker_sync_at_pass"] = target_after["image_bbox_sync"]
            cycle["independent_snapshot_matches_at_pass"] = target_after[
                "independent_latest_snapshot_matches_preview"]
            cycle["zero_pipeline_errors_at_pass"] = target_after["pipeline_errors"] == 0
            cycle["target_reconnect_count_at_pass"] = target_after["reconnects"]
            cycle["expected_recovery_count"] = expected_recovery_count
            cycle["recovery_count_in_native_pass_event"] = event_detail_int(rows[-1], "recovery_count")
            cycle["passed"] = bool(
                cycle["event_order_pass"] and target_progress and all(peer_progress.values())
                and cycle["max_socket_count_during_test"] <= 6 and cycle["socket_count_at_pass"] == 6
                and cycle["exact_preview_tracker_sync_at_pass"] and cycle["zero_pipeline_errors_at_pass"]
                and cycle["target_reconnect_count_at_pass"] == expected_recovery_count
                and cycle["recovery_count_in_native_pass_event"] == expected_recovery_count
            )
        # Each event contains its own pixel capture, sampled immediately when
        # the native recovery-pass event was observed (not reused at run end).
        for cycle in cycles:
            if not cycle.get("events"):
                continue
            target = cycle["camera_id"]
            capture = cycle["events"][-1].get("post_recovery_pixel_capture")
            cycle["post_recovery_pixel_capture"] = capture
            begin_index = int(cycle["events"][0]["sample_index"])
            before_preview = observations[begin_index]["cameras"][target].get("preview")
            if capture and before_preview:
                capture["sequence_advanced_since_stall"] = capture["sequence"] > before_preview["sequence"]
            cycle["passed"] = bool(cycle.get("passed") and capture and capture["image_bbox_sync"]
                                    and capture.get("sequence_advanced_since_stall")
                                    and capture.get("age_ms", 1e9) <= 1000
                                    and not capture["payload_all_zero"])
        if stop_sent and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                stop_reason = "four recoveries passed but graceful shutdown exceeded 90 seconds"
        shutdown = {}
        try:
            shutdown = json.loads((stack_output / "shutdown.json").read_text())
        except (OSError, json.JSONDecodeError):
            pass
        after = {
            "rtsp_socket_count": len(established_rtsp_sockets()),
            "service_listeners": surveillance_listeners(),
            "candidate_containers": docker_analytics_containers(),
            "container_ids_seen": sorted(container_ids),
            "max_rtsp_socket_count": max_sockets,
            "supervisor_return_code": proc.poll(),
            "shutdown": shutdown,
            "protected_hashes": protected_hashes(candidate),
        }
        write_json(output / "postflight.json", after)
        write_json(output / "protected-after.json", after["protected_hashes"])
        final_cameras = observations[-1]["cameras"] if observations else {}
        expected_final_reconnects = {camera: (3 if camera == "CAM-02" else 1 if camera == "CAM-03" else 0)
                                     for camera in CAMERAS}
        final_reconnect_counts_exact = all(
            int(final_cameras.get(camera, {}).get("reconnects", -1)) == expected
            for camera, expected in expected_final_reconnects.items()
        )
        qualified_observations = [sample for sample in observations if sample["ready"]]
        queue_values = [
            (camera, int(row.get("queue_depth", -1)), int(row.get("queue_high_water", -1)))
            for sample in qualified_observations for camera, row in sample["cameras"].items()
        ]
        pipeline_errors_zero = bool(qualified_observations) and all(
            int(row.get("pipeline_errors", 0)) == 0
            for sample in qualified_observations for row in sample["cameras"].values()
        )
        bounded_queues = bool(queue_values) and all(
            0 <= depth <= 12 and 0 <= high_water <= 12
            for _, depth, high_water in queue_values
        )
        sync_checked_pairs = [
            row for sample in qualified_observations for row in sample["cameras"].values()
            if row.get("preview") is not None
        ]
        preview_sync_mismatches = sum(
            not bool(row["preview"].get("embedded_metadata_matches"))
            for row in sync_checked_pairs
        )
        expected_stall_windows = {
            (cycle["camera_id"], index)
            for cycle in cycles if cycle.get("events")
            for index in range(int(cycle["events"][0]["sample_index"]),
                               int(cycle["events"][-1]["sample_index"]) + 1)
        }
        preview_missing_by_camera = {camera: 0 for camera in CAMERAS}
        preview_missing_unexpected_by_camera = {camera: 0 for camera in CAMERAS}
        for sample in qualified_observations:
            for camera, row in sample["cameras"].items():
                if row.get("preview") is None:
                    preview_missing_by_camera[camera] += 1
                    if (camera, int(sample["sample_index"])) not in expected_stall_windows:
                        preview_missing_unexpected_by_camera[camera] += 1
        no_pipeline_restart = len(container_ids) == 1 and all(
            len(sample["containers"]) == 1 for sample in observations if sample["ready"]
        )
        final_all_cameras_current = bool(final_cameras) and all(
            row.get("preview") and row["preview"]["age_ms"] <= 1000 and row["image_bbox_sync"]
            and row["source_fps"] >= 15.0 and row["queue_depth"] <= 12
            for row in final_cameras.values()
        )
        write_json(output / "cycle-results.json", {"status": "PENDING", "stop_reason": stop_reason,
            "plan": plan, "cycles": cycles, "sample_count": len(observations),
            "events": events, "all_cycles_pass": bool(cycles) and all(cycle.get("passed") for cycle in cycles),
            "unexpected_recovery_events": unexpected_events,
            "no_unplanned_source_recoveries": not unexpected_events,
            "shutdown_clean": (shutdown.get("status") == "PASS" and not shutdown.get("forced_kill")
                and not after["rtsp_socket_count"] and not after["candidate_containers"]
                and after["service_listeners"]["api_8100"] == 0 and after["service_listeners"]["ml_8101"] == 0),
            "amc_listener_count_before": before["service_listeners"]["amc_8000"],
            "amc_listener_count_after": after["service_listeners"]["amc_8000"],
            "exact_final_reconnect_counts": final_reconnect_counts_exact,
            "expected_final_reconnect_counts": expected_final_reconnects,
            "pipeline_errors_zero": pipeline_errors_zero,
            "queues_bounded": bounded_queues,
            "queue_measurement_samples": len(queue_values),
            "max_queue_depth_by_camera": {
                camera: max((int(sample["cameras"][camera].get("queue_depth", -1))
                             for sample in qualified_observations), default=-1)
                for camera in CAMERAS
            },
            "max_queue_high_water_by_camera": {
                camera: max((int(sample["cameras"][camera].get("queue_high_water", -1))
                             for sample in qualified_observations), default=-1)
                for camera in CAMERAS
            },
            "preview_image_metadata_pairs_checked": len(sync_checked_pairs),
            "preview_image_metadata_sync_mismatches": preview_sync_mismatches,
            "preview_missing_samples_by_camera": preview_missing_by_camera,
            "unexpected_preview_missing_samples_by_camera": preview_missing_unexpected_by_camera,
            "single_pipeline_container_no_restart": no_pipeline_restart,
            "all_six_final_previews_fresh_synced_and_at_rate": final_all_cameras_current,
            "all_six_final_cameras": final_cameras})
        results = json.loads((output / "cycle-results.json").read_text())
        results["protected_hashes_unchanged"] = before["protected_hashes"] == after["protected_hashes"]
        all_pass = all((results["all_cycles_pass"], results["shutdown_clean"],
                        results["exact_final_reconnect_counts"], results["pipeline_errors_zero"],
                        results["queues_bounded"], results["single_pipeline_container_no_restart"],
                        results["all_six_final_previews_fresh_synced_and_at_rate"],
                        results["preview_image_metadata_sync_mismatches"] == 0,
                        results["no_unplanned_source_recoveries"],
                        all(value == 0 for value in results["unexpected_preview_missing_samples_by_camera"].values()),
                        results["protected_hashes_unchanged"],
                        results["amc_listener_count_before"] == results["amc_listener_count_after"],
                        after["max_rtsp_socket_count"] <= 6, after["supervisor_return_code"] == 0))
        results["status"] = "PASS" if all_pass else "FAIL"
        write_json(output / "cycle-results.json", results)
        return 0 if all_pass else 1
    finally:
        for reader in readers.values():
            reader.close()
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                # Never SIGKILL the native candidate as normal cleanup.
                pass
        stack_log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-limit", type=int, default=360)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    args = parser.parse_args()
    try:
        return run(args.output, args.candidate, args.duration_limit, args.plan)
    except Exception as exc:
        print(f"SOURCE_RECOVERY_ACCEPTANCE_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
