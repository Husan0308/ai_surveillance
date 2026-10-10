"""Offline audit of source-stall recovery events and continuous health samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.run_six_camera_v13_wall_candidate import CAMERAS

ROOT = Path(__file__).resolve().parents[1]
PLAN = [("CAM-02", 30), ("CAM-02", 110), ("CAM-02", 190), ("CAM-03", 270)]
EVENT_NAMES = ("TEST_SOURCE_STALL_BEGIN", "SOURCE_STALL_RECOVERY_BEGIN",
               "SOURCE_STALL_OLD_OWNER_REMOVED", "SOURCE_STALL_RECOVERY_RECONNECTED",
               "SOURCE_STALL_RECOVERY_PASS")
COUNTERS = ("decoded_frames_total", "analytics_input_frames_total", "pgie_frames_total",
            "canonical_tracker_updates_total", "preview_frames_total")


def parse_detail(detail: str) -> dict[str, str]:
    result = {}
    for token in detail.split():
        key, separator, value = token.partition("=")
        if separator:
            result[key] = value.rstrip(";,.")
    return result


def read_native_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(errors="strict").splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) >= 2 and fields[1] in EVENT_NAMES:
            events.append({"camera_id": fields[0], "event": fields[1],
                           "detail": fields[2] if len(fields) > 2 else "",
                           "values": parse_detail(fields[2] if len(fields) > 2 else "")})
    return events


def read_observations(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def closest_sample(observations: list[dict], camera: str, event: dict) -> int:
    values = event["values"]
    targets = {
        "decoded_frames_total": int(values["decoded_frames"]),
        "analytics_input_frames_total": int(values.get("input_frames", values["decoded_frames"])),
        "pgie_frames_total": int(values["pgie_frames"]),
        "canonical_tracker_updates_total": int(values["tracker_updates"]),
        "preview_frames_total": int(values["preview_frames"]),
    }
    indices = [index for index, sample in enumerate(observations)
               if sample["ready"] and camera in sample["cameras"]]
    if not indices:
        raise RuntimeError("no ready monitoring samples")
    return min(indices, key=lambda index: sum(
        abs(int(observations[index]["cameras"][camera][key]) - value)
        for key, value in targets.items()))


def event_groups(events: list[dict]) -> list[list[dict]]:
    groups = []
    index = 0
    for target, _second in PLAN:
        while index < len(events) and not (
            events[index]["event"] == "TEST_SOURCE_STALL_BEGIN"
            and events[index]["camera_id"] == target
        ):
            index += 1
        group = []
        for expected in EVENT_NAMES:
            if index >= len(events):
                break
            group.append(events[index])
            index += 1
        groups.append(group)
    return groups


def command_count(cmd: list[str]) -> int:
    return sum(bool(line.strip()) for line in subprocess.run(
        cmd, check=True, capture_output=True, text=True).stdout.splitlines())


def current_cleanup() -> dict:
    rtsp = command_count(["ss", "-Htnp", "state", "established", "( dport = :554 )"])
    # Parenthesized ss filters are parsed by ss, not the shell.
    api = command_count(["ss", "-Hlnpt", "( sport = :8100 )"])
    ml = command_count(["ss", "-Hlnpt", "( sport = :8101 )"])
    amc = command_count(["ss", "-Hlnpt", "( sport = :8000 )"])
    containers = subprocess.run(["docker", "ps", "--filter", "name=surv-sixcam-analytics",
                                 "--format", "{{.ID}}\t{{.Names}}"], check=True,
                                capture_output=True, text=True).stdout.splitlines()
    return {"rtsp_sockets": rtsp, "api_8100_listeners": api, "ml_8101_listeners": ml,
            "amc_8000_listeners": amc, "candidate_containers": containers}


def audit(run_dir: Path) -> dict:
    stack = run_dir / "stack"
    native_log = stack / "analytics" / "native.log"
    events = read_native_events(native_log)
    observations = read_observations(run_dir / "observations.jsonl")
    groups = event_groups(events)
    cycles = []
    expected_recovery_by_camera: dict[str, int] = {}
    for (camera, at_sec), group in zip(PLAN, groups):
        expected_recovery_by_camera[camera] = expected_recovery_by_camera.get(camera, 0) + 1
        ordered = len(group) == len(EVENT_NAMES) and [row["event"] for row in group] == list(EVENT_NAMES)
        same_camera = ordered and all(row["camera_id"] == camera for row in group)
        cycle = {"camera_id": camera, "scheduled_second": at_sec, "events": group,
                 "event_order_pass": bool(ordered and same_camera), "passed": False}
        if not cycle["event_order_pass"]:
            cycles.append(cycle)
            continue
        begin, recovery, removed, reconnected, passed = group
        begin_values, recovery_values = begin["values"], recovery["values"]
        removed_values, reconnected_values, pass_values = removed["values"], reconnected["values"], passed["values"]
        expected_count = expected_recovery_by_camera[camera]
        expected_generation = int(recovery_values["generation"]) + 1
        ordering_proven = (
            int(recovery_values["owner_count_before"]) == 1
            and int(removed_values["owner_count"]) == 0
            and int(reconnected_values["owner_count"]) == 1
            and int(reconnected_values["generation"]) == expected_generation
            and int(pass_values["generation"]) == expected_generation
            and int(pass_values["recovery_count"]) == expected_count
            and float(recovery_values["input_age_ms"]) >= 2500.0
            and int(recovery_values["queue_depth"]) == 0
        )
        before_index = closest_sample(observations, camera, begin)
        after_index = closest_sample(observations, camera, passed)
        before_sample = observations[before_index]
        after_sample = observations[after_index]
        target_before = before_sample["cameras"][camera]
        target_after = after_sample["cameras"][camera]
        target_progress = all(target_after[key] > target_before[key] for key in COUNTERS)
        preview_before, preview_after = target_before.get("preview"), target_after.get("preview")
        preview_recovered = bool(
            preview_after and target_after["image_bbox_sync"]
            and preview_after["embedded_metadata_matches"] and preview_after["age_ms"] <= 1000
            and preview_after["sequence"] > (preview_before or {}).get("sequence", 0)
            and preview_after["source_frame_num"] > (preview_before or {}).get("source_frame_num", -1)
        )
        peers = {}
        for peer in CAMERAS:
            if peer == camera:
                continue
            left, right = before_sample["cameras"][peer], after_sample["cameras"][peer]
            peers[peer] = all(right[key] > left[key] for key in COUNTERS)
        cycle.update({
            "ordering_and_single_owner_proven": ordering_proven,
            "event_counter_before": {key: begin_values.get(key) for key in
                ("decoded_frames", "input_frames", "pgie_frames", "tracker_updates", "preview_frames")},
            "event_counter_after": {key: pass_values.get(key) for key in
                ("decoded_frames", "input_frames", "pgie_frames", "tracker_updates", "preview_frames")},
            "target_progress_all_stages": target_progress,
            "target_preview_fresh_and_synced": preview_recovered,
            "unaffected_camera_progress": peers,
            "socket_max_during_run": max((sample["socket_count"] for sample in
                observations[before_index:after_index + 1]), default=0),
            "selected_observation_indices": [before_index, after_index],
        })
        cycle["passed"] = bool(ordering_proven and target_progress and preview_recovered
                                and all(peers.values()) and cycle["socket_max_during_run"] <= 6
                                and after_sample["socket_count"] == 6)
        cycles.append(cycle)

    ready = (stack / "ready.json").is_file()
    shutdown = json.loads((stack / "shutdown.json").read_text())
    clean = current_cleanup()
    preflight = json.loads((run_dir / "preflight.json").read_text())
    protected_after = json.loads((run_dir / "protected-after.json").read_text())
    protected_before = json.loads((run_dir / "protected-before.json").read_text())
    ready_samples = [sample for sample in observations if sample["ready"]]
    max_sockets = max((sample["socket_count"] for sample in observations), default=0)
    max_queue = max((row["queue_depth"] for sample in observations
                     for row in sample["cameras"].values()), default=-1)
    errors = sum(row["pipeline_errors"] for sample in observations for row in sample["cameras"].values())
    final = ready_samples[-1]["cameras"] if ready_samples else {}
    expected_reconnects = {camera: (3 if camera == "CAM-02" else 1 if camera == "CAM-03" else 0)
                           for camera in CAMERAS}
    reconnect_counts = {camera: int(final.get(camera, {}).get("source_stall_recoveries",
        final.get(camera, {}).get("reconnects", -1))) for camera in CAMERAS}
    single_container = {tuple(row) for sample in observations for row in sample["containers"]}
    no_pipeline_restart = len(single_container) == 1 and all(
        len(sample["containers"]) == 1 for sample in ready_samples)
    result = {
        "status": "PENDING",
        "run_id": run_dir.name,
        "candidate_binary_sha256": json.loads((run_dir / "stack" / "launch.json").read_text())["analytics_binary_sha256"],
        "cycles": cycles,
        "four_cycles_pass": len(cycles) == 4 and all(cycle.get("passed") for cycle in cycles),
        "rtsp_socket_max": max_sockets,
        "rtsp_socket_count_after_shutdown": clean["rtsp_sockets"],
        "candidate_container_count_after_shutdown": len(clean["candidate_containers"]),
        "single_pipeline_container_no_restart": no_pipeline_restart,
        "all_six_reconnect_counts_exact": reconnect_counts == expected_reconnects,
        "reconnect_counts": reconnect_counts,
        "pipeline_errors_total": errors,
        "queue_depth_max": max_queue,
        "bounded_queue": 0 <= max_queue <= 12,
        "all_six_last_preview_synced_fresh_and_normal_rate": bool(final) and all(
            final[camera]["image_bbox_sync"] and final[camera]["preview"]
            and final[camera]["preview"]["age_ms"] <= 1000 and final[camera]["source_fps"] >= 15
            for camera in CAMERAS),
        "startup_ready": ready,
        "shutdown": shutdown,
        "cleanup": clean,
        "ports_released": clean["api_8100_listeners"] == 0 and clean["ml_8101_listeners"] == 0,
        "amc_8000_untouched": clean["amc_8000_listeners"] == preflight["service_listeners"]["amc_8000"],
        "protected_hashes_unchanged": protected_before == protected_after,
        "forced_kill": shutdown.get("forced_kill"),
        "test_harness_wrapper_exit": json.loads((run_dir / "cycle-results.json").read_text()).get("status"),
    }
    passed = all((result["four_cycles_pass"], result["rtsp_socket_max"] <= 6,
                  result["rtsp_socket_count_after_shutdown"] == 0,
                  result["candidate_container_count_after_shutdown"] == 0,
                  result["single_pipeline_container_no_restart"],
                  result["all_six_reconnect_counts_exact"], result["pipeline_errors_total"] == 0,
                  result["bounded_queue"], result["all_six_last_preview_synced_fresh_and_normal_rate"],
                  result["startup_ready"], shutdown.get("status") == "PASS",
                  not result["forced_kill"], result["ports_released"], result["amc_8000_untouched"],
                  result["protected_hashes_unchanged"]))
    result["status"] = "PASS" if passed else "FAIL"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    result = audit(run_dir)
    output = args.output or run_dir / "recovery-audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cycles"}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
