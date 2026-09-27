#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

CAMERAS = ("CAM-01", "CAM-04")
EXPECTED_EXPERIMENTS = {
    "recall-003": {
        "experiment": "recall-003",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
    },
    "recall-004": {
        "experiment": "recall-004",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
        "min_iou_diff_new_target": 0.5,
    },
    "recall-005": {
        "experiment": "recall-005",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
        "min_iou_diff_new_target": 0.5,
        "early_termination_age": 2,
    },
    "recall-006": {
        "experiment": "recall-006",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
        "min_iou_diff_new_target": 0.5,
        "early_termination_age": 2,
        "probation_age": 3,
    },
    "recall-007": {
        "experiment": "recall-007",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
        "min_iou_diff_new_target": 0.3,
        "early_termination_age": 2,
        "probation_age": 3,
    },
    "recall-008": {
        "experiment": "recall-008",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
        "min_iou_diff_new_target": 0.5,
        "early_termination_age": 2,
        "probation_age": 2,
    },
    "recall-009": {
        "experiment": "recall-009",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
        "min_iou_diff_new_target": 0.5,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
    },
    "recall-010": {
        "experiment": "recall-010",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
        "min_iou_diff_new_target": 0.5,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.045,
    },
    "recall-011": {
        "experiment": "recall-011",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
        "min_iou_diff_new_target": 0.5,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.035,
    },
    "recall-012": {
        "experiment": "recall-012",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
        "min_iou_diff_new_target": 0.45,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
    },
    "recall-013": {
        "experiment": "recall-013",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
        "min_iou_diff_new_target": 0.46,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
    },
    "recall-014": {
        "experiment": "recall-014",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
        "min_iou_diff_new_target": 0.46,
        "early_termination_age": 2,
        "probation_age": 3,
        "min_tracker_confidence": 0.03,
    },
    "recall-015": {
        "experiment": "recall-015",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.25,
        "min_iou_diff_new_target": 0.46,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
    },
    "recall-016": {
        "experiment": "recall-016",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.15,
        "min_iou_diff_new_target": 0.46,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
    },
    "recall-017": {
        "experiment": "recall-017",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
        "min_iou_diff_new_target": 0.46,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.02,
    },
    "recall-018": {
        "experiment": "recall-018",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
        "min_iou_diff_new_target": 0.46,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
        "min_detector_confidence": 0.04,
    },
    "recall-019": {
        "experiment": "recall-019",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
        "min_iou_diff_new_target": 0.46,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
        "min_detector_confidence": 0.036,
    },
    "recall-020": {
        "experiment": "recall-020",
        "production_profile_modified": False,
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.2,
        "min_iou_diff_new_target": 0.47,
        "early_termination_age": 2,
        "probation_age": 2,
        "min_tracker_confidence": 0.03,
    },
}
SOURCE_FPS = 20.0
MIN_SOURCE_FPS = 19.0
MIN_PGIE_TO_TRACKER_COUNT_COVERAGE = 0.95
MAX_DEFICIT_FRAMES = 10  # 0.5 s at 20 FPS

RECALL_020_PRODUCTION_VALUES = {
    "pre_cluster_threshold": 0.03,
    "tentative_detector_confidence": 0.03,
    "data_associator_min_matching_score": 0.20,
    "min_iou_diff_new_target": 0.47,
    "min_tracker_confidence": 0.03,
    "probation_age": 2,
    "early_termination_age": 2,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def person_objects(row: dict) -> list[dict]:
    return [obj for obj in (row.get("objects") or []) if int(obj.get("class_id", -1)) == 0]


def person_count(row: dict) -> int:
    return len(person_objects(row))


def contiguous_true_runs(frames: list[int], values: list[bool]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start = previous = None
    for frame, value in zip(frames, values):
        if value:
            if start is None or previous is None or frame != previous + 1:
                if start is not None and previous is not None:
                    runs.append((start, previous, previous - start + 1))
                start = frame
            previous = frame
        elif start is not None and previous is not None:
            runs.append((start, previous, previous - start + 1))
            start = previous = None
    if start is not None and previous is not None:
        runs.append((start, previous, previous - start + 1))
    return runs


def _text_section(text: str, name: str, opener: str, closer: str) -> str | None:
    match = re.search(rf"(?ms)^{re.escape(opener)}{re.escape(name)}{re.escape(closer)}\n(.*?)(?=^{re.escape(opener)}[^\n]+{re.escape(closer)}\s*$|\Z)", text)
    return match.group(1) if match else None


def _section_number(section: str | None, key: str, line_prefix: str = "") -> float | None:
    if section is None:
        return None
    match = re.search(rf"(?m)^\s*{re.escape(line_prefix)}{re.escape(key)}\s*[:=]\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$", section)
    return float(match.group(1)) if match else None


def production_recall_values(run: Path) -> dict | None:
    """Read recall settings from the staged production config copies.

    In particular, read minMatchingScore4Overall only from DataAssociator;
    TrajectoryManagement contains a separate key with the same name.
    """
    pgie_path = run / "run" / "config_pgie.txt"
    tracker_path = run / "run" / "config_tracker.yml"
    if not pgie_path.is_file() or not tracker_path.is_file():
        return None
    pgie = pgie_path.read_text()
    tracker = tracker_path.read_text()
    class_attrs = _text_section(pgie, "class-attrs-0", "[", "]")
    data_associator = _text_section(tracker, "DataAssociator", "", ":")
    target_management = _text_section(tracker, "TargetManagement", "", ":")
    values = {
        "pre_cluster_threshold": _section_number(class_attrs, "pre-cluster-threshold"),
        "tentative_detector_confidence": _section_number(data_associator, "tentativeDetectorConfidence"),
        "data_associator_min_matching_score": _section_number(data_associator, "minMatchingScore4Overall"),
        "min_iou_diff_new_target": _section_number(target_management, "minIouDiff4NewTarget"),
        "min_tracker_confidence": _section_number(target_management, "minTrackerConfidence"),
        "probation_age": _section_number(target_management, "probationAge"),
        "early_termination_age": _section_number(target_management, "earlyTerminationAge"),
    }
    if any(value is None for value in values.values()):
        return values
    return values


def _recall020_values_match(actual: dict | None) -> bool:
    if actual is None or any(value is None for value in actual.values()):
        return False
    return all(
        abs(float(actual[key]) - float(expected)) <= 1e-8
        for key, expected in RECALL_020_PRODUCTION_VALUES.items()
    )


def _pts_span_fps(points: dict[int, int]) -> float | None:
    """Measure replay cadence from unique frame PTS, excluding process startup/drain."""
    valid = [(int(frame), int(pts)) for frame, pts in points.items() if pts is not None]
    valid.sort()
    if len(valid) < 2:
        return None
    _, first_pts = valid[0]
    _, last_pts = valid[-1]
    elapsed_seconds = (last_pts - first_pts) / 1_000_000_000.0
    frame_intervals = len(valid) - 1
    if elapsed_seconds <= 0 or frame_intervals <= 0:
        return None
    return frame_intervals / elapsed_seconds


def audit(run: Path) -> dict:
    exp_path = run / "experiment_overrides.json"
    report_path = run / "identity-live" / "runtime_report.json"
    ready_path = run / "run" / "logs" / "probe" / "readiness.json"
    audit_path = run / "run" / "logs" / "probe" / "frame_path_audit.jsonl"

    required = [report_path, ready_path, audit_path]
    if exp_path.exists():
        required.append(exp_path)
    else:
        required.extend((run / "run" / "config_pgie.txt", run / "run" / "config_tracker.yml"))
    missing = [str(p.relative_to(run)) for p in required if not p.exists()]
    if missing:
        return {"pass": False, "reason": "missing required evidence", "missing": missing}

    exp = load_json(exp_path) if exp_path.exists() else {}
    runtime = load_json(report_path)
    readiness = load_json(ready_path)

    stage_rows: dict[str, dict[str, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    stage_objects: dict[str, dict[str, dict[int, list[dict]]]] = defaultdict(lambda: defaultdict(dict))
    stage_pts: dict[str, dict[str, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    malformed = 0
    duplicate_frame_rows = 0
    for line in audit_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if row.get("record") != "frame":
            continue
        stage = str(row.get("stage", ""))
        camera = str(row.get("mapped_camera_id", ""))
        if camera not in CAMERAS:
            continue
        frame = int(row.get("frame_num", -1))
        if frame < 0:
            continue
        objects = person_objects(row)
        if frame in stage_rows[stage][camera]:
            # Frame rows are latest-state observations, not an event stream.
            # Keep one row per stage/camera/frame so stale duplicate emissions
            # cannot inflate person-instance totals.
            duplicate_frame_rows += 1
        stage_rows[stage][camera][frame] = len(objects)
        stage_objects[stage][camera][frame] = objects
        try:
            if row.get("pts") is not None:
                stage_pts[stage][camera][frame] = int(row["pts"])
        except (TypeError, ValueError):
            pass

    elapsed = float(runtime.get("elapsed_seconds") or 0.0)
    source_mode_path = run / "source_mode.json"
    source_mode = load_json(source_mode_path).get("source_mode") if source_mode_path.exists() else None
    replay_pts_fps = source_mode == "replay"
    source_by_id = {int(row["source_id"]): row for row in readiness.get("sources", [])}
    camera_source = {"CAM-01": 0, "CAM-04": 1}

    cameras = {}
    camera_passes = []
    for camera in CAMERAS:
        health = source_by_id.get(camera_source[camera], {})
        mux_frames = int(health.get("mux_frames", 0))
        pgie_frames_health = int(health.get("pgie_frames", 0))
        tracker_frames_health = int(health.get("tracker_frames", 0))
        if replay_pts_fps:
            # Replay process elapsed time includes pipeline/model startup and
            # sidecar drain. Derive media cadence from the audited frame PTS
            # span instead; the acceptance threshold itself is unchanged.
            mux_fps = _pts_span_fps(stage_pts.get("nvstreammux", {}).get(camera, {}))
            pgie_fps = _pts_span_fps(stage_pts.get("pgie", {}).get(camera, {}))
            tracker_fps = _pts_span_fps(stage_pts.get("tracker", {}).get(camera, {}))
        else:
            mux_fps = mux_frames / elapsed if elapsed > 0 else 0.0
            pgie_fps = pgie_frames_health / elapsed if elapsed > 0 else 0.0
            tracker_fps = tracker_frames_health / elapsed if elapsed > 0 else 0.0

        pgie = stage_rows.get("pgie", {}).get(camera, {})
        tracker = stage_rows.get("tracker", {}).get(camera, {})
        common_frames = sorted(set(pgie) & set(tracker))
        pgie_person_frames = [f for f in common_frames if pgie[f] > 0]
        pgie_person_instances = sum(pgie[f] for f in pgie_person_frames)
        retained_instances = sum(min(pgie[f], tracker[f]) for f in pgie_person_frames)
        count_coverage = retained_instances / pgie_person_instances if pgie_person_instances else None

        deficits = [pgie[f] > tracker[f] for f in pgie_person_frames]
        deficit_runs = sorted(
            contiguous_true_runs(pgie_person_frames, deficits),
            key=lambda item: (-item[2], item[0]),
        )
        max_deficit_frames = deficit_runs[0][2] if deficit_runs else 0
        max_deficit_sec = max_deficit_frames / SOURCE_FPS

        deficit_details = []
        for start, end, length in deficit_runs[:5]:
            sample_frames = sorted(set([start, min(start + 1, end), (start + end) // 2, max(start, end - 1), end]))
            samples = []
            for frame in sample_frames:
                p_objs = stage_objects.get("pgie", {}).get(camera, {}).get(frame, [])
                t_objs = stage_objects.get("tracker", {}).get(camera, {}).get(frame, [])
                samples.append({
                    "frame": frame,
                    "pgie_count": len(p_objs),
                    "tracker_count": len(t_objs),
                    "pgie_confidences": [round(float(obj.get("confidence", 0.0)), 6) for obj in p_objs],
                    "tracker_detector_confidences": [round(float(obj.get("confidence", 0.0)), 6) for obj in t_objs],
                    "tracker_confidences": [round(float(obj.get("tracker_confidence", 0.0)), 6) for obj in t_objs],
                    "tracker_object_ids": [str(obj.get("object_id")) for obj in t_objs],
                })
            deficit_details.append({
                "start_frame": start,
                "end_frame": end,
                "frames": length,
                "seconds": length / SOURCE_FPS,
                "samples": samples,
            })
        reconnects = int(health.get("reconnect_attempts", 0))

        pass_camera = (
            bool(health.get("mux_seen"))
            and bool(health.get("pgie_seen"))
            and bool(health.get("tracker_seen"))
            and mux_fps >= MIN_SOURCE_FPS
            and pgie_fps >= MIN_SOURCE_FPS
            and tracker_fps >= MIN_SOURCE_FPS
            and reconnects == 0
            and count_coverage is not None
            and count_coverage >= MIN_PGIE_TO_TRACKER_COUNT_COVERAGE
            and max_deficit_frames <= MAX_DEFICIT_FRAMES
        )
        camera_passes.append(pass_camera)
        cameras[camera] = {
            "source_health": {
                "mux_frames": mux_frames,
                "pgie_frames": pgie_frames_health,
                "tracker_frames": tracker_frames_health,
                "mux_fps": mux_fps,
                "pgie_fps": pgie_fps,
                "tracker_fps": tracker_fps,
                "fps_measurement": "frame_audit_pts_span" if replay_pts_fps else "readiness_frames_over_runtime_elapsed",
                "reconnect_attempts": reconnects,
                "person_seen": bool(health.get("person_seen")),
            },
            "pgie_to_tracker": {
                "common_audited_frames": len(common_frames),
                "pgie_person_frames": len(pgie_person_frames),
                "pgie_person_instances": pgie_person_instances,
                "retained_person_instances": retained_instances,
                "count_coverage": count_coverage,
                "max_continuous_deficit_frames": max_deficit_frames,
                "max_continuous_deficit_seconds": max_deficit_sec,
                "top_deficit_intervals": deficit_details,
            },
            "pass": pass_camera,
        }

    lifecycle = runtime.get("identity_lifecycle") or {}
    crop = runtime.get("crop_delivery") or {}
    osnet = runtime.get("osnet") or {}
    identity_safety_checks = {
        "pending_to_new_confirmed_zero": int(lifecycle.get("pending_to_new_confirmed", 0)) == 0,
        "positive_novelty_confirmations_zero": int(lifecycle.get("positive_novelty_confirmations", 0)) == 0,
        "crop_failures_zero": int(crop.get("crop_failures", 0)) == 0,
        "osnet_ready": bool((osnet.get("embedder") or {}).get("ready")),
        "osnet_cuda": (osnet.get("embedder") or {}).get("device") == "cuda",
    }
    identity_safe = all(identity_safety_checks.values())
    experiment_name = exp.get("experiment")
    if experiment_name:
        expected_experiment = EXPECTED_EXPERIMENTS.get(str(experiment_name))
        experiment_ok = bool(expected_experiment) and all(
            exp.get(k) == v for k, v in expected_experiment.items()
        )
        configuration_source = "experiment_overrides"
        production_values = None
    else:
        production_values = production_recall_values(run)
        experiment_ok = (
            exp.get("production_profile_modified", False) is False
            and _recall020_values_match(production_values)
        )
        configuration_source = "production_config"
    readiness_ok = bool(readiness.get("ready")) and readiness.get("status") == "ready"

    recall_pass = experiment_ok and readiness_ok and all(camera_passes)
    automatic_pass = recall_pass

    return {
        "automatic_pass": automatic_pass,
        "recall_pass": recall_pass,
        "experiment_ok": experiment_ok,
        "configuration_source": configuration_source,
        "production_recall_values": production_values,
        "readiness_ok": readiness_ok,
        "identity_safety_ok": identity_safe,
        "identity_safety_scope": "informational here; identity acceptance is a later gate",
        "identity_safety_checks": identity_safety_checks,
        "identity_safety_values": {
            "pending_to_new_confirmed": int(lifecycle.get("pending_to_new_confirmed", 0)),
            "positive_novelty_confirmations": int(lifecycle.get("positive_novelty_confirmations", 0)),
            "crop_failures": int(crop.get("crop_failures", 0)),
            "osnet_device": (osnet.get("embedder") or {}).get("device"),
        },
        "malformed_audit_lines": malformed,
        "duplicate_frame_audit_rows": duplicate_frame_rows,
        "cameras": cameras,
        "thresholds": {
            "min_source_fps": MIN_SOURCE_FPS,
            "min_pgie_to_tracker_count_coverage": MIN_PGIE_TO_TRACKER_COUNT_COVERAGE,
            "max_continuous_deficit_frames": MAX_DEFICIT_FRAMES,
            "max_continuous_deficit_seconds": MAX_DEFICIT_FRAMES / SOURCE_FPS,
        },
        "ground_truth_note": (
            "Automatic PASS proves source health and PGIE-to-NvDCF retention only. "
            "Detector recall against a physically visible human still requires labeled/manual ground truth."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run.resolve())
    print(json.dumps(result, indent=2))
    return 0 if result.get("automatic_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
