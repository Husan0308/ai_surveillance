#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
}
SOURCE_FPS = 20.0
MIN_SOURCE_FPS = 19.0
MIN_PGIE_TO_TRACKER_COUNT_COVERAGE = 0.95
MAX_DEFICIT_FRAMES = 10  # 0.5 s at 20 FPS


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


def audit(run: Path) -> dict:
    exp_path = run / "experiment_overrides.json"
    report_path = run / "identity-live" / "runtime_report.json"
    ready_path = run / "run" / "logs" / "probe" / "readiness.json"
    audit_path = run / "run" / "logs" / "probe" / "frame_path_audit.jsonl"

    missing = [str(p.relative_to(run)) for p in (exp_path, report_path, ready_path, audit_path) if not p.exists()]
    if missing:
        return {"pass": False, "reason": "missing required evidence", "missing": missing}

    exp = load_json(exp_path)
    runtime = load_json(report_path)
    readiness = load_json(ready_path)

    stage_rows: dict[str, dict[str, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    stage_objects: dict[str, dict[str, dict[int, list[dict]]]] = defaultdict(lambda: defaultdict(dict))
    malformed = 0
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
        stage_rows[stage][camera][frame] = len(objects)
        stage_objects[stage][camera][frame] = objects

    elapsed = float(runtime.get("elapsed_seconds") or 0.0)
    source_by_id = {int(row["source_id"]): row for row in readiness.get("sources", [])}
    camera_source = {"CAM-01": 0, "CAM-04": 1}

    cameras = {}
    camera_passes = []
    for camera in CAMERAS:
        health = source_by_id.get(camera_source[camera], {})
        mux_frames = int(health.get("mux_frames", 0))
        pgie_frames_health = int(health.get("pgie_frames", 0))
        tracker_frames_health = int(health.get("tracker_frames", 0))
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
    expected_experiment = EXPECTED_EXPERIMENTS.get(str(exp.get("experiment")))
    experiment_ok = bool(expected_experiment) and all(
        exp.get(k) == v for k, v in expected_experiment.items()
    )
    readiness_ok = bool(readiness.get("ready")) and readiness.get("status") == "ready"

    recall_pass = experiment_ok and readiness_ok and all(camera_passes)
    automatic_pass = recall_pass

    return {
        "automatic_pass": automatic_pass,
        "recall_pass": recall_pass,
        "experiment_ok": experiment_ok,
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
