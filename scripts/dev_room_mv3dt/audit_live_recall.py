#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CAMERAS = ("CAM-01", "CAM-04")
EXPECTED_EXPERIMENT = {
    "experiment": "recall-003",
    "production_profile_modified": False,
    "pre_cluster_threshold": 0.03,
    "tentative_detector_confidence": 0.03,
    "data_associator_min_matching_score": 0.2,
}
SOURCE_FPS = 20.0
MIN_SOURCE_FPS = 19.0
MIN_PGIE_TO_TRACKER_COUNT_COVERAGE = 0.95
MAX_DEFICIT_FRAMES = 10  # 0.5 s at 20 FPS


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def person_count(row: dict) -> int:
    total = 0
    for obj in row.get("objects") or []:
        if int(obj.get("class_id", -1)) == 0:
            total += 1
    return total


def longest_true_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


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
        stage_rows[stage][camera][frame] = person_count(row)

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
        max_deficit_frames = longest_true_run(deficits)
        max_deficit_sec = max_deficit_frames / SOURCE_FPS
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
            },
            "pass": pass_camera,
        }

    lifecycle = runtime.get("identity_lifecycle") or {}
    crop = runtime.get("crop_delivery") or {}
    osnet = runtime.get("osnet") or {}
    identity_safe = (
        int(lifecycle.get("pending_to_new_confirmed", 0)) == 0
        and int(lifecycle.get("positive_novelty_confirmations", 0)) == 0
        and int(crop.get("crop_failures", 0)) == 0
        and bool((osnet.get("embedder") or {}).get("ready"))
        and (osnet.get("embedder") or {}).get("device") == "cuda"
    )
    experiment_ok = all(exp.get(k) == v for k, v in EXPECTED_EXPERIMENT.items())
    readiness_ok = bool(readiness.get("ready")) and readiness.get("status") == "ready"

    automatic_pass = experiment_ok and readiness_ok and all(camera_passes) and identity_safe

    return {
        "automatic_pass": automatic_pass,
        "experiment_ok": experiment_ok,
        "readiness_ok": readiness_ok,
        "identity_safety_ok": identity_safe,
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
