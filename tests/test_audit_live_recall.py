from __future__ import annotations

import json

from scripts.dev_room_mv3dt.audit_live_recall import audit, production_recall_values


PGIE_CONFIG = """[property]
batch-size=2

[class-attrs-0]
pre-cluster-threshold=0.03
"""


def _tracker_config(data_associator_score: float = 0.20) -> str:
    return f"""BaseConfig:
  minDetectorConfidence: 0.027087304322979212
TargetManagement:
  minIouDiff4NewTarget: 0.47
  minTrackerConfidence: 0.03
  probationAge: 2
  maxShadowTrackingAge: 162
  earlyTerminationAge: 2
TrajectoryManagement:
  minMatchingScore4Overall: 0.9349462651721144
DataAssociator:
  minMatchingScore4Overall: {data_associator_score}
  tentativeDetectorConfidence: 0.03
"""


def _write_live_run(root, *, data_associator_score: float = 0.20):
    (root / "run/logs/probe").mkdir(parents=True)
    (root / "identity-live").mkdir()
    (root / "run/config_pgie.txt").write_text(PGIE_CONFIG)
    (root / "run/config_tracker.yml").write_text(_tracker_config(data_associator_score))
    (root / "identity-live/runtime_report.json").write_text(json.dumps({"elapsed_seconds": 100.0}))
    (root / "run/logs/probe/readiness.json").write_text(json.dumps({
        "ready": True,
        "status": "ready",
        "sources": [
            {
                "source_id": source_id,
                "mux_frames": 2000,
                "pgie_frames": 2000,
                "tracker_frames": 2000,
                "mux_seen": True,
                "pgie_seen": True,
                "tracker_seen": True,
                "reconnect_attempts": 0,
            }
            for source_id in (0, 1)
        ],
    }))
    rows = []
    for camera, source_id in (("CAM-01", 0), ("CAM-04", 1)):
        for stage in ("pgie", "tracker"):
            for frame in range(20):
                objects = []
                if frame in (4, 5):
                    objects = [{
                        "class_id": 0,
                        "object_id": "7",
                        "confidence": 0.9,
                        "tracker_confidence": 0.8 if stage == "tracker" else 0.0,
                    }]
                rows.append({
                    "record": "frame",
                    "stage": stage,
                    "frame_num": frame,
                    "source_id": source_id,
                    "mapped_camera_id": camera,
                    "objects": objects,
                })
    audit_path = root / "run/logs/probe/frame_path_audit.jsonl"
    audit_path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_production_live_audit_reads_promoted_profile_without_experiment_file(tmp_path):
    _write_live_run(tmp_path)

    actual = production_recall_values(tmp_path)
    assert actual == {
        "pre_cluster_threshold": 0.03,
        "tentative_detector_confidence": 0.03,
        "data_associator_min_matching_score": 0.20,
        "min_iou_diff_new_target": 0.47,
        "min_tracker_confidence": 0.03,
        "probation_age": 2.0,
        "early_termination_age": 2.0,
    }

    result = audit(tmp_path)
    assert result["configuration_source"] == "production_config"
    assert result["experiment_ok"] is True
    assert result["recall_pass"] is True
    assert result["automatic_pass"] is True


def test_production_live_audit_checks_data_associator_section_not_trajectory_key(tmp_path):
    _write_live_run(tmp_path, data_associator_score=0.4)

    actual = production_recall_values(tmp_path)
    assert actual["data_associator_min_matching_score"] == 0.4
    result = audit(tmp_path)
    assert result["experiment_ok"] is False
    assert result["recall_pass"] is False
    assert result["automatic_pass"] is False


def test_replay_fps_uses_frame_pts_not_process_startup_and_sidecar_drain(tmp_path):
    _write_live_run(tmp_path)
    (tmp_path / "source_mode.json").write_text(json.dumps({"source_mode": "replay"}))
    report_path = tmp_path / "identity-live/runtime_report.json"
    report = json.loads(report_path.read_text())
    report["elapsed_seconds"] = 126.0
    report_path.write_text(json.dumps(report))

    readiness_path = tmp_path / "run/logs/probe/readiness.json"
    readiness = json.loads(readiness_path.read_text())
    for source in readiness["sources"]:
        source.update(mux_frames=3, pgie_frames=3, tracker_frames=3)
    readiness_path.write_text(json.dumps(readiness))

    audit_path = tmp_path / "run/logs/probe/frame_path_audit.jsonl"
    rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
    expanded = []
    for row in rows:
        row["pts"] = str(100_000_000 + int(row["frame_num"]) * 50_000_000)
        expanded.append(row)
        if row["stage"] == "pgie":
            mux_row = dict(row, stage="nvstreammux", objects=[])
            expanded.append(mux_row)
    audit_path.write_text("".join(json.dumps(row) + "\n" for row in expanded))

    result = audit(tmp_path)
    assert result["cameras"]["CAM-01"]["source_health"]["mux_fps"] == 20.0
    assert result["cameras"]["CAM-01"]["source_health"]["pgie_fps"] == 20.0
    assert result["cameras"]["CAM-01"]["source_health"]["tracker_fps"] == 20.0
    assert result["cameras"]["CAM-01"]["source_health"]["fps_measurement"] == "frame_audit_pts_span"
    assert result["recall_pass"] is True

    # An interior dropped frame must lower measured throughput even though
    # the first/last frame numbers and PTS remain unchanged.
    audit_path.write_text("".join(
        json.dumps(row) + "\n" for row in expanded if int(row["frame_num"]) != 10
    ))
    with_gap = audit(tmp_path)
    measured = with_gap["cameras"]["CAM-01"]["source_health"]["mux_fps"]
    assert abs(measured - (18 / 0.95)) < 1e-6
    assert with_gap["cameras"]["CAM-01"]["pass"] is False


def test_duplicate_frame_rows_do_not_inflate_person_instance_coverage(tmp_path):
    _write_live_run(tmp_path)
    audit_path = tmp_path / "run/logs/probe/frame_path_audit.jsonl"
    lines = audit_path.read_text().splitlines()
    result_before = audit(tmp_path)
    person_instances_before = result_before["cameras"]["CAM-01"]["pgie_to_tracker"]["pgie_person_instances"]

    duplicate = next(
        line for line in lines
        if (row := json.loads(line)).get("stage") == "pgie"
        and row.get("mapped_camera_id") == "CAM-01"
        and row.get("frame_num") == 4
    )
    audit_path.write_text("\n".join([*lines, duplicate]) + "\n")

    result_after = audit(tmp_path)
    assert result_after["duplicate_frame_audit_rows"] == 1
    assert result_after["cameras"]["CAM-01"]["pgie_to_tracker"]["pgie_person_instances"] == person_instances_before
    assert result_after["cameras"]["CAM-01"]["pgie_to_tracker"]["count_coverage"] == 1.0
