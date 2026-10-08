from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_gate_a_direct_file_replay.py"
SPEC = importlib.util.spec_from_file_location("gate_a_direct_file_replay", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def rows(frames: list[int], pts: list[int] | None = None) -> list[dict]:
    if pts is None:
        pts = [100_000_000 + i * MODULE.EXPECTED_PTS_STEP_NS for i in range(len(frames))]
    return [
        {
            "source_id": 0,
            "source_local_frame_num": frame,
            "decoded_sequence": index + 1,
            "source_pts_ns": pts[index],
        }
        for index, frame in enumerate(frames)
    ]


def test_direct_file_integrity_requires_exact_source_local_timeline() -> None:
    report = MODULE.validate_source_frame_rows(rows(list(range(4))), [0], expected_frames=4)
    assert report["0"]["decoded_frames"] == 4
    assert report["0"]["source_local_first"] == 0
    assert report["0"]["source_local_last"] == 3
    assert report["0"]["unique_local_frames"] == 4


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        ([0, 1, 2, 0], "source-local frame restart/gap/repeat"),
        ([0, 1, 3, 4], "source-local frame restart/gap/repeat"),
        ([0, 1, 1, 2], "source-local frame restart/gap/repeat"),
        ([0, 1, 2], "expected 4 decoded frames"),
    ],
)
def test_direct_file_integrity_rejects_restarted_gapped_or_repeated_frames(
    frames: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MODULE.validate_source_frame_rows(rows(frames), [0], expected_frames=4)


def test_direct_file_integrity_rejects_pts_restart_or_duplicate() -> None:
    with pytest.raises(ValueError, match="source PTS restarted or repeated"):
        MODULE.validate_source_frame_rows(
            rows([0, 1, 2, 3], [100, 150, 200, 150]), [0], expected_frames=4
        )


def test_direct_file_integrity_rejects_unexpected_pts_cadence() -> None:
    with pytest.raises(ValueError, match="unexpected source PTS cadence"):
        MODULE.validate_source_frame_rows(
            rows([0, 1, 2, 3], [100, 150, 250, 300]), [0], expected_frames=4
        )


def test_normalization_uses_source_local_lookup_not_global_frame_number() -> None:
    source = MODULE.write_normalized_rows
    text = SCRIPT.read_text()
    assert "source_local_frame_num" in text
    assert "pipeline_frame_index" in text
    assert "source_pts_ns" in text
    assert "nvds_frame_num" in text
    assert "eval(" not in text
    assert callable(source)


def test_amended_reference_comparison_uses_canonical_rows_and_reviewed_frames(tmp_path: Path) -> None:
    comparator_path = Path(__file__).resolve().parents[1] / "scripts/compare_cam05_gate_a_reference.py"
    comparator_spec = importlib.util.spec_from_file_location("cam05_gate_a_compare", comparator_path)
    assert comparator_spec and comparator_spec.loader
    comparator = importlib.util.module_from_spec(comparator_spec)
    comparator_spec.loader.exec_module(comparator)
    run_dir, reference_dir = tmp_path / "run", tmp_path / "reference"
    run_dir.mkdir()
    reference_dir.mkdir()
    (reference_dir / "CAM01-per-frame.csv").write_text(
        "camera_id,frame,pts,chosen_bbox,chosen_source\n"
        'CAM-01,0,100,"[10,20,30,40]",LOCAL_ACTIVE\n'
        'CAM-01,1,150,"[11,20,30,40]",LOCAL_ACTIVE\n'
    )
    (reference_dir / "CAM04-per-frame.csv").write_text(
        "camera_id,frame,pts,chosen_bbox,chosen_source\n"
        'CAM-04,0,100,"[10,20,30,40]",LOCAL_ACTIVE\n'
    )
    (run_dir / "canonical-objects.jsonl").write_text("\n".join([
        json.dumps({"camera_id": "CAM-01", "source_frame_num": 0, "objects": [
            {"bbox": [10, 20, 40, 60], "bbox_source": "TRACKER", "native_track_id": 7}]}),
        json.dumps({"camera_id": "CAM-01", "source_frame_num": 1, "objects": []}),
        json.dumps({"camera_id": "CAM-04", "source_frame_num": 0, "objects": [
            {"bbox": [10, 20, 40, 60], "bbox_source": "PGIE_SHADOW_FALLBACK", "native_track_id": 9}]}),
    ]) + "\n")
    report = comparator.compare(run_dir, reference_dir)
    assert report["CAM-01"]["missed_reference_boxes"] == 1
    assert report["CAM-01"]["unmatched_candidate_boxes"] == 0
    assert report["CAM-01"]["matched_box_source_counts"] == {"TRACKER": 1}
    assert report["CAM-04"]["matched_box_source_counts"] == {"PGIE_SHADOW_FALLBACK": 1}


def test_cam05_direct_runner_runtime_budget_covers_longest_file_plus_eos_drain() -> None:
    runner_path = Path(__file__).resolve().parents[1] / "scripts/run_cam05_candidate_direct_replay.py"
    runner_spec = importlib.util.spec_from_file_location("cam05_direct_runner", runner_path)
    assert runner_spec and runner_spec.loader
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    assert runner.native_runtime_budget_seconds([{"duration_sec": 120.0}]) == 150
    assert runner.native_runtime_budget_seconds([
        {"duration_sec": 41.0}, {"duration_sec": 119.25}
    ]) == 150
    assert runner.native_runtime_budget_seconds([{"duration_sec": 2.05}]) == 120


def test_cam05_direct_runner_uses_verifier_config_batch_contract() -> None:
    runner_path = Path(__file__).resolve().parents[1] / "scripts/run_cam05_candidate_direct_replay.py"
    runner_spec = importlib.util.spec_from_file_location("cam05_direct_runner_batch_contract", runner_path)
    assert runner_spec and runner_spec.loader
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    manifest = {"yolo_context_verifier": {"runtime_batch_size": 2}}
    manifest["yolo_context_verifier"]["config"] = "yolo26m_context_verifier_448.txt"
    assert runner.resolve_verifier_config(manifest, None) == "yolo26m_context_verifier_448.txt"
    assert runner.resolve_verifier_config(manifest, "peoplenet_lowconf_verifier.txt") == (
        "peoplenet_lowconf_verifier.txt")
    assert runner.verifier_runtime_contract(manifest, "peoplenet_lowconf_verifier.txt") == (
        "PeopleNet-2.6.3", 1)
    assert runner.verifier_runtime_contract(manifest, "yolo26m_context_verifier_448.txt") == (
        "YOLO26m-640-FP16", 2)


def test_cam05_direct_runner_rejects_unmanifested_verifier_path() -> None:
    runner_path = Path(__file__).resolve().parents[1] / "scripts/run_cam05_candidate_direct_replay.py"
    runner_spec = importlib.util.spec_from_file_location("cam05_direct_runner_config_path", runner_path)
    assert runner_spec and runner_spec.loader
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    with pytest.raises(ValueError, match="plain filename"):
        runner.resolve_verifier_config(
            {"yolo_context_verifier": {"config": "nested/verifier.txt"}}, None)
    assert runner.resolve_verifier_config({}, None) == "yolo26m_context_verifier.txt"


def test_cam05_diagnostic_verifier_batch_override_is_engine_profile_bounded() -> None:
    runner_path = Path(__file__).resolve().parents[1] / "scripts/run_cam05_candidate_direct_replay.py"
    runner_spec = importlib.util.spec_from_file_location("cam05_direct_runner_batch_override", runner_path)
    assert runner_spec and runner_spec.loader
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    variant = {"batch": {"runtime": 2, "engine_min": 1, "engine_opt": 2, "engine_max": 2}}
    assert runner.resolve_verifier_batch_size(2, None, variant) == 2
    assert runner.resolve_verifier_batch_size(2, 1, variant) == 1
    with pytest.raises(ValueError, match="engine profile"):
        runner.resolve_verifier_batch_size(2, 3, variant)
    with pytest.raises(ValueError, match="manifest-declared"):
        runner.resolve_verifier_batch_size(1, 2, None)
    batch6 = {"batch": {"runtime": 6, "engine_min": 1, "engine_opt": 6, "engine_max": 6}}
    assert runner.resolve_verifier_batch_size(6, None, batch6) == 6
    assert runner.resolve_verifier_batch_size(2, 6, batch6) == 6
    with pytest.raises(ValueError, match="engine profile"):
        runner.resolve_verifier_batch_size(2, 6, {"batch": {"engine_min": 1, "engine_opt": 2, "engine_max": 2}})


def test_cam05_auxiliary_a_b_switch_is_explicit_and_preserves_primary_default() -> None:
    runner_path = Path(__file__).resolve().parents[1] / "scripts/run_cam05_candidate_direct_replay.py"
    runner_spec = importlib.util.spec_from_file_location("cam05_direct_runner_aux", runner_path)
    assert runner_spec and runner_spec.loader
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    assert runner.auxiliary_detector_environment(True) == [
        "V11_AUX_YOLO896_ENABLE=1",
        "V11_AUX_YOLO896_CONFIG=/config/config_infer_aux_yolo26m_896.txt",
        "V11_AUX_YOLO896_METAMUX_CONFIG=/config/config_metamux_yolo26m_aux896.txt",
    ]
    assert runner.auxiliary_detector_environment(False) == ["V11_AUX_YOLO896_ENABLE=0"]


def test_tracker_nan_diagnostics_are_opt_in_and_diagnostic_only() -> None:
    runner_path = Path(__file__).resolve().parents[1] / "scripts/run_cam05_candidate_direct_replay.py"
    runner_spec = importlib.util.spec_from_file_location("cam05_direct_runner_nan_diagnostics", runner_path)
    assert runner_spec and runner_spec.loader
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    assert runner.tracker_nan_diagnostic_environment(False) == []
    assert runner.tracker_nan_diagnostic_environment(True) == ["V11_TRACKER_NAN_DIAGNOSTICS=1"]


def test_cam05_direct_replay_timestamp_remux_requires_identical_video_packets() -> None:
    runner_path = Path(__file__).resolve().parents[1] / "scripts/run_cam05_candidate_direct_replay.py"
    runner_spec = importlib.util.spec_from_file_location("cam05_direct_runner_remux", runner_path)
    assert runner_spec and runner_spec.loader
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    same = {"width": 960, "height": 540, "fps": "20/1", "frames": 41}
    runner.validate_stream_copy_provenance(same, same, "packet-hash", "packet-hash")
    with pytest.raises(RuntimeError, match="compressed video packet payloads"):
        runner.validate_stream_copy_provenance(same, same, "source-packets", "different-packets")
    changed = {**same, "frames": 40}
    with pytest.raises(RuntimeError, match="changed input frames"):
        runner.validate_stream_copy_provenance(same, changed, "packet-hash", "packet-hash")
    source = runner_path.read_text()
    assert '"-c:v", "copy"' in source
    assert '"-avoid_negative_ts", "make_zero"' in source
    assert '"-fflags", "+genpts"' in source
