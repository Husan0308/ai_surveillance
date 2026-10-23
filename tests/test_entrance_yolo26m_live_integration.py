from pathlib import Path

import pytest

from services.mv3dt_staging.live_tracking import live_bridge, yolo_candidate


def test_yolo_batch_two_config_preserves_frozen_detection_contract():
    original = yolo_candidate.FROZEN_PGIE.read_text()
    rendered = yolo_candidate.render_batch_two_config(original, engine_only=True)
    expected = original.replace("batch-size=6", "batch-size=2", 1)
    expected = expected.replace("onnx-file=/models/yolo26m_raw_otm.onnx\n", "", 1)
    assert rendered == expected
    assert "batch-size=2" in rendered
    assert "onnx-file=" not in rendered
    assert "network-mode=2" in rendered  # TensorRT FP16
    assert "parse-bbox-func-name=NvDsInferParseYolo26RawPerson" in rendered
    assert "cluster-mode=2" in rendered  # DeepStream NMS for raw OTM outputs
    assert "pre-cluster-threshold=0.25" in rendered
    assert "nms-iou-threshold=0.45" in rendered
    assert "infer-dims=3;640;640" in rendered


def test_frozen_yolo_assets_and_config_hash_contract():
    hashes = yolo_candidate.validate_yolo_assets(yolo_candidate.DEFAULT_MODEL_DIR)
    assert hashes["yolo26m_raw_otm_b6_fp16.engine"] == (
        "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"
    )
    assert hashes["libyolo26rawperson.so"] == (
        "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19"
    )
    assert yolo_candidate.sha256_file(yolo_candidate.FROZEN_PGIE) == (
        "fbfa196338b6781b7d65c3058d4f02069e9116963804d0c3b23d1bf2301e4113"
    )


def test_pnt_command_remains_the_default_rollback_path():
    command = live_bridge.docker_command(mux_only=False, output_root=Path("/tmp/no-write-test"))
    joined = " ".join(command)
    assert live_bridge.DS_IMAGE in command
    assert live_bridge.YOLO_DS_IMAGE not in command
    assert "--pull=never" not in command
    assert "/workspace/app/config_pgie_pnt_batch2.txt" in joined
    assert "/workspace/models/PeopleNetTransformer,readonly" in joined
    assert "--network none" in joined
    assert "USE_NEW_NVSTREAMMUX=no" in joined


def test_yolo_command_is_pinned_engine_only_and_isolated(tmp_path, monkeypatch):
    staged_dir = tmp_path / "staged-models"
    config_file = tmp_path / "staged-pgie.txt"
    profile = {
        "model_dir": str(staged_dir),
        "config_file": str(config_file),
        "engine_only": True,
        "batch_size": 2,
    }
    monkeypatch.setattr(yolo_candidate, "stage_yolo_batch_two", lambda *args, **kwargs: profile)
    command = live_bridge.docker_command(
        mux_only=False,
        output_root=tmp_path,
        detector="yolo26m",
        yolo_artifacts=yolo_candidate.DEFAULT_MODEL_DIR,
    )
    joined = " ".join(command)
    assert live_bridge.YOLO_DS_IMAGE in command
    assert "--pull=never" in command
    assert "--network none" in joined
    assert "USE_NEW_NVSTREAMMUX=no" in joined
    assert f"source={staged_dir},target=/models,readonly" in joined
    assert f"source={config_file},target={yolo_candidate.CONTAINER_CONFIG},readonly" in joined
    assert "PeopleNetTransformer" not in joined
    assert "--gpus all" in joined
    assert "onnx" not in joined


def test_detector_class_ids_are_explicit_and_unknown_detectors_fail_closed():
    assert live_bridge.detector_person_class("yolo26m") == 0
    assert live_bridge.detector_person_class("peoplenet") == 1
    with pytest.raises(ValueError):
        live_bridge.detector_person_class("fallback")


def test_smoke_preflight_requires_three_available_samples_below_limit():
    safe = [{"status": "AVAILABLE", "gpu_utilization_percent": 94}] * 3
    assert live_bridge.safe_for_smoke(safe)
    assert not live_bridge.safe_for_smoke(safe[:-1])
    assert not live_bridge.safe_for_smoke(
        [{"status": "AVAILABLE", "gpu_utilization_percent": 95}] + safe[1:]
    )
    assert not live_bridge.safe_for_smoke([{"status": "UNAVAILABLE"}] * 3)


def test_smoke_only_stops_even_when_extended_review_guard_would_allow_continuation():
    low = [{"status": "AVAILABLE", "gpu_utilization_percent": 10}] * 3
    assert live_bridge.post_smoke_action(smoke_only=True, gpu_samples=low) == "STOP_SMOKE_ONLY"
    assert live_bridge.post_smoke_action(smoke_only=False, gpu_samples=low) == "CONTINUE_REVIEW"
    assert live_bridge.post_smoke_action(smoke_only=False, gpu_samples=[]) == "STOP_RESOURCE_GUARD"


def test_high_iou_metric_reports_candidates_not_identity_or_duplicate_truth():
    records = [{
        "objects": [
            {"class_id": 0, "left": 10, "top": 10, "width": 20, "height": 40},
            {"class_id": 0, "left": 10, "top": 10, "width": 20, "height": 40},
            {"class_id": 1, "left": 10, "top": 10, "width": 20, "height": 40},
        ]
    }]
    assert live_bridge.high_overlap_pairs(records, class_id=0) == 1
    assert live_bridge.high_overlap_pairs(records, class_id=1) == 0


def test_track_frame_summary_keeps_camera_local_ids_and_class_filtering():
    rows = [
        {"source_id": 0, "input_frame_index": 2,
         "objects": [{"class_id": 0, "tracker_id": 9}, {"class_id": 1, "tracker_id": 4}]},
        {"source_id": 0, "input_frame_index": 5,
         "objects": [{"class_id": 0, "tracker_id": 9}]},
        {"source_id": 1, "input_frame_index": 3,
         "objects": [{"class_id": 0, "tracker_id": 9}]},
    ]
    summary = live_bridge.track_frame_summary(rows, class_id=0)
    assert summary["0"]["9"] == {
        "frames_observed": 2,
        "first_input_frame": 2,
        "last_input_frame": 5,
        "observed_frame_gaps": 2,
    }
    assert summary["1"]["9"]["frames_observed"] == 1
    assert "4" not in summary["0"]
