from pathlib import Path

from scripts.run_six_camera_v13_wall_candidate import auxiliary_detector_environment


ROOT = Path(__file__).resolve().parents[1]


def test_supplemental_detector_is_latest_only_and_cannot_backpressure_primary():
    source = (ROOT / "services/camera_v11/native/six_camera_v13/detection.hpp").read_text()
    primary_policy = '''for (auto *queue : {primary_queue, primary_meta_queue})
    g_object_set(queue, "max-size-buffers", 2u, "max-size-bytes", 0u,
        "max-size-time", guint64(0), "leaky", 0, "silent", TRUE, nullptr);'''
    auxiliary_policy = '''for (auto *queue : {auxiliary_queue, auxiliary_meta_queue})
    g_object_set(queue, "max-size-buffers", 1u, "max-size-bytes", 0u,
        "max-size-time", guint64(0), "leaky", 2, "silent", TRUE, nullptr);'''

    assert primary_policy in source
    assert auxiliary_policy in source
    assert "AuxiliaryExactJoin<AuxiliaryProposal, 6, 64>" in source
    assert "auxiliary_exact_join.take_until(key, deadline)" in source
    assert "auxiliary_exact_join.publish(key, std::move(proposals))" in source
    assert 'make("nvdsmetamux"' not in source
    assert "GST_BUFFER_PTS(buffer)" in source
    assert 'milliseconds > 100' in source
    assert "missing auxiliary frames fail closed" in source


def test_auxiliary_queue_pressure_is_observed_without_changing_inference_settings():
    source = (ROOT / "services/camera_v11/native/six_camera_v13/detection.hpp").read_text()
    aux_config = (ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r9-sparse-aux4"
                  / "config/config_infer_aux_yolo26m_896.txt")

    assert '"overrun", G_CALLBACK(auxiliary_queue_overrun)' in source
    assert "AUX_QUEUE input_level=" in source
    if aux_config.exists():
        config = aux_config.read_text()
        assert "interval=4" in config
        assert "pre-cluster-threshold=0.25" in config
        assert "nms-iou-threshold=0.45" in config


def test_auxiliary_exact_pts_wait_is_manifested_and_bounded():
    profile = {
        "name": "yolo26m-fp16-640-primary-896-aux-interval-4-context-verified",
        "aux_input": [896, 896],
        "aux_config": "config_infer_aux_yolo26m_896.txt",
        "metamux_config": "config_metamux_yolo26m_aux896.txt",
        "aux_engine_sha256": "e" * 64,
        "aux_parser_sha256": "f" * 64,
        "aux_unique_id": 3,
        "aux_join_mode": "exact-key-sidecar-v1",
        "exact_join_timeout_ms": 25,
        "metamux_loaded": False,
    }
    environment = auxiliary_detector_environment(profile)
    assert "V11_AUX_EXACT_JOIN_TIMEOUT_MS=25" in environment
    native_source = (ROOT / "services/camera_v11/native/six_camera_v13/detection.hpp").read_text()
    assert 'milliseconds > 100' in native_source
    assert "AUX_EXACT_JOIN_TIMEOUT_INVALID" in native_source
    assert "AUX_EXACT_JOIN_TIMEOUT" in native_source
    assert "AUX_BRANCH queued_frames=" in native_source
