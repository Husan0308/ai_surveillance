from pathlib import Path

import pytest

from scripts.cam_six_validation.camera_selection import (
    CAMERA_IDS,
    camera_ids_for_run,
    recording_filename,
    validate_camera_mode,
)


ROOT = Path(__file__).resolve().parents[1]


def test_default_selection_preserves_six_camera_order_and_recording_name():
    assert camera_ids_for_run() == CAMERA_IDS
    assert recording_filename(camera_ids_for_run()) == (
        "CAM-01_CAM-02_CAM-03_CAM-04_CAM-05_CAM-06.mkv"
    )


def test_single_camera_selection_keeps_camera_identity_and_output_name():
    selected = camera_ids_for_run("CAM-02")
    assert selected == ("CAM-02",)
    assert recording_filename(selected) == "CAM-02.mkv"


@pytest.mark.parametrize("camera_id", ["", "cam-02", "CAM-00", "CAM-07", "CAM-02/CAM-05"])
def test_invalid_camera_selections_fail_closed(camera_id):
    with pytest.raises(ValueError):
        camera_ids_for_run(camera_id)


def test_single_camera_mode_is_capture_only():
    validate_camera_mode("CAM-02", person_detection=False)
    with pytest.raises(ValueError, match="capture-only"):
        validate_camera_mode("CAM-02", person_detection=True)


def test_native_source_creation_uses_plan_and_distinct_logical_and_mux_ids():
    source = (ROOT / "scripts/cam_six_validation/main.cpp").read_text()
    assert "for (const auto &entry : source_plan)" in source
    assert "load_source(f, entry)" in source
    assert source.count("load_source(f, entry)") == 1
    assert source.count("ctx->source = create_source_element(ctx)") == 1
    assert "ctx->mux_pad_index" in source
    assert 'std::to_string(ctx->mux_pad_index)' in source
    assert '"source-id", ctx->source_id' in source
    assert "build_source_plan(only_camera, source_plan)" in source
    selected_at = source.index("build_source_plan(only_camera, source_plan)")
    assert selected_at < source.index("gst_init(nullptr, nullptr)") < source.index('gst_pipeline_new(')
    assert "if (only_camera_seen && only_camera.empty())" in source
    assert "if (only_camera_seen)" in source  # YOLO builds fail closed before startup


def test_launcher_passes_only_selected_credentials_and_native_selection():
    launcher = (ROOT / "scripts/validate_cam_six.py").read_text()
    assert "for cid in selected_camera_ids:" in launcher
    assert '*(["--only-camera", args.only_camera] if args.only_camera else [])' in launcher
    assert "recording_filename(tuple(selected_camera_ids))" in launcher
    assert "single-camera diagnostic output path must not already exist" in launcher


def test_single_camera_pipeline_shape_and_capture_mode_are_dynamic():
    source = (ROOT / "scripts/cam_six_validation/main.cpp").read_text()
    launcher = (ROOT / "scripts/validate_cam_six.py").read_text()
    selection = (ROOT / "scripts/cam_six_validation/camera_selection.py").read_text()
    assert "pipeline_shape(source_plan.size(), shape)" in source
    assert '"batch-size", shape.batch_size' in source
    assert '"rows", shape.tile_rows' in source
    assert '"columns", shape.tile_columns' in source
    assert "YOLO26_PERSON" in source
    assert "validate_camera_mode(args.only_camera, person_detection)" in launcher
    assert "--only-camera is restricted to the capture-only validator" in selection
    assert '" -DYOLO26_PERSON' in launcher
    assert "active_sources_json()" in source
    assert "camera_id" in source and "mux_pad_index" in source
    assert 'g_snprintf(cam, sizeof(cam), "CAM-%02d", ctx->source_id + 1);' in source
