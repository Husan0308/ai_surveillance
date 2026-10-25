import json
from pathlib import Path

from scripts.cam_six_validation.stage_capture_manifest import (
    compare_mosaic_to_decoded,
    compare_source_to_mosaic,
    finalize_stage_capture,
)


ROOT = Path(__file__).resolve().parents[1]


def _mosaic(pts, *, target=5, frames=None):
    return {
        "stage": "pre_nvenc",
        "target_sec": target,
        "filename": f"pre-nvenc-t{target:02d}s.nv12",
        "buffer_pts_ns": pts,
        "nvds_frames": frames or [],
    }


def test_synthetic_zero_metadata_cannot_produce_exact_join():
    source = {"camera_id": "CAM-01", "source_id": 0, "buffer_pts_ns": 0,
              "target_sec": 5, "elapsed_monotonic_us": 5_010_000}
    mosaic = _mosaic(0, frames=[{"source_id": 0, "frame_num": 0, "buf_pts_ns": 0}])
    mosaic.update(status="saved", elapsed_monotonic_us=5_020_000)
    result = compare_source_to_mosaic(source, [mosaic])
    assert result["relation"] == "time_adjacent_only"
    assert result["delta_monotonic_us"] == 10_000


def test_even_equal_pts_and_claimed_identity_do_not_prove_tiled_camera_frame():
    source = {"source_id": 1, "buffer_pts_ns": 123, "target_sec": 30,
              "elapsed_monotonic_us": 30_010_000}
    mosaic = _mosaic(123, target=30, frames=[{"source_id": 1, "frame_num": 99, "buf_pts_ns": 123}])
    mosaic.update(status="saved", elapsed_monotonic_us=30_010_000)
    assert compare_source_to_mosaic(source, [mosaic])["relation"] == "time_adjacent_only"


def test_missing_or_distant_sample_metadata_is_uncorrelated():
    source = {"source_id": 3, "target_sec": 5, "elapsed_monotonic_us": 5_000_000}
    mosaic = _mosaic(5_000_000_000)
    assert compare_source_to_mosaic(source, [mosaic])["relation"] == "uncorrelated"
    mosaic.update(status="saved", elapsed_monotonic_us=8_000_001)
    assert compare_source_to_mosaic(source, [mosaic])["relation"] == "uncorrelated"
    source["source_id"] = None
    assert compare_source_to_mosaic(source, [mosaic])["relation"] == "uncorrelated"


def test_equal_time_candidates_are_ambiguous_and_other_targets_not_substituted():
    source = {"source_id": 0, "target_sec": 5, "elapsed_monotonic_us": 5_000_000}
    mosaic = _mosaic(1000)
    mosaic.update(status="saved", elapsed_monotonic_us=5_000_000)
    assert compare_source_to_mosaic(source, [mosaic, dict(mosaic)])["relation"] == "uncorrelated"
    mosaic["target_sec"] = 30
    assert compare_source_to_mosaic(source, [mosaic])["relation"] == "uncorrelated"


def test_zero_sampling_times_cannot_claim_even_time_adjacency():
    source = {"source_id": 0, "target_sec": 5, "buffer_pts_ns": 0,
              "elapsed_monotonic_us": 0}
    mosaic = _mosaic(0, frames=[{"source_id": 0, "frame_num": 0, "buf_pts_ns": 0}])
    mosaic.update(status="saved", elapsed_monotonic_us=0)
    assert compare_source_to_mosaic(source, [mosaic])["relation"] == "uncorrelated"


def test_mosaic_to_decoded_mkv_requires_same_sample_and_labels_pts_delta():
    mosaic = _mosaic(20_000, target=30)
    decoded = [
        {"target_sec": 5, "status": "saved", "pts_ns": 20_000, "frame_index": 1, "timestamp_candidate_count": 1},
        {"target_sec": 30, "status": "saved", "pts_ns": 21_000, "frame_index": 900, "image": "mkv-t30.png", "timestamp_candidate_count": 1},
    ]
    result = compare_mosaic_to_decoded(mosaic, decoded)
    assert result["relation"] == "time_adjacent_only"
    assert result["delta_ns"] == 1_000
    assert result["mkv_frame_index"] == 900


def test_mosaic_to_decoded_mkv_exact_pts_and_missing_pts():
    mosaic = _mosaic(20_000, target=55)
    exact = compare_mosaic_to_decoded(
        mosaic, [{"target_sec": 55, "status": "saved", "pts_ns": 20_000, "frame_index": 1650, "timestamp_candidate_count": 1}]
    )
    assert exact["relation"] == "exact_mosaic_pts"

    missing = compare_mosaic_to_decoded(
        mosaic, [{"target_sec": 55, "status": "saved", "pts_ns": None, "frame_index": 1650}]
    )
    assert missing["relation"] == "unmatched"


def test_duplicate_decoded_pts_prevent_exact_claim():
    result = compare_mosaic_to_decoded(
        _mosaic(1000), [{"target_sec": 5, "status": "saved", "pts_ns": 1000,
                        "timestamp_candidate_count": 2}]
    )
    assert result["relation"] == "ambiguous_duplicate_pts"
    missing = compare_mosaic_to_decoded(
        _mosaic(1000), [{"target_sec": 5, "status": "saved", "pts_ns": 1000}]
    )
    assert missing["relation"] == "uncorrelated"


def test_finalizer_marks_missing_native_summary_incomplete(tmp_path, monkeypatch):
    import scripts.cam_six_validation.stage_capture_manifest as manifest_module

    monkeypatch.setattr(manifest_module.shutil, "which", lambda _: None)
    capture_dir = tmp_path / "stage-captures"
    capture_dir.mkdir()
    (capture_dir / "stage-frames.jsonl").write_text(
        '{"kind":"header","schema":1}\n'
        '{"kind":"point_status","stage":"source_decoded","target_sec":5,"status":"capture_error"}\n'
    )

    result_path = finalize_stage_capture(capture_dir, tmp_path / "missing.mkv")
    result = json.loads(result_path.read_text())

    assert result["capture_session_complete"] is False
    assert result["queue_dropped"] is None
    assert result["decoded_mkv_samples"] == []
    assert result["decoded_mkv_status"]["status"] == "recording_missing"
    assert result_path.stat().st_mode & 0o777 == 0o600


def test_capture_is_opt_in_and_stream_probe_only_enqueues_owned_buffer_refs():
    main_cpp = (ROOT / "scripts/cam_six_validation/main.cpp").read_text()
    validator = (ROOT / "scripts/validate_cam_six.py").read_text()
    callback = main_cpp.split("static GstPadProbeReturn stage_capture_probe(", 1)[1].split(
        "static bool attach_stage_capture_probe", 1
    )[0]
    enqueue = main_cpp.split("bool enqueue(StageCaptureJob &job)", 1)[1].split(
        "void stop()", 1
    )[0]

    assert '\"--capture-stage-samples\"' in main_cpp
    assert '\"--capture-stage-samples\"' in validator
    assert "store_true" in validator
    assert "NvBufSurface2Raw" not in callback
    assert "write_all" not in callback
    assert "gst_buffer_ref(job.buffer)" in enqueue
    assert "std::try_to_lock" in enqueue
    assert "queue_size_ >= STAGE_CAPTURE_QUEUE_CAPACITY" in enqueue
    assert "STAGE_CAPTURE_TARGET_SEC[3] = {5, 30, 55}" in main_cpp


def test_queue_capacity_has_single_source_for_storage_header_and_log():
    main_cpp = (ROOT / "scripts/cam_six_validation/main.cpp").read_text()
    assert "STAGE_CAPTURE_QUEUE_CAPACITY = 8;" in main_cpp
    assert "std::array<StageCaptureJob, STAGE_CAPTURE_QUEUE_CAPACITY>" in main_cpp
    assert main_cpp.count("std::to_string(STAGE_CAPTURE_QUEUE_CAPACITY)") == 2
    assert "queue_capacity=16" not in main_cpp


def test_packed_preview_rejects_old_short_capture_and_decodes_correct_nv12(tmp_path):
    import shutil
    import pytest
    from scripts.cam_six_validation.stage_capture_manifest import _make_nv12_preview
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is unavailable")
    raw = tmp_path / "test.nv12"
    record = {"filename": raw.name, "width": 16, "height": 8,
              "raw_plane_width": [16, 8], "raw_plane_height": [8, 4],
              "packed_row_bytes": [16, 16], "packed_layout": "NV12_Y_UV_tight"}
    raw.write_bytes(bytes([100]) * 128 + bytes([37, 193]) * 16)
    assert _make_nv12_preview(record, tmp_path, ffmpeg) is None  # old 1.25WH
    raw.write_bytes(bytes([100]) * 128 + bytes([37, 193]) * 32)
    image = _make_nv12_preview(record, tmp_path, ffmpeg)
    assert image == "test.png"
    assert (tmp_path / image).stat().st_mode & 0o777 == 0o600
    record["packed_layout"] = None
    assert _make_nv12_preview(record, tmp_path, ffmpeg) is None
