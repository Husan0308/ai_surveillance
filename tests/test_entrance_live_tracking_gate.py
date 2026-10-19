from services.mv3dt_staging.live_tracking.live_bridge import (
    safe_for_extended_review,
    smoke_gate_passed,
)


def _source_stats(*, missing=0, mismatch=0, nonmonotonic=0):
    return {
        "mux_frames": 21,
        "pgie_frames": 21,
        "tracker_frames": 21,
        "display_frame_matched": 20,
        "display_frame_missing": missing,
        "source_id_mismatch": mismatch,
        "mux_frame_num_nonmonotonic": nonmonotonic,
    }


def test_live_smoke_allows_bounded_late_preview_records_but_counts_them():
    baseline = {0: {key: 10 for key in ("mux_frames", "pgie_frames", "tracker_frames", "display_frame_matched")},
                1: {key: 10 for key in ("mux_frames", "pgie_frames", "tracker_frames", "display_frame_matched")}}
    stats = {0: _source_stats(missing=1), 1: _source_stats(missing=2)}
    assert smoke_gate_passed(stats, baseline)
    assert stats[0]["display_frame_missing"] == 1
    assert stats[1]["display_frame_missing"] == 2


def test_live_smoke_fails_when_source_frame_crosswalk_is_missing():
    baseline = {0: {key: 10 for key in ("mux_frames", "pgie_frames", "tracker_frames", "display_frame_matched")},
                1: {key: 10 for key in ("mux_frames", "pgie_frames", "tracker_frames", "display_frame_matched")}}
    stats = {0: _source_stats(), 1: _source_stats(mismatch=1)}
    assert not smoke_gate_passed(stats, baseline)


def test_live_smoke_requires_all_stages_and_exact_preview_matches_to_advance():
    baseline = {0: {key: 10 for key in ("mux_frames", "pgie_frames", "tracker_frames", "display_frame_matched")},
                1: {key: 10 for key in ("mux_frames", "pgie_frames", "tracker_frames", "display_frame_matched")}}
    no_tracker_progress = _source_stats()
    no_tracker_progress["tracker_frames"] = 10
    stats = {0: _source_stats(), 1: no_tracker_progress}
    assert not smoke_gate_passed(stats, baseline)


def test_extended_review_fails_closed_on_missing_gpu_telemetry():
    assert not safe_for_extended_review([])
    assert not safe_for_extended_review([{"status": "UNAVAILABLE"}] * 3)


def test_extended_review_skips_sustained_device_gpu_saturation():
    samples = [
        {"status": "AVAILABLE", "gpu_utilization_percent": 98},
        {"status": "AVAILABLE", "gpu_utilization_percent": 99},
        {"status": "AVAILABLE", "gpu_utilization_percent": 96},
    ]
    assert not safe_for_extended_review(samples)
    assert safe_for_extended_review([
        {"status": "AVAILABLE", "gpu_utilization_percent": 20},
        {"status": "AVAILABLE", "gpu_utilization_percent": 65},
        {"status": "AVAILABLE", "gpu_utilization_percent": 71},
    ])

