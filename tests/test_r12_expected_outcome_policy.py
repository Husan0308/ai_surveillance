from scripts.run_r11_auth_quarantine_recovery import (
    CAMERA_IDS,
    combine_auth_test_outcome,
    evaluate_peer_progress,
)


STAGES = (
    "analytics_input_frames_total", "decoded_frames", "pgie_frames_total",
    "canonical_tracker_updates_total", "preview_frames_total",
)
AGES = (
    "age_since_last_decoded_frame_ms", "age_since_last_pgie_frame_ms",
    "age_since_last_tracker_output_ms", "age_since_last_preview_ms",
)


def _sample(monotonic_sec: float, count: int, offline_cam02: bool = True) -> dict:
    cameras = {}
    for camera in CAMERA_IDS:
        if camera == "CAM-02" and offline_cam02:
            cameras[camera] = {"online": False, "current_state": "QUARANTINED"}
            continue
        row = {key: count for key in STAGES}
        row.update({key: 30.0 for key in AGES})
        row.update({"online": True, "current_state": "ONLINE", "pipeline_errors": 0})
        cameras[camera] = row
    return {"monotonic_sec": monotonic_sec, "cameras": cameras}


def test_expected_auth_quarantine_does_not_count_failed_camera_as_peer():
    samples = [_sample(19.0, 100), _sample(40.0, 520)]

    result = evaluate_peer_progress(samples, {"CAM-02"}, 19.0, 40.0)

    assert result["pass"] is True
    assert set(result["peers"]) == set(CAMERA_IDS) - {"CAM-02"}
    assert all(row["pass"] and row["sample_count"] == 2
               for row in result["peers"].values())


def test_unexpectedly_excluding_camera_keeps_expected_quarantine_visible():
    samples = [_sample(19.0, 100), _sample(40.0, 520)]

    result = evaluate_peer_progress(samples, set(), 19.0, 40.0)

    assert result["pass"] is False
    assert result["peers"]["CAM-02"]["pass"] is False


def test_source_local_expected_failure_still_fails_for_stale_healthy_peer():
    samples = [_sample(19.0, 100), _sample(40.0, 520)]
    samples[1]["cameras"]["CAM-05"]["age_since_last_tracker_output_ms"] = 1000.0

    result = evaluate_peer_progress(samples, {"CAM-02"}, 19.0, 40.0)

    assert result["pass"] is False
    assert result["peers"]["CAM-05"]["pass"] is False


def _passing_auth_report() -> dict:
    return {
        "unauthorized_seen": True,
        "authentication_recovery_started": True,
        "quarantine_300s_recorded": True,
        "application_created_generation_2": True,
        "source_epoch_advanced_to_2": True,
        "recovered_through_all_stages": True,
        "native_retry_warning_absent": True,
        "interval_properties_read_back_zero": True,
        "fixture_auth_was_corrected": True,
        "other_cameras_progressed": True,
        "preview_sync_mismatches_zero": True,
        "candidate_clean_exit": True,
        "fixture_max_established_owners": 1,
        "fixture_client_sessions": 2,
        "cam02_final_online": True,
        "cam02_final_state": "ONLINE",
    }


def test_expected_auth_401_generic_soak_fail_does_not_override_scenario_pass():
    outcome = combine_auth_test_outcome(_passing_auth_report(), "FAIL", 1)

    assert outcome["status"] == "PASS"
    assert outcome["result_code"] == 0
    assert outcome["generic_soak_status"] == "FAIL"
    assert outcome["generic_soak_exit_code"] == 1
    assert outcome["failed_scenario_checks"] == []


def test_auth_scenario_still_fails_when_a_healthy_peer_did_not_progress():
    auth = _passing_auth_report()
    auth["other_cameras_progressed"] = False

    outcome = combine_auth_test_outcome(auth, "FAIL", 1)

    assert outcome["status"] == "FAIL"
    assert outcome["result_code"] == 1
    assert "other_cameras_progressed" in outcome["failed_scenario_checks"]
