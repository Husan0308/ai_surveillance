from pathlib import Path
import json

from scripts import calibration_recording_controller as controller
from scripts import validate_calibration_recording as validator


ROOT = Path(__file__).resolve().parents[1]


def test_room_camera_mapping_is_the_configured_pairing():
    assert controller.CAMERA_PAIRS == {
        "entrance": ("CAM-02", "CAM-05"),
        "main_room": ("CAM-03", "CAM-06"),
    }
    assert [row[0] for row in validator.ROOM_CAMERAS["entrance"]] == ["CAM-02", "CAM-05"]
    assert [row[0] for row in validator.ROOM_CAMERAS["main_room"]] == ["CAM-03", "CAM-06"]


def test_interactive_schedule_keeps_baseline_exit_and_post_exit_windows():
    assert controller.GREEN_SECONDS == 15
    assert controller.SESSION_SECONDS == 180
    assert controller.SUPPORTED_SESSION_SECONDS == (120, 180)
    for duration in controller.SUPPORTED_SESSION_SECONDS:
        assert duration - controller.POST_EXIT_MARGIN_SECONDS >= 100 / 20


def test_active_recording_gate_accepts_room_specific_camera_pair(tmp_path):
    status = tmp_path / "status.json"
    status.write_text(json.dumps({
        "state": "ACTIVE",
        "cameras": [
            {"camera_id": camera, "video_bytes": 16, "input_frames": 3, "encoded_frames": 3}
            for camera in ("CAM-03", "CAM-06")
        ],
    }))
    active = controller.wait_recording_active(status, ("CAM-03", "CAM-06"))
    assert active["state"] == "ACTIVE"


def test_native_recorder_is_optional_and_uses_existing_source_tees_only():
    source = (ROOT / "services/camera_v11/native/six_camera_v13/calibration_recording.hpp").read_text()
    main = (ROOT / "services/camera_v11/native/six_camera_v13/main.cpp").read_text()
    assert '"V11_CALIBRATION_RECORDING_ENABLED"' in main
    assert 'std::strcmp(recording_enabled, "1") == 0' in main
    assert '"CAM-02", "CAM-05"' in source
    assert '"CAM-03", "CAM-06"' in source
    assert 'gst_element_request_pad_simple(source.tee, "src_%u")' in source
    assert 'create("nvurisrcbin"' not in source
    assert 'video/x-raw(memory:NVMM),format=NV12' in source
    assert 'create("nvv4l2h264enc"' in source
    assert 'create("qtmux"' in source
    assert '(fields[4] != "120" && fields[4] != "180")' in source
    assert 'recording->duration_sec = static_cast<guint>(std::stoul(fields[4]))' in source
    assert '\\"faststart\\":false' in source
    assert '"max-size-buffers", 24u' in source
    assert '"leaky", 2' in source
    assert "metadata_writer_loop" in source
    assert "std::try_to_lock" in source
    assert "metadata_queue_drops" in source
    assert "queue_output_frames" in source
    assert "encoder_output_frames" in source


def test_streaming_probe_never_writes_jsonl_and_recorder_failure_fails_closed():
    source = (ROOT / "services/camera_v11/native/six_camera_v13/calibration_recording.hpp").read_text()
    probe_start = source.index("static GstPadProbeReturn input_probe(")
    probe_end = source.index("static GstPadProbeReturn encoded_probe(", probe_start)
    probe = source[probe_start:probe_end]
    assert "enqueue_metadata(branch" in probe
    assert "branch->rows <<" not in probe
    assert "metadata_rows_enqueued.load() == b->metadata_rows_written.load()" in source
    assert "metadata_queue_drops.load() == 0" in source
    assert "queue_overruns.load() == 0" in source
    assert "metadata_writer_failed.load()" in source
    assert '"bounded JSONL metadata queue overflow; recording evidence is incomplete"' in source
    assert '"recorder branch queue overrun; frames may be missing"' in source


def test_native_eos_is_scheduled_once_and_sent_inside_callback():
    source = (ROOT / "services/camera_v11/native/six_camera_v13/calibration_recording.hpp").read_text()
    assert "eos_scheduled.exchange(true)" in source
    assert "eos_sent.compare_exchange_strong(expected, true)" in source
    assert "if (!branch->eos_sent.exchange(true)) g_idle_add(eos_callback" not in source


def test_recorder_bounds_media_duration_using_source_pts_without_rewriting_pts():
    source = (ROOT / "services/camera_v11/native/six_camera_v13/calibration_recording.hpp").read_text()
    validator_source = (ROOT / "scripts/validate_calibration_recording.py").read_text()
    assert "source_pts_duration_elapsed(" in source
    assert "first_pts, current_pts, session->duration_sec" in source
    assert "source_pts_duration_boundary_reached" in source
    assert "included source PTS retained" in source
    assert "shared monotonic start; each camera stops at the earlier of host-duration or first-source-PTS duration; included source PTS retained" in validator_source
    assert 'abs(duration - requested_duration) > 0.25' in validator_source


def test_native_stop_command_is_session_scoped_and_never_reports_complete_early():
    source = (ROOT / "services/camera_v11/native/six_camera_v13/calibration_recording.hpp").read_text()
    assert 'fields[0] == "STOP"' in source
    assert "fields[1] != session_->id" in source
    assert 'request_stop("operator cancellation")' in source
    assert 'session_->requested_duration_completed = reason == "duration elapsed"' in source
    assert "session_->error.empty() && session_->requested_duration_completed" in source
    assert "stop_reason" in source


def test_controller_detaches_evidence_child_and_gracefully_stops_native_on_ctrl_c():
    source = (ROOT / "scripts/calibration_recording_controller.py").read_text()
    assert "start_new_session=True" in source
    assert "signal.signal(signal.SIGINT, capture_sigint)" in source
    assert 'f"STOP\\t{session_id}\\n"' in source
    assert "requesting orderly native EOS/finalization" in source
    assert "waiting for native EOS and MP4 finalization" in source


def test_post_exit_window_uses_consecutive_source_frame_numbers():
    source = (ROOT / "scripts/calibration_recording_controller.py").read_text()
    assert "last_source_frame" in source
    assert "frame.source_frame_num == prior_source_frame + 1" in source
    assert "camera == lifecycle_camera" in source


def test_calibration_video_validation_is_strict_about_pair_and_timing():
    source = (ROOT / "scripts/validate_calibration_recording.py").read_text()
    assert 'abs(duration - requested_duration) > 0.25' in source
    assert 'requested_duration not in (120.0, 180.0)' in source
    assert 'decoded_count != encoded_count' in source
    assert 'queue_overruns' in source
    assert 'avg_frame_rate' in source
    assert 'source PTS is empty, repeated, or non-monotonic' in source
    assert 'recorder first frame loss boundary' in source


def test_offline_harness_can_exercise_three_minute_session():
    source = (ROOT / "scripts/test_calibration_recording_offline.py").read_text()
    assert 'parser.add_argument("--duration-sec", type=int, choices=(120, 180)' in source
    assert 'recording_duration_sec": duration_sec' in source
    assert 'duration_sec + 30' in source


def test_single_command_entrance_runner_only_uses_active_validated_owner():
    source = (ROOT / "scripts/record_entrance_calibration.sh").read_text()
    assert "--duration-sec 180" in source
    assert '"$PY" -B -m scripts.calibration_recording_controller' in source
    assert "EXPECTED_BINARY_SHA256" in source
    assert "dport = :554" in source
    assert "calibration-recording-status.json" in source
    assert "scripts/start_full_live_stack.sh" not in source
    assert "nvurisrcbin" not in source


def test_controller_launches_human_evidence_recorder_as_project_module():
    source = (ROOT / "scripts/calibration_recording_controller.py").read_text()
    assert '"-m", "scripts.capture_six_camera_human_validation"' in source


def test_evidence_child_uses_sanitized_declared_runtime_and_safe_failure_path():
    env = controller.recorder_environment({
        "PYTHONHOME": "/stale/python",
        "PYTHONPATH": "/stale/site-packages",
        "KEEP_ME": "yes",
    })
    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["KEEP_ME"] == "yes"
    assert "summary_path = directory if session_started or directory.exists() else evidence_output" in (
        ROOT / "scripts/calibration_recording_controller.py"
    ).read_text()


def test_venv_python_symlink_is_not_dereferenced(tmp_path):
    target = tmp_path / "system-python"
    target.write_text("python executable placeholder")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(target)
    assert controller.declared_python_path(venv_python) == venv_python
    assert controller.declared_python_path(venv_python) != venv_python.resolve()
