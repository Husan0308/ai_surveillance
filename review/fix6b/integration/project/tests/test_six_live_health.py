"""CPU-only tests for official DeepStream six-source FPS log monitoring."""
import json
import stat

import pytest

from services.gpu.runtime.live_health import CAMERA_IDS, SourceFpsMonitor


def perf(values=None):
    values = [20.0] * 6 if values is None else values
    return "**PERF: " + "\t".join(f"{fps:.2f} ({fps:.2f})" for fps in values)


def six_progress(source_id, at_ns, count, *, tracker_count=None):
    camera_id = CAMERA_IDS[source_id]
    tracker_count = count if tracker_count is None else tracker_count
    stages = [
        {"count": count, "last_ns": at_ns - 10_000_000},  # decoded source
        {"count": count, "last_ns": at_ns - 10_000_000},  # streammux
        {"count": count, "last_ns": at_ns - 10_000_000},  # PGIE
        {"count": tracker_count, "last_ns": at_ns - 10_000_000},  # NvDCF
    ]
    return "SIX_PROGRESS " + json.dumps({
        "source_id": source_id, "camera_id": camera_id,
        "monotonic_ns": at_ns, "fatal_metadata": False, "stages": stages,
    })


def feed_native_round(mon, second, counts, *, now=None):
    for source_id, count in enumerate(counts):
        assert mon.ingest(six_progress(source_id, second * 1_000_000_000, count),
                          now=float(second if now is None else now))


def test_two_complete_samples_enable_readiness():
    mon = SourceFpsMonitor()
    assert not mon.snapshot(now=0)["ready"]
    assert mon.ingest(perf(), now=0)
    assert not mon.snapshot(now=0)["ready"]
    assert mon.ingest(perf(), now=5)
    snap = mon.snapshot(now=6)
    assert snap["ready"]
    assert snap["samples"] == 2
    assert len(snap["camera_fps"]) == 6
    assert tuple(snap["camera_fps"]) == CAMERA_IDS


def test_native_source_progress_keeps_readiness_when_perf_lines_are_absent():
    mon = SourceFpsMonitor(min_fps=5.0, max_age_sec=15.0, min_consecutive=2)
    feed_native_round(mon, 1, [20] * 6)
    assert not mon.snapshot(now=1.0)["ready"]
    feed_native_round(mon, 2, [40] * 6)
    assert not mon.snapshot(now=2.0)["ready"]
    feed_native_round(mon, 3, [60] * 6)
    snap = mon.snapshot(now=3.1)
    assert snap["ready"]
    assert snap["source_progress_mode"] == "native_decode_and_tracker_counters"
    assert all(abs(fps - 20.0) < 0.01 for fps in snap["camera_fps"].values())
    assert all(abs(value["decode"] - 20.0) < 0.01
               and abs(value["tracker"] - 20.0) < 0.01
               for value in snap["camera_stage_fps"].values())


def test_native_source_stall_fails_closed_and_recovers_after_two_good_samples():
    mon = SourceFpsMonitor(min_fps=5.0, max_age_sec=15.0, min_consecutive=2)
    feed_native_round(mon, 1, [20] * 6)
    feed_native_round(mon, 2, [40] * 6)
    feed_native_round(mon, 3, [60] * 6)
    assert mon.snapshot(now=3.0)["ready"]

    # CAM-03 still reports empty-room frames in the other stages, but its
    # decoded and tracker frame counts have stopped advancing.
    feed_native_round(mon, 4, [80, 80, 60, 80, 80, 80])
    assert not mon.snapshot(now=4.0)["ready"]
    feed_native_round(mon, 5, [100, 100, 80, 100, 100, 100])
    assert not mon.snapshot(now=5.0)["ready"]
    feed_native_round(mon, 6, [120] * 6)
    assert mon.snapshot(now=6.0)["ready"]


def test_one_fresh_zero_delta_sample_fails_closed_before_staleness_threshold():
    mon = SourceFpsMonitor(min_fps=5.0, max_age_sec=15.0, min_consecutive=2)
    feed_native_round(mon, 1, [20] * 6)
    feed_native_round(mon, 2, [40] * 6)
    feed_native_round(mon, 3, [60] * 6)
    assert mon.snapshot(now=3.0)["ready"]

    # Source 0's record itself is fresh, but its decode and tracker counters
    # did not advance. The current guard intentionally fails closed on that
    # sample rather than waiting for max_age_seconds.
    feed_native_round(mon, 4, [60, 80, 80, 80, 80, 80])
    snapshot = mon.snapshot(now=4.01)
    assert not snapshot["ready"]
    assert snapshot["fresh"]
    assert snapshot["guard_thresholds"] == {
        "minimum_fps": 5.0,
        "maximum_age_seconds": 15.0,
        "required_consecutive_good_samples": 2,
    }
    assert snapshot["camera_progress"]["CAM-01"]["decode_count"] == 60
    assert snapshot["camera_progress"]["CAM-01"]["decode_fps"] == 0.0
    assert snapshot["camera_progress"]["CAM-01"]["consecutive_good"] == 0
    assert snapshot["camera_progress"]["CAM-02"]["decode_fps"] == 20.0


def test_counter_reset_and_reconnecting_state_do_not_fabricate_progress():
    mon = SourceFpsMonitor(min_fps=5.0, max_age_sec=15.0, min_consecutive=2)
    feed_native_round(mon, 1, [20] * 6)
    feed_native_round(mon, 2, [40] * 6)
    feed_native_round(mon, 3, [60] * 6)
    assert mon.snapshot(now=3.0)["ready"]

    # Lifecycle state is retained separately; it cannot make a repeated or
    # reset frame counter count as progress.
    assert not mon.ingest(
        'SIX_SOURCE_STATE {"source_id":0,"camera_id":"CAM-01",'
        '"state":"reconnecting","source_generation":1,"monotonic_ns":3500000000}',
        now=3.5,
    )
    feed_native_round(mon, 4, [5, 80, 80, 80, 80, 80])
    snapshot = mon.snapshot(now=4.0)
    assert not snapshot["ready"]
    assert snapshot["camera_progress"]["CAM-01"]["decode_count"] == 5
    assert snapshot["camera_progress"]["CAM-01"]["decode_fps"] == 0.0

    # Recovery is still detected, and still requires two good counter deltas.
    feed_native_round(mon, 5, [25, 100, 100, 100, 100, 100])
    assert not mon.snapshot(now=5.0)["ready"]
    feed_native_round(mon, 6, [45, 120, 120, 120, 120, 120])
    assert mon.snapshot(now=6.0)["ready"]


def test_native_progress_rejects_wrong_logical_mapping_and_fatal_metadata():
    mon = SourceFpsMonitor()
    wrong = json.loads(six_progress(0, 1_000_000_000, 20).split(" ", 1)[1])
    wrong["camera_id"] = "CAM-06"
    assert not mon.ingest("SIX_PROGRESS " + json.dumps(wrong), now=1)
    fatal = json.loads(six_progress(1, 1_000_000_000, 20).split(" ", 1)[1])
    fatal["fatal_metadata"] = True
    assert not mon.ingest("SIX_PROGRESS " + json.dumps(fatal), now=1)
    assert mon.snapshot(now=1)["samples"] == 0


def test_parent_output_pump_uses_native_progress_for_readiness():
    from io import StringIO
    from services.gpu.runtime import live_launcher
    from services.shared.credential_redaction import CredentialRedactor

    lines = []
    for second, count in ((1, 20), (2, 40), (3, 60)):
        lines.extend(six_progress(source_id, second * 1_000_000_000, count)
                     for source_id in range(6))

    class ChildOutput:
        stdout = StringIO("\n".join(lines) + "\n")

    monitor = SourceFpsMonitor()
    live_launcher._pump_lines(ChildOutput(), CredentialRedactor(), monitor,
                              live_launcher.ChildOutputDiagnostics())
    assert monitor.snapshot()["ready"]
    assert monitor.snapshot()["source_progress_mode"] == "native_decode_and_tracker_counters"


def test_zero_fps_causes_fail_closed():
    mon = SourceFpsMonitor()
    mon.ingest(perf(), now=0)
    mon.ingest(perf(), now=5)
    assert mon.snapshot(now=6)["ready"]
    values = [20.0] * 6
    values[4] = 0.0
    assert mon.ingest(perf(values), now=10)
    assert not mon.snapshot(now=11)["ready"]
    assert mon.snapshot(now=11)["camera_fps"]["CAM-05"] == 0.0


def test_stale_stats_are_not_live_frame_progress():
    mon = SourceFpsMonitor(max_age_sec=15)
    mon.ingest(perf(), now=0)
    mon.ingest(perf(), now=5)
    assert mon.snapshot(now=20)["ready"]
    assert not mon.snapshot(now=20.01)["ready"]
    assert not mon.snapshot(now=20.01)["fresh"]


def test_bad_or_partial_lines_do_not_advance_counter():
    mon = SourceFpsMonitor()
    bad = [
        "**PERF: FPS 0 (Avg)\t FPS 1 (Avg)",
        "**PERF: 20 (20)\t20 (20)",
        "INFO: " + perf(),
        "**PERF: 20.0 (20.0) " * 8,
        "PERF(0): 20.0 (20.0)",
        "**PERF: 0 (Avg)\t1 (Avg)\t2 (Avg)",
        "",
    ]
    for line in bad:
        assert not mon.ingest(line, now=0)
    assert mon.snapshot(now=0)["samples"] == 0


def test_stale_gap_resets_good_streak():
    mon = SourceFpsMonitor(max_age_sec=15)
    mon.ingest(perf(), now=0)
    mon.ingest(perf(), now=5)
    assert mon.snapshot(now=5)["ready"]
    mon.ingest(perf(), now=25)
    assert not mon.snapshot(now=25)["ready"]
    mon.ingest(perf(), now=30)
    assert mon.snapshot(now=30)["ready"]


def test_threshold_and_instrumentation_limits_are_explicit():
    with pytest.raises(ValueError):
        SourceFpsMonitor(min_fps=0)
    with pytest.raises(ValueError):
        SourceFpsMonitor(max_age_sec=-1)
    assert "not exact frame PTS" in SourceFpsMonitor().snapshot(now=0)["limited_to"]


def test_live_launcher_uses_monitor_before_and_during_timer():
    from pathlib import Path
    launcher = Path(__file__).resolve().parents[1] / "services/gpu/runtime/live_launcher.py"
    code = launcher.read_text()
    assert "deadline=stop_deadline" in code
    assert "run_deadline = invocation_started + args.duration" in code
    assert "stop_deadline = run_deadline - BOUNDED_SHUTDOWN_RESERVE_SECONDS" in code
    assert "run_deadline_monotonic" in code
    assert "timeout --foreground --verbose --signal=KILL" in code
    assert "container_kill_deadline = stop_deadline + DAILY_STOP_GRACE_SECONDS" in code
    assert "_ask_operator(None, None, mosaic_preview=args.mosaic_preview," in code
    assert code.index("_ask_operator(None, None") < code.index("proc = subprocess.Popen(")
    assert "runtime_limit_seconds=runtime_limit" in code
    assert "watchdog = threading.Timer(" in code
    assert "--duration must be between 1 and 600 seconds" in code
    assert "monitor_snapshot = monitor.snapshot()" in code
    assert "if not monitor_snapshot[\"ready\"]:" in code
    assert "monitor = SourceFpsMonitor(" in code
    assert "LIVE SOURCE FPS:" in code
    assert '["docker", "inspect", "--format"' in code
    assert "_remove_owned_container(owned_name, deadline=run_deadline)" in code


def test_operator_exit_error_includes_docker_run_status(monkeypatch):
    from services.gpu.runtime import live_launcher

    class Input:
        @staticmethod
        def isatty():
            return True

    class ExitedProcess:
        @staticmethod
        def poll():
            return 17

    monkeypatch.setattr(live_launcher.sys, "stdin", Input())
    monitor = SourceFpsMonitor()
    with pytest.raises(RuntimeError, match="docker_run_exit_code=17"):
        live_launcher._ask_operator(ExitedProcess(), monitor)


def test_failure_report_is_private_bounded_and_redacted(tmp_path, monkeypatch):
    from services.gpu.runtime import live_launcher
    from services.shared.credential_redaction import CredentialRedactor

    monkeypatch.setattr(live_launcher, "ROOT", tmp_path)
    redactor = CredentialRedactor()
    redactor.add_credentials(
        username="synthetic-user-marker",
        password="synthetic-password-marker",
        uri="rtsp://synthetic-user-marker:synthetic-password-marker@example.invalid/secret-channel",
    )
    diagnostics = live_launcher.ChildOutputDiagnostics(
        max_tail=3, max_events=2, max_errors=2, max_line_chars=300,
    )
    diagnostics.ingest(redactor.sanitize_child_output(
        "ERROR from source0: internal data stream error "
        "rtsp://synthetic-user-marker:synthetic-password-marker@example.invalid/secret-channel"
    ))
    diagnostics.ingest(redactor.sanitize_child_output(
        "Debug info: auth=synthetic-password-marker"
    ))
    diagnostics.ingest(redactor.sanitize_child_output("Received EOS"))
    camera_wall_output = live_launcher.CameraWallOutputDiagnostics(max_lines=2)
    camera_wall_output.ingest("Traceback (most recent call last):")
    camera_wall_output.ingest(redactor.sanitize_child_output(
        "RuntimeError: auth=synthetic-password-marker"
    ))

    class ExitedProcess:
        returncode = 23

    container_state = {
        "Status": "exited", "Running": False, "OOMKilled": False,
        "ExitCode": 23, "Error": "rtsp://synthetic-user-marker:"
        "synthetic-password-marker@example.invalid/secret-channel",
        "Pid": 0,
    }
    calls = []

    def fake_inspect(*args, **kwargs):
        calls.append(args[0])
        return type("Result", (), {
            "returncode": 0, "stdout": json.dumps(container_state), "stderr": "",
        })()

    monkeypatch.setattr(live_launcher.subprocess, "run", fake_inspect)
    state = live_launcher._inspect_owned_container("unique-test-container", redactor)

    path = live_launcher._write_failure_report(
        stage="operator_confirmation",
        exc=RuntimeError("viewer exited; docker_run_exit_code=23"),
        redactor=redactor,
        diagnostics=diagnostics,
        proc=ExitedProcess(),
        exit_code_at_detection=23,
        cleanup_stop_requested=False,
        container_state_before_cleanup=state,
        container_removed=True,
        camera_wall_output=camera_wall_output,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    encoded = path.read_text(encoding="utf-8")

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert payload["failure_stage"] == "operator_confirmation"
    assert payload["exit_status"]["docker_run_returncode_at_failure_detection"] == 23
    assert payload["exit_status"]["docker_run_returncode_after_cleanup"] == 23
    assert payload["exit_status"]["cleanup_stop_requested"] is False
    assert payload["exit_status"]["container_state_before_cleanup"]["ExitCode"] == 23
    assert payload["exit_status"]["container_state_before_cleanup"]["Status"] == "exited"
    assert payload["exit_status"]["container_removed_after_capture"] is True
    assert payload["camera_wall_output"] == [
        "Traceback (most recent call last):", "RuntimeError: auth=<redacted>",
    ]
    assert payload["gstreamer_errors"][0]["following_lines"] == [
        "Debug info: auth=<redacted>", "Received EOS",
    ]
    assert "Received EOS" in payload["diagnostic_events"]
    for marker in ("synthetic-user-marker", "synthetic-password-marker",
                   "secret-channel", "example.invalid"):
        assert marker not in encoded
    assert calls == [[
        "docker", "inspect", "--format", "{{json .State}}", "unique-test-container",
    ]]


def test_child_output_pump_collects_only_redacted_gstreamer_diagnostics(capsys):
    from io import StringIO
    from services.gpu.runtime import live_launcher
    from services.shared.credential_redaction import CredentialRedactor

    redactor = CredentialRedactor()
    redactor.add_credentials(
        username="pump-user-marker", password="pump-password-marker",
        uri="rtsp://pump-user-marker:pump-password-marker@example.invalid/private",
    )

    class ChildOutput:
        stdout = StringIO(
            "ERROR from source2: failed rtsp://pump-user-marker:"
            "pump-password-marker@example.invalid/private\n"
            "Debug info: token=pump-password-marker\n"
        )

    diagnostics = live_launcher.ChildOutputDiagnostics()
    live_launcher._pump_lines(
        ChildOutput(), redactor, SourceFpsMonitor(), diagnostics,
    )
    captured = capsys.readouterr().out
    serialized = json.dumps(diagnostics.snapshot())
    assert "<redacted-rtsp-uri>" in captured
    assert "pump-user-marker" not in captured + serialized
    assert "pump-password-marker" not in captured + serialized
    assert "ERROR from source2" in serialized
    assert "Debug info: token=<redacted>" in serialized


def test_six_progress_records_are_not_classified_as_gstreamer_errors():
    from services.gpu.runtime.live_launcher import ChildOutputDiagnostics

    diagnostics = ChildOutputDiagnostics()
    diagnostics.ingest(
        'SIX_PROGRESS {"source_id":2,"fatal_metadata":false,"source_errors":0}'
    )
    snapshot = diagnostics.snapshot()

    assert snapshot["source_progress_messages"] == [
        'SIX_PROGRESS {"source_id":2,"fatal_metadata":false,"source_errors":0}'
    ]
    assert snapshot["gstreamer_errors"] == []
    assert snapshot["diagnostic_events"] == []


def test_source_stall_diagnostic_is_disabled_by_default():
    from services.gpu.runtime.live_launcher import ChildOutputDiagnostics

    diagnostics = ChildOutputDiagnostics()
    diagnostics.ingest(six_progress(0, 1_000_000_000, 20))
    diagnostics.ingest("ERROR from src_elem0: synthetic timeout")
    snapshot = diagnostics.snapshot()["source_stall_diagnostics"]
    assert snapshot == {
        "enabled": False,
        "progress_samples": [],
        "progress_samples_observed": 0,
        "progress_samples_dropped": 0,
        "events": [],
        "events_observed": 0,
        "events_dropped": 0,
        "guard_failure": None,
    }


def test_opt_in_source_stall_diagnostic_is_bounded_and_redacts_event_text(monkeypatch):
    from services.gpu.runtime.live_launcher import ChildOutputDiagnostics
    from services.shared.credential_redaction import CredentialRedactor

    diagnostics = ChildOutputDiagnostics(
        source_stall_diagnostics=True,
        max_source_progress_samples=2,
        max_source_events=2,
    )
    redactor = CredentialRedactor()

    def progress_line(source_id, at_ns, count):
        stages = [
            {"count": count, "last_ns": at_ns - 1, "pts": count * 50_000_000}
            for _ in range(4)
        ]
        return "SIX_PROGRESS " + json.dumps({
            "source_id": source_id,
            "camera_id": CAMERA_IDS[source_id],
            "monotonic_ns": at_ns,
            "source_generation": 0,
            "fatal_metadata": False,
            "stages": stages,
        })

    for at_ns, count in ((1_000_000_000, 20), (2_000_000_000, 40),
                         (3_000_000_000, 60)):
        diagnostics.ingest(progress_line(0, at_ns, count))
    diagnostics.ingest(
        redactor.sanitize_child_output(
            "ERROR from src_elem0: RTSP timeout rtsp://fixture-user:fixture-secret@host/path"
        )
    )
    diagnostics.ingest(
        'SIX_SOURCE_STATE {"source_id":0,"camera_id":"CAM-01",'
        '"state":"reconnecting","source_generation":1,"monotonic_ns":4000000000}'
    )
    diagnostics.ingest("Received EOS. Exiting ...")

    snapshot = diagnostics.snapshot()["source_stall_diagnostics"]
    assert snapshot["enabled"] is True
    assert snapshot["progress_samples_observed"] == 3
    assert snapshot["progress_samples_dropped"] == 1
    assert len(snapshot["progress_samples"]) == 2
    assert snapshot["progress_samples"][-1]["decode_count"] == 60
    assert snapshot["progress_samples"][-1]["source_id"] == 0
    assert snapshot["events_observed"] == 3
    assert snapshot["events_dropped"] == 1
    assert [event["event"] for event in snapshot["events"]] == [
        "source_reconnecting", "eos",
    ]
    assert snapshot["events"][0]["source_generation"] == 1
    assert all(event["host_receipt_monotonic_ns"] > 0 for event in snapshot["events"])
    serialized = json.dumps(snapshot)
    assert "fixture-secret" not in serialized
    assert "rtsp://" not in serialized


def test_source_guard_failure_snapshot_keeps_counts_and_thresholds():
    from services.gpu.runtime.live_launcher import ChildOutputDiagnostics

    mon = SourceFpsMonitor(min_fps=5.0, max_age_sec=15.0, min_consecutive=2)
    feed_native_round(mon, 1, [20] * 6)
    feed_native_round(mon, 2, [40] * 6)
    feed_native_round(mon, 3, [60] * 6)
    feed_native_round(mon, 4, [60, 80, 80, 80, 80, 80])
    snapshot = mon.snapshot(now=4.0)

    diagnostics = ChildOutputDiagnostics(source_stall_diagnostics=True)
    diagnostics.record_source_guard_failure(snapshot)
    guard = diagnostics.snapshot()["source_stall_diagnostics"]["guard_failure"]
    assert guard["ready"] is False
    assert guard["guard_thresholds"]["maximum_age_seconds"] == 15.0
    assert guard["camera_progress"]["CAM-01"]["decode_count"] == 60
    assert guard["camera_progress"]["CAM-01"]["consecutive_good"] == 0


def test_opt_in_diagnostic_distinguishes_gstreamer_severity_from_timeout_class():
    from services.gpu.runtime.live_launcher import ChildOutputDiagnostics

    diagnostics = ChildOutputDiagnostics(source_stall_diagnostics=True)
    diagnostics.ingest("ERROR from src_elem0: simulated RTSP timeout")
    diagnostics.ingest("WARNING from src_elem0: simulated source warning")
    diagnostics.ingest("Received EOS. Exiting ...")
    events = diagnostics.snapshot()["source_stall_diagnostics"]["events"]
    assert [(event["event"], event["source_id"], event["severity"], event["condition"])
            for event in events] == [
        ("timeout", 0, "error", "gstreamer_timeout"),
        ("gstreamer_warning", 0, "warning", None),
        ("eos", None, None, "end_of_stream"),
    ]


def test_tracker_metadata_is_not_retained_in_child_failure_diagnostics():
    from services.gpu.runtime.live_launcher import ChildOutputDiagnostics

    diagnostics = ChildOutputDiagnostics()
    diagnostics.ingest(
        'SIX_TRACKER_FRAME {"source_id":0,"camera_id":"CAM-01",'
        '"objects":[{"native_track_id":123,"bbox":[1,2,3,4]}]}'
    )
    snapshot = diagnostics.snapshot()

    assert snapshot["recent_child_output"] == []
    assert snapshot["diagnostic_events"] == []
    assert snapshot["source_progress_messages"] == []


def test_camera_wall_output_is_captured_bounded_and_credential_redacted():
    from io import StringIO
    from services.gpu.runtime import live_launcher
    from services.shared.credential_redaction import CredentialRedactor

    redactor = CredentialRedactor()
    redactor.add_credentials(
        username="wall-user-marker", password="wall-password-marker",
        uri="rtsp://wall-user-marker:wall-password-marker@example.invalid/private",
    )

    class CameraWallProcess:
        stdout = StringIO(
            "Traceback (most recent call last):\n"
            "RuntimeError: Qt startup failed\n"
            "Qt warning: token=wall-password-marker\n"
        )

    diagnostics = live_launcher.CameraWallOutputDiagnostics(max_lines=2)
    live_launcher._pump_camera_wall_lines(CameraWallProcess(), redactor, diagnostics)
    lines = diagnostics.snapshot()

    assert len(lines) == 2
    assert lines[0] == "RuntimeError: Qt startup failed"
    assert lines[1] == "Qt warning: token=<redacted>"
    assert "wall-password-marker" not in "\n".join(lines)


def test_child_diagnostics_keep_bounded_tail_and_event_lines():
    from services.gpu.runtime.live_launcher import ChildOutputDiagnostics

    diagnostics = ChildOutputDiagnostics(
        max_tail=2, max_events=1, max_errors=1, max_line_chars=20,
    )
    diagnostics.ingest("ordinary line 000000000000")
    diagnostics.ingest("WARNING source 1")
    diagnostics.ingest("ordinary line 2")
    diagnostics.ingest("ERROR source 2 failure with a long line")
    snapshot = diagnostics.snapshot()

    assert len(snapshot["recent_child_output"]) == 2
    assert len(snapshot["diagnostic_events"]) == 1
    assert len(snapshot["gstreamer_errors"]) == 1
    assert all(len(line) <= 20 for line in snapshot["recent_child_output"])


def test_native_freshness_uses_oldest_required_stage():
    import json

    mon = SourceFpsMonitor(
        min_fps=5.0,
        max_age_sec=15.0,
        min_consecutive=2,
    )

    now_ns = 100_000_000_000

    for source_id in range(6):
        decode_age = 16 if source_id == 0 else 1

        sample = {
            "source_id": source_id,
            "camera_id": f"CAM-{source_id + 1:02d}",
            "monotonic_ns": now_ns,
            "fatal_metadata": False,
            "stages": [
                {
                    "count": 20,
                    "last_ns": now_ns - decode_age * 1_000_000_000,
                },
                {},
                {},
                {
                    "count": 20,
                    "last_ns": now_ns - 1_000_000_000,
                },
            ],
        }

        assert mon.ingest(
            "SIX_PROGRESS " + json.dumps(sample),
            now=1.0,
        )

    snap = mon.snapshot(now=1.0)

    assert snap["camera_stage_age_seconds"]["CAM-01"] == 16.0
    assert snap["camera_stage_age_seconds"]["CAM-02"] == 1.0
    assert snap["fresh"] is False

    # Future stage timestamps must remain invalid.
    sample["source_id"] = 0
    sample["camera_id"] = "CAM-01"
    sample["stages"][0]["last_ns"] = now_ns + 1

    assert not mon.ingest(
        "SIX_PROGRESS " + json.dumps(sample),
        now=2.0,
    )
