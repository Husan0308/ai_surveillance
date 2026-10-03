import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from scripts.run_v13_room_candidate import stage_v13, V13, V13_SHA, verify_pose, sidecar_environment
from scripts.run_v13_room_candidate import verify_video_contract, complete_native_frames
from scripts.dev_room_mv3dt.verify_validated_assets import sha256
from scripts.run_room_candidate import ROOT
from scripts.v13_monitoring import MixedMonitoring, PAIR
from scripts.validate_v13_full_stack import (
    latency, clean_child_shutdown, current_websocket_samples, surveillance_ports_released,
)


def test_surveillance_port_release_checks_listeners_not_time_wait():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    deployment = SimpleNamespace(api_host="127.0.0.1", api_port=port,
                                 ml_host="127.0.0.1", ml_port=port)
    try:
        assert not surveillance_ports_released(deployment)
    finally:
        listener.close()
    assert surveillance_ports_released(deployment)


def test_exact_v13_copy_preserves_all_scored_parameters(tmp_path):
    stage = stage_v13(tmp_path, "replay")
    assert sha256(stage / "config_tracker.yml") == V13_SHA == sha256(V13)
    assert (stage / "config_tracker.yml").read_bytes() == V13.read_bytes()
    app = (stage / "config_deepstream.txt").read_text()
    assert "[tracker]\nenable=1" in app
    assert "[sink3]\nenable=1" in app
    assert "PeopleNet" not in app
    assert "batched-push-timeout=-1" in app
    assert "NvDCF_stable_person" not in app
    detector = (stage / "config_pgie.txt").read_text()
    assert detector.replace("batch-size=2", "batch-size=6", 1) == (ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt").read_text()


def test_pose_runtime_guard_rejects_changed_engine(tmp_path):
    engine = tmp_path / "pose.engine"
    engine.write_bytes(b"actual")
    path = tmp_path / "build.json"
    path.write_text(json.dumps({"engine": str(engine), "engine_sha256": "not-the-same"}))
    with pytest.raises(ValueError, match="pose engine"):
        verify_pose(path)


def test_sidecar_imports_are_explicit_and_never_global(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/untrusted/global")
    monkeypatch.setenv("PYTHONHOME", "/untrusted/home")
    env = sidecar_environment()
    assert env["PYTHONPATH"] == f"{ROOT}:{ROOT / 'services/mv3dt_room'}"
    assert "Trash" not in env["PYTHONPATH"]
    assert "PYTHONHOME" not in env
    assert env["PYTHONNOUSERSITE"] == "1"


def test_mixed_adapter_rejects_second_pair_owner(tmp_path):
    import time
    (tmp_path / "preview").mkdir()
    (tmp_path / "preview/camera-health.json").write_text(json.dumps({"CAM-01": {}}))
    monitor = MixedMonitoring(tmp_path, time.monotonic_ns())
    try:
        with pytest.raises(ValueError, match="never own"):
            monitor.snapshot()
    finally:
        monitor.close()


def test_mixed_adapter_uses_frozen_preview_freshness_without_rtsp(tmp_path, monkeypatch):
    import time
    from services.camera_v11.monitoring_telemetry_ipc_v1 import offline_snapshot
    started = time.monotonic_ns() - 1_000_000
    monitor = MixedMonitoring(tmp_path, started)
    monkeypatch.setattr(monitor.bridge, "snapshot", offline_snapshot)
    snapshot = monitor.snapshot()
    assert snapshot["runtime"]["native_analytics_ready"] is False
    assert all(not row["online"] for row in snapshot["cameras"])
    assert all(row["source_fps"] is None for row in snapshot["cameras"])
    monitor.close()


def test_old_native_health_cannot_create_current_analytics(tmp_path, monkeypatch):
    import time
    from services.camera_v11.monitoring_telemetry_ipc_v1 import offline_snapshot
    folder = tmp_path / "room-pair/live-current"
    folder.mkdir(parents=True)
    started = time.monotonic_ns()
    (folder / "live-health.json").write_text(json.dumps({"mono_ns": started - 1,
        "readiness": {"ready": True, "sources": []}, "decoder": []}))
    monitor = MixedMonitoring(tmp_path, started)
    monkeypatch.setattr(monitor.bridge, "snapshot", offline_snapshot)
    assert monitor.snapshot()["runtime"]["native_analytics_ready"] is False
    assert json.loads((tmp_path / "merged-stats.json").read_text()) == {}
    monitor.close()


def test_latency_rejects_invalid_chain_without_removing_row(tmp_path):
    path = tmp_path / "paint.jsonl"
    path.write_text('{"camera_id":"CAM-01"}\nnot-json\n')
    report = latency(path)
    assert report["status"] == "FAIL"
    assert len(report["invalid_rows"]) == 2
    assert all(c["samples"] == 0 and not c["pass"] for c in report["cameras"].values())


def test_latency_keeps_slow_frames_and_requires_every_camera(tmp_path):
    path = tmp_path / "paint.jsonl"
    rows = []
    for frame in range(200):
        base = 1_000_000_000 + frame * 50_000_000
        rows.append({"camera_id": "CAM-01", "sequence": frame + 1,
            "t0_decoder_reference_monotonic_ns": base,
            "t1_decoder_out_monotonic_ns": base + 1_000_000,
            "t6_ipc_publish_monotonic_ns": base + 2_000_000,
            "t7_ui_receive_monotonic_ns": base + 3_000_000,
            "t8_ui_paint_monotonic_ns": base + 100_000_000})
    path.write_text("\n".join(map(json.dumps, rows)))
    report = latency(path)
    assert report["status"] == "FAIL"
    assert report["cameras"]["CAM-01"]["samples"] == 200
    assert report["cameras"]["CAM-01"]["timing_ms"]["total"]["p95"] == 100


def test_live_candidate_does_not_use_frozen_camera_only_launcher():
    script = (ROOT / "scripts/validate_v13_full_stack.py").read_text()
    assert '"scripts.run_v13_room_candidate"' in script
    assert '"scripts.run_preview_foundation"' in script
    assert '"--cameras", ",".join(OTHER)' in script
    assert 'SURVEILLANCE_ANALYTICS_ENABLED="1"' in script
    assert "deployment = DEPLOYMENT" in script
    assert "production_promoted\": False" in script
    assert script.index("FRONTEND_SETTINGS = load_frontend_settings()") < script.index("from scripts.freeze_app_contract")
    assert 'FRONTEND_FRAME_REFRESH_INTERVAL_MS=str(FRONTEND_SETTINGS.frame_refresh_interval_ms)' in script


def test_complete_empty_input_and_native_decode_are_both_mandatory():
    verify_video_contract("empty-room", {"nb_frames": "12031", "nb_read_packets": "12031"})
    with pytest.raises(ValueError, match="complete archived"):
        verify_video_contract("empty-room", {"nb_frames": "12031", "nb_read_packets": "12030"})
    frames = {"CAM-01": {"pgie_frames": 12031, "tracker_frames": 12031},
              "CAM-04": {"pgie_frames": 12031, "tracker_frames": 12031}}
    assert complete_native_frames({"cameras": frames}, "empty-room")
    frames["CAM-04"]["tracker_frames"] = 12030
    assert not complete_native_frames({"cameras": frames}, "empty-room")
    assert not complete_native_frames({"cameras": {}}, "empty-room")


def test_visible_body_review_never_reuses_one_box_for_two_people():
    from scripts.freeze_v13_live import match_references
    refs = [{"bbox": [10, 20, 30, 40]}, {"bbox": [12, 22, 30, 40]}]
    assert len(match_references(refs, [{"bbox": [10, 20, 30, 40]}])) == 1
    assert match_references(refs, [{"bbox": [400, 500, 30, 40]}]) == []


def test_expected_uvicorn_signal_exit_requires_owned_lifespan_completion():
    import signal
    log = "Application shutdown complete.\nFinished server process [1234]"
    assert clean_child_shutdown("api", -signal.SIGTERM, 1234, log)
    assert clean_child_shutdown("ml", -signal.SIGTERM, 1234, log)
    assert not clean_child_shutdown("api", -signal.SIGTERM, 9999, log)
    assert not clean_child_shutdown("api", -signal.SIGTERM, 1234, "Shutting down")
    assert not clean_child_shutdown("native", 137, 1234, log)
    assert not clean_child_shutdown("api", 0, 1234, log, forced=True)


def test_websocket_acceptance_does_not_pause_current_producer_observation():
    import asyncio
    class Bridge:
        sequence = 0
        def snapshot(self):
            self.sequence += 1
            return {"sequence": self.sequence}
    class Writer:
        rows = []
        def publish(self, row):
            self.rows.append(row)
    bridge, writer = Bridge(), Writer()
    async def collect(url):
        result = []
        for _ in range(3):
            await asyncio.sleep(.03)
            result.append(writer.rows[-1])
        return result
    rows = asyncio.run(current_websocket_samples("test", bridge, writer, collect, .01))
    assert len({row["sequence"] for row in rows}) == 3


def test_debugger_only_changes_container_inspection_permissions():
    from scripts.native_shutdown_diagnostics import debugger_command
    command = ["docker", "run", "-v", "models:/models:ro", "image", "/workspace/proto-app", "-c", "config"]
    updated = debugger_command(command, "image")
    assert command == ["docker", "run", "-v", "models:/models:ro", "image", "/workspace/proto-app", "-c", "config"]
    assert updated[-4:] == command[-4:]
    assert "--privileged" not in updated
    assert "--cap-add=SYS_PTRACE" in updated


def test_mutex_diagnostic_is_read_only_and_does_not_claim_register_guess_is_owner():
    from scripts.native_shutdown_diagnostics import debugger_arguments, MUTEX_DIAGNOSTIC
    args = debugger_arguments()
    attach_at = args.index("attach 1")
    info_at = args.index("info threads")
    backtrace_at = args.index("thread apply all bt")
    assert attach_at < info_at < backtrace_at
    assert args[-1] == "detach"
    assert "frame-arguments none" in " ".join(args)
    assert "UNPROVEN_UNTIL_ARGUMENT_OR_DISASSEMBLY_MATCH" in MUTEX_DIAGNOSTIC
    assert "F5_MUTEX_OWNER_CONFIRMED" in MUTEX_DIAGNOSTIC
    assert "F5_MUTEX_OWNER_INFERRED_GLIBC_X86_64" in MUTEX_DIAGNOSTIC
    assert "thread_by_lwp" in MUTEX_DIAGNOSTIC
    assert "source == \"register_rdi\"" in MUTEX_DIAGNOSTIC
    assert "__owner" in MUTEX_DIAGNOSTIC
    assert "owner_thread_matches" in MUTEX_DIAGNOSTIC
    assert "read_memory" in MUTEX_DIAGNOSTIC
    assert "write_memory" not in MUTEX_DIAGNOSTIC
    assert "inferior().call" not in MUTEX_DIAGNOSTIC
    assert "read_var(\"mutex\")" in MUTEX_DIAGNOSTIC


def test_deficit_diagnosis_preserves_missing_proposals_and_unknown_private_reason():
    from scripts.diagnose_v13_live import window_evidence
    proposals = [{"camera": "CAM-04", "frame": f, "bbox_xywh": [10, 20, 100, 200],
                  "tracker_association_emitted": f != 2, "native_track_id": "4" if f != 2 else None}
                 for f in (1, 2, 3)]
    states = [{"camera_id": "CAM-04", "frame": "2", "native_track_id": "4", "nvdcf_state": "INACTIVE"}]
    result = window_evidence("CAM-04", {"start": 2, "end": 2}, proposals, states)
    item = result["unassociated_proposals"][0]
    assert item["proposal"] is proposals[1]
    assert item["previous_associated_proposal"]["native_track_id"] == "4"
    assert item["first_geometrically_matching_return"]["native_track_id"] == "4"
    assert item["classification"] == "inactive_target_during_association_deficit"
    assert "NOT_EXPORTED" in item["private_rejection_reason"]
    assert len(proposals) == 3


def test_shutdown_experiment_only_changes_explicit_live_teardown_points():
    from scripts.build_v13_shutdown_candidate import (
        shutdown_source, native_shutdown_source, EOS, LIVE_CLOSE,
        NATIVE_DESTROY, NATIVE_DESTROY_FLUSH,
    )
    text = "prefix\n" + EOS + "\nsuffix"
    changed = shutdown_source(text)
    assert changed.replace(LIVE_CLOSE, EOS) == text
    assert "NV_DS_SOURCE_RTSP" in LIVE_CLOSE
    assert "if (!live_close)" in LIVE_CLOSE
    assert LIVE_CLOSE.count("gst_element_send_event(appCtx->pipeline.pipeline, gst_event_new_eos());") == 1
    assert "sleep (1);" in LIVE_CLOSE

    native = native_shutdown_source(NATIVE_DESTROY)
    assert native == NATIVE_DESTROY_FLUSH
    assert "NV_DS_SOURCE_RTSP" in native
    assert "gst_event_new_flush_start" in native
    assert "gst_event_new_flush_start ())) {" in native
    assert "gst_event_new_flush_start ()))) {" not in native
    assert "FLUSH_START completed" in native
    assert "sleep (" not in native
    assert native.count("destroy_pipeline (appCtx[i]);") == 1

    with pytest.raises(ValueError, match="exact SDK"):
        shutdown_source("different source")
    with pytest.raises(ValueError, match="exact native"):
        native_shutdown_source("different source")


def test_drain_builder_requires_all_admitted_pts_to_reach_tracker_before_flush():
    from scripts.build_v13_drain_shutdown_candidate import (
        drain_source, drained, NATIVE_DESTROY_DRAIN, C_DRAIN,
    )
    complete = {"blocked": True, "last_admitted_pts": 120,
        "mux_last_pts": 120, "pgie_last_pts": 120, "tracker_last_pts": 120,
        "pgie_frames": 20, "tracker_frames": 20}
    assert drained(complete)
    assert not drained(dict(complete, tracker_last_pts=0, tracker_frames=19))
    assert "gst_element_get_static_pad (bin->bin, \"src\")" in C_DRAIN
    assert "GST_PAD_PROBE_TYPE_IDLE" in C_DRAIN
    assert "last_pts[SOURCE_HEALTH_MUX] == source->last_admitted_pts" in C_DRAIN
    assert "last_pts[SOURCE_HEALTH_PGIE] == source->last_admitted_pts" in C_DRAIN
    assert "last_pts[SOURCE_HEALTH_TRACKER] == source->last_admitted_pts" in C_DRAIN
    assert "admission_buffers_since_barrier" in C_DRAIN
    assert "last_admitted_source_frame" not in C_DRAIN
    assert "g_usleep (1000)" in C_DRAIN
    assert "g_get_monotonic_time () < deadline" in C_DRAIN
    assert NATIVE_DESTROY_DRAIN.index("f5_drain_admitted_frames") < NATIVE_DESTROY_DRAIN.index("gst_event_new_flush_start")
    failed = NATIVE_DESTROY_DRAIN.split("} else {", 1)[1]
    assert "return_value = -1" in failed
    assert "gst_event_new_flush_start" not in failed
    staged = drain_source((ROOT / "services/mv3dt_room/native/deepstream_test5_app_main.c").read_text())
    assert staged.count("f5_drain_admitted_frames (appCtx[i])") == 1
    assert staged.count("f5_release_drain_probes ();") == 1
    assert "last_pts[stage_index] = frame_meta->buf_pts" in staged


def test_drain_log_audit_requires_ordered_events_and_equal_terminal_pts(tmp_path):
    from scripts.run_v13_room_candidate import audit_shutdown_drain
    path = tmp_path / "shutdown-drain.jsonl"
    names = ("shutdown_requested", "source_admission_stopped", "drain_completed",
             "flush_start", "destroy_pipeline", "native_exit")
    source = {"blocked": True, "last_admitted_pts": 7, "mux_last_pts": 7,
        "pgie_last_pts": 7, "tracker_last_pts": 7, "pgie_frames": 9, "tracker_frames": 9}
    path.write_text("\n".join(json.dumps({"event": name, "mono_ns": i + 10,
        "sources": [source, source]}) for i, name in enumerate(names)))
    assert audit_shutdown_drain(path)["status"] == "PASS"
    bad = dict(source, tracker_last_pts=6)
    path.write_text("\n".join(json.dumps({"event": name, "mono_ns": i + 10,
        "sources": [source, bad]}) for i, name in enumerate(names)))
    assert audit_shutdown_drain(path)["status"] == "FAIL"
    path.write_text("\n".join(json.dumps({"event": name, "mono_ns": i + 10,
        "sources": [source, source]}) for i, name in enumerate(reversed(names))))
    assert audit_shutdown_drain(path)["status"] == "FAIL"


def test_drain_build_verifier_is_selected_without_weakening_existing_candidate_verifier(tmp_path):
    from scripts.run_v13_room_candidate import verify_native_build

    record = {"candidate": "F5_DRAIN_BEFORE_FLUSH"}
    path = tmp_path / "build.json"
    path.write_text(json.dumps(record))
    # The drain-specific verifier is selected; it fails on its provenance checks,
    # rather than passing this record through the older candidate schema.
    with pytest.raises(ValueError, match="remain experimental"):
        verify_native_build(path)


def test_regular_staged_build_uses_existing_candidate_verifier(tmp_path, monkeypatch):
    from scripts import build_v13_shutdown_candidate
    from scripts.run_v13_room_candidate import verify_native_build

    path = tmp_path / "build.json"
    path.write_text(json.dumps({"candidate": "F5_UI_DIAGNOSTICS"}))
    expected = {"candidate": "verified-existing-schema"}
    monkeypatch.setattr(build_v13_shutdown_candidate, "verify_candidate",
                        lambda supplied: expected if supplied == path else None)

    assert verify_native_build(path) == expected


def test_drain_shutdown_audit_is_live_only_and_replay_behavior_is_unchanged():
    from scripts.run_v13_room_candidate import shutdown_drain_required

    candidate = {"candidate": "F5_DRAIN_BEFORE_FLUSH"}
    assert shutdown_drain_required("live", candidate)
    assert not shutdown_drain_required("canonical", candidate)
    assert not shutdown_drain_required("person-present", candidate)
    assert not shutdown_drain_required("empty-room", candidate)


def test_final_gate_requires_same_binary_and_fresh_gallery_in_every_run():
    import copy
    from scripts.freeze_v13_live import same_candidate_chain
    first = {"native": {"binary_sha256": "binary", "source_sha256": "source"},
             "tracker_sha256": "tracker", "pose": {"engine_sha256": "pose"},
             "configs": {"config_pgie.txt": "pgie"}, "gallery_preexisting": False, "gallery": "one"}
    second = copy.deepcopy(first)
    second["gallery"] = "two"
    assert same_candidate_chain([first, second])
    second["native"]["binary_sha256"] = "different"
    assert not same_candidate_chain([first, second])
    second["native"]["binary_sha256"] = "binary"
    second["gallery"] = "one"
    assert not same_candidate_chain([first, second])
    drain_first = {"native": {"binary_sha256": "binary", "staged_source_sha256": "source"},
             "tracker_sha256": "tracker", "pose": {"engine_sha256": "pose"},
             "configs": {"config_pgie.txt": "pgie"}, "gallery_preexisting": False, "gallery": "one"}
    drain_second = copy.deepcopy(drain_first)
    drain_second["gallery"] = "two"
    assert same_candidate_chain([drain_first, drain_second])
    drain_second["native"]["staged_source_sha256"] = "different"
    assert not same_candidate_chain([drain_first, drain_second])
    assert not same_candidate_chain([])


def test_empty_live_room_is_unmeasured_not_perfect_retention():
    from scripts.validate_v13_full_stack import retention_accepted
    assert not retention_accepted(None, 0)
    assert not retention_accepted(float("nan"), 0)
    assert not retention_accepted(99.5, 50)
    assert retention_accepted(99.5, 2)


@pytest.mark.parametrize("override,expected", [(None, (16, 200)), ((19, 222), (19, 222))])
def test_candidate_preserves_frontend_settings_before_credential_dotenv(override, expected):
    import os
    import subprocess
    import sys
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    for key in ("FRONTEND_FRAME_REFRESH_INTERVAL_MS", "FRONTEND_REFRESH_INTERVAL_MS"):
        env.pop(key, None)
    if override:
        env["FRONTEND_FRAME_REFRESH_INTERVAL_MS"], env["FRONTEND_REFRESH_INTERVAL_MS"] = map(str, override)
    command = [sys.executable, "-B", "-c", "from scripts.validate_v13_full_stack import FRONTEND_SETTINGS; import json; print(json.dumps([FRONTEND_SETTINGS.frame_refresh_interval_ms, FRONTEND_SETTINGS.refresh_interval_ms]))"]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert tuple(json.loads(result.stdout)) == expected


def test_teardown_cycle_gate_requires_clean_exit_and_no_leaks():
    from scripts.run_v13_teardown_cycles import cycle_pass, safe_to_continue_after_failure
    row = {
        "runner_returncode": 0,
        "result": {
            "status": "PASS", "native_exit": 0, "identity_exit": 0, "capture_exit": 0,
            "retention": {"cameras": {
                "CAM-01": {"association_retention_gate": True,
                           "association_deficit_windows": [{"frames": 5}]},
                "CAM-04": {"association_retention_gate": True,
                           "association_deficit_windows": [{"frames": 1}]},
                }},
                "shutdown_drain": {"status": "PASS", "sources": [
                    {"pgie_frames": 12, "tracker_frames": 12},
                    {"pgie_frames": 12, "tracker_frames": 12},
                ]},
            },
        "shutdown_backtrace_present": False,
        "rtsp_sockets_after": [],
        "candidate_containers_after": [],
    }
    assert cycle_pass(row)
    assert safe_to_continue_after_failure(row)
    row["result"]["shutdown_drain"]["sources"][1]["tracker_frames"] = 11
    assert not cycle_pass(row)
    row["result"]["shutdown_drain"]["sources"][1]["tracker_frames"] = 12
    for key, bad in (
        ("runner_returncode", 1),
        ("shutdown_backtrace_present", True),
        ("rtsp_sockets_after", ["socket"]),
        ("candidate_containers_after", ["container"]),
    ):
        changed = dict(row)
        changed[key] = bad
        assert not cycle_pass(changed)


    too_long = json.loads(json.dumps(row))
    too_long["result"]["retention"]["cameras"]["CAM-04"]["association_deficit_windows"] = [{"frames": 11}]
    assert not cycle_pass(too_long)

    retention_fail = json.loads(json.dumps(row))
    retention_fail["result"]["retention"]["cameras"]["CAM-01"]["association_retention_gate"] = False
    assert not cycle_pass(retention_fail)
