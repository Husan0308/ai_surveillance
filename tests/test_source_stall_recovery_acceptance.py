from scripts.run_source_stall_recovery_acceptance import (
    all_camera_feeds_ready, capture_recovered_preview, event_detail_int, planned_recovery_complete,
    preview_pair_matches, read_latest_fresh, recovery_cycles, recovery_verifier_environment,
    unexpected_recovery_events,
)
from scripts.run_six_camera_v13_wall_candidate import parse_source_stall_test_plan
from types import SimpleNamespace


def test_acceptance_plan_has_three_cam02_cycles_then_cam03():
    plan = parse_source_stall_test_plan("CAM-02@30,CAM-02@110,CAM-02@190,CAM-03@270", "1")
    assert [item["camera_id"] for item in plan] == ["CAM-02", "CAM-02", "CAM-02", "CAM-03"]
    assert [item["at_sec"] for item in plan] == [30, 110, 190, 270]


def test_recovery_uses_active_only_fail_closed_verifier_for_896_candidate():
    env = recovery_verifier_environment({
        "detector_profile": {"name": "yolo26m-fp16-896-context-verified"}
    })
    assert env == {
        "FULL_STACK_LOWCONF_VERIFIER": "0",
        "FULL_STACK_ACTIVE_TRACK_VERIFIER": "1",
        "FULL_STACK_YOLO_CONTEXT_VERIFIER": "0",
    }


def test_recovery_uses_dual_context_verifier_for_cadence_configured_896_cascade_candidate():
    env = recovery_verifier_environment({
        "detector_profile": {
            "name": "yolo26m-fp16-640-primary-896-aux-interval-2-context-verified",
            "aux_config": "config_infer_aux_yolo26m_896.txt",
            "aux_input": [896, 896],
            "aux_unique_id": 3,
            "aux_engine_sha256": "engine",
            "aux_parser_sha256": "parser",
            "aux_interval": 2,
        }
    })
    assert env == {
        "FULL_STACK_LOWCONF_VERIFIER": "0",
        "FULL_STACK_ACTIVE_TRACK_VERIFIER": "0",
        "FULL_STACK_YOLO_CONTEXT_VERIFIER": "1",
    }


def test_recovery_keeps_lowconf_mode_for_frozen_640_candidate():
    env = recovery_verifier_environment({})
    assert env["FULL_STACK_LOWCONF_VERIFIER"] == "1"
    assert env["FULL_STACK_ACTIVE_TRACK_VERIFIER"] == "0"


def test_recovery_event_audit_requires_remove_before_recreate_and_pass():
    names = ["TEST_SOURCE_STALL_BEGIN", "SOURCE_STALL_RECOVERY_BEGIN",
             "SOURCE_STALL_OLD_OWNER_REMOVED", "SOURCE_STALL_RECOVERY_RECONNECTED",
             "SOURCE_STALL_RECOVERY_PASS"]
    events = [{"camera_id": "CAM-02", "event": name, "detail": "recovery_count=1",
               "sample_index": index} for index, name in enumerate(names)]
    cycles = recovery_cycles(events, [{"camera_id": "CAM-02", "at_sec": 30}])
    assert cycles[0]["event_order_pass"] is True
    assert event_detail_int(events[-1], "recovery_count") == 1


def test_recovery_event_audit_rejects_wrong_camera_or_order():
    events = [
        {"camera_id": "CAM-02", "event": "TEST_SOURCE_STALL_BEGIN", "detail": ""},
        {"camera_id": "CAM-02", "event": "SOURCE_STALL_RECOVERY_BEGIN", "detail": ""},
        {"camera_id": "CAM-02", "event": "SOURCE_STALL_RECOVERY_RECONNECTED", "detail": ""},
        {"camera_id": "CAM-02", "event": "SOURCE_STALL_OLD_OWNER_REMOVED", "detail": ""},
        {"camera_id": "CAM-02", "event": "SOURCE_STALL_RECOVERY_PASS", "detail": ""},
    ]
    cycles = recovery_cycles(events, [{"camera_id": "CAM-02", "at_sec": 30}])
    assert cycles[0]["event_order_pass"] is False


def test_recovery_audit_never_uses_unplanned_camera_pass_to_finish_schedule():
    names = ["TEST_SOURCE_STALL_BEGIN", "SOURCE_STALL_RECOVERY_BEGIN",
             "SOURCE_STALL_OLD_OWNER_REMOVED", "SOURCE_STALL_RECOVERY_RECONNECTED",
             "SOURCE_STALL_RECOVERY_PASS"]
    events = [{"camera_id": "CAM-02", "event": name, "detail": "recovery_count=1",
               "sample_index": index} for index, name in enumerate(names)]
    events += [{"camera_id": "CAM-01", "event": name, "detail": "recovery_count=1",
                "sample_index": 10 + index} for index, name in enumerate(names[1:])]
    plan = [{"camera_id": "CAM-02", "at_sec": 30}, {"camera_id": "CAM-03", "at_sec": 110}]
    cycles = recovery_cycles(events, plan)
    assert cycles[0]["event_order_pass"] is True
    assert cycles[1]["event_order_pass"] is False
    assert planned_recovery_complete(events, plan) is False
    assert [row["camera_id"] for row in unexpected_recovery_events(events, cycles)] == [
        "CAM-01", "CAM-01", "CAM-01", "CAM-01"]


def test_recovery_schedule_completes_only_after_all_planned_cycles_pass():
    names = ["TEST_SOURCE_STALL_BEGIN", "SOURCE_STALL_RECOVERY_BEGIN",
             "SOURCE_STALL_OLD_OWNER_REMOVED", "SOURCE_STALL_RECOVERY_RECONNECTED",
             "SOURCE_STALL_RECOVERY_PASS"]
    events = []
    for camera, base in (("CAM-02", 0), ("CAM-03", 10)):
        events.extend({"camera_id": camera, "event": name, "detail": "recovery_count=1",
                       "sample_index": base + offset} for offset, name in enumerate(names))
    plan = [{"camera_id": "CAM-02", "at_sec": 30}, {"camera_id": "CAM-03", "at_sec": 110}]
    assert planned_recovery_complete(events[:-1], plan) is False
    assert planned_recovery_complete(events, plan) is True


def test_image_bbox_sync_uses_atomic_preview_embedded_metadata():
    frame = SimpleNamespace(
        source_frame_num=102,
        tracker_frame_num=101,
        pts_ns=5_100_000_000,
        tracker_snapshot={"source_frame_num": 102, "frame_num": 101,
                          "pts_ns": 5_100_000_000, "objects": []},
    )
    assert preview_pair_matches(frame) is True
    frame.tracker_snapshot["pts_ns"] += 1
    assert preview_pair_matches(frame) is False


def test_recovery_startup_readiness_requires_six_fresh_valid_preview_pairs():
    preview = {"age_ms": 40.0, "embedded_metadata_matches": True}
    base = {
        "online": True, "source_fps": 19.9,
        "decoded_frames_total": 10, "pgie_frames_total": 10,
        "canonical_tracker_updates_total": 10,
        "queue_depth": 0, "queue_high_water": 2, "preview": preview,
    }
    rows = {f"CAM-{index:02d}": dict(base) for index in range(1, 7)}
    assert all_camera_feeds_ready(rows) is True
    rows["CAM-04"]["preview"] = None
    assert all_camera_feeds_ready(rows) is False
    rows["CAM-04"]["preview"] = {"age_ms": 1001.0, "embedded_metadata_matches": True}
    assert all_camera_feeds_ready(rows) is False


def test_recovery_preview_capture_binds_pixels_to_embedded_same_frame_metadata():
    frame = SimpleNamespace(
        sequence=7,
        source_frame_num=102, tracker_frame_num=101, pts_ns=5_100_000_000,
        width=2, height=1, timestamp_ns=1, payload=bytes([20, 40]),
        tracker_snapshot={"source_frame_num": 102, "frame_num": 101,
                          "pts_ns": 5_100_000_000, "objects": []},
    )

    class Reader:
        @staticmethod
        def read_latest(**kwargs):
            assert kwargs == {"max_age_sec": 1.0, "metadata_only": False}
            return frame

    result = capture_recovered_preview("CAM-02", Reader())
    assert result is not None
    assert result["image_bbox_sync"] is True
    assert result["payload_all_zero"] is False
    assert result["source_frame_num"] == 102
    assert result["tracker_frame_num"] == 101
    frame.tracker_snapshot["frame_num"] = 99
    assert capture_recovered_preview("CAM-02", Reader())["image_bbox_sync"] is False


def test_recovery_audit_retries_transient_uncommitted_preview_read_and_still_fails_closed():
    frame = SimpleNamespace(sequence=3)

    class Reader:
        calls = 0

        def read_latest(self, **kwargs):
            assert kwargs == {"max_age_sec": 1.0, "metadata_only": True}
            self.calls += 1
            return frame if self.calls == 3 else None

    reader = Reader()
    result, attempts = read_latest_fresh(
        reader, metadata_only=True, attempts=4, retry_delay_sec=0)
    assert result is frame
    assert attempts == 3
    assert reader.calls == 3

    class InvalidReader:
        @staticmethod
        def read_latest(**kwargs):
            return None

    result, attempts = read_latest_fresh(
        InvalidReader(), metadata_only=True, attempts=3, retry_delay_sec=0)
    assert result is None
    assert attempts == 3
