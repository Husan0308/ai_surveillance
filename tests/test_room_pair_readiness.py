from __future__ import annotations

import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from services.mv3dt_room.global_identity_manager import GlobalIdentityManager
from services.mv3dt_room.identity_metrics import IdentityMetrics, monotonic_ns
from services.mv3dt_room.publication_readiness import PublicationBarrier, read_readiness
from services.mv3dt_room.room_pair_state import RoomPairState


def health(ready=True, *, now_ms=None):
    return {
        "ready": ready, "status": "ready" if ready else "not_ready",
        "updated_epoch_ms": time.time() * 1000 if now_ms is None else now_ms,
        "sources": [{"source_id": source, **{
            name: value for stage in ("mux", "pgie", "tracker")
            for name, value in ((f"{stage}_seen", True), (f"last_{stage}_age_ms", 0))
        }} for source in (0, 1)],
    }


def write_health(root, ready=True):
    path = root / "run/logs/probe/readiness.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(health(ready)))


@pytest.fixture
def worker(tmp_path, monkeypatch):
    # Exercise the actual production writer without opening CUDA or a camera.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "services/mv3dt_room"))
    from services.mv3dt_room.live_identity_worker import LiveIdentityWorker, TrackState

    value = LiveIdentityWorker.__new__(LiveIdentityWorker)
    value.output_dir = tmp_path / "identity-live"
    value.output_dir.mkdir()
    value.args = Namespace(source_mode="live", session_id="test")
    value.publication = PublicationBarrier()
    value.metrics = IdentityMetrics()
    value.manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    value.public_ids = {}
    value.next_public = 1
    value.output_rows = []
    value.publication_events = []
    value.manager_event_cursor = value.event_row_cursor = 0
    value.last_readiness_check_ns = 0
    value.latest_camera_frame = {"CAM-01": 1, "CAM-04": -1}
    value.latest_camera_rows = {"CAM-01": [], "CAM-04": []}
    value.last_published_presence = set()
    value.track_states = {}
    value._audit = lambda *args, **kwargs: None
    value._track_state_class = TrackState
    return value


def observe(worker, *, frame=1, native=1, internal=None, pending=False, receive_ns=None):
    row = {"camera_id": "CAM-01", "frame": frame, "native_track_id": native,
           "receive_monotonic_ns": monotonic_ns() if receive_ns is None else receive_ns,
           "bbox": [10, 20, 100, 200], "world": [1.0, 2.0], "confidence": 0.9}
    if internal is None:
        internal = worker.manager.confirm_new(row, np.asarray([1, 0], np.float32), "test")
    state = worker._track_state_class("CAM-01", native, latest=row,
                                      canonical_internal_identity=None if pending else internal,
                                      identity_state="PENDING" if pending else "KNOWN")
    worker.track_states[("CAM-01", native)] = state
    worker.latest_camera_rows["CAM-01"] = [row]
    worker.latest_camera_frame["CAM-01"] = frame
    return row, internal


def state_of(worker):
    return json.loads((worker.output_dir / "current_state.json").read_text())


def test_warming_keeps_production_empty_and_has_no_deferred_event_backlog(worker, tmp_path):
    write_health(tmp_path, False)
    row, internal = observe(worker)
    for _ in range(2000):
        worker._record_identity_row({**row, "canonical_internal_identity": internal,
                                    "identity_state": "KNOWN"})
        worker._collect_publication_events()
    worker._publish_state()
    assert state_of(worker)["people"] == []
    assert state_of(worker)["publication"]["status"] == "WARMING"
    assert worker.output_rows == worker.publication_events == []
    assert worker.publication.snapshot()["withheld_event_queue_depth"] == 0


def test_ready_begins_with_new_observations_not_warming_state_or_events(worker, tmp_path):
    write_health(tmp_path, False)
    old, internal = observe(worker)
    worker._record_identity_row({**old, "canonical_internal_identity": internal,
                                "identity_state": "KNOWN"})
    worker._collect_publication_events()
    write_health(tmp_path)
    worker._publish_state()
    assert state_of(worker)["people"] == []
    assert worker.publication_events == []
    current, _ = observe(worker, frame=2, internal=internal)
    worker._record_identity_row({**current, "canonical_internal_identity": internal,
                                "identity_state": "KNOWN"})
    worker._publish_state()
    assert [row["application_id"] for row in state_of(worker)["people"]] == ["Person_01"]
    assert worker.manager.next_number == 2


def test_ready_loss_clears_presence_and_recovery_reuses_gallery_without_ghost(worker, tmp_path):
    write_health(tmp_path)
    worker._publish_state()
    _, internal = observe(worker)
    worker._publish_state()
    assert len(state_of(worker)["people"]) == 1
    write_health(tmp_path, False)
    worker._publish_state()
    assert state_of(worker)["people"] == []
    assert worker.last_published_presence == set()
    write_health(tmp_path)
    worker._publish_state()
    assert state_of(worker)["people"] == []
    observe(worker, frame=3, internal=internal)
    worker._publish_state()
    assert [row["application_id"] for row in state_of(worker)["people"]] == ["Person_01"]
    assert worker.manager.next_number == 2


def test_pending_is_not_published_as_trustworthy_presence(worker, tmp_path):
    write_health(tmp_path)
    worker._publish_state()
    observe(worker, pending=True)
    worker._publish_state()
    assert state_of(worker)["people"] == []


def test_batch_events_are_published_for_both_accepted_rows_not_just_first(worker, tmp_path):
    write_health(tmp_path)
    worker._publish_state()
    for native in (1, 2):
        row, internal = observe(worker, native=native)
        worker._record_identity_row({**row, "canonical_internal_identity": internal,
                                    "identity_state": "NEW_CONFIRMED"})
    worker._collect_publication_events()
    assert len([event for event in worker.publication_events if event["event"] == "CREATE"]) == 2


def test_stale_reid_completion_does_not_publish_old_warming_observation(worker, tmp_path):
    old, internal = observe(worker)
    write_health(tmp_path)
    worker._publish_state()
    worker._record_identity_row({**old, "canonical_internal_identity": internal,
                                "identity_state": "NEW_CONFIRMED"})
    worker._collect_publication_events()
    assert worker.output_rows == worker.publication_events == []


def test_ready_guard_and_merge_events_are_not_hidden_by_publication_filter(worker, tmp_path):
    write_health(tmp_path)
    worker._publish_state()
    row, internal = observe(worker)
    worker.manager.events.extend([
        {"event": "REASSIGN_REJECTED", "frame": row["frame"], "camera_id": row["camera_id"],
         "native_track_id": row["native_track_id"]},
        {"event": "MERGE", "frame": row["frame"], "kept_global_person_id": internal,
         "removed_global_person_id": 2},
    ])
    worker._record_identity_row({**row, "canonical_internal_identity": internal,
                                "identity_state": "KNOWN"})
    worker._collect_publication_events()
    assert {event["event"] for event in worker.publication_events} >= {"MERGE", "REASSIGN_REJECTED"}


def test_native_readiness_missing_incomplete_and_stale_fail_closed(tmp_path):
    assert not read_readiness(tmp_path)["ready"]
    path = tmp_path / "run/logs/probe/readiness.json"
    path.parent.mkdir(parents=True)
    payload = health(now_ms=10_000)
    path.write_text(json.dumps(payload))
    assert read_readiness(tmp_path, now_epoch_ms=14_999)["ready"]
    assert not read_readiness(tmp_path, now_epoch_ms=15_001)["ready"]
    payload["sources"][1]["tracker_seen"] = False
    path.write_text(json.dumps(payload))
    assert not read_readiness(tmp_path, now_epoch_ms=10_000)["ready"]
    payload["sources"] = []
    path.write_text(json.dumps(payload))
    assert not read_readiness(tmp_path, now_epoch_ms=10_000)["ready"]


def test_api_hides_stale_presence_even_if_worker_is_frozen(worker, tmp_path):
    write_health(tmp_path)
    worker._publish_state()
    observe(worker)
    worker._publish_state()
    api = RoomPairState(tmp_path)
    assert len(api.snapshot()["people"]) == 1
    write_health(tmp_path, False)
    snapshot = api.snapshot()
    assert snapshot["people"] == []
    assert snapshot["presence"]["rendered_ids"] == []
    assert not snapshot["readiness"]["ready"]


def test_barrier_never_controls_preview_and_opens_exactly_once_per_epoch(tmp_path):
    from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader, PreviewFrameWriter

    barrier = PublicationBarrier()
    writer = PreviewFrameWriter(str(tmp_path / "preview.bin"), width=16, height=8)
    reader = PreviewFrameReader(str(tmp_path / "preview.bin"))
    try:
        for index, ready in enumerate((False, True, True, False, True), 1):
            transition = barrier.update(health(ready), index)
            writer.publish(bytes([index] * 16 * 8 * 4), object_count=0)
            assert reader.read_latest().payload[0] == index
            if index == 3:
                assert transition is None
        assert barrier.epoch == 3
        assert not barrier.allows({"receive_monotonic_ns": 4})
        assert barrier.allows({"receive_monotonic_ns": 5})
    finally:
        reader.close()
        writer.close()


def test_runner_enables_native_health_without_a_diagnostic_flag(tmp_path, monkeypatch):
    from scripts.dev_room_mv3dt.run_room_pair import deepstream_command
    monkeypatch.delenv("MV3DT_SOURCE_HEALTH_DIR", raising=False)
    (tmp_path / "config_deepstream.txt").write_text("[source0]\nenable=1\n[source1]\nenable=1\n")
    (tmp_path / "config_msgconv.txt").write_text("[sensor0]\nid=CAM-01\n[sensor1]\nid=CAM-04\n")
    assert "MV3DT_SOURCE_HEALTH_DIR=/workspace/experiments/logs/probe" in deepstream_command(
        tmp_path, tmp_path / "binary", "image", "live", "test"
    )

def test_native_recovery_requires_sustained_all_stage_progress_and_retries_safely():
    source = (
        Path(__file__).resolve().parents[1]
        / "services/mv3dt_room/native/deepstream_test5_app_main.c"
    ).read_text()

    assert "SOURCE_HEALTH_RECOVERY_MIN_FRAMES 5" in source
    assert "if (health->recovery_active)\n      return FALSE;" in source
    assert "health->recovery_start_frames[recovery_stage]" in source
    assert "health->frames[recovery_stage] <" in source
    assert "SOURCE_HEALTH_RETRY_SEC * G_USEC_PER_SEC" in source
    assert "src_bin->reconfiguring ? \"RECONFIGURING\" : \"RECONNECTING\"" in source
    assert "g_timeout_add (0, reset_source_pipeline, src_bin);" in source
    assert "health->recovery_active && !src_bin->reconfiguring" not in source


def test_native_recovery_does_not_reset_sources_while_parent_pipeline_is_paused():
    source = (
        Path(__file__).resolve().parents[1]
        / "services/mv3dt_room/native/deepstream_test5_app_main.c"
    ).read_text()

    assert "pipeline_playing = pipeline_state == GST_STATE_PLAYING;" in source
    assert "health->recovery_active && mux_stalled && pipeline_playing" in source

