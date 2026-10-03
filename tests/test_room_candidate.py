"""Fail-closed candidate staging; no cameras or production mutations."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.build_room_candidate import detector_only_source, diagnostic_source, OLD, NEW, verify_build
from scripts.run_room_candidate import ROOT, section_value, two_source_pgie, stage_profile, command


def test_detector_only_guard_only_affects_tracker_disabled():
    text = (ROOT / "services/mv3dt_room/native/deepstream_test5_app_main.c").read_text()
    result = detector_only_source(text)
    assert result.replace(NEW, OLD, 1) == text
    assert result.count(NEW) == 1
    assert "config.tracker_config.enable && !pn263_bbox_correction_attach" in result
    with pytest.raises(ValueError):
        detector_only_source(text + OLD)


def test_batch_two_changes_nothing_else():
    original = (ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt").read_text()
    staged = two_source_pgie(original)
    assert staged.replace("batch-size=2", "batch-size=6") == original
    for setting in ("pre-cluster-threshold=0.25", "nms-iou-threshold=0.45", "interval=0", "infer-dims=3;640;640"):
        assert setting in staged


def test_bbox_instrumentation_only_touches_existing_gated_logger():
    source = (ROOT / "services/mv3dt_room/native/deepstream_test5_app_main.c").read_text()
    staged = diagnostic_source(source)
    assert 'obj_meta->rect_params.width, obj_meta->rect_params.height);' in staged
    assert staged.count('\\"bbox\\":[%.3f,%.3f,%.3f,%.3f]') == 1
    assert 'if (!frame_audit_open () || !batch_meta)' in staged


def test_section_update_is_scoped_and_unique():
    before = "[tracker]\nenable=1\nfoo=2\n[sink]\nenable=1\n"
    assert section_value(before, "tracker", "enable", "0") == before.replace("enable=1", "enable=0", 1)
    with pytest.raises(ValueError):
        section_value(before, "tracker", "absent", "0")


def test_detector_profile_never_publishes_untracked_state(tmp_path):
    original = (ROOT / "config/mv3dt_dev_room/config_deepstream.txt").read_bytes()
    stage = stage_profile(tmp_path, "replay", "off")
    app = (stage / "config_deepstream.txt").read_text()
    assert "PeopleNet" not in app
    assert "[tracker]\nenable=0" in app
    assert "[sink3]\nenable=0" in app
    assert "batched-push-timeout=-1" in app
    assert (ROOT / "config/mv3dt_dev_room/config_deepstream.txt").read_bytes() == original
    assert (stage / "config_tracker.yml").read_bytes() == (ROOT / "config/mv3dt_dev_room/config_tracker.yml").read_bytes()


def test_native_pin_mismatch_fails_before_any_camera(tmp_path):
    record = {"status": "EXPERIMENTAL_NOT_ACCEPTED", "production_accepted": False,
              "base_inputs": {"f": "incorrect"}}
    path = tmp_path / "build.json"
    path.write_text(json.dumps(record))
    with patch("scripts.build_room_candidate.input_hashes", return_value={"f": "correct"}):
        with pytest.raises(ValueError, match="frozen checkout"):
            verify_build(path)


def test_command_uses_engine_recorded_runtime_and_readonly_artifacts(tmp_path):
    stage = stage_profile(tmp_path, "replay", "off")
    cmd = command(stage, {"binary": str(tmp_path / "native-app")}, "replay", "candidate-test")
    platform = json.loads((ROOT / "config/deepstream-platform.json").read_text())
    assert platform["image"] in cmd
    assert "--privileged" not in cmd
    assert f"{ROOT / '.runtime/models/yolo26m'}:/models:ro" in cmd
    assert "--pull=never" in cmd
