from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

from scripts.build_six_camera_v13_wall_candidate import CANDIDATE, EXPECTED, SOURCE
from scripts.run_six_camera_v13_wall_candidate import (
    CAMERAS,
    camera_config,
    docker_command,
    has_auxiliary_yolo_profile,
    parse_source_stall_test_plan,
    validate_candidate,
    validate_lowconf_verifier_artifacts,
    validate_candidate_source_snapshot,
)


class _Camera:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.uri = f"rtsp://camera.invalid/{camera_id}"
        self.username = "secret-user"
        self.password = "secret-password"
        self.decoder_low_latency_mode = True
        self.decoder_extra_surfaces = None

    @staticmethod
    def effective_latency_ms(default: int) -> int:
        return default


def test_candidate_manifest_pins_local6_yolo_engine_and_frozen_v13_reference():
    # This test reads the historical local candidate snapshot only as a data
    # fixture.  It deliberately does not call validate_candidate(): that
    # function must reject a build whose native source hash no longer matches
    # today's tree.  Fresh-build validation is performed against the new r2
    # candidate after compilation, never by rewriting the old snapshot.
    import json

    manifest = json.loads((CANDIDATE / "build-manifest.json").read_text())
    assert manifest["config_sha256"]["config_tracker_NvDCF_v13_local6.yml"] == EXPECTED[
        "config_tracker_NvDCF_v13_local6.yml"
    ]
    profile = manifest.get("detector_profile") or {}
    if profile:
        assert profile["name"] == "yolo26m-fp16-896-context-verified"
        assert profile["input"] == [896, 896] and profile["batch_size"] == 6
        assert profile["confidence"] == 0.25 and profile["nms_iou"] == 0.45
        assert manifest["config_sha256"]["config_infer_primary_yolo26m_raw_otm.txt"] == profile["config_sha256"]
        assert manifest["model_sha256"]["yolo26m_raw_otm_896_b6_fp16.engine"] == profile["engine_sha256"]
        assert manifest["model_sha256"]["libyolo26rawperson_896.so"] == profile["parser_sha256"]
    else:
        assert manifest["config_sha256"]["config_infer_primary_yolo26m_raw_otm.txt"] == EXPECTED[
            "config_infer_primary_yolo26m_raw_otm.txt"
        ]
        assert manifest["model_sha256"]["yolo26m_raw_otm_b6_fp16.engine"] == EXPECTED[
            "yolo26m_raw_otm_b6_fp16.engine"
        ]
    recorded_sources = set(manifest["source_sha256"])
    current_sources = {path.name for path in SOURCE.iterdir() if path.is_file()}
    # The accepted r12 manifest is an immutable historical fixture. New
    # diagnostic/runtime modules may be added to the source tree later, but
    # every source named by the fixture must still exist and remain hashed.
    assert {"main.cpp", "tracker.hpp", "detection.hpp"} <= recorded_sources
    assert recorded_sources <= current_sources
    assert all(len(value) == 64 for value in manifest["source_sha256"].values())
    assert manifest["lowconf_verifier"]["enabled_by_default"] is True
    policy = manifest["lowconf_verifier"]["async_policy"]
    assert "per-camera ordered canonical publisher" in policy
    assert "uniquely continuous current INACTIVE-shadow candidate" in policy
    assert "deadline=26ms" in policy
    assert "streaming callbacks never wait" in policy


def test_release_source_integrity_validates_embedded_snapshot_not_mutable_worktree(tmp_path, monkeypatch):
    import hashlib
    import json

    from scripts import build_six_camera_v13_wall_candidate as mutable_build

    candidate = tmp_path / "release"
    native = candidate / "source/native"
    native.mkdir(parents=True)
    source_file = native / "main.cpp"
    source_file.write_text("int main() { return 0; }\n", encoding="utf-8")
    source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
    files = {"main.cpp": source_hash}
    tree_hash = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_manifest = candidate / "source/source-manifest.json"
    source_manifest.write_text(json.dumps({"source_sha256": files, "tree_sha256": tree_hash}),
                               encoding="utf-8")
    manifest = {
        "source_sha256": files,
        "source_snapshot": {
            "directory": "source/native",
            "manifest": "source/source-manifest.json",
            "manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            "tree_sha256": tree_hash,
        },
    }
    monkeypatch.setattr(mutable_build, "source_hashes", lambda: (_ for _ in ()).throw(
        AssertionError("mutable development tree must not be consulted")))
    assert validate_candidate_source_snapshot(candidate, manifest) == files

    source_file.write_text("tampered release source\n", encoding="utf-8")
    try:
        validate_candidate_source_snapshot(candidate, manifest)
    except RuntimeError as exc:
        assert "snapshot" in str(exc)
    else:
        raise AssertionError("tampered embedded native source was accepted")


def test_legacy_release_source_integrity_keeps_strict_current_tree_check(monkeypatch):
    from scripts import build_six_camera_v13_wall_candidate as mutable_build

    monkeypatch.setattr(mutable_build, "source_hashes", lambda: {"main.cpp": "current"})
    assert validate_candidate_source_snapshot(Path("unused"), {"source_sha256": {"main.cpp": "current"}}) == {
        "main.cpp": "current"
    }
    try:
        validate_candidate_source_snapshot(Path("unused"), {"source_sha256": {"main.cpp": "stale"}})
    except RuntimeError as exc:
        assert "native source changed" in str(exc)
    else:
        raise AssertionError("legacy mutable-tree mismatch was accepted")


def test_camera_runtime_secret_profile_is_six_source_and_encodes_credentials():
    settings = SimpleNamespace(
        cameras=[_Camera(camera) for camera in CAMERAS],
        deepstream=SimpleNamespace(latency_ms=20, drop_on_latency=True, decoder_extra_surfaces=4),
    )
    encoded = camera_config(settings)
    assert [line for line in encoded.splitlines() if line.startswith("[CAM-")] == [
        f"[{camera}]" for camera in CAMERAS
    ]
    assert "secret-password" not in encoded
    assert base64.b64encode(b"secret-password").decode() in encoded
    assert "[validation]" in encoded and "preview_enabled=false" in encoded


def test_live_candidate_disables_verbose_evidence_io_by_default(monkeypatch, tmp_path):
    candidate = tmp_path / "candidate"
    output = tmp_path / "run"
    camera_file = tmp_path / "camera.ini"
    monkeypatch.delenv("V11_VERBOSE_EVIDENCE", raising=False)
    command = docker_command(candidate, output, camera_file, 300)
    assert "V11_VERBOSE_EVIDENCE=0" in command
    monkeypatch.setenv("V11_VERBOSE_EVIDENCE", "1")
    command = docker_command(candidate, output, camera_file, 300)
    assert "V11_VERBOSE_EVIDENCE=1" in command
    monkeypatch.setenv("V11_VERBOSE_EVIDENCE", "true")
    try:
        docker_command(candidate, output, camera_file, 300)
    except ValueError as exc:
        assert "must be 0 or 1" in str(exc)
    else:
        raise AssertionError("verbose evidence flag accepted an ambiguous value")


def test_source_stall_recovery_is_explicit_and_strictly_boolean(monkeypatch):
    monkeypatch.delenv("V11_SOURCE_STALL_RECOVERY", raising=False)
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    assert "V11_SOURCE_STALL_RECOVERY=0" in command
    monkeypatch.setenv("V11_SOURCE_STALL_RECOVERY", "1")
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    assert "V11_SOURCE_STALL_RECOVERY=1" in command
    monkeypatch.setenv("V11_SOURCE_STALL_RECOVERY", "true")
    try:
        docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    except ValueError as exc:
        assert "must be 0 or 1" in str(exc)
    else:
        raise AssertionError("source-stall recovery accepted an ambiguous value")


def test_auxiliary_detector_profile_is_enabled_by_artifact_contract_not_cadence_or_name(tmp_path):
    import json

    candidate = tmp_path / "cascade-candidate"
    candidate.mkdir()
    profile = {
        "name": "yolo26m-fp16-640-primary-896-aux-interval-2-context-verified",
        "aux_config": "config_infer_aux_yolo26m_896.txt",
        "aux_input": [896, 896],
        "aux_unique_id": 3,
        "aux_engine_sha256": "aux-engine",
        "aux_parser_sha256": "aux-parser",
        "aux_interval": 2,
    }
    (candidate / "build-manifest.json").write_text(json.dumps({
        "detector_profile": profile
    }))
    assert has_auxiliary_yolo_profile(profile)
    command = docker_command(candidate, tmp_path / "run", tmp_path / "camera.ini", 60)
    rendered = " ".join(command)
    assert "V11_AUX_YOLO896_ENABLE=1" in rendered
    assert "V11_AUX_YOLO896_CONFIG=/config/config_infer_aux_yolo26m_896.txt" in rendered
    assert "V11_AUX_YOLO896_METAMUX_CONFIG=/config/config_metamux_yolo26m_aux896.txt" in rendered


def test_non_auxiliary_profiles_do_not_enable_cascade_by_name_alone():
    assert not has_auxiliary_yolo_profile({
        "name": "yolo26m-fp16-640-primary-896-aux-context-verified"
    })


def test_generic_768_auxiliary_profile_uses_its_declared_config_and_input_size(tmp_path):
    import json

    candidate = tmp_path / "generic-aux-candidate"
    candidate.mkdir()
    profile = {
        "name": "yolo26m-fp16-640-primary-768-aux-interval-1-context-verified",
        "aux_config": "config_infer_aux_yolo26m_768.txt",
        "metamux_config": "config_metamux_yolo26m_aux768.txt",
        "aux_input": [768, 768], "aux_unique_id": 3,
        "aux_engine_sha256": "aux-engine", "aux_parser_sha256": "aux-parser",
        "aux_interval": 1, "aux_environment_contract": "generic-v1",
    }
    (candidate / "build-manifest.json").write_text(json.dumps({"detector_profile": profile}))
    assert has_auxiliary_yolo_profile(profile)
    command = docker_command(candidate, tmp_path / "run", tmp_path / "camera.ini", 60)
    rendered = " ".join(command)
    assert "V11_AUX_YOLO_ENABLE=1" in rendered
    assert "V11_AUX_YOLO_CONFIG=/config/config_infer_aux_yolo26m_768.txt" in rendered
    assert "V11_AUX_YOLO_METAMUX_CONFIG=/config/config_metamux_yolo26m_aux768.txt" in rendered
    assert "V11_AUX_YOLO_INPUT_SIZE=768" in rendered


def test_parallel_auxiliary_inference_gets_branch_local_batch_meta_without_touching_primary():
    detection = (SOURCE / "detection.hpp").read_text()
    aux = detection.split("GstElement *input_tee = make(\"tee\", \"yolo640-supplemental-input-tee\")", 1)[1]
    aux = aux.split("#else", 1)[0]
    assert "GstElement *auxiliary_isolation_convert = make(\"nvvideoconvert\"" in aux
    assert "auxiliary_strip_inherited_objects" in aux
    assert 'video/x-raw(memory:NVMM),format=RGBA' in aux
    assert aux.index("link(primary_queue, gie)") < aux.index("link(gie, primary_meta_queue)")
    assert aux.index("link(auxiliary_queue, auxiliary_isolation_convert)") < aux.index(
        "link(auxiliary_isolation_convert, auxiliary_isolation_caps)")
    assert aux.index("link(auxiliary_isolation_caps, auxiliary_gie)") < aux.index(
        "link(auxiliary_gie, auxiliary_meta_queue)")
    assert "The primary PGIE branch remains byte-for-byte and element-for-element intact" in aux


def test_parallel_preview_join_cache_is_bounded_for_measured_branch_skew():
    detection = (SOURCE / "detection.hpp").read_text()
    assert "static constexpr size_t F5R_JOIN_ENTRY_LIMIT = 32;" in detection
    prune = detection[detection.index("static void f5r_prune_join_entries_locked"):]
    prune = prune.split("static void f5r_add_prepared_frame", 1)[0]
    assert "entries.size() > F5R_JOIN_ENTRY_LIMIT" in prune
    assert "f5r_preview_join_misses[source_id]" in prune
    assert "f5r_preview_sync_mismatches[source_id]" not in prune
    # Pairing remains exact: increasing the bounded retention window must not
    # permit nearest-PTS or cross-frame fallback.
    writer = detection[detection.index("static bool f5r_write_joined_preview"):]
    assert "frame.frame_num != tracker.source_frame_num || frame.pts != tracker.pts" in writer


def test_auxiliary_birth_verification_is_scoped_and_fail_closed_in_native_source():
    detection = (SOURCE / "detection.hpp").read_text()
    tracker = (SOURCE / "tracker.hpp").read_text()
    # Metadata transfer is an exact source/frame/PTS sidecar join. A MetaMux
    # PTS tolerance property is no longer the synchronization contract.
    assert "AuxiliaryExactJoin<AuxiliaryProposal, 6, 64>" in detection
    assert "auxiliary_exact_join.take_until(key, deadline)" in detection
    assert 'make("nvdsmetamux"' not in detection
    assert 'TRACKER_AUXILIARY_GIE_ID' in detection
    assert "auxiliary_duplicate_of_primary(obj, primary_boxes)" in detection
    assert "if (!row.requires_birth_verification) return false;" in tracker
    assert "if (!row.requires_birth_verification) return false;" in tracker
    assert "tracker_birth_verification.verified(source_id, row.id)" in tracker
    assert "tracker_auxiliary_target_ids[frame->source_id].insert(row.id)" in tracker
    assert "row.requires_birth_verification = active_track_verifier_enabled() ||" in tracker
    builder = (Path(__file__).resolve().parents[1] / "scripts/build_cam05_cascade_candidate.py").read_text()
    assert '"canonical_birth_safety": "all newly born 640-primary and 896-auxiliary native targets' in builder


def test_controlled_recovery_test_plan_is_opt_in_sequential_and_passed_to_native(monkeypatch):
    plan = "CAM-02@30,CAM-02@110,CAM-02@190,CAM-03@270"
    monkeypatch.setenv("V11_SOURCE_STALL_RECOVERY", "1")
    monkeypatch.setenv("V11_TEST_SOURCE_STALL_PLAN", plan)
    parsed = parse_source_stall_test_plan(plan, "1")
    assert parsed == [
        {"camera_id": "CAM-02", "at_sec": 30},
        {"camera_id": "CAM-02", "at_sec": 110},
        {"camera_id": "CAM-02", "at_sec": 190},
        {"camera_id": "CAM-03", "at_sec": 270},
    ]
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-run"),
                             Path("/tmp/private-camera.ini"), 310)
    assert f"V11_TEST_SOURCE_STALL_PLAN={plan}" in command
    monkeypatch.setenv("V11_SOURCE_STALL_RECOVERY", "0")
    try:
        parse_source_stall_test_plan(plan, "0")
    except ValueError as exc:
        assert "requires V11_SOURCE_STALL_RECOVERY=1" in str(exc)
    else:
        raise AssertionError("test-only stalls were allowed with recovery disabled")


def test_controlled_recovery_plan_rejects_cycles_inside_cooldown():
    assert parse_source_stall_test_plan("CAM-04@30", "1") == [
        {"camera_id": "CAM-04", "at_sec": 30}
    ]
    for plan in ("CAM-02@30,CAM-02@90", "CAM-07@30", "CAM-02@5"):
        try:
            parse_source_stall_test_plan(plan, "1")
        except ValueError:
            continue
        raise AssertionError(f"unsafe controlled source-stall plan accepted: {plan}")


def test_source_recovery_is_source_local_and_removes_old_owner_before_recreation():
    native = (SOURCE / "main.cpp").read_text()
    helper = native[native.index("static bool replace_stalled_source"):native.index("static gboolean source_stall_recovery_tick")]
    assert "GST_STATE_NULL" in helper
    assert helper.index("gst_bin_remove") < helper.index("create_source_element(target)")
    assert "refusing to create a second RTSP owner" in helper
    assert "queue_level != 0" in helper
    assert "decoded_last && now_us - decoded_last <= SOURCE_STALL_RECOVERY_AFTER_US" in helper
    assert "source_stall_recovery_tick" in native
    assert 'g_getenv("V11_SOURCE_STALL_RECOVERY")' in native
    assert 'g_timeout_add(500, source_stall_recovery_tick, nullptr)' in native
    assert 'V11_TEST_SOURCE_STALL_PLAN' in native
    assert 'TEST_SOURCE_STALL_BEGIN' in native
    assert helper.index('gst_bin_remove') < helper.index('create_source_element(target)')
    assert 'source_pgie_frames[ctx->source_id]' in native
    assert 'source_tracker_updates[ctx->source_id]' in native
    assert 'source_preview_published[ctx->source_id]' in native
    removal = helper.index("gst_bin_remove")
    assert helper.index("const std::string old_source_name = GST_ELEMENT_NAME(old_source)") < removal
    assert "std::string(GST_ELEMENT_NAME(old_source))" not in helper[removal:]
    assert 'gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, count_buffer, &ctx->decoded, nullptr)' in native


def test_container_command_uses_single_pinned_pipeline_without_external_detector(monkeypatch):
    monkeypatch.delenv("V11_TRACKER_DECISION_DIAGNOSTICS", raising=False)
    monkeypatch.delenv("V11_TRACKER_FRAME_STATE_DIAGNOSTICS", raising=False)
    monkeypatch.delenv("V11_TRACKER_NAN_DIAGNOSTICS", raising=False)
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    rendered = " ".join(command)
    assert "--gpus all" in rendered
    assert "--network=host" in rendered
    assert "nvinfer" not in rendered  # The in-process native binary builds the DS graph.
    assert rendered.count("--mount") == 6
    assert "workspace/models" not in rendered
    assert "camInfo" not in rendered
    assert "pub_sub_info" not in rendered
    assert command[-3] == "/candidate/bin/cam-six-yolo-local6-posttracker"
    assert "/run/camera-six-secret.ini" in rendered
    assert "--tracker" not in rendered
    assert "V11_TRACKER_DECISION_DIAGNOSTICS=1" not in rendered
    assert "V11_TRACKER_FRAME_STATE_DIAGNOSTICS=1" not in rendered


def test_lowconf_context_verifier_is_explicit_opt_in_and_keeps_normal_nvdcf_floor():
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-lowconf-run"),
                             Path("/tmp/private-camera.ini"), 60, lowconf_verifier=True)
    rendered = " ".join(command)
    assert "V11_LOWCONF_VERIFIER_ENABLE=1" in rendered
    assert "V11_LOWCONF_PRIMARY_CONFIG=/config/config_infer_primary_yolo26m_raw_otm_lowconf023.txt" in rendered
    assert "V11_LOWCONF_VERIFIER_CONFIG=/config/peoplenet_lowconf_verifier.txt" in rendered


def test_active_track_verifier_is_fail_closed_without_changing_lowconf_or_primary_config(monkeypatch):
    monkeypatch.delenv("V11_SOURCE_STALL_RECOVERY", raising=False)
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-active-track-run"),
                             Path("/tmp/private-camera.ini"), 60,
                             active_track_verifier=True)
    rendered = " ".join(command)
    assert "V11_ACTIVE_TRACK_VERIFIER_ENABLE=1" in rendered
    assert "V11_ACTIVE_TRACK_VERIFIER_FAIL_CLOSED=1" in rendered
    assert "V11_LOWCONF_VERIFIER_CONFIG=/config/yolo26m_context_verifier.txt" in rendered
    assert "V11_LOWCONF_VERIFIER_ENABLE=1" not in rendered
    assert "V11_LOWCONF_PRIMARY_CONFIG=" not in rendered
    assert "V11_CONTEXT_VERIFIER_NAME=YOLO26m-640-FP16" in rendered


def test_active_track_verifier_switch_is_opt_in_and_runtime_gates_new_ids():
    from scripts.run_six_camera_v13_wall_candidate import ROOT

    tracker = (SOURCE / "tracker.hpp").read_text()
    runner = (ROOT / "scripts/run_six_camera_v13_wall_candidate.py").read_text()
    assert '"--active-track-verifier"' in runner
    assert '"V11_ACTIVE_TRACK_VERIFIER_FAIL_CLOSED=1"' in runner
    assert '"V11_ACTIVE_TRACK_VERIFIER_ENABLE=1"' in runner
    apply = tracker.split("static void tracker_apply_active_birth_verification(", 1)[1].split(
        "static GstPadProbeReturn tracker_enter", 1)[0]
    assert 'g_getenv("V11_ACTIVE_TRACK_VERIFIER_FAIL_CLOSED")' in apply
    assert "tracker_birth_verification.verified(source_id, row.id)" in apply
    assert "candidate.source_frame_num != frame_num" in apply
    assert "candidate.source_pts_ns != pts_ns" in apply
    assert "const bool reject = require_positive &&" in apply


def test_lowconf_verifier_artifacts_are_hash_and_contract_validated():
    import json

    # Verify the frozen sidecar artifacts independently of a stale binary's
    # native-source manifest.  Full candidate provenance is checked after a
    # fresh isolated rebuild.
    manifest = json.loads((CANDIDATE / "build-manifest.json").read_text())
    validate_lowconf_verifier_artifacts(CANDIDATE, manifest)


def test_verifier_validation_fails_closed_when_engine_is_changed(tmp_path):
    import json

    staged = tmp_path / "candidate"
    (staged / "config").mkdir(parents=True)
    engine = staged / "models/peopleNet/benchmarks/engines/peoplenet_dynamic_b2.engine"
    engine.parent.mkdir(parents=True)
    for name in ("config_infer_primary_yolo26m_raw_otm_lowconf023.txt", "peoplenet_lowconf_verifier.txt"):
        (staged / "config" / name).symlink_to(CANDIDATE / "config" / name)
    files = {
        "model/resnet34_peoplenet.onnx": "models/peopleNet/model/resnet34_peoplenet.onnx",
        "config/labels.txt": "models/peopleNet/config/labels.txt",
    }
    for relative, destination in files.items():
        target = staged / "models/peopleNet" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(CANDIDATE / destination)
    source_engine = CANDIDATE / "models/peopleNet/benchmarks/engines/peoplenet_dynamic_b2.engine"
    engine.write_bytes(source_engine.read_bytes() + b"tamper")
    manifest = json.loads((CANDIDATE / "build-manifest.json").read_text())
    try:
        validate_lowconf_verifier_artifacts(staged, manifest)
    except RuntimeError as exc:
        assert "hash mismatch" in str(exc) or "frozen artifact mismatch" in str(exc)
    else:
        raise AssertionError("tampered verifier engine must be rejected")


def test_tracker_decision_diagnostics_are_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("V11_TRACKER_DECISION_DIAGNOSTICS", "1")
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    assert "V11_TRACKER_DECISION_DIAGNOSTICS=1" in " ".join(command)


def test_tracker_frame_state_diagnostics_are_explicit_opt_in_and_strictly_boolean(monkeypatch):
    monkeypatch.delenv("V11_TRACKER_FRAME_STATE_DIAGNOSTICS", raising=False)
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    assert "V11_TRACKER_FRAME_STATE_DIAGNOSTICS=1" not in " ".join(command)
    monkeypatch.setenv("V11_TRACKER_FRAME_STATE_DIAGNOSTICS", "1")
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    assert "V11_TRACKER_FRAME_STATE_DIAGNOSTICS=1" in " ".join(command)
    monkeypatch.setenv("V11_TRACKER_FRAME_STATE_DIAGNOSTICS", "true")
    try:
        docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    except ValueError as exc:
        assert "must be 0 or 1" in str(exc)
    else:
        raise AssertionError("ambiguous tracker frame-state diagnostic flag must fail closed")


def test_tracker_nan_diagnostics_are_explicit_opt_in_and_strictly_boolean(monkeypatch):
    monkeypatch.delenv("V11_TRACKER_NAN_DIAGNOSTICS", raising=False)
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    assert "V11_TRACKER_NAN_DIAGNOSTICS=1" not in " ".join(command)
    monkeypatch.setenv("V11_TRACKER_NAN_DIAGNOSTICS", "1")
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    assert "V11_TRACKER_NAN_DIAGNOSTICS=1" in " ".join(command)
    monkeypatch.setenv("V11_TRACKER_NAN_DIAGNOSTICS", "true")
    try:
        docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    except ValueError as exc:
        assert "must be 0 or 1" in str(exc)
    else:
        raise AssertionError("ambiguous tracker NaN diagnostic flag must fail closed")


def test_pipeline_stage_timing_is_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("V11_PIPELINE_STAGE_TIMING", raising=False)
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    assert "V11_PIPELINE_STAGE_TIMING=1" not in " ".join(command)
    monkeypatch.setenv("V11_PIPELINE_STAGE_TIMING", "1")
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    assert "V11_PIPELINE_STAGE_TIMING=1" in " ".join(command)
    monkeypatch.setenv("V11_PIPELINE_STAGE_TIMING", "true")
    try:
        docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    except ValueError as exc:
        assert "must be 0 or 1" in str(exc)
    else:
        raise AssertionError("ambiguous stage timing diagnostic flag must fail closed")


def test_primary_only_diagnostic_retains_bounded_pgie_queue_topology():
    source = (SOURCE / "detection.hpp").read_text()
    start = source.index("#else\n  auto *in=gst_element_get_static_pad(gie,\"sink\")")
    end = source.index("#endif", start)
    primary_only_branch = source[start:end]
    assert 'make("queue", "primary-pgie-queue")' in primary_only_branch
    assert '"max-size-buffers", 2u' in primary_only_branch
    assert '"max-size-bytes", 0u' in primary_only_branch
    assert '"max-size-time", guint64(0)' in primary_only_branch
    assert '"leaky", 0' in primary_only_branch
    assert "link(from, primary_queue);" in primary_only_branch
    assert "link(primary_queue, gie);" in primary_only_branch


def test_preview_capture_is_opt_in_and_only_passes_staged_capture_settings(monkeypatch):
    monkeypatch.setenv("V11_PREVIEW_CAPTURE_DIR", "/work/preview-captures")
    monkeypatch.setenv("V11_PREVIEW_CAPTURE_SOURCE_FRAME", "120")
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    rendered = " ".join(command)
    assert "V11_PREVIEW_CAPTURE_DIR=/work/preview-captures" in rendered
    assert "V11_PREVIEW_CAPTURE_SOURCE_FRAME=120" in rendered
    assert "camera.invalid" not in rendered


def test_mux_timeout_override_is_isolated_and_range_checked(monkeypatch):
    monkeypatch.setenv("V11_MUX_BATCHED_PUSH_TIMEOUT_US", "10000")
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    assert "V11_MUX_BATCHED_PUSH_TIMEOUT_US=10000" in command
    monkeypatch.setenv("V11_MUX_BATCHED_PUSH_TIMEOUT_US", "999")
    try:
        docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    except ValueError as exc:
        assert "1000..50000" in str(exc)
    else:
        raise AssertionError("invalid diagnostic mux timeout must fail closed")


def test_mux_timeout_defaults_to_measured_six_camera_candidate_value(monkeypatch):
    monkeypatch.delenv("V11_MUX_BATCHED_PUSH_TIMEOUT_US", raising=False)
    command = docker_command(CANDIDATE, Path("/tmp/sixcam-run"), Path("/tmp/private-camera.ini"), 60)
    assert "V11_MUX_BATCHED_PUSH_TIMEOUT_US=1000" in command


def test_native_graph_draws_only_post_tracker_objects_before_shared_memory():
    detection = (SOURCE / "detection.hpp").read_text()
    tracker = (SOURCE / "tracker.hpp").read_text()
    assert "f5r_make_preview_branch(sources[i], false)" in detection
    assert "f5r_make_preview_branch(sources[i], true)" in detection
    assert "decoded tee->{analytics:" in detection
    assert "exact source-frame/PTS join" in detection
    assert "version = 5" in detection
    assert "source_frame_num" in tracker
    assert "frame_snapshot.data(), frame_snapshot.size()" in tracker
    assert "ui-preview-osd-" not in detection
    assert "system_convert" not in detection
    assert "tracker_atomic_snapshot" in tracker
    assert "NVDS_TRACKER_SHADOW_LIST_META" in tracker
    assert "PGIE_SHADOW_FALLBACK" in tracker
    assert "tracker_center_distance" in tracker
    assert "shadow.age" in tracker
    assert "sources[i]->tee,q" not in detection
    assert '"/config/config_tracker_NvDCF_v13_local6.yml"' in tracker
    assert "bbox_source" in tracker
    assert "tracker_record_detections" in detection
    assert "bbox_coordinate_space" in tracker and "pipeline" in tracker
    assert "bbox_coordinate_width" in tracker and "bbox_coordinate_height" in tracker


def test_f5r_reference_tree_is_not_used_as_candidate_output():
    assert CANDIDATE.name != "F5R-six-camera-yolo-resolution-20261003"
    assert CANDIDATE.is_relative_to(CANDIDATE.parents[1])


def test_local6_profile_is_calibration_free_and_keeps_v13_local_modules():
    import yaml

    from scripts.run_six_camera_v13_wall_candidate import ROOT

    frozen = yaml.safe_load((ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml").read_text().replace("%YAML:1.0", ""))
    local_text = (ROOT / "config/deepstream/config_tracker_NvDCF_v13_local6.yml").read_text()
    local = yaml.safe_load(local_text.replace("%YAML:1.0", ""))
    assert set(local) == {"BaseConfig", "TargetManagement", "DataAssociator", "StateEstimator", "VisualTracker"}
    assert local["BaseConfig"] == frozen["BaseConfig"]
    for section in ("TargetManagement", "VisualTracker"):
        for key, value in local[section].items():
            if section == "TargetManagement" and key == "outputShadowTracks":
                continue
            assert frozen[section][key] == value, (section, key)
    for key, value in local["DataAssociator"].items():
        if key == "minMatchingScore4PeerAssocIou":
            continue
        assert frozen["DataAssociator"][key] == value, ("DataAssociator", key)
    assert "minMatchingScore4PeerAssocIou" not in local["DataAssociator"]
    assert "TrajectoryManagement" not in local
    assert "ObjectModelProjection" not in local
    assert "cameraModelFilepath" not in local_text
    assert "Communicator" not in local and "PoseEstimator" not in local
    assert local["StateEstimator"]["stateEstimatorType"] == 2
    assert local["TargetManagement"]["outputShadowTracks"] == 1
    assert local["StateEstimator"] == {
        "stateEstimatorType": 2,
        "processNoiseVar4Loc": 6810.8668,
        "processNoiseVar4Size": 1541.8647,
        "processNoiseVar4Vel": 1348.4874,
        "measurementNoiseVar4Detector": 100.0,
        "measurementNoiseVar4Tracker": 293.3238,
    }
    assert "SV3DT 3D foot-location estimator" in local_text
