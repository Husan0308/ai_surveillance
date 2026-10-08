from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "services/camera_v11/native/six_camera_v13"


def test_primary_detections_reach_nvdcf_unchanged() -> None:
    source = (NATIVE / "detection.hpp").read_text()
    normal = source.split("if (obj->confidence < .25f)", 1)[1].split("++count;", 1)[1]
    assert "nvds_remove_obj_meta_from_frame" not in normal
    assert "tracker_detections.push_back({obj->confidence, r.left, r.top, r.width, r.height," in normal
    assert "static_cast<guint>(std::max(0, obj->unique_component_id)), auxiliary" in normal
    # The 640 primary >=.25 path retains its same confidence and bbox unchanged;
    # the appended fields only record GIE provenance for the isolated auxiliary.
    assert "if (obj->confidence < .25f)" in source
    assert "if (auxiliary && auxiliary_duplicate_of_primary(obj, primary_boxes))" in source


def test_active_context_verifier_is_async_and_suppresses_definitively_unmatched_tracks() -> None:
    tracker = (NATIVE / "tracker.hpp").read_text()
    pretracker = tracker.split("static GstPadProbeReturn tracker_enter(", 1)[1].split(
        "static GstPadProbeReturn tracker_exit(", 1)[0]
    posttracker = tracker.split("static GstPadProbeReturn tracker_exit(", 1)[1].split(
        "static void tracker_stats(", 1)[0]
    assert "tracker_schedule_shadow_context_before_nvdcf(buffer, frame)" in pretracker
    assert "candidate, false, true" not in pretracker
    assert "detection_to_active" in posttracker
    assert "candidate.detector_index = row.detection_index" in posttracker
    assert "candidate, true);" in posttracker  # typed as an active birth at submission
    assert "tracker_birth_verification.begin(frame->source_id, row.id" in tracker
    assert "row.detection_index = static_cast<int>(mapping.first)" in tracker
    assert "lowconf_verifier_assign_active_track(" in tracker
    assert "lowconf_verifier_submit(buffer, frame->batch_id" in posttracker
    assert "tracker_apply_active_birth_verification(source_id, tracker_frame_num" in tracker
    assert "lowconf_verifier_take_completed_active(frame->source_id)" in tracker
    wait = tracker.split("static bool tracker_lowconf_wait_eligible(", 1)[1].split(
        "static void tracker_log_shadow_context_fallback", 1)[0]
    assert "candidate.active_track_birth" in wait
    assert "candidate.source_frame_num != frame_num" in wait
    assert "candidate.source_pts_ns != pts_ns" in wait
    assert "track.detection_index == candidate.detector_index" in wait
    apply = tracker.split("static void tracker_apply_active_birth_verification(", 1)[1].split(
        "static GstPadProbeReturn tracker_enter", 1)[0]
    assert "published->erase(std::remove_if" in apply
    assert "tracker_birth_verification.rejected(source_id, row.id)" in apply
    assert "V11_ACTIVE_TRACK_VERIFIER_FAIL_CLOSED" in apply
    assert "tracker_birth_verification.verified(source_id, row.id)" in apply
    assert "!decision || !decision->verifier_done || !decision->verifier_confirmed" in apply
    assert "bool active_birth_wait" not in tracker
    assert "const bool wait_for_verifier = context_verifier_enabled() &&" in tracker


def test_queue_backpressure_retries_without_false_negative_backoff() -> None:
    tracker = (NATIVE / "tracker.hpp").read_text()
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    assert "return LowConfSubmitResult::Backpressure;" in verifier
    assert "LowConfSubmitResult lowconf_verifier_submit" in verifier
    # The birth ledger becomes Pending only when an exact PGIE row was joined
    # to a native target. Backpressure leaves it Unverified, so the next frame
    # can retry rather than suppressing output or applying a false negative.
    assert "if (submitted != LowConfSubmitResult::Submitted ||" in tracker
    assert "tracker_birth_verification.cancel(frame->source_id, row.id" in tracker


def test_context_location_match_requires_a_unique_best_iou_margin() -> None:
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    assert "uniqueness_margin >= .10f" in verifier
    assert "static bool lowconf_verifier_assign_active_track" in verifier
    assert "request.candidate.detector_index != detector_index" in verifier
    assert "request.candidate.active_track_id_assigned = true" in verifier
    assert "request.candidate.verifier_runner_up_iou = runner_up_iou;" in verifier
    assert "request.candidate.verifier_uniqueness_margin = uniqueness_margin;" in verifier


def test_lowconf_fallback_can_reuse_only_recent_verified_track_support() -> None:
    tracker = (NATIVE / "tracker.hpp").read_text()
    ledger = (NATIVE / "track_birth_verification.hpp").read_text()
    apply = tracker.split("static void tracker_apply_verified_lowconf(", 1)[1].split(
        "// Context verification affects only canonical visibility", 1)[0]
    wait = tracker.split("static bool tracker_lowconf_wait_eligible(", 1)[1].split(
        "static void tracker_log_shadow_context_fallback", 1)[0]
    assert "TRUSTED_LOWCONF_SUPPORT_MAX_AGE_FRAMES = 5" in tracker
    assert "support_verified_target(frame->source_id, row.id" in tracker
    assert "tracker_birth_verification.rejected(source_id, shadow.id)" in apply
    assert "candidate.verifier_done && candidate.verifier_confirmed" in apply
    assert "lowconf_fallback_authorized(" in apply
    assert "!candidate.verifier_done || !candidate.verifier_confirmed" not in apply
    assert "verified_support_age(" in apply and "source_id, shadow.id" in apply
    assert "PGIE_RECENT_VERIFIED_TRACK_LOWCONF_FALLBACK" in apply
    assert "candidate.source_frame_num != frame_num" in apply
    assert "candidate.source_pts_ns != pts_ns" in apply
    assert "shadow.immediate_continuity" in apply
    assert "score >= .30f" in apply
    assert "<= 50.0f" in apply
    assert "width_ratio <= 2.0f && height_ratio <= 2.0f" in apply
    assert "verified_with_recent_support(source_id, native_id" in wait
    assert "void support_verified_target" in ledger
    assert "frame - support_frame <= max_support_age" in ledger
    assert "candidate_result_done) return candidate_result_confirmed" in ledger
    assert "return verified_with_recent_support(source, id, frame, max_support_age)" in ledger


def test_lowconf_fallback_visible_rectangle_is_always_current_primary_box() -> None:
    tracker = (NATIVE / "tracker.hpp").read_text()
    apply = tracker.split("static void tracker_apply_verified_lowconf(", 1)[1].split(
        "// Context verification affects only canonical visibility", 1)[0]
    assert "row.left = candidate.left; row.top = candidate.top;" in apply
    assert "row.width = candidate.width; row.height = candidate.height;" in apply
    assert "row.left = shadow.left" not in apply
    assert "row.top = shadow.top" not in apply


def test_active_context_match_uses_candidate_center_and_keeps_shadow_gates_separate() -> None:
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    active = verifier.split("static bool lowconf_active_location_match(", 1)[1].split(
        "static GstElement *lowconf_make_element", 1)[0]
    assert "cx >= proposal.left && cx <= proposal.left + proposal.width" in active
    assert "cy >= proposal.top && cy <= proposal.top + proposal.height" in active
    complete = verifier.split("static void lowconf_complete_result(", 1)[1].split(
        "static GstPadProbeReturn lowconf_infer_exit", 1)[0]
    assert "request.candidate.active_track_birth" in complete
    assert "request.candidate.verifier_no_local_person" in complete
    assert "lowconf_strict_match" in complete


def test_context_verifier_keeps_lowconf_sidecar_out_of_nvdcf() -> None:
    source = (NATIVE / "detection.hpp").read_text()
    low = source.split("if (obj->confidence < .25f)", 1)[1].split("++count;", 1)[0]
    assert "lowconf_sidecar.push_back(candidate)" in low
    assert "lowconf_verifier_submit(b, frame->batch_id, lowconf_key" in low
    assert "candidate, false, false" in low
    assert "nvds_remove_obj_meta_from_frame(frame, obj)" in low
    assert "tracker_record_lowconf_candidates(frame->source_id" in source
    tracker = (NATIVE / "tracker.hpp").read_text()
    assert "tracker_take_lowconf_candidates(frame->source_id" in tracker
    assert "tracker_lowconf_safe_shadow_candidates(" in tracker
    assert "const auto eligible_lowconf = tracker_lowconf_safe_shadow_candidates(" not in tracker
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    assert "candidate.confidence >= .25f" in verifier
    assert "candidate.active_track_birth = active_birth" in verifier


def test_shadow_refresh_is_current_frame_verified_and_never_uses_shadow_geometry() -> None:
    tracker = (NATIVE / "tracker.hpp").read_text()
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    pretracker = tracker.split("static GstPadProbeReturn tracker_enter(", 1)[1].split(
        "static GstPadProbeReturn tracker_exit(", 1)[0]
    refresh_submit = tracker.split(
        "static void tracker_schedule_shadow_context_before_nvdcf(", 1
    )[1].split("static GstPadProbeReturn tracker_enter(", 1)[0]
    posttracker = tracker.split("static GstPadProbeReturn tracker_exit(", 1)[1].split(
        "static void tracker_stats(", 1)[0]
    assert "tracker_previous_active_boxes" in refresh_submit
    assert "tracker_previous_shadows" in refresh_submit
    assert "tracker_birth_verification.verified(source_id, native_id)" in refresh_submit
    assert "lowconf_verifier_submit(buffer, frame->batch_id, key" in refresh_submit
    assert "candidate.native_track_id = anchor.native_id" in refresh_submit
    assert "candidate.shadow_targets.push_back" in refresh_submit
    assert "shadow_context_group_hints(surface_width, surface_height" in refresh_submit
    assert "candidate, false, true" in refresh_submit
    assert "tracker_schedule_shadow_context_before_nvdcf(buffer, frame)" in pretracker
    refresh_block = posttracker.split("// Shadow context requests are scheduled", 1)[1].split(
        "if (tracker_decision_evidence)", 1)[0]
    assert "lowconf_verifier_submit(buffer, frame->batch_id" not in refresh_block
    assert "candidate.shadow_context_refresh" in posttracker
    apply = tracker.split("static void tracker_apply_verified_shadow_context(", 1)[1].split(
        "// Context verification affects only canonical visibility", 1)[0]
    assert "candidate.source_frame_num != frame_num" in apply
    assert "candidate.source_pts_ns != pts_ns" in apply
    assert "shadow.state == INACTIVE" in apply
    assert "shadow.immediate_continuity" in apply
    assert "shadow_context_match_one_to_one(target_rects, proposal_rects)" in apply
    assert "row.left = box.left" in apply
    assert "row.top = box.top" in apply
    assert 'row.source = "SHADOW_CONTEXT_VERIFIED"' in apply
    assert "shadow_it->left" in apply  # current shadow geometry is only a matching hint
    assert "matched_candidate.verifier_bbox_left = box.left" in apply
    assert "row.left = box.left" in apply
    assert "row.top = box.top" in apply
    assert "row.width = box.width" in apply
    assert "row.height = box.height" in apply
    complete = verifier.split("static void lowconf_complete_result(", 1)[1].split(
        "static GstPadProbeReturn lowconf_infer_exit", 1)[0]
    assert "shadow_context_to_frame" in complete
    assert "request.candidate.verifier_boxes.push_back" in complete
    assert "SHADOW_CONTEXT_REFRESH" in verifier
    assert "verifier_bbox_width = 0" in complete


def test_context_verifier_queue_is_bounded_without_silent_gstreamer_drops() -> None:
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    assert '"block", TRUE' in verifier
    assert '"leaky", 0' in verifier
    assert '"max-size-buffers", 1u' in verifier


def test_diagnostic_camera_lane_reuses_private_crop_and_trt_stream() -> None:
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    runtime = (NATIVE / "context_trt_runtime.hpp").read_text()
    crop = verifier.split("static NvBufSurface *lowconf_prepare_gpu_nvmm_crop_surface(", 1)[1].split(
        "static GstBuffer *lowconf_prepare_gpu_nvmm_crop(", 1)[0]
    lane = verifier.split("static void lowconf_context_infer_camera_job(", 1)[1].split(
        "static GstPadProbeReturn lowconf_infer_exit", 1)[0]
    assert 'std::strcmp(g_getenv("V11_CONTEXT_VERIFIER_PER_CAMERA"), "1") == 0' in verifier
    assert "lowconf_context_preprocess_lanes[job.source_id].stream" in crop
    assert "std::lock_guard<std::mutex> transform_guard(lowconf_nvbuf_transform_session_mutex)" in crop
    assert "preprocess.rgb(preprocess.stream" in lane
    assert "lowconf_context_trt_infer_lane(job.source_id" in lane
    assert "cudaStreamSynchronize(cuda_stream)" in runtime
    assert "six-persistent-per-camera-async-lanes" in runtime
    live_runner = (ROOT / "scripts/run_six_camera_v13_wall_candidate.py").read_text()
    assert 'if manifest.get("per_camera_context_backend") is True:' in live_runner


def test_bbox_extent_uses_effective_mux_canvas_not_source_metadata_fallback() -> None:
    tracker = (NATIVE / "tracker.hpp").read_text()
    detection = (NATIVE / "detection.hpp").read_text()
    helper = tracker.split("static guint tracker_bbox_coordinate_extent(", 1)[1].split(
        "static LatencyHistogram tracker_element_latency", 1)[0]
    assert "pgie_isolation_width" in helper
    assert "pgie_isolation_height" in helper
    assert "v11_mux_width" in helper
    assert "v11_mux_height" in helper
    assert "frame->pipeline_width" not in helper
    assert "frame->source_frame_width" not in helper
    assert "#ifdef PGIE_ISOLATION" in helper
    assert "tracker_bbox_coordinate_extent(frame, true)" in detection
    assert "tracker_bbox_coordinate_extent(frame, false)" in detection
    assert "lowconf_key, tracker_bbox_coordinate_extent(frame, true)" in tracker
    assert "const guint proposal_width = tracker_bbox_coordinate_extent(frame, true)" in tracker
    assert "const guint proposal_height = tracker_bbox_coordinate_extent(frame, false)" in tracker


def test_canonical_snapshot_keeps_source_and_bbox_coordinate_extents_distinct() -> None:
    tracker = (NATIVE / "tracker.hpp").read_text()
    serializer = tracker.split("static bool tracker_atomic_snapshot(", 1)[1].split(
        "static bool tracker_write_line", 1)[0]
    assert '"source_frame_width\\":" << width' in serializer
    assert '"source_frame_height\\":" << height' in serializer
    assert '"bbox_coordinate_width\\":" << proposal_width' in serializer
    assert '"bbox_coordinate_height\\":" << proposal_height' in serializer
    # Detector proposals carry source dimensions as source metadata; bbox
    # coordinates remain in the separately declared mux/pipeline extent.
    assert '"source_frame_width\\":" << proposal_width' not in serializer
    assert '"source_frame_height\\":" << proposal_height' not in serializer


def test_auxiliary_branch_strips_inherited_objects_before_its_inference() -> None:
    source = (NATIVE / "detection.hpp").read_text()
    function = source.split("static GstPadProbeReturn auxiliary_strip_inherited_objects(", 1)[1].split(
        "static bool write_frame_json", 1)[0]
    graph = source.split("GstElement *input_tee = make(\"tee\", \"yolo640-supplemental-input-tee\")", 1)[1].split(
        "#else", 1)[0]
    assert "NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(buffer);" in function
    assert "nvds_remove_obj_meta_from_frame(frame, object)" in function
    assert "frame_node = frame_node->next" in function
    assert "object_node = next" in function
    assert graph.index("auxiliary_strip_inherited_objects") < graph.index(
        "link(auxiliary_isolation_caps, auxiliary_gie)")
    assert "nvds_remove_frame_meta" not in function


def test_verifier_config_is_isolated_batch1_frozen_yolo_and_launcher_opt_in() -> None:
    builder = (ROOT / "scripts/build_six_camera_v13_wall_candidate.py").read_text()
    runner = (ROOT / "scripts/run_six_camera_v13_wall_candidate.py").read_text()
    assert '"yolo26m_context_verifier.txt"' in builder
    assert "batch-size=1" in builder
    assert "model-engine-file=/models/yolo26m_raw_otm_b6_fp16.engine" in builder
    assert "pre-cluster-threshold=0.20" in builder
    assert '"V11_ACTIVE_TRACK_VERIFIER_ENABLE=1"' in runner
    assert "validate_yolo_context_verifier(candidate, manifest)" in runner
    assert "--yolo-context-verifier" in runner


def test_primary_only_context_verifier_profile_is_explicit_and_not_mistaken_for_896_cascade() -> None:
    from scripts.run_six_camera_v13_wall_candidate import (
        has_auxiliary_yolo_profile,
        supports_yolo_context_profile,
    )

    profile = {
        "name": "yolo26m-fp16-640-primary-448-context-verified",
        "context_verifier_mode": "yolo26m-current-frame-crop-confirmation",
        "primary_config": "config_infer_primary_yolo26m_raw_otm_lowconf023.txt",
    }
    assert supports_yolo_context_profile(profile)
    assert not has_auxiliary_yolo_profile(profile)
    assert not supports_yolo_context_profile({"name": "yolo26m-fp16-640-only"})
