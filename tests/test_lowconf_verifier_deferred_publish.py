from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "services/camera_v11/native/six_camera_v13"


def test_lowconf_sidecar_is_removed_before_nvdcf_and_high_conf_path_is_unchanged() -> None:
    detection = (NATIVE / "detection.hpp").read_text()
    assert "if (obj->confidence < .25f)" in detection
    low_branch = detection.split("if (obj->confidence < .25f)", 1)[1].split("++count;", 1)[0]
    assert "lowconf_sidecar.push_back(candidate)" in low_branch
    assert "lowconf_verifier_submit(b, frame->batch_id, lowconf_key" in low_branch
    assert "pgie_frame_width, pgie_frame_height, candidate, false, false" in low_branch
    assert "nvds_remove_obj_meta_from_frame(frame, obj)" in low_branch
    assert "candidate, false, false" in low_branch
    assert "tracker_detections.push_back({obj->confidence" in detection
    assert "tracker_record_lowconf_candidates(frame->source_id" in detection
    tracker = (NATIVE / "tracker.hpp").read_text()
    assert "tracker_take_lowconf_candidates(frame->source_id" in tracker
    assert "tracker_lowconf_safe_shadow_candidates(" in tracker
    assert "const auto eligible_lowconf = tracker_lowconf_safe_shadow_candidates(" not in tracker


def test_only_canonical_publisher_waits_for_safe_same_frame_verifier_result() -> None:
    tracker = (NATIVE / "tracker.hpp").read_text()
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    assert "tracker_lowconf_wait_eligible(source_id, lowconf_candidates" in tracker
    assert "lowconf_verifier_defer_frame(lowconf_key, std::move(finalize), wait_for_verifier)" in tracker
    assert "lowconf_verifier_take_frame_wait(frame.key)" not in tracker
    assert "lowconf_verifier_frame_candidates(lowconf_key)" in tracker
    assert "tracker_lowconf_candidates_with_safe_shadow" not in tracker
    assert "frame.wait_for_verifier\n        ? lowconf_verifier_take_frame_wait(frame.key)" in verifier
    assert "LOWCONF_PUBLISH_QUEUE_LIMIT = 32" in verifier
    assert "latest_deadline_ns = std::max" in verifier
    assert "LOWCONF_RESULT_DEADLINE_NS = 26000000ULL" in verifier
    assert "lowconf_publish_workers[source_id] = std::thread" in verifier
    assert "if (worker.joinable()) worker.join();" in verifier
    assert "late.verifier_done = false" in verifier
    assert "request.consumed = true" in verifier
    assert "lowconf_pts_to_serial.find(verifier_pts)" in verifier


def test_active_birth_exact_frame_wait_has_separate_bounded_budget() -> None:
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    assert "LOWCONF_RESULT_DEADLINE_NS = 26000000ULL" in verifier
    assert "ACTIVE_BIRTH_RESULT_DEADLINE_NS = 200000000ULL" in verifier
    deadline = verifier.split("static guint64 lowconf_verifier_result_deadline_ns(", 1)[1].split(
        "static constexpr size_t LOWCONF_PUBLISH_QUEUE_LIMIT", 1)[0]
    assert "candidate.active_track_birth" in deadline
    assert "ACTIVE_BIRTH_RESULT_DEADLINE_NS" in deadline
    assert "LOWCONF_RESULT_DEADLINE_NS" in deadline
    take = verifier.split("static std::vector<LowConfCandidate> lowconf_verifier_take_frame(", 1)[1].split(
        "static bool lowconf_request_needs_exact_frame_result", 1)[0]
    assert "lowconf_verifier_result_deadline_ns(completed)" in take
    wait = verifier.split("static std::vector<LowConfCandidate> lowconf_verifier_take_frame_wait(", 1)[1].split(
        "static void lowconf_verifier_publish_worker_main", 1)[0]
    assert "lowconf_verifier_result_deadline_ns(request_it->second.candidate)" in wait
    # Only active births receive the larger bound. Low-confidence fallback and
    # shadow-refresh decisions remain on the original fail-closed 26 ms limit.
    assert "candidate.active_track_birth\n      ? ACTIVE_BIRTH_RESULT_DEADLINE_NS\n      : LOWCONF_RESULT_DEADLINE_NS" in deadline


def test_selective_wait_uses_existing_shadow_continuity_and_spatial_uniqueness_gates() -> None:
    tracker = (NATIVE / "tracker.hpp").read_text()
    helper = tracker.split("static bool tracker_lowconf_wait_eligible(", 1)[1].split(
        "static void tracker_apply_verified_lowconf(", 1)[0]
    assert "candidate.confidence < .23f ||" in helper
    assert "candidate.confidence >= .25f" in helper
    assert "shadow.state != INACTIVE" in helper
    assert "shadow.immediate_continuity" in helper
    assert "score >= .30f" in helper
    assert "<= 50.0f" in helper
    assert "width_ratio <= 2.0f" in helper
    assert "height_ratio <= 2.0f" in helper
    assert "choices[ci][0].score - choices[ci][1].score < .10f" in helper
    assert "target_claims" in helper
    assert "return false;" in helper


def test_tracker_snapshots_continuity_before_deferred_publish_without_copying_history() -> None:
    tracker = (NATIVE / "tracker.hpp").read_text()
    assert "immediate_continuity" in tracker
    assert "const auto prior_shadow_state = tracker_previous_shadows" not in tracker
    assert "const auto prior_active_state = tracker_previous_active_ids" not in tracker
    assert "shadows = std::move(shadows)" in tracker


def test_verifier_branch_drains_to_eos_before_replay_aggregate_eos() -> None:
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    detection = (NATIVE / "detection.hpp").read_text()
    worker = verifier.split("static void lowconf_verifier_worker_main(guint source_id)", 1)[1].split("static void lowconf_verifier_finish_input", 1)[0]
    assert "if (lowconf_worker_stop.load() && lowconf_jobs[source_id].empty()) break;" in worker
    assert "gst_app_src_end_of_stream(\n        GST_APP_SRC(lowconf_appsrc[source_id]))" in worker
    assert "lowconf_verifier_primary_event" in detection
    assert "GST_EVENT_TYPE(event) == GST_EVENT_EOS" in verifier


def test_verifier_crop_work_is_bounded_and_isolated_per_camera() -> None:
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    assert "std::array<std::deque<LowConfVerifierJob>, 6> lowconf_jobs" in verifier
    assert "std::array<std::thread, 6> lowconf_workers" in verifier
    assert "LOWCONF_JOB_QUEUE_LIMIT_PER_CAMERA = 8" in verifier
    assert "LOWCONF_REQUEST_LIMIT = 64" in verifier
    assert "lowconf_jobs[key.source_id].size() >= LOWCONF_JOB_QUEUE_LIMIT_PER_CAMERA" in verifier
    assert "lowconf_requests.size() >= LOWCONF_REQUEST_LIMIT" in verifier
    assert "lowconf_jobs[key.source_id].push_back" in verifier
    assert "lowconf_appsrc[source_id]" in verifier
    assert "per-camera-appsrc[6]" in verifier


def test_context_verifier_supports_three_persistent_batch2_groups_without_changing_camera_queues() -> None:
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    assert "parsed == 1 || parsed == 2" in verifier
    assert "(static_cast<guint>(lowconf_appsrc.size()) + verifier_batch - 1)" in verifier
    assert '"batch-size", verifier_batch' in verifier
    assert "group * verifier_batch + slot" in verifier
    assert "parsed == 1 || parsed == 2 || parsed == 6" in verifier
    assert "V11_CONTEXT_VERIFIER_BATCH_SIZE" in verifier
    # Requests remain queued/per-camera; batch groups affect only downstream
    # inference contexts, not NvDCF inputs or source identity.
    assert "lowconf_jobs[key.source_id].push_back" in verifier
    assert "gst_app_src_push_buffer(\n            GST_APP_SRC(lowconf_appsrc[source_id]), out)" in verifier


def test_opt_in_gpu_crop_keeps_context_verifier_in_nvmm_without_primary_path_changes() -> None:
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    direct_runner = (ROOT / "scripts/run_cam05_candidate_direct_replay.py").read_text()
    wall_runner = (ROOT / "scripts/run_six_camera_v13_wall_candidate.py").read_text()
    assert 'g_getenv("V11_CONTEXT_VERIFIER_GPU_CROP")' in verifier
    assert "lowconf_prepare_gpu_nvmm_crop(job, &request, &reason)" in verifier
    assert 'video/x-raw(memory:NVMM),format=NV12,width=640,height=640' in verifier
    assert "gst_element_link(source, queue)" in verifier
    assert 'V11_CONTEXT_VERIFIER_GPU_CROP=1' in direct_runner
    assert 'V11_CONTEXT_VERIFIER_GPU_CROP=1' in wall_runner
    # The low-confidence row remains removed before the primary NvDCF path;
    # GPU cropping only prepares the verifier's independent appsrc input.
    detection = (NATIVE / "detection.hpp").read_text()
    low = detection.split("if (obj->confidence < .25f)", 1)[1].split("++count;", 1)[0]
    assert "nvds_remove_obj_meta_from_frame(frame, obj)" in low
    assert "tracker_detections.push_back" not in low
    assert "LOWCONF_GPU_SURFACE_POOL_SIZE = 4" in verifier
    assert "lowconf_gpu_surface_pools_initialize(0)" in verifier
    assert '"nvmm-pool-empty"' in verifier
    assert "cuda_device_to_device = 3" in verifier


def test_native_context_backend_is_persistent_and_bypasses_secondary_gstreamer_scheduling() -> None:
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    builder = (ROOT / "scripts/build_cam05_context448_batch6_candidate.py").read_text()
    direct_runner = (ROOT / "scripts/run_cam05_candidate_direct_replay.py").read_text()
    live_runner = (ROOT / "scripts/run_six_camera_v13_wall_candidate.py").read_text()
    assert 'g_getenv("V11_CONTEXT_VERIFIER_BACKEND")' in verifier
    assert "NvDsInferContext_Create(" in verifier
    assert "NvDsInferContext_QueueInputBatch(" in verifier
    assert "NvDsInferContext_DequeueOutputBatch(" in verifier
    assert "NvDsInferContext_ReleaseBatchOutput(" in verifier
    assert "NvDsInferContext_Destroy(handle)" in verifier
    assert "? lowconf_verifier_batch_size() : 1u" in verifier
    assert "NvDsInferContextBatchInput input{}" in verifier
    assert "input.numInputFrames = static_cast<unsigned int>(jobs.size())" in verifier
    assert "std::chrono::microseconds(750)" in verifier
    assert "lowconf_context_queue_high_water" in verifier
    assert "batch-input" not in verifier
    assert "native_context && !lowconf_verifier_gpu_crop_enabled()" in verifier
    assert "NVBUF_COLOR_FORMAT_RGB" in verifier
    assert "job.surface->surfaceList[0].width == lowconf_context_input_side.load()" in verifier
    assert "input_width == 320 || input_width == 448 || input_width == 512" in verifier
    assert "const guint context_side = lowconf_context_input_side.load()" in verifier
    assert "V11_CONTEXT_VERIFIER_BACKEND=nvdsinfer-context" in direct_runner
    assert "V11_CONTEXT_VERIFIER_BACKEND=nvdsinfer-context" in live_runner
    assert "-lnvds_infer" in builder
    assert 'manifest["native_context_backend"] = True' in builder


def test_direct_trt_verifier_uses_staged_parser_and_same_preprocessed_crop_stream() -> None:
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    runtime = (NATIVE / "context_trt_runtime.hpp").read_text()
    builder = (ROOT / "scripts/build_cam05_context448_trt_candidate.py").read_text()
    direct_runner = (ROOT / "scripts/run_cam05_candidate_direct_replay.py").read_text()
    live_runner = (ROOT / "scripts/run_six_camera_v13_wall_candidate.py").read_text()
    assert 'std::strcmp(value, "trt-runtime") == 0' in verifier
    assert "lowconf_context_trt_initialize(engine, parser, parse_function" in verifier
    assert "lowconf_context_trt_infer(api.device_buffer, api.stream" in verifier
    assert "lowconf_context_trt_shutdown()" in verifier
    assert "deserializeCudaEngine" in runtime
    assert "enqueueV3(cuda_stream)" in runtime
    assert "NvDsInferParseDetectionParams" in runtime
    assert "lowconf_context_trt_parser(layers, network, parse_params, parsed)" in runtime
    assert "lowconf_context_trt_iou(candidate, accepted) > lowconf_context_trt_nms_iou" in runtime
    assert "NvDsInferContext_QueueInputBatchPreprocessed" not in runtime
    assert 'V11_CONTEXT_VERIFIER_BACKEND=trt-runtime' in direct_runner
    assert 'V11_CONTEXT_VERIFIER_BACKEND=trt-runtime' in live_runner
    assert 'manifest["native_trt_runtime_backend"] = True' in builder
    assert "-lnvinfer_plugin -L/usr/local/cuda/lib64 -lcudart -ldl -pthread" in builder


def test_diagnostic_context_size_is_read_from_hashed_config_before_surface_pool_allocation() -> None:
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    assert verifier.index("if (native_context && !lowconf_verifier_contexts_initialize(config))") < verifier.index(
        "if (lowconf_verifier_gpu_crop_enabled() && !lowconf_gpu_surface_pools_initialize(0))")
    assert "params.width = native_context ? context_side : 640" in verifier
    assert "const guint output_side = native_context ? lowconf_context_input_side.load() : 640u" in verifier


def test_diagnostic_320_context_candidate_is_isolated_and_hash_pinned() -> None:
    builder = (ROOT / "scripts/build_cam05_context320_candidate.py").read_text()
    assert "sixcam-cam05-context448-direct-context-20261008-r64" in builder
    assert "sixcam-cam05-context320-direct-context-20261008-r65" in builder
    assert "yolo26m_context_320_b6_fp16.engine" in builder
    assert "batch-size=6" in builder
    assert "pre-cluster-threshold=0.20" in builder
    assert "NvDCF local6 config unchanged" in builder


def test_native_context_errors_fail_closed_without_claiming_person_absence() -> None:
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    helper = verifier.split("static void lowconf_fail_request(", 1)[1].split(
        "static bool lowconf_context_infer(", 1)[0]
    assert "verifier_confirmed = false" in helper
    assert "verifier_no_local_person = false" in helper
    assert "lowconf_queue_active_decision_locked" in helper
    assert '"CONTEXT_QUEUE_FAILED"' in verifier
    assert '"CONTEXT_OUTPUT_INVALID"' in verifier


def test_direct_file_canonical_evidence_is_batched_not_per_frame_unbuffered_io() -> None:
    tracker = (NATIVE / "tracker.hpp").read_text()
    setup = tracker.split('canonical_evidence = std::fopen("/work/canonical-objects.jsonl", "w");', 1)[1].split(
        "#endif", 1)[0]
    assert "std::setvbuf(canonical_evidence, nullptr, _IOFBF, 1024 * 1024);" in setup
    assert "_IOLBF" not in setup


def test_crop_copy_uses_persistent_nonblocking_stream_and_pinned_scratch() -> None:
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    assert 'dlsym(cuda_runtime, "cudaMemcpy2DAsync")' in verifier
    assert "cuda_stream_create(&scratch.stream, 1u)" in verifier
    assert "cuda_host_alloc(reinterpret_cast<void**>(&scratch.host)" in verifier
    assert "cuda_stream_sync(scratch.stream)" in verifier
    assert 'dlsym(cuda_runtime, "cudaMemcpy2D")' not in verifier
    assert "std::free(pinned_scratch)" not in verifier


def test_preprocessed_native_context_is_explicit_opt_in_and_does_not_touch_primary_path() -> None:
    verifier = (NATIVE / "lowconf_verifier.hpp").read_text()
    kernel = (NATIVE / "context_preprocess.cu").read_text()
    builder = (ROOT / "scripts/build_cam05_context320_preprocessed_candidate.py").read_text()
    direct_runner = (ROOT / "scripts/run_cam05_candidate_direct_replay.py").read_text()
    live_runner = (ROOT / "scripts/run_six_camera_v13_wall_candidate.py").read_text()
    assert 'g_getenv("V11_CONTEXT_VERIFIER_PREPROCESSED")' in verifier
    assert "params.inputFromPreprocessedTensor =" in verifier
    assert "lowconf_context_handles[0]->queueInputBatchPreprocessed(input)" in verifier
    assert "RGB-HWC->normalized-FP32-NCHW" in verifier
    assert "NVBUF_COLOR_FORMAT_RGB" in verifier
    assert "lowconf_context_net_scale" in verifier
    assert "rgb_hwc_to_normalized_nchw_fp32" in kernel
    assert "static_cast<float>(row[0]) * scale" in kernel
    assert 'manifest["native_preprocessed_context_backend"] = True' in builder
    assert 'V11_CONTEXT_PREPROCESS_LIBRARY={preprocessor[\'library\']}' in direct_runner
    assert 'V11_CONTEXT_PREPROCESS_LIBRARY={preprocessor[\'library\']}' in live_runner
    assert 'sha256(library) != preprocessor.get("library_sha256")' in direct_runner
    assert "normal >=0.25 NvDCF proposal input unchanged" in builder
    assert "const guint batch_limit = lowconf_verifier_context_backend_enabled()" in verifier
    assert "while (jobs.size() < batch_limit" in verifier
    assert '"PGIE_ISOLATION_REALTIME_PACE=1"' in direct_runner


def test_pgie_sidecar_uses_bounded_exact_frame_pts_cache() -> None:
    tracker = (NATIVE / "tracker.hpp").read_text()
    ring = (NATIVE / "exact_frame_ring.hpp").read_text()
    assert "TRACKER_DETECTION_CACHE_LIMIT = 64" in tracker
    assert "ExactFrameRing<TrackerDetection, 6, TRACKER_DETECTION_CACHE_LIMIT>" in tracker
    assert "std::move(detections)" in tracker
    assert "tracker_take_detections(frame->source_id, frame->frame_num," in tracker
    assert "frame->buf_pts, current_source_epoch)" in tracker
    assert "slot.source_epoch != source_epoch" in ring
    assert "slot.frame != frame || slot.pts != pts ||" in ring
    assert "std::move(slot.values)" in ring
    assert "tracker_detection_cache_misses" in tracker
    assert "tracker-detection-join.jsonl" in tracker
    assert "tracker_detector_frame[source_id]" not in tracker


def test_direct_file_clock_pacing_is_opt_in_and_file_only() -> None:
    main = (NATIVE / "main.cpp").read_text()
    runner = (ROOT / "scripts/run_lowconf_gate_a_replay.py").read_text()
    assert 'g_getenv("PGIE_ISOLATION_REALTIME_PACE")' in main
    pacing_branch = main.split('g_getenv("PGIE_ISOLATION_REALTIME_PACE")', 1)[1].split("#endif", 1)[0]
    assert 'std::strcmp(pace_files, "1") == 0' in pacing_branch
    assert 'ctx->uri.rfind("file://", 0) == 0' in pacing_branch
    assert 'make("identity", pacer_name.c_str())' in pacing_branch
    assert '"sync", pace_to_pts' in pacing_branch
    assert 'PGIE_ISOLATION_REALTIME_PACE=1' in runner
    # The opt-in is compiled only into the PGIE isolation harness, never the
    # production camera graph.
    before_loop = main.split("for (auto *ctx : sources)", 1)[1].split("GstElement *tiler", 1)[0]
    assert "#ifdef PGIE_ISOLATION" in before_loop


def test_gate_a_canonical_feed_uses_tmpfs_not_host_evidence_bind_mount() -> None:
    runner = (ROOT / "scripts/run_lowconf_gate_a_replay.py").read_text()
    assert '"-e", "V11_TRACKED_OBJECT_FEED_DIR=/dev/shm"' in runner
    assert '"-e", "V11_TRACKED_OBJECT_FEED_DIR=/work/tracked-feed"' not in runner
    assert '"canonical_frames": len(read_rows(run_dir / "canonical-objects.jsonl"))' in runner
