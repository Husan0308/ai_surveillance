// Included only in the detection build of the frozen six-camera native runner.
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <unistd.h>
#include <mutex>
#include <fstream>
#include <set>
#include <sstream>
#include <map>
#include <deque>
#include <array>
#include <condition_variable>
#include <thread>
#include <optional>
#include <tuple>
#include <nvbufsurface.h>
#include "latency_histogram.hpp"
#include "person_proposal_dedupe.hpp"
#include "auxiliary_exact_join.hpp"
#include "auxiliary_batch_cadence.hpp"
#if defined(YOLO26_TRACKER) || defined(PGIE_ISOLATION)
static bool verbose_evidence_enabled();
static guint32 f5r_source_frame_for_pts(guint source_id, guint64 pts_ns);
static void f5r_tracker_frame_ready(guint source_id, guint tracker_frame_num,
    guint source_frame_num, guint64 pts_ns, guint object_count,
    guint64 tracker_ready_ns, const char *snapshot_json, size_t snapshot_size);
// Optional per-frame stage trace used only by isolated latency diagnostics.
// Stages: 1=mux src, 2=PGIE sink, 3=PGIE src, 4=nvtracker sink, 5=nvtracker src.
static void pipeline_stage_record_batch(GstBuffer *buffer, guint stage, guint64 now_ns);
#include "lowconf_contract.hpp"
#include "tracker.hpp"
#include "lowconf_verifier.hpp"
#endif

// Per-frame JSONL is forensic evidence, not runtime state. On a six-source
// inference path, synchronously formatting/flushing every detector, tracker,
// and preview row from streaming callbacks adds avoidable I/O latency. Direct
// file-isolation builds keep their mandatory evidence; live candidates opt in
// explicitly with V11_VERBOSE_EVIDENCE=1.
static bool verbose_evidence_enabled() {
#ifdef PGIE_ISOLATION
  return true;
#else
  const char *value = g_getenv("V11_VERBOSE_EVIDENCE");
  return value && std::strcmp(value, "1") == 0;
#endif
}

struct DetectionCounter {
  std::atomic<guint64> frames{0}, persons{0}, batches{0}, errors{0}, last_unix_ms{0};
};
static DetectionCounter detections[6];
static std::atomic<guint64> infer_batches{0}, infer_frames{0}, infer_errors{0};
static std::atomic<guint64> infer_us{0}, infer_timed{0}, infer_max_us{0};
static std::atomic<guint64> auxiliary_detector_objects{0};
static std::atomic<guint64> auxiliary_duplicate_suppressed{0};
static std::atomic<guint64> auxiliary_detector_admitted{0};
static std::atomic<guint64> auxiliary_input_queue_overruns{0};
static std::atomic<guint64> auxiliary_meta_queue_overruns{0};
static std::atomic<guint> auxiliary_input_queue_high_water{0};
static std::atomic<guint> auxiliary_meta_queue_high_water{0};
static std::atomic<guint64> auxiliary_input_batches{0}, auxiliary_infer_batches{0},
    auxiliary_join_batches{0}, auxiliary_join_timeouts{0}, auxiliary_join_wait_ns{0},
    auxiliary_join_max_wait_ns{0}, auxiliary_injection_errors{0},
    auxiliary_join_events_reported{0};
static std::atomic<guint64> auxiliary_join_timeout_ms_configured{0};
static AuxiliaryBatchCadence auxiliary_preprocess_cadence{20, 0};
static std::atomic<guint64> auxiliary_preprocess_batches{0};
static std::atomic<guint64> auxiliary_sparse_skip_batches{0};
static std::atomic<guint64> auxiliary_sparse_skip_join_frames{0};
static std::atomic<guint64> auxiliary_sparse_skip_join_errors{0};
static std::atomic<guint64> auxiliary_input_frames[6], auxiliary_infer_frames[6],
    auxiliary_infer_objects[6], auxiliary_joined_objects[6], auxiliary_joined_frames[6],
    auxiliary_join_timeout_frames[6];
struct AuxiliaryProposal {
  float confidence = 0.0f;
  float left = 0.0f, top = 0.0f, width = 0.0f, height = 0.0f;
};
static AuxiliaryExactJoin<AuxiliaryProposal, 6, 64> auxiliary_exact_join;
static GstElement *auxiliary_input_queue_observed = nullptr;
static GstElement *auxiliary_meta_queue_observed = nullptr;
static std::atomic<guint64> same_component_duplicate_suppressed{0};
static LatencyHistogram infer_latency_histogram;
struct PipelineStageKey {
  guint source_id = 0;
  guint frame_num = 0;
  guint64 pts = 0;
  bool operator<(const PipelineStageKey &other) const {
    return std::tie(source_id, frame_num, pts) <
        std::tie(other.source_id, other.frame_num, other.pts);
  }
};
struct PipelineStageStamp {
  guint64 decoder_input_ns = 0, decoder_output_ns = 0;
  guint64 mux_output_ns = 0, pgie_input_ns = 0, pgie_output_ns = 0;
  guint64 tracker_input_ns = 0, tracker_output_ns = 0;
  guint mux_batch = 0, pgie_input_batch = 0, pgie_output_batch = 0;
  guint tracker_input_batch = 0, tracker_output_batch = 0;
};
static FILE *pipeline_stage_evidence = nullptr;
static std::mutex pipeline_stage_mutex;
static std::map<PipelineStageKey, PipelineStageStamp> pipeline_stage_stamps;
using ParserCount = unsigned long long (*)();
static ParserCount parser_errors, parser_rejected, parser_calls, parser_proposals;
static FILE *detection_evidence = nullptr, *frame_evidence = nullptr;
static FILE *lowconf_candidate_evidence = nullptr;
static FILE *proposal_duplicate_evidence = nullptr;
static GQuark infer_stamp;

static void auxiliary_queue_overrun(GstElement *, gpointer opaque) {
  auto *counter = static_cast<std::atomic<guint64> *>(opaque);
  if (counter) counter->fetch_add(1, std::memory_order_relaxed);
}

static guint auxiliary_queue_level(GstElement *queue) {
  guint level = 0;
  if (queue) g_object_get(queue, "current-level-buffers", &level, nullptr);
  return level;
}

#if defined(YOLO26_AUXILIARY_SPARSE_PREPROCESS)
static GstPadProbeReturn auxiliary_sparse_preprocess_gate(
    GstPad *, GstPadProbeInfo *info, gpointer) {
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer) return GST_PAD_PROBE_OK;
  if (!gst_buffer_get_nvds_batch_meta(buffer)) {
    fail("sparse auxiliary branch lost NvDsBatchMeta before preprocessing gate");
    return GST_PAD_PROBE_DROP;
  }
  auxiliary_preprocess_batches.fetch_add(1, std::memory_order_relaxed);
  if (auxiliary_preprocess_cadence.admit_next()) return GST_PAD_PROBE_OK;

  // Every primary frame performs an exact-key join. A deliberately skipped
  // auxiliary batch must therefore complete that same key with an empty
  // result before the auxiliary conversion queue is bypassed; otherwise the
  // primary branch waits for the full timeout on 19 of every 20 batches.
  auto *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
  auxiliary_sparse_skip_batches.fetch_add(1, std::memory_order_relaxed);
  for (auto *node = batch_meta->frame_meta_list; node; node = node->next) {
    auto *frame = static_cast<NvDsFrameMeta *>(node->data);
    if (!frame) continue;
    const AuxiliaryJoinKey key{frame->source_id, frame->frame_num, frame->buf_pts};
    auto result = auxiliary_exact_join.publish(key, std::vector<AuxiliaryProposal>{});
    if (!result.accepted) {
      auxiliary_sparse_skip_join_errors.fetch_add(1, std::memory_order_relaxed);
      fail("sparse auxiliary skip could not publish exact empty source/frame/PTS result");
      return GST_PAD_PROBE_DROP;
    }
    auxiliary_sparse_skip_join_frames.fetch_add(1, std::memory_order_relaxed);
  }
  return GST_PAD_PROBE_DROP;
}
#endif

static guint64 auxiliary_exact_join_timeout_ms() {
  const char *raw = g_getenv("V11_AUX_EXACT_JOIN_TIMEOUT_MS");
  if (!raw || !*raw) return 25;
  char *end = nullptr;
  const guint64 milliseconds = g_ascii_strtoull(raw, &end, 10);
  if (end == raw || !end || *end != '\0' || milliseconds > 100) {
    g_printerr("AUX_EXACT_JOIN_TIMEOUT_INVALID value=%s max_ms=100\n", raw);
    std::exit(2);
  }
  return milliseconds;
}

static GstPadProbeReturn auxiliary_input_observer(GstPad *, GstPadProbeInfo *info, gpointer) {
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer) return GST_PAD_PROBE_OK;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
  if (!batch_meta) return GST_PAD_PROBE_OK;
  auxiliary_input_batches.fetch_add(1, std::memory_order_relaxed);
  for (auto *node = batch_meta->frame_meta_list; node; node = node->next) {
    auto *frame = static_cast<NvDsFrameMeta *>(node->data);
    if (frame && frame->source_id < 6)
      auxiliary_input_frames[frame->source_id].fetch_add(1, std::memory_order_relaxed);
  }
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn auxiliary_infer_observer(GstPad *, GstPadProbeInfo *info, gpointer) {
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer) return GST_PAD_PROBE_OK;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
  if (!batch_meta) return GST_PAD_PROBE_OK;
  auxiliary_infer_batches.fetch_add(1, std::memory_order_relaxed);
  for (auto *frame_node = batch_meta->frame_meta_list; frame_node; frame_node = frame_node->next) {
    auto *frame = static_cast<NvDsFrameMeta *>(frame_node->data);
    if (!frame || frame->source_id >= 6) continue;
    auxiliary_infer_frames[frame->source_id].fetch_add(1, std::memory_order_relaxed);
    std::vector<AuxiliaryProposal> proposals;
    for (auto *object_node = frame->obj_meta_list; object_node; object_node = object_node->next) {
      auto *object = static_cast<NvDsObjectMeta *>(object_node->data);
      if (object && object->unique_component_id == 3 && object->class_id == 0) {
        auxiliary_infer_objects[frame->source_id].fetch_add(1, std::memory_order_relaxed);
        const auto &rect = object->rect_params;
        proposals.push_back({object->confidence, rect.left, rect.top, rect.width, rect.height});
      }
    }
    const AuxiliaryJoinKey key{frame->source_id, frame->frame_num, frame->buf_pts};
    const auto result = auxiliary_exact_join.publish(key, std::move(proposals));
    if (!result.accepted) {
      g_printerr("AUX_EXACT_JOIN_PUBLISH_REJECT source=%u frame=%u pts=%" G_GUINT64_FORMAT "\n",
          frame->source_id, frame->frame_num, frame->buf_pts);
    }
  }
  return GST_PAD_PROBE_OK;
}

#if defined(YOLO26_AUXILIARY_896) || defined(YOLO26_AUXILIARY_DETECTOR)
static GstPadProbeReturn infer_exit(GstPad*, GstPadProbeInfo*, gpointer);

static bool auxiliary_join_inject_object(
    NvDsBatchMeta *batch, NvDsFrameMeta *frame, const AuxiliaryProposal &proposal) {
  if (!batch || !frame || frame->source_id >= 6 ||
      !std::isfinite(proposal.confidence) || proposal.confidence < 0.0f ||
      proposal.confidence > 1.0f || !std::isfinite(proposal.left) ||
      !std::isfinite(proposal.top) || !std::isfinite(proposal.width) ||
      !std::isfinite(proposal.height) || proposal.width <= 0.0f || proposal.height <= 0.0f)
    return false;
  nvds_acquire_meta_lock(batch);
  NvDsObjectMeta *object = nvds_acquire_obj_meta_from_pool(batch);
  if (!object) {
    nvds_release_meta_lock(batch);
    return false;
  }
  object->unique_component_id = TRACKER_AUXILIARY_GIE_ID;
  object->confidence = proposal.confidence;
  object->object_id = UNTRACKED_OBJECT_ID;
  object->class_id = 0;
  g_strlcpy(object->obj_label, "person", sizeof(object->obj_label));
  auto &rect = object->rect_params;
  rect.left = proposal.left;
  rect.top = proposal.top;
  rect.width = proposal.width;
  rect.height = proposal.height;
  object->detector_bbox_info.org_bbox_coords.left = rect.left;
  object->detector_bbox_info.org_bbox_coords.top = rect.top;
  object->detector_bbox_info.org_bbox_coords.width = rect.width;
  object->detector_bbox_info.org_bbox_coords.height = rect.height;
  nvds_add_obj_meta_to_frame(frame, object, nullptr);
  nvds_release_meta_lock(batch);
  return true;
}

static GstPadProbeReturn auxiliary_exact_join_before_tracker(
    GstPad *pad, GstPadProbeInfo *info, gpointer) {
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer) return GST_PAD_PROBE_OK;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
  if (!batch_meta) return GST_PAD_PROBE_OK;
  const auto start = std::chrono::steady_clock::now();
  const auto deadline = start + std::chrono::milliseconds(
      auxiliary_join_timeout_ms_configured.load(std::memory_order_relaxed));
  guint64 joined_objects = 0;
  guint64 joined_frames = 0;
  guint64 timeout_frames = 0;
  for (auto *frame_node = batch_meta->frame_meta_list; frame_node; frame_node = frame_node->next) {
    auto *frame = static_cast<NvDsFrameMeta *>(frame_node->data);
    if (!frame || frame->source_id >= 6) continue;
    const AuxiliaryJoinKey key{frame->source_id, frame->frame_num, frame->buf_pts};
    auto result = auxiliary_exact_join.take_until(key, deadline);
    if (result.status != AuxiliaryExactJoin<AuxiliaryProposal, 6, 64>::WaitStatus::Ready) {
      ++timeout_frames;
      auxiliary_join_timeout_frames[frame->source_id].fetch_add(1, std::memory_order_relaxed);
      if (auxiliary_join_timeouts.fetch_add(1, std::memory_order_relaxed) < 24) {
        g_printerr("AUX_EXACT_JOIN_TIMEOUT camera=CAM-%02u source=%u frame=%u pts=%" G_GUINT64_FORMAT " status=%s timeout_ms=%" G_GUINT64_FORMAT "\n",
            frame->source_id + 1, frame->source_id, frame->frame_num, frame->buf_pts,
            result.status == AuxiliaryExactJoin<AuxiliaryProposal, 6, 64>::WaitStatus::InvalidKey
                ? "INVALID_KEY" : "NOT_READY",
            auxiliary_join_timeout_ms_configured.load(std::memory_order_relaxed));
      }
      continue;
    }
    ++joined_frames;
    auxiliary_joined_frames[frame->source_id].fetch_add(1, std::memory_order_relaxed);
    for (const auto &proposal : result.values) {
      if (!auxiliary_join_inject_object(batch_meta, frame, proposal)) {
        auxiliary_injection_errors.fetch_add(1, std::memory_order_relaxed);
        g_printerr("AUX_EXACT_JOIN_INVALID_OBJECT camera=CAM-%02u frame=%u pts=%" G_GUINT64_FORMAT "\n",
            frame->source_id + 1, frame->frame_num, frame->buf_pts);
        return GST_PAD_PROBE_DROP;
      }
      ++joined_objects;
      auxiliary_joined_objects[frame->source_id].fetch_add(1, std::memory_order_relaxed);
    }
  }
  auxiliary_join_batches.fetch_add(1, std::memory_order_relaxed);
  const guint64 elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - start).count();
  auxiliary_join_wait_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
  guint64 previous_max = auxiliary_join_max_wait_ns.load(std::memory_order_relaxed);
  while (elapsed_ns > previous_max && !auxiliary_join_max_wait_ns.compare_exchange_weak(
      previous_max, elapsed_ns, std::memory_order_relaxed)) {}
  if ((joined_frames || timeout_frames) &&
      auxiliary_join_events_reported.fetch_add(1, std::memory_order_relaxed) < 12)
    g_print("AUX_EXACT_JOIN batch_pts=%" G_GUINT64_FORMAT
        " matched_frames=%" G_GUINT64_FORMAT
        " timeout_frames=%" G_GUINT64_FORMAT
        " injected=%" G_GUINT64_FORMAT " wait_ms=%.3f\n",
        GST_BUFFER_PTS(buffer), joined_frames, timeout_frames, joined_objects,
        elapsed_ns / 1e6);
  return infer_exit(pad, info, nullptr);
}
#endif

static void auxiliary_queue_update_high_water(std::atomic<guint> &high_water, guint level) {
  guint old = high_water.load(std::memory_order_relaxed);
  while (level > old && !high_water.compare_exchange_weak(
      old, level, std::memory_order_relaxed)) {}
}

#ifdef PGIE_ISOLATION
#ifdef PGIE_ISOLATION_CAPTURE_PIXELS
static std::set<guint> pgie_pixel_frames() {
  std::set<guint> frames;
  const char *raw = g_getenv("PGIE_ISOLATION_CAPTURE_FRAMES");
  if (!raw) return frames;
  std::stringstream input(raw);
  std::string token;
  while (std::getline(input, token, ',')) {
    char *end = nullptr;
    const unsigned long value = std::strtoul(token.c_str(), &end, 10);
    if (end && *end == '\0' && !token.empty() && value <= G_MAXUINT) frames.insert(value);
  }
  return frames;
}

static void dump_mux_pixels(GstBuffer *buffer, NvDsBatchMeta *batch_meta) {
  static const std::set<guint> selected = pgie_pixel_frames();
  static FILE *manifest = std::fopen("/work/mux-pixels.csv", "w");
  static FILE *debug = std::fopen("/work/mux-pixel-debug.log", "w");
  if (selected.empty() || !manifest || !batch_meta) return;
  GstMapInfo map = GST_MAP_INFO_INIT;
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    if (debug) { std::fprintf(debug, "buffer_map_failed\n"); std::fflush(debug); }
    return;
  }
  auto *surface = reinterpret_cast<NvBufSurface*>(map.data);
  for (auto *node = batch_meta->frame_meta_list; node; node = node->next) {
    auto *frame = static_cast<NvDsFrameMeta*>(node->data);
    if (!selected.count(frame->frame_num) || frame->batch_id >= surface->batchSize) continue;
    auto &entry = surface->surfaceList[frame->batch_id];
    if (debug) { std::fprintf(debug, "surface source=%u frame=%u batch=%u mem=%u fmt=%u width=%u height=%u planes=%u data=%p offset1=%u pitch0=%u pitch1=%u\n",
        frame->source_id, frame->frame_num, frame->batch_id, surface->memType,
        static_cast<guint>(entry.colorFormat), entry.width, entry.height,
        entry.planeParams.num_planes, entry.dataPtr, entry.planeParams.offset[1],
        entry.planeParams.pitch[0], entry.planeParams.pitch[1]); std::fflush(debug); }
    const bool nv12_8bit = entry.colorFormat == NVBUF_COLOR_FORMAT_NV12 ||
        entry.colorFormat == NVBUF_COLOR_FORMAT_NV12_ER ||
        entry.colorFormat == NVBUF_COLOR_FORMAT_NV12_709 ||
        entry.colorFormat == NVBUF_COLOR_FORMAT_NV12_709_ER ||
        entry.colorFormat == NVBUF_COLOR_FORMAT_NV12_2020;
    if (nv12_8bit && entry.dataPtr) {
      std::ostringstream path;
      path << "/work/pixels/source_" << frame->source_id << "_frame_" << frame->frame_num << ".nv12";
      std::ofstream out(path.str(), std::ios::binary | std::ios::trunc);
      const guint width = entry.width, height = entry.height;
      const guint y_pitch = entry.planeParams.pitch[0], uv_pitch = entry.planeParams.pitch[1];
      std::vector<guint8> y(static_cast<size_t>(width) * height);
      std::vector<guint8> uv(static_cast<size_t>(width) * height / 2);
      using CudaSetDeviceFn = int (*)(int);
      using CudaMemcpy2DFn = int (*)(void*, size_t, const void*, size_t, size_t, size_t, int);
      static void *cuda_runtime = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
      static auto cuda_set_device = reinterpret_cast<CudaSetDeviceFn>(
          cuda_runtime ? dlsym(cuda_runtime, "cudaSetDevice") : nullptr);
      static auto cuda_memcpy_2d = reinterpret_cast<CudaMemcpy2DFn>(
          cuda_runtime ? dlsym(cuda_runtime, "cudaMemcpy2D") : nullptr);
      const int set_device = cuda_set_device ? cuda_set_device(static_cast<int>(surface->gpuId)) : -1;
      constexpr int cuda_device_to_host = 2;
      const int y_status = set_device == 0 && cuda_memcpy_2d ? cuda_memcpy_2d(
          y.data(), width, static_cast<const guint8*>(entry.dataPtr), y_pitch,
          width, height, cuda_device_to_host) : (set_device ? set_device : -1);
      const int uv_status = y_status == 0 && cuda_memcpy_2d ? cuda_memcpy_2d(
          uv.data(), width, static_cast<const guint8*>(entry.dataPtr) + entry.planeParams.offset[1],
          uv_pitch, width, height / 2, cuda_device_to_host) : y_status;
      if (uv_status != 0) {
        if (debug) { std::fprintf(debug, "cuda_copy_failed source=%u frame=%u status=%d\n",
            frame->source_id, frame->frame_num, uv_status); std::fflush(debug); }
        continue;
      }
      out.write(reinterpret_cast<const char*>(y.data()), y.size());
      out.write(reinterpret_cast<const char*>(uv.data()), uv.size());
      std::fprintf(manifest, "%u,%u,%lu,%u,%u,%u,%u,%u,%lu,%s\n", frame->source_id,
          frame->frame_num, frame->buf_pts, width, height, y_pitch, uv_pitch,
          static_cast<guint>(entry.colorFormat), static_cast<guint64>(width) * height * 3 / 2,
          path.str().c_str());
      std::fflush(manifest);
    }
    NvBufSurfaceUnMap(surface, frame->batch_id, -1);
  }
  gst_buffer_unmap(buffer, &map);
}
#endif
#endif

// Staged-only latest-frame shared-memory output for the existing PySide wall.
// Each source has one bounded/leaky demux branch and one locked shared slot.
struct F5RPreviewWriter {
  int fd = -1;
  guint8 *map = nullptr;
  size_t mapped_size = 0;
  guint width = 0, height = 0, stride = 0, payload_size = 0;
  std::string path;
  guint64 sequence = 0;
  guint64 last_ns = 0;
  double fps = 0.0;
  std::mutex mutex;
};
struct F5RPreviewBranch {
  guint source_id = 0;
  guint width = 0, height = 0;
  bool fullscreen = false;
};
static F5RPreviewWriter f5r_preview[6];
static F5RPreviewWriter f5r_fullscreen_preview;
static F5RPreviewBranch f5r_preview_branches[6], f5r_fullscreen_branches[6];
static GstElement *f5r_fullscreen_valves[6] = {};
static std::atomic<guint64> f5r_preview_frames[6];
static std::atomic<guint64> f5r_preview_errors{0};
static std::atomic<guint64> f5r_preview_reference_misses[6];
static std::atomic<guint64> f5r_decoder_input_reference_misses[6];
static std::atomic<guint64> f5r_preview_joined[6], f5r_preview_join_misses[6],
    f5r_preview_sync_mismatches[6], f5r_preview_stale_drops[6], f5r_preview_queue_drops{0};
static std::atomic<guint64> f5r_preview_join_cache_high_water[6];
static std::atomic<guint64> f5r_preview_prepare_frames[6], f5r_fullscreen_prepare_frames[6];
static std::atomic<guint64> f5r_tile_queue_in[6], f5r_tile_queue_out[6],
    f5r_tile_convert_out[6], f5r_tile_sink_in[6], f5r_tile_samples[6];
static std::atomic<guint64> f5r_fullscreen_queue_in[6], f5r_fullscreen_queue_out[6],
    f5r_fullscreen_convert_out[6], f5r_fullscreen_sink_in[6], f5r_fullscreen_samples[6];
static std::atomic<gint> f5r_fullscreen_selected{-1};
// Version 5 keeps a JSON snapshot beside the image in the same flock-protected
// latest-only slot. A reader can therefore never pair pixels with another PTS.
static constexpr guint F5R_PREVIEW_W = 960, F5R_PREVIEW_H = 540;
static constexpr guint F5R_FULLSCREEN_W = 1920, F5R_FULLSCREEN_H = 1080;
// Preview and analytics are parallel branches. Keep enough exact-frame entries
// to cover measured startup/steady-state skew while remaining strictly bounded.
// At 20 FPS this is 1.6s; entries older than the bound are discarded, never
// paired approximately or reused for a later image.
static constexpr size_t F5R_JOIN_ENTRY_LIMIT = 32;
static constexpr guint F5R_PREVIEW_HEADER = 65536;
static constexpr guint F5R_PREVIEW_METADATA_OFFSET = 128;
static constexpr guint F5R_PREVIEW_METADATA_MAX = F5R_PREVIEW_HEADER - F5R_PREVIEW_METADATA_OFFSET;
static constexpr const char *F5R_PREVIEW_NOTIFY_PATH = "/dev/shm/v11_ui_preview_notify_v1.sock";
static int f5r_preview_notify_fd = -1;
static sockaddr_un f5r_preview_notify_address{};

using F5RJoinKey = std::pair<guint32, guint64>;
struct F5RPreparedFrame {
  guint source_id = 0, frame_num = 0, width = 0, height = 0, stride = 0;
  guint64 pts = 0, dts = 0, decoder_input_ns = 0, decoder_output_ns = 0;
  guint64 prepare_start_ns = 0, prepare_complete_ns = 0;
  bool fullscreen = false;
  GstBuffer *buffer = nullptr;
  F5RPreparedFrame() = default;
  F5RPreparedFrame(const F5RPreparedFrame&) = delete;
  F5RPreparedFrame& operator=(const F5RPreparedFrame&) = delete;
  F5RPreparedFrame(F5RPreparedFrame&& other) noexcept
      : source_id(other.source_id), frame_num(other.frame_num), width(other.width),
        height(other.height), stride(other.stride), pts(other.pts), dts(other.dts),
        decoder_input_ns(other.decoder_input_ns), decoder_output_ns(other.decoder_output_ns),
        prepare_start_ns(other.prepare_start_ns), prepare_complete_ns(other.prepare_complete_ns),
        fullscreen(other.fullscreen), buffer(other.buffer) {
    other.buffer = nullptr;
  }
  F5RPreparedFrame& operator=(F5RPreparedFrame&& other) noexcept {
    if (this == &other) return *this;
    if (buffer) gst_buffer_unref(buffer);
    source_id = other.source_id; frame_num = other.frame_num;
    width = other.width; height = other.height; stride = other.stride;
    pts = other.pts; dts = other.dts;
    decoder_input_ns = other.decoder_input_ns; decoder_output_ns = other.decoder_output_ns;
    prepare_start_ns = other.prepare_start_ns; prepare_complete_ns = other.prepare_complete_ns;
    fullscreen = other.fullscreen; buffer = other.buffer; other.buffer = nullptr;
    return *this;
  }
  ~F5RPreparedFrame() { if (buffer) gst_buffer_unref(buffer); }
};
struct F5RTrackerReady {
  guint tracker_frame_num = 0, source_frame_num = 0, object_count = 0;
  guint64 pts = 0, ready_ns = 0;
  std::string snapshot_json;
};
struct F5RJoinedPreview {
  F5RPreparedFrame frame;
  F5RTrackerReady tracker;
};
struct F5RJoinEntry {
  std::optional<F5RPreparedFrame> tile;
  std::optional<F5RPreparedFrame> fullscreen;
  std::optional<F5RTrackerReady> tracker;
  guint64 tile_prepare_start_ns = 0, fullscreen_prepare_start_ns = 0;
  bool tile_published = false, fullscreen_published = false;
  bool fullscreen_expected = false;
};
static std::mutex f5r_join_mutex;
static std::map<F5RJoinKey, F5RJoinEntry> f5r_join_entries[6];
static std::map<F5RJoinKey, guint64> f5r_prepare_starts[6];
static std::mutex f5r_publish_mutex;
static std::condition_variable f5r_publish_cv;
static std::deque<F5RJoinedPreview> f5r_publish_jobs;
static std::thread f5r_publish_thread;
static bool f5r_publish_stopping = false;
static FILE *f5r_join_evidence = nullptr;
static std::string f5r_fullscreen_request_path = "/dev/shm/v11_ui_fullscreen_request_v1";

static void f5r_preview_notify_open() {
  if (f5r_preview_notify_fd >= 0) return;
  f5r_preview_notify_fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (f5r_preview_notify_fd < 0) return;
  f5r_preview_notify_address.sun_family = AF_UNIX;
  std::strncpy(f5r_preview_notify_address.sun_path, F5R_PREVIEW_NOTIFY_PATH,
      sizeof(f5r_preview_notify_address.sun_path) - 1);
}

static void f5r_preview_notify(guint source_id, bool fullscreen) {
  if (f5r_preview_notify_fd < 0 || source_id >= 6) return;
  const guint8 event[2] = {static_cast<guint8>(fullscreen ? 'F' : 'T'),
                           static_cast<guint8>(source_id)};
  (void)sendto(f5r_preview_notify_fd, event, sizeof(event), MSG_DONTWAIT,
      reinterpret_cast<const sockaddr*>(&f5r_preview_notify_address),
      sizeof(f5r_preview_notify_address));
}

static guint f5r_preview_capture_frame() {
  static const guint frame = [] {
    const char *raw = g_getenv("V11_PREVIEW_CAPTURE_SOURCE_FRAME");
    if (!raw || !*raw) return G_MAXUINT;
    char *end = nullptr;
    const unsigned long value = std::strtoul(raw, &end, 10);
    return end && *end == '\0' && value <= G_MAXUINT ? static_cast<guint>(value) : G_MAXUINT;
  }();
  return frame;
}

static const char *f5r_preview_capture_dir() {
  static const char *path = g_getenv("V11_PREVIEW_CAPTURE_DIR");
  return path;
}

static std::atomic_bool f5r_preview_quality_capture_done[2] = {};

static void f5r_preview_quality_capture_nv12(
    GstBuffer *buffer, guint source_id, guint32 source_frame_num,
    guint64 pts_ns, const char *stage, guint surface_index) {
  const char *directory = f5r_preview_capture_dir();
  const guint selected_frame = f5r_preview_capture_frame();
  if (!directory || !*directory || source_id != 0 || source_frame_num != selected_frame) return;
  const guint stage_index = std::strcmp(stage, "decoded-source") == 0 ? 0u : 1u;
  bool expected = false;
  if (!f5r_preview_quality_capture_done[stage_index].compare_exchange_strong(expected, true)) return;

  GstMapInfo map = GST_MAP_INFO_INIT;
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    g_printerr("PREVIEW_QUALITY_CAPTURE_FAIL stage=%s reason=buffer-map\n", stage);
    return;
  }
  auto *surface = reinterpret_cast<NvBufSurface*>(map.data);
  if (!surface || surface_index >= surface->batchSize) {
    g_printerr("PREVIEW_QUALITY_CAPTURE_FAIL stage=%s reason=surface-index index=%u\n", stage, surface_index);
    gst_buffer_unmap(buffer, &map);
    return;
  }
  const auto &entry = surface->surfaceList[surface_index];
  const bool nv12 = entry.colorFormat == NVBUF_COLOR_FORMAT_NV12 ||
      entry.colorFormat == NVBUF_COLOR_FORMAT_NV12_ER ||
      entry.colorFormat == NVBUF_COLOR_FORMAT_NV12_709 ||
      entry.colorFormat == NVBUF_COLOR_FORMAT_NV12_709_ER ||
      entry.colorFormat == NVBUF_COLOR_FORMAT_NV12_2020;
  if (!nv12 || !entry.dataPtr || entry.width == 0 || entry.height == 0 ||
      entry.width % 2 || entry.height % 2) {
    g_printerr("PREVIEW_QUALITY_CAPTURE_FAIL stage=%s reason=unsupported-surface format=%u size=%ux%u\n",
        stage, static_cast<guint>(entry.colorFormat), entry.width, entry.height);
    gst_buffer_unmap(buffer, &map);
    return;
  }

  using CudaSetDeviceFn = int (*)(int);
  using CudaMemcpy2DFn = int (*)(void*, size_t, const void*, size_t, size_t, size_t, int);
  static void *cuda_runtime = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
  static auto cuda_set_device = reinterpret_cast<CudaSetDeviceFn>(
      cuda_runtime ? dlsym(cuda_runtime, "cudaSetDevice") : nullptr);
  static auto cuda_memcpy_2d = reinterpret_cast<CudaMemcpy2DFn>(
      cuda_runtime ? dlsym(cuda_runtime, "cudaMemcpy2D") : nullptr);
  const int set_device = cuda_set_device ? cuda_set_device(static_cast<int>(surface->gpuId)) : -1;
  std::vector<guint8> y(static_cast<size_t>(entry.width) * entry.height);
  std::vector<guint8> uv(static_cast<size_t>(entry.width) * entry.height / 2);
  constexpr int cuda_device_to_host = 2;
  const guint y_pitch = entry.planeParams.pitch[0];
  const guint uv_pitch = entry.planeParams.pitch[1];
  const int y_status = set_device == 0 && cuda_memcpy_2d ? cuda_memcpy_2d(
      y.data(), entry.width, static_cast<const guint8*>(entry.dataPtr), y_pitch,
      entry.width, entry.height, cuda_device_to_host) : (set_device ? set_device : -1);
  const int uv_status = y_status == 0 && cuda_memcpy_2d ? cuda_memcpy_2d(
      uv.data(), entry.width, static_cast<const guint8*>(entry.dataPtr) + entry.planeParams.offset[1],
      uv_pitch, entry.width, entry.height / 2, cuda_device_to_host) : y_status;

  if (uv_status == 0 && g_mkdir_with_parents(directory, 0755) == 0) {
    const std::string filename = std::string(directory) + "/cam01-" + stage + "-frame-" +
        std::to_string(source_frame_num) + ".nv12";
    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(y.data()), static_cast<std::streamsize>(y.size()));
    out.write(reinterpret_cast<const char*>(uv.data()), static_cast<std::streamsize>(uv.size()));
    out.close();
    std::ofstream manifest(std::string(directory) + "/native-captures.jsonl", std::ios::app);
    manifest << "{\"camera_id\":\"CAM-01\",\"stage\":\"" << stage
        << "\",\"source_frame_num\":" << source_frame_num
        << ",\"pts_ns\":" << pts_ns << ",\"width\":" << entry.width
        << ",\"height\":" << entry.height << ",\"stride_y\":" << y_pitch
        << ",\"stride_uv\":" << uv_pitch << ",\"format\":\"NV12\",\"color_format_id\":"
        << static_cast<guint>(entry.colorFormat) << ",\"layout_id\":"
        << static_cast<guint>(entry.layout) << ",\"memory_type\":" << surface->memType
        << ",\"gpu_id\":" << surface->gpuId << ",\"payload_bytes\":"
        << y.size() + uv.size() << ",\"path\":\"" << filename << "\"}\n";
    std::fflush(stdout);
    g_print("PREVIEW_QUALITY_CAPTURE stage=%s frame=%u pts=%lu size=%ux%u format=NV12 strides=%u/%u\n",
        stage, source_frame_num, pts_ns, entry.width, entry.height, y_pitch, uv_pitch);
  } else {
    g_printerr("PREVIEW_QUALITY_CAPTURE_FAIL stage=%s reason=cuda-or-write status=%d\n", stage, uv_status);
  }
  gst_buffer_unmap(buffer, &map);
}

static guint32 f5r_source_frame_for_pts(guint source_id, guint64 pts_ns) {
  if (source_id >= sources.size()) return G_MAXUINT32;
  SourceCtx *owner = sources[source_id];
  std::lock_guard<std::mutex> guard(owner->decoded_clock_mutex);
  auto found = owner->decoded_clock_ns.find(pts_ns);
  return found == owner->decoded_clock_ns.end() ? G_MAXUINT32 : found->second.frame_num;
}

static GstPadProbeReturn f5r_preview_input_capture_probe(
    GstPad*, GstPadProbeInfo *info, gpointer) {
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer || !f5r_preview_capture_dir() || f5r_preview_capture_frame() == G_MAXUINT) return GST_PAD_PROBE_OK;
  NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(buffer);
  if (!batch) return GST_PAD_PROBE_OK;
  for (auto *node = batch->frame_meta_list; node; node = node->next) {
    auto *frame = static_cast<NvDsFrameMeta*>(node->data);
    if (!frame || frame->source_id != 0 || frame->batch_id == G_MAXUINT) continue;
    const guint32 source_frame_num = f5r_source_frame_for_pts(frame->source_id, frame->buf_pts);
    if (source_frame_num == f5r_preview_capture_frame()) {
      f5r_preview_quality_capture_nv12(buffer, frame->source_id, source_frame_num,
          frame->buf_pts, "preview-input", frame->batch_id);
      break;
    }
  }
  return GST_PAD_PROBE_OK;
}

static DecoderFrameStamp f5r_decode_stamp_for_pts(guint source_id, guint64 pts) {
  if (source_id >= sources.size()) return {};
  SourceCtx *owner = sources[source_id];
  std::lock_guard<std::mutex> guard(owner->decoded_clock_mutex);
  auto found = owner->decoded_clock_ns.find(pts);
  return found == owner->decoded_clock_ns.end() ? DecoderFrameStamp{} : found->second;
}

static bool f5r_writer_open(F5RPreviewWriter &writer, const std::string &path,
                            guint width, guint height) {
  writer.width = width;
  writer.height = height;
  writer.stride = width * 4;
  writer.payload_size = writer.stride * height;
  writer.path = path;
  writer.mapped_size = static_cast<size_t>(F5R_PREVIEW_HEADER) + writer.payload_size;
  writer.fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (writer.fd < 0 || ftruncate(writer.fd, static_cast<off_t>(writer.mapped_size)) != 0) return false;
  writer.map = static_cast<guint8*>(mmap(nullptr, writer.mapped_size,
      PROT_READ | PROT_WRITE, MAP_SHARED, writer.fd, 0));
  if (writer.map == MAP_FAILED) { writer.map = nullptr; return false; }
  std::memset(writer.map, 0, F5R_PREVIEW_HEADER);
  std::memcpy(writer.map, "V11UI01\0", 8);
  const guint32 version = 5;
  std::memcpy(writer.map + 8, &version, sizeof(version));
  return true;
}

static void f5r_writer_close(F5RPreviewWriter &writer) {
  if (writer.map) { munmap(writer.map, writer.mapped_size); writer.map = nullptr; }
  if (writer.fd >= 0) { close(writer.fd); writer.fd = -1; }
  if (!writer.path.empty()) unlink(writer.path.c_str());
}

static bool f5r_write_joined_preview(F5RJoinedPreview &joined) {
  auto &frame = joined.frame;
  auto &tracker = joined.tracker;
  F5RPreviewWriter &writer = frame.fullscreen ? f5r_fullscreen_preview : f5r_preview[frame.source_id];
  if (frame.fullscreen && f5r_fullscreen_selected.load() != static_cast<gint>(frame.source_id)) {
    ++f5r_preview_stale_drops[frame.source_id];
    return false;
  }
  if (!writer.map || !frame.buffer ||
      tracker.snapshot_json.size() > F5R_PREVIEW_METADATA_MAX ||
      frame.frame_num != tracker.source_frame_num || frame.pts != tracker.pts) {
    ++f5r_preview_sync_mismatches[frame.source_id];
    return false;
  }
  const guint64 publish_start_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  std::lock_guard<std::mutex> guard(writer.mutex);
  if (writer.last_ns && publish_start_ns > writer.last_ns) {
    const double instant = 1e9 / static_cast<double>(publish_start_ns - writer.last_ns);
    writer.fps = writer.fps ? writer.fps * 0.85 + instant * 0.15 : instant;
  }
  writer.last_ns = publish_start_ns;
  ++writer.sequence;
  flock(writer.fd, LOCK_EX);
  GstVideoInfo video_info;
  GstVideoFrame video_frame;
  bool mapped = gst_video_info_set_format(&video_info, GST_VIDEO_FORMAT_BGRx,
      frame.width, frame.height) &&
      gst_video_frame_map(&video_frame, &video_info, frame.buffer, GST_MAP_READ);
  if (!mapped) {
    flock(writer.fd, LOCK_UN);
    ++f5r_preview_errors;
    return false;
  }
  const guint8 *src = static_cast<const guint8*>(GST_VIDEO_FRAME_PLANE_DATA(&video_frame, 0));
  const gint src_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&video_frame, 0);
  if (src_stride < static_cast<gint>(writer.stride)) {
    gst_video_frame_unmap(&video_frame);
    flock(writer.fd, LOCK_UN);
    ++f5r_preview_errors;
    return false;
  }
  for (guint y = 0; y < frame.height; ++y)
    std::memcpy(writer.map + F5R_PREVIEW_HEADER + y * writer.stride,
        src + y * src_stride, writer.stride);
  gst_video_frame_unmap(&video_frame);
  if (!tracker.snapshot_json.empty())
    std::memcpy(writer.map + F5R_PREVIEW_METADATA_OFFSET,
        tracker.snapshot_json.data(), tracker.snapshot_json.size());
  const guint32 fields[6] = {writer.width, writer.height, writer.stride,
      writer.payload_size, tracker.object_count,
      static_cast<guint32>(std::max(0.0, writer.fps * 1000.0))};
  std::memcpy(writer.map, "V11UI01\0", 8);
  const guint32 version = 5;
  std::memcpy(writer.map + 8, &version, sizeof(version));
  std::memcpy(writer.map + 12, &writer.sequence, 8);
  std::memcpy(writer.map + 28, fields, sizeof(fields));
  std::memcpy(writer.map + 52, &frame.decoder_input_ns, 8);
  std::memcpy(writer.map + 60, &frame.decoder_output_ns, 8);
  std::memcpy(writer.map + 68, &frame.pts, 8);
  std::memcpy(writer.map + 76, &frame.dts, 8);
  std::memcpy(writer.map + 84, &frame.frame_num, 4);
  const guint32 reserved = 0;
  std::memcpy(writer.map + 88, &reserved, 4);
  std::memcpy(writer.map + 92, &frame.prepare_start_ns, 8);
  std::memcpy(writer.map + 100, &frame.prepare_complete_ns, 8);
  std::memcpy(writer.map + 108, &tracker.ready_ns, 8);
  std::memcpy(writer.map + 116, &tracker.tracker_frame_num, 4);
  const guint32 metadata_size = static_cast<guint32>(tracker.snapshot_json.size());
  std::memcpy(writer.map + 120, &metadata_size, 4);
  const guint64 publish_done_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  std::memcpy(writer.map + 20, &publish_done_ns, 8);
  flock(writer.fd, LOCK_UN);
  f5r_preview_notify(frame.source_id, frame.fullscreen);
  if (!frame.fullscreen) {
    ++f5r_preview_frames[frame.source_id];
    ++source_preview_published[frame.source_id];
  }
  ++f5r_preview_joined[frame.source_id];
  if (f5r_join_evidence) {
    char row[768];
    const int length = std::snprintf(row, sizeof(row),
        "{\"camera_id\":\"CAM-%02u\",\"frame_num\":%u,\"tracker_frame_num\":%u,"
        "\"pts_ns\":%lu,\"preview_width\":%u,\"preview_height\":%u,\"preview_source\":\"%s\","
        "\"t0_decoder_reference_ns\":%lu,\"t1_decoder_out_ns\":%lu,"
        "\"t1_preview_prepare_start_ns\":%lu,\"t2_preview_prepare_complete_ns\":%lu,"
        "\"t3_tracker_metadata_ready_ns\":%lu,\"t4_shared_publish_ns\":%lu,"
        "\"image_bbox_pts_match\":true,\"image_bbox_frame_match\":true}\n",
        frame.source_id + 1, frame.frame_num, tracker.tracker_frame_num, frame.pts,
        frame.width, frame.height, frame.fullscreen ? "fullscreen" : "tile",
        frame.decoder_input_ns, frame.decoder_output_ns, frame.prepare_start_ns,
        frame.prepare_complete_ns, tracker.ready_ns, publish_done_ns);
    if (length > 0 && static_cast<size_t>(length) < sizeof(row))
      tracker_write_line(f5r_join_evidence, row, static_cast<size_t>(length));
  }
  return true;
}

static void f5r_publish_worker() {
  for (;;) {
    F5RJoinedPreview job;
    {
      std::unique_lock<std::mutex> guard(f5r_publish_mutex);
      f5r_publish_cv.wait(guard, [] { return f5r_publish_stopping || !f5r_publish_jobs.empty(); });
      if (f5r_publish_jobs.empty() && f5r_publish_stopping) return;
      job = std::move(f5r_publish_jobs.front());
      f5r_publish_jobs.pop_front();
    }
    f5r_write_joined_preview(job);
  }
}

static void f5r_enqueue_preview(F5RJoinedPreview &&job) {
  std::lock_guard<std::mutex> guard(f5r_publish_mutex);
  constexpr size_t maximum_pending_jobs = 12;
  if (f5r_publish_jobs.size() >= maximum_pending_jobs) {
    f5r_publish_jobs.pop_front();
    ++f5r_preview_queue_drops;
  }
  f5r_publish_jobs.push_back(std::move(job));
  f5r_publish_cv.notify_one();
}

static void f5r_collect_ready_locked(guint source_id, F5RJoinEntry &entry,
                                     std::vector<F5RJoinedPreview> &ready) {
  if (!entry.tracker) return;
  if (entry.tile && !entry.tile_published) {
    ready.push_back({std::move(*entry.tile), *entry.tracker});
    entry.tile.reset();
    entry.tile_published = true;
  }
  if (entry.fullscreen_expected && entry.fullscreen && !entry.fullscreen_published) {
    ready.push_back({std::move(*entry.fullscreen), *entry.tracker});
    entry.fullscreen.reset();
    entry.fullscreen_published = true;
  }
  (void)source_id;
}

static void f5r_prune_join_entries_locked(guint source_id) {
  auto &entries = f5r_join_entries[source_id];
  guint64 high_water = f5r_preview_join_cache_high_water[source_id].load();
  while (entries.size() > high_water && !f5r_preview_join_cache_high_water[source_id].compare_exchange_weak(
      high_water, entries.size())) {}
  while (entries.size() > F5R_JOIN_ENTRY_LIMIT) {
    auto it = entries.begin();
    if (!it->second.tile_published) ++f5r_preview_join_misses[source_id];
    if (it->second.fullscreen_expected && !it->second.fullscreen_published)
      ++f5r_preview_join_misses[source_id];
    ++f5r_preview_stale_drops[source_id];
    entries.erase(it);
  }
  auto &starts = f5r_prepare_starts[source_id];
  while (starts.size() > 12) starts.erase(starts.begin());
}

static void f5r_add_prepared_frame(F5RPreparedFrame &&frame) {
  const guint source_id = frame.source_id;
  const F5RJoinKey key{frame.frame_num, frame.pts};
  std::vector<F5RJoinedPreview> ready;
  {
    std::lock_guard<std::mutex> guard(f5r_join_mutex);
    auto &entry = f5r_join_entries[source_id][key];
    auto &start = frame.fullscreen ? entry.fullscreen_prepare_start_ns : entry.tile_prepare_start_ns;
    auto found_start = f5r_prepare_starts[source_id].find(key);
    if (found_start != f5r_prepare_starts[source_id].end()) {
      start = found_start->second;
      f5r_prepare_starts[source_id].erase(found_start);
    }
    frame.prepare_start_ns = start;
    if (frame.fullscreen) {
      entry.fullscreen = std::move(frame);
      ++f5r_fullscreen_prepare_frames[source_id];
    } else {
      entry.tile = std::move(frame);
      ++f5r_preview_prepare_frames[source_id];
    }
    f5r_collect_ready_locked(source_id, entry, ready);
    if (entry.tile_published && entry.fullscreen_published)
      f5r_join_entries[source_id].erase(key);
    f5r_prune_join_entries_locked(source_id);
  }
  for (auto &job : ready) f5r_enqueue_preview(std::move(job));
}

static void f5r_tracker_frame_ready(guint source_id, guint tracker_frame_num,
    guint source_frame_num, guint64 pts_ns, guint object_count,
    guint64 tracker_ready_ns, const char *snapshot_json, size_t snapshot_size) {
  if (source_id >= 6 || source_frame_num == G_MAXUINT32 || pts_ns == GST_CLOCK_TIME_NONE ||
      !snapshot_json || snapshot_size == 0) {
    if (source_id < 6) ++f5r_preview_reference_misses[source_id];
    return;
  }
  F5RTrackerReady tracker;
  tracker.tracker_frame_num = tracker_frame_num;
  tracker.source_frame_num = source_frame_num;
  tracker.object_count = object_count;
  tracker.pts = pts_ns;
  tracker.ready_ns = tracker_ready_ns;
  tracker.snapshot_json.assign(snapshot_json, snapshot_size);
  const F5RJoinKey key{source_frame_num, pts_ns};
  std::vector<F5RJoinedPreview> ready;
  {
    std::lock_guard<std::mutex> guard(f5r_join_mutex);
    auto &entry = f5r_join_entries[source_id][key];
    entry.tracker = std::move(tracker);
    entry.fullscreen_expected = f5r_fullscreen_selected.load() == static_cast<gint>(source_id);
    if (!entry.fullscreen_expected) entry.fullscreen_published = true;
    f5r_collect_ready_locked(source_id, entry, ready);
    if (entry.tile_published && entry.fullscreen_published)
      f5r_join_entries[source_id].erase(key);
    f5r_prune_join_entries_locked(source_id);
  }
  for (auto &job : ready) f5r_enqueue_preview(std::move(job));
}

static GstPadProbeReturn f5r_preview_prepare_start_probe(
    GstPad*, GstPadProbeInfo *info, gpointer opaque) {
  auto *branch = static_cast<F5RPreviewBranch*>(opaque);
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer) return GST_PAD_PROBE_OK;
  const guint64 pts = GST_BUFFER_PTS(buffer);
  const guint32 frame = f5r_source_frame_for_pts(branch->source_id, pts);
  if (frame == G_MAXUINT32 || pts == GST_CLOCK_TIME_NONE) return GST_PAD_PROBE_OK;
  const F5RJoinKey key{frame, pts};
  std::lock_guard<std::mutex> guard(f5r_join_mutex);
  f5r_prepare_starts[branch->source_id][key] =
      static_cast<guint64>(g_get_monotonic_time()) * 1000;
  return GST_PAD_PROBE_OK;
}

static GstFlowReturn f5r_preview_sample(GstAppSink *sink, gpointer opaque) {
  auto *branch = static_cast<F5RPreviewBranch*>(opaque);
  if (branch->fullscreen) ++f5r_fullscreen_samples[branch->source_id];
  else ++f5r_tile_samples[branch->source_id];
  GstSample *sample = gst_app_sink_pull_sample(sink);
  if (!sample) return GST_FLOW_EOS;
  GstCaps *caps = gst_sample_get_caps(sample);
  GstVideoInfo info;
  GstBuffer *buffer = gst_sample_get_buffer(sample);
  GstVideoFrame video_frame;
  if (!caps || !buffer || !gst_video_info_from_caps(&info, caps) ||
      GST_VIDEO_INFO_FORMAT(&info) != GST_VIDEO_FORMAT_BGRx ||
      GST_VIDEO_INFO_WIDTH(&info) != branch->width || GST_VIDEO_INFO_HEIGHT(&info) != branch->height ||
      GST_VIDEO_INFO_PLANE_STRIDE(&info, 0) < static_cast<gint>(branch->width * 4)) {
    const guint64 prior = f5r_preview_errors.fetch_add(1);
    if (prior < 8) {
      gchar *caps_text = caps ? gst_caps_to_string(caps) : g_strdup("<no-caps>");
      g_printerr("PREVIEW_PREPARE_MAP_FAIL source=%u fullscreen=%d caps=%s\n",
          branch->source_id, int(branch->fullscreen), caps_text);
      g_free(caps_text);
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
  }
  const guint64 pts = GST_BUFFER_PTS(buffer);
  const guint32 source_frame = f5r_source_frame_for_pts(branch->source_id, pts);
  const DecoderFrameStamp stamp = f5r_decode_stamp_for_pts(branch->source_id, pts);
  if (source_frame == G_MAXUINT32 || !stamp.output_ns) {
    ++f5r_preview_reference_misses[branch->source_id];
    gst_sample_unref(sample);
    return GST_FLOW_OK;
  }
  F5RPreparedFrame prepared;
  prepared.source_id = branch->source_id;
  prepared.frame_num = source_frame;
  prepared.width = branch->width;
  prepared.height = branch->height;
  prepared.stride = branch->width * 4;
  prepared.pts = pts;
  prepared.dts = GST_BUFFER_DTS(buffer);
  prepared.decoder_input_ns = stamp.input_ns;
  prepared.decoder_output_ns = stamp.output_ns;
  prepared.fullscreen = branch->fullscreen;
  prepared.buffer = gst_buffer_ref(buffer);
  prepared.prepare_complete_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  gst_sample_unref(sample);
  f5r_add_prepared_frame(std::move(prepared));
  return GST_FLOW_OK;
}

static gboolean f5r_fullscreen_request_tick(gpointer) {
  gint selected = -1;
  FILE *request = std::fopen(f5r_fullscreen_request_path.c_str(), "r");
  if (request) {
    char text[32] = {};
    if (std::fgets(text, sizeof(text), request)) {
      guint camera = 0;
      if (std::sscanf(text, "CAM-%02u", &camera) == 1 && camera >= 1 && camera <= 6)
        selected = static_cast<gint>(camera - 1);
    }
    std::fclose(request);
  }
  if (selected == f5r_fullscreen_selected.load()) return G_SOURCE_CONTINUE;
  for (guint i = 0; i < 6; ++i)
    if (f5r_fullscreen_valves[i])
      g_object_set(f5r_fullscreen_valves[i], "drop", static_cast<gint>(i) != selected, nullptr);
  f5r_fullscreen_selected = selected;
  event(nullptr, "FULLSCREEN_PREVIEW_SELECT",
      selected < 0 ? "none" : "CAM-" + std::string(selected + 1 < 10 ? "0" : "") +
          std::to_string(selected + 1) + " 1920x1080");
  return G_SOURCE_CONTINUE;
}

static bool f5r_preview_open() {
  f5r_preview_notify_open();
  for (guint i = 0; i < 6; ++i) {
    char path[128];
    std::snprintf(path, sizeof(path), "/dev/shm/v11_ui_preview_cam%02u_v1.bin", i + 1);
    if (!f5r_writer_open(f5r_preview[i], path, F5R_PREVIEW_W, F5R_PREVIEW_H)) return false;
  }
  if (!f5r_writer_open(f5r_fullscreen_preview,
        "/dev/shm/v11_ui_preview_fullscreen_v1.bin", F5R_FULLSCREEN_W, F5R_FULLSCREEN_H))
    return false;
  if (verbose_evidence_enabled()) {
    f5r_join_evidence = std::fopen("/work/preview-join.jsonl", "w");
    if (!f5r_join_evidence) return false;
    std::setvbuf(f5r_join_evidence, nullptr, _IOFBF, 1024 * 1024);
  }
  f5r_publish_stopping = false;
  f5r_publish_thread = std::thread(f5r_publish_worker);
  return true;
}

static void f5r_preview_close() {
  {
    std::lock_guard<std::mutex> guard(f5r_publish_mutex);
    f5r_publish_stopping = true;
  }
  f5r_publish_cv.notify_all();
  if (f5r_publish_thread.joinable()) f5r_publish_thread.join();
  if (f5r_preview_notify_fd >= 0) {
    close(f5r_preview_notify_fd);
    f5r_preview_notify_fd = -1;
  }
  for (guint i = 0; i < 6; ++i) f5r_writer_close(f5r_preview[i]);
  f5r_writer_close(f5r_fullscreen_preview);
  if (f5r_join_evidence) { std::fclose(f5r_join_evidence); f5r_join_evidence = nullptr; }
  std::lock_guard<std::mutex> guard(f5r_join_mutex);
  for (auto &entries : f5r_join_entries) entries.clear();
  for (auto &starts : f5r_prepare_starts) starts.clear();
}

static bool f5r_tee_link(GstElement *tee, GstElement *downstream) {
  GstPad *src = gst_element_request_pad_simple(tee, "src_%u");
  GstPad *sink = gst_element_get_static_pad(downstream, "sink");
  const bool ok = src && sink && gst_pad_link(src, sink) == GST_PAD_LINK_OK;
  if (src) gst_object_unref(src);
  if (sink) gst_object_unref(sink);
  return ok;
}

static bool write_all_json_line(FILE *file, const char *data, size_t len) {
  if (!file || !data || !len) return false;
  flockfile(file);
  clearerr(file);
  const size_t written = std::fwrite(data, 1, len, file);
  const bool ok = written == len && std::ferror(file) == 0;
  funlockfile(file);
  return ok;
}

static bool write_detection_json(
    FILE *file, guint source_id, gint frame_num, guint64 pts_ns, gint class_id,
    guint component_id, float confidence, float left, float top, float width, float height) {
  char line[544];
  const int n = std::snprintf(
      line, sizeof(line),
      "{\"source_id\":%u,\"frame\":%d,\"pts_ns\":%lu,"
      "\"class_id\":%d,\"component_id\":%u,\"confidence\":%.6f,"
      "\"box\":[%.3f,%.3f,%.3f,%.3f]}\n",
      source_id, frame_num, pts_ns, class_id, component_id, confidence,
      left, top, width, height);
  if (n <= 0 || static_cast<size_t>(n) >= sizeof(line)) return false;
  return write_all_json_line(file, line, static_cast<size_t>(n));
}

static void suppress_same_component_person_duplicates(NvDsFrameMeta *frame) {
  if (!frame) return;
  std::vector<NvDsObjectMeta*> objects;
  std::vector<PersonProposalBox> boxes;
  for (NvDsMetaList *node = frame->obj_meta_list; node; node = node->next) {
    auto *object = static_cast<NvDsObjectMeta*>(node->data);
    if (!object || object->class_id != 0) continue;
#if defined(YOLO26_AUXILIARY_896) || defined(YOLO26_AUXILIARY_DETECTOR)
    if (object->unique_component_id != 1 &&
        object->unique_component_id != TRACKER_AUXILIARY_GIE_ID) continue;
#else
    if (object->unique_component_id != 1) continue;
#endif
    const auto &rect = object->rect_params;
    if (!std::isfinite(object->confidence) || object->confidence < 0 ||
        object->confidence > 1 || !std::isfinite(rect.left) ||
        !std::isfinite(rect.top) || !std::isfinite(rect.width) ||
        !std::isfinite(rect.height) || rect.width <= 0 || rect.height <= 0)
      continue;  // Preserve malformed metadata for the normal fail-closed validator.
    objects.push_back(object);
    boxes.push_back({static_cast<unsigned>(object->unique_component_id), object->confidence,
        rect.left, rect.top, rect.width, rect.height});
  }
  const auto suppressions = person_proposal_duplicates(boxes);
  for (const auto &decision : suppressions) {
    if (decision.suppressed_index >= objects.size() ||
        decision.retained_index >= objects.size()) continue;
    const auto *suppressed = objects[decision.suppressed_index];
    const auto *retained = objects[decision.retained_index];
    if (proposal_duplicate_evidence) {
      const auto &s = suppressed->rect_params;
      const auto &k = retained->rect_params;
      std::fprintf(proposal_duplicate_evidence,
          "{\"camera_id\":\"CAM-%02u\",\"frame_num\":%d,\"pts_ns\":%lu,"
          "\"component_id\":%u,\"decision\":\"SUPPRESS_LOWER_CONFIDENCE_NESTED_DUPLICATE\","
          "\"suppressed_confidence\":%.6f,\"suppressed_bbox_xywh\":[%.3f,%.3f,%.3f,%.3f],"
          "\"retained_confidence\":%.6f,\"retained_bbox_xywh\":[%.3f,%.3f,%.3f,%.3f],"
          "\"iou\":%.6f,\"containment\":%.6f,\"center_over_max_height\":%.6f,"
          "\"width_ratio\":%.6f,\"height_ratio\":%.6f}\n",
          frame->source_id + 1, frame->frame_num, frame->buf_pts,
          suppressed->unique_component_id, suppressed->confidence,
          s.left, s.top, s.width, s.height, retained->confidence,
          k.left, k.top, k.width, k.height, decision.metrics.iou,
          decision.metrics.containment, decision.metrics.center_over_max_height,
          decision.metrics.width_ratio, decision.metrics.height_ratio);
    }
    nvds_remove_obj_meta_from_frame(frame, const_cast<NvDsObjectMeta*>(suppressed));
    ++same_component_duplicate_suppressed;
  }
}

static bool auxiliary_duplicate_of_primary(
    const NvDsObjectMeta *auxiliary, const std::vector<std::array<float, 5>> &primary) {
  if (!auxiliary) return false;
  const auto &a = auxiliary->rect_params;
  size_t strong_matches = 0;
  for (const auto &candidate : primary) {
    const float iou = tracker_iou(a.left, a.top, a.width, a.height,
        candidate[0], candidate[1], candidate[2], candidate[3]);
    const float center = tracker_center_distance(a.left, a.top, a.width, a.height,
        candidate[0], candidate[1], candidate[2], candidate[3]);
    const float width_ratio = std::max(a.width, candidate[2]) /
        std::max(1.0f, std::min(a.width, candidate[2]));
    const float height_ratio = std::max(a.height, candidate[3]) /
        std::max(1.0f, std::min(a.height, candidate[3]));
    const float center_limit = .25f * std::max(a.height, candidate[3]);
    if (iou >= .30f && center <= center_limit && width_ratio <= 2.0f && height_ratio <= 2.0f)
      ++strong_matches;
  }
  // One or more strong primary overlaps means retain the 640 proposal and
  // withhold the auxiliary candidate. This is conservative under ambiguity.
  return strong_matches != 0;
}

// The auxiliary branch receives a tee reference to the decoded buffer. Its
// out-of-place NVMM conversion separates video surfaces, but the DeepStream
// metadata transform may still carry objects produced by the primary PGIE.
// Remove inherited objects on the auxiliary branch only, before its nvinfer,
// so nvdsmetamux cannot merge primary detections a second time. Frame/source/
// PTS metadata remains intact for exact-frame matching.
static GstPadProbeReturn auxiliary_strip_inherited_objects(
    GstPad*, GstPadProbeInfo *info, gpointer) {
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer) return GST_PAD_PROBE_OK;
  NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(buffer);
  if (!batch) {
    fail("auxiliary inference branch lost NvDsBatchMeta before metadata isolation");
    return GST_PAD_PROBE_DROP;
  }
  for (NvDsMetaList *frame_node = batch->frame_meta_list; frame_node;
       frame_node = frame_node->next) {
    auto *frame = static_cast<NvDsFrameMeta*>(frame_node->data);
    if (!frame) continue;
    NvDsMetaList *object_node = frame->obj_meta_list;
    while (object_node) {
      NvDsMetaList *next = object_node->next;
      auto *object = static_cast<NvDsObjectMeta*>(object_node->data);
      if (object) nvds_remove_obj_meta_from_frame(frame, object);
      object_node = next;
    }
  }
  return GST_PAD_PROBE_OK;
}

static bool write_frame_json(
    FILE *file, guint source_id, gint frame_num, guint64 pts_ns,
    guint64 batch_pts_ns, guint persons, guint batch_id,
    guint source_width, guint source_height, guint pipeline_width, guint pipeline_height) {
  char line[384];
  const int n = std::snprintf(
      line, sizeof(line),
      "{\"source_id\":%u,\"frame\":%d,\"pts_ns\":%lu,"
      "\"batch_pts_ns\":%lu,\"persons\":%u,\"batch_id\":%u,"
      "\"source_width\":%u,\"source_height\":%u,"
      "\"pipeline_width\":%u,\"pipeline_height\":%u}\n",
      source_id, frame_num, pts_ns, batch_pts_ns, persons, batch_id,
      source_width, source_height, pipeline_width, pipeline_height);
  if (n <= 0 || static_cast<size_t>(n) >= sizeof(line)) return false;
  return write_all_json_line(file, line, static_cast<size_t>(n));
}

static GstPadProbeReturn infer_enter(GstPad*, GstPadProbeInfo *info, gpointer) {
  GstBuffer *b = GST_PAD_PROBE_INFO_BUFFER(info);
  if (b) {
    pipeline_stage_record_batch(b, 2,
        static_cast<guint64>(g_get_monotonic_time()) * 1000);
#ifdef PGIE_ISOLATION
#ifdef PGIE_ISOLATION_CAPTURE_PIXELS
    dump_mux_pixels(b, gst_buffer_get_nvds_batch_meta(b));
#endif
#endif
    auto *stamp = g_new(gint64, 1); *stamp = g_get_monotonic_time();
    gst_mini_object_set_qdata(GST_MINI_OBJECT(b), infer_stamp, stamp, g_free);
  }
  return GST_PAD_PROBE_OK;
}
static GstPadProbeReturn infer_exit(GstPad*, GstPadProbeInfo *info, gpointer) {
  GstBuffer *b = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!b) return GST_PAD_PROBE_OK;
  auto *meta = gst_buffer_get_nvds_batch_meta(b);
  if (!meta) { ++infer_errors; fail("YOLO missing batch metadata"); return GST_PAD_PROBE_DROP; }
  if (auto *stamp = static_cast<gint64*>(gst_mini_object_get_qdata(GST_MINI_OBJECT(b), infer_stamp))) {
    const guint64 dt = g_get_monotonic_time() - *stamp;
    infer_us += dt; ++infer_timed;
    if (dt > infer_max_us) infer_max_us = dt;
    infer_latency_histogram.add(dt * 1000);
  }
  ++infer_batches;
#ifdef PGIE_ISOLATION
  // The offline context-crop test deliberately runs the verifier's .20 floor.
  // Keep the normal direct-file audit floor at .23 unless this diagnostic-only
  // override is explicitly supplied; live/production builds cannot see it.
  float isolation_min_confidence = .23f;
  if (const char *value = g_getenv("PGIE_ISOLATION_MIN_CONFIDENCE")) {
    char *end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (!end || *end || !std::isfinite(parsed) || parsed < 0.0f || parsed > .25f) {
      ++infer_errors; fail("invalid PGIE_ISOLATION_MIN_CONFIDENCE");
      return GST_PAD_PROBE_DROP;
    }
    isolation_min_confidence = parsed;
  }
#else
  constexpr float isolation_min_confidence = .23f;
#endif
  for (auto *node = meta->frame_meta_list; node; node = node->next) {
    auto *frame = static_cast<NvDsFrameMeta*>(node->data);
    if (frame->source_id >= 6 || !frame->bInferDone) {
      ++infer_errors; fail("YOLO unknown source or frame not inferred"); return GST_PAD_PROBE_DROP;
    }
    const guint pgie_frame_width = tracker_bbox_coordinate_extent(frame, true);
    const guint pgie_frame_height = tracker_bbox_coordinate_extent(frame, false);
    auto &c = detections[frame->source_id];
    ++c.frames; ++c.batches; ++infer_frames;
    ++source_pgie_frames[frame->source_id];
    guint count = 0;
    std::vector<TrackerDetection> tracker_detections;
    std::vector<LowConfCandidate> lowconf_sidecar;
    std::vector<std::array<float, 5>> primary_boxes;
    const LowConfFrameKey lowconf_key{frame->source_id, frame->frame_num, frame->buf_pts};
    suppress_same_component_person_duplicates(frame);
    for (NvDsMetaList *scan = frame->obj_meta_list; scan; scan = scan->next) {
      auto *candidate = static_cast<NvDsObjectMeta*>(scan->data);
      if (!candidate || candidate->class_id != 0 || candidate->unique_component_id != 1) continue;
      const auto &box = candidate->rect_params;
      if (std::isfinite(box.left) && std::isfinite(box.top) &&
          std::isfinite(box.width) && std::isfinite(box.height) &&
          box.width > 0 && box.height > 0)
        primary_boxes.push_back({box.left, box.top, box.width, box.height, candidate->confidence});
    }
    auto *obj_node = frame->obj_meta_list;
    while (obj_node) {
      auto *next_obj_node = obj_node->next;
      auto *obj = static_cast<NvDsObjectMeta*>(obj_node->data);
      if (!obj) { obj_node = next_obj_node; continue; }
      const bool auxiliary = obj->unique_component_id == TRACKER_AUXILIARY_GIE_ID;
      if (auxiliary) ++auxiliary_detector_objects;
#if defined(YOLO26_AUXILIARY_896) || defined(YOLO26_AUXILIARY_DETECTOR)
      const bool component_allowed = obj->unique_component_id == 1 || auxiliary;
#else
      const bool component_allowed = obj->unique_component_id == 1;
#endif
      auto &r = obj->rect_params;
      const float frame_width = static_cast<float>(pgie_frame_width);
      const float frame_height = static_cast<float>(pgie_frame_height);
#ifdef PGIE_ISOLATION
      // Context-verifier diagnostics use the generic COCO parser so their raw
      // output can include non-person classes. These are not person proposals
      // and must not make the person-only audit abort. Preserve a raw record so
      // the diagnostic still accounts for what the model emitted.
      if (obj->class_id != 0) {
        g_printerr("PGIE_IGNORED_NONPERSON source=%u frame=%u pts=%" G_GUINT64_FORMAT
            " class=%u component=%u confidence=%.9g rect=[%.9g,%.9g,%.9g,%.9g]\n",
            frame->source_id, frame->frame_num, frame->buf_pts, obj->class_id,
            obj->unique_component_id, obj->confidence, r.left, r.top, r.width, r.height);
        obj_node = next_obj_node;
        continue;
      }
#endif
      const bool invalid_object = obj->class_id != 0 || !component_allowed ||
          !std::isfinite(obj->confidence) || obj->confidence < isolation_min_confidence || obj->confidence > 1 ||
          !std::isfinite(r.left) || !std::isfinite(r.top) || !std::isfinite(r.width) || !std::isfinite(r.height) ||
          r.width <= 0 || r.height <= 0 || r.left < 0 || r.top < 0 ||
          r.left+r.width > frame_width + .1f || r.top+r.height > frame_height + .1f;
      if (invalid_object) {
#ifdef PGIE_ISOLATION
        g_printerr("PGIE_INVALID_OBJECT source=%u frame=%u pts=%" G_GUINT64_FORMAT
            " class=%u component=%u confidence=%.9g rect=[%.9g,%.9g,%.9g,%.9g]"
            " frame_size=%.9g,%.9g audit_floor=%.9g\n",
            frame->source_id, frame->frame_num, frame->buf_pts, obj->class_id,
            obj->unique_component_id, obj->confidence, r.left, r.top, r.width,
            r.height, frame_width, frame_height, isolation_min_confidence);
#endif
        ++c.errors; ++infer_errors; fail("YOLO invalid/non-person/under-floor object metadata"); return GST_PAD_PROBE_DROP;
      }
      if (auxiliary && auxiliary_duplicate_of_primary(obj, primary_boxes)) {
        ++auxiliary_duplicate_suppressed;
        nvds_remove_obj_meta_from_frame(frame, obj);
        obj_node = next_obj_node;
        continue;
      }
      if (obj->confidence < .25f) {
        if (lowconf_verifier_enabled()) {
          LowConfCandidate candidate;
          candidate.confidence = obj->confidence;
          candidate.left = r.left; candidate.top = r.top;
          candidate.width = r.width; candidate.height = r.height;
          candidate.source_frame_num = frame->frame_num;
          candidate.source_pts_ns = frame->buf_pts;
          lowconf_sidecar.push_back(candidate);
          // Start appearance verification while the exact PGIE frame is still
          // upstream of NvDCF. The crop worker holds its own GstBuffer ref and
          // reads pixels only; this metadata is removed below and never enters
          // NvDCF. Later publication still requires the same-frame result plus
          // the existing current-shadow/geometry/uniqueness gates. This overlaps
          // verifier work with tracking instead of adding it after tracking.
          (void)lowconf_verifier_submit(b, frame->batch_id, lowconf_key,
              pgie_frame_width, pgie_frame_height, candidate, false, false);
          if (lowconf_candidate_evidence) {
            std::fprintf(lowconf_candidate_evidence,
                "{\"camera_id\":\"CAM-%02u\",\"frame_num\":%d,\"pts_ns\":%lu,"
                "\"bbox\":[%.3f,%.3f,%.3f,%.3f],\"confidence\":%.6f,"
                "\"source_frame_width\":%u,\"source_frame_height\":%u,"
                "\"sidecar_stored\":true,\"verifier_submitted\":true,"
                "\"nvdcf_input\":false}\n",
                frame->source_id + 1, frame->frame_num, frame->buf_pts,
                r.left, r.top, r.width, r.height, obj->confidence,
                static_cast<guint>(frame_width), static_cast<guint>(frame_height));
          }
          // This proposal is intentionally absent from NvDCF's input metadata.
          // A verified current-frame copy may only be emitted after tracking.
          nvds_remove_obj_meta_from_frame(frame, obj);
          obj_node = next_obj_node;
          continue;
        }
#ifndef PGIE_ISOLATION
        ++c.errors; ++infer_errors;
        fail("low-confidence proposal reached PGIE output while verifier mode is disabled");
        return GST_PAD_PROBE_DROP;
#endif
        // A PGIE_ISOLATION detector-only build may retain below-.25 proposals
        // in evidence. It has no NvDCF path and cannot publish them as boxes.
      }
      ++count;
      tracker_detections.push_back({obj->confidence, r.left, r.top, r.width, r.height,
          static_cast<guint>(std::max(0, obj->unique_component_id)), auxiliary});
      if (auxiliary) ++auxiliary_detector_admitted;
      obj->object_id = UNTRACKED_OBJECT_ID;
      g_strlcpy(obj->obj_label, "person", sizeof(obj->obj_label));
      r.border_width = 3;
      r.border_color = {0.1, 1.0, 0.2, 1.0};
      r.has_bg_color = 0;
      auto &text = obj->text_params;
      g_free(text.display_text);
      text.display_text = g_strdup_printf("person %.2f", obj->confidence);
      text.x_offset = guint(r.left); text.y_offset = guint(std::max(0.0f, r.top - 22));
      text.font_params.font_name = const_cast<char*>("Sans");
      text.font_params.font_size = 13;
      text.font_params.font_color = {1.0, 1.0, 1.0, 1.0};
      text.set_bg_clr = 1; text.text_bg_clr = {0.0, 0.0, 0.0, 0.65};
      // Save every inferred frame's object metadata; no pixels or CPU inference.
      if (detection_evidence && !write_detection_json(
              detection_evidence, frame->source_id, frame->frame_num, frame->buf_pts,
              obj->class_id, obj->unique_component_id, obj->confidence,
              r.left, r.top, r.width, r.height)) {
        ++c.errors; ++infer_errors; fail("YOLO detection evidence write failed");
        return GST_PAD_PROBE_DROP;
      }
      obj_node = next_obj_node;
    }
    if (frame_evidence && !write_frame_json(
            frame_evidence, frame->source_id, frame->frame_num, frame->buf_pts,
            GST_BUFFER_PTS(b), count, frame->batch_id,
            frame->source_frame_width, frame->source_frame_height,
            pgie_frame_width, pgie_frame_height)) {
      ++c.errors; ++infer_errors; fail("YOLO frame evidence write failed");
      return GST_PAD_PROBE_DROP;
    }
#ifdef YOLO26_TRACKER
    // Transfer this exact source/frame/PTS proposal list into the bounded
    // tracker-side ring without a per-frame vector copy.
    tracker_record_detections(frame->source_id, frame->frame_num, frame->buf_pts,
                              std::move(tracker_detections));
    tracker_record_lowconf_candidates(frame->source_id, frame->frame_num,
        frame->buf_pts, std::move(lowconf_sidecar));
#endif
    c.persons += count;
    if (count) c.last_unix_ms = g_get_real_time()/1000;
  }
  if (parser_errors() || parser_rejected()) {
    ++infer_errors; fail("YOLO parser reported malformed tensors or detections");
    return GST_PAD_PROBE_DROP;
  }
  return GST_PAD_PROBE_OK;
}

static void pipeline_stage_record_batch(GstBuffer *buffer, guint stage, guint64 now_ns) {
  if (!pipeline_stage_evidence || !buffer || stage < 1 || stage > 5) return;
  NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(buffer);
  if (!batch) return;
  guint actual_batch = 0;
  for (NvDsMetaList *node = batch->frame_meta_list; node; node = node->next)
    if (node->data) ++actual_batch;
  if (!actual_batch) return;
  std::lock_guard<std::mutex> guard(pipeline_stage_mutex);
  for (NvDsMetaList *node = batch->frame_meta_list; node; node = node->next) {
    auto *frame = static_cast<NvDsFrameMeta *>(node->data);
    if (!frame || frame->source_id >= 6 || frame->frame_num < 0 ||
        frame->buf_pts == GST_CLOCK_TIME_NONE) continue;
    const PipelineStageKey key{frame->source_id, static_cast<guint>(frame->frame_num),
        frame->buf_pts};
    auto &stamp = pipeline_stage_stamps[key];
    switch (stage) {
      case 1: {
        stamp.mux_output_ns = now_ns;
        stamp.mux_batch = actual_batch;
        if (frame->source_id < sources.size() && sources[frame->source_id]) {
          auto *source = sources[frame->source_id];
          std::lock_guard<std::mutex> source_guard(source->decoded_clock_mutex);
          const auto found = source->decoded_clock_ns.find(frame->buf_pts);
          if (found != source->decoded_clock_ns.end()) {
            stamp.decoder_input_ns = found->second.input_ns;
            stamp.decoder_output_ns = found->second.output_ns;
          }
        }
        break;
      }
      case 2: stamp.pgie_input_ns = now_ns; stamp.pgie_input_batch = actual_batch; break;
      case 3: stamp.pgie_output_ns = now_ns; stamp.pgie_output_batch = actual_batch; break;
      case 4: stamp.tracker_input_ns = now_ns; stamp.tracker_input_batch = actual_batch; break;
      case 5: {
        stamp.tracker_output_ns = now_ns;
        stamp.tracker_output_batch = actual_batch;
        std::fprintf(pipeline_stage_evidence,
            "{\"camera_id\":\"CAM-%02u\",\"source_frame_num\":%u,\"mux_frame_num\":%u,\"pts_ns\":%lu,"
            "\"decoder_in_ns\":%lu,\"decoder_out_ns\":%lu,\"mux_out_ns\":%lu,"
            "\"pgie_in_ns\":%lu,\"pgie_out_ns\":%lu,\"tracker_in_ns\":%lu,\"tracker_out_ns\":%lu,"
            "\"mux_batch\":%u,\"pgie_input_batch\":%u,\"pgie_output_batch\":%u,"
            "\"tracker_input_batch\":%u,\"tracker_output_batch\":%u}\n",
            frame->source_id + 1, f5r_source_frame_for_pts(frame->source_id, frame->buf_pts),
            frame->frame_num, frame->buf_pts, stamp.decoder_input_ns, stamp.decoder_output_ns,
            stamp.mux_output_ns, stamp.pgie_input_ns, stamp.pgie_output_ns,
            stamp.tracker_input_ns, stamp.tracker_output_ns, stamp.mux_batch,
            stamp.pgie_input_batch, stamp.pgie_output_batch, stamp.tracker_input_batch,
            stamp.tracker_output_batch);
        pipeline_stage_stamps.erase(key);
        break;
      }
    }
  }
  // Drop only orphaned diagnostic stamps if a frame did not reach the tracker.
  while (pipeline_stage_stamps.size() > 8192) pipeline_stage_stamps.erase(pipeline_stage_stamps.begin());
}

static GstPadProbeReturn pipeline_stage_probe(GstPad *, GstPadProbeInfo *info, gpointer opaque) {
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  const guint stage = GPOINTER_TO_UINT(opaque);
  pipeline_stage_record_batch(buffer, stage,
      static_cast<guint64>(g_get_monotonic_time()) * 1000);
  return GST_PAD_PROBE_OK;
}
static void detection_stats() {
  guint64 persons = 0;
  for (int i=0;i<6;++i) {
    auto &c=detections[i]; persons += c.persons.load();
    g_print("CAM-%02d DETECT frames=%lu detections=%lu persons=%lu batches=%lu errors=%lu last_detection_unix_ms=%lu\n",
      i+1, c.frames.load(), c.persons.load(), c.persons.load(), c.batches.load(),
      c.errors.load(), c.last_unix_ms.load());
  }
  g_print("YOLO STATS batch=6 precision=FP16 batches=%lu inferred_frames=%lu persons=%lu errors=%lu parser_errors=%llu parser_rejected=%llu parser_calls=%llu proposals=%llu element_latency_mean_ms=%.3f element_latency_p50_ms=%.3f element_latency_p95_ms=%.3f element_latency_p99_ms=%.3f element_latency_max_ms=%.3f\n",
    infer_batches.load(),infer_frames.load(),persons,infer_errors.load(),
    parser_errors(),parser_rejected(),parser_calls(),parser_proposals(),
    infer_timed ? infer_us.load()/1000.0/infer_timed.load() : 0.0,
    infer_latency_histogram.percentile_ns(.50) / 1e6,
    infer_latency_histogram.percentile_ns(.95) / 1e6,
    infer_latency_histogram.percentile_ns(.99) / 1e6,
    infer_max_us.load()/1000.0);
  const char *aux_config = g_getenv("V11_AUX_YOLO_CONFIG");
  if (!aux_config || !*aux_config) aux_config = g_getenv("V11_AUX_YOLO896_CONFIG");
  g_print("AUX_DETECTOR objects=%lu admitted=%lu duplicate_suppressed=%lu input_size=%s config=%s\n",
      auxiliary_detector_objects.load(), auxiliary_detector_admitted.load(),
      auxiliary_duplicate_suppressed.load(),
      g_getenv("V11_AUX_YOLO_INPUT_SIZE") ? g_getenv("V11_AUX_YOLO_INPUT_SIZE") : "config-defined",
      aux_config ? aux_config : "configured-by-runtime");
  const guint aux_input_level = auxiliary_queue_level(auxiliary_input_queue_observed);
  const guint aux_meta_level = auxiliary_queue_level(auxiliary_meta_queue_observed);
  auxiliary_queue_update_high_water(auxiliary_input_queue_high_water, aux_input_level);
  auxiliary_queue_update_high_water(auxiliary_meta_queue_high_water, aux_meta_level);
  g_print("AUX_QUEUE input_level=%u input_hwm=%u input_overruns=%lu meta_level=%u meta_hwm=%u meta_overruns=%lu policy=latest-only-1\n",
      aux_input_level, auxiliary_input_queue_high_water.load(),
      auxiliary_input_queue_overruns.load(), aux_meta_level,
      auxiliary_meta_queue_high_water.load(), auxiliary_meta_queue_overruns.load());
  g_print("AUX_JOIN mode=exact-key-sidecar input_batches=%lu infer_batches=%lu matched_batches=%lu timeouts=%lu timeout_ms=%lu mean_wait_ms=%.3f max_wait_ms=%.3f injection_errors=%lu\n",
      auxiliary_input_batches.load(), auxiliary_infer_batches.load(),
      auxiliary_join_batches.load(), auxiliary_join_timeouts.load(),
      auxiliary_join_timeout_ms_configured.load(),
      auxiliary_join_batches.load() ? auxiliary_join_wait_ns.load() / 1e6 /
          auxiliary_join_batches.load() : 0.0,
      auxiliary_join_max_wait_ns.load() / 1e6, auxiliary_injection_errors.load());
#ifdef YOLO26_AUXILIARY_SPARSE_PREPROCESS
  g_print("AUX_PREPROCESS_GATE period=%lu phase=%lu seen_batches=%lu admitted_batches=%lu skipped_batches=%lu skip_empty_join_frames=%lu skip_join_errors=%lu policy=exact-empty-result-on-skipped-batch\n",
      static_cast<unsigned long>(20), static_cast<unsigned long>(0),
      auxiliary_preprocess_cadence.seen(), auxiliary_preprocess_cadence.admitted(),
      auxiliary_sparse_skip_batches.load(), auxiliary_sparse_skip_join_frames.load(),
      auxiliary_sparse_skip_join_errors.load());
#endif
  for (guint source = 0; source < 6; ++source)
    g_print("CAM-%02u AUX_BRANCH queued_frames=%lu inferred_frames=%lu inferred_objects=%lu joined_frames=%lu joined_objects=%lu join_timeouts=%lu\n",
        source + 1, auxiliary_input_frames[source].load(),
        auxiliary_infer_frames[source].load(), auxiliary_infer_objects[source].load(),
        auxiliary_joined_frames[source].load(), auxiliary_joined_objects[source].load(),
        auxiliary_join_timeout_frames[source].load());
  g_print("PGIE_NESTED_DUPLICATES suppressed=%lu policy=same-component,unique-match,iou>=0.30,containment>=0.95,center/max_height<=0.25,size_ratios<=2,confidence_margin>=0.05\n",
      same_component_duplicate_suppressed.load());
  lowconf_verifier_stats();
  if (detection_evidence) std::fflush(detection_evidence);
  if (frame_evidence) std::fflush(frame_evidence);
  if (lowconf_candidate_evidence) std::fflush(lowconf_candidate_evidence);
  for (guint i = 0; i < 6; ++i)
    g_print("CAM-%02u PREVIEW tile_queue_in=%lu tile_queue_out=%lu tile_convert_out=%lu tile_sink_in=%lu tile_samples=%lu source_prepare=%lu fullscreen_queue_in=%lu fullscreen_queue_out=%lu fullscreen_convert_out=%lu fullscreen_sink_in=%lu fullscreen_samples=%lu fullscreen_prepare=%lu joined=%lu join_misses=%lu join_cache_hwm=%lu sync_mismatches=%lu stale_drops=%lu published=%lu fps_ema=%.3f errors=%lu decoder_reference_misses=%lu\n", i + 1,
        f5r_tile_queue_in[i].load(), f5r_tile_queue_out[i].load(),
        f5r_tile_convert_out[i].load(), f5r_tile_sink_in[i].load(),
        f5r_tile_samples[i].load(),
        f5r_preview_prepare_frames[i].load(),
        f5r_fullscreen_queue_in[i].load(), f5r_fullscreen_queue_out[i].load(),
        f5r_fullscreen_convert_out[i].load(), f5r_fullscreen_sink_in[i].load(),
        f5r_fullscreen_samples[i].load(), f5r_fullscreen_prepare_frames[i].load(),
        f5r_preview_joined[i].load(), f5r_preview_join_misses[i].load(),
        f5r_preview_join_cache_high_water[i].load(),
        f5r_preview_sync_mismatches[i].load(), f5r_preview_stale_drops[i].load(),
        f5r_preview_frames[i].load(), f5r_preview[i].fps, f5r_preview_errors.load(),
        f5r_preview_reference_misses[i].load());
  g_print("PREVIEW_PUBLISH_QUEUE dropped=%lu fullscreen_selected=%d\n",
      f5r_preview_queue_drops.load(), f5r_fullscreen_selected.load());
}
static GstElement *f5r_make_preview_branch(SourceCtx *source, bool fullscreen) {
  const guint source_id = source->source_id;
  const guint width = fullscreen ? F5R_FULLSCREEN_W : F5R_PREVIEW_W;
  const guint height = fullscreen ? F5R_FULLSCREEN_H : F5R_PREVIEW_H;
  const std::string suffix = fullscreen ? "fullscreen" : "tile";
  auto make_name = [&](const char *stem) {
    return source->id + "-preview-" + suffix + "-" + stem;
  };
  GstElement *queue = make("queue", make_name("queue").c_str());
  g_object_set(queue, "max-size-buffers", 1u, "max-size-bytes", 0u,
      "max-size-time", guint64(0), "leaky", 2, "silent", TRUE, nullptr);
  GstElement *valve = nullptr;
  if (fullscreen) {
    valve = make("valve", make_name("valve").c_str());
    g_object_set(valve, "drop", TRUE, nullptr);
    f5r_fullscreen_valves[source_id] = valve;
  }
  GstElement *convert = make("nvvideoconvert", make_name("convert").c_str());
  g_object_set(convert, "interpolation-method", 4, nullptr);
  GstElement *filter = make("capsfilter", make_name("caps").c_str());
  const std::string caps_text = "video/x-raw,format=BGRx,width=" +
      std::to_string(width) + ",height=" + std::to_string(height) +
      ",pixel-aspect-ratio=1/1";
  GstCaps *caps = gst_caps_from_string(caps_text.c_str());
  g_object_set(filter, "caps", caps, nullptr);
  gst_caps_unref(caps);
  GstElement *sink = make("appsink", make_name("appsink").c_str());
  g_object_set(sink, "emit-signals", FALSE, "max-buffers", 1u, "drop", TRUE,
      "sync", FALSE, "async", FALSE, "enable-last-sample", FALSE, nullptr);
  auto &branch = fullscreen ? f5r_fullscreen_branches[source_id] : f5r_preview_branches[source_id];
  branch = {source_id, width, height, fullscreen};
  GstAppSinkCallbacks callbacks{};
  callbacks.new_sample = f5r_preview_sample;
  gst_app_sink_set_callbacks(GST_APP_SINK(sink), &callbacks, &branch, nullptr);
  auto add_counter = [&](GstElement *element, const char *pad_name,
                         std::atomic<guint64> *counter) {
    GstPad *pad = gst_element_get_static_pad(element, pad_name);
    if (!pad) return;
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER,
        [](GstPad*, GstPadProbeInfo*, gpointer opaque) -> GstPadProbeReturn {
          ++*static_cast<std::atomic<guint64>*>(opaque);
          return GST_PAD_PROBE_OK;
        }, counter, nullptr);
    gst_object_unref(pad);
  };
  if (fullscreen) {
    add_counter(queue, "sink", &f5r_fullscreen_queue_in[source_id]);
    add_counter(queue, "src", &f5r_fullscreen_queue_out[source_id]);
    add_counter(convert, "src", &f5r_fullscreen_convert_out[source_id]);
    add_counter(sink, "sink", &f5r_fullscreen_sink_in[source_id]);
  } else {
    add_counter(queue, "sink", &f5r_tile_queue_in[source_id]);
    add_counter(queue, "src", &f5r_tile_queue_out[source_id]);
    add_counter(convert, "src", &f5r_tile_convert_out[source_id]);
    add_counter(sink, "sink", &f5r_tile_sink_in[source_id]);
  }
  GstPad *convert_sink = gst_element_get_static_pad(convert, "sink");
  if (convert_sink) {
    gst_pad_add_probe(convert_sink, GST_PAD_PROBE_TYPE_BUFFER,
        f5r_preview_prepare_start_probe, &branch, nullptr);
    gst_object_unref(convert_sink);
  }
  if (!f5r_tee_link(source->tee, queue)) {
    fail(source->id + " decoded preview tee link failed");
    return nullptr;
  }
  if (valve) {
    link(queue, valve); link(valve, convert);
  } else {
    link(queue, convert);
  }
  link(convert, filter); link(filter, sink);
  return sink;
}

static void detection_link(GstElement *from, GstElement *tiler, GstElement *to) {
  const char *stage_timing = g_getenv("V11_PIPELINE_STAGE_TIMING");
  if (stage_timing && std::strcmp(stage_timing, "1") == 0) {
    pipeline_stage_evidence = std::fopen("/work/pipeline-stage-timings.jsonl", "w");
    if (!pipeline_stage_evidence) {
      fail("cannot open optional pipeline stage timing evidence");
      std::exit(2);
    }
    std::setvbuf(pipeline_stage_evidence, nullptr, _IOFBF, 256 * 1024);
    GstPad *mux_src = gst_element_get_static_pad(from, "src");
    if (!mux_src || !gst_pad_add_probe(mux_src, GST_PAD_PROBE_TYPE_BUFFER,
            pipeline_stage_probe, GUINT_TO_POINTER(1), nullptr)) {
      if (mux_src) gst_object_unref(mux_src);
      fail("optional stage timing could not probe nvstreammux output");
      std::exit(2);
    }
    gst_object_unref(mux_src);
  }
  void *library = dlopen("/models/libyolo26rawperson.so", RTLD_NOW|RTLD_LOCAL);
  if (!library) { fail("YOLO parser library cannot load"); std::exit(2); }
  parser_errors=reinterpret_cast<ParserCount>(dlsym(library,"Yolo26ParserErrors"));
  parser_rejected=reinterpret_cast<ParserCount>(dlsym(library,"Yolo26ParserRejected"));
  parser_calls=reinterpret_cast<ParserCount>(dlsym(library,"Yolo26ParserCalls"));
  parser_proposals=reinterpret_cast<ParserCount>(dlsym(library,"Yolo26ParserProposals"));
  if(!parser_errors||!parser_rejected||!parser_calls||!parser_proposals){fail("YOLO parser ABI mismatch");std::exit(2);}
  if (verbose_evidence_enabled()) {
    detection_evidence=std::fopen("/work/detections.jsonl","w");
    frame_evidence=std::fopen("/work/frames.jsonl","w");
    proposal_duplicate_evidence=std::fopen("/work/proposal-duplicate-suppressions.jsonl","w");
    if(!detection_evidence || !frame_evidence || !proposal_duplicate_evidence){fail("cannot write YOLO metadata evidence");std::exit(2);}
    std::setvbuf(detection_evidence, nullptr, _IOFBF, 1024 * 1024);
    std::setvbuf(frame_evidence, nullptr, _IOFBF, 256 * 1024);
    std::setvbuf(proposal_duplicate_evidence, nullptr, _IOFBF, 256 * 1024);
  }
  if (lowconf_verifier_enabled() && verbose_evidence_enabled()) {
    lowconf_candidate_evidence = std::fopen("/work/lowconf-candidates.jsonl", "w");
    if (!lowconf_candidate_evidence) { fail("cannot write low-confidence candidate evidence"); std::exit(2); }
    std::setvbuf(lowconf_candidate_evidence, nullptr, _IOFBF, 256 * 1024);
#ifndef YOLO26_TRACKER
    fail("low-confidence verifier requires NvDCF canonical output"); std::exit(2);
#endif
  }
#ifdef PGIE_ISOLATION
  direct_file_frame_evidence = std::fopen("/work/source-frames.jsonl", "w");
  if (!direct_file_frame_evidence) { fail("cannot write direct-file source frame evidence"); std::exit(2); }
  std::setvbuf(direct_file_frame_evidence, nullptr, _IOLBF, 0);
#endif
  infer_stamp=g_quark_from_static_string("yolo26-infer-entry");
#ifndef PGIE_ISOLATION
  if (!f5r_preview_open()) { fail("cannot initialize isolated F5R shared-memory preview slots"); std::exit(2); }
#endif
  auto *gie=make("nvinfer","YOLO26m-person");
  const char *pgie_config = g_getenv("PGIE_ISOLATION_CONFIG");
  if (!pgie_config || !*pgie_config) pgie_config = "/config/config_infer_primary_yolo26m_raw_otm.txt";
  if (lowconf_verifier_enabled()) {
    const char *candidate_config = g_getenv("V11_LOWCONF_PRIMARY_CONFIG");
    pgie_config = candidate_config && *candidate_config
        ? candidate_config : "/config/config_infer_primary_yolo26m_raw_otm_lowconf023.txt";
  }
  if (context_verifier_enabled()) {
    lowconf_verifier_start(pipeline);
  }
  g_object_set(gie,"config-file-path",pgie_config,nullptr);
  if (pipeline_stage_evidence) {
    GstPad *pgie_src = gst_element_get_static_pad(gie, "src");
    if (!pgie_src || !gst_pad_add_probe(pgie_src, GST_PAD_PROBE_TYPE_BUFFER,
            pipeline_stage_probe, GUINT_TO_POINTER(3), nullptr)) {
      if (pgie_src) gst_object_unref(pgie_src);
      fail("optional stage timing could not probe PGIE output");
      std::exit(2);
    }
    gst_object_unref(pgie_src);
  }
  GstElement *tracker_input = gie;
#if defined(YOLO26_AUXILIARY_896) || defined(YOLO26_AUXILIARY_DETECTOR)
  const char *aux_enabled = g_getenv("V11_AUX_YOLO_ENABLE");
  if (!aux_enabled || !*aux_enabled) aux_enabled = g_getenv("V11_AUX_YOLO896_ENABLE");
  if (!aux_enabled || std::strcmp(aux_enabled, "1") != 0) {
    fail("supplemental YOLO auxiliary path was compiled but not explicitly enabled");
    std::exit(2);
  }
  const char *aux_config = g_getenv("V11_AUX_YOLO_CONFIG");
  if (!aux_config || !*aux_config) aux_config = g_getenv("V11_AUX_YOLO896_CONFIG");
  if (!aux_config || !*aux_config) aux_config = "/config/config_infer_aux_yolo26m_896.txt";
  GstElement *input_tee = make("tee", "yolo640-supplemental-input-tee");
  GstElement *primary_queue = make("queue", "yolo640-branch-queue");
  GstElement *auxiliary_queue = make("queue", "supplemental-yolo-branch-queue");
  // A tee only refs the same GstBuffer. DeepStream inference elements append
  // NvDsObjectMeta in-place, so feeding that shared NvDsBatchMeta to both
  // branches lets the auxiliary branch observe/re-merge primary detections.
  // The conversion provides an auxiliary-owned surface; an explicit probe
  // below strips any object metadata propagated by DeepStream's meta transform.
  // The primary PGIE branch remains byte-for-byte and element-for-element intact.
  GstElement *auxiliary_isolation_convert = make("nvvideoconvert", "supplemental-yolo-branch-meta-isolation");
  GstElement *auxiliary_isolation_caps = make("capsfilter", "supplemental-yolo-branch-isolation-caps");
  GstCaps *auxiliary_caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");
  if (!auxiliary_caps) { fail("cannot construct auxiliary branch isolation caps"); std::exit(2); }
  g_object_set(auxiliary_isolation_caps, "caps", auxiliary_caps, nullptr);
  gst_caps_unref(auxiliary_caps);
  GstElement *primary_meta_queue = make("queue", "yolo640-exact-join-queue");
  GstElement *auxiliary_meta_queue = make("queue", "supplemental-yolo-result-drain-queue");
  GstElement *auxiliary_sink = make("fakesink", "supplemental-yolo-result-drain");
  g_object_set(auxiliary_sink, "sync", FALSE, "async", FALSE, nullptr);
  for (auto *queue : {primary_queue, primary_meta_queue})
    g_object_set(queue, "max-size-buffers", 2u, "max-size-bytes", 0u,
        "max-size-time", guint64(0), "leaky", 0, "silent", TRUE, nullptr);
  // Supplemental inference is recall assistance only. Its output is copied
  // into a bounded exact (source, frame, PTS) sidecar before the leaky drain
  // queue. The primary branch waits at most the configured bound; a missing
  // or late result fails closed and cannot be paired to another frame.
  for (auto *queue : {auxiliary_queue, auxiliary_meta_queue})
    g_object_set(queue, "max-size-buffers", 1u, "max-size-bytes", 0u,
        "max-size-time", guint64(0), "leaky", 2, "silent", TRUE, nullptr);
#ifdef YOLO26_AUXILIARY_SPARSE_PREPROCESS
  auto *auxiliary_admission = gst_element_get_static_pad(auxiliary_queue, "sink");
  if (!auxiliary_admission || !gst_pad_add_probe(
          auxiliary_admission, GST_PAD_PROBE_TYPE_BUFFER,
          auxiliary_sparse_preprocess_gate, nullptr, nullptr)) {
    if (auxiliary_admission) gst_object_unref(auxiliary_admission);
    fail("sparse auxiliary preprocessing gate could not be installed");
    std::exit(2);
  }
  gst_object_unref(auxiliary_admission);
#endif
  auxiliary_input_queue_observed = auxiliary_queue;
  auxiliary_meta_queue_observed = auxiliary_meta_queue;
  g_signal_connect(auxiliary_queue, "overrun", G_CALLBACK(auxiliary_queue_overrun),
      &auxiliary_input_queue_overruns);
  g_signal_connect(auxiliary_meta_queue, "overrun", G_CALLBACK(auxiliary_queue_overrun),
      &auxiliary_meta_queue_overruns);
  GstElement *auxiliary_gie = make("nvinfer", "YOLO26m-supplemental");
  g_object_set(auxiliary_gie, "config-file-path", aux_config, nullptr);
  const guint64 auxiliary_join_wait_ms = auxiliary_exact_join_timeout_ms();
  auxiliary_join_timeout_ms_configured.store(auxiliary_join_wait_ms, std::memory_order_relaxed);

  auto *auxiliary_isolated = gst_element_get_static_pad(auxiliary_isolation_caps, "src");
  if (!auxiliary_isolated || !gst_pad_add_probe(auxiliary_isolated,
          GST_PAD_PROBE_TYPE_BUFFER, auxiliary_strip_inherited_objects, nullptr, nullptr)) {
    if (auxiliary_isolated) gst_object_unref(auxiliary_isolated);
    fail("auxiliary branch inherited-object metadata isolation probe could not be installed");
    std::exit(2);
  }
  gst_object_unref(auxiliary_isolated);

  if (!f5r_tee_link(input_tee, primary_queue) ||
      !f5r_tee_link(input_tee, auxiliary_queue)) {
    fail("sparse auxiliary detector input tee link failed");
    std::exit(2);
  }
  link(from, input_tee);
  link(primary_queue, gie);
  link(gie, primary_meta_queue);
  link(auxiliary_queue, auxiliary_isolation_convert);
  link(auxiliary_isolation_convert, auxiliary_isolation_caps);
  link(auxiliary_isolation_caps, auxiliary_gie);
  link(auxiliary_gie, auxiliary_meta_queue);
  link(auxiliary_meta_queue, auxiliary_sink);
  auto add_auxiliary_observer = [](GstElement *element, const char *pad_name,
                                  GstPadProbeCallback callback, const char *error) {
    GstPad *pad = gst_element_get_static_pad(element, pad_name);
    const gulong probe_id = pad ? gst_pad_add_probe(
        pad, GST_PAD_PROBE_TYPE_BUFFER, callback, nullptr, nullptr) : 0;
    if (pad) gst_object_unref(pad);
    if (!probe_id) {
      fail(error);
      std::exit(2);
    }
  };
  add_auxiliary_observer(auxiliary_queue, "src", auxiliary_input_observer,
      "auxiliary input observation probe unavailable");
  add_auxiliary_observer(auxiliary_gie, "src", auxiliary_infer_observer,
      "auxiliary inference observation probe unavailable");
  auto *in = gst_element_get_static_pad(gie, "sink");
  if (!in) { fail("YOLO-640 input pad unavailable"); std::exit(2); }
  gst_pad_add_probe(in, GST_PAD_PROBE_TYPE_BUFFER, infer_enter, nullptr, nullptr);
  gst_object_unref(in);
  auto *out = gst_element_get_static_pad(primary_meta_queue, "src");
  if (!out) { fail("YOLO primary exact-join queue output pad unavailable"); std::exit(2); }
  if (!gst_pad_add_probe(out, GST_PAD_PROBE_TYPE_BUFFER,
          auxiliary_exact_join_before_tracker, nullptr, nullptr)) {
    gst_object_unref(out);
    fail("auxiliary exact-key sidecar join probe unavailable");
    std::exit(2);
  }
  gst_object_unref(out);
  tracker_input = primary_meta_queue;
  event(nullptr, "AUXILIARY_DETECTOR_GRAPH",
      std::string("YOLO26m-640 batch=6 primary unchanged; parallel supplemental YOLO batch=6 using the staged auxiliary config; "
      "auxiliary branch strips inherited objects after its out-of-place NVMM conversion and before nvinfer; "
#ifdef YOLO26_AUXILIARY_SPARSE_PREPROCESS
      "branch-local preprocessing gate admits one of every 20 mux batches before nvvideoconvert and publishes exact empty sidecar results for skipped source/frame/PTS keys; auxiliary interval=0; "
#endif
      "supplemental input/output queues are bounded latest-only and non-blocking to primary; "
      "bounded exact (source,frame,PTS) sidecar join timeout_ms=" + std::to_string(auxiliary_join_wait_ms) +
      "; missing auxiliary frames fail closed; duplicate auxiliary proposals are withheld; "
      "only supplemental native targets require frozen-640 context verification before publication; ") +
      "primary_config=" + pgie_config + " auxiliary_config=" + aux_config +
      " auxiliary_input_size=" + (g_getenv("V11_AUX_YOLO_INPUT_SIZE") ? g_getenv("V11_AUX_YOLO_INPUT_SIZE") : "config-defined") +
      " join_mode=exact-key-sidecar");
#else
  auto *in=gst_element_get_static_pad(gie,"sink"), *out=gst_element_get_static_pad(gie,"src");
  gst_pad_add_probe(in,GST_PAD_PROBE_TYPE_BUFFER,infer_enter,nullptr,nullptr);
  gst_pad_add_probe(out,GST_PAD_PROBE_TYPE_BUFFER,infer_exit,nullptr,nullptr);
  gst_object_unref(in);gst_object_unref(out);
  // Keep mux collection decoupled from PGIE execution even in a primary-only
  // diagnostic build. The 640+auxiliary graph already uses this same bounded
  // queue on its primary branch; omitting it in the no-aux path changes mux
  // scheduling and makes detector-load A/B comparisons invalid.
  GstElement *primary_queue = make("queue", "primary-pgie-queue");
  g_object_set(primary_queue, "max-size-buffers", 2u, "max-size-bytes", 0u,
      "max-size-time", guint64(0), "leaky", 0, "silent", TRUE, nullptr);
  link(from, primary_queue);
  link(primary_queue, gie);
#endif
#ifdef YOLO26_TRACKER
  auto *tracker=create_nvdcf_tracker();
  if (pipeline_stage_evidence) {
    GstPad *tracker_sink = gst_element_get_static_pad(tracker, "sink");
    GstPad *tracker_src = gst_element_get_static_pad(tracker, "src");
    const gulong sink_probe = tracker_sink ? gst_pad_add_probe(tracker_sink,
        GST_PAD_PROBE_TYPE_BUFFER, pipeline_stage_probe, GUINT_TO_POINTER(4), nullptr) : 0;
    const gulong src_probe = tracker_src ? gst_pad_add_probe(tracker_src,
        GST_PAD_PROBE_TYPE_BUFFER, pipeline_stage_probe, GUINT_TO_POINTER(5), nullptr) : 0;
    if (tracker_sink) gst_object_unref(tracker_sink);
    if (tracker_src) gst_object_unref(tracker_src);
    if (!sink_probe || !src_probe) {
      fail("optional stage timing could not probe NvDCF input/output");
      std::exit(2);
    }
  }
  link(tracker_input,tracker);
  auto *tracked_tee=make("tee","post-tracker-tee");
  if(!f5r_tee_link(tracked_tee,tiler)){
    fail("post-tracker tee branch link failed");std::exit(2);
  }
  link(tracker,tracked_tee);
#ifdef PGIE_ISOLATION
  if (context_verifier_enabled()) {
    GstPad *tee_sink = gst_element_get_static_pad(tracked_tee, "sink");
    if (!tee_sink || !gst_pad_add_probe(tee_sink, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
            lowconf_verifier_primary_event, nullptr, nullptr)) {
      if (tee_sink) gst_object_unref(tee_sink);
      fail("could not install low-confidence verifier EOS drain hook");
      std::exit(2);
    }
    gst_object_unref(tee_sink);
  }
#endif
#ifndef PGIE_ISOLATION
  if (const char *capture_dir = g_getenv("V11_PREVIEW_CAPTURE_DIR"); capture_dir && *capture_dir) {
    GstPad *tee_sink = gst_element_get_static_pad(tracked_tee, "sink");
    if (tee_sink) {
      gst_pad_add_probe(tee_sink, GST_PAD_PROBE_TYPE_BUFFER,
          f5r_preview_input_capture_probe, nullptr, nullptr);
      gst_object_unref(tee_sink);
    }
  }
  for(guint i=0;i<6;++i){
    if (!f5r_make_preview_branch(sources[i], false) ||
        !f5r_make_preview_branch(sources[i], true)) {
      fail("decoded-source preview preparation branch creation failed");
      std::exit(2);
    }
  }
  g_timeout_add(25, f5r_fullscreen_request_tick, nullptr);
  event(nullptr,"DETECTION_GRAPH","six RTSP/NVDEC owners->decoded tee->{analytics: nvstreammux(batch=6)->nvinfer(YOLO26m FP16 640)->NvDCF(local6,type2,shadow-output)->canonical object feed; preview: parallel GPU conversion to 960x540 for tiles and request-gated single 1920x1080 fullscreen branch}->exact source-frame/PTS join->atomic image+tracker metadata shared-memory slot");
#else
  event(nullptr,"DETECTION_GRAPH","direct MP4 sources with explicit no-loop/no-reconnect->nvstreammux isolation geometry->YOLO26m->local NvDCF->raw PGIE/tracker evidence; identity/MV3DT/UI preview disabled");
#endif
#else
  auto *convert=make("nvvideoconvert","osd-gpu-convert"), *filter=make("capsfilter","osd-rgba-nvmm");
  auto *rgba=gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");
  g_object_set(filter,"caps",rgba,nullptr);gst_caps_unref(rgba);
  auto *osd=make("nvdsosd","person-overlay");
  g_object_set(osd,"process-mode",1,"display-text",TRUE,"display-bbox",TRUE,"display-mask",FALSE,"display-clock",FALSE,nullptr);
  link(gie,tiler);
#endif
#ifndef YOLO26_TRACKER
  link(tiler,convert);link(convert,filter);link(filter,osd);link(osd,to);
#else
  (void)to;
#endif
#ifdef YOLO26_TRACKER
  event(nullptr,"PREVIEW_SEMANTICS","clean current-frame pixels in shared memory; PySide overlays only canonical post-NvDCF active or current-PGIE/shadow-fallback objects by exact PTS");
#else
  event(nullptr,"DETECTION_GRAPH","mux->nvinfer(YOLO26m,FP16,batch=6,interval=0)->DeepStream-NMS(iou=0.45,conf=0.25)->tiler->NVMM/RGBA->nvdsosd(GPU,person+confidence)->NVENC inference=1");
#endif
}
