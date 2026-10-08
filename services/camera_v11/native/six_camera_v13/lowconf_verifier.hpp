// Opt-in, bounded PeopleNet context-crop verifier for low-confidence YOLO
// candidates. This branch never feeds the primary NvDCF path.
#pragma once

#include "lowconf_contract.hpp"
#include <gst/app/gstappsrc.h>
#include <nvdsmeta.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <nvdsinfer_context.h>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <dlfcn.h>

struct LowConfVerifierProposal {
  float confidence = 0;
  float left = 0, top = 0, width = 0, height = 0;
};

static bool lowconf_verifier_surface_dimensions(
    GstBuffer *buffer, guint batch_id, guint *width, guint *height) {
  if (!buffer || !width || !height) return false;
  GstMapInfo map = GST_MAP_INFO_INIT;
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) return false;
  auto *batch = reinterpret_cast<NvBufSurface*>(map.data);
  const bool valid = batch && batch_id < batch->batchSize && batch->surfaceList;
  if (valid) {
    *width = batch->surfaceList[batch_id].width;
    *height = batch->surfaceList[batch_id].height;
  }
  gst_buffer_unmap(buffer, &map);
  return valid && *width > 0 && *height > 0;
}

#include "context_trt_runtime.hpp"

struct LowConfVerifierRequest {
  LowConfFrameKey key;
  LowConfCandidate candidate;
  guint frame_width = 0, frame_height = 0;
  guint64 verifier_pts = GST_CLOCK_TIME_NONE;
  guint64 worker_started_ns = 0;
  guint64 crop_completed_ns = 0;
  guint64 push_completed_ns = 0;
  guint64 infer_completed_ns = 0;
  float expected_left = 0, expected_top = 0, expected_width = 0, expected_height = 0;
  ShadowContextTransform transform;
  size_t selected_proposal_index = 0;
  bool crop_ready = false;
  bool complete = false;
  bool consumed = false;
  bool active_decision_queued = false;
};

struct LowConfVerifierJob {
  GstBuffer *buffer = nullptr;
  guint batch_id = 0;
  guint source_id = 0;
  guint64 serial = 0;
};

struct LowConfDeferredFrame {
  LowConfFrameKey key;
  bool wait_for_verifier = false;
  std::function<void(std::vector<LowConfCandidate>)> publish;
};

struct LowConfCudaScratch {
  void *stream = nullptr;
  guint8 *host = nullptr;
  size_t capacity = 0;
  int gpu_id = -1;
  ~LowConfCudaScratch() {
    void *runtime = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
    if (!runtime) return;
    using FreeHostFn = int (*)(void*);
    using StreamDestroyFn = int (*)(void*);
    auto free_host = reinterpret_cast<FreeHostFn>(dlsym(runtime, "cudaFreeHost"));
    auto stream_destroy = reinterpret_cast<StreamDestroyFn>(dlsym(runtime, "cudaStreamDestroy"));
    if (host && free_host) free_host(host);
    if (stream && stream_destroy) stream_destroy(stream);
    dlclose(runtime);
  }
};
static thread_local LowConfCudaScratch lowconf_cuda_scratch;

static std::array<GstElement*, 6> lowconf_appsrc{};
static FILE *lowconf_evidence = nullptr;
static FILE *lowconf_assignment_evidence = nullptr;
static std::mutex lowconf_mutex;
static std::condition_variable lowconf_condition;
// Keep verifier scheduling isolated by source. A single FIFO let simultaneous
// target births on one camera delay every other camera, so a valid person could
// miss its same-frame decision simply because unrelated crops were ahead of it.
static std::array<std::deque<LowConfVerifierJob>, 6> lowconf_jobs;
static std::map<guint64, LowConfVerifierRequest> lowconf_requests;
static std::map<guint64, guint64> lowconf_pts_to_serial;
static std::map<LowConfFrameKey, std::vector<guint64>> lowconf_frame_requests;
static std::array<std::deque<ActiveTrackVerificationDecision>, 6>
    lowconf_completed_active_decisions;
static std::array<std::thread, 6> lowconf_workers;
static std::atomic_bool lowconf_accepting{false};
static std::atomic_bool lowconf_worker_stop{false};
static std::array<std::atomic_bool, 6> lowconf_eos_sent{};
static std::atomic<guint64> lowconf_next_serial{0};
static std::atomic<guint64> lowconf_submitted{0}, lowconf_queue_drops{0};
static std::atomic<guint64> lowconf_active_assign_calls{0};
static std::atomic<guint64> lowconf_active_assign_hits{0};
static std::atomic<guint64> lowconf_active_assign_misses{0};
static std::atomic<guint64> lowconf_crop_failures{0}, lowconf_inferred{0};
static std::atomic<guint64> lowconf_confirmed{0}, lowconf_rejected{0};
static std::atomic<guint64> lowconf_late{0}, lowconf_unmatched_result{0};
static std::atomic<guint64> lowconf_infer_total_us{0}, lowconf_infer_max_us{0};
static std::atomic<guint64> shadow_context_submitted{0}, shadow_context_confirmed{0};
static constexpr guint64 LOWCONF_RESULT_DEADLINE_NS = 26000000ULL;
// A newly active tracker birth is held only on the per-camera canonical
// publisher until its exact-frame appearance check completes. In the six-
// camera r4 replay, verified active-birth results took up to 170.392 ms while
// the low-confidence/shadow path remained on its 26 ms fail-closed budget.
// Keep this wait bounded and isolated from DeepStream streaming callbacks.
static constexpr guint64 ACTIVE_BIRTH_RESULT_DEADLINE_NS = 200000000ULL;
static constexpr size_t LOWCONF_JOB_QUEUE_LIMIT_PER_CAMERA = 8;
static constexpr size_t LOWCONF_REQUEST_LIMIT = 64;

static guint64 lowconf_verifier_result_deadline_ns(
    const LowConfCandidate &candidate) {
  return candidate.active_track_birth
      ? ACTIVE_BIRTH_RESULT_DEADLINE_NS
      : LOWCONF_RESULT_DEADLINE_NS;
}
static constexpr size_t LOWCONF_PUBLISH_QUEUE_LIMIT = 32;
static std::mutex lowconf_publish_mutex;
static std::condition_variable lowconf_publish_condition;
static std::array<std::deque<LowConfDeferredFrame>, 6> lowconf_publish_queues;
static std::array<std::thread, 6> lowconf_publish_workers;
static std::atomic_bool lowconf_publish_accepting{false};
static std::atomic_bool lowconf_publish_stop{false};
static std::atomic<guint64> lowconf_selective_wait_frames{0};
static std::atomic<guint64> lowconf_publish_queue_full{0};
static std::atomic<guint64> lowconf_publish_queue_high_water{0};
static std::atomic<guint64> lowconf_deadline_expired{0};
static void lowconf_verifier_publish_worker_main(guint source_id);
static constexpr size_t LOWCONF_GPU_SURFACE_POOL_SIZE = 4;
struct LowConfGpuSurfacePool {
  std::mutex mutex;
  std::vector<NvBufSurface*> free;
  std::vector<NvBufSurface*> all;
};
struct LowConfGpuSurfaceLease {
  guint source_id = 0;
  NvBufSurface *surface = nullptr;
};
static std::array<LowConfGpuSurfacePool, 6> lowconf_gpu_surface_pools;
static std::array<NvDsInferContextHandle, 6> lowconf_context_handles{};
static std::atomic<guint> lowconf_context_input_side{448};
static std::array<std::array<NvDsInferDetectionParams, 80>, 6>
    lowconf_context_detection_params{};
static std::array<char, 64> lowconf_context_output_name{};
static std::array<char*, 1> lowconf_context_output_names{};
static std::array<char, 64> lowconf_context_input_name{};
static float lowconf_context_net_scale = 1.0f / 255.0f;
struct LowConfContextPreprocessApi {
  using CreateFn = int (*)(int, size_t, void**, void**);
  using RgbFn = int (*)(void*, const unsigned char*, size_t, void*, unsigned int,
      unsigned int, unsigned int, float);
  using SyncFn = int (*)(void*);
  using DestroyFn = void (*)(void*, void*);
  void *library = nullptr;
  void *stream = nullptr;
  void *device_buffer = nullptr;
  CreateFn create = nullptr;
  RgbFn rgb = nullptr;
  SyncFn sync = nullptr;
  DestroyFn destroy = nullptr;
};
static LowConfContextPreprocessApi lowconf_context_preprocess_api;
static std::array<LowConfContextPreprocessApi, 6> lowconf_context_preprocess_lanes{};
static std::mutex lowconf_nvbuf_transform_session_mutex;
static bool lowconf_context_per_camera_trt_enabled() {
  return g_getenv("V11_CONTEXT_VERIFIER_PER_CAMERA") &&
      std::strcmp(g_getenv("V11_CONTEXT_VERIFIER_PER_CAMERA"), "1") == 0;
}
struct LowConfContextInputJob {
  guint source_id = 0;
  guint64 serial = 0;
  NvBufSurface *surface = nullptr;
};
static std::mutex lowconf_context_queue_mutex;
static std::condition_variable lowconf_context_queue_condition;
static std::deque<LowConfContextInputJob> lowconf_context_queue;
static std::thread lowconf_context_worker;
static std::atomic_bool lowconf_context_worker_stop{false};
static std::atomic<guint64> lowconf_context_queue_high_water{0};

static void lowconf_queue_active_decision_locked(const LowConfVerifierRequest &request) {
  if (!request.candidate.active_track_birth || request.key.source_id >=
      lowconf_completed_active_decisions.size() ||
      !request.candidate.active_track_id_assigned) return;
  auto &queue = lowconf_completed_active_decisions[request.key.source_id];
  // The source tracker callback drains these decisions continuously. Bound the
  // handoff during a stopped source; a dropped decision leaves the track
  // unverified and therefore fails closed.
  constexpr size_t kDecisionQueueLimit = 128;
  if (queue.size() >= kDecisionQueueLimit) queue.pop_front();
  queue.push_back({request.candidate.native_track_id, request.key.frame_num,
      request.candidate.verifier_done && request.candidate.verifier_confirmed,
      request.candidate.verifier_done && request.candidate.verifier_no_local_person});
}

static bool lowconf_verifier_enabled() {
  const char *value = g_getenv("V11_LOWCONF_VERIFIER_ENABLE");
  return value && std::strcmp(value, "1") == 0;
}

static bool active_track_verifier_enabled() {
  const char *value = g_getenv("V11_ACTIVE_TRACK_VERIFIER_ENABLE");
  return value && std::strcmp(value, "1") == 0;
}

static bool shadow_context_refresh_enabled() {
  const char *value = g_getenv("V11_SHADOW_CONTEXT_REFRESH_ENABLE");
  return value && std::strcmp(value, "1") == 0;
}

static bool lowconf_verifier_gpu_crop_enabled() {
  const char *value = g_getenv("V11_CONTEXT_VERIFIER_GPU_CROP");
  return value && std::strcmp(value, "1") == 0;
}

static bool lowconf_candidate_centered_crop_enabled() {
  const char *value = g_getenv("V11_CONTEXT_CANDIDATE_CROP");
  return value && std::strcmp(value, "1") == 0;
}

static ShadowContextCrop lowconf_compute_verifier_crop(
    unsigned int surface_width, unsigned int surface_height,
    unsigned int frame_width, unsigned int frame_height,
    const ShadowContextRect &candidate, unsigned int verifier_side) {
  if (lowconf_candidate_centered_crop_enabled())
    return shadow_context_compute_candidate_crop(surface_width, surface_height,
        frame_width, frame_height, candidate, verifier_side);
  return shadow_context_compute_crop(surface_width, surface_height,
      frame_width, frame_height, candidate, verifier_side);
}

static bool lowconf_verifier_native_context_enabled() {
  const char *value = g_getenv("V11_CONTEXT_VERIFIER_BACKEND");
  return value && std::strcmp(value, "nvdsinfer-context") == 0;
}

static bool lowconf_verifier_direct_trt_enabled() {
  const char *value = g_getenv("V11_CONTEXT_VERIFIER_BACKEND");
  return value && std::strcmp(value, "trt-runtime") == 0;
}

static bool lowconf_verifier_context_backend_enabled() {
  return lowconf_verifier_native_context_enabled() ||
      lowconf_verifier_direct_trt_enabled();
}

static bool lowconf_verifier_preprocessed_context_enabled() {
  const char *value = g_getenv("V11_CONTEXT_VERIFIER_PREPROCESSED");
  return value && std::strcmp(value, "1") == 0;
}

static bool lowconf_context_preprocessor_initialize(guint side) {
  const char *library = g_getenv("V11_CONTEXT_PREPROCESS_LIBRARY");
  if (!library || !*library) {
    g_printerr("LOWCONF_CONTEXT_PREPROCESS_FAIL missing_library_path\n");
    return false;
  }
  auto &api = lowconf_context_preprocess_api;
  api.library = dlopen(library, RTLD_NOW | RTLD_LOCAL);
  if (!api.library) {
    g_printerr("LOWCONF_CONTEXT_PREPROCESS_FAIL dlopen=%s\n", dlerror());
    return false;
  }
  api.create = reinterpret_cast<LowConfContextPreprocessApi::CreateFn>(
      dlsym(api.library, "lowconf_context_preprocess_create"));
  api.rgb = reinterpret_cast<LowConfContextPreprocessApi::RgbFn>(
      dlsym(api.library, "lowconf_context_preprocess_rgb"));
  api.sync = reinterpret_cast<LowConfContextPreprocessApi::SyncFn>(
      dlsym(api.library, "lowconf_context_preprocess_sync"));
  api.destroy = reinterpret_cast<LowConfContextPreprocessApi::DestroyFn>(
      dlsym(api.library, "lowconf_context_preprocess_destroy"));
  if (!api.create || !api.rgb || !api.sync || !api.destroy) {
    g_printerr("LOWCONF_CONTEXT_PREPROCESS_FAIL incomplete_api\n");
    return false;
  }
  constexpr size_t max_batch = 6;
  const size_t bytes = 3 * static_cast<size_t>(side) * side * sizeof(float);
  if (lowconf_context_per_camera_trt_enabled()) {
    for (size_t index = 0; index < lowconf_context_preprocess_lanes.size(); ++index) {
      auto &lane = lowconf_context_preprocess_lanes[index];
      lane = api;
      lane.library = nullptr;  // the parent API owns the single dlopen handle
      const int status = lane.create(0, bytes, &lane.stream, &lane.device_buffer);
      if (status != 0 || !lane.stream || !lane.device_buffer) {
        g_printerr("LOWCONF_CONTEXT_PREPROCESS_FAIL lane=%zu cuda_create status=%d\n",
            index, status);
        return false;
      }
    }
    g_print("LOWCONF_CONTEXT_PREPROCESS_READY side=%u lanes=6 batch=1 input=RGB-HWC->normalized-FP32-NCHW\n",
        side);
    return true;
  }
  const int status = api.create(0, bytes * max_batch, &api.stream, &api.device_buffer);
  if (status != 0 || !api.stream || !api.device_buffer) {
    g_printerr("LOWCONF_CONTEXT_PREPROCESS_FAIL cuda_create status=%d\n", status);
    return false;
  }
  g_print("LOWCONF_CONTEXT_PREPROCESS_READY side=%u batch=%zu input=RGB-HWC->normalized-FP32-NCHW\n",
      side, max_batch);
  return true;
}

static guint lowconf_verifier_batch_size() {
  const char *value = g_getenv("V11_CONTEXT_VERIFIER_BATCH_SIZE");
  if (!value || !*value) return 1;
  char *end = nullptr;
  const guint64 parsed = g_ascii_strtoull(value, &end, 10);
  return end && *end == '\0' && (parsed == 1 || parsed == 2 || parsed == 6)
      ? static_cast<guint>(parsed) : 0;
}

static bool context_verifier_enabled() {
  return lowconf_verifier_enabled() || active_track_verifier_enabled();
}

static bool lowconf_nv12_format(NvBufSurfaceColorFormat format) {
  return format == NVBUF_COLOR_FORMAT_NV12 || format == NVBUF_COLOR_FORMAT_NV12_ER ||
      format == NVBUF_COLOR_FORMAT_NV12_709 || format == NVBUF_COLOR_FORMAT_NV12_709_ER ||
      format == NVBUF_COLOR_FORMAT_NV12_2020;
}

static float lowconf_iou(float al, float at, float aw, float ah,
                         float bl, float bt, float bw, float bh) {
  const float x1 = std::max(al, bl), y1 = std::max(at, bt);
  const float x2 = std::min(al + aw, bl + bw), y2 = std::min(at + ah, bt + bh);
  const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  const float total = aw * ah + bw * bh - intersection;
  return total > 0.0f ? intersection / total : 0.0f;
}

static float lowconf_center_distance(float al, float at, float aw, float ah,
                                     float bl, float bt, float bw, float bh) {
  return std::hypot(al + aw * .5f - bl - bw * .5f,
                    at + ah * .5f - bt - bh * .5f);
}

static bool lowconf_strict_match(float expected_left, float expected_top,
                                 float expected_width, float expected_height,
                                 const LowConfVerifierProposal &proposal,
                                 float *iou_out, float *distance_out) {
  const float score = lowconf_iou(expected_left, expected_top, expected_width, expected_height,
      proposal.left, proposal.top, proposal.width, proposal.height);
  const float distance = lowconf_center_distance(expected_left, expected_top,
      expected_width, expected_height, proposal.left, proposal.top,
      proposal.width, proposal.height);
  const float wr = std::max(expected_width, proposal.width) /
      std::max(1.0f, std::min(expected_width, proposal.width));
  const float hr = std::max(expected_height, proposal.height) /
      std::max(1.0f, std::min(expected_height, proposal.height));
  if (iou_out) *iou_out = score;
  if (distance_out) *distance_out = distance;
  return score >= .30f && distance <= 50.0f && wr <= 2.0f && hr <= 2.0f;
}

// An active-track appearance verifier answers whether the candidate location
// contains a person, not whether the crop detector reproduces the primary
// detector's exact box geometry. A context crop often yields a fuller box, so
// IoU/size-ratio gates intended for shadow fallback are inappropriate here.
// Require the candidate center to lie inside exactly one local person proposal;
// ambiguous overlaps fail closed as inconclusive.
static bool lowconf_active_location_match(float expected_left, float expected_top,
    float expected_width, float expected_height,
    const LowConfVerifierProposal &proposal, float *iou_out) {
  const float cx = expected_left + expected_width * 0.5f;
  const float cy = expected_top + expected_height * 0.5f;
  const bool contains = cx >= proposal.left && cx <= proposal.left + proposal.width &&
      cy >= proposal.top && cy <= proposal.top + proposal.height;
  if (iou_out) *iou_out = contains ? lowconf_iou(expected_left, expected_top,
      expected_width, expected_height, proposal.left, proposal.top,
      proposal.width, proposal.height) : 0.0f;
  return contains;
}

static GstElement *lowconf_make_element(const char *factory, const char *name) {
  GstElement *element = gst_element_factory_make(factory, name);
  if (!element) {
    g_printerr("LOWCONF_VERIFIER_START_FAIL missing_plugin=%s\n", factory);
    std::exit(2);
  }
  gst_bin_add(GST_BIN(pipeline), element);
  return element;
}

static NvBufSurface *lowconf_gpu_surface_acquire(guint source_id) {
  if (source_id >= lowconf_gpu_surface_pools.size()) return nullptr;
  auto &pool = lowconf_gpu_surface_pools[source_id];
  std::lock_guard<std::mutex> guard(pool.mutex);
  if (pool.free.empty()) return nullptr;
  NvBufSurface *surface = pool.free.back();
  pool.free.pop_back();
  return surface;
}

static void lowconf_gpu_surface_release(guint source_id, NvBufSurface *surface) {
  if (!surface || source_id >= lowconf_gpu_surface_pools.size()) return;
  auto &pool = lowconf_gpu_surface_pools[source_id];
  std::lock_guard<std::mutex> guard(pool.mutex);
  pool.free.push_back(surface);
}

static void lowconf_release_gpu_surface_lease(gpointer user_data) {
  auto *lease = static_cast<LowConfGpuSurfaceLease*>(user_data);
  if (!lease) return;
  lowconf_gpu_surface_release(lease->source_id, lease->surface);
  delete lease;
}

static bool lowconf_gpu_surface_pools_initialize(guint gpu_id) {
  NvBufSurfaceCreateParams params{};
  params.gpuId = gpu_id;
  const bool native_context = lowconf_verifier_context_backend_enabled();
  const guint context_side = lowconf_context_input_side.load();
  params.width = native_context ? context_side : 640;
  params.height = native_context ? context_side : 640;
  params.colorFormat = native_context ? NVBUF_COLOR_FORMAT_RGB : NVBUF_COLOR_FORMAT_NV12;
  params.layout = NVBUF_LAYOUT_PITCH;
  params.memType = NVBUF_MEM_CUDA_DEVICE;
  params.isContiguous = TRUE;
  for (guint source_id = 0; source_id < lowconf_gpu_surface_pools.size(); ++source_id) {
    auto &pool = lowconf_gpu_surface_pools[source_id];
    for (size_t i = 0; i < LOWCONF_GPU_SURFACE_POOL_SIZE; ++i) {
      NvBufSurface *surface = nullptr;
      if (NvBufSurfaceCreate(&surface, 1, &params) != 0 || !surface ||
          !surface->surfaceList || !surface->surfaceList[0].dataPtr) {
        if (surface) NvBufSurfaceDestroy(surface);
        for (auto &cleanup_pool : lowconf_gpu_surface_pools) {
          for (NvBufSurface *allocated : cleanup_pool.all) NvBufSurfaceDestroy(allocated);
          cleanup_pool.all.clear();
          cleanup_pool.free.clear();
        }
        return false;
      }
      surface->numFilled = 1;
      pool.all.push_back(surface);
      pool.free.push_back(surface);
    }
  }
  g_print("LOWCONF_GPU_SURFACE_POOL initialized_per_source=%zu sources=%zu surfaces=%zu\n",
      LOWCONF_GPU_SURFACE_POOL_SIZE, lowconf_gpu_surface_pools.size(),
      LOWCONF_GPU_SURFACE_POOL_SIZE * lowconf_gpu_surface_pools.size());
  return true;
}

static bool lowconf_context_read_string(GKeyFile *file, const char *group,
                                        const char *key, std::string *value) {
  GError *error = nullptr;
  gchar *loaded = g_key_file_get_string(file, group, key, &error);
  if (!loaded) {
    if (error) {
      g_printerr("LOWCONF_CONTEXT_CONFIG_FAIL key=%s reason=%s\n", key, error->message);
      g_error_free(error);
    }
    return false;
  }
  *value = loaded;
  g_free(loaded);
  return true;
}

static void lowconf_context_log(NvDsInferContextHandle, unsigned int unique_id,
    NvDsInferLogLevel level, const char *message, void *) {
  g_printerr("LOWCONF_CONTEXT_LOG id=%u level=%d message=%s\n", unique_id,
      static_cast<int>(level), message ? message : "");
}

static bool lowconf_verifier_contexts_initialize(const char *config_path) {
  GKeyFile *file = g_key_file_new();
  GError *error = nullptr;
  if (!g_key_file_load_from_file(file, config_path, G_KEY_FILE_NONE, &error)) {
    g_printerr("LOWCONF_CONTEXT_CONFIG_FAIL path=%s reason=%s\n", config_path,
        error ? error->message : "unknown");
    if (error) g_error_free(error);
    g_key_file_unref(file);
    return false;
  }
  std::string engine, parser, parse_function, labels, dims, output_name;
  if (!lowconf_context_read_string(file, "property", "model-engine-file", &engine) ||
      !lowconf_context_read_string(file, "property", "custom-lib-path", &parser) ||
      !lowconf_context_read_string(file, "property", "parse-bbox-func-name", &parse_function) ||
      !lowconf_context_read_string(file, "property", "labelfile-path", &labels) ||
      !lowconf_context_read_string(file, "property", "infer-dims", &dims) ||
      !lowconf_context_read_string(file, "property", "output-blob-names", &output_name)) {
    g_key_file_unref(file);
    return false;
  }
  gint gpu_id = 0, network_mode = -1, num_classes = 0, cluster_mode = -1;
  gdouble scale = 0.0, threshold = -1.0, nms_iou = -1.0;
  gint top_k = 0;
  if (!g_key_file_has_key(file, "property", "gpu-id", nullptr) ||
      !g_key_file_has_key(file, "property", "network-mode", nullptr) ||
      !g_key_file_has_key(file, "property", "num-detected-classes", nullptr) ||
      !g_key_file_has_key(file, "property", "cluster-mode", nullptr) ||
      !g_key_file_has_key(file, "class-attrs-all", "pre-cluster-threshold", nullptr) ||
      !g_key_file_has_key(file, "class-attrs-all", "nms-iou-threshold", nullptr) ||
      !g_key_file_has_key(file, "class-attrs-all", "topk", nullptr)) {
    g_printerr("LOWCONF_CONTEXT_CONFIG_FAIL missing required inference field\n");
    g_key_file_unref(file);
    return false;
  }
  gpu_id = g_key_file_get_integer(file, "property", "gpu-id", nullptr);
  network_mode = g_key_file_get_integer(file, "property", "network-mode", nullptr);
  num_classes = g_key_file_get_integer(file, "property", "num-detected-classes", nullptr);
  cluster_mode = g_key_file_get_integer(file, "property", "cluster-mode", nullptr);
  scale = g_key_file_get_double(file, "property", "net-scale-factor", nullptr);
  threshold = g_key_file_get_double(file, "class-attrs-all", "pre-cluster-threshold", nullptr);
  nms_iou = g_key_file_get_double(file, "class-attrs-all", "nms-iou-threshold", nullptr);
  top_k = g_key_file_get_integer(file, "class-attrs-all", "topk", nullptr);
  gint channels = 0, input_height = 0, input_width = 0;
  const bool dims_ok = std::sscanf(dims.c_str(), "%d;%d;%d", &channels,
      &input_height, &input_width) == 3;
  const bool supported_side = input_height == input_width &&
      (input_width == 320 || input_width == 448 || input_width == 512);
  const bool params_ok = dims_ok && channels == 3 && supported_side && gpu_id == 0 &&
      network_mode == 2 && num_classes == 80 &&
      cluster_mode == 2 && std::isfinite(scale) && scale > 0.0 &&
      std::isfinite(threshold) && threshold >= 0.0 && threshold <= 1.0 &&
      std::isfinite(nms_iou) && nms_iou >= 0.0 && nms_iou <= 1.0 && top_k > 0 &&
      output_name == "output0";
  g_key_file_unref(file);
  if (!params_ok || !g_file_test(engine.c_str(), G_FILE_TEST_IS_REGULAR) ||
      !g_file_test(parser.c_str(), G_FILE_TEST_IS_REGULAR)) {
    g_printerr("LOWCONF_CONTEXT_CONFIG_FAIL unsupported_or_missing_artifact config=%s dims=%s output=%s\n",
        config_path, dims.c_str(), output_name.c_str());
    return false;
  }

  lowconf_context_input_side = static_cast<guint>(input_width);
  lowconf_context_net_scale = static_cast<float>(scale);

  g_strlcpy(lowconf_context_output_name.data(), output_name.c_str(),
      lowconf_context_output_name.size());
  lowconf_context_output_names[0] = lowconf_context_output_name.data();
  const bool direct_trt = lowconf_verifier_direct_trt_enabled();
  const guint context_count = lowconf_verifier_context_backend_enabled() ? 1u :
      static_cast<guint>(lowconf_context_handles.size());
  if (direct_trt) {
    if (!lowconf_verifier_preprocessed_context_enabled() ||
        !lowconf_context_trt_initialize(engine, parser, parse_function,
            static_cast<guint>(input_width), lowconf_verifier_batch_size(),
            static_cast<float>(threshold), static_cast<float>(nms_iou),
            static_cast<guint>(top_k)) ||
        !lowconf_context_preprocessor_initialize(static_cast<guint>(input_width))) {
      g_printerr("LOWCONF_CONTEXT_TRT_INIT_FAIL config=%s\n", config_path);
      return false;
    }
  } else for (guint source_id = 0; source_id < context_count; ++source_id) {
    NvDsInferContextInitParams params{};
    NvDsInferContext_ResetInitParams(&params);
    params.uniqueID = 9200 + source_id;
    params.networkMode = NvDsInferNetworkMode_FP16;
    params.netInputOrder = NvDsInferTensorOrder_kNCHW;
    g_strlcpy(params.modelEngineFilePath, engine.c_str(), sizeof(params.modelEngineFilePath));
    g_strlcpy(params.customLibPath, parser.c_str(), sizeof(params.customLibPath));
    g_strlcpy(params.customBBoxParseFuncName, parse_function.c_str(),
        sizeof(params.customBBoxParseFuncName));
    g_strlcpy(params.labelsFilePath, labels.c_str(), sizeof(params.labelsFilePath));
    params.maxBatchSize = lowconf_verifier_context_backend_enabled()
        ? lowconf_verifier_batch_size() : 1u;
    params.networkScaleFactor = static_cast<float>(scale);
    params.networkInputFormat = NvDsInferFormat_RGB;
    params.offsets[0] = params.offsets[1] = params.offsets[2] = 0.0f;
    params.numOffsets = 3;
    params.networkType = NvDsInferNetworkType_Detector;
    params.numDetectedClasses = static_cast<unsigned int>(num_classes);
    params.perClassDetectionParams = lowconf_context_detection_params[source_id].data();
    for (auto &class_params : lowconf_context_detection_params[source_id]) {
      class_params.preClusterThreshold = static_cast<float>(threshold);
      class_params.nmsIOUThreshold = static_cast<float>(nms_iou);
      class_params.topK = top_k;
    }
    params.gpuID = static_cast<unsigned int>(gpu_id);
    params.outputBufferPoolSize = 2;
    params.inferInputDims.c = static_cast<unsigned int>(channels);
    params.inferInputDims.h = static_cast<unsigned int>(input_height);
    params.inferInputDims.w = static_cast<unsigned int>(input_width);
    params.clusterMode = NVDSINFER_CLUSTER_NMS;
    params.outputLayerNames = lowconf_context_output_names.data();
    params.numOutputLayers = 1;
    params.copyInputToHostBuffers = 0;
    // This is an opt-in diagnostic backend. The primary PGIE path and the
    // normal context-verifier preprocessing path remain unchanged unless the
    // isolated candidate explicitly provides the native CUDA preprocessor.
    params.inputFromPreprocessedTensor =
        lowconf_verifier_preprocessed_context_enabled() ? 1 : 0;
    g_printerr("LOWCONF_CONTEXT_CREATE_BEGIN source=%u engine=%s batch_max=%u dims=%ux%u format=RGB classes=%d threshold=%.3f nms=%.3f topk=%d\n",
        source_id, engine.c_str(), params.maxBatchSize, params.inferInputDims.w,
        params.inferInputDims.h, num_classes, threshold, nms_iou, top_k);
    const NvDsInferStatus status = NvDsInferContext_Create(
        &lowconf_context_handles[source_id], &params, nullptr, lowconf_context_log);
    g_printerr("LOWCONF_CONTEXT_CREATE_END source=%u status=%d handle=%p\n",
        source_id, static_cast<int>(status),
        static_cast<void*>(lowconf_context_handles[source_id]));
    if (status != NVDSINFER_SUCCESS || !lowconf_context_handles[source_id]) {
      g_printerr("LOWCONF_CONTEXT_CREATE_FAIL source=%u status=%d engine=%s\n",
          source_id, static_cast<int>(status), engine.c_str());
      for (auto &handle : lowconf_context_handles) {
        if (handle) NvDsInferContext_Destroy(handle);
        handle = nullptr;
      }
      return false;
    }
  }
  if (lowconf_verifier_preprocessed_context_enabled() && !direct_trt) {
    const guint layers_count = NvDsInferContext_GetNumLayersInfo(lowconf_context_handles[0]);
    std::vector<NvDsInferLayerInfo> layers(layers_count);
    if (!layers_count) {
      g_printerr("LOWCONF_CONTEXT_PREPROCESS_FAIL layer_info\n");
      return false;
    }
    NvDsInferContext_FillLayersInfo(lowconf_context_handles[0], layers.data());
    const NvDsInferLayerInfo *input_layer = nullptr;
    for (const auto &layer : layers) {
      if (!layer.isInput) continue;
      if (input_layer) {
        g_printerr("LOWCONF_CONTEXT_PREPROCESS_FAIL multiple_input_layers\n");
        return false;
      }
      input_layer = &layer;
    }
    if (!input_layer || input_layer->dataType != FLOAT || !input_layer->layerName ||
        std::strlen(input_layer->layerName) >= lowconf_context_input_name.size()) {
      g_printerr("LOWCONF_CONTEXT_PREPROCESS_FAIL unsupported_input_layer\n");
      return false;
    }
    g_strlcpy(lowconf_context_input_name.data(), input_layer->layerName,
        lowconf_context_input_name.size());
    if (!lowconf_context_preprocessor_initialize(lowconf_context_input_side.load()))
      return false;
  }
  g_print("LOWCONF_CONTEXT_READY contexts=%u engine=%s parser=%s input=%ux%u RGB threshold=%.3f nms=%.3f max_batch=%u scheduling=%s\n",
      context_count, engine.c_str(), parser.c_str(), lowconf_context_input_side.load(),
      lowconf_context_input_side.load(), threshold, nms_iou,
      lowconf_verifier_context_backend_enabled() ? lowconf_verifier_batch_size() : 1u,
      direct_trt ? "direct-tensorrt-context" :
          (lowconf_verifier_native_context_enabled() ? "shared-bounded-dynamic-batch" : "per-source"));
  return true;
}

static void lowconf_log_result(const LowConfVerifierRequest &request,
                               const std::vector<LowConfVerifierProposal> &proposals,
                               guint strict_matches, float selected_confidence,
                               float selected_iou, float selected_distance,
                               const char *status, guint64 done_ns) {
  if (!lowconf_evidence) return;
  flockfile(lowconf_evidence);
  char detector_confidence[32];
  if (request.candidate.shadow_context_refresh) {
    std::snprintf(detector_confidence, sizeof(detector_confidence), "null");
  } else {
    std::snprintf(detector_confidence, sizeof(detector_confidence), "%.6f",
        request.candidate.confidence);
  }
  std::fprintf(lowconf_evidence,
      "{\"camera_id\":\"CAM-%02u\",\"frame_num\":%d,\"pts_ns\":%lu,"
      "\"detector_confidence\":%s,\"detector_bbox\":[%.3f,%.3f,%.3f,%.3f],"
      "\"verification_kind\":\"%s\",\"native_track_id\":%lu,"
      "\"source_frame_num\":%d,\"source_pts_ns\":%lu,\"shadow_hint_confidence\":%.6f,"
      "\"verifier\":\"%s\",\"verifier_confidence\":%.6f,"
      "\"verifier_iou\":%.6f,\"verifier_runner_up_iou\":%.6f,"
      "\"verifier_uniqueness_margin\":%.6f,\"verifier_center_distance\":%.3f,"
      "\"strict_match_count\":%u,\"status\":\"%s\","
      "\"submitted_monotonic_ns\":%lu,\"completed_monotonic_ns\":%lu,"
      "\"worker_started_ns\":%lu,\"crop_completed_ns\":%lu,\"push_completed_ns\":%lu,"
      "\"queue_wait_ms\":%.3f,\"crop_ms\":%.3f,\"push_ms\":%.3f,\"infer_after_push_ms\":%.3f,"
      "\"proposals\":[",
      request.key.source_id + 1, request.key.frame_num, request.key.pts_ns,
      detector_confidence, request.candidate.left, request.candidate.top,
      request.candidate.width, request.candidate.height,
      request.candidate.shadow_context_refresh ? "SHADOW_CONTEXT_REFRESH" :
          (request.candidate.active_track_birth ? "ACTIVE_TRACK_BIRTH" : "LOWCONF_FALLBACK"),
      request.candidate.native_track_id,
      request.candidate.source_frame_num, request.candidate.source_pts_ns,
      request.candidate.shadow_context_refresh ? request.candidate.confidence : 0.0f,
      g_getenv("V11_CONTEXT_VERIFIER_NAME") ? g_getenv("V11_CONTEXT_VERIFIER_NAME") : "configured-nvinfer",
      selected_confidence,
      selected_iou, request.candidate.verifier_runner_up_iou,
      request.candidate.verifier_uniqueness_margin, selected_distance, strict_matches, status,
      request.candidate.submitted_monotonic_ns, done_ns,
      request.worker_started_ns, request.crop_completed_ns, request.push_completed_ns,
      (request.worker_started_ns >= request.candidate.submitted_monotonic_ns
          ? (request.worker_started_ns - request.candidate.submitted_monotonic_ns) / 1.0e6 : 0.0),
      (request.crop_completed_ns >= request.worker_started_ns
          ? (request.crop_completed_ns - request.worker_started_ns) / 1.0e6 : 0.0),
      (request.push_completed_ns >= request.crop_completed_ns
          ? (request.push_completed_ns - request.crop_completed_ns) / 1.0e6 : 0.0),
      (done_ns >= request.push_completed_ns ? (done_ns - request.push_completed_ns) / 1.0e6 : 0.0));
  for (size_t i = 0; i < proposals.size(); ++i) {
    const auto &proposal = proposals[i];
    if (i) std::fputc(',', lowconf_evidence);
    std::fprintf(lowconf_evidence,
        "{\"class_id\":0,\"confidence\":%.6f,\"bbox\":[%.3f,%.3f,%.3f,%.3f]}",
        proposal.confidence, proposal.left, proposal.top, proposal.width, proposal.height);
  }
  std::fprintf(lowconf_evidence, "]}\n");
  funlockfile(lowconf_evidence);
}

static void lowconf_complete_result(guint64 verifier_pts,
                                   const std::vector<LowConfVerifierProposal> &proposals,
                                   guint64 completed_ns) {
  std::lock_guard<std::mutex> guard(lowconf_mutex);
  const auto serial_it = lowconf_pts_to_serial.find(verifier_pts);
  if (serial_it == lowconf_pts_to_serial.end()) {
    ++lowconf_unmatched_result;
    return;
  }
  const guint64 serial = serial_it->second;
  auto request_it = lowconf_requests.find(serial);
  if (request_it == lowconf_requests.end()) {
    lowconf_pts_to_serial.erase(serial_it);
    ++lowconf_unmatched_result;
    return;
  }
  auto &request = request_it->second;
  request.infer_completed_ns = completed_ns;
  guint strict_count = 0;
  guint active_location_count = 0;
  float selected_confidence = 0, selected_iou = 0, runner_up_iou = 0, selected_distance = 0;
  for (size_t proposal_index = 0; proposal_index < proposals.size(); ++proposal_index) {
    const auto &proposal = proposals[proposal_index];
    float score = 0, distance = 0;
    const ShadowContextRect expected{request.expected_left, request.expected_top,
        request.expected_width, request.expected_height};
    const ShadowContextRect current{proposal.left, proposal.top,
        proposal.width, proposal.height};
    const bool match = request.candidate.active_track_birth
        ? lowconf_active_location_match(request.expected_left, request.expected_top,
            request.expected_width, request.expected_height, proposal, &score)
        : request.candidate.shadow_context_refresh
            ? shadow_context_spatial_match(expected, current, &score, &distance)
        : lowconf_strict_match(request.expected_left, request.expected_top,
            request.expected_width, request.expected_height, proposal, &score, &distance);
    if (match) {
      ++strict_count;
      if (request.candidate.active_track_birth) ++active_location_count;
      if (score > selected_iou) {
        runner_up_iou = selected_iou;
        selected_confidence = proposal.confidence;
        selected_iou = score;
        selected_distance = distance;
        request.selected_proposal_index = proposal_index;
      } else if (score > runner_up_iou) {
        runner_up_iou = score;
      }
    }
  }
  // Multiple context boxes may describe the same person at different scales.
  // Accept only a clearly best location match; otherwise remain fail-closed.
  const float uniqueness_margin = selected_iou - runner_up_iou;
  const bool uniquely_confirmed = strict_count == 1 ||
      (strict_count > 1 && uniqueness_margin >= .10f);
  request.candidate.verifier_done = true;
  request.candidate.verifier_confirmed = uniquely_confirmed;
  request.candidate.verifier_no_local_person = request.candidate.active_track_birth &&
      active_location_count == 0;
  request.candidate.verifier_confidence = selected_confidence;
  request.candidate.verifier_iou = selected_iou;
  request.candidate.verifier_runner_up_iou = runner_up_iou;
  request.candidate.verifier_uniqueness_margin = uniqueness_margin;
  request.candidate.verifier_center_distance = selected_distance;
  if (request.candidate.shadow_context_refresh && request.crop_ready) {
    request.candidate.verifier_boxes.clear();
    request.candidate.verifier_boxes.reserve(proposals.size());
    for (const auto &proposal : proposals) {
      if (!std::isfinite(proposal.confidence) || proposal.confidence < .20f ||
          !std::isfinite(proposal.left) || !std::isfinite(proposal.top) ||
          !std::isfinite(proposal.width) || !std::isfinite(proposal.height) ||
          proposal.width <= 0 || proposal.height <= 0) continue;
      const auto mapped = shadow_context_to_frame(
          ShadowContextRect{proposal.left, proposal.top, proposal.width, proposal.height},
          request.transform);
      const float right = mapped[0] + mapped[2];
      const float bottom = mapped[1] + mapped[3];
      if (!std::isfinite(mapped[0]) || !std::isfinite(mapped[1]) ||
          !std::isfinite(mapped[2]) || !std::isfinite(mapped[3]) ||
          mapped[0] < 0 || mapped[1] < 0 || mapped[2] <= 0 || mapped[3] <= 0 ||
          right > request.frame_width + .1f || bottom > request.frame_height + .1f)
        continue;
      request.candidate.verifier_boxes.push_back({proposal.confidence,
          mapped[0], mapped[1], mapped[2], mapped[3]});
    }
    request.candidate.verifier_confirmed = !request.candidate.verifier_boxes.empty();
  }
  if (uniquely_confirmed && request.candidate.shadow_context_refresh &&
      request.selected_proposal_index < proposals.size()) {
    const auto &proposal = proposals[request.selected_proposal_index];
    const auto mapped = shadow_context_to_frame(
        ShadowContextRect{proposal.left, proposal.top, proposal.width, proposal.height},
        request.transform);
    request.candidate.verifier_bbox_left = mapped[0];
    request.candidate.verifier_bbox_top = mapped[1];
    request.candidate.verifier_bbox_width = mapped[2];
    request.candidate.verifier_bbox_height = mapped[3];
    const float right = mapped[0] + mapped[2];
    const float bottom = mapped[1] + mapped[3];
    if (!std::isfinite(mapped[0]) || !std::isfinite(mapped[1]) ||
        !std::isfinite(mapped[2]) || !std::isfinite(mapped[3]) ||
        mapped[0] < 0 || mapped[1] < 0 || mapped[2] <= 0 || mapped[3] <= 0 ||
        right > request.frame_width + 0.1f || bottom > request.frame_height + 0.1f) {
      request.candidate.verifier_confirmed = false;
      request.candidate.verifier_bbox_width = 0;
      request.candidate.verifier_bbox_height = 0;
    }
  }
  request.candidate.verifier_ready_monotonic_ns = completed_ns;
  request.candidate.verifier_latency_ms = completed_ns >= request.candidate.submitted_monotonic_ns
      ? (completed_ns - request.candidate.submitted_monotonic_ns) / 1.0e6f : 0.0f;
  request.complete = true;
  if (!request.active_decision_queued && request.candidate.active_track_id_assigned) {
    lowconf_queue_active_decision_locked(request);
    request.active_decision_queued = true;
  }
  ++lowconf_inferred;
  if (request.candidate.verifier_confirmed) ++lowconf_confirmed;
  else ++lowconf_rejected;
  if (request.candidate.shadow_context_refresh && request.candidate.verifier_confirmed)
    ++shadow_context_confirmed;
  lowconf_infer_total_us += static_cast<guint64>(request.candidate.verifier_latency_ms * 1000.0f);
  const guint64 latency_us = static_cast<guint64>(request.candidate.verifier_latency_ms * 1000.0f);
  guint64 maximum = lowconf_infer_max_us.load();
  while (latency_us > maximum && !lowconf_infer_max_us.compare_exchange_weak(maximum, latency_us)) {}
  lowconf_log_result(request, proposals, strict_count, selected_confidence,
      selected_iou, selected_distance, request.candidate.verifier_confirmed ? "CONFIRMED" :
      (strict_count > 1 ? "AMBIGUOUS" : "NO_LOCAL_PERSON"), completed_ns);
  if (request.consumed) {
    lowconf_pts_to_serial.erase(serial_it);
    lowconf_requests.erase(request_it);
  }
  lowconf_condition.notify_all();
}

static void lowconf_fail_request(guint64 serial, const char *reason) {
  std::lock_guard<std::mutex> guard(lowconf_mutex);
  auto found = lowconf_requests.find(serial);
  if (found == lowconf_requests.end()) return;
  const guint64 pts = found->second.verifier_pts;
  lowconf_pts_to_serial.erase(pts);
  auto failed = found->second;
  found->second.complete = true;
  found->second.candidate.verifier_done = true;
  found->second.candidate.verifier_confirmed = false;
  // An execution failure is not evidence that a person is absent. In
  // particular, do not invalidate an active track on a verifier timeout/error.
  found->second.candidate.verifier_no_local_person = false;
  if (!found->second.active_decision_queued &&
      found->second.candidate.active_track_id_assigned) {
    lowconf_queue_active_decision_locked(found->second);
    found->second.active_decision_queued = true;
  }
  lowconf_log_result(failed, {}, 0, 0, 0, 0, reason,
      static_cast<guint64>(g_get_monotonic_time()) * 1000);
  if (found->second.consumed) lowconf_requests.erase(found);
  lowconf_condition.notify_all();
}

static void lowconf_context_infer_batch(const std::vector<LowConfContextInputJob> &jobs) {
  if (jobs.empty()) return;
  const bool direct_trt = lowconf_verifier_direct_trt_enabled();
  if ((direct_trt ? !lowconf_context_trt_execution : !lowconf_context_handles[0]) ||
      jobs.size() > 6) {
    for (const auto &job : jobs) {
      lowconf_fail_request(job.serial, "CONTEXT_INPUT_INVALID");
      lowconf_gpu_surface_release(job.source_id, job.surface);
    }
    return;
  }
  std::array<void*, 6> frames{};
  guint input_pitch = 0;
  bool valid = true;
  for (size_t i = 0; i < jobs.size(); ++i) {
    const auto &job = jobs[i];
    valid = valid && job.source_id < lowconf_gpu_surface_pools.size() && job.surface &&
        job.surface->surfaceList && job.surface->surfaceList[0].dataPtr &&
        job.surface->surfaceList[0].colorFormat == NVBUF_COLOR_FORMAT_RGB &&
        job.surface->surfaceList[0].width == lowconf_context_input_side.load() &&
        job.surface->surfaceList[0].height == lowconf_context_input_side.load();
    if (!valid) break;
        frames[i] = job.surface->surfaceList[0].dataPtr;
    const guint pitch = job.surface->surfaceList[0].planeParams.pitch[0];
    if (i == 0) input_pitch = pitch;
    else valid = valid && pitch == input_pitch;
    std::lock_guard<std::mutex> guard(lowconf_mutex);
    auto found = lowconf_requests.find(job.serial);
    if (found != lowconf_requests.end())
      found->second.push_completed_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  }
  if (!valid) {
    for (const auto &job : jobs) {
      lowconf_fail_request(job.serial, "CONTEXT_INPUT_INVALID");
      lowconf_gpu_surface_release(job.source_id, job.surface);
    }
    return;
  }
  NvDsInferStatus queued = NVDSINFER_SUCCESS;
  const guint side = lowconf_context_input_side.load();
  if (lowconf_verifier_preprocessed_context_enabled()) {
    auto &api = lowconf_context_preprocess_api;
    for (size_t i = 0; i < jobs.size(); ++i) {
      const auto &surface = jobs[i].surface->surfaceList[0];
      const int status = api.rgb(api.stream,
          static_cast<const unsigned char*>(surface.dataPtr),
          surface.planeParams.pitch[0], api.device_buffer, side,
          static_cast<unsigned int>(i), 6, lowconf_context_net_scale);
      if (status != 0) {
        for (const auto &job : jobs) {
          lowconf_fail_request(job.serial, "PREPROCESSED_TENSOR_CONVERT_FAILED");
          lowconf_gpu_surface_release(job.source_id, job.surface);
        }
        return;
      }
    }
    if (!direct_trt) {
      const int sync_status = api.sync(api.stream);
      if (sync_status != 0) {
        for (const auto &job : jobs) {
          lowconf_fail_request(job.serial, "PREPROCESSED_TENSOR_SYNC_FAILED");
          lowconf_gpu_surface_release(job.source_id, job.surface);
        }
        return;
      }
    }
    if (direct_trt) {
      for (const auto &job : jobs) {
        std::lock_guard<std::mutex> guard(lowconf_mutex);
        auto found = lowconf_requests.find(job.serial);
        if (found != lowconf_requests.end())
          found->second.push_completed_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
      }
      std::vector<std::vector<LowConfVerifierProposal>> results;
      const guint64 infer_start_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
      const bool inferred = lowconf_context_trt_infer(api.device_buffer, api.stream,
          static_cast<guint>(jobs.size()), &results);
      const guint64 completed_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
      if (g_getenv("V11_CONTEXT_VERIFIER_TRACE"))
        g_printerr("LOWCONF_TRT_BATCH count=%zu infer_ms=%.3f success=%d\n",
            jobs.size(), (completed_ns - infer_start_ns) / 1.0e6, inferred ? 1 : 0);
      if (!inferred || results.size() != jobs.size()) {
        for (const auto &job : jobs) {
          lowconf_fail_request(job.serial, "DIRECT_TRT_INFERENCE_FAILED");
          lowconf_gpu_surface_release(job.source_id, job.surface);
        }
        return;
      }
      for (size_t index = 0; index < jobs.size(); ++index) {
        guint64 verifier_pts = GST_CLOCK_TIME_NONE;
        {
          std::lock_guard<std::mutex> guard(lowconf_mutex);
          const auto found = lowconf_requests.find(jobs[index].serial);
          if (found != lowconf_requests.end()) verifier_pts = found->second.verifier_pts;
        }
        if (verifier_pts != GST_CLOCK_TIME_NONE)
          lowconf_complete_result(verifier_pts, results[index], completed_ns);
      }
      for (const auto &job : jobs)
        lowconf_gpu_surface_release(job.source_id, job.surface);
      return;
    }
    NvDsInferLayerInfo tensor{};
    tensor.dataType = FLOAT;
    tensor.isInput = 1;
    tensor.bindingIndex = -1;
    tensor.layerName = lowconf_context_input_name.data();
    tensor.buffer = api.device_buffer;
    tensor.inferDims.numDims = 4;
    tensor.inferDims.d[0] = static_cast<unsigned int>(jobs.size());
    tensor.inferDims.d[1] = 3;
    tensor.inferDims.d[2] = side;
    tensor.inferDims.d[3] = side;
    tensor.inferDims.numElements = static_cast<unsigned int>(jobs.size() * 3 * side * side);
    NvDsInferContextBatchPreprocessedInput input{};
    input.tensors = &tensor;
    input.numInputTensors = 1;
    const guint64 queue_start_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
    // DeepStream 9.1 exposes this as the INvDsInferContext virtual method;
    // unlike QueueInputBatch, it has no C wrapper in this SDK release.
    queued = lowconf_context_handles[0]->queueInputBatchPreprocessed(input);
    const guint64 queue_end_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
    if (g_getenv("V11_CONTEXT_VERIFIER_TRACE"))
      g_printerr("LOWCONF_CONTEXT_PREPROCESSED_BATCH count=%zu side=%u queue_ms=%.3f status=%d\n",
          jobs.size(), side, (queue_end_ns - queue_start_ns) / 1.0e6,
          static_cast<int>(queued));
  } else {
    NvDsInferContextBatchInput input{};
    input.inputFrames = frames.data();
    input.numInputFrames = static_cast<unsigned int>(jobs.size());
    input.inputFormat = NvDsInferFormat_RGB;
    input.inputPitch = input_pitch;
    const guint64 queue_start_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
    queued = NvDsInferContext_QueueInputBatch(lowconf_context_handles[0], &input);
    const guint64 queue_end_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
    if (g_getenv("V11_CONTEXT_VERIFIER_TRACE"))
      g_printerr("LOWCONF_CONTEXT_BATCH_QUEUE count=%zu queue_ms=%.3f status=%d\n",
          jobs.size(), (queue_end_ns - queue_start_ns) / 1.0e6,
          static_cast<int>(queued));
  }
  if (queued != NVDSINFER_SUCCESS) {
    for (const auto &job : jobs) {
      lowconf_fail_request(job.serial, "CONTEXT_QUEUE_FAILED");
      lowconf_gpu_surface_release(job.source_id, job.surface);
    }
    return;
  }
  NvDsInferContextBatchOutput output{};
  const guint64 dequeue_start_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  const NvDsInferStatus dequeued = NvDsInferContext_DequeueOutputBatch(
      lowconf_context_handles[0], &output);
  const guint64 dequeue_end_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  if (g_getenv("V11_CONTEXT_VERIFIER_TRACE"))
    g_printerr("LOWCONF_CONTEXT_BATCH_DEQUEUE count=%zu dequeue_ms=%.3f status=%d frames=%u\n",
        jobs.size(), (dequeue_end_ns - dequeue_start_ns) / 1.0e6,
        static_cast<int>(dequeued), output.numFrames);
  if (dequeued != NVDSINFER_SUCCESS || output.numFrames != jobs.size() || !output.frames) {
    if (dequeued == NVDSINFER_SUCCESS)
      NvDsInferContext_ReleaseBatchOutput(lowconf_context_handles[0], &output);
    for (const auto &job : jobs) {
      lowconf_fail_request(job.serial, "CONTEXT_OUTPUT_INVALID");
      lowconf_gpu_surface_release(job.source_id, job.surface);
    }
    return;
  }
  const float context_to_existing_crop_scale =
      640.0f / static_cast<float>(lowconf_context_input_side.load());
  const guint64 completed_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  for (size_t batch_index = 0; batch_index < jobs.size(); ++batch_index) {
    const auto &frame_output = output.frames[batch_index];
    if (frame_output.outputType != NvDsInferNetworkType_Detector) {
      lowconf_fail_request(jobs[batch_index].serial, "CONTEXT_OUTPUT_INVALID");
      continue;
    }
    std::vector<LowConfVerifierProposal> proposals;
    const auto &detected = frame_output.detectionOutput;
    proposals.reserve(detected.numObjects);
    for (unsigned int i = 0; i < detected.numObjects; ++i) {
      const auto &object = detected.objects[i];
      if (object.classIndex != 0 || !std::isfinite(object.confidence) ||
          object.confidence < .20f || !std::isfinite(object.left) ||
          !std::isfinite(object.top) || !std::isfinite(object.width) ||
          !std::isfinite(object.height) || object.width <= 0 || object.height <= 0) continue;
      proposals.push_back({object.confidence, object.left * context_to_existing_crop_scale,
          object.top * context_to_existing_crop_scale,
          object.width * context_to_existing_crop_scale,
          object.height * context_to_existing_crop_scale});
    }
    guint64 verifier_pts = GST_CLOCK_TIME_NONE;
    {
      std::lock_guard<std::mutex> guard(lowconf_mutex);
      const auto found = lowconf_requests.find(jobs[batch_index].serial);
      if (found != lowconf_requests.end()) verifier_pts = found->second.verifier_pts;
    }
    if (verifier_pts != GST_CLOCK_TIME_NONE)
      lowconf_complete_result(verifier_pts, proposals, completed_ns);
  }
  NvDsInferContext_ReleaseBatchOutput(lowconf_context_handles[0], &output);
  for (const auto &job : jobs)
    lowconf_gpu_surface_release(job.source_id, job.surface);
}

static void lowconf_context_worker_main() {
  const guint batch_limit = lowconf_verifier_context_backend_enabled()
      ? lowconf_verifier_batch_size() : 1u;
  for (;;) {
    std::vector<LowConfContextInputJob> jobs;
    {
      std::unique_lock<std::mutex> guard(lowconf_context_queue_mutex);
      lowconf_context_queue_condition.wait(guard, [] {
        return lowconf_context_worker_stop.load() || !lowconf_context_queue.empty();
      });
      if (lowconf_context_queue.empty() && lowconf_context_worker_stop.load()) break;
      jobs.push_back(lowconf_context_queue.front());
      lowconf_context_queue.pop_front();
      // A very short opportunistic wait lets already-ready cameras share the
      // engine's dynamic batch without imposing frame-period batching latency.
      const auto batch_deadline = std::chrono::steady_clock::now() +
          std::chrono::microseconds(750);
      while (jobs.size() < batch_limit && lowconf_context_queue.size() < batch_limit &&
             !lowconf_context_worker_stop.load()) {
        if (lowconf_context_queue_condition.wait_until(guard, batch_deadline) ==
            std::cv_status::timeout) break;
      }
      while (jobs.size() < batch_limit && !lowconf_context_queue.empty()) {
        jobs.push_back(lowconf_context_queue.front());
        lowconf_context_queue.pop_front();
      }
    }
    lowconf_context_infer_batch(jobs);
  }
}

static void lowconf_context_infer_camera_job(const LowConfContextInputJob &job) {
  if (job.source_id >= lowconf_context_preprocess_lanes.size() || !job.surface ||
      !job.surface->surfaceList || !job.surface->surfaceList[0].dataPtr ||
      job.surface->surfaceList[0].colorFormat != NVBUF_COLOR_FORMAT_RGB ||
      job.surface->surfaceList[0].width != lowconf_context_input_side.load() ||
      job.surface->surfaceList[0].height != lowconf_context_input_side.load()) {
    lowconf_fail_request(job.serial, "PER_CAMERA_CONTEXT_INPUT_INVALID");
    lowconf_gpu_surface_release(job.source_id, job.surface);
    return;
  }
  auto &preprocess = lowconf_context_preprocess_lanes[job.source_id];
  const auto &surface = job.surface->surfaceList[0];
  const guint side = lowconf_context_input_side.load();
  const int converted = preprocess.rgb(preprocess.stream,
      static_cast<const unsigned char*>(surface.dataPtr),
      surface.planeParams.pitch[0], preprocess.device_buffer, side, 0, 1,
      lowconf_context_net_scale);
  if (converted != 0) {
    lowconf_fail_request(job.serial, "PER_CAMERA_PREPROCESS_FAILED");
    lowconf_gpu_surface_release(job.source_id, job.surface);
    return;
  }
  const guint64 submitted_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  guint64 verifier_pts = GST_CLOCK_TIME_NONE;
  {
    std::lock_guard<std::mutex> guard(lowconf_mutex);
    auto found = lowconf_requests.find(job.serial);
    if (found != lowconf_requests.end()) {
      found->second.push_completed_ns = submitted_ns;
      verifier_pts = found->second.verifier_pts;
    }
  }
  std::vector<LowConfVerifierProposal> proposals;
  const guint64 infer_start_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  const bool inferred = lowconf_context_trt_infer_lane(job.source_id,
      preprocess.device_buffer, preprocess.stream, &proposals);
  const guint64 completed_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  if (g_getenv("V11_CONTEXT_VERIFIER_TRACE"))
    g_printerr("LOWCONF_TRT_LANE source=%u infer_ms=%.3f success=%d\n",
        job.source_id, (completed_ns - infer_start_ns) / 1.0e6, inferred ? 1 : 0);
  if (inferred && verifier_pts != GST_CLOCK_TIME_NONE)
    lowconf_complete_result(verifier_pts, proposals, completed_ns);
  else
    lowconf_fail_request(job.serial, "PER_CAMERA_TRT_INFERENCE_FAILED");
  lowconf_gpu_surface_release(job.source_id, job.surface);
}

static GstPadProbeReturn lowconf_infer_exit(GstPad*, GstPadProbeInfo *info, gpointer) {
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer) return GST_PAD_PROBE_OK;
  NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(buffer);
  if (!batch) return GST_PAD_PROBE_OK;
  for (NvDsMetaList *fn = batch->frame_meta_list; fn; fn = fn->next) {
    auto *frame = static_cast<NvDsFrameMeta*>(fn->data);
    if (!frame) continue;
    std::vector<LowConfVerifierProposal> proposals;
    for (NvDsMetaList *on = frame->obj_meta_list; on; on = on->next) {
      auto *object = static_cast<NvDsObjectMeta*>(on->data);
      if (!object || object->class_id != 0 || object->unique_component_id != 2 ||
          !std::isfinite(object->confidence) || object->confidence < .20f) continue;
      const auto &rect = object->rect_params;
      if (!std::isfinite(rect.left) || !std::isfinite(rect.top) ||
          !std::isfinite(rect.width) || !std::isfinite(rect.height) ||
          rect.width <= 0 || rect.height <= 0) continue;
      proposals.push_back({object->confidence, rect.left, rect.top, rect.width, rect.height});
    }
    lowconf_complete_result(frame->buf_pts, proposals,
        static_cast<guint64>(g_get_monotonic_time()) * 1000);
  }
  return GST_PAD_PROBE_OK;
}

// Direct-file isolation sends EOS through the primary post-tracker tee. End
// the independent appsrc branch only after stopping admission; the worker
// drains all already accepted jobs before it emits verifier EOS. This lets
// GStreamer aggregate EOS across both sink branches without dropping work.
static GstPadProbeReturn lowconf_verifier_primary_event(GstPad*, GstPadProbeInfo *info, gpointer) {
  if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
    GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
    if (event && GST_EVENT_TYPE(event) == GST_EVENT_EOS) lowconf_verifier_finish_input();
  }
  return GST_PAD_PROBE_OK;
}

static bool lowconf_prepare_nv12_crop(const LowConfVerifierJob &job,
                                      std::vector<guint8> *cropped,
                                      LowConfVerifierRequest *request,
                                      std::string *reason) {
  GstMapInfo map = GST_MAP_INFO_INIT;
  if (!gst_buffer_map(job.buffer, &map, GST_MAP_READ)) { *reason = "buffer-map"; return false; }
  auto *batch = reinterpret_cast<NvBufSurface*>(map.data);
  if (!batch || job.batch_id >= batch->batchSize || !batch->surfaceList) {
    gst_buffer_unmap(job.buffer, &map); *reason = "batch-index"; return false;
  }
  const auto &surface = batch->surfaceList[job.batch_id];
  const guint width = surface.width, height = surface.height;
  if (!lowconf_nv12_format(surface.colorFormat) || surface.layout != NVBUF_LAYOUT_PITCH ||
      !surface.dataPtr || width < 640 || height < 640 || width % 2 || height % 2) {
    gst_buffer_unmap(job.buffer, &map); *reason = "unsupported-source-surface"; return false;
  }

  using CudaSetDeviceFn = int (*)(int);
  using CudaStreamCreateFn = int (*)(void**, unsigned int);
  using CudaStreamSyncFn = int (*)(void*);
  using CudaMemcpy2DAsyncFn = int (*)(void*, size_t, const void*, size_t,
      size_t, size_t, int, void*);
  using CudaHostAllocFn = int (*)(void**, size_t, unsigned int);
  using CudaHostFreeFn = int (*)(void*);
  using CudaStreamDestroyFn = int (*)(void*);
  static void *cuda_runtime = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
  static auto cuda_set_device = reinterpret_cast<CudaSetDeviceFn>(
      cuda_runtime ? dlsym(cuda_runtime, "cudaSetDevice") : nullptr);
  static auto cuda_stream_create = reinterpret_cast<CudaStreamCreateFn>(
      cuda_runtime ? dlsym(cuda_runtime, "cudaStreamCreateWithFlags") : nullptr);
  static auto cuda_stream_sync = reinterpret_cast<CudaStreamSyncFn>(
      cuda_runtime ? dlsym(cuda_runtime, "cudaStreamSynchronize") : nullptr);
  static auto cuda_memcpy_2d_async = reinterpret_cast<CudaMemcpy2DAsyncFn>(
      cuda_runtime ? dlsym(cuda_runtime, "cudaMemcpy2DAsync") : nullptr);
  static auto cuda_host_alloc = reinterpret_cast<CudaHostAllocFn>(
      cuda_runtime ? dlsym(cuda_runtime, "cudaHostAlloc") : nullptr);
  static auto cuda_host_free = reinterpret_cast<CudaHostFreeFn>(
      cuda_runtime ? dlsym(cuda_runtime, "cudaFreeHost") : nullptr);
  static auto cuda_stream_destroy = reinterpret_cast<CudaStreamDestroyFn>(
      cuda_runtime ? dlsym(cuda_runtime, "cudaStreamDestroy") : nullptr);
  if (!cuda_set_device || !cuda_stream_create || !cuda_stream_sync ||
      !cuda_memcpy_2d_async || !cuda_host_alloc || !cuda_host_free || !cuda_stream_destroy ||
      cuda_set_device(static_cast<int>(batch->gpuId)) != 0) {
    gst_buffer_unmap(job.buffer, &map); *reason = "cuda-runtime"; return false;
  }

  if (!request->frame_width || !request->frame_height) {
    gst_buffer_unmap(job.buffer, &map); *reason = "missing-frame-dimensions"; return false;
  }
  constexpr int output_side = 640;
  const auto crop = lowconf_compute_verifier_crop(width, height,
      request->frame_width, request->frame_height,
      ShadowContextRect{request->candidate.left, request->candidate.top,
          request->candidate.width, request->candidate.height}, output_side);
  if (!crop.valid) {
    gst_buffer_unmap(job.buffer, &map); *reason = "context-too-small-or-invalid-geometry"; return false;
  }
  const int side = crop.side;
  const int crop_x = crop.left;
  const int crop_y = crop.top;
  const float scale_x = crop.surface_per_frame_x;
  const float scale_y = crop.surface_per_frame_y;
  request->transform = {static_cast<float>(crop_x), static_cast<float>(crop_y),
      static_cast<float>(side), static_cast<float>(output_side), scale_x, scale_y};

  const size_t source_y_bytes = static_cast<size_t>(side) * side;
  const size_t source_bytes = source_y_bytes + source_y_bytes / 2;
  auto &scratch = lowconf_cuda_scratch;
  if (scratch.gpu_id != static_cast<int>(batch->gpuId)) {
    if (scratch.host) cuda_host_free(scratch.host);
    if (scratch.stream) cuda_stream_destroy(scratch.stream);
    scratch.host = nullptr;
    scratch.stream = nullptr;
    scratch.capacity = 0;
    scratch.gpu_id = static_cast<int>(batch->gpuId);
  }
  // One worker owns this persistent nonblocking CUDA stream and pinned host
  // scratch. Avoid the legacy-default-stream synchronization caused by
  // cudaMemcpy2D, which can serialize unrelated primary PGIE/NvDCF GPU work.
  if (!scratch.stream && cuda_stream_create(&scratch.stream, 1u) != 0) {
    gst_buffer_unmap(job.buffer, &map); *reason = "cuda-stream-create"; return false;
  }
  if (scratch.capacity < source_bytes) {
    if (scratch.host) cuda_host_free(scratch.host);
    scratch.host = nullptr;
    if (cuda_host_alloc(reinterpret_cast<void**>(&scratch.host), source_bytes, 0u) != 0) {
      gst_buffer_unmap(job.buffer, &map); *reason = "cuda-pinned-host-alloc"; return false;
    }
    scratch.capacity = source_bytes;
  }
  const size_t output_y_bytes = output_side * output_side;
  cropped->resize(output_y_bytes + output_y_bytes / 2);
  const auto *base = static_cast<const guint8*>(surface.dataPtr);
  const guint y_pitch = surface.planeParams.pitch[0];
  const guint uv_pitch = surface.planeParams.pitch[1];
  constexpr int cuda_copy_device_to_host = 2;
  int y_status = cuda_memcpy_2d_async(scratch.host, side,
      base + static_cast<size_t>(crop_y) * y_pitch + crop_x, y_pitch,
      side, side, cuda_copy_device_to_host, scratch.stream);
  int uv_status = y_status == 0 ? cuda_memcpy_2d_async(scratch.host + source_y_bytes, side,
      base + surface.planeParams.offset[1] + static_cast<size_t>(crop_y / 2) * uv_pitch + crop_x,
      uv_pitch, side, side / 2, cuda_copy_device_to_host, scratch.stream) : y_status;
  if (uv_status == 0) uv_status = cuda_stream_sync(scratch.stream);
  if (uv_status != 0) {
    gst_buffer_unmap(job.buffer, &map); *reason = "cuda-crop-copy"; return false;
  }
  if (side == output_side) {
    // The two historical sources have a 1920x1080 mux, so the crop is already
    // the verifier tensor size; only copy the pinned bytes into GstBuffer data.
    std::memcpy(cropped->data(), scratch.host, cropped->size());
  } else {
    const guint8 *source_y = scratch.host;
    const guint8 *source_uv = scratch.host + source_y_bytes;
    const int input_side = side;
    auto sample_plane = [](const guint8 *input, int input_width, int input_height,
                           int input_stride, guint8 *output, int output_width, int output_height,
                           int output_stride, int channels) {
      for (int oy = 0; oy < output_height; ++oy) {
        const float fy = std::max(0.0f, std::min(static_cast<float>(input_height - 1),
            (oy + .5f) * input_height / output_height - .5f));
        const int y0 = static_cast<int>(fy), y1 = std::min(input_height - 1, y0 + 1);
        const float wy = fy - y0;
        for (int ox = 0; ox < output_width; ++ox) {
          const float fx = std::max(0.0f, std::min(static_cast<float>(input_width - 1),
              (ox + .5f) * input_width / output_width - .5f));
          const int x0 = static_cast<int>(fx), x1 = std::min(input_width - 1, x0 + 1);
          const float wx = fx - x0;
          for (int c = 0; c < channels; ++c) {
            const float a = input[y0 * input_stride + x0 * channels + c];
            const float b = input[y0 * input_stride + x1 * channels + c];
            const float d = input[y1 * input_stride + x0 * channels + c];
            const float e = input[y1 * input_stride + x1 * channels + c];
            output[oy * output_stride + ox * channels + c] = static_cast<guint8>(std::lround(
                (a * (1.0f - wx) + b * wx) * (1.0f - wy) +
                (d * (1.0f - wx) + e * wx) * wy));
          }
        }
      }
  };
    sample_plane(source_y, input_side, input_side, input_side, cropped->data(),
        output_side, output_side, output_side, 1);
    sample_plane(source_uv, input_side / 2, input_side / 2, input_side,
        cropped->data() + output_y_bytes, output_side / 2, output_side / 2,
        output_side, 2);
  }
  // Optional forensic capture of the exact NV12 payload sent to appsrc. This
  // is disabled in normal operation and may be restricted to one camera,
  // frame, and native target by environment variables in an isolated replay.
  if (const char *dump_dir = g_getenv("V11_CONTEXT_VERIFIER_DUMP_DIR")) {
    auto env_matches = [](const char *name, guint64 actual) {
      const char *raw = g_getenv(name);
      if (!raw || !*raw) return true;
      char *end = nullptr;
      const guint64 expected = g_ascii_strtoull(raw, &end, 10);
      return end && *end == '\0' && expected == actual;
    };
    if (*dump_dir && env_matches("V11_CONTEXT_VERIFIER_DUMP_CAMERA", request->key.source_id + 1) &&
        env_matches("V11_CONTEXT_VERIFIER_DUMP_FRAME", static_cast<guint64>(request->key.frame_num)) &&
        env_matches("V11_CONTEXT_VERIFIER_DUMP_NATIVE_ID", request->candidate.native_track_id)) {
      g_mkdir_with_parents(dump_dir, 0750);
      gchar *filename = g_strdup_printf("CAM-%02u-frame-%d-native-%lu.nv12",
          request->key.source_id + 1, request->key.frame_num,
          request->candidate.native_track_id);
      gchar *path = g_build_filename(dump_dir, filename, nullptr);
      GError *dump_error = nullptr;
      if (!g_file_set_contents(path, reinterpret_cast<const gchar*>(cropped->data()),
              static_cast<gssize>(cropped->size()), &dump_error)) {
        g_printerr("CONTEXT_VERIFIER_CROP_DUMP_FAIL path=%s reason=%s\n", path,
            dump_error ? dump_error->message : "unknown");
      } else {
        g_print("CONTEXT_VERIFIER_CROP_DUMP path=%s bytes=%zu camera=%u frame=%d native_id=%lu\n",
            path, cropped->size(), request->key.source_id + 1, request->key.frame_num,
            request->candidate.native_track_id);
      }
      if (dump_error) g_error_free(dump_error);
      g_free(path);
      g_free(filename);
    }
  }
  gst_buffer_unmap(job.buffer, &map);

  const float local_scale = output_side / static_cast<float>(side);
  const float original_to_surface_x = scale_x;
  const float original_to_surface_y = scale_y;
  request->expected_left = ((request->candidate.left * original_to_surface_x) - crop_x) * local_scale;
  request->expected_top = ((request->candidate.top * original_to_surface_y) - crop_y) * local_scale;
  request->expected_width = request->candidate.width * original_to_surface_x * local_scale;
  request->expected_height = request->candidate.height * original_to_surface_y * local_scale;
  request->crop_ready = true;
  *reason = "ok";
  return true;
}

// Prepare the exact context crop entirely in NVMM/CUDA memory. The older
// diagnostic path copies the crop GPU->CPU, repacks it, then converts it back
// to NVMM; this path performs crop/resize on the GPU and wraps the resulting
// NvBufSurface directly for appsrc. Geometry and candidate matching remain
// identical to lowconf_prepare_nv12_crop.
static NvBufSurface *lowconf_prepare_gpu_nvmm_crop_surface(const LowConfVerifierJob &job,
    LowConfVerifierRequest *request, std::string *reason) {
  GstMapInfo map = GST_MAP_INFO_INIT;
  if (!gst_buffer_map(job.buffer, &map, GST_MAP_READ)) {
    *reason = "buffer-map"; return nullptr;
  }
  auto *batch = reinterpret_cast<NvBufSurface*>(map.data);
  if (!batch || job.batch_id >= batch->batchSize || !batch->surfaceList) {
    gst_buffer_unmap(job.buffer, &map); *reason = "batch-index"; return nullptr;
  }
  const auto &surface = batch->surfaceList[job.batch_id];
  const guint width = surface.width, height = surface.height;
  if (!lowconf_nv12_format(surface.colorFormat) || surface.layout != NVBUF_LAYOUT_PITCH ||
      !surface.dataPtr || width < 640 || height < 640 || width % 2 || height % 2) {
    gst_buffer_unmap(job.buffer, &map); *reason = "unsupported-source-surface"; return nullptr;
  }
  if (!request->frame_width || !request->frame_height) {
    gst_buffer_unmap(job.buffer, &map); *reason = "missing-frame-dimensions"; return nullptr;
  }

  constexpr guint verifier_input_side = 640;
  const auto crop = lowconf_compute_verifier_crop(width, height,
      request->frame_width, request->frame_height,
      ShadowContextRect{request->candidate.left, request->candidate.top,
          request->candidate.width, request->candidate.height}, verifier_input_side);
  if (!crop.valid) {
    gst_buffer_unmap(job.buffer, &map); *reason = "context-too-small-or-invalid-geometry"; return nullptr;
  }
  const int side = crop.side;
  const int crop_x = crop.left;
  const int crop_y = crop.top;
  const float scale_x = crop.surface_per_frame_x;
  const float scale_y = crop.surface_per_frame_y;
  request->transform = {static_cast<float>(crop_x), static_cast<float>(crop_y),
      static_cast<float>(side), static_cast<float>(verifier_input_side), scale_x, scale_y};

  NvBufSurface *output = lowconf_gpu_surface_acquire(job.source_id);
  if (!output || !output->surfaceList || !output->surfaceList[0].dataPtr) {
    if (output) lowconf_gpu_surface_release(job.source_id, output);
    gst_buffer_unmap(job.buffer, &map); *reason = "nvmm-pool-empty"; return nullptr;
  }
  output->numFilled = 1;
  const bool native_context = lowconf_verifier_context_backend_enabled();
  const guint output_side = native_context ? lowconf_context_input_side.load() : 640u;

  // At the accepted 1920x1080 mux, context width is exactly 640. Copy the
  // current NV12 ROI device-to-device with no resize or host round trip.
  // Other source geometries retain the GPU transform path below.
  bool crop_ready = false;
  if (!native_context && side == 640) {
    using CudaSetDeviceFn = int (*)(int);
    using CudaStreamCreateFn = int (*)(void**, unsigned int);
    using CudaStreamSyncFn = int (*)(void*);
    using CudaMemcpy2DAsyncFn = int (*)(void*, size_t, const void*, size_t,
        size_t, size_t, int, void*);
    using CudaStreamDestroyFn = int (*)(void*);
    static void *cuda_runtime = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
    static auto cuda_set_device = reinterpret_cast<CudaSetDeviceFn>(
        cuda_runtime ? dlsym(cuda_runtime, "cudaSetDevice") : nullptr);
    static auto cuda_stream_create = reinterpret_cast<CudaStreamCreateFn>(
        cuda_runtime ? dlsym(cuda_runtime, "cudaStreamCreateWithFlags") : nullptr);
    static auto cuda_stream_sync = reinterpret_cast<CudaStreamSyncFn>(
        cuda_runtime ? dlsym(cuda_runtime, "cudaStreamSynchronize") : nullptr);
    static auto cuda_memcpy_2d_async = reinterpret_cast<CudaMemcpy2DAsyncFn>(
        cuda_runtime ? dlsym(cuda_runtime, "cudaMemcpy2DAsync") : nullptr);
    static auto cuda_stream_destroy = reinterpret_cast<CudaStreamDestroyFn>(
        cuda_runtime ? dlsym(cuda_runtime, "cudaStreamDestroy") : nullptr);
    auto &scratch = lowconf_cuda_scratch;
    if (scratch.gpu_id != static_cast<int>(batch->gpuId)) {
      if (scratch.stream && cuda_stream_destroy) cuda_stream_destroy(scratch.stream);
      scratch.stream = nullptr;
      scratch.gpu_id = static_cast<int>(batch->gpuId);
    }
    if (cuda_set_device && cuda_stream_create && cuda_stream_sync &&
        cuda_memcpy_2d_async && cuda_set_device(static_cast<int>(batch->gpuId)) == 0) {
      if (!scratch.stream && cuda_stream_create(&scratch.stream, 1u) != 0)
        scratch.stream = nullptr;
      if (scratch.stream) {
        const auto &dst = output->surfaceList[0];
        const auto *src = static_cast<const guint8*>(surface.dataPtr);
        auto *dst_bytes = static_cast<guint8*>(dst.dataPtr);
        constexpr int cuda_device_to_device = 3;
        const int y_status = cuda_memcpy_2d_async(
            dst_bytes, dst.planeParams.pitch[0],
            src + static_cast<size_t>(crop_y) * surface.planeParams.pitch[0] + crop_x,
            surface.planeParams.pitch[0], 640, 640, cuda_device_to_device, scratch.stream);
        const int uv_status = y_status == 0 ? cuda_memcpy_2d_async(
            dst_bytes + dst.planeParams.offset[1], dst.planeParams.pitch[1],
            src + surface.planeParams.offset[1] +
                static_cast<size_t>(crop_y / 2) * surface.planeParams.pitch[1] + crop_x,
            surface.planeParams.pitch[1], 640, 320, cuda_device_to_device, scratch.stream)
            : y_status;
        crop_ready = uv_status == 0 && cuda_stream_sync(scratch.stream) == 0;
      }
    }
  }

  // The transform API works on one surface per entry. Keep the decoded batch
  // object alive via job.buffer while exposing only its selected source slot.
  NvBufSurface source_view = *batch;
  source_view.surfaceList = const_cast<NvBufSurfaceParams*>(&surface);
  source_view.batchSize = 1;
  source_view.numFilled = 1;
  NvBufSurfTransformRect source_rect{};
  source_rect.top = static_cast<guint>(crop_y);
  source_rect.left = static_cast<guint>(crop_x);
  source_rect.width = static_cast<guint>(side);
  source_rect.height = static_cast<guint>(side);
  NvBufSurfTransformRect destination_rect{};
  destination_rect.width = output_side;
  destination_rect.height = output_side;
  NvBufSurfTransform_Error transform_status = NvBufSurfTransformError_Success;
  if (!crop_ready) {
  NvBufSurfTransformParams transform{};
  transform.transform_flag = NVBUFSURF_TRANSFORM_CROP_SRC |
      NVBUFSURF_TRANSFORM_CROP_DST | NVBUFSURF_TRANSFORM_FILTER;
  transform.transform_flip = NvBufSurfTransform_None;
  transform.transform_filter = NvBufSurfTransformInter_Bilinear;
  transform.src_rect = &source_rect;
  transform.dst_rect = &destination_rect;
  NvBufSurfTransformConfigParams config{};
  config.compute_mode = NvBufSurfTransformCompute_GPU;
  config.gpu_id = static_cast<gint32>(batch->gpuId);
  config.cuda_stream = lowconf_context_per_camera_trt_enabled() &&
          job.source_id < lowconf_context_preprocess_lanes.size()
      ? static_cast<cudaStream_t>(lowconf_context_preprocess_lanes[job.source_id].stream)
      : nullptr;
  {
    // Session parameters are process-global in DeepStream 9.1. Keep the
    // set+enqueue pair atomic; each source still owns its CUDA stream and all
    // subsequent preprocessing/inference work remains private to that lane.
    std::lock_guard<std::mutex> transform_guard(lowconf_nvbuf_transform_session_mutex);
    transform_status = NvBufSurfTransformSetSessionParams(&config);
    if (transform_status == NvBufSurfTransformError_Success)
      transform_status = NvBufSurfTransform(&source_view, output, &transform);
  }
  }
  gst_buffer_unmap(job.buffer, &map);
  if (!crop_ready && transform_status != NvBufSurfTransformError_Success) {
    lowconf_gpu_surface_release(job.source_id, output);
    *reason = "nvbufsurftransform";
    return nullptr;
  }

  const float local_scale = 640.0f / static_cast<float>(side);
  request->expected_left = ((request->candidate.left * scale_x) - crop_x) * local_scale;
  request->expected_top = ((request->candidate.top * scale_y) - crop_y) * local_scale;
  request->expected_width = request->candidate.width * scale_x * local_scale;
  request->expected_height = request->candidate.height * scale_y * local_scale;
  request->crop_ready = true;
  request->crop_completed_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  *reason = "ok";
  return output;
}

static GstBuffer *lowconf_prepare_gpu_nvmm_crop(const LowConfVerifierJob &job,
    LowConfVerifierRequest *request, std::string *reason) {
  NvBufSurface *output = lowconf_prepare_gpu_nvmm_crop_surface(job, request, reason);
  if (!output) return nullptr;
  auto *lease = new LowConfGpuSurfaceLease{job.source_id, output};
  GstBuffer *buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY,
      output, sizeof(NvBufSurface), 0, sizeof(NvBufSurface), lease,
      lowconf_release_gpu_surface_lease);
  if (!buffer) {
    lowconf_gpu_surface_release(job.source_id, output);
    delete lease;
    *reason = "nvmm-wrapped-buffer";
    return nullptr;
  }
  return buffer;
}

static void lowconf_verifier_worker_main(guint source_id) {
  for (;;) {
    LowConfVerifierJob job;
    {
      std::unique_lock<std::mutex> guard(lowconf_mutex);
      lowconf_condition.wait(guard, [source_id] {
        return lowconf_worker_stop.load() || !lowconf_jobs[source_id].empty();
      });
      if (lowconf_worker_stop.load() && lowconf_jobs[source_id].empty()) break;
      job = lowconf_jobs[source_id].front();
      lowconf_jobs[source_id].pop_front();
    }
    LowConfVerifierRequest request;
    bool present = false;
    {
      std::lock_guard<std::mutex> guard(lowconf_mutex);
      const auto found = lowconf_requests.find(job.serial);
      if (found != lowconf_requests.end()) { request = found->second; present = true; }
    }
    if (present) {
      request.worker_started_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
      if (lowconf_verifier_context_backend_enabled()) {
        std::string reason;
        NvBufSurface *surface = lowconf_prepare_gpu_nvmm_crop_surface(
            job, &request, &reason);
        if (surface) {
          {
            std::lock_guard<std::mutex> guard(lowconf_mutex);
            auto found = lowconf_requests.find(job.serial);
            if (found != lowconf_requests.end()) {
              found->second.expected_left = request.expected_left;
              found->second.expected_top = request.expected_top;
              found->second.expected_width = request.expected_width;
              found->second.expected_height = request.expected_height;
              found->second.transform = request.transform;
              found->second.crop_ready = true;
              found->second.worker_started_ns = request.worker_started_ns;
              found->second.crop_completed_ns = request.crop_completed_ns;
            }
          }
          if (lowconf_context_per_camera_trt_enabled()) {
            lowconf_context_infer_camera_job({source_id, job.serial, surface});
          } else {
            std::lock_guard<std::mutex> guard(lowconf_context_queue_mutex);
            if (lowconf_context_queue.size() >= 32) {
              lowconf_gpu_surface_release(source_id, surface);
              lowconf_fail_request(job.serial, "CONTEXT_QUEUE_FULL");
            } else {
              lowconf_context_queue.push_back({source_id, job.serial, surface});
              guint64 observed = lowconf_context_queue_high_water.load();
              while (lowconf_context_queue.size() > observed &&
                  !lowconf_context_queue_high_water.compare_exchange_weak(
                      observed, lowconf_context_queue.size())) {}
              lowconf_context_queue_condition.notify_one();
            }
          }
        } else {
          ++lowconf_crop_failures;
          lowconf_fail_request(job.serial, reason.c_str());
        }
        if (job.buffer) gst_buffer_unref(job.buffer);
        continue;
      }
      std::vector<guint8> crop;
      std::string reason;
      GstBuffer *out = nullptr;
      if (lowconf_verifier_gpu_crop_enabled()) {
        out = lowconf_prepare_gpu_nvmm_crop(job, &request, &reason);
      } else if (lowconf_prepare_nv12_crop(job, &crop, &request, &reason)) {
        out = gst_buffer_new_allocate(nullptr, crop.size(), nullptr);
        if (!out || gst_buffer_fill(out, 0, crop.data(), crop.size()) != crop.size()) {
          if (out) gst_buffer_unref(out);
          out = nullptr;
          reason = "gst-buffer-allocation";
        }
      }
      if (out) {
        request.crop_completed_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
        const guint64 synthetic_pts = request.verifier_pts;
        {
          std::lock_guard<std::mutex> guard(lowconf_mutex);
          auto found = lowconf_requests.find(job.serial);
          if (found != lowconf_requests.end()) {
            found->second.expected_left = request.expected_left;
            found->second.expected_top = request.expected_top;
            found->second.expected_width = request.expected_width;
            found->second.expected_height = request.expected_height;
            found->second.transform = request.transform;
            found->second.crop_ready = true;
            found->second.worker_started_ns = request.worker_started_ns;
            found->second.crop_completed_ns = request.crop_completed_ns;
          }
        }
        GST_BUFFER_PTS(out) = synthetic_pts;
        GST_BUFFER_DTS(out) = GST_CLOCK_TIME_NONE;
        GST_BUFFER_DURATION(out) = GST_MSECOND;
        GST_BUFFER_OFFSET(out) = job.serial;
        const GstFlowReturn flow = gst_app_src_push_buffer(
            GST_APP_SRC(lowconf_appsrc[source_id]), out);
        const guint64 push_completed_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
        {
          std::lock_guard<std::mutex> guard(lowconf_mutex);
          auto found = lowconf_requests.find(job.serial);
          if (found != lowconf_requests.end()) found->second.push_completed_ns = push_completed_ns;
        }
        if (flow != GST_FLOW_OK) {
          std::lock_guard<std::mutex> guard(lowconf_mutex);
          lowconf_pts_to_serial.erase(synthetic_pts);
          auto found = lowconf_requests.find(job.serial);
          if (found != lowconf_requests.end()) {
            auto failed = found->second;
            found->second.complete = true;
            found->second.candidate.verifier_done = true;
            found->second.candidate.verifier_confirmed = false;
            lowconf_queue_active_decision_locked(found->second);
            lowconf_log_result(failed, {}, 0, 0, 0, 0, "APP_SRC_REJECTED", static_cast<guint64>(g_get_monotonic_time()) * 1000);
            if (found->second.consumed) lowconf_requests.erase(found);
          }
          lowconf_condition.notify_all();
        }
      }
      if (reason != "ok") {
        ++lowconf_crop_failures;
        std::lock_guard<std::mutex> guard(lowconf_mutex);
        lowconf_pts_to_serial.erase(request.verifier_pts);
        auto found = lowconf_requests.find(job.serial);
        if (found != lowconf_requests.end()) {
          auto failed = found->second;
          found->second.complete = true;
          found->second.candidate.verifier_done = true;
          found->second.candidate.verifier_confirmed = false;
          lowconf_queue_active_decision_locked(found->second);
          lowconf_log_result(failed, {}, 0, 0, 0, 0, reason.c_str(), static_cast<guint64>(g_get_monotonic_time()) * 1000);
          if (found->second.consumed) lowconf_requests.erase(found);
        }
        lowconf_condition.notify_all();
      }
    }
    if (job.buffer) gst_buffer_unref(job.buffer);
  }
  if (lowconf_appsrc[source_id] && !lowconf_eos_sent[source_id].exchange(true)) {
    const GstFlowReturn flow = gst_app_src_end_of_stream(
        GST_APP_SRC(lowconf_appsrc[source_id]));
    g_print("LOWCONF_VERIFIER_INPUT_EOS source=%u flow=%s\n", source_id,
        gst_flow_get_name(flow));
  }
}

static void lowconf_verifier_finish_input() {
  lowconf_accepting = false;
  lowconf_worker_stop = true;
  lowconf_condition.notify_all();
}

static void lowconf_verifier_start(GstElement *active_pipeline) {
  if (!context_verifier_enabled()) return;
  const char *config = g_getenv("V11_LOWCONF_VERIFIER_CONFIG");
  if (!config || !*config) config = "/config/peoplenet_lowconf_verifier.txt";
  if (!g_file_test(config, G_FILE_TEST_IS_REGULAR)) {
    g_printerr("LOWCONF_VERIFIER_START_FAIL missing_config=%s\n", config);
    std::exit(2);
  }
  const guint verifier_batch = lowconf_verifier_batch_size();
  if (!verifier_batch) {
    g_printerr("LOWCONF_VERIFIER_START_FAIL invalid_batch_size\n");
    std::exit(2);
  }
  gchar *config_text = nullptr;
  gsize config_length = 0;
  if (!g_file_get_contents(config, &config_text, &config_length, nullptr)) {
    g_printerr("LOWCONF_VERIFIER_START_FAIL unreadable_config=%s\n", config);
    std::exit(2);
  }
  gchar *batch_line = g_strdup_printf("batch-size=%u", verifier_batch);
  const bool batch_matches = g_strstr_len(config_text, static_cast<gssize>(config_length), batch_line) != nullptr;
  g_free(batch_line);
  g_free(config_text);
  if (!batch_matches) {
    g_printerr("LOWCONF_VERIFIER_START_FAIL config_batch_size_mismatch config=%s expected=%u\n",
        config, verifier_batch);
    std::exit(2);
  }
  const char *capture_context = g_getenv("V11_CONTEXT_VERIFIER_EVIDENCE");
  if (verbose_evidence_enabled() || (capture_context && std::strcmp(capture_context, "1") == 0)) {
    lowconf_evidence = std::fopen("/work/lowconf-verifier.jsonl", "w");
    lowconf_assignment_evidence = std::fopen("/work/lowconf-assignments.jsonl", "w");
    if (!lowconf_evidence || !lowconf_assignment_evidence) {
      g_printerr("LOWCONF_VERIFIER_START_FAIL cannot_write_evidence\n");
      std::exit(2);
    }
    std::setvbuf(lowconf_evidence, nullptr, _IOFBF, 256 * 1024);
    std::setvbuf(lowconf_assignment_evidence, nullptr, _IOFBF, 64 * 1024);
  }

  const bool native_context = lowconf_verifier_context_backend_enabled();
  if (native_context && !lowconf_verifier_gpu_crop_enabled()) {
    g_printerr("LOWCONF_VERIFIER_START_FAIL native_context_requires_gpu_crop\n");
    std::exit(2);
  }
  if (!native_context) {
  const guint group_count = (static_cast<guint>(lowconf_appsrc.size()) + verifier_batch - 1) /
      verifier_batch;
  for (guint group = 0; group < group_count; ++group) {
    gchar *mux_name = g_strdup_printf("lowconf-crop-mux-%u", group);
    gchar *infer_name = g_strdup_printf("context-crop-verifier-%u", group);
    gchar *sink_name = g_strdup_printf("lowconf-verifier-sink-%u", group);
    GstElement *mux = lowconf_make_element("nvstreammux", mux_name);
    GstElement *infer = lowconf_make_element("nvinfer", infer_name);
    GstElement *sink = lowconf_make_element("fakesink", sink_name);
    g_free(mux_name);
    g_free(infer_name);
    g_free(sink_name);
    g_object_set(mux, "batch-size", verifier_batch, "width", 640u, "height", 640u,
        "live-source", TRUE, "batched-push-timeout", 1000u, "sync-inputs", FALSE,
        "gpu-id", 0u, nullptr);
    g_object_set(infer, "config-file-path", config, nullptr);
    g_object_set(sink, "sync", FALSE, "async", FALSE, nullptr);

    for (guint slot = 0; slot < verifier_batch; ++slot) {
      const guint source_id = group * verifier_batch + slot;
      if (source_id >= lowconf_appsrc.size()) break;
      gchar *source_name = g_strdup_printf("lowconf-crop-appsrc-%u", source_id);
      gchar *queue_name = g_strdup_printf("lowconf-crop-queue-%u", source_id);
      gchar *convert_name = g_strdup_printf("lowconf-crop-nvconvert-%u", source_id);
      gchar *caps_name = g_strdup_printf("lowconf-crop-nvmm-caps-%u", source_id);
      GstElement *source = lowconf_make_element("appsrc", source_name);
      GstElement *queue = lowconf_make_element("queue", queue_name);
      g_free(source_name);
      g_free(queue_name);
      const bool gpu_crop = lowconf_verifier_gpu_crop_enabled();
      GstElement *convert = gpu_crop ? nullptr : lowconf_make_element(
          "nvvideoconvert", convert_name);
      GstElement *capsfilter = gpu_crop ? nullptr : lowconf_make_element(
          "capsfilter", caps_name);
      g_free(convert_name);
      g_free(caps_name);
      lowconf_appsrc[source_id] = source;

      GstCaps *input_caps = gst_caps_from_string(gpu_crop
          ? "video/x-raw(memory:NVMM),format=NV12,width=640,height=640,framerate=20/1"
          : "video/x-raw,format=NV12,width=640,height=640,framerate=20/1");
      gst_app_src_set_caps(GST_APP_SRC(source), input_caps);
      gst_caps_unref(input_caps);
      gst_app_src_set_stream_type(GST_APP_SRC(source), GST_APP_STREAM_TYPE_STREAM);
      // One-frame queues preserve bounded latest work while each source has its
      // own worker. Paired appsrcs share only the downstream inference context.
      gst_app_src_set_max_bytes(GST_APP_SRC(source), 640u * 640u * 3u / 2u);
      g_object_set(source, "is-live", TRUE, "format", GST_FORMAT_TIME,
          "block", TRUE, "do-timestamp", FALSE, "emit-signals", FALSE, nullptr);
      g_object_set(queue, "max-size-buffers", 1u, "max-size-bytes", 0u,
          "max-size-time", guint64(0), "leaky", 0, "silent", TRUE, nullptr);
      if (!gpu_crop) {
        GstCaps *nvmm_caps = gst_caps_from_string(
            "video/x-raw(memory:NVMM),format=NV12,width=640,height=640,framerate=20/1");
        g_object_set(capsfilter, "caps", nvmm_caps, nullptr);
        gst_caps_unref(nvmm_caps);
      }
      const gboolean linked = gpu_crop
          ? gst_element_link(source, queue)
          : gst_element_link_many(source, queue, convert, capsfilter, nullptr);
      if (!linked) {
        g_printerr("LOWCONF_VERIFIER_START_FAIL appsrc_video_link source=%u\n", source_id);
        std::exit(2);
      }
      GstPad *src_pad = gst_element_get_static_pad(gpu_crop ? queue : capsfilter, "src");
      gchar *mux_pad_name = g_strdup_printf("sink_%u", slot);
      GstPad *mux_pad = gst_element_request_pad_simple(mux, mux_pad_name);
      g_free(mux_pad_name);
      const GstPadLinkReturn pad_link = src_pad && mux_pad
          ? gst_pad_link(src_pad, mux_pad) : GST_PAD_LINK_REFUSED;
      if (src_pad) gst_object_unref(src_pad);
      if (mux_pad) gst_object_unref(mux_pad);
      if (pad_link != GST_PAD_LINK_OK) {
        g_printerr("LOWCONF_VERIFIER_START_FAIL mux_input_link source=%u result=%d\n",
            source_id, pad_link);
        std::exit(2);
      }
    }
    if (!gst_element_link_many(mux, infer, sink, nullptr)) {
      g_printerr("LOWCONF_VERIFIER_START_FAIL mux_infer_link group=%u\n", group);
      std::exit(2);
    }
    GstPad *infer_src = gst_element_get_static_pad(infer, "src");
    if (!infer_src) {
      g_printerr("LOWCONF_VERIFIER_START_FAIL infer_src_pad group=%u\n", group);
      std::exit(2);
    }
    gst_pad_add_probe(infer_src, GST_PAD_PROBE_TYPE_BUFFER,
        lowconf_infer_exit, nullptr, nullptr);
    gst_object_unref(infer_src);
  }
  }

  if (native_context && !lowconf_verifier_contexts_initialize(config)) {
    g_printerr("LOWCONF_VERIFIER_START_FAIL native_context_init\n");
    std::exit(2);
  }
  if (lowconf_verifier_gpu_crop_enabled() && !lowconf_gpu_surface_pools_initialize(0)) {
    g_printerr("LOWCONF_VERIFIER_START_FAIL gpu_crop_surface_pool_init\n");
    std::exit(2);
  }
  lowconf_accepting = true;
  lowconf_worker_stop = false;
  lowconf_context_worker_stop = false;
  lowconf_context_queue.clear();
  lowconf_publish_stop = false;
  lowconf_publish_accepting = true;
  if (native_context && !lowconf_context_per_camera_trt_enabled())
    lowconf_context_worker = std::thread(lowconf_context_worker_main);
  for (guint source_id = 0; source_id < lowconf_workers.size(); ++source_id)
    lowconf_workers[source_id] = std::thread(lowconf_verifier_worker_main, source_id);
  for (guint source_id = 0; source_id < lowconf_publish_workers.size(); ++source_id)
    lowconf_publish_workers[source_id] = std::thread(lowconf_verifier_publish_worker_main, source_id);
  if (native_context) {
    g_print("CONTEXT_VERIFIER_BACKEND per-camera-crop[6]->%s; exact-frame; crop_policy=%s; low-confidence sidecar excluded from NvDCF\n",
        lowconf_context_per_camera_trt_enabled()
            ? "six-persistent-TensorRT-context-and-CUDA-stream-lanes"
            : "bounded-ready-queue->persistent-NvDsInferContext[1,batch<=6]-750us-opportunistic-batch",
        lowconf_candidate_centered_crop_enabled()
            ? "candidate-centered-zoom-v1" : "legacy-scene-context-v1");
  } else {
    const guint group_count = (static_cast<guint>(lowconf_appsrc.size()) + verifier_batch - 1) /
        verifier_batch;
    g_print("CONTEXT_VERIFIER_GRAPH per-camera-appsrc[6]->bounded-queue->%s->nvstreammux(groups=%u,batch=%u)->nvinfer config=%s name=%s; independent camera queues; no RTSP, no NvDCF input\n",
        lowconf_verifier_gpu_crop_enabled() ? "NVMM-GPU-crop" : "nvvideoconvert",
        group_count, verifier_batch, config,
        g_getenv("V11_CONTEXT_VERIFIER_NAME") ? g_getenv("V11_CONTEXT_VERIFIER_NAME") : "configured-nvinfer");
  }
  (void)active_pipeline;
}

static LowConfSubmitResult lowconf_verifier_submit(GstBuffer *buffer, guint batch_id,
    const LowConfFrameKey &key, guint frame_width, guint frame_height,
    const LowConfCandidate &candidate, bool active_birth, bool shadow_refresh) {
  const bool confidence_allowed = shadow_refresh
      ? (candidate.confidence >= 0.0f && candidate.confidence <= 1.0f)
      : active_birth ? (candidate.confidence >= .25f && candidate.confidence <= 1.0f)
      : (candidate.confidence >= .23f && candidate.confidence < .25f);
  if (!context_verifier_enabled() ||
      (active_birth && !active_track_verifier_enabled()) ||
      (shadow_refresh && (!active_track_verifier_enabled() ||
          !shadow_context_refresh_enabled())) ||
      (!active_birth && !shadow_refresh && !lowconf_verifier_enabled()))
    return LowConfSubmitResult::Disabled;
  if (!buffer || key.source_id >= 6 || !confidence_allowed ||
      !std::isfinite(candidate.confidence) || frame_width == 0 || frame_height == 0)
    return LowConfSubmitResult::Invalid;
  const guint64 serial = ++lowconf_next_serial;
  const guint64 pts = serial * GST_MSECOND;
  LowConfVerifierRequest request;
  request.key = key;
  request.candidate = candidate;
  request.candidate.active_track_birth = active_birth;
  request.candidate.shadow_context_refresh = shadow_refresh;
  request.candidate.shadow_context_pretracker_scheduled = shadow_refresh;
  request.candidate.source_frame_num = key.frame_num;
  request.candidate.source_pts_ns = key.pts_ns;
  request.frame_width = frame_width;
  request.frame_height = frame_height;
  request.candidate.submitted_monotonic_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  request.verifier_pts = pts;
  std::lock_guard<std::mutex> guard(lowconf_mutex);
  if (!lowconf_accepting.load() ||
      lowconf_jobs[key.source_id].size() >= LOWCONF_JOB_QUEUE_LIMIT_PER_CAMERA ||
      lowconf_requests.size() >= LOWCONF_REQUEST_LIMIT) {
    ++lowconf_queue_drops;
    return LowConfSubmitResult::Backpressure;
  }
  lowconf_requests.emplace(serial, request);
  lowconf_pts_to_serial.emplace(pts, serial);
  lowconf_frame_requests[key].push_back(serial);
  lowconf_jobs[key.source_id].push_back({gst_buffer_ref(buffer), batch_id, key.source_id, serial});
  ++lowconf_submitted;
  if (shadow_refresh) ++shadow_context_submitted;
  lowconf_condition.notify_all();
  return LowConfSubmitResult::Submitted;
}

static std::vector<LowConfCandidate> lowconf_verifier_frame_candidates(
    const LowConfFrameKey &key) {
  std::vector<LowConfCandidate> result;
  if (!context_verifier_enabled()) return result;
  std::lock_guard<std::mutex> guard(lowconf_mutex);
  const auto frame_it = lowconf_frame_requests.find(key);
  if (frame_it == lowconf_frame_requests.end()) return result;
  result.reserve(frame_it->second.size());
  for (guint64 serial : frame_it->second) {
    const auto request_it = lowconf_requests.find(serial);
    if (request_it != lowconf_requests.end()) result.push_back(request_it->second.candidate);
  }
  return result;
}

// Attach the native ID to a verifier request only after NvDCF has associated
// this exact detector row. A result that completes early can then be handed to
// the tracker ledger; a later result enqueues the same decision on completion.
static bool lowconf_verifier_assign_active_track(const LowConfFrameKey &key,
    gint detector_index, guint64 native_track_id) {
  if (!context_verifier_enabled() || key.source_id >= 6 || detector_index < 0) return false;
  ++lowconf_active_assign_calls;
  std::lock_guard<std::mutex> guard(lowconf_mutex);
  const auto frame_it = lowconf_frame_requests.find(key);
  if (frame_it == lowconf_frame_requests.end()) {
    ++lowconf_active_assign_misses;
    if (lowconf_assignment_evidence)
      std::fprintf(lowconf_assignment_evidence,
          "{\"camera_id\":\"CAM-%02u\",\"frame_num\":%d,\"pts_ns\":%lu,"
          "\"event\":\"assign\",\"track_id\":%lu,\"tracker_detection_index\":%d,"
          "\"result\":\"NO_FRAME_REQUESTS\"}\n",
          key.source_id + 1, key.frame_num, key.pts_ns, native_track_id, detector_index);
    return false;
  }
  bool assigned = false;
  guint active_birth_requests = 0;
  guint matched_requests = 0;
  gint requested_indices[8]{};
  for (guint64 serial : frame_it->second) {
    const auto request_it = lowconf_requests.find(serial);
    if (request_it == lowconf_requests.end()) continue;
    auto &request = request_it->second;
    if (!request.candidate.active_track_birth) continue;
    if (active_birth_requests < G_N_ELEMENTS(requested_indices))
      requested_indices[active_birth_requests] = request.candidate.detector_index;
    ++active_birth_requests;
    if (request.candidate.detector_index != detector_index) continue;
    request.candidate.native_track_id = native_track_id;
    request.candidate.active_track_id_assigned = true;
    assigned = true;
    ++matched_requests;
    if (request.complete && !request.active_decision_queued) {
      lowconf_queue_active_decision_locked(request);
      request.active_decision_queued = true;
    }
  }
  if (assigned) ++lowconf_active_assign_hits;
  else ++lowconf_active_assign_misses;
  if (lowconf_assignment_evidence) {
    std::fprintf(lowconf_assignment_evidence,
        "{\"camera_id\":\"CAM-%02u\",\"frame_num\":%d,\"pts_ns\":%lu,"
        "\"event\":\"assign\",\"track_id\":%lu,\"tracker_detection_index\":%d,"
        "\"active_birth_requests\":%u,\"matched_requests\":%u,\"request_indices\":[",
        key.source_id + 1, key.frame_num, key.pts_ns, native_track_id, detector_index,
        active_birth_requests, matched_requests);
    const guint logged = std::min<guint>(active_birth_requests, G_N_ELEMENTS(requested_indices));
    for (guint i = 0; i < logged; ++i) {
      if (i) std::fputc(',', lowconf_assignment_evidence);
      std::fprintf(lowconf_assignment_evidence, "%d", requested_indices[i]);
    }
    std::fprintf(lowconf_assignment_evidence, "],\"result\":\"%s\"}\n",
        assigned ? "ASSIGNED" : "INDEX_MISMATCH");
  }
  lowconf_condition.notify_all();
  return assigned;
}

static std::vector<ActiveTrackVerificationDecision>
lowconf_verifier_take_completed_active(guint source_id) {
  if (!context_verifier_enabled() || source_id >= lowconf_completed_active_decisions.size()) return {};
  std::lock_guard<std::mutex> guard(lowconf_mutex);
  auto &queue = lowconf_completed_active_decisions[source_id];
  std::vector<ActiveTrackVerificationDecision> result;
  result.reserve(queue.size());
  while (!queue.empty()) {
    result.push_back(queue.front());
    queue.pop_front();
  }
  return result;
}

static std::vector<LowConfCandidate> lowconf_verifier_take_frame(const LowConfFrameKey &key) {
  std::vector<LowConfCandidate> result;
  if (!context_verifier_enabled()) return result;
  std::lock_guard<std::mutex> guard(lowconf_mutex);
  const auto frame_it = lowconf_frame_requests.find(key);
  if (frame_it == lowconf_frame_requests.end()) return result;
  for (guint64 serial : frame_it->second) {
    const auto request_it = lowconf_requests.find(serial);
    if (request_it == lowconf_requests.end()) continue;
    auto &request = request_it->second;
    if (request.complete) {
      LowConfCandidate completed = request.candidate;
      const bool within_deadline = completed.verifier_ready_monotonic_ns == 0 ||
          (completed.verifier_ready_monotonic_ns >= completed.submitted_monotonic_ns &&
           completed.verifier_ready_monotonic_ns - completed.submitted_monotonic_ns <=
               lowconf_verifier_result_deadline_ns(completed));
      if (!within_deadline) {
        completed.verifier_done = false;
        ++lowconf_late;
      }
      result.push_back(completed);
      lowconf_pts_to_serial.erase(request.verifier_pts);
      lowconf_requests.erase(request_it);
    } else {
      LowConfCandidate late = request.candidate;
      late.verifier_done = false;
      result.push_back(late);
      request.consumed = true;
      ++lowconf_late;
    }
  }
  lowconf_frame_requests.erase(frame_it);
  return result;
}

static bool lowconf_request_needs_exact_frame_result(const LowConfCandidate &candidate) {
  return (candidate.active_track_birth && candidate.active_track_id_assigned) ||
      candidate.shadow_context_refresh || (!candidate.active_track_birth &&
      candidate.confidence >= .23f && candidate.confidence < .25f);
}

static bool lowconf_frame_complete_locked(const LowConfFrameKey &key) {
  const auto frame_it = lowconf_frame_requests.find(key);
  if (frame_it == lowconf_frame_requests.end()) return true;
  for (guint64 serial : frame_it->second) {
    const auto request_it = lowconf_requests.find(serial);
    if (request_it != lowconf_requests.end() &&
        lowconf_request_needs_exact_frame_result(request_it->second.candidate) &&
        !request_it->second.complete) return false;
  }
  return true;
}

// Wait only on the per-camera canonical publisher, never on a DeepStream
// streaming callback. The deadline is measured from each exact-frame request;
// late verifier results are rejected by take_frame even if they arrive while
// another candidate from the same frame is still inside its deadline.
static std::vector<LowConfCandidate> lowconf_verifier_take_frame_wait(
    const LowConfFrameKey &key) {
  if (!context_verifier_enabled()) return {};
  {
    std::unique_lock<std::mutex> guard(lowconf_mutex);
    guint64 latest_deadline_ns = 0;
    const auto frame_it = lowconf_frame_requests.find(key);
    if (frame_it != lowconf_frame_requests.end()) {
      for (guint64 serial : frame_it->second) {
        const auto request_it = lowconf_requests.find(serial);
        if (request_it == lowconf_requests.end() ||
            !lowconf_request_needs_exact_frame_result(request_it->second.candidate)) continue;
        latest_deadline_ns = std::max(latest_deadline_ns,
            request_it->second.candidate.submitted_monotonic_ns +
                lowconf_verifier_result_deadline_ns(request_it->second.candidate));
      }
    }
    while (latest_deadline_ns && !lowconf_frame_complete_locked(key)) {
      const guint64 now_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
      if (now_ns >= latest_deadline_ns) break;
      lowconf_condition.wait_for(guard, std::chrono::nanoseconds(latest_deadline_ns - now_ns));
    }
  }
  auto result = lowconf_verifier_take_frame(key);
  for (const auto &candidate : result)
    if (!candidate.verifier_done) ++lowconf_deadline_expired;
  return result;
}

static void lowconf_verifier_publish_worker_main(guint source_id) {
  for (;;) {
    LowConfDeferredFrame frame;
    {
      std::unique_lock<std::mutex> guard(lowconf_publish_mutex);
      lowconf_publish_condition.wait(guard, [source_id] {
        return lowconf_publish_stop.load() || !lowconf_publish_queues[source_id].empty();
      });
      if (lowconf_publish_stop.load() && lowconf_publish_queues[source_id].empty()) break;
      frame = std::move(lowconf_publish_queues[source_id].front());
      lowconf_publish_queues[source_id].pop_front();
    }
    auto verified = frame.wait_for_verifier
        ? lowconf_verifier_take_frame_wait(frame.key)
        : lowconf_verifier_take_frame(frame.key);
    if (frame.publish) frame.publish(std::move(verified));
  }
}

static bool lowconf_verifier_defer_frame(const LowConfFrameKey &key,
    std::function<void(std::vector<LowConfCandidate>)> publish,
    bool wait_for_verifier) {
  if (!context_verifier_enabled() || key.source_id >= lowconf_publish_queues.size() || !publish ||
      !lowconf_publish_accepting.load()) return false;
  std::lock_guard<std::mutex> guard(lowconf_publish_mutex);
  auto &queue = lowconf_publish_queues[key.source_id];
  if (queue.size() >= LOWCONF_PUBLISH_QUEUE_LIMIT) {
    ++lowconf_publish_queue_full;
    return false;
  }
  queue.push_back({key, wait_for_verifier, std::move(publish)});
  guint64 observed = lowconf_publish_queue_high_water.load();
  while (queue.size() > observed && !lowconf_publish_queue_high_water.compare_exchange_weak(
      observed, queue.size())) {}
  if (wait_for_verifier) ++lowconf_selective_wait_frames;
  lowconf_publish_condition.notify_all();
  return true;
}

static void lowconf_verifier_stats() {
  if (!context_verifier_enabled()) return;
  const guint64 completed = lowconf_inferred.load();
  size_t pending = 0, queued = 0;
  { std::lock_guard<std::mutex> guard(lowconf_mutex);
    pending = lowconf_requests.size();
    for (const auto &queue : lowconf_jobs) queued += queue.size(); }
  size_t publish_pending = 0;
  { std::lock_guard<std::mutex> guard(lowconf_publish_mutex);
    for (const auto &queue : lowconf_publish_queues) publish_pending += queue.size(); }
  g_print("LOWCONF_VERIFIER submitted=%lu queue_drops=%lu queued=%zu crop_failures=%lu inferred=%lu confirmed=%lu rejected=%lu late_at_tracker=%lu deadline_expired=%lu unmatched_results=%lu inference_mean_ms=%.3f inference_max_ms=%.3f selective_wait_frames=%lu publisher_queue_high_water=%lu publisher_queue_full=%lu publisher_pending=%zu verifier_pending=%zu active_assign_calls=%lu active_assign_hits=%lu active_assign_misses=%lu\n",
      lowconf_submitted.load(), lowconf_queue_drops.load(), queued,
      lowconf_crop_failures.load(),
      completed, lowconf_confirmed.load(), lowconf_rejected.load(), lowconf_late.load(),
      lowconf_deadline_expired.load(), lowconf_unmatched_result.load(),
      completed ? lowconf_infer_total_us.load() / 1000.0 / completed : 0.0,
      lowconf_infer_max_us.load() / 1000.0, lowconf_selective_wait_frames.load(),
      lowconf_publish_queue_high_water.load(), lowconf_publish_queue_full.load(),
      publish_pending, pending, lowconf_active_assign_calls.load(),
      lowconf_active_assign_hits.load(), lowconf_active_assign_misses.load());
  if (lowconf_evidence) std::fflush(lowconf_evidence);
  if (lowconf_assignment_evidence) std::fflush(lowconf_assignment_evidence);
}

static void lowconf_verifier_shutdown() {
  if (!context_verifier_enabled()) return;
  lowconf_verifier_finish_input();
  for (auto &worker : lowconf_workers)
    if (worker.joinable()) worker.join();
  if (lowconf_context_worker.joinable()) {
    lowconf_context_worker_stop = true;
    lowconf_context_queue_condition.notify_all();
    lowconf_context_worker.join();
  }
  lowconf_publish_accepting = false;
  lowconf_publish_stop = true;
  lowconf_publish_condition.notify_all();
  for (auto &worker : lowconf_publish_workers)
    if (worker.joinable()) worker.join();
  for (auto &handle : lowconf_context_handles) {
    if (handle) NvDsInferContext_Destroy(handle);
    handle = nullptr;
  }
  if (lowconf_verifier_direct_trt_enabled())
    lowconf_context_trt_shutdown();
  if (lowconf_context_preprocess_api.destroy) {
    for (auto &lane : lowconf_context_preprocess_lanes) {
      if (lane.destroy && (lane.stream || lane.device_buffer))
        lane.destroy(lane.stream, lane.device_buffer);
      lane = {};
    }
    lowconf_context_preprocess_api.destroy(lowconf_context_preprocess_api.stream,
        lowconf_context_preprocess_api.device_buffer);
    lowconf_context_preprocess_api.stream = nullptr;
    lowconf_context_preprocess_api.device_buffer = nullptr;
    if (lowconf_context_preprocess_api.library) dlclose(lowconf_context_preprocess_api.library);
    lowconf_context_preprocess_api = {};
  }
  {
    std::lock_guard<std::mutex> guard(lowconf_mutex);
    for (auto &queue : lowconf_jobs) {
      for (auto &job : queue) if (job.buffer) gst_buffer_unref(job.buffer);
      queue.clear();
    }
    lowconf_requests.clear();
    lowconf_pts_to_serial.clear();
    lowconf_frame_requests.clear();
    for (auto &queue : lowconf_completed_active_decisions) queue.clear();
  }
  if (lowconf_verifier_gpu_crop_enabled()) {
    for (guint source_id = 0; source_id < lowconf_gpu_surface_pools.size(); ++source_id) {
      auto &pool = lowconf_gpu_surface_pools[source_id];
      std::lock_guard<std::mutex> guard(pool.mutex);
      if (pool.free.size() != pool.all.size()) {
        g_printerr("LOWCONF_GPU_SURFACE_POOL shutdown_incomplete source=%u free=%zu total=%zu; retaining borrowed surfaces until process exit\n",
            source_id, pool.free.size(), pool.all.size());
        continue;
      }
      for (NvBufSurface *surface : pool.all) NvBufSurfaceDestroy(surface);
      pool.all.clear();
      pool.free.clear();
    }
  }
  if (lowconf_evidence) { std::fclose(lowconf_evidence); lowconf_evidence = nullptr; }
  if (lowconf_assignment_evidence) {
    std::fclose(lowconf_assignment_evidence);
    lowconf_assignment_evidence = nullptr;
  }
}
