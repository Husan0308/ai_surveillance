// CAM-01 + CAM-02 + CAM-03 + CAM-04 + CAM-05 + CAM-06 DeepStream 9.1 validation. No inference, tracker, OpenCV or CPU pixel copies.
#include <gst/gst.h>
#include <glib-unix.h>
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "camera_source_plan.hpp"
#include "credential_redaction.hpp"
#include <sys/resource.h>
#include <unistd.h>
#ifdef STAGE_CAPTURE
#include <gstnvdsmeta.h>
#include <nvbufsurface.h>
#include "nv12_capture.hpp"
#include <memory>
#include <array>
#include <condition_variable>
#include <dlfcn.h>
#include <fcntl.h>
#include <mutex>
#include <thread>
#include <sys/stat.h>
#endif

#ifdef STAGE_CAPTURE
enum class StageCaptureKind { SOURCE_DECODED, PRE_NVENC };

struct StageCaptureContext {
  StageCaptureKind kind = StageCaptureKind::SOURCE_DECODED;
  int source_id = -1;
  int mux_pad_index = -1;
  std::string camera_id;
  std::string raw_format;
  std::string caps_format;
  guint width = 0;
  guint height = 0;
  guint caps_width = 0;
  guint caps_height = 0;
  bool probe_attached = false;
  guint64 pad_buffer_sequence = 0;
  GstSegment segment{};
  bool have_segment = false;
  std::array<std::atomic<int>, 3> point_state{};  // pending, queued, dropped, missed, written, error
  StageCaptureContext() {
    for (auto &state : point_state) state.store(0, std::memory_order_relaxed);
  }
};

struct StageCaptureJob {
  StageCaptureContext *context = nullptr;
  int target_index = -1;
  GstBuffer *buffer = nullptr;
  guint64 buffer_pts_ns = GST_CLOCK_TIME_NONE;
  guint64 running_time_ns = GST_CLOCK_TIME_NONE;
  gint64 elapsed_us = 0;
  guint64 pad_buffer_sequence = 0;
  guint64 buffer_offset = GST_BUFFER_OFFSET_NONE;
  guint64 buffer_offset_end = GST_BUFFER_OFFSET_NONE;

};
#endif

struct Counter {
  std::atomic<guint64> frames{0};
  std::atomic<guint64> pts{GST_CLOCK_TIME_NONE};
  std::atomic<guint64> backwards{0};
  std::atomic<guint64> duplicates{0};
  std::atomic<gint64> last_us{0};
  std::atomic<gint64> max_gap_us{0};
};

struct SourceCtx {
  std::string id;
  int source_id = 0;
  int mux_pad_index = 0;
  std::string uri;
  std::string username;
  std::string password;
  int latency_ms = 300;
  GstElement *source = nullptr;
  GstElement *queue = nullptr;
  GstElement *jitterbuffer = nullptr;
  GMutex jitter_lock;
  Counter ingress;
  Counter input;
  std::atomic_bool decoder_seen{false};
  guint errors = 0;
  guint warnings = 0;
  guint64 previous_ingress_frames = 0;
  guint64 previous_frames = 0;
#ifdef STAGE_CAPTURE
  StageCaptureContext stage_capture;
#endif
};

static GstElement *pipeline = nullptr;
static GstElement *mux = nullptr;
static GstElement *encoder = nullptr;
static GstElement *sink_element = nullptr;
static GMainLoop *loop = nullptr;
static std::vector<SourceCtx*> sources;
static Counter output_counter;
static std::atomic_bool fatal{false};
static bool stopping = false;
static guint shared_errors = 0, shared_warnings = 0;
static gint64 start_us = 0, previous_us = 0, last_message_us = 0;
static guint64 previous_output = 0;
static double previous_cpu = 0.0;
static int duration_sec = 60;
static bool preview_enabled = false;
static int preview_port = 5600;
#ifdef STAGE_CAPTURE
static bool stage_capture_enabled = false;
static StageCaptureContext pre_nvenc_capture;
static class StageCapture *stage_capture = nullptr;
static void *nvbufsurface_handle = nullptr;
static decltype(&NvBufSurfaceCreate) capture_surface_create = nullptr;
static decltype(&NvBufSurfaceCopy) capture_surface_copy = nullptr;
static decltype(&NvBufSurfaceDestroy) capture_surface_destroy = nullptr;
// CUDA runtime C ABI: cudaError_t is returned as an int on pinned x86_64.
// Dynamically resolved so compile-only containers need no driver injection.
static void *capture_cuda_handle = nullptr;
static int (*capture_cuda_set_device)(int) = nullptr;
static int (*capture_cuda_synchronize)() = nullptr;
struct CaptureSurfaceInfo {
  stage_nv12::Layout layout;
  guint memory_type = 0;
  guint surface_layout = 0;
};
#endif

static std::string interrupt_camera = "none";
static int interrupt_at = 20;
static int interrupt_seconds = 12;
static bool interruption_started = false;
static bool interruption_resumed = false;
static bool interruption_reported = false;
static guint64 target_start_frames = 0, peer_start_frames = 0, target_resume_frames = 0, peer_resume_frames = 0;

#ifdef STAGE_CAPTURE
static void event(SourceCtx *ctx, const char *kind, const std::string &detail);
static constexpr int STAGE_CAPTURE_TARGET_SEC[3] = {5, 30, 55};
static constexpr gint64 STAGE_CAPTURE_TOLERANCE_US = 2000000;
// One six-camera sample group plus the mosaic fits without retaining an
// unbounded number of decoder/mux surfaces if the disk worker falls behind.
static constexpr size_t STAGE_CAPTURE_QUEUE_CAPACITY = 8;
static constexpr const char *STAGE_CAPTURE_DIRECTORY = "/work/stage-captures";

static const char *stage_capture_name(StageCaptureKind kind) {
  return kind == StageCaptureKind::SOURCE_DECODED ? "source_decoded" : "pre_nvenc";
}

static bool write_all(int fd, const guint8 *data, size_t size) {
  size_t offset = 0;
  while (offset < size) {
    const ssize_t n = ::write(fd, data + offset, size - offset);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    offset += static_cast<size_t>(n);
  }
  return true;
}

class StageCapture {
 public:
  bool start(const std::vector<StageCaptureContext*> &contexts) {
    contexts_ = contexts;
    if (::mkdir(STAGE_CAPTURE_DIRECTORY, 0700) != 0) return false;
    if (::chmod(STAGE_CAPTURE_DIRECTORY, 0700) != 0) return false;
    const std::string manifest_path = std::string(STAGE_CAPTURE_DIRECTORY) + "/stage-frames.jsonl";
    manifest_fd_ = ::open(manifest_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (manifest_fd_ < 0) return false;
    const std::string header =
        "{\"kind\":\"header\",\"schema\":2,\"clock\":\"monotonic_elapsed_us\","
        "\"targets_sec\":[5,30,55],\"queue_capacity\":" +
        std::to_string(STAGE_CAPTURE_QUEUE_CAPACITY) +
        ",\"format\":\"packed_nv12_raw\",\"packed_layout\":\"NV12_Y_UV_tight\","
        "\"active_sources\":" + active_sources_json() +
        ",\"buffer_offset_note\":\"GstBuffer offsets are not assumed to be frame numbers\"}\n";
    if (!write_all(manifest_fd_, reinterpret_cast<const guint8*>(header.data()), header.size())) {
      ::close(manifest_fd_);
      manifest_fd_ = -1;
      return false;
    }
    try {
      worker_ = std::thread(&StageCapture::run, this);
    } catch (...) {
      ::close(manifest_fd_);
      manifest_fd_ = -1;
      return false;
    }
    return true;
  }

  bool enqueue(StageCaptureJob &job) {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock() || stopping_ || queue_size_ >= STAGE_CAPTURE_QUEUE_CAPACITY) {
      ++queue_dropped_;
      return false;
    }
    job.buffer = gst_buffer_ref(job.buffer);
    jobs_[queue_tail_] = job;
    queue_tail_ = (queue_tail_ + 1) % STAGE_CAPTURE_QUEUE_CAPACITY;
    ++queue_size_;
    lock.unlock();
    ready_.notify_one();
    return true;
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (manifest_fd_ >= 0) {
      ::fsync(manifest_fd_);
      ::close(manifest_fd_);
      manifest_fd_ = -1;
    }
  }

  guint64 queue_dropped() const { return queue_dropped_.load(); }

 private:
  std::string active_sources_json() const {
    std::string json = "[";
    bool first = true;
    for (const auto *ctx : contexts_) {
      if (!ctx || ctx->kind != StageCaptureKind::SOURCE_DECODED) continue;
      if (!first) json += ",";
      first = false;
      json += "{\"camera_id\":\"" + ctx->camera_id + "\",\"source_id\":" +
              std::to_string(ctx->source_id) + ",\"mux_pad_index\":" +
              std::to_string(ctx->mux_pad_index) + "}";
    }
    json += "]";
    return json;
  }

  void write_line(const std::string &line) {
    if (manifest_fd_ >= 0) {
      write_all(manifest_fd_, reinterpret_cast<const guint8*>(line.data()), line.size());
    }
  }

  static bool is_nv12(NvBufSurfaceColorFormat format) {
    return format == NVBUF_COLOR_FORMAT_NV12 || format == NVBUF_COLOR_FORMAT_NV12_ER ||
           format == NVBUF_COLOR_FORMAT_NV12_709 || format == NVBUF_COLOR_FORMAT_NV12_709_ER ||
           format == NVBUF_COLOR_FORMAT_NV12_2020;
  }

  static stage_nv12::Layout describe(const NvBufSurfaceParams &p) {
    stage_nv12::Layout out;
    out.width = p.width;
    out.height = p.height;
    out.allocation = p.dataSize;
    out.plane_count = p.planeParams.num_planes;
    out.pitch_linear = p.layout == NVBUF_LAYOUT_PITCH;
    for (guint i = 0; i < 2; ++i) {
      out.planes[i] = {p.planeParams.width[i], p.planeParams.height[i],
                       p.planeParams.pitch[i], p.planeParams.bytesPerPix[i],
                       p.planeParams.offset[i], p.planeParams.psize[i]};
    }
    return out;
  }

  bool copy_nv12(GstBuffer *buffer, std::vector<guint8> &packed,
                 guint &width, guint &height, guint &format_code,
                 std::array<guint, 2> &plane_width,
                 std::array<guint, 2> &plane_height,
                 CaptureSurfaceInfo &info, std::string &error) {
    GstMapInfo map = GST_MAP_INFO_INIT;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
      error = "GstBuffer descriptor map failed";
      return false;
    }
    struct Unmap {
      GstBuffer *buffer; GstMapInfo *map;
      ~Unmap() { gst_buffer_unmap(buffer, map); }
    } unmap{buffer, &map};
    auto fail = [&](const char *reason) { error = reason; return false; };
    if (map.size < sizeof(NvBufSurface) || !map.data ||
        reinterpret_cast<std::uintptr_t>(map.data) % alignof(NvBufSurface))
      return fail("short or unaligned NvBufSurface descriptor");
    auto *source = reinterpret_cast<NvBufSurface*>(map.data);
    if (source->batchSize != 1 || source->numFilled != 1 || !source->surfaceList)
      return fail("capture requires exactly one valid surface");
    // GstBuffer mapping exposes the descriptor only, not CPU-addressable pixels.
    // NvBufSurfaceMap does not support CUDA_DEVICE or CUDA_PINNED on dGPU.
    // Use an SDK cross-memory copy to a private CUDA pinned host
    // allocation. Never dereference the source dataPtr on the CPU.
    if (source->memType != NVBUF_MEM_CUDA_DEVICE &&
        source->memType != NVBUF_MEM_CUDA_UNIFIED &&
        source->memType != NVBUF_MEM_CUDA_PINNED)
      return fail("unsupported or implicit NVMM memory type");
    const auto &p = source->surfaceList[0];
    width = p.width; height = p.height;
    format_code = static_cast<guint>(p.colorFormat);
    info.memory_type = static_cast<guint>(source->memType);
    info.surface_layout = static_cast<guint>(p.layout);
    info.layout = describe(p);
    for (guint i = 0; i < 2; ++i) {
      plane_width[i] = p.planeParams.width[i];
      plane_height[i] = p.planeParams.height[i];
    }
    if (!is_nv12(p.colorFormat) || p.layout != NVBUF_LAYOUT_PITCH ||
        p.planeParams.num_planes != 2 || p.pitch != p.planeParams.pitch[0] || !p.dataPtr)
      return fail("unsupported format, layout, plane count, or source pointer");
    std::size_t total = 0;
    if (!stage_nv12::validate(info.layout, total, error)) return false;
    if (source->gpuId > static_cast<guint>(G_MAXINT) ||
        capture_cuda_set_device(static_cast<int>(source->gpuId)) != 0 ||
        capture_cuda_synchronize() != 0)
      return fail("CUDA device selection or pre-copy completion barrier failed");
    NvBufSurfaceCreateParams create{};
    create.gpuId = source->gpuId;
    create.width = width; create.height = height;
    create.colorFormat = p.colorFormat;
    create.layout = NVBUF_LAYOUT_PITCH;
    create.memType = NVBUF_MEM_CUDA_PINNED;
    NvBufSurface *host_raw = nullptr;
    if (capture_surface_create(&host_raw, 1, &create) != 0 || !host_raw) {
      if (host_raw) capture_surface_destroy(host_raw);
      return fail("pinned host surface allocation failed");
    }
    const auto release_host = [](NvBufSurface *surface) {
      if (capture_surface_destroy(surface) != 0)
        event(nullptr, "STAGE_CAPTURE", "pinned host surface destruction failed");
    };
    std::unique_ptr<NvBufSurface, decltype(release_host)> host(host_raw, release_host);
    if (host->batchSize != 1 || !host->surfaceList ||
        host->memType != NVBUF_MEM_CUDA_PINNED)
      return fail("unexpected host allocation type or batch");
    auto &h = host->surfaceList[0];
    auto host_layout = describe(h);
    std::size_t host_total = 0;
    if (h.layout != NVBUF_LAYOUT_PITCH || h.colorFormat != p.colorFormat ||
        h.planeParams.num_planes != 2 || h.pitch != h.planeParams.pitch[0] || !h.dataPtr ||
        !stage_nv12::validate(host_layout, host_total, error))
      return fail("invalid pinned host allocation metadata");
    // The SDK requires matching buffer and batch sizes. Reject differing
    // allocation geometry instead of asking it to copy into a smaller buffer.
    if (host_total != total || host_layout.allocation != info.layout.allocation)
      return fail("SDK source/host allocation sizes differ");
    for (guint i = 0; i < 2; ++i) {
      const auto &a = info.layout.planes[i]; const auto &b = host_layout.planes[i];
      if (a.pitch != b.pitch || a.offset != b.offset || a.allocation != b.allocation)
        return fail("SDK source/host plane allocations differ");
    }
    const auto allocated_host_layout = host_layout;
    const auto *allocated_host_pointer = h.dataPtr;
    const int copy_result = capture_surface_copy(source, host.get());
    const int sync_result = capture_cuda_synchronize();
    if (copy_result != 0 || sync_result != 0)
      return fail("NvBufSurfaceCopy or post-copy CUDA completion barrier failed");
    // Explicit device synchronization completed. Pinned host dataPtr is
    // CPU-accessible; SyncForCpu is documented for SURFACE_ARRAY/HANDLE only,
    // which this dGPU adapter explicitly rejects. No mappedAddr is assumed.
    host_layout = describe(h);
    if (host->memType != NVBUF_MEM_CUDA_PINNED || host->batchSize != 1 ||
        h.layout != NVBUF_LAYOUT_PITCH || h.colorFormat != p.colorFormat ||
        h.planeParams.num_planes != 2 || h.pitch != h.planeParams.pitch[0] || !h.dataPtr ||
        h.dataPtr != allocated_host_pointer ||
        host_layout.allocation != allocated_host_layout.allocation ||
        !stage_nv12::validate(host_layout, host_total, error) || host_total != total)
      return fail("host metadata changed or became invalid after SDK copy");
    for (guint i = 0; i < 2; ++i) {
      const auto &a = host_layout.planes[i]; const auto &b = allocated_host_layout.planes[i];
      if (a.pitch != b.pitch || a.offset != b.offset || a.allocation != b.allocation)
        return fail("host plane allocations changed after SDK copy");
    }
    packed.resize(total);
    return stage_nv12::pack(host_layout, static_cast<const guint8*>(h.dataPtr),
                           host_layout.allocation, packed.data(), packed.size(), error);
  }

  void process(StageCaptureJob &job) {
    std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)>
        owned_buffer(job.buffer, gst_buffer_unref);
    StageCaptureContext *ctx = job.context;
    const int target_sec = STAGE_CAPTURE_TARGET_SEC[job.target_index];
    guint width = 0, height = 0, format_code = 0;
    std::array<guint, 2> plane_width{};
    std::array<guint, 2> plane_height{};
    CaptureSurfaceInfo surface_info;
    std::string copy_error;
    std::vector<guint8> packed;
    const bool copied = copy_nv12(job.buffer, packed, width, height, format_code,
                                  plane_width, plane_height, surface_info, copy_error);
    std::string filename;
    bool saved = false;
    if (copied) {
      char name[96];
      if (ctx->kind == StageCaptureKind::SOURCE_DECODED)
        g_snprintf(name, sizeof(name), "source-%s-t%02ds.nv12", ctx->camera_id.c_str(), target_sec);
      else
        g_snprintf(name, sizeof(name), "pre-nvenc-t%02ds.nv12", target_sec);
      filename = name;
      const std::string full_path = std::string(STAGE_CAPTURE_DIRECTORY) + "/" + filename;
      const int fd = ::open(full_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
      saved = fd >= 0 && write_all(fd, packed.data(), packed.size());
      if (fd >= 0) {
        if (::fsync(fd) != 0) saved = false;
        ::close(fd);
      }
      if (!saved) ::unlink(full_path.c_str());
    }

    std::string line = "{\"kind\":\"snapshot\",\"stage\":\"";
    line += stage_capture_name(ctx->kind);
    line += "\",\"target_sec\":" + std::to_string(target_sec);
    line += ",\"status\":\"" + std::string(saved ? "saved" : "copy_failed") + "\"";
    if (ctx->source_id >= 0) {
      char cam[8];
      g_snprintf(cam, sizeof(cam), "CAM-%02d", ctx->source_id + 1);
      line += ",\"camera_id\":\"" + std::string(cam) + "\",\"source_id\":" + std::to_string(ctx->source_id);
      line += ",\"mux_pad_index\":" + std::to_string(ctx->mux_pad_index);
      line += ",\"pad_buffer_sequence\":" + std::to_string(job.pad_buffer_sequence);
    } else {
      line += ",\"camera_id\":null,\"source_id\":null,\"mux_pad_index\":null,\"pad_buffer_sequence\":" +
              std::to_string(job.pad_buffer_sequence);
    }
    line += ",\"buffer_pts_ns\":" + std::to_string(job.buffer_pts_ns);
    line += ",\"buffer_offset\":" + std::to_string(job.buffer_offset);
    line += ",\"buffer_offset_end\":" + std::to_string(job.buffer_offset_end);
    line += ",\"running_time_ns\":" + std::to_string(job.running_time_ns);
    line += ",\"elapsed_monotonic_us\":" + std::to_string(job.elapsed_us);
    line += ",\"width\":" + std::to_string(width) + ",\"height\":" + std::to_string(height);
    line += ",\"caps_format\":\"" + ctx->caps_format + "\"";
    line += ",\"caps_width\":" + std::to_string(ctx->caps_width);
    line += ",\"caps_height\":" + std::to_string(ctx->caps_height);
    line += ",\"copy_error\":\"" + copy_error + "\"";
    line += ",\"packed_layout\":\"NV12_Y_UV_tight\",\"packed_row_bytes\":[" +
            std::to_string(width) + "," + std::to_string(width) + "]";
    line += ",\"memory_type\":" + std::to_string(surface_info.memory_type);
    line += ",\"surface_layout\":" + std::to_string(surface_info.surface_layout);
    line += ",\"surface_allocation_bytes\":" + std::to_string(surface_info.layout.allocation);
    line += ",\"plane_metadata\":[";
    for (guint i = 0; i < 2; ++i) {
      if (i) line += ",";
      const auto &plane = surface_info.layout.planes[i];
      line += "{\"pitch\":" + std::to_string(plane.pitch) +
              ",\"bytes_per_pixel\":" + std::to_string(plane.bytes_per_pixel) +
              ",\"offset\":" + std::to_string(plane.offset) +
              ",\"allocation_bytes\":" + std::to_string(plane.allocation) + "}";
    }
    line += "]";
    line += ",\"format\":\"NV12\",\"format_code\":" + std::to_string(format_code);
    line += ",\"raw_plane_width\":[" + std::to_string(plane_width[0]) + "," +
            std::to_string(plane_width[1]) + "]";
    line += ",\"raw_plane_height\":[" + std::to_string(plane_height[0]) + "," +
            std::to_string(plane_height[1]) + "]";
    line += ",\"filename\":" + std::string(saved ? "\"" + filename + "\"" : "null");
    line += ",\"bytes\":" + std::to_string(saved ? packed.size() : 0);
    line += ",\"frame_identity_available\":false,\"nvds_frames\":[],"
            "\"frame_identity_reason\":\"";
    line += ctx->kind == StageCaptureKind::PRE_NVENC
        ? "post-tiler metadata cannot identify contributing camera frames"
        : "camera and pad sequence are known; original frame index is unavailable";
    line += "\"}\n";
    write_line(line);
    ctx->point_state[job.target_index].store(saved ? 4 : 5, std::memory_order_release);
  }

  void write_final_statuses() {
    for (auto *ctx : contexts_) {
      for (int i = 0; i < 3; ++i) {
        int state = ctx->point_state[i].load(std::memory_order_acquire);
        if (state == 0) {
          state = ctx->probe_attached ? 3 : 5;
          ctx->point_state[i].store(state, std::memory_order_release);
        }
        const char *status = state == 2 ? "queue_dropped" : state == 3 ? "not_reached" :
                             state == 5 ? "capture_error" : "captured";
        if (state == 3 && ctx->probe_attached) {
          const gint64 end_elapsed = (g_get_monotonic_time() - start_us) / 1000000;
          if (end_elapsed >= STAGE_CAPTURE_TARGET_SEC[i]) status = "sample_missed";
        }
        std::string line = "{\"kind\":\"point_status\",\"stage\":\"";
        line += stage_capture_name(ctx->kind);
        line += "\",\"target_sec\":" + std::to_string(STAGE_CAPTURE_TARGET_SEC[i]);
        line += ",\"status\":\"" + std::string(status) + "\"";
        if (ctx->source_id >= 0) {
          char cam[8];
          g_snprintf(cam, sizeof(cam), "CAM-%02d", ctx->source_id + 1);
          line += ",\"camera_id\":\"" + std::string(cam) + "\",\"source_id\":" + std::to_string(ctx->source_id);
        }
        line += "}\n";
        write_line(line);
      }
    }
    const std::string summary = "{\"kind\":\"summary\",\"queue_dropped\":" +
        std::to_string(queue_dropped_.load()) + "}\n";
    write_line(summary);
  }

  void run() {
    for (;;) {
      StageCaptureJob job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || queue_size_ > 0; });
        if (queue_size_ == 0 && stopping_) break;
        job = jobs_[queue_head_];
        jobs_[queue_head_].buffer = nullptr;
        queue_head_ = (queue_head_ + 1) % STAGE_CAPTURE_QUEUE_CAPACITY;
        --queue_size_;
      }
      try {
        process(job);
      } catch (const std::exception &) {
        job.context->point_state[job.target_index].store(5, std::memory_order_release);
        event(nullptr, "STAGE_CAPTURE", "worker processing exception; sample rejected");
      }
    }
    write_final_statuses();
  }

  std::vector<StageCaptureContext*> contexts_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::array<StageCaptureJob, STAGE_CAPTURE_QUEUE_CAPACITY> jobs_{};
  size_t queue_head_ = 0;
  size_t queue_tail_ = 0;
  size_t queue_size_ = 0;
  std::thread worker_;
  bool stopping_ = false;
  int manifest_fd_ = -1;
  std::atomic<guint64> queue_dropped_{0};
};

static GstPadProbeReturn stage_capture_probe(GstPad*, GstPadProbeInfo *info, gpointer data) {
  auto *ctx = static_cast<StageCaptureContext*>(data);
  if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
    GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
    if (event && GST_EVENT_TYPE(event) == GST_EVENT_SEGMENT) {
      const GstSegment *segment = nullptr;
      gst_event_parse_segment(event, &segment);
      if (segment && segment->format == GST_FORMAT_TIME) {
        ctx->segment = *segment;
        ctx->have_segment = true;
      }
    }
    return GST_PAD_PROBE_OK;
  }
  if (!(GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) || !stage_capture_enabled || !stage_capture)
    return GST_PAD_PROBE_OK;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer || start_us <= 0) return GST_PAD_PROBE_OK;
  const guint64 pad_buffer_sequence = ctx->pad_buffer_sequence++;
  const gint64 elapsed_us = g_get_monotonic_time() - start_us;
  if (elapsed_us < 0) return GST_PAD_PROBE_OK;
  int target_index = -1;
  for (int i = 0; i < 3; ++i) {
    if (ctx->point_state[i].load(std::memory_order_relaxed) != 0) continue;
    const gint64 target_us = static_cast<gint64>(STAGE_CAPTURE_TARGET_SEC[i]) * 1000000;
    if (elapsed_us < target_us) continue;
    if (elapsed_us > target_us + STAGE_CAPTURE_TOLERANCE_US) {
      int pending = 0;
      ctx->point_state[i].compare_exchange_strong(pending, 3, std::memory_order_relaxed);
      continue;
    }
    target_index = i;
    break;
  }
  if (target_index < 0) return GST_PAD_PROBE_OK;

  StageCaptureJob job;
  job.context = ctx;
  job.target_index = target_index;
  job.buffer = buffer;
  job.buffer_pts_ns = GST_BUFFER_PTS(buffer);
  job.elapsed_us = elapsed_us;
  job.pad_buffer_sequence = pad_buffer_sequence;
  job.buffer_offset = GST_BUFFER_OFFSET(buffer);
  job.buffer_offset_end = GST_BUFFER_OFFSET_END(buffer);
  if (ctx->have_segment && GST_CLOCK_TIME_IS_VALID(job.buffer_pts_ns)) {
    job.running_time_ns = gst_segment_to_running_time(&ctx->segment, GST_FORMAT_TIME, job.buffer_pts_ns);
  }
  int pending = 0;
  if (!ctx->point_state[target_index].compare_exchange_strong(pending, 1, std::memory_order_relaxed))
    return GST_PAD_PROBE_OK;
  if (!stage_capture->enqueue(job))
    ctx->point_state[target_index].store(2, std::memory_order_release);
  return GST_PAD_PROBE_OK;
}

static bool attach_stage_capture_probe(GstPad *pad, StageCaptureContext *ctx) {
  if (!pad || !ctx) return false;
  const gulong probe_id = gst_pad_add_probe(
      pad, static_cast<GstPadProbeType>(GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM),
      stage_capture_probe, ctx, nullptr);
  ctx->probe_attached = probe_id != 0;
  return ctx->probe_attached;
}

static bool start_stage_capture() {
  if (!stage_capture_enabled) return true;
  nvbufsurface_handle = dlopen("libnvbufsurface.so", RTLD_NOW | RTLD_LOCAL);
  if (!nvbufsurface_handle) {
    event(nullptr, "STAGE_CAPTURE", "libnvbufsurface unavailable; capture disabled");
    return false;
  }
  capture_surface_create = reinterpret_cast<decltype(capture_surface_create)>(
      dlsym(nvbufsurface_handle, "NvBufSurfaceCreate"));
  capture_surface_copy = reinterpret_cast<decltype(capture_surface_copy)>(
      dlsym(nvbufsurface_handle, "NvBufSurfaceCopy"));
  capture_surface_destroy = reinterpret_cast<decltype(capture_surface_destroy)>(
      dlsym(nvbufsurface_handle, "NvBufSurfaceDestroy"));
  if (!capture_surface_create || !capture_surface_copy || !capture_surface_destroy) {
    event(nullptr, "STAGE_CAPTURE", "required NvBufSurface APIs unavailable");
    dlclose(nvbufsurface_handle);
    nvbufsurface_handle = nullptr;
    capture_surface_create = nullptr;
    capture_surface_copy = nullptr;
    capture_surface_destroy = nullptr;
    return false;
  }
  capture_cuda_handle = dlopen("libcudart.so.13", RTLD_NOW | RTLD_LOCAL);
  if (capture_cuda_handle) {
    capture_cuda_set_device = reinterpret_cast<decltype(capture_cuda_set_device)>(
        dlsym(capture_cuda_handle, "cudaSetDevice"));
    capture_cuda_synchronize = reinterpret_cast<decltype(capture_cuda_synchronize)>(
        dlsym(capture_cuda_handle, "cudaDeviceSynchronize"));
  }
  if (!capture_cuda_handle || !capture_cuda_set_device || !capture_cuda_synchronize) {
    event(nullptr, "STAGE_CAPTURE", "required CUDA synchronization APIs unavailable");
    if (capture_cuda_handle) dlclose(capture_cuda_handle);
    capture_cuda_handle = nullptr;
    capture_cuda_set_device = nullptr;
    capture_cuda_synchronize = nullptr;
    dlclose(nvbufsurface_handle);
    nvbufsurface_handle = nullptr;
    capture_surface_create = nullptr;
    capture_surface_copy = nullptr;
    capture_surface_destroy = nullptr;
    return false;
  }
  std::vector<StageCaptureContext*> contexts;
  for (auto *ctx : sources) contexts.push_back(&ctx->stage_capture);
  contexts.push_back(&pre_nvenc_capture);
  stage_capture = new StageCapture();
  if (!stage_capture->start(contexts)) {
    delete stage_capture;
    stage_capture = nullptr;
    dlclose(nvbufsurface_handle);
    nvbufsurface_handle = nullptr;
    capture_surface_create = nullptr;
    capture_surface_copy = nullptr;
    capture_surface_destroy = nullptr;
    dlclose(capture_cuda_handle);
    capture_cuda_handle = nullptr;
    capture_cuda_set_device = nullptr;
    capture_cuda_synchronize = nullptr;
    return false;
  }
  event(nullptr, "STAGE_CAPTURE", "enabled targets_sec=5,30,55 queue_capacity=" +
        std::to_string(STAGE_CAPTURE_QUEUE_CAPACITY) + " format=packed_nv12_raw");
  return true;
}

static void stop_stage_capture() {
  if (!stage_capture) return;
  stage_capture->stop();
  event(nullptr, "STAGE_CAPTURE", "worker_stopped queue_dropped=" + std::to_string(stage_capture->queue_dropped()));
  delete stage_capture;
  stage_capture = nullptr;
  if (nvbufsurface_handle) dlclose(nvbufsurface_handle);
  nvbufsurface_handle = nullptr;
  capture_surface_create = nullptr;
  capture_surface_copy = nullptr;
  capture_surface_destroy = nullptr;
  if (capture_cuda_handle) dlclose(capture_cuda_handle);
  capture_cuda_handle = nullptr;
  capture_cuda_set_device = nullptr;
  capture_cuda_synchronize = nullptr;
}
#endif

static void safe_print(const std::string &prefix, const std::string &detail) {
  std::vector<std::string> credentials;
  for (const auto *s : sources) {
    credentials.push_back(s->username);
    credentials.push_back(s->password);
    credentials.push_back(s->uri);
    if (!s->username.empty() && !s->password.empty())
      credentials.push_back(s->username + ":" + s->password);
  }
  std::string text = credential_redaction::sanitize(prefix + " " + detail, credentials);
  for (auto &c : text) if (c == '\n' || c == '\r') c = ' ';
  g_print("%s\n", text.c_str());
}

static void event(SourceCtx *ctx, const char *kind, const std::string &detail) {
  safe_print((ctx ? ctx->id : std::string("GROUP")) + " " + kind, detail);
}

static void fail(const std::string &detail) {
  event(nullptr, "FATAL", detail);
  fatal = true;
  if (loop) g_main_loop_quit(loop);
}

static GstElement *make(const char *factory, const char *name) {
  GstElement *e = gst_element_factory_make(factory, name);
  if (!e) {
    fail(std::string("missing plugin ") + factory);
    std::exit(2);
  }
  gst_bin_add(GST_BIN(pipeline), e);
  return e;
}

static void link(GstElement *a, GstElement *b) {
  if (!gst_element_link(a, b)) {
    fail(std::string("static link failed: ") + GST_ELEMENT_NAME(a) + " -> " + GST_ELEMENT_NAME(b));
    std::exit(2);
  }
}

static GstPadProbeReturn count_buffer(GstPad *pad, GstPadProbeInfo *info, gpointer data) {
  auto *c = static_cast<Counter*>(data);
  GstBuffer *b = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!b) return GST_PAD_PROBE_OK;
  const gint64 now = g_get_monotonic_time();
  const gint64 last = c->last_us.exchange(now);
  if (last && now - last > c->max_gap_us.load()) c->max_gap_us = now - last;
  const guint64 pts = GST_BUFFER_PTS(b);
  const guint64 before = c->pts.exchange(pts);
  if (pts != GST_CLOCK_TIME_NONE && before != GST_CLOCK_TIME_NONE) {
    if (pts < before) ++c->backwards;
    if (pts == before) ++c->duplicates;
  }
  if (c->frames.fetch_add(1) == 0) {
    GstCaps *caps = gst_pad_get_current_caps(pad);
    gchar *s = caps ? gst_caps_to_string(caps) : g_strdup("unknown");
    safe_print("PAIR FIRST_CAPS", s);
    g_free(s);
    if (caps) gst_caps_unref(caps);
  }
  return GST_PAD_PROBE_OK;
}

static void probe(GstElement *e, const char *pad_name, Counter *c) {
  GstPad *pad = gst_element_get_static_pad(e, pad_name);
  if (!pad) {
    fail(std::string("missing pad ") + pad_name);
    return;
  }
  gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, count_buffer, c, nullptr);
  gst_object_unref(pad);
}

static void configure_rtsp(SourceCtx *ctx, GstElement *e) {
  g_object_set(
      e,
      "user-id", ctx->username.c_str(),
      "user-pw", ctx->password.c_str(),
      "protocols", 4,
      "latency", ctx->latency_ms,
      "drop-on-latency", FALSE,
      "tcp-timeout", guint64(10000000),
      "do-rtsp-keep-alive", TRUE,
      nullptr);
}

static void child_added(GstBin*, GstBin*, GstElement *e, gpointer data) {
  auto *ctx = static_cast<SourceCtx*>(data);
  GstElementFactory *f = gst_element_get_factory(e);
  if (!f) return;
  const char *name = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(f));
  if (!std::strcmp(name, "rtspsrc")) {
    configure_rtsp(ctx, e);
    event(ctx, "RTSP_CHILD", "tcp=1 timeout_sec=10");
  }
  if (!std::strcmp(name, "nvv4l2decoder")) {
    ctx->decoder_seen = true;
    g_object_set(e, "drop-frame-interval", 0u, "skip-frames", 0, nullptr);
    event(ctx, "DECODER", "nvv4l2decoder hardware=1 skip=0 drop=0");
  }
  if (!std::strcmp(name, "rtpjitterbuffer")) {
    g_mutex_lock(&ctx->jitter_lock);
    if (ctx->jitterbuffer) gst_object_unref(ctx->jitterbuffer);
    ctx->jitterbuffer = GST_ELEMENT(gst_object_ref(e));
    g_mutex_unlock(&ctx->jitter_lock);
  }
  if (!std::strcmp(name, "queue")) {
    g_object_set(e, "leaky", 0, "max-size-buffers", 12u, "max-size-bytes", 0u,
                 "max-size-time", guint64(0), nullptr);
  }
}

static void dynamic_pad(GstElement*, GstPad *pad, gpointer data) {
  auto *ctx = static_cast<SourceCtx*>(data);
  GstCaps *caps = gst_pad_get_current_caps(pad);
  if (!caps) caps = gst_pad_query_caps(pad, nullptr);
  if (!caps || gst_caps_is_any(caps) || gst_caps_is_empty(caps)) {
    if (caps) gst_caps_unref(caps);
    event(ctx, "ERROR", "source pad has no usable caps");
    return;
  }
  auto *st = gst_caps_get_structure(caps, 0);
  const char *name = gst_structure_get_name(st);
  if (g_str_has_prefix(name, "video/")) {
    GstPad *dest = gst_element_get_static_pad(ctx->queue, "sink");
    if (!gst_pad_is_linked(dest)) {
      if (gst_pad_link(pad, dest) != GST_PAD_LINK_OK) {
        event(ctx, "ERROR", "dynamic video link failed");
      } else {
        gchar *s = gst_caps_to_string(caps);
        const bool nvmm = gst_caps_features_contains(gst_caps_get_features(caps, 0), "memory:NVMM");
        event(ctx, "SOURCE_LINK", std::string(GST_PAD_NAME(pad)) + " linked " + (nvmm ? "memory:NVMM " : "") + s);
        g_free(s);
      }
    }
    gst_object_unref(dest);
  }
#ifdef STAGE_CAPTURE
  if (stage_capture_enabled && g_str_has_prefix(name, "video/x-raw") &&
      gst_caps_features_contains(gst_caps_get_features(caps, 0), "memory:NVMM")) {
    auto &capture = ctx->stage_capture;
    capture.kind = StageCaptureKind::SOURCE_DECODED;
    capture.source_id = ctx->source_id;
    capture.camera_id = ctx->id;
    const char *format = gst_structure_get_string(st, "format");
    capture.raw_format = format ? format : "unknown";
    gint width = 0, height = 0;
    gst_structure_get_int(st, "width", &width);
    gst_structure_get_int(st, "height", &height);
    capture.width = width > 0 ? static_cast<guint>(width) : 0;
    capture.height = height > 0 ? static_cast<guint>(height) : 0;
    capture.caps_format = capture.raw_format;
    capture.caps_width = capture.width;
    capture.caps_height = capture.height;
    if (!attach_stage_capture_probe(pad, &capture))
      event(ctx, "STAGE_CAPTURE", "source decoded pad probe attachment failed");
  }
#endif
  gst_caps_unref(caps);
}

static GstElement *create_source_element(SourceCtx *ctx) {
  const std::string source_name = ctx->id + "-source";
  GstElement *source = gst_element_factory_make("nvurisrcbin", source_name.c_str());
  if (!source) {
    fail(ctx->id + ": missing nvurisrcbin");
    return nullptr;
  }
  gst_bin_add(GST_BIN(pipeline), source);
  g_signal_connect(source, "deep-element-added", G_CALLBACK(child_added), ctx);
  g_signal_connect(source, "pad-added", G_CALLBACK(dynamic_pad), ctx);
  g_object_set(
      source,
      "uri", ctx->uri.c_str(),
      "gpu-id", 0u,
      "source-id", ctx->source_id,
      "disable-audio", TRUE,
      "select-rtp-protocol", 4,
      "latency", guint(ctx->latency_ms),
      "drop-on-latency", FALSE,
      "num-extra-surfaces", 4u,
      "cudadec-memtype", 0,
      "drop-frame-interval", 0u,
      "dec-skip-frames", 0,
      "low-latency-mode", FALSE,
      "rtsp-reconnect-interval", 5u,
      "init-rtsp-reconnect-interval", 5u,
      "rtsp-reconnect-attempts", -1,
      "max-size-buffers", 12u,
      "leaky", 0,
      "message-forward", TRUE,
      nullptr);
  return source;
}

static SourceCtx *owner_for_message(GstMessage *m) {
  for (auto *ctx : sources) {
    if (m->src == GST_OBJECT(ctx->source) ||
        gst_object_has_as_ancestor(m->src, GST_OBJECT(ctx->source))) return ctx;
  }
  return nullptr;
}

static gboolean bus_message(GstBus*, GstMessage *m, gpointer) {
  const auto type = GST_MESSAGE_TYPE(m);
  if (type == GST_MESSAGE_ERROR || type == GST_MESSAGE_WARNING) {
    const bool error = type == GST_MESSAGE_ERROR;
    GError *e = nullptr;
    gchar *debug = nullptr;
    if (error) gst_message_parse_error(m, &e, &debug);
    else gst_message_parse_warning(m, &e, &debug);
    std::string message = std::string(GST_OBJECT_NAME(m->src)) + ": " + (e ? e->message : "unknown");
    if (debug) message += " | " + std::string(debug);
    SourceCtx *ctx = owner_for_message(m);
    if (ctx) {
      if (error) ++ctx->errors; else ++ctx->warnings;
    } else {
      if (error) ++shared_errors; else ++shared_warnings;
    }
    const gint64 now = g_get_monotonic_time();
    if (now - last_message_us > 2000000 || (ctx ? ctx->errors + ctx->warnings : shared_errors + shared_warnings) < 3) {
      event(ctx, error ? "ERROR" : "WARNING", message);
      last_message_us = now;
    }
    if (error && !ctx) fail("shared mux/output error: " + message);
    if (error && message.find("Unauthorized") != std::string::npos) fail("RTSP authentication rejected");
    if (e) g_error_free(e);
    g_free(debug);
  } else if (type == GST_MESSAGE_EOS && stopping) {
    g_main_loop_quit(loop);
  } else if (type == GST_MESSAGE_EOS && !stopping) {
    fail("unexpected shared pipeline EOS");
  }
  return G_SOURCE_CONTINUE;
}

static double cpu_seconds() {
  rusage r{};
  getrusage(RUSAGE_SELF, &r);
  return r.ru_utime.tv_sec + r.ru_utime.tv_usec / 1e6 +
         r.ru_stime.tv_sec + r.ru_stime.tv_usec / 1e6;
}

static void jitter_stats(SourceCtx *ctx, guint64 &lost, guint64 &late, guint64 &pushed) {
  GstStructure *stats = nullptr;
  g_mutex_lock(&ctx->jitter_lock);
  if (ctx->jitterbuffer) {
    g_object_get(ctx->jitterbuffer, "stats", &stats, nullptr);
    if (stats) {
      gst_structure_get_uint64(stats, "num-lost", &lost);
      gst_structure_get_uint64(stats, "num-late", &late);
      gst_structure_get_uint64(stats, "num-pushed", &pushed);
      gst_structure_free(stats);
    }
  }
  g_mutex_unlock(&ctx->jitter_lock);
}

static SourceCtx *find_source(const std::string &id) {
  for (auto *ctx : sources) if (ctx->id == id) return ctx;
  return nullptr;
}

static guint64 peer_frames_total(SourceCtx *target) {
  guint64 total = 0;
  for (auto *ctx : sources) if (ctx != target) total += ctx->input.frames.load();
  return total;
}

static gboolean resume_interrupted(gpointer data) {
  auto *target = static_cast<SourceCtx*>(data);
  target_resume_frames = target->input.frames.load();
  peer_resume_frames = peer_frames_total(target);

  event(target, "ISOLATION", "recreating only this nvurisrcbin");
  if (target->source) {
    gst_element_set_state(target->source, GST_STATE_NULL);
    if (!gst_bin_remove(GST_BIN(pipeline), target->source)) {
      fail("failed to remove isolated source");
      return G_SOURCE_REMOVE;
    }
    target->source = nullptr;
  }

  g_mutex_lock(&target->jitter_lock);
  if (target->jitterbuffer) {
    gst_object_unref(target->jitterbuffer);
    target->jitterbuffer = nullptr;
  }
  g_mutex_unlock(&target->jitter_lock);

  target->decoder_seen = false;
  target->source = create_source_element(target);
  if (!target->source || !gst_element_sync_state_with_parent(target->source)) {
    fail("failed to recreate isolated source");
  } else {
    interruption_resumed = true;
  }
  return G_SOURCE_REMOVE;
}

#ifdef YOLO26_PERSON
#include "detection.hpp"
static std::atomic_bool pgie_diag_pipeline_started{false};

static void pgie_diag_atexit_export() {
  if (!pgie_diag_enabled.load(std::memory_order_relaxed) ||
      pgie_diag_pipeline_started.load(std::memory_order_acquire)) return;
  // Setup-only exits occur before the pipeline can stream or invoke probes.
  pgie_diag_export_ready.store(true, std::memory_order_release);
  pgie_diag_dump();
}
#endif

static gboolean tick(gpointer) {
  const gint64 now = g_get_monotonic_time();
#ifdef YOLO26_PERSON
  pgie_diag_sample_queues();
#endif
  const double elapsed = (now - start_us) / 1e6;
  const double dt = (now - previous_us) / 1e6;
  const double cpu = cpu_seconds();

  for (auto *ctx : sources) {
    guint q = 0;
    guint64 qtime = 0, lost = 0, late = 0, pushed = 0;
    g_object_get(ctx->queue, "current-level-buffers", &q, "current-level-time", &qtime, nullptr);
    jitter_stats(ctx, lost, late, pushed);
    const guint64 ingress_frames = ctx->ingress.frames.load();
    const guint64 frames = ctx->input.frames.load();
    const double ingress_age = ctx->ingress.last_us
        ? std::max(0.0, (now - ctx->ingress.last_us) / 1e6) : elapsed;
    const double age = ctx->input.last_us
        ? std::max(0.0, (now - ctx->input.last_us) / 1e6) : elapsed;
    g_print(
      "%s STATS elapsed=%.3f ingress=%lu ingress_fps=%.3f ingress_age=%.3f "
      "ingress_max_gap_ms=%.3f input=%lu fps=%.3f age=%.3f pts_ns=%lu "
      "pts_backwards=%lu pts_duplicates=%lu max_gap_ms=%.3f queue=%u queue_ms=%.3f "
      "rtp_lost=%lu rtp_late=%lu rtp_pushed=%lu errors=%u warnings=%u decoder=%d\n",
      ctx->id.c_str(), elapsed,
      ingress_frames, (ingress_frames - ctx->previous_ingress_frames) / dt,
      ingress_age, ctx->ingress.max_gap_us / 1000.0,
      frames, (frames - ctx->previous_frames) / dt, age,
      ctx->input.pts.load(), ctx->input.backwards.load(), ctx->input.duplicates.load(),
      ctx->input.max_gap_us / 1000.0, q, qtime / 1e6, lost, late, pushed,
      ctx->errors, ctx->warnings, int(ctx->decoder_seen.load()));
    ctx->previous_ingress_frames = ingress_frames;
    ctx->previous_frames = frames;
  }

  long pages = 0, resident = 0;
  std::ifstream statm("/proc/self/statm");
  statm >> pages >> resident;
  GstStructure *stats = nullptr;
  guint64 rendered = 0, dropped = 0;
  g_object_get(sink_element, "stats", &stats, nullptr);
  if (stats) {
    gst_structure_get_uint64(stats, "rendered", &rendered);
    gst_structure_get_uint64(stats, "dropped", &dropped);
    gst_structure_free(stats);
  }
  const guint64 out = output_counter.frames.load();
  const double out_age = output_counter.last_us ? std::max(0.0, (now - output_counter.last_us) / 1e6) : elapsed;
  g_print(
    "GROUP STATS elapsed=%.3f output=%lu fps=%.3f age=%.3f pts_ns=%lu "
    "pts_backwards=%lu pts_duplicates=%lu max_gap_ms=%.3f rss_mib=%.3f cpu_pct=%.3f "
    "rendered=%lu dropped=%lu shared_errors=%u shared_warnings=%u\n",
    elapsed, out, (out - previous_output) / dt, out_age, output_counter.pts.load(),
    output_counter.backwards.load(), output_counter.duplicates.load(),
    output_counter.max_gap_us / 1000.0,
    resident * sysconf(_SC_PAGESIZE) / 1048576.0,
    100 * (cpu - previous_cpu) / dt, rendered, dropped, shared_errors, shared_warnings);

#ifdef YOLO26_PERSON
  detection_stats();
#ifdef YOLO26_TRACKER
  tracker_stats();
#endif
#endif
  previous_output = out;
  previous_us = now;
  previous_cpu = cpu;

  if (interrupt_camera != "none" && !interruption_started && elapsed >= interrupt_at) {
    SourceCtx *target = find_source(interrupt_camera);
    if (!target) {
      fail("invalid isolation target");
      return G_SOURCE_REMOVE;
    }
    target_start_frames = target->input.frames.load();
    peer_start_frames = peer_frames_total(target);
    interruption_started = true;
    event(target, "ISOLATION", "setting only this source to NULL");
    gst_element_set_state(target->source, GST_STATE_NULL);
    g_timeout_add_seconds(interrupt_seconds, resume_interrupted, target);
  }

  if (interruption_resumed && !interruption_reported) {
    SourceCtx *target = find_source(interrupt_camera);
    if (target && target->input.frames.load() >= target_resume_frames + 100) {
      const guint64 peer_off_gain = peer_resume_frames - peer_start_frames;
      const guint64 target_off_gain = target_resume_frames - target_start_frames;
      const guint64 target_recovered = target->input.frames.load() - target_resume_frames;
      const bool ok = peer_off_gain >= guint64(interrupt_seconds * 75) && target_off_gain <= 5;
      g_print(
        "GROUP ISOLATION_SUMMARY target=%s healthy_peers=5 outage_sec=%d peer_frames_during=%lu "
        "target_frames_during=%lu target_recovered_frames=%lu status=%s\n",
        target->id.c_str(), interrupt_seconds, peer_off_gain, target_off_gain,
        target_recovered, ok ? "PASS" : "BLOCKED");
      interruption_reported = true;
    }
  }

  for (auto *ctx : sources) {
    const double age = ctx->input.last_us ? std::max(0.0, (now - ctx->input.last_us) / 1e6) : elapsed;
    const bool target_is_off = interruption_started && !interruption_resumed && ctx->id == interrupt_camera;
    if (!target_is_off && age > 90) {
      fail(ctx->id + " has no advancing frames for 90 seconds");
      return G_SOURCE_REMOVE;
    }
  }

  if (out_age > 30) {
    fail("mux/tiled output stalled for more than 30 seconds");
    return G_SOURCE_REMOVE;
  }

  if (elapsed >= duration_sec) {
    stopping = true;
    gst_element_send_event(encoder, gst_event_new_eos());
    g_timeout_add_seconds(5, [](gpointer)->gboolean {
      fail("output EOS did not complete");
      return G_SOURCE_REMOVE;
    }, nullptr);
    return G_SOURCE_REMOVE;
  }
  return G_SOURCE_CONTINUE;
}

static gboolean stop_signal(gpointer) {
  stopping = true;
  if (loop) g_main_loop_quit(loop);
  return G_SOURCE_REMOVE;
}

static std::string b64_secret(GKeyFile *f, const char *group, const char *key) {
  gchar *s = g_key_file_get_string(f, group, key, nullptr);
  if (!s) return "";
  gsize n = 0;
  guchar *raw = g_base64_decode(s, &n);
  std::string v(reinterpret_cast<char*>(raw), n);
  g_free(raw);
  g_free(s);
  return v;
}

static SourceCtx *load_source(GKeyFile *f, const cam_six::SourcePlanEntry &entry) {
  auto *ctx = new SourceCtx();
  ctx->id = entry.camera_id;
  ctx->source_id = entry.logical_source_id;
  ctx->mux_pad_index = entry.mux_pad_index;
#ifdef STAGE_CAPTURE
  ctx->stage_capture.kind = StageCaptureKind::SOURCE_DECODED;
  ctx->stage_capture.source_id = entry.logical_source_id;
  ctx->stage_capture.mux_pad_index = entry.mux_pad_index;
  ctx->stage_capture.camera_id = entry.camera_id;
#endif
  ctx->uri = b64_secret(f, entry.camera_id, "uri");
  ctx->username = b64_secret(f, entry.camera_id, "username");
  ctx->password = b64_secret(f, entry.camera_id, "password");
  ctx->latency_ms = g_key_file_get_integer(f, entry.camera_id, "latency_ms", nullptr);
  g_mutex_init(&ctx->jitter_lock);
  if (ctx->uri.rfind("rtsp://", 0) != 0 || ctx->latency_ms < 100) {
    delete ctx;
    return nullptr;
  }
  return ctx;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    g_printerr("usage: cam-six-validator CONFIG DURATION [--capture-stage-samples] [--only-camera CAM-01..CAM-06]\n");
    return 2;
  }
  std::string only_camera;
  bool only_camera_seen = false;
  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--capture-stage-samples") == 0) {
#ifdef STAGE_CAPTURE
      if (stage_capture_enabled) return 2;
      stage_capture_enabled = true;
#else
      g_printerr("capture option is unavailable in this build\n");
      return 2;
#endif
    } else if (std::strcmp(argv[i], "--only-camera") == 0) {
      if (only_camera_seen || i + 1 >= argc) return 2;
      only_camera = argv[++i];
      only_camera_seen = true;
    } else {
      g_printerr("unknown option\n");
      return 2;
    }
  }
#ifdef YOLO26_PERSON
  if (only_camera_seen) {
    g_printerr("--only-camera is restricted to the capture-only validator\n");
    return 2;
  }
#endif
  if (only_camera_seen && only_camera.empty()) {
    g_printerr("empty camera selection is invalid\n");
    return 2;
  }
  std::vector<cam_six::SourcePlanEntry> source_plan;
  if (!cam_six::build_source_plan(only_camera, source_plan)) {
    g_printerr("invalid camera selection; use CAM-01 through CAM-06\n");
    return 2;
  }
  cam_six::PipelineShape shape{};
  if (!cam_six::pipeline_shape(source_plan.size(), shape)) {
    g_printerr("unsupported active source count\n");
    return 2;
  }
  duration_sec = std::atoi(argv[2]);
  if (duration_sec < 20) return 2;

  GKeyFile *f = g_key_file_new();
  GError *err = nullptr;
  if (!g_key_file_load_from_file(f, argv[1], G_KEY_FILE_NONE, &err)) {
    g_printerr("GROUP config read failed\n");
    if (err) g_error_free(err);
    return 2;
  }

  for (const auto &entry : source_plan) {
    SourceCtx *ctx = load_source(f, entry);
    if (!ctx) {
      g_printerr("selected camera config invalid\n");
      for (auto *loaded : sources) delete loaded;
      g_key_file_free(f);
      return 2;
    }
    sources.push_back(ctx);
  }

  gchar *ival = g_key_file_get_string(f, "validation", "interrupt_camera", nullptr);
  if (ival) { interrupt_camera = ival; g_free(ival); }
  if (interrupt_camera != "none" &&
      std::none_of(sources.begin(), sources.end(), [](const SourceCtx *ctx) {
        return ctx->id == interrupt_camera;
      })) {
    g_printerr("interruption target is not an active source\n");
    for (auto *ctx : sources) delete ctx;
    g_key_file_free(f);
    return 2;
  }
  if (sources.size() == 1 && interrupt_camera != "none") {
    g_printerr("source interruption testing requires all six cameras\n");
    for (auto *ctx : sources) delete ctx;
    g_key_file_free(f);
    return 2;
  }
  interrupt_at = g_key_file_get_integer(f, "validation", "interrupt_at", nullptr);
  interrupt_seconds = g_key_file_get_integer(f, "validation", "interrupt_seconds", nullptr);
  preview_enabled = g_key_file_get_boolean(f, "validation", "preview_enabled", nullptr);
  preview_port = g_key_file_get_integer(f, "validation", "preview_port", nullptr);
  g_key_file_free(f);

  gst_init(nullptr, nullptr);
#ifdef YOLO26_PERSON
  pgie_diag_configure();
  if (pgie_diag_enabled.load(std::memory_order_relaxed))
    std::atexit(pgie_diag_atexit_export);
#endif
  loop = g_main_loop_new(nullptr, FALSE);
  const std::string pipeline_name = source_plan.size() == 1
      ? std::string(source_plan.front().camera_id) + "-validation"
      : "CAM-01-CAM-02-CAM-03-CAM-04-CAM-05-CAM-06-validation";
  pipeline = gst_pipeline_new(pipeline_name.c_str());
  mux = make("nvstreammux", "mux");
  g_object_set(
      mux,
      "batch-size", shape.batch_size,
      "width", 2560u,
      "height", 1440u,
      "live-source", TRUE,
      "batched-push-timeout", 50000,
      "sync-inputs", FALSE,
      "attach-sys-ts", TRUE,
      "nvbuf-memory-type", 2,
      "cache-buffer", FALSE,
      "drop-pipeline-eos", TRUE,
      nullptr);

  for (auto *ctx : sources) {
    const std::string queue_name = ctx->id + "-queue";
    ctx->queue = make("queue", queue_name.c_str());
    g_object_set(ctx->queue, "max-size-buffers", 12u, "max-size-bytes", 0u,
                 "max-size-time", guint64(0), "leaky", 0, nullptr);
    probe(ctx->queue, "sink", &ctx->ingress);
    probe(ctx->queue, "src", &ctx->input);
#ifdef YOLO26_PERSON
    pgie_diag_probe(ctx->queue, "sink", PGIE_DIAG_CAMERA_QUEUE_BASE + ctx->source_id, 0, ctx->source_id);
    pgie_diag_probe(ctx->queue, "src", PGIE_DIAG_CAMERA_QUEUE_BASE + ctx->source_id, 1, ctx->source_id);
    pgie_diag_register_queue(ctx->queue, PGIE_DIAG_CAMERA_QUEUE_BASE + ctx->source_id);
#endif

    ctx->source = create_source_element(ctx);
    if (!ctx->source) return 2;

    const std::string sink_name = "sink_" + std::to_string(ctx->mux_pad_index);
    GstPad *mp = gst_element_request_pad_simple(mux, sink_name.c_str());
    GstPad *qp = gst_element_get_static_pad(ctx->queue, "src");
    if (!mp || !qp || gst_pad_link(qp, mp) != GST_PAD_LINK_OK) {
      fail(ctx->id + " -> nvstreammux link failed");
      return 2;
    }
    gst_object_unref(qp);
    gst_object_unref(mp);
  }

  GstElement *tiler = make("nvmultistreamtiler", "tiler");
  g_object_set(tiler, "rows", shape.tile_rows, "columns", shape.tile_columns,
               "width", shape.output_width, "height", shape.output_height, nullptr);
  GstElement *conv = make("nvvideoconvert", "converter");
  GstElement *capsfilter = make("capsfilter", "nvmm-caps");
  const std::string capture_caps = "video/x-raw(memory:NVMM),format=NV12,width=" +
      std::to_string(shape.output_width) + ",height=" + std::to_string(shape.output_height);
  GstCaps *caps = gst_caps_from_string(capture_caps.c_str());
  g_object_set(capsfilter, "caps", caps, nullptr);
  gst_caps_unref(caps);
#ifdef STAGE_CAPTURE
  pre_nvenc_capture.kind = StageCaptureKind::PRE_NVENC;
  pre_nvenc_capture.source_id = -1;
  pre_nvenc_capture.width = shape.output_width;
  pre_nvenc_capture.height = shape.output_height;
  pre_nvenc_capture.raw_format = "NV12";
  pre_nvenc_capture.caps_format = "NV12";
  pre_nvenc_capture.caps_width = shape.output_width;
  pre_nvenc_capture.caps_height = shape.output_height;
  if (stage_capture_enabled) {
    GstPad *capture_pad = gst_element_get_static_pad(capsfilter, "src");
    if (!attach_stage_capture_probe(capture_pad, &pre_nvenc_capture))
      fail("stage capture could not attach before NVENC");
    if (capture_pad) gst_object_unref(capture_pad);
  }
#endif
  encoder = make("nvv4l2h264enc", "NVENC");
  g_object_set(encoder, "bitrate", 20000000u, "iframeinterval", 40u, nullptr);
  GstElement *encoded_tee = make("tee", "encoded-tee");

  GstElement *file_queue = make("queue", "file-queue");
  g_object_set(file_queue, "max-size-buffers", 12u, "max-size-bytes", 0u,
               "max-size-time", guint64(0), "leaky", 0, "silent", TRUE, nullptr);
  GstElement *file_parse = make("h264parse", "output-parse");
  GstElement *filemux = make("matroskamux", "filemux");
  sink_element = make("filesink", "sink");
#ifdef YOLO26_PERSON
  pgie_diag_trace_element(capsfilter, PGIE_DIAG_OUTPUT_CAPS);
  pgie_diag_trace_element(encoder, PGIE_DIAG_ENCODER);
  pgie_diag_trace_element(file_queue, PGIE_DIAG_FILE_QUEUE);
  pgie_diag_register_queue(file_queue, PGIE_DIAG_FILE_QUEUE);
  pgie_diag_trace_element(file_parse, PGIE_DIAG_FILE_PARSE);
  pgie_diag_trace_element(filemux, PGIE_DIAG_FILE_MUX);
  pgie_diag_trace_element(sink_element, PGIE_DIAG_FILE_SINK);
#endif
  const std::string recording_path = source_plan.size() == 1
      ? std::string("/work/") + source_plan.front().camera_id + ".mkv"
      : "/work/CAM-01_CAM-02_CAM-03_CAM-04_CAM-05_CAM-06.mkv";
  g_object_set(sink_element, "location", recording_path.c_str(), "sync", FALSE,
               "async", FALSE, "enable-last-sample", FALSE, nullptr);

#ifdef YOLO26_PERSON
  detection_link(mux, tiler, conv);
#else
  link(mux, tiler);
  link(tiler, conv);
#endif
  link(conv, capsfilter);
  link(capsfilter, encoder);
  link(encoder, encoded_tee);

  link(encoded_tee, file_queue);
  link(file_queue, file_parse);
  link(file_parse, filemux);
  link(filemux, sink_element);

  if (preview_enabled) {
    GstElement *preview_queue = make("queue", "preview-queue");
    g_object_set(preview_queue,
                 "max-size-buffers", 4u,
                 "max-size-bytes", 0u,
                 "max-size-time", guint64(0),
                 "leaky", 2,
                 "silent", TRUE,
                 nullptr);
    GstElement *preview_parse = make("h264parse", "preview-parse");
    g_object_set(preview_parse, "config-interval", -1, nullptr);
    GstElement *preview_caps = make("capsfilter", "preview-caps");
    GstCaps *pcaps = gst_caps_from_string(
        "video/x-h264,stream-format=byte-stream,alignment=au");
    g_object_set(preview_caps, "caps", pcaps, nullptr);
    gst_caps_unref(pcaps);
    GstElement *tsmux = make("mpegtsmux", "preview-tsmux");
    g_object_set(tsmux, "alignment", 7, nullptr);
    GstElement *udp = make("udpsink", "preview-udp");
    g_object_set(udp,
                 "host", "host.docker.internal",
                 "port", preview_port,
                 "sync", FALSE,
                 "async", FALSE,
                 nullptr);

#ifdef YOLO26_PERSON
    pgie_diag_trace_element(preview_queue, PGIE_DIAG_PREVIEW_QUEUE);
    pgie_diag_register_queue(preview_queue, PGIE_DIAG_PREVIEW_QUEUE);
    pgie_diag_trace_element(preview_parse, PGIE_DIAG_PREVIEW_PARSE);
    pgie_diag_trace_element(preview_caps, PGIE_DIAG_PREVIEW_CAPS);
    pgie_diag_trace_element(tsmux, PGIE_DIAG_PREVIEW_MUX);
    pgie_diag_trace_element(udp, PGIE_DIAG_PREVIEW_SINK);
#endif

    link(encoded_tee, preview_queue);
    link(preview_queue, preview_parse);
    link(preview_parse, preview_caps);
    link(preview_caps, tsmux);
    link(tsmux, udp);
  }

  probe(encoder, "src", &output_counter);

#ifndef YOLO26_PERSON
  std::string active_names;
  for (const auto *ctx : sources) {
    if (!active_names.empty()) active_names += "+";
    active_names += ctx->id;
  }
  event(nullptr, "GRAPH",
        active_names + " nvurisrcbin/NVDEC->NVMM->queues->nvstreammux(batch=" +
        std::to_string(shape.batch_size) + ",2560x1440,live,50000us,sync-inputs=false)->tiler(" +
        std::to_string(shape.tile_rows) + "x" + std::to_string(shape.tile_columns) + "," +
        std::to_string(shape.output_width) + "x" + std::to_string(shape.output_height) +
        ")->NVENC->record" + (preview_enabled ? "+UDP-preview inference=0" : " inference=0"));

#endif
  GstBus *bus = gst_element_get_bus(pipeline);
  gst_bus_add_watch(bus, bus_message, nullptr);
  gst_object_unref(bus);
  g_unix_signal_add(SIGINT, stop_signal, nullptr);
  g_unix_signal_add(SIGTERM, stop_signal, nullptr);

  start_us = previous_us = g_get_monotonic_time();
  previous_cpu = cpu_seconds();
#ifdef STAGE_CAPTURE
  if (stage_capture_enabled && !start_stage_capture())
    fail("stage capture initialization failed before PLAYING");
#endif
#ifdef YOLO26_PERSON
  pgie_diag_pipeline_started.store(true, std::memory_order_release);
#endif
#ifdef STAGE_CAPTURE
  if (stage_capture_enabled && fatal.load()) {
    event(nullptr, "STAGE_CAPTURE", "PLAYING skipped after capture initialization failure");
  } else
#endif
  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    fail("initial PLAYING failed");
  }
  g_timeout_add_seconds(
#ifdef YOLO26_PERSON
      pgie_diag_enabled.load(std::memory_order_relaxed) ? 1 : 5,
#else
      5,
#endif
      tick, nullptr);
  if (!fatal) g_main_loop_run(loop);

  stopping = true;
  const GstStateChangeReturn null_set = gst_element_set_state(pipeline, GST_STATE_NULL);
  GstState current = GST_STATE_VOID_PENDING, pending = GST_STATE_VOID_PENDING;
  const GstStateChangeReturn null_get =
      gst_element_get_state(pipeline, &current, &pending, 0);
  [[maybe_unused]] bool pipeline_stopped =
      null_set != GST_STATE_CHANGE_FAILURE && null_get != GST_STATE_CHANGE_FAILURE &&
      current == GST_STATE_NULL && pending == GST_STATE_VOID_PENDING;
#ifdef STAGE_CAPTURE
  if (stage_capture_enabled && !pipeline_stopped) {
    gst_element_get_state(pipeline, &current, &pending, 2 * GST_SECOND);
    pipeline_stopped = current == GST_STATE_NULL && pending == GST_STATE_VOID_PENDING;
  }
#endif
#ifdef YOLO26_PERSON
  detection_stats();
#ifdef YOLO26_TRACKER
  tracker_stats();
  if (tracker_evidence) std::fclose(tracker_evidence);
#endif
  if (detection_evidence) std::fclose(detection_evidence);
  if (frame_evidence) std::fclose(frame_evidence);
#endif
#ifdef STAGE_CAPTURE
  if (stage_capture_enabled && !pipeline_stopped) {
    event(nullptr, "STAGE_CAPTURE",
          "finalization aborted: pipeline NULL state unverified; capture may be partial");
#ifdef YOLO26_PERSON
    event(nullptr, "DIAGNOSTIC", "trace export skipped: pipeline NULL state unverified");
#endif
    // Keep the pipeline, probe contexts, and capture object alive until process
    // exit. Freeing them while streaming-task termination is unverified could
    // leave pad callbacks with dangling context pointers. The nonzero process
    // exit is the bounded failure cleanup for this exceptional path.
    return 2;
  }
#endif
  gst_object_unref(pipeline);
#ifdef STAGE_CAPTURE
  if (pipeline_stopped) stop_stage_capture();
#endif
#ifdef YOLO26_PERSON
  // GST_STATE_NULL stops the streaming tasks; unref the pipeline before
  // exporting so no probe callback can still append to the diagnostic ring.
  if (pipeline_stopped) {
    pgie_diag_export_ready.store(true, std::memory_order_release);
    pgie_diag_dump();
  } else {
    event(nullptr, "DIAGNOSTIC", "trace export skipped: pipeline NULL state unverified");
  }
#endif

  for (auto *ctx : sources) {
    if (ctx->jitterbuffer) gst_object_unref(ctx->jitterbuffer);
    g_print("%s END ingress=%lu input=%lu hardware_decoder=%d errors=%u warnings=%u pts_backwards=%lu pts_duplicates=%lu\n",
            ctx->id.c_str(), ctx->ingress.frames.load(), ctx->input.frames.load(), int(ctx->decoder_seen.load()),
            ctx->errors, ctx->warnings, ctx->input.backwards.load(), ctx->input.duplicates.load());
  }
  g_print("GROUP END output=%lu fatal=%d shared_errors=%u shared_warnings=%u isolation=%s\n",
          output_counter.frames.load(), int(fatal.load()), shared_errors, shared_warnings,
          interruption_started ? (interruption_reported ? "completed" : "incomplete") : "not-requested");

  bool ok = !fatal && output_counter.frames.load() > 0;
  for (auto *ctx : sources) ok = ok && ctx->decoder_seen && ctx->input.frames.load() > 0;
  if (interruption_started) ok = ok && interruption_reported;

  for (auto *ctx : sources) delete ctx;
  g_main_loop_unref(loop);
  return ok ? 0 : 1;
}
