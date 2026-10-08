// CAM-01 + CAM-02 + CAM-03 + CAM-04 + CAM-05 + CAM-06 DeepStream 9.1 validation. No inference, tracker, OpenCV or CPU pixel copies.
#include <gst/gst.h>
#include <glib-unix.h>
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <deque>
#include <utility>
#include <mutex>
#include <sys/resource.h>
#include <unistd.h>
#include <fcntl.h>

struct Counter {
  std::atomic<guint64> frames{0};
  std::atomic<guint64> pts{GST_CLOCK_TIME_NONE};
  std::atomic<guint64> backwards{0};
  std::atomic<guint64> duplicates{0};
  std::atomic<gint64> last_us{0};
  std::atomic<gint64> max_gap_us{0};
};

struct DecoderFrameStamp {
  guint64 input_ns = 0;
  guint64 output_ns = 0;
  guint32 frame_num = 0;
  guint64 sequence = 0;
};

struct DecoderInputStamp {
  guint64 monotonic_ns = 0;
  guint64 sequence = 0;
};

struct SourceCtx {
  std::string id;
  int source_id = 0;
  std::string uri;
  std::string username;
  std::string password;
  int latency_ms = 300;
  bool drop_on_latency = false;
  bool decoder_low_latency_mode = false;
  guint decoder_extra_surfaces = 4;
  GstElement *source = nullptr;
  GstElement *tee = nullptr;
  GstElement *queue = nullptr;
#ifdef PGIE_ISOLATION
  GstElement *replay_pacer = nullptr;
#endif
  GstElement *jitterbuffer = nullptr;
  GMutex jitter_lock;
  Counter decoded;
  Counter ingress;
  Counter input;
  std::atomic_bool decoder_seen{false};
  guint errors = 0;
  guint warnings = 0;
  std::atomic<guint> queue_high_water{0};
  bool stale_recovery_pending = false;
  guint64 stale_recovery_baseline_frames = 0;
  guint64 stale_recovery_baseline_decoded_frames = 0;
  guint64 stale_recovery_baseline_pgie_frames = 0;
  guint64 stale_recovery_baseline_tracker_updates = 0;
  guint64 stale_recovery_baseline_preview_frames = 0;
  guint stale_recovery_count = 0;
  guint consecutive_stale_recoveries = 0;
  gint64 stale_recovery_started_us = 0;
  gint64 last_stale_recovery_us = 0;
  guint source_generation = 0;
  guint64 previous_ingress_frames = 0;
  guint64 previous_frames = 0;
  std::mutex decoded_clock_mutex;
  std::unordered_map<guint64, DecoderInputStamp> decoder_input_clock_ns;
  std::deque<std::pair<guint64, guint64>> decoder_input_order;
  guint64 decoder_input_sequence = 0;
  std::unordered_map<guint64, DecoderFrameStamp> decoded_clock_ns;
  std::deque<std::pair<guint64, guint64>> decoded_clock_order;
  guint64 decoded_clock_sequence = 0;
  std::atomic<guint32> decoded_output_frames{0};
};

#ifdef YOLO26_TRACKER
static void f5r_preview_quality_capture_nv12(
    GstBuffer *buffer, guint source_id, guint32 source_frame_num,
    guint64 pts_ns, const char *stage, guint surface_index);
#endif

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
static std::string monitoring_path;
// Recovery-completion counters deliberately span the translation-unit
// boundaries: they let the watchdog require progress through PGIE, canonical
// tracking, and preview publication before declaring a source healthy.
static std::atomic<guint64> source_pgie_frames[6]{};
static std::atomic<guint64> source_tracker_updates[6]{};
static std::atomic<guint64> source_preview_published[6]{};
struct RecoveryTestStall {
  std::string camera_id;
  gint64 at_us = 0;
  bool injected = false;
};
static std::vector<RecoveryTestStall> recovery_test_stalls;
static size_t recovery_test_next_stall = 0;
// Local-NvDCF six-camera wall candidate: 1 ms measured decoder-reference to
// UI P95 below 40 ms across all six sources.
static guint mux_batched_push_timeout_us = 1000;
// Diagnostic candidate overrides for isolating mux/tracker geometry. Defaults
// retain the existing six-camera candidate behavior; replay evidence records
// the effective values for each run.
static guint v11_mux_width = 2560;
static guint v11_mux_height = 1440;
#ifdef PGIE_ISOLATION
static guint pgie_isolation_batch_size = 2;
static guint pgie_isolation_width = 1920;
static guint pgie_isolation_height = 1080;
static guint pgie_isolation_memory_type = 0;
static bool pgie_isolation_live_source = false;
static FILE *direct_file_frame_evidence = nullptr;
static std::atomic_bool direct_file_frame_write_failed{false};

static void direct_file_log_decoded_frame(guint source_id, guint32 local_frame,
                                          guint64 decoded_sequence, guint64 pts_ns) {
  if (!direct_file_frame_evidence || local_frame == G_MAXUINT32 ||
      pts_ns == GST_CLOCK_TIME_NONE) {
    direct_file_frame_write_failed = true;
    return;
  }
  char line[256];
  const int length = std::snprintf(line, sizeof(line),
      "{\"source_id\":%u,\"source_local_frame_num\":%u,\"decoded_sequence\":%lu,\"source_pts_ns\":%lu}\n",
      source_id, local_frame, decoded_sequence, pts_ns);
  if (length <= 0 || static_cast<size_t>(length) >= sizeof(line)) {
    direct_file_frame_write_failed = true;
    return;
  }
  flockfile(direct_file_frame_evidence);
  const size_t written = std::fwrite(line, 1, static_cast<size_t>(length), direct_file_frame_evidence);
  funlockfile(direct_file_frame_evidence);
  if (written != static_cast<size_t>(length)) direct_file_frame_write_failed = true;
}
#endif
static guint64 monitoring_sequence = 0;
static guint64 monitoring_prev_input[6] = {};
static guint64 monitoring_prev_infer[6] = {};
static guint64 monitoring_prev_preview[6] = {};
static gint64 monitoring_prev_us = 0;
static int duration_sec = 60;
static bool preview_enabled = false;
static int preview_port = 5600;
static bool source_stall_recovery_enabled = false;
static constexpr gint64 SOURCE_STALL_RECOVERY_AFTER_US = 2'500'000;
static constexpr gint64 SOURCE_STALL_RECOVERY_CONFIRM_US = 15'000'000;
static constexpr gint64 SOURCE_STALL_RECOVERY_COOLDOWN_US = 60'000'000;

static std::string interrupt_camera = "none";
static int interrupt_at = 20;
static int interrupt_seconds = 12;
static bool interruption_started = false;
static bool interruption_resumed = false;
static bool interruption_reported = false;
static guint64 target_start_frames = 0, peer_start_frames = 0, target_resume_frames = 0, peer_resume_frames = 0;

static void safe_print(const std::string &prefix, const std::string &detail) {
  std::string text = detail;
  for (const auto *s : sources) {
    for (const auto &secret : {s->password, s->uri}) {
      if (secret.empty()) continue;
      size_t at = 0;
      while ((at = text.find(secret, at)) != std::string::npos) {
        text.replace(at, secret.size(), "<redacted>");
        at += 10;
      }
    }
  }
  for (auto &c : text) if (c == '\n' || c == '\r') c = ' ';
  g_print("%s %s\n", prefix.c_str(), text.c_str());
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
      "drop-on-latency", ctx->drop_on_latency,
      "tcp-timeout", guint64(10000000),
      "do-rtsp-keep-alive", TRUE,
      nullptr);
}

static GstPadProbeReturn decoder_input_stamp(GstPad*, GstPadProbeInfo *info, gpointer opaque) {
  auto *ctx = static_cast<SourceCtx*>(opaque);
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer) return GST_PAD_PROBE_OK;
  const guint64 pts = GST_BUFFER_PTS(buffer);
  if (pts == GST_CLOCK_TIME_NONE) return GST_PAD_PROBE_OK;
  std::lock_guard<std::mutex> guard(ctx->decoded_clock_mutex);
  const guint64 sequence = ++ctx->decoder_input_sequence;
  ctx->decoder_input_clock_ns[pts] = {
      static_cast<guint64>(g_get_monotonic_time()) * 1000, sequence};
  ctx->decoder_input_order.emplace_back(pts, sequence);
  while (ctx->decoder_input_order.size() > 512) {
    const auto oldest = ctx->decoder_input_order.front();
    ctx->decoder_input_order.pop_front();
    const auto found = ctx->decoder_input_clock_ns.find(oldest.first);
    if (found != ctx->decoder_input_clock_ns.end() &&
        found->second.sequence == oldest.second)
      ctx->decoder_input_clock_ns.erase(found);
  }
  return GST_PAD_PROBE_OK;
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
    GParamSpec *low_latency = g_object_class_find_property(G_OBJECT_GET_CLASS(e), "low-latency-mode");
    if (low_latency) {
      g_object_set(e, "low-latency-mode", ctx->decoder_low_latency_mode, nullptr);
      gboolean actual = FALSE;
      g_object_get(e, "low-latency-mode", &actual, nullptr);
      if (actual != ctx->decoder_low_latency_mode) {
        fail(ctx->id + " decoder low-latency-mode did not match staged runtime config");
      }
    } else if (ctx->decoder_low_latency_mode) {
      fail(ctx->id + " decoder lacks configured low-latency-mode property");
    }
    GParamSpec *surfaces = g_object_class_find_property(G_OBJECT_GET_CLASS(e), "num-extra-surfaces");
    if (surfaces) {
      g_object_set(e, "num-extra-surfaces", ctx->decoder_extra_surfaces, nullptr);
      guint actual = 0;
      g_object_get(e, "num-extra-surfaces", &actual, nullptr);
      if (actual != ctx->decoder_extra_surfaces) {
        fail(ctx->id + " decoder extra-surface count did not match staged runtime config");
      }
    } else if (ctx->decoder_extra_surfaces != 4) {
      fail(ctx->id + " decoder lacks configured extra-surface property");
    }
    event(ctx, "DECODER_PROFILE", "low_latency=" + std::to_string(ctx->decoder_low_latency_mode) +
        " extra_surfaces=" + std::to_string(ctx->decoder_extra_surfaces) +
        " rtspsrc_latency_ms=" + std::to_string(ctx->latency_ms) +
        " drop_on_latency=" + std::to_string(ctx->drop_on_latency));
    GstPad *decoder_sink = gst_element_get_static_pad(e, "sink");
    if (!decoder_sink) {
      event(ctx, "ERROR", "hardware decoder has no sink pad for timestamp probe");
    } else {
      gst_pad_add_probe(decoder_sink, GST_PAD_PROBE_TYPE_BUFFER, decoder_input_stamp, ctx, nullptr);
      gst_object_unref(decoder_sink);
    }
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
    // This is nvurisrcbin's decoded video output (NVDEC); track it separately
    // from the analytics queue so a freeze is localized before publication.
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, count_buffer, &ctx->decoded, nullptr);
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER,
      [](GstPad*, GstPadProbeInfo *info, gpointer opaque) -> GstPadProbeReturn {
        auto *owner = static_cast<SourceCtx*>(opaque);
        GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
        if (!buffer) return GST_PAD_PROBE_OK;
        const guint64 pts = GST_BUFFER_PTS(buffer);
        guint32 decoded_frame_num = G_MAXUINT32;
        guint64 decoded_sequence = 0;
        if (pts != GST_CLOCK_TIME_NONE) {
          std::lock_guard<std::mutex> guard(owner->decoded_clock_mutex);
          DecoderFrameStamp stamp;
          const auto input_stamp = owner->decoder_input_clock_ns.find(pts);
          if (input_stamp != owner->decoder_input_clock_ns.end()) {
            stamp.input_ns = input_stamp->second.monotonic_ns;
            owner->decoder_input_clock_ns.erase(input_stamp);
          }
          stamp.output_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
          stamp.frame_num = owner->decoded_output_frames.fetch_add(1);
          stamp.sequence = ++owner->decoded_clock_sequence;
          decoded_frame_num = stamp.frame_num;
          decoded_sequence = stamp.sequence;
          owner->decoded_clock_ns[pts] = stamp;
          owner->decoded_clock_order.emplace_back(pts, stamp.sequence);
          while (owner->decoded_clock_order.size() > 512) {
            const auto oldest = owner->decoded_clock_order.front();
            owner->decoded_clock_order.pop_front();
            const auto found = owner->decoded_clock_ns.find(oldest.first);
            if (found != owner->decoded_clock_ns.end() &&
                found->second.sequence == oldest.second)
              owner->decoded_clock_ns.erase(found);
          }
        }
#ifdef PGIE_ISOLATION
        if (decoded_frame_num != G_MAXUINT32 && owner->uri.rfind("file://", 0) == 0)
          direct_file_log_decoded_frame(owner->source_id, decoded_frame_num, decoded_sequence, pts);
#endif
#if defined(YOLO26_TRACKER) && !defined(PGIE_ISOLATION)
        if (decoded_frame_num != G_MAXUINT32) {
          f5r_preview_quality_capture_nv12(buffer, owner->source_id, decoded_frame_num,
              pts, "decoded-source", 0);
        }
#endif
        return GST_PAD_PROBE_OK;
      }, ctx, nullptr);
    GstElement *link_target = ctx->tee;
#ifdef PGIE_ISOLATION
    if (ctx->replay_pacer) link_target = ctx->replay_pacer;
#endif
    GstPad *dest = gst_element_get_static_pad(link_target, "sink");
    if (!gst_pad_is_linked(dest)) {
      if (gst_pad_link(pad, dest) != GST_PAD_LINK_OK) {
        event(ctx, "ERROR", std::string("dynamic video link failed to ") + GST_ELEMENT_NAME(link_target));
      } else {
        gchar *s = gst_caps_to_string(caps);
        const bool nvmm = gst_caps_features_contains(gst_caps_get_features(caps, 0), "memory:NVMM");
        event(ctx, "SOURCE_LINK", std::string(GST_PAD_NAME(pad)) + " linked " + (nvmm ? "memory:NVMM " : "") + s);
        g_free(s);
      }
    }
    gst_object_unref(dest);
  }
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
  ++ctx->source_generation;
  event(ctx, "SOURCE_OWNER_CREATED", "source=" + source_name +
      " generation=" + std::to_string(ctx->source_generation) + " owner_count=1");
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
      "drop-on-latency", ctx->drop_on_latency,
      "num-extra-surfaces", ctx->decoder_extra_surfaces,
      "cudadec-memtype", 0,
      "drop-frame-interval", 0u,
      "dec-skip-frames", 0,
      "low-latency-mode", ctx->decoder_low_latency_mode,
      "rtsp-reconnect-interval", 5u,
      "init-rtsp-reconnect-interval", 5u,
      "rtsp-reconnect-attempts", -1,
      "max-size-buffers", 12u,
      "leaky", 0,
      "message-forward", TRUE,
      nullptr);
#ifdef PGIE_ISOLATION
  if (ctx->uri.rfind("file://", 0) == 0) {
    GParamSpec *file_loop = g_object_class_find_property(G_OBJECT_GET_CLASS(source), "file-loop");
    if (!file_loop) {
      fail(ctx->id + " nvurisrcbin has no file-loop property; cannot prove non-looping replay");
      return source;
    }
    g_object_set(source, "file-loop", FALSE, nullptr);
    gboolean file_loop_enabled = TRUE;
    g_object_get(source, "file-loop", &file_loop_enabled, nullptr);
    if (file_loop_enabled) {
      fail(ctx->id + " direct-file source did not verify file-loop=0");
      return source;
    }
    event(ctx, "DIRECT_FILE_SOURCE", "uri_scheme=file file_loop=0 rtsp_reconnect=not_applicable");
  }
#endif
  return source;
}

static bool parse_recovery_test_plan(const char *raw, std::string *error) {
  recovery_test_stalls.clear();
  recovery_test_next_stall = 0;
  if (!raw || !*raw) return true;
  if (!source_stall_recovery_enabled) {
    *error = "V11_TEST_SOURCE_STALL_PLAN requires V11_SOURCE_STALL_RECOVERY=1";
    return false;
  }
  std::stringstream input(raw);
  std::string item;
  gint64 previous_at_us = -1;
  while (std::getline(input, item, ',')) {
    const size_t delimiter = item.find('@');
    if (delimiter == std::string::npos || delimiter == 0 || delimiter + 1 >= item.size() ||
        item.find('@', delimiter + 1) != std::string::npos) {
      *error = "V11_TEST_SOURCE_STALL_PLAN entries must be CAM-XX@SECONDS";
      return false;
    }
    const std::string camera = item.substr(0, delimiter);
    const bool known_camera = camera == "CAM-01" || camera == "CAM-02" || camera == "CAM-03" ||
        camera == "CAM-04" || camera == "CAM-05" || camera == "CAM-06";
    char *end = nullptr;
    const long seconds = std::strtol(item.c_str() + delimiter + 1, &end, 10);
    if (!known_camera || !end || *end || seconds < 20 || seconds > 86400) {
      *error = "V11_TEST_SOURCE_STALL_PLAN has an invalid camera or time (20..86400 seconds)";
      return false;
    }
    const gint64 at_us = static_cast<gint64>(seconds) * G_USEC_PER_SEC;
    if (previous_at_us >= 0 && at_us - previous_at_us < SOURCE_STALL_RECOVERY_COOLDOWN_US + 15'000'000) {
      *error = "V11_TEST_SOURCE_STALL_PLAN entries must be at least 75 seconds apart";
      return false;
    }
    recovery_test_stalls.push_back({camera, at_us, false});
    previous_at_us = at_us;
  }
  if (recovery_test_stalls.empty()) {
    *error = "V11_TEST_SOURCE_STALL_PLAN must contain at least one stall";
    return false;
  }
  return true;
}

static SourceCtx *find_source(const std::string &id);

static bool replace_stalled_source(SourceCtx *target, gint64 now_us) {
  if (!target || !target->source || target->stale_recovery_pending) return false;
  const gint64 input_last = target->input.last_us.load();
  const gint64 ingress_last = target->ingress.last_us.load();
  const gint64 decoded_last = target->decoded.last_us.load();
  const double input_age_ms = input_last ? (now_us - input_last) / 1000.0 : -1.0;
  const double ingress_age_ms = ingress_last ? (now_us - ingress_last) / 1000.0 : -1.0;
  const double decoded_age_ms = decoded_last ? (now_us - decoded_last) / 1000.0 : -1.0;
  guint queue_level = 0;
  g_object_get(target->queue, "current-level-buffers", &queue_level, nullptr);

  // Do not reset a source when decoded buffers are reaching its analytics
  // queue. That points to downstream backpressure, not an RTSP/decoder stall.
  if (queue_level != 0 ||
      (decoded_last && now_us - decoded_last <= SOURCE_STALL_RECOVERY_AFTER_US) ||
      (input_last && now_us - input_last <= SOURCE_STALL_RECOVERY_AFTER_US) ||
      (ingress_last && now_us - ingress_last <= SOURCE_STALL_RECOVERY_AFTER_US)) {
    return false;
  }
  if (target->stale_recovery_count > 0 &&
      now_us - target->last_stale_recovery_us < SOURCE_STALL_RECOVERY_COOLDOWN_US) {
    fail(target->id + " source remained stale after its bounded recovery; refusing a reconnect loop");
    return false;
  }

  target->stale_recovery_pending = true;
  target->stale_recovery_baseline_frames = target->input.frames.load();
  target->stale_recovery_baseline_decoded_frames = target->decoded.frames.load();
  target->stale_recovery_baseline_pgie_frames = source_pgie_frames[target->source_id].load();
  target->stale_recovery_baseline_tracker_updates = source_tracker_updates[target->source_id].load();
  target->stale_recovery_baseline_preview_frames = source_preview_published[target->source_id].load();
  target->stale_recovery_started_us = now_us;
  ++target->stale_recovery_count;
  ++target->consecutive_stale_recoveries;
  target->last_stale_recovery_us = now_us;
  event(target, "SOURCE_STALL_RECOVERY_BEGIN",
      "elapsed_ms=" + std::to_string((now_us - start_us) / 1000) +
      " decoded ingress and analytics input stale; input_age_ms=" +
      std::to_string(input_age_ms) + " decoded_age_ms=" + std::to_string(decoded_age_ms) +
      " ingress_age_ms=" + std::to_string(ingress_age_ms) +
      " queue_depth=0 owner_count_before=1 generation=" + std::to_string(target->source_generation) +
      " decoded_frames=" + std::to_string(target->stale_recovery_baseline_decoded_frames) +
      " input_frames=" + std::to_string(target->stale_recovery_baseline_frames) +
      " pgie_frames=" + std::to_string(target->stale_recovery_baseline_pgie_frames) +
      " tracker_updates=" + std::to_string(target->stale_recovery_baseline_tracker_updates) +
      " preview_frames=" + std::to_string(target->stale_recovery_baseline_preview_frames) +
      "; replacing existing nvurisrcbin owner");

  GstElement *old_source = target->source;
  const std::string old_source_name = GST_ELEMENT_NAME(old_source);
  gst_element_set_state(old_source, GST_STATE_NULL);
  const GstStateChangeReturn stopped = gst_element_get_state(
      old_source, nullptr, nullptr, 3 * GST_SECOND);
  if (stopped == GST_STATE_CHANGE_FAILURE || stopped == GST_STATE_CHANGE_ASYNC) {
    target->stale_recovery_pending = false;
    fail(target->id + " stale source did not stop cleanly; refusing to create a second RTSP owner");
    return false;
  }
  if (!gst_bin_remove(GST_BIN(pipeline), old_source)) {
    target->stale_recovery_pending = false;
    fail(target->id + " stale source could not be removed; refusing to create a second RTSP owner");
    return false;
  }
  target->source = nullptr;
  event(target, "SOURCE_STALL_OLD_OWNER_REMOVED",
      "old_source=" + old_source_name +
      " elapsed_ms=" + std::to_string((g_get_monotonic_time() - start_us) / 1000) +
      " owner_count=0; replacement creation may now begin");

  g_mutex_lock(&target->jitter_lock);
  if (target->jitterbuffer) {
    gst_object_unref(target->jitterbuffer);
    target->jitterbuffer = nullptr;
  }
  g_mutex_unlock(&target->jitter_lock);
  {
    std::lock_guard<std::mutex> guard(target->decoded_clock_mutex);
    target->decoder_input_clock_ns.clear();
    target->decoder_input_order.clear();
    target->decoded_clock_ns.clear();
    target->decoded_clock_order.clear();
  }
  target->decoder_seen = false;
  target->source = create_source_element(target);
  if (!target->source || !gst_element_sync_state_with_parent(target->source)) {
    target->stale_recovery_pending = false;
    fail(target->id + " failed to recreate its single RTSP/NVDEC source owner");
    return false;
  }
  event(target, "SOURCE_STALL_RECOVERY_RECONNECTED",
      "elapsed_ms=" + std::to_string((g_get_monotonic_time() - start_us) / 1000) +
      " old owner stopped and removed before replacement; owner_count=1 generation=" +
      std::to_string(target->source_generation) +
      " awaiting decoded+PGIE+tracker+preview progress");
  return true;
}

static void inject_scheduled_source_stall(gint64 now_us) {
  if (recovery_test_next_stall >= recovery_test_stalls.size()) return;
  auto &test = recovery_test_stalls[recovery_test_next_stall];
  if (test.injected || now_us - start_us < test.at_us) return;
  SourceCtx *target = find_source(test.camera_id);
  if (!target || !target->source || target->stale_recovery_pending) {
    fail("controlled source-stall test could not resolve an idle target owner");
    return;
  }
  const guint64 decoded = target->decoded.frames.load();
  const guint64 input = target->input.frames.load();
  const guint64 pgie = source_pgie_frames[target->source_id].load();
  const guint64 tracker = source_tracker_updates[target->source_id].load();
  const guint64 preview = source_preview_published[target->source_id].load();
  const std::string old_name = GST_ELEMENT_NAME(target->source);
  event(target, "TEST_SOURCE_STALL_BEGIN",
      "elapsed_ms=" + std::to_string((now_us - start_us) / 1000) +
      " mode=safe_source_state_null equivalent_to_decoded_stall=true old_source=" + old_name +
      " generation=" + std::to_string(target->source_generation) +
      " decoded_frames=" + std::to_string(decoded) + " input_frames=" + std::to_string(input) +
      " pgie_frames=" + std::to_string(pgie) + " tracker_updates=" + std::to_string(tracker) +
      " preview_frames=" + std::to_string(preview) + " owner_count=1");
  const GstStateChangeReturn result = gst_element_set_state(target->source, GST_STATE_NULL);
  if (result == GST_STATE_CHANGE_FAILURE) {
    fail(target->id + " controlled source interruption could not reach NULL state");
    return;
  }
  test.injected = true;
  ++recovery_test_next_stall;
}

static gboolean source_stall_recovery_tick(gpointer) {
  if (!source_stall_recovery_enabled || stopping || fatal) return G_SOURCE_CONTINUE;
  const gint64 now_us = g_get_monotonic_time();
  if ((now_us - start_us) < 15'000'000) return G_SOURCE_CONTINUE;
  inject_scheduled_source_stall(now_us);
  if (fatal) return G_SOURCE_REMOVE;
  for (auto *ctx : sources) {
    const guint64 input_frames = ctx->input.frames.load();
    if (ctx->stale_recovery_pending) {
      const guint64 decoded_frames = ctx->decoded.frames.load();
      const guint64 pgie_frames = source_pgie_frames[ctx->source_id].load();
      const guint64 tracker_updates = source_tracker_updates[ctx->source_id].load();
      const guint64 preview_frames = source_preview_published[ctx->source_id].load();
      if (input_frames > ctx->stale_recovery_baseline_frames &&
          decoded_frames > ctx->stale_recovery_baseline_decoded_frames &&
          pgie_frames > ctx->stale_recovery_baseline_pgie_frames &&
          tracker_updates > ctx->stale_recovery_baseline_tracker_updates &&
          preview_frames > ctx->stale_recovery_baseline_preview_frames) {
        ctx->stale_recovery_pending = false;
        ctx->consecutive_stale_recoveries = 0;
        ctx->stale_recovery_started_us = 0;
        event(ctx, "SOURCE_STALL_RECOVERY_PASS",
            "elapsed_ms=" + std::to_string((now_us - start_us) / 1000) +
            " decoded+analytics input+PGIE+canonical tracker+preview all advanced after replacement; "
            "decoded_frames=" + std::to_string(decoded_frames) +
            " input_frames=" + std::to_string(input_frames) +
            " pgie_frames=" + std::to_string(pgie_frames) +
            " tracker_updates=" + std::to_string(tracker_updates) +
            " preview_frames=" + std::to_string(preview_frames) +
            " recovery_count=" + std::to_string(ctx->stale_recovery_count) +
            " generation=" + std::to_string(ctx->source_generation));
        continue;
      }
      if (now_us - ctx->stale_recovery_started_us > SOURCE_STALL_RECOVERY_CONFIRM_US) {
        ctx->stale_recovery_pending = false;
        fail(ctx->id + " source replacement produced no decoded frame within 15 seconds");
        return G_SOURCE_REMOVE;
      }
      continue;
    }
    const gint64 decoded_last = ctx->decoded.last_us.load();
    const gint64 input_last = ctx->input.last_us.load();
    const gint64 ingress_last = ctx->ingress.last_us.load();
    const bool decoded_stale = !decoded_last || now_us - decoded_last > SOURCE_STALL_RECOVERY_AFTER_US;
    const bool input_stale = !input_last || now_us - input_last > SOURCE_STALL_RECOVERY_AFTER_US;
    const bool ingress_stale = !ingress_last || now_us - ingress_last > SOURCE_STALL_RECOVERY_AFTER_US;
    if (decoded_stale && input_stale && ingress_stale) {
      if (!replace_stalled_source(ctx, now_us) && fatal) return G_SOURCE_REMOVE;
    }
  }
  return G_SOURCE_CONTINUE;
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
  }
#ifdef PGIE_ISOLATION
  else if (type == GST_MESSAGE_EOS && !stopping) {
    stopping = true;
    g_print("PGIE_ISOLATION EOS all file sources drained\n");
    g_main_loop_quit(loop);
  }
#else
  else if (type == GST_MESSAGE_EOS && !stopping) {
    fail("unexpected shared pipeline EOS");
  }
#endif
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
#endif

static bool write_monitoring_snapshot() {
  if (monitoring_path.empty()) return true;
  const gint64 now_us = g_get_monotonic_time();
  const double elapsed = monitoring_prev_us ? std::max(0.001, (now_us - monitoring_prev_us) / 1e6) : 0.5;
  const guint64 mono_ns = static_cast<guint64>(now_us) * 1000;
  const guint64 epoch_ms = static_cast<guint64>(g_get_real_time()) / 1000;
  const guint64 next_sequence = ++monitoring_sequence;
  std::ostringstream out;
  out << "{\"schema_version\":1,\"sequence\":" << next_sequence
      << ",\"generated_monotonic_ns\":" << mono_ns << ",\"generated_epoch_ms\":" << epoch_ms
      << ",\"runtime\":{\"status\":\"live\",\"camera_count\":6,\"rtsp_source_count\":6,"
         "\"ui_rtsp_extra\":0,\"shared_trt_workers\":0,\"detector_enabled\":true,"
         "\"uptime_sec\":" << std::max(0.0, (now_us - start_us) / 1e6)
      << ",\"warnings_total\":" << shared_warnings << ",\"errors_total\":" << shared_errors << "},\"cameras\":[";
  for (guint i = 0; i < 6; ++i) {
    auto *ctx = sources[i];
    if (i) out << ',';
    const guint64 source_frames = ctx->input.frames.load();
    const guint64 decoded_frames = ctx->decoded.frames.load();
    const guint64 infer_frames_now = detections[i].frames.load();
    const guint64 preview_frames_now = f5r_preview_frames[i].load();
    const double source_fps = (source_frames - monitoring_prev_input[i]) / elapsed;
    const double infer_fps = (infer_frames_now - monitoring_prev_infer[i]) / elapsed;
    const double preview_fps = (preview_frames_now - monitoring_prev_preview[i]) / elapsed;
    const gint64 last = ctx->input.last_us.load();
    const double age_ms = last ? std::max(0.0, (now_us - last) / 1000.0) : 1.0e9;
    const gint64 decoded_last = ctx->decoded.last_us.load();
    const double decoded_age_ms = decoded_last ? std::max(0.0, (now_us - decoded_last) / 1000.0) : 1.0e9;
    guint queue = 0;
    if (ctx->queue) g_object_get(ctx->queue, "current-level-buffers", &queue, nullptr);
    guint previous_high = ctx->queue_high_water.load();
    while (queue > previous_high && !ctx->queue_high_water.compare_exchange_weak(previous_high, queue)) {}
    const gint track_frame = tracker_canonical_frames[i].load();
    const guint count = tracker_canonical_count_for_frame(i, track_frame);
    out << "{\"camera_id\":\"CAM-" << std::setw(2) << std::setfill('0') << i + 1
        << "\",\"online\":" << ((ctx->decoder_seen.load() && age_ms < 2000) ? "true" : "false")
        << ",\"source_fps\":" << std::fixed << std::setprecision(3) << source_fps
        << ",\"decoded_frames\":" << decoded_frames
        << ",\"analytics_input_frames_total\":" << source_frames
        << ",\"pgie_frames_total\":" << infer_frames_now
        << ",\"canonical_tracker_updates_total\":" << source_tracker_updates[i].load()
        << ",\"canonical_frame_num\":" << track_frame
        << ",\"preview_frames_total\":" << preview_frames_now
        << ",\"decoded_age_ms\":" << decoded_age_ms
        << ",\"render_fps\":" << source_fps << ",\"infer_hz\":" << infer_fps
        << ",\"queue_depth\":" << queue << ",\"detector_drops\":"
        << std::max<gint64>(0, static_cast<gint64>(source_frames) - static_cast<gint64>(infer_frames_now))
        << ",\"current_box_count\":" << count << ",\"positive_inferences\":" << detections[i].persons.load()
        << ",\"detections_total\":" << detections[i].persons.load()
        << ",\"result_age_ms\":" << age_ms << ",\"infer_p95_ms\":null,\"render_gap_p95_ms\":null"
        << ",\"reconnects\":" << ctx->stale_recovery_count
        << ",\"source_stall_recovery_pending\":" << (ctx->stale_recovery_pending ? "true" : "false")
        << ",\"source_stall_recovery_enabled\":" << (source_stall_recovery_enabled ? "true" : "false")
        << ",\"copy_errors\":0,\"infer_errors\":" << detections[i].errors.load()
        << ",\"meta_errors\":" << tracker_counters[i].duplicate_ids.load()
        << ",\"pipeline_errors\":" << ctx->errors + shared_errors
        << ",\"warnings\":" << ctx->warnings + shared_warnings
        << ",\"preview_exported\":" << preview_frames_now
        << ",\"preview_errors\":" << f5r_preview_errors.load()
        << ",\"preview_fps\":" << preview_fps;
    guint64 preview_last_ns = 0;
    {
      std::lock_guard<std::mutex> guard(f5r_preview[i].mutex);
      preview_last_ns = f5r_preview[i].last_ns;
    }
    out << ",\"preview_age_ms\":" << (preview_last_ns ? std::max(0.0, (mono_ns - preview_last_ns) / 1e6) : 1.0e9)
        << ",\"queue_high_water\":" << ctx->queue_high_water.load() << '}';
    monitoring_prev_input[i] = source_frames;
    monitoring_prev_infer[i] = infer_frames_now;
    monitoring_prev_preview[i] = preview_frames_now;
  }
  out << "]}\n";
  const std::string data = out.str();
  const std::string temporary = monitoring_path + ".tmp." + std::to_string(getpid());
  int fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (fd < 0) return false;
  size_t offset = 0;
  while (offset < data.size()) {
    const ssize_t n = write(fd, data.data() + offset, data.size() - offset);
    if (n <= 0) { close(fd); unlink(temporary.c_str()); return false; }
    offset += static_cast<size_t>(n);
  }
  close(fd);
  if (rename(temporary.c_str(), monitoring_path.c_str()) != 0) { unlink(temporary.c_str()); return false; }
  monitoring_prev_us = now_us;
  return true;
}
static gboolean monitoring_tick(gpointer) {
  if (!write_monitoring_snapshot()) {
    fail("monitoring telemetry publish failed");
    return G_SOURCE_REMOVE;
  }
  return G_SOURCE_CONTINUE;
}

static gboolean tick(gpointer) {
  const gint64 now = g_get_monotonic_time();
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
  guint64 rendered = 0, dropped = 0;
#ifndef YOLO26_TRACKER
  GstStructure *stats = nullptr;
  g_object_get(sink_element, "stats", &stats, nullptr);
  if (stats) {
    gst_structure_get_uint64(stats, "rendered", &rendered);
    gst_structure_get_uint64(stats, "dropped", &dropped);
    gst_structure_free(stats);
  }
#endif
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
    // Candidate-run lifecycle: stop the event loop, then perform the existing
    // full pipeline NULL transition. Avoid injecting EOS into a shared tee
    // whose independent latest-only UI branch has its own downstream sink.
    g_main_loop_quit(loop);
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

static SourceCtx *load_source(GKeyFile *f, const char *group, int source_id) {
  auto *ctx = new SourceCtx();
  ctx->id = group;
  ctx->source_id = source_id;
  ctx->uri = b64_secret(f, group, "uri");
  ctx->username = b64_secret(f, group, "username");
  ctx->password = b64_secret(f, group, "password");
  ctx->latency_ms = g_key_file_get_integer(f, group, "latency_ms", nullptr);
  ctx->drop_on_latency = g_key_file_get_boolean(f, group, "drop_on_latency", nullptr);
  ctx->decoder_low_latency_mode = g_key_file_get_boolean(f, group, "decoder_low_latency_mode", nullptr);
  ctx->decoder_extra_surfaces = g_key_file_get_integer(f, group, "decoder_extra_surfaces", nullptr);
  g_mutex_init(&ctx->jitter_lock);
  const bool uri_ok = ctx->uri.rfind("rtsp://", 0) == 0
#ifdef PGIE_ISOLATION
      || ctx->uri.rfind("file://", 0) == 0
#endif
      ;
  if (!uri_ok || ctx->latency_ms < 0 || ctx->decoder_extra_surfaces > 55) {
    delete ctx;
    return nullptr;
  }
  return ctx;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    g_printerr("usage: cam-six-validator CONFIG DURATION\n");
    return 2;
  }
  duration_sec = std::atoi(argv[2]);
  if (duration_sec < 20) return 2;
  auto dimension_override = [](const char *key, guint fallback) {
    const char *raw = g_getenv(key);
    if (!raw || !*raw) return fallback;
    char *end = nullptr;
    const unsigned long value = std::strtoul(raw, &end, 10);
    if (!end || *end || value < 320 || value > 4096) {
      g_printerr("GROUP invalid %s (valid range 320..4096)\n", key);
      std::exit(2);
    }
    return static_cast<guint>(value);
  };
  v11_mux_width = dimension_override("V11_MUX_WIDTH", v11_mux_width);
  v11_mux_height = dimension_override("V11_MUX_HEIGHT", v11_mux_height);
#ifdef PGIE_ISOLATION
  auto env_uint = [](const char *key, guint fallback, guint minimum, guint maximum) {
    const char *raw = g_getenv(key);
    if (!raw || !*raw) return fallback;
    char *end = nullptr;
    const unsigned long value = std::strtoul(raw, &end, 10);
    if (!end || *end || value < minimum || value > maximum) {
      g_printerr("PGIE_ISOLATION invalid %s\n", key);
      std::exit(2);
    }
    return static_cast<guint>(value);
  };
  const guint source_count = env_uint("PGIE_ISOLATION_SOURCE_COUNT", 2, 2, 6);
  pgie_isolation_batch_size = env_uint("PGIE_ISOLATION_MUX_BATCH", source_count, source_count, 6);
  pgie_isolation_width = env_uint("PGIE_ISOLATION_MUX_WIDTH", 1920, 320, 4096);
  pgie_isolation_height = env_uint("PGIE_ISOLATION_MUX_HEIGHT", 1080, 240, 2160);
  pgie_isolation_memory_type = env_uint("PGIE_ISOLATION_MUX_MEMORY_TYPE", 0, 0, 3);
  const char *live_source = g_getenv("PGIE_ISOLATION_LIVE_SOURCE");
  pgie_isolation_live_source = live_source && std::strcmp(live_source, "0") != 0;
  const char *timeout = g_getenv("PGIE_ISOLATION_BATCHED_PUSH_TIMEOUT_US");
  if (timeout && *timeout) {
    char *end = nullptr;
    const long value = std::strtol(timeout, &end, 10);
    if (!end || *end || value < -1 || value > 50000) {
      g_printerr("PGIE_ISOLATION invalid PGIE_ISOLATION_BATCHED_PUSH_TIMEOUT_US\n");
      return 2;
    }
    mux_batched_push_timeout_us = static_cast<guint>(value);
  }
#endif
  const char *telemetry = g_getenv("V11_MONITORING_TELEMETRY_PATH");
  monitoring_path = telemetry && *telemetry ? telemetry : "/work/monitoring.json";
  const char *mux_timeout = g_getenv("V11_MUX_BATCHED_PUSH_TIMEOUT_US");
  if (mux_timeout && *mux_timeout) {
    char *end = nullptr;
    const long parsed = std::strtol(mux_timeout, &end, 10);
    if (!end || *end || parsed < 1000 || parsed > 50000) {
      g_printerr("GROUP invalid V11_MUX_BATCHED_PUSH_TIMEOUT_US (valid range 1000..50000)\n");
      return 2;
    }
    mux_batched_push_timeout_us = static_cast<guint>(parsed);
  }
  const char *source_recovery = g_getenv("V11_SOURCE_STALL_RECOVERY");
  if (source_recovery && *source_recovery) {
    if (std::strcmp(source_recovery, "0") && std::strcmp(source_recovery, "1")) {
      g_printerr("GROUP invalid V11_SOURCE_STALL_RECOVERY (valid values 0 or 1)\n");
      return 2;
    }
    source_stall_recovery_enabled = std::strcmp(source_recovery, "1") == 0;
  }
  g_print("SOURCE_STALL_RECOVERY enabled=%d stale_after_ms=2500 confirm_timeout_ms=15000 cooldown_ms=60000\n",
          int(source_stall_recovery_enabled));
  std::string recovery_test_error;
  if (!parse_recovery_test_plan(g_getenv("V11_TEST_SOURCE_STALL_PLAN"), &recovery_test_error)) {
    g_printerr("GROUP invalid controlled source-stall plan: %s\n", recovery_test_error.c_str());
    return 2;
  }
  if (!recovery_test_stalls.empty()) {
    std::string plan;
    for (const auto &stall : recovery_test_stalls) {
      if (!plan.empty()) plan += ',';
      plan += stall.camera_id + "@" + std::to_string(stall.at_us / G_USEC_PER_SEC);
    }
    g_print("SOURCE_STALL_TEST_PLAN enabled=1 schedule=%s mechanism=source_set_null watchdog_owned_recovery=1\n",
        plan.c_str());
  }

  GKeyFile *f = g_key_file_new();
  GError *err = nullptr;
  if (!g_key_file_load_from_file(f, argv[1], G_KEY_FILE_NONE, &err)) {
    g_printerr("GROUP config read failed\n");
    if (err) g_error_free(err);
    return 2;
  }

#ifdef PGIE_ISOLATION
  SourceCtx *cam01 = load_source(f, "CAM-01", 0);
  if (source_count == 2) {
    SourceCtx *cam04 = load_source(f, "CAM-04", 1);
    if (!cam01 || !cam04) {
      g_printerr("PGIE_ISOLATION two-source config invalid\n");
      return 2;
    }
    sources = {cam01, cam04};
  } else {
    SourceCtx *cam02 = load_source(f, "CAM-02", 1);
    SourceCtx *cam03 = load_source(f, "CAM-03", 2);
    SourceCtx *cam04 = load_source(f, "CAM-04", 3);
    SourceCtx *cam05 = load_source(f, "CAM-05", 4);
    SourceCtx *cam06 = load_source(f, "CAM-06", 5);
    if (!cam01 || !cam02 || !cam03 || !cam04 || !cam05 || !cam06) {
      g_printerr("PGIE_ISOLATION six-source config invalid\n");
      return 2;
    }
    sources = {cam01, cam02, cam03, cam04, cam05, cam06};
  }
#else
  SourceCtx *cam01 = load_source(f, "CAM-01", 0);
  SourceCtx *cam02 = load_source(f, "CAM-02", 1);
  SourceCtx *cam03 = load_source(f, "CAM-03", 2);
  SourceCtx *cam04 = load_source(f, "CAM-04", 3);
  SourceCtx *cam05 = load_source(f, "CAM-05", 4);
  SourceCtx *cam06 = load_source(f, "CAM-06", 5);
  if (!cam01 || !cam02 || !cam03 || !cam04 || !cam05 || !cam06) {
    g_printerr("GROUP camera config invalid\n");
    return 2;
  }
  sources = {cam01, cam02, cam03, cam04, cam05, cam06};
#endif

  gchar *ival = g_key_file_get_string(f, "validation", "interrupt_camera", nullptr);
  if (ival) { interrupt_camera = ival; g_free(ival); }
  interrupt_at = g_key_file_get_integer(f, "validation", "interrupt_at", nullptr);
  interrupt_seconds = g_key_file_get_integer(f, "validation", "interrupt_seconds", nullptr);
  preview_enabled = g_key_file_get_boolean(f, "validation", "preview_enabled", nullptr);
  preview_port = g_key_file_get_integer(f, "validation", "preview_port", nullptr);
  g_key_file_free(f);

  gst_init(nullptr, nullptr);
  loop = g_main_loop_new(nullptr, FALSE);
  pipeline = gst_pipeline_new("CAM-01-CAM-02-CAM-03-CAM-04-CAM-05-CAM-06-validation");
  mux = make("nvstreammux", "mux");
#ifdef PGIE_ISOLATION
  g_object_set(
      mux,
      "batch-size", pgie_isolation_batch_size,
      "width", pgie_isolation_width,
      "height", pgie_isolation_height,
      "live-source", pgie_isolation_live_source,
      "batched-push-timeout", mux_batched_push_timeout_us,
      "sync-inputs", FALSE,
      "attach-sys-ts", TRUE,
      "nvbuf-memory-type", pgie_isolation_memory_type,
      "cache-buffer", FALSE,
      "drop-pipeline-eos", FALSE,
      nullptr);
  event(nullptr, "PGIE_ISOLATION_MUX", "batch=" + std::to_string(pgie_isolation_batch_size) +
      " width=" + std::to_string(pgie_isolation_width) + " height=" + std::to_string(pgie_isolation_height) +
      " live_source=" + std::to_string(pgie_isolation_live_source) +
      " timeout_us=" + std::to_string(static_cast<gint32>(mux_batched_push_timeout_us)) +
      " memory_type=" + std::to_string(pgie_isolation_memory_type));
#else
  g_object_set(
      mux,
      "batch-size", 6u,
      "width", v11_mux_width,
      "height", v11_mux_height,
      "live-source", TRUE,
      "batched-push-timeout", mux_batched_push_timeout_us,
      "sync-inputs", FALSE,
      "attach-sys-ts", TRUE,
      "nvbuf-memory-type", 2,
      "cache-buffer", FALSE,
      "drop-pipeline-eos", TRUE,
      nullptr);
  event(nullptr, "MUX_CONFIG", "batch=6 width=" + std::to_string(v11_mux_width) + " height=" + std::to_string(v11_mux_height) + " live_source=1 batched_push_timeout_us=" +
      std::to_string(mux_batched_push_timeout_us) + " sync_inputs=0");
#endif

  for (auto *ctx : sources) {
    const std::string queue_name = ctx->id + "-analytics-queue";
    ctx->queue = make("queue", queue_name.c_str());
    g_object_set(ctx->queue, "max-size-buffers", 12u, "max-size-bytes", 0u,
                 "max-size-time", guint64(0), "leaky", 0, nullptr);
    const std::string tee_name = ctx->id + "-decoded-frame-tee";
    ctx->tee = make("tee", tee_name.c_str());
#ifdef PGIE_ISOLATION
    const char *pace_files = g_getenv("PGIE_ISOLATION_REALTIME_PACE");
    if (pace_files && std::strcmp(pace_files, "1") == 0 &&
        ctx->uri.rfind("file://", 0) == 0) {
      const std::string pacer_name = ctx->id + "-direct-file-clock-pacer";
      ctx->replay_pacer = make("identity", pacer_name.c_str());
      g_object_set(ctx->replay_pacer, "sync", TRUE, "silent", TRUE, nullptr);
      if (!gst_element_link(ctx->replay_pacer, ctx->tee)) {
        fail(ctx->id + " direct-file replay pacer -> decoded tee link failed");
        return 2;
      }
      event(ctx, "DIRECT_FILE_PACING", "identity sync=1; source PTS preserved; file-only diagnostic mode");
    }
#endif
    if (!f5r_tee_link(ctx->tee, ctx->queue)) {
      fail(ctx->id + " decoded tee -> analytics queue link failed");
      return 2;
    }
    probe(ctx->queue, "sink", &ctx->ingress);
    probe(ctx->queue, "src", &ctx->input);

    ctx->source = create_source_element(ctx);
    if (!ctx->source) return 2;

    const std::string sink_name = "sink_" + std::to_string(ctx->source_id);
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
  g_object_set(tiler, "rows", 3u, "columns", 2u, "width", 1920u, "height", 1620u, nullptr);
#ifdef YOLO26_TRACKER
  sink_element = make("fakesink", "analytics-output-sink");
  g_object_set(sink_element, "sync", FALSE, "async", FALSE, "enable-last-sample", FALSE, nullptr);
#else
  GstElement *conv = make("nvvideoconvert", "converter");
  GstElement *capsfilter = make("capsfilter", "nvmm-caps");
  GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12,width=1920,height=1620");
  g_object_set(capsfilter, "caps", caps, nullptr);
  gst_caps_unref(caps);
  encoder = make("nvv4l2h264enc", "NVENC");
  g_object_set(encoder, "bitrate", 20000000u, "iframeinterval", 40u, nullptr);
  GstElement *encoded_tee = make("tee", "encoded-tee");

  GstElement *file_queue = make("queue", "file-queue");
  g_object_set(file_queue, "max-size-buffers", 12u, "max-size-bytes", 0u,
               "max-size-time", guint64(0), "leaky", 0, "silent", TRUE, nullptr);
  GstElement *file_parse = make("h264parse", "output-parse");
  GstElement *filemux = make("matroskamux", "filemux");
  sink_element = make("filesink", "sink");
  g_object_set(sink_element, "location", "/work/CAM-01_CAM-02_CAM-03_CAM-04_CAM-05_CAM-06.mkv", "sync", FALSE,
               "async", FALSE, "enable-last-sample", FALSE, nullptr);
#endif

#ifdef YOLO26_PERSON
#ifdef YOLO26_TRACKER
  detection_link(mux, tiler, nullptr);
#else
  detection_link(mux, tiler, conv);
#endif
#else
  link(mux, tiler);
  link(tiler, conv);
#endif
#ifdef YOLO26_TRACKER
  link(tiler, sink_element);
  probe(tiler, "src", &output_counter);
#else
  link(conv, capsfilter);
  link(capsfilter, encoder);
  link(encoder, encoded_tee);

  link(encoded_tee, file_queue);
  link(file_queue, file_parse);
  link(file_parse, filemux);
  link(filemux, sink_element);
#endif

#ifndef YOLO26_TRACKER
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

    link(encoded_tee, preview_queue);
    link(preview_queue, preview_parse);
    link(preview_parse, preview_caps);
    link(preview_caps, tsmux);
    link(tsmux, udp);
  }

  probe(encoder, "src", &output_counter);
#endif

#ifndef YOLO26_PERSON
  event(nullptr, "GRAPH",
        std::string("CAM-01+CAM-02+CAM-03+CAM-04+CAM-05+CAM-06 nvurisrcbin/NVDEC->NVMM->queues->nvstreammux(batch=6,2560x1440,live,50000us,sync-inputs=false)->tiler(3x2,1920x1620)->NVENC->record") +
        (preview_enabled ? "+UDP-preview inference=0" : " inference=0"));

#endif
  GstBus *bus = gst_element_get_bus(pipeline);
  gst_bus_add_watch(bus, bus_message, nullptr);
  gst_object_unref(bus);
  g_unix_signal_add(SIGINT, stop_signal, nullptr);
  g_unix_signal_add(SIGTERM, stop_signal, nullptr);

  start_us = previous_us = g_get_monotonic_time();
  previous_cpu = cpu_seconds();
  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    fail("initial PLAYING failed");
  }
#ifndef PGIE_ISOLATION
  g_timeout_add(500, monitoring_tick, nullptr);
  if (source_stall_recovery_enabled) g_timeout_add(500, source_stall_recovery_tick, nullptr);
#endif
  g_timeout_add_seconds(5, tick, nullptr);
  if (!fatal) g_main_loop_run(loop);

  stopping = true;
  gst_element_set_state(pipeline, GST_STATE_NULL);
#ifdef YOLO26_PERSON
  lowconf_verifier_shutdown();
  detection_stats();
#ifndef PGIE_ISOLATION
  f5r_preview_close();
#endif
#ifdef YOLO26_TRACKER
  tracker_stats();
  if (tracker_evidence) std::fclose(tracker_evidence);
  if (canonical_evidence) { std::fclose(canonical_evidence); canonical_evidence = nullptr; }
  if (lowconf_fallback_evidence) { std::fclose(lowconf_fallback_evidence); lowconf_fallback_evidence = nullptr; }
#endif
  if (detection_evidence) std::fclose(detection_evidence);
  if (proposal_duplicate_evidence) { std::fclose(proposal_duplicate_evidence); proposal_duplicate_evidence = nullptr; }
  if (frame_evidence) std::fclose(frame_evidence);
  if (lowconf_candidate_evidence) { std::fclose(lowconf_candidate_evidence); lowconf_candidate_evidence = nullptr; }
#ifdef PGIE_ISOLATION
  if (direct_file_frame_evidence) {
    std::fclose(direct_file_frame_evidence);
    direct_file_frame_evidence = nullptr;
  }
  if (direct_file_frame_write_failed.load()) fail("direct-file source frame evidence write failed");
#endif
#endif
  gst_object_unref(pipeline);

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
