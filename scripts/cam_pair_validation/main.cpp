// CAM-01 + CAM-02 DeepStream 9.1 validation. No inference, tracker, OpenCV or CPU pixel copies.
#include <gst/gst.h>
#include <glib-unix.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>

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
  std::string uri;
  std::string username;
  std::string password;
  int latency_ms = 300;
  GstElement *source = nullptr;
  GstElement *queue = nullptr;
  GstElement *jitterbuffer = nullptr;
  GMutex jitter_lock;
  Counter input;
  std::atomic_bool decoder_seen{false};
  guint errors = 0;
  guint warnings = 0;
  guint64 previous_frames = 0;
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
  safe_print((ctx ? ctx->id : std::string("PAIR")) + " " + kind, detail);
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
  gst_caps_unref(caps);
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

static SourceCtx *peer_source(SourceCtx *target) {
  for (auto *ctx : sources) if (ctx != target) return ctx;
  return nullptr;
}

static gboolean resume_interrupted(gpointer data) {
  auto *target = static_cast<SourceCtx*>(data);
  target_resume_frames = target->input.frames.load();
  SourceCtx *peer = peer_source(target);
  peer_resume_frames = peer ? peer->input.frames.load() : 0;
  event(target, "ISOLATION", "restoring source to PLAYING");
  if (gst_element_set_state(target->source, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    fail("failed to restore isolated source");
  } else {
    interruption_resumed = true;
  }
  return G_SOURCE_REMOVE;
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
    const guint64 frames = ctx->input.frames.load();
    const double age = ctx->input.last_us ? (now - ctx->input.last_us) / 1e6 : elapsed;
    g_print(
      "%s STATS elapsed=%.3f input=%lu fps=%.3f age=%.3f pts_ns=%lu "
      "pts_backwards=%lu pts_duplicates=%lu max_gap_ms=%.3f queue=%u queue_ms=%.3f "
      "rtp_lost=%lu rtp_late=%lu rtp_pushed=%lu errors=%u warnings=%u decoder=%d\n",
      ctx->id.c_str(), elapsed, frames, (frames - ctx->previous_frames) / dt, age,
      ctx->input.pts.load(), ctx->input.backwards.load(), ctx->input.duplicates.load(),
      ctx->input.max_gap_us / 1000.0, q, qtime / 1e6, lost, late, pushed,
      ctx->errors, ctx->warnings, int(ctx->decoder_seen.load()));
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
  const double out_age = output_counter.last_us ? (now - output_counter.last_us) / 1e6 : elapsed;
  g_print(
    "PAIR STATS elapsed=%.3f output=%lu fps=%.3f age=%.3f pts_ns=%lu "
    "pts_backwards=%lu pts_duplicates=%lu max_gap_ms=%.3f rss_mib=%.3f cpu_pct=%.3f "
    "rendered=%lu dropped=%lu shared_errors=%u shared_warnings=%u\n",
    elapsed, out, (out - previous_output) / dt, out_age, output_counter.pts.load(),
    output_counter.backwards.load(), output_counter.duplicates.load(),
    output_counter.max_gap_us / 1000.0,
    resident * sysconf(_SC_PAGESIZE) / 1048576.0,
    100 * (cpu - previous_cpu) / dt, rendered, dropped, shared_errors, shared_warnings);

  previous_output = out;
  previous_us = now;
  previous_cpu = cpu;

  if (interrupt_camera != "none" && !interruption_started && elapsed >= interrupt_at) {
    SourceCtx *target = find_source(interrupt_camera);
    SourceCtx *peer = peer_source(target);
    if (!target || !peer) {
      fail("invalid isolation target");
      return G_SOURCE_REMOVE;
    }
    target_start_frames = target->input.frames.load();
    peer_start_frames = peer->input.frames.load();
    interruption_started = true;
    event(target, "ISOLATION", "setting only this source to NULL");
    gst_element_set_state(target->source, GST_STATE_NULL);
    g_timeout_add_seconds(interrupt_seconds, resume_interrupted, target);
  }

  if (interruption_resumed && !interruption_reported) {
    SourceCtx *target = find_source(interrupt_camera);
    SourceCtx *peer = peer_source(target);
    if (target && peer && target->input.frames.load() >= target_resume_frames + 100) {
      const guint64 peer_gain = peer->input.frames.load() - peer_start_frames;
      const guint64 target_off_gain = target_resume_frames - target_start_frames;
      const bool ok = peer_gain >= guint64(interrupt_seconds * 15) && target_off_gain <= 5;
      g_print(
        "PAIR ISOLATION_SUMMARY target=%s peer=%s outage_sec=%d peer_frames_during=%lu "
        "target_frames_during=%lu target_recovered_frames=%lu status=%s\n",
        target->id.c_str(), peer->id.c_str(), interrupt_seconds, peer_gain, target_off_gain,
        target->input.frames.load() - target_resume_frames, ok ? "PASS" : "BLOCKED");
      interruption_reported = true;
    }
  }

  for (auto *ctx : sources) {
    const double age = ctx->input.last_us ? (now - ctx->input.last_us) / 1e6 : elapsed;
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

static SourceCtx *load_source(GKeyFile *f, const char *group, int source_id) {
  auto *ctx = new SourceCtx();
  ctx->id = group;
  ctx->source_id = source_id;
  ctx->uri = b64_secret(f, group, "uri");
  ctx->username = b64_secret(f, group, "username");
  ctx->password = b64_secret(f, group, "password");
  ctx->latency_ms = g_key_file_get_integer(f, group, "latency_ms", nullptr);
  g_mutex_init(&ctx->jitter_lock);
  if (ctx->uri.rfind("rtsp://", 0) != 0 || ctx->latency_ms < 100) {
    delete ctx;
    return nullptr;
  }
  return ctx;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    g_printerr("usage: cam-pair-validator CONFIG DURATION\n");
    return 2;
  }
  duration_sec = std::atoi(argv[2]);
  if (duration_sec < 20) return 2;

  GKeyFile *f = g_key_file_new();
  GError *err = nullptr;
  if (!g_key_file_load_from_file(f, argv[1], G_KEY_FILE_NONE, &err)) {
    g_printerr("PAIR config read failed\n");
    if (err) g_error_free(err);
    return 2;
  }

  SourceCtx *cam01 = load_source(f, "CAM-01", 0);
  SourceCtx *cam02 = load_source(f, "CAM-02", 1);
  if (!cam01 || !cam02) {
    g_printerr("PAIR camera config invalid\n");
    return 2;
  }
  sources = {cam01, cam02};

  gchar *ival = g_key_file_get_string(f, "validation", "interrupt_camera", nullptr);
  if (ival) { interrupt_camera = ival; g_free(ival); }
  interrupt_at = g_key_file_get_integer(f, "validation", "interrupt_at", nullptr);
  interrupt_seconds = g_key_file_get_integer(f, "validation", "interrupt_seconds", nullptr);
  g_key_file_free(f);

  gst_init(nullptr, nullptr);
  loop = g_main_loop_new(nullptr, FALSE);
  pipeline = gst_pipeline_new("CAM-01-CAM-02-validation");
  mux = make("nvstreammux", "mux");
  g_object_set(
      mux,
      "batch-size", 2u,
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
    const std::string source_name = ctx->id + "-source";
    const std::string queue_name = ctx->id + "-queue";
    ctx->queue = make("queue", queue_name.c_str());
    g_object_set(ctx->queue, "max-size-buffers", 12u, "max-size-bytes", 0u,
                 "max-size-time", guint64(0), "leaky", 0, nullptr);
    probe(ctx->queue, "src", &ctx->input);

    ctx->source = make("nvurisrcbin", source_name.c_str());
    g_signal_connect(ctx->source, "deep-element-added", G_CALLBACK(child_added), ctx);
    g_signal_connect(ctx->source, "pad-added", G_CALLBACK(dynamic_pad), ctx);
    g_object_set(
        ctx->source,
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
  g_object_set(tiler, "rows", 1u, "columns", 2u, "width", 2560u, "height", 720u, nullptr);
  GstElement *conv = make("nvvideoconvert", "converter");
  GstElement *capsfilter = make("capsfilter", "nvmm-caps");
  GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12,width=2560,height=720");
  g_object_set(capsfilter, "caps", caps, nullptr);
  gst_caps_unref(caps);
  encoder = make("nvv4l2h264enc", "NVENC");
  g_object_set(encoder, "bitrate", 12000000u, "iframeinterval", 40u, nullptr);
  GstElement *parse = make("h264parse", "output-parse");
  GstElement *filemux = make("matroskamux", "filemux");
  sink_element = make("filesink", "sink");
  g_object_set(sink_element, "location", "/work/CAM-01_CAM-02.mkv", "sync", FALSE,
               "async", FALSE, "enable-last-sample", FALSE, nullptr);

  link(mux, tiler);
  link(tiler, conv);
  link(conv, capsfilter);
  link(capsfilter, encoder);
  link(encoder, parse);
  link(parse, filemux);
  link(filemux, sink_element);
  probe(encoder, "src", &output_counter);

  event(nullptr, "GRAPH",
        "CAM-01+CAM-02 nvurisrcbin/NVDEC->NVMM->queues->nvstreammux(batch=2,2560x1440,live,50000us,sync-inputs=false)->tiler(1x2)->NVENC->CAM-01_CAM-02.mkv inference=0");

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
  g_timeout_add_seconds(5, tick, nullptr);
  if (!fatal) g_main_loop_run(loop);

  stopping = true;
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);

  for (auto *ctx : sources) {
    if (ctx->jitterbuffer) gst_object_unref(ctx->jitterbuffer);
    g_print("%s END input=%lu hardware_decoder=%d errors=%u warnings=%u pts_backwards=%lu pts_duplicates=%lu\n",
            ctx->id.c_str(), ctx->input.frames.load(), int(ctx->decoder_seen.load()),
            ctx->errors, ctx->warnings, ctx->input.backwards.load(), ctx->input.duplicates.load());
  }
  g_print("PAIR END output=%lu fatal=%d shared_errors=%u shared_warnings=%u isolation=%s\n",
          output_counter.frames.load(), int(fatal.load()), shared_errors, shared_warnings,
          interruption_started ? (interruption_reported ? "completed" : "incomplete") : "not-requested");

  bool ok = !fatal && output_counter.frames.load() > 0;
  for (auto *ctx : sources) ok = ok && ctx->decoder_seen && ctx->input.frames.load() > 0;
  if (interruption_started) ok = ok && interruption_reported;

  for (auto *ctx : sources) delete ctx;
  g_main_loop_unref(loop);
  return ok ? 0 : 1;
}
