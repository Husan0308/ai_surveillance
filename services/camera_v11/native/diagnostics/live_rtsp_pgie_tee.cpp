#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static GMainLoop *loop = nullptr;
static FILE *rows = nullptr;
static std::atomic<guint64> decode_frames{0};
static std::string video_caps;

static std::string read64(GKeyFile *key, const char *name) {
  GError *error = nullptr;
  gchar *encoded = g_key_file_get_string(key, "source", name, &error);
  if (error) { g_error_free(error); throw std::runtime_error(std::string("missing source field: ") + name); }
  gsize length = 0;
  guchar *decoded = g_base64_decode(encoded, &length);
  std::string value(reinterpret_cast<char *>(decoded), length);
  g_free(decoded); g_free(encoded);
  return value;
}

static GstElement *make(const char *factory, const std::string &name) {
  GstElement *element = gst_element_factory_make(factory, name.c_str());
  if (!element) throw std::runtime_error(std::string("missing GStreamer element: ") + factory);
  return element;
}

static void link_or_throw(GstElement *a, GstElement *b) {
  if (!gst_element_link(a, b)) throw std::runtime_error(std::string("link failed: ") +
      GST_ELEMENT_NAME(a) + " -> " + GST_ELEMENT_NAME(b));
}

static GstPadProbeReturn decoder_caps(GstPad *, GstPadProbeInfo *info, gpointer) {
  if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
    GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
    if (GST_EVENT_TYPE(event) == GST_EVENT_CAPS) {
      GstCaps *caps = nullptr; gst_event_parse_caps(event, &caps);
      gchar *text = gst_caps_to_string(caps); video_caps = text ? text : ""; g_free(text);
      g_print("DECODE_CAPS %s\n", video_caps.c_str());
    }
  }
  if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) ++decode_frames;
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn pgie_probe(GstPad *, GstPadProbeInfo *info, gpointer) {
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer) return GST_PAD_PROBE_OK;
  NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(buffer);
  if (!batch) return GST_PAD_PROBE_OK;
  for (NvDsMetaList *fn = batch->frame_meta_list; fn; fn = fn->next) {
    auto *frame = static_cast<NvDsFrameMeta *>(fn->data);
    if (!frame || frame->source_id != 0) continue;
    for (NvDsMetaList *on = frame->obj_meta_list; on; on = on->next) {
      auto *obj = static_cast<NvDsObjectMeta *>(on->data);
      if (!obj || obj->class_id != 0) continue;
      const auto &r = obj->rect_params;
      std::fprintf(rows,
        "{\"frame\":%u,\"pts_ns\":%lu,\"class_id\":%d,\"confidence\":%.7f,\"bbox\":[%.3f,%.3f,%.3f,%.3f],\"mux_width\":1920,\"mux_height\":1080}\n",
        frame->frame_num, frame->buf_pts, obj->class_id, obj->confidence,
        r.left, r.top, r.width, r.height);
    }
  }
  std::fflush(rows);
  return GST_PAD_PROBE_OK;
}

static gboolean stop_timer(gpointer pipeline) {
  gst_element_send_event(GST_ELEMENT(pipeline), gst_event_new_eos());
  return G_SOURCE_REMOVE;
}

int main(int argc, char **argv) {
  if (argc != 5) { g_printerr("usage: live-tee CAMERA_INI DURATION RAW_MP4 PGIE_CONFIG\n"); return 2; }
  const int duration = std::atoi(argv[2]);
  if (duration < 20) return 2;
  try {
    gst_init(&argc, &argv);
    GError *error = nullptr;
    GKeyFile *key = g_key_file_new();
    if (!g_key_file_load_from_file(key, argv[1], G_KEY_FILE_NONE, &error))
      throw std::runtime_error("cannot read source config");
    const std::string uri = read64(key, "uri"), username = read64(key, "username"), password = read64(key, "password");
    const int latency = g_key_file_get_integer(key, "source", "latency_ms", nullptr);
    const bool drop = g_key_file_get_boolean(key, "source", "drop_on_latency", nullptr);
    g_key_file_unref(key);
    rows = std::fopen("/work/live-detections.jsonl", "w");
    if (!rows) throw std::runtime_error("cannot create detection evidence");
    std::setvbuf(rows, nullptr, _IOLBF, 0);

    GstElement *pipeline = gst_pipeline_new("single-owner-live-tee-pgie");
    GstElement *source = make("rtspsrc", "one-rtsp-owner");
    GstElement *depay = make("rtph264depay", "rtp-depay");
    GstElement *tee = make("tee", "encoded-tee");
    GstElement *record_q = make("queue", "record-q");
    GstElement *record_parse = make("h264parse", "record-parse");
    GstElement *mux_file = make("qtmux", "streamcopy-mux");
    GstElement *file_sink = make("filesink", "streamcopy-file");
    GstElement *decode_q = make("queue", "decode-q");
    GstElement *decode_parse = make("h264parse", "decode-parse");
    GstElement *decoder = make("nvv4l2decoder", "nvidia-decoder");
    GstElement *raw_tee = make("tee", "decoded-frame-tee");
    GstElement *mux = make("nvstreammux", "six-slot-mux");
    GstElement *pgie = make("nvinfer", "frozen-yolo26m");
    GstElement *sink = make("fakesink", "no-tracker-sink");
    std::vector<GstElement *> elements = {source, depay, tee, record_q, record_parse, mux_file,
        file_sink, decode_q, decode_parse, decoder, raw_tee, mux, pgie, sink};
    for (auto *e : elements) gst_bin_add(GST_BIN(pipeline), e);
    g_object_set(source, "location", uri.c_str(), "user-id", username.c_str(),
        "user-pw", password.c_str(), "protocols", 4, "latency", latency,
        "drop-on-latency", drop, nullptr);
    g_object_set(file_sink, "location", argv[3], "sync", FALSE, nullptr);
    g_object_set(mux, "batch-size", 6u, "width", 1920u, "height", 1080u,
        "live-source", TRUE, "batched-push-timeout", 1000u, "sync-inputs", FALSE,
        "attach-sys-ts", TRUE, "nvbuf-memory-type", 2u, "drop-pipeline-eos", FALSE, nullptr);
    g_object_set(pgie, "config-file-path", argv[4], "batch-size", 6u, nullptr);
    g_object_set(sink, "sync", FALSE, "async", FALSE, nullptr);
    g_object_set(record_parse, "config-interval", -1, nullptr);
    link_or_throw(depay, tee);
    link_or_throw(tee, record_q); link_or_throw(record_q, record_parse);
    link_or_throw(record_parse, mux_file); link_or_throw(mux_file, file_sink);
    link_or_throw(tee, decode_q); link_or_throw(decode_q, decode_parse);
    link_or_throw(decode_parse, decoder); link_or_throw(decoder, raw_tee);
    link_or_throw(mux, pgie); link_or_throw(pgie, sink);
    GstPad *decoder_src = gst_element_get_static_pad(decoder, "src");
    gst_pad_add_probe(decoder_src, static_cast<GstPadProbeType>(GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM | GST_PAD_PROBE_TYPE_BUFFER), decoder_caps, nullptr, nullptr);
    gst_object_unref(decoder_src);
    GstPad *pgie_src = gst_element_get_static_pad(pgie, "src");
    gst_pad_add_probe(pgie_src, GST_PAD_PROBE_TYPE_BUFFER, pgie_probe, nullptr, nullptr);
    gst_object_unref(pgie_src);
    for (guint i = 0; i < 6; ++i) {
      GstElement *q = make("queue", "duplicate-queue-" + std::to_string(i));
      gst_bin_add(GST_BIN(pipeline), q);
      link_or_throw(raw_tee, q);
      GstPad *src_pad = gst_element_get_static_pad(q, "src");
      GstPad *mux_pad = gst_element_request_pad_simple(mux, ("sink_" + std::to_string(i)).c_str());
      if (!mux_pad || gst_pad_link(src_pad, mux_pad) != GST_PAD_LINK_OK)
        throw std::runtime_error("cannot attach duplicate batch slot");
      gst_object_unref(src_pad); gst_object_unref(mux_pad);
    }
    g_signal_connect(source, "pad-added", G_CALLBACK(+[](GstElement *, GstPad *pad, gpointer data) {
      auto *depay_element = static_cast<GstElement *>(data);
      GstCaps *caps = gst_pad_get_current_caps(pad);
      if (!caps) caps = gst_pad_query_caps(pad, nullptr);
      const GstStructure *s = caps && gst_caps_get_size(caps) ? gst_caps_get_structure(caps, 0) : nullptr;
      const char *media = s ? gst_structure_get_string(s, "media") : nullptr;
      const char *encoding = s ? gst_structure_get_string(s, "encoding-name") : nullptr;
      if (media && std::strcmp(media, "video") == 0 && encoding && std::strcmp(encoding, "H264") == 0) {
        GstPad *target = gst_element_get_static_pad(depay_element, "sink");
        if (!gst_pad_is_linked(target) && gst_pad_link(pad, target) == GST_PAD_LINK_OK)
          g_print("VIDEO_RTP_LINK codec=H264\n");
        gst_object_unref(target);
      }
      if (caps) gst_caps_unref(caps);
    }), depay);

    GstBus *bus = gst_element_get_bus(pipeline); gst_bus_add_signal_watch(bus);
    g_signal_connect(bus, "message", G_CALLBACK(+[](GstBus *, GstMessage *m, gpointer) {
      if (GST_MESSAGE_TYPE(m) == GST_MESSAGE_ERROR) {
        GError *e = nullptr; gchar *debug = nullptr; gst_message_parse_error(m, &e, &debug);
        g_printerr("PIPELINE_ERROR source=%s message=%s\n", GST_OBJECT_NAME(m->src), e ? e->message : "unknown");
        if (e) g_error_free(e); g_free(debug); g_main_loop_quit(loop);
      } else if (GST_MESSAGE_TYPE(m) == GST_MESSAGE_EOS) {
        g_print("PIPELINE_EOS decoded_frames=%lu\n", decode_frames.load()); g_main_loop_quit(loop);
      }
    }), nullptr);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    loop = g_main_loop_new(nullptr, FALSE);
    g_timeout_add_seconds(duration, stop_timer, pipeline);
    g_print("RUNTIME mux=1920x1080 batch=6 live=1 timeout_us=1000 detector=YOLO26m tracker=OFF identity=OFF rtsp_owners=1\n");
    g_main_loop_run(loop);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_bus_remove_signal_watch(bus); gst_object_unref(bus);
    gst_object_unref(pipeline); g_main_loop_unref(loop); loop = nullptr;
    std::fclose(rows); rows = nullptr;
    return 0;
  } catch (const std::exception &e) {
    g_printerr("LIVE_TEE_ERROR %s\n", e.what());
    if (rows) std::fclose(rows);
    return 2;
  }
}
