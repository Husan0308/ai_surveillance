// Included only in the detection build of the frozen six-camera native runner.
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#include <cmath>
#include <dlfcn.h>

struct DetectionCounter {
  std::atomic<guint64> frames{0}, persons{0}, batches{0}, errors{0}, last_unix_ms{0};
};
static DetectionCounter detections[6];
static std::atomic<guint64> infer_batches{0}, infer_frames{0}, infer_errors{0};
static std::atomic<guint64> infer_us{0}, infer_timed{0}, infer_max_us{0};
using ParserCount = unsigned long long (*)();
static ParserCount parser_errors, parser_rejected, parser_calls, parser_proposals;
static FILE *detection_evidence = nullptr, *frame_evidence = nullptr;
static GQuark infer_stamp;

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
    float confidence, float left, float top, float width, float height) {
  char line[512];
  const int n = std::snprintf(
      line, sizeof(line),
      "{\"source_id\":%u,\"frame\":%d,\"pts_ns\":%lu,"
      "\"class_id\":%d,\"confidence\":%.6f,"
      "\"box\":[%.3f,%.3f,%.3f,%.3f]}\n",
      source_id, frame_num, pts_ns, class_id, confidence,
      left, top, width, height);
  if (n <= 0 || static_cast<size_t>(n) >= sizeof(line)) return false;
  return write_all_json_line(file, line, static_cast<size_t>(n));
}

static bool write_frame_json(
    FILE *file, guint source_id, gint frame_num, guint64 pts_ns,
    guint64 batch_pts_ns, guint persons) {
  char line[384];
  const int n = std::snprintf(
      line, sizeof(line),
      "{\"source_id\":%u,\"frame\":%d,\"pts_ns\":%lu,"
      "\"batch_pts_ns\":%lu,\"persons\":%u}\n",
      source_id, frame_num, pts_ns, batch_pts_ns, persons);
  if (n <= 0 || static_cast<size_t>(n) >= sizeof(line)) return false;
  return write_all_json_line(file, line, static_cast<size_t>(n));
}

static GstPadProbeReturn infer_enter(GstPad*, GstPadProbeInfo *info, gpointer) {
  GstBuffer *b = GST_PAD_PROBE_INFO_BUFFER(info);
  if (b) {
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
  }
  ++infer_batches;
  for (auto *node = meta->frame_meta_list; node; node = node->next) {
    auto *frame = static_cast<NvDsFrameMeta*>(node->data);
    if (frame->source_id >= 6 || !frame->bInferDone) {
      ++infer_errors; fail("YOLO unknown source or frame not inferred"); return GST_PAD_PROBE_DROP;
    }
    auto &c = detections[frame->source_id];
    ++c.frames; ++c.batches; ++infer_frames;
    guint count = 0;
    for (auto *obj_node = frame->obj_meta_list; obj_node; obj_node = obj_node->next) {
      auto *obj = static_cast<NvDsObjectMeta*>(obj_node->data);
      auto &r = obj->rect_params;
      if (obj->class_id != 0 || obj->unique_component_id != 1 ||
          !std::isfinite(obj->confidence) || obj->confidence < .25f || obj->confidence > 1 ||
          !std::isfinite(r.left) || !std::isfinite(r.top) || !std::isfinite(r.width) || !std::isfinite(r.height) ||
          r.width <= 0 || r.height <= 0 || r.left < 0 || r.top < 0 ||
          r.left+r.width > 2560.1f || r.top+r.height > 1440.1f) {
        ++c.errors; ++infer_errors; fail("YOLO invalid/non-person object metadata"); return GST_PAD_PROBE_DROP;
      }
      ++count;
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
      if (!write_detection_json(
              detection_evidence, frame->source_id, frame->frame_num, frame->buf_pts,
              obj->class_id, obj->confidence, r.left, r.top, r.width, r.height)) {
        ++c.errors; ++infer_errors; fail("YOLO detection evidence write failed");
        return GST_PAD_PROBE_DROP;
      }
    }
    if (!write_frame_json(
            frame_evidence, frame->source_id, frame->frame_num, frame->buf_pts,
            GST_BUFFER_PTS(b), count)) {
      ++c.errors; ++infer_errors; fail("YOLO frame evidence write failed");
      return GST_PAD_PROBE_DROP;
    }
    c.persons += count;
    if (count) c.last_unix_ms = g_get_real_time()/1000;
  }
  if (parser_errors() || parser_rejected()) {
    ++infer_errors; fail("YOLO parser reported malformed tensors or detections");
    return GST_PAD_PROBE_DROP;
  }
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
  g_print("YOLO STATS batch=6 precision=FP16 batches=%lu inferred_frames=%lu persons=%lu errors=%lu parser_errors=%llu parser_rejected=%llu parser_calls=%llu proposals=%llu element_latency_mean_ms=%.3f element_latency_max_ms=%.3f\n",
    infer_batches.load(),infer_frames.load(),persons,infer_errors.load(),
    parser_errors(),parser_rejected(),parser_calls(),parser_proposals(),
    infer_timed ? infer_us.load()/1000.0/infer_timed.load() : 0.0,infer_max_us.load()/1000.0);
  if (detection_evidence) std::fflush(detection_evidence);
  if (frame_evidence) std::fflush(frame_evidence);
}
static void detection_link(GstElement *from, GstElement *tiler, GstElement *to) {
  void *library = dlopen("/models/libyolo26rawperson.so", RTLD_NOW|RTLD_LOCAL);
  if (!library) { fail("YOLO parser library cannot load"); std::exit(2); }
  parser_errors=reinterpret_cast<ParserCount>(dlsym(library,"Yolo26ParserErrors"));
  parser_rejected=reinterpret_cast<ParserCount>(dlsym(library,"Yolo26ParserRejected"));
  parser_calls=reinterpret_cast<ParserCount>(dlsym(library,"Yolo26ParserCalls"));
  parser_proposals=reinterpret_cast<ParserCount>(dlsym(library,"Yolo26ParserProposals"));
  if(!parser_errors||!parser_rejected||!parser_calls||!parser_proposals){fail("YOLO parser ABI mismatch");std::exit(2);}
  detection_evidence=std::fopen("/work/detections.jsonl","w");
  frame_evidence=std::fopen("/work/frames.jsonl","w");
  if(!detection_evidence || !frame_evidence){fail("cannot write YOLO metadata evidence");std::exit(2);}
  std::setvbuf(detection_evidence, nullptr, _IOLBF, 0);
  std::setvbuf(frame_evidence, nullptr, _IOLBF, 0);
  infer_stamp=g_quark_from_static_string("yolo26-infer-entry");
  auto *gie=make("nvinfer","YOLO26m-person");
  g_object_set(gie,"config-file-path","/config/config_infer_primary_yolo26m_raw_otm.txt",nullptr);
  auto *in=gst_element_get_static_pad(gie,"sink"), *out=gst_element_get_static_pad(gie,"src");
  gst_pad_add_probe(in,GST_PAD_PROBE_TYPE_BUFFER,infer_enter,nullptr,nullptr);
  gst_pad_add_probe(out,GST_PAD_PROBE_TYPE_BUFFER,infer_exit,nullptr,nullptr);
  gst_object_unref(in);gst_object_unref(out);
  auto *convert=make("nvvideoconvert","osd-gpu-convert"), *filter=make("capsfilter","osd-rgba-nvmm");
  auto *rgba=gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");
  g_object_set(filter,"caps",rgba,nullptr);gst_caps_unref(rgba);
  auto *osd=make("nvdsosd","person-overlay");
  g_object_set(osd,"process-mode",1,"display-text",TRUE,"display-bbox",TRUE,"display-mask",FALSE,"display-clock",FALSE,nullptr);
  link(from,gie);link(gie,tiler);link(tiler,convert);link(convert,filter);link(filter,osd);link(osd,to);
  event(nullptr,"DETECTION_GRAPH","mux->nvinfer(YOLO26m,FP16,batch=6,interval=0)->DeepStream-NMS(iou=0.70,conf=0.25)->tiler->NVMM/RGBA->nvdsosd(GPU,person+confidence)->NVENC inference=1");
}
