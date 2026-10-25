// Included only in the detection build of the frozen six-camera native runner.
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#include <cmath>
#include <dlfcn.h>
#include <array>
#include <cstdint>
#include <atomic>
#ifdef YOLO26_TRACKER
#include "tracker.hpp"
#endif

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

// Enabled only by the validator's existing --trace-latency switch. Streaming
// callbacks append fixed-size records to this one-shot bounded ring; they do
// no file I/O, logging, allocation, or locking.
enum PgieDiagStage : guint16 {
  PGIE_DIAG_MUX = 1, PGIE_DIAG_PGIE, PGIE_DIAG_TRACKER, PGIE_DIAG_TILER,
  PGIE_DIAG_OSD_CONVERT, PGIE_DIAG_OSD_CAPS, PGIE_DIAG_OSD,
  PGIE_DIAG_OUTPUT_CONVERT, PGIE_DIAG_OUTPUT_CAPS, PGIE_DIAG_ENCODER,
  PGIE_DIAG_FILE_QUEUE, PGIE_DIAG_FILE_PARSE, PGIE_DIAG_FILE_MUX,
  PGIE_DIAG_FILE_SINK, PGIE_DIAG_PREVIEW_QUEUE, PGIE_DIAG_PREVIEW_PARSE,
  PGIE_DIAG_PREVIEW_CAPS, PGIE_DIAG_PREVIEW_MUX, PGIE_DIAG_PREVIEW_SINK,
  PGIE_DIAG_CAMERA_QUEUE_BASE = 100
};
static constexpr size_t PGIE_DIAG_CAPACITY = 131072;
struct PgieDiagEvent {
  gint64 monotonic_us;
  guint64 pts_ns;
  guint64 buffer_cookie;
  guint64 batch_no;
  guint64 source_mask;
  guint64 frame_key;
  guint64 value_a;
  guint64 value_b;
  guint32 frame_count;
  guint16 stage;
  guint8 edge; // 0=sink/input, 1=src/output, 2=queue sample
  guint8 kind; // 0=buffer boundary, 1=queue levels
};
struct InferStamp { gint64 entered_us; guint64 batch_no; };
struct PgieDiagProbeCtx { guint16 stage; guint8 edge; gint source_id; };
struct PgieDiagQueue { GstElement *element; guint16 stage; };
static std::array<PgieDiagEvent, PGIE_DIAG_CAPACITY> pgie_diag_events{};
static std::atomic<guint64> pgie_diag_claimed{0}, pgie_diag_dropped{0};
static std::atomic<guint64> pgie_diag_batch_seq{0};
static std::atomic_bool pgie_diag_enabled{false};
static std::atomic_bool pgie_diag_export_ready{false};
static std::atomic_bool pgie_diag_dump_started{false};
static PgieDiagQueue pgie_diag_queues[16]{};
static guint pgie_diag_queue_count = 0;

#ifndef PGIE_DIAG_TRACE_PATH
#define PGIE_DIAG_TRACE_PATH "/work/pgie-stage-trace.jsonl"
#endif

static const char *pgie_diag_stage_name(guint16 stage) {
  switch (stage) {
    case PGIE_DIAG_MUX: return "streammux";
    case PGIE_DIAG_PGIE: return "nvinfer";
    case PGIE_DIAG_TRACKER: return "nvtracker";
    case PGIE_DIAG_TILER: return "tiler";
    case PGIE_DIAG_OSD_CONVERT: return "osd_convert";
    case PGIE_DIAG_OSD_CAPS: return "osd_caps";
    case PGIE_DIAG_OSD: return "nvdsosd";
    case PGIE_DIAG_OUTPUT_CONVERT: return "output_convert";
    case PGIE_DIAG_OUTPUT_CAPS: return "output_caps";
    case PGIE_DIAG_ENCODER: return "nvenc";
    case PGIE_DIAG_FILE_QUEUE: return "file_queue";
    case PGIE_DIAG_FILE_PARSE: return "file_parse";
    case PGIE_DIAG_FILE_MUX: return "file_mux";
    case PGIE_DIAG_FILE_SINK: return "file_sink";
    case PGIE_DIAG_PREVIEW_QUEUE: return "preview_queue";
    case PGIE_DIAG_PREVIEW_PARSE: return "preview_parse";
    case PGIE_DIAG_PREVIEW_CAPS: return "preview_caps";
    case PGIE_DIAG_PREVIEW_MUX: return "preview_mux";
    case PGIE_DIAG_PREVIEW_SINK: return "preview_sink";
    default:
      if (stage >= PGIE_DIAG_CAMERA_QUEUE_BASE &&
          stage < PGIE_DIAG_CAMERA_QUEUE_BASE + 6) return "camera_queue";
      return "unknown";
  }
}

static guint64 pgie_diag_frame_key(NvDsBatchMeta *meta, guint64 &mask, guint32 &count) {
  // FNV-1a over (source_id, frame_num) pairs, preserving an exact batch key
  // when NvDs frame metadata is available at both sides of an element.
  guint64 hash = 1469598103934665603ULL;
  mask = 0; count = 0;
  if (!meta) return 0;
  for (NvDsMetaList *node = meta->frame_meta_list; node; node = node->next) {
    auto *frame = static_cast<NvDsFrameMeta*>(node->data);
    if (!frame) continue;
    if (frame->source_id < 64) mask |= (guint64(1) << frame->source_id);
    const guint32 values[2] = {frame->source_id, guint32(frame->frame_num)};
    for (guint32 value : values) {
      for (int byte = 0; byte < 4; ++byte) {
        hash ^= guint8((value >> (byte * 8)) & 0xff);
        hash *= 1099511628211ULL;
      }
    }
    ++count;
  }
  return count ? hash : 0;
}

static void pgie_diag_append(PgieDiagEvent event) {
  if (!pgie_diag_enabled.load(std::memory_order_relaxed)) return;
  const guint64 slot = pgie_diag_claimed.fetch_add(1, std::memory_order_relaxed);
  if (slot >= PGIE_DIAG_CAPACITY) {
    pgie_diag_dropped.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  event.monotonic_us = g_get_monotonic_time();
  pgie_diag_events[slot] = event;
}

static void pgie_diag_buffer(guint16 stage, guint8 edge, GstBuffer *buffer, gint source_id = -1) {
  if (!pgie_diag_enabled.load(std::memory_order_relaxed) || !buffer) return;
  PgieDiagEvent event{};
  event.pts_ns = GST_BUFFER_PTS(buffer);
  event.buffer_cookie = guint64(reinterpret_cast<uintptr_t>(buffer));
  auto *meta = gst_buffer_get_nvds_batch_meta(buffer);
  event.frame_key = pgie_diag_frame_key(meta, event.source_mask, event.frame_count);
  if (source_id >= 0 && source_id < 64) {
    event.source_mask |= (guint64(1) << source_id);
    if (!event.frame_count) event.frame_count = 1;
  }
  if (auto *stamp = static_cast<InferStamp*>(gst_mini_object_get_qdata(
          GST_MINI_OBJECT(buffer), infer_stamp))) event.batch_no = stamp->batch_no;
  event.stage = stage;
  event.edge = edge;
  event.kind = 0;
  pgie_diag_append(event);
}

static GstPadProbeReturn pgie_diag_probe_cb(GstPad*, GstPadProbeInfo *info, gpointer data) {
  auto *ctx = static_cast<PgieDiagProbeCtx*>(data);
  pgie_diag_buffer(ctx->stage, ctx->edge, GST_PAD_PROBE_INFO_BUFFER(info), ctx->source_id);
  return GST_PAD_PROBE_OK;
}

static void pgie_diag_probe(GstElement *element, const char *pad_name,
                            guint16 stage, guint8 edge, gint source_id = -1) {
  if (!pgie_diag_enabled.load(std::memory_order_relaxed) || !element) return;
  GstPad *pad = gst_element_get_static_pad(element, pad_name);
  if (!pad) return; // Diagnostics must never turn a missing optional pad into a failure.
  auto *ctx = g_new(PgieDiagProbeCtx, 1);
  *ctx = {stage, edge, source_id};
  gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, pgie_diag_probe_cb, ctx,
                    [](gpointer p) { g_free(p); });
  gst_object_unref(pad);
}

static void pgie_diag_trace_element(GstElement *element, guint16 stage) {
  pgie_diag_probe(element, "sink", stage, 0);
  pgie_diag_probe(element, "src", stage, 1);
}

static void pgie_diag_register_queue(GstElement *element, guint16 stage) {
  if (!pgie_diag_enabled.load(std::memory_order_relaxed) || !element ||
      pgie_diag_queue_count >= G_N_ELEMENTS(pgie_diag_queues)) return;
  pgie_diag_queues[pgie_diag_queue_count++] = {element, stage};
}

static void pgie_diag_configure() {
  const char *tracers = g_getenv("GST_TRACERS");
  pgie_diag_enabled = tracers && std::strstr(tracers, "latency");
  pgie_diag_export_ready = false;
  pgie_diag_dump_started = false;
  g_print("PGIE_DIAG enabled=%d capacity=%zu\n",
          int(pgie_diag_enabled.load(std::memory_order_relaxed)), PGIE_DIAG_CAPACITY);
}

static void pgie_diag_sample_queues() {
  if (!pgie_diag_enabled.load(std::memory_order_relaxed)) return;
  for (guint i = 0; i < pgie_diag_queue_count; ++i) {
    guint buffers = 0, bytes = 0;
    guint64 time_ns = 0;
    g_object_get(pgie_diag_queues[i].element,
                 "current-level-buffers", &buffers,
                 "current-level-bytes", &bytes,
                 "current-level-time", &time_ns, nullptr);
    PgieDiagEvent event{};
    event.stage = pgie_diag_queues[i].stage;
    event.edge = 2;
    event.kind = 1;
    event.value_a = buffers;
    event.value_b = bytes;
    event.pts_ns = time_ns;
    pgie_diag_append(event);
  }
}

static void pgie_diag_dump() {
  if (!pgie_diag_enabled.load(std::memory_order_relaxed) ||
      !pgie_diag_export_ready.load(std::memory_order_acquire)) return;
  bool expected = false;
  if (!pgie_diag_dump_started.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) return;
  FILE *file = std::fopen(PGIE_DIAG_TRACE_PATH, "w");
  if (!file) {
    g_print("PGIE_DIAG dump=FAILED reason=trace_file_open\n");
    return;
  }
  bool write_ok = true;
  const guint64 claimed = pgie_diag_claimed.load(std::memory_order_relaxed);
  const guint64 count = std::min<guint64>(claimed, PGIE_DIAG_CAPACITY);
  write_ok = std::fprintf(file,
      "{\"type\":\"header\",\"schema\":1,\"clock\":\"g_get_monotonic_time_us\","
      "\"capacity\":%zu,\"claimed\":%llu,\"recorded\":%llu,\"dropped\":%llu}\n",
      PGIE_DIAG_CAPACITY, (unsigned long long)claimed, (unsigned long long)count,
      (unsigned long long)pgie_diag_dropped.load(std::memory_order_relaxed)) >= 0;
  for (guint64 i = 0; i < count; ++i) {
    const auto &e = pgie_diag_events[i];
    int written = -1;
    if (e.kind == 1) {
      written = std::fprintf(file,
          "{\"type\":\"queue\",\"monotonic_us\":%lld,\"stage\":\"%s\","
          "\"source_id\":%d,\"buffers\":%llu,\"bytes\":%llu,\"level_time_ns\":%llu}\n",
          (long long)e.monotonic_us, pgie_diag_stage_name(e.stage),
          e.stage >= PGIE_DIAG_CAMERA_QUEUE_BASE && e.stage < PGIE_DIAG_CAMERA_QUEUE_BASE + 6
              ? int(e.stage - PGIE_DIAG_CAMERA_QUEUE_BASE) : -1,
          (unsigned long long)e.value_a, (unsigned long long)e.value_b,
          (unsigned long long)e.pts_ns);
    } else {
      written = std::fprintf(file,
          "{\"type\":\"buffer\",\"monotonic_us\":%lld,\"stage\":\"%s\","
          "\"edge\":\"%s\",\"pts_ns\":%llu,\"buffer_cookie\":%llu,"
          "\"batch_no\":%llu,\"source_mask\":%llu,\"frame_count\":%u,\"frame_key\":%llu}\n",
          (long long)e.monotonic_us, pgie_diag_stage_name(e.stage),
          e.edge == 0 ? "sink" : "src", (unsigned long long)e.pts_ns,
          (unsigned long long)e.buffer_cookie, (unsigned long long)e.batch_no,
          (unsigned long long)e.source_mask, e.frame_count, (unsigned long long)e.frame_key);
    }
    if (written < 0) {
      write_ok = false;
      break;
    }
  }
  if (std::fflush(file) != 0 || std::ferror(file)) write_ok = false;
  if (std::fclose(file) != 0) write_ok = false;
  if (!write_ok) {
    g_print("PGIE_DIAG dump=FAILED reason=trace_file_write\n");
    return;
  }
  g_print("PGIE_DIAG dump=OK file=pgie-stage-trace.jsonl capacity=%zu claimed=%llu recorded=%llu dropped=%llu\n",
      PGIE_DIAG_CAPACITY, (unsigned long long)claimed, (unsigned long long)count,
      (unsigned long long)pgie_diag_dropped.load(std::memory_order_relaxed));
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
    auto *stamp = g_new(InferStamp, 1);
    stamp->entered_us = g_get_monotonic_time();
    stamp->batch_no = pgie_diag_enabled.load(std::memory_order_relaxed)
        ? pgie_diag_batch_seq.fetch_add(1, std::memory_order_relaxed) + 1 : 0;
    gst_mini_object_set_qdata(GST_MINI_OBJECT(b), infer_stamp, stamp, g_free);
    pgie_diag_buffer(PGIE_DIAG_PGIE, 0, b);
  }
  return GST_PAD_PROBE_OK;
}
static GstPadProbeReturn infer_exit(GstPad*, GstPadProbeInfo *info, gpointer) {
  GstBuffer *b = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!b) return GST_PAD_PROBE_OK;
  pgie_diag_buffer(PGIE_DIAG_PGIE, 1, b);
  auto *meta = gst_buffer_get_nvds_batch_meta(b);
  if (!meta) { ++infer_errors; fail("YOLO missing batch metadata"); return GST_PAD_PROBE_DROP; }
  if (auto *stamp = static_cast<InferStamp*>(gst_mini_object_get_qdata(GST_MINI_OBJECT(b), infer_stamp))) {
    const guint64 dt = g_get_monotonic_time() - stamp->entered_us;
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
  pgie_diag_probe(from, "src", PGIE_DIAG_MUX, 1);
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
  pgie_diag_trace_element(tiler, PGIE_DIAG_TILER);
  pgie_diag_trace_element(convert, PGIE_DIAG_OSD_CONVERT);
  pgie_diag_trace_element(filter, PGIE_DIAG_OSD_CAPS);
  pgie_diag_trace_element(osd, PGIE_DIAG_OSD);
  pgie_diag_trace_element(to, PGIE_DIAG_OUTPUT_CONVERT);
  link(from,gie);
#ifdef YOLO26_TRACKER
  auto *tracker=create_nvdcf_tracker();
  pgie_diag_trace_element(tracker, PGIE_DIAG_TRACKER);
  link(gie,tracker);link(tracker,tiler);
#else
  link(gie,tiler);
#endif
  link(tiler,convert);link(convert,filter);link(filter,osd);link(osd,to);
#ifdef YOLO26_TRACKER
  event(nullptr,"DETECTION_GRAPH","mux->nvinfer(YOLO26m,FP16,batch=6,interval=0)->DeepStream-NMS(iou=0.45,conf=0.25)->nvtracker(NvDCF_stable_person,960x544)->tiler->NVMM/RGBA->nvdsosd(GPU,person+ID)->NVENC inference=1 tracker=1");
#else
  event(nullptr,"DETECTION_GRAPH","mux->nvinfer(YOLO26m,FP16,batch=6,interval=0)->DeepStream-NMS(iou=0.45,conf=0.25)->tiler->NVMM/RGBA->nvdsosd(GPU,person+confidence)->NVENC inference=1");
#endif
}
