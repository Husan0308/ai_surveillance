// NvDCF per-camera tracker stage. Included after detector helpers.
#pragma once
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <unordered_set>

struct TrackerCounter {
  std::atomic<guint64> frames{0};
  std::atomic<guint64> objects{0};
  std::atomic<guint64> untracked{0};
  std::atomic<guint64> duplicate_ids{0};
};
static TrackerCounter tracker_counters[6];
static FILE *tracker_evidence = nullptr;
static std::mutex tracker_ids_lock;
static std::unordered_set<guint64> tracker_unique_ids[6];

static bool tracker_write_line(FILE *file, const char *data, size_t len) {
  if (!file || !data || !len) return false;
  flockfile(file);
  clearerr(file);
  const size_t written = std::fwrite(data, 1, len, file);
  const bool ok = written == len && std::ferror(file) == 0;
  funlockfile(file);
  return ok;
}

static bool tracker_write_object(
    guint source_id, gint frame_num, guint64 pts_ns, guint64 object_id,
    float detector_confidence, float tracker_confidence,
    float left, float top, float width, float height) {
  char line[640];
  const int n = std::snprintf(
      line, sizeof(line),
      "{\"source_id\":%u,\"frame\":%d,\"pts_ns\":%lu,"
      "\"object_id\":%lu,\"detector_confidence\":%.6f,"
      "\"tracker_confidence\":%.6f,"
      "\"box\":[%.3f,%.3f,%.3f,%.3f]}\n",
      source_id, frame_num, pts_ns, object_id,
      detector_confidence, tracker_confidence,
      left, top, width, height);
  if (n <= 0 || static_cast<size_t>(n) >= sizeof(line)) return false;
  return tracker_write_line(tracker_evidence, line, static_cast<size_t>(n));
}

static GstPadProbeReturn tracker_exit(GstPad*, GstPadProbeInfo *info, gpointer) {
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer) return GST_PAD_PROBE_OK;
  NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(buffer);
  if (!batch) {
    fail("NvDCF missing batch metadata");
    return GST_PAD_PROBE_DROP;
  }

  for (NvDsMetaList *fn = batch->frame_meta_list; fn; fn = fn->next) {
    auto *frame = static_cast<NvDsFrameMeta*>(fn->data);
    if (!frame || frame->source_id >= 6) {
      fail("NvDCF invalid source metadata");
      return GST_PAD_PROBE_DROP;
    }
    auto &counter = tracker_counters[frame->source_id];
    ++counter.frames;
    std::unordered_set<guint64> frame_ids;

    for (NvDsMetaList *on = frame->obj_meta_list; on; on = on->next) {
      auto *obj = static_cast<NvDsObjectMeta*>(on->data);
      if (!obj || obj->class_id != 0) continue;
      ++counter.objects;

      if (obj->object_id == UNTRACKED_OBJECT_ID) {
        ++counter.untracked;
        continue;
      }
      if (!frame_ids.insert(obj->object_id).second) {
        ++counter.duplicate_ids;
      }
      {
        std::lock_guard<std::mutex> guard(tracker_ids_lock);
        tracker_unique_ids[frame->source_id].insert(obj->object_id);
      }

      auto &r = obj->rect_params;
      if (!std::isfinite(r.left) || !std::isfinite(r.top) ||
          !std::isfinite(r.width) || !std::isfinite(r.height) ||
          r.width <= 0 || r.height <= 0 ||
          !std::isfinite(obj->tracker_confidence)) {
        fail("NvDCF invalid tracked object metadata");
        return GST_PAD_PROBE_DROP;
      }

      if (!tracker_write_object(
              frame->source_id, frame->frame_num, frame->buf_pts,
              obj->object_id, obj->confidence, obj->tracker_confidence,
              r.left, r.top, r.width, r.height)) {
        fail("NvDCF track evidence write failed");
        return GST_PAD_PROBE_DROP;
      }

      auto &text = obj->text_params;
      g_free(text.display_text);
      text.display_text = g_strdup_printf("person ID=%lu", obj->object_id);
      text.x_offset = guint(std::max(0.0f, r.left));
      text.y_offset = guint(std::max(0.0f, r.top - 22));
      text.font_params.font_name = const_cast<char*>("Sans");
      text.font_params.font_size = 13;
      text.font_params.font_color = {1.0, 1.0, 1.0, 1.0};
      text.set_bg_clr = 1;
      text.text_bg_clr = {0.0, 0.0, 0.0, 0.65};
    }
  }
  return GST_PAD_PROBE_OK;
}

static GstElement *create_nvdcf_tracker() {
  const char *lib =
      "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so";
  const char *cfg =
      "/config/config_tracker_NvDCF_stable_person.yml";

  tracker_evidence = std::fopen("/work/tracks.jsonl", "w");
  if (!tracker_evidence) {
    fail("cannot write NvDCF track evidence");
    std::exit(2);
  }
  std::setvbuf(tracker_evidence, nullptr, _IOLBF, 0);

  GstElement *tracker = make("nvtracker", "NvDCF-person");
  g_object_set(
      tracker,
      "tracker-width", 960u,
      "tracker-height", 544u,
      "ll-lib-file", lib,
      "ll-config-file", cfg,
      "gpu-id", 0u,
      "display-tracking-id", FALSE,
      "compute-hw", 1u,
      "tracking-id-reset-mode", 1u,
      nullptr);

  GstPad *out = gst_element_get_static_pad(tracker, "src");
  gst_pad_add_probe(out, GST_PAD_PROBE_TYPE_BUFFER, tracker_exit, nullptr, nullptr);
  gst_object_unref(out);

  event(nullptr, "TRACKER_GRAPH",
        "nvtracker(NvDCF_stable_person,width=960,height=544,batch=6,gpu=0,reid=0)->"
        "per-camera-object-id");
  return tracker;
}

static void tracker_stats() {
  for (int i = 0; i < 6; ++i) {
    guint64 unique = 0;
    {
      std::lock_guard<std::mutex> guard(tracker_ids_lock);
      unique = tracker_unique_ids[i].size();
    }
    g_print(
        "CAM-%02d TRACK frames=%lu objects=%lu untracked=%lu "
        "duplicate_ids=%lu unique_ids=%lu\n",
        i + 1,
        tracker_counters[i].frames.load(),
        tracker_counters[i].objects.load(),
        tracker_counters[i].untracked.load(),
        tracker_counters[i].duplicate_ids.load(),
        unique);
  }
  if (tracker_evidence) std::fflush(tracker_evidence);
}
