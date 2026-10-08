// NvDCF per-camera tracker stage. Included after detector helpers.
#pragma once
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#include <nvds_tracker_meta.h>
#include "exact_frame_ring.hpp"
#include "shadow_continuity.hpp"
#include "latency_histogram.hpp"
#include "track_birth_verification.hpp"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <vector>

struct TrackerCounter {
  std::atomic<guint64> frames{0};
  std::atomic<guint64> objects{0};
  std::atomic<guint64> untracked{0};
  std::atomic<guint64> duplicate_ids{0};
  std::atomic<guint64> canonical_objects{0};
  std::atomic<guint64> fallbacks{0};
  std::atomic<guint64> verified_lowconf_fallbacks{0};
  std::atomic<guint64> trusted_track_lowconf_fallbacks{0};
  std::atomic<guint64> shadow_context_fallbacks{0};
  std::atomic<guint64> active_birth_verifier_confirmed{0};
  std::atomic<guint64> active_birth_verifier_withheld{0};
  std::atomic<guint64> ambiguous_matches{0};
  std::atomic<guint64> invalid_shadow_geometry{0};
};
static TrackerCounter tracker_counters[6];
// Bounded live-join diagnostics: record the first few tracker keys per source
// so they can be compared to the corresponding decoded preview keys.

// PGIE and NvDCF rectangles are in the nvstreammux output coordinate space.
// DeepStream 9.1 may leave NvDsFrameMeta::pipeline_width/height unset (or
// populated from source caps) even when nvstreammux resizes the surface. The
// source_frame_* fields describe the decoded source, not these rectangles.
// Use the actual dimensions configured on the mux for validation, verifier
// crop mapping, and the canonical bbox-coordinate contract.
static guint tracker_bbox_coordinate_extent(const NvDsFrameMeta *frame, bool horizontal) {
  (void)frame;
#ifdef PGIE_ISOLATION
  return horizontal ? pgie_isolation_width : pgie_isolation_height;
#else
  return horizontal ? v11_mux_width : v11_mux_height;
#endif
}

static LatencyHistogram tracker_element_latency[6];
static LatencyHistogram tracker_shadow_scan_latency[6];
static LatencyHistogram tracker_callback_to_metadata_latency[6];
static std::atomic<guint64> tracker_input_stamp_misses[6];
struct TrackerInputStamp {
  gint frame_num = -1;
  guint64 pts_ns = 0;
  guint64 entered_ns = 0;
};
static constexpr size_t TRACKER_INPUT_STAMP_CAPACITY = 128;
static TrackerInputStamp tracker_input_stamps[6][TRACKER_INPUT_STAMP_CAPACITY];
static std::mutex tracker_input_stamp_mutex[6];
static FILE *tracker_evidence = nullptr;
static FILE *fallback_evidence = nullptr;
static FILE *lowconf_fallback_evidence = nullptr;
static FILE *canonical_evidence = nullptr;
static FILE *invalid_shadow_evidence = nullptr;
static FILE *tracker_decision_evidence = nullptr;
static FILE *tracker_frame_state_evidence = nullptr;
static std::mutex tracker_ids_lock;
static std::unordered_set<guint64> tracker_unique_ids[6];
static constexpr guint TRACKER_AUXILIARY_GIE_ID = 3;
// At 20 FPS this permits at most 250 ms between an appearance-verified
// target's ordinary detector association and a current low-confidence box.
// Shadow lifetime by itself never refreshes this support window.
static constexpr std::uint32_t TRUSTED_LOWCONF_SUPPORT_MAX_AGE_FRAMES = 5;
// Auxiliary high-resolution proposals are admitted only on sparse diagnostic
// frames. Their NvDCF targets remain hidden from canonical/UI output until the
// frozen 640 crop verifier positively confirms the same native target. Normal
// 640 proposals keep their historical publication behavior unchanged.
static std::unordered_set<guint64> tracker_auxiliary_target_ids[6];

struct TrackerDetection {
  float confidence = 0;
  float left = 0, top = 0, width = 0, height = 0;
  guint component_id = 1;
  bool auxiliary = false;
};
struct ShadowStamp {
  guint32 frame_num = 0;
  guint32 age = 0;
  float left = 0, top = 0, width = 0, height = 0;
  float confidence = 0;
  TRACKER_STATE state = EMPTY;
};
struct ActiveStamp {
  guint32 frame_num = 0;
  float left = 0, top = 0, width = 0, height = 0;
  float confidence = 0;
};
struct ShadowCandidate {
  guint64 id = 0;
  guint32 age = 0;
  TRACKER_STATE state = EMPTY;
  float confidence = 0;
  float visibility = 0;
  float left = 0, top = 0, width = 0, height = 0;
  bool immediate_continuity = false;
};
struct PublishedTrack {
  guint64 id = 0;
  const char *source = "TRACKER";
  float detector_confidence = NAN;
  float tracker_confidence = NAN;
  float verifier_confidence = NAN;
  float left = 0, top = 0, width = 0, height = 0;
  guint32 shadow_age = 0;
  float association_iou = 0;
  int detection_index = -1;
  bool requires_birth_verification = false;
};
// nvinfer and nvtracker may run on different streaming tasks. Keep a bounded
// exact-key ring instead of one mutable "latest frame" slot: otherwise a
// downstream tracker buffer can observe detections overwritten by a later
// PGIE callback, silently disabling safe current-frame shadow association.
static constexpr size_t TRACKER_DETECTION_CACHE_LIMIT = 64;
static ExactFrameRing<TrackerDetection, 6, TRACKER_DETECTION_CACHE_LIMIT> tracker_detector_frames;
// Low-confidence proposals are removed from NvDCF input but retained under
// the same exact source/frame/PTS key. They are submitted to the persistent
// appearance verifier only after tracker_exit proves a current, unique
// inactive-shadow association. This prevents unrelated background proposals
// from consuming verifier capacity or delaying a valid fallback.
static ExactFrameRing<LowConfCandidate, 6, TRACKER_DETECTION_CACHE_LIMIT>
    tracker_lowconf_candidate_frames;
static std::atomic<guint64> tracker_detection_cache_hits{0};
static std::atomic<guint64> tracker_detection_cache_misses{0};
static std::atomic<guint64> tracker_detection_cache_evictions{0};
static FILE *tracker_detection_join_evidence = nullptr;
static bool tracker_write_line(FILE *file, const char *data, size_t len);
static std::unordered_map<guint64, ShadowStamp> tracker_previous_shadows[6];
static std::unordered_map<guint64, ActiveStamp> tracker_previous_active_boxes[6];
static std::unordered_map<guint64, guint32> tracker_last_primary_association_frame[6];
static std::mutex tracker_previous_active_mutex[6];
static std::unordered_set<guint64> tracker_previous_active_ids[6];
static gint tracker_previous_active_frame[6] = {-1,-1,-1,-1,-1,-1};
static std::atomic<guint32> tracker_canonical_counts[6]{};
static std::atomic<gint> tracker_canonical_frames[6]{-1,-1,-1,-1,-1,-1};
static std::string tracker_generation;
static guint v11_tracker_width = 960;
static guint v11_tracker_height = 544;
static std::mutex tracker_feed_mutex;
static TrackBirthVerificationLedger tracker_birth_verification;
static std::unordered_map<guint64, guint> tracker_birth_attempt_diag_count[6];

static void tracker_record_detections(guint source_id, gint frame_num, guint64 pts_ns,
                                      std::vector<TrackerDetection> &&detections) {
  if (source_id >= 6) return;
  const auto result = tracker_detector_frames.put(source_id, frame_num, pts_ns,
      std::move(detections));
  if (result.overwritten) {
    if (tracker_detection_join_evidence) {
      char line[256];
      const int n = std::snprintf(line, sizeof(line),
          "{\"event\":\"evict\",\"source_id\":%u,\"frame\":%d,\"pts_ns\":%lu}\n",
          source_id, static_cast<gint>(result.old_frame), result.old_pts);
      if (n > 0 && static_cast<size_t>(n) < sizeof(line))
        tracker_write_line(tracker_detection_join_evidence, line, static_cast<size_t>(n));
    }
    ++tracker_detection_cache_evictions;
  }
}

static std::vector<TrackerDetection> tracker_take_detections(
    guint source_id, gint frame_num, guint64 pts_ns) {
  if (source_id >= 6) return {};
  auto result = tracker_detector_frames.take(source_id, frame_num, pts_ns);
  if (result) {
    ++tracker_detection_cache_hits;
    return std::move(*result);
  }
  ++tracker_detection_cache_misses;
  if (tracker_detection_join_evidence) {
    char line[512];
    const int n = std::snprintf(line, sizeof(line),
        "{\"event\":\"miss\",\"source_id\":%u,\"frame\":%d,\"pts_ns\":%lu,"
        "\"cache_capacity\":%zu}\n",
        source_id, frame_num, pts_ns, TRACKER_DETECTION_CACHE_LIMIT);
    if (n > 0 && static_cast<size_t>(n) < sizeof(line))
      tracker_write_line(tracker_detection_join_evidence, line, static_cast<size_t>(n));
  }
  return {};
}

static void tracker_record_lowconf_candidates(
    guint source_id, gint frame_num, guint64 pts_ns,
    std::vector<LowConfCandidate> &&candidates) {
  if (source_id >= 6) return;
  tracker_lowconf_candidate_frames.put(source_id, frame_num, pts_ns,
      std::move(candidates));
}

static std::vector<LowConfCandidate> tracker_take_lowconf_candidates(
    guint source_id, gint frame_num, guint64 pts_ns) {
  if (source_id >= 6) return {};
  auto result = tracker_lowconf_candidate_frames.take(source_id, frame_num, pts_ns);
  if (!result) return {};
  return std::move(*result);
}
static guint tracker_canonical_count_for_frame(guint source_id, gint frame_num) {
  if (source_id >= 6 || tracker_canonical_frames[source_id].load() != frame_num) return 0;
  return tracker_canonical_counts[source_id].load();
}

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
  if (!tracker_evidence) return true;
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

static float tracker_iou(float al, float at, float aw, float ah,
                         float bl, float bt, float bw, float bh) {
  const float x1 = std::max(al, bl), y1 = std::max(at, bt);
  const float x2 = std::min(al + aw, bl + bw), y2 = std::min(at + ah, bt + bh);
  const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  const float uni = aw * ah + bw * bh - inter;
  return uni > 0 ? inter / uni : 0;
}
static float tracker_center_distance(float al, float at, float aw, float ah,
                                     float bl, float bt, float bw, float bh) {
  return std::hypot(al + aw / 2 - bl - bw / 2, at + ah / 2 - bt - bh / 2);
}
static void tracker_json_number(std::ostringstream &out, float value) {
  if (std::isfinite(value) && value >= 0) out << std::fixed << std::setprecision(6) << value;
  else out << "null";
}
static bool tracker_atomic_snapshot(guint source_id, gint frame_num, guint64 pts_ns,
                                    guint64 monotonic_ns, guint64 epoch_ns,
                                    guint32 source_frame_num,
                                    guint width, guint height, guint proposal_width, guint proposal_height,
                                    const std::vector<PublishedTrack> &objects,
                                    const std::vector<TrackerDetection> &detections,
                                    std::string *serialized_out = nullptr) {
  const char *dir = g_getenv("V11_TRACKED_OBJECT_FEED_DIR");
  if (!dir || !*dir) dir = "/dev/shm";
  char path[512], temporary[560];
  std::snprintf(path, sizeof(path), "%s/v11_tracked_objects_cam%02u_v1.json", dir, source_id + 1);
  std::snprintf(temporary, sizeof(temporary), "%s.tmp.%d", path, getpid());
  std::ostringstream out;
  out << "{\"schema_version\":1,\"camera_id\":\"CAM-" << std::setw(2) << std::setfill('0') << source_id + 1
      << "\",\"generation\":\"" << tracker_generation << "\",\"frame_num\":" << frame_num
      << ",\"source_frame_num\":" << source_frame_num
      << ",\"pts_ns\":" << pts_ns << ",\"timestamp_monotonic_ns\":" << monotonic_ns
      << ",\"timestamp_epoch_ns\":" << epoch_ns << ",\"source_frame_width\":" << width
      << ",\"source_frame_height\":" << height
      << ",\"pipeline_width\":" << proposal_width << ",\"pipeline_height\":" << proposal_height
      << ",\"bbox_coordinate_space\":\"pipeline\",\"bbox_coordinate_width\":" << proposal_width
      << ",\"bbox_coordinate_height\":" << proposal_height << ",\"objects\":[";
  for (size_t i = 0; i < objects.size(); ++i) {
    const auto &o = objects[i];
    if (i) out << ',';
    out << "{\"camera_id\":\"CAM-" << std::setw(2) << std::setfill('0') << source_id + 1
        << "\",\"frame_num\":" << frame_num << ",\"frame\":" << frame_num
        << ",\"pts_ns\":" << pts_ns << ",\"native_track_id\":" << o.id
        << ",\"bbox\":[" << std::fixed << std::setprecision(3) << o.left << ',' << o.top << ','
        << o.left + o.width << ',' << o.top + o.height << "],\"bbox_source\":\"" << o.source
        << "\",\"bbox_coordinate_space\":\"pipeline\",\"bbox_coordinate_width\":" << proposal_width
        << ",\"bbox_coordinate_height\":" << proposal_height
        << ",\"detector_confidence\":";
    tracker_json_number(out, o.detector_confidence);
    out << ",\"tracker_confidence\":";
    tracker_json_number(out, o.tracker_confidence);
    out << ",\"verifier_confidence\":";
    tracker_json_number(out, o.verifier_confidence);
    const bool shadow_fallback = std::strcmp(o.source, "PGIE_SHADOW_FALLBACK") == 0 ||
        std::strcmp(o.source, "PGIE_VERIFIED_LOWCONF_FALLBACK") == 0 ||
        std::strcmp(o.source, "SHADOW_CONTEXT_VERIFIED") == 0;
    out << ",\"shadow_age\":" << (shadow_fallback ? std::to_string(o.shadow_age) : "null")
        << ",\"association_iou\":";
    if (shadow_fallback) out << std::fixed << std::setprecision(6) << o.association_iou;
    else out << "null";
    out << ",\"generation\":\"" << tracker_generation << "\",\"timestamp_monotonic_ns\":" << monotonic_ns << '}';
  }
  out << "],\"detector_proposals\":[";
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto &d = detections[i];
    if (i) out << ',';
    out << "{\"camera_id\":\"CAM-" << std::setw(2) << std::setfill('0') << source_id + 1
        << "\",\"frame_num\":" << frame_num << ",\"pts_ns\":" << pts_ns
        << ",\"proposal_index\":" << i << ",\"class_id\":0,\"confidence\":"
        << std::fixed << std::setprecision(6) << d.confidence << ",\"bbox\":["
        << d.left << ',' << d.top << ',' << d.left + d.width << ',' << d.top + d.height << "],"
        << "\"bbox_coordinate_space\":\"pipeline\",\"bbox_coordinate_width\":" << proposal_width
        << ",\"bbox_coordinate_height\":" << proposal_height << ','
        << "\"source_frame_width\":" << width << ",\"source_frame_height\":" << height << '}';
  }
  out << "]}\n";
  const std::string bytes = out.str();
  if (serialized_out) *serialized_out = bytes;
  int fd = open(temporary, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (fd < 0) return false;
  size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count = write(fd, bytes.data() + offset, bytes.size() - offset);
    if (count <= 0) { close(fd); unlink(temporary); return false; }
    offset += static_cast<size_t>(count);
  }
  close(fd);
  if (rename(temporary, path) != 0) { unlink(temporary); return false; }
  return true;
}

static std::vector<ShadowCandidate> tracker_current_shadows(NvDsBatchMeta *batch, NvDsFrameMeta *frame) {
  std::vector<ShadowCandidate> result;
  for (NvDsMetaList *node = batch->batch_user_meta_list; node; node = node->next) {
    auto *user = static_cast<NvDsUserMeta*>(node->data);
    if (!user || user->base_meta.meta_type != NVDS_TRACKER_SHADOW_LIST_META || !user->user_meta_data) continue;
    auto *shadow_batch = static_cast<NvDsTargetMiscDataBatch*>(user->user_meta_data);
    if (!shadow_batch->list) continue;
    for (guint si = 0; si < shadow_batch->numFilled; ++si) {
      auto *stream = &shadow_batch->list[si];
      if (stream->streamID != frame->pad_index) continue;
      for (guint ti = 0; ti < stream->numFilled; ++ti) {
        auto *target = &stream->list[ti];
        if (target->classId != 0 || !target->list) continue;
        const auto *current = find_current_tracker_sample(target->list,
            target->numObj, frame->frame_num,
            [](const NvDsTargetMiscDataFrame &sample) { return sample.frameNum; });
        if (current) {
          const auto &sample = *current;
          const auto &r = sample.tBbox;
          if (!std::isfinite(r.left) || !std::isfinite(r.top) || !std::isfinite(r.width) ||
              !std::isfinite(r.height) || r.width <= 0 || r.height <= 0) {
            ++tracker_counters[frame->source_id].invalid_shadow_geometry;
            if (invalid_shadow_evidence) {
              char line[512];
              const int n = std::snprintf(line, sizeof(line),
                  "{\"source_id\":%u,\"camera_id\":\"CAM-%02u\",\"frame\":%d,\"pts_ns\":%lu,"
                  "\"native_track_id\":%lu,\"age\":%u,\"state_value\":%d,\"confidence_finite\":%s,"
                  "\"bbox_finite\":[%s,%s,%s,%s],\"bbox_positive\":%s}\n",
                  frame->source_id, frame->source_id + 1, frame->frame_num, frame->buf_pts,
                  target->uniqueId, sample.age, static_cast<int>(sample.trackerState),
                  std::isfinite(sample.confidence) ? "true" : "false",
                  std::isfinite(r.left) ? "true" : "false", std::isfinite(r.top) ? "true" : "false",
                  std::isfinite(r.width) ? "true" : "false", std::isfinite(r.height) ? "true" : "false",
                  (r.width > 0 && r.height > 0) ? "true" : "false");
              if (n > 0 && static_cast<size_t>(n) < sizeof(line))
                tracker_write_line(invalid_shadow_evidence, line, static_cast<size_t>(n));
            }
            continue;
          }
          const auto previous_shadow = tracker_previous_shadows[frame->source_id].find(target->uniqueId);
          const bool has_previous_shadow = previous_shadow != tracker_previous_shadows[frame->source_id].end();
          const bool previous_active_id_present =
              tracker_previous_active_ids[frame->source_id].count(target->uniqueId) != 0;
          const bool immediate_continuity = tracker_has_immediate_shadow_continuity(
              frame->frame_num,
              has_previous_shadow,
              has_previous_shadow ? static_cast<int>(previous_shadow->second.frame_num) : -1,
              has_previous_shadow ? previous_shadow->second.age : 0,
              sample.age,
              tracker_previous_active_frame[frame->source_id],
              previous_active_id_present);
          result.push_back({target->uniqueId, sample.age, sample.trackerState,
              sample.confidence, sample.visibility, r.left, r.top, r.width, r.height,
              immediate_continuity});
        }
      }
    }
  }
  return result;
}

static void tracker_log_fallback(guint source_id, gint frame_num, guint64 pts_ns,
                                 const TrackerDetection &det, const ShadowCandidate &shadow,
                                 float score) {
  if (!fallback_evidence) return;
  char line[768];
  int n = std::snprintf(line, sizeof(line),
      "{\"camera_id\":\"CAM-%02u\",\"frame_num\":%d,\"pts_ns\":%lu,"
      "\"detector_bbox\":[%.3f,%.3f,%.3f,%.3f],\"detector_confidence\":%.6f,"
      "\"matched_shadow_native_id\":%lu,\"shadow_age\":%u,\"shadow_state\":\"INACTIVE\","
      "\"shadow_confidence\":%.6f,\"association_iou\":%.6f,\"bbox_source\":\"PGIE_FALLBACK\"}\n",
      source_id + 1, frame_num, pts_ns, det.left, det.top, det.width, det.height,
      det.confidence, shadow.id, shadow.age, shadow.confidence, score);
  if (n > 0 && static_cast<size_t>(n) < sizeof(line)) tracker_write_line(fallback_evidence, line, static_cast<size_t>(n));
}

static void tracker_log_lowconf_fallback(guint source_id, gint frame_num, guint64 pts_ns,
                                         const LowConfCandidate &candidate,
                                         const ShadowCandidate &shadow, float score,
                                         const char *verification_basis,
                                         std::int32_t support_age_frames) {
  if (!lowconf_fallback_evidence) return;
  char line[1024];
  const int n = std::snprintf(line, sizeof(line),
      "{\"camera_id\":\"CAM-%02u\",\"frame_num\":%d,\"pts_ns\":%lu,"
      "\"detector_bbox\":[%.3f,%.3f,%.3f,%.3f],\"detector_confidence\":%.6f,"
      "\"verifier_confidence\":%.6f,\"verifier_iou\":%.6f,"
      "\"verifier_latency_ms\":%.3f,\"matched_shadow_native_id\":%lu,"
      "\"shadow_age\":%u,\"shadow_confidence\":%.6f,\"association_iou\":%.6f,"
      "\"verification_basis\":\"%s\",\"verified_track_support_age_frames\":%d,"
      "\"bbox_source\":\"%s\"}\n",
      source_id + 1, frame_num, pts_ns, candidate.left, candidate.top,
      candidate.width, candidate.height, candidate.confidence,
      candidate.verifier_confidence, candidate.verifier_iou,
      candidate.verifier_latency_ms, shadow.id, shadow.age,
      shadow.confidence, score, verification_basis, support_age_frames,
      std::strcmp(verification_basis, "same_frame_context") == 0
          ? "PGIE_VERIFIED_LOWCONF_FALLBACK"
          : "PGIE_RECENT_VERIFIED_TRACK_LOWCONF_FALLBACK");
  if (n > 0 && static_cast<size_t>(n) < sizeof(line))
    tracker_write_line(lowconf_fallback_evidence, line, static_cast<size_t>(n));
}

static const char *tracker_state_name(TRACKER_STATE state) {
  switch (state) {
    case ACTIVE: return "ACTIVE";
    case INACTIVE: return "INACTIVE";
    case TENTATIVE: return "TENTATIVE";
    case PROJECTED: return "PROJECTED";
    case QUASIACTIVE: return "QUASIACTIVE";
    default: return "OTHER";
  }
}

// Diagnostic-only per-frame snapshot of the public NvDCF metadata boundary.
// This is intentionally independent of detector rows: it makes it possible to
// distinguish a detector miss from a native active/shadow lifecycle change.
static void tracker_log_frame_state(
    NvDsFrameMeta *frame, const std::vector<ShadowCandidate> &shadows) {
  if (!tracker_frame_state_evidence || !frame) return;
  std::ostringstream out;
  out << "{\"camera_id\":\"CAM-" << std::setw(2) << std::setfill('0')
      << frame->source_id + 1 << "\",\"source_id\":" << frame->source_id
      << ",\"frame\":" << frame->frame_num << ",\"pts_ns\":" << frame->buf_pts
      << ",\"batch_id\":" << frame->batch_id
      << ",\"pad_index\":" << frame->pad_index
      << ",\"object_meta_count\":" << frame->num_obj_meta
      << ",\"active_objects\":[";
  bool first = true;
  for (NvDsMetaList *node = frame->obj_meta_list; node; node = node->next) {
    auto *obj = static_cast<NvDsObjectMeta *>(node->data);
    if (!obj || obj->class_id != 0 || obj->object_id == UNTRACKED_OBJECT_ID) continue;
    const auto &rect = obj->rect_params;
    if (!first) out << ',';
    first = false;
    out << "{\"id\":" << obj->object_id << ",\"component_id\":"
        << obj->unique_component_id << ",\"confidence\":";
    tracker_json_number(out, obj->confidence);
    out << ",\"tracker_confidence\":";
    tracker_json_number(out, obj->tracker_confidence);
    out << ",\"bbox\":[";
    tracker_json_number(out, rect.left); out << ',';
    tracker_json_number(out, rect.top); out << ',';
    tracker_json_number(out, rect.width); out << ',';
    tracker_json_number(out, rect.height); out << "]}";
  }
  out << "],\"shadow_targets\":[";
  first = true;
  for (const auto &shadow : shadows) {
    if (!first) out << ',';
    first = false;
    out << "{\"id\":" << shadow.id << ",\"state\":\""
        << tracker_state_name(shadow.state) << "\",\"age\":" << shadow.age
        << ",\"confidence\":";
    tracker_json_number(out, shadow.confidence);
    out << ",\"visibility\":";
    tracker_json_number(out, shadow.visibility);
    out << ",\"immediate_continuity\":"
        << (shadow.immediate_continuity ? "true" : "false")
        << ",\"bbox\":[";
    tracker_json_number(out, shadow.left); out << ',';
    tracker_json_number(out, shadow.top); out << ',';
    tracker_json_number(out, shadow.width); out << ',';
    tracker_json_number(out, shadow.height); out << "]}";
  }
  out << "]}\n";
  const std::string bytes = out.str();
  tracker_write_line(tracker_frame_state_evidence, bytes.data(), bytes.size());
}

// Decide whether a canonical publisher needs to wait for the exact-frame
// appearance-verifier result. This mirrors the existing low-confidence
// fallback geometry/continuity gates but does not create an output row. A
// wait is scheduled only for a candidate that has a safe, exact current-frame
// publication dependency: a newly active target joined to the precise detector
// row, or a low-confidence candidate with one unique current shadow match.
// Ambiguous and unsafe frames publish immediately without a fallback.
static bool tracker_lowconf_wait_eligible(
    guint source_id,
    const std::vector<LowConfCandidate> &candidates,
    const std::vector<PublishedTrack> &published,
    const std::vector<ShadowCandidate> &shadows,
    gint frame_num, guint64 pts_ns) {
  struct Choice { size_t shadow_index; float score; };
  std::vector<std::vector<Choice>> choices(candidates.size());
  std::vector<bool> active_duplicate(candidates.size(), false);
  std::vector<bool> ambiguous(candidates.size(), false);
  std::unordered_map<size_t, std::vector<size_t>> target_claims;
  for (size_t ci = 0; ci < candidates.size(); ++ci) {
    const auto &candidate = candidates[ci];
    if (candidate.active_track_birth) {
      if (candidate.verifier_done || !candidate.active_track_id_assigned ||
          candidate.detector_index < 0 || candidate.source_frame_num != frame_num ||
          candidate.source_pts_ns != pts_ns) continue;
      const auto matches = std::count_if(published.begin(), published.end(),
          [&](const PublishedTrack &track) {
            return track.id == candidate.native_track_id &&
                track.requires_birth_verification &&
                track.detection_index == candidate.detector_index;
          });
      if (matches == 1) return true;
      continue;
    }
    if (candidate.shadow_context_refresh) {
      if (!candidate.verifier_done && candidate.shadow_context_pretracker_scheduled) {
        for (const auto &hint : candidate.shadow_targets) {
          if (!hint.native_track_id) continue;
          const bool active_id = std::any_of(published.begin(), published.end(),
              [&](const PublishedTrack &track) { return track.id == hint.native_track_id; });
          const bool current_shadow = std::any_of(shadows.begin(), shadows.end(),
              [&](const ShadowCandidate &shadow) {
                return shadow.id == hint.native_track_id && shadow.state == INACTIVE &&
                    shadow.immediate_continuity;
              });
          if (!active_id && current_shadow) return true;
        }
      }
      continue;
    }
    if (candidate.verifier_done || !std::isfinite(candidate.confidence) || candidate.confidence < .23f ||
        candidate.confidence >= .25f || !std::isfinite(candidate.left) ||
        !std::isfinite(candidate.top) || !std::isfinite(candidate.width) ||
        !std::isfinite(candidate.height) || candidate.width <= 0 || candidate.height <= 0 ||
        candidate.source_frame_num != frame_num || candidate.source_pts_ns != pts_ns)
      continue;
    for (const auto &visible : published) {
      if (tracker_iou(candidate.left, candidate.top, candidate.width, candidate.height,
              visible.left, visible.top, visible.width, visible.height) >= .30f) {
        active_duplicate[ci] = true;
        break;
      }
    }
    if (active_duplicate[ci]) continue;
    for (size_t si = 0; si < shadows.size(); ++si) {
      const auto &shadow = shadows[si];
      if (shadow.state != INACTIVE) continue;
      if (!shadow.immediate_continuity) continue;
      const float width_ratio = std::max(candidate.width, shadow.width) /
          std::max(1.0f, std::min(candidate.width, shadow.width));
      const float height_ratio = std::max(candidate.height, shadow.height) /
          std::max(1.0f, std::min(candidate.height, shadow.height));
      const float score = tracker_iou(candidate.left, candidate.top, candidate.width,
          candidate.height, shadow.left, shadow.top, shadow.width, shadow.height);
      if (score >= .30f && tracker_center_distance(candidate.left, candidate.top,
              candidate.width, candidate.height, shadow.left, shadow.top,
              shadow.width, shadow.height) <= 50.0f && width_ratio <= 2.0f &&
          height_ratio <= 2.0f)
        choices[ci].push_back({si, score});
    }
    std::sort(choices[ci].begin(), choices[ci].end(),
        [](const auto &a, const auto &b) { return a.score > b.score; });
    if (choices[ci].size() > 1 && choices[ci][0].score - choices[ci][1].score < .10f)
      ambiguous[ci] = true;
    if (!ambiguous[ci] && !choices[ci].empty())
      target_claims[choices[ci][0].shadow_index].push_back(ci);
  }
  for (const auto &claim : target_claims)
    if (claim.second.size() > 1)
      for (size_t ci : claim.second) ambiguous[ci] = true;

  std::unordered_set<guint64> active_ids;
  for (const auto &track : published) active_ids.insert(track.id);
  for (size_t ci = 0; ci < candidates.size(); ++ci) {
    if (active_duplicate[ci] || ambiguous[ci] || choices[ci].empty()) continue;
    const guint64 native_id = shadows[choices[ci][0].shadow_index].id;
    if (!active_ids.count(native_id) &&
        !tracker_birth_verification.verified_with_recent_support(source_id, native_id,
            static_cast<std::uint32_t>(frame_num), TRUSTED_LOWCONF_SUPPORT_MAX_AGE_FRAMES))
      return true;
  }
  return false;
}

// Select only low-confidence proposals that already satisfy every current
// shadow association gate before consuming verifier work. The returned
// proposals still do not enter NvDCF; the verifier is only a canonical
// visibility check. Exact frame/PTS and one-to-one matching are rechecked by
// the existing wait and output paths after inference completes.
static std::vector<LowConfCandidate> tracker_lowconf_safe_shadow_candidates(
    const std::vector<LowConfCandidate> &candidates,
    const std::vector<PublishedTrack> &published,
    const std::vector<ShadowCandidate> &shadows,
    gint frame_num, guint64 pts_ns) {
  struct Choice { size_t shadow_index; float score; };
  std::vector<std::vector<Choice>> choices(candidates.size());
  std::vector<bool> ambiguous(candidates.size(), false);
  std::unordered_map<size_t, std::vector<size_t>> shadow_claims;
  for (size_t ci = 0; ci < candidates.size(); ++ci) {
    const auto &candidate = candidates[ci];
    if (candidate.active_track_birth || candidate.shadow_context_refresh ||
        candidate.source_frame_num != frame_num || candidate.source_pts_ns != pts_ns ||
        !std::isfinite(candidate.confidence) || candidate.confidence < .23f ||
        candidate.confidence >= .25f || !std::isfinite(candidate.left) ||
        !std::isfinite(candidate.top) || !std::isfinite(candidate.width) ||
        !std::isfinite(candidate.height) || candidate.left < 0 || candidate.top < 0 ||
        candidate.width <= 0 || candidate.height <= 0) continue;

    const bool duplicates_active = std::any_of(published.begin(), published.end(),
        [&](const PublishedTrack &visible) {
          return tracker_iou(candidate.left, candidate.top, candidate.width,
              candidate.height, visible.left, visible.top, visible.width,
              visible.height) >= .30f;
        });
    if (duplicates_active) continue;

    for (size_t si = 0; si < shadows.size(); ++si) {
      const auto &shadow = shadows[si];
      if (shadow.state != INACTIVE || !shadow.immediate_continuity) continue;
      const bool target_already_active = std::any_of(published.begin(), published.end(),
          [&](const PublishedTrack &visible) { return visible.id == shadow.id; });
      if (target_already_active) continue;
      const float width_ratio = std::max(candidate.width, shadow.width) /
          std::max(1.0f, std::min(candidate.width, shadow.width));
      const float height_ratio = std::max(candidate.height, shadow.height) /
          std::max(1.0f, std::min(candidate.height, shadow.height));
      const float score = tracker_iou(candidate.left, candidate.top,
          candidate.width, candidate.height, shadow.left, shadow.top,
          shadow.width, shadow.height);
      const float center = tracker_center_distance(candidate.left, candidate.top,
          candidate.width, candidate.height, shadow.left, shadow.top,
          shadow.width, shadow.height);
      if (score >= .30f && center <= 50.0f && width_ratio <= 2.0f &&
          height_ratio <= 2.0f) choices[ci].push_back({si, score});
    }
    std::sort(choices[ci].begin(), choices[ci].end(),
        [](const Choice &a, const Choice &b) { return a.score > b.score; });
    if (choices[ci].size() > 1 &&
        choices[ci][0].score - choices[ci][1].score < .10f)
      ambiguous[ci] = true;
    if (!ambiguous[ci] && !choices[ci].empty())
      shadow_claims[choices[ci][0].shadow_index].push_back(ci);
  }
  for (const auto &claim : shadow_claims)
    if (claim.second.size() > 1)
      for (size_t ci : claim.second) ambiguous[ci] = true;

  std::vector<LowConfCandidate> result;
  for (size_t ci = 0; ci < candidates.size(); ++ci)
    if (!ambiguous[ci] && !choices[ci].empty()) result.push_back(candidates[ci]);
  return result;
}

static void tracker_log_shadow_context_fallback(
    guint source_id, gint frame_num, guint64 pts_ns,
    const LowConfCandidate &candidate, const ShadowCandidate &shadow, float score) {
  if (!lowconf_fallback_evidence) return;
  char line[1152];
  const int n = std::snprintf(line, sizeof(line),
      "{\"camera_id\":\"CAM-%02u\",\"frame_num\":%d,\"pts_ns\":%lu,"
      "\"native_track_id\":%lu,\"source_frame_num\":%d,\"source_pts_ns\":%lu,"
      "\"shadow_state\":\"INACTIVE\",\"shadow_age\":%u,\"shadow_confidence\":%.6f,"
      "\"shadow_bbox_hint\":[%.3f,%.3f,%.3f,%.3f],"
      "\"visible_bbox\":[%.3f,%.3f,%.3f,%.3f],\"verifier_confidence\":%.6f,"
      "\"association_iou\":%.6f,\"verifier_latency_ms\":%.3f,"
      "\"bbox_source\":\"SHADOW_CONTEXT_VERIFIED\"}\n",
      source_id + 1, frame_num, pts_ns, candidate.native_track_id,
      candidate.source_frame_num, candidate.source_pts_ns, shadow.age, shadow.confidence,
      shadow.left, shadow.top, shadow.width, shadow.height,
      candidate.verifier_bbox_left, candidate.verifier_bbox_top,
      candidate.verifier_bbox_width, candidate.verifier_bbox_height,
      candidate.verifier_confidence, score, candidate.verifier_latency_ms);
  if (n > 0 && static_cast<size_t>(n) < sizeof(line))
    tracker_write_line(lowconf_fallback_evidence, line, static_cast<size_t>(n));
}

static void tracker_apply_verified_shadow_context(
    guint source_id, gint frame_num, guint64 pts_ns, guint frame_width,
    guint frame_height, std::vector<PublishedTrack> *published,
    const std::vector<LowConfCandidate> &verified_candidates,
    const std::vector<ShadowCandidate> &shadows,
    const std::unordered_set<guint64> &eligible_native_ids) {
  if (!published || source_id >= 6 || verified_candidates.empty() ||
      eligible_native_ids.empty()) return;
  for (const auto &candidate : verified_candidates) {
    if (!candidate.shadow_context_refresh ||
        !candidate.shadow_context_pretracker_scheduled || !candidate.verifier_done ||
        candidate.source_frame_num != frame_num || candidate.source_pts_ns != pts_ns ||
        candidate.shadow_targets.empty() || candidate.verifier_boxes.empty()) continue;

    std::vector<const LowConfShadowTarget*> accepted_targets;
    std::vector<const ShadowCandidate*> current_shadows;
    std::vector<ShadowContextRect> target_rects;
    for (const auto &hint : candidate.shadow_targets) {
      if (!hint.native_track_id || !eligible_native_ids.count(hint.native_track_id)) continue;
      const auto shadow_it = std::find_if(shadows.begin(), shadows.end(),
          [&](const ShadowCandidate &shadow) {
            return shadow.id == hint.native_track_id && shadow.state == INACTIVE &&
                shadow.immediate_continuity;
          });
      if (shadow_it == shadows.end()) continue;
      const bool already_active = std::any_of(published->begin(), published->end(),
          [&](const PublishedTrack &track) { return track.id == hint.native_track_id; });
      if (already_active) continue;
      accepted_targets.push_back(&hint);
      current_shadows.push_back(&*shadow_it);
      target_rects.push_back({shadow_it->left, shadow_it->top,
          shadow_it->width, shadow_it->height});
    }
    if (target_rects.empty()) continue;

    std::vector<const LowConfVerifiedBox*> valid_boxes;
    std::vector<ShadowContextRect> proposal_rects;
    for (const auto &box : candidate.verifier_boxes) {
      if (!std::isfinite(box.confidence) || box.confidence < .20f ||
          !std::isfinite(box.left) || !std::isfinite(box.top) ||
          !std::isfinite(box.width) || !std::isfinite(box.height) ||
          box.left < 0 || box.top < 0 || box.width <= 0 || box.height <= 0 ||
          box.left + box.width > frame_width + .1f ||
          box.top + box.height > frame_height + .1f) continue;
      valid_boxes.push_back(&box);
      proposal_rects.push_back({box.left, box.top, box.width, box.height});
    }
    const auto matches = shadow_context_match_one_to_one(target_rects, proposal_rects);
    for (const auto &match : matches) {
      const auto &hint = *accepted_targets[match.target_index];
      const auto &shadow = *current_shadows[match.target_index];
      const auto &box = *valid_boxes[match.proposal_index];
      const bool already_present = std::any_of(published->begin(), published->end(),
          [&](const PublishedTrack &track) {
            return track.id == hint.native_track_id ||
                tracker_iou(box.left, box.top, box.width, box.height,
                    track.left, track.top, track.width, track.height) >= .30f;
          });
      if (already_present) continue;

    PublishedTrack row;
      row.id = hint.native_track_id;
    row.source = "SHADOW_CONTEXT_VERIFIED";
      row.detector_confidence = box.confidence;
    row.tracker_confidence = shadow.confidence;
      row.verifier_confidence = box.confidence;
      row.left = box.left;
      row.top = box.top;
      row.width = box.width;
      row.height = box.height;
    row.shadow_age = shadow.age;
      row.association_iou = match.iou;
    published->push_back(row);
      LowConfCandidate matched_candidate = candidate;
      matched_candidate.native_track_id = hint.native_track_id;
      matched_candidate.verifier_confidence = box.confidence;
      matched_candidate.verifier_iou = match.iou;
      matched_candidate.verifier_bbox_left = box.left;
      matched_candidate.verifier_bbox_top = box.top;
      matched_candidate.verifier_bbox_width = box.width;
      matched_candidate.verifier_bbox_height = box.height;
      tracker_log_shadow_context_fallback(source_id, frame_num, pts_ns,
          matched_candidate, shadow, match.iou);
    ++tracker_counters[source_id].shadow_context_fallbacks;
    }
  }
}

static void tracker_apply_verified_lowconf(
    guint source_id, gint frame_num, guint64 pts_ns,
    std::vector<PublishedTrack> *published,
    const std::vector<LowConfCandidate> &verified_lowconf,
    const std::vector<ShadowCandidate> &shadows) {
  if (!published || source_id >= 6 || verified_lowconf.empty()) return;
  struct LowEligible { size_t shadow_index; float score; };
  std::vector<std::vector<LowEligible>> low_eligible(verified_lowconf.size());
  std::vector<bool> low_active_duplicate(verified_lowconf.size(), false);
  std::vector<bool> low_ambiguous(verified_lowconf.size(), false);
  std::unordered_map<size_t, std::vector<size_t>> low_shadow_claims;
  for (size_t ci = 0; ci < verified_lowconf.size(); ++ci) {
    const auto &candidate = verified_lowconf[ci];
    if (candidate.confidence < .23f || candidate.confidence >= .25f ||
        candidate.source_frame_num != frame_num || candidate.source_pts_ns != pts_ns) continue;
    for (const auto &visible : *published) {
      if (tracker_iou(candidate.left, candidate.top, candidate.width, candidate.height,
              visible.left, visible.top, visible.width, visible.height) >= .30f) {
        low_active_duplicate[ci] = true;
        break;
      }
    }
    if (low_active_duplicate[ci]) continue;
    for (size_t si = 0; si < shadows.size(); ++si) {
      const auto &shadow = shadows[si];
      if (shadow.state != INACTIVE) continue;
      if (!shadow.immediate_continuity) continue;
      const float width_ratio = std::max(candidate.width, shadow.width) /
          std::max(1.0f, std::min(candidate.width, shadow.width));
      const float height_ratio = std::max(candidate.height, shadow.height) /
          std::max(1.0f, std::min(candidate.height, shadow.height));
      const float score = tracker_iou(candidate.left, candidate.top, candidate.width,
          candidate.height, shadow.left, shadow.top, shadow.width, shadow.height);
      if (score >= .30f && tracker_center_distance(candidate.left, candidate.top,
              candidate.width, candidate.height, shadow.left, shadow.top,
              shadow.width, shadow.height) <= 50.0f && width_ratio <= 2.0f && height_ratio <= 2.0f)
        low_eligible[ci].push_back({si, score});
    }
    std::sort(low_eligible[ci].begin(), low_eligible[ci].end(),
        [](const auto &a, const auto &b) { return a.score > b.score; });
    if (low_eligible[ci].size() > 1 &&
        low_eligible[ci][0].score - low_eligible[ci][1].score < .10f)
      low_ambiguous[ci] = true;
    if (!low_ambiguous[ci] && !low_eligible[ci].empty())
      low_shadow_claims[low_eligible[ci][0].shadow_index].push_back(ci);
  }
  for (const auto &claim : low_shadow_claims)
    if (claim.second.size() > 1)
      for (size_t ci : claim.second) low_ambiguous[ci] = true;

  std::unordered_set<guint64> active_ids;
  for (const auto &row : *published) active_ids.insert(row.id);
  for (size_t ci = 0; ci < verified_lowconf.size(); ++ci) {
    const auto &candidate = verified_lowconf[ci];
    if (low_active_duplicate[ci] || low_ambiguous[ci] || low_eligible[ci].empty()) continue;
    const auto &choice = low_eligible[ci][0];
    const auto &shadow = shadows[choice.shadow_index];
    if (active_ids.count(shadow.id)) continue;
    if (tracker_birth_verification.rejected(source_id, shadow.id)) continue;
    const bool same_frame_context = candidate.verifier_done && candidate.verifier_confirmed;
    const std::int32_t support_age = tracker_birth_verification.verified_support_age(
        source_id, shadow.id, static_cast<std::uint32_t>(frame_num));
    const bool verification_authorized =
        tracker_birth_verification.lowconf_fallback_authorized(source_id, shadow.id,
            static_cast<std::uint32_t>(frame_num), candidate.verifier_done,
            candidate.verifier_confirmed, TRUSTED_LOWCONF_SUPPORT_MAX_AGE_FRAMES);
    if (!verification_authorized) continue;
    PublishedTrack row;
    row.id = shadow.id;
    row.source = same_frame_context ? "PGIE_VERIFIED_LOWCONF_FALLBACK"
                                    : "PGIE_RECENT_VERIFIED_TRACK_LOWCONF_FALLBACK";
    row.detector_confidence = candidate.confidence;
    row.tracker_confidence = shadow.confidence;
    row.verifier_confidence = same_frame_context ? candidate.verifier_confidence : NAN;
    row.left = candidate.left; row.top = candidate.top;
    row.width = candidate.width; row.height = candidate.height;
    row.shadow_age = shadow.age; row.association_iou = choice.score;
    published->push_back(row);
    active_ids.insert(shadow.id);
    tracker_log_lowconf_fallback(source_id, frame_num, pts_ns, candidate, shadow,
        choice.score, same_frame_context ? "same_frame_context" : "recent_verified_track",
        same_frame_context ? -1 : support_age);
    if (same_frame_context) ++tracker_counters[source_id].verified_lowconf_fallbacks;
    else ++tracker_counters[source_id].trusted_track_lowconf_fallbacks;
  }
}

// Context verification affects only canonical visibility. The original PGIE
// object and its NvDCF association remain intact; an unknown/new native ID is
// withheld until one exact-frame crop result uniquely confirms that location.
static void tracker_apply_active_birth_verification(
    guint source_id, gint frame_num, guint64 pts_ns,
    std::vector<PublishedTrack> *published,
    const std::vector<LowConfCandidate> &verifier_results) {
  if (!active_track_verifier_enabled() || !published || source_id >= 6) return;
  // Birth crops are submitted at the NvDCF input and may finish while the
  // tracker runs. Join only a completed decision for this exact source frame,
  // PTS, detector index and native target. Late or ambiguous results fail
  // closed; the visible rectangle remains the current NvDCF bbox.
  const char *fail_closed = g_getenv("V11_ACTIVE_TRACK_VERIFIER_FAIL_CLOSED");
  const bool require_positive = fail_closed && std::strcmp(fail_closed, "1") == 0;
  published->erase(std::remove_if(published->begin(), published->end(),
      [source_id, frame_num, pts_ns, require_positive, &verifier_results]
      (const PublishedTrack &row) {
        if (!row.requires_birth_verification) return false;
        if (tracker_birth_verification.rejected(source_id, row.id)) {
          ++tracker_counters[source_id].active_birth_verifier_withheld;
          return true;
        }
        if (tracker_birth_verification.verified(source_id, row.id)) return false;
        const LowConfCandidate *decision = nullptr;
        for (const auto &candidate : verifier_results) {
          if (!candidate.active_track_birth || !candidate.active_track_id_assigned ||
              candidate.native_track_id != row.id ||
              candidate.detector_index != row.detection_index ||
              candidate.source_frame_num != frame_num || candidate.source_pts_ns != pts_ns)
            continue;
          if (decision) { decision = nullptr; break; }
          decision = &candidate;
        }
        if (decision && decision->verifier_done && decision->verifier_confirmed) {
          tracker_birth_verification.trust(source_id, row.id,
              static_cast<std::uint32_t>(frame_num));
          return false;
        }
        if (decision && decision->verifier_done && decision->verifier_no_local_person)
          tracker_birth_verification.reject(source_id, row.id,
              static_cast<std::uint32_t>(frame_num));
        const bool reject = require_positive &&
            (!decision || !decision->verifier_done || !decision->verifier_confirmed);
        if (reject) ++tracker_counters[source_id].active_birth_verifier_withheld;
        return reject || tracker_birth_verification.rejected(source_id, row.id);
      }), published->end());
}

static void tracker_schedule_shadow_context_before_nvdcf(
    GstBuffer *buffer, NvDsFrameMeta *frame) {
  if (!buffer || !frame || frame->source_id >= 6 || frame->frame_num <= 0 ||
      !active_track_verifier_enabled() || !shadow_context_refresh_enabled()) return;

  const guint source_id = frame->source_id;
  const guint frame_width = tracker_bbox_coordinate_extent(frame, true);
  const guint frame_height = tracker_bbox_coordinate_extent(frame, false);
  const LowConfFrameKey key{source_id, frame->frame_num, frame->buf_pts};
  std::unordered_map<guint64, ActiveStamp> prior_active;
  {
    std::lock_guard<std::mutex> guard(tracker_previous_active_mutex[source_id]);
    for (const auto &entry : tracker_previous_active_boxes[source_id])
      if (entry.second.frame_num + 1 == static_cast<guint32>(frame->frame_num))
        prior_active.emplace(entry.first, entry.second);
  }

  // Current detector boxes are used only to avoid redundant context jobs.
  // They do not affect inference or alter NvDCF's object metadata.
  std::vector<ShadowContextRect> current_detector_boxes;
  for (NvDsMetaList *node = frame->obj_meta_list; node; node = node->next) {
    const auto *object = static_cast<NvDsObjectMeta*>(node->data);
    if (!object || object->class_id != 0) continue;
    const auto &rect = object->rect_params;
    current_detector_boxes.push_back({rect.left, rect.top, rect.width, rect.height});
  }
  const auto current_lowconf = lowconf_verifier_frame_candidates(key);
  std::unordered_map<guint64, ShadowStamp> prior_shadows;
  for (const auto &entry : tracker_previous_shadows[source_id])
    if (entry.second.frame_num + 1 == static_cast<guint32>(frame->frame_num) &&
        entry.second.state == INACTIVE)
      prior_shadows.emplace(entry.first, entry.second);

  struct EligibleHint {
    guint64 native_id = 0;
    guint32 prior_frame = 0;
    bool was_active = false;
    bool was_inactive_shadow = false;
    ShadowContextRect rect;
  };
  std::vector<EligibleHint> eligible_hints;
  auto collect_hint = [&](guint64 native_id, guint32 prior_frame,
                          bool was_active, bool was_inactive_shadow,
                          const ShadowContextRect &hint) {
    if (!tracker_auxiliary_target_ids[source_id].count(native_id) ||
        !tracker_birth_verification.verified(source_id, native_id)) return;
    const auto primary_support = tracker_last_primary_association_frame[source_id].find(native_id);
    if (primary_support != tracker_last_primary_association_frame[source_id].end() &&
        shadow_context_has_recent_primary_support(frame->frame_num,
            static_cast<int>(primary_support->second), 2)) return;
    const bool detector_match =
        std::any_of(current_detector_boxes.begin(), current_detector_boxes.end(),
            [&](const ShadowContextRect &box) { return shadow_context_spatial_match(hint, box); }) ||
        std::any_of(current_lowconf.begin(), current_lowconf.end(),
            [&](const LowConfCandidate &candidate) {
              return !candidate.shadow_context_refresh &&
                  shadow_context_spatial_match(hint, ShadowContextRect{
                      candidate.left, candidate.top, candidate.width, candidate.height});
            });
    if (!shadow_context_pretracker_hint_eligible(frame->frame_num,
            static_cast<int>(prior_frame), native_id, true, true,
            was_active, was_inactive_shadow, detector_match, hint,
            frame_width, frame_height)) return;
    eligible_hints.push_back({native_id, prior_frame, was_active,
        was_inactive_shadow, hint});
  };

  for (const auto &entry : prior_active) {
    const auto &stamp = entry.second;
    collect_hint(entry.first, stamp.frame_num, true, false,
        ShadowContextRect{stamp.left, stamp.top, stamp.width, stamp.height});
  }
  for (const auto &entry : prior_shadows) {
    const auto &stamp = entry.second;
    collect_hint(entry.first, stamp.frame_num, false, true,
        ShadowContextRect{stamp.left, stamp.top, stamp.width, stamp.height});
  }
  std::sort(eligible_hints.begin(), eligible_hints.end(),
      [](const EligibleHint &a, const EligibleHint &b) { return a.native_id < b.native_id; });
  eligible_hints.erase(std::unique(eligible_hints.begin(), eligible_hints.end(),
      [](const EligibleHint &a, const EligibleHint &b) { return a.native_id == b.native_id; }),
      eligible_hints.end());

  guint surface_width = 0, surface_height = 0;
  if (!eligible_hints.empty() && lowconf_verifier_surface_dimensions(
          buffer, frame->batch_id, &surface_width, &surface_height)) {
    std::vector<ShadowContextRect> hint_rects;
    hint_rects.reserve(eligible_hints.size());
    for (const auto &hint : eligible_hints) hint_rects.push_back(hint.rect);
    const auto groups = shadow_context_group_hints(surface_width, surface_height,
        frame_width, frame_height, hint_rects);
    for (const auto &group : groups) {
      if (group.empty()) continue;
      const auto &anchor = eligible_hints[group.front()];
      LowConfCandidate candidate;
      candidate.confidence = 0.0f;  // prior target geometry is only a crop hint
      candidate.left = anchor.rect.left;
      candidate.top = anchor.rect.top;
      candidate.width = anchor.rect.width;
      candidate.height = anchor.rect.height;
      candidate.native_track_id = anchor.native_id;
      candidate.shadow_targets.reserve(group.size());
      for (size_t index : group) {
        const auto &hint = eligible_hints[index];
        candidate.shadow_targets.push_back({hint.native_id, hint.rect.left,
            hint.rect.top, hint.rect.width, hint.rect.height});
      }
      (void)lowconf_verifier_submit(buffer, frame->batch_id, key,
          frame_width, frame_height, candidate, false, true);
    }
  }
}

static GstPadProbeReturn tracker_enter(GstPad*, GstPadProbeInfo *info, gpointer) {
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  NvDsBatchMeta *batch = buffer ? gst_buffer_get_nvds_batch_meta(buffer) : nullptr;
  if (!batch) return GST_PAD_PROBE_OK;
  const guint64 now_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  for (NvDsMetaList *node = batch->frame_meta_list; node; node = node->next) {
    auto *frame = static_cast<NvDsFrameMeta*>(node->data);
    if (!frame || frame->source_id >= 6 || frame->frame_num < 0 ||
        frame->buf_pts == GST_CLOCK_TIME_NONE) continue;
    const size_t slot_index = static_cast<size_t>(frame->frame_num) % TRACKER_INPUT_STAMP_CAPACITY;
    {
      std::lock_guard<std::mutex> guard(tracker_input_stamp_mutex[frame->source_id]);
      tracker_input_stamps[frame->source_id][slot_index] = {
          frame->frame_num, frame->buf_pts, now_ns};
    }
    tracker_schedule_shadow_context_before_nvdcf(buffer, frame);
  }
  // Active-target birth checks remain post-tracker because they require a new
  // NvDCF ID association. Shadow refresh is scheduled above only for a
  // previously appearance-verified auxiliary target, using the immediately
  // preceding target box as a crop hint and this exact current GstBuffer.
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn tracker_exit(GstPad*, GstPadProbeInfo *info, gpointer) {
  const guint64 callback_start_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
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
    {
      const size_t slot_index = static_cast<size_t>(frame->frame_num) % TRACKER_INPUT_STAMP_CAPACITY;
      std::lock_guard<std::mutex> guard(tracker_input_stamp_mutex[frame->source_id]);
      const auto &stamp = tracker_input_stamps[frame->source_id][slot_index];
      if (stamp.frame_num == frame->frame_num && stamp.pts_ns == frame->buf_pts &&
          stamp.entered_ns && callback_start_ns >= stamp.entered_ns) {
        tracker_element_latency[frame->source_id].add(callback_start_ns - stamp.entered_ns);
        tracker_input_stamps[frame->source_id][slot_index] = {};
      } else {
        ++tracker_input_stamp_misses[frame->source_id];
      }
    }
    ++counter.frames;
    std::unordered_set<guint64> frame_ids;
    std::unordered_set<guint64> current_active_ids;
    std::vector<PublishedTrack> published;
    const guint64 shadow_scan_start_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
    std::vector<ShadowCandidate> shadows = tracker_current_shadows(batch, frame);
    const guint64 shadow_scan_done_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
    tracker_shadow_scan_latency[frame->source_id].add(shadow_scan_done_ns - shadow_scan_start_ns);
    tracker_log_frame_state(frame, shadows);
    std::vector<TrackerDetection> detections;
    detections = tracker_take_detections(frame->source_id, frame->frame_num, frame->buf_pts);
    const LowConfFrameKey lowconf_key{frame->source_id, frame->frame_num, frame->buf_pts};
    std::vector<LowConfCandidate> lowconf_sidecars = context_verifier_enabled()
        ? tracker_take_lowconf_candidates(frame->source_id, frame->frame_num, frame->buf_pts)
        : std::vector<LowConfCandidate>{};

    for (NvDsMetaList *on = frame->obj_meta_list; on; on = on->next) {
      auto *obj = static_cast<NvDsObjectMeta*>(on->data);
      if (!obj || obj->class_id != 0) continue;
      if (obj->object_id == UNTRACKED_OBJECT_ID) {
        ++counter.untracked;
        continue;
      }
      ++counter.objects;
      current_active_ids.insert(obj->object_id);
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
      PublishedTrack row;
      row.id = obj->object_id;
      row.tracker_confidence = obj->tracker_confidence;
      row.left = r.left; row.top = r.top; row.width = r.width; row.height = r.height;
      if (obj->unique_component_id == TRACKER_AUXILIARY_GIE_ID)
        tracker_auxiliary_target_ids[frame->source_id].insert(obj->object_id);
      // Preserve the user-reviewed CAM-02 jacket safeguard: when the context
      // verifier is enabled, every new canonical target (including a
      // high-confidence 640-primary target) must be appearance-verified.
      // The auxiliary ID test remains as a defensive requirement if an
      // auxiliary track survives a verifier-mode transition. This changes
      // publication eligibility only; detector proposals still reach NvDCF
      // with their original geometry/confidence.
      row.requires_birth_verification = active_track_verifier_enabled() ||
          tracker_auxiliary_target_ids[frame->source_id].count(obj->object_id) != 0;
      published.push_back(row);
    }

    // Match detections to active objects with the approved v4 greedy IoU gate.
    std::vector<std::pair<float, size_t>> active_pairs;
    for (size_t di = 0; di < detections.size(); ++di) {
      for (size_t ai = 0; ai < published.size(); ++ai) {
        const auto &a = detections[di]; const auto &t = published[ai];
        const float score = tracker_iou(a.left, a.top, a.width, a.height, t.left, t.top, t.width, t.height);
        if (score >= 0.30f) active_pairs.emplace_back(score, di * published.size() + ai);
      }
    }
    std::sort(active_pairs.begin(), active_pairs.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    std::unordered_map<size_t, size_t> detection_to_active;
    std::unordered_set<size_t> used_detections, used_active;
    for (const auto &pair : active_pairs) {
      const size_t di = pair.second / published.size(), ai = pair.second % published.size();
      if (!published.empty() && !used_detections.count(di) && !used_active.count(ai)) {
        detection_to_active[di] = ai; used_detections.insert(di); used_active.insert(ai);
      }
    }
    for (const auto &mapping : detection_to_active) {
      auto &row = published[mapping.second];
      const auto &detection = detections[mapping.first];
      row.detector_confidence = detection.confidence;
      row.detection_index = static_cast<int>(mapping.first);
      if (detection.auxiliary) {
        tracker_auxiliary_target_ids[frame->source_id].insert(row.id);
        row.requires_birth_verification = true;
      } else {
        tracker_last_primary_association_frame[frame->source_id][row.id] =
            static_cast<guint32>(frame->frame_num);
      }
    }
    for (auto it = tracker_last_primary_association_frame[frame->source_id].begin();
         it != tracker_last_primary_association_frame[frame->source_id].end();) {
      if (frame->frame_num >= 0 &&
          static_cast<guint32>(frame->frame_num) > it->second + 5)
        it = tracker_last_primary_association_frame[frame->source_id].erase(it);
      else ++it;
    }
    std::vector<int> active_to_detection(published.size(), -1);
    for (const auto &mapping : detection_to_active)
      active_to_detection[mapping.second] = static_cast<int>(mapping.first);
    const size_t active_output_count = published.size();
    if (active_track_verifier_enabled()) {
      // Verify only detector rows that NvDCF actually associated with a
      // current native target. This excludes proposal-only work that would
      // otherwise queue behind the person target the verifier is meant to
      // validate. The crop still uses this exact source frame and the original
      // PGIE rectangle; NvDCF input/state are already complete and unchanged.
      for (size_t ai = 0; ai < active_output_count; ++ai) {
        const auto &row = published[ai];
        if (!row.requires_birth_verification || row.detection_index < 0 ||
            static_cast<size_t>(row.detection_index) >= detections.size()) continue;
        const auto state = tracker_birth_verification.state(frame->source_id, row.id);
        if (state == TrackBirthVerificationLedger::State::Verified ||
            state == TrackBirthVerificationLedger::State::Pending ||
            state == TrackBirthVerificationLedger::State::Rejected) continue;
        if (!tracker_birth_verification.begin(frame->source_id, row.id,
                static_cast<std::uint32_t>(frame->frame_num))) continue;
        const auto &detection = detections[static_cast<size_t>(row.detection_index)];
        LowConfCandidate candidate;
        candidate.confidence = detection.confidence;
        candidate.left = detection.left; candidate.top = detection.top;
        candidate.width = detection.width; candidate.height = detection.height;
        candidate.detector_index = row.detection_index;
        candidate.native_track_id = row.id;
        candidate.active_track_id_assigned = true;
        const auto submitted = lowconf_verifier_submit(buffer, frame->batch_id,
            lowconf_key, tracker_bbox_coordinate_extent(frame, true),
            tracker_bbox_coordinate_extent(frame, false), candidate, true);
        if (submitted != LowConfSubmitResult::Submitted ||
            !lowconf_verifier_assign_active_track(lowconf_key,
                row.detection_index, row.id)) {
          // A queue/correlation failure is not a negative appearance result.
          // Keep the target unpublished and allow one controlled retry rather
          // than leaving its birth ledger permanently pending.
          tracker_birth_verification.cancel(frame->source_id, row.id,
              frame->frame_num, 1);
        }
      }
    }

    // Low-confidence crops were submitted at the PGIE boundary to overlap
    // verifier work with NvDCF. They remain only a sidecar here: the candidate
    // was removed before the tracker, and canonical publication rechecks the
    // exact current shadow association before a fallback can be emitted.
    (void)lowconf_sidecars;

    {
      std::unordered_map<guint64, ActiveStamp> current_active_boxes;
      for (size_t ai = 0; ai < active_output_count; ++ai) {
        const auto &row = published[ai];
        current_active_boxes.emplace(row.id, ActiveStamp{
            static_cast<guint32>(frame->frame_num), row.left, row.top,
            row.width, row.height, row.tracker_confidence});
      }
      std::lock_guard<std::mutex> guard(
          tracker_previous_active_mutex[frame->source_id]);
      tracker_previous_active_boxes[frame->source_id] =
          std::move(current_active_boxes);
    }

    struct Eligible { size_t shadow_index; float score; };
    std::unordered_map<size_t, std::vector<Eligible>> eligible;
    std::vector<bool> continuous_shadow(detections.size(), false);
    for (size_t di = 0; di < detections.size(); ++di) {
      if (detection_to_active.count(di)) continue;
      const auto &det = detections[di];
      for (size_t si = 0; si < shadows.size(); ++si) {
        const auto &sh = shadows[si];
        if (sh.state != INACTIVE) continue;
        const bool continuous = sh.immediate_continuity;
        if (continuous) continuous_shadow[di] = true;
        if (!continuous) continue;
        const float width_ratio = std::max(det.width, sh.width) / std::max(1.0f, std::min(det.width, sh.width));
        const float height_ratio = std::max(det.height, sh.height) / std::max(1.0f, std::min(det.height, sh.height));
        const float score = tracker_iou(det.left, det.top, det.width, det.height, sh.left, sh.top, sh.width, sh.height);
        if (score >= 0.30f && tracker_center_distance(det.left, det.top, det.width, det.height,
              sh.left, sh.top, sh.width, sh.height) <= 50.0f && width_ratio <= 2.0f && height_ratio <= 2.0f)
          eligible[di].push_back({si, score});
      }
      std::sort(eligible[di].begin(), eligible[di].end(), [](const auto &a, const auto &b) { return a.score > b.score; });
    }
    std::unordered_set<size_t> ambiguous;
    for (const auto &entry : eligible) {
      const auto &options = entry.second;
      if (options.size() > 1 && options[0].score - options[1].score < 0.10f) ambiguous.insert(entry.first);
    }
    std::unordered_map<size_t, std::vector<size_t>> shadow_claims;
    for (const auto &entry : eligible)
      if (!ambiguous.count(entry.first) && !entry.second.empty()) shadow_claims[entry.second[0].shadow_index].push_back(entry.first);
    for (const auto &entry : shadow_claims)
      if (entry.second.size() > 1) for (size_t di : entry.second) ambiguous.insert(di);
    counter.ambiguous_matches += ambiguous.size();
    std::unordered_set<guint64> active_ids;
    for (const auto &row : published) active_ids.insert(row.id);
    if (active_track_verifier_enabled()) {
      for (const auto &decision : lowconf_verifier_take_completed_active(frame->source_id)) {
        const gint age = frame->frame_num - decision.request_frame_num;
        const bool recent = age >= 0 && age <= 20;
        const bool target_current = active_ids.count(decision.native_track_id) != 0 ||
            std::any_of(shadows.begin(), shadows.end(), [&](const ShadowCandidate &shadow) {
              return shadow.id == decision.native_track_id;
            });
        const bool applicable = recent && target_current;
        if (applicable && decision.confirmed) {
          tracker_birth_verification.trust(frame->source_id, decision.native_track_id,
              static_cast<std::uint32_t>(frame->frame_num));
          ++counter.active_birth_verifier_confirmed;
        } else if (applicable && decision.definitive_no_local_person) {
          tracker_birth_verification.reject(frame->source_id, decision.native_track_id,
              static_cast<std::uint32_t>(frame->frame_num));
          ++counter.active_birth_verifier_withheld;
        } else {
          ++counter.active_birth_verifier_withheld;
        }
      }
    }
    // Refresh recent support only from a detector proposal that NvDCF matched
    // to this same native target. Extrapolated active boxes and INACTIVE shadow
    // age do not refresh the bounded low-confidence fallback authorization.
    for (const auto &mapping : detection_to_active) {
      const auto &row = published[mapping.second];
      tracker_birth_verification.support_verified_target(frame->source_id, row.id,
          static_cast<std::uint32_t>(frame->frame_num));
    }
    for (size_t di = 0; di < detections.size(); ++di) {
      if (detection_to_active.count(di) || ambiguous.count(di) || eligible[di].empty()) continue;
      const auto &choice = eligible[di][0]; const auto &sh = shadows[choice.shadow_index];
      if (active_ids.count(sh.id)) { ++counter.ambiguous_matches; continue; }
      const auto &det = detections[di];
      PublishedTrack row;
      row.id = sh.id; row.source = "PGIE_SHADOW_FALLBACK";
      row.requires_birth_verification = active_track_verifier_enabled() ||
          tracker_auxiliary_target_ids[frame->source_id].count(sh.id) != 0;
      row.detector_confidence = det.confidence; row.tracker_confidence = sh.confidence;
      row.left = det.left; row.top = det.top; row.width = det.width; row.height = det.height;
      row.shadow_age = sh.age; row.association_iou = choice.score;
      row.detection_index = static_cast<int>(di);
      published.push_back(row);
      active_ids.insert(sh.id);
      tracker_log_fallback(frame->source_id, frame->frame_num, frame->buf_pts, det, sh, choice.score);
      ++counter.fallbacks;
    }

    if (active_track_verifier_enabled()) {
      for (const auto &row : published)
        if (row.requires_birth_verification)
          tracker_birth_verification.touch(frame->source_id, row.id,
              static_cast<guint32>(frame->frame_num));
      for (const auto &shadow : shadows)
        if (tracker_auxiliary_target_ids[frame->source_id].count(shadow.id))
          tracker_birth_verification.touch(frame->source_id, shadow.id,
              static_cast<guint32>(frame->frame_num));

    }

    // Shadow context requests are scheduled at tracker_enter, before NvDCF,
    // from immediately preceding active/shadow geometry. Recover exact-frame
    // request IDs here; only same-frame INACTIVE shadows can consume results.
    std::unordered_set<guint64> shadow_refresh_ids;
    if (active_track_verifier_enabled() && shadow_context_refresh_enabled()) {
      const auto pending_frame_candidates = lowconf_verifier_frame_candidates(lowconf_key);
      for (const auto &candidate : pending_frame_candidates) {
        if (!candidate.shadow_context_refresh) continue;
        if (candidate.native_track_id) shadow_refresh_ids.insert(candidate.native_track_id);
        for (const auto &target : candidate.shadow_targets)
          if (target.native_track_id) shadow_refresh_ids.insert(target.native_track_id);
      }
    }

    if (tracker_decision_evidence) {
      for (size_t di = 0; di < detections.size(); ++di) {
        const auto &det = detections[di];
        const char *outcome = "NO_CURRENT_SHADOW";
        guint64 selected_id = 0;
        if (detection_to_active.count(di)) {
          outcome = "ACTIVE";
          selected_id = published[detection_to_active[di]].id;
        } else if (ambiguous.count(di)) {
          outcome = "AMBIGUOUS";
        } else if (eligible[di].empty()) {
          bool had_inactive = false, failed_spatial = false;
          for (const auto &sh : shadows) {
            if (sh.state != INACTIVE) continue;
            had_inactive = true;
            const float iou = tracker_iou(det.left, det.top, det.width, det.height,
                                          sh.left, sh.top, sh.width, sh.height);
            const float center = tracker_center_distance(det.left, det.top, det.width, det.height,
                                                          sh.left, sh.top, sh.width, sh.height);
            const float wr = std::max(det.width, sh.width) /
                std::max(1.0f, std::min(det.width, sh.width));
            const float hr = std::max(det.height, sh.height) /
                std::max(1.0f, std::min(det.height, sh.height));
            if (iou < 0.30f || center > 50.0f || wr > 2.0f || hr > 2.0f) failed_spatial = true;
          }
          if (had_inactive && !continuous_shadow[di]) outcome = "NO_CURRENT_SHADOW";
          else if (had_inactive && failed_spatial) outcome = "SPATIAL_GATE_REJECT";
          else if (shadows.empty()) outcome = "NO_CURRENT_SHADOW";
          else outcome = "NO_INACTIVE_SHADOW";
        } else {
          outcome = "PGIE_FALLBACK";
          selected_id = shadows[eligible[di][0].shadow_index].id;
        }
        std::ostringstream row;
        row << "{\"camera_id\":\"CAM-" << std::setw(2) << std::setfill('0') << frame->source_id + 1
            << "\",\"source_id\":" << frame->source_id << ",\"frame\":" << frame->frame_num
            << ",\"pts_ns\":" << frame->buf_pts << ",\"detection\":{\"bbox\":["
            << det.left << ',' << det.top << ',' << det.width << ',' << det.height
            << "],\"confidence\":" << det.confidence << "},\"outcome\":\"" << outcome
            << "\",\"selected_native_id\":" << selected_id << ",\"shadow_candidates\":[";
        bool first = true;
        for (const auto &sh : shadows) {
          const bool continuous = sh.immediate_continuity;
          const float score = tracker_iou(det.left, det.top, det.width, det.height,
                                          sh.left, sh.top, sh.width, sh.height);
          const float center = tracker_center_distance(det.left, det.top, det.width, det.height,
                                                        sh.left, sh.top, sh.width, sh.height);
          const float wr = std::max(det.width, sh.width) / std::max(1.0f, std::min(det.width, sh.width));
          const float hr = std::max(det.height, sh.height) / std::max(1.0f, std::min(det.height, sh.height));
          if (!first) row << ',';
          first = false;
          row << "{\"native_id\":" << sh.id << ",\"state\":\"" << tracker_state_name(sh.state)
              << "\",\"state_value\":" << static_cast<int>(sh.state) << ",\"age\":" << sh.age
              << ",\"confidence\":" << sh.confidence << ",\"bbox\":[" << sh.left << ',' << sh.top
              << ',' << sh.width << ',' << sh.height << "],\"continuous\":" << (continuous ? "true" : "false")
              << ",\"iou\":" << score << ",\"center_distance_px\":" << center
              << ",\"width_ratio\":" << wr << ",\"height_ratio\":" << hr << '}';
        }
        row << "]}\n";
        const std::string bytes = row.str();
        tracker_write_line(tracker_decision_evidence, bytes.data(), bytes.size());
      }
    }
    for (const auto &sh : shadows)
      tracker_previous_shadows[frame->source_id][sh.id] = {
          static_cast<guint32>(frame->frame_num), sh.age,
          sh.left, sh.top, sh.width, sh.height, sh.confidence, sh.state};
    // Shadow history is latest-only and bounded by NvDCF's max targets.
    for (auto it = tracker_previous_shadows[frame->source_id].begin();
         it != tracker_previous_shadows[frame->source_id].end();) {
      if (it->second.frame_num + 1 < static_cast<guint32>(frame->frame_num))
        it = tracker_previous_shadows[frame->source_id].erase(it);
      else ++it;
    }
    tracker_previous_active_ids[frame->source_id] = std::move(current_active_ids);
    tracker_previous_active_frame[frame->source_id] = frame->frame_num;
    const guint source_id = frame->source_id;
    const gint tracker_frame_num = frame->frame_num;
    const guint64 pts_ns = frame->buf_pts;
    const guint64 monotonic_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
    const guint64 epoch_ns = static_cast<guint64>(g_get_real_time()) * 1000;
    const guint width = frame->source_frame_width ? frame->source_frame_width : 2560;
    const guint height = frame->source_frame_height ? frame->source_frame_height : 1440;
    const guint proposal_width = tracker_bbox_coordinate_extent(frame, true);
    const guint proposal_height = tracker_bbox_coordinate_extent(frame, false);
    const guint32 source_frame_num = f5r_source_frame_for_pts(source_id, pts_ns);
    const guint64 tracker_ready_ns = monotonic_ns;
    tracker_callback_to_metadata_latency[source_id].add(tracker_ready_ns - callback_start_ns);
    const auto lowconf_candidates = context_verifier_enabled()
        ? lowconf_verifier_frame_candidates(lowconf_key) : std::vector<LowConfCandidate>{};
    const bool wait_for_verifier = context_verifier_enabled() &&
        tracker_lowconf_wait_eligible(source_id, lowconf_candidates, published, shadows,
            tracker_frame_num, pts_ns);

    // Canonical publication is ordered per camera on a separate worker. Only a
    // current low-confidence proposal with a unique continuous inactive-shadow
    // match may consume the measured verifier deadline; the NvDCF streaming
    // callback itself never waits for appearance inference.
    auto finalize = [source_id, tracker_frame_num, pts_ns, monotonic_ns, epoch_ns,
        source_frame_num, width, height, proposal_width, proposal_height, tracker_ready_ns,
        published = std::move(published), detections = std::move(detections),
        shadows = std::move(shadows), shadow_refresh_ids = std::move(shadow_refresh_ids)]
        (std::vector<LowConfCandidate> verified_lowconf) mutable {
      tracker_apply_active_birth_verification(source_id, tracker_frame_num,
          pts_ns, &published, verified_lowconf);
      tracker_apply_verified_lowconf(source_id, tracker_frame_num, pts_ns, &published,
          verified_lowconf, shadows);
      tracker_apply_verified_shadow_context(source_id, tracker_frame_num, pts_ns,
          proposal_width, proposal_height, &published, verified_lowconf, shadows,
          shadow_refresh_ids);
      std::string frame_snapshot;
      if (!tracker_atomic_snapshot(source_id, tracker_frame_num, pts_ns,
              monotonic_ns, epoch_ns, source_frame_num, width, height,
              proposal_width, proposal_height, published, detections, &frame_snapshot)) {
        fail("canonical tracked-object IPC publish failed");
        return;
      }
      tracker_canonical_counts[source_id] = static_cast<guint32>(published.size());
      tracker_canonical_frames[source_id] = tracker_frame_num;
      ++source_tracker_updates[source_id];
      tracker_counters[source_id].canonical_objects += published.size();
#ifdef PGIE_ISOLATION
      if (!canonical_evidence || !tracker_write_line(canonical_evidence,
              frame_snapshot.data(), frame_snapshot.size())) {
        fail("direct-file canonical object evidence write failed");
        return;
      }
#else
      f5r_tracker_frame_ready(source_id, tracker_frame_num, source_frame_num,
          pts_ns, static_cast<guint>(published.size()), tracker_ready_ns,
          frame_snapshot.data(), frame_snapshot.size());
#endif
    };
    if (context_verifier_enabled()) {
      if (!lowconf_verifier_defer_frame(lowconf_key, std::move(finalize), wait_for_verifier)) {
        fail("bounded per-camera canonical publisher queue is unavailable or full");
        return GST_PAD_PROBE_DROP;
      }
    } else {
      finalize({});
    }
  }
  return GST_PAD_PROBE_OK;
}

static GstElement *create_nvdcf_tracker() {
  const char *lib =
      "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so";
  const char *cfg = "/config/config_tracker_NvDCF_v13_local6.yml";

  tracker_generation = g_uuid_string_random();
  for (guint i = 0; i < 6; ++i) {
    char path[160]; std::snprintf(path, sizeof(path), "/dev/shm/v11_tracked_objects_cam%02u_v1.json", i + 1);
    unlink(path);
  }
  if (verbose_evidence_enabled()) {
    tracker_evidence = std::fopen("/work/tracks.jsonl", "w");
    fallback_evidence = std::fopen("/work/shadow_fallbacks.jsonl", "w");
    if (!tracker_evidence || !fallback_evidence) {
      fail("cannot write NvDCF evidence");
      std::exit(2);
    }
    std::setvbuf(tracker_evidence, nullptr, _IOFBF, 1024 * 1024);
    std::setvbuf(fallback_evidence, nullptr, _IOFBF, 256 * 1024);
  }
#ifdef PGIE_ISOLATION
  canonical_evidence = std::fopen("/work/canonical-objects.jsonl", "w");
  if (!canonical_evidence) { fail("cannot write direct-file canonical object evidence"); std::exit(2); }
  // The direct-file harness emits one serialized canonical frame per source
  // frame from per-camera publisher workers. Full buffering avoids forcing a
  // shared-filesystem write for every row while preserving complete evidence
  // at fclose/EOS; periodic tracker_stats() still flushes bounded progress.
  std::setvbuf(canonical_evidence, nullptr, _IOFBF, 1024 * 1024);
#endif
  if (lowconf_verifier_enabled() && verbose_evidence_enabled()) {
    lowconf_fallback_evidence = std::fopen("/work/lowconf-fallbacks.jsonl", "w");
    if (!lowconf_fallback_evidence) { fail("cannot write verified low-confidence fallback evidence"); std::exit(2); }
    std::setvbuf(lowconf_fallback_evidence, nullptr, _IOFBF, 256 * 1024);
  }
  if (g_getenv("V11_TRACKER_NAN_DIAGNOSTICS")) {
    invalid_shadow_evidence = std::fopen("/work/invalid_shadow_geometry.jsonl", "w");
    if (!invalid_shadow_evidence) { fail("cannot write invalid shadow geometry diagnostics"); std::exit(2); }
    std::setvbuf(invalid_shadow_evidence, nullptr, _IOFBF, 256 * 1024);
  }
  if (g_getenv("V11_TRACKER_DECISION_DIAGNOSTICS")) {
    tracker_decision_evidence = std::fopen("/work/tracker_decisions.jsonl", "w");
    if (!tracker_decision_evidence) { fail("cannot write tracker decision diagnostics"); std::exit(2); }
    std::setvbuf(tracker_decision_evidence, nullptr, _IOFBF, 1024 * 1024);
    tracker_detection_join_evidence = std::fopen("/work/tracker-detection-join.jsonl", "w");
    if (!tracker_detection_join_evidence) { fail("cannot write tracker detection join diagnostics"); std::exit(2); }
    std::setvbuf(tracker_detection_join_evidence, nullptr, _IOFBF, 256 * 1024);
  }
  if (g_getenv("V11_TRACKER_FRAME_STATE_DIAGNOSTICS")) {
    tracker_frame_state_evidence = std::fopen("/work/tracker-frame-states.jsonl", "w");
    if (!tracker_frame_state_evidence) {
      fail("cannot write NvDCF per-frame state diagnostics");
      std::exit(2);
    }
    std::setvbuf(tracker_frame_state_evidence, nullptr, _IOFBF, 1024 * 1024);
  }

  GstElement *tracker = make("nvtracker", "NvDCF-person");
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
  v11_tracker_width = dimension_override("V11_TRACKER_WIDTH", v11_tracker_width);
  v11_tracker_height = dimension_override("V11_TRACKER_HEIGHT", v11_tracker_height);
  guint tracker_compute_hw = 1u;
  guint tracker_id_reset_mode = 1u;
#ifdef PGIE_ISOLATION
  auto tracker_property_override = [](const char *key, guint fallback, guint max_value) {
    const char *raw = g_getenv(key);
    if (!raw || !*raw) return fallback;
    char *end = nullptr;
    const unsigned long value = std::strtoul(raw, &end, 10);
    if (!end || *end || value > max_value) {
      g_printerr("GROUP invalid %s (valid range 0..%u)\n", key, max_value);
      std::exit(2);
    }
    return static_cast<guint>(value);
  };
  // Runtime-property isolation only: normal builds keep the existing fixed values.
  tracker_compute_hw = tracker_property_override("PGIE_ISOLATION_COMPUTE_HW", tracker_compute_hw, 2u);
  tracker_id_reset_mode = tracker_property_override("PGIE_ISOLATION_ID_RESET_MODE", tracker_id_reset_mode, 3u);
#endif
  g_object_set(
      tracker,
      "tracker-width", v11_tracker_width,
      "tracker-height", v11_tracker_height,
      "ll-lib-file", lib,
      "ll-config-file", cfg,
      "gpu-id", 0u,
      "display-tracking-id", FALSE,
      "compute-hw", tracker_compute_hw,
      "tracking-id-reset-mode", tracker_id_reset_mode,
      nullptr);

  GstPad *in = gst_element_get_static_pad(tracker, "sink");
  if (in) {
    gst_pad_add_probe(in, GST_PAD_PROBE_TYPE_BUFFER, tracker_enter, nullptr, nullptr);
    gst_object_unref(in);
  }
  GstPad *out = gst_element_get_static_pad(tracker, "src");
  gst_pad_add_probe(out, GST_PAD_PROBE_TYPE_BUFFER, tracker_exit, nullptr, nullptr);
  gst_object_unref(out);

  event(nullptr, "TRACKER_GRAPH",
        "nvtracker(config_tracker_NvDCF_v13_local6.yml,width=" + std::to_string(v11_tracker_width) + ",height=" + std::to_string(v11_tracker_height) + ",compute_hw=" + std::to_string(tracker_compute_hw) + ",id_reset_mode=" + std::to_string(tracker_id_reset_mode) + ",batch=6,gpu=0,local-only)->"
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
        "duplicate_ids=%lu unique_ids=%lu canonical_objects=%lu fallbacks=%lu verified_lowconf_fallbacks=%lu trusted_track_lowconf_fallbacks=%lu active_birth_verified=%lu active_birth_withheld=%lu ambiguous=%lu invalid_shadow_geometry=%lu\n",
        i + 1,
        tracker_counters[i].frames.load(),
        tracker_counters[i].objects.load(),
        tracker_counters[i].untracked.load(),
        tracker_counters[i].duplicate_ids.load(),
        unique, tracker_counters[i].canonical_objects.load(), tracker_counters[i].fallbacks.load(),
        tracker_counters[i].verified_lowconf_fallbacks.load(),
        tracker_counters[i].trusted_track_lowconf_fallbacks.load(),
        tracker_counters[i].active_birth_verifier_confirmed.load(),
        tracker_counters[i].active_birth_verifier_withheld.load(),
        tracker_counters[i].ambiguous_matches.load(),
        tracker_counters[i].invalid_shadow_geometry.load());
    g_print("CAM-%02d TRACK_LATENCY element_ms_p50=%.3f element_ms_p95=%.3f element_ms_p99=%.3f element_ms_max=%.3f shadow_scan_ms_p50=%.3f shadow_scan_ms_p95=%.3f shadow_scan_ms_p99=%.3f callback_to_metadata_ms_p50=%.3f callback_to_metadata_ms_p95=%.3f callback_to_metadata_ms_p99=%.3f input_stamp_misses=%lu\n",
        i + 1,
        tracker_element_latency[i].percentile_ns(.50) / 1e6,
        tracker_element_latency[i].percentile_ns(.95) / 1e6,
        tracker_element_latency[i].percentile_ns(.99) / 1e6,
        tracker_element_latency[i].max_ns.load() / 1e6,
        tracker_shadow_scan_latency[i].percentile_ns(.50) / 1e6,
        tracker_shadow_scan_latency[i].percentile_ns(.95) / 1e6,
        tracker_shadow_scan_latency[i].percentile_ns(.99) / 1e6,
        tracker_callback_to_metadata_latency[i].percentile_ns(.50) / 1e6,
        tracker_callback_to_metadata_latency[i].percentile_ns(.95) / 1e6,
        tracker_callback_to_metadata_latency[i].percentile_ns(.99) / 1e6,
        tracker_input_stamp_misses[i].load());
  }
  g_print("TRACKER_DETECTION_JOIN hits=%lu misses=%lu evictions=%lu cache_limit=%zu\n",
      tracker_detection_cache_hits.load(), tracker_detection_cache_misses.load(),
      tracker_detection_cache_evictions.load(), TRACKER_DETECTION_CACHE_LIMIT);
  if (tracker_evidence) std::fflush(tracker_evidence);
  if (canonical_evidence) std::fflush(canonical_evidence);
  if (fallback_evidence) std::fflush(fallback_evidence);
  if (lowconf_fallback_evidence) std::fflush(lowconf_fallback_evidence);
  if (invalid_shadow_evidence) std::fflush(invalid_shadow_evidence);
  if (tracker_decision_evidence) std::fflush(tracker_decision_evidence);
  if (tracker_detection_join_evidence) std::fflush(tracker_detection_join_evidence);
  for (guint i = 0; i < 6; ++i) {
    char path[160]; std::snprintf(path, sizeof(path), "/dev/shm/v11_tracked_objects_cam%02u_v1.json", i + 1);
    unlink(path);
  }
}
