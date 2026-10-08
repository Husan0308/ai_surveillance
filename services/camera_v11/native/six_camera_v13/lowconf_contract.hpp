#pragma once

#include <gst/gst.h>
#include <cstdint>
#include <functional>
#include <vector>

struct LowConfFrameKey {
  guint source_id = 0;
  gint frame_num = -1;
  guint64 pts_ns = GST_CLOCK_TIME_NONE;
  bool operator<(const LowConfFrameKey &other) const {
    if (source_id != other.source_id) return source_id < other.source_id;
    if (frame_num != other.frame_num) return frame_num < other.frame_num;
    return pts_ns < other.pts_ns;
  }
};

struct LowConfVerifiedBox {
  float confidence = 0;
  float left = 0, top = 0, width = 0, height = 0;
};

struct LowConfShadowTarget {
  guint64 native_track_id = 0;
  float hint_left = 0, hint_top = 0, hint_width = 0, hint_height = 0;
};

struct LowConfCandidate {
  float confidence = 0;
  float left = 0, top = 0, width = 0, height = 0;
  // Nonzero when the context result gates first canonical publication for an
  // ACTIVE NvDCF native ID. The primary detector row remains untouched.
  guint64 native_track_id = 0;
  // Same-frame detector-row index associated with the native target by NvDCF.
  gint detector_index = -1;
  bool active_track_id_assigned = false;
  // These fields preserve the original frame key through the asynchronous crop
  // worker. They are checked again before a shadow-assisted box is published.
  gint source_frame_num = -1;
  guint64 source_pts_ns = GST_CLOCK_TIME_NONE;
  bool active_track_birth = false;
  // A current-frame crop confirmation may refresh a previously verified
  // auxiliary-born native target that is presently an INACTIVE NvDCF shadow.
  // Shadow geometry is used only to center/match the crop, never as output.
  bool shadow_context_refresh = false;
  bool shadow_context_pretracker_scheduled = false;
  // One context crop can validate multiple nearby shadow targets. Every target
  // still needs its own current INACTIVE state and unique detector-box match.
  std::vector<LowConfShadowTarget> shadow_targets;
  std::vector<LowConfVerifiedBox> verifier_boxes;
  bool verifier_done = false;
  bool verifier_confirmed = false;
  bool verifier_no_local_person = false;
  float verifier_confidence = 0;
  float verifier_iou = 0;
  float verifier_runner_up_iou = 0;
  float verifier_uniqueness_margin = 0;
  float verifier_center_distance = 0;
  float verifier_bbox_left = 0;
  float verifier_bbox_top = 0;
  float verifier_bbox_width = 0;
  float verifier_bbox_height = 0;
  guint64 submitted_monotonic_ns = 0;
  guint64 verifier_ready_monotonic_ns = 0;
  float verifier_latency_ms = 0;
};

struct ActiveTrackVerificationDecision {
  guint64 native_track_id = 0;
  gint request_frame_num = -1;
  bool confirmed = false;
  bool definitive_no_local_person = false;
};

enum class LowConfSubmitResult : std::uint8_t {
  Submitted,
  Disabled,
  Invalid,
  Backpressure,
};

static bool lowconf_verifier_enabled();
static bool active_track_verifier_enabled();
static bool context_verifier_enabled();
static bool shadow_context_refresh_enabled();
static void lowconf_verifier_start(GstElement *pipeline);
static LowConfSubmitResult lowconf_verifier_submit(GstBuffer *buffer, guint batch_id,
    const LowConfFrameKey &key, guint frame_width, guint frame_height,
    const LowConfCandidate &candidate, bool active_track_birth = false,
    bool shadow_context_refresh = false);
static bool lowconf_verifier_surface_dimensions(
    GstBuffer *buffer, guint batch_id, guint *width, guint *height);
static std::vector<LowConfCandidate> lowconf_verifier_take_frame(
    const LowConfFrameKey &key);
static std::vector<LowConfCandidate> lowconf_verifier_frame_candidates(
    const LowConfFrameKey &key);
static bool lowconf_verifier_assign_active_track(const LowConfFrameKey &key,
    gint detector_index, guint64 native_track_id);
static std::vector<ActiveTrackVerificationDecision>
lowconf_verifier_take_completed_active(guint source_id);
static bool lowconf_verifier_defer_frame(const LowConfFrameKey &key,
    std::function<void(std::vector<LowConfCandidate>)> publish,
    bool wait_for_verifier);
static void lowconf_verifier_stats();
static void lowconf_verifier_finish_input();
static GstPadProbeReturn lowconf_verifier_primary_event(GstPad*, GstPadProbeInfo*, gpointer);
static void lowconf_verifier_shutdown();
