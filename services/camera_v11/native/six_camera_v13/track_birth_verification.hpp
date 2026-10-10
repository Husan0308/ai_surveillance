#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <unordered_map>

// Canonical-only state for a same-frame context check at first active-track
// publication. It never modifies detector rows or NvDCF input/state.
class TrackBirthVerificationLedger {
 public:
  enum class State : std::uint8_t { Unverified, Pending, Verified, Rejected };

  // A context-crop detector may return a larger person proposal than the
  // primary detector, but center containment alone can let an adjacent
  // garment/background proposal confirm an unrelated track. Require both
  // containment and a small, explicit minimum geometry overlap. This is only
  // used for active-track birth verification; shadow fallback keeps its own
  // stricter association policy.
  static bool active_context_location_match(
      float target_left, float target_top, float target_width, float target_height,
      float context_left, float context_top, float context_width, float context_height,
      float *iou_out = nullptr, float minimum_iou = 0.25f) {
    if (iou_out) *iou_out = 0.0f;
    if (!valid_box(target_left, target_top, target_width, target_height) ||
        !valid_box(context_left, context_top, context_width, context_height) ||
        !std::isfinite(minimum_iou) || minimum_iou < 0.0f || minimum_iou > 1.0f)
      return false;
    const float cx = target_left + target_width * 0.5f;
    const float cy = target_top + target_height * 0.5f;
    const bool contains = cx >= context_left && cx <= context_left + context_width &&
        cy >= context_top && cy <= context_top + context_height;
    const float overlap = box_iou(target_left, target_top, target_width, target_height,
        context_left, context_top, context_width, context_height);
    if (iou_out) *iou_out = overlap;
    return contains && overlap >= minimum_iou;
  }

  bool begin(std::uint32_t source, std::uint64_t id, std::int32_t frame) {
    // NvDCF's native ID zero is a valid target ID (the first target on a
    // stream commonly receives it). Do not conflate it with a missing ID.
    if (source >= entries_.size() || frame < 0) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    prune_locked(source, static_cast<std::uint32_t>(frame));
    auto &entry = entries_[source][id];
    entry.last_seen = static_cast<std::uint32_t>(frame);
    if (entry.state != State::Unverified ||
        static_cast<std::uint32_t>(frame) < entry.retry_after) return false;
    entry.state = State::Pending;
    entry.pending_frame = frame;
    return true;
  }

  // A definitive negative is scoped to the crop geometry that was checked.
  // One bounded recheck is allowed only after the same native target receives
  // a materially different current detector box. The caller still requires a
  // positive exact-frame verifier result before canonical publication.
  bool begin_geometry_recheck(std::uint32_t source, std::uint64_t id,
                              std::int32_t frame, float left, float top,
                              float width, float height,
                              float max_iou = 0.30f,
                              std::uint32_t min_delay_frames = 5) {
    if (source >= entries_.size() || frame < 0 ||
        !valid_box(left, top, width, height) ||
        !std::isfinite(max_iou) || max_iou < 0.0f || max_iou > 1.0f) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    prune_locked(source, static_cast<std::uint32_t>(frame));
    auto camera = entries_[source].find(id);
    if (camera == entries_[source].end()) return false;
    auto &entry = camera->second;
    if (entry.state != State::Rejected || !entry.has_rejected_geometry ||
        entry.geometry_rechecks != 0 ||
        static_cast<std::uint32_t>(frame) < entry.last_rejected_frame + min_delay_frames)
      return false;
    if (box_iou(left, top, width, height,
                entry.rejected_left, entry.rejected_top,
                entry.rejected_width, entry.rejected_height) >= max_iou)
      return false;
    entry.state = State::Pending;
    entry.pending_frame = frame;
    entry.last_seen = static_cast<std::uint32_t>(frame);
    entry.geometry_rechecks = 1;
    entry.geometry_recheck_pending = true;
    return true;
  }

  void cancel_geometry_recheck(std::uint32_t source, std::uint64_t id,
                               std::int32_t frame,
                               std::uint32_t retry_delay_frames = 1) {
    if (source >= entries_.size() || frame < 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto camera = entries_[source].find(id);
    if (camera == entries_[source].end()) return;
    auto &entry = camera->second;
    if (entry.state != State::Pending || entry.pending_frame != frame ||
        !entry.geometry_recheck_pending) return;
    entry.state = State::Rejected;
    entry.pending_frame = -1;
    entry.geometry_recheck_pending = false;
    entry.geometry_rechecks = 0;
    entry.retry_after = static_cast<std::uint32_t>(frame) + retry_delay_frames;
    entry.last_seen = std::max(entry.last_seen, static_cast<std::uint32_t>(frame));
  }

  void finish(std::uint32_t source, std::uint64_t id, std::int32_t frame,
              std::uint32_t applied_frame, bool confirmed,
              bool definitive_no_local_person, std::uint32_t retry_delay_frames = 5) {
    if (source >= entries_.size() || frame < 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto camera = entries_[source].find(id);
    if (camera == entries_[source].end()) return;
    auto &entry = camera->second;
    if (entry.state != State::Pending || entry.pending_frame != frame) return;
    if (confirmed) {
      entry.state = State::Verified;
      entry.negative_results = 0;
      entry.last_verified_support_frame = applied_frame;
      entry.geometry_recheck_pending = false;
    } else if (definitive_no_local_person) {
      ++entry.negative_results;
      entry.state = entry.negative_results >= kRequiredNegativeResults
          ? State::Rejected : State::Unverified;
      if (entry.state == State::Rejected) entry.last_verified_support_frame = kNoFrame;
      entry.geometry_recheck_pending = false;
    } else {
      entry.state = State::Unverified;
    }
    entry.pending_frame = -1;
    // Only repeated, completed no-local-person results reject an active target.
    // Backpressure, ambiguity, and late results are inconclusive and retryable.
    entry.retry_after = (confirmed || entry.state == State::Rejected) ? 0u :
        applied_frame + retry_delay_frames;
  }

  void finish(std::uint32_t source, std::uint64_t id, std::int32_t frame,
              bool confirmed, std::uint32_t retry_delay_frames = 30) {
    finish(source, id, frame, static_cast<std::uint32_t>(frame), confirmed, false,
        retry_delay_frames);
  }

  void cancel(std::uint32_t source, std::uint64_t id, std::int32_t frame,
              std::uint32_t retry_delay_frames = 2) {
    finish(source, id, frame, static_cast<std::uint32_t>(frame), false, false,
        retry_delay_frames);
  }

  void trust(std::uint32_t source, std::uint64_t id, std::uint32_t frame) {
    if (source >= entries_.size()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto &entry = entries_[source][id];
    entry.state = State::Verified;
    entry.pending_frame = -1;
    entry.retry_after = 0;
    entry.last_seen = frame;
    entry.last_verified_support_frame = static_cast<std::int32_t>(frame);
    entry.geometry_recheck_pending = false;
  }

  void reject(std::uint32_t source, std::uint64_t id, std::uint32_t frame) {
    reject_impl(source, id, frame, 0.0f, 0.0f, 0.0f, 0.0f, false);
  }

  void reject(std::uint32_t source, std::uint64_t id, std::uint32_t frame,
              float left, float top, float width, float height) {
    reject_impl(source, id, frame, left, top, width, height, true);
  }

  // A source reconnect starts a new local-ID namespace. Drop every pending,
  // rejected, and verified decision for that source so IDs reused by NvDCF
  // cannot inherit appearance evidence from the previous stream epoch.
  void clear_source(std::uint32_t source) {
    if (source >= entries_.size()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    entries_[source].clear();
  }

 private:
  void reject_impl(std::uint32_t source, std::uint64_t id, std::uint32_t frame,
                   float left, float top, float width, float height,
                   bool has_geometry) {
    if (source >= entries_.size()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto &entry = entries_[source][id];
    // A delayed result from an earlier frame cannot undo a newer positive or
    // replace a more recent rejection decision.
    if (entry.state != State::Pending && frame < entry.last_seen) return;
    entry.state = State::Rejected;
    entry.pending_frame = -1;
    entry.retry_after = 0;
    entry.last_seen = frame;
    entry.last_rejected_frame = frame;
    entry.geometry_recheck_pending = false;
    entry.has_rejected_geometry = has_geometry && valid_box(left, top, width, height);
    if (entry.has_rejected_geometry) {
      entry.rejected_left = left;
      entry.rejected_top = top;
      entry.rejected_width = width;
      entry.rejected_height = height;
    }
  }

 public:

  void touch(std::uint32_t source, std::uint64_t id, std::uint32_t frame) {
    if (source >= entries_.size()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto &entry = entries_[source][id];
    entry.last_seen = frame;
    prune_locked(source, frame);
  }

  bool verified(std::uint32_t source, std::uint64_t id) const {
    if (source >= entries_.size()) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_[source].find(id);
    return found != entries_[source].end() && found->second.state == State::Verified;
  }

  // A prior appearance-confirmed target can support a current low-confidence
  // detector box only while ordinary >=threshold detector associations keep
  // refreshing the same native ID. Shadow lifetime alone never refreshes this
  // support window.
  void support_verified_target(std::uint32_t source, std::uint64_t id,
                               std::uint32_t frame) {
    if (source >= entries_.size() || frame > static_cast<std::uint32_t>(INT32_MAX)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto camera = entries_[source].find(id);
    if (camera == entries_[source].end() || camera->second.state != State::Verified) return;
    const auto signed_frame = static_cast<std::int32_t>(frame);
    if (camera->second.last_verified_support_frame < 0 ||
        signed_frame >= camera->second.last_verified_support_frame) {
      camera->second.last_verified_support_frame = signed_frame;
      camera->second.last_seen = frame;
    }
  }

  bool verified_with_recent_support(std::uint32_t source, std::uint64_t id,
                                    std::uint32_t frame,
                                    std::uint32_t max_support_age) const {
    if (source >= entries_.size() || frame > static_cast<std::uint32_t>(INT32_MAX)) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_[source].find(id);
    if (found == entries_[source].end() || found->second.state != State::Verified ||
        found->second.last_verified_support_frame < 0) return false;
    const auto support_frame = static_cast<std::uint32_t>(found->second.last_verified_support_frame);
    return frame >= support_frame && frame - support_frame <= max_support_age;
  }

  // A low-confidence current-frame candidate is authorized by either its own
  // timely appearance confirmation, or by a previously verified native target
  // with fresh support from ordinary detector-to-NvDCF associations. A timely
  // negative/inconclusive result is not overridden by old support. If the
  // candidate-local result is late, it is not used for this frame; the recent
  // target-support path remains independent and bounded.
  bool lowconf_fallback_authorized(std::uint32_t source, std::uint64_t id,
                                  std::uint32_t frame,
                                  bool candidate_result_done,
                                  bool candidate_result_confirmed,
                                  std::uint32_t max_support_age) const {
    if (candidate_result_done) return candidate_result_confirmed;
    return verified_with_recent_support(source, id, frame, max_support_age);
  }

  std::int32_t verified_support_age(std::uint32_t source, std::uint64_t id,
                                    std::uint32_t frame) const {
    if (source >= entries_.size() || frame > static_cast<std::uint32_t>(INT32_MAX)) return -1;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_[source].find(id);
    if (found == entries_[source].end() || found->second.state != State::Verified ||
        found->second.last_verified_support_frame < 0 ||
        frame < static_cast<std::uint32_t>(found->second.last_verified_support_frame)) return -1;
    return static_cast<std::int32_t>(frame -
        static_cast<std::uint32_t>(found->second.last_verified_support_frame));
  }

  bool rejected(std::uint32_t source, std::uint64_t id) const {
    if (source >= entries_.size()) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_[source].find(id);
    return found != entries_[source].end() && found->second.state == State::Rejected;
  }

  State state(std::uint32_t source, std::uint64_t id) const {
    if (source >= entries_.size()) return State::Unverified;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_[source].find(id);
    return found == entries_[source].end() ? State::Unverified : found->second.state;
  }

 private:
  struct Entry {
    State state = State::Unverified;
    std::int32_t pending_frame = -1;
    std::uint32_t retry_after = 0;
    std::uint32_t last_seen = 0;
    std::uint8_t negative_results = 0;
    std::int32_t last_verified_support_frame = -1;
    std::uint32_t last_rejected_frame = 0;
    std::uint8_t geometry_rechecks = 0;
    bool geometry_recheck_pending = false;
    bool has_rejected_geometry = false;
    float rejected_left = 0.0f;
    float rejected_top = 0.0f;
    float rejected_width = 0.0f;
    float rejected_height = 0.0f;
  };

  static constexpr std::int32_t kNoFrame = -1;

  // Active-target verification is a location-specific, same-frame check: a
  // definitive result means the crop contained no person proposal covering
  // this target's center. Unlike a global confidence heuristic, a single such
  // completed result is sufficient to suppress this native target. Inconclusive
  // and late results still do not count as negatives.
  static constexpr std::uint8_t kRequiredNegativeResults = 1;

  static bool valid_box(float left, float top, float width, float height) {
    return std::isfinite(left) && std::isfinite(top) &&
        std::isfinite(width) && std::isfinite(height) &&
        width > 0.0f && height > 0.0f;
  }

  static float box_iou(float al, float at, float aw, float ah,
                       float bl, float bt, float bw, float bh) {
    const float x1 = std::max(al, bl), y1 = std::max(at, bt);
    const float x2 = std::min(al + aw, bl + bw), y2 = std::min(at + ah, bt + bh);
    const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float area_a = aw * ah, area_b = bw * bh;
    const float total = area_a + area_b - intersection;
    return total > 0.0f ? intersection / total : 0.0f;
  }

  void prune_locked(std::uint32_t source, std::uint32_t frame) {
    // Retain decisions through the local tracker's full shadow lifetime plus
    // margin; native IDs are not allowed to inherit another session's result.
    constexpr std::uint32_t kRetentionFrames = 256;
    auto &camera = entries_[source];
    for (auto it = camera.begin(); it != camera.end();) {
      const bool expired = it->second.state != State::Pending &&
          frame > it->second.last_seen && frame - it->second.last_seen > kRetentionFrames;
      if (expired) it = camera.erase(it);
      else ++it;
    }
  }

  mutable std::mutex mutex_;
  std::array<std::unordered_map<std::uint64_t, Entry>, 6> entries_;
};
