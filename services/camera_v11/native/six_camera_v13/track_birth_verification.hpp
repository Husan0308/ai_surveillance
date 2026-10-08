#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <unordered_map>

// Canonical-only state for a same-frame context check at first active-track
// publication. It never modifies detector rows or NvDCF input/state.
class TrackBirthVerificationLedger {
 public:
  enum class State : std::uint8_t { Unverified, Pending, Verified, Rejected };

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
    } else if (definitive_no_local_person) {
      ++entry.negative_results;
      entry.state = entry.negative_results >= kRequiredNegativeResults
          ? State::Rejected : State::Unverified;
      if (entry.state == State::Rejected) entry.last_verified_support_frame = kNoFrame;
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
  }

  void reject(std::uint32_t source, std::uint64_t id, std::uint32_t frame) {
    if (source >= entries_.size()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto &entry = entries_[source][id];
    entry.state = State::Rejected;
    entry.pending_frame = -1;
    entry.retry_after = 0;
    entry.last_seen = frame;
  }

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
  };

  static constexpr std::int32_t kNoFrame = -1;

  // Active-target verification is a location-specific, same-frame check: a
  // definitive result means the crop contained no person proposal covering
  // this target's center. Unlike a global confidence heuristic, a single such
  // completed result is sufficient to suppress this native target. Inconclusive
  // and late results still do not count as negatives.
  static constexpr std::uint8_t kRequiredNegativeResults = 1;

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
