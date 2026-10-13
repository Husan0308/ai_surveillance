#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

namespace camera_recovery {

enum class SourceHealthState {
  STARTING,
  ONLINE,
  SUSPECT,
  RECOVERING,
  OFFLINE,
  BACKOFF,
  QUARANTINED,
};

enum class FailureScope { SOURCE_LOCAL, SHARED_PIPELINE, PROGRAMMER };

inline const char *state_name(SourceHealthState state) {
  switch (state) {
    case SourceHealthState::STARTING: return "STARTING";
    case SourceHealthState::ONLINE: return "ONLINE";
    case SourceHealthState::SUSPECT: return "SUSPECT";
    case SourceHealthState::RECOVERING: return "RECOVERING";
    case SourceHealthState::OFFLINE: return "OFFLINE";
    case SourceHealthState::BACKOFF: return "BACKOFF";
    case SourceHealthState::QUARANTINED: return "QUARANTINED";
  }
  return "QUARANTINED";
}

inline bool should_fail_process(FailureScope scope) {
  return scope != FailureScope::SOURCE_LOCAL;
}

inline int64_t retry_delay_us(uint32_t consecutive_failures, bool authentication_failure) {
  if (authentication_failure) return 300'000'000;
  const uint32_t exponent = std::min<uint32_t>(consecutive_failures ? consecutive_failures - 1 : 0, 6);
  return std::min<int64_t>(60'000'000, (int64_t{1} << exponent) * 1'000'000);
}

class SourceRecoveryPolicy {
 public:
  static constexpr int64_t kStableOnlineResetUs = 30'000'000;

  SourceHealthState state() const { return state_; }
  const std::string &last_failure_reason() const { return last_failure_reason_; }
  uint32_t consecutive_failures() const { return consecutive_failures_; }
  uint64_t source_epoch() const { return source_epoch_; }
  int64_t next_retry_us() const { return next_retry_us_; }
  int64_t state_since_us() const { return state_since_us_; }

  void mark_online(int64_t now_us) {
    const bool was_online = state_ == SourceHealthState::ONLINE;
    state_ = SourceHealthState::ONLINE;
    if (!was_online) state_since_us_ = now_us;
    next_retry_us_ = 0;
    owner_stopped_ = false;
  }

  // A source that advances briefly and fails again is still flapping. Keep its
  // failure streak until it has remained online for a complete stability
  // window; otherwise every one-frame reconnect resets backoff to one second.
  bool observe_online_stability(int64_t now_us) {
    if (state_ != SourceHealthState::ONLINE || consecutive_failures_ == 0 ||
        now_us - state_since_us_ < kStableOnlineResetUs) return false;
    consecutive_failures_ = 0;
    last_failure_reason_.clear();
    return true;
  }

  void mark_suspect(int64_t now_us, std::string reason) {
    if (state_ == SourceHealthState::RECOVERING || state_ == SourceHealthState::BACKOFF ||
        state_ == SourceHealthState::QUARANTINED) return;
    state_ = SourceHealthState::SUSPECT;
    state_since_us_ = now_us;
    if (!reason.empty()) last_failure_reason_ = std::move(reason);
  }

  bool begin_recovery(int64_t now_us) {
    if (state_ != SourceHealthState::SUSPECT && state_ != SourceHealthState::OFFLINE &&
        state_ != SourceHealthState::STARTING) return false;
    state_ = SourceHealthState::RECOVERING;
    state_since_us_ = now_us;
    owner_stopped_ = false;
    return true;
  }

  void owner_removed(int64_t now_us, bool authentication_failure) {
    owner_stopped_ = true;
    ++consecutive_failures_;
    const int64_t delay = retry_delay_us(consecutive_failures_, authentication_failure);
    next_retry_us_ = now_us + delay;
    state_ = authentication_failure ? SourceHealthState::QUARANTINED : SourceHealthState::BACKOFF;
    state_since_us_ = now_us;
  }

  bool retry_due(int64_t now_us) const {
    return owner_stopped_ && next_retry_us_ > 0 && now_us >= next_retry_us_ &&
        (state_ == SourceHealthState::BACKOFF || state_ == SourceHealthState::QUARANTINED ||
         state_ == SourceHealthState::OFFLINE);
  }

  bool source_started(int64_t now_us) {
    if (!owner_stopped_ && state_ != SourceHealthState::RECOVERING &&
        state_ != SourceHealthState::STARTING) return false;
    state_ = SourceHealthState::STARTING;
    state_since_us_ = now_us;
    next_retry_us_ = 0;
    owner_stopped_ = false;
    ++source_epoch_;
    return true;
  }

  void recovery_failed(int64_t now_us, std::string reason, bool authentication_failure = false) {
    if (!reason.empty()) last_failure_reason_ = std::move(reason);
    owner_removed(now_us, authentication_failure);
  }

  void mark_offline(int64_t now_us, std::string reason, bool authentication_failure = false) {
    state_ = authentication_failure ? SourceHealthState::QUARANTINED : SourceHealthState::OFFLINE;
    state_since_us_ = now_us;
    next_retry_us_ = 0;
    owner_stopped_ = false;
    if (!reason.empty()) last_failure_reason_ = std::move(reason);
  }

  bool owner_stopped() const { return owner_stopped_; }

 private:
  SourceHealthState state_ = SourceHealthState::STARTING;
  int64_t state_since_us_ = 0;
  int64_t next_retry_us_ = 0;
  uint32_t consecutive_failures_ = 0;
  uint64_t source_epoch_ = 0;
  bool owner_stopped_ = false;
  std::string last_failure_reason_;
};

}  // namespace camera_recovery
