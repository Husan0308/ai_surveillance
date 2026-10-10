#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

// A bounded handoff between parallel inference branches. An empty proposal
// vector is still a completed result. Matching requires source, frame and PTS;
// no nearest-frame or stale-result fallback is provided.
struct AuxiliaryJoinKey {
  std::uint32_t source_id = 0;
  std::int64_t frame_num = -1;
  std::uint64_t pts_ns = 0;
  std::uint64_t source_epoch = 0;

  bool operator<(const AuxiliaryJoinKey& other) const {
    if (source_id != other.source_id) return source_id < other.source_id;
    if (source_epoch != other.source_epoch) return source_epoch < other.source_epoch;
    if (frame_num != other.frame_num) return frame_num < other.frame_num;
    return pts_ns < other.pts_ns;
  }
  bool operator==(const AuxiliaryJoinKey& other) const {
    return source_id == other.source_id && source_epoch == other.source_epoch &&
        frame_num == other.frame_num &&
        pts_ns == other.pts_ns;
  }
};

template <typename Value, std::size_t CameraCount = 6, std::size_t Capacity = 64>
class AuxiliaryExactJoin {
  static_assert(CameraCount > 0, "at least one camera is required");
  static_assert(Capacity > 1, "join capacity must exceed one frame");

 public:
  enum class WaitStatus { Ready, Timeout, InvalidKey };
  struct WaitResult {
    WaitStatus status = WaitStatus::InvalidKey;
    std::vector<Value> values;
  };
  struct PublishResult {
    bool accepted = false;
    bool evicted = false;
    AuxiliaryJoinKey evicted_key{};
  };

  PublishResult publish(const AuxiliaryJoinKey& key, std::vector<Value>&& values) {
    if (key.source_id >= CameraCount || key.frame_num < 0) return {};
    std::lock_guard<std::mutex> guard(mutex_);
    auto& order = order_[key.source_id];
    auto existing = values_.find(key);
    if (existing == values_.end()) {
      order.push_back(key);
      values_.emplace(key, std::move(values));
    } else {
      existing->second = std::move(values);
    }
    PublishResult result;
    result.accepted = true;
    while (order.size() > Capacity) {
      result.evicted = true;
      result.evicted_key = order.front();
      auto entry = values_.find(order.front());
      if (entry != values_.end()) values_.erase(entry);
      order.pop_front();
    }
    ready_.notify_all();
    return result;
  }

  template <typename Clock, typename Duration>
  WaitResult take_until(const AuxiliaryJoinKey& key,
                        const std::chrono::time_point<Clock, Duration>& deadline) {
    if (key.source_id >= CameraCount || key.frame_num < 0)
      return {WaitStatus::InvalidKey, {}};
    std::unique_lock<std::mutex> lock(mutex_);
    if (!ready_.wait_until(lock, deadline, [&] { return values_.find(key) != values_.end(); }))
      return {WaitStatus::Timeout, {}};
    auto found = values_.find(key);
    if (found == values_.end()) return {WaitStatus::Timeout, {}};
    auto result = std::move(found->second);
    values_.erase(found);
    auto& order = order_[key.source_id];
    for (auto it = order.begin(); it != order.end(); ++it) {
      if (*it == key) {
        order.erase(it);
        break;
      }
    }
    return {WaitStatus::Ready, std::move(result)};
  }

  std::size_t pending(std::size_t source_id) const {
    if (source_id >= CameraCount) return 0;
    std::lock_guard<std::mutex> guard(mutex_);
    return order_[source_id].size();
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::map<AuxiliaryJoinKey, std::vector<Value>> values_;
  std::array<std::deque<AuxiliaryJoinKey>, CameraCount> order_{};
};
