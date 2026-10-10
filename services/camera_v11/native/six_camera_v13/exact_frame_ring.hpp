#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

// Fixed-capacity exact-key handoff for PGIE -> NvDCF metadata. Source frame
// and PTS are both part of the key; a modulo collision is never accepted as a
// match. The ring avoids per-frame tree-node allocation and proposal-vector
// copies on the inference callback.
template <typename Value, std::size_t CameraCount = 6, std::size_t Capacity = 64>
class ExactFrameRing {
  static_assert(CameraCount > 0, "at least one camera is required");
  static_assert(Capacity > 1, "ring capacity must exceed one frame");

 public:
  using FrameNumber = std::int64_t;
  using Pts = std::uint64_t;

  struct PutResult {
    bool overwritten = false;
    FrameNumber old_frame = -1;
    Pts old_pts = 0;
  };

  PutResult put(std::size_t camera, FrameNumber frame, Pts pts,
                std::vector<Value>&& values, std::uint64_t source_epoch = 0) {
    if (camera >= CameraCount || frame < 0) return {};
    std::lock_guard<std::mutex> guard(mutex_);
    Slot& slot = slots_[camera][static_cast<std::size_t>(frame) % Capacity];
    PutResult result;
    result.overwritten = slot.valid && (slot.frame != frame || slot.pts != pts ||
        slot.source_epoch != source_epoch);
    if (result.overwritten) {
      result.old_frame = slot.frame;
      result.old_pts = slot.pts;
    }
    slot.frame = frame;
    slot.pts = pts;
    slot.source_epoch = source_epoch;
    slot.values = std::move(values);
    slot.valid = true;
    return result;
  }

  std::optional<std::vector<Value>> take(std::size_t camera,
                                         FrameNumber frame, Pts pts,
                                         std::uint64_t source_epoch = 0) {
    if (camera >= CameraCount || frame < 0) return std::nullopt;
    std::lock_guard<std::mutex> guard(mutex_);
    Slot& slot = slots_[camera][static_cast<std::size_t>(frame) % Capacity];
    if (!slot.valid || slot.frame != frame || slot.pts != pts ||
        slot.source_epoch != source_epoch) return std::nullopt;
    slot.valid = false;
    return std::move(slot.values);
  }

 private:
  struct Slot {
    FrameNumber frame = -1;
    Pts pts = 0;
    std::uint64_t source_epoch = 0;
    bool valid = false;
    std::vector<Value> values;
  };

  std::mutex mutex_;
  std::array<std::array<Slot, Capacity>, CameraCount> slots_{};
};

// NvDCF's public per-target history normally appends one sample per frame.
// Check the newest sample first; retain the original forward search as the
// safe fallback if a metadata producer presents a non-tail ordering.
template <typename Sample, typename FrameNumber, typename FrameAccessor>
const Sample* find_current_tracker_sample(const Sample* samples,
                                          std::size_t count,
                                          FrameNumber frame,
                                          FrameAccessor frame_of) {
  if (!samples || count == 0) return nullptr;
  if (frame_of(samples[count - 1]) == frame) return &samples[count - 1];
  for (std::size_t index = 0; index < count; ++index) {
    if (frame_of(samples[index]) == frame) return &samples[index];
  }
  return nullptr;
}
