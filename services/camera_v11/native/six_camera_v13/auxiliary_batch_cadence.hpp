#pragma once

#include <atomic>
#include <cstdint>

// Keeps one in every `period` mux batches for an optional auxiliary detector.
// The first batch is admitted so the auxiliary nvinfer keeps the same startup
// cadence as DeepStream's interval=N implementation (first batch, then N
// skipped batches). This gate is branch-local and must never touch primary
// detector/tracker buffers.
class AuxiliaryBatchCadence {
 public:
  explicit AuxiliaryBatchCadence(std::uint64_t period = 20,
                                 std::uint64_t phase = 0) noexcept
      : period_(period ? period : 1), phase_(phase % (period ? period : 1)) {}

  bool admit_next() noexcept {
    const auto sequence = seen_.fetch_add(1, std::memory_order_relaxed);
    const bool admitted = sequence % period_ == phase_;
    if (admitted) admitted_.fetch_add(1, std::memory_order_relaxed);
    else dropped_.fetch_add(1, std::memory_order_relaxed);
    return admitted;
  }

  std::uint64_t seen() const noexcept { return seen_.load(std::memory_order_relaxed); }
  std::uint64_t admitted() const noexcept { return admitted_.load(std::memory_order_relaxed); }
  std::uint64_t dropped() const noexcept { return dropped_.load(std::memory_order_relaxed); }

 private:
  const std::uint64_t period_;
  const std::uint64_t phase_;
  std::atomic<std::uint64_t> seen_{0};
  std::atomic<std::uint64_t> admitted_{0};
  std::atomic<std::uint64_t> dropped_{0};
};
