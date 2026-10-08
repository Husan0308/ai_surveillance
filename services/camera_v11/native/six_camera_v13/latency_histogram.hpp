#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>

// Allocation-free stage timing for live diagnostics. Buckets are 0.25 ms wide;
// recording only touches atomics and does no formatting or file I/O on the
// GStreamer streaming threads.
struct LatencyHistogram {
  static constexpr std::uint64_t BUCKET_NS = 250000;
  static constexpr std::size_t BUCKETS = 512;  // 0..128 ms, last bucket saturates

  std::array<std::atomic<std::uint64_t>, BUCKETS> bins{};
  std::atomic<std::uint64_t> count{0};
  std::atomic<std::uint64_t> max_ns{0};

  void add(std::uint64_t duration_ns) noexcept {
    const auto bucket = static_cast<std::size_t>(duration_ns / BUCKET_NS);
    bins[bucket < BUCKETS ? bucket : BUCKETS - 1].fetch_add(1, std::memory_order_relaxed);
    count.fetch_add(1, std::memory_order_relaxed);
    auto prior = max_ns.load(std::memory_order_relaxed);
    while (prior < duration_ns &&
           !max_ns.compare_exchange_weak(prior, duration_ns, std::memory_order_relaxed)) {}
  }

  std::uint64_t percentile_ns(double q) const noexcept {
    const auto n = count.load(std::memory_order_relaxed);
    if (!n) return 0;
    const auto rank = static_cast<std::uint64_t>(std::ceil(q * static_cast<double>(n)));
    std::uint64_t seen = 0;
    for (std::size_t i = 0; i < BUCKETS; ++i) {
      seen += bins[i].load(std::memory_order_relaxed);
      if (seen >= rank) return static_cast<std::uint64_t>(i) * BUCKET_NS + BUCKET_NS / 2;
    }
    return BUCKETS * BUCKET_NS;
  }
};
