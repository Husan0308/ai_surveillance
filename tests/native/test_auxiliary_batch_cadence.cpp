#include "services/camera_v11/native/six_camera_v13/auxiliary_batch_cadence.hpp"
#include "services/camera_v11/native/six_camera_v13/auxiliary_exact_join.hpp"

#include <cassert>
#include <cstdint>
#include <thread>
#include <vector>

int main() {
  AuxiliaryBatchCadence cadence(20, 0);
  for (std::uint64_t i = 0; i < 1000; ++i) assert(cadence.admit_next() == (i % 20 == 0));
  assert(cadence.seen() == 1000);
  assert(cadence.admitted() == 50);
  assert(cadence.dropped() == 950);

  // Cadence is safe if GStreamer hands buffers from more than one streaming
  // context, and the aggregate admission count remains exact.
  AuxiliaryBatchCadence concurrent(7, 3);
  std::atomic<std::uint64_t> accepted{0};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 4; ++worker) {
    workers.emplace_back([&] {
      for (int i = 0; i < 1000; ++i)
        if (concurrent.admit_next()) accepted.fetch_add(1, std::memory_order_relaxed);
    });
  }
  for (auto &worker : workers) worker.join();
  assert(concurrent.seen() == 4000);
  assert(concurrent.admitted() == accepted.load(std::memory_order_relaxed));
  assert(concurrent.admitted() + concurrent.dropped() == concurrent.seen());

  AuxiliaryBatchCadence invalid(0, 99);
  assert(invalid.admit_next());
  assert(invalid.admit_next());
  assert(invalid.seen() == 2);
  assert(invalid.admitted() == 2);
  assert(invalid.dropped() == 0);

  // A skipped auxiliary batch must still complete every exact frame key with
  // an empty result so the primary branch never pays the bounded timeout.
  AuxiliaryBatchCadence sparse(20, 0);
  AuxiliaryExactJoin<int, 6, 64> join;
  for (std::uint64_t batch = 0; batch < 40; ++batch) {
    const bool admitted = sparse.admit_next();
    for (std::uint32_t source = 0; source < 6; ++source) {
      const AuxiliaryJoinKey key{source, static_cast<std::int64_t>(batch), batch * 50'000'000};
      auto publish = join.publish(key, admitted ? std::vector<int>{static_cast<int>(batch)}
                                                : std::vector<int>{});
      assert(publish.accepted);
      auto result = join.take_until(key, std::chrono::steady_clock::now() +
          std::chrono::milliseconds(1));
      assert((result.status == AuxiliaryExactJoin<int, 6, 64>::WaitStatus::Ready));
      if (admitted) assert(result.values == std::vector<int>{static_cast<int>(batch)});
      else assert(result.values.empty());
    }
  }
  assert(sparse.seen() == 40);
  assert(sparse.admitted() == 2);
  assert(sparse.dropped() == 38);
  return 0;
}
