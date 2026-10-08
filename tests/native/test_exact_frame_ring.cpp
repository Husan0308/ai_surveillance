#include "services/camera_v11/native/six_camera_v13/exact_frame_ring.hpp"
#include "services/camera_v11/native/six_camera_v13/shadow_continuity.hpp"
#include "services/camera_v11/native/six_camera_v13/latency_histogram.hpp"

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

struct Sample {
  std::uint32_t frameNum;
};

int main() {
  // The deferred fallback keeps the exact old rule: either the same shadow
  // continued one age step, or the same ID transitioned ACTIVE -> INACTIVE
  // from the immediately preceding frame. No older or absent ID is eligible.
  assert(tracker_has_immediate_shadow_continuity(11, true, 10, 4, 5, 9, false));
  assert(tracker_has_immediate_shadow_continuity(11, false, -1, 0, 5, 10, true));
  assert(!tracker_has_immediate_shadow_continuity(11, true, 9, 4, 5, 9, false));
  assert(!tracker_has_immediate_shadow_continuity(11, true, 10, 3, 5, 9, false));
  assert(!tracker_has_immediate_shadow_continuity(11, false, -1, 0, 5, 9, true));

  ExactFrameRing<int, 2, 4> ring;
  std::vector<int> camera_zero{7, 8};
  assert(!ring.put(0, 10, 1000, std::move(camera_zero)).overwritten);
  std::vector<int> camera_one{9};
  assert(!ring.put(1, 10, 1000, std::move(camera_one)).overwritten);

  // A wrong PTS must not consume the exact frame entry.
  assert(!ring.take(0, 10, 1001));
  auto got_zero = ring.take(0, 10, 1000);
  assert(got_zero && *got_zero == std::vector<int>({7, 8}));
  assert(!ring.take(0, 10, 1000));
  auto got_one = ring.take(1, 10, 1000);
  assert(got_one && *got_one == std::vector<int>({9}));

  // A modulo collision evicts the old key and never aliases it to the new one.
  std::vector<int> old_value{1};
  assert(!ring.put(0, 2, 2000, std::move(old_value)).overwritten);
  std::vector<int> new_value{2};
  const auto collision = ring.put(0, 6, 6000, std::move(new_value));
  assert(collision.overwritten && collision.old_frame == 2 && collision.old_pts == 2000);
  assert(!ring.take(0, 2, 2000));
  auto got_new = ring.take(0, 6, 6000);
  assert(got_new && *got_new == std::vector<int>({2}));

  // The current-frame fast path is O(1) for the normal append order, while
  // the fallback preserves a search for a non-tail current sample.
  const Sample appended[]{{1}, {2}, {3}};
  unsigned comparisons = 0;
  auto frame_of = [&comparisons](const Sample& sample) {
    ++comparisons;
    return sample.frameNum;
  };
  const auto* current = find_current_tracker_sample(appended, 3, 3u, frame_of);
  assert(current == &appended[2] && comparisons == 1);

  const Sample unordered[]{{1}, {3}, {2}};
  comparisons = 0;
  current = find_current_tracker_sample(unordered, 3, 3u, frame_of);
  assert(current == &unordered[1] && comparisons == 3);
  assert(find_current_tracker_sample(unordered, 3, 4u, frame_of) == nullptr);
  assert(find_current_tracker_sample<Sample>(nullptr, 0, 4u, frame_of) == nullptr);

  LatencyHistogram latency;
  latency.add(100000);
  latency.add(1000000);
  latency.add(3000000);
  assert(latency.count.load() == 3);
  assert(latency.percentile_ns(.50) == 1125000);
  assert(latency.percentile_ns(.95) == 3125000);
  assert(latency.max_ns.load() == 3000000);
  return 0;
}
