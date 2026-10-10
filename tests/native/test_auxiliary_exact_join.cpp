#include "services/camera_v11/native/six_camera_v13/auxiliary_exact_join.hpp"

#include <cassert>
#include <chrono>
#include <thread>
#include <vector>

int main() {
  using Join = AuxiliaryExactJoin<int, 2, 2>;
  using namespace std::chrono_literals;
  Join join;
  const AuxiliaryJoinKey a{0, 11, 1001};
  const AuxiliaryJoinKey b{1, 11, 1001};

  // Completion with no detections is still distinguishable from missing work.
  assert(join.publish(a, {}).accepted);
  auto empty = join.take_until(a, std::chrono::steady_clock::now() + 1ms);
  assert(empty.status == Join::WaitStatus::Ready && empty.values.empty());

  // No approximate PTS or cross-source matching.
  assert(join.publish(a, {7, 8}).accepted);
  assert(join.take_until(AuxiliaryJoinKey{0, 11, 1002},
      std::chrono::steady_clock::now() + 1ms).status == Join::WaitStatus::Timeout);
  assert(join.take_until(b, std::chrono::steady_clock::now() + 1ms).status ==
      Join::WaitStatus::Timeout);
  auto exact = join.take_until(a, std::chrono::steady_clock::now() + 1ms);
  assert(exact.status == Join::WaitStatus::Ready && exact.values == std::vector<int>({7, 8}));

  // A post-reconnect source epoch cannot consume a pre-reconnect auxiliary
  // inference result even if local frame and PTS values are reused.
  const AuxiliaryJoinKey old_epoch{0, 4, 400, 8};
  const AuxiliaryJoinKey new_epoch{0, 4, 400, 9};
  assert(join.publish(old_epoch, {4}).accepted);
  assert(join.take_until(new_epoch,
      std::chrono::steady_clock::now() + 1ms).status == Join::WaitStatus::Timeout);
  auto old_epoch_result = join.take_until(old_epoch,
      std::chrono::steady_clock::now() + 1ms);
  assert(old_epoch_result.status == Join::WaitStatus::Ready &&
      old_epoch_result.values == std::vector<int>({4}));

  // A bounded timeout fails closed while a producer can still wake an exact waiter.
  const AuxiliaryJoinKey c{0, 12, 1200};
  std::thread producer([&] {
    std::this_thread::sleep_for(2ms);
    assert(join.publish(c, {12}).accepted);
  });
  auto arrived = join.take_until(c, std::chrono::steady_clock::now() + 100ms);
  producer.join();
  assert(arrived.status == Join::WaitStatus::Ready && arrived.values == std::vector<int>({12}));

  // Capacity eviction cannot alias an old frame to a new one.
  assert(join.publish(AuxiliaryJoinKey{0, 20, 2000}, {20}).accepted);
  assert(join.publish(AuxiliaryJoinKey{0, 21, 2100}, {21}).accepted);
  const auto overflow = join.publish(AuxiliaryJoinKey{0, 22, 2200}, {22});
  assert(overflow.accepted && overflow.evicted);
  assert((overflow.evicted_key == AuxiliaryJoinKey{0, 20, 2000}));
  assert(join.take_until(AuxiliaryJoinKey{0, 20, 2000},
      std::chrono::steady_clock::now() + 1ms).status == Join::WaitStatus::Timeout);
  auto newest = join.take_until(AuxiliaryJoinKey{0, 22, 2200},
      std::chrono::steady_clock::now() + 1ms);
  assert(newest.status == Join::WaitStatus::Ready && newest.values == std::vector<int>({22}));
  assert(join.pending(9) == 0);
  return 0;
}
