#include "services/camera_v11/native/six_camera_v13/camera_recovery_policy.hpp"

#include <cassert>
#include <cstdio>

int main() {
  using namespace camera_recovery;

  SourceRecoveryPolicy cam05;
  cam05.mark_online(1'000'000);
  assert(cam05.state() == SourceHealthState::ONLINE);

  // A repeated stale event after a successful recovery remains camera-local;
  // it must never become a process-wide fatal condition.
  cam05.mark_suspect(10'000'000, "no decoded frames");
  assert(cam05.begin_recovery(12'000'000));
  cam05.owner_removed(12'100'000, false);
  assert(cam05.retry_due(cam05.next_retry_us()));
  cam05.source_started(13'100'000);
  cam05.mark_online(14'000'000);
  assert(cam05.consecutive_failures() == 1);
  cam05.mark_suspect(20'000'000, "stale again during old cooldown window");
  assert(cam05.begin_recovery(22'000'000));
  cam05.owner_removed(22'100'000, false);
  assert(cam05.state() == SourceHealthState::BACKOFF);
  assert(cam05.consecutive_failures() == 2);
  assert(cam05.next_retry_us() - 22'100'000 == 2'000'000);
  assert(!should_fail_process(FailureScope::SOURCE_LOCAL));

  cam05.source_started(cam05.next_retry_us());
  cam05.mark_online(25'000'000);
  assert(!cam05.observe_online_stability(54'999'999));
  assert(cam05.observe_online_stability(55'000'000));
  assert(cam05.consecutive_failures() == 0);

  // Camera-local backoff is bounded, grows after repeat failures, and auth
  // rejection is quarantined much longer to avoid a credential retry storm.
  const auto first_delay = retry_delay_us(1, false);
  const auto second_delay = retry_delay_us(2, false);
  const auto capped_delay = retry_delay_us(100, false);
  assert(first_delay == 1'000'000);
  assert(second_delay == 2'000'000);
  assert(capped_delay == 60'000'000);
  assert(retry_delay_us(1, true) == 300'000'000);

  SourceRecoveryPolicy cam04;
  cam04.mark_online(1'000'000);
  cam04.mark_suspect(5'000'000, "decoder error");
  cam04.begin_recovery(6'000'000);
  cam04.owner_removed(6'100'000, true);
  assert(cam04.state() == SourceHealthState::QUARANTINED);
  assert(cam04.next_retry_us() - 6'100'000 == 300'000'000);
  assert(cam04.state() != cam05.state());

  SourceRecoveryPolicy cam02;
  cam02.mark_online(1'000'000);
  cam02.mark_suspect(2'000'000, "offline for two minutes");
  cam02.begin_recovery(3'000'000);
  cam02.owner_removed(3'100'000, false);
  assert(cam02.state() == SourceHealthState::BACKOFF);
  assert(!should_fail_process(FailureScope::SOURCE_LOCAL));

  // Independent failures do not couple camera health. Shared failures remain
  // process-fatal.
  SourceRecoveryPolicy cam01;
  cam01.mark_online(1'000'000);
  assert(cam01.state() == SourceHealthState::ONLINE);
  assert(should_fail_process(FailureScope::SHARED_PIPELINE));
  assert(should_fail_process(FailureScope::PROGRAMMER));

  // A source epoch advances only when a replacement owner is actually
  // started, allowing native track IDs to be safely reused after resets.
  cam05.mark_suspect(56'000'000, "new interruption after stable period");
  assert(cam05.begin_recovery(56'100'000));
  cam05.owner_removed(56'200'000, false);
  cam05.source_started(cam05.next_retry_us());
  assert(cam05.source_epoch() == 3);

  std::puts("PASS: camera-local recovery, bounded backoff, quarantine and shared-fatal policy");
}
