#include "track_birth_verification.hpp"

#include <cassert>

int main() {
  TrackBirthVerificationLedger ledger;
  constexpr std::uint64_t id = 810;

  // NvDCF ID zero is a valid native target ID, not an uninitialized sentinel.
  assert(ledger.begin(0, 0, 0));
  assert(ledger.state(0, 0) == TrackBirthVerificationLedger::State::Pending);
  ledger.finish(0, 0, 0, true);
  assert(ledger.verified(0, 0));
  assert(!ledger.begin(0, 0, 1));
  assert(ledger.verified_with_recent_support(0, 0, 0, 5));

  // A current low-confidence rectangle may reuse a prior positive appearance
  // result only while ordinary detector rows keep associating to that same
  // verified native ID. Shadow touches do not refresh this support clock.
  assert(ledger.begin(4, 900, 27));
  ledger.finish(4, 900, 27, 27, true, false);
  assert(ledger.verified_with_recent_support(4, 900, 27, 5));
  ledger.support_verified_target(4, 900, 27);
  assert(ledger.verified_with_recent_support(4, 900, 32, 5));
  assert(ledger.verified_support_age(4, 900, 30) == 3);
  assert(ledger.lowconf_fallback_authorized(4, 900, 30, false, false, 5));
  assert(ledger.lowconf_fallback_authorized(4, 900, 30, true, true, 5));
  // A timely negative/inconclusive current-frame appearance result is not
  // overridden by prior track support.
  assert(!ledger.lowconf_fallback_authorized(4, 900, 30, true, false, 5));
  assert(!ledger.verified_with_recent_support(4, 900, 33, 5));
  assert(!ledger.lowconf_fallback_authorized(4, 900, 33, false, false, 5));
  ledger.touch(4, 900, 33);  // tracker/shadow lifecycle is not detector support
  assert(!ledger.verified_with_recent_support(4, 900, 33, 5));
  ledger.support_verified_target(4, 900, 34);
  assert(ledger.verified_with_recent_support(4, 900, 39, 5));
  assert(!ledger.verified_with_recent_support(3, 900, 34, 5));

  assert(!ledger.verified(1, id));
  assert(ledger.begin(1, id, 10));
  assert(ledger.state(1, id) == TrackBirthVerificationLedger::State::Pending);
  assert(!ledger.begin(1, id, 11));
  ledger.finish(1, id, 9, true);  // wrong exact frame cannot authorize output
  assert(!ledger.verified(1, id));
  ledger.finish(1, id, 10, false, 3);  // fail closed and back off before retry
  assert(!ledger.verified(1, id));
  assert(!ledger.begin(1, id, 11));
  assert(!ledger.begin(1, id, 12));
  assert(ledger.begin(1, id, 13));
  ledger.finish(1, id, 13, true);
  assert(ledger.verified(1, id));
  assert(!ledger.begin(1, id, 14));

  // An inference result may arrive well after its source frame. Pending state
  // remains single-flight until that exact request is resolved, then authorizes
  // only future current-frame boxes for the same native ID.
  assert(ledger.begin(4, 91, 20));
  assert(!ledger.begin(4, 91, 50));
  ledger.finish(4, 91, 20, true);
  assert(ledger.verified(4, 91));

  // Verification state is isolated per camera and can be explicitly trusted
  // only by a separate exact-frame verified-fallback path.
  assert(!ledger.verified(2, id));
  assert(ledger.begin(2, id, 12));
  ledger.finish(2, id, 12, true);
  assert(ledger.verified(2, id));

  // A decision ages out after the track/shadow retention horizon so a reused
  // native ID cannot inherit a stale answer.
  assert(ledger.begin(3, id, 1));
  ledger.finish(3, id, 1, true);
  assert(ledger.begin(3, id, 300));
  assert(!ledger.verified(3, id));

  // Queue backpressure is not a negative appearance result: it must allow a
  // retry on the next current frame rather than imposing detector-frame delay.
  assert(ledger.begin(5, 810, 100));
  ledger.cancel(5, 810, 100, 0);
  assert(ledger.begin(5, 810, 101));

  // One completed, definitive no-local-person result rejects this target.
  assert(ledger.begin(2, 44, 10));
  ledger.finish(2, 44, 10, 10, false, true, 5);
  assert(ledger.rejected(2, 44));
  assert(!ledger.verified_with_recent_support(2, 44, 10, 5));
  assert(!ledger.begin(2, 44, 11));

  // Ambiguity / timeout / non-definitive results never count as negatives.
  assert(ledger.begin(2, 45, 10));
  ledger.finish(2, 45, 10, 10, false, false, 1);
  assert(!ledger.rejected(2, 45));
  assert(ledger.begin(2, 45, 11));
  ledger.finish(2, 45, 11, 11, true, false, 1);
  assert(ledger.verified(2, 45));
  return 0;
}
