#include "track_birth_verification.hpp"

#include <cassert>

int main() {
  // Context-crop verification must be spatially tied to the active target.
  // A broad adjacent-person crop can contain the target center but must not
  // confirm a static garment/background proposal at IoU ~= 0.162.
  float context_iou = -1.0f;
  assert(!TrackBirthVerificationLedger::active_context_location_match(
      1099.875f, 734.625f, 194.250f, 201.750f,
      959.0f, 584.0f, 482.0f, 501.0f, &context_iou));
  assert(context_iou > 0.15f && context_iou < 0.17f);
  assert(TrackBirthVerificationLedger::active_context_location_match(
      100.0f, 100.0f, 100.0f, 200.0f,
      95.0f, 96.0f, 110.0f, 210.0f, &context_iou));
  assert(context_iou >= 0.25f);
  assert(!TrackBirthVerificationLedger::active_context_location_match(
      100.0f, 100.0f, 100.0f, 200.0f,
      210.0f, 100.0f, 100.0f, 200.0f, &context_iou));
  assert(context_iou == 0.0f);
  assert(!TrackBirthVerificationLedger::active_context_location_match(
      100.0f, 100.0f, 0.0f, 200.0f,
      95.0f, 96.0f, 110.0f, 210.0f, &context_iou));

  TrackBirthVerificationLedger ledger;
  constexpr std::uint64_t id = 810;

  // NvDCF ID zero is a valid native target ID, not an uninitialized sentinel.
  assert(ledger.begin(0, 0, 0));
  assert(ledger.state(0, 0) == TrackBirthVerificationLedger::State::Pending);
  ledger.finish(0, 0, 0, true);
  assert(ledger.verified(0, 0));
  assert(!ledger.begin(0, 0, 1));
  assert(ledger.verified_with_recent_support(0, 0, 0, 5));
  assert(ledger.begin(1, 0, 1));
  ledger.finish(1, 0, 1, true);
  ledger.clear_source(0);
  assert(!ledger.verified(0, 0));
  assert(ledger.verified(1, 0));  // source reset must not affect another camera
  assert(ledger.begin(0, 0, 2));  // a reused local ID starts a new source epoch

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

  // A negative applies to the exact crop geometry that was checked. A later
  // current detector box for the same native target may receive one bounded
  // exact-frame recheck only after a material geometry change. The changed
  // box remains withheld until that recheck positively confirms a person.
  constexpr std::uint64_t changed_geometry_id = 8101;
  assert(ledger.begin(2, changed_geometry_id, 100));
  ledger.reject(2, changed_geometry_id, 100, 1236.0f, 853.5f, 566.0f, 493.0f);
  assert(ledger.rejected(2, changed_geometry_id));
  assert(!ledger.begin_geometry_recheck(2, changed_geometry_id, 104,
      1319.0f, 1082.0f, 320.0f, 320.0f));  // minimum delay not met
  assert(!ledger.begin_geometry_recheck(2, changed_geometry_id, 105,
      1236.0f, 853.5f, 566.0f, 493.0f));  // unchanged jacket-like geometry
  assert(ledger.begin_geometry_recheck(2, changed_geometry_id, 105,
      1319.0f, 1082.0f, 320.0f, 320.0f));
  assert(ledger.state(2, changed_geometry_id) == TrackBirthVerificationLedger::State::Pending);
  ledger.finish(2, changed_geometry_id, 105, 105, true, false);
  assert(ledger.verified(2, changed_geometry_id));
  assert(!ledger.begin_geometry_recheck(2, changed_geometry_id, 120,
      1319.0f, 1082.0f, 320.0f, 320.0f));

  constexpr std::uint64_t retry_cancel_id = 8102;
  assert(ledger.begin(2, retry_cancel_id, 200));
  ledger.reject(2, retry_cancel_id, 200, 100.0f, 100.0f, 200.0f, 300.0f);
  assert(ledger.begin_geometry_recheck(2, retry_cancel_id, 205,
      400.0f, 400.0f, 200.0f, 300.0f));
  ledger.cancel_geometry_recheck(2, retry_cancel_id, 205, 1);
  assert(ledger.rejected(2, retry_cancel_id));
  assert(ledger.begin_geometry_recheck(2, retry_cancel_id, 206,
      400.0f, 400.0f, 200.0f, 300.0f));
  ledger.reject(2, retry_cancel_id, 206, 400.0f, 400.0f, 200.0f, 300.0f);
  assert(!ledger.begin_geometry_recheck(2, retry_cancel_id, 220,
      700.0f, 700.0f, 200.0f, 300.0f));  // only one completed recheck

  constexpr std::uint64_t geometry_unknown_id = 8103;
  assert(ledger.begin(2, geometry_unknown_id, 300));
  ledger.reject(2, geometry_unknown_id, 300);  // no stored geometry: fail closed
  assert(!ledger.begin_geometry_recheck(2, geometry_unknown_id, 305,
      700.0f, 700.0f, 200.0f, 300.0f));

  // Ambiguity / timeout / non-definitive results never count as negatives.
  assert(ledger.begin(2, 45, 10));
  ledger.finish(2, 45, 10, 10, false, false, 1);
  assert(!ledger.rejected(2, 45));
  assert(ledger.begin(2, 45, 11));
  ledger.finish(2, 45, 11, 11, true, false, 1);
  assert(ledger.verified(2, 45));
  return 0;
}
