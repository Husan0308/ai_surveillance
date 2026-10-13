#include "services/camera_v11/native/six_camera_v13/cross_id_context_duplicate.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static CrossIdDuplicateDecision admit(
    std::uint64_t candidate_id, const std::vector<CrossIdTrack> &tracks,
    const std::vector<CrossIdContextProposal> &proposals, std::int32_t frame,
    std::uint64_t pts) {
  return cross_id_context_duplicate_birth(1, 7, 7, frame, frame, pts, pts,
      true, true, candidate_id, tracks, proposals, 1920, 1080);
}

int main() {
  // Exact-r1 CAM-02 frame-39 active birth that later creates the confirmed
  // frame-113 same-person/different-native-ID duplicate (IDs 3 and 20).
  const std::vector<CrossIdTrack> cam02_f39{
      {20, {431.812f, 540.750f, 126.376f, 126.000f}, false},
      {3, {376.875f, 414.000f, 178.500f, 205.500f}, true},
      {4, {550.500f, 493.875f, 262.500f, 275.250f}, true},
      {5, {803.250f, 699.000f, 288.000f, 244.500f}, true},
  };
  const std::vector<CrossIdContextProposal> cam02_f39_context{
      {.806152f, {376.768f, 414.357f, 183.036f, 249.286f}},
      {.819336f, {549.714f, 492.571f, 263.571f, 300.714f}},
  };
  const auto f39 = admit(20, cam02_f39, cam02_f39_context, 39, 1950000000ULL);
  assert(f39.suppress_candidate && f39.owner_id == 3 && f39.proposal_index == 0);
  assert(std::fabs(f39.owner_proposal_iou - .801f) < .01f);
  assert(std::fabs(f39.candidate_proposal_iou - .337f) < .01f);

  // If two new IDs claim the same established owner in one source frame,
  // candidate iteration order must not decide which ID is hidden.
  auto competing_birth_tracks = cam02_f39;
  competing_birth_tracks.push_back({21, cam02_f39.front().box, false});
  const std::vector<CrossIdBirthContext> competing_births{
      {20, true, cam02_f39_context}, {21, true, cam02_f39_context}};
  const auto competing = cross_id_context_duplicate_birth_batch(
      1, 7, 7, 39, 39, 1950000000ULL, 1950000000ULL, true,
      competing_birth_tracks, competing_births, 1920, 1080);
  assert(competing.size() == 2);
  assert(!competing[0].suppress_candidate && !competing[1].suppress_candidate);
  assert(std::strcmp(competing[0].reason,
      "AMBIGUOUS_MULTIPLE_BIRTHS_CLAIM_OWNER") == 0);
  assert(std::strcmp(competing[1].reason,
      "AMBIGUOUS_MULTIPLE_BIRTHS_CLAIM_OWNER") == 0);

  // Exact-r1 CAM-02 second duplicate birth at frame 995 (IDs 4 and 34).
  const std::vector<CrossIdTrack> cam02_f995{
      {4, {629.438f, 457.875f, 229.124f, 254.250f}, true},
      {34, {596.062f, 511.500f, 139.876f, 220.500f}, false},
      {3, {317.250f, 441.000f, 227.250f, 228.000f}, true},
      {5, {639.938f, 630.000f, 419.624f, 273.000f}, true},
  };
  const std::vector<CrossIdContextProposal> cam02_f995_context{
      {.821777f, {597.339f, 459.768f, 258.036f, 320.179f}},
      {.910645f, {346.179f, 440.929f, 194.821f, 226.786f}},
      {.920410f, {638.857f, 628.786f, 346.429f, 300.000f}},
  };
  const auto f995 = admit(34, cam02_f995, cam02_f995_context, 995, 49778500000ULL);
  assert(f995.suppress_candidate && f995.owner_id == 4 && f995.proposal_index == 0);
  assert(std::fabs(f995.owner_proposal_iou - .680f) < .02f);
  assert(std::fabs(f995.candidate_proposal_iou - .369f) < .02f);

  // CAM-05 frame 1300: two real seated people have overlapping boxes. A broad
  // context box is closer to ID 22, but the new track's IoU is too high and
  // ownership margin too small to discard; fail open and preserve both.
  const std::vector<CrossIdTrack> cam05_f1300{
      {28, {970.875f, 148.500f, 110.250f, 140.250f}, false},
      {22, {916.125f, 141.000f, 120.750f, 185.250f}, true},
      {6, {1067.625f, 148.500f, 105.750f, 285.000f}, true},
  };
  const std::vector<CrossIdContextProposal> cam05_f1300_context{
      {.553711f, {915.643f, 146.518f, 160.357f, 189.821f}},
      {.854980f, {1068.857f, 149.018f, 105.714f, 284.821f}},
  };
  const auto f1300 = admit(28, cam05_f1300, cam05_f1300_context, 1300, 65000000000ULL);
  assert(!f1300.suppress_candidate);

  // CAM-05 frame 2761: saved candidate replay shows new ID 85 overlapping
  // established ID 82 on the same seated person. Context proposal 0 maps to
  // the established track at IoU .853 while only weakly covering the new box
  // at IoU .281. Other context proposals resolve the surrounding people
  // separately, so this is a positive cross-ID duplicate fixture, not an
  // overlap-only merge.
  const std::vector<CrossIdTrack> cam05_f2761{
      {85, {1000.875f, 129.750f, 62.250f, 137.250f}, false},
      {82, {1001.250f, 134.062f, 118.500f, 211.875f}, true},
      {75, {873.000f, 108.938f, 114.000f, 214.125f}, true},
      {71, {1126.500f, 250.125f, 156.000f, 429.750f}, true},
      {6, {1108.500f, 103.500f, 124.500f, 316.500f}, true},
  };
  const std::vector<CrossIdContextProposal> cam05_f2761_context{
      {.870605f, {999.143f, 142.411f, 119.643f, 224.464f}},
      {.903809f, {1105.750f, 102.768f, 126.786f, 316.607f}},
      {.908203f, {870.348f, 108.125f, 118.661f, 213.750f}},
      {.929199f, {1127.536f, 249.464f, 154.643f, 389.643f}},
  };
  const auto cam05_f2761_decision = admit(85, cam05_f2761,
      cam05_f2761_context, 2761, 138050000000ULL);
  assert(cam05_f2761_decision.suppress_candidate &&
      cam05_f2761_decision.owner_id == 82 &&
      cam05_f2761_decision.proposal_index == 0);
  assert(std::fabs(cam05_f2761_decision.owner_proposal_iou - .853f) < .01f);
  assert(std::fabs(cam05_f2761_decision.candidate_proposal_iou - .281f) < .01f);

  // CAM-05 IDs 6 and 71 are two visually distinct people at frame 2763 and
  // overlap at IoU ~0.25. Neither is a newly born track there: no birth event
  // means the candidate must make no canonical suppression decision.
  const std::vector<CrossIdTrack> cam05_f2763{
      {6, {1107.750f, 104.812f, 123.000f, 316.876f}, true},
      {71, {1130.625f, 262.125f, 162.750f, 444.750f}, true},
  };
  assert(std::strcmp(cross_id_context_duplicate_birth(2, 7, 7, 2763, 2761,
      138150000000ULL, 138050000000ULL, true, true, 71,
      cam05_f2763, {}, 1920, 1080).reason,
      "STALE_OR_INCOMPLETE_EXACT_FRAME_EVIDENCE") == 0);

  // Their boxes overlap at .25 but neither ID is linked by a fresh birth
  // decision. The ledger must leave both untouched: geometry alone cannot
  // merge the two visibly distinct people.
  CrossIdDuplicateLedger distinct_people_ledger;
  const auto distinct_people = distinct_people_ledger.filter_frame(
      4, 7, 2763, cam05_f2763);
  assert(distinct_people.suppressed_ids.empty());
  assert(distinct_people.released_ids.empty());

  // Supplemental CAM-02 full replay frame 1354 has a raw NvDCF ID 44 at the
  // competing location, but the active-birth/canonical path does not publish
  // ID 44 on that exact frame. The filter receives only canonical-eligible
  // tracks, so it must not suppress the verified current person ID 51 based
  // on an unpublished raw tracker row.
  const std::vector<CrossIdTrack> cam02_f1354_published{
      {51, {102.938f, 161.812f, 103.688f, 250.875f}, false},
      {1, {463.688f, 332.812f, 196.125f, 285.375f}, true},
      {2, {610.500f, 483.750f, 235.500f, 300.000f}, true},
      {4, {768.750f, 679.125f, 267.000f, 240.750f}, true},
  };
  const std::vector<CrossIdContextProposal> cam02_f1354_context{
      {.759766f, {462.857f, 332.143f, 177.143f, 305.714f}},
      {.910645f, {103.929f, 157.500f, 214.286f, 331.071f}},
  };
  const auto no_published_owner = admit(51, cam02_f1354_published,
      cam02_f1354_context, 1354, 67770500000ULL);
  assert(!no_published_owner.suppress_candidate);

  // Missing exact-frame history, epoch mismatch, ambiguous candidate output,
  // and invalid coordinates all fail open.
  auto stale = cross_id_context_duplicate_birth(1, 8, 7, 39, 39,
      1950000000ULL, 1950000000ULL, true, true, 20,
      cam02_f39, cam02_f39_context, 1920, 1080);
  assert(!stale.suppress_candidate);
  auto missing_history = cross_id_context_duplicate_birth(1, 7, 7, 39, 39,
      1950000000ULL, 1950000000ULL, false, true, 20,
      cam02_f39, cam02_f39_context, 1920, 1080);
  assert(!missing_history.suppress_candidate);
  auto duplicate_candidate = cam02_f39;
  duplicate_candidate.push_back(cam02_f39.front());
  assert(!admit(20, duplicate_candidate, cam02_f39_context, 39,
      1950000000ULL).suppress_candidate);

  // The source-epoch-scoped continuity link suppresses only while both tracks
  // remain current and their birth relative geometry is stable. Owner loss,
  // movement divergence, frame gaps, and reset release instead of merging IDs.
  CrossIdDuplicateLedger ledger;
  assert(ledger.link(1, 7, 39, cam02_f39[0], cam02_f39[1]));
  auto at_birth = ledger.filter_frame(1, 7, 39, cam02_f39);
  assert(at_birth.suppressed_ids.size() == 1 && at_birth.suppressed_ids[0] == 20);
  const std::vector<CrossIdTrack> cam02_f40{
      {20, {432.375f, 541.125f, 125.250f, 123.750f}, true},
      {3, {376.969f, 414.000f, 178.312f, 220.500f}, true},
      {4, {550.500f, 493.500f, 263.250f, 277.500f}, true},
      {5, {803.250f, 699.375f, 288.000f, 245.250f}, true},
  };
  auto next = ledger.filter_frame(1, 7, 40, cam02_f40);
  assert(next.suppressed_ids.size() == 1 && next.suppressed_ids[0] == 20);

  // A temporary active-output gap is retained only when NvDCF explicitly
  // reports the same target as a shadow. It is never rendered while absent.
  const std::unordered_map<std::uint64_t, CrossIdRect> shadow20{
      {20, {432.360f, 541.336f, 125.526f, 124.302f}}};
  auto inactive = ledger.filter_frame(1, 7, 41,
      {cam02_f40[1], cam02_f40[2], cam02_f40[3]}, shadow20);
  assert(inactive.suppressed_ids.empty() && inactive.released_ids.empty());
  const std::vector<CrossIdTrack> cam02_f42{
      {20, {432.750f, 541.500f, 124.500f, 126.750f}, true},
      {3, {377.250f, 414.375f, 178.500f, 219.750f}, true},
      {4, {550.125f, 493.125f, 263.625f, 277.125f}, true},
      {5, {803.625f, 699.000f, 288.000f, 245.625f}, true},
  };
  auto reactivated = ledger.filter_frame(1, 7, 42, cam02_f42);
  assert(reactivated.suppressed_ids.size() == 1 && reactivated.suppressed_ids[0] == 20);

  // When the historical owner drops to shadow but the new ID is active, keep
  // the current box visible. If the old owner becomes active again, normal
  // simultaneous-active duplicate suppression resumes.
  const std::unordered_map<std::uint64_t, CrossIdRect> shadow_owner{
      {3, {376.2f, 414.1f, 179.0f, 238.0f}}};
  const std::vector<CrossIdTrack> candidate_only{cam02_f42[0], cam02_f42[2], cam02_f42[3]};
  auto owner_inactive = ledger.filter_frame(1, 7, 43, candidate_only, shadow_owner);
  assert(owner_inactive.suppressed_ids.empty() && owner_inactive.released_ids.empty());
  // A detector-backed current fallback for the owner is different from a
  // shadow-only row: once both linked targets have current canonical geometry,
  // publish exactly the established owner and suppress the duplicate ID.
  CrossIdDuplicateLedger current_fallback_ledger;
  assert(current_fallback_ledger.link(1, 7, 42, cam02_f39[0], cam02_f39[1]));
  assert(current_fallback_ledger.filter_frame(1, 7, 42, cam02_f39)
      .suppressed_ids.size() == 1);
  const std::vector<CrossIdTrack> active_plus_current_fallback{
      {20, {432.750f, 541.500f, 124.500f, 126.750f}, true},
      {3, {377.250f, 414.375f, 178.500f, 219.750f}, false},
      {4, {550.125f, 493.125f, 263.625f, 277.125f}, true},
      {5, {803.625f, 699.000f, 288.000f, 245.625f}, true},
  };
  const auto active_plus_fallback = current_fallback_ledger.filter_frame(
      1, 7, 43, active_plus_current_fallback);
  assert(active_plus_fallback.suppressed_ids.size() == 1 &&
      active_plus_fallback.suppressed_ids[0] == 20);
  auto both_active_again = ledger.filter_frame(1, 7, 44, cam02_f42);
  assert(both_active_again.suppressed_ids.size() == 1 &&
      both_active_again.suppressed_ids[0] == 20);

  // Exact CAM-02 frame-104..113 posture-change continuation from the saved
  // r1 replay. The smaller target alternates ACTIVE/INACTIVE and its relative
  // area ratio changes .332 -> .527 -> .587 -> .669 -> .493 while remaining
  // inside the same established target. A birth-relative ratio delta wrongly
  // released this proven link at frame 109, just before both IDs became active
  // together at frame 113. Compare consecutive shape changes instead, while
  // retaining the birth-relative center envelope and exact active/shadow
  // presence requirements.
  CrossIdDuplicateLedger posture_ledger;
  const CrossIdTrack posture_owner_104{
      1, {376.594f, 416.25f, 179.812f, 241.5f}, true};
  const CrossIdTrack posture_candidate_104{
      6, {431.376f, 533.091f, 124.712f, 115.488f}, false};
  assert(posture_ledger.link(0, 7, 104, posture_candidate_104,
      posture_owner_104));
  auto posture_104 = posture_ledger.filter_frame(0, 7, 104,
      {posture_owner_104}, {{6, posture_candidate_104.box}});
  assert(posture_104.suppressed_ids.empty());
  const std::vector<CrossIdTrack> posture_107_candidate{
      {6, {434.25f, 540.375f, 129.75f, 176.25f}, false}};
  const CrossIdRect posture_107_owner_shadow{
      376.866f, 414.114f, 179.888f, 241.331f};
  assert(posture_ledger.filter_frame(0, 7, 105, {}, {
      {1, {376.736f, 414.675f, 179.892f, 241.296f}},
      {6, {431.245f, 532.360f, 124.712f, 115.488f}}}).released_ids.empty());
  assert(posture_ledger.filter_frame(0, 7, 106, {}, {
      {1, {376.808f, 414.280f, 179.889f, 241.320f}},
      {6, {431.181f, 532.229f, 124.712f, 115.488f}}}).released_ids.empty());
  assert(posture_ledger.filter_frame(0, 7, 107, posture_107_candidate, {
      {1, posture_107_owner_shadow}}).released_ids.empty());
  assert(posture_ledger.filter_frame(0, 7, 108,
      {{1, {377.625f, 416.25f, 179.25f, 229.5f}, true}}, {
       {6, {433.855f, 535.869f, 130.491f, 185.192f}}}).released_ids.empty());
  assert(posture_ledger.filter_frame(0, 7, 109, {}, {
      {1, {377.457f, 416.879f, 179.402f, 232.323f}},
      {6, {434.241f, 533.258f, 132.575f, 210.317f}}}).released_ids.empty());
  assert(posture_ledger.filter_frame(0, 7, 110,
      {{6, {433.5f, 540.0f, 130.5f, 157.5f}, false}}, {
       {1, {377.409f, 417.294f, 179.394f, 232.190f}}}).released_ids.empty());
  assert(posture_ledger.filter_frame(0, 7, 111,
      {{6, {434.25f, 541.875f, 130.5f, 158.25f}, false}}, {
       {1, {377.497f, 417.258f, 179.391f, 232.126f}}}).released_ids.empty());
  assert(posture_ledger.filter_frame(0, 7, 112,
      {{6, {434.438f, 541.5f, 130.125f, 156.0f}, false}}, {
       {1, {377.534f, 417.427f, 179.389f, 232.097f}}}).released_ids.empty());
  const auto posture_frame_113 = posture_ledger.filter_frame(0, 7, 113,
      {{1, {377.719f, 415.5f, 179.062f, 234.0f}, true},
       {6, {432.938f, 544.125f, 131.625f, 153.75f}, false}});
  assert(posture_frame_113.suppressed_ids.size() == 1 &&
      posture_frame_113.suppressed_ids[0] == 6);

  // A disjoint movement breaks the relative geometry link and releases the
  // ID, preserving it as a separate person rather than merging by history.
  const std::vector<CrossIdTrack> separated{
      {20, {700.0f, 500.0f, 126.0f, 126.0f}, true},
      {3, {378.0f, 414.0f, 179.0f, 235.0f}, true},
  };
  auto diverged = ledger.filter_frame(1, 7, 45, separated);
  assert(diverged.suppressed_ids.empty() && diverged.released_ids.size() == 1);

  // Without an explicit current shadow row, absence releases immediately.
  assert(ledger.link(1, 7, 46, cam02_f39[0], cam02_f39[1]));
  auto linked_no_shadow = ledger.filter_frame(1, 7, 46, cam02_f39);
  assert(linked_no_shadow.suppressed_ids.size() == 1);
  auto no_shadow = ledger.filter_frame(1, 7, 47,
      {cam02_f40[1], cam02_f40[2], cam02_f40[3]});
  assert(no_shadow.suppressed_ids.empty() && no_shadow.released_ids.size() == 1);

  // Owner loss clears a linked duplicate, even if the candidate remains.
  assert(ledger.link(1, 7, 48, cam02_f39[0], cam02_f39[1]));
  auto linked_owner_lost = ledger.filter_frame(1, 7, 48, cam02_f39);
  assert(linked_owner_lost.suppressed_ids.size() == 1);
  const std::vector<CrossIdTrack> owner_lost{cam02_f40[0], cam02_f40[2], cam02_f40[3]};
  auto released = ledger.filter_frame(1, 7, 49, owner_lost);
  assert(released.suppressed_ids.empty() && released.released_ids.size() == 1);
  assert(ledger.link(1, 7, 42, cam02_f39[0], cam02_f39[1]));
  ledger.clear_source(1);
  assert(ledger.filter_frame(1, 8, 43, cam02_f40).suppressed_ids.empty());
  return 0;
}
