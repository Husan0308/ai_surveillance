#include "services/camera_v11/native/six_camera_v13/person_proposal_dedupe.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
  // Confirmed CAM-02 single-person duplicate at frame 4803, converted from
  // the evidence's xyxy convention to the native NvDsObjectMeta xywh form.
  const PersonProposalBox cam02_tight{1, .631836f, 1252, 1088, 352, 308};
  const PersonProposalBox cam02_loose{1, .395996f, 1249.5f, 932.5f, 517, 467};
  PersonProposalDuplicateMetrics cam02_metrics;
  assert(person_proposal_duplicate_metrics(cam02_tight, cam02_loose, &cam02_metrics));
  assert(std::fabs(cam02_metrics.iou - .44904f) < .001f);
  assert(cam02_metrics.containment > .999f);
  const auto cam02_suppressed = person_proposal_duplicates({cam02_tight, cam02_loose});
  assert(cam02_suppressed.size() == 1);
  assert(cam02_suppressed[0].suppressed_index == 1);
  assert(cam02_suppressed[0].retained_index == 0);

  // Human-positive CAM-02 white-shirt frame 5656 contains the same person as
  // two near-fully nested PGIE proposals. The loose proposal is slightly more
  // confident, but extends over the chair; the tighter current-person box is
  // the reviewed geometry to retain. This pair was just outside the former
  // center-distance gate (0.263 versus 0.25 of the taller box).
  const PersonProposalBox cam02_r1_tight{1, .507812f, 1271.5f, 1090.0f, 349.0f, 320.0f};
  const PersonProposalBox cam02_r1_loose{1, .562012f, 1272.5f, 932.5f, 551.0f, 483.0f};
  PersonProposalDuplicateMetrics cam02_r1_metrics;
  assert(person_proposal_duplicate_metrics(cam02_r1_tight, cam02_r1_loose,
      &cam02_r1_metrics));
  assert(std::fabs(cam02_r1_metrics.iou - .4183f) < .001f);
  assert(cam02_r1_metrics.containment > .99f);
  assert(cam02_r1_metrics.center_over_max_height > .25f &&
      cam02_r1_metrics.center_over_max_height < .28f);
  const auto cam02_r1_suppressed = person_proposal_duplicates(
      {cam02_r1_tight, cam02_r1_loose});
  assert(cam02_r1_suppressed.size() == 1);
  assert(cam02_r1_suppressed[0].suppressed_index == 1);
  assert(cam02_r1_suppressed[0].retained_index == 0);
  assert(cam02_r1_suppressed[0].retained_tighter_geometry);

  // The CAM-01 accepted frame has two real people plus a lower-confidence,
  // nested duplicate of the seated person's detector box. Keep both people.
  const std::vector<PersonProposalBox> cam01_two_people{
      {1, .599121f, 2008.5f, 644.5f, 183, 219},
      {1, .519531f, 1948.5f, 901, 163, 210},
      {1, .274414f, 1913.5f, 906, 197, 376},
  };
  const auto cam01_suppressed = person_proposal_duplicates(cam01_two_people);
  assert(cam01_suppressed.size() == 1);
  assert(cam01_suppressed[0].suppressed_index == 2);
  assert(cam01_suppressed[0].retained_index == 1);

  // CAM-04 frame 6748 includes two nearby people with overlapping boxes.
  // Their boxes are not a nested same-person duplicate and must both survive.
  const PersonProposalBox cam04_person_a{1, .373047f, 686.964f, 158.929f,
      164.643f, 162.142f};
  const PersonProposalBox cam04_person_b{1, .341553f, 690.75f, 197,
      168.5f, 227};
  PersonProposalDuplicateMetrics cam04_metrics;
  assert(!person_proposal_duplicate_metrics(cam04_person_a, cam04_person_b,
      &cam04_metrics));
  assert(cam04_metrics.iou > .30f && cam04_metrics.iou < .45f);
  assert(cam04_metrics.containment < .95f);
  assert(person_proposal_duplicates({cam04_person_a, cam04_person_b}).empty());

  // Current CAM-02 live evidence: one person produced a loose full-body box
  // and a tight nested box with IoU below the NMS cutoff. Their high
  // containment, center proximity and aligned bottom edge identify a unique
  // nested duplicate; keep the higher-confidence tight proposal.
  const PersonProposalBox cam02_live_loose{1, .320801f, 1316, 898, 476, 506};
  const PersonProposalBox cam02_live_tight{1, .503906f, 1325, 1146, 214, 244};
  PersonProposalDuplicateMetrics cam02_live_metrics;
  assert(person_proposal_duplicate_metrics(cam02_live_loose, cam02_live_tight,
      &cam02_live_metrics));
  assert(cam02_live_metrics.iou < .30f);
  assert(cam02_live_metrics.containment > .99f);
  assert(cam02_live_metrics.center_over_max_height > .30f);
  assert(cam02_live_metrics.bottom_edge_delta_over_max_height < .04f);
  const auto cam02_live_suppressed = person_proposal_duplicates(
      {cam02_live_loose, cam02_live_tight});
  assert(cam02_live_suppressed.size() == 1);
  assert(cam02_live_suppressed[0].suppressed_index == 0);
  assert(cam02_live_suppressed[0].retained_index == 1);

  // Similar overlap without bottom-edge alignment is insufficient evidence
  // to merge an occluded nearby person.
  const PersonProposalBox distinct_occluded{1, .51f, 1325, 1050, 214, 244};
  assert(person_proposal_duplicates({cam02_live_loose, distinct_occluded}).empty());

  // Cross-component pairs use the existing auxiliary/primary rule and are not
  // silently consumed by this same-component policy.
  PersonProposalBox cross_component = cam02_loose;
  cross_component.component_id = 3;
  assert(!person_proposal_duplicate_metrics(cam02_tight, cross_component));

  // Exact r1 CAM-02 replay at source-local frame 2443: the short upper-body
  // proposal is a lower-quality duplicate, while the full-body proposal
  // matches the immediately previous active ID 4 much more strongly.
  const std::vector<PersonProposalBox> cam02_history_pair{
      {1, .370117f, 564.750f, 450.000f, 243.750f, 150.000f},
      {1, .702637f, 576.750f, 466.500f, 239.250f, 267.000f},
  };
  const std::vector<PersonProposalHistoryTarget> cam02_previous{
      {4, 2442, 579.000f, 469.500f, 246.750f, 267.000f},
  };
  const auto cam02_history_decisions =
      person_proposal_duplicates_supported_by_previous_tracks(
          cam02_history_pair, cam02_previous, 2443);
  assert(cam02_history_decisions.size() == 1);
  assert(cam02_history_decisions[0].target_id == 4);
  assert(cam02_history_decisions[0].suppressed_index == 0);
  assert(cam02_history_decisions[0].retained_index == 1);
  assert(cam02_history_decisions[0].retained_history_iou -
      cam02_history_decisions[0].suppressed_history_iou >= .15f);

  // CAM-05 frame 2757's nested proposal follows the same target less well;
  // retain the current geometry that agrees with the immediately previous ID.
  const std::vector<PersonProposalBox> cam05_history_pair{
      {1, .299072f, 1004.250f, 133.500f, 90.000f, 200.250f},
      {1, .484375f, 1001.625f, 129.938f, 63.750f, 139.125f},
  };
  const std::vector<PersonProposalHistoryTarget> cam05_previous{
      {71, 2756, 1002.000f, 132.750f, 108.000f, 225.000f},
  };
  const auto cam05_history_decisions =
      person_proposal_duplicates_supported_by_previous_tracks(
          cam05_history_pair, cam05_previous, 2757);
  assert(cam05_history_decisions.size() == 1);
  assert(cam05_history_decisions[0].target_id == 71);
  assert(cam05_history_decisions[0].suppressed_index == 1);
  assert(cam05_history_decisions[0].retained_index == 0);

  // CAM-05 frame 2805 is a cross-confidence nested pair on prior ID 85. The
  // short current upper-body box follows the established target; the taller
  // duplicate is not allowed to spawn a second local ID.
  const std::vector<PersonProposalBox> cam05_occlusion_pair{
      {1, .385742f, 894.750f, 98.250f, 70.500f, 225.750f},
      {1, .460938f, 895.500f, 99.375f, 61.500f, 74.250f},
  };
  const std::vector<PersonProposalHistoryTarget> cam05_occlusion_previous{
      {85, 2804, 904.875f, 99.750f, 53.250f, 72.000f},
  };
  const auto cam05_occlusion_decisions =
      person_proposal_duplicates_supported_by_previous_tracks(
          cam05_occlusion_pair, cam05_occlusion_previous, 2805);
  assert(cam05_occlusion_decisions.size() == 1);
  assert(cam05_occlusion_decisions[0].target_id == 85);
  assert(cam05_occlusion_decisions[0].suppressed_index == 0);
  assert(cam05_occlusion_decisions[0].retained_index == 1);

  // Distinct close people can overlap and still remain separate when each
  // proposal has its own exact, unambiguous previous-frame track.
  const std::vector<PersonProposalBox> two_close_people{
      {1, .71f, 100.0f, 100.0f, 80.0f, 160.0f},
      {1, .69f, 130.0f, 104.0f, 80.0f, 160.0f},
  };
  const std::vector<PersonProposalHistoryTarget> two_close_previous{
      {41, 100, 100.0f, 100.0f, 80.0f, 160.0f},
      {42, 100, 130.0f, 104.0f, 80.0f, 160.0f},
  };
  assert(person_proposal_duplicates_supported_by_previous_tracks(
      two_close_people, two_close_previous, 101).empty());

  // A stale frame, an ambiguous old-track match, and below-normal confidence
  // all fail open; no proposal is removed on incomplete evidence.
  assert(person_proposal_duplicates_supported_by_previous_tracks(
      cam02_history_pair, cam02_previous, 2444).empty());
  const std::vector<PersonProposalHistoryTarget> ambiguous_previous{
      {4, 2442, 579.000f, 469.500f, 246.750f, 267.000f},
      {9, 2442, 579.000f, 469.500f, 246.750f, 267.000f},
  };
  assert(person_proposal_duplicates_supported_by_previous_tracks(
      cam02_history_pair, ambiguous_previous, 2443).empty());
  const std::vector<PersonProposalBox> lowconf_pair{
      {1, .249f, 564.750f, 450.000f, 243.750f, 150.000f},
      {1, .702637f, 576.750f, 466.500f, 239.250f, 267.000f},
  };
  assert(person_proposal_duplicates_supported_by_previous_tracks(
      lowconf_pair, cam02_previous, 2443).empty());

  // Equal-confidence geometry is ambiguous; fail open rather than selecting
  // an arbitrary native proposal.
  PersonProposalBox equal_conf = cam02_loose;
  equal_conf.confidence = cam02_tight.confidence;
  assert(person_proposal_duplicates({cam02_tight, equal_conf}).empty());

  std::puts("PASS: same-component nested duplicate suppression and close-person preservation");
}
