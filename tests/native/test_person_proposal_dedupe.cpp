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

  // Cross-component pairs use the existing auxiliary/primary rule and are not
  // silently consumed by this same-component policy.
  PersonProposalBox cross_component = cam02_loose;
  cross_component.component_id = 3;
  assert(!person_proposal_duplicate_metrics(cam02_tight, cross_component));

  // Equal-confidence geometry is ambiguous; fail open rather than selecting
  // an arbitrary native proposal.
  PersonProposalBox equal_conf = cam02_loose;
  equal_conf.confidence = cam02_tight.confidence;
  assert(person_proposal_duplicates({cam02_tight, equal_conf}).empty());

  std::puts("PASS: same-component nested duplicate suppression and close-person preservation");
}
