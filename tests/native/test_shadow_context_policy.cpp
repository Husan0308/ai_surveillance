#include "../../services/camera_v11/native/six_camera_v13/shadow_continuity.hpp"

#include <cassert>
#include <cmath>

int main() {
  const ShadowContextRect shadow{100.0f, 120.0f, 80.0f, 160.0f};
  const std::vector<ShadowContextRect> one_person{{102.0f, 121.0f, 78.0f, 158.0f}};
  const auto unique = shadow_context_select_unique(shadow, one_person);
  assert(unique.accepted);
  assert(unique.proposal_index == 0);
  assert(unique.iou > 0.90f);

  // Two spatially equivalent person proposals are not silently selected.
  const std::vector<ShadowContextRect> ambiguous{
      {101.0f, 120.0f, 80.0f, 160.0f},
      {100.0f, 121.0f, 80.0f, 160.0f}};
  assert(!shadow_context_select_unique(shadow, ambiguous).accepted);

  // Existing strict spatial limits remain the safety boundary.
  assert(!shadow_context_spatial_match(shadow,
      ShadowContextRect{151.0f, 120.0f, 80.0f, 160.0f}));
  assert(!shadow_context_spatial_match(shadow,
      ShadowContextRect{100.0f, 120.0f, 161.0f, 160.0f}));

  // A crop-local detector rectangle maps back exactly once into frame space.
  const ShadowContextTransform transform{640.0f, 120.0f, 640.0f, 640.0f,
      1.0f, 1.0f};
  const auto mapped = shadow_context_to_frame(
      ShadowContextRect{100.0f, 80.0f, 80.0f, 160.0f}, transform);
  assert(std::abs(mapped[0] - 740.0f) < 1e-5f);
  assert(std::abs(mapped[1] - 200.0f) < 1e-5f);
  assert(std::abs(mapped[2] - 80.0f) < 1e-5f);
  assert(std::abs(mapped[3] - 160.0f) < 1e-5f);

  // The same crop transform accounts for a larger decoded surface while the
  // declared frame coordinate space remains the mux/pipeline dimensions.
  const ShadowContextTransform high_res{1280.0f, 240.0f, 640.0f, 640.0f,
      1.0f, 1.0f};
  const auto high_res_mapped = shadow_context_to_frame(
      ShadowContextRect{100.0f, 80.0f, 80.0f, 160.0f}, high_res);
  assert(std::abs(high_res_mapped[0] - 1380.0f) < 1e-5f);
  assert(std::abs(high_res_mapped[1] - 320.0f) < 1e-5f);
  assert(std::abs(high_res_mapped[2] - 80.0f) < 1e-5f);
  assert(std::abs(high_res_mapped[3] - 160.0f) < 1e-5f);

  const ShadowContextRect prior_box{400.0f, 120.0f, 90.0f, 180.0f};
  // A verified auxiliary target can schedule a current-frame crop from the
  // immediately preceding ACTIVE or INACTIVE observation, only if there is
  // no matching detector proposal on this frame.
  assert(shadow_context_pretracker_hint_eligible(434, 433, 39, true, true,
      true, false, false, prior_box, 1920, 1080));
  assert(shadow_context_pretracker_hint_eligible(434, 433, 39, true, true,
      false, true, false, prior_box, 1920, 1080));
  assert(!shadow_context_pretracker_hint_eligible(434, 432, 39, true, true,
      true, false, false, prior_box, 1920, 1080));
  assert(!shadow_context_pretracker_hint_eligible(434, 433, 39, false, true,
      true, false, false, prior_box, 1920, 1080));
  assert(!shadow_context_pretracker_hint_eligible(434, 433, 39, true, false,
      true, false, false, prior_box, 1920, 1080));
  assert(!shadow_context_pretracker_hint_eligible(434, 433, 39, true, true,
      false, false, false, prior_box, 1920, 1080));
  assert(!shadow_context_pretracker_hint_eligible(434, 433, 39, true, true,
      true, false, true, prior_box, 1920, 1080));
  assert(!shadow_context_pretracker_hint_eligible(434, 433, 0, true, true,
      true, false, false, prior_box, 1920, 1080));
  assert(!shadow_context_pretracker_hint_eligible(434, 433, 39, true, true,
      true, false, false, ShadowContextRect{1900, 1000, 40, 100}, 1920, 1080));
  assert(!shadow_context_pretracker_hint_eligible(434, 433, 39, true, true,
      true, false, false, ShadowContextRect{400, 120, NAN, 180}, 1920, 1080));
  assert(shadow_context_has_recent_primary_support(434, 432, 2));
  assert(!shadow_context_has_recent_primary_support(434, 431, 2));
  assert(!shadow_context_has_recent_primary_support(434, -1, 5));

  // The CAM-05 two-person context frame places both complete prior boxes in
  // one exact mux-space crop, so one verifier invocation may serve both hints.
  const std::vector<ShadowContextRect> nearby_hints{
      {848.3f, 97.6f, 101.3f, 167.9f},
      {994.2f, 169.0f, 117.9f, 148.4f}};
  const auto groups = shadow_context_group_hints(1920, 1080, 1920, 1080,
      nearby_hints);
  assert(groups.size() == 1);
  assert(groups[0].size() == 2);
  const auto group_crop = shadow_context_compute_crop(1920, 1080, 1920, 1080,
      nearby_hints[0]);
  assert(group_crop.valid && group_crop.side == 640);
  assert(shadow_context_crop_contains(group_crop, nearby_hints[1]));

  // Candidate-centered diagnostic crop narrows context enough that a nearby
  // person's center cannot be returned as confirmation for the seated target.
  const ShadowContextRect cam05_seated{994.2f, 169.0f, 117.9f, 148.4f};
  const ShadowContextRect cam05_neighbor{850.5f, 101.8f, 81.0f, 160.9f};
  const auto candidate_crop = shadow_context_compute_candidate_crop(
      1920, 1080, 1920, 1080, cam05_seated);
  assert(candidate_crop.valid);
  assert(candidate_crop.side >= 256 && candidate_crop.side < 320);
  assert(shadow_context_crop_contains(candidate_crop, cam05_seated));
  const float neighbor_center_x = cam05_neighbor.left + cam05_neighbor.width * .5f;
  const float candidate_crop_left = candidate_crop.left / candidate_crop.surface_per_frame_x;
  const float candidate_crop_right = (candidate_crop.left + candidate_crop.side) /
      candidate_crop.surface_per_frame_x;
  assert(neighbor_center_x < candidate_crop_left || neighbor_center_x > candidate_crop_right);
  assert(shadow_context_crop_contains(group_crop, cam05_neighbor));

  const auto clipped_candidate_crop = shadow_context_compute_candidate_crop(
      1920, 1080, 1920, 1080, ShadowContextRect{1850, 100, 100, 500});
  assert(!clipped_candidate_crop.valid);

  const std::vector<ShadowContextRect> two_targets{
      {100, 100, 80, 160}, {300, 100, 80, 160}};
  const std::vector<ShadowContextRect> two_people{
      {101, 101, 79, 158}, {301, 99, 80, 162}};
  const auto one_to_one = shadow_context_match_one_to_one(two_targets, two_people);
  assert(one_to_one.size() == 2);
  assert(one_to_one[0].target_index == 0 && one_to_one[0].proposal_index == 0);
  assert(one_to_one[1].target_index == 1 && one_to_one[1].proposal_index == 1);

  // One verifier box cannot be used to rescue two nearby native targets.
  const std::vector<ShadowContextRect> nearby_targets{
      {100, 100, 80, 160}, {120, 100, 80, 160}};
  const std::vector<ShadowContextRect> one_nearby_proposal{{110, 100, 80, 160}};
  assert(shadow_context_match_one_to_one(nearby_targets, one_nearby_proposal).empty());

  const std::vector<ShadowContextRect> distant_hints{
      {100, 100, 80, 160}, {1450, 100, 80, 160}};
  assert(shadow_context_group_hints(1920, 1080, 1920, 1080,
      distant_hints).size() == 2);
  return 0;
}
