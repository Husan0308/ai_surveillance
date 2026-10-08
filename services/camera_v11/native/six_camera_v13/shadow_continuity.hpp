#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

// Preserve the approved immediate-frame continuity rule without copying the
// full previous-frame target maps into every deferred publication closure.
inline bool tracker_has_immediate_shadow_continuity(
    int frame_num,
    bool previous_shadow_present,
    int previous_shadow_frame,
    unsigned int previous_shadow_age,
    unsigned int current_shadow_age,
    int previous_active_frame,
    bool previous_active_id_present) {
  const bool previous_inactive_continuity = previous_shadow_present &&
      previous_shadow_frame + 1 == frame_num &&
      previous_shadow_age + 1 == current_shadow_age;
  const bool previous_active_transition = previous_active_frame + 1 == frame_num &&
      previous_active_id_present;
  return previous_inactive_continuity || previous_active_transition;
}

// Geometry contract for the current-frame, shadow-supported context verifier.
// Shadow geometry is only an association hint; its box is never emitted.
struct ShadowContextRect {
  float left = 0;
  float top = 0;
  float width = 0;
  float height = 0;
};

struct ShadowContextCrop {
  int left = 0;
  int top = 0;
  int side = 0;
  float surface_per_frame_x = 0;
  float surface_per_frame_y = 0;
  bool valid = false;
};

inline bool shadow_context_crop_contains(
    const ShadowContextCrop &crop, const ShadowContextRect &rect);

// Keep crop geometry in one place so target grouping and the actual CUDA crop
// use precisely the same aligned source rectangle.
inline ShadowContextCrop shadow_context_compute_crop(
    unsigned int surface_width, unsigned int surface_height,
    unsigned int frame_width, unsigned int frame_height,
    const ShadowContextRect &anchor, unsigned int verifier_side = 640) {
  ShadowContextCrop crop;
  if (!surface_width || !surface_height || !frame_width || !frame_height ||
      !verifier_side || surface_width % 2 || surface_height % 2 ||
      !std::isfinite(anchor.left) || !std::isfinite(anchor.top) ||
      !std::isfinite(anchor.width) || !std::isfinite(anchor.height) ||
      anchor.left < 0 || anchor.top < 0 || anchor.width <= 0 || anchor.height <= 0)
    return crop;
  int side = static_cast<int>(std::nearbyint(surface_width / 3.0));
  side = std::min(side, static_cast<int>(surface_height)) & ~1;
  if (side < static_cast<int>(verifier_side)) return crop;
  const float scale_x = static_cast<float>(surface_width) / frame_width;
  const float scale_y = static_cast<float>(surface_height) / frame_height;
  const float cx = (anchor.left + anchor.width * .5f) * scale_x;
  const float cy = (anchor.top + anchor.height * .5f) * scale_y;
  int left = static_cast<int>(std::nearbyint(cx - side * .5f));
  int top = static_cast<int>(std::nearbyint(cy - side * .5f));
  left = std::max(0, std::min(static_cast<int>(surface_width) - side, left)) & ~1;
  top = std::max(0, std::min(static_cast<int>(surface_height) - side, top)) & ~1;
  crop.left = left;
  crop.top = top;
  crop.side = side;
  crop.surface_per_frame_x = scale_x;
  crop.surface_per_frame_y = scale_y;
  crop.valid = true;
  return crop;
}

// Candidate-centered verifier crop for diagnostic/candidate use. The legacy
// crop above intentionally includes broad scene context. For nearby people,
// that can make a crop detector return the neighbor and fail the location
// check even when the center candidate is real. This zoom crop scales with
// the candidate geometry, keeps a bounded context margin, and never changes
// the bbox used for rendering or the NvDCF input.
inline ShadowContextCrop shadow_context_compute_candidate_crop(
    unsigned int surface_width, unsigned int surface_height,
    unsigned int frame_width, unsigned int frame_height,
    const ShadowContextRect &anchor, unsigned int verifier_side = 640,
    unsigned int minimum_source_side = 256,
    unsigned int maximum_source_side = 640,
    float context_scale = 1.8f) {
  ShadowContextCrop crop;
  if (!surface_width || !surface_height || !frame_width || !frame_height ||
      !verifier_side || minimum_source_side < 2 || maximum_source_side < minimum_source_side ||
      surface_width % 2 || surface_height % 2 ||
      !std::isfinite(anchor.left) || !std::isfinite(anchor.top) ||
      !std::isfinite(anchor.width) || !std::isfinite(anchor.height) ||
      !std::isfinite(context_scale) || context_scale < 1.0f ||
      anchor.left < 0 || anchor.top < 0 || anchor.width <= 0 || anchor.height <= 0)
    return crop;

  const float scale_x = static_cast<float>(surface_width) / frame_width;
  const float scale_y = static_cast<float>(surface_height) / frame_height;
  const float anchor_width = anchor.width * scale_x;
  const float anchor_height = anchor.height * scale_y;
  const float requested = std::ceil(std::max(anchor_width, anchor_height) * context_scale);
  if (!std::isfinite(requested) || requested <= 0) return crop;

  unsigned int side = static_cast<unsigned int>(requested);
  side = std::max(minimum_source_side, std::min(maximum_source_side, side));
  side = std::min(side, std::min(surface_width, surface_height));
  side &= ~1u;
  if (side < 2) return crop;

  const float center_x = (anchor.left + anchor.width * .5f) * scale_x;
  const float center_y = (anchor.top + anchor.height * .5f) * scale_y;
  int left = static_cast<int>(std::nearbyint(center_x - side * .5f));
  int top = static_cast<int>(std::nearbyint(center_y - side * .5f));
  left = std::max(0, std::min(static_cast<int>(surface_width - side), left)) & ~1;
  top = std::max(0, std::min(static_cast<int>(surface_height - side), top)) & ~1;
  crop.left = left;
  crop.top = top;
  crop.side = static_cast<int>(side);
  crop.surface_per_frame_x = scale_x;
  crop.surface_per_frame_y = scale_y;
  crop.valid = true;

  // Never run a verifier on a crop that clipped part of its target. At frame
  // edges the caller may retry with the legacy crop, or fail closed.
  if (!shadow_context_crop_contains(crop, anchor)) return ShadowContextCrop{};
  return crop;
}

inline bool shadow_context_crop_contains(
    const ShadowContextCrop &crop, const ShadowContextRect &rect) {
  if (!crop.valid || !std::isfinite(rect.left) || !std::isfinite(rect.top) ||
      !std::isfinite(rect.width) || !std::isfinite(rect.height) ||
      rect.left < 0 || rect.top < 0 || rect.width <= 0 || rect.height <= 0)
    return false;
  const float left = crop.left / crop.surface_per_frame_x;
  const float top = crop.top / crop.surface_per_frame_y;
  const float right = (crop.left + crop.side) / crop.surface_per_frame_x;
  const float bottom = (crop.top + crop.side) / crop.surface_per_frame_y;
  constexpr float edge_tolerance = 1.0f;
  return rect.left >= left - edge_tolerance && rect.top >= top - edge_tolerance &&
      rect.left + rect.width <= right + edge_tolerance &&
      rect.top + rect.height <= bottom + edge_tolerance;
}

inline std::vector<std::vector<std::size_t>> shadow_context_group_hints(
    unsigned int surface_width, unsigned int surface_height,
    unsigned int frame_width, unsigned int frame_height,
    const std::vector<ShadowContextRect> &hints,
    unsigned int verifier_side = 640) {
  std::vector<std::vector<std::size_t>> groups;
  std::vector<bool> assigned(hints.size(), false);
  for (std::size_t seed = 0; seed < hints.size(); ++seed) {
    if (assigned[seed]) continue;
    const auto crop = shadow_context_compute_crop(surface_width, surface_height,
        frame_width, frame_height, hints[seed], verifier_side);
    if (!crop.valid || !shadow_context_crop_contains(crop, hints[seed])) {
      assigned[seed] = true;
      continue;
    }
    std::vector<std::size_t> group{seed};
    assigned[seed] = true;
    for (std::size_t candidate = seed + 1; candidate < hints.size(); ++candidate) {
      if (!assigned[candidate] && shadow_context_crop_contains(crop, hints[candidate])) {
        group.push_back(candidate);
        assigned[candidate] = true;
      }
    }
    groups.push_back(std::move(group));
  }
  return groups;
}

struct ShadowContextPair {
  std::size_t target_index = 0;
  std::size_t proposal_index = 0;
  float iou = 0;
};

inline bool shadow_context_spatial_match(const ShadowContextRect &hint,
    const ShadowContextRect &proposal, float *iou_out, float *distance_out);

// Fail closed unless a target/proposal pair is each other's unique best
// spatial match. This enforces one-to-one association for grouped crops and
// prevents a single context detection from rescuing two nearby tracks.
inline std::vector<ShadowContextPair> shadow_context_match_one_to_one(
    const std::vector<ShadowContextRect> &targets,
    const std::vector<ShadowContextRect> &proposals,
    float uniqueness_margin = .10f) {
  const float invalid = -1.0f;
  std::vector<std::vector<float>> scores(
      targets.size(), std::vector<float>(proposals.size(), invalid));
  for (std::size_t ti = 0; ti < targets.size(); ++ti)
    for (std::size_t pi = 0; pi < proposals.size(); ++pi) {
      float score = 0.0f;
      if (shadow_context_spatial_match(targets[ti], proposals[pi], &score, nullptr))
        scores[ti][pi] = score;
    }

  std::vector<int> target_best(targets.size(), -1);
  for (std::size_t ti = 0; ti < targets.size(); ++ti) {
    float best = invalid, second = invalid;
    for (std::size_t pi = 0; pi < proposals.size(); ++pi) {
      const float score = scores[ti][pi];
      if (score < 0) continue;
      if (score > best) { second = best; best = score; target_best[ti] = static_cast<int>(pi); }
      else if (score > second) second = score;
    }
    if (target_best[ti] >= 0 && second >= 0 && best - second < uniqueness_margin)
      target_best[ti] = -1;
  }

  std::vector<int> proposal_best(proposals.size(), -1);
  for (std::size_t pi = 0; pi < proposals.size(); ++pi) {
    float best = invalid, second = invalid;
    for (std::size_t ti = 0; ti < targets.size(); ++ti) {
      const float score = scores[ti][pi];
      if (score < 0) continue;
      if (score > best) { second = best; best = score; proposal_best[pi] = static_cast<int>(ti); }
      else if (score > second) second = score;
    }
    if (proposal_best[pi] >= 0 && second >= 0 && best - second < uniqueness_margin)
      proposal_best[pi] = -1;
  }

  std::vector<ShadowContextPair> result;
  for (std::size_t ti = 0; ti < targets.size(); ++ti) {
    const int pi = target_best[ti];
    if (pi < 0 || proposal_best[static_cast<std::size_t>(pi)] != static_cast<int>(ti)) continue;
    result.push_back({ti, static_cast<std::size_t>(pi),
        scores[ti][static_cast<std::size_t>(pi)]});
  }
  return result;
}

// A context crop may be prepared before NvDCF processes the current frame,
// but only from a target observation on the immediately preceding frame.
// This is scheduling evidence, not publication authorization: the current
// frame must still produce the same INACTIVE native target and a current
// verifier bbox that passes the existing strict association checks.
inline bool shadow_context_pretracker_hint_eligible(
    int current_frame, int prior_frame, unsigned long long native_id,
    bool auxiliary_target, bool appearance_verified,
    bool prior_active, bool prior_inactive_shadow,
    bool current_detector_match, const ShadowContextRect &hint,
    unsigned int frame_width, unsigned int frame_height) {
  const bool valid_geometry = std::isfinite(hint.left) && std::isfinite(hint.top) &&
      std::isfinite(hint.width) && std::isfinite(hint.height) &&
      hint.left >= 0.0f && hint.top >= 0.0f && hint.width > 0.0f &&
      hint.height > 0.0f && frame_width > 0 && frame_height > 0 &&
      hint.left + hint.width <= static_cast<float>(frame_width) + 0.1f &&
      hint.top + hint.height <= static_cast<float>(frame_height) + 0.1f;
  return current_frame > 0 && prior_frame == current_frame - 1 &&
      native_id != 0 && auxiliary_target && appearance_verified &&
      (prior_active || prior_inactive_shadow) && !current_detector_match &&
      valid_geometry;
}

inline bool shadow_context_has_recent_primary_support(
    int current_frame, int last_primary_frame, unsigned int max_age_frames) {
  return current_frame >= 0 && last_primary_frame >= 0 &&
      last_primary_frame <= current_frame &&
      static_cast<unsigned int>(current_frame - last_primary_frame) <= max_age_frames;
}

struct ShadowContextMatch {
  bool accepted = false;
  std::size_t proposal_index = 0;
  float iou = 0;
  float runner_up_iou = 0;
  float center_distance = 0;
};

inline float shadow_context_iou(const ShadowContextRect &a,
                                const ShadowContextRect &b) {
  const float x1 = std::max(a.left, b.left);
  const float y1 = std::max(a.top, b.top);
  const float x2 = std::min(a.left + a.width, b.left + b.width);
  const float y2 = std::min(a.top + a.height, b.top + b.height);
  const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  const float union_area = a.width * a.height + b.width * b.height - intersection;
  return union_area > 0.0f ? intersection / union_area : 0.0f;
}

inline float shadow_context_center_distance(const ShadowContextRect &a,
                                             const ShadowContextRect &b) {
  return std::hypot(a.left + a.width * 0.5f - b.left - b.width * 0.5f,
                    a.top + a.height * 0.5f - b.top - b.height * 0.5f);
}

inline bool shadow_context_spatial_match(const ShadowContextRect &hint,
                                          const ShadowContextRect &proposal,
                                          float *iou_out = nullptr,
                                          float *distance_out = nullptr) {
  const float overlap = shadow_context_iou(hint, proposal);
  const float distance = shadow_context_center_distance(hint, proposal);
  const float width_ratio = std::max(hint.width, proposal.width) /
      std::max(1.0f, std::min(hint.width, proposal.width));
  const float height_ratio = std::max(hint.height, proposal.height) /
      std::max(1.0f, std::min(hint.height, proposal.height));
  if (iou_out) *iou_out = overlap;
  if (distance_out) *distance_out = distance;
  return std::isfinite(overlap) && std::isfinite(distance) &&
      std::isfinite(width_ratio) && std::isfinite(height_ratio) &&
      overlap >= 0.30f && distance <= 50.0f &&
      width_ratio <= 2.0f && height_ratio <= 2.0f;
}

inline ShadowContextMatch shadow_context_select_unique(
    const ShadowContextRect &hint,
    const std::vector<ShadowContextRect> &proposals) {
  ShadowContextMatch result;
  std::size_t matches = 0;
  for (std::size_t i = 0; i < proposals.size(); ++i) {
    float overlap = 0.0f, distance = 0.0f;
    if (!shadow_context_spatial_match(hint, proposals[i], &overlap, &distance)) continue;
    ++matches;
    if (overlap > result.iou) {
      result.runner_up_iou = result.iou;
      result.iou = overlap;
      result.center_distance = distance;
      result.proposal_index = i;
    } else if (overlap > result.runner_up_iou) {
      result.runner_up_iou = overlap;
    }
  }
  result.accepted = matches == 1 ||
      (matches > 1 && result.iou - result.runner_up_iou >= 0.10f);
  return result;
}

struct ShadowContextTransform {
  float crop_x = 0;
  float crop_y = 0;
  float crop_side = 0;
  float verifier_side = 640;
  float surface_per_frame_x = 1;
  float surface_per_frame_y = 1;
};

inline std::array<float, 4> shadow_context_to_frame(
    const ShadowContextRect &proposal,
    const ShadowContextTransform &transform) {
  const float local_scale = transform.verifier_side / transform.crop_side;
  return {{(proposal.left / local_scale + transform.crop_x) /
               transform.surface_per_frame_x,
           (proposal.top / local_scale + transform.crop_y) /
               transform.surface_per_frame_y,
           proposal.width / local_scale / transform.surface_per_frame_x,
           proposal.height / local_scale / transform.surface_per_frame_y}};
}
