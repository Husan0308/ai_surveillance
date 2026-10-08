#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

// DeepStream's per-GIE NMS can leave two boxes for one person when the IoU is
// just below its configured cutoff. This second, conservative same-GIE check
// only removes a lower-confidence box when it is almost wholly contained by a
// unique, higher-confidence box with a nearby center. It deliberately leaves
// cross-GIE arbitration to the separately reviewed auxiliary/primary policy.
struct PersonProposalBox {
  unsigned component_id = 0;
  float confidence = 0;
  float left = 0, top = 0, width = 0, height = 0;
};

struct PersonProposalDuplicateMetrics {
  float iou = 0;
  float containment = 0;
  float center_over_max_height = 0;
  float width_ratio = 0;
  float height_ratio = 0;
};

struct PersonProposalSuppression {
  std::size_t suppressed_index = 0;
  std::size_t retained_index = 0;
  PersonProposalDuplicateMetrics metrics;
};

inline bool person_proposal_duplicate_metrics(
    const PersonProposalBox &a, const PersonProposalBox &b,
    PersonProposalDuplicateMetrics *metrics = nullptr) {
  if (a.component_id == 0 || a.component_id != b.component_id ||
      !std::isfinite(a.confidence) || !std::isfinite(b.confidence) ||
      !std::isfinite(a.left) || !std::isfinite(a.top) ||
      !std::isfinite(a.width) || !std::isfinite(a.height) ||
      !std::isfinite(b.left) || !std::isfinite(b.top) ||
      !std::isfinite(b.width) || !std::isfinite(b.height) ||
      a.width <= 0 || a.height <= 0 || b.width <= 0 || b.height <= 0)
    return false;

  const float x1 = std::max(a.left, b.left);
  const float y1 = std::max(a.top, b.top);
  const float x2 = std::min(a.left + a.width, b.left + b.width);
  const float y2 = std::min(a.top + a.height, b.top + b.height);
  const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  const float area_a = a.width * a.height;
  const float area_b = b.width * b.height;
  const float union_area = area_a + area_b - intersection;
  const float smaller_area = std::min(area_a, area_b);
  const float max_height = std::max(a.height, b.height);
  if (union_area <= 0 || smaller_area <= 0 || max_height <= 0) return false;

  PersonProposalDuplicateMetrics value;
  value.iou = intersection / union_area;
  value.containment = intersection / smaller_area;
  value.center_over_max_height = std::hypot(
      a.left + a.width / 2 - b.left - b.width / 2,
      a.top + a.height / 2 - b.top - b.height / 2) / max_height;
  value.width_ratio = std::max(a.width, b.width) / std::min(a.width, b.width);
  value.height_ratio = std::max(a.height, b.height) / std::min(a.height, b.height);
  if (metrics) *metrics = value;

  return value.iou >= 0.30f && value.containment >= 0.95f &&
      value.center_over_max_height <= 0.25f &&
      value.width_ratio <= 2.0f && value.height_ratio <= 2.0f &&
      std::fabs(a.confidence - b.confidence) >= 0.05f;
}

// Return indices to withhold. Confidence determines the retained proposal.
// A candidate is suppressed only when exactly one already-retained,
// same-component proposal satisfies the strict nested-duplicate policy;
// ambiguous multi-match configurations fail open.
inline std::vector<PersonProposalSuppression> person_proposal_duplicates(
    const std::vector<PersonProposalBox> &proposals) {
  std::vector<std::size_t> order(proposals.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return proposals[a].confidence > proposals[b].confidence;
  });

  std::vector<std::size_t> retained;
  std::vector<PersonProposalSuppression> suppressed;
  for (const std::size_t candidate : order) {
    std::vector<std::pair<std::size_t, PersonProposalDuplicateMetrics>> matches;
    for (const std::size_t prior : retained) {
      PersonProposalDuplicateMetrics metrics;
      if (person_proposal_duplicate_metrics(proposals[prior], proposals[candidate], &metrics))
        matches.emplace_back(prior, metrics);
    }
    if (matches.size() == 1) {
      suppressed.push_back({candidate, matches.front().first, matches.front().second});
    } else {
      retained.push_back(candidate);
    }
  }
  return suppressed;
}
