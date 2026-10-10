#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// DeepStream's per-GIE NMS can leave nested boxes for one person when their
// IoU is below its configured cutoff. This second conservative check removes
// only a unique, same-component proposal pair with strong nested geometry.
struct PersonProposalBox {
  unsigned component_id = 0;
  float confidence = 0;
  float left = 0, top = 0, width = 0, height = 0;
};

struct PersonProposalDuplicateMetrics {
  float iou = 0;
  float containment = 0;
  float center_over_max_height = 0;
  float center_over_max_width = 0;
  float bottom_edge_delta_over_max_height = 0;
  float width_ratio = 0;
  float height_ratio = 0;
};

struct PersonProposalSuppression {
  std::size_t suppressed_index = 0;
  std::size_t retained_index = 0;
  PersonProposalDuplicateMetrics metrics;
  bool retained_tighter_geometry = false;
};

inline bool person_proposal_geometry_metrics(
    const PersonProposalBox &a, const PersonProposalBox &b,
    PersonProposalDuplicateMetrics *metrics = nullptr) {
  if (!std::isfinite(a.confidence) || !std::isfinite(b.confidence) ||
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
  if (!std::isfinite(area_a) || !std::isfinite(area_b) ||
      !std::isfinite(union_area) || union_area <= 0 || smaller_area <= 0 ||
      max_height <= 0) return false;

  PersonProposalDuplicateMetrics value;
  value.iou = intersection / union_area;
  value.containment = intersection / smaller_area;
  value.center_over_max_height = std::hypot(
      a.left + a.width / 2 - b.left - b.width / 2,
      a.top + a.height / 2 - b.top - b.height / 2) / max_height;
  value.center_over_max_width = std::fabs(
      a.left + a.width / 2 - b.left - b.width / 2) /
      std::max(a.width, b.width);
  value.bottom_edge_delta_over_max_height = std::fabs(
      a.top + a.height - b.top - b.height) / max_height;
  value.width_ratio = std::max(a.width, b.width) / std::min(a.width, b.width);
  value.height_ratio = std::max(a.height, b.height) / std::min(a.height, b.height);
  if (metrics) *metrics = value;

  return true;
}

inline bool person_proposal_duplicate_metrics(
    const PersonProposalBox &a, const PersonProposalBox &b,
    PersonProposalDuplicateMetrics *metrics = nullptr) {
  if (a.component_id == 0 || a.component_id != b.component_id) return false;
  PersonProposalDuplicateMetrics value;
  if (!person_proposal_geometry_metrics(a, b, &value)) return false;
  if (metrics) *metrics = value;
  const float area_a = a.width * a.height;
  const float area_b = b.width * b.height;
  const float smaller_area = std::min(area_a, area_b);
  const float area_ratio = std::max(area_a, area_b) / smaller_area;
  const bool established_nested_gate = value.containment >= 0.95f &&
      value.center_over_max_height <= 0.25f;
  // Some people produce a tight box and a much looser box that contains it
  // almost completely. Permit a small center offset only in that much stricter
  // containment regime; ordinary overlapping people still fail the gate.
  const bool strict_nested_extension = value.containment >= 0.99f &&
      value.center_over_max_height <= 0.28f && area_ratio >= 1.25f;
  const bool same_person_nested_extension = value.containment >= 0.99f &&
      value.center_over_max_height <= 0.35f &&
      value.center_over_max_width <= 0.30f &&
      value.bottom_edge_delta_over_max_height <= 0.04f &&
      value.width_ratio <= 2.40f && value.height_ratio <= 2.20f &&
      area_ratio >= 1.25f && area_ratio <= 6.0f &&
      std::fabs(a.confidence - b.confidence) >= 0.05f;
  return (value.iou >= 0.30f &&
          (established_nested_gate || strict_nested_extension) &&
          value.width_ratio <= 2.0f && value.height_ratio <= 2.0f &&
          std::fabs(a.confidence - b.confidence) >= 0.05f) ||
      same_person_nested_extension;
}

// Return indices to withhold. Confidence normally determines the retained
// proposal. For the narrow strict-nesting extension only, when confidence
// differs by less than 0.10, prefer the tighter box because the wider box is
// demonstrably a loose duplicate. A candidate is suppressed only when exactly
// one already-retained same-component proposal matches; ambiguity fails open.
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
      const std::size_t prior = matches.front().first;
      const auto &metrics = matches.front().second;
      const auto &candidate_box = proposals[candidate];
      const auto &prior_box = proposals[prior];
      const float candidate_area = candidate_box.width * candidate_box.height;
      const float prior_area = prior_box.width * prior_box.height;
      const bool strict_extension = metrics.center_over_max_height > 0.25f &&
          metrics.containment >= 0.99f &&
          std::fabs(candidate_box.confidence - prior_box.confidence) < 0.10f;
      const bool keep_candidate_tight = strict_extension && candidate_area < prior_area;
      if (keep_candidate_tight) {
        suppressed.push_back({prior, candidate, metrics, true});
        retained.erase(std::remove(retained.begin(), retained.end(), prior), retained.end());
        retained.push_back(candidate);
      } else {
        suppressed.push_back({candidate, prior, metrics, false});
      }
    } else {
      retained.push_back(candidate);
    }
  }
  return suppressed;
}

// An exact previous-frame NvDCF target can disambiguate two otherwise
// overlapping proposals that represent the same person. This check is applied
// only when both proposals independently associate to exactly one identical
// target, their current boxes are tightly aligned, and one proposal has a
// materially stronger match to that target's immediately preceding geometry.
// It deliberately fails open on missing/stale history, multiple proposals for
// one target, ambiguous target association, or a small score margin.
struct PersonProposalHistoryTarget {
  std::uint64_t id = 0;
  std::uint32_t frame_num = 0;
  float left = 0, top = 0, width = 0, height = 0;
};

struct PersonProposalHistorySuppression {
  std::size_t suppressed_index = 0;
  std::size_t retained_index = 0;
  std::uint64_t target_id = 0;
  PersonProposalDuplicateMetrics pair_metrics;
  float suppressed_history_iou = 0;
  float retained_history_iou = 0;
};

inline float person_proposal_iou_xywh(float al, float at, float aw, float ah,
                                      float bl, float bt, float bw, float bh) {
  if (!std::isfinite(al) || !std::isfinite(at) || !std::isfinite(aw) ||
      !std::isfinite(ah) || !std::isfinite(bl) || !std::isfinite(bt) ||
      !std::isfinite(bw) || !std::isfinite(bh) || aw <= 0 || ah <= 0 ||
      bw <= 0 || bh <= 0) return 0;
  const float x1 = std::max(al, bl), y1 = std::max(at, bt);
  const float x2 = std::min(al + aw, bl + bw), y2 = std::min(at + ah, bt + bh);
  const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  const float union_area = aw * ah + bw * bh - intersection;
  return union_area > 0 ? intersection / union_area : 0;
}

inline std::vector<PersonProposalHistorySuppression>
person_proposal_duplicates_supported_by_previous_tracks(
    const std::vector<PersonProposalBox> &proposals,
    const std::vector<PersonProposalHistoryTarget> &previous_targets,
    std::uint32_t current_frame) {
  struct Match {
    std::size_t proposal_index = 0;
    std::uint64_t target_id = 0;
    float history_iou = 0;
  };
  std::unordered_map<std::uint64_t, std::vector<Match>> matches_by_target;
  std::unordered_set<std::uint64_t> ambiguous_targets;
  if (current_frame == 0) return {};
  for (std::size_t pi = 0; pi < proposals.size(); ++pi) {
    const auto &proposal = proposals[pi];
    if (!std::isfinite(proposal.confidence) || proposal.confidence < 0.25f ||
        proposal.confidence > 1.0f) continue;
    std::vector<Match> matches;
    for (const auto &target : previous_targets) {
      if (target.frame_num != current_frame - 1) continue;
      PersonProposalBox target_box{proposal.component_id, 1.0f,
          target.left, target.top, target.width, target.height};
      PersonProposalDuplicateMetrics metrics;
      if (!person_proposal_geometry_metrics(proposal, target_box, &metrics)) continue;
      const bool history_match = metrics.iou >= 0.30f ||
          (metrics.center_over_max_height <= 0.35f &&
           metrics.center_over_max_width <= 0.30f &&
           metrics.width_ratio <= 2.0f && metrics.height_ratio <= 3.20f);
      if (history_match)
        matches.push_back({pi, target.id, metrics.iou});
    }
    if (matches.size() == 1)
      matches_by_target[matches.front().target_id].push_back(matches.front());
    else if (matches.size() > 1)
      for (const auto &match : matches)
        ambiguous_targets.insert(match.target_id);
  }

  std::vector<PersonProposalHistorySuppression> result;
  for (const auto &entry : matches_by_target) {
    if (ambiguous_targets.count(entry.first)) continue;
    const auto &target_matches = entry.second;
    // More than two detections around the same prior target is ambiguous; do
    // not guess which one belongs to another nearby or occluded person.
    if (target_matches.size() != 2) continue;
    const auto &first = target_matches[0];
    const auto &second = target_matches[1];
    PersonProposalDuplicateMetrics pair_metrics;
    if (!person_proposal_geometry_metrics(proposals[first.proposal_index],
            proposals[second.proposal_index], &pair_metrics)) continue;
    const bool same_current_target = pair_metrics.iou >= 0.25f &&
        pair_metrics.containment >= 0.75f &&
        pair_metrics.center_over_max_height <= 0.35f &&
        pair_metrics.center_over_max_width <= 0.22f &&
        pair_metrics.width_ratio <= 2.20f && pair_metrics.height_ratio <= 3.20f;
    if (!same_current_target ||
        std::fabs(first.history_iou - second.history_iou) < 0.15f) continue;
    const bool first_retained = first.history_iou > second.history_iou;
    result.push_back({
        first_retained ? second.proposal_index : first.proposal_index,
        first_retained ? first.proposal_index : second.proposal_index,
        entry.first, pair_metrics,
        first_retained ? second.history_iou : first.history_iou,
        first_retained ? first.history_iou : second.history_iou});
  }
  std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
    return a.suppressed_index < b.suppressed_index;
  });
  return result;
}
