#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

struct TrackerOutputBox {
  std::uint64_t id = 0;
  std::uint32_t frame_num = 0;
  float left = 0, top = 0, width = 0, height = 0;
};

struct TrackerOutputSuppression {
  std::uint64_t id = 0;
  std::size_t suppressed_index = 0;
  std::size_t retained_index = 0;
  float suppressed_previous_iou = 0;
  float retained_previous_iou = 0;
};

struct TrackerOutputDedupeResult {
  std::vector<std::size_t> keep_indices;
  std::vector<TrackerOutputSuppression> suppressions;
  std::vector<std::uint64_t> withheld_ambiguous_ids;
};

inline float tracker_output_iou(const TrackerOutputBox &a, const TrackerOutputBox &b) {
  if (!std::isfinite(a.left) || !std::isfinite(a.top) ||
      !std::isfinite(a.width) || !std::isfinite(a.height) ||
      !std::isfinite(b.left) || !std::isfinite(b.top) ||
      !std::isfinite(b.width) || !std::isfinite(b.height) ||
      a.width <= 0 || a.height <= 0 || b.width <= 0 || b.height <= 0)
    return 0;
  const float x1 = std::max(a.left, b.left);
  const float y1 = std::max(a.top, b.top);
  const float x2 = std::min(a.left + a.width, b.left + b.width);
  const float y2 = std::min(a.top + a.height, b.top + b.height);
  const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  const float union_area = a.width * a.height + b.width * b.height - intersection;
  return union_area > 0 ? intersection / union_area : 0;
}

// NvDCF output should contain one row per native ID per frame. If a malformed
// frame contains the same ID more than once, retain a row only when exactly one
// candidate has strong, immediate-frame geometric continuity with that ID.
// Ambiguous groups fail closed. This does not merge IDs or infer a replacement
// identity; distinct IDs are always retained independently.
inline TrackerOutputDedupeResult dedupe_tracker_outputs_by_id_continuity(
    const std::vector<TrackerOutputBox> &current,
    const std::vector<TrackerOutputBox> &previous) {
  TrackerOutputDedupeResult result;
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> indices_by_id;
  for (std::size_t i = 0; i < current.size(); ++i)
    indices_by_id[current[i].id].push_back(i);

  std::vector<bool> retain(current.size(), false);
  for (const auto &entry : indices_by_id) {
    const auto &indices = entry.second;
    if (indices.size() == 1) {
      retain[indices.front()] = true;
      continue;
    }

    const auto prior = std::find_if(previous.begin(), previous.end(), [&](const auto &box) {
      return box.id == entry.first && !current.empty() &&
          box.frame_num + 1 == current[indices.front()].frame_num;
    });
    const auto prior_count = std::count_if(previous.begin(), previous.end(), [&](const auto &box) {
      return box.id == entry.first && !current.empty() &&
          box.frame_num + 1 == current[indices.front()].frame_num;
    });
    if (prior == previous.end() || prior_count != 1) {
      result.withheld_ambiguous_ids.push_back(entry.first);
      continue;
    }

    std::size_t best_index = indices.front();
    float best_iou = -1.0f;
    float second_iou = -1.0f;
    for (const auto index : indices) {
      const float score = tracker_output_iou(current[index], *prior);
      if (score > best_iou) {
        second_iou = best_iou;
        best_iou = score;
        best_index = index;
      } else if (score > second_iou) {
        second_iou = score;
      }
    }

    // Require one near-current continuation and no plausible competing match.
    // The wide separation is deliberately conservative: when NvDCF's duplicate
    // rows cannot be resolved from the immediately previous emitted target,
    // no row for that duplicated ID is published.
    if (best_iou < 0.50f || second_iou > 0.10f) {
      result.withheld_ambiguous_ids.push_back(entry.first);
      continue;
    }
    retain[best_index] = true;
    for (const auto index : indices) {
      if (index == best_index) continue;
      result.suppressions.push_back({entry.first, index, best_index,
          tracker_output_iou(current[index], *prior), best_iou});
    }
  }

  for (std::size_t i = 0; i < retain.size(); ++i)
    if (retain[i]) result.keep_indices.push_back(i);
  std::sort(result.withheld_ambiguous_ids.begin(), result.withheld_ambiguous_ids.end());
  result.withheld_ambiguous_ids.erase(
      std::unique(result.withheld_ambiguous_ids.begin(), result.withheld_ambiguous_ids.end()),
      result.withheld_ambiguous_ids.end());
  return result;
}
