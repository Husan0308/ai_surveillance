#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Conservative canonical-only resolution for a new NvDCF ID that competes
// with an established ID for the same exact-frame context-detector instance.
// The helper never merges IDs or changes PGIE/NvDCF input. Missing, stale,
// ambiguous, or geometrically inconsistent evidence fails open.
struct CrossIdRect {
  float left = 0, top = 0, width = 0, height = 0;
};

struct CrossIdTrack {
  std::uint64_t id = 0;
  CrossIdRect box;
  bool active_previous_frame = false;
};

struct CrossIdContextProposal {
  float confidence = 0;
  CrossIdRect box;
};

struct CrossIdDuplicateDecision {
  bool suppress_candidate = false;
  std::uint64_t candidate_id = 0;
  std::uint64_t owner_id = 0;
  std::size_t proposal_index = 0;
  float owner_proposal_iou = 0;
  float candidate_proposal_iou = 0;
  float proposal_iou_margin = 0;
  float candidate_owner_area_ratio = 0;
  const char *reason = "NO_UNAMBIGUOUS_CONTEXT_EXCLUSIVITY";
};

struct CrossIdBirthContext {
  std::uint64_t candidate_id = 0;
  bool verifier_confirmed = false;
  std::vector<CrossIdContextProposal> proposals;
};

inline bool cross_id_valid_rect(const CrossIdRect &r, float frame_width,
                                float frame_height) {
  return std::isfinite(r.left) && std::isfinite(r.top) &&
      std::isfinite(r.width) && std::isfinite(r.height) &&
      r.left >= 0 && r.top >= 0 && r.width > 0 && r.height > 0 &&
      r.left + r.width <= frame_width + 0.1f &&
      r.top + r.height <= frame_height + 0.1f;
}

inline float cross_id_iou(const CrossIdRect &a, const CrossIdRect &b) {
  if (!std::isfinite(a.left) || !std::isfinite(a.top) ||
      !std::isfinite(a.width) || !std::isfinite(a.height) ||
      !std::isfinite(b.left) || !std::isfinite(b.top) ||
      !std::isfinite(b.width) || !std::isfinite(b.height) ||
      a.width <= 0 || a.height <= 0 || b.width <= 0 || b.height <= 0) return 0;
  const float x1 = std::max(a.left, b.left), y1 = std::max(a.top, b.top);
  const float x2 = std::min(a.left + a.width, b.left + b.width);
  const float y2 = std::min(a.top + a.height, b.top + b.height);
  const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  const float union_area = a.width * a.height + b.width * b.height - intersection;
  return union_area > 0 && std::isfinite(union_area) ? intersection / union_area : 0;
}

inline bool cross_id_center_inside(const CrossIdRect &inner,
                                   const CrossIdRect &outer) {
  const float x = inner.left + inner.width * 0.5f;
  const float y = inner.top + inner.height * 0.5f;
  return x >= outer.left && x <= outer.left + outer.width &&
      y >= outer.top && y <= outer.top + outer.height;
}

// Admission gates deliberately target strong asymmetry in a single context
// crop, not raw track-box IoU. IoU alone overlaps the CAM-02 duplicate and the
// distinct CAM-05 people, so it is never sufficient evidence here.
inline CrossIdDuplicateDecision cross_id_context_duplicate_birth(
    std::uint32_t source_id, std::uint64_t current_epoch,
    std::uint64_t result_epoch, std::int32_t current_frame,
    std::int32_t result_frame, std::uint64_t current_pts,
    std::uint64_t result_pts, bool exact_previous_frame_available,
    bool verifier_confirmed, std::uint64_t candidate_id,
    const std::vector<CrossIdTrack> &tracks,
    const std::vector<CrossIdContextProposal> &proposals,
    float frame_width, float frame_height) {
  CrossIdDuplicateDecision result;
  result.candidate_id = candidate_id;
  if (source_id >= 6 || current_epoch != result_epoch || current_frame <= 0 ||
      current_frame != result_frame || current_pts != result_pts ||
      !exact_previous_frame_available || !verifier_confirmed ||
      !std::isfinite(frame_width) || !std::isfinite(frame_height) ||
      frame_width <= 0 || frame_height <= 0) {
    result.reason = "STALE_OR_INCOMPLETE_EXACT_FRAME_EVIDENCE";
    return result;
  }
  std::size_t candidate_count = 0;
  const CrossIdTrack *candidate = nullptr;
  std::unordered_set<std::uint64_t> ids;
  for (const auto &track : tracks) {
    if (!ids.insert(track.id).second ||
        !cross_id_valid_rect(track.box, frame_width, frame_height)) {
      result.reason = "INVALID_OR_DUPLICATE_CURRENT_TRACK_METADATA";
      return result;
    }
    if (track.id == candidate_id) {
      ++candidate_count;
      candidate = &track;
    }
  }
  if (candidate_count != 1 || !candidate || candidate->active_previous_frame) {
    result.reason = "CANDIDATE_NOT_A_UNIQUE_NEW_ACTIVE_BIRTH";
    return result;
  }
  if (proposals.empty()) {
    result.reason = "NO_CONTEXT_INSTANCES";
    return result;
  }
  for (const auto &proposal : proposals) {
    if (!std::isfinite(proposal.confidence) || proposal.confidence < 0.20f ||
        proposal.confidence > 1.0f ||
        !cross_id_valid_rect(proposal.box, frame_width, frame_height)) {
      result.reason = "INVALID_CONTEXT_PROPOSAL";
      return result;
    }
  }

  std::vector<CrossIdDuplicateDecision> eligible;
  for (const auto &owner : tracks) {
    if (owner.id == candidate_id || !owner.active_previous_frame) continue;
    if (!cross_id_center_inside(candidate->box, owner.box)) continue;
    const float owner_area = owner.box.width * owner.box.height;
    const float candidate_area = candidate->box.width * candidate->box.height;
    if (!std::isfinite(owner_area) || !std::isfinite(candidate_area) || owner_area <= 0)
      continue;
    const float area_ratio = candidate_area / owner_area;
    if (area_ratio <= 0 || area_ratio > 0.60f) continue;

    std::vector<float> owner_scores, candidate_scores;
    owner_scores.reserve(proposals.size());
    candidate_scores.reserve(proposals.size());
    for (const auto &proposal : proposals) {
      owner_scores.push_back(cross_id_iou(owner.box, proposal.box));
      candidate_scores.push_back(cross_id_iou(candidate->box, proposal.box));
    }
    const auto owner_best = static_cast<std::size_t>(std::distance(
        owner_scores.begin(), std::max_element(owner_scores.begin(), owner_scores.end())));
    const auto candidate_best = static_cast<std::size_t>(std::distance(
        candidate_scores.begin(), std::max_element(candidate_scores.begin(), candidate_scores.end())));
    // A simple indexed second-best scan avoids relying on a score sort whose
    // ties might choose an arbitrary proposal.
    float owner_runner_up = 0.0f;
    float candidate_runner_up = 0.0f;
    for (std::size_t pi = 0; pi < owner_scores.size(); ++pi)
      if (pi != owner_best) owner_runner_up = std::max(owner_runner_up, owner_scores[pi]);
    for (std::size_t pi = 0; pi < candidate_scores.size(); ++pi)
      if (pi != candidate_best) candidate_runner_up =
          std::max(candidate_runner_up, candidate_scores[pi]);
    const float owner_iou = owner_scores[owner_best];
    const float candidate_iou = candidate_scores[owner_best];
    if (owner_best != candidate_best || owner_iou < 0.65f ||
        owner_iou - owner_runner_up < 0.10f ||
        candidate_scores[candidate_best] - candidate_runner_up < 0.10f ||
        candidate_iou < 0.25f || candidate_iou > 0.40f ||
        owner_iou - candidate_iou < 0.28f) continue;

    // The context detector must also resolve at least one other previously
    // active person to a different proposal. A single broad box around two
    // nearby people is not enough to authorize suppression.
    bool independent_person_resolved = false;
    for (std::size_t pi = 0; pi < proposals.size(); ++pi) {
      if (pi == owner_best) continue;
      for (const auto &other : tracks) {
        if (other.id == candidate_id || other.id == owner.id ||
            !other.active_previous_frame) continue;
        const float other_iou = cross_id_iou(other.box, proposals[pi].box);
        float other_runner_up = 0.0f;
        for (std::size_t qi = 0; qi < proposals.size(); ++qi)
          if (qi != pi) other_runner_up = std::max(other_runner_up,
              cross_id_iou(other.box, proposals[qi].box));
        if (other_iou >= 0.65f && other_iou - other_runner_up >= 0.10f) {
          independent_person_resolved = true;
          break;
        }
      }
      if (independent_person_resolved) break;
    }
    if (!independent_person_resolved) continue;

    bool owner_is_unique_track = true;
    for (const auto &other : tracks) {
      if (other.id == owner.id || other.id == candidate_id) continue;
      if (cross_id_iou(other.box, proposals[owner_best].box) >= owner_iou - 0.10f) {
        owner_is_unique_track = false;
        break;
      }
    }
    if (!owner_is_unique_track) continue;

    CrossIdDuplicateDecision decision;
    decision.suppress_candidate = true;
    decision.candidate_id = candidate_id;
    decision.owner_id = owner.id;
    decision.proposal_index = owner_best;
    decision.owner_proposal_iou = owner_iou;
    decision.candidate_proposal_iou = candidate_iou;
    decision.proposal_iou_margin = owner_iou - candidate_iou;
    decision.candidate_owner_area_ratio = area_ratio;
    decision.reason = "UNIQUE_ESTABLISHED_TRACK_OWNS_CONTEXT_INSTANCE";
    eligible.push_back(decision);
  }
  if (eligible.size() == 1) return eligible.front();
  result.reason = eligible.empty() ? "NO_STRONG_CONTEXT_EXCLUSIVITY" :
      "MULTIPLE_PLAUSIBLE_OWNERS";
  return result;
}

// Evaluate all births from a frame together. Per-birth context evidence may
// independently point at the same established target; suppressing whichever
// candidate happens to be iterated first would make publication order decide
// identity. If multiple births claim one owner, preserve all and report the
// ambiguity instead.
inline std::vector<CrossIdDuplicateDecision> cross_id_context_duplicate_birth_batch(
    std::uint32_t source_id, std::uint64_t current_epoch,
    std::uint64_t result_epoch, std::int32_t current_frame,
    std::int32_t result_frame, std::uint64_t current_pts,
    std::uint64_t result_pts, bool exact_previous_frame_available,
    const std::vector<CrossIdTrack> &tracks,
    const std::vector<CrossIdBirthContext> &births,
    float frame_width, float frame_height) {
  std::vector<CrossIdDuplicateDecision> decisions;
  decisions.reserve(births.size());
  for (const auto &birth : births) {
    decisions.push_back(cross_id_context_duplicate_birth(
        source_id, current_epoch, result_epoch, current_frame, result_frame,
        current_pts, result_pts, exact_previous_frame_available,
        birth.verifier_confirmed, birth.candidate_id, tracks, birth.proposals,
        frame_width, frame_height));
  }
  std::unordered_map<std::uint64_t, std::size_t> owner_claims;
  for (const auto &decision : decisions)
    if (decision.suppress_candidate) ++owner_claims[decision.owner_id];
  for (auto &decision : decisions) {
    const auto owner = owner_claims.find(decision.owner_id);
    if (decision.suppress_candidate && owner != owner_claims.end() && owner->second > 1) {
      decision.suppress_candidate = false;
      decision.reason = "AMBIGUOUS_MULTIPLE_BIRTHS_CLAIM_OWNER";
    }
  }
  return decisions;
}

class CrossIdDuplicateLedger {
 public:
  static constexpr std::size_t kSourceCount = 6;
  struct Result {
    std::vector<std::uint64_t> suppressed_ids;
    std::vector<std::uint64_t> released_ids;
  };

  bool link(std::uint32_t source_id, std::uint64_t epoch, std::int32_t frame,
            const CrossIdTrack &candidate, const CrossIdTrack &owner) {
    if (source_id >= kSourceCount || frame < 1 || candidate.id == owner.id ||
        !cross_id_valid_rect(candidate.box, 1.0e9f, 1.0e9f) ||
        !cross_id_valid_rect(owner.box, 1.0e9f, 1.0e9f) ||
        !cross_id_center_inside(candidate.box, owner.box)) return false;
    const float ratio = candidate.box.width * candidate.box.height /
        (owner.box.width * owner.box.height);
    if (!std::isfinite(ratio) || ratio <= 0 || ratio > 0.60f) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto &camera = entries_[source_id];
    for (const auto &entry : camera)
      if (entry.candidate_id == candidate.id || entry.owner_id == owner.id) return false;
    Relation relation;
    relation.epoch = epoch;
    relation.birth_frame = frame;
    relation.last_frame = frame - 1;
    relation.last_candidate_frame = frame;
    relation.candidate_id = candidate.id;
    relation.owner_id = owner.id;
    relation.offset_x = ((candidate.box.left + candidate.box.width * 0.5f) -
        (owner.box.left + owner.box.width * 0.5f)) / owner.box.width;
    relation.offset_y = ((candidate.box.top + candidate.box.height * 0.5f) -
        (owner.box.top + owner.box.height * 0.5f)) / owner.box.height;
    relation.area_ratio = ratio;
    relation.last_area_ratio = ratio;
    camera.push_back(relation);
    return true;
  }

  Result filter_frame(std::uint32_t source_id, std::uint64_t epoch,
                      std::int32_t frame,
                      const std::vector<CrossIdTrack> &active_tracks,
                      const std::unordered_map<std::uint64_t, CrossIdRect> &shadow_tracks = {}) {
    Result result;
    if (source_id >= kSourceCount || frame < 0) return result;
    std::lock_guard<std::mutex> lock(mutex_);
    auto &camera = entries_[source_id];
    for (auto it = camera.begin(); it != camera.end();) {
      if (it->epoch != epoch || it->last_frame + 1 != frame) {
        result.released_ids.push_back(it->candidate_id);
        it = camera.erase(it);
        continue;
      }
      const auto owner = std::find_if(active_tracks.begin(), active_tracks.end(),
          [&](const CrossIdTrack &track) { return track.id == it->owner_id; });
      const auto candidate = std::find_if(active_tracks.begin(), active_tracks.end(),
          [&](const CrossIdTrack &track) { return track.id == it->candidate_id; });
      const auto owner_shadow = shadow_tracks.find(it->owner_id);
      const auto candidate_shadow = shadow_tracks.find(it->candidate_id);
      const bool owner_current = owner != active_tracks.end() || owner_shadow != shadow_tracks.end();
      const bool candidate_current = candidate != active_tracks.end() ||
          candidate_shadow != shadow_tracks.end();
      if (!owner_current || !candidate_current) {
        result.released_ids.push_back(it->candidate_id);
        it = camera.erase(it);
        continue;
      }
      const CrossIdRect &owner_box = owner != active_tracks.end()
          ? owner->box : owner_shadow->second;
      const CrossIdRect &candidate_box = candidate != active_tracks.end()
          ? candidate->box : candidate_shadow->second;
      const float ratio = candidate_box.width * candidate_box.height /
          (owner_box.width * owner_box.height);
      const float offset_x = ((candidate_box.left + candidate_box.width * 0.5f) -
          (owner_box.left + owner_box.width * 0.5f)) / owner_box.width;
      const float offset_y = ((candidate_box.top + candidate_box.height * 0.5f) -
          (owner_box.top + owner_box.height * 0.5f)) / owner_box.height;
      const bool continuous = cross_id_valid_rect(candidate_box, 1.0e9f, 1.0e9f) &&
          cross_id_valid_rect(owner_box, 1.0e9f, 1.0e9f) &&
          cross_id_center_inside(candidate_box, owner_box) &&
          std::isfinite(ratio) && ratio > 0.05f && ratio <= 0.75f &&
          std::fabs(offset_x - it->offset_x) <= 0.25f &&
          std::fabs(offset_y - it->offset_y) <= 0.24f &&
          // Person posture and detector/tracker boxes can change height
          // abruptly while the same explicitly shadowed target pair remains
          // spatially nested. Compare shape to the immediately preceding
          // frame, not to the birth-frame shape; the absolute ratio bound,
          // birth-relative center envelope, exact-frame continuity, source
          // epoch, and explicit current active/shadow rows remain mandatory.
          std::fabs(ratio - it->last_area_ratio) <= 0.25f;
      if (!continuous) {
        result.released_ids.push_back(it->candidate_id);
        it = camera.erase(it);
        continue;
      }
      // Never remove the only current active box: if exactly one of the
      // linked NvDCF targets is ACTIVE, publish that one and retain the link
      // while the other remains an explicit current shadow. Suppression is
      // needed only when both tracks are ACTIVE in the same frame.
      if (owner != active_tracks.end() && candidate != active_tracks.end()) {
        result.suppressed_ids.push_back(it->candidate_id);
        it->last_candidate_frame = frame;
      }
      it->last_area_ratio = ratio;
      it->last_frame = frame;
      ++it;
    }
    return result;
  }

  void clear_source(std::uint32_t source_id) {
    if (source_id >= kSourceCount) return;
    std::lock_guard<std::mutex> lock(mutex_);
    entries_[source_id].clear();
  }

 private:
  struct Relation {
    std::uint64_t epoch = 0;
    std::int32_t birth_frame = -1, last_frame = -1, last_candidate_frame = -1;
    std::uint64_t candidate_id = 0, owner_id = 0;
    float offset_x = 0, offset_y = 0, area_ratio = 0, last_area_ratio = 0;
  };
  std::mutex mutex_;
  std::vector<Relation> entries_[6];
};
