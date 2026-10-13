#include "services/camera_v11/native/six_camera_v13/tracker_output_dedupe.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
  // CAM-02 source frame 3782: native ID 3 is emitted once on the previous
  // white-shirt person, then twice on the current frame. Only the right-hand
  // box follows the immediate previous ID-3 location; the other box overlaps
  // the independently tracked ID 2 and is withheld.
  const std::vector<TrackerOutputBox> previous{
      {2, 3781, 522.75f, 548.0f, 314.5f, 322.0f},
      {65, 3781, 754.75f, 577.5f, 322.5f, 437.0f},
      {3, 3781, 1112.5f, 1002.0f, 375.0f, 260.0f},
  };
  const std::vector<TrackerOutputBox> current{
      {3, 3782, 552.25f, 678.0f, 277.5f, 264.0f},
      {2, 3782, 525.25f, 547.0f, 319.5f, 318.0f},
      {65, 3782, 765.0f, 577.0f, 311.0f, 438.0f},
      {3, 3782, 1112.0f, 1002.0f, 376.0f, 260.0f},
  };
  const auto decision = dedupe_tracker_outputs_by_id_continuity(current, previous);
  assert(decision.withheld_ambiguous_ids.empty());
  assert(decision.suppressions.size() == 1);
  assert(decision.suppressions[0].id == 3);
  assert(decision.suppressions[0].suppressed_index == 0);
  assert(decision.suppressions[0].retained_index == 3);
  assert(decision.suppressions[0].retained_previous_iou > .99f);
  assert(decision.suppressions[0].suppressed_previous_iou == 0.0f);
  assert(decision.keep_indices.size() == 3);
  assert((decision.keep_indices == std::vector<std::size_t>{1, 2, 3}));

  // Different nearby people with distinct native IDs remain distinct, even
  // when their boxes overlap substantially.
  const std::vector<TrackerOutputBox> close_people{
      {10, 12, 100.0f, 100.0f, 120.0f, 220.0f},
      {11, 12, 150.0f, 110.0f, 120.0f, 220.0f},
  };
  const auto distinct = dedupe_tracker_outputs_by_id_continuity(close_people, {});
  assert(distinct.keep_indices.size() == 2);
  assert(distinct.suppressions.empty());
  assert(distinct.withheld_ambiguous_ids.empty());

  // No prior-frame proof, multiple plausible continuations, or a PTS/frame
  // reset must fail closed rather than choose an arbitrary ID-duplicate row.
  const std::vector<TrackerOutputBox> no_history{
      {7, 30, 0, 0, 100, 200}, {7, 30, 300, 0, 100, 200},
  };
  const auto missing_history = dedupe_tracker_outputs_by_id_continuity(no_history, {});
  assert(missing_history.keep_indices.empty());
  assert((missing_history.withheld_ambiguous_ids == std::vector<std::uint64_t>{7}));

  const std::vector<TrackerOutputBox> reset_previous{{7, 99, 0, 0, 100, 200}};
  const std::vector<TrackerOutputBox> reset_current{
      {7, 0, 0, 0, 100, 200}, {7, 0, 300, 0, 100, 200},
  };
  const auto reset = dedupe_tracker_outputs_by_id_continuity(reset_current, reset_previous);
  assert(reset.keep_indices.empty());
  assert((reset.withheld_ambiguous_ids == std::vector<std::uint64_t>{7}));

  const std::vector<TrackerOutputBox> ambiguous_previous{{8, 49, 0, 0, 100, 200}};
  const std::vector<TrackerOutputBox> ambiguous_current{
      {8, 50, 0, 0, 100, 200}, {8, 50, 2, 0, 100, 200},
  };
  const auto ambiguous = dedupe_tracker_outputs_by_id_continuity(
      ambiguous_current, ambiguous_previous);
  assert(ambiguous.keep_indices.empty());
  assert((ambiguous.withheld_ambiguous_ids == std::vector<std::uint64_t>{8}));

  std::puts("PASS: duplicate native ID output is continuity-resolved or withheld; distinct IDs remain separate");
}
