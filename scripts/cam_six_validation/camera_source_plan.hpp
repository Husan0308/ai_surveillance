#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace cam_six {

inline constexpr std::array<const char *, 6> CAMERA_IDS = {
    "CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06"};

struct SourcePlanEntry {
  const char *camera_id;
  int logical_source_id;
  int mux_pad_index;
};

struct PipelineShape {
  unsigned batch_size;
  unsigned tile_rows;
  unsigned tile_columns;
  unsigned output_width;
  unsigned output_height;
};

// Empty selection preserves the historical six-camera order. A selected
// camera retains its stable logical source ID while receiving a compact mux
// pad index for this process's active source list.
inline bool build_source_plan(const std::string &only_camera,
                              std::vector<SourcePlanEntry> &plan) {
  plan.clear();
  if (only_camera.empty()) {
    for (std::size_t i = 0; i < CAMERA_IDS.size(); ++i) {
      plan.push_back({CAMERA_IDS[i], static_cast<int>(i), static_cast<int>(i)});
    }
    return true;
  }
  for (std::size_t i = 0; i < CAMERA_IDS.size(); ++i) {
    if (only_camera == CAMERA_IDS[i]) {
      plan.push_back({CAMERA_IDS[i], static_cast<int>(i), 0});
      return true;
    }
  }
  return false;
}

inline bool pipeline_shape(std::size_t active_sources, PipelineShape &shape) {
  if (active_sources == 1) {
    shape = {1, 1, 1, 1920, 1080};
    return true;
  }
  if (active_sources == CAMERA_IDS.size()) {
    shape = {6, 3, 2, 1920, 1620};
    return true;
  }
  return false;
}

}  // namespace cam_six
