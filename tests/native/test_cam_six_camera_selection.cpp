#include "../../scripts/cam_six_validation/camera_source_plan.hpp"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
  std::vector<cam_six::SourcePlanEntry> plan;
  assert(cam_six::build_source_plan("", plan));
  assert(plan.size() == 6);
  for (std::size_t i = 0; i < plan.size(); ++i) {
    assert(plan[i].camera_id == cam_six::CAMERA_IDS[i]);
    assert(plan[i].logical_source_id == static_cast<int>(i));
    assert(plan[i].mux_pad_index == static_cast<int>(i));
  }

  assert(cam_six::build_source_plan("CAM-02", plan));
  assert(plan.size() == 1);
  assert(std::string(plan[0].camera_id) == "CAM-02");
  assert(plan[0].logical_source_id == 1);
  assert(plan[0].mux_pad_index == 0);

  assert(!cam_six::build_source_plan("CAM-07", plan));
  assert(plan.empty());
  plan.resize(2);
  cam_six::PipelineShape shape{};
  assert(!cam_six::pipeline_shape(plan.size(), shape));

  assert(cam_six::build_source_plan("CAM-02", plan));
  assert(cam_six::pipeline_shape(plan.size(), shape));
  assert(shape.batch_size == 1 && shape.tile_rows == 1 && shape.tile_columns == 1);
  assert(shape.output_width == 1920 && shape.output_height == 1080);

  assert(cam_six::build_source_plan("", plan));
  assert(cam_six::pipeline_shape(plan.size(), shape));
  assert(shape.batch_size == 6 && shape.tile_rows == 3 && shape.tile_columns == 2);
  assert(shape.output_width == 1920 && shape.output_height == 1620);

  std::cout << "camera selection and pipeline shapes PASS\n";
}
