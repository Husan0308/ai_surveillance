#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <cstdio>
#include <vector>

extern "C" bool NvDsInferParseYolo26RawPerson(
    const std::vector<NvDsInferLayerInfo> &,
    const NvDsInferNetworkInfo &,
    const NvDsInferParseDetectionParams &,
    std::vector<NvDsInferObjectDetectionInfo> &);

int main() {
#ifndef YOLO_PARSER_TEST_SIDE
#define YOLO_PARSER_TEST_SIDE 768
#endif
  constexpr int side = YOLO_PARSER_TEST_SIDE;
  constexpr int anchors = (side / 8) * (side / 8) +
                          (side / 16) * (side / 16) +
                          (side / 32) * (side / 32);
  static_assert(anchors == 12096);
  std::vector<float> data(84 * anchors, 0.0f);
  NvDsInferLayerInfo layer{};
  layer.dataType = FLOAT;
  layer.buffer = data.data();
  layer.inferDims.numDims = 2;
  layer.inferDims.d[0] = 84;
  layer.inferDims.d[1] = anchors;
  layer.inferDims.numElements = 84 * anchors;
  NvDsInferNetworkInfo network{side, side, 3};
  NvDsInferParseDetectionParams params{};
  params.numClassesConfigured = 80;
  params.perClassPreclusterThreshold.assign(80, .25f);
  data[0] = side / 2.0f;
  data[anchors] = side / 2.0f;
  data[2 * anchors] = side / 4.0f;
  data[3 * anchors] = side / 2.0f;
  data[4 * anchors] = .8f;
  std::vector<NvDsInferObjectDetectionInfo> objects;
  assert(NvDsInferParseYolo26RawPerson({layer}, network, params, objects));
  assert(objects.size() == 1);
  assert(objects[0].classId == 0 && objects[0].detectionConfidence == .8f);
  assert(objects[0].left == side * 3 / 8.0f);
  assert(objects[0].top == side / 4.0f);
  assert(objects[0].width == side / 4.0f);
  assert(objects[0].height == side / 2.0f);
  std::puts("PASS: dynamic raw YOLO26 parser 768x768 shape and box semantics");
}
