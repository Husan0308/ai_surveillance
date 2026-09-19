// DeepStream 9.1 YOLO26 one-to-one parser. Network pixel coordinates; no NMS.
#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>

static std::atomic<unsigned long long> errors{0}, rejected{0}, calls{0};
extern "C" unsigned long long Yolo26ParserErrors() { return errors.load(); }
extern "C" unsigned long long Yolo26ParserRejected() { return rejected.load(); }
extern "C" unsigned long long Yolo26ParserCalls() { return calls.load(); }
static bool invalid(const char *why) {
  if (errors.fetch_add(1) == 0) std::fprintf(stderr, "YOLO PARSER_ERROR %s\n", why);
  return false;
}
extern "C" bool NvDsInferParseYolo26Person(
    const std::vector<NvDsInferLayerInfo> &layers, const NvDsInferNetworkInfo &net,
    const NvDsInferParseDetectionParams &params,
    std::vector<NvDsInferObjectDetectionInfo> &objects) {
  objects.clear();
  ++calls;
  if (layers.size() != 1 || !layers[0].buffer || layers[0].isInput ||
      layers[0].dataType != FLOAT || net.width != 640 || net.height != 640 ||
      params.numClassesConfigured != 80 || params.perClassPreclusterThreshold.size() < 80)
    return invalid("invalid layer, network or class configuration");
  const auto &dim = layers[0].inferDims;
  // nvinfer calls this API per frame and removes the batch dimension.
  if (dim.numDims != 2 || dim.d[0] != 300 || dim.d[1] != 6 || dim.numElements != 1800)
    return invalid("expected per-frame FLOAT tensor (300,6)");
  const float threshold = params.perClassPreclusterThreshold[0];
  if (!std::isfinite(threshold) || threshold < 0 || threshold > 1)
    return invalid("confidence threshold must be in [0,1]");
  const auto *data = static_cast<const float*>(layers[0].buffer);
  for (unsigned i = 0; i < 300; ++i) {
    const float *p = data + i * 6;
    bool finite = true;
    for (int k = 0; k < 6; ++k) finite &= std::isfinite(p[k]);
    if (!finite || p[4] < 0 || p[4] > 1 || p[5] < 0 || p[5] >= 80 || std::trunc(p[5]) != p[5]) {
      ++rejected; continue;
    }
    if (p[4] < threshold || p[5] != 0) continue;
    if (p[2] <= p[0] || p[3] <= p[1]) { ++rejected; continue; }
    const float x1 = std::clamp(p[0], 0.0f, float(net.width));
    const float y1 = std::clamp(p[1], 0.0f, float(net.height));
    const float x2 = std::clamp(p[2], 0.0f, float(net.width));
    const float y2 = std::clamp(p[3], 0.0f, float(net.height));
    if (x2 <= x1 || y2 <= y1) { ++rejected; continue; }
    NvDsInferObjectDetectionInfo obj{};  // Includes rotation_angle=0 on DS9.1.
    obj.classId = 0;
    obj.left = x1; obj.top = y1; obj.width = x2-x1; obj.height = y2-y1;
    obj.detectionConfidence = p[4];
    objects.push_back(obj);
  }
  return true;
}
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo26Person);
