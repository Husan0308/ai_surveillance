// F5R-only raw YOLO26 parser. Same class/confidence/box/NMS semantics as the
// validated parser; derive the raw anchor count from DeepStream's input size.
#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>

static std::atomic<unsigned long long> errors{0}, rejected{0}, calls{0}, proposals{0};
extern "C" unsigned long long Yolo26ParserErrors() { return errors.load(); }
extern "C" unsigned long long Yolo26ParserRejected() { return rejected.load(); }
extern "C" unsigned long long Yolo26ParserCalls() { return calls.load(); }
extern "C" unsigned long long Yolo26ParserProposals() { return proposals.load(); }

static bool invalid(const char *why) {
  if (errors.fetch_add(1) == 0) std::fprintf(stderr, "YOLO PARSER_ERROR %s\n", why);
  return false;
}

static int expected_anchors(unsigned width, unsigned height) {
  if (width != height || width < 32 || width % 32 != 0) return -1;
  int total = 0;
  for (const int stride : {8, 16, 32}) {
    const int side = static_cast<int>(width) / stride;
    total += side * side;
  }
  return total;
}

extern "C" bool NvDsInferParseYolo26RawPerson(
    const std::vector<NvDsInferLayerInfo> &layers,
    const NvDsInferNetworkInfo &net,
    const NvDsInferParseDetectionParams &params,
    std::vector<NvDsInferObjectDetectionInfo> &objects) {
  objects.clear();
  ++calls;
  if (layers.size() != 1 || !layers[0].buffer || layers[0].isInput ||
      layers[0].dataType != FLOAT || net.channels != 3 ||
      params.numClassesConfigured != 80 ||
      params.perClassPreclusterThreshold.size() != 80) {
    return invalid("invalid raw layer/network/classes");
  }
  const int anchors = expected_anchors(net.width, net.height);
  const auto &dims = layers[0].inferDims;
  const unsigned expected = anchors > 0 ? static_cast<unsigned>(anchors) : 0u;
  if (anchors <= 0 || dims.numDims != 2 || dims.d[0] != 84u ||
      dims.d[1] != expected || dims.numElements != 84u * expected) {
    return invalid("raw shape does not match 80-class YOLO26 strides 8/16/32");
  }
  const float threshold = params.perClassPreclusterThreshold[0];
  if (!std::isfinite(threshold) || threshold < 0 || threshold > 1) {
    return invalid("invalid confidence threshold");
  }

  const float *data = static_cast<const float *>(layers[0].buffer);
  for (int a = 0; a < anchors; ++a) {
    float score = -1;
    int cls = -1;
    bool good = true;
    for (int c = 0; c < 80; ++c) {
      const float value = data[(4 + c) * anchors + a];
      if (!std::isfinite(value) || value < 0 || value > 1) {
        good = false;
        break;
      }
      if (value > score) {
        score = value;
        cls = c;
      }
    }
    if (!good) {
      ++rejected;
      continue;
    }
    if (cls != 0 || score < threshold) continue;

    const float cx = data[a];
    const float cy = data[anchors + a];
    const float w = data[2 * anchors + a];
    const float h = data[3 * anchors + a];
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) ||
        !std::isfinite(h) || w <= 0 || h <= 0) {
      ++rejected;
      continue;
    }
    const float x1 = std::clamp(double(cx) - double(w) / 2, 0.0, double(net.width));
    const float y1 = std::clamp(double(cy) - double(h) / 2, 0.0, double(net.height));
    const float x2 = std::clamp(double(cx) + double(w) / 2, 0.0, double(net.width));
    const float y2 = std::clamp(double(cy) + double(h) / 2, 0.0, double(net.height));
    if (x2 <= x1 || y2 <= y1) {
      ++rejected;
      continue;
    }
    NvDsInferObjectDetectionInfo obj{};
    obj.classId = 0;
    obj.left = x1;
    obj.top = y1;
    obj.width = x2 - x1;
    obj.height = y2 - y1;
    obj.detectionConfidence = score;
    objects.push_back(obj);  // DeepStream NMS remains the only suppression stage.
  }
  proposals += objects.size();
  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo26RawPerson);
