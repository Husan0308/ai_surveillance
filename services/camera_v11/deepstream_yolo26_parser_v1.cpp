#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {
constexpr int kValuesPerDetection = 6;
constexpr int kPersonClass = 0;

float clampf(float value, float low, float high) {
    return std::max(low, std::min(high, value));
}
}

extern "C" bool NvDsInferParseCustomYolo26V1(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    objectList.clear();
    if (networkInfo.width == 0 || networkInfo.height == 0) {
        return false;
    }

    NvDsInferLayerInfo const *detections = nullptr;
    for (auto const &layer : outputLayersInfo) {
        if (layer.buffer == nullptr || layer.dataType != FLOAT) {
            continue;
        }
        const auto elements = static_cast<std::size_t>(layer.inferDims.numElements);
        if (elements >= static_cast<std::size_t>(kValuesPerDetection) &&
            elements % static_cast<std::size_t>(kValuesPerDetection) == 0) {
            detections = &layer;
            break;
        }
    }
    if (detections == nullptr) {
        return false;
    }

    const float threshold =
        detectionParams.perClassPreclusterThreshold.empty()
            ? 0.0f
            : detectionParams.perClassPreclusterThreshold.front();
    const auto count = static_cast<std::size_t>(detections->inferDims.numElements) /
                       static_cast<std::size_t>(kValuesPerDetection);
    const auto *rows = static_cast<const float *>(detections->buffer);
    objectList.reserve(std::min<std::size_t>(count, 300));

    for (std::size_t index = 0; index < count; ++index) {
        const float *row = rows + index * kValuesPerDetection;
        const float x1_raw = row[0];
        const float y1_raw = row[1];
        const float x2_raw = row[2];
        const float y2_raw = row[3];
        const float confidence = row[4];
        const int class_id = static_cast<int>(std::lround(row[5]));

        if (!std::isfinite(x1_raw) || !std::isfinite(y1_raw) ||
            !std::isfinite(x2_raw) || !std::isfinite(y2_raw) ||
            !std::isfinite(confidence) || confidence < threshold ||
            class_id != kPersonClass) {
            continue;
        }

        const float max_x = static_cast<float>(networkInfo.width);
        const float max_y = static_cast<float>(networkInfo.height);
        const float x1 = clampf(x1_raw, 0.0f, max_x);
        const float y1 = clampf(y1_raw, 0.0f, max_y);
        const float x2 = clampf(x2_raw, 0.0f, max_x);
        const float y2 = clampf(y2_raw, 0.0f, max_y);
        if (x2 <= x1 + 1.0f || y2 <= y1 + 1.0f) {
            continue;
        }

        NvDsInferObjectDetectionInfo object{};
        object.classId = kPersonClass;
        object.left = x1;
        object.top = y1;
        object.width = x2 - x1;
        object.height = y2 - y1;
        object.detectionConfidence = clampf(confidence, 0.0f, 1.0f);
        objectList.push_back(object);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolo26V1);
