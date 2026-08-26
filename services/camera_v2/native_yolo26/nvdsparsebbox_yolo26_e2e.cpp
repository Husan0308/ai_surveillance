#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

const NvDsInferLayerInfo* find_detection_layer(
    const std::vector<NvDsInferLayerInfo>& layers) {
    for (const auto& layer : layers) {
        const auto elements = static_cast<std::size_t>(layer.inferDims.numElements);
        if (elements >= 6 && elements % 6 == 0 && layer.buffer != nullptr) {
            return &layer;
        }
    }
    return nullptr;
}

}  // namespace

extern "C" bool NvDsInferParseCustomYolo26E2E(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {
    const NvDsInferLayerInfo* layer = find_detection_layer(outputLayersInfo);
    if (layer == nullptr) {
        return false;
    }

    // YOLO26 end-to-end detect export: [x1, y1, x2, y2, confidence, class_id].
    // Gst-nvinfer invokes the parser per frame, so inferDims excludes the batch
    // dimension and contains rows*6 values for that frame.
    const auto* data = static_cast<const float*>(layer->buffer);
    const std::size_t elements = static_cast<std::size_t>(layer->inferDims.numElements);
    const std::size_t rows = elements / 6;

    const float person_threshold =
        detectionParams.perClassPreclusterThreshold.empty()
            ? 0.10F
            : detectionParams.perClassPreclusterThreshold[0];

    const float max_x = std::max(1U, networkInfo.width) - 1.0F;
    const float max_y = std::max(1U, networkInfo.height) - 1.0F;

    for (std::size_t row = 0; row < rows; ++row) {
        const float* p = data + row * 6;
        const float confidence = p[4];
        const int class_id = static_cast<int>(std::lround(p[5]));

        if (!std::isfinite(confidence) || confidence < person_threshold || class_id != 0) {
            continue;
        }

        float x1 = std::clamp(p[0], 0.0F, max_x);
        float y1 = std::clamp(p[1], 0.0F, max_y);
        float x2 = std::clamp(p[2], 0.0F, max_x);
        float y2 = std::clamp(p[3], 0.0F, max_y);
        if (!(x2 > x1 && y2 > y1)) {
            continue;
        }

        NvDsInferObjectDetectionInfo object{};
        object.classId = 0;
        object.detectionConfidence = confidence;
        object.left = x1;
        object.top = y1;
        object.width = x2 - x1;
        object.height = y2 - y1;
        objectList.push_back(object);
    }

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolo26E2E);
