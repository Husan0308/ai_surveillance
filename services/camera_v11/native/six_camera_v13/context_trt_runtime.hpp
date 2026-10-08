// Diagnostic-only direct TensorRT runtime for context verification. The
// primary nvinfer/NvDCF branch never calls this code.
#pragma once

#include "lowconf_contract.hpp"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <nvdsinfer_custom_impl.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

struct LowConfVerifierProposal;

class LowConfContextTrtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char *message) noexcept override {
    if (severity <= Severity::kWARNING)
      g_printerr("LOWCONF_TRT_LOG severity=%d message=%s\n",
          static_cast<int>(severity), message ? message : "");
  }
};

using LowConfContextParser = bool (*)(
    const std::vector<NvDsInferLayerInfo>&, const NvDsInferNetworkInfo&,
    const NvDsInferParseDetectionParams&,
    std::vector<NvDsInferObjectDetectionInfo>&);

static LowConfContextTrtLogger lowconf_context_trt_logger;
static nvinfer1::IRuntime *lowconf_context_trt_runtime = nullptr;
static nvinfer1::ICudaEngine *lowconf_context_trt_engine = nullptr;
static nvinfer1::IExecutionContext *lowconf_context_trt_execution = nullptr;
struct LowConfContextTrtLane {
  nvinfer1::IExecutionContext *execution = nullptr;
  float *device_output = nullptr;
  float *host_output = nullptr;
};
static std::array<LowConfContextTrtLane, 6> lowconf_context_trt_lanes{};
static bool lowconf_context_trt_per_camera = false;
static void *lowconf_context_trt_parser_library = nullptr;
static LowConfContextParser lowconf_context_trt_parser = nullptr;
static std::string lowconf_context_trt_input_name;
static std::string lowconf_context_trt_output_name;
static float *lowconf_context_trt_device_output = nullptr;
static float *lowconf_context_trt_host_output = nullptr;
static size_t lowconf_context_trt_output_elements = 0;
static guint lowconf_context_trt_side = 0;
static guint lowconf_context_trt_batch_limit = 0;
static float lowconf_context_trt_threshold = 0.20f;
static float lowconf_context_trt_nms_iou = 0.45f;
static guint lowconf_context_trt_topk = 300;

static bool lowconf_context_trt_per_camera_enabled() {
  const char *value = g_getenv("V11_CONTEXT_VERIFIER_PER_CAMERA");
  return value && std::strcmp(value, "1") == 0;
}

static float lowconf_context_trt_iou(const NvDsInferObjectDetectionInfo &a,
                                    const NvDsInferObjectDetectionInfo &b) {
  const float left = std::max(a.left, b.left);
  const float top = std::max(a.top, b.top);
  const float right = std::min(a.left + a.width, b.left + b.width);
  const float bottom = std::min(a.top + a.height, b.top + b.height);
  const float intersection = std::max(0.0f, right - left) *
      std::max(0.0f, bottom - top);
  const float union_area = a.width * a.height + b.width * b.height - intersection;
  return union_area > 0.0f ? intersection / union_area : 0.0f;
}

static bool lowconf_context_trt_initialize(const std::string &engine_path,
    const std::string &parser_path, const std::string &parser_symbol,
    guint side, guint batch_limit, float threshold, float nms_iou, guint topk) {
  if (lowconf_context_trt_engine || lowconf_context_trt_runtime ||
      lowconf_context_trt_execution || side == 0 || batch_limit == 0 || batch_limit > 6)
    return false;
  std::ifstream engine_file(engine_path, std::ios::binary);
  if (!engine_file) return false;
  const std::vector<char> serialized((std::istreambuf_iterator<char>(engine_file)),
      std::istreambuf_iterator<char>());
  if (serialized.empty()) return false;
  if (!initLibNvInferPlugins(&lowconf_context_trt_logger, "")) {
    g_printerr("LOWCONF_TRT_INIT_FAIL plugin_init\n");
    return false;
  }
  lowconf_context_trt_runtime = nvinfer1::createInferRuntime(lowconf_context_trt_logger);
  if (!lowconf_context_trt_runtime) return false;
  lowconf_context_trt_engine = lowconf_context_trt_runtime->deserializeCudaEngine(
      serialized.data(), serialized.size());
  if (!lowconf_context_trt_engine) return false;
  lowconf_context_trt_per_camera = lowconf_context_trt_per_camera_enabled();
  if (!lowconf_context_trt_per_camera) {
    lowconf_context_trt_execution = lowconf_context_trt_engine->createExecutionContext();
    if (!lowconf_context_trt_execution) return false;
  }

  for (int index = 0; index < lowconf_context_trt_engine->getNbIOTensors(); ++index) {
    const char *name = lowconf_context_trt_engine->getIOTensorName(index);
    if (!name) return false;
    if (lowconf_context_trt_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      if (!lowconf_context_trt_input_name.empty()) return false;
      lowconf_context_trt_input_name = name;
    } else {
      if (!lowconf_context_trt_output_name.empty()) return false;
      lowconf_context_trt_output_name = name;
    }
  }
  if (lowconf_context_trt_input_name.empty() ||
      lowconf_context_trt_output_name != "output0" ||
      lowconf_context_trt_engine->getTensorDataType(lowconf_context_trt_input_name.c_str()) !=
          nvinfer1::DataType::kFLOAT ||
      lowconf_context_trt_engine->getTensorDataType(lowconf_context_trt_output_name.c_str()) !=
          nvinfer1::DataType::kFLOAT) {
    g_printerr("LOWCONF_TRT_INIT_FAIL unsupported_io input=%s output=%s\n",
        lowconf_context_trt_input_name.c_str(), lowconf_context_trt_output_name.c_str());
    return false;
  }
  const int side_int = static_cast<int>(side);
  nvinfer1::Dims4 input_dims{1, 3, side_int, side_int};
  nvinfer1::Dims output_dims{};
  if (lowconf_context_trt_per_camera) {
    for (auto &lane : lowconf_context_trt_lanes) {
      lane.execution = lowconf_context_trt_engine->createExecutionContext();
      if (!lane.execution || !lane.execution->setInputShape(
              lowconf_context_trt_input_name.c_str(), input_dims)) return false;
      output_dims = lane.execution->getTensorShape(lowconf_context_trt_output_name.c_str());
      if (output_dims.nbDims != 3 || output_dims.d[0] != 1 || output_dims.d[1] != 84 ||
          output_dims.d[2] <= 0) return false;
    }
  } else {
    if (!lowconf_context_trt_execution->setInputShape(
            lowconf_context_trt_input_name.c_str(), input_dims)) return false;
    output_dims = lowconf_context_trt_execution->getTensorShape(
        lowconf_context_trt_output_name.c_str());
    if (output_dims.nbDims != 3 || output_dims.d[0] != 1 || output_dims.d[1] != 84 ||
        output_dims.d[2] <= 0) {
      g_printerr("LOWCONF_TRT_INIT_FAIL output_shape dims=%d %ld %ld %ld\n",
          output_dims.nbDims, output_dims.d[0], output_dims.d[1], output_dims.d[2]);
      return false;
    }
  }
  const size_t per_frame = static_cast<size_t>(output_dims.d[1]) * output_dims.d[2];
  lowconf_context_trt_output_elements = per_frame *
      (lowconf_context_trt_per_camera ? 1u : batch_limit);
  lowconf_context_trt_side = side;
  lowconf_context_trt_batch_limit = batch_limit;
  lowconf_context_trt_threshold = threshold;
  lowconf_context_trt_nms_iou = nms_iou;
  lowconf_context_trt_topk = topk;
  if (cudaSetDevice(0) != cudaSuccess) return false;
  if (lowconf_context_trt_per_camera) {
    for (auto &lane : lowconf_context_trt_lanes) {
      if (cudaMalloc(reinterpret_cast<void **>(&lane.device_output),
              lowconf_context_trt_output_elements * sizeof(float)) != cudaSuccess ||
          cudaHostAlloc(reinterpret_cast<void **>(&lane.host_output),
              lowconf_context_trt_output_elements * sizeof(float), cudaHostAllocPortable) != cudaSuccess) {
        g_printerr("LOWCONF_TRT_INIT_FAIL per_camera_output_buffer_allocation\n");
        return false;
      }
    }
  } else if (cudaMalloc(reinterpret_cast<void **>(&lowconf_context_trt_device_output),
          lowconf_context_trt_output_elements * sizeof(float)) != cudaSuccess ||
      cudaHostAlloc(reinterpret_cast<void **>(&lowconf_context_trt_host_output),
          lowconf_context_trt_output_elements * sizeof(float), cudaHostAllocPortable) != cudaSuccess) {
    g_printerr("LOWCONF_TRT_INIT_FAIL output_buffer_allocation\n");
    return false;
  }
  lowconf_context_trt_parser_library = dlopen(parser_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!lowconf_context_trt_parser_library) {
    g_printerr("LOWCONF_TRT_INIT_FAIL parser_load=%s\n", dlerror());
    return false;
  }
  lowconf_context_trt_parser = reinterpret_cast<LowConfContextParser>(
      dlsym(lowconf_context_trt_parser_library, parser_symbol.c_str()));
  if (!lowconf_context_trt_parser) {
    g_printerr("LOWCONF_TRT_INIT_FAIL parser_symbol=%s\n", parser_symbol.c_str());
    return false;
  }
  g_print("LOWCONF_TRT_READY engine=%s parser=%s input=%s output=%s side=%u batch_limit=%u binding=FP32 engine_compute=FP16 scheduling=%s\n",
      engine_path.c_str(), parser_path.c_str(), lowconf_context_trt_input_name.c_str(),
      lowconf_context_trt_output_name.c_str(), side, batch_limit,
      lowconf_context_trt_per_camera ? "six-persistent-per-camera-async-lanes" : "shared-dynamic-batch");
  return true;
}

static bool lowconf_context_trt_infer(const void *device_input, void *stream,
    guint batch, std::vector<std::vector<LowConfVerifierProposal>> *results) {
  if (!device_input || !stream || !results || !lowconf_context_trt_execution ||
      batch == 0 || batch > lowconf_context_trt_batch_limit || cudaSetDevice(0) != cudaSuccess)
    return false;
  results->clear();
  results->resize(batch);
  nvinfer1::Dims4 input_dims{static_cast<int>(batch), 3,
      static_cast<int>(lowconf_context_trt_side), static_cast<int>(lowconf_context_trt_side)};
  const guint64 shape_start_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  if (!lowconf_context_trt_execution->setInputShape(
          lowconf_context_trt_input_name.c_str(), input_dims) ||
      !lowconf_context_trt_execution->setTensorAddress(
          lowconf_context_trt_input_name.c_str(), const_cast<void *>(device_input)) ||
      !lowconf_context_trt_execution->setTensorAddress(
          lowconf_context_trt_output_name.c_str(), lowconf_context_trt_device_output)) {
    return false;
  }
  const guint64 enqueue_start_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  const nvinfer1::Dims output_dims = lowconf_context_trt_execution->getTensorShape(
      lowconf_context_trt_output_name.c_str());
  if (output_dims.nbDims != 3 || output_dims.d[0] != static_cast<int>(batch) ||
      output_dims.d[1] != 84 || output_dims.d[2] <= 0) return false;
  const size_t anchors = static_cast<size_t>(output_dims.d[2]);
  const size_t frame_elements = 84 * anchors;
  const size_t output_elements = frame_elements * batch;
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  cudaEvent_t gpu_start = nullptr;
  cudaEvent_t gpu_done = nullptr;
  const bool have_events = cudaEventCreate(&gpu_start) == cudaSuccess &&
      cudaEventCreate(&gpu_done) == cudaSuccess;
  if (have_events && cudaEventRecord(gpu_start, cuda_stream) != cudaSuccess) return false;
  if (!lowconf_context_trt_execution->enqueueV3(cuda_stream)) return false;
  const guint64 enqueue_done_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  if (cudaMemcpyAsync(lowconf_context_trt_host_output, lowconf_context_trt_device_output,
          output_elements * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream) != cudaSuccess)
    return false;
  if (have_events && cudaEventRecord(gpu_done, cuda_stream) != cudaSuccess) return false;
  const guint64 sync_start_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  if (cudaStreamSynchronize(cuda_stream) != cudaSuccess) return false;
  const guint64 sync_done_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  float gpu_elapsed_ms = -1.0f;
  if (have_events && cudaEventElapsedTime(&gpu_elapsed_ms, gpu_start, gpu_done) != cudaSuccess)
    gpu_elapsed_ms = -1.0f;
  if (gpu_start) cudaEventDestroy(gpu_start);
  if (gpu_done) cudaEventDestroy(gpu_done);
  if (g_getenv("V11_CONTEXT_VERIFIER_TRACE"))
    g_printerr("LOWCONF_TRT_TIMING batch=%u set_shape_ms=%.3f enqueue_host_ms=%.3f sync_wait_ms=%.3f gpu_stream_ms=%.3f\n",
        batch, (enqueue_start_ns - shape_start_ns) / 1.0e6,
        (enqueue_done_ns - enqueue_start_ns) / 1.0e6,
        (sync_done_ns - sync_start_ns) / 1.0e6, gpu_elapsed_ms);

  const guint side = lowconf_context_trt_side;
  NvDsInferNetworkInfo network{};
  network.width = side;
  network.height = side;
  network.channels = 3;
  NvDsInferParseDetectionParams parse_params{};
  parse_params.numClassesConfigured = 80;
  parse_params.perClassPreclusterThreshold.assign(80, lowconf_context_trt_threshold);
  parse_params.perClassPostclusterThreshold.assign(80, 0.0f);
  const float to_640 = 640.0f / static_cast<float>(side);
  const guint64 parse_start_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
  for (guint frame = 0; frame < batch; ++frame) {
    NvDsInferLayerInfo output{};
    output.dataType = FLOAT;
    output.isInput = FALSE;
    output.layerName = const_cast<char *>(lowconf_context_trt_output_name.c_str());
    output.buffer = lowconf_context_trt_host_output + frame * frame_elements;
    output.inferDims.numDims = 2;
    output.inferDims.d[0] = 84;
    output.inferDims.d[1] = static_cast<unsigned int>(anchors);
    output.inferDims.numElements = static_cast<unsigned int>(frame_elements);
    std::vector<NvDsInferLayerInfo> layers{output};
    std::vector<NvDsInferObjectDetectionInfo> parsed;
    if (!lowconf_context_trt_parser(layers, network, parse_params, parsed)) return false;
    std::stable_sort(parsed.begin(), parsed.end(), [](const auto &left, const auto &right) {
      return left.detectionConfidence > right.detectionConfidence;
    });
    std::vector<NvDsInferObjectDetectionInfo> kept;
    kept.reserve(std::min<size_t>(parsed.size(), lowconf_context_trt_topk));
    for (const auto &candidate : parsed) {
      bool suppressed = false;
      for (const auto &accepted : kept) {
        if (lowconf_context_trt_iou(candidate, accepted) > lowconf_context_trt_nms_iou) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) {
        kept.push_back(candidate);
        if (kept.size() >= lowconf_context_trt_topk) break;
      }
    }
    for (const auto &object : kept) {
      (*results)[frame].push_back({object.detectionConfidence,
          object.left * to_640, object.top * to_640,
          object.width * to_640, object.height * to_640});
    }
  }
  if (g_getenv("V11_CONTEXT_VERIFIER_TRACE")) {
    const guint64 parse_done_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
    g_printerr("LOWCONF_TRT_PARSE batch=%u parse_ms=%.3f\n", batch,
        (parse_done_ns - parse_start_ns) / 1.0e6);
  }
  return true;
}

// Diagnostic low-latency path: each camera's crop worker owns one persistent
// execution context, output buffer and CUDA stream. No context/buffer is shared
// concurrently, and every result is still joined by exact source PTS upstream.
static bool lowconf_context_trt_infer_lane(guint lane_index,
    const void *device_input, void *stream,
    std::vector<LowConfVerifierProposal> *results) {
  if (!lowconf_context_trt_per_camera || lane_index >= lowconf_context_trt_lanes.size() ||
      !device_input || !stream || !results || cudaSetDevice(0) != cudaSuccess)
    return false;
  auto &lane = lowconf_context_trt_lanes[lane_index];
  if (!lane.execution || !lane.device_output || !lane.host_output) return false;
  nvinfer1::Dims4 input_dims{1, 3, static_cast<int>(lowconf_context_trt_side),
      static_cast<int>(lowconf_context_trt_side)};
  if (!lane.execution->setInputShape(lowconf_context_trt_input_name.c_str(), input_dims) ||
      !lane.execution->setTensorAddress(lowconf_context_trt_input_name.c_str(),
          const_cast<void *>(device_input)) ||
      !lane.execution->setTensorAddress(lowconf_context_trt_output_name.c_str(),
          lane.device_output)) return false;
  const nvinfer1::Dims output_dims = lane.execution->getTensorShape(
      lowconf_context_trt_output_name.c_str());
  if (output_dims.nbDims != 3 || output_dims.d[0] != 1 || output_dims.d[1] != 84 ||
      output_dims.d[2] <= 0) return false;
  const size_t anchors = static_cast<size_t>(output_dims.d[2]);
  const size_t output_elements = 84 * anchors;
  if (output_elements > lowconf_context_trt_output_elements) return false;
  const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (!lane.execution->enqueueV3(cuda_stream) ||
      cudaMemcpyAsync(lane.host_output, lane.device_output,
          output_elements * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream) != cudaSuccess ||
      cudaStreamSynchronize(cuda_stream) != cudaSuccess) return false;

  const guint side = lowconf_context_trt_side;
  NvDsInferNetworkInfo network{};
  network.width = side;
  network.height = side;
  network.channels = 3;
  NvDsInferParseDetectionParams parse_params{};
  parse_params.numClassesConfigured = 80;
  parse_params.perClassPreclusterThreshold.assign(80, lowconf_context_trt_threshold);
  parse_params.perClassPostclusterThreshold.assign(80, 0.0f);
  std::vector<NvDsInferLayerInfo> layers(1);
  auto &output = layers.front();
  output.dataType = FLOAT;
  output.isInput = FALSE;
  output.layerName = const_cast<char *>(lowconf_context_trt_output_name.c_str());
  output.buffer = lane.host_output;
  output.inferDims.numDims = 2;
  output.inferDims.d[0] = 84;
  output.inferDims.d[1] = static_cast<unsigned int>(anchors);
  output.inferDims.numElements = static_cast<unsigned int>(output_elements);
  std::vector<NvDsInferObjectDetectionInfo> parsed;
  if (!lowconf_context_trt_parser(layers, network, parse_params, parsed)) return false;
  std::stable_sort(parsed.begin(), parsed.end(), [](const auto &left, const auto &right) {
    return left.detectionConfidence > right.detectionConfidence;
  });
  std::vector<NvDsInferObjectDetectionInfo> kept;
  kept.reserve(std::min<size_t>(parsed.size(), lowconf_context_trt_topk));
  for (const auto &candidate : parsed) {
    bool suppressed = false;
    for (const auto &accepted : kept) {
      if (lowconf_context_trt_iou(candidate, accepted) > lowconf_context_trt_nms_iou) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push_back(candidate);
      if (kept.size() >= lowconf_context_trt_topk) break;
    }
  }
  results->clear();
  results->reserve(kept.size());
  const float to_640 = 640.0f / static_cast<float>(side);
  for (const auto &object : kept) {
    results->push_back({object.detectionConfidence, object.left * to_640,
        object.top * to_640, object.width * to_640, object.height * to_640});
  }
  return true;
}

static void lowconf_context_trt_shutdown() {
  for (auto &lane : lowconf_context_trt_lanes) {
    if (lane.execution) delete lane.execution;
    if (lane.device_output) cudaFree(lane.device_output);
    if (lane.host_output) cudaFreeHost(lane.host_output);
    lane = {};
  }
  if (lowconf_context_trt_execution) delete lowconf_context_trt_execution;
  if (lowconf_context_trt_engine) delete lowconf_context_trt_engine;
  if (lowconf_context_trt_runtime) delete lowconf_context_trt_runtime;
  if (lowconf_context_trt_device_output) cudaFree(lowconf_context_trt_device_output);
  if (lowconf_context_trt_host_output) cudaFreeHost(lowconf_context_trt_host_output);
  if (lowconf_context_trt_parser_library) dlclose(lowconf_context_trt_parser_library);
  lowconf_context_trt_execution = nullptr;
  lowconf_context_trt_engine = nullptr;
  lowconf_context_trt_runtime = nullptr;
  lowconf_context_trt_device_output = nullptr;
  lowconf_context_trt_host_output = nullptr;
  lowconf_context_trt_parser_library = nullptr;
  lowconf_context_trt_parser = nullptr;
  lowconf_context_trt_input_name.clear();
  lowconf_context_trt_output_name.clear();
  lowconf_context_trt_output_elements = 0;
}
