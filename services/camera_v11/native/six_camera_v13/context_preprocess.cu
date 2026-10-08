#include <cuda_runtime.h>
#include <cstddef>

namespace {

__global__ void rgb_hwc_to_normalized_nchw_fp32(const unsigned char *source,
    size_t source_pitch, float *destination, unsigned int side,
    unsigned int batch_index, unsigned int batch_capacity, float scale) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= side || y >= side) return;
  const size_t pixel = static_cast<size_t>(y) * side + x;
  const size_t chw = static_cast<size_t>(side) * side;
  const size_t batch_offset = static_cast<size_t>(batch_index) * 3 * chw;
  const unsigned char *row = source + static_cast<size_t>(y) * source_pitch + x * 3;
  destination[batch_offset + pixel] = static_cast<float>(row[0]) * scale;
  destination[batch_offset + chw + pixel] = static_cast<float>(row[1]) * scale;
  destination[batch_offset + 2 * chw + pixel] = static_cast<float>(row[2]) * scale;
  (void)batch_capacity;
}

}  // namespace

extern "C" int lowconf_context_preprocess_create(int gpu_id, size_t bytes,
    void **stream_out, void **device_buffer_out) {
  if (!stream_out || !device_buffer_out || bytes == 0) return static_cast<int>(cudaErrorInvalidValue);
  *stream_out = nullptr;
  *device_buffer_out = nullptr;
  cudaError_t status = cudaSetDevice(gpu_id);
  if (status != cudaSuccess) return static_cast<int>(status);
  cudaStream_t stream = nullptr;
  status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) return static_cast<int>(status);
  void *buffer = nullptr;
  status = cudaMalloc(&buffer, bytes);
  if (status != cudaSuccess) {
    cudaStreamDestroy(stream);
    return static_cast<int>(status);
  }
  *stream_out = stream;
  *device_buffer_out = buffer;
  return static_cast<int>(cudaSuccess);
}

extern "C" int lowconf_context_preprocess_rgb(void *stream, const unsigned char *source,
    size_t source_pitch, void *device_buffer, unsigned int side,
    unsigned int batch_index, unsigned int batch_capacity, float scale) {
  if (!stream || !source || !device_buffer || side == 0 || batch_index >= batch_capacity)
    return static_cast<int>(cudaErrorInvalidValue);
  dim3 block(16, 16);
  dim3 grid((side + block.x - 1) / block.x, (side + block.y - 1) / block.y);
  rgb_hwc_to_normalized_nchw_fp32<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      source, source_pitch, static_cast<float*>(device_buffer), side,
      batch_index, batch_capacity, scale);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int lowconf_context_preprocess_sync(void *stream) {
  return stream ? static_cast<int>(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)))
                : static_cast<int>(cudaErrorInvalidValue);
}

extern "C" void lowconf_context_preprocess_destroy(void *stream, void *device_buffer) {
  if (device_buffer) cudaFree(device_buffer);
  if (stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}
