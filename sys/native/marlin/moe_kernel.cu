// SPDX-License-Identifier: Apache-2.0

#define MARLIN_NAMESPACE_NAME mircuda_marlin_moe
#include "vendor/moe_template.h"

namespace mircuda_marlin_moe {

using NvFp4Bf16Kernel = void (*)(
    const int4*, const int4*, int4*, int4*, const int4*, const float*,
    const int4*, const float*, const int4*, const int*, const int32_t*,
    const int32_t*, const int32_t*, const float*, int, bool, int, int, int,
    int, int*, bool, bool, bool);

constexpr auto nvfp4_bf16_block8_n128_k128 = Marlin<
    vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
    vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 4, 1, false>;
constexpr auto nvfp4_bf16_block8_n128_k64 = Marlin<
    vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
    vllm::kFE4M3fn.id(), 128, 1, 8, 4, true, 4, 1, false>;
constexpr auto nvfp4_bf16_block8_n64_k128 = Marlin<
    vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
    vllm::kFE4M3fn.id(), 128, 1, 4, 8, true, 4, 1, false>;

template __global__ void Marlin<
    vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
    vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 4, 1, false>(
    const int4*, const int4*, int4*, int4*, const int4*, const float*,
    const int4*, const float*, const int4*, const int*, const int32_t*,
    const int32_t*, const int32_t*, const float*, int, bool, int, int, int,
    int, int*, bool, bool, bool);

template __global__ void Marlin<
    vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
    vllm::kFE4M3fn.id(), 128, 1, 8, 4, true, 4, 1, false>(
    const int4*, const int4*, int4*, int4*, const int4*, const float*,
    const int4*, const float*, const int4*, const int*, const int32_t*,
    const int32_t*, const int32_t*, const float*, int, bool, int, int, int,
    int, int*, bool, bool, bool);

template __global__ void Marlin<
    vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
    vllm::kFE4M3fn.id(), 128, 1, 4, 8, true, 4, 1, false>(
    const int4*, const int4*, int4*, int4*, const int4*, const float*,
    const int4*, const float*, const int4*, const int*, const int32_t*,
    const int32_t*, const int32_t*, const float*, int, bool, int, int, int,
    int, int*, bool, bool, bool);

struct LaunchConfig {
  NvFp4Bf16Kernel kernel = nullptr;
  int device = -1;
  int threads = 0;
  int shared = 0;
  int blocks = 0;
  cudaError_t status = cudaSuccess;
};

int shared_bytes(int thread_n, int thread_k) {
  constexpr int stages = 4;
  constexpr int thread_m = 16;
  const int metadata = thread_m * 16;
  const int activation = stages * thread_m * thread_k * 2;
  const int weight = stages * (thread_k * thread_n / 8) * 4;
  const int reduction = thread_m * (thread_n + 8) * 2;
  const int bias = thread_n * 2;
  const int overlap = (weight < reduction ? weight : reduction) + bias;
  const int temporary = weight > reduction ? weight : reduction;
  const int workspace = temporary > overlap ? temporary : overlap;
  const int scale_groups = (thread_k + 15) / 16;
  const int scales = scale_groups * thread_n * 2 * stages;
  constexpr int allocation_granularity = 4 * 1024;
  const int required = metadata + activation + workspace + scales;
  return ((required + allocation_granularity - 1) / allocation_granularity) *
         allocation_granularity;
}

LaunchConfig prepare_launch(int selection, int device) {
  LaunchConfig result;
  result.device = device;
  int thread_n = 0;
  int thread_k = 0;
  if (selection == 0) {
    result.kernel = nvfp4_bf16_block8_n128_k128;
    result.threads = 256;
    thread_n = 128;
    thread_k = 128;
  } else if (selection == 1) {
    result.kernel = nvfp4_bf16_block8_n128_k64;
    result.threads = 128;
    thread_n = 128;
    thread_k = 64;
  } else if (selection == 2) {
    result.kernel = nvfp4_bf16_block8_n64_k128;
    result.threads = 128;
    thread_n = 64;
    thread_k = 128;
  } else {
    result.status = cudaErrorInvalidValue;
    return result;
  }
  int sms = 0;
  int maximum = 0;
  result.shared = shared_bytes(thread_n, thread_k);
  result.status = cudaDeviceGetAttribute(
      &sms, cudaDevAttrMultiProcessorCount, device);
  if (result.status != cudaSuccess) return result;
  result.status = cudaDeviceGetAttribute(
      &maximum, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (result.status != cudaSuccess) return result;
  if (result.shared > maximum) {
    result.status = cudaErrorInvalidConfiguration;
    return result;
  }
  result.status = cudaFuncSetAttribute(
      result.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, result.shared);
  if (result.status != cudaSuccess) return result;
  int active = 0;
  result.status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, result.kernel, result.threads, result.shared);
  if (result.status != cudaSuccess) return result;
  result.blocks = sms * active;
  return result;
}

}  // namespace mircuda_marlin_moe

extern "C" int mircuda_marlin_nvfp4_moe_execute(
    void* stream, const void* input, const void* weight, void* output,
    void* temporary, const void* scales, const void* global_scales,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* padded_routes, const float* topk_weights, int top_k,
    bool multiply_topk, int tokens, int output_features, int input_features,
    int thread_config, int* locks, bool atomic_reduce) {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return static_cast<int>(cudaGetLastError());
  }
  thread_local mircuda_marlin_moe::LaunchConfig cache[3];
  if (thread_config < 0 || thread_config >= 3) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  auto& launch = cache[thread_config];
  if (launch.kernel == nullptr || launch.device != device) {
    launch = mircuda_marlin_moe::prepare_launch(thread_config, device);
  }
  if (launch.status != cudaSuccess) return static_cast<int>(launch.status);
  launch.kernel<<<launch.blocks, launch.threads, launch.shared,
                  static_cast<cudaStream_t>(stream)>>>(
      static_cast<const int4*>(input), static_cast<const int4*>(weight),
      static_cast<int4*>(output), static_cast<int4*>(temporary), nullptr,
      nullptr, static_cast<const int4*>(scales),
      static_cast<const float*>(global_scales), nullptr, nullptr,
      sorted_token_ids, expert_ids, padded_routes, topk_weights, top_k,
      multiply_topk, input_features / 16, tokens, output_features,
      input_features, locks, false, atomic_reduce, true);
  return static_cast<int>(cudaGetLastError());
}
