// SPDX-License-Identifier: Apache-2.0

#define MARLIN_NAMESPACE_NAME mircuda_marlin_mxfp4_moe
#include "vendor/moe_template.h"

namespace mircuda_marlin_mxfp4_moe {

using Kernel = void (*)(
    const int4*, const int4*, int4*, int4*, const int4*, const float*,
    const int4*, const float*, const int4*, const int*, const int32_t*,
    const int32_t*, const int32_t*, const float*, int, bool, int, int, int,
    int, int*, bool, bool, bool);

#define MIRCUDA_MXFP4_KERNEL(THREADS, M_BLOCKS, N_BLOCKS, K_BLOCKS, BLOCK8) \
  Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),                        \
         vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(), THREADS, M_BLOCKS,   \
         N_BLOCKS, K_BLOCKS, BLOCK8, 4, 2, false>

constexpr auto n128_k128 = MIRCUDA_MXFP4_KERNEL(256, 1, 8, 8, true);
constexpr auto n128_k64 = MIRCUDA_MXFP4_KERNEL(128, 1, 8, 4, true);
constexpr auto n64_k128 = MIRCUDA_MXFP4_KERNEL(128, 1, 4, 8, true);
constexpr auto m64_n256_k64 = MIRCUDA_MXFP4_KERNEL(256, 4, 16, 4, false);
constexpr auto m64_n128_k64 = MIRCUDA_MXFP4_KERNEL(128, 4, 8, 4, false);
constexpr auto m64_n64_k128 = MIRCUDA_MXFP4_KERNEL(128, 4, 4, 8, false);

template __global__ void MIRCUDA_MXFP4_KERNEL(256, 1, 8, 8, true)(
    const int4*, const int4*, int4*, int4*, const int4*, const float*,
    const int4*, const float*, const int4*, const int*, const int32_t*,
    const int32_t*, const int32_t*, const float*, int, bool, int, int, int,
    int, int*, bool, bool, bool);
template __global__ void MIRCUDA_MXFP4_KERNEL(128, 1, 8, 4, true)(
    const int4*, const int4*, int4*, int4*, const int4*, const float*,
    const int4*, const float*, const int4*, const int*, const int32_t*,
    const int32_t*, const int32_t*, const float*, int, bool, int, int, int,
    int, int*, bool, bool, bool);
template __global__ void MIRCUDA_MXFP4_KERNEL(128, 1, 4, 8, true)(
    const int4*, const int4*, int4*, int4*, const int4*, const float*,
    const int4*, const float*, const int4*, const int*, const int32_t*,
    const int32_t*, const int32_t*, const float*, int, bool, int, int, int,
    int, int*, bool, bool, bool);
template __global__ void MIRCUDA_MXFP4_KERNEL(256, 4, 16, 4, false)(
    const int4*, const int4*, int4*, int4*, const int4*, const float*,
    const int4*, const float*, const int4*, const int*, const int32_t*,
    const int32_t*, const int32_t*, const float*, int, bool, int, int, int,
    int, int*, bool, bool, bool);
template __global__ void MIRCUDA_MXFP4_KERNEL(128, 4, 8, 4, false)(
    const int4*, const int4*, int4*, int4*, const int4*, const float*,
    const int4*, const float*, const int4*, const int*, const int32_t*,
    const int32_t*, const int32_t*, const float*, int, bool, int, int, int,
    int, int*, bool, bool, bool);
template __global__ void MIRCUDA_MXFP4_KERNEL(128, 4, 4, 8, false)(
    const int4*, const int4*, int4*, int4*, const int4*, const float*,
    const int4*, const float*, const int4*, const int*, const int32_t*,
    const int32_t*, const int32_t*, const float*, int, bool, int, int, int,
    int, int*, bool, bool, bool);

struct LaunchConfig {
  Kernel kernel = nullptr;
  int device = -1;
  int threads = 0;
  int shared = 0;
  int blocks = 0;
  cudaError_t status = cudaSuccess;
};

int shared_bytes(int thread_m, int thread_n, int thread_k) {
  constexpr int stages = 4;
  const int metadata = thread_m * 16;
  const int activation = stages * thread_m * thread_k * 2;
  const int weight = stages * (thread_k * thread_n / 8) * 4;
  const int reduction = thread_m * (thread_n + 8) * 2;
  const int workspace = weight > reduction ? weight : reduction;
  const int scales = ((thread_k + 31) / 32) * thread_n * 2 * stages;
  constexpr int granularity = 4 * 1024;
  const int required = metadata + activation + workspace + scales;
  return ((required + granularity - 1) / granularity) * granularity;
}

LaunchConfig prepare_launch(int selection, int device) {
  LaunchConfig result;
  result.device = device;
  int thread_n = 0;
  int thread_k = 0;
  int thread_m = 16;
  if (selection == 0) {
    result.kernel = n128_k128;
    result.threads = 256;
    thread_n = 128;
    thread_k = 128;
  } else if (selection == 1) {
    result.kernel = n128_k64;
    result.threads = 128;
    thread_n = 128;
    thread_k = 64;
  } else if (selection == 2) {
    result.kernel = n64_k128;
    result.threads = 128;
    thread_n = 64;
    thread_k = 128;
  } else if (selection == 3) {
    result.kernel = m64_n256_k64;
    result.threads = 256;
    thread_m = 64;
    thread_n = 256;
    thread_k = 64;
  } else if (selection == 4) {
    result.kernel = m64_n128_k64;
    result.threads = 128;
    thread_m = 64;
    thread_n = 128;
    thread_k = 64;
  } else if (selection == 5) {
    result.kernel = m64_n64_k128;
    result.threads = 128;
    thread_m = 64;
    thread_n = 64;
    thread_k = 128;
  } else {
    result.status = cudaErrorInvalidValue;
    return result;
  }
  int sms = 0;
  int maximum = 0;
  result.shared = shared_bytes(thread_m, thread_n, thread_k);
  result.status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (result.status != cudaSuccess) return result;
  result.status = cudaDeviceGetAttribute(
      &maximum, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (result.status != cudaSuccess || result.shared > maximum) {
    result.status = cudaErrorInvalidConfiguration;
    return result;
  }
  result.status = cudaFuncSetAttribute(
      result.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, result.shared);
  if (result.status != cudaSuccess) return result;
  int active = 0;
  result.status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, result.kernel, result.threads, result.shared);
  if (result.status == cudaSuccess) result.blocks = sms * active;
  return result;
}

}  // namespace mircuda_marlin_mxfp4_moe

extern "C" int mircuda_marlin_mxfp4_moe_execute(
    void* stream, const void* input, const void* weight, void* output,
    void* temporary, const void* scales, const int32_t* sorted_token_ids,
    const int32_t* expert_ids, const int32_t* padded_routes,
    const float* topk_weights, int top_k, bool multiply_topk, int tokens,
    int output_features, int input_features, int thread_config, int* locks,
    bool atomic_reduce) {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess)
    return static_cast<int>(cudaGetLastError());
  thread_local mircuda_marlin_mxfp4_moe::LaunchConfig cache[6];
  if (thread_config < 0 || thread_config >= 6)
    return static_cast<int>(cudaErrorInvalidValue);
  auto& launch = cache[thread_config];
  if (launch.kernel == nullptr || launch.device != device)
    launch = mircuda_marlin_mxfp4_moe::prepare_launch(thread_config, device);
  if (launch.status != cudaSuccess) return static_cast<int>(launch.status);
  launch.kernel<<<launch.blocks, launch.threads, launch.shared,
                  static_cast<cudaStream_t>(stream)>>>(
      static_cast<const int4*>(input), static_cast<const int4*>(weight),
      static_cast<int4*>(output), static_cast<int4*>(temporary), nullptr,
      nullptr, static_cast<const int4*>(scales), nullptr, nullptr, nullptr,
      sorted_token_ids, expert_ids, padded_routes, topk_weights, top_k,
      multiply_topk, input_features / 32, tokens, output_features,
      input_features, locks, false, atomic_reduce, true);
  return static_cast<int>(cudaGetLastError());
}
