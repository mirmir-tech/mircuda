// SPDX-License-Identifier: Apache-2.0

#define MARLIN_NAMESPACE_NAME mircuda_marlin_dense
#include "vendor/dense_template.h"

namespace mircuda_marlin_dense {

#define MIRCUDA_DENSE_PARAMS                                                  \
  const int4* A, const int4* B, int4* C, int4* C_tmp, const int4* bias,      \
      const float* a_scales, const int4* scales, const float* global_scale,  \
      const int4* zero_points, const int* group_indices, int groups, int m,  \
      int n, int k, int lda, int* locks, bool has_bias, bool atomic_reduce,  \
      bool fp32_reduce, int maximum_shared

using Kernel = void (*)(MIRCUDA_DENSE_PARAMS);

constexpr auto nvfp4_bf16_n128_k128 = Marlin<
    vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
    vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 4, 1, false>;
constexpr auto nvfp4_bf16_n128_k64 = Marlin<
    vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
    vllm::kFE4M3fn.id(), 128, 1, 8, 4, true, 4, 1, false>;
constexpr auto nvfp4_bf16_n64_k128 = Marlin<
    vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
    vllm::kFE4M3fn.id(), 128, 1, 4, 8, true, 4, 1, false>;

template __global__ void Marlin<
    vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
    vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 4, 1, false>(MIRCUDA_DENSE_PARAMS);
template __global__ void Marlin<
    vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
    vllm::kFE4M3fn.id(), 128, 1, 8, 4, true, 4, 1, false>(MIRCUDA_DENSE_PARAMS);
template __global__ void Marlin<
    vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
    vllm::kFE4M3fn.id(), 128, 1, 4, 8, true, 4, 1, false>(MIRCUDA_DENSE_PARAMS);

Kernel select_kernel(int selection, int* threads) {
  if (selection == 0) {
    *threads = 256;
    return nvfp4_bf16_n128_k128;
  }
  *threads = 128;
  if (selection == 1) return nvfp4_bf16_n128_k64;
  if (selection == 2) return nvfp4_bf16_n64_k128;
  return nullptr;
}

}  // namespace mircuda_marlin_dense

extern "C" int mircuda_marlin_nvfp4_dense_execute(
    void* stream, const void* input, const void* weight, void* output,
    void* temporary, const void* scales, const float* global_scale, int tokens,
    int output_features, int input_features, int thread_config, int* locks,
    bool atomic_reduce) {
  int device = 0;
  int sms = 0;
  int maximum_shared = 0;
  int threads = 0;
  auto kernel = mircuda_marlin_dense::select_kernel(thread_config, &threads);
  if (kernel == nullptr) return static_cast<int>(cudaErrorInvalidValue);
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&maximum_shared,
                             cudaDevAttrMaxSharedMemoryPerBlockOptin, device) !=
          cudaSuccess)
    return static_cast<int>(cudaGetLastError());
  auto status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maximum_shared);
  if (status != cudaSuccess) return static_cast<int>(status);
  kernel<<<sms, threads, maximum_shared, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const int4*>(input), static_cast<const int4*>(weight),
      static_cast<int4*>(output), static_cast<int4*>(temporary), nullptr,
      nullptr, static_cast<const int4*>(scales), global_scale, nullptr, nullptr,
      input_features / 16, tokens, output_features, input_features,
      input_features, locks, false, atomic_reduce, true, maximum_shared);
  return static_cast<int>(cudaGetLastError());
}
