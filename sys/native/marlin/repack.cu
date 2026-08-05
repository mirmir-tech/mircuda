// SPDX-License-Identifier: Apache-2.0
// Layout conversions and route preparation for the vendored Marlin kernel.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ uint32_t nibble(const uint8_t* left, const uint8_t* right,
                           int expert, int n, int k, int size_n, int size_k) {
  const bool paired = right != nullptr;
  const int source_n = paired ? size_n / 2 : size_n;
  const uint8_t* source = paired && n >= source_n ? right : left;
  const int row = paired && n >= source_n ? n - source_n : n;
  const uint8_t packed =
      source[(expert * source_n + row) * (size_k / 2) + k / 2];
  return (packed >> ((k & 1) * 4)) & 0x0f;
}

__global__ void repack_nvfp4(const uint8_t* left, const uint8_t* right,
                             uint32_t* output, int experts, int size_n,
                             int size_k) {
  constexpr int tile_k = 16;
  constexpr int tile_n = 64;
  constexpr int words_per_tile = tile_k * tile_n / 8;
  const int tiles_n = size_n / tile_n;
  const int words_per_expert = size_n * size_k / 8;
  const int word = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = experts * words_per_expert;
  if (word >= total) return;

  const int expert = word / words_per_expert;
  const int expert_word = word % words_per_expert;
  const int tile = expert_word / words_per_tile;
  const int local = expert_word % words_per_tile;
  const int tile_k_id = tile / tiles_n;
  const int tile_n_id = tile % tiles_n;
  const int lane = local / 4;
  const int warp = local % 4;
  const int column = lane / 4;
  const int row = (lane % 4) * 2;
  const int base_n = tile_n_id * tile_n + warp * 16 + column;
  const int base_k = tile_k_id * tile_k;
  constexpr int offsets[4] = {0, 1, 8, 9};
  uint32_t values[8];
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const int k = base_k + row + offsets[index];
    values[index] = nibble(left, right, expert, base_n, k, size_n, size_k);
    values[4 + index] =
        nibble(left, right, expert, base_n + 8, k, size_n, size_k);
  }
  constexpr int order[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  uint32_t packed = 0;
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    packed |= values[order[index]] << (index * 4);
  }
  output[word] = packed;
}

__device__ float fp8_value(uint8_t bits) {
  __nv_fp8_e4m3 value;
  value.__x = bits;
  return static_cast<float>(value);
}

__global__ void maximum_scale(const uint8_t* left, const uint8_t* right,
                              uint32_t* maximum, int elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  float value = fp8_value(left[index]);
  if (right != nullptr) value = fmaxf(value, fp8_value(right[index]));
  atomicMax(maximum, __float_as_uint(value));
}

__device__ int scale_destination(int linear) {
  const int chunk = linear / 64;
  const int local = linear % 64;
  const int permuted = chunk * 64 + 8 * (local % 8) + local / 8;
  constexpr int reorder[4] = {0, 2, 1, 3};
  return (permuted / 4) * 4 + reorder[permuted % 4];
}

__global__ void prepare_scales(const uint8_t* left, const uint8_t* right,
                               const float* globals, uint8_t* output,
                               float* output_globals, const uint32_t* maximum,
                               int experts, int size_n, int size_k) {
  const int groups = size_k / 16;
  const int elements_per_expert = size_n * groups;
  const int total = experts * elements_per_expert;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const float max_value = __uint_as_float(maximum[0]);
  const float max_scaled = max_value * 128.0f;
  const float factor = max_scaled > 0.0f && max_scaled < 57344.0f
                           ? exp2f(floorf(log2f(57344.0f / max_scaled)))
                           : 1.0f;
  if (index < total) {
    const int expert = index / elements_per_expert;
    const int logical = index % elements_per_expert;
    const int group = logical / size_n;
    const int n = logical % size_n;
    const bool paired = right != nullptr;
    const int source_n = paired ? size_n / 2 : size_n;
    const uint8_t* source = paired && n >= source_n ? right : left;
    const int source_row = paired && n >= source_n ? n - source_n : n;
    const int source_index =
        (expert * source_n + source_row) * groups + group;
    __half converted = __float2half_rn(fp8_value(source[source_index]) * factor);
    converted = __hmul(converted, __float2half_rn(128.0f));
    uint16_t bits = __half_as_ushort(converted);
    const uint8_t encoded = __half2float(converted) < 2.0f
                                ? 0
                                : static_cast<uint8_t>((bits << 1) >> 8);
    output[expert * elements_per_expert + scale_destination(logical)] = encoded;
  }
  if (index < experts) {
    output_globals[index] = globals[index] * exp2f(119.0f) / factor;
  }
}

__global__ void prepare_routes(const uint32_t* selected,
                               const __nv_bfloat16* routing, int32_t* sorted,
                               int32_t* expert_ids, int32_t* padded,
                               int32_t* offsets, float* routing_f32,
                               int assignments, int experts) {
  extern __shared__ int32_t shared[];
  int32_t* counts = shared;
  int32_t* cursors = shared + experts;
  const int capacity = assignments + experts * 7;
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    counts[expert] = 0;
    cursors[expert] = 0;
  }
  for (int index = threadIdx.x; index < capacity; index += blockDim.x) {
    sorted[index] = assignments;
  }
  for (int assignment = threadIdx.x; assignment < assignments;
       assignment += blockDim.x) {
    const uint32_t expert = selected[assignment];
    if (expert < static_cast<uint32_t>(experts)) atomicAdd(counts + expert, 1);
    routing_f32[assignment] = __bfloat162float(routing[assignment]);
  }
  __syncthreads();
  if (experts <= blockDim.x) {
    const int expert = threadIdx.x;
    if (expert < experts) cursors[expert] = ((counts[expert] + 7) / 8) * 8;
    __syncthreads();
    for (int stride = 1; stride < experts; stride *= 2) {
      const int value = expert < experts && expert >= stride
                            ? cursors[expert - stride]
                            : 0;
      __syncthreads();
      if (expert < experts) cursors[expert] += value;
      __syncthreads();
    }
    if (expert < experts) {
      const int count = ((counts[expert] + 7) / 8) * 8;
      const int offset = cursors[expert] - count;
      offsets[expert] = offset;
      for (int block = 0; block < count / 8; ++block) {
        expert_ids[offset / 8 + block] = expert;
      }
      if (expert == experts - 1) padded[0] = cursors[expert];
    }
  } else if (threadIdx.x == 0) {
    int offset = 0;
    for (int expert = 0; expert < experts; ++expert) {
      offsets[expert] = offset;
      const int blocks = (counts[expert] + 7) / 8;
      for (int block = 0; block < blocks; ++block) {
        expert_ids[offset / 8 + block] = expert;
      }
      offset += blocks * 8;
    }
    padded[0] = offset;
  }
  __syncthreads();
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    cursors[expert] = 0;
  }
  __syncthreads();
  for (int assignment = threadIdx.x; assignment < assignments;
       assignment += blockDim.x) {
    const uint32_t expert = selected[assignment];
    if (expert >= static_cast<uint32_t>(experts)) continue;
    const int rank = atomicAdd(cursors + expert, 1);
    sorted[offsets[expert] + rank] = assignment;
  }
}

int launch_repack(void* stream, const void* left, const void* right,
                  void* output, int experts, int size_n, int size_k) {
  const int words = experts * size_n * size_k / 8;
  int threads = 32;
  while (threads < experts && threads < 1024) threads *= 2;
  repack_nvfp4<<<(words + threads - 1) / threads, threads, 0,
                  static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(left), static_cast<const uint8_t*>(right),
      static_cast<uint32_t*>(output), experts, size_n, size_k);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" int mircuda_marlin_nvfp4_repack(
    void* stream, const void* input, void* output, int experts, int size_n,
    int size_k) {
  return launch_repack(stream, input, nullptr, output, experts, size_n, size_k);
}

extern "C" int mircuda_marlin_nvfp4_repack_pair(
    void* stream, const void* left, const void* right, void* output,
    int experts, int size_n, int size_k) {
  return launch_repack(stream, left, right, output, experts, size_n, size_k);
}

extern "C" int mircuda_marlin_nvfp4_prepare_scales(
    void* stream, const void* left, const void* right, const void* globals,
    void* output, void* output_globals, void* maximum, int experts,
    int size_n, int size_k) {
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  cudaError_t status = cudaMemsetAsync(maximum, 0, sizeof(uint32_t), cuda_stream);
  if (status != cudaSuccess) return static_cast<int>(status);
  const int input_elements = experts * (right == nullptr ? size_n : size_n / 2) *
                             size_k / 16;
  constexpr int threads = 256;
  maximum_scale<<<(input_elements + threads - 1) / threads, threads, 0,
                  cuda_stream>>>(static_cast<const uint8_t*>(left),
                                 static_cast<const uint8_t*>(right),
                                 static_cast<uint32_t*>(maximum), input_elements);
  const int output_elements = experts * size_n * size_k / 16;
  prepare_scales<<<(output_elements + threads - 1) / threads, threads, 0,
                   cuda_stream>>>(
      static_cast<const uint8_t*>(left), static_cast<const uint8_t*>(right),
      static_cast<const float*>(globals), static_cast<uint8_t*>(output),
      static_cast<float*>(output_globals), static_cast<const uint32_t*>(maximum),
      experts, size_n, size_k);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int mircuda_marlin_prepare_moe_routes(
    void* stream, const void* selected, const void* routing, void* sorted,
    void* expert_ids, void* padded, void* offsets, void* routing_f32,
    int assignments, int experts) {
  constexpr int threads = 256;
  prepare_routes<<<1, threads, experts * 2 * sizeof(int32_t),
                   static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint32_t*>(selected),
      static_cast<const __nv_bfloat16*>(routing), static_cast<int32_t*>(sorted),
      static_cast<int32_t*>(expert_ids), static_cast<int32_t*>(padded),
      static_cast<int32_t*>(offsets), static_cast<float*>(routing_f32),
      assignments, experts);
  return static_cast<int>(cudaGetLastError());
}
