#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>

#include <new>

namespace mircuda::fp4_vector {

using Output = cutlass::bfloat16_t;
using Scale = cutlass::float_ue4m3_t;

constexpr int kWarpSize = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kWarpSize * kWarps;
constexpr int kValuesPerLane = 16;
constexpr int kValuesPerTile = kWarpSize * kValuesPerLane;
constexpr int kPackedPerTile = kValuesPerTile / 2;

struct Plan {
  int n;
  int k;
  cudaStream_t stream;
};

struct SharedInput {
  alignas(16) uint8_t values[kPackedPerTile];
  alignas(16) uint8_t scales[kWarpSize];
};

__device__ __forceinline__ int scale_index(int row, int scale_column,
                                            int columns) {
  const int blocks_k = columns / 64;
  const int row_block = ((row >> 7) * blocks_k) << 9;
  const int row_offset = ((row & 31) << 4) + (((row & 127) >> 5) << 2);
  const int column_offset = ((scale_column >> 2) << 9) + (scale_column & 3);
  return row_block + row_offset + column_offset;
}

__device__ __forceinline__ uint32_t decode_fp4_pair(uint32_t packed) {
  uint32_t decoded;
  asm volatile(
      "{\n"
      ".reg .b8 source, unused1, unused2, unused3;\n"
      "mov.b32 {source, unused1, unused2, unused3}, %1;\n"
      "cvt.rn.f16x2.e2m1x2 %0, source;\n"
      "}\n"
      : "=r"(decoded)
      : "r"(packed));
  return decoded;
}

__device__ __forceinline__ __half2 as_half2(uint32_t bits) {
  union {
    uint32_t bits;
    __half2 values;
  } value{bits};
  return value.values;
}

__device__ __forceinline__ float dot_16(uint64_t left, uint64_t right) {
  float sum = 0.0f;
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const auto shift = pair * 8;
    const uint32_t left_pair = static_cast<uint32_t>((left >> shift) & 0xff);
    const uint32_t right_pair = static_cast<uint32_t>((right >> shift) & 0xff);
    const __half2 left_values = as_half2(decode_fp4_pair(left_pair));
    const __half2 right_values = as_half2(decode_fp4_pair(right_pair));
    sum = __half2float(__low2half(left_values)) *
              __half2float(__low2half(right_values)) +
          sum;
    sum = __half2float(__high2half(left_values)) *
              __half2float(__high2half(right_values)) +
          sum;
  }
  return sum;
}

__device__ __forceinline__ float decode_scale(uint8_t bits) {
  return static_cast<float>(Scale::bitcast(bits));
}

__global__ void fp4_vector(const uint8_t* input, const uint8_t* input_scales,
                           const uint8_t* weight,
                           const uint8_t* weight_scales, Output* output, int n,
                           int k, float alpha) {
  __shared__ SharedInput shared;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row = blockIdx.x * kWarps + warp;
  float accumulator = 0.0f;

  for (int tile = 0; tile < k; tile += kValuesPerTile) {
    const int packed_remaining = (k - tile) / 2;
    if (threadIdx.x < packed_remaining && threadIdx.x < kPackedPerTile) {
      shared.values[threadIdx.x] = input[tile / 2 + threadIdx.x];
    }
    if (threadIdx.x < kWarpSize && tile + threadIdx.x * kValuesPerLane < k) {
      const int column = tile / kValuesPerLane + threadIdx.x;
      shared.scales[threadIdx.x] = input_scales[scale_index(0, column, k)];
    }
    __syncthreads();

    const int element = tile + lane * kValuesPerLane;
    if (row < n && element < k) {
      const auto* shared_values =
          reinterpret_cast<const uint64_t*>(shared.values + lane * 8);
      const auto* row_weight = weight + row * k / 2;
      const auto* packed_weight =
          reinterpret_cast<const uint64_t*>(row_weight + element / 2);
      const int column = element / kValuesPerLane;
      const float scales = decode_scale(shared.scales[lane]) *
                           decode_scale(weight_scales[scale_index(row, column, k)]);
      accumulator += dot_16(*shared_values, *packed_weight) * scales;
    }
    __syncthreads();
  }

  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    accumulator += __shfl_down_sync(0xffffffff, accumulator, offset);
  }
  if (lane == 0 && row < n) output[row] = Output(accumulator * alpha);
}

}  // namespace mircuda::fp4_vector

extern "C" int mircuda_fp4_vector_create(int n, int k, void* stream,
                                           void** output) {
  using namespace mircuda::fp4_vector;
  if (n <= 0 || k <= 0 || k % 64 != 0 || stream == nullptr ||
      output == nullptr) {
    return -1;
  }
  auto* plan =
      new (std::nothrow) Plan{n, k, static_cast<cudaStream_t>(stream)};
  if (plan == nullptr) return -2;
  *output = plan;
  return 0;
}

extern "C" int mircuda_fp4_vector_execute(
    void* raw, void* stream, const void* input, const void* input_scales,
    const void* weight, const void* weight_scales, void* output, float alpha) {
  using namespace mircuda::fp4_vector;
  if (raw == nullptr || stream == nullptr || input == nullptr ||
      input_scales == nullptr || weight == nullptr || weight_scales == nullptr ||
      output == nullptr) {
    return -1;
  }
  auto* plan = static_cast<Plan*>(raw);
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (cuda_stream != plan->stream) return -1;
  const dim3 block(kThreads);
  const dim3 grid((plan->n + kWarps - 1) / kWarps);
  fp4_vector<<<grid, block, 0, cuda_stream>>>(
      static_cast<const uint8_t*>(input),
      static_cast<const uint8_t*>(input_scales),
      static_cast<const uint8_t*>(weight),
      static_cast<const uint8_t*>(weight_scales), static_cast<Output*>(output),
      plan->n, plan->k, alpha);
  return static_cast<int>(cudaPeekAtLastError());
}

extern "C" void mircuda_fp4_vector_destroy(void* raw) {
  delete static_cast<mircuda::fp4_vector::Plan*>(raw);
}
