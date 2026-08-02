#include "dense_vector_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <new>

namespace mircuda::dense_vector {

constexpr int kWarps = 8;
constexpr int kRowsPerWarp = 4;
constexpr int kRowsPerBlock = kWarps * kRowsPerWarp;

template <typename Element>
struct Convert;

template <>
struct Convert<__half> {
  using Pair = __half2;

  __device__ static float to_float(__half value) { return __half2float(value); }
  __device__ static float2 pair_to_float(const __half* values) {
    return __half22float2(*reinterpret_cast<const Pair*>(values));
  }
  __device__ static float2 pair_bits_to_float(unsigned int values) {
    return __half22float2(*reinterpret_cast<const Pair*>(&values));
  }
  __device__ static __half from_float(float value) { return __float2half_rn(value); }
};

template <>
struct Convert<__nv_bfloat16> {
  using Pair = __nv_bfloat162;

  __device__ static float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }
  __device__ static float2 pair_to_float(const __nv_bfloat16* values) {
    return __bfloat1622float2(*reinterpret_cast<const Pair*>(values));
  }
  __device__ static float2 pair_bits_to_float(unsigned int values) {
    return __bfloat1622float2(*reinterpret_cast<const Pair*>(&values));
  }
  __device__ static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
  }
};

template <typename Element>
__global__ void kernel(const Element* input, const Element* weight,
                       Element* output, int n, int k, float alpha, float beta) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int first_row = blockIdx.x * kRowsPerBlock + warp * kRowsPerWarp;
  float sums[kRowsPerWarp] = {};
  if ((k & 7) == 0) {
    for (int column = lane * 8; column < k; column += 256) {
      const uint4 input_bits =
          *reinterpret_cast<const uint4*>(input + column);
      const float2 input_pairs[4] = {
          Convert<Element>::pair_bits_to_float(input_bits.x),
          Convert<Element>::pair_bits_to_float(input_bits.y),
          Convert<Element>::pair_bits_to_float(input_bits.z),
          Convert<Element>::pair_bits_to_float(input_bits.w),
      };
#pragma unroll
      for (int item = 0; item < kRowsPerWarp; ++item) {
        const int row = first_row + item;
        if (row < n) {
          const uint4 weight_bits =
              *reinterpret_cast<const uint4*>(weight + row * k + column);
          const unsigned int pairs[4] = {
              weight_bits.x, weight_bits.y, weight_bits.z, weight_bits.w};
#pragma unroll
          for (int pair = 0; pair < 4; ++pair) {
            const float2 values =
                Convert<Element>::pair_bits_to_float(pairs[pair]);
            sums[item] = fmaf(input_pairs[pair].x, values.x, sums[item]);
            sums[item] = fmaf(input_pairs[pair].y, values.y, sums[item]);
          }
        }
      }
    }
  } else if ((k & 1) == 0) {
    const int pairs = k / 2;
    for (int pair = lane; pair < pairs; pair += 32) {
      const float2 input_pair =
          Convert<Element>::pair_to_float(input + 2 * pair);
#pragma unroll
      for (int item = 0; item < kRowsPerWarp; ++item) {
        const int row = first_row + item;
        if (row < n) {
          const float2 values =
              Convert<Element>::pair_to_float(weight + row * k + 2 * pair);
          sums[item] = fmaf(input_pair.x, values.x, sums[item]);
          sums[item] = fmaf(input_pair.y, values.y, sums[item]);
        }
      }
    }
  } else {
    for (int column = lane; column < k; column += 32) {
      const float value = Convert<Element>::to_float(input[column]);
#pragma unroll
      for (int item = 0; item < kRowsPerWarp; ++item) {
        const int row = first_row + item;
        if (row < n) {
          const float scale = Convert<Element>::to_float(weight[row * k + column]);
          sums[item] = fmaf(value, scale, sums[item]);
        }
      }
    }
  }
#pragma unroll
  for (int item = 0; item < kRowsPerWarp; ++item) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      sums[item] += __shfl_down_sync(0xffffffffu, sums[item], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int item = 0; item < kRowsPerWarp; ++item) {
      const int row = first_row + item;
      if (row < n) {
        float result = alpha * sums[item];
        if (beta != 0.0f) {
          result = fmaf(beta, Convert<Element>::to_float(output[row]), result);
        }
        output[row] = Convert<Element>::from_float(result);
      }
    }
  }
}

template <typename Element>
int launch(Plan* plan, const void* input, const void* weight, void* output,
           float alpha, float beta) {
  const int blocks = (plan->n + kRowsPerBlock - 1) / kRowsPerBlock;
  kernel<<<blocks, kWarps * 32, 0, plan->stream>>>(
      static_cast<const Element*>(input), static_cast<const Element*>(weight),
      static_cast<Element*>(output), plan->n, plan->k, alpha, beta);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace mircuda::dense_vector

extern "C" int mircuda_dense_vector_create(int data_type, int n, int k,
                                             void* stream, void** output) {
  using namespace mircuda::dense_vector;
  if ((data_type != 0 && data_type != 1) || n <= 0 || k <= 0 ||
      stream == nullptr || output == nullptr) return -1;
  auto* plan = new (std::nothrow)
      Plan{data_type, n, k, static_cast<cudaStream_t>(stream)};
  if (plan == nullptr) return -2;
  *output = plan;
  return 0;
}

extern "C" int mircuda_dense_vector_execute(
    void* raw, void* stream, const void* input, const void* weight,
    void* output, float alpha, float beta) {
  using namespace mircuda::dense_vector;
  if (raw == nullptr || stream == nullptr || input == nullptr ||
      weight == nullptr || output == nullptr) return -1;
  auto* plan = static_cast<Plan*>(raw);
  if (static_cast<cudaStream_t>(stream) != plan->stream) return -1;
  return plan->data_type == 0
      ? launch<__half>(plan, input, weight, output, alpha, beta)
      : launch<__nv_bfloat16>(plan, input, weight, output, alpha, beta);
}

extern "C" void mircuda_dense_vector_destroy(void* raw) {
  delete static_cast<mircuda::dense_vector::Plan*>(raw);
}
