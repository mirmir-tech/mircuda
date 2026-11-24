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

// One block per row, its warps splitting the input columns (k % 8 == 0). Few
// output rows (routers) get one block each instead of one 32-row block; the
// plan's kernel, measured by the caller's tuner, selects it over `kernel`.
template <typename Element>
__global__ void row_kernel(const Element* input, const Element* weight,
                           Element* output, int n, int k, float alpha,
                           float beta) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x;
  if (row >= n) return;
  const uint4* in = reinterpret_cast<const uint4*>(input);
  const uint4* weights = reinterpret_cast<const uint4*>(weight + static_cast<long long>(row) * k);
  const int vectors = k / 8;
  float sum = 0.0f;
  for (int vector = threadIdx.x; vector < vectors; vector += kWarps * 32) {
    const uint4 input_bits = in[vector];
    const uint4 weight_bits = weights[vector];
    const unsigned int inputs[4] = {input_bits.x, input_bits.y, input_bits.z, input_bits.w};
    const unsigned int values[4] = {weight_bits.x, weight_bits.y, weight_bits.z, weight_bits.w};
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const float2 x = Convert<Element>::pair_bits_to_float(inputs[pair]);
      const float2 w = Convert<Element>::pair_bits_to_float(values[pair]);
      sum = fmaf(x.x, w.x, sum);
      sum = fmaf(x.y, w.y, sum);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  __shared__ float warps[kWarps];
  if (lane == 0) warps[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
#pragma unroll
    for (int item = 0; item < kWarps; ++item) total += warps[item];
    float result = alpha * total;
    if (beta != 0.0f) {
      result = fmaf(beta, Convert<Element>::to_float(output[row]), result);
    }
    output[row] = Convert<Element>::from_float(result);
  }
}

// Counts one arrived slice: releases this thread's partial and acquires the
// partials of slices that arrived earlier, without a full `__threadfence`.
__device__ unsigned int arrive(unsigned int* counter) {
  unsigned int previous;
  asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
               : "=r"(previous)
               : "l"(counter)
               : "memory");
  return previous;
}

// `n * splits` blocks: each reduces one balanced slice of one row's k / 8
// vectors into an FP32 partial; the last block to arrive for a row sums that
// row's partials in slice order, so results do not depend on block timing.
// Counters return to zero after every launch.
template <typename Element>
__global__ void split_kernel(const Element* input, const Element* weight,
                             Element* output, int n, int k, int splits,
                             unsigned int* counters, float* partials,
                             float alpha, float beta) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x / splits;
  const int slice = blockIdx.x - row * splits;
  if (row >= n) return;
  const long long vectors = k / 8;
  const int first = static_cast<int>(slice * vectors / splits);
  const int last = static_cast<int>((slice + 1) * vectors / splits);
  const uint4* in = reinterpret_cast<const uint4*>(input);
  const uint4* weights = reinterpret_cast<const uint4*>(weight + static_cast<long long>(row) * k);
  float sum = 0.0f;
#pragma unroll 4
  for (int vector = first + threadIdx.x; vector < last; vector += kWarps * 32) {
    const uint4 input_bits = in[vector];
    const uint4 weight_bits = weights[vector];
    const unsigned int inputs[4] = {input_bits.x, input_bits.y, input_bits.z, input_bits.w};
    const unsigned int values[4] = {weight_bits.x, weight_bits.y, weight_bits.z, weight_bits.w};
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const float2 x = Convert<Element>::pair_bits_to_float(inputs[pair]);
      const float2 w = Convert<Element>::pair_bits_to_float(values[pair]);
      sum = fmaf(x.x, w.x, sum);
      sum = fmaf(x.y, w.y, sum);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  __shared__ float warps[kWarps];
  if (lane == 0) warps[warp] = sum;
  __syncthreads();
  if (warp != 0) return;
  float total = 0.0f;
  if (lane == 0) {
#pragma unroll
    for (int item = 0; item < kWarps; ++item) total += warps[item];
  }
  if (splits > 1) {
    unsigned int arrived = 0;
    if (lane == 0) {
      partials[static_cast<long long>(row) * splits + slice] = total;
      arrived = arrive(counters + row);
    }
    arrived = __shfl_sync(0xffffffffu, arrived, 0);
    if (arrived != static_cast<unsigned int>(splits - 1)) return;
    // Extends lane 0's acquire to the lanes that read the partials.
    __syncwarp();
    const float* row_partials = partials + static_cast<long long>(row) * splits;
    total = 0.0f;
    for (int item = lane; item < splits; item += 32) {
      total += __ldcg(row_partials + item);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(0xffffffffu, total, offset);
    }
    if (lane == 0) counters[row] = 0;
  }
  if (lane == 0) {
    float result = alpha * total;
    if (beta != 0.0f) {
      result = fmaf(beta, Convert<Element>::to_float(output[row]), result);
    }
    output[row] = Convert<Element>::from_float(result);
  }
}

template <typename Element>
int launch(Plan* plan, const void* input, const void* weight, void* output,
           float alpha, float beta) {
  const auto* typed_input = static_cast<const Element*>(input);
  const auto* typed_weight = static_cast<const Element*>(weight);
  auto* typed_output = static_cast<Element*>(output);
  if (plan->kernel == Kernel::SplitK) {
    auto* counters = static_cast<unsigned int*>(plan->workspace);
    auto* partials = reinterpret_cast<float*>(counters + plan->n);
    split_kernel<<<plan->n * plan->splits, kWarps * 32, 0, plan->stream>>>(
        typed_input, typed_weight, typed_output, plan->n, plan->k,
        plan->splits, counters, partials, alpha, beta);
  } else if (plan->kernel == Kernel::BlockPerRow) {
    row_kernel<<<plan->n, kWarps * 32, 0, plan->stream>>>(
        typed_input, typed_weight, typed_output, plan->n, plan->k, alpha, beta);
  } else {
    const int blocks = (plan->n + kRowsPerBlock - 1) / kRowsPerBlock;
    kernel<<<blocks, kWarps * 32, 0, plan->stream>>>(
        typed_input, typed_weight, typed_output, plan->n, plan->k, alpha, beta);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

bool admits(int n, int k, int kernel, int splits) {
  switch (static_cast<Kernel>(kernel)) {
    case Kernel::RowGroups:
      return splits == 1;
    case Kernel::BlockPerRow:
      return splits == 1 && (k & 7) == 0;
    case Kernel::SplitK:
      return (k & 7) == 0 && splits >= 1 && splits <= k / 8 &&
             static_cast<long long>(n) * splits <= 0x7fffffffLL;
  }
  return false;
}

// Allocates and zeroes the split-K workspace on the plan stream; refused
// while that stream captures, so the allocation never belongs to a graph.
int prepare(Plan* plan) {
  if (plan->kernel != Kernel::SplitK) return 0;
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  int error = static_cast<int>(cudaStreamIsCapturing(plan->stream, &capture));
  if (error != 0) return error;
  if (capture != cudaStreamCaptureStatusNone) return -1;
  const size_t counters = static_cast<size_t>(plan->n) * sizeof(unsigned int);
  plan->workspace_bytes =
      counters + static_cast<size_t>(plan->n) * plan->splits * sizeof(float);
  error = static_cast<int>(
      cudaMallocAsync(&plan->workspace, plan->workspace_bytes, plan->stream));
  if (error != 0) return error;
  return static_cast<int>(
      cudaMemsetAsync(plan->workspace, 0, counters, plan->stream));
}

}  // namespace mircuda::dense_vector

extern "C" int mircuda_dense_vector_create(int data_type, int n, int k,
                                             int kernel, int splits,
                                             void* stream, void** output) {
  using namespace mircuda::dense_vector;
  if ((data_type != 0 && data_type != 1) || n <= 0 || k <= 0 ||
      !admits(n, k, kernel, splits) || stream == nullptr || output == nullptr)
    return -1;
  auto* plan = new (std::nothrow) Plan{data_type, n, k, static_cast<Kernel>(kernel),
                                       splits, static_cast<cudaStream_t>(stream),
                                       nullptr, 0};
  if (plan == nullptr) return -2;
  const int status = prepare(plan);
  if (status != 0) {
    if (plan->workspace != nullptr) cudaFreeAsync(plan->workspace, plan->stream);
    delete plan;
    return status;
  }
  *output = plan;
  return 0;
}

extern "C" size_t mircuda_dense_vector_workspace_bytes(const void* raw) {
  auto* plan = static_cast<const mircuda::dense_vector::Plan*>(raw);
  return plan == nullptr ? 0 : plan->workspace_bytes;
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
  auto* plan = static_cast<mircuda::dense_vector::Plan*>(raw);
  if (plan == nullptr) return;
  if (plan->workspace != nullptr) cudaFreeAsync(plan->workspace, plan->stream);
  delete plan;
}
