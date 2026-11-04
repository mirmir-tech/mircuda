#include <cuda_runtime_api.h>

#include "kernel_forward.h"

namespace mircuda::fmha {

template <int QueriesPerBlock>
struct GqaToBatchHook {
  template <typename Params>
  CUTLASS_DEVICE static bool advance_to_batch(
      Params& params, int64_t& query_start, int64_t& key_start) {
    if (params.seqstart_q_ptr != nullptr) {
      const int batch = int(blockIdx.z);
      params.seqstart_q_ptr += batch;
      params.seqstart_k_ptr += batch;
      query_start = params.seqstart_q_ptr[0];
      key_start = params.seqstart_k_ptr[0];
      params.num_queries = params.seqstart_q_ptr[1] - query_start;
      params.num_keys = params.seqstart_k_ptr[1] - key_start;
      if (int(blockIdx.x) * QueriesPerBlock >= params.num_queries) {
        return false;
      }
    }
    const int query_heads = params.q_strideM / params.head_dim;
    const int kv_heads = params.k_strideM / params.head_dim;
    const int kv_head = int(blockIdx.y) / (query_heads / kv_heads);
    params.key_ptr += kv_head * params.head_dim;
    params.value_ptr += kv_head * params.head_dim_value;
    params.k_strideH = 0;
    params.v_strideH = 0;
    return true;
  }
};

// Heads up to 128 keep the output in registers over one key tile. Wider heads
// iterate the value dimension in 64-column tiles and accumulate the output in
// a caller-owned FP32 buffer.
template <int HeadDim>
struct Tile {
  static constexpr int kQueries = 32;
  static constexpr int kKeys = HeadDim;
};

template <>
struct Tile<512> {
  static constexpr int kQueries = 64;
  static constexpr int kKeys = 64;
};

template <int HeadDim>
using Attention = AttentionKernel<
    cutlass::bfloat16_t, cutlass::arch::Sm80, true, Tile<HeadDim>::kQueries,
    Tile<HeadDim>::kKeys, HeadDim, false, false,
    GqaToBatchHook<Tile<HeadDim>::kQueries>>;

template <int HeadDim>
int launch_typed(
    const void* query, const void* keys, const void* values, void* output,
    float* accumulator, int query_tokens, int context_tokens, int query_heads,
    int kv_heads, float scale, cudaStream_t stream) {
  using Kernel = Attention<HeadDim>;
  typename Kernel::Params params;
  params.query_ptr = reinterpret_cast<cutlass::bfloat16_t*>(
      const_cast<void*>(query));
  params.key_ptr =
      reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(keys));
  params.value_ptr =
      reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(values));
  params.output_ptr = reinterpret_cast<cutlass::bfloat16_t*>(output);
  if (Kernel::kNeedsOutputAccumulatorBuffer) {
    if (accumulator == nullptr) return -1;
    params.output_accum_ptr = accumulator;
  }
  params.scale = scale;
  params.num_heads = query_heads;
  params.num_batches = 1;
  params.head_dim = HeadDim;
  params.head_dim_value = HeadDim;
  params.num_queries = query_tokens;
  params.num_keys = context_tokens;
  params.custom_mask_type = Kernel::CausalFromBottomRight;
  params.q_strideH = HeadDim;
  params.k_strideH = HeadDim;
  params.v_strideH = HeadDim;
  params.q_strideM = query_heads * HeadDim;
  params.k_strideM = kv_heads * HeadDim;
  params.v_strideM = kv_heads * HeadDim;
  params.q_strideB = params.q_strideM * query_tokens;
  params.k_strideB = params.k_strideM * context_tokens;
  params.v_strideB = params.v_strideM * context_tokens;
  params.o_strideM = query_heads * HeadDim;
  if (!Kernel::check_supported(params)) return -2;
  const int shared_bytes = sizeof(typename Kernel::SharedStorage);
  attention_kernel_batched_impl<Kernel>
      <<<params.getBlocksGrid(), params.getThreadsGrid(), shared_bytes, stream>>>(
          params);
  return static_cast<int>(cudaPeekAtLastError());
}

template <int HeadDim>
int launch_varlen_typed(
    const void* query, const void* keys, const void* values, void* output,
    const int* query_starts, const int* key_starts, int max_query_tokens,
    int max_context_tokens, int batch_size, int query_heads, int kv_heads,
    float scale, cudaStream_t stream) {
  using Kernel = Attention<HeadDim>;
  typename Kernel::Params params;
  params.query_ptr = reinterpret_cast<cutlass::bfloat16_t*>(
      const_cast<void*>(query));
  params.key_ptr =
      reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(keys));
  params.value_ptr =
      reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(values));
  params.output_ptr = reinterpret_cast<cutlass::bfloat16_t*>(output);
  params.seqstart_q_ptr = const_cast<int*>(query_starts);
  params.seqstart_k_ptr = const_cast<int*>(key_starts);
  params.scale = scale;
  params.num_heads = query_heads;
  params.num_batches = batch_size;
  params.head_dim = HeadDim;
  params.head_dim_value = HeadDim;
  params.num_queries = max_query_tokens;
  params.num_keys = max_context_tokens;
  params.custom_mask_type = Kernel::CausalFromBottomRight;
  params.q_strideH = HeadDim;
  params.k_strideH = HeadDim;
  params.v_strideH = HeadDim;
  params.q_strideM = query_heads * HeadDim;
  params.k_strideM = kv_heads * HeadDim;
  params.v_strideM = kv_heads * HeadDim;
  params.o_strideM = query_heads * HeadDim;
  if (!Kernel::check_supported(params)) return -2;
  const int shared_bytes = sizeof(typename Kernel::SharedStorage);
  attention_kernel_batched_impl<Kernel>
      <<<params.getBlocksGrid(), params.getThreadsGrid(), shared_bytes, stream>>>(
          params);
  return static_cast<int>(cudaPeekAtLastError());
}

template <int HeadDim>
int configure() {
  using Kernel = Attention<HeadDim>;
  const int shared_bytes = sizeof(typename Kernel::SharedStorage);
  if (shared_bytes <= 0xc000) return 0;
  return static_cast<int>(cudaFuncSetAttribute(
      attention_kernel_batched_impl<Kernel>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
}

int launch(
    const void* query, const void* keys, const void* values, void* output,
    float* accumulator, int query_tokens, int context_tokens, int query_heads,
    int kv_heads, int head_dim, float scale, cudaStream_t stream) {
  switch (head_dim) {
    case 64:
      return launch_typed<64>(
          query, keys, values, output, accumulator, query_tokens,
          context_tokens, query_heads, kv_heads, scale, stream);
    case 128:
      return launch_typed<128>(
          query, keys, values, output, accumulator, query_tokens,
          context_tokens, query_heads, kv_heads, scale, stream);
    default:
      return launch_typed<512>(
          query, keys, values, output, accumulator, query_tokens,
          context_tokens, query_heads, kv_heads, scale, stream);
  }
}

int configure_fixed(int head_dim) {
  switch (head_dim) {
    case 64:
      return configure<64>();
    case 128:
      return configure<128>();
    default:
      return configure<512>();
  }
}

int launch_varlen(
    const void* query, const void* keys, const void* values, void* output,
    const int* query_starts, const int* key_starts, int max_query_tokens,
    int max_context_tokens, int batch_size, int query_heads, int kv_heads,
    int head_dim, float scale, cudaStream_t stream) {
  return head_dim == 64
      ? launch_varlen_typed<64>(
            query, keys, values, output, query_starts, key_starts,
            max_query_tokens, max_context_tokens, batch_size, query_heads,
            kv_heads, scale, stream)
      : launch_varlen_typed<128>(
            query, keys, values, output, query_starts, key_starts,
            max_query_tokens, max_context_tokens, batch_size, query_heads,
            kv_heads, scale, stream);
}

}  // namespace mircuda::fmha

extern "C" int mircuda_fmha_bf16_execute(
    const void* query, const void* keys, const void* values, void* output,
    void* accumulator, int query_tokens, int context_tokens, int query_heads,
    int kv_heads, int head_dim, int value_head_dim, float scale,
    void* stream) {
  using namespace mircuda::fmha;
  if (query == nullptr || keys == nullptr || values == nullptr ||
      output == nullptr || stream == nullptr || query_tokens <= 0 ||
      context_tokens < query_tokens || query_heads <= 0 || kv_heads <= 0 ||
      query_heads % kv_heads != 0 || head_dim != value_head_dim ||
      (head_dim != 64 && head_dim != 128 && head_dim != 512)) {
    return -1;
  }
  const int configured = configure_fixed(head_dim);
  if (configured != 0) return configured;
  return launch(
      query, keys, values, output, static_cast<float*>(accumulator),
      query_tokens, context_tokens, query_heads, kv_heads, head_dim, scale,
      static_cast<cudaStream_t>(stream));
}

extern "C" int mircuda_fmha_bf16_varlen_execute(
    const void* query, const void* keys, const void* values, void* output,
    const int* query_starts, const int* key_starts, int max_query_tokens,
    int max_context_tokens, int batch_size, int query_heads, int kv_heads,
    int head_dim, int value_head_dim, float scale, void* stream) {
  using namespace mircuda::fmha;
  if (query == nullptr || keys == nullptr || values == nullptr ||
      output == nullptr || query_starts == nullptr || key_starts == nullptr ||
      stream == nullptr || max_query_tokens <= 0 || max_context_tokens <= 0 ||
      batch_size <= 0 || query_heads <= 0 || kv_heads <= 0 ||
      query_heads % kv_heads != 0 || head_dim != value_head_dim ||
      (head_dim != 64 && head_dim != 128)) {
    return -1;
  }
  const int configured =
      head_dim == 64 ? configure<64>() : configure<128>();
  if (configured != 0) return configured;
  return launch_varlen(
      query, keys, values, output, query_starts, key_starts, max_query_tokens,
      max_context_tokens, batch_size, query_heads, kv_heads, head_dim,
      scale, static_cast<cudaStream_t>(stream));
}
