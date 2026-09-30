// Bidirectional fused attention over sequences padded to one length inside a
// token-major fused `[tokens, 3, heads, head_dim]` projection. Sequence `b`
// owns tokens `[b · length, (b + 1) · length)`; each of its queries attends
// to its first `lengths[b]` keys, clamped to `[1, length]` so a bad length
// never reads outside the sequence.
#include <cuda_runtime_api.h>

#include "kernel_forward.h"

namespace mircuda::fmha_padded {

// Matches `mircuda_sys::PaddedAttentionDataType`.
enum class DataType : int { Bf16 = 0, F32 = 1 };

struct PaddedBatchHook {
  template <typename Params>
  CUTLASS_DEVICE static bool advance_to_batch(
      Params& params, int64_t& query_start, int64_t& key_start) {
    const int batch = int(blockIdx.z);
    query_start = int64_t(batch) * params.num_queries;
    key_start = query_start;
    const int valid = params.seqlen_k_ptr[batch];
    params.num_keys = valid < 1 ? 1 : (valid > params.num_queries ? params.num_queries : valid);
    return true;
  }
};

template <typename T, int HeadDim>
using Attention =
    AttentionKernel<T, cutlass::arch::Sm80, true, 64, 64, HeadDim, false, false, PaddedBatchHook>;

template <typename T, int HeadDim>
int launch(
    const void* qkv, void* output, const int* lengths, int sequences, int length, int heads,
    float scale, cudaStream_t stream) {
  using Kernel = Attention<T, HeadDim>;
  const int shared_bytes = sizeof(typename Kernel::SharedStorage);
  if (shared_bytes > 0xc000) {
    const cudaError_t configured = cudaFuncSetAttribute(
        attention_kernel_batched_impl<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes);
    if (configured != cudaSuccess) return static_cast<int>(configured);
  }
  auto* base = reinterpret_cast<T*>(const_cast<void*>(qkv));
  const int hidden = heads * HeadDim;
  typename Kernel::Params params;
  params.query_ptr = base;
  params.key_ptr = base + hidden;
  params.value_ptr = base + 2 * hidden;
  params.output_ptr = reinterpret_cast<T*>(output);
  params.seqlen_k_ptr = const_cast<int*>(lengths);
  params.scale = scale;
  params.num_heads = heads;
  params.num_batches = sequences;
  params.head_dim = HeadDim;
  params.head_dim_value = HeadDim;
  params.num_queries = length;
  params.num_keys = length;
  params.custom_mask_type = Kernel::NoCustomMask;
  params.q_strideH = HeadDim;
  params.k_strideH = HeadDim;
  params.v_strideH = HeadDim;
  params.q_strideM = 3 * hidden;
  params.k_strideM = 3 * hidden;
  params.v_strideM = 3 * hidden;
  params.o_strideM = hidden;
  if (!Kernel::check_supported(params)) return -2;
  attention_kernel_batched_impl<Kernel>
      <<<params.getBlocksGrid(), params.getThreadsGrid(), shared_bytes, stream>>>(params);
  return static_cast<int>(cudaPeekAtLastError());
}

template <typename T>
int launch_head_dim(
    int head_dim, const void* qkv, void* output, const int* lengths, int sequences, int length,
    int heads, float scale, cudaStream_t stream) {
  return head_dim == 64
      ? launch<T, 64>(qkv, output, lengths, sequences, length, heads, scale, stream)
      : launch<T, 128>(qkv, output, lengths, sequences, length, heads, scale, stream);
}

}  // namespace mircuda::fmha_padded

extern "C" int mircuda_fmha_padded_execute(
    int data_type, const void* qkv, void* output, const int* lengths, int sequences, int length,
    int heads, int head_dim, float scale, void* stream) {
  using namespace mircuda::fmha_padded;
  if (qkv == nullptr || output == nullptr || lengths == nullptr || stream == nullptr ||
      sequences <= 0 || length <= 0 || heads <= 0 || (head_dim != 64 && head_dim != 128)) {
    return -1;
  }
  const auto target = static_cast<cudaStream_t>(stream);
  switch (static_cast<DataType>(data_type)) {
    case DataType::Bf16:
      return launch_head_dim<cutlass::bfloat16_t>(
          head_dim, qkv, output, lengths, sequences, length, heads, scale, target);
    case DataType::F32:
      return launch_head_dim<float>(
          head_dim, qkv, output, lengths, sequences, length, heads, scale, target);
  }
  return -1;
}
