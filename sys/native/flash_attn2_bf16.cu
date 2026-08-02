#include <cmath>
#include <cstdint>

#include <cuda_runtime_api.h>
#include <cutlass/numeric_types.h>

#include "flash.h"

namespace mircuda::flash_attn2 {

template <int HeadDim>
int launch_typed(
    const void* query, const void* key_pages, const void* value_pages,
    void* output, const int* query_starts, const int* token_counts,
    const int* context_starts, const int* block_table, float* softmax_lse,
    int total_query_tokens,
    int max_query_tokens, int max_context_tokens, int batch_size,
    int max_blocks, int page_block_size, int query_heads, int kv_heads,
    float scale, cudaStream_t stream) {
  flash::Flash_fwd_params params{};
  params.q_ptr = const_cast<void*>(query);
  params.k_ptr = const_cast<void*>(key_pages);
  params.v_ptr = const_cast<void*>(value_pages);
  params.o_ptr = output;
  params.softmax_lse_ptr = softmax_lse;
  params.cu_seqlens_q = const_cast<int*>(query_starts);
  params.cu_seqlens_k = const_cast<int*>(context_starts);
  params.seqused_k = const_cast<int*>(token_counts);
  params.block_table = const_cast<int*>(block_table);
  params.q_row_stride = query_heads * HeadDim;
  params.k_row_stride = kv_heads * HeadDim;
  params.v_row_stride = kv_heads * HeadDim;
  params.o_row_stride = query_heads * HeadDim;
  params.q_head_stride = HeadDim;
  params.k_head_stride = HeadDim;
  params.v_head_stride = HeadDim;
  params.o_head_stride = HeadDim;
  params.k_batch_stride = page_block_size * params.k_row_stride;
  params.v_batch_stride = page_block_size * params.v_row_stride;
  params.block_table_batch_stride = max_blocks;
  params.page_block_size = page_block_size;
  params.b = batch_size;
  params.h = query_heads;
  params.h_k = kv_heads;
  params.h_h_k_ratio = query_heads / kv_heads;
  params.seqlen_q = max_query_tokens;
  params.seqlen_k = max_context_tokens;
  params.seqlen_q_rounded = ((max_query_tokens + 127) / 128) * 128;
  params.seqlen_k_rounded = ((max_context_tokens + 127) / 128) * 128;
  params.d = HeadDim;
  params.d_rounded = HeadDim;
  params.total_q = total_query_tokens;
  params.scale_softmax = scale;
  params.scale_softmax_log2 = scale * static_cast<float>(M_LOG2E);
  params.p_dropout = 1.0F;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0F;
  params.scale_softmax_rp_dropout = scale;
  params.window_size_left = -1;
  params.window_size_right = 0;
  params.is_bf16 = true;
  params.is_causal = true;
  params.is_seqlens_k_cumulative = true;
  params.unpadded_lse = true;
  params.num_splits = 1;
  flash::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, HeadDim, true>(
      params, stream);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch(
    const void* query, const void* key_pages, const void* value_pages,
    void* output, const int* query_starts, const int* token_counts,
    const int* context_starts, const int* block_table, float* softmax_lse,
    int total_query_tokens, int max_query_tokens, int max_context_tokens,
    int batch_size, int max_blocks, int page_block_size, int query_heads,
    int kv_heads, int head_dim, float scale, cudaStream_t stream) {
  if (head_dim == 64) {
    return launch_typed<64>(
        query, key_pages, value_pages, output, query_starts, token_counts,
        context_starts, block_table, softmax_lse, total_query_tokens,
        max_query_tokens, max_context_tokens, batch_size, max_blocks,
        page_block_size, query_heads, kv_heads, scale, stream);
  }
  return launch_typed<128>(
      query, key_pages, value_pages, output, query_starts, token_counts,
      context_starts, block_table, softmax_lse, total_query_tokens,
      max_query_tokens, max_context_tokens, batch_size, max_blocks,
      page_block_size, query_heads, kv_heads, scale, stream);
}

}  // namespace mircuda::flash_attn2

extern "C" int mircuda_flash_attn2_paged_bf16_execute(
    const void* query, const void* key_pages, const void* value_pages,
    void* output, const int* query_starts, const int* token_counts,
    const int* context_starts, const int* block_table, float* softmax_lse,
    int total_query_tokens,
    int max_query_tokens, int max_context_tokens, int batch_size,
    int max_blocks, int page_block_size, int query_heads, int kv_heads,
    int head_dim, float scale, void* stream) {
  if (query == nullptr || key_pages == nullptr || value_pages == nullptr ||
      output == nullptr || query_starts == nullptr || token_counts == nullptr ||
      context_starts == nullptr || block_table == nullptr ||
      softmax_lse == nullptr || stream == nullptr ||
      total_query_tokens <= 0 || max_query_tokens <= 0 ||
      max_context_tokens < max_query_tokens || batch_size <= 0 ||
      max_blocks <= 0 || page_block_size <= 0 ||
      page_block_size % 16 != 0 || query_heads <= 0 || kv_heads <= 0 ||
      query_heads % kv_heads != 0 ||
      (head_dim != 64 && head_dim != 128)) {
    return -1;
  }
  return mircuda::flash_attn2::launch(
      query, key_pages, value_pages, output, query_starts, token_counts,
      context_starts, block_table, softmax_lse, total_query_tokens,
      max_query_tokens,
      max_context_tokens, batch_size, max_blocks, page_block_size, query_heads,
      kv_heads, head_dim, scale, static_cast<cudaStream_t>(stream));
}
