use super::{FmhaBf16Plan, FmhaCausalWindow};
use crate::{DeviceBuffer, DeviceElement, Result, Stream, bf16};

impl FmhaBf16Plan {
    /// Enqueues split-KV causal `FlashAttention` 2 over paged K/V and merges partials.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_paged_varlen_split<T: DeviceElement>(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
        query_starts: &DeviceBuffer<u32>,
        token_counts: &DeviceBuffer<u32>,
        context_starts: &DeviceBuffer<u32>,
        block_table: &DeviceBuffer<u32>,
        softmax_lse_workspace: &mut DeviceBuffer<T>,
        output_accum: &mut DeviceBuffer<f32>,
        softmax_lse_accum: &mut DeviceBuffer<f32>,
        num_splits: usize,
        batch_size: usize,
        total_query_tokens: usize,
        max_query_tokens: usize,
        max_context_tokens: usize,
        max_blocks: usize,
        page_block_size: usize,
        scale: f32,
    ) -> Result<()> {
        Ok(self.native.execute_paged_varlen_split(
            &stream.native,
            &query.native,
            &key_pages.native,
            &value_pages.native,
            &output.native,
            &query_starts.native,
            &token_counts.native,
            &context_starts.native,
            &block_table.native,
            &softmax_lse_workspace.native,
            &output_accum.native,
            &softmax_lse_accum.native,
            num_splits,
            batch_size,
            total_query_tokens,
            max_query_tokens,
            max_context_tokens,
            max_blocks,
            page_block_size,
            scale,
        )?)
    }

    /// Enqueues causal `FlashAttention` 2 directly over packed queries and paged K/V.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_paged_varlen<T: DeviceElement>(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
        query_starts: &DeviceBuffer<u32>,
        token_counts: &DeviceBuffer<u32>,
        context_starts: &DeviceBuffer<u32>,
        block_table: &DeviceBuffer<u32>,
        softmax_lse_workspace: &mut DeviceBuffer<T>,
        batch_size: usize,
        total_query_tokens: usize,
        max_query_tokens: usize,
        max_context_tokens: usize,
        max_blocks: usize,
        page_block_size: usize,
        scale: f32,
    ) -> Result<()> {
        self.execute_paged_varlen_windowed(
            stream,
            query,
            key_pages,
            value_pages,
            output,
            query_starts,
            token_counts,
            context_starts,
            block_table,
            softmax_lse_workspace,
            batch_size,
            total_query_tokens,
            max_query_tokens,
            max_context_tokens,
            max_blocks,
            page_block_size,
            FmhaCausalWindow::Full,
            scale,
        )
    }

    /// Enqueues full or sliding causal `FlashAttention` 2 over paged K/V.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_paged_varlen_windowed<T: DeviceElement>(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
        query_starts: &DeviceBuffer<u32>,
        token_counts: &DeviceBuffer<u32>,
        context_starts: &DeviceBuffer<u32>,
        block_table: &DeviceBuffer<u32>,
        softmax_lse_workspace: &mut DeviceBuffer<T>,
        batch_size: usize,
        total_query_tokens: usize,
        max_query_tokens: usize,
        max_context_tokens: usize,
        max_blocks: usize,
        page_block_size: usize,
        window: FmhaCausalWindow,
        scale: f32,
    ) -> Result<()> {
        Ok(self.native.execute_paged_varlen_windowed(
            &stream.native,
            &query.native,
            &key_pages.native,
            &value_pages.native,
            &output.native,
            &query_starts.native,
            &token_counts.native,
            &context_starts.native,
            &block_table.native,
            &softmax_lse_workspace.native,
            batch_size,
            total_query_tokens,
            max_query_tokens,
            max_context_tokens,
            max_blocks,
            page_block_size,
            window.left_tokens(),
            scale,
        )?)
    }
}
