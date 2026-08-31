use std::ffi::c_void;

use super::{FmhaBf16Plan, check, validate};
use crate::{
    Error, Result,
    platform::{driver::Stream, memory::DeviceBuffer},
};

unsafe extern "C" {
    fn mircuda_flash_attn2_paged_bf16_execute_split(
        query: *const c_void,
        key_pages: *const c_void,
        value_pages: *const c_void,
        output: *mut c_void,
        query_starts: *const i32,
        token_counts: *const i32,
        context_starts: *const i32,
        block_table: *const i32,
        softmax_lse: *mut f32,
        output_accum: *mut f32,
        softmax_lse_accum: *mut f32,
        num_splits: i32,
        total_query_tokens: i32,
        max_query_tokens: i32,
        max_context_tokens: i32,
        batch_size: i32,
        max_blocks: i32,
        page_block_size: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        window_size_left: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
}

impl FmhaBf16Plan {
    #[allow(clippy::too_many_arguments)]
    pub fn execute_paged_varlen_split(
        &self,
        stream: &Stream,
        query: &DeviceBuffer,
        key_pages: &DeviceBuffer,
        value_pages: &DeviceBuffer,
        output: &DeviceBuffer,
        query_starts: &DeviceBuffer,
        token_counts: &DeviceBuffer,
        context_starts: &DeviceBuffer,
        block_table: &DeviceBuffer,
        softmax_lse: &DeviceBuffer,
        output_accum: &DeviceBuffer,
        softmax_lse_accum: &DeviceBuffer,
        num_splits: usize,
        batch_size: usize,
        total_query_tokens: usize,
        max_query_tokens: usize,
        max_context_tokens: usize,
        max_blocks: usize,
        page_block_size: usize,
        scale: f32,
    ) -> Result<()> {
        validate(
            self,
            stream,
            query,
            key_pages,
            value_pages,
            output,
            query_starts,
            token_counts,
            context_starts,
            block_table,
            softmax_lse,
            batch_size,
            total_query_tokens,
            max_query_tokens,
            max_context_tokens,
            max_blocks,
            page_block_size,
        )?;
        let output_elements = num_splits
            .saturating_mul(total_query_tokens)
            .saturating_mul(self.spec.query_heads)
            .saturating_mul(self.spec.head_dim);
        let lse_elements = num_splits
            .saturating_mul(total_query_tokens)
            .saturating_mul(self.spec.query_heads);
        if num_splits < 2
            || output_accum.bytes() < output_elements.saturating_mul(4)
            || softmax_lse_accum.bytes() < lse_elements.saturating_mul(4)
        {
            return Err(Error::InvalidMatmulBuffer);
        }
        self.stream.context().bind_to_thread()?;
        // SAFETY: all buffers and dimensions were validated above, and the CUDA
        // context owning them is bound to this thread.
        let status = unsafe {
            mircuda_flash_attn2_paged_bf16_execute_split(
                query.pointer() as *const c_void,
                key_pages.pointer() as *const c_void,
                value_pages.pointer() as *const c_void,
                output.pointer() as *mut c_void,
                query_starts.pointer() as *const i32,
                token_counts.pointer() as *const i32,
                context_starts.pointer() as *const i32,
                block_table.pointer() as *const i32,
                softmax_lse.pointer() as *mut f32,
                output_accum.pointer() as *mut f32,
                softmax_lse_accum.pointer() as *mut f32,
                i32::try_from(num_splits)?,
                i32::try_from(total_query_tokens)?,
                i32::try_from(max_query_tokens)?,
                i32::try_from(max_context_tokens)?,
                i32::try_from(batch_size)?,
                i32::try_from(max_blocks)?,
                i32::try_from(page_block_size)?,
                i32::try_from(self.spec.query_heads)?,
                i32::try_from(self.spec.kv_heads)?,
                i32::try_from(self.spec.head_dim)?,
                -1,
                scale,
                stream.inner.cu_stream().cast(),
            )
        };
        check(status)
    }
}
