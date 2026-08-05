use std::ffi::c_void;

use super::{FmhaBf16Plan, validation::check};
use crate::{
    Error, Result,
    platform::{
        driver::Stream,
        memory::{DeviceBuffer, ensure_stream},
    },
};

mod split;

unsafe extern "C" {
    fn mircuda_flash_attn2_paged_bf16_execute(
        query: *const c_void,
        key_pages: *const c_void,
        value_pages: *const c_void,
        output: *mut c_void,
        query_starts: *const i32,
        token_counts: *const i32,
        context_starts: *const i32,
        block_table: *const i32,
        softmax_lse: *mut f32,
        total_query_tokens: i32,
        max_query_tokens: i32,
        max_context_tokens: i32,
        batch_size: i32,
        max_blocks: i32,
        page_block_size: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
}

impl FmhaBf16Plan {
    #[allow(clippy::too_many_arguments)]
    pub fn execute_paged_varlen(
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
        self.stream.context().bind_to_thread()?;
        let status = unsafe {
            mircuda_flash_attn2_paged_bf16_execute(
                query.pointer() as *const c_void,
                key_pages.pointer() as *const c_void,
                value_pages.pointer() as *const c_void,
                output.pointer() as *mut c_void,
                query_starts.pointer() as *const i32,
                token_counts.pointer() as *const i32,
                context_starts.pointer() as *const i32,
                block_table.pointer() as *const i32,
                softmax_lse.pointer() as *mut f32,
                i32::try_from(total_query_tokens)?,
                i32::try_from(max_query_tokens)?,
                i32::try_from(max_context_tokens)?,
                i32::try_from(batch_size)?,
                i32::try_from(max_blocks)?,
                i32::try_from(page_block_size)?,
                i32::try_from(self.spec.query_heads)?,
                i32::try_from(self.spec.kv_heads)?,
                i32::try_from(self.spec.head_dim)?,
                scale,
                stream.inner.cu_stream().cast(),
            )
        };
        check(status)
    }
}

#[allow(clippy::too_many_arguments)]
fn validate(
    plan: &FmhaBf16Plan,
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
    batch_size: usize,
    total_query_tokens: usize,
    max_query_tokens: usize,
    max_context_tokens: usize,
    max_blocks: usize,
    page_block_size: usize,
) -> Result<()> {
    if !std::sync::Arc::ptr_eq(&plan.stream, &stream.inner) {
        return Err(Error::StreamMismatch);
    }
    for buffer in [
        query, key_pages, value_pages, output, query_starts, token_counts, context_starts,
        block_table, softmax_lse,
    ] {
        ensure_stream(buffer, stream)?;
    }
    let query_bytes = elements(total_query_tokens, plan.spec.query_heads, plan.spec.head_dim, 2)?;
    let page_bytes = elements(page_block_size, plan.spec.kv_heads, plan.spec.head_dim, 2)?;
    let valid = batch_size > 0
        && total_query_tokens >= max_query_tokens
        && max_query_tokens > 0
        && page_block_size > 0
        && page_block_size.is_multiple_of(16)
        && max_blocks > 0
        && max_context_tokens <= max_blocks.saturating_mul(page_block_size)
        && query.bytes() == query_bytes
        && output.bytes() == query_bytes
        && key_pages.bytes() >= page_bytes
        && key_pages.bytes().is_multiple_of(page_bytes)
        && value_pages.bytes() == key_pages.bytes()
        && query_starts.bytes() >= (batch_size + 1).saturating_mul(4)
        && context_starts.bytes() >= (batch_size + 1).saturating_mul(4)
        && token_counts.bytes() >= batch_size.saturating_mul(4)
        && block_table.bytes() >= batch_size.saturating_mul(max_blocks).saturating_mul(4)
        && softmax_lse.bytes()
            >= total_query_tokens.saturating_mul(plan.spec.query_heads).saturating_mul(4);
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidMatmulBuffer)
    }
}

fn elements(rows: usize, heads: usize, columns: usize, bytes: usize) -> Result<usize> {
    rows.checked_mul(heads)
        .and_then(|value| value.checked_mul(columns))
        .and_then(|value| value.checked_mul(bytes))
        .ok_or(Error::InvalidMatmulBuffer)
}
