use super::super::{Context, DeviceBuffer, Stream, unsupported};
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FmhaBf16Spec {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_head_dim: usize,
}

#[derive(Debug)]
pub struct FmhaBf16Plan;

impl Context {
    pub const fn create_fmha_bf16_plan(
        &self,
        _stream: &Stream,
        _spec: FmhaBf16Spec,
    ) -> Result<FmhaBf16Plan> {
        Err(unsupported())
    }
}

impl FmhaBf16Plan {
    #[allow(clippy::too_many_arguments)]
    pub const fn execute_paged_varlen_split(
        &self,
        _stream: &Stream,
        _query: &DeviceBuffer,
        _key_pages: &DeviceBuffer,
        _value_pages: &DeviceBuffer,
        _output: &DeviceBuffer,
        _query_starts: &DeviceBuffer,
        _token_counts: &DeviceBuffer,
        _context_starts: &DeviceBuffer,
        _block_table: &DeviceBuffer,
        _softmax_lse: &DeviceBuffer,
        _output_accum: &DeviceBuffer,
        _softmax_lse_accum: &DeviceBuffer,
        _num_splits: usize,
        _batch_size: usize,
        _total_query_tokens: usize,
        _max_query_tokens: usize,
        _max_context_tokens: usize,
        _max_blocks: usize,
        _page_block_size: usize,
        _scale: f32,
    ) -> Result<()> {
        Err(unsupported())
    }

    #[allow(clippy::too_many_arguments)]
    pub const fn execute_paged_varlen(
        &self,
        _stream: &Stream,
        _query: &DeviceBuffer,
        _key_pages: &DeviceBuffer,
        _value_pages: &DeviceBuffer,
        _output: &DeviceBuffer,
        _query_starts: &DeviceBuffer,
        _token_counts: &DeviceBuffer,
        _context_starts: &DeviceBuffer,
        _block_table: &DeviceBuffer,
        _softmax_lse: &DeviceBuffer,
        _batch_size: usize,
        _total_query_tokens: usize,
        _max_query_tokens: usize,
        _max_context_tokens: usize,
        _max_blocks: usize,
        _page_block_size: usize,
        _scale: f32,
    ) -> Result<()> {
        Err(unsupported())
    }

    #[allow(clippy::too_many_arguments)]
    pub const fn execute(
        &self,
        _stream: &Stream,
        _query: &DeviceBuffer,
        _keys: &DeviceBuffer,
        _values: &DeviceBuffer,
        _output: &DeviceBuffer,
        _query_tokens: usize,
        _context_tokens: usize,
        _key_offset_bytes: usize,
        _value_offset_bytes: usize,
        _scale: f32,
    ) -> Result<()> {
        Err(unsupported())
    }

    #[allow(clippy::too_many_arguments)]
    pub const fn execute_varlen(
        &self,
        _stream: &Stream,
        _query: &DeviceBuffer,
        _keys: &DeviceBuffer,
        _values: &DeviceBuffer,
        _output: &DeviceBuffer,
        _query_starts: &DeviceBuffer,
        _key_starts: &DeviceBuffer,
        _batch_size: usize,
        _total_query_tokens: usize,
        _total_context_tokens: usize,
        _max_query_tokens: usize,
        _max_context_tokens: usize,
        _scale: f32,
    ) -> Result<()> {
        Err(unsupported())
    }
}
