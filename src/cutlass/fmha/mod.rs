use std::num::NonZeroUsize;

use crate::{Context, DeviceBuffer, Error, Result, Stream, bf16};

mod paged;

/// Causal K/V visibility used by paged fused attention.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FmhaCausalWindow {
    /// Every causal key before and including the query is visible.
    Full,
    /// At most this many causal keys, including the current position, are visible.
    Sliding(NonZeroUsize),
}

impl FmhaCausalWindow {
    /// Creates a non-empty sliding causal window measured in visible tokens.
    pub const fn sliding(tokens: usize) -> Result<Self> {
        match NonZeroUsize::new(tokens) {
            Some(tokens) => Ok(Self::Sliding(tokens)),
            None => Err(Error::InvalidMatmulShape),
        }
    }

    pub(super) const fn left_tokens(self) -> Option<usize> {
        match self {
            Self::Full => None,
            Self::Sliding(tokens) => Some(tokens.get() - 1),
        }
    }
}

/// Head geometry supported by the BF16 CUTLASS fused-attention plan.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FmhaBf16Spec {
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_head_dim: usize,
}

impl FmhaBf16Spec {
    /// Creates a GQA/MHA fused-attention geometry.
    pub const fn new(
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Self> {
        if query_heads == 0
            || kv_heads == 0
            || !query_heads.is_multiple_of(kv_heads)
            || head_dim != value_head_dim
            || !matches!(head_dim, 64 | 128 | 256 | 512)
        {
            return Err(Error::InvalidMatmulShape);
        }
        Ok(Self {
            query_heads,
            kv_heads,
            head_dim,
            value_head_dim,
        })
    }

    /// Query-head count.
    #[must_use]
    pub const fn query_heads(self) -> usize {
        self.query_heads
    }

    /// K/V-head count.
    #[must_use]
    pub const fn kv_heads(self) -> usize {
        self.kv_heads
    }

    /// FP32 elements of the output accumulator that fixed attention over
    /// `query_tokens` needs; heads wider than 256 accumulate outside registers.
    #[must_use]
    pub const fn accumulator_elements(self, query_tokens: usize) -> usize {
        if self.value_head_dim > 256 {
            query_tokens * self.query_heads * self.value_head_dim
        } else {
            0
        }
    }
}

/// Stream-bound BF16 fused-attention plan using a bottom-right causal mask.
#[derive(Debug)]
pub struct FmhaBf16Plan {
    native: mircuda_sys::FmhaBf16Plan,
    spec: FmhaBf16Spec,
}

impl FmhaBf16Plan {
    /// Creates an asynchronous fixed-head plan.
    pub fn new(context: &Context, stream: &Stream, spec: FmhaBf16Spec) -> Result<Self> {
        let native = context.native.create_fmha_bf16_plan(
            &stream.native,
            mircuda_sys::FmhaBf16Spec {
                query_heads: spec.query_heads,
                kv_heads: spec.kv_heads,
                head_dim: spec.head_dim,
                value_head_dim: spec.value_head_dim,
            },
        )?;
        Ok(Self { native, spec })
    }

    /// Head geometry of this plan.
    #[must_use]
    pub const fn spec(&self) -> FmhaBf16Spec {
        self.spec
    }

    /// Enqueues causal attention for queries aligned to the end of the K/V context.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
        query_tokens: usize,
        context_tokens: usize,
        first_page_token: usize,
        scale: f32,
    ) -> Result<()> {
        let pages = FixedPages { key_pages, value_pages, first_page_token };
        self.execute_fixed(stream, query, pages, output, None, query_tokens, context_tokens, scale)
    }

    /// Enqueues [`Self::execute`] for heads that accumulate their output in an
    /// FP32 buffer of at least [`FmhaBf16Spec::accumulator_elements`].
    #[allow(clippy::too_many_arguments)]
    pub fn execute_accumulated(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
        accumulator: &mut DeviceBuffer<f32>,
        query_tokens: usize,
        context_tokens: usize,
        first_page_token: usize,
        scale: f32,
    ) -> Result<()> {
        let pages = FixedPages { key_pages, value_pages, first_page_token };
        self.execute_fixed(
            stream,
            query,
            pages,
            output,
            Some(&*accumulator),
            query_tokens,
            context_tokens,
            scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_fixed(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        pages: FixedPages<'_>,
        output: &DeviceBuffer<bf16>,
        accumulator: Option<&DeviceBuffer<f32>>,
        query_tokens: usize,
        context_tokens: usize,
        scale: f32,
    ) -> Result<()> {
        let FixedPages { key_pages, value_pages, first_page_token } = pages;
        let key_row = self.spec.kv_heads * self.spec.head_dim;
        let value_row = self.spec.kv_heads * self.spec.value_head_dim;
        let key_offset = first_page_token
            .checked_mul(key_row)
            .and_then(|value| value.checked_mul(size_of::<bf16>()))
            .ok_or(Error::InvalidMatmulShape)?;
        let value_offset = first_page_token
            .checked_mul(value_row)
            .and_then(|value| value.checked_mul(size_of::<bf16>()))
            .ok_or(Error::InvalidMatmulShape)?;
        Ok(self.native.execute(
            &stream.native,
            &query.native,
            &key_pages.native,
            &value_pages.native,
            &output.native,
            accumulator.map(|buffer| &*buffer.native),
            query_tokens,
            context_tokens,
            key_offset,
            value_offset,
            scale,
        )?)
    }

    /// Enqueues one bottom-right causal attention launch over packed variable-length rows.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_varlen(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        keys: &DeviceBuffer<bf16>,
        values: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        query_starts: &DeviceBuffer<u32>,
        key_starts: &DeviceBuffer<u32>,
        batch_size: usize,
        total_query_tokens: usize,
        total_context_tokens: usize,
        max_query_tokens: usize,
        max_context_tokens: usize,
        scale: f32,
    ) -> Result<()> {
        Ok(self.native.execute_varlen(
            &stream.native,
            &query.native,
            &keys.native,
            &values.native,
            &output.native,
            &query_starts.native,
            &key_starts.native,
            batch_size,
            total_query_tokens,
            total_context_tokens,
            max_query_tokens,
            max_context_tokens,
            scale,
        )?)
    }
}

#[derive(Clone, Copy)]
struct FixedPages<'a> {
    key_pages: &'a DeviceBuffer<u8>,
    value_pages: &'a DeviceBuffer<u8>,
    first_page_token: usize,
}
