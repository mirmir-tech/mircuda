use std::marker::PhantomData;

use crate::{Context, DeviceBuffer, DeviceElement, Result, Stream, bf16};

mod sealed {
    pub trait Sealed {}
}

/// Element type of every operand of padded fused attention.
pub trait PaddedAttentionElement: DeviceElement + sealed::Sealed {
    #[doc(hidden)]
    const DATA_TYPE: mircuda_sys::PaddedAttentionDataType;
}

impl sealed::Sealed for bf16 {}
impl sealed::Sealed for f32 {}

impl PaddedAttentionElement for bf16 {
    const DATA_TYPE: mircuda_sys::PaddedAttentionDataType =
        mircuda_sys::PaddedAttentionDataType::Bf16;
}

impl PaddedAttentionElement for f32 {
    const DATA_TYPE: mircuda_sys::PaddedAttentionDataType =
        mircuda_sys::PaddedAttentionDataType::F32;
}

/// Bidirectional fused attention of sequences padded to one length.
///
/// It reads a token-major fused `[tokens, 3, heads, head_dim]` projection and
/// writes `[tokens, heads × head_dim]`; scores never reach device memory.
#[derive(Debug)]
pub struct PaddedAttentionPlan<T: PaddedAttentionElement> {
    native: mircuda_sys::PaddedAttentionPlan,
    marker: PhantomData<T>,
}

impl<T: PaddedAttentionElement> PaddedAttentionPlan<T> {
    /// Supports head widths 64 and 128.
    pub fn new(context: &Context, stream: &Stream, heads: usize, head_dim: usize) -> Result<Self> {
        let spec = mircuda_sys::PaddedAttentionSpec { heads, head_dim, data_type: T::DATA_TYPE };
        Ok(Self {
            native: context.native.create_padded_attention_plan(&stream.native, spec)?,
            marker: PhantomData,
        })
    }

    /// Sequence `b` owns tokens `[b · length, (b + 1) · length)` and attends
    /// to its first `lengths[b]` keys; queries past that length still get
    /// outputs, over the same keys.
    pub fn execute(
        &self,
        stream: &Stream,
        (qkv, lengths): (&DeviceBuffer<T>, &DeviceBuffer<u32>),
        length: usize,
        scale: f32,
        output: &mut DeviceBuffer<T>,
    ) -> Result<()> {
        Ok(self.native.execute(
            &stream.native,
            (&qkv.native, &lengths.native, &output.native),
            length,
            scale,
        )?)
    }
}
