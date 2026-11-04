use super::super::{Context, DeviceBuffer, Stream, unsupported};
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PaddedAttentionDataType {
    Bf16,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PaddedAttentionSpec {
    pub heads: usize,
    pub head_dim: usize,
    pub data_type: PaddedAttentionDataType,
}

#[derive(Debug)]
pub struct PaddedAttentionPlan;

impl Context {
    pub const fn create_padded_attention_plan(
        &self,
        _stream: &Stream,
        _spec: PaddedAttentionSpec,
    ) -> Result<PaddedAttentionPlan> {
        Err(unsupported())
    }
}

impl PaddedAttentionPlan {
    pub const fn execute(
        &self,
        _stream: &Stream,
        _buffers: (&DeviceBuffer, &DeviceBuffer, &DeviceBuffer),
        _length: usize,
        _scale: f32,
    ) -> Result<()> {
        Err(unsupported())
    }
}
