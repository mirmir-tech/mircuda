use super::{Context, DeviceBuffer, Stream, unsupported};
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockScaledFp4VectorSpec {
    pub n: usize,
    pub k: usize,
}

#[derive(Debug)]
pub struct BlockScaledFp4VectorPlan;

impl Context {
    pub const fn create_block_scaled_fp4_vector_plan(
        &self,
        _stream: &Stream,
        _spec: BlockScaledFp4VectorSpec,
    ) -> Result<BlockScaledFp4VectorPlan> {
        Err(unsupported())
    }
}

impl BlockScaledFp4VectorPlan {
    #[allow(clippy::too_many_arguments)]
    pub const fn execute(
        &mut self,
        _stream: &Stream,
        _input: &DeviceBuffer,
        _input_scales: &DeviceBuffer,
        _weight: &DeviceBuffer,
        _weight_scales: &DeviceBuffer,
        _output: &DeviceBuffer,
        _alpha: f32,
    ) -> Result<()> {
        Err(unsupported())
    }
}
