use super::super::{Context, DeviceBuffer, Stream, unsupported};
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockwiseFp8VectorSpec {
    pub n: usize,
    pub k: usize,
}

#[derive(Debug)]
pub struct BlockwiseFp8VectorPlan;

impl Context {
    pub const fn create_blockwise_fp8_vector_plan(
        &self,
        _stream: &Stream,
        _spec: BlockwiseFp8VectorSpec,
    ) -> Result<BlockwiseFp8VectorPlan> {
        Err(unsupported())
    }
}

impl BlockwiseFp8VectorPlan {
    #[must_use]
    pub const fn workspace_bytes(&self) -> usize {
        0
    }

    #[allow(clippy::too_many_arguments)]
    pub const fn execute(
        &mut self,
        _stream: &Stream,
        _input: &DeviceBuffer,
        _input_scales: &DeviceBuffer,
        _weight: &DeviceBuffer,
        _weight_scales: &DeviceBuffer,
        _output: &DeviceBuffer,
    ) -> Result<()> {
        Err(unsupported())
    }
}
