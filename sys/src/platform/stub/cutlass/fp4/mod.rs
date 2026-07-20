use super::super::{Context, DeviceBuffer, Stream, unsupported};
use crate::Result;

mod vector;

pub use vector::{BlockScaledFp4VectorPlan, BlockScaledFp4VectorSpec};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockScaledFp4Spec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Debug)]
pub struct BlockScaledFp4Plan;

impl Context {
    pub const fn create_block_scaled_fp4_plan(
        &self,
        _stream: &Stream,
        _spec: BlockScaledFp4Spec,
    ) -> Result<BlockScaledFp4Plan> {
        Err(unsupported())
    }
}

impl BlockScaledFp4Plan {
    #[must_use]
    pub const fn workspace_bytes(&self) -> usize {
        0
    }

    #[allow(clippy::too_many_arguments)]
    pub const fn execute(
        &mut self,
        _stream: &Stream,
        _a: &DeviceBuffer,
        _a_scales: &DeviceBuffer,
        _b: &DeviceBuffer,
        _b_scales: &DeviceBuffer,
        _c: &DeviceBuffer,
        _alpha: f32,
    ) -> Result<()> {
        Err(unsupported())
    }
}
