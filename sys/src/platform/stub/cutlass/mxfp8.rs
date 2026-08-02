use super::{Context, DeviceBuffer, Stream, unsupported};
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockScaledMxFp8Spec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Debug)]
pub struct BlockScaledMxFp8Plan;

impl Context {
    pub const fn create_block_scaled_mxfp8_plan(
        &self,
        _stream: &Stream,
        _spec: BlockScaledMxFp8Spec,
    ) -> Result<BlockScaledMxFp8Plan> {
        Err(unsupported())
    }
}

impl BlockScaledMxFp8Plan {
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
    ) -> Result<()> {
        Err(unsupported())
    }
}
