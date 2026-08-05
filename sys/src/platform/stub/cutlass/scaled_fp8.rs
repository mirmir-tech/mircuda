use super::{Context, DeviceBuffer, Stream, unsupported};
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaledFp8ScaleType {
    F32,
    Bf16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaledFp8WeightScaleType {
    Tensor,
    OutputChannel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaledFp8Tile {
    M16N64K128,
    M16N128K64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScaledFp8Spec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub scale_type: ScaledFp8ScaleType,
    pub weight_scale_type: ScaledFp8WeightScaleType,
    pub has_bias: bool,
    pub tile: ScaledFp8Tile,
}

#[derive(Debug)]
pub struct ScaledFp8Plan;

impl Context {
    pub const fn create_scaled_fp8_plan(
        &self,
        _stream: &Stream,
        _spec: ScaledFp8Spec,
    ) -> Result<ScaledFp8Plan> {
        Err(unsupported())
    }
}

impl ScaledFp8Plan {
    #[must_use]
    pub const fn workspace_bytes(&self) -> usize {
        0
    }

    #[allow(clippy::too_many_arguments)]
    pub const fn execute(
        &self,
        _stream: &Stream,
        _input: &DeviceBuffer,
        _weight: &DeviceBuffer,
        _input_scales: &DeviceBuffer,
        _weight_scales: &DeviceBuffer,
        _bias: Option<&DeviceBuffer>,
        _output: &DeviceBuffer,
    ) -> Result<()> {
        Err(unsupported())
    }
}
