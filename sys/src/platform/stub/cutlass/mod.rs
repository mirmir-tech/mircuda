use super::{Context, DeviceBuffer, Stream, unsupported};
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexedGroupedFp4Spec {
    pub groups: usize,
    pub matrices: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub broadcast_input: bool,
}

#[derive(Debug)]
pub struct IndexedGroupedFp4Plan;

impl Context {
    pub const fn create_indexed_grouped_fp4_plan(
        &self,
        _stream: &Stream,
        _spec: IndexedGroupedFp4Spec,
    ) -> Result<IndexedGroupedFp4Plan> {
        Err(unsupported())
    }
}

impl IndexedGroupedFp4Plan {
    #[allow(clippy::too_many_arguments)]
    pub const fn execute(
        &mut self,
        _stream: &Stream,
        _a: &DeviceBuffer,
        _a_scales: &DeviceBuffer,
        _b: &DeviceBuffer,
        _b_scales: &DeviceBuffer,
        _alphas: &DeviceBuffer,
        _indices: &DeviceBuffer,
        _c: &DeviceBuffer,
    ) -> Result<()> {
        Err(unsupported())
    }
}
mod dense;
mod fmha;
mod fp4;
mod fp8_vector;
mod scaled_fp8;
mod variable_grouped_bf16;

pub use dense::{
    DenseMatmulDataType, DenseMatmulPlan, DenseMatmulSpec, DenseVectorPlan, DenseVectorSpec,
};
pub use fmha::{FmhaBf16Plan, FmhaBf16Spec};
pub use fp4::{
    BlockScaledFp4Plan, BlockScaledFp4Spec, BlockScaledFp4VectorPlan, BlockScaledFp4VectorSpec,
};
pub use fp8_vector::{BlockwiseFp8VectorPlan, BlockwiseFp8VectorSpec};
pub use mxfp8::{BlockScaledMxFp8Plan, BlockScaledMxFp8Spec};
pub use scaled_fp8::{ScaledFp8Plan, ScaledFp8ScaleType, ScaledFp8Spec, ScaledFp8WeightScaleType};
pub use variable_grouped_bf16::{VariableGroupedBf16Plan, VariableGroupedBf16Spec};
mod mxfp8;
mod variable_grouped_fp4;
pub use variable_grouped_fp4::{
    PairedVariableGroupedFp4Plan, VariableGroupedFp4Plan, VariableGroupedFp4Spec,
};
