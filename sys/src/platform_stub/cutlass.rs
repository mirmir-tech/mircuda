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
#[path = "cutlass/dense.rs"]
mod dense;
#[path = "cutlass/dense_vector.rs"]
mod dense_vector;
#[path = "cutlass/fp4.rs"]
mod fp4;
#[path = "cutlass/fp4_vector.rs"]
mod fp4_vector;
#[path = "cutlass/fp8_vector.rs"]
mod fp8_vector;

pub use dense::{DenseMatmulDataType, DenseMatmulPlan, DenseMatmulSpec};
pub use dense_vector::{DenseVectorPlan, DenseVectorSpec};
pub use fp4::{BlockScaledFp4Plan, BlockScaledFp4Spec};
pub use fp4_vector::{BlockScaledFp4VectorPlan, BlockScaledFp4VectorSpec};
pub use fp8_vector::{BlockwiseFp8VectorPlan, BlockwiseFp8VectorSpec};
#[path = "cutlass/variable_grouped_fp4.rs"]
mod variable_grouped_fp4;
pub use variable_grouped_fp4::{
    PairedVariableGroupedFp4Plan, VariableGroupedFp4Plan, VariableGroupedFp4Spec,
};
