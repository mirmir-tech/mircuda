use super::{
    super::super::{Context, DeviceBuffer, Stream, unsupported},
    DenseMatmulDataType,
};
use crate::Result;

/// Work decomposition of a matrix-vector plan.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DenseVectorKernel {
    /// Each block reduces 32 rows, four per warp, across the whole input; any `k`.
    RowGroups,
    /// Each block reduces one row, its warps splitting the input; `k % 8 == 0`.
    BlockPerRow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseVectorSpec {
    pub n: usize,
    pub k: usize,
    pub data_type: DenseMatmulDataType,
    pub kernel: DenseVectorKernel,
}

#[derive(Debug)]
pub struct DenseVectorPlan;

impl Context {
    pub const fn create_dense_vector_plan(
        &self,
        _stream: &Stream,
        _spec: DenseVectorSpec,
    ) -> Result<DenseVectorPlan> {
        Err(unsupported())
    }
}

impl DenseVectorPlan {
    #[allow(clippy::too_many_arguments)]
    pub const fn execute(
        &mut self,
        _stream: &Stream,
        _input: &DeviceBuffer,
        _weight: &DeviceBuffer,
        _output: &DeviceBuffer,
        _alpha: f32,
        _beta: f32,
    ) -> Result<()> {
        Err(unsupported())
    }
}
