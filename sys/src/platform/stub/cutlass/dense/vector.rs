use std::num::NonZeroU16;

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
    /// `splits` blocks per row each reduce a balanced slice of the input into
    /// an FP32 partial in plan-owned workspace; the last block of a row sums
    /// the partials in slice order. `k % 8 == 0` and `splits <= k / 8`.
    SplitK {
        /// Input slices per output row.
        splits: NonZeroU16,
    },
}

impl DenseVectorKernel {
    /// Byte alignment the input and weight must have: the 16-byte vector
    /// loads of block-per-row and split-K kernels; row groups read whatever
    /// width both operands' alignment allows.
    #[must_use]
    pub const fn operand_alignment(self) -> usize {
        match self {
            Self::RowGroups => size_of::<u16>(),
            Self::BlockPerRow | Self::SplitK { .. } => 16,
        }
    }
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
    #[must_use]
    pub const fn workspace_bytes(&self) -> usize {
        0
    }

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
