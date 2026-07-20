use super::super::{Context, DeviceBuffer, Stream, unsupported};
use crate::Result;

mod vector;

pub use vector::{DenseVectorPlan, DenseVectorSpec};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseMatmulDataType {
    F16,
    Bf16,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseMatmulSpec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub input_type: DenseMatmulDataType,
    pub output_type: DenseMatmulDataType,
}

#[derive(Debug)]
pub struct DenseMatmulPlan;

impl Context {
    pub const fn create_dense_matmul_plan(
        &self,
        _stream: &Stream,
        _spec: DenseMatmulSpec,
    ) -> Result<DenseMatmulPlan> {
        Err(unsupported())
    }
}

impl DenseMatmulPlan {
    #[must_use]
    pub const fn workspace_bytes(&self) -> usize {
        0
    }

    #[allow(clippy::too_many_arguments)]
    pub const fn execute(
        &mut self,
        _stream: &Stream,
        _a: &DeviceBuffer,
        _b: &DeviceBuffer,
        _c: &DeviceBuffer,
        _alpha: f32,
        _beta: f32,
    ) -> Result<()> {
        Err(unsupported())
    }
}
