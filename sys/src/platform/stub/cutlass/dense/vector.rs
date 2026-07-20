use super::{
    super::super::{Context, DeviceBuffer, Stream, unsupported},
    DenseMatmulDataType,
};
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseVectorSpec {
    pub n: usize,
    pub k: usize,
    pub data_type: DenseMatmulDataType,
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
