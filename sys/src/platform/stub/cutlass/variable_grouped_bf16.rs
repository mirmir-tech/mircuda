use super::super::{Context, DeviceBuffer, Stream, unsupported};
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariableGroupedBf16Spec {
    pub groups: usize,
    pub max_rows: usize,
    pub n: usize,
    pub k: usize,
    pub capacity_rows: usize,
}

#[derive(Debug)]
pub struct VariableGroupedBf16Plan;

impl Context {
    pub const fn create_variable_grouped_bf16_plan(
        &self,
        _stream: &Stream,
        _spec: VariableGroupedBf16Spec,
    ) -> Result<VariableGroupedBf16Plan> {
        Err(unsupported())
    }
}

impl VariableGroupedBf16Plan {
    #[allow(clippy::too_many_arguments)]
    pub const fn execute(
        &mut self,
        _stream: &Stream,
        _input: &DeviceBuffer,
        _weights: &DeviceBuffer,
        _rows: &DeviceBuffer,
        _offsets: &DeviceBuffer,
        _output: &DeviceBuffer,
        _beta: f32,
    ) -> Result<()> {
        Err(unsupported())
    }
}
