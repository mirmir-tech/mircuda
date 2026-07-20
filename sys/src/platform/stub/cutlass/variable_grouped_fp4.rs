use super::super::{Context, DeviceBuffer, Stream, unsupported};
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariableGroupedFp4Spec {
    pub groups: usize,
    pub matrices: usize,
    pub max_m: usize,
    pub n: usize,
    pub k: usize,
    pub capacity_rows: usize,
}

#[derive(Debug)]
pub struct VariableGroupedFp4Plan;

#[derive(Debug)]
pub struct PairedVariableGroupedFp4Plan;

impl Context {
    pub const fn create_variable_grouped_fp4_plan(
        &self,
        _stream: &Stream,
        _spec: VariableGroupedFp4Spec,
    ) -> Result<VariableGroupedFp4Plan> {
        Err(unsupported())
    }

    pub const fn create_paired_variable_grouped_fp4_plan(
        &self,
        _stream: &Stream,
        _spec: VariableGroupedFp4Spec,
    ) -> Result<PairedVariableGroupedFp4Plan> {
        Err(unsupported())
    }
}

impl PairedVariableGroupedFp4Plan {
    #[allow(clippy::too_many_arguments)]
    pub const fn execute(
        &mut self,
        _stream: &Stream,
        _left_a: &DeviceBuffer,
        _left_a_scales: &DeviceBuffer,
        _left_b: &DeviceBuffer,
        _left_b_scales: &DeviceBuffer,
        _left_alphas: &DeviceBuffer,
        _right_a: &DeviceBuffer,
        _right_a_scales: &DeviceBuffer,
        _right_b: &DeviceBuffer,
        _right_b_scales: &DeviceBuffer,
        _right_alphas: &DeviceBuffer,
        _indices: &DeviceBuffer,
        _rows: &DeviceBuffer,
        _offsets: &DeviceBuffer,
        _left_c: &DeviceBuffer,
        _right_c: &DeviceBuffer,
    ) -> Result<()> {
        Err(unsupported())
    }
}

impl VariableGroupedFp4Plan {
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
        _rows: &DeviceBuffer,
        _offsets: &DeviceBuffer,
        _c: &DeviceBuffer,
    ) -> Result<()> {
        Err(unsupported())
    }
}
