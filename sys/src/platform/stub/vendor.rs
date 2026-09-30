use super::{Context, DeviceBuffer, Stream, unsupported};
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CublasDataType {
    Bf16,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CublasDenseSpec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub data_type: CublasDataType,
}

#[derive(Debug)]
pub struct CublasDensePlan;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CublasLtBf16Spec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Debug)]
pub struct CublasLtBf16Plan;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CublasLtFp8Spec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Debug)]
pub struct CublasLtFp8Plan;

impl Context {
    pub const fn create_cublas_dense_plan(
        &self,
        _stream: &Stream,
        _spec: CublasDenseSpec,
    ) -> Result<CublasDensePlan> {
        Err(unsupported())
    }

    pub const fn create_cublaslt_bf16_plan(
        &self,
        _stream: &Stream,
        _spec: CublasLtBf16Spec,
    ) -> Result<CublasLtBf16Plan> {
        Err(unsupported())
    }

    pub const fn create_cublaslt_fp8_plan(
        &self,
        _stream: &Stream,
        _spec: CublasLtFp8Spec,
    ) -> Result<CublasLtFp8Plan> {
        Err(unsupported())
    }
}

impl CublasDensePlan {
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

impl CublasLtBf16Plan {
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

impl CublasLtFp8Plan {
    #[allow(clippy::too_many_arguments)]
    pub const fn execute(
        &mut self,
        _stream: &Stream,
        _a: &DeviceBuffer,
        _b: &DeviceBuffer,
        _a_scale: &DeviceBuffer,
        _b_scale: &DeviceBuffer,
        _c: &DeviceBuffer,
    ) -> Result<()> {
        Err(unsupported())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CublasGemmOperand {
    pub leading: usize,
    pub stride: usize,
    pub transposed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CublasGemmSpec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
    pub first: CublasGemmOperand,
    pub second: CublasGemmOperand,
    pub output_leading: usize,
    pub output_stride: usize,
    pub input_type: CublasDataType,
    pub output_type: CublasDataType,
}

#[derive(Debug)]
pub struct CublasGemmPlan;

impl Context {
    pub const fn create_cublas_gemm_plan(
        &self,
        _stream: &Stream,
        _spec: CublasGemmSpec,
    ) -> Result<CublasGemmPlan> {
        Err(unsupported())
    }
}

impl CublasGemmPlan {
    pub const fn execute(
        &mut self,
        _stream: &Stream,
        _buffers: (&DeviceBuffer, &DeviceBuffer, &DeviceBuffer),
        _offsets: [usize; 3],
        _alpha: f32,
        _beta: f32,
    ) -> Result<()> {
        Err(unsupported())
    }
}
