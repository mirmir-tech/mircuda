use super::{Context, DeviceBuffer, Stream, unsupported};
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CublasBf16Spec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Debug)]
pub struct CublasBf16Plan;

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
    pub const fn create_cublas_bf16_plan(
        &self,
        _stream: &Stream,
        _spec: CublasBf16Spec,
    ) -> Result<CublasBf16Plan> {
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

impl CublasBf16Plan {
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
