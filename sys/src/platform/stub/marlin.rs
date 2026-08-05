use super::{Context, DeviceBuffer, Stream, unsupported};
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MarlinNvFp4RepackSpec {
    pub experts: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MarlinNvFp4MoeSpec {
    pub experts: usize,
    pub tokens: usize,
    pub top_k: usize,
    pub n: usize,
    pub k: usize,
    pub thread_config: MarlinNvFp4ThreadConfig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum MarlinNvFp4ThreadConfig {
    N128K128 = 0,
    N128K64 = 1,
    N64K128 = 2,
}

impl Context {
    #[allow(clippy::too_many_arguments)]
    pub const fn marlin_nvfp4_dense(
        &self,
        _stream: &Stream,
        _spec: MarlinNvFp4MoeSpec,
        _input: &DeviceBuffer,
        _weight: &DeviceBuffer,
        _scales: &DeviceBuffer,
        _global_scale: &DeviceBuffer,
        _temporary: &DeviceBuffer,
        _locks: &DeviceBuffer,
        _output: &DeviceBuffer,
        _atomic_reduce: bool,
    ) -> Result<()> {
        Err(unsupported())
    }

    pub const fn marlin_repack_nvfp4(
        &self,
        _stream: &Stream,
        _spec: MarlinNvFp4RepackSpec,
        _input: &DeviceBuffer,
        _output: &DeviceBuffer,
    ) -> Result<()> {
        Err(unsupported())
    }

    pub const fn marlin_repack_nvfp4_pair(
        &self,
        _stream: &Stream,
        _spec: MarlinNvFp4RepackSpec,
        _left: &DeviceBuffer,
        _right: &DeviceBuffer,
        _output: &DeviceBuffer,
    ) -> Result<()> {
        Err(unsupported())
    }

    #[allow(clippy::too_many_arguments)]
    pub const fn marlin_prepare_nvfp4_scales(
        &self,
        _stream: &Stream,
        _spec: MarlinNvFp4RepackSpec,
        _left: &DeviceBuffer,
        _right: Option<&DeviceBuffer>,
        _globals: &DeviceBuffer,
        _output: &DeviceBuffer,
        _output_globals: &DeviceBuffer,
        _maximum: &DeviceBuffer,
    ) -> Result<()> {
        Err(unsupported())
    }

    #[allow(clippy::too_many_arguments)]
    pub const fn marlin_prepare_moe_routes(
        &self,
        _stream: &Stream,
        _spec: MarlinNvFp4MoeSpec,
        _selected: &DeviceBuffer,
        _routing: &DeviceBuffer,
        _sorted: &DeviceBuffer,
        _expert_ids: &DeviceBuffer,
        _padded: &DeviceBuffer,
        _offsets: &DeviceBuffer,
        _routing_f32: &DeviceBuffer,
    ) -> Result<()> {
        Err(unsupported())
    }

    #[allow(clippy::too_many_arguments)]
    pub const fn marlin_nvfp4_moe(
        &self,
        _stream: &Stream,
        _spec: MarlinNvFp4MoeSpec,
        _input: &DeviceBuffer,
        _weight: &DeviceBuffer,
        _scales: &DeviceBuffer,
        _globals: &DeviceBuffer,
        _routing: &DeviceBuffer,
        _sorted: &DeviceBuffer,
        _expert_ids: &DeviceBuffer,
        _padded: &DeviceBuffer,
        _temporary: &DeviceBuffer,
        _locks: &DeviceBuffer,
        _output: &DeviceBuffer,
        _multiply_routing: bool,
        _atomic_reduce: bool,
    ) -> Result<()> {
        Err(unsupported())
    }
}
