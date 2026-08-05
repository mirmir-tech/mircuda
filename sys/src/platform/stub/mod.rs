use crate::{Error, Result};

mod compiler;
mod graph;
#[cfg(feature = "marlin")]
mod marlin;
mod memory;
mod profile;
#[cfg(feature = "cublaslt")]
mod vendor;
pub use compiler::{CompileSpec, CompiledPtx, compiler_version};
pub use graph::{CaptureMode, Graph, KernelNode};
#[cfg(feature = "marlin")]
pub use marlin::{MarlinNvFp4MoeSpec, MarlinNvFp4RepackSpec, MarlinNvFp4ThreadConfig};
pub use profile::ProfilerRange;
#[cfg(feature = "cublaslt")]
pub use vendor::{
    CublasBf16Plan, CublasBf16Spec, CublasLtBf16Plan, CublasLtBf16Spec, CublasLtFp8Plan,
    CublasLtFp8Spec,
};

#[cfg(feature = "cutlass")]
mod cutlass;
#[cfg(feature = "cutlass")]
pub use cutlass::{
    BlockScaledFp4Plan, BlockScaledFp4Spec, BlockScaledFp4VectorPlan, BlockScaledFp4VectorSpec,
    BlockScaledMxFp8Plan, BlockScaledMxFp8Spec, BlockwiseFp8VectorPlan, BlockwiseFp8VectorSpec,
    DenseMatmulDataType, DenseMatmulPlan, DenseMatmulSpec, DenseVectorPlan, DenseVectorSpec,
    FmhaBf16Plan, FmhaBf16Spec, IndexedGroupedFp4Plan, IndexedGroupedFp4Spec,
    PairedVariableGroupedFp4Plan, ScaledFp8Plan, ScaledFp8ScaleType, ScaledFp8Spec, ScaledFp8Tile,
    ScaledFp8WeightScaleType, VariableGroupedBf16Plan, VariableGroupedBf16Spec,
    VariableGroupedFp4Plan, VariableGroupedFp4Spec,
};

#[derive(Debug)]
pub struct Driver;

impl Driver {
    pub const fn initialize() -> Result<Self> {
        Err(unsupported())
    }

    pub const fn device_count(&self) -> Result<usize> {
        Err(unsupported())
    }

    pub const fn create_context(&self, _ordinal: usize) -> Result<Context> {
        Err(unsupported())
    }
}

#[derive(Clone, Debug)]
pub struct Context;

impl Context {
    pub const fn memory_info(&self) -> Result<(usize, usize)> {
        Err(unsupported())
    }

    pub const fn device_info(&self) -> Result<DeviceInfo> {
        Err(unsupported())
    }

    pub const fn create_stream(&self) -> Result<Stream> {
        Err(unsupported())
    }

    pub const fn create_event(&self, _timing: bool) -> Result<Event> {
        Err(unsupported())
    }

    pub const fn default_memory_pool(&self) -> Result<MemoryPool> {
        Err(unsupported())
    }

    pub const fn allocate_pinned(&self, _bytes: usize) -> Result<PinnedBuffer> {
        Err(unsupported())
    }
}

#[derive(Clone, Debug)]
pub struct Stream;

impl Stream {
    pub const fn synchronize(&self) -> Result<()> {
        Err(unsupported())
    }

    pub const fn wait(&self, _event: &Event) -> Result<()> {
        Err(unsupported())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub multiprocessor_count: i32,
    pub total_memory: usize,
    pub memory_pools: bool,
    pub integrated: bool,
}

#[derive(Debug)]
pub struct Event;

impl Event {
    pub const fn record(&self, _stream: &Stream) -> Result<()> {
        Err(unsupported())
    }

    pub const fn synchronize(&self) -> Result<()> {
        Err(unsupported())
    }

    pub const fn elapsed_ms(&self, _end: &Self) -> Result<f32> {
        Err(unsupported())
    }
}

#[derive(Clone, Debug)]
pub struct MemoryPool;

impl MemoryPool {
    pub const fn set_release_threshold(&self, _bytes: u64) -> Result<()> {
        Err(unsupported())
    }

    pub const fn stats(&self) -> Result<MemoryPoolStats> {
        Err(unsupported())
    }

    pub const fn trim_to(&self, _bytes: usize) -> Result<()> {
        Err(unsupported())
    }

    pub const fn allocate(
        &self,
        _stream: &Stream,
        _bytes: usize,
        _zeroed: bool,
    ) -> Result<DeviceBuffer> {
        Err(unsupported())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryPoolStats {
    pub reserved: u64,
    pub used: u64,
}

#[derive(Debug)]
pub struct DeviceBuffer;

impl DeviceBuffer {
    #[must_use]
    pub const fn bytes(&self) -> usize {
        0
    }

    #[must_use]
    pub const fn argument(&self) -> KernelArgument {
        KernelArgument::Pointer { value: 0, stream: 0 }
    }
}

#[derive(Debug)]
pub struct PinnedBuffer;

impl PinnedBuffer {
    #[must_use]
    pub const fn bytes(&self) -> usize {
        0
    }

    pub fn with_slice<T, R>(&self, _len: usize, _read: impl FnOnce(&[T]) -> R) -> Result<R> {
        Err(unsupported())
    }

    pub fn with_mut_slice<T, R>(
        &mut self,
        _len: usize,
        _write: impl FnOnce(&mut [T]) -> R,
    ) -> Result<R> {
        Err(unsupported())
    }
}

#[derive(Clone, Debug)]
pub struct Module {
    pub marker: usize,
}

impl Module {
    pub const fn kernel(&self, _name: &str) -> Result<Kernel> {
        Err(unsupported())
    }
}

#[derive(Clone, Debug)]
pub struct Kernel {
    pub marker: usize,
}

impl Kernel {
    pub const fn launch(
        &self,
        _stream: &Stream,
        _config: LaunchConfig,
        _arguments: &mut [KernelArgument],
    ) -> Result<()> {
        Err(unsupported())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_memory_bytes: u32,
}

#[derive(Debug)]
pub enum KernelArgument {
    Pointer { value: usize, stream: usize },
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    Usize(usize),
    Isize(isize),
    F32(f32),
    F64(f64),
}

const fn unsupported() -> Error {
    Error::UnsupportedPlatform(std::env::consts::OS)
}
