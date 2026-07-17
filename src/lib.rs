#![doc = include_str!("../README.md")]
#![deny(missing_docs)]

mod compiler;
#[cfg(feature = "cutlass")]
mod cutlass;
mod driver;
mod error;
mod event;
mod graph;
mod launch;
mod memory;
mod profile;
mod source;

pub use compiler::{CompileCacheStats, CompileOptions, Compiler, CompilerConfig, Module};
#[cfg(feature = "cutlass")]
pub use cutlass::{
    BlockScaledFp4Plan, BlockScaledFp4Spec, BlockScaledFp4VectorPlan, BlockScaledFp4VectorSpec,
    BlockwiseFp8VectorPlan, BlockwiseFp8VectorSpec, DenseMatmulElement, DenseMatmulOutput,
    DenseMatmulPlan, DenseMatmulSpec, DenseVectorPlan, DenseVectorSpec, IndexedGroupedFp4Plan,
    IndexedGroupedFp4Spec, PairedVariableGroupedFp4Launch, PairedVariableGroupedFp4Plan,
    VariableGroupedFp4Metadata, VariableGroupedFp4Operands, VariableGroupedFp4Plan,
    VariableGroupedFp4Spec,
};
pub use driver::{Context, Device, DeviceInfo, Driver, Stream};
pub use error::{Error, Result};
pub use event::Event;
pub use graph::{CaptureMode, Graph, KernelNode};
pub use half::{bf16, f16};
pub use launch::{
    KernelArguments, KernelScalar, KernelSignature, LaunchConfig, ScalarValue, TypedKernel,
};
pub use memory::{DeviceBuffer, DeviceElement, MemoryPool, MemoryPoolStats, PinnedBuffer};
pub use mircuda_macros::{cuda_export, cuda_kernel};
pub use profile::ProfilerRange;

/// Version of the linked CUTLASS AOT backend encoded as `major * 10000 + minor * 100 + patch`.
#[cfg(all(target_os = "linux", feature = "cutlass"))]
#[must_use]
pub fn cutlass_version() -> u32 {
    mircuda_sys::cutlass_version()
}
pub use source::KernelSource;

/// Embeds a CUDA translation unit from a file relative to the calling crate.
#[macro_export]
macro_rules! cuda_kernel_file {
    ($path:literal) => {
        $crate::KernelSource::embedded($path, include_str!($path))
    };
}

/// Embeds ordered CUDA fragments as one self-contained translation unit.
#[macro_export]
macro_rules! cuda_kernel_files {
    ($name:literal; $($path:literal),+ $(,)?) => {
        $crate::KernelSource::composed($name, &[$(include_str!($path)),+])
    };
}
