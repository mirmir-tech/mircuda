#![doc = include_str!("../README.md")]
#![deny(missing_docs)]

extern crate self as mircuda;

mod compiler;
#[cfg(feature = "cutlass")]
mod cutlass;
mod driver;
mod error;
mod event;
mod graph;
mod launch;
#[cfg(feature = "marlin")]
mod marlin;
mod memory;
mod mxfp8;
mod profile;
mod source;
#[cfg(feature = "cublaslt")]
mod vendor;

pub use compiler::{CompileCacheStats, CompileOptions, Compiler, CompilerConfig, Module};
#[cfg(feature = "cutlass")]
pub use cutlass::{
    BlockScaledFp4Plan, BlockScaledFp4Spec, BlockScaledFp4VectorPlan, BlockScaledFp4VectorSpec,
    BlockScaledMxFp8Plan, BlockScaledMxFp8Spec, BlockwiseFp8VectorPlan, BlockwiseFp8VectorSpec,
    DenseMatmulElement, DenseMatmulOutput, DenseMatmulPlan, DenseMatmulSpec, DenseVectorPlan,
    DenseVectorSpec, FmhaBf16Plan, FmhaBf16Spec, IndexedGroupedFp4Plan, IndexedGroupedFp4Spec,
    PairedVariableGroupedFp4Launch, PairedVariableGroupedFp4Plan, ScaledFp8Plan, ScaledFp8Scale,
    ScaledFp8Spec, ScaledFp8Tile, ScaledFp8WeightScale, VariableGroupedBf16Plan,
    VariableGroupedBf16Spec, VariableGroupedFp4Metadata, VariableGroupedFp4Operands,
    VariableGroupedFp4Plan, VariableGroupedFp4Spec,
};
pub use driver::{Context, Device, DeviceInfo, Driver, Stream};
pub use error::{Error, Result};
pub use event::Event;
pub use graph::{CaptureMode, Graph, KernelNode};
pub use half::{bf16, f16};
pub use launch::{
    KernelArguments, KernelScalar, KernelSignature, LaunchConfig, ScalarValue, TypedKernel,
};
#[cfg(feature = "marlin")]
pub use marlin::{
    MarlinNvFp4DenseOperands, MarlinNvFp4MoeOperands, MarlinNvFp4MoeSpec, MarlinNvFp4RepackSpec,
    MarlinNvFp4ThreadConfig,
};
pub use memory::{DeviceBuffer, DeviceElement, MemoryPool, MemoryPoolStats, PinnedBuffer};
pub use mircuda_macros::{cuda_export, cuda_kernel};
pub use mxfp8::{
    MxFp8Embedding, MxFp8EmbeddingOperands, MxFp8EmbeddingSpec, MxFp8Gathered,
    MxFp8GatheredOperands, MxFp8GatheredSpec, MxFp8Matmul, MxFp8Spec,
};
#[cfg(feature = "cutlass")]
pub use mxfp8::{MxFp8TensorCore, MxFp8TensorCoreScratch};
pub use profile::ProfilerRange;
#[cfg(feature = "cublaslt")]
pub use vendor::{
    CublasBf16Plan, CublasBf16Spec, CublasLtBf16Plan, CublasLtBf16Spec, CublasLtFp8Plan,
    CublasLtFp8Spec,
};

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
