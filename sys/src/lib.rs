#[cfg(target_os = "linux")]
mod platform;
#[cfg(not(target_os = "linux"))]
#[path = "platform/stub/mod.rs"]
mod platform;

#[cfg(feature = "cutlass")]
pub use platform::{
    BlockScaledFp4Plan, BlockScaledFp4Spec, BlockScaledFp4VectorPlan, BlockScaledFp4VectorSpec,
    BlockScaledMxFp8Plan, BlockScaledMxFp8Spec, BlockwiseFp8VectorPlan, BlockwiseFp8VectorSpec,
    DenseMatmulDataType, DenseMatmulPlan, DenseMatmulSpec, DenseVectorPlan, DenseVectorSpec,
    FmhaBf16Plan, FmhaBf16Spec, IndexedGroupedFp4Plan, IndexedGroupedFp4Spec,
    PairedVariableGroupedFp4Plan, ScaledFp8Plan, ScaledFp8ScaleType, ScaledFp8Spec,
    ScaledFp8WeightScaleType, VariableGroupedBf16Plan, VariableGroupedBf16Spec,
    VariableGroupedFp4Plan, VariableGroupedFp4Spec,
};
pub use platform::{
    CaptureMode, CompileSpec, CompiledPtx, Context, DeviceBuffer, DeviceInfo, Driver, Event, Graph,
    Kernel, KernelArgument, KernelNode, LaunchConfig, MemoryPool, MemoryPoolStats, Module,
    PinnedBuffer, ProfilerRange, Stream, compiler_version,
};
#[cfg(feature = "cublaslt")]
pub use platform::{CublasBf16Plan, CublasBf16Spec, CublasLtBf16Plan, CublasLtBf16Spec};

#[cfg(all(target_os = "linux", feature = "cutlass"))]
unsafe extern "C" {
    fn mircuda_cutlass_version() -> u32;
}

/// Version of the linked, pinned CUTLASS AOT dependency.
#[cfg(all(target_os = "linux", feature = "cutlass"))]
#[must_use]
pub fn cutlass_version() -> u32 {
    // SAFETY: the feature links the matching no-argument C ABI symbol.
    unsafe { mircuda_cutlass_version() }
}

/// Native CUDA boundary failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// CUDA is currently supported only on Linux hosts.
    #[error("CUDA execution is unsupported on {0}")]
    UnsupportedPlatform(&'static str),
    /// The CUDA Driver API returned an error.
    #[cfg(target_os = "linux")]
    #[error(transparent)]
    Driver(#[from] cudarc::driver::DriverError),
    /// NVRTC rejected a CUDA translation unit.
    #[cfg(target_os = "linux")]
    #[error(transparent)]
    Nvrtc(#[from] cudarc::nvrtc::CompileError),
    /// The NVRTC runtime API rejected a metadata query.
    #[cfg(target_os = "linux")]
    #[error(transparent)]
    NvrtcApi(#[from] cudarc::nvrtc::result::NvrtcError),
    /// A driver value could not be represented by the public Rust type.
    #[cfg(target_os = "linux")]
    #[error(transparent)]
    IntegerConversion(#[from] std::num::TryFromIntError),
    /// A loaded module and launch stream belong to different contexts.
    #[error("CUDA module and stream belong to different contexts")]
    ContextMismatch,
    /// Device memory was used on a stream other than its allocation stream.
    #[error("CUDA buffer must be used on its allocation stream")]
    StreamMismatch,
    /// Stream capture completed without producing a CUDA graph.
    #[error("CUDA stream capture produced an empty graph")]
    EmptyGraph,
    /// A graph node does not belong to the graph being updated.
    #[error("CUDA graph node does not belong to this graph")]
    GraphNodeMismatch,
    /// A captured launch did not produce one active kernel dependency.
    #[error("CUDA capture did not expose one active kernel dependency")]
    MissingCapturedKernel,
    /// NVRTC did not return a loadable PTX image.
    #[error("NVRTC returned no loadable PTX image")]
    MissingPtx,
    /// A persisted PTX image is empty or is not NUL-terminated.
    #[error("PTX image is not a NUL-terminated driver module")]
    InvalidPtx,
    /// CUDA returned a null host allocation.
    #[error("CUDA returned a null host allocation")]
    NullAllocation,
    /// A CUDA symbol contains an interior NUL byte.
    #[cfg(target_os = "linux")]
    #[error(transparent)]
    Nul(#[from] std::ffi::NulError),
    /// A typed host view does not exactly cover its pinned allocation.
    #[error("typed pinned-host view does not match its allocation size")]
    InvalidHostView,
    /// Host and device transfer byte lengths differ.
    #[error("CUDA transfer byte lengths differ")]
    TransferSizeMismatch,
    /// A device-to-device transfer range exceeds one of its allocations.
    #[error("CUDA device transfer range exceeds its allocation")]
    InvalidTransferRange,
    /// Native matrix buffers do not cover their declared shape.
    #[error("CUDA matrix buffers do not match the plan shape")]
    InvalidMatmulBuffer,
    /// An ahead-of-time CUTLASS plan rejected construction or execution.
    #[error("CUTLASS plan failed with status {0}")]
    Cutlass(i32),
    /// A persistent cuBLASLt plan rejected construction or execution.
    #[error("cuBLASLt plan failed with status {0}")]
    CublasLt(i32),
    /// A persistent cuBLAS plan rejected construction or execution.
    #[error("cuBLAS plan failed with status {0}")]
    Cublas(i32),
}

/// Result returned by the native CUDA boundary.
pub type Result<T> = std::result::Result<T, Error>;
