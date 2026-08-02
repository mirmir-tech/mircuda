/// Result returned by mircuda operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Failure reported by the safe CUDA gateway.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The native CUDA boundary rejected the operation.
    #[error(transparent)]
    Native(#[from] mircuda_sys::Error),
    /// A public dimension cannot be represented by CUDA's scalar ABI.
    #[error(transparent)]
    IntegerConversion(#[from] std::num::TryFromIntError),
    /// A device or pinned allocation cannot contain zero elements.
    #[error("CUDA allocations must contain at least one element")]
    EmptyAllocation,
    /// An element count overflowed the addressable byte size.
    #[error("CUDA allocation size overflow for {elements} elements of {element_bytes} bytes")]
    AllocationOverflow {
        /// Requested element count.
        elements: usize,
        /// Size of one element.
        element_bytes: usize,
    },
    /// The CUDA memory pool rejected a device allocation.
    #[error("CUDA device allocation of {bytes} bytes failed: {source}")]
    DeviceAllocation {
        /// Requested allocation size in bytes.
        bytes: usize,
        /// Failure returned by the native CUDA boundary.
        #[source]
        source: mircuda_sys::Error,
    },
    /// Source and target buffers have different lengths.
    #[error("CUDA transfer length mismatch: source {source_len}, target {target_len}")]
    LengthMismatch {
        /// Number of source elements.
        source_len: usize,
        /// Number of target elements.
        target_len: usize,
    },
    /// A device-to-device copy range is empty, overflows, or exceeds an allocation.
    #[error("invalid CUDA device transfer range")]
    InvalidTransferRange,
    /// A typed device view does not divide the underlying allocation exactly.
    #[error("CUDA device allocation cannot be viewed as the requested element type")]
    InvalidDeviceView,
    /// Launch geometry is empty or exceeds CUDA's representable grid.
    #[error("invalid CUDA launch geometry")]
    InvalidLaunch,
    /// A matrix dimension is zero or overflows an addressable element count.
    #[error("invalid matrix multiplication shape")]
    InvalidMatmulShape,
    /// A typed matrix allocation does not match the fixed plan geometry.
    #[error("matrix {operand} length mismatch: expected {expected}, got {actual}")]
    MatmulLengthMismatch {
        /// Matrix operand name.
        operand: &'static str,
        /// Element count required by the plan.
        expected: usize,
        /// Element count supplied at execution.
        actual: usize,
    },
}
