use std::{marker::PhantomData, num::NonZeroU16};

pub use mircuda_sys::DenseVectorKernel;

use super::DenseMatmulElement;
use crate::{Context, DeviceBuffer, Error, Result, Stream};

/// Largest output count that defaults to [`DenseVectorKernel::BlockPerRow`].
const BLOCK_PER_ROW_DEFAULT_OUTPUTS: usize = 512;
/// The widest accepted input: a kernel's column loop steps at most 256 past
/// its last column, which then stays an `i32`.
pub const DENSE_VECTOR_MAX_COLUMNS: usize = i32::MAX as usize - 255;
/// Waves of one block per SM that a split-K grid aims to fill.
const SPLIT_K_TARGET_WAVES: u32 = 2;

/// Fixed geometry and kernel for
/// `output[N] = alpha * weight[N,K] * input[K] + beta * output[N]`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DenseVectorSpec {
    n: usize,
    k: usize,
    kernel: DenseVectorKernel,
}

impl DenseVectorSpec {
    /// Creates a validated row-major matrix-vector shape with the default
    /// kernel: [`DenseVectorKernel::BlockPerRow`] for router-sized outputs
    /// (`n <= 512`) when `k` admits it, otherwise
    /// [`DenseVectorKernel::RowGroups`]. Tuners choose other kernels through
    /// [`Self::with_kernel`]. `n` may reach `i32::MAX`; `k`, at most
    /// [`DENSE_VECTOR_MAX_COLUMNS`].
    pub fn new(n: usize, k: usize) -> Result<Self> {
        if n == 0 || k == 0 {
            return Err(Error::InvalidMatmulShape);
        }
        for value in [n, k] {
            let _ = i32::try_from(value)?;
        }
        if k > DENSE_VECTOR_MAX_COLUMNS {
            return Err(Error::InvalidMatmulShape);
        }
        let _ = n.checked_mul(k).ok_or(Error::InvalidMatmulShape)?;
        let kernel = if n <= BLOCK_PER_ROW_DEFAULT_OUTPUTS
            && Self::admits(n, k, DenseVectorKernel::BlockPerRow)
        {
            DenseVectorKernel::BlockPerRow
        } else {
            DenseVectorKernel::RowGroups
        };
        Ok(Self { n, k, kernel })
    }

    /// Selects another kernel for the same shape, as a tuner measures them.
    ///
    /// [`DenseVectorKernel::BlockPerRow`] needs `k % 8 == 0`;
    /// [`DenseVectorKernel::SplitK`] also needs `splits <= k / 8` and an
    /// `n * splits` grid that CUDA can launch.
    pub const fn with_kernel(self, kernel: DenseVectorKernel) -> Result<Self> {
        if Self::admits(self.n, self.k, kernel) {
            Ok(Self { kernel, ..self })
        } else {
            Err(Error::VectorKernelShape { kernel, n: self.n, k: self.k })
        }
    }

    /// Split-K kernels worth measuring for this shape on a device with
    /// `multiprocessors` SMs, each when the shape admits it: the slice count
    /// whose `n * splits` grid first fills two waves of one block per SM and
    /// its doubled neighbour. Outputs that fill two waves on their own (such
    /// as `n >= 96` on 48 SMs) measure only two slices, which halves each
    /// block's pass over a row.
    #[must_use]
    pub fn split_k_candidates(self, multiprocessors: u32) -> Vec<DenseVectorKernel> {
        let blocks = u64::from(multiprocessors) * u64::from(SPLIT_K_TARGET_WAVES);
        let target = blocks.div_ceil(self.n as u64);
        let splits = if target > 1 {
            [Some(target), Some(target * 2)]
        } else {
            [Some(2), None]
        };
        splits
            .into_iter()
            .flatten()
            .filter_map(|splits| u16::try_from(splits).ok().and_then(NonZeroU16::new))
            .map(|splits| DenseVectorKernel::SplitK { splits })
            .filter(|kernel| Self::admits(self.n, self.k, *kernel))
            .collect()
    }

    /// Kernel that executes this shape.
    #[must_use]
    pub const fn kernel(self) -> DenseVectorKernel {
        self.kernel
    }

    const fn admits(n: usize, k: usize, kernel: DenseVectorKernel) -> bool {
        match kernel {
            DenseVectorKernel::RowGroups => true,
            DenseVectorKernel::BlockPerRow => k.is_multiple_of(8),
            DenseVectorKernel::SplitK { splits } => {
                let splits = splits.get() as usize;
                k.is_multiple_of(8)
                    && splits <= k / 8
                    && matches!(n.checked_mul(splits), Some(blocks) if blocks <= i32::MAX as usize)
            },
        }
    }

    /// Output element count.
    #[must_use]
    pub const fn n(self) -> usize {
        self.n
    }

    /// Reduction dimension.
    #[must_use]
    pub const fn k(self) -> usize {
        self.k
    }
}

/// Stream-bound native matrix-vector plan; split-K kernels retain a small
/// FP32 workspace allocated with the plan, so launches stay capture-safe.
#[derive(Debug)]
pub struct DenseVectorPlan<T: DenseMatmulElement> {
    native: mircuda_sys::DenseVectorPlan,
    spec: DenseVectorSpec,
    marker: PhantomData<T>,
}

impl<T: DenseMatmulElement> DenseVectorPlan<T> {
    /// Creates a fixed-shape plan without synchronizing the stream.
    pub fn new(context: &Context, stream: &Stream, spec: DenseVectorSpec) -> Result<Self> {
        let native_spec = mircuda_sys::DenseVectorSpec {
            n: spec.n,
            k: spec.k,
            data_type: T::NATIVE_TYPE,
            kernel: spec.kernel,
        };
        Ok(Self {
            native: context.native.create_dense_vector_plan(&stream.native, native_spec)?,
            spec,
            marker: PhantomData,
        })
    }

    /// Plan-owned workspace bytes; nonzero only for split-K kernels.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn workspace_bytes(&self) -> usize {
        self.native.workspace_bytes()
    }

    /// Returns the immutable plan geometry.
    #[must_use]
    pub const fn spec(&self) -> DenseVectorSpec {
        self.spec
    }

    /// Enqueues the fixed-shape matrix-vector multiplication.
    pub fn execute(
        &mut self,
        stream: &Stream,
        input: &DeviceBuffer<T>,
        weight: &DeviceBuffer<T>,
        output: &mut DeviceBuffer<T>,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        validate_len("input", self.spec.k, input.len())?;
        validate_len("weight", self.spec.n * self.spec.k, weight.len())?;
        validate_len("output", self.spec.n, output.len())?;
        let (kernel, alignment) = (self.spec.kernel, self.spec.kernel.operand_alignment());
        for (operand, buffer) in [("input", &input.native), ("weight", &weight.native)] {
            if !buffer.is_aligned(alignment) {
                return Err(Error::MisalignedOperand { operand, alignment, kernel });
            }
        }
        Ok(self
            .native
            .execute(&stream.native, &input.native, &weight.native, &output.native, alpha, beta)?)
    }
}

const fn validate_len(operand: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::MatmulLengthMismatch { operand, expected, actual })
    }
}
