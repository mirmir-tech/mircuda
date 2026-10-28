use std::marker::PhantomData;

pub use mircuda_sys::CublasGemmOperand;

use super::CublasElement;
use crate::{Context, DeviceBuffer, Error, Result, Stream};

/// Row-major `output[M,N] = op(left)[M,K] · op(right)[K,N]` over `batch`
/// members, each operand addressed by leading dimension and batch stride so
/// views into a fused projection need no copy.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CublasGemmSpec {
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    left: CublasGemmOperand,
    right: CublasGemmOperand,
    output: CublasGemmOperand,
}

impl CublasGemmSpec {
    /// Validates that every stored row holds its operand's row width.
    pub fn new(
        (m, n, k, batch): (usize, usize, usize, usize),
        left: CublasGemmOperand,
        right: CublasGemmOperand,
        output: CublasGemmOperand,
    ) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 || batch == 0 || output.transposed {
            return Err(Error::InvalidMatmulShape);
        }
        let left_width = if left.transposed {
            m
        } else {
            k
        };
        let right_width = if right.transposed {
            k
        } else {
            n
        };
        if left.leading < left_width || right.leading < right_width || output.leading < n {
            return Err(Error::InvalidMatmulShape);
        }
        for value in [m, n, k, batch] {
            let _ = i32::try_from(value)?;
        }
        Ok(Self { m, n, k, batch, left, right, output })
    }

    /// cuBLAS is column-major: a row-major output is the column-major
    /// transpose, computed as `op(right)ᵀ · op(left)ᵀ` from the same memory.
    const fn native(
        self,
        input: mircuda_sys::CublasDataType,
        output: mircuda_sys::CublasDataType,
    ) -> mircuda_sys::CublasGemmSpec {
        mircuda_sys::CublasGemmSpec {
            m: self.n,
            n: self.m,
            k: self.k,
            batch: self.batch,
            first: self.right,
            second: self.left,
            output_leading: self.output.leading,
            output_stride: self.output.stride,
            input_type: input,
            output_type: output,
        }
    }
}

/// Element offsets of the first batch member of each operand.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CublasGemmOffsets {
    /// Offset of the left operand.
    pub left: usize,
    /// Offset of the right operand.
    pub right: usize,
    /// Offset of the output.
    pub output: usize,
}

/// Stream-bound strided batched cuBLAS plan. `f32` inputs run in full single
/// precision; `bf16` inputs use tensor cores with F32 accumulation.
#[derive(Debug)]
pub struct CublasGemmPlan<I: CublasElement, O: CublasElement> {
    native: mircuda_sys::CublasGemmPlan,
    spec: CublasGemmSpec,
    marker: PhantomData<(I, O)>,
}

impl<I: CublasElement, O: CublasElement> CublasGemmPlan<I, O> {
    /// Creates the fixed-geometry plan without synchronizing its stream.
    pub fn new(context: &Context, stream: &Stream, spec: CublasGemmSpec) -> Result<Self> {
        let native = spec.native(I::DATA_TYPE, O::DATA_TYPE);
        Ok(Self {
            native: context.native.create_cublas_gemm_plan(&stream.native, native)?,
            spec,
            marker: PhantomData,
        })
    }

    /// Returns the immutable plan geometry.
    #[must_use]
    pub const fn spec(&self) -> CublasGemmSpec {
        self.spec
    }

    /// Enqueues `output = alpha · op(left) · op(right) + beta · output` for
    /// every batch member.
    pub fn execute(
        &mut self,
        stream: &Stream,
        (left, right, output): (&DeviceBuffer<I>, &DeviceBuffer<I>, &mut DeviceBuffer<O>),
        offsets: CublasGemmOffsets,
        (alpha, beta): (f32, f32),
    ) -> Result<()> {
        Ok(self.native.execute(
            &stream.native,
            [&right.native, &left.native, &output.native],
            [offsets.right, offsets.left, offsets.output],
            alpha,
            beta,
        )?)
    }
}
