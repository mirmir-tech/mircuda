mod vector;

pub use vector::{BlockScaledFp4VectorPlan, BlockScaledFp4VectorSpec};

use crate::{Context, DeviceBuffer, Error, Result, Stream, bf16};

/// Fixed geometry for block-scaled `FP4 × FP4 → BF16` multiplication.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BlockScaledFp4Spec {
    m: usize,
    n: usize,
    k: usize,
}

impl BlockScaledFp4Spec {
    /// Creates `C[M,N] = alpha * A[M,K] * B[N,K]^T`.
    pub fn new(m: usize, n: usize, k: usize) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 || !k.is_multiple_of(64) {
            return Err(Error::InvalidMatmulShape);
        }
        for value in [m, n, k] {
            let _ = i32::try_from(value)?;
        }
        let _ = m.checked_mul(k).ok_or(Error::InvalidMatmulShape)?;
        let _ = n.checked_mul(k).ok_or(Error::InvalidMatmulShape)?;
        let _ = m.checked_mul(n).ok_or(Error::InvalidMatmulShape)?;
        Ok(Self { m, n, k })
    }

    /// Output row count.
    #[must_use]
    pub const fn m(self) -> usize {
        self.m
    }

    /// Output column count.
    #[must_use]
    pub const fn n(self) -> usize {
        self.n
    }

    /// Reduction dimension.
    #[must_use]
    pub const fn k(self) -> usize {
        self.k
    }

    const fn native(self) -> mircuda_sys::BlockScaledFp4Spec {
        mircuda_sys::BlockScaledFp4Spec { m: self.m, n: self.n, k: self.k }
    }
}

/// Stream-bound CUTLASS FP4 plan with persistent workspace.
#[derive(Debug)]
pub struct BlockScaledFp4Plan {
    native: mircuda_sys::BlockScaledFp4Plan,
    spec: BlockScaledFp4Spec,
}

impl BlockScaledFp4Plan {
    /// Creates the fixed-shape plan without synchronizing the stream.
    pub fn new(context: &Context, stream: &Stream, spec: BlockScaledFp4Spec) -> Result<Self> {
        Ok(Self {
            native: context.native.create_block_scaled_fp4_plan(&stream.native, spec.native())?,
            spec,
        })
    }

    /// Returns immutable plan geometry.
    #[must_use]
    pub const fn spec(&self) -> BlockScaledFp4Spec {
        self.spec
    }

    /// Native workspace retained for repeated launches.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn workspace_bytes(&self) -> usize {
        self.native.workspace_bytes()
    }

    /// Enqueues one block-scaled multiplication on the plan stream.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        stream: &Stream,
        a: &DeviceBuffer<u8>,
        a_scales: &DeviceBuffer<u8>,
        b: &DeviceBuffer<u8>,
        b_scales: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
        alpha: f32,
    ) -> Result<()> {
        validate("A", self.spec.m * self.spec.k / 2, a.len())?;
        validate("A scales", scale_elements(self.spec.m, self.spec.k)?, a_scales.len())?;
        validate("B", self.spec.n * self.spec.k / 2, b.len())?;
        validate("B scales", scale_elements(self.spec.n, self.spec.k)?, b_scales.len())?;
        validate("C", self.spec.m * self.spec.n, output.len())?;
        Ok(self.native.execute(
            &stream.native, &a.native, &a_scales.native, &b.native, &b_scales.native,
            &output.native, alpha,
        )?)
    }
}

fn scale_elements(rows: usize, columns: usize) -> Result<usize> {
    rows.div_ceil(128)
        .checked_mul(columns / 64)
        .and_then(|tiles| tiles.checked_mul(512))
        .ok_or(Error::InvalidMatmulShape)
}

const fn validate(operand: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::MatmulLengthMismatch { operand, expected, actual })
    }
}
