use crate::{Context, DeviceBuffer, Error, Result, Stream, bf16};

/// Fixed row-major `E4M3[M,K] × E4M3[N,K]^T → BF16[M,N]` MXFP8 geometry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BlockScaledMxFp8Spec {
    m: usize,
    n: usize,
    k: usize,
}

impl BlockScaledMxFp8Spec {
    /// Creates a shape using one E8M0 scale per 32 consecutive K values.
    pub fn new(m: usize, n: usize, k: usize) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 || !k.is_multiple_of(128) {
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

    /// Activation row count.
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

    /// Padded bytes required by the native 128-by-4 scale-factor layout.
    pub fn scale_bytes(self, rows: usize) -> Result<usize> {
        rows.div_ceil(128)
            .checked_mul(128)
            .and_then(|outer| outer.checked_mul((self.k / 32).div_ceil(4) * 4))
            .ok_or(Error::InvalidMatmulShape)
    }

    const fn native(self) -> mircuda_sys::BlockScaledMxFp8Spec {
        mircuda_sys::BlockScaledMxFp8Spec { m: self.m, n: self.n, k: self.k }
    }
}

/// Stream-bound CUTLASS MXFP8 plan with persistent workspace.
#[derive(Debug)]
pub struct BlockScaledMxFp8Plan {
    native: mircuda_sys::BlockScaledMxFp8Plan,
    spec: BlockScaledMxFp8Spec,
}

impl BlockScaledMxFp8Plan {
    /// Creates the fixed-shape plan without synchronizing the stream.
    pub fn new(context: &Context, stream: &Stream, spec: BlockScaledMxFp8Spec) -> Result<Self> {
        Ok(Self {
            native: context.native.create_block_scaled_mxfp8_plan(&stream.native, spec.native())?,
            spec,
        })
    }

    /// Returns immutable plan geometry.
    #[must_use]
    pub const fn spec(&self) -> BlockScaledMxFp8Spec {
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
        b: &DeviceBuffer<u32>,
        b_scales: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        validate("A", self.spec.m * self.spec.k, a.len())?;
        validate("A scales", scale_elements(self.spec.m, self.spec.k)?, a_scales.len())?;
        validate("B", self.spec.n * self.spec.k / 4, b.len())?;
        validate("B scales", scale_elements(self.spec.n, self.spec.k)?, b_scales.len())?;
        validate("C", self.spec.m * self.spec.n, output.len())?;
        Ok(self.native.execute(
            &stream.native, &a.native, &a_scales.native, &b.native, &b_scales.native,
            &output.native,
        )?)
    }
}

fn scale_elements(rows: usize, columns: usize) -> Result<usize> {
    rows.div_ceil(128)
        .checked_mul(128)
        .and_then(|outer| outer.checked_mul((columns / 32).div_ceil(4) * 4))
        .ok_or(Error::InvalidMatmulShape)
}

const fn validate(operand: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::MatmulLengthMismatch { operand, expected, actual })
    }
}
