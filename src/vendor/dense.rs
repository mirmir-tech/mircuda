use crate::{Context, DeviceBuffer, Error, Result, Stream, bf16};

/// Fixed geometry for a cuBLASLt BF16 projection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CublasLtBf16Spec {
    m: usize,
    n: usize,
    k: usize,
}

impl CublasLtBf16Spec {
    /// Creates a validated row-major `C[M,N] = A[M,K] * B[N,K]^T` shape.
    pub fn new(m: usize, n: usize, k: usize) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
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
}

/// Stream-bound cuBLASLt BF16 plan with persistent descriptors and workspace.
#[derive(Debug)]
pub struct CublasLtBf16Plan {
    native: mircuda_sys::CublasLtBf16Plan,
    spec: CublasLtBf16Spec,
}

impl CublasLtBf16Plan {
    /// Selects one vendor algorithm for the fixed shape without synchronizing.
    pub fn new(context: &Context, stream: &Stream, spec: CublasLtBf16Spec) -> Result<Self> {
        let native_spec = mircuda_sys::CublasLtBf16Spec { m: spec.m, n: spec.n, k: spec.k };
        Ok(Self {
            native: context.native.create_cublaslt_bf16_plan(&stream.native, native_spec)?,
            spec,
        })
    }

    /// Returns the immutable plan geometry.
    #[must_use]
    pub const fn spec(&self) -> CublasLtBf16Spec {
        self.spec
    }

    /// Persistent vendor workspace retained by this plan.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn workspace_bytes(&self) -> usize {
        self.native.workspace_bytes()
    }

    /// Enqueues the fixed-shape multiplication on the plan stream.
    pub fn execute(
        &mut self,
        stream: &Stream,
        a: &DeviceBuffer<bf16>,
        b: &DeviceBuffer<bf16>,
        c: &mut DeviceBuffer<bf16>,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        validate_len("A", self.spec.m * self.spec.k, a.len())?;
        validate_len("B", self.spec.n * self.spec.k, b.len())?;
        validate_len("C", self.spec.m * self.spec.n, c.len())?;
        Ok(self
            .native
            .execute(&stream.native, &a.native, &b.native, &c.native, alpha, beta)?)
    }
}

const fn validate_len(operand: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::MatmulLengthMismatch { operand, expected, actual })
    }
}
