use crate::{Context, DeviceBuffer, Error, Result, Stream, bf16};

/// Fixed geometry for a scaled cuBLASLt E4M3 projection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CublasLtFp8Spec {
    m: usize,
    n: usize,
    k: usize,
}

impl CublasLtFp8Spec {
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
}

/// Stream-bound scaled E4M3 cuBLASLt plan with a persistent algorithm.
#[derive(Debug)]
pub struct CublasLtFp8Plan {
    native: mircuda_sys::CublasLtFp8Plan,
    spec: CublasLtFp8Spec,
}

impl CublasLtFp8Plan {
    /// Selects one vendor algorithm for the fixed shape without synchronizing.
    pub fn new(context: &Context, stream: &Stream, spec: CublasLtFp8Spec) -> Result<Self> {
        let native_spec = mircuda_sys::CublasLtFp8Spec { m: spec.m, n: spec.n, k: spec.k };
        Ok(Self {
            native: context.native.create_cublaslt_fp8_plan(&stream.native, native_spec)?,
            spec,
        })
    }

    /// Enqueues the scaled multiplication on the plan stream.
    pub fn execute(
        &mut self,
        stream: &Stream,
        a: &DeviceBuffer<u8>,
        b: &DeviceBuffer<u8>,
        a_scale: &DeviceBuffer<f32>,
        b_scale: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        validate_len("A", self.spec.m * self.spec.k, a.len())?;
        validate_len("B", self.spec.n * self.spec.k, b.len())?;
        validate_len("A scale", 1, a_scale.len())?;
        validate_len("B scale", 1, b_scale.len())?;
        validate_len("C", self.spec.m * self.spec.n, c.len())?;
        Ok(self.native.execute(
            &stream.native, &a.native, &b.native, &a_scale.native, &b_scale.native, &c.native,
        )?)
    }
}

const fn validate_len(operand: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::MatmulLengthMismatch { operand, expected, actual })
    }
}
