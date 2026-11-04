use std::marker::PhantomData;

use crate::{Context, DeviceBuffer, DeviceElement, Error, Result, Stream, bf16};

mod sealed {
    pub trait Sealed {}
}

/// Element type a classic cuBLAS dense plan multiplies. `f32` runs in full
/// single precision, never TF32.
pub trait CublasElement: DeviceElement + sealed::Sealed {
    #[doc(hidden)]
    const DATA_TYPE: mircuda_sys::CublasDataType;
}

impl sealed::Sealed for bf16 {}
impl sealed::Sealed for f32 {}

impl CublasElement for bf16 {
    const DATA_TYPE: mircuda_sys::CublasDataType = mircuda_sys::CublasDataType::Bf16;
}

impl CublasElement for f32 {
    const DATA_TYPE: mircuda_sys::CublasDataType = mircuda_sys::CublasDataType::F32;
}

/// Fixed geometry for a classic cuBLAS projection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CublasDenseSpec {
    m: usize,
    n: usize,
    k: usize,
}

/// The spec of a BF16 plan.
pub type CublasBf16Spec = CublasDenseSpec;

impl CublasDenseSpec {
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

/// Stream-bound classic cuBLAS plan with a persistent handle.
#[derive(Debug)]
pub struct CublasDensePlan<T: CublasElement> {
    native: mircuda_sys::CublasDensePlan,
    spec: CublasDenseSpec,
    marker: PhantomData<T>,
}

/// A plan multiplying BF16 operands with F32 accumulation.
pub type CublasBf16Plan = CublasDensePlan<bf16>;

/// A plan multiplying F32 operands in full single precision.
pub type CublasF32Plan = CublasDensePlan<f32>;

impl<T: CublasElement> CublasDensePlan<T> {
    /// Creates the fixed-shape plan without synchronizing its stream.
    pub fn new(context: &Context, stream: &Stream, spec: CublasDenseSpec) -> Result<Self> {
        let native_spec = mircuda_sys::CublasDenseSpec {
            m: spec.m,
            n: spec.n,
            k: spec.k,
            data_type: T::DATA_TYPE,
        };
        Ok(Self {
            native: context.native.create_cublas_dense_plan(&stream.native, native_spec)?,
            spec,
            marker: PhantomData,
        })
    }

    /// Returns the immutable plan geometry.
    #[must_use]
    pub const fn spec(&self) -> CublasDenseSpec {
        self.spec
    }

    /// Enqueues `c = alpha · a · bᵀ + beta · c` on the plan stream.
    pub fn execute(
        &mut self,
        stream: &Stream,
        a: &DeviceBuffer<T>,
        b: &DeviceBuffer<T>,
        c: &mut DeviceBuffer<T>,
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
