use std::marker::PhantomData;

use crate::{Context, DeviceBuffer, DeviceElement, Error, Result, Stream, bf16, f16};

mod sealed {
    pub trait Sealed {}
}

/// Storage type accepted as the output of a native dense CUTLASS plan.
pub trait DenseMatmulOutput: DeviceElement + sealed::Sealed {
    #[doc(hidden)]
    const NATIVE_TYPE: mircuda_sys::DenseMatmulDataType;
}

/// Storage type accepted as an input of a native dense CUTLASS plan.
pub trait DenseMatmulElement: DenseMatmulOutput {
    #[doc(hidden)]
    const INPUT_NATIVE_TYPE: mircuda_sys::DenseMatmulDataType;
}

impl sealed::Sealed for f16 {}
impl sealed::Sealed for bf16 {}
impl sealed::Sealed for f32 {}

impl DenseMatmulElement for f16 {
    const INPUT_NATIVE_TYPE: mircuda_sys::DenseMatmulDataType =
        mircuda_sys::DenseMatmulDataType::F16;
}

impl DenseMatmulElement for bf16 {
    const INPUT_NATIVE_TYPE: mircuda_sys::DenseMatmulDataType =
        mircuda_sys::DenseMatmulDataType::Bf16;
}

impl DenseMatmulOutput for f16 {
    const NATIVE_TYPE: mircuda_sys::DenseMatmulDataType = mircuda_sys::DenseMatmulDataType::F16;
}

impl DenseMatmulOutput for bf16 {
    const NATIVE_TYPE: mircuda_sys::DenseMatmulDataType = mircuda_sys::DenseMatmulDataType::Bf16;
}

impl DenseMatmulOutput for f32 {
    const NATIVE_TYPE: mircuda_sys::DenseMatmulDataType = mircuda_sys::DenseMatmulDataType::F32;
}

/// Fixed geometry for `C[M,N] = alpha * A[M,K] * B[N,K]^T + beta * C[M,N]`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DenseMatmulSpec {
    m: usize,
    n: usize,
    k: usize,
}

impl DenseMatmulSpec {
    /// Creates a validated row-major dense projection shape.
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

/// Stream-bound dense CUTLASS plan with persistent native workspace.
#[derive(Debug)]
pub struct DenseMatmulPlan<T: DenseMatmulElement, O: DenseMatmulOutput = T> {
    native: mircuda_sys::DenseMatmulPlan,
    spec: DenseMatmulSpec,
    marker: PhantomData<(T, O)>,
}

impl<T: DenseMatmulElement, O: DenseMatmulOutput> DenseMatmulPlan<T, O> {
    /// Creates a fixed-shape plan without synchronizing the stream.
    pub fn new(context: &Context, stream: &Stream, spec: DenseMatmulSpec) -> Result<Self> {
        let native_spec = mircuda_sys::DenseMatmulSpec {
            m: spec.m,
            n: spec.n,
            k: spec.k,
            input_type: T::INPUT_NATIVE_TYPE,
            output_type: O::NATIVE_TYPE,
        };
        Ok(Self {
            native: context.native.create_dense_matmul_plan(&stream.native, native_spec)?,
            spec,
            marker: PhantomData,
        })
    }

    /// Returns the immutable plan geometry.
    #[must_use]
    pub const fn spec(&self) -> DenseMatmulSpec {
        self.spec
    }

    /// Native workspace retained for repeated launches.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn workspace_bytes(&self) -> usize {
        self.native.workspace_bytes()
    }

    /// Enqueues the fixed-shape multiplication on the plan stream.
    pub fn execute(
        &mut self,
        stream: &Stream,
        a: &DeviceBuffer<T>,
        b: &DeviceBuffer<T>,
        c: &mut DeviceBuffer<O>,
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
