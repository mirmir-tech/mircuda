use std::marker::PhantomData;

use super::DenseMatmulElement;
use crate::{Context, DeviceBuffer, Error, Result, Stream};

/// Fixed geometry for `output[N] = alpha * weight[N,K] * input[K] + beta * output[N]`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DenseVectorSpec {
    n: usize,
    k: usize,
}

impl DenseVectorSpec {
    /// Creates a validated row-major matrix-vector shape.
    pub fn new(n: usize, k: usize) -> Result<Self> {
        if n == 0 || k == 0 {
            return Err(Error::InvalidMatmulShape);
        }
        for value in [n, k] {
            let _ = i32::try_from(value)?;
        }
        let _ = n.checked_mul(k).ok_or(Error::InvalidMatmulShape)?;
        Ok(Self { n, k })
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

/// Stream-bound native matrix-vector plan without persistent workspace.
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
        };
        Ok(Self {
            native: context.native.create_dense_vector_plan(&stream.native, native_spec)?,
            spec,
            marker: PhantomData,
        })
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
