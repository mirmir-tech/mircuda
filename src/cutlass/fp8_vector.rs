use crate::{Context, DeviceBuffer, Error, Result, Stream, bf16};

/// Fixed geometry for blockwise E4M3 vector projection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BlockwiseFp8VectorSpec {
    n: usize,
    k: usize,
}

impl BlockwiseFp8VectorSpec {
    /// Creates `output[N] = input[1,K] * weight[N,K]^T`.
    pub fn new(n: usize, k: usize) -> Result<Self> {
        if n == 0 || k == 0 || !n.is_multiple_of(128) || !k.is_multiple_of(128) {
            return Err(Error::InvalidMatmulShape);
        }
        for value in [n, k] {
            let _ = i32::try_from(value)?;
        }
        let _ = n.checked_mul(k).ok_or(Error::InvalidMatmulShape)?;
        Ok(Self { n, k })
    }

    #[must_use]
    /// Returns the number of output rows.
    pub const fn n(self) -> usize {
        self.n
    }

    #[must_use]
    /// Returns the reduction dimension in E4M3 elements.
    pub const fn k(self) -> usize {
        self.k
    }

    #[must_use]
    /// Returns the number of FP32 scales required by the input vector.
    pub const fn input_scale_elements(self) -> usize {
        self.k / 128
    }

    /// Returns the number of FP32 scales required by the weight matrix.
    pub fn weight_scale_elements(self) -> Result<usize> {
        (self.n / 128).checked_mul(self.k / 128).ok_or(Error::InvalidMatmulShape)
    }

    const fn native(self) -> mircuda_sys::BlockwiseFp8VectorSpec {
        mircuda_sys::BlockwiseFp8VectorSpec { n: self.n, k: self.k }
    }
}

/// Stream-bound CUTLASS blockwise FP8 plan with retained workspace.
#[derive(Debug)]
pub struct BlockwiseFp8VectorPlan {
    native: mircuda_sys::BlockwiseFp8VectorPlan,
    spec: BlockwiseFp8VectorSpec,
}

impl BlockwiseFp8VectorPlan {
    /// Creates a reusable plan and allocates its persistent workspace.
    pub fn new(context: &Context, stream: &Stream, spec: BlockwiseFp8VectorSpec) -> Result<Self> {
        Ok(Self {
            native: context
                .native
                .create_blockwise_fp8_vector_plan(&stream.native, spec.native())?,
            spec,
        })
    }

    #[must_use]
    /// Returns the fixed geometry of this plan.
    pub const fn spec(&self) -> BlockwiseFp8VectorSpec {
        self.spec
    }

    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    /// Returns the retained native workspace size in bytes.
    pub fn workspace_bytes(&self) -> usize {
        self.native.workspace_bytes()
    }

    /// Executes one blockwise E4M3 vector projection without allocating or synchronizing.
    pub fn execute(
        &mut self,
        stream: &Stream,
        input: &DeviceBuffer<u8>,
        input_scales: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        weight_scales: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        validate("input", self.spec.k, input.len())?;
        validate("input scales", self.spec.input_scale_elements(), input_scales.len())?;
        validate("weight", self.spec.n * self.spec.k, weight.len())?;
        validate("weight scales", self.spec.weight_scale_elements()?, weight_scales.len())?;
        validate("output", self.spec.n, output.len())?;
        Ok(self.native.execute(
            &stream.native,
            &input.native,
            &input_scales.native,
            &weight.native,
            &weight_scales.native,
            &output.native,
        )?)
    }
}

const fn validate(operand: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::MatmulLengthMismatch { operand, expected, actual })
    }
}
