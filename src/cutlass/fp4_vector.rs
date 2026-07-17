use crate::{Context, DeviceBuffer, Error, Result, Stream, bf16};

/// Fixed geometry for block-scaled `FP4 x FP4 -> BF16` vector projection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BlockScaledFp4VectorSpec {
    n: usize,
    k: usize,
}

impl BlockScaledFp4VectorSpec {
    /// Creates `output[N] = alpha * weight[N,K] * input[K]`.
    pub fn new(n: usize, k: usize) -> Result<Self> {
        if n == 0 || k == 0 || !k.is_multiple_of(64) {
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
    /// Returns the reduction dimension in packed FP4 elements.
    pub const fn k(self) -> usize {
        self.k
    }

    const fn native(self) -> mircuda_sys::BlockScaledFp4VectorSpec {
        mircuda_sys::BlockScaledFp4VectorSpec { n: self.n, k: self.k }
    }
}

/// Stream-bound block-scaled FP4 vector plan with no transient allocation.
#[derive(Debug)]
pub struct BlockScaledFp4VectorPlan {
    native: mircuda_sys::BlockScaledFp4VectorPlan,
    spec: BlockScaledFp4VectorSpec,
}

impl BlockScaledFp4VectorPlan {
    /// Creates a reusable plan and allocates all persistent native state.
    pub fn new(context: &Context, stream: &Stream, spec: BlockScaledFp4VectorSpec) -> Result<Self> {
        Ok(Self {
            native: context
                .native
                .create_block_scaled_fp4_vector_plan(&stream.native, spec.native())?,
            spec,
        })
    }

    #[must_use]
    /// Returns the fixed geometry of this plan.
    pub const fn spec(&self) -> BlockScaledFp4VectorSpec {
        self.spec
    }

    #[allow(clippy::too_many_arguments)]
    /// Executes one FP4 vector projection without allocating or synchronizing.
    ///
    /// Packed input and weight values use two FP4 elements per byte. Scale
    /// buffers use the tiled UE4M3 layout required by the native SM120 kernel.
    pub fn execute(
        &mut self,
        stream: &Stream,
        input: &DeviceBuffer<u8>,
        input_scales: &DeviceBuffer<u8>,
        weight: &DeviceBuffer<u8>,
        weight_scales: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
        alpha: f32,
    ) -> Result<()> {
        validate("input", self.spec.k / 2, input.len())?;
        validate("input scales", scale_elements(1, self.spec.k)?, input_scales.len())?;
        validate("weight", self.spec.n * self.spec.k / 2, weight.len())?;
        validate("weight scales", scale_elements(self.spec.n, self.spec.k)?, weight_scales.len())?;
        validate("output", self.spec.n, output.len())?;
        Ok(self.native.execute(
            &stream.native,
            &input.native,
            &input_scales.native,
            &weight.native,
            &weight_scales.native,
            &output.native,
            alpha,
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
