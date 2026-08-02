use crate::{
    CompileOptions, Compiler, DeviceBuffer, Error, LaunchConfig, Result, Stream, TypedKernel, bf16,
    cuda_export, cuda_kernel_file,
};

cuda_export!(
    MxFp8Kernel = "mircuda_mxfp8_bf16"(
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u32>,
        scales: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
        tokens: u32,
        input_features: u32,
        output_features: u32,
    )
);

cuda_export!(
    MxFp8TiledKernel = "mircuda_mxfp8_bf16_tiled"(
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u32>,
        scales: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
        tokens: u32,
        input_features: u32,
        output_features: u32,
    )
);

/// Fixed geometry for weight-only `BF16 × MXFP8 → BF16` multiplication.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MxFp8Spec {
    tokens: usize,
    input_features: usize,
    output_features: usize,
}

impl MxFp8Spec {
    /// Creates a matrix operation over complete 32-value MXFP8 blocks.
    pub fn new(tokens: usize, input_features: usize, output_features: usize) -> Result<Self> {
        if tokens == 0
            || input_features == 0
            || output_features == 0
            || !input_features.is_multiple_of(32)
        {
            return Err(Error::InvalidMatmulShape);
        }
        let _ = u32::try_from(tokens)?;
        let _ = u32::try_from(input_features)?;
        let _ = u32::try_from(output_features)?;
        let _ = tokens.checked_mul(input_features).ok_or(Error::InvalidMatmulShape)?;
        let _ = output_features.checked_mul(input_features).ok_or(Error::InvalidMatmulShape)?;
        let _ = tokens.checked_mul(output_features).ok_or(Error::InvalidMatmulShape)?;
        Ok(Self { tokens, input_features, output_features })
    }

    #[must_use]
    /// Returns the number of activation rows.
    pub const fn tokens(self) -> usize {
        self.tokens
    }

    #[must_use]
    /// Returns the reduction dimension.
    pub const fn input_features(self) -> usize {
        self.input_features
    }

    #[must_use]
    /// Returns the output width.
    pub const fn output_features(self) -> usize {
        self.output_features
    }

    /// Returns packed 32-bit weight words required by this geometry.
    pub fn weight_words(self) -> Result<usize> {
        self.output_features
            .checked_mul(self.input_features / 4)
            .ok_or(Error::InvalidMatmulShape)
    }

    /// Returns E8M0 scale bytes required by this geometry.
    pub fn scale_elements(self) -> Result<usize> {
        self.output_features
            .checked_mul(self.input_features / 32)
            .ok_or(Error::InvalidMatmulShape)
    }
}

/// Reusable portable CUDA MXFP8 projection compiled for the active device.
#[derive(Debug)]
pub struct MxFp8Matmul {
    kernel: TypedKernel<MxFp8Kernel>,
    tiled: TypedKernel<MxFp8TiledKernel>,
    spec: MxFp8Spec,
}

impl MxFp8Matmul {
    /// Compiles the portable, numerically explicit MXFP8 kernel.
    pub fn compile(compiler: &Compiler, spec: MxFp8Spec) -> Result<Self> {
        let source = cuda_kernel_file!("../../kernels/mxfp8.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            kernel: module.kernel()?,
            tiled: module.kernel()?,
            spec,
        })
    }

    #[must_use]
    /// Returns fixed operation geometry.
    pub const fn spec(&self) -> MxFp8Spec {
        self.spec
    }

    /// Enqueues one weight-only MXFP8 projection without host synchronization.
    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u32>,
        scales: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        minimum("input", self.spec.tokens * self.spec.input_features, input.len())?;
        validate("weight", self.spec.weight_words()?, weight.len())?;
        validate("scales", self.spec.scale_elements()?, scales.len())?;
        minimum("output", self.spec.tokens * self.spec.output_features, output.len())?;
        let rows = u32::try_from(self.spec.output_features)?;
        let tokens = u32::try_from(self.spec.tokens)?;
        let launch = LaunchConfig {
            grid: (
                rows,
                if tokens == 1 {
                    1
                } else {
                    tokens.div_ceil(4)
                },
                1,
            ),
            block: (256, 1, 1),
            shared_memory_bytes: 0,
        };
        let operands = (
            input,
            weight,
            scales,
            output,
            tokens,
            u32::try_from(self.spec.input_features)?,
            rows,
        );
        if tokens == 1 {
            self.kernel.launch(stream, launch, operands)
        } else {
            self.tiled.launch(stream, launch, operands)
        }
    }
}

const fn validate(operand: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::MatmulLengthMismatch { operand, expected, actual })
    }
}

const fn minimum(operand: &'static str, expected: usize, actual: usize) -> Result<()> {
    if actual >= expected {
        Ok(())
    } else {
        Err(Error::MatmulLengthMismatch { operand, expected, actual })
    }
}
