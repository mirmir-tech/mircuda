use crate::{
    CompileOptions, Compiler, DeviceBuffer, Error, LaunchConfig, Result, Stream, TypedKernel, bf16,
    cuda_export, cuda_kernel_file,
};

cuda_export!(
    MxFp8EmbeddingKernel = "mircuda_mxfp8_embedding_bf16"(
        weight: &DeviceBuffer<u32>,
        scales: &DeviceBuffer<u8>,
        selected: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
        selected_start: u32,
        tokens: u32,
        vocab: u32,
        hidden: u32,
        output_scale: f32,
    )
);

/// Fixed checkpoint geometry for selected-row MXFP8 dequantization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MxFp8EmbeddingSpec {
    vocab: usize,
    hidden: usize,
    output_scale: f32,
}

impl MxFp8EmbeddingSpec {
    /// Creates an embedding operation over complete 32-value MXFP8 blocks.
    pub fn new(vocab: usize, hidden: usize, output_scale: f32) -> Result<Self> {
        if vocab == 0 || hidden == 0 || !hidden.is_multiple_of(32) || !output_scale.is_finite() {
            return Err(Error::InvalidMatmulShape);
        }
        let _ = u32::try_from(vocab)?;
        let _ = u32::try_from(hidden)?;
        Ok(Self { vocab, hidden, output_scale })
    }

    /// Returns packed weight words required by the checkpoint matrix.
    pub fn weight_words(self) -> Result<usize> {
        self.vocab.checked_mul(self.hidden / 4).ok_or(Error::InvalidMatmulShape)
    }

    /// Returns scale bytes required by the checkpoint matrix.
    pub fn scale_elements(self) -> Result<usize> {
        self.vocab.checked_mul(self.hidden / 32).ok_or(Error::InvalidMatmulShape)
    }
}

/// Reusable selected-row MXFP8 embedding operation.
#[derive(Debug)]
pub struct MxFp8Embedding {
    kernel: TypedKernel<MxFp8EmbeddingKernel>,
    spec: MxFp8EmbeddingSpec,
}

/// Device buffers consumed by one selected-row embedding launch.
pub struct MxFp8EmbeddingOperands<'a> {
    /// Packed U32 E4M3 checkpoint matrix.
    pub weight: &'a DeviceBuffer<u32>,
    /// Per-32-value U8 E8M0 block scales.
    pub scales: &'a DeviceBuffer<u8>,
    /// Device token identifiers.
    pub selected: &'a DeviceBuffer<u32>,
    /// BF16 destination rows.
    pub output: &'a mut DeviceBuffer<bf16>,
}

impl MxFp8Embedding {
    /// Compiles selected-row MXFP8 dequantization for the active device.
    pub fn compile(compiler: &Compiler, spec: MxFp8EmbeddingSpec) -> Result<Self> {
        let module = compiler
            .compile(cuda_kernel_file!("../../kernels/mxfp8.cu"), &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    /// Enqueues selected rows without materializing the full BF16 matrix.
    pub fn execute(
        &self,
        stream: &Stream,
        operands: MxFp8EmbeddingOperands<'_>,
        selected_start: usize,
        tokens: usize,
    ) -> Result<()> {
        let MxFp8EmbeddingOperands { weight, scales, selected, output } = operands;
        exact("weight", self.spec.weight_words()?, weight.len())?;
        exact("scales", self.spec.scale_elements()?, scales.len())?;
        minimum(
            "selected",
            selected_start.checked_add(tokens).ok_or(Error::InvalidMatmulShape)?,
            selected.len(),
        )?;
        minimum(
            "output",
            tokens.checked_mul(self.spec.hidden).ok_or(Error::InvalidMatmulShape)?,
            output.len(),
        )?;
        let tokens = u32::try_from(tokens)?;
        self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (tokens, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                weight,
                scales,
                selected,
                output,
                u32::try_from(selected_start)?,
                tokens,
                u32::try_from(self.spec.vocab)?,
                u32::try_from(self.spec.hidden)?,
                self.spec.output_scale,
            ),
        )
    }
}

const fn exact(operand: &'static str, expected: usize, actual: usize) -> Result<()> {
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
