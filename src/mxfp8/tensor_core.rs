use std::sync::Arc;

use parking_lot::Mutex;

use super::MxFp8Spec;
use crate::{
    BlockScaledMxFp8Plan, BlockScaledMxFp8Spec, CompileOptions, Compiler, Context, DeviceBuffer,
    Error, LaunchConfig, MemoryPool, Result, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

cuda_export!(
    MxFp8QuantizeKernel = "mircuda_mxfp8_quantize_bf16"(
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        rows: u32,
        columns: u32,
    )
);

cuda_export!(
    MxFp8SwizzleKernel = "mircuda_mxfp8_swizzle_scales"(
        input: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<u8>,
        rows: u32,
        columns: u32,
    )
);

/// Reusable W8A8 MXFP8 Tensor Core projection with device-side activation quantization.
#[derive(Debug)]
pub struct MxFp8TensorCore {
    quantize: TypedKernel<MxFp8QuantizeKernel>,
    swizzle: TypedKernel<MxFp8SwizzleKernel>,
    plan: Mutex<BlockScaledMxFp8Plan>,
    scratch: Arc<MxFp8TensorCoreScratch>,
    spec: MxFp8Spec,
}

/// Shared sequential-execution buffers for one MXFP8 activation geometry.
#[derive(Debug)]
pub struct MxFp8TensorCoreScratch {
    quantized_input: Mutex<DeviceBuffer<u8>>,
    input_scales: Mutex<DeviceBuffer<u8>>,
    tokens: usize,
    input_features: usize,
}

impl MxFp8TensorCore {
    /// Creates persistent quantization buffers and a vendor plan for one shape.
    pub fn new(
        compiler: &Compiler,
        context: &Context,
        pool: &MemoryPool,
        stream: &Stream,
        spec: MxFp8Spec,
    ) -> Result<Self> {
        let scratch = Arc::new(MxFp8TensorCoreScratch::new(context, pool, stream, spec)?);
        Self::new_with_scratch(compiler, context, stream, spec, scratch)
    }

    /// Creates a plan reusing sequential-execution buffers with matching M and K.
    pub fn new_with_scratch(
        compiler: &Compiler,
        context: &Context,
        stream: &Stream,
        spec: MxFp8Spec,
        scratch: Arc<MxFp8TensorCoreScratch>,
    ) -> Result<Self> {
        scratch.validate(spec)?;
        let native = BlockScaledMxFp8Spec::new(
            spec.tokens(),
            spec.output_features(),
            spec.input_features(),
        )?;
        let module = compiler
            .compile(cuda_kernel_file!("../../kernels/mxfp8.cu"), &CompileOptions::default())?;
        Ok(Self {
            quantize: module.kernel()?,
            swizzle: module.kernel()?,
            plan: Mutex::new(BlockScaledMxFp8Plan::new(context, stream, native)?),
            scratch,
            spec,
        })
    }

    /// Converts row-major checkpoint scales once into the native padded layout.
    pub fn swizzle_weight_scales(
        &self,
        pool: &MemoryPool,
        stream: &Stream,
        scales: &DeviceBuffer<u8>,
    ) -> Result<DeviceBuffer<u8>> {
        let rows = self.spec.output_features();
        let columns = self.spec.input_features() / 32;
        exact("MXFP8 weight scales", rows * columns, scales.len())?;
        let native = self.plan.lock().spec();
        let mut output = pool.allocate_zeroed::<u8>(stream, native.scale_bytes(rows)?)?;
        self.swizzle.launch(
            stream,
            LaunchConfig::for_elements(rows * columns, 256)?,
            (scales, &mut output, u32::try_from(rows)?, u32::try_from(columns)?),
        )?;
        self.prepare_weight_scales(stream, &output)?;
        Ok(output)
    }

    /// Validates an already-swizzled checkpoint scale buffer.
    pub fn prepare_weight_scales(
        &self,
        _stream: &Stream,
        weight_scales: &DeviceBuffer<u8>,
    ) -> Result<()> {
        let rows = self.spec.output_features();
        let expected = self.plan.lock().spec().scale_bytes(rows)?;
        exact("swizzled MXFP8 weight scales", expected, weight_scales.len())
    }

    /// Persistent workspace selected for this fixed geometry.
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        self.plan.lock().workspace_bytes()
    }

    /// Quantizes BF16 activations and executes one MXFP8 Tensor Core projection.
    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u32>,
        weight_scales: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        exact("MXFP8 input", self.spec.tokens() * self.spec.input_features(), input.len())?;
        exact("MXFP8 weight", self.spec.weight_words()?, weight.len())?;
        exact("MXFP8 output", self.spec.tokens() * self.spec.output_features(), output.len())?;
        let mut quantized = self.scratch.quantized_input.lock();
        let mut input_scales = self.scratch.input_scales.lock();
        self.quantize.launch(
            stream,
            LaunchConfig {
                grid: (
                    u32::try_from(self.spec.input_features() / 32)?,
                    u32::try_from(self.spec.tokens())?,
                    1,
                ),
                block: (32, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                &mut quantized,
                &mut input_scales,
                u32::try_from(self.spec.tokens())?,
                u32::try_from(self.spec.input_features())?,
            ),
        )?;
        self.plan
            .lock()
            .execute(stream, &quantized, &input_scales, weight, weight_scales, output)
    }
}

impl MxFp8TensorCoreScratch {
    /// Allocates one reusable activation and scale pair for M-by-K execution.
    pub fn new(
        _context: &Context,
        pool: &MemoryPool,
        stream: &Stream,
        spec: MxFp8Spec,
    ) -> Result<Self> {
        let native = BlockScaledMxFp8Spec::new(
            spec.tokens(),
            spec.output_features(),
            spec.input_features(),
        )?;
        Ok(Self {
            quantized_input: Mutex::new(
                pool.allocate::<u8>(stream, spec.tokens() * spec.input_features())?,
            ),
            input_scales: Mutex::new(
                pool.allocate_zeroed::<u8>(stream, native.scale_bytes(spec.tokens())?)?,
            ),
            tokens: spec.tokens(),
            input_features: spec.input_features(),
        })
    }

    const fn validate(&self, spec: MxFp8Spec) -> Result<()> {
        if self.tokens == spec.tokens() && self.input_features == spec.input_features() {
            Ok(())
        } else {
            Err(Error::InvalidMatmulShape)
        }
    }
}

const fn exact(operand: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::MatmulLengthMismatch { operand, expected, actual })
    }
}
