use crate::{
    CompileOptions, Compiler, DeviceBuffer, Error, LaunchConfig, Result, Stream, TypedKernel, bf16,
    cuda_export, cuda_kernel_file,
};

cuda_export!(
    MxFp8GatheredKernel = "mircuda_mxfp8_bf16_gathered"(
        input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u32>, scales: &DeviceBuffer<u8>,
        bias: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>, output: &mut DeviceBuffer<bf16>,
        assignments: u32, matrices: u32, rows: u32, columns: u32,
        selections_per_input: u32, has_bias: u32,
    )
);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Fixed geometry for selected projections from an MXFP8 matrix bank.
pub struct MxFp8GatheredSpec {
    input_rows: usize,
    selections_per_input: usize,
    assignments: usize,
    matrices: usize,
    input_features: usize,
    output_features: usize,
}

impl MxFp8GatheredSpec {
    /// Creates a routed geometry that reuses each input row across its selections.
    pub fn new_routed(
        input_rows: usize,
        selections_per_input: usize,
        matrices: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        if input_rows == 0 || selections_per_input == 0 || matrices == 0 {
            return Err(Error::InvalidMatmulShape);
        }
        let assignments =
            input_rows.checked_mul(selections_per_input).ok_or(Error::InvalidMatmulShape)?;
        let matrix = super::MxFp8Spec::new(1, input_features, output_features)?;
        let spec = Self {
            input_rows,
            selections_per_input,
            assignments,
            matrices,
            input_features,
            output_features,
        };
        let _ = matrices.checked_mul(matrix.weight_words()?).ok_or(Error::InvalidMatmulShape)?;
        let _ = matrices
            .checked_mul(matrix.scale_elements()?)
            .ok_or(Error::InvalidMatmulShape)?;
        Ok(spec)
    }

    fn input_elements(self) -> Result<usize> {
        self.input_rows
            .checked_mul(self.input_features)
            .ok_or(Error::InvalidMatmulShape)
    }

    fn weight_words(self) -> Result<usize> {
        self.matrices
            .checked_mul(self.output_features)
            .and_then(|value| value.checked_mul(self.input_features / 4))
            .ok_or(Error::InvalidMatmulShape)
    }

    fn scale_elements(self) -> Result<usize> {
        self.matrices
            .checked_mul(self.output_features)
            .and_then(|value| value.checked_mul(self.input_features / 32))
            .ok_or(Error::InvalidMatmulShape)
    }

    fn output_elements(self) -> Result<usize> {
        self.assignments
            .checked_mul(self.output_features)
            .ok_or(Error::InvalidMatmulShape)
    }

    fn bias_elements(self) -> Result<usize> {
        self.matrices.checked_mul(self.output_features).ok_or(Error::InvalidMatmulShape)
    }
}

/// Device-resident operands for one gathered MXFP8 projection.
pub struct MxFp8GatheredOperands<'a> {
    /// BF16 input rows.
    pub input: &'a DeviceBuffer<bf16>,
    /// Packed U32 E4M3 matrix banks.
    pub weight: &'a DeviceBuffer<u32>,
    /// U8 E8M0 scale banks.
    pub scales: &'a DeviceBuffer<u8>,
    /// Optional BF16 output bias bank.
    pub bias: Option<&'a DeviceBuffer<bf16>>,
    /// Selected matrix index for every assignment.
    pub selected: &'a DeviceBuffer<u32>,
    /// BF16 assignment-major output rows.
    pub output: &'a mut DeviceBuffer<bf16>,
}

#[derive(Debug)]
/// Portable gathered BF16-by-MXFP8 projection.
pub struct MxFp8Gathered {
    kernel: TypedKernel<MxFp8GatheredKernel>,
    spec: MxFp8GatheredSpec,
    warps_per_block: u32,
}

impl MxFp8Gathered {
    /// Compiles a gathered kernel for a fixed geometry.
    pub fn compile(compiler: &Compiler, spec: MxFp8GatheredSpec) -> Result<Self> {
        Self::compile_warps(compiler, spec, 8)
    }

    /// Compiles a gathered kernel with one of the supported reduction widths.
    pub fn compile_warps(
        compiler: &Compiler,
        spec: MxFp8GatheredSpec,
        warps_per_block: u32,
    ) -> Result<Self> {
        if !matches!(warps_per_block, 4 | 8) {
            return Err(Error::InvalidLaunch);
        }
        let module = compiler
            .compile(cuda_kernel_file!("../../kernels/mxfp8.cu"), &CompileOptions::default())?;
        Ok(Self {
            kernel: module.kernel()?,
            spec,
            warps_per_block,
        })
    }

    /// Enqueues gathered execution without allocation or host synchronization.
    pub fn execute(&self, stream: &Stream, operands: &mut MxFp8GatheredOperands<'_>) -> Result<()> {
        validate("input", self.spec.input_elements()?, operands.input.len())?;
        validate("weight", self.spec.weight_words()?, operands.weight.len())?;
        validate("scales", self.spec.scale_elements()?, operands.scales.len())?;
        validate("selected", self.spec.assignments, operands.selected.len())?;
        if let Some(bias) = operands.bias {
            validate("bias", self.spec.bias_elements()?, bias.len())?;
        }
        validate("output", self.spec.output_elements()?, operands.output.len())?;
        self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (
                    u32::try_from(self.spec.output_features)?,
                    u32::try_from(self.spec.assignments)?,
                    1,
                ),
                block: (self.warps_per_block * 32, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                operands.input,
                operands.weight,
                operands.scales,
                operands.bias.unwrap_or(operands.input),
                operands.selected,
                &mut *operands.output,
                u32::try_from(self.spec.assignments)?,
                u32::try_from(self.spec.matrices)?,
                u32::try_from(self.spec.output_features)?,
                u32::try_from(self.spec.input_features)?,
                u32::try_from(self.spec.selections_per_input)?,
                u32::from(operands.bias.is_some()),
            ),
        )
    }
}

const fn validate(operand: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::MatmulLengthMismatch { operand, expected, actual })
    }
}
