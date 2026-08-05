use crate::{Context, DeviceBuffer, DeviceElement, Error, Result, Stream, bf16};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Element type of the per-output-channel scale vector.
pub enum ScaledFp8Scale {
    /// IEEE single-precision scales.
    F32,
    /// Brain floating-point scales retained directly from a checkpoint.
    Bf16,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Geometry of the model-owned weight scale.
pub enum ScaledFp8WeightScale {
    /// One scalar applies to the complete weight matrix.
    Tensor,
    /// One scalar applies to each output channel.
    OutputChannel,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Decode tile geometry for SM120 scaled FP8 matrix multiplication.
pub enum ScaledFp8Tile {
    /// 16 by 64 output tile with a 128-wide reduction tile.
    M16N64K128,
    /// 16 by 128 output tile with a 64-wide reduction tile.
    M16N128K64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Fixed geometry for E4M3 by E4M3 scaled matrix multiplication.
pub struct ScaledFp8Spec {
    /// Input row count.
    pub tokens: usize,
    /// Output column count.
    pub output_features: usize,
    /// Reduction dimension.
    pub input_features: usize,
    /// Weight scale storage type.
    pub scale: ScaledFp8Scale,
    /// Weight scale geometry.
    pub weight_scale: ScaledFp8WeightScale,
    /// Whether a BF16 output bias is fused.
    pub has_bias: bool,
    /// Decode tile selected by the caller or its autotuner.
    pub tile: ScaledFp8Tile,
}

impl ScaledFp8Spec {
    /// Validates Tensor Core alignment and buffer geometry.
    pub fn new(
        tokens: usize,
        input_features: usize,
        output_features: usize,
        scale: ScaledFp8Scale,
        weight_scale: ScaledFp8WeightScale,
        has_bias: bool,
    ) -> Result<Self> {
        if tokens == 0
            || input_features == 0
            || output_features == 0
            || !input_features.is_multiple_of(16)
            || !output_features.is_multiple_of(16)
        {
            return Err(Error::InvalidMatmulShape);
        }
        let _ = tokens.checked_mul(input_features).ok_or(Error::InvalidMatmulShape)?;
        let _ = output_features.checked_mul(input_features).ok_or(Error::InvalidMatmulShape)?;
        Ok(Self {
            tokens,
            output_features,
            input_features,
            scale,
            weight_scale,
            has_bias,
            tile: ScaledFp8Tile::M16N64K128,
        })
    }

    /// Overrides the default decode tile for this plan.
    #[must_use]
    pub const fn with_tile(mut self, tile: ScaledFp8Tile) -> Self {
        self.tile = tile;
        self
    }

    const fn native(self) -> mircuda_sys::ScaledFp8Spec {
        mircuda_sys::ScaledFp8Spec {
            m: self.tokens,
            n: self.output_features,
            k: self.input_features,
            scale_type: match self.scale {
                ScaledFp8Scale::F32 => mircuda_sys::ScaledFp8ScaleType::F32,
                ScaledFp8Scale::Bf16 => mircuda_sys::ScaledFp8ScaleType::Bf16,
            },
            weight_scale_type: match self.weight_scale {
                ScaledFp8WeightScale::Tensor => mircuda_sys::ScaledFp8WeightScaleType::Tensor,
                ScaledFp8WeightScale::OutputChannel => {
                    mircuda_sys::ScaledFp8WeightScaleType::OutputChannel
                },
            },
            has_bias: self.has_bias,
            tile: match self.tile {
                ScaledFp8Tile::M16N64K128 => mircuda_sys::ScaledFp8Tile::M16N64K128,
                ScaledFp8Tile::M16N128K64 => mircuda_sys::ScaledFp8Tile::M16N128K64,
            },
        }
    }
}

#[derive(Debug)]
/// Stream-bound native SM120 FP8 scaled-matmul plan.
pub struct ScaledFp8Plan {
    native: mircuda_sys::ScaledFp8Plan,
    spec: ScaledFp8Spec,
}

impl ScaledFp8Plan {
    /// Creates a reusable plan and retains its native workspace.
    pub fn new(context: &Context, stream: &Stream, spec: ScaledFp8Spec) -> Result<Self> {
        Ok(Self {
            native: context.native.create_scaled_fp8_plan(&stream.native, spec.native())?,
            spec,
        })
    }

    /// Returns retained workspace bytes.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn workspace_bytes(&self) -> usize {
        self.native.workspace_bytes()
    }

    /// Executes a projection whose channel scales are FP32.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_f32_scales(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u8>,
        weight: &DeviceBuffer<u8>,
        input_scales: &DeviceBuffer<f32>,
        weight_scales: &DeviceBuffer<f32>,
        bias: Option<&DeviceBuffer<bf16>>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if self.spec.scale != ScaledFp8Scale::F32 {
            return Err(Error::InvalidMatmulShape);
        }
        self.execute(stream, input, weight, input_scales, weight_scales, bias, output)
    }

    /// Executes a projection whose channel scales are BF16.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_bf16_scales(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u8>,
        weight: &DeviceBuffer<u8>,
        input_scales: &DeviceBuffer<f32>,
        weight_scales: &DeviceBuffer<bf16>,
        bias: Option<&DeviceBuffer<bf16>>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if self.spec.scale != ScaledFp8Scale::Bf16 {
            return Err(Error::InvalidMatmulShape);
        }
        self.execute(stream, input, weight, input_scales, weight_scales, bias, output)
    }

    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_ref_mut)]
    fn execute<T: DeviceElement>(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u8>,
        weight: &DeviceBuffer<u8>,
        input_scales: &DeviceBuffer<f32>,
        weight_scales: &DeviceBuffer<T>,
        bias: Option<&DeviceBuffer<bf16>>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        Ok(self.native.execute(
            &stream.native,
            &input.native,
            &weight.native,
            &input_scales.native,
            &weight_scales.native,
            bias.map(|value| value.native.as_ref()),
            &output.native,
        )?)
    }
}
