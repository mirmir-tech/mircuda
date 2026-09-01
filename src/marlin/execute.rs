use super::MarlinNvFp4MoeSpec;
use crate::{Context, DeviceBuffer, Result, Stream, bf16};

/// Device buffers consumed by one standard dense Marlin NVFP4 projection.
pub struct MarlinNvFp4DenseOperands<'a> {
    /// BF16 input rows.
    pub input: &'a DeviceBuffer<bf16>,
    /// Repacked NVFP4 weight matrix.
    pub weight: &'a DeviceBuffer<u8>,
    /// Marlin-formatted E4M3 group scales.
    pub scales: &'a DeviceBuffer<u8>,
    /// Compensated global weight scale.
    pub global_scale: &'a DeviceBuffer<f32>,
    /// FP32 cross-block reduction scratch.
    pub temporary: &'a mut DeviceBuffer<f32>,
    /// Marlin reduction workspace.
    pub locks: &'a mut DeviceBuffer<i32>,
    /// BF16 output rows.
    pub output: &'a mut DeviceBuffer<bf16>,
    /// Uses atomic partial reductions when the shape benefits from them.
    pub atomic_reduce: bool,
}

/// Device buffers consumed by one Marlin NVFP4 `MoE` projection.
pub struct MarlinNvFp4MoeOperands<'a> {
    /// BF16 input rows.
    pub input: &'a DeviceBuffer<bf16>,
    /// Repacked NVFP4 expert weights.
    pub weight: &'a DeviceBuffer<u8>,
    /// Marlin-formatted E4M3 group scales.
    pub scales: &'a DeviceBuffer<u8>,
    /// Per-expert compensated global scales.
    pub global_scales: &'a DeviceBuffer<f32>,
    /// Original-order FP32 routing weights.
    pub routing: &'a DeviceBuffer<f32>,
    /// Block-padded assignment indices.
    pub sorted: &'a DeviceBuffer<i32>,
    /// Expert id for each padded block.
    pub expert_ids: &'a DeviceBuffer<i32>,
    /// Device scalar holding the valid padded row count.
    pub padded: &'a DeviceBuffer<i32>,
    /// FP32 cross-block reduction scratch.
    pub temporary: &'a mut DeviceBuffer<f32>,
    /// Marlin lock workspace.
    pub locks: &'a mut DeviceBuffer<i32>,
    /// BF16 output rows.
    pub output: &'a mut DeviceBuffer<bf16>,
    /// Applies routing weights inside the projection.
    pub multiply_routing: bool,
    /// Uses independent atomic partial reductions instead of Marlin's blocking locks.
    pub atomic_reduce: bool,
}

/// Device buffers consumed by one Marlin OCP MXFP4 `MoE` projection.
pub struct MarlinMxFp4MoeOperands<'a> {
    /// BF16 input rows.
    pub input: &'a DeviceBuffer<bf16>,
    /// Repacked OCP MXFP4 expert weights.
    pub weight: &'a DeviceBuffer<u8>,
    /// Marlin-formatted E8M0 block scales.
    pub scales: &'a DeviceBuffer<u8>,
    /// Original-order FP32 routing weights.
    pub routing: &'a DeviceBuffer<f32>,
    /// Block-padded assignment indices.
    pub sorted: &'a DeviceBuffer<i32>,
    /// Expert id for every padded block.
    pub expert_ids: &'a DeviceBuffer<i32>,
    /// Device scalar holding the valid padded row count.
    pub padded: &'a DeviceBuffer<i32>,
    /// FP32 cross-block reduction scratch.
    pub temporary: &'a mut DeviceBuffer<f32>,
    /// Marlin lock workspace.
    pub locks: &'a mut DeviceBuffer<i32>,
    /// BF16 projection output.
    pub output: &'a mut DeviceBuffer<bf16>,
    /// Applies routing weights inside the projection.
    pub multiply_routing: bool,
    /// Uses atomic partial reductions instead of blocking locks.
    pub atomic_reduce: bool,
}

impl Context {
    /// Enqueues one Torch-free Marlin OCP MXFP4 `MoE` projection.
    pub fn marlin_mxfp4_moe(
        &self,
        stream: &Stream,
        spec: MarlinNvFp4MoeSpec,
        operands: &MarlinMxFp4MoeOperands<'_>,
    ) -> Result<()> {
        Ok(self.native.marlin_mxfp4_moe(
            &stream.native,
            spec.native(),
            &operands.input.native,
            &operands.weight.native,
            &operands.scales.native,
            &operands.routing.native,
            &operands.sorted.native,
            &operands.expert_ids.native,
            &operands.padded.native,
            &operands.temporary.native,
            &operands.locks.native,
            &operands.output.native,
            operands.multiply_routing,
            operands.atomic_reduce,
        )?)
    }

    /// Enqueues one Torch-free standard dense Marlin NVFP4 projection.
    pub fn marlin_nvfp4_dense(
        &self,
        stream: &Stream,
        spec: MarlinNvFp4MoeSpec,
        operands: &MarlinNvFp4DenseOperands<'_>,
    ) -> Result<()> {
        Ok(self.native.marlin_nvfp4_dense(
            &stream.native,
            spec.native(),
            &operands.input.native,
            &operands.weight.native,
            &operands.scales.native,
            &operands.global_scale.native,
            &operands.temporary.native,
            &operands.locks.native,
            &operands.output.native,
            operands.atomic_reduce,
        )?)
    }

    /// Buckets assignments, converts routing weights, and pads expert blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn marlin_prepare_moe_routes(
        &self,
        stream: &Stream,
        spec: MarlinNvFp4MoeSpec,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        sorted: &mut DeviceBuffer<i32>,
        expert_ids: &mut DeviceBuffer<i32>,
        padded: &mut DeviceBuffer<i32>,
        offsets: &mut DeviceBuffer<i32>,
        routing_f32: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        Ok(self.native.marlin_prepare_moe_routes(
            &stream.native,
            spec.native(),
            &selected.native,
            &routing.native,
            &sorted.native,
            &expert_ids.native,
            &padded.native,
            &offsets.native,
            &routing_f32.native,
        )?)
    }

    /// Enqueues one Torch-free Marlin NVFP4 `MoE` projection.
    pub fn marlin_nvfp4_moe(
        &self,
        stream: &Stream,
        spec: MarlinNvFp4MoeSpec,
        operands: &MarlinNvFp4MoeOperands<'_>,
    ) -> Result<()> {
        Ok(self.native.marlin_nvfp4_moe(
            &stream.native,
            spec.native(),
            &operands.input.native,
            &operands.weight.native,
            &operands.scales.native,
            &operands.global_scales.native,
            &operands.routing.native,
            &operands.sorted.native,
            &operands.expert_ids.native,
            &operands.padded.native,
            &operands.temporary.native,
            &operands.locks.native,
            &operands.output.native,
            operands.multiply_routing,
            operands.atomic_reduce,
        )?)
    }
}
