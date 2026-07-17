use super::VariableGroupedFp4Spec;
use crate::{Context, DeviceBuffer, Result, Stream, bf16};

/// One operand set for a paired variable-row grouped FP4 dispatch.
pub struct VariableGroupedFp4Operands<'a> {
    /// Compact packed FP4 activation rows.
    pub a: &'a DeviceBuffer<u8>,
    /// CUTLASS-formatted activation scales.
    pub a_scales: &'a DeviceBuffer<u8>,
    /// Packed matrix bank.
    pub b: &'a DeviceBuffer<u8>,
    /// CUTLASS-formatted matrix scales.
    pub b_scales: &'a DeviceBuffer<u8>,
    /// Per-matrix FP32 epilogue multipliers.
    pub alphas: &'a DeviceBuffer<f32>,
    /// Compact BF16 result rows.
    pub output: &'a mut DeviceBuffer<bf16>,
}

/// Shared device-resident scheduling metadata for a paired dispatch.
#[derive(Clone, Copy)]
pub struct VariableGroupedFp4Metadata<'a> {
    /// Matrix index selected for every logical group.
    pub indices: &'a DeviceBuffer<u32>,
    /// Active row count for every logical group.
    pub rows: &'a DeviceBuffer<u32>,
    /// Compact row offset for every logical group.
    pub offsets: &'a DeviceBuffer<u32>,
}

/// Two independent grouped products sharing only scheduling metadata.
pub struct PairedVariableGroupedFp4Launch<'a> {
    /// First grouped product.
    pub left: VariableGroupedFp4Operands<'a>,
    /// Second grouped product.
    pub right: VariableGroupedFp4Operands<'a>,
    /// Metadata shared by both products.
    pub metadata: VariableGroupedFp4Metadata<'a>,
}

/// Persistent CUTLASS plan issuing two same-geometry grouped products through
/// one device scheduler.
#[derive(Debug)]
pub struct PairedVariableGroupedFp4Plan {
    native: mircuda_sys::PairedVariableGroupedFp4Plan,
    spec: VariableGroupedFp4Spec,
}

impl PairedVariableGroupedFp4Plan {
    /// Creates a paired plan without allocating or joining operand banks.
    pub fn new(context: &Context, stream: &Stream, spec: VariableGroupedFp4Spec) -> Result<Self> {
        Ok(Self {
            native: context
                .native
                .create_paired_variable_grouped_fp4_plan(&stream.native, spec.native())?,
            spec,
        })
    }

    /// Returns the geometry of each logical product set.
    #[must_use]
    pub const fn spec(&self) -> VariableGroupedFp4Spec {
        self.spec
    }

    /// Enqueues both products in one grouped CUTLASS dispatch.
    pub fn execute(
        &mut self,
        stream: &Stream,
        launch: &mut PairedVariableGroupedFp4Launch<'_>,
    ) -> Result<()> {
        Ok(self.native.execute(
            &stream.native,
            &launch.left.a.native,
            &launch.left.a_scales.native,
            &launch.left.b.native,
            &launch.left.b_scales.native,
            &launch.left.alphas.native,
            &launch.right.a.native,
            &launch.right.a_scales.native,
            &launch.right.b.native,
            &launch.right.b_scales.native,
            &launch.right.alphas.native,
            &launch.metadata.indices.native,
            &launch.metadata.rows.native,
            &launch.metadata.offsets.native,
            &launch.left.output.native,
            &launch.right.output.native,
        )?)
    }
}
