use crate::{Context, DeviceBuffer, Error, Result, Stream, bf16};

mod paired;

pub use paired::{
    PairedVariableGroupedFp4Launch, PairedVariableGroupedFp4Plan, VariableGroupedFp4Metadata,
    VariableGroupedFp4Operands,
};

/// Capacity and maximum geometry for variable-row grouped FP4 multiplication.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct VariableGroupedFp4Spec {
    groups: usize,
    matrices: usize,
    max_rows: usize,
    n: usize,
    k: usize,
    capacity_rows: usize,
}

impl VariableGroupedFp4Spec {
    /// Validates a compact grouped product with device-resident row counts and offsets.
    pub fn new(
        groups: usize,
        matrices: usize,
        max_rows: usize,
        n: usize,
        k: usize,
        capacity_rows: usize,
    ) -> Result<Self> {
        if groups == 0
            || matrices == 0
            || max_rows == 0
            || n == 0
            || k == 0
            || capacity_rows == 0
            || !k.is_multiple_of(64)
        {
            return Err(Error::InvalidMatmulShape);
        }
        for value in [groups, matrices, max_rows, n, k, capacity_rows] {
            let _ = u32::try_from(value)?;
        }
        capacity_rows.checked_mul(k / 2).ok_or(Error::InvalidMatmulShape)?;
        capacity_rows.checked_mul(n).ok_or(Error::InvalidMatmulShape)?;
        Ok(Self {
            groups,
            matrices,
            max_rows,
            n,
            k,
            capacity_rows,
        })
    }

    /// Number of products described by device metadata.
    #[must_use]
    pub const fn groups(self) -> usize {
        self.groups
    }

    /// Number of matrices in the B bank.
    #[must_use]
    pub const fn matrices(self) -> usize {
        self.matrices
    }

    /// Maximum rows accepted for one product.
    #[must_use]
    pub const fn max_rows(self) -> usize {
        self.max_rows
    }

    /// Output columns in every product.
    #[must_use]
    pub const fn n(self) -> usize {
        self.n
    }

    /// Reduction dimension in every product.
    #[must_use]
    pub const fn k(self) -> usize {
        self.k
    }

    /// Total compact A and output row capacity.
    #[must_use]
    pub const fn capacity_rows(self) -> usize {
        self.capacity_rows
    }

    const fn native(self) -> mircuda_sys::VariableGroupedFp4Spec {
        mircuda_sys::VariableGroupedFp4Spec {
            groups: self.groups,
            matrices: self.matrices,
            max_m: self.max_rows,
            n: self.n,
            k: self.k,
            capacity_rows: self.capacity_rows,
        }
    }
}

/// Persistent CUTLASS plan whose row counts and compact offsets stay on CUDA.
#[derive(Debug)]
pub struct VariableGroupedFp4Plan {
    native: mircuda_sys::VariableGroupedFp4Plan,
    spec: VariableGroupedFp4Spec,
}

impl VariableGroupedFp4Plan {
    /// Creates a capacity-bounded plan without synchronizing the stream.
    pub fn new(context: &Context, stream: &Stream, spec: VariableGroupedFp4Spec) -> Result<Self> {
        Ok(Self {
            native: context
                .native
                .create_variable_grouped_fp4_plan(&stream.native, spec.native())?,
            spec,
        })
    }

    /// Returns the immutable capacity associated with this plan.
    #[must_use]
    pub const fn spec(&self) -> VariableGroupedFp4Spec {
        self.spec
    }

    /// Enqueues all products without reading row counts or offsets on the host.
    ///
    /// `rows[group]` must not exceed `max_rows`. `offsets[group]` must identify
    /// a non-overlapping region of `capacity_rows`. `scale_offsets[group]`
    /// identifies its 128-row-aligned activation-scale region.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        stream: &Stream,
        a: &DeviceBuffer<u8>,
        a_scales: &DeviceBuffer<u8>,
        b: &DeviceBuffer<u8>,
        b_scales: &DeviceBuffer<u8>,
        alphas: &DeviceBuffer<f32>,
        indices: &DeviceBuffer<u32>,
        rows: &DeviceBuffer<u32>,
        offsets: &DeviceBuffer<u32>,
        scale_offsets: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        Ok(self.native.execute(
            &stream.native,
            &a.native,
            &a_scales.native,
            &b.native,
            &b_scales.native,
            &alphas.native,
            &indices.native,
            &rows.native,
            &offsets.native,
            &scale_offsets.native,
            &output.native,
        )?)
    }
}
