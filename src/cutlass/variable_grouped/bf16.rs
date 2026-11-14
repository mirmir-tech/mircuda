use crate::{Context, DeviceBuffer, Error, Result, Stream, bf16};

/// Capacity of a variable-row grouped BF16 product.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct VariableGroupedBf16Spec {
    groups: usize,
    max_rows: usize,
    n: usize,
    k: usize,
    capacity_rows: usize,
}

impl VariableGroupedBf16Spec {
    /// Validates a grouped product driven by device-resident rows and offsets.
    pub fn new(
        groups: usize,
        max_rows: usize,
        n: usize,
        k: usize,
        capacity_rows: usize,
    ) -> Result<Self> {
        if groups == 0
            || max_rows == 0
            || n == 0
            || k == 0
            || capacity_rows == 0
            || !n.is_multiple_of(8)
            || !k.is_multiple_of(8)
        {
            return Err(Error::InvalidMatmulShape);
        }
        for value in [groups, max_rows, n, k, capacity_rows] {
            let _ = u32::try_from(value)?;
        }
        capacity_rows.checked_mul(k).ok_or(Error::InvalidMatmulShape)?;
        capacity_rows.checked_mul(n).ok_or(Error::InvalidMatmulShape)?;
        Ok(Self { groups, max_rows, n, k, capacity_rows })
    }

    /// Number of metadata-driven products.
    #[must_use]
    pub const fn groups(self) -> usize {
        self.groups
    }

    /// Maximum number of compact rows assigned to one group.
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

    /// Total compact input and output row capacity.
    #[must_use]
    pub const fn capacity_rows(self) -> usize {
        self.capacity_rows
    }

    const fn native(self) -> mircuda_sys::VariableGroupedBf16Spec {
        mircuda_sys::VariableGroupedBf16Spec {
            groups: self.groups,
            max_rows: self.max_rows,
            n: self.n,
            k: self.k,
            capacity_rows: self.capacity_rows,
        }
    }
}

/// Persistent grouped BF16 tensor-core plan with GPU-resident routing metadata.
#[derive(Debug)]
pub struct VariableGroupedBf16Plan {
    native: mircuda_sys::VariableGroupedBf16Plan,
    spec: VariableGroupedBf16Spec,
}

impl VariableGroupedBf16Plan {
    /// Creates a capacity-bounded plan without synchronizing the stream.
    pub fn new(context: &Context, stream: &Stream, spec: VariableGroupedBf16Spec) -> Result<Self> {
        Ok(Self {
            native: context
                .native
                .create_variable_grouped_bf16_plan(&stream.native, spec.native())?,
            spec,
        })
    }

    /// Returns the immutable capacity associated with this plan.
    #[must_use]
    pub const fn spec(&self) -> VariableGroupedBf16Spec {
        self.spec
    }

    /// Enqueues all products without reading routing metadata on the host.
    ///
    /// Weights use canonical `[group, N, K]` row-major storage. Compact input
    /// and output use `[capacity_rows, K]` and `[capacity_rows, N]` row-major.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weights: &DeviceBuffer<bf16>,
        rows: &DeviceBuffer<u32>,
        offsets: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
        beta: f32,
    ) -> Result<()> {
        Ok(self.native.execute(
            &stream.native, &input.native, &weights.native, &rows.native, &offsets.native,
            &output.native, beta,
        )?)
    }
}
