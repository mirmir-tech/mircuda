use crate::{Context, DeviceBuffer, Error, Result, Stream, bf16};

/// Fixed geometry for device-indexed grouped block-scaled FP4 multiplication.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct IndexedGroupedFp4Spec {
    groups: usize,
    matrices: usize,
    m: usize,
    n: usize,
    k: usize,
    broadcast_input: bool,
}

impl IndexedGroupedFp4Spec {
    /// Validates fixed-row `M x K` by `N x K` products selected from a matrix bank.
    pub fn new(groups: usize, matrices: usize, m: usize, n: usize, k: usize) -> Result<Self> {
        if groups == 0 || matrices == 0 || m == 0 || n == 0 || k == 0 || !k.is_multiple_of(64) {
            return Err(Error::InvalidMatmulShape);
        }
        for value in [groups, matrices, m, n, k] {
            let _ = i32::try_from(value)?;
        }
        Ok(Self {
            groups,
            matrices,
            m,
            n,
            k,
            broadcast_input: false,
        })
    }

    /// Reuses one A matrix and its block scales for every grouped product.
    #[must_use]
    pub const fn with_broadcast_input(mut self) -> Self {
        self.broadcast_input = true;
        self
    }

    /// Number of independent products.
    #[must_use]
    pub const fn groups(self) -> usize {
        self.groups
    }

    /// Number of matrices in the device-resident B bank.
    #[must_use]
    pub const fn matrices(self) -> usize {
        self.matrices
    }

    /// Rows in each grouped product.
    #[must_use]
    pub const fn m(self) -> usize {
        self.m
    }

    /// Output columns in each product.
    #[must_use]
    pub const fn n(self) -> usize {
        self.n
    }

    /// Reduction dimension in each product.
    #[must_use]
    pub const fn k(self) -> usize {
        self.k
    }

    const fn native(self) -> mircuda_sys::IndexedGroupedFp4Spec {
        mircuda_sys::IndexedGroupedFp4Spec {
            groups: self.groups,
            matrices: self.matrices,
            m: self.m,
            n: self.n,
            k: self.k,
            broadcast_input: self.broadcast_input,
        }
    }
}

/// Persistent SM120 CUTLASS plan with device-resident matrix selection.
#[derive(Debug)]
pub struct IndexedGroupedFp4Plan {
    native: mircuda_sys::IndexedGroupedFp4Plan,
    spec: IndexedGroupedFp4Spec,
}

impl IndexedGroupedFp4Plan {
    /// Enqueues persistent CUTLASS metadata and workspace creation on `stream`.
    pub fn new(context: &Context, stream: &Stream, spec: IndexedGroupedFp4Spec) -> Result<Self> {
        Ok(Self {
            native: context
                .native
                .create_indexed_grouped_fp4_plan(&stream.native, spec.native())?,
            spec,
        })
    }

    /// Returns the immutable geometry associated with this plan.
    #[must_use]
    pub const fn spec(&self) -> IndexedGroupedFp4Spec {
        self.spec
    }

    /// Enqueues all indexed products without host selection, allocation, or synchronization.
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
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        Ok(self.native.execute(
            &stream.native, &a.native, &a_scales.native, &b.native, &b_scales.native,
            &alphas.native, &indices.native, &output.native,
        )?)
    }
}
