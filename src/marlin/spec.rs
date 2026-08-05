use crate::{Error, Result};

/// Geometry of a persistent NVFP4 expert-bank repack into Marlin tiles.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MarlinNvFp4RepackSpec {
    pub(super) experts: usize,
    pub(super) n: usize,
    pub(super) k: usize,
}

impl MarlinNvFp4RepackSpec {
    /// Creates a bank repack for canonical `[expert, N, K / 2]` weights.
    pub fn new(experts: usize, n: usize, k: usize) -> Result<Self> {
        validate_geometry(experts, n, k, 16)?;
        Ok(Self { experts, n, k })
    }

    pub(super) const fn native(self) -> mircuda_sys::MarlinNvFp4RepackSpec {
        mircuda_sys::MarlinNvFp4RepackSpec {
            experts: self.experts,
            n: self.n,
            k: self.k,
        }
    }
}

/// Geometry of one BF16-by-NVFP4 Marlin `MoE` projection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MarlinNvFp4MoeSpec {
    pub(super) experts: usize,
    pub(super) tokens: usize,
    pub(super) top_k: usize,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) thread_config: MarlinNvFp4ThreadConfig,
}

/// Thread-tile geometry measured independently by the Marlin autotuner.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MarlinNvFp4ThreadConfig {
    /// 128 output columns by 128 reduction elements, using 256 threads.
    N128K128,
    /// 128 output columns by 64 reduction elements, using 128 threads.
    N128K64,
    /// 64 output columns by 128 reduction elements, using 128 threads.
    N64K128,
}

impl MarlinNvFp4MoeSpec {
    /// Creates a block-8 Marlin `MoE` projection geometry.
    pub fn new(
        experts: usize,
        tokens: usize,
        top_k: usize,
        n: usize,
        k: usize,
        thread_config: MarlinNvFp4ThreadConfig,
    ) -> Result<Self> {
        validate_geometry(experts, n, k, 128)?;
        if tokens == 0 || top_k == 0 || top_k > experts {
            return Err(Error::InvalidMatmulShape);
        }
        for value in [tokens, top_k] {
            let _ = i32::try_from(value)?;
        }
        tokens.checked_mul(top_k).ok_or(Error::InvalidMatmulShape)?;
        Ok(Self {
            experts,
            tokens,
            top_k,
            n,
            k,
            thread_config,
        })
    }

    /// Number of routed rows before block padding.
    #[must_use]
    pub const fn assignments(self) -> usize {
        self.tokens * self.top_k
    }

    /// Worst-case number of routed rows after block-8 expert padding.
    #[must_use]
    pub const fn padded_capacity(self) -> usize {
        self.assignments() + self.experts * 7
    }

    pub(super) const fn native(self) -> mircuda_sys::MarlinNvFp4MoeSpec {
        mircuda_sys::MarlinNvFp4MoeSpec {
            experts: self.experts,
            tokens: self.tokens,
            top_k: self.top_k,
            n: self.n,
            k: self.k,
            thread_config: match self.thread_config {
                MarlinNvFp4ThreadConfig::N128K128 => mircuda_sys::MarlinNvFp4ThreadConfig::N128K128,
                MarlinNvFp4ThreadConfig::N128K64 => mircuda_sys::MarlinNvFp4ThreadConfig::N128K64,
                MarlinNvFp4ThreadConfig::N64K128 => mircuda_sys::MarlinNvFp4ThreadConfig::N64K128,
            },
        }
    }
}

fn validate_geometry(experts: usize, n: usize, k: usize, k_tile: usize) -> Result<()> {
    if experts == 0 || n == 0 || k == 0 || !n.is_multiple_of(64) || !k.is_multiple_of(k_tile) {
        return Err(Error::InvalidMatmulShape);
    }
    for value in [experts, n, k] {
        let _ = i32::try_from(value)?;
    }
    experts
        .checked_mul(n)
        .and_then(|value| value.checked_mul(k))
        .ok_or(Error::InvalidMatmulShape)?;
    Ok(())
}
