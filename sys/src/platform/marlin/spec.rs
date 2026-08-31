#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MarlinNvFp4RepackSpec {
    pub experts: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MarlinMxFp4RepackSpec {
    pub experts: usize,
    pub n: usize,
    pub k: usize,
    pub logical_n: usize,
    pub logical_k: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MarlinNvFp4MoeSpec {
    pub experts: usize,
    pub tokens: usize,
    pub top_k: usize,
    pub n: usize,
    pub k: usize,
    pub thread_config: MarlinNvFp4ThreadConfig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum MarlinNvFp4ThreadConfig {
    N128K128 = 0,
    N128K64 = 1,
    N64K128 = 2,
    M64N256K64 = 3,
    M64N128K64 = 4,
    M64N64K128 = 5,
}

impl MarlinNvFp4ThreadConfig {
    pub(super) const fn moe_block_size(self) -> usize {
        match self {
            Self::N128K128 | Self::N128K64 | Self::N64K128 => 8,
            Self::M64N256K64 | Self::M64N128K64 | Self::M64N64K128 => 64,
        }
    }
}
