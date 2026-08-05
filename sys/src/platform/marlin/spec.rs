#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MarlinNvFp4RepackSpec {
    pub experts: usize,
    pub n: usize,
    pub k: usize,
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
}
