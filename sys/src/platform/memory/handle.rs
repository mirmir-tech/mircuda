use cudarc::driver::sys;

#[derive(Clone, Copy, Debug)]
pub(super) struct MemoryPoolHandle(pub(super) sys::CUmemoryPool);

// SAFETY: CUDA pool handles are thread-safe and operations bind the retained context.
unsafe impl Send for MemoryPoolHandle {}
// SAFETY: shared policy updates use synchronized CUDA Driver API calls.
unsafe impl Sync for MemoryPoolHandle {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryPoolStats {
    pub reserved: u64,
    pub used: u64,
}
