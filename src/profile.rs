use crate::{Context, Result};

/// Explicit CUDA Profiler API capture range.
#[derive(Debug)]
pub struct ProfilerRange {
    native: mircuda_sys::ProfilerRange,
}

impl Context {
    /// Starts collection for tools configured to use the CUDA Profiler API.
    pub fn start_profiler_range(&self) -> Result<ProfilerRange> {
        Ok(ProfilerRange {
            native: self.native.start_profiler_range()?,
        })
    }
}

impl ProfilerRange {
    /// Stops collection and reports any CUDA Driver API failure.
    pub fn stop(self) -> Result<()> {
        Ok(self.native.stop()?)
    }
}
