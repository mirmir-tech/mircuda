use super::{Context, unsupported};
use crate::Result;

#[derive(Debug)]
pub struct ProfilerRange;

impl Context {
    pub const fn start_profiler_range(&self) -> Result<ProfilerRange> {
        Err(unsupported())
    }
}

impl ProfilerRange {
    pub const fn stop(self) -> Result<()> {
        Err(unsupported())
    }
}
