use std::sync::Arc;

use cudarc::driver::{CudaContext, profiler_start, profiler_stop};

use super::Context;
use crate::Result;

#[derive(Debug)]
pub struct ProfilerRange {
    context: Arc<CudaContext>,
    active: bool,
}

impl Context {
    pub fn start_profiler_range(&self) -> Result<ProfilerRange> {
        self.inner.bind_to_thread()?;
        profiler_start()?;
        Ok(ProfilerRange {
            context: Arc::clone(&self.inner),
            active: true,
        })
    }
}

impl ProfilerRange {
    pub fn stop(mut self) -> Result<()> {
        self.context.bind_to_thread()?;
        profiler_stop()?;
        self.active = false;
        Ok(())
    }
}

impl Drop for ProfilerRange {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        match self.context.bind_to_thread() {
            Ok(()) => self.context.record_err(profiler_stop()),
            Err(error) => self.context.record_err::<()>(Err(error)),
        }
    }
}
