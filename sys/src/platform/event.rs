use cudarc::driver::{CudaEvent, sys};

use super::driver::{Context, Stream};
use crate::Result;

#[derive(Debug)]
pub struct Event {
    pub(super) inner: CudaEvent,
}

impl Context {
    pub fn create_event(&self, timing: bool) -> Result<Event> {
        let flags = if timing {
            sys::CUevent_flags::CU_EVENT_DEFAULT
        } else {
            sys::CUevent_flags::CU_EVENT_DISABLE_TIMING
        };
        Ok(Event {
            inner: self.inner.new_event(Some(flags))?,
        })
    }
}

impl Event {
    pub fn record(&self, stream: &Stream) -> Result<()> {
        Ok(self.inner.record(&stream.inner)?)
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(self.inner.synchronize()?)
    }

    pub fn elapsed_ms(&self, end: &Self) -> Result<f32> {
        Ok(self.inner.elapsed_ms(&end.inner)?)
    }
}

impl Stream {
    pub fn wait(&self, event: &Event) -> Result<()> {
        Ok(self.inner.wait(&event.inner)?)
    }
}
