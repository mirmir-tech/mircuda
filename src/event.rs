use crate::{Context, Result, Stream};

/// Explicit CUDA event used for ordering and device timing.
#[derive(Debug)]
pub struct Event {
    native: mircuda_sys::Event,
}

impl Context {
    /// Creates an event, optionally retaining device timing information.
    pub fn create_event(&self, timing: bool) -> Result<Event> {
        Ok(Event {
            native: self.native.create_event(timing)?,
        })
    }
}

impl Event {
    /// Records this event after all work currently queued on `stream`.
    pub fn record(&self, stream: &Stream) -> Result<()> {
        Ok(self.native.record(&stream.native)?)
    }

    /// Explicitly blocks until this event is complete.
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.native.synchronize()?)
    }

    /// Returns device elapsed time from this event to `end`.
    pub fn elapsed_ms(&self, end: &Self) -> Result<f32> {
        Ok(self.native.elapsed_ms(&end.native)?)
    }
}

impl Stream {
    /// Enqueues a dependency on an event without synchronizing the host.
    pub fn wait(&self, event: &Event) -> Result<()> {
        Ok(self.native.wait(&event.native)?)
    }
}
