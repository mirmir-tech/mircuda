use std::sync::{Arc, Mutex, PoisonError};

use cudarc::driver::{CudaStream, DriverError, sys};

/// Streams other than the allocation stream that queued work reading or
/// writing one device allocation.
///
/// Releasing the allocation orders its stream-ordered free after the work
/// queued on exactly these streams, so dropping a buffer never synchronizes
/// the whole context: a context-wide synchronization would block the host and
/// invalidate stream captures running on other threads.
#[derive(Debug, Default)]
pub struct ForeignStreams(Mutex<Vec<Arc<CudaStream>>>);

impl ForeignStreams {
    /// Records that `stream` queued work on the allocation.
    pub(in crate::platform) fn record(&self, stream: &Arc<CudaStream>) {
        let mut streams = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if !streams.iter().any(|known| known.cu_stream() == stream.cu_stream()) {
            streams.push(stream.clone());
        }
    }

    /// Makes `allocation` wait for the work already queued on every recorded
    /// stream, without blocking the host. Called once, when the allocation is
    /// released, so it drains the recorded streams.
    pub(super) fn order_before(&self, allocation: &CudaStream) -> Result<(), DriverError> {
        let streams = std::mem::take(&mut *self.0.lock().unwrap_or_else(PoisonError::into_inner));
        for stream in &streams {
            let event = allocation
                .context()
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))?;
            event.record(stream)?;
            allocation.wait(&event)?;
        }
        Ok(())
    }
}
