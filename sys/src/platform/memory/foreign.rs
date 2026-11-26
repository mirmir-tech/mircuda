use std::sync::{Arc, Mutex, PoisonError};

use cudarc::driver::{CudaStream, DriverError, result, sys};

/// Streams other than the allocation stream that queued work reading or
/// writing one device allocation.
///
/// Releasing the allocation orders its stream-ordered free after the work
/// queued on exactly these streams, so dropping a buffer never synchronizes
/// the whole context: a context-wide synchronization would block the host and
/// invalidate stream captures running on other threads. While one of these
/// streams or the allocation stream records into a capture other than the
/// one that allocated the buffer, the release waits for that capture to end
/// instead of ordering into it.
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

    /// Takes the recorded streams. Called once, when the allocation is
    /// released.
    pub(super) fn take(&self) -> Vec<Arc<CudaStream>> {
        std::mem::take(&mut *self.0.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

/// Makes `allocation` wait for the work already queued on every one of
/// `streams`, without blocking the host. The context must be current, and
/// the streams must capture nothing or the same capture as `allocation`: an
/// event recorded into one capture cannot order work outside it. Raw driver
/// calls leave errors other operations recorded on the context in place.
pub(super) fn order_before(
    streams: &[Arc<CudaStream>],
    allocation: &CudaStream,
) -> Result<(), DriverError> {
    for stream in streams {
        let event = result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)?;
        // SAFETY: the event was just created, and both streams are live and
        // belong to the current context.
        let ordered =
            unsafe { result::event::record(event, stream.cu_stream()) }.and_then(|()| unsafe {
                result::stream::wait_event(
                    allocation.cu_stream(),
                    event,
                    sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                )
            });
        // SAFETY: the event is owned here; a wait already enqueued on it stays
        // valid after its destruction.
        let destroyed = unsafe { result::event::destroy(event) };
        ordered.and(destroyed)?;
    }
    Ok(())
}
