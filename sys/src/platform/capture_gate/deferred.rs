//! Teardown waiting in the capture gate, and the capture state it waits on.

use std::{
    any::Any,
    sync::{Arc, atomic::Ordering},
};

use cudarc::driver::{CudaStream, DriverError, sys};

use super::{super::driver::bind_current, ACTIVE, Teardown};

/// The capture a stream records into, as the driver identifies it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::platform) struct CaptureId(u64);

/// Teardown waiting in the gate.
pub(super) enum Deferred {
    /// Device-synchronizing teardown, waiting for every capture to end.
    Device(Teardown),
    /// Stream-ordered teardown, waiting for its streams' captures to end.
    Ordered(Ordered),
}

pub(super) struct Ordered {
    /// Streams the teardown enqueues work on.
    pub(super) streams: Vec<Arc<CudaStream>>,
    /// The capture the teardown belongs to, which it may enqueue into.
    pub(super) own: Option<CaptureId>,
    /// The teardown, returning its device-synchronizing fallback on failure.
    pub(super) release: Box<dyn FnOnce() -> Result<(), Teardown> + Send>,
}

impl Deferred {
    /// Whether running now would disturb a capture, given `active` threads
    /// capturing. A failed capture query counts as disturbing one.
    pub(super) fn blocked(&self, active: usize) -> bool {
        if active == 0 {
            return false;
        }
        match self {
            Self::Device(_) => true,
            Self::Ordered(ordered) => ordered.streams.iter().any(|stream| match capture(stream) {
                Ok(None) => false,
                Ok(Some(id)) => ordered.own != Some(id),
                Err(_) => true,
            }),
        }
    }

    /// Runs the teardown. A failed stream ordering returns its
    /// device-synchronizing fallback, for the gate to run once no capture
    /// is in progress.
    pub(super) fn run(self) -> Option<Self> {
        match self {
            Self::Device(teardown) => {
                teardown();
                None
            },
            Self::Ordered(ordered) => (ordered.release)().err().map(Self::Device),
        }
    }
}

/// A panic raised by a deferred teardown.
pub(super) type Panic = Box<dyn Any + Send>;

/// The capture `stream` records into, or `None` when it is not capturing.
/// The query needs the stream's context current, whichever context the
/// releasing thread last used.
fn capture(stream: &CudaStream) -> Result<Option<CaptureId>, DriverError> {
    let _current = bind_current(stream.context())?;
    let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let mut id = 0;
    // SAFETY: the stream is live and its context current, both outputs are
    // writable, and the optional graph and dependency outputs are null. The
    // query enqueues nothing.
    unsafe {
        sys::cuStreamGetCaptureInfo_v3(
            stream.cu_stream(),
            &raw mut status,
            &raw mut id,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    }
    .result()?;
    let capturing = status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    Ok(capturing.then_some(CaptureId(id)))
}

/// The capture work enqueued on `stream` now records into. Without a capture
/// in progress anywhere it is `None` without asking the driver.
pub(in crate::platform) fn capture_of(stream: &CudaStream) -> Option<CaptureId> {
    if ACTIVE.load(Ordering::Acquire) == 0 {
        return None;
    }
    capture(stream).ok().flatten()
}
