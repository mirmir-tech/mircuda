//! Teardown waiting in the capture gate, and the capture state it waits on.

use std::{
    any::Any,
    panic::AssertUnwindSafe,
    sync::{Arc, atomic::Ordering},
};

use cudarc::driver::{CudaStream, DriverError, sys};

use super::{ACTIVE, State, Teardown};

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
            Self::Ordered(ordered) => {
                ordered.streams.iter().any(|stream| match capture(stream.cu_stream()) {
                    Ok(None) => false,
                    Ok(Some(id)) => ordered.own != Some(id),
                    Err(_) => true,
                })
            },
        }
    }

    /// Runs the teardown. A failed stream ordering leaves its fallback to run
    /// once no capture is in progress.
    pub(super) fn run(self, state: &mut State) {
        match self {
            Self::Device(teardown) => teardown(),
            Self::Ordered(ordered) => {
                if let Err(fallback) = (ordered.release)() {
                    let fallback = Self::Device(fallback);
                    if fallback.blocked(state.active) {
                        state.deferred.push(fallback);
                    } else {
                        fallback.run(state);
                    }
                }
            },
        }
    }
}

/// A panic raised by a deferred teardown.
pub(super) type Panic = Box<dyn Any + Send>;

/// Runs every deferred teardown that no capture blocks any more and keeps the
/// rest. A panicking teardown does not keep the others from running; the
/// first panic is returned for the caller to resume once the gate is free.
pub(super) fn run_ready(state: &mut State) -> Option<Panic> {
    let mut panic = None;
    for entry in std::mem::take(&mut state.deferred) {
        if entry.blocked(state.active) {
            state.deferred.push(entry);
        } else if let Err(payload) = std::panic::catch_unwind(AssertUnwindSafe(|| entry.run(state)))
        {
            panic.get_or_insert(payload);
        }
    }
    panic
}

/// The capture `stream` records into, or `None` when it is not capturing.
fn capture(stream: sys::CUstream) -> Result<Option<CaptureId>, DriverError> {
    let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let mut id = 0;
    // SAFETY: the caller keeps `stream` alive, both outputs are writable, and
    // the optional graph and dependency outputs are null. The query enqueues
    // nothing.
    unsafe {
        sys::cuStreamGetCaptureInfo_v3(
            stream,
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
    capture(stream.cu_stream()).ok().flatten()
}
