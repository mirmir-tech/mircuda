//! Keeps device-synchronizing teardown out of stream captures.
//!
//! `cublasDestroy` and `cublasLtDestroy` synchronize the whole device. Called
//! on any thread while any thread, itself included, captures a stream, that
//! synchronization invalidates the capture even in thread-local mode. Such
//! teardown therefore runs only while no capture is in progress: during a
//! capture it is deferred and runs when the last capture ends. Deferring
//! instead of waiting means a drop never blocks on another thread's capture,
//! so it cannot deadlock with locks that capture takes.
//!
//! Stream-ordered teardown, which enqueues work on given streams, is deferred
//! only while one of those streams is capturing a capture other than the one
//! the teardown belongs to, and runs as soon as none is.

use std::{
    cell::Cell,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicUsize, Ordering},
    },
};

use cudarc::driver::CudaStream;

mod deferred;
#[cfg(test)]
pub(in crate::platform) mod tests;

pub(super) use self::deferred::{CaptureId, capture_of};
use self::deferred::{Deferred, Ordered};

pub(super) type Teardown = Box<dyn FnOnce() + Send>;

struct State {
    /// Threads with a capture in progress.
    active: usize,
    /// Teardown waiting for the captures it would disturb to end.
    deferred: Vec<Deferred>,
}

static STATE: Mutex<State> = Mutex::new(State { active: 0, deferred: Vec::new() });

/// `State::active`, readable without the lock so that a release outside any
/// capture stays cheap.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Captures this thread has begun and not yet ended.
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Begins a capture on this thread through `begin` and marks it when it
/// succeeds. Nested captures on the same thread count once. Holding the state
/// across `begin` keeps a capture from beginning while teardown checks or uses
/// its stream.
pub(super) fn enter<T, E>(begin: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    let mut state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    let capture = begin()?;
    if DEPTH.get() == 0 {
        state.active += 1;
        ACTIVE.store(state.active, Ordering::Release);
    }
    DEPTH.set(DEPTH.get() + 1);
    drop(state);
    Ok(capture)
}

/// Ends the capture marked by the matching [`enter`], then runs the deferred
/// teardown that no capture blocks any longer before another capture can
/// begin. A thread with no capture marked leaves nothing, so it cannot end
/// another thread's.
pub(super) fn leave() {
    let depth = DEPTH.get();
    debug_assert!(depth > 0, "a thread left a capture it never entered");
    if depth == 0 {
        return;
    }
    DEPTH.set(depth - 1);
    let mut state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    if depth == 1 {
        state.active = state.active.saturating_sub(1);
        ACTIVE.store(state.active, Ordering::Release);
    }
    let panic = deferred::run_ready(&mut state);
    drop(state);
    if let Some(panic) = panic
        && !std::thread::panicking()
    {
        std::panic::resume_unwind(panic);
    }
}

/// Whether no capture is in progress anywhere. A release that sees this may
/// skip the gate, as long as its own thread begins no capture before it ends.
pub(super) fn idle() -> bool {
    ACTIVE.load(Ordering::Acquire) == 0
}

/// Captures in progress and the teardown waiting for them.
// Not part of the supported API: exists for tests and diagnostics.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CaptureActivity {
    /// Threads with a capture in progress.
    pub capturing_threads: usize,
    /// Teardown deferred until those captures end.
    pub deferred_teardown: usize,
}

/// The process-wide capture activity, for diagnostics and tests.
#[doc(hidden)]
#[must_use]
pub fn activity() -> CaptureActivity {
    let state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    CaptureActivity {
        capturing_threads: state.active,
        deferred_teardown: state.deferred.len(),
    }
}

/// Runs device-synchronizing teardown now when no capture is in progress,
/// otherwise once the last capture ends, even when the only capture is the
/// calling thread's own. Teardown runs with the gate held, so it must not
/// release anything through the gate itself.
pub(super) fn release(teardown: impl FnOnce() + Send + 'static) {
    let mut state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    if state.active > 0 {
        state.deferred.push(Deferred::Device(Box::new(teardown)));
    } else {
        // Holding the state keeps a capture from beginning during teardown.
        teardown();
    }
    drop(state);
}

/// Runs teardown that enqueues work on `streams` now when none of them is
/// capturing a capture other than `own`, otherwise as soon as none is.
/// `ordered` returns the device-synchronizing fallback it needs when stream
/// ordering fails, which then runs as [`release`] would run it.
///
/// Holding the state from the check through `ordered` keeps a capture from
/// beginning on those streams in between.
pub(super) fn release_ordered<F>(streams: Vec<Arc<CudaStream>>, own: Option<CaptureId>, ordered: F)
where
    F: FnOnce() -> Result<(), Teardown> + Send + 'static,
{
    let mut state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    let entry = Deferred::Ordered(Ordered { streams, own, release: Box::new(ordered) });
    if entry.blocked(state.active) {
        state.deferred.push(entry);
    } else {
        entry.run(&mut state);
    }
    drop(state);
}

/// A native plan handle moved into deferred teardown.
pub(super) struct NativePlan(*mut std::ffi::c_void);

// SAFETY: the plan is owned by its teardown, which is its only remaining user.
unsafe impl Send for NativePlan {}

impl NativePlan {
    pub(super) const fn new(raw: *mut std::ffi::c_void) -> Self {
        Self(raw)
    }

    pub(super) const fn into_raw(self) -> *mut std::ffi::c_void {
        self.0
    }
}
