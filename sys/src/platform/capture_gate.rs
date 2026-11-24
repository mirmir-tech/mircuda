//! Keeps device-synchronizing teardown out of stream captures.
//!
//! `cublasDestroy` and `cublasLtDestroy` synchronize the whole device. Called
//! on any thread while another thread captures a stream, that synchronization
//! invalidates the capture even in thread-local mode. Such teardown therefore
//! runs only while no capture is in progress: during a capture it is deferred
//! and runs when the last capture ends. Deferring instead of waiting means a
//! drop never blocks on another thread's capture, so it cannot deadlock with
//! locks that capture takes.

use std::{
    cell::Cell,
    sync::{Mutex, PoisonError},
};

type Teardown = Box<dyn FnOnce() + Send>;

struct State {
    /// Threads with a capture in progress.
    active: usize,
    /// Teardown waiting for `active` to reach zero.
    deferred: Vec<Teardown>,
}

static STATE: Mutex<State> = Mutex::new(State { active: 0, deferred: Vec::new() });

thread_local! {
    /// Captures this thread has begun and not yet ended.
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Marks a capture this thread is beginning. Nested captures on the same
/// thread count once.
pub(super) fn enter() {
    if DEPTH.get() == 0 {
        STATE.lock().unwrap_or_else(PoisonError::into_inner).active += 1;
    }
    DEPTH.set(DEPTH.get() + 1);
}

/// Ends the capture marked by the matching [`enter`]. The last capture to end
/// runs the deferred teardown before another capture can begin.
pub(super) fn leave() {
    let depth = DEPTH.get().saturating_sub(1);
    DEPTH.set(depth);
    if depth > 0 {
        return;
    }
    let mut state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    state.active = state.active.saturating_sub(1);
    if state.active == 0 {
        for teardown in std::mem::take(&mut state.deferred) {
            teardown();
        }
    }
    drop(state);
}

/// Calls [`leave`] when dropped, so every exit from ending a capture
/// releases its mark.
pub(super) struct Leave;

impl Drop for Leave {
    fn drop(&mut self) {
        leave();
    }
}

/// Runs device-synchronizing teardown now when no capture is in progress,
/// otherwise once the last capture ends.
///
/// A thread inside its own capture runs `teardown` directly, as before: the
/// synchronization then reports that misuse through the capture result.
pub(super) fn release(teardown: impl FnOnce() + Send + 'static) {
    if DEPTH.get() > 0 {
        teardown();
        return;
    }
    let mut state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    if state.active > 0 {
        state.deferred.push(Box::new(teardown));
    } else {
        // Holding the state keeps a capture from beginning during teardown.
        teardown();
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
