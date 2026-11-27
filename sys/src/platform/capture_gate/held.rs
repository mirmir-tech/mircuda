//! The gate held by the calling thread, and teardown released while it is.

use std::{
    cell::Cell,
    panic::AssertUnwindSafe,
    sync::{Mutex, MutexGuard, PoisonError},
};

use super::{
    STATE, State,
    deferred::{Deferred, Panic},
};

/// Whether a thread holds the gate.
#[derive(Clone, Copy, Eq, PartialEq)]
enum Holding {
    No,
    /// Held, with no teardown released under it yet.
    Yes,
    /// Held, with teardown released under it waiting in [`NESTED`].
    Nested,
}

thread_local! {
    /// Whether this thread holds the gate. Like every thread-local of the
    /// gate, it is const-initialized and needs no destructor, so it stays
    /// accessible while other thread-locals, and buffers they own, are
    /// destroyed at thread exit.
    static HOLDING: Cell<Holding> = const { Cell::new(Holding::No) };
}

/// Teardown released by teardown that runs under the gate. Locking the gate
/// again there would deadlock, so the holder settles it before letting go.
/// Only the holder pushes to it, and it is never locked while teardown runs.
static NESTED: Mutex<Vec<Deferred>> = Mutex::new(Vec::new());

/// Whether this thread holds the gate, running teardown or beginning a
/// capture.
pub(super) fn holding() -> bool {
    HOLDING.get() != Holding::No
}

/// Leaves `entry` for the gate this thread holds to settle, or returns it
/// when this thread does not hold the gate.
pub(super) fn nest(entry: Deferred) -> Option<Deferred> {
    if !holding() {
        return Some(entry);
    }
    NESTED.lock().unwrap_or_else(PoisonError::into_inner).push(entry);
    HOLDING.set(Holding::Nested);
    None
}

/// The gate, held by this thread. Teardown runs under it, so no capture can
/// begin while teardown checks or uses its streams; a teardown that panics
/// leaves the others to run, and its panic is resumed once the gate is free.
pub(super) struct Held {
    pub(super) state: MutexGuard<'static, State>,
    panic: Option<Panic>,
}

impl Held {
    /// Takes the gate. Teardown running under the gate cannot take it again:
    /// that fails here instead of deadlocking, and the gate resumes the
    /// failure once it is free.
    pub(super) fn lock() -> Self {
        assert!(!holding(), "teardown took the capture gate it runs under");
        let state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
        HOLDING.set(Holding::Yes);
        Self { state, panic: None }
    }

    /// Runs `entry` now when no capture would see it, otherwise defers it.
    pub(super) fn settle(&mut self, entry: Deferred) {
        if entry.blocked(self.state.active) {
            self.state.deferred.push(entry);
            return;
        }
        match std::panic::catch_unwind(AssertUnwindSafe(|| entry.run())) {
            Ok(Some(fallback)) => self.settle(fallback),
            Ok(None) => {},
            Err(panic) => {
                self.panic.get_or_insert(panic);
            },
        }
    }

    /// Runs every deferred teardown that no capture blocks any more and keeps
    /// the rest.
    pub(super) fn run_ready(&mut self) {
        for entry in std::mem::take(&mut self.state.deferred) {
            self.settle(entry);
        }
    }

    /// Settles the teardown that teardown run under the gate released.
    fn settle_nested(&mut self) {
        while HOLDING.get() == Holding::Nested {
            HOLDING.set(Holding::Yes);
            let nested =
                std::mem::take(&mut *NESTED.lock().unwrap_or_else(PoisonError::into_inner));
            for entry in nested {
                self.settle(entry);
            }
        }
    }

    /// Lets go of the gate, then resumes the first panic a teardown raised.
    pub(super) fn unlock(mut self) {
        self.settle_nested();
        let panic = self.panic.take();
        drop(self);
        if let Some(panic) = panic
            && !std::thread::panicking()
        {
            std::panic::resume_unwind(panic);
        }
    }
}

impl Drop for Held {
    /// Settles nested teardown even when the holder unwinds, so that none is
    /// lost and the next lock on this thread starts clean. A panic a
    /// teardown raised is dropped here; [`Self::unlock`] resumes it instead.
    fn drop(&mut self) {
        self.settle_nested();
        HOLDING.set(Holding::No);
    }
}
