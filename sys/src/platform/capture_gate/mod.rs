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
//!
//! Every release decides and runs holding the gate, which beginning a capture
//! also takes, so no capture can begin between finding a stream idle and
//! enqueuing work on it. Teardown released by teardown running under the gate
//! settles before the gate is let go, instead of locking it again.

use std::{
    cell::Cell,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use cudarc::driver::CudaStream;

mod deferred;
mod held;
#[cfg(any(feature = "cublaslt", feature = "cutlass"))]
mod plan;
#[cfg(test)]
pub(in crate::platform) mod tests;

pub(super) use self::deferred::{CaptureId, capture_of};
#[cfg(feature = "cublaslt")]
pub(super) use self::plan::release_device_plan;
#[cfg(feature = "cutlass")]
pub(super) use self::plan::{destroy_plan, release_stream_plan};
use self::{
    deferred::{Deferred, Ordered},
    held::Held,
};

pub(super) type Teardown = Box<dyn FnOnce() + Send>;

struct State {
    /// Threads with a capture in progress or beginning.
    active: usize,
    /// Teardown waiting for the captures it would disturb to end.
    deferred: Vec<Deferred>,
}

static STATE: Mutex<State> = Mutex::new(State { active: 0, deferred: Vec::new() });

/// `State::active`, readable without the gate so that recording where an
/// allocation belongs stays cheap outside any capture. It counts a capture
/// before the capture begins, so that work the driver orders after the begin
/// also observes the count.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Captures this thread has begun and not yet ended. Const-initialized
    /// and without a destructor, so it stays accessible at thread exit.
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Begins a capture on this thread through `begin` and marks it when it
/// succeeds. Nested captures on the same thread count once. Holding the gate
/// across `begin` keeps a capture from beginning while teardown checks or
/// uses its stream.
pub(super) fn enter<T, E>(begin: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    let mut held = Held::lock();
    let mut beginning = (DEPTH.get() == 0).then(|| Beginning::count(&mut held.state));
    let capture = begin()?;
    beginning.as_mut().map(Beginning::begun);
    drop(beginning);
    DEPTH.set(DEPTH.get() + 1);
    // No teardown runs while a capture begins, so there is no panic to
    // resume that could unwind past the capture just begun.
    drop(held);
    Ok(capture)
}

/// A capture counted before it begins, so that work the driver orders after
/// the begin observes the count. Unless it begins, failing or panicking, the
/// count is taken back.
struct Beginning<'state> {
    state: &'state mut State,
    begun: bool,
}

impl<'state> Beginning<'state> {
    fn count(state: &'state mut State) -> Self {
        state.active += 1;
        ACTIVE.store(state.active, Ordering::Release);
        Self { state, begun: false }
    }

    const fn begun(&mut self) {
        self.begun = true;
    }
}

impl Drop for Beginning<'_> {
    fn drop(&mut self) {
        if !self.begun {
            self.state.active -= 1;
            ACTIVE.store(self.state.active, Ordering::Release);
        }
    }
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
    let mut held = Held::lock();
    if depth == 1 {
        held.state.active = held.state.active.saturating_sub(1);
        ACTIVE.store(held.state.active, Ordering::Release);
    }
    held.run_ready();
    held.unlock();
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

/// The process-wide capture activity, for diagnostics and tests. Teardown
/// running under the gate cannot ask: that panics instead of deadlocking.
#[doc(hidden)]
#[must_use]
pub fn activity() -> CaptureActivity {
    let held = Held::lock();
    CaptureActivity {
        capturing_threads: held.state.active,
        deferred_teardown: held.state.deferred.len(),
    }
}

/// Runs `release` now, holding the gate, when no capture is in progress, and
/// returns whether it ran. Holding the gate from the check through `release`
/// keeps a capture from beginning in between, on any thread; teardown
/// running under the gate on this thread gets `false` and goes through
/// [`release_ordered`] instead.
pub(super) fn release_when_idle(release: impl FnOnce()) -> bool {
    if held::holding() {
        return false;
    }
    let held = Held::lock();
    if held.state.active > 0 {
        return false;
    }
    release();
    held.unlock();
    true
}

/// Runs device-synchronizing teardown now when no capture is in progress,
/// otherwise once the last capture ends, even when the only capture is the
/// calling thread's own. Teardown runs with the gate held; whatever it
/// releases through the gate in turn settles before the gate is let go.
#[cfg(any(feature = "cublaslt", test))]
pub(super) fn release(teardown: impl FnOnce() + Send + 'static) {
    settle(Deferred::Device(Box::new(teardown)));
}

/// Runs teardown that enqueues work on `streams` now when none of them is
/// capturing a capture other than `own`, otherwise as soon as none is.
/// `ordered` returns the device-synchronizing fallback it needs when stream
/// ordering fails, which then runs as [`release`] would run it.
///
/// Holding the gate from the check through `ordered` keeps a capture from
/// beginning on those streams in between.
pub(super) fn release_ordered<F>(streams: Vec<Arc<CudaStream>>, own: Option<CaptureId>, ordered: F)
where
    F: FnOnce() -> Result<(), Teardown> + Send + 'static,
{
    settle(Deferred::Ordered(Ordered { streams, own, release: Box::new(ordered) }));
}

/// Runs `entry` now when no capture would see it, otherwise defers it. Under
/// a gate this thread already holds, the holder settles it.
fn settle(entry: Deferred) {
    if let Some(entry) = held::nest(entry) {
        let mut held = Held::lock();
        held.settle(entry);
        held.unlock();
    }
}
