use std::sync::{
    Arc, Mutex, MutexGuard, PoisonError,
    atomic::{AtomicBool, Ordering},
};

use super::{ACTIVE, CaptureActivity, activity, enter, leave, release};

mod context;
mod nested;
#[cfg(any(feature = "cublaslt", feature = "cutlass"))]
mod plan;

/// The gate is process-wide: unit tests that capture run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

pub(in crate::platform) fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Marks a capture on this thread without beginning one on any stream.
fn mark_capture() {
    assert!(enter(|| Ok::<_, ()>(())).is_ok());
}

/// A flag that teardown sets when it runs.
fn flag() -> (Arc<AtomicBool>, impl FnOnce() + Send + 'static) {
    let ran = Arc::new(AtomicBool::new(false));
    let set = ran.clone();
    (ran, move || set.store(true, Ordering::SeqCst))
}

#[expect(clippy::panic, reason = "the test drives a panic through deferred teardown")]
fn panicking_teardown() {
    std::panic::panic_any("the deferred teardown panics");
}

/// A deferred teardown that panics neither keeps the teardown after it from
/// running nor loses its panic.
#[test]
fn a_panicking_teardown_leaves_the_others_to_run() {
    let _serial = serial();
    mark_capture();
    release(panicking_teardown);
    let (ran, set) = flag();
    release(set);
    assert_eq!(activity().deferred_teardown, 2);
    assert!(std::panic::catch_unwind(leave).is_err(), "the teardown's panic was lost");
    assert!(ran.load(Ordering::SeqCst), "a teardown after the panic never ran");
    assert_eq!(activity(), CaptureActivity::default());
}

/// A teardown that panics while it runs at once raises its panic only after
/// the gate is let go, which leaves the gate usable.
#[test]
fn a_teardown_panicking_at_once_lets_go_of_the_gate() {
    let _serial = serial();
    assert!(std::panic::catch_unwind(|| release(panicking_teardown)).is_err());
    let (ran, set) = flag();
    release(set);
    assert!(ran.load(Ordering::SeqCst), "the gate stayed held after a panic");
    assert_eq!(activity(), CaptureActivity::default());
}

#[expect(clippy::panic, reason = "the test drives a panic through a capture's begin")]
fn panicking_begin() -> Result<(), ()> {
    std::panic::panic_any("the capture's begin panics")
}

/// A begin that panics counts no capture, so later teardown is not kept
/// waiting for a capture that never began.
#[test]
fn a_panicking_begin_counts_no_capture() {
    let _serial = serial();
    assert!(std::panic::catch_unwind(|| enter(panicking_begin)).is_err());
    assert_eq!(activity(), CaptureActivity::default());
    assert_eq!(ACTIVE.load(Ordering::SeqCst), 0);
    let (ran, set) = flag();
    release(set);
    assert!(ran.load(Ordering::SeqCst), "teardown waited for a capture that never began");
}

/// Teardown asking for the gate it runs under fails instead of deadlocking,
/// and the gate stays usable.
#[test]
fn teardown_taking_its_gate_again_fails() {
    let _serial = serial();
    let taken = std::panic::catch_unwind(|| {
        release(|| {
            let _activity = activity();
        });
    });
    assert!(taken.is_err(), "teardown took the gate it runs under");
    let (ran, set) = flag();
    release(set);
    assert!(ran.load(Ordering::SeqCst));
    assert_eq!(activity(), CaptureActivity::default());
}
