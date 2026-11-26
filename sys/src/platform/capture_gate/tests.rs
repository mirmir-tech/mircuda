use std::sync::{
    Arc, Mutex, MutexGuard, PoisonError,
    atomic::{AtomicBool, Ordering},
};

use super::{CaptureActivity, activity, enter, leave, release};

/// The gate is process-wide: unit tests that capture run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

pub(in crate::platform) fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
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
    assert!(enter(|| Ok::<_, ()>(())).is_ok());
    let ran = Arc::new(AtomicBool::new(false));
    release(panicking_teardown);
    let flag = ran.clone();
    release(move || flag.store(true, Ordering::SeqCst));
    assert_eq!(activity().deferred_teardown, 2);
    assert!(std::panic::catch_unwind(leave).is_err(), "the teardown's panic was lost");
    assert!(ran.load(Ordering::SeqCst), "a teardown after the panic never ran");
    assert_eq!(activity(), CaptureActivity::default());
}
