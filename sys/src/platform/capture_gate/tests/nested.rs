//! The gate stays held from finding no capture through the release it
//! guards, and teardown released while it is held settles before it is let
//! go instead of locking it again.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::Duration,
};

use super::{
    super::{release_ordered, release_when_idle},
    CaptureActivity, activity, enter, flag, leave, mark_capture, release, serial,
};

/// Long enough for a capture that does not wait for the gate to begin.
const BEGIN: Duration = Duration::from_millis(50);

/// A capture cannot begin on another thread between finding the gate idle
/// and the release that relies on it.
#[test]
fn a_release_when_idle_keeps_captures_from_beginning() {
    let _serial = serial();
    let began = Arc::new(AtomicBool::new(false));
    let mut capture: Option<JoinHandle<()>> = None;
    let ran = release_when_idle(|| {
        let marked = began.clone();
        capture = Some(std::thread::spawn(move || {
            let mark = || {
                marked.store(true, Ordering::SeqCst);
                Ok::<_, ()>(())
            };
            assert!(enter(mark).is_ok());
            leave();
        }));
        std::thread::sleep(BEGIN);
        assert!(!began.load(Ordering::SeqCst), "a capture began during the release");
    });
    assert!(ran, "an idle gate did not run the release");
    if let Some(capture) = capture
        && let Err(panic) = capture.join()
    {
        std::panic::resume_unwind(panic);
    }
    assert!(began.load(Ordering::SeqCst));
    assert_eq!(activity(), CaptureActivity::default());
}

/// During a capture the release is left to the caller, unrun.
#[test]
fn a_release_when_idle_declines_during_a_capture() {
    let _serial = serial();
    mark_capture();
    let ran = AtomicBool::new(false);
    assert!(!release_when_idle(|| ran.store(true, Ordering::SeqCst)));
    assert!(!ran.load(Ordering::SeqCst));
    leave();
    assert!(release_when_idle(|| ran.store(true, Ordering::SeqCst)));
    assert!(ran.load(Ordering::SeqCst));
}

/// Teardown releasing teardown, as a plan holding a buffer would, runs
/// both without locking the gate twice, at once or when the capture ends.
#[test]
fn teardown_released_by_teardown_settles_under_the_same_gate() {
    let _serial = serial();
    let (inner, set) = flag();
    release(move || {
        release(set);
        assert!(!release_when_idle(|| ()), "teardown locked the gate it runs under");
    });
    assert!(inner.load(Ordering::SeqCst), "nested teardown was lost");
    mark_capture();
    let (inner, set) = flag();
    release(move || release(set));
    assert_eq!(activity().deferred_teardown, 1);
    leave();
    assert!(inner.load(Ordering::SeqCst), "nested deferred teardown was lost");
    assert_eq!(activity(), CaptureActivity::default());
}

/// Device teardown released by stream-ordered teardown that runs during a
/// capture waits for that capture to end.
#[test]
fn nested_device_teardown_waits_for_the_capture() {
    let _serial = serial();
    mark_capture();
    let (inner, set) = flag();
    release_ordered(Vec::new(), None, move || {
        release(set);
        Ok(())
    });
    assert!(!inner.load(Ordering::SeqCst), "device teardown ran during a capture");
    assert_eq!(activity().deferred_teardown, 1);
    leave();
    assert!(inner.load(Ordering::SeqCst));
    assert_eq!(activity(), CaptureActivity::default());
}
