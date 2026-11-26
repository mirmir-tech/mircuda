//! A capture that ends early, by a panic, a failed begin or a failed
//! operation, releases the capture gate and leaves its stream able to capture.

use std::panic::AssertUnwindSafe;

use mircuda::{CaptureMode, DeviceBuffer, LaunchConfig, Stream, TypedKernel};

use super::{Fixture, Increment};

type Increments<'a> = (&'a TypedKernel<Increment>, &'a Stream, &'a mut DeviceBuffer<u32>);

#[expect(clippy::panic, reason = "the test drives a panic through a capture")]
fn panic_during_capture(_: &mut ()) -> mircuda::Result<()> {
    std::panic::panic_any("the captured operation panics")
}

#[cfg(feature = "cublaslt")]
#[test]
fn a_panicking_capture_releases_the_gate() -> mircuda::Result<()> {
    let fixture = Fixture::new()?;
    let stream = fixture.context.create_stream()?;
    let panicked = std::panic::catch_unwind(AssertUnwindSafe(|| {
        stream.capture((), panic_during_capture).map(drop)
    }));
    assert!(panicked.is_err());
    assert_eq!(mircuda::capture_activity().capturing_threads, 0);
    let spec = mircuda::CublasLtBf16Spec::new(64, 64, 64)?;
    drop(mircuda::CublasLtBf16Plan::new(&fixture.context, &stream, spec)?);
    assert_eq!(mircuda::capture_activity(), mircuda::CaptureActivity::default());
    // The stream captures again after the discarded capture.
    fixture.capture_during(|_| Ok(()))
}

/// A capture that begins inside another on the same stream fails to begin.
fn nested_capture(sequence: &mut Increments<'_>) -> mircuda::Result<()> {
    let (kernel, stream, value) = sequence;
    kernel.launch_captured(stream, LaunchConfig::for_elements(1, 1)?, (&mut **value, 1))?;
    assert!(stream.capture((), |()| Ok::<_, mircuda::Error>(())).is_err());
    assert_eq!(mircuda::capture_activity().capturing_threads, 1);
    Ok(())
}

#[test]
fn a_failed_begin_leaves_no_capture_behind() -> mircuda::Result<()> {
    let fixture = Fixture::new()?;
    let stream = fixture.context.create_stream()?;
    let mut value = fixture.pool.allocate_zeroed::<u32>(&stream, 1)?;
    // The outer capture may end invalidated by the failed begin; either way
    // both release the gate.
    let outer = stream.capture((&fixture.kernel, &stream, &mut value), nested_capture);
    drop(outer);
    assert_eq!(mircuda::capture_activity(), mircuda::CaptureActivity::default());
    fixture.capture_during(|_| Ok(()))
}

/// Synchronizing a capturing stream fails and invalidates its capture.
fn synchronize_during_capture(sequence: &mut Increments<'_>) -> mircuda::Result<()> {
    let (kernel, stream, value) = sequence;
    kernel.launch_captured(stream, LaunchConfig::for_elements(1, 1)?, (&mut **value, 1))?;
    stream.synchronize()
}

/// A failed operation discards its capture instead of instantiating it, and
/// the failed end of that invalidated capture stays out of the context: the
/// next call on the context succeeds.
#[test]
fn a_failed_operation_discards_its_capture() -> mircuda::Result<()> {
    let fixture = Fixture::new()?;
    let stream = fixture.context.create_stream()?;
    let mut value = fixture.pool.allocate_zeroed::<u32>(&stream, 1)?;
    let failed = stream.capture_or_recover(
        CaptureMode::ThreadLocal,
        (&fixture.kernel, &stream, &mut value),
        synchronize_during_capture,
    );
    assert!(failed.is_err(), "synchronizing a capturing stream succeeded");
    drop(failed);
    assert_eq!(mircuda::capture_activity(), mircuda::CaptureActivity::default());
    stream.synchronize()?;
    let mut host = fixture.context.allocate_pinned::<u32>(1)?;
    stream.copy_to_host(&value, &mut host)?;
    assert_eq!(host.to_vec()?, [0], "the discarded capture ran");
    fixture.capture_during(|_| Ok(()))
}
