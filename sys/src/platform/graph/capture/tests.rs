use cudarc::driver::{DriverError, sys};

use crate::platform::{CaptureMode, Driver, Stream, capture_activity, capture_gate::tests::serial};

const UNKNOWN: DriverError = DriverError(sys::cudaError_enum::CUDA_ERROR_UNKNOWN);

fn stream() -> crate::Result<Stream> {
    Driver::initialize()?.create_context(0)?.create_stream()
}

fn assert_ended(stream: &Stream) -> crate::Result<()> {
    assert_eq!(
        stream.inner.capture_status()?,
        sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    );
    assert_eq!(capture_activity().capturing_threads, 0);
    Ok(())
}

/// An error another operation recorded on the context belongs to that
/// context's next caller: ending a capture neither fails on it nor consumes
/// it, and the capture ends.
#[test]
fn ending_a_capture_leaves_a_recorded_error_in_place() -> crate::Result<()> {
    let _serial = serial();
    let stream = stream()?;
    let context = stream.inner.context().clone();
    let capture = stream.begin_capture(CaptureMode::ThreadLocal)?;
    context.record_err::<()>(Err(UNKNOWN));
    // An empty capture may or may not instantiate; either way it has ended.
    drop(capture.end());
    assert_eq!(context.check_err(), Err(UNKNOWN));
    assert_ended(&stream)?;
    let capture = stream.begin_capture(CaptureMode::ThreadLocal)?;
    context.record_err::<()>(Err(UNKNOWN));
    drop(capture);
    assert_eq!(context.check_err(), Err(UNKNOWN));
    assert_ended(&stream)
}
