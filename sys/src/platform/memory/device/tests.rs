use std::{
    ffi::c_void,
    sync::Arc,
    time::{Duration, Instant},
};

use cudarc::driver::{CudaContext, DriverError, result, sys};

use super::synchronized_free;
use crate::platform::{Driver, capture_gate::tests::serial};

const EARLIER: DriverError = DriverError(sys::cudaError_enum::CUDA_ERROR_UNKNOWN);
const DURING: DriverError = DriverError(sys::cudaError_enum::CUDA_ERROR_NOT_READY);
const BUSY: Duration = Duration::from_millis(200);

/// Host work on another stream that keeps the context busy, then records an
/// error the way a drop on another thread would while the free waits.
unsafe extern "C" fn busy_then_record(context: *mut c_void) {
    std::thread::sleep(BUSY);
    // SAFETY: the test passes a pointer from `Arc::into_raw` and keeps that
    // context alive until the callback has run.
    let context = unsafe { Arc::from_raw(context.cast_const().cast::<CudaContext>()) };
    context.record_err::<()>(Err(DURING));
}

/// The synchronized fallback free waits for the whole context although an
/// earlier error is recorded on it, and an error recorded meanwhile stays in
/// place for the context's next caller instead of being overwritten.
#[test]
fn a_synchronized_free_waits_and_keeps_errors_recorded_meanwhile() -> crate::Result<()> {
    let _serial = serial();
    let context = Driver::initialize()?.create_context(0)?;
    let (stream, busy) = (context.create_stream()?, context.create_stream()?);
    let cuda = stream.inner.context().clone();
    cuda.bind_to_thread()?;
    // SAFETY: the stream is live; the allocation is freed by the teardown.
    let pointer = unsafe { result::malloc_async(stream.inner.cu_stream(), 4) }?;
    let user = Arc::into_raw(cuda.clone()).cast_mut().cast::<c_void>();
    // SAFETY: the callback takes the context reference passed as user data.
    unsafe { sys::cuLaunchHostFunc(busy.inner.cu_stream(), Some(busy_then_record), user) }
        .result()?;
    cuda.record_err::<()>(Err(EARLIER));
    let start = Instant::now();
    synchronized_free(pointer, stream.inner)();
    assert!(start.elapsed() >= BUSY, "the free did not wait for the context");
    assert_eq!(cuda.check_err(), Err(DURING), "an error recorded meanwhile was lost");
    Ok(())
}
