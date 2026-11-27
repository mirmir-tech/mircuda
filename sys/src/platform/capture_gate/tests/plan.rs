//! Native plans are destroyed with their own context current, even when an
//! earlier operation left an error recorded on it or another context, or
//! none, is current on the releasing thread, which keeps its own afterwards.

use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cudarc::driver::{CudaStream, DriverError, result, sys};

#[cfg(feature = "cublaslt")]
use super::super::release_device_plan;
#[cfg(feature = "cutlass")]
use super::super::{CaptureId, release_stream_plan};
use super::serial;
use crate::platform::Driver;

const EARLIER: DriverError = DriverError(sys::cudaError_enum::CUDA_ERROR_UNKNOWN);

/// A plan that records the context current when it is destroyed.
type Plan = Option<sys::CUcontext>;

unsafe extern "C" fn record_current(raw: *mut c_void) {
    let current = result::ctx::get_current().ok().flatten();
    // SAFETY: the test passes a live, writable plan and destroys it once.
    unsafe { raw.cast::<Plan>().write(current) };
}

type Release = fn(&Arc<CudaStream>, NonNull<c_void>);

fn destroyed_under(release: Release) -> crate::Result<()> {
    let _serial = serial();
    let context = Driver::initialize()?.create_context(0)?;
    let stream = context.create_stream()?.inner;
    let plan = NonNull::from(Box::leak(Box::new(Plan::None)));
    context.inner.record_err::<()>(Err(EARLIER));
    super::context::clear_current()?;
    release(&stream, plan.cast());
    let after = result::ctx::get_current()?;
    // SAFETY: the plan was leaked above and its destructor has run.
    let current = *unsafe { Box::from_raw(plan.as_ptr()) };
    assert_eq!(current, Some(context.inner.cu_ctx()), "the plan was destroyed off its context");
    assert_eq!(context.inner.check_err(), Err(EARLIER), "the recorded error was lost");
    assert_eq!(after, None, "the teardown left its context current");
    Ok(())
}

#[cfg(feature = "cublaslt")]
#[test]
fn a_device_plan_is_destroyed_under_its_context() -> crate::Result<()> {
    destroyed_under(|stream, plan| {
        // SAFETY: the plan is solely owned here and enqueues nothing.
        unsafe { release_device_plan(stream, plan, record_current) };
    })
}

#[cfg(feature = "cutlass")]
#[test]
fn a_stream_plan_is_destroyed_under_its_context() -> crate::Result<()> {
    destroyed_under(|stream, plan| {
        let own: Option<CaptureId> = None;
        // SAFETY: the plan is solely owned here and enqueues nothing.
        unsafe { release_stream_plan(stream, own, plan, record_current) };
    })
}
