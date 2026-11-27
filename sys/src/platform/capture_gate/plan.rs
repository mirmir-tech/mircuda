//! Teardown of native plans bound to one stream.

use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cudarc::driver::CudaStream;

use super::super::driver::bind_current;
#[cfg(feature = "cublaslt")]
use super::release;
#[cfg(feature = "cutlass")]
use super::{CaptureId, release_ordered};

/// A native plan destructor.
pub(in crate::platform) type Destroy = unsafe extern "C" fn(*mut c_void);

/// A native plan handle moved into teardown.
struct NativePlan(NonNull<c_void>);

// SAFETY: the plan is owned by its teardown, which is its only remaining user.
unsafe impl Send for NativePlan {}

impl NativePlan {
    /// Destroys the plan with its stream's context current. When that
    /// context cannot be made current the plan is leaked and the failure
    /// recorded: destroying it under another context would free its memory
    /// there.
    fn destroy(self, stream: &CudaStream, destroy: Destroy) {
        let context = stream.context();
        match bind_current(context) {
            // SAFETY: the teardown solely owns the plan, its context is
            // current, and `stream` outlives the release it enqueues.
            Ok(_current) => unsafe { destroy(self.0.as_ptr()) },
            Err(error) => context.record_err::<()>(Err(error)),
        }
    }
}

/// Destroys at once a plan whose destructor neither enqueues work nor
/// synchronizes, so no capture can see it.
///
/// # Safety
///
/// `raw` must be a live plan that `destroy` accepts, solely owned by the
/// caller, whose destructor only releases host memory.
#[cfg(feature = "cutlass")]
pub(in crate::platform) unsafe fn destroy_plan(
    stream: &CudaStream,
    raw: NonNull<c_void>,
    destroy: Destroy,
) {
    NativePlan(raw).destroy(stream, destroy);
}

/// Destroys a plan whose destructor frees memory ordered on `stream`, once
/// `stream` captures nothing or only `own`, the capture that allocated that
/// memory. A free enqueued into another capture would become a node of that
/// graph for memory it did not allocate.
///
/// # Safety
///
/// `raw` must be a live plan that `destroy` accepts, solely owned by the
/// caller, whose destructor enqueues work on no stream other than `stream`.
#[cfg(feature = "cutlass")]
pub(in crate::platform) unsafe fn release_stream_plan(
    stream: &Arc<CudaStream>,
    own: Option<CaptureId>,
    raw: NonNull<c_void>,
    destroy: Destroy,
) {
    let (plan, stream) = (NativePlan(raw), stream.clone());
    release_ordered(vec![stream.clone()], own, move || {
        plan.destroy(&stream, destroy);
        Ok(())
    });
}

/// Destroys a plan whose destructor may synchronize the device once no
/// capture is in progress anywhere.
///
/// # Safety
///
/// `raw` must be a live plan that `destroy` accepts, solely owned by the
/// caller, whose destructor enqueues work on no stream other than `stream`.
#[cfg(feature = "cublaslt")]
pub(in crate::platform) unsafe fn release_device_plan(
    stream: &Arc<CudaStream>,
    raw: NonNull<c_void>,
    destroy: Destroy,
) {
    let (plan, stream) = (NativePlan(raw), stream.clone());
    release(move || plan.destroy(&stream, destroy));
}
