//! Asking the driver which capture a stream records into needs the stream's
//! context current, whichever context the releasing thread last used.

use std::sync::atomic::Ordering;

use cudarc::driver::{result, sys};

use super::{super::release_ordered, CaptureActivity, activity, flag, serial};
use crate::platform::{CaptureMode, Driver};

/// A context of its own on the same device, current on this thread until it
/// is dropped.
struct OtherContext(sys::CUcontext);

impl OtherContext {
    fn current(device: sys::CUdevice) -> crate::Result<Self> {
        let mut context = std::ptr::null_mut();
        // SAFETY: the output is writable and no creation parameters are
        // given; the new context becomes current.
        unsafe { sys::cuCtxCreate_v4(&raw mut context, std::ptr::null_mut(), 0, device) }
            .result()?;
        Ok(Self(context))
    }
}

impl Drop for OtherContext {
    fn drop(&mut self) {
        // SAFETY: this guard solely owns the context it created.
        let _destroyed = unsafe { sys::cuCtxDestroy_v2(self.0) };
    }
}

/// Leaves no context current on this thread. Setting none pops only the top
/// of the thread's context stack, so it repeats until the stack is empty.
pub(super) fn clear_current() -> crate::Result<()> {
    while result::ctx::get_current()?.is_some() {
        // SAFETY: unbinding a context leaves it alive.
        unsafe { result::ctx::set_current(std::ptr::null_mut()) }?;
    }
    Ok(())
}

/// Stream-ordered teardown on a stream that captures nothing runs at once
/// during another stream's capture, whether another context or none is
/// current when it is released, and leaves that context current.
#[test]
fn a_release_under_another_context_still_finds_its_stream_idle() -> crate::Result<()> {
    let _serial = serial();
    let context = Driver::initialize()?.create_context(0)?;
    let (captured, released) = (context.create_stream()?, context.create_stream()?);
    let capture = captured.begin_capture(CaptureMode::ThreadLocal)?;
    let other = OtherContext::current(context.inner.cu_device())?;
    let (other_ran, set) = flag();
    release_ordered(vec![released.inner.clone()], None, move || {
        set();
        Ok(())
    });
    let under_other = activity().deferred_teardown;
    let kept_other = result::ctx::get_current()? == Some(other.0);
    clear_current()?;
    let (none_ran, set) = flag();
    // The stream outlives the teardown: cudarc destroying it binds its
    // context, leaving it current.
    release_ordered(vec![released.inner.clone()], None, move || {
        set();
        Ok(())
    });
    let under_none = activity().deferred_teardown;
    let kept_none = result::ctx::get_current()?.is_none();
    drop(capture);
    drop(other);
    drop(released);
    assert!(other_ran.load(Ordering::SeqCst) && none_ran.load(Ordering::SeqCst));
    assert_eq!(under_other, 0, "teardown under another context waited for a capture");
    assert_eq!(under_none, 0, "teardown under no context waited for a capture");
    assert!(kept_other && kept_none, "the release left its context current");
    assert_eq!(activity(), CaptureActivity::default());
    Ok(())
}
