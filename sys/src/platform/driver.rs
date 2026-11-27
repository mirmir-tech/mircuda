use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaStream, DriverError, result,
    sys::{CUcontext, CUdevice_attribute},
};

use crate::Result;

#[derive(Debug)]
pub struct Driver;

impl Driver {
    pub fn initialize() -> Result<Self> {
        let _ = CudaContext::device_count()?;
        Ok(Self)
    }

    pub fn device_count(&self) -> Result<usize> {
        Ok(usize::try_from(CudaContext::device_count()?)?)
    }

    pub fn create_context(&self, ordinal: usize) -> Result<Context> {
        let inner = CudaContext::new(ordinal)?;
        // SAFETY: event tracking only orders cudarc's own `CudaSlice` views,
        // which mircuda never creates: device buffers record their foreign
        // streams and order their frees explicitly. Left on, cudarc would
        // synchronize the shared primary context when the first stream is
        // created, invalidating stream captures on other threads.
        unsafe { inner.disable_event_tracking() };
        Ok(Context { inner })
    }
}

#[derive(Clone, Debug)]
pub struct Context {
    pub(super) inner: Arc<CudaContext>,
}

impl Context {
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        Ok(self.inner.mem_get_info()?)
    }

    pub fn device_info(&self) -> Result<DeviceInfo> {
        Ok(DeviceInfo {
            ordinal: self.inner.ordinal(),
            name: self.inner.name()?,
            compute_capability: self.inner.compute_capability()?,
            multiprocessor_count: self
                .inner
                .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            total_memory: self.inner.total_mem()?,
            memory_pools: self.inner.has_async_alloc(),
            integrated: self.inner.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)?
                != 0,
        })
    }

    pub fn create_stream(&self) -> Result<Stream> {
        Ok(Stream { inner: self.inner.new_stream()? })
    }
}

#[derive(Clone, Debug)]
pub struct Stream {
    pub(super) inner: Arc<CudaStream>,
}

impl Stream {
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.inner.synchronize()?)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub multiprocessor_count: i32,
    pub total_memory: usize,
    pub memory_pools: bool,
    pub integrated: bool,
}

/// Makes `context` current on this thread through the driver alone. Unlike
/// cudarc's bind, it neither consumes nor fails on an error that an earlier
/// operation recorded on the context for its next caller.
pub(super) fn make_current(context: &CudaContext) -> std::result::Result<(), DriverError> {
    // SAFETY: the borrowed context is live.
    unsafe { result::ctx::set_current(context.cu_ctx()) }
}

/// A context made current by [`bind_current`], until dropped.
#[must_use = "dropping the binding makes the previous context current again"]
pub(super) struct CurrentContext {
    /// The context current before, made current again on drop; `None` when
    /// the bound context was current already.
    previous: Option<CUcontext>,
}

/// Makes `context` current on this thread, like [`make_current`], until the
/// returned binding drops, which makes current again whatever context the
/// thread had: teardown and capture queries run on threads that may be using
/// another context of their own. A thread with the context current already
/// pays one query.
pub(super) fn bind_current(
    context: &CudaContext,
) -> std::result::Result<CurrentContext, DriverError> {
    let current = result::ctx::get_current()?;
    if current == Some(context.cu_ctx()) {
        return Ok(CurrentContext { previous: None });
    }
    make_current(context)?;
    Ok(CurrentContext {
        previous: Some(current.unwrap_or(std::ptr::null_mut())),
    })
}

impl Drop for CurrentContext {
    fn drop(&mut self) {
        if let Some(previous) = self.previous {
            // SAFETY: restoring the thread's previous binding, or none; a
            // context destroyed since only makes the driver refuse it.
            let _restored = unsafe { result::ctx::set_current(previous) };
        }
    }
}
