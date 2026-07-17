mod handle;
mod transfer;

use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cudarc::driver::{CudaContext, CudaEvent, CudaStream, result, sys};

use self::handle::MemoryPoolHandle;
pub use self::handle::MemoryPoolStats;
use super::driver::{Context, Stream};
use crate::Result;

#[derive(Clone, Debug)]
pub struct MemoryPool {
    handle: MemoryPoolHandle,
    context: Arc<CudaContext>,
}

impl Context {
    pub fn default_memory_pool(&self) -> Result<MemoryPool> {
        self.inner.bind_to_thread()?;
        // SAFETY: cu_device is retained by this live context.
        let handle = unsafe { result::device::get_default_mem_pool(self.inner.cu_device()) }?;
        Ok(MemoryPool {
            handle: MemoryPoolHandle(handle),
            context: self.inner.clone(),
        })
    }

    pub fn allocate_pinned(&self, bytes: usize) -> Result<PinnedBuffer> {
        self.inner.bind_to_thread()?;
        // SAFETY: bytes is non-zero and the returned allocation is owned below.
        let raw = unsafe { result::malloc_host(bytes, 0) }?;
        let pointer = NonNull::new(raw.cast::<u8>()).ok_or(crate::Error::NullAllocation)?;
        // SAFETY: pointer covers exactly `bytes` writable bytes returned by CUDA.
        unsafe { pointer.as_ptr().write_bytes(0, bytes) };
        let event = self.inner.new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))?;
        Ok(PinnedBuffer { pointer, bytes, event })
    }
}

impl MemoryPool {
    pub fn set_release_threshold(&self, bytes: u64) -> Result<()> {
        self.context.bind_to_thread()?;
        set_attribute(
            self.handle.0,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            bytes,
        )
    }

    pub fn stats(&self) -> Result<MemoryPoolStats> {
        self.context.bind_to_thread()?;
        Ok(MemoryPoolStats {
            reserved: get_attribute(
                self.handle.0,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            )?,
            used: get_attribute(
                self.handle.0,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            )?,
        })
    }

    pub fn trim_to(&self, bytes: usize) -> Result<()> {
        self.context.bind_to_thread()?;
        // SAFETY: this handle is the live default pool retained by the context.
        Ok(unsafe { result::mem_pool::trim_to(self.handle.0, bytes) }?)
    }

    pub fn allocate(&self, stream: &Stream, bytes: usize, zeroed: bool) -> Result<DeviceBuffer> {
        if !Arc::ptr_eq(&self.context, stream.inner.context()) {
            return Err(crate::Error::ContextMismatch);
        }
        self.context.bind_to_thread()?;
        // SAFETY: pool and stream share the checked context and stay live in the allocation.
        let pointer = unsafe {
            result::mem_pool::alloc_async(self.handle.0, bytes, stream.inner.cu_stream())
        }?;
        if zeroed {
            // SAFETY: pointer covers `bytes` and uses its allocation stream.
            unsafe { result::memset_d8_async(pointer, 0, bytes, stream.inner.cu_stream()) }?;
        }
        Ok(DeviceBuffer {
            pointer,
            bytes,
            stream: stream.inner.clone(),
        })
    }
}

#[derive(Debug)]
pub struct DeviceBuffer {
    pointer: sys::CUdeviceptr,
    bytes: usize,
    stream: Arc<CudaStream>,
}

impl DeviceBuffer {
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    #[must_use]
    pub fn argument(&self) -> super::compiler::KernelArgument {
        super::compiler::KernelArgument::Pointer {
            value: self.pointer,
            stream: self.stream.cu_stream(),
        }
    }

    #[must_use]
    pub(super) const fn pointer(&self) -> sys::CUdeviceptr {
        self.pointer
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        self.stream.context().record_err(self.stream.context().bind_to_thread());
        // SAFETY: this is the sole owner and free is ordered on the allocation stream.
        self.stream
            .context()
            .record_err(unsafe { result::free_async(self.pointer, self.stream.cu_stream()) });
    }
}

#[derive(Debug)]
pub struct PinnedBuffer {
    pointer: NonNull<u8>,
    bytes: usize,
    event: CudaEvent,
}

// SAFETY: access is mediated by a CUDA event and mutable Rust borrows.
unsafe impl Send for PinnedBuffer {}

impl PinnedBuffer {
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn with_slice<T, R>(&self, len: usize, read: impl FnOnce(&[T]) -> R) -> Result<R> {
        validate_view::<T>(len, self.bytes)?;
        self.event.synchronize()?;
        // SAFETY: validate_view proves the typed slice covers exactly the allocation.
        let values = unsafe { std::slice::from_raw_parts(self.pointer.as_ptr().cast(), len) };
        Ok(read(values))
    }

    pub fn with_mut_slice<T, R>(
        &mut self,
        len: usize,
        write: impl FnOnce(&mut [T]) -> R,
    ) -> Result<R> {
        validate_view::<T>(len, self.bytes)?;
        self.event.synchronize()?;
        // SAFETY: validate_view proves the mutable slice covers the unique allocation.
        let values = unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr().cast(), len) };
        Ok(write(values))
    }

    fn record(&self, stream: &Stream) -> Result<()> {
        Ok(self.event.record(&stream.inner)?)
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        self.event.context().record_err(self.event.synchronize());
        // SAFETY: this is the sole owner and the event completed queued transfers.
        self.event
            .context()
            .record_err(unsafe { result::free_host(self.pointer.as_ptr().cast::<c_void>()) });
    }
}

pub(super) fn ensure_stream(buffer: &DeviceBuffer, stream: &Stream) -> Result<()> {
    if Arc::ptr_eq(&buffer.stream, &stream.inner) {
        Ok(())
    } else {
        Err(crate::Error::StreamMismatch)
    }
}

fn get_attribute(pool: sys::CUmemoryPool, attribute: sys::CUmemPool_attribute) -> Result<u64> {
    let mut value = 0_u64;
    // SAFETY: queried attributes use a writable u64 according to the Driver API.
    unsafe { result::mem_pool::get_attribute(pool, attribute, (&raw mut value).cast()) }?;
    Ok(value)
}

fn set_attribute(
    pool: sys::CUmemoryPool,
    attribute: sys::CUmemPool_attribute,
    mut value: u64,
) -> Result<()> {
    // SAFETY: configured attributes consume a readable u64 according to the Driver API.
    Ok(unsafe { result::mem_pool::set_attribute(pool, attribute, (&raw mut value).cast()) }?)
}

fn validate_view<T>(len: usize, bytes: usize) -> Result<()> {
    if std::mem::size_of::<T>().checked_mul(len) == Some(bytes) {
        Ok(())
    } else {
        Err(crate::Error::InvalidHostView)
    }
}

pub(super) const fn ensure_transfer_size(source: usize, target: usize) -> Result<()> {
    if source == target {
        Ok(())
    } else {
        Err(crate::Error::TransferSizeMismatch)
    }
}
