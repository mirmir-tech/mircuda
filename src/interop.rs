//! Borrowed driver handles for foreign kernel providers.
//!
//! Experiments that launch another runtime's kernels on `mircuda` streams and
//! memory use these handles instead of copying. `mircuda` keeps ownership: a
//! handle is valid only while the `Context`, `Stream`, or `DeviceBuffer` it was
//! taken from is alive, and foreign work must be ordered on the stream whose
//! handle it uses.

use std::{ffi::c_void, ptr::NonNull};

use crate::{Context, DeviceBuffer, DeviceElement, Result, Stream};

/// Borrowed CUDA primary context and its device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForeignContext {
    /// Raw `CUcontext`.
    pub context: NonNull<c_void>,
    /// Raw `CUdevice`.
    pub device: i32,
    /// Device ordinal used to create the context.
    pub ordinal: usize,
}

/// Borrowed device allocation range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForeignPointer {
    /// Raw `CUdeviceptr` of the first element.
    pub address: u64,
    /// Bytes covered by the typed view.
    pub bytes: usize,
}

impl Context {
    /// Returns the retained primary context for a foreign runtime to borrow.
    pub fn foreign_context(&self) -> Result<ForeignContext> {
        Ok(ForeignContext {
            context: self.native.raw_context()?,
            device: self.native.raw_device(),
            ordinal: self.device_info()?.ordinal,
        })
    }
}

impl Stream {
    /// Returns the raw `CUstream` for a foreign runtime to launch on.
    pub fn foreign_stream(&self) -> Result<NonNull<c_void>> {
        Ok(self.native.raw_stream()?)
    }
}

impl<T: DeviceElement> DeviceBuffer<T> {
    /// Returns this view's device range for foreign work queued on `stream`.
    ///
    /// Use on a stream other than the allocation stream is recorded exactly as
    /// for native launches, so dropping the buffer synchronizes first.
    pub fn foreign_pointer(&self, stream: &Stream) -> Result<ForeignPointer> {
        Ok(ForeignPointer {
            address: self.native.raw_pointer_for(&stream.native)?,
            bytes: self.len() * size_of::<T>(),
        })
    }
}
