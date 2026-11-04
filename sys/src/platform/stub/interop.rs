use std::{ffi::c_void, ptr::NonNull};

use super::{Context, DeviceBuffer, Stream, unsupported};
use crate::Result;

impl Context {
    pub const fn raw_context(&self) -> Result<NonNull<c_void>> {
        Err(unsupported())
    }

    #[must_use]
    pub const fn raw_device(&self) -> i32 {
        -1
    }
}

impl Stream {
    pub const fn raw_stream(&self) -> Result<NonNull<c_void>> {
        Err(unsupported())
    }
}

impl DeviceBuffer {
    pub const fn raw_pointer_for(&self, _stream: &Stream) -> Result<u64> {
        Err(unsupported())
    }
}
