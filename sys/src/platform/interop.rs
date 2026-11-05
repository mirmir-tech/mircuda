use std::{ffi::c_void, ptr::NonNull};

use super::{Context, Stream};
use crate::{Error, Result};

impl Context {
    pub fn raw_context(&self) -> Result<NonNull<c_void>> {
        NonNull::new(self.inner.cu_ctx().cast()).ok_or(Error::NullHandle)
    }

    #[must_use]
    pub fn raw_device(&self) -> i32 {
        self.inner.cu_device()
    }
}

impl Stream {
    pub fn raw_stream(&self) -> Result<NonNull<c_void>> {
        NonNull::new(self.inner.cu_stream().cast()).ok_or(Error::NullHandle)
    }
}
