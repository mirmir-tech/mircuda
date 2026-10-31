use std::sync::{Arc, atomic::Ordering};

use super::{DeviceBuffer, Stream};
use crate::{Error, Result};

impl DeviceBuffer {
    /// Returns the device address for foreign work queued on `stream`. Use on
    /// another stream of the same context is recorded like a native launch, so
    /// destruction synchronizes the context before freeing.
    pub fn raw_pointer_for(&self, stream: &Stream) -> Result<u64> {
        if !Arc::ptr_eq(self.stream.context(), stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        if self.stream.cu_stream() != stream.inner.cu_stream() {
            self.cross_stream.store(true, Ordering::Release);
        }
        Ok(self.pointer)
    }
}
