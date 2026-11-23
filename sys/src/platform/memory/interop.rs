use std::sync::Arc;

use super::{DeviceBuffer, Stream};
use crate::{Error, Result};

impl DeviceBuffer {
    /// Returns the device address for foreign work queued on `stream`. Use on
    /// another stream of the same context is recorded like a native launch, so
    /// the stream-ordered free waits for the work queued on that stream.
    pub fn raw_pointer_for(&self, stream: &Stream) -> Result<u64> {
        if !Arc::ptr_eq(self.stream.context(), stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        if self.stream.cu_stream() != stream.inner.cu_stream() {
            self.foreign.record(&stream.inner);
        }
        Ok(self.pointer)
    }
}
